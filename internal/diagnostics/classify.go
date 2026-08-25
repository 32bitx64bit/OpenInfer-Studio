package diagnostics

import (
	"regexp"
	"strconv"
	"strings"
)

// FailureClass is a stable machine-readable classification.
type FailureClass string

const (
	ClassUnknown            FailureClass = "unknown"
	ClassUnsupportedArch    FailureClass = "unsupported-architecture"
	ClassRuntimeTooOld      FailureClass = "runtime-too-old"
	ClassMissingProjector   FailureClass = "missing-projector"
	ClassWrongProjector     FailureClass = "incorrect-projector"
	ClassAudioUnsupported   FailureClass = "audio-unsupported"
	ClassCorruptGGUF        FailureClass = "corrupt-gguf"
	ClassMissingShard       FailureClass = "missing-shard"
	ClassInsufficientRAM    FailureClass = "insufficient-ram"
	ClassInsufficientVRAM   FailureClass = "insufficient-vram"
	ClassContextAlloc       FailureClass = "context-allocation-failure"
	ClassUnsupportedBackend FailureClass = "unsupported-backend"
	ClassMissingDriver      FailureClass = "missing-driver"
	ClassInvalidFlag        FailureClass = "invalid-flag"
	ClassPortConflict       FailureClass = "port-conflict"
	ClassPermission         FailureClass = "permission-error"
	ClassInvalidPath        FailureClass = "invalid-path"
	ClassDraftModel         FailureClass = "draft-model"
	ClassRuntimeCrash       FailureClass = "runtime-crash"
	ClassTimeout            FailureClass = "timeout"
	ClassTensorShape        FailureClass = "tensor-shape"
)

// Classification explains a failure conservatively.
type Classification struct {
	Class      FailureClass `json:"class"`
	Summary    string       `json:"summary"`
	Suggestion string       `json:"suggestion"`
}

type rule struct {
	class   FailureClass
	pattern *regexp.Regexp
	summary string
	fix     string
}

var rules = []rule{
	{ClassUnsupportedArch, regexp.MustCompile(`(?i)(unknown model architecture|unsupported architecture|architecture.*not supported)`),
		"The runtime does not support this model architecture.",
		"Update to a newer llama.cpp runtime, or try a different build."},
	{ClassTensorShape, regexp.MustCompile(`(?i)(check_tensor_dims|has wrong shape)`),
		"A GGUF tensor has the wrong shape for this architecture (often vocab size vs embedding rows).",
		"Reconvert from the Hugging Face weights so embedding rows, output rows, and tokenizer length agree. Or use a GGUF built for this llama.cpp version."},
	{ClassRuntimeTooOld, regexp.MustCompile(`(?i)(requires newer|gguf version.*not supported|unsupported gguf version)`),
		"This model uses a GGUF format newer than the selected runtime understands.",
		"Install the latest llama.cpp runtime and pin it to this model."},
	{ClassMissingShard, regexp.MustCompile(`(?i)(split file|shard).*(missing|not found|failed to open)`),
		"A shard of a split GGUF model is missing.",
		"Verify all -0000N-of-0000M files are present in the model directory."},
	{ClassCorruptGGUF, regexp.MustCompile(`(?i)(bad magic|invalid gguf|corrupt|failed to read.*header|tensor.*out of bounds|checksum|has offset \d+, expected \d+|failed to read tensor data)`),
		"The GGUF file appears corrupt or incomplete (tensor layout inconsistent).",
		"If the download checksum passed, the uploaded file itself is broken — report it to the model author or try a different quantization. Otherwise delete and re-download."},
	{ClassMissingProjector, regexp.MustCompile(`(?i)(mmproj|projector).*(not found|missing|failed to open)`),
		"A multimodal projector file is missing or the path is wrong.",
		"Select the correct mmproj file in the model's load settings."},
	{ClassWrongProjector, regexp.MustCompile(`(?i)(mmproj|projector).*(mismatch|incompatible|does not match)`),
		"The projector file does not match this model.",
		"Use the projector published with this exact model variant."},
	{ClassAudioUnsupported, regexp.MustCompile(`(?i)audio input is not supported`),
		"The runtime rejected audio input (missing or incompatible multimodal projector, or an older llama-server).",
		"Load the matching mmproj, enable Jinja for chat templates, and update to a recent multimodal llama.cpp runtime. Audio support is experimental upstream."},
	{ClassInsufficientVRAM, regexp.MustCompile(`(?i)(cuda|vulkan|hip|metal).*(out of memory|oom|allocation failed)|ggml_cuda.*alloc`),
		"The GPU ran out of memory while loading or allocating context.",
		"Reduce GPU layers, lower the context size, or switch KV cache to q8_0. A CPU fallback is available."},
	{ClassInsufficientRAM, regexp.MustCompile(`(?i)(out of memory|cannot allocate memory|std::bad_alloc|memory allocation failed)`),
		"System memory was insufficient for this configuration.",
		"Reduce context size, pick a smaller quantization, or enable memory mapping."},
	{ClassContextAlloc, regexp.MustCompile(`(?i)(failed to allocate.*(context|kv)|kv cache.*(failed|allocation))`),
		"Allocating the KV cache / context failed.",
		"Lower the context size or use a quantized KV cache type."},
	{ClassMissingDriver, regexp.MustCompile(`(?i)(libcuda|nvidia.*driver|vk_icd|vulkan.*driver|libamdhip|rocm).*(not found|cannot open|missing)|no vulkan device`),
		"A required GPU driver or runtime library was not found.",
		"Install or repair the GPU driver, or switch to a CPU runtime."},
	{ClassUnsupportedBackend, regexp.MustCompile(`(?i)(backend.*(not supported|unavailable)|no suitable (device|backend)|device.*not found)`),
		"The selected acceleration backend is unavailable on this machine.",
		"Choose a different backend (Vulkan or CPU) for this model."},
	{ClassRuntimeCrash, regexp.MustCompile(`(?i)(failed to (decode|process) (image|mtmd)|failed to find a memory slot|failed to prepare attention ubatches)`),
		"The runtime aborted while processing a request — the attention batch ran out of memory (seen with parallel multimodal requests).",
		"Lower parallel slots, increase the context size, or reduce image sizes for this model."},
	{ClassInvalidFlag, regexp.MustCompile(`(?i)(unknown (argument|option|flag)|invalid argument|error: unrecognized|error while handling argument|unknown value for)`),
		"The runtime rejected a command-line argument.",
		"Check the generated command against this runtime's supported flags; remove expert raw arguments."},
	{ClassPortConflict, regexp.MustCompile(`(?i)(address already in use|bind.*failed|EADDRINUSE)`),
		"The inference server could not bind its local port.",
		"Retry — OpenInfer will allocate a different port."},
	{ClassPermission, regexp.MustCompile(`(?i)(permission denied|EACCES|operation not permitted)`),
		"The operating system denied access to a file or resource.",
		"Check permissions on the model file and runtime directory."},
	{ClassInvalidPath, regexp.MustCompile(`(?i)(no such file or directory|ENOENT|failed to open.*model)`),
		"A required file path does not exist.",
		"Rescan the library; the model files may have moved."},
	{ClassDraftModel, regexp.MustCompile(`(?i)(draft model|model.draft|spec.draft|speculative).*(failed|error|incompatible|mismatch|not found|vocab)`),
		"The speculative decoding draft model failed to load or is incompatible with the target.",
		"Pick a smaller draft of the same architecture and tokenizer, or disable draft filtering in Settings if using a specialized draft (EAGLE/DFlash)."},
	{ClassTimeout, regexp.MustCompile(`(?i)(timed? ?out|deadline exceeded)`),
		"The model did not become ready within the startup timeout.",
		"Large models can take minutes on CPU; watch the live log, or increase the timeout in Settings."},
}

// Classify examines combined stderr/log text and returns the most specific
// matching classification, or a conservative unknown. Raw errors are always
// shown alongside this in the UI.
func Classify(logText string, exitCode int, timedOut bool) Classification {
	if timedOut {
		for _, r := range rules {
			if r.class == ClassTimeout {
				return Classification{r.class, r.summary, r.fix}
			}
		}
	}
	for _, r := range rules {
		if r.pattern.MatchString(logText) {
			return Classification{r.class, r.summary, r.fix}
		}
	}
	summary := "The model process exited unexpectedly."
	if exitCode != 0 {
		summary = "llama-server exited with code " + strconv.Itoa(exitCode) + "."
	}
	return Classification{ClassUnknown, summary,
		"Review the full log for the underlying error. Retrying with safe settings may help."}
}

// RedactPaths optionally replaces the user's home directory in exports.
func RedactPaths(s, home string) string {
	if home == "" {
		return s
	}
	return strings.ReplaceAll(s, home, "~")
}
