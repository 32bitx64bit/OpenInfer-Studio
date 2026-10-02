package mediagen

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/sdmodel"
)

// packedFile is a file a launch hands to sd-server that is stored in a
// quantization pack stable-diffusion.cpp is known to abort on.
type packedFile struct {
	Role string // "diffusion model", "VAE", "LLM text encoder", …
	Path string
	Why  string
}

// packedFiles checks the model and every pipeline file the settings name.
func packedFiles(modelPath string, s LoadSettings) []packedFile {
	parts := []struct{ role, path string }{
		{"diffusion model", modelPath},
		{CompanionLabel(CompanionVAE), s.VAE},
		{CompanionLabel(CompanionLLM), s.LLM},
		{CompanionLabel(CompanionT5XXL), s.T5XXL},
		{CompanionLabel(CompanionClipL), s.ClipL},
		{CompanionLabel(CompanionClipG), s.ClipG},
		{CompanionLabel(CompanionClipVision), s.ClipVision},
	}
	var out []packedFile
	for _, p := range parts {
		if p.path == "" {
			continue
		}
		if why := sdmodel.PackedReason(p.path); why != "" {
			out = append(out, packedFile{Role: p.role, Path: p.path, Why: why})
		}
	}
	return out
}

// packedWarnings are the pre-launch warnings for packedFiles: shown in the
// load dialog and written to the log before sd-server is started.
func packedWarnings(modelPath string, s LoadSettings) []string {
	var out []string
	for _, p := range packedFiles(modelPath, s) {
		out = append(out, fmt.Sprintf(
			"%s %s is stored as %s, a ComfyUI-only quantization. stable-diffusion.cpp usually aborts while loading it; a GGUF, BF16/FP16 or plain FP8 build loads.",
			p.Role, filepath.Base(p.Path), p.Why))
	}
	return out
}

var (
	// gdbFrameRe is a frame line of the backtrace ggml prints through gdb when
	// it aborts: "#9  0x00007f… in GGMLBlock::init(…) () from /…/lib.so".
	gdbFrameRe = regexp.MustCompile(`^#\d+\s`)
	// gdbNoiseRe are the other lines gdb prints around it.
	gdbNoiseRe = regexp.MustCompile(`^(0x[0-9a-fA-F]+ in |\[New LWP|\[Thread debugging|Using host libthread_db|\[Inferior \d+ |\(No debugging symbols|warning: |Download failed|This GDB supports)`)
)

// cleanLogTail returns the last n lines of an sd-server log with the
// backtrace noise removed, so the line that says why it died (the assertion
// ggml printed before calling gdb) is not pushed out by thirty stack frames.
// A log that is nothing but noise comes back as its raw tail.
func cleanLogTail(text string, n int) string {
	var kept []string
	for _, line := range strings.Split(strings.TrimRight(text, "\n"), "\n") {
		trimmed := strings.TrimSpace(line)
		if gdbFrameRe.MatchString(trimmed) || gdbNoiseRe.MatchString(trimmed) {
			continue
		}
		if len(kept) > 0 && kept[len(kept)-1] == line {
			continue // repeated line
		}
		kept = append(kept, line)
	}
	if len(kept) == 0 {
		return lastLines(text, n)
	}
	if len(kept) > n {
		kept = kept[len(kept)-n:]
	}
	return strings.Join(kept, "\n")
}

// looksLikeLoaderAbort reports whether a log ends in an abort inside
// stable-diffusion.cpp's model construction rather than an ordinary error.
func looksLikeLoaderAbort(text string) bool {
	lower := strings.ToLower(text)
	for _, marker := range []string{"ggml_abort", "ggml_assert", "aborted", "sigabrt", "libstable-diffusion"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

// startupFailure builds the error for an sd-server that died before it was
// ready: the cause first (named packed files when the abort fits them), the
// cleaned log tail, and where the full log is.
func startupFailure(logText, logPath, modelPath string, s LoadSettings) string {
	var b strings.Builder
	b.WriteString("sd-server exited during startup")
	if packed := packedFiles(modelPath, s); len(packed) > 0 && looksLikeLoaderAbort(logText) {
		b.WriteString(": it aborted while loading weights stored in a format stable-diffusion.cpp cannot read.\n")
		for _, p := range packed {
			fmt.Fprintf(&b, "  - %s %s: %s\n", p.Role, filepath.Base(p.Path), p.Why)
		}
		b.WriteString("These packs are meant for ComfyUI. Use a GGUF, BF16/FP16 or plain FP8 build of the model, VAE and text encoders.\n")
	} else {
		b.WriteString(":\n")
	}
	tail := cleanLogTail(logText, 12)
	if strings.TrimSpace(tail) == "" {
		tail = "no output captured"
	}
	b.WriteString("\nsd-server output:\n")
	b.WriteString(tail)
	if logPath != "" {
		b.WriteString("\n\nFull log: " + logPath)
	}
	return b.String()
}
