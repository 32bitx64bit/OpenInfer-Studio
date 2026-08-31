package instances

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

// LoadSettings is the user-facing model-load configuration.
// Numeric zero values select automatic defaults unless noted.
type LoadSettings struct {
	ContextLength  int     `json:"context_length"`  // 0 = model default
	GPUOffload     string  `json:"gpu_offload"`     // auto|all|none|custom
	GPULayers      int     `json:"gpu_layers"`      // used when GPUOffload == custom
	Threads        int     `json:"threads"`         // 0 = auto
	FlashAttention string  `json:"flash_attention"` // auto|on|off
	Parallel       int     `json:"parallel"`        // 0 = default 1
	BatchSize      int     `json:"batch_size"`
	UBatchSize     int     `json:"ubatch_size"`
	CacheTypeK     string  `json:"cache_type_k"` // "" = default
	CacheTypeV     string  `json:"cache_type_v"`
	NoMmap         bool    `json:"no_mmap"`
	MLock          bool    `json:"mlock"`
	NUMA           string  `json:"numa"`
	MainGPU        int     `json:"main_gpu"` // -1 = unset
	Device         string  `json:"device"`
	SplitMode      string  `json:"split_mode"`
	TensorSplit    string  `json:"tensor_split"`
	ContBatching   *bool   `json:"cont_batching"`
	CacheReuse     int     `json:"cache_reuse"`
	ThreadsBatch   int     `json:"threads_batch"` // 0 = same as --threads
	Prio           int     `json:"prio"`          // -2 = unset; -1..3 = llama --prio
	Poll           int     `json:"poll"`          // -1 = unset; 0..100 = --poll
	CPUMoe         bool    `json:"cpu_moe"`
	NCPUMoe        int     `json:"n_cpu_moe"`  // 0 = unset
	KVOffload      string  `json:"kv_offload"` // ""|on|off → --kv-offload / --no-kv-offload
	OpOffload      string  `json:"op_offload"` // ""|on|off → --op-offload / --no-op-offload
	KVUnified      string  `json:"kv_unified"` // ""|on|off → --kv-unified / --no-kv-unified
	SWAFull        bool    `json:"swa_full"`
	Fit            string  `json:"fit"` // ""|on|off → --fit
	NoWarmup       bool    `json:"no_warmup"`
	RopeScaling    string  `json:"rope_scaling"`
	RopeFreqBase   float64 `json:"rope_freq_base"`
	RopeFreqScale  float64 `json:"rope_freq_scale"`
	SleepIdleSec   int     `json:"sleep_idle_seconds"`
	Alias          string  `json:"alias"`
	MediaPath      string  `json:"media_path"`
	ChatTemplate   string  `json:"chat_template"`
	// ChatTemplateKwargs is passed as --chat-template-kwargs (JSON object).
	// Muse Glimmer uses {"reasoning_strength":"low"|"medium"|"high"|"xhigh"}.
	ChatTemplateKwargs string  `json:"chat_template_kwargs"`
	Jinja              *bool   `json:"jinja"`
	NoMmproj           bool    `json:"no_mmproj"` // skip paired multimodal projector
	NoMmprojOffload    bool    `json:"no_mmproj_offload"`
	LoraPath           string  `json:"lora_path"`
	LoraScale          float64 `json:"lora_scale"`
	DraftModel         string  `json:"draft_model"` // path to draft GGUF
	DraftMax           int     `json:"draft_max"`   // max draft tokens (0 = runtime default)
	DraftMin           int     `json:"draft_min"`   // min draft tokens (0 = runtime default)
	SpecType           string  `json:"spec_type"`   // e.g. draft-simple; empty = auto when draft set
	// Embedding enables --embedding (dedicated embedder / reranker use case).
	// Nil = unset (defaults applied for detected embedders); false disables.
	Embedding *bool `json:"embedding"`
	// Pooling is --pooling none|mean|cls|last|rank; empty = model / runtime default.
	Pooling         string `json:"pooling"`
	ReasoningFormat string `json:"reasoning_format"`
	// ReasoningBudget is an optional server-wide --reasoning-budget. Nil =
	// unset so per-request chat params.reasoning_budget can take effect
	// (older llama.cpp ignores the request body when this CLI flag is set).
	// 0 disables thinking; -1 is unlimited.
	ReasoningBudget *int `json:"reasoning_budget,omitempty"`
	// ReasoningPreserve keeps prior-turn reasoning in the chat template
	// (--reasoning-preserve). Nil = unset (defaults on when the GGUF
	// template supports it); false disables.
	ReasoningPreserve *bool             `json:"reasoning_preserve"`
	RawArgs           string            `json:"raw_args"`      // expert: space-separated, validated, never shell
	EnvOverrides      map[string]string `json:"env_overrides"` // allowlisted keys only

	// RuntimeID selects a specific installed llama-server build for this load.
	// Empty = model pin, else global preferred. Not a llama-server flag.
	RuntimeID string `json:"runtime_id"`

	// SaveOnSuccess persists these settings as the model's last-known-good
	// preset once the instance reaches ready. Failed loads leave any previous
	// good preset untouched. Not a llama-server flag.
	SaveOnSuccess bool `json:"save_on_success"`
}

// DefaultSettings returns safe automatic defaults. Context defaults to 4096:
// larger windows cost KV-cache memory linearly, so the model's maximum is an
// explicit opt-in, not the default. GPU offload defaults to all layers.
func DefaultSettings() LoadSettings {
	return LoadSettings{
		ContextLength: 4096, GPUOffload: "all", FlashAttention: "auto",
		MainGPU: -1, Prio: -2, Poll: -1,
	}
}

// envAllowlist restricts expert environment overrides.
var envAllowlist = map[string]bool{
	"GGML_CUDA_NO_PINNED": true, "GGML_VULKAN_DEVICE": true,
	"GGML_SCHED_DEBUG": true, "LLAMA_ARG_THREADS": true,
	"CUDA_VISIBLE_DEVICES": true, "ROCR_VISIBLE_DEVICES": true,
	"GGML_LOG_LEVEL": true,
	"NGL":            true, "MAXTOK": true,
}

// Resolution records how an Auto value resolved, for transparent display.
type Resolution struct {
	Setting  string `json:"setting"`
	Auto     string `json:"auto"`
	Resolved string `json:"resolved"`
}

// BuildResult is the generated command plus transparency data.
type BuildResult struct {
	Args        []string     `json:"args"`
	Command     string       `json:"command"` // display-only, quoted
	Resolutions []Resolution `json:"resolutions"`
	Warnings    []string     `json:"warnings"` // unsupported settings dropped
}

// BuildArgs generates the llama-server argument vector. Flags unsupported by
// the selected runtime (per its captured --help) are dropped with warnings,
// never blindly passed. host/port/apiKey are backend-generated.
func BuildArgs(s LoadSettings, modelPath, projectorPath string,
	caps []string, help, host string, port int, apiKey string) BuildResult {

	var args []string
	var res []Resolution
	var warn []string

	add := func(flag string, values ...string) {
		if !runtimes.SupportsFlag(caps, help, flag) {
			warn = append(warn, fmt.Sprintf("%s is not supported by this runtime; setting skipped", flag))
			return
		}
		args = append(args, flag)
		args = append(args, values...)
	}

	// addFirst picks the first supported spelling of a renamed flag
	// (e.g. --draft-max was removed in favor of --spec-draft-n-max).
	addFirst := func(values []string, flags ...string) {
		for _, f := range flags {
			if runtimes.SupportsFlag(caps, help, f) {
				args = append(args, f)
				args = append(args, values...)
				return
			}
		}
		warn = append(warn, fmt.Sprintf("none of %s supported by this runtime; setting skipped", strings.Join(flags, ", ")))
	}

	args = append(args, "--model", modelPath)
	if projectorPath != "" && !s.NoMmproj {
		add("--mmproj", projectorPath)
	}
	if s.NoMmproj {
		// Prevent auto-discovery of a same-directory mmproj on newer runtimes.
		add("--no-mmproj")
		if projectorPath != "" {
			res = append(res, Resolution{"Multimodal projector", "paired", "skipped (--no-mmproj)"})
		}
	}
	add("--host", host)
	add("--port", strconv.Itoa(port))
	add("--api-key", apiKey)
	// GET /slots drives the library/detail activity meters. Older llama.cpp
	// disables the endpoint unless --slots is passed; current builds default
	// it on, so this is a no-op there.
	add("--slots")

	// Context. llama.cpp splits --ctx-size across --parallel slots, so the
	// flag receives ctx × slots: every request gets the full window, and
	// total KV memory scales linearly with slots (by design).
	slots := s.Parallel
	if slots < 1 {
		slots = 1
	}
	if s.ContextLength > 0 {
		total := s.ContextLength * slots
		add("--ctx-size", strconv.Itoa(total))
		resolved := strconv.Itoa(s.ContextLength)
		if slots > 1 {
			resolved = fmt.Sprintf("%d per slot × %d slots (%d total)", s.ContextLength, slots, total)
		}
		res = append(res, Resolution{"Context length", "model default", resolved})
	} else {
		res = append(res, Resolution{"Context length", "auto", "model default"})
		if slots > 1 {
			warn = append(warn, "context length is model-default; slots share it. Set an explicit context length to give every slot the full window")
		}
	}

	// GPU offload.
	switch s.GPUOffload {
	case "all":
		add("--n-gpu-layers", "999")
		res = append(res, Resolution{"GPU offload", "all", "all layers (999)"})
	case "none":
		add("--n-gpu-layers", "0")
		res = append(res, Resolution{"GPU offload", "none", "CPU only"})
	case "custom":
		if s.GPULayers > 0 {
			add("--n-gpu-layers", strconv.Itoa(s.GPULayers))
			res = append(res, Resolution{"GPU offload", "custom", fmt.Sprintf("%d layers", s.GPULayers)})
		}
	default: // auto
		add("--n-gpu-layers", "999")
		res = append(res, Resolution{"GPU offload", "auto", "all layers when a GPU backend is active"})
	}

	if s.Threads > 0 {
		add("--threads", strconv.Itoa(s.Threads))
		res = append(res, Resolution{"CPU threads", "auto", strconv.Itoa(s.Threads)})
	} else {
		res = append(res, Resolution{"CPU threads", "auto", "runtime default"})
	}
	if s.ThreadsBatch > 0 {
		add("--threads-batch", strconv.Itoa(s.ThreadsBatch))
		res = append(res, Resolution{"Batch threads", "same as --threads", strconv.Itoa(s.ThreadsBatch)})
	}

	// Flash attention. Newer llama.cpp builds take a value
	// (--flash-attn on|off|auto); older ones use a bare boolean switch.
	faValued := runtimes.FlagTakesValue(help, "--flash-attn")
	switch s.FlashAttention {
	case "on":
		if faValued {
			add("--flash-attn", "on")
		} else {
			add("--flash-attn")
		}
		res = append(res, Resolution{"Flash Attention", "on", "enabled"})
	case "off":
		if faValued {
			add("--flash-attn", "off")
		} // boolean builds: omitting the flag disables it
		res = append(res, Resolution{"Flash Attention", "off", "disabled"})
	default:
		if faValued {
			add("--flash-attn", "auto")
		} else {
			add("--flash-attn")
		}
		res = append(res, Resolution{"Flash Attention", "auto", "enabled (safe on all current backends)"})
	}

	// Speculative decoding (especially draft-mtp) needs a single slot. Recent
	// llama-server defaults can be >1 when --parallel is omitted, which silently
	// disables speculation — pin to 1 when unset and a draft/spec type is active.
	speculative := s.DraftModel != "" || strings.TrimSpace(s.SpecType) != ""
	switch {
	case speculative && s.Parallel <= 0:
		add("--parallel", "1")
		res = append(res, Resolution{"Parallel slots", "auto for speculative", "1"})
	case s.Parallel > 0:
		add("--parallel", strconv.Itoa(s.Parallel))
	}
	if s.BatchSize > 0 {
		add("--batch-size", strconv.Itoa(s.BatchSize))
	}
	if s.UBatchSize > 0 {
		add("--ubatch-size", strconv.Itoa(s.UBatchSize))
	}
	if s.CacheTypeK != "" {
		add("--cache-type-k", s.CacheTypeK)
		res = append(res, Resolution{"KV cache K", "auto", s.CacheTypeK})
	}
	if s.CacheTypeV != "" {
		add("--cache-type-v", s.CacheTypeV)
		res = append(res, Resolution{"KV cache V", "auto", s.CacheTypeV})
	}
	if s.NoMmap {
		add("--no-mmap")
	}
	if s.MLock {
		add("--mlock")
	}
	if s.NUMA != "" {
		add("--numa", s.NUMA)
	}
	if s.MainGPU >= 0 {
		add("--main-gpu", strconv.Itoa(s.MainGPU))
	}
	if s.Device != "" {
		add("--device", s.Device)
	}
	if s.SplitMode != "" {
		add("--split-mode", s.SplitMode)
	}
	if s.TensorSplit != "" {
		add("--tensor-split", s.TensorSplit)
	}
	if s.ContBatching != nil {
		if *s.ContBatching {
			add("--cont-batching")
		} else {
			add("--no-cont-batching")
		}
	}
	if s.CacheReuse > 0 {
		add("--cache-reuse", strconv.Itoa(s.CacheReuse))
	}
	// Process scheduling / latency knobs (expert).
	if s.Prio >= -1 && s.Prio <= 3 {
		add("--prio", strconv.Itoa(s.Prio))
	}
	if s.Poll >= 0 && s.Poll <= 100 {
		add("--poll", strconv.Itoa(s.Poll))
	}
	if s.CPUMoe {
		add("--cpu-moe")
	}
	if s.NCPUMoe > 0 {
		add("--n-cpu-moe", strconv.Itoa(s.NCPUMoe))
	}
	addOnOff := func(v, onFlag, offFlag, label string) {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "on", "1", "true":
			add(onFlag)
			res = append(res, Resolution{label, "default", "on"})
		case "off", "0", "false":
			add(offFlag)
			res = append(res, Resolution{label, "default", "off"})
		}
	}
	addOnOff(s.KVOffload, "--kv-offload", "--no-kv-offload", "KV offload")
	addOnOff(s.OpOffload, "--op-offload", "--no-op-offload", "Op offload")
	addOnOff(s.KVUnified, "--kv-unified", "--no-kv-unified", "Unified KV")
	if s.SWAFull {
		add("--swa-full")
	}
	switch strings.ToLower(strings.TrimSpace(s.Fit)) {
	case "on":
		if runtimes.FlagTakesValue(help, "--fit") {
			add("--fit", "on")
		} else {
			add("--fit")
		}
		res = append(res, Resolution{"Fit to VRAM", "default", "on"})
	case "off":
		if runtimes.FlagTakesValue(help, "--fit") {
			add("--fit", "off")
		}
		res = append(res, Resolution{"Fit to VRAM", "default", "off"})
	}
	if s.NoWarmup {
		add("--no-warmup")
	}
	if s.RopeScaling != "" {
		add("--rope-scaling", s.RopeScaling)
	}
	if s.RopeFreqBase > 0 {
		add("--rope-freq-base", strconv.FormatFloat(s.RopeFreqBase, 'f', -1, 64))
	}
	if s.RopeFreqScale > 0 {
		add("--rope-freq-scale", strconv.FormatFloat(s.RopeFreqScale, 'f', -1, 64))
	}
	if s.SleepIdleSec > 0 {
		add("--sleep-idle-seconds", strconv.Itoa(s.SleepIdleSec))
	}
	if s.Alias != "" {
		add("--alias", s.Alias)
	}
	if s.MediaPath != "" {
		add("--media-path", s.MediaPath)
	}
	if s.ChatTemplate != "" {
		add("--chat-template", s.ChatTemplate)
	}
	kwargs := s.ChatTemplateKwargs
	preserveFlag := runtimes.SupportsFlag(caps, help, "--reasoning-preserve")
	if s.ReasoningPreserve != nil && !preserveFlag {
		kwargs = mergeChatTemplateKwargs(kwargs, preserveReasoningKwargs(*s.ReasoningPreserve))
	}
	if kwargs != "" {
		add("--chat-template-kwargs", kwargs)
	}
	if s.Jinja != nil && *s.Jinja {
		add("--jinja")
	}
	if s.ReasoningPreserve != nil {
		if *s.ReasoningPreserve {
			if preserveFlag {
				add("--reasoning-preserve")
				res = append(res, Resolution{"Preserve reasoning", "template default", "on"})
			} else if kwargs != s.ChatTemplateKwargs {
				res = append(res, Resolution{"Preserve reasoning", "template default", "on (chat-template-kwargs)"})
			}
		} else if runtimes.SupportsFlag(caps, help, "--no-reasoning-preserve") {
			add("--no-reasoning-preserve")
			res = append(res, Resolution{"Preserve reasoning", "template default", "off"})
		}
	}
	if s.NoMmprojOffload && !s.NoMmproj {
		add("--no-mmproj-offload")
	}
	if s.LoraPath != "" {
		if s.LoraScale > 0 {
			add("--lora-scaled", s.LoraPath, strconv.FormatFloat(s.LoraScale, 'f', -1, 64))
		} else {
			add("--lora", s.LoraPath)
		}
	}

	// Resolve speculative decoding. llama.cpp defaults --spec-type to none;
	// without an explicit type a draft can load while speculation never runs.
	specType := resolveSpecType(s)
	if s.DraftModel != "" {
		addFirst([]string{s.DraftModel}, "--model-draft", "--spec-draft-model")
		if specType != "" {
			if runtimes.SupportsFlag(caps, help, "--spec-type") {
				add("--spec-type", specType)
				label := "auto"
				if strings.TrimSpace(s.SpecType) != "" {
					label = "explicit"
				}
				res = append(res, Resolution{"Speculative type", label, specType})
			} else if s.SpecType != "" {
				warn = append(warn, "--spec-type is not supported by this runtime; setting skipped")
			}
		}
		if s.Parallel > 1 {
			warn = append(warn, "parallel slots > 1 often disable or weaken draft speculative decoding; use 1 for MTP/EAGLE")
		}
	} else if specType != "" {
		// Fused-trunk MTP / ngram modes: --spec-type without a draft GGUF.
		add("--spec-type", specType)
		res = append(res, Resolution{"Speculative type", "explicit", specType})
	}
	if s.DraftMax > 0 {
		addFirst([]string{strconv.Itoa(s.DraftMax)}, "--draft-max", "--spec-draft-n-max")
	}
	if s.DraftMin > 0 {
		addFirst([]string{strconv.Itoa(s.DraftMin)}, "--draft-min", "--spec-draft-n-min")
	}
	if s.Embedding != nil && *s.Embedding {
		added := false
		for _, flag := range []string{"--embedding", "--embeddings"} {
			if runtimes.SupportsFlag(caps, help, flag) {
				add(flag)
				added = true
				res = append(res, Resolution{"Embedding mode", "explicit", "on"})
				break
			}
		}
		if !added {
			warn = append(warn, "--embedding is not supported by this runtime; setting skipped")
		}
	}
	if p := strings.ToLower(strings.TrimSpace(s.Pooling)); p != "" {
		switch p {
		case "none", "mean", "cls", "last", "rank":
			if runtimes.SupportsFlag(caps, help, "--pooling") {
				add("--pooling", p)
				res = append(res, Resolution{"Pooling", "explicit", p})
			} else {
				warn = append(warn, "--pooling is not supported by this runtime; setting skipped")
			}
		default:
			warn = append(warn, "invalid pooling value "+p+"; use none|mean|cls|last|rank")
		}
	}
	if s.ReasoningFormat != "" {
		add("--reasoning-format", s.ReasoningFormat)
	}
	if s.ReasoningBudget != nil {
		add("--reasoning-budget", strconv.Itoa(*s.ReasoningBudget))
		res = append(res, Resolution{"Reasoning budget", "explicit", strconv.Itoa(*s.ReasoningBudget) + " tokens"})
	}

	// Expert raw arguments: the field is rejected wholesale if it contains
	// shell metacharacters (execution never involves a shell, but this keeps
	// the vector clean). Otherwise flags are filtered against runtime support;
	// non-flag tokens are only accepted as values of a preceding supported flag.
	if strings.TrimSpace(s.RawArgs) != "" {
		tokens := strings.Fields(s.RawArgs)
		unsafe := false
		for _, tok := range tokens {
			if strings.ContainsAny(tok, ";&|<>$`\\\"'") {
				unsafe = true
				break
			}
		}
		if unsafe {
			warn = append(warn, "raw arguments contain unsafe characters and were rejected entirely")
		} else {
			prevFlagAccepted := false
			for _, tok := range tokens {
				if strings.HasPrefix(tok, "--") {
					if !runtimes.SupportsFlag(caps, help, tok) {
						warn = append(warn, fmt.Sprintf("raw flag %s not supported by this runtime; skipped", tok))
						prevFlagAccepted = false
						continue
					}
					args = append(args, tok)
					prevFlagAccepted = true
					continue
				}
				if prevFlagAccepted {
					args = append(args, tok)
					prevFlagAccepted = false // one value per flag; next bare token is rejected
					continue
				}
				warn = append(warn, fmt.Sprintf("raw token %q rejected (not a flag value)", tok))
			}
		}
	}

	return BuildResult{Args: args, Command: quoteCommand(args), Resolutions: res, Warnings: warn}
}

// resolveSpecType picks --spec-type from explicit settings and draft GGUF signals
// (official mtp-/eagle3-/dflash-/dspark- sidecars, dedicated arches, name hints).
func resolveSpecType(s LoadSettings) string {
	var draftSpec gguf.SpecType
	if s.DraftModel != "" {
		if md, err := gguf.ParseFile(s.DraftModel); err == nil {
			md.ApplySpeculativeFlags(s.DraftModel)
			draftSpec = md.SpecType
		} else {
			_, draftSpec = gguf.DetectSpeculativeDraft("", "", s.DraftModel, nil)
		}
	}
	return string(gguf.InferSpecType(s.SpecType, s.DraftModel, draftSpec))
}

// ApplyTemplateDefaults fills Jinja, Muse Glimmer chat-template kwargs,
// reasoning-preserve, and --reasoning-format auto when the client omitted
// them. Glimmer requires --jinja even without an mmproj: OID quants are
// written to a new folder and lose the projector pairing that previously
// flipped Jinja on. Without Jinja, stop tokens and reasoning_content split
// break and chat looks like it never streamed. Thinking models
// (reasoning_effort / enable_thinking / …) also force Jinja so per-request
// chat_template_kwargs actually reach the template. Templates that can keep
// prior-turn reasoning default --reasoning-preserve on (the llama.cpp log
// hint). Think-token caps live in chat params (reasoning_budget), not here:
// a CLI --reasoning-budget would pin older llama.cpp and ignore per-request
// overrides.
func ApplyTemplateDefaults(s *LoadSettings, arch string, multimodal, draft, embedding bool, reasoning gguf.Reasoning) {
	if s == nil || draft || embedding {
		return
	}
	if !reasoning.Controllable() && !reasoning.CanPreserve {
		reasoning = gguf.DetectReasoning("", arch)
	}
	if reasoning.CanPreserve && s.ReasoningPreserve == nil {
		t := true
		s.ReasoningPreserve = &t
	}
	glimmer := gguf.IsMuseGlimmerChat(arch)
	preserveOn := s.ReasoningPreserve != nil && *s.ReasoningPreserve
	forceJinja := glimmer || reasoning.Controllable() || preserveOn
	needJinja := gguf.NeedsJinja(arch, multimodal) || reasoning.Controllable() || preserveOn
	if needJinja && (s.Jinja == nil || forceJinja) {
		t := true
		s.Jinja = &t
	}
	if glimmer && strings.TrimSpace(s.ChatTemplateKwargs) == "" {
		s.ChatTemplateKwargs = gguf.GlimmerChatTemplateKwargs
	}
	thinking := glimmer || reasoning.Controllable()
	if thinking && s.ReasoningFormat == "" {
		s.ReasoningFormat = "auto"
	}
}

func mergeChatTemplateKwargs(existing string, extra map[string]any) string {
	if len(extra) == 0 {
		return existing
	}
	out := map[string]any{}
	if strings.TrimSpace(existing) != "" {
		if err := json.Unmarshal([]byte(existing), &out); err != nil {
			out = map[string]any{}
		}
	}
	for k, v := range extra {
		out[k] = v
	}
	b, err := json.Marshal(out)
	if err != nil {
		return existing
	}
	return string(b)
}

func preserveReasoningKwargs(on bool) map[string]any {
	return map[string]any{
		"preserve_reasoning":        on,
		"preserve_thinking":         on,
		"clear_thinking":            !on,
		"truncate_history_thinking": !on,
		"drop_thinking":             !on,
	}
}

// quoteCommand renders the argument vector for the preview pane. It is
// display-only; execution always uses the vector directly.
func quoteCommand(args []string) string {
	var b strings.Builder
	for i, a := range args {
		if i > 0 {
			b.WriteByte(' ')
		}
		if strings.ContainsAny(a, " \t\"'") {
			b.WriteString(strconv.Quote(a))
		} else {
			b.WriteString(a)
		}
	}
	return b.String()
}

// FilterEnv applies the override allowlist.
func FilterEnv(overrides map[string]string) (map[string]string, []string) {
	out := map[string]string{}
	var rejected []string
	for k, v := range overrides {
		if envAllowlist[k] {
			out[k] = v
		} else {
			rejected = append(rejected, k)
		}
	}
	return out, rejected
}
