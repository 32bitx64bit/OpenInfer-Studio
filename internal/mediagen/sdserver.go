package mediagen

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

// sdServerNames are the executable basenames that count as a usable sd.cpp
// HTTP server. sd-cli is deliberately excluded: it is a one-shot CLI with no
// --listen-port/HTTP server, so launching it with server flags can never
// become ready (bug: it used to be accepted here and callers would poll a
// port that no listener ever opened).
var sdServerNames = []string{"sd-server", "sd-server.exe"}

// sdCLINames are one-shot sd.cpp CLI basenames: real sd.cpp builds, but not
// servers.
var sdCLINames = []string{"sd_cli", "sd_cli.exe", "sd-cli", "sd-cli.exe"}

func isSDServerName(base string) bool {
	base = strings.ToLower(base)
	for _, n := range sdServerNames {
		if base == n {
			return true
		}
	}
	return false
}

func isSDCLIName(base string) bool {
	base = strings.ToLower(base)
	for _, n := range sdCLINames {
		if base == n {
			return true
		}
	}
	return false
}

// SDServerPath resolves the sd-server executable for a runtime record.
// Official sd.cpp installs point ExecutablePath at sd-server directly;
// custom llama.cpp imports may point at llama-server, in which case a
// colocated sd-server sibling is used when present. sd-cli (no HTTP server)
// is never accepted, whether as the record itself or as a sibling.
func SDServerPath(executablePath string) (string, error) {
	if executablePath == "" {
		return "", fmt.Errorf("runtime has no executable path")
	}
	base := filepath.Base(executablePath)
	if isSDServerName(base) {
		if _, err := os.Stat(executablePath); err != nil {
			return "", fmt.Errorf("sd-server executable missing: %w", err)
		}
		return executablePath, nil
	}
	if isSDCLIName(base) {
		return "", fmt.Errorf("runtime %s is sd-cli, a one-shot CLI with no HTTP server; install or import sd-server instead", base)
	}
	// llama-server (or other) record: look only for an sd-server sibling in
	// the same directory — never sd-cli, which cannot serve requests.
	dir := filepath.Dir(executablePath)
	for _, name := range sdServerNames {
		cand := filepath.Join(dir, name)
		if st, err := os.Stat(cand); err == nil && !st.IsDir() {
			return cand, nil
		}
	}
	return "", fmt.Errorf("runtime %s has no sd-server executable (sd-cli alone cannot serve requests)", filepath.Base(dir))
}

// IsSDRuntime reports whether a runtime record is a usable stable-diffusion
// server runtime. A directory that only has sd-cli is a real sd.cpp build
// but not a usable server, so it does not count.
func IsSDRuntime(rt *runtimes.Runtime) bool {
	if rt == nil {
		return false
	}
	base := filepath.Base(rt.ExecutablePath)
	if isSDServerName(base) {
		return true
	}
	if isSDCLIName(base) {
		return false
	}
	dir := filepath.Dir(rt.ExecutablePath)
	if dir == "" || dir == "." {
		return false
	}
	for _, n := range sdServerNames {
		if st, err := os.Stat(filepath.Join(dir, n)); err == nil && !st.IsDir() {
			return true
		}
	}
	return false
}

// FindSDServer locates the sd-server executable inside an extracted runtime
// tree (used by runtime install + custom import). sd-cli is not returned:
// callers need a process they can launch as an HTTP server.
func FindSDServer(root string) string {
	var found string
	_ = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil || found != "" {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if isSDServerName(d.Name()) {
			found = p
		}
		return nil
	})
	return found
}

// ParseSDCapabilities extracts capability ids from `sd-server --help`
// output. Only flags present in help are considered supported, and only
// supported flags are ever passed (same rule as llama.cpp capabilities).
func ParseSDCapabilities(help string) []string {
	caps := runtimes.ParseCapabilities(help) // shared llama-style flags
	seen := map[string]bool{}
	for _, c := range caps {
		seen[c] = true
	}
	for flag, cap := range sdKnownFlags {
		_ = flag
		if seen[cap] {
			continue
		}
		if strings.Contains(help, flag) && !removedInHelp(help, flag) {
			caps = append(caps, cap)
		}
	}
	return caps
}

// sdKnownFlags maps sd.cpp CLI flags to capability ids (superset of the
// llama.cpp table in runtimes/capabilities.go).
var sdKnownFlags = map[string]string{
	"--model": "--model", "--clip_l": "clip-l", "--clip_g": "clip-g",
	"--clip_vision": "clip-vision", "--t5xxl": "t5xxl", "--llm": "llm",
	"--llm_vision": "llm-vision", "--tokenizer": "tokenizer",
	"--diffusion-model":            "diffusion-model",
	"--high-noise-diffusion-model": "high-noise-diffusion-model",
	"--vae":                        "vae", "--vae-format": "vae-format", "--taesd": "taesd", "--tae": "taesd",
	"--control-net": "control-net", "--ip-adapter": "ip-adapter",
	"--motion-module": "motion-module", "--embd-dir": "embd-dir",
	"--lora-model-dir": "lora-model-dir", "--hires-upscalers-dir": "hires-upscalers-dir",
	"--upscale-model": "upscale-model", "--type": "weight-type",
	"--threads": "threads", "--backend": "sd-backend", "--params-backend": "params-backend",
	"--split-mode": "split-mode", "--rpc-servers": "rpc-servers",
	"--max-vram": "max-vram", "--auto-fit": "auto-fit",
	"--offload-to-cpu": "offload-to-cpu", "--mmap": "mmap",
	"--clip-on-cpu": "clip-on-cpu", "--vae-on-cpu": "vae-on-cpu",
	"--control-net-cpu": "control-net-cpu", "--fa": "flash-attn",
	"--diffusion-fa": "diffusion-fa", "--sage-attn": "sage-attn",
	"--diffusion-conv-direct": "diffusion-conv-direct",
	"--vae-conv-direct":       "vae-conv-direct",
	"--vae-tiling":            "vae-tiling", "--temporal-tiling": "temporal-tiling",
	"--eager-load": "eager-load",
	"--hires":      "hires",
	"--listen-ip":  "listen-ip", "--listen-port": "listen-port",
	"--serve-html-path": "serve-html-path",
	"--prompt":          "prompt", "--negative-prompt": "negative-prompt",
	"--width": "width", "--height": "height", "--steps": "steps",
	"--cfg-scale": "cfg-scale", "--img-cfg-scale": "img-cfg-scale",
	"--guidance": "guidance", "--seed": "seed",
	"--sampler": "sampler", "--sampling-method": "sampler",
	"--scheduler": "scheduler", "--batch-count": "batch-count",
	"--video-frames": "video-frames", "--fps": "fps",
	"--strength": "strength", "--output": "output",
	"--init-img": "init-img", "--mask": "mask",
	"--lora-apply-mode": "lora-apply-mode",
	"--linear-scale":    "linear-scale", "--attn-scale": "attn-scale",
	"--rng": "rng", "--prediction": "prediction",
}

// SupportsSDFlag reports whether a raw CLI flag may be passed to an sd.cpp
// runtime with the given capability list.
func SupportsSDFlag(caps []string, help, flag string) bool {
	if removedInHelp(help, flag) {
		return false
	}
	if cap, known := sdKnownFlags[flag]; known {
		for _, c := range caps {
			if c == cap || c == strings.TrimPrefix(flag, "--") {
				return true
			}
		}
		if help != "" && strings.Contains(help, flag) {
			return true
		}
		return false
	}
	return runtimes.SupportsFlag(caps, help, flag)
}

func removedInHelp(help, flag string) bool {
	for _, line := range strings.Split(help, "\n") {
		idx := strings.Index(line, flag)
		if idx < 0 {
			continue
		}
		lower := strings.ToLower(line)
		if strings.Contains(lower, "has been removed") || strings.Contains(lower, "no longer supported") ||
			strings.Contains(lower, "no longer available") {
			return true
		}
	}
	return false
}

// BuildServerArgs assembles the sd-server argv for a model. host/port bind
// loopback only. port 0 means "patched later by EnsureServer".
//
// componentOnly selects --diffusion-model instead of --model: community
// GGUF dumps (ComfyUI-style, no KV metadata) are transformer-only files
// that sd.cpp cannot version-detect through --model, while --diffusion-model
// classifies them from the tensor layout and takes companions via --vae /
// --llm / --clip_*.
func BuildServerArgs(s LoadSettings, modelPath string, componentOnly bool, caps []string, help, host string, port int) ([]string, error) {
	if strings.TrimSpace(modelPath) == "" {
		return nil, fmt.Errorf("model path is required")
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model file missing: %w", err)
	}
	// Companion overrides are user-typed paths; a typo here must surface as
	// a clear error now, not as an sd-server crash after the process launches.
	type pathSetting struct {
		label string
		path  string
		dir   bool
	}
	for _, c := range []pathSetting{
		{"vae", s.VAE, false},
		{"taesd", s.TAESD, false},
		{"esrgan", s.ESRGAN, false},
		{"control_net", s.ControlNet, false},
		{"llm", s.LLM, false},
		{"llm_vision", s.LLMVision, false},
		{"t5xxl", s.T5XXL, false},
		{"clip_l", s.ClipL, false},
		{"clip_g", s.ClipG, false},
		{"clip_vision", s.ClipVision, false},
		{"tokenizer", s.Tokenizer, false},
		{"lora_model_dir", s.LoraModelDir, true},
	} {
		if c.path == "" {
			continue
		}
		st, err := os.Stat(c.path)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", c.label, err)
		}
		if c.dir && !st.IsDir() {
			return nil, fmt.Errorf("%s: %s is not a directory", c.label, c.path)
		}
		if !c.dir && st.IsDir() {
			return nil, fmt.Errorf("%s: %s is a directory, expected a file", c.label, c.path)
		}
	}
	modelFlag := "--model"
	if componentOnly {
		modelFlag = "--diffusion-model"
	}
	args := []string{modelFlag, modelPath}
	add := func(flag, value string) {
		if value == "" || !SupportsSDFlag(caps, help, flag) {
			return
		}
		args = append(args, flag, value)
	}
	addBool := func(flag string, on bool) {
		if !on || !SupportsSDFlag(caps, help, flag) {
			return
		}
		args = append(args, flag)
	}
	if s.VAE != "" {
		add("--vae", s.VAE)
	}
	if s.TAESD != "" {
		if SupportsSDFlag(caps, help, "--taesd") {
			args = append(args, "--taesd", s.TAESD)
		} else {
			add("--tae", s.TAESD)
		}
	}
	// Pipeline companions (text encoders / tokenizer). Required by
	// transformer-only checkpoints; harmless for full checkpoints that do
	// not advertise the flags.
	add("--llm", s.LLM)
	add("--llm_vision", s.LLMVision)
	add("--t5xxl", s.T5XXL)
	add("--clip_l", s.ClipL)
	add("--clip_g", s.ClipG)
	add("--clip_vision", s.ClipVision)
	add("--tokenizer", s.Tokenizer)
	if s.ESRGAN != "" {
		add("--upscale-model", s.ESRGAN)
	}
	if s.ControlNet != "" {
		add("--control-net", s.ControlNet)
	}
	add("--lora-model-dir", s.LoraModelDir)
	if s.Threads > 0 {
		add("--threads", strconv.Itoa(s.Threads))
	}
	if s.WeightType != "" {
		add("--type", strings.ToLower(s.WeightType))
	}
	add("--vae-format", s.VAEFormat)
	if s.Backend != "" {
		add("--backend", s.Backend)
	}
	if s.ParamsBackend != "" {
		add("--params-backend", s.ParamsBackend)
	}
	if s.MaxVRAM != "" {
		add("--max-vram", s.MaxVRAM)
	}
	switch strings.ToLower(strings.TrimSpace(s.AutoFit)) {
	case "on", "off":
		add("--auto-fit", strings.ToLower(strings.TrimSpace(s.AutoFit)))
	}
	addBool("--offload-to-cpu", s.OffloadCPU)
	addBool("--mmap", s.Mmap)
	addBool("--eager-load", s.EagerLoad)
	addBool("--diffusion-conv-direct", s.DiffusionConvDirect)
	addBool("--vae-conv-direct", s.VAEConvDirect)
	if s.FlashAttention {
		if SupportsSDFlag(caps, help, "--diffusion-fa") {
			args = append(args, "--diffusion-fa")
		} else {
			addBool("--fa", true)
		}
	}
	addBool("--vae-tiling", s.VAETiling)
	addBool("--temporal-tiling", s.TemporalTiling)
	// Network bind: loopback only (same invariant as llama-server).
	if SupportsSDFlag(caps, help, "--listen-ip") {
		args = append(args, "--listen-ip", host)
	}
	if SupportsSDFlag(caps, help, "--listen-port") {
		args = append(args, "--listen-port", strconv.Itoa(port))
	}
	return args, nil
}

// sdForbiddenRawFlags are flags raw_args may never carry — they are always
// managed by the launcher (loopback bind, the model file itself) or by the
// desktop shell (the served HTML path), never by user-typed overrides.
var sdForbiddenRawFlags = map[string]bool{
	"--listen-ip": true, "-l": true,
	"--listen-port": true,
	"--model":       true, "-m": true,
	"--diffusion-model": true,
	"--serve-html-path": true,
}

// parseSDRawArgs tokenizes LoadSettings.RawArgs without a shell (mirrors the
// llama.cpp raw-args tokenizer in internal/instances/args.go): whitespace
// split, reject the whole string on shell metacharacters, keep only
// "--flag [value]" pairs the runtime's --help advertises, and never allow
// overriding the managed network/model flags.
func parseSDRawArgs(raw string, caps []string, help string) (args []string, warnings []string) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	tokens := strings.Fields(raw)
	for _, tok := range tokens {
		if strings.ContainsAny(tok, ";&|<>$`\"'\\") {
			return nil, []string{"raw arguments contain unsafe characters and were rejected entirely"}
		}
	}
	prevFlagAccepted := false
	for _, tok := range tokens {
		if strings.HasPrefix(tok, "-") {
			if sdForbiddenRawFlags[tok] {
				warnings = append(warnings, fmt.Sprintf("raw flag %s may not override the managed network/model flags; skipped", tok))
				prevFlagAccepted = false
				continue
			}
			if !strings.HasPrefix(tok, "--") {
				warnings = append(warnings, fmt.Sprintf("raw token %q rejected (short flags are not accepted in raw arguments)", tok))
				prevFlagAccepted = false
				continue
			}
			if !SupportsSDFlag(caps, help, tok) {
				warnings = append(warnings, fmt.Sprintf("raw flag %s not supported by this runtime; skipped", tok))
				prevFlagAccepted = false
				continue
			}
			args = append(args, tok)
			prevFlagAccepted = true
			continue
		}
		if prevFlagAccepted {
			args = append(args, tok)
			prevFlagAccepted = false
			continue
		}
		warnings = append(warnings, fmt.Sprintf("raw token %q rejected (not a flag value)", tok))
	}
	return args, warnings
}

// PrepareLaunch resolves everything needed to run sd-server for a model:
// the sd-server binary (never sd-cli), local companion pairing, the
// full-checkpoint vs transformer-only --model/--diffusion-model choice,
// capability parsing and the final argv (host 127.0.0.1, port 0 — the
// caller patches the allocated port into "--listen-port"). startServer and
// previewDiffusionLoad both call this so the load dialog shows exactly what
// will run.
//
// help is the runtime's captured --help snapshot; it may be empty or stale
// (sd builds' recorded Capabilities come from the llama.cpp parser and are
// nearly empty for sd-server). When empty, sd-server --help is probed
// directly (argv only, bounded timeout) instead of launching with an
// unknown flag set: without an advertised flag list, --listen-ip/
// --listen-port would silently be dropped, sd-server would bind its
// hardcoded default port 1234, and the launcher would poll a different
// port until its full timeout every time.
func PrepareLaunch(rt *runtimes.Runtime, help, modelPath string, s LoadSettings) (
	exe string, args []string, resolved LoadSettings, resolvedHelp string, caps []string, warnings []string, err error) {
	resolved = s
	if rt == nil {
		return "", nil, resolved, help, nil, nil, fmt.Errorf("no diffusion runtime selected")
	}
	exe, err = SDServerPath(rt.ExecutablePath)
	if err != nil {
		return "", nil, resolved, help, nil, nil, err
	}

	// Pair local companions (…/vae/…, …/text_encoders/…) for transformer-only
	// checkpoints. Explicit settings always win.
	ApplyCompanions(&resolved, ModelRoot(modelPath), modelPath)

	resolvedHelp = help
	if strings.TrimSpace(resolvedHelp) == "" {
		probed, perr := runtimes.ProbeHelp(exe)
		if perr != nil || strings.TrimSpace(probed) == "" {
			return "", nil, resolved, "", nil, nil, fmt.Errorf(
				"could not read sd-server --help output (%v); refusing to guess which flags this build supports", perr)
		}
		resolvedHelp = probed
		warnings = append(warnings, "runtime capability snapshot was empty; probed sd-server --help directly")
	}
	caps = ParseSDCapabilities(resolvedHelp)

	componentOnly := IsComponentDiffusionFile(modelPath)
	args, err = BuildServerArgs(resolved, modelPath, componentOnly, caps, resolvedHelp, "127.0.0.1", 0)
	if err != nil {
		return "", nil, resolved, resolvedHelp, caps, warnings, err
	}
	rawArgs, rawWarnings := parseSDRawArgs(resolved.RawArgs, caps, resolvedHelp)
	args = append(args, rawArgs...)
	warnings = append(warnings, rawWarnings...)
	return exe, args, resolved, resolvedHelp, caps, warnings, nil
}

// SDRequestBody converts portable GenerateParams to the native sdcpp API
// schema (same shape sd_cpp_extra_args accepts).
func SDRequestBody(p GenerateParams) map[string]any {
	body := map[string]any{"prompt": p.Prompt}
	if p.NegativePrompt != "" {
		body["negative_prompt"] = p.NegativePrompt
	}
	if p.Width > 0 {
		body["width"] = p.Width
	}
	if p.Height > 0 {
		body["height"] = p.Height
	}
	if p.Steps > 0 {
		body["sample_params"] = map[string]any{"sample_steps": p.Steps}
	}
	if p.CFGScale != 0 {
		sp, _ := body["sample_params"].(map[string]any)
		if sp == nil {
			sp = map[string]any{}
			body["sample_params"] = sp
		}
		guid, _ := sp["guidance"].(map[string]any)
		if guid == nil {
			guid = map[string]any{}
			sp["guidance"] = guid
		}
		guid["txt_cfg"] = p.CFGScale
	}
	if p.Seed != 0 {
		body["seed"] = p.Seed
	}
	if p.Sampler != "" {
		sp, _ := body["sample_params"].(map[string]any)
		if sp == nil {
			sp = map[string]any{}
			body["sample_params"] = sp
		}
		sp["sample_method"] = strings.ToLower(p.Sampler)
	}
	if p.Scheduler != "" {
		sp, _ := body["sample_params"].(map[string]any)
		if sp == nil {
			sp = map[string]any{}
			body["sample_params"] = sp
		}
		sp["scheduler"] = strings.ToLower(p.Scheduler)
	}
	if p.Kind == KindVideo {
		if p.VideoFrames > 0 {
			body["video_frames"] = p.VideoFrames
		}
		if p.FPS > 0 {
			body["fps"] = p.FPS
		}
	} else {
		if p.BatchCount > 1 {
			body["batch_count"] = p.BatchCount
		}
		if p.InitImage != "" {
			body["init_image"] = p.InitImage
			if p.Strength > 0 {
				body["strength"] = p.Strength
			}
		}
	}
	if p.OutputFormat != "" {
		body["output_format"] = strings.ToLower(p.OutputFormat)
	}
	return body
}

// ValidateGenerateParams applies safe defaults and clamps.
func ValidateGenerateParams(p *GenerateParams) {
	if p.Width < 0 {
		p.Width = 0
	}
	if p.Height < 0 {
		p.Height = 0
	}
	if p.Width > 4096 {
		p.Width = 4096
	}
	if p.Height > 4096 {
		p.Height = 4096
	}
	if p.Steps < 0 {
		p.Steps = 0
	}
	if p.Steps > 300 {
		p.Steps = 300
	}
	if p.CFGScale < 0 {
		p.CFGScale = 0
	}
	if p.CFGScale > 30 {
		p.CFGScale = 30
	}
	if p.Strength < 0 {
		p.Strength = 0
	}
	if p.Strength > 1 {
		p.Strength = 1
	}
	if p.BatchCount < 0 {
		p.BatchCount = 0
	}
	if p.BatchCount > 8 {
		p.BatchCount = 8
	}
	if p.VideoFrames < 0 {
		p.VideoFrames = 0
	}
	if p.VideoFrames > 512 {
		p.VideoFrames = 512
	}
	if p.FPS < 0 {
		p.FPS = 0
	}
	if p.FPS > 60 {
		p.FPS = 60
	}
	if p.Seed == 0 {
		// sd.cpp treats any seed < 0 as random. 0 is never a request for the
		// literal seed 0 here — the client sends 0 when the user left the
		// field blank (portable "unset" for an int64 request field) — so it
		// is mapped to -1 (random) rather than passed through.
		p.Seed = -1
	}
	// Init images over the API are capped: base64 payloads ride inside the
	// 4 MiB control-API body limit (json.go maxRequestBody).
	if len(p.InitImage) > 3<<20 {
		p.InitImage = ""
	}
	_ = runtime.GOOS
}
