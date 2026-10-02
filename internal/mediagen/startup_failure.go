package mediagen

import (
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
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
	missing := parseMissingTensors(logText)
	shapes := parseShapeMismatches(logText)
	packed := packedFiles(modelPath, s)
	switch {
	case len(missing) > 0 || len(shapes) > 0:
		b.WriteString(": the files it was given do not match the model.\n")
		if len(missing) > 0 {
			explainMissing(&b, missing, modelPath, s)
		}
		if len(shapes) > 0 {
			explainShapes(&b, shapes, modelPath, s)
		}
	case len(packed) > 0 && looksLikeLoaderAbort(logText):
		b.WriteString(": it aborted while loading weights stored in a format stable-diffusion.cpp cannot read.\n")
		for _, p := range packed {
			fmt.Fprintf(&b, "  - %s %s: %s\n", p.Role, filepath.Base(p.Path), p.Why)
		}
		b.WriteString("These packs are meant for ComfyUI. Use a GGUF, BF16/FP16 or plain FP8 build of the model, VAE and text encoders.\n")
	default:
		b.WriteString(":\n")
	}
	tail := cleanLogTail(withoutShapeLines(withoutMissingTensorLines(logText)), 12)
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

// missingTensorRe matches the validation errors stable-diffusion.cpp prints
// when a file lacks tensors the model needs:
// "[ERROR ] model_manager.cpp:763 - VAE tensor 'first_stage_model.…' not in model metadata".
var missingTensorRe = regexp.MustCompile(`^\[ERROR\s*\]\s+\S+:\d+\s+-\s+(.+?) tensor '([^']+)' not in model metadata`)

// missingGroup is the tensors of one component the model expected and did
// not find.
type missingGroup struct {
	Kind  string // as sd.cpp names it: "VAE", "LLM", …
	Names []string
}

// parseMissingTensors groups sd.cpp's missing-tensor errors by component, in
// order of appearance, each name once.
func parseMissingTensors(text string) []missingGroup {
	var out []missingGroup
	idx := map[string]int{}
	seen := map[string]bool{}
	for _, line := range strings.Split(text, "\n") {
		m := missingTensorRe.FindStringSubmatch(strings.TrimSpace(line))
		if m == nil {
			continue
		}
		kind, name := m[1], m[2]
		if seen[kind+"\x00"+name] {
			continue
		}
		seen[kind+"\x00"+name] = true
		i, ok := idx[kind]
		if !ok {
			i = len(out)
			idx[kind] = i
			out = append(out, missingGroup{Kind: kind})
		}
		out[i].Names = append(out[i].Names, name)
	}
	return out
}

// withoutMissingTensorLines removes the missing-tensor error lines: they are
// summarized separately, and left in they push everything else out of the tail.
func withoutMissingTensorLines(text string) string {
	var kept []string
	for _, line := range strings.Split(text, "\n") {
		if !missingTensorRe.MatchString(strings.TrimSpace(line)) {
			kept = append(kept, line)
		}
	}
	return strings.Join(kept, "\n")
}

// fileForKind maps the component sd.cpp complained about to the file this
// launch passed for it ("" when it is part of the model file itself).
func fileForKind(kind, modelPath string, s LoadSettings) (role, path string) {
	k := strings.ToLower(kind)
	switch {
	case strings.Contains(k, "conditioner") || strings.Contains(k, "text encoder"):
		// sd.cpp's name for whichever text encoder the model uses.
		for _, c := range []struct{ role, path string }{
			{"LLM text encoder", s.LLM}, {"T5 text encoder", s.T5XXL},
			{"CLIP-L text encoder", s.ClipL}, {"CLIP-G text encoder", s.ClipG},
		} {
			if c.path != "" {
				return c.role, c.path
			}
		}
		return "text encoder", ""
	case strings.Contains(k, "vae"):
		return "VAE", s.VAE
	case strings.Contains(k, "llm"):
		return "LLM text encoder", s.LLM
	case strings.Contains(k, "t5"):
		return "T5 text encoder", s.T5XXL
	case strings.Contains(k, "clip") && (strings.Contains(k, "_g") || strings.Contains(k, "-g") || strings.Contains(k, " g")):
		return "CLIP-G text encoder", s.ClipG
	case strings.Contains(k, "clip"):
		return "CLIP-L text encoder", s.ClipL
	}
	return "diffusion model", modelPath
}

// tensorGroups summarizes a weight file's tensor names by their first path
// segment ("decoder x90, post_quant_conv x2"), the way the model's VAE or
// encoder is laid out, most numerous first. It also reports the total.
func tensorGroups(path string) (groups []string, total int, keys map[string]int, ok bool) {
	names, err := sdmodel.TensorNames(path)
	if err != nil || len(names) == 0 {
		return nil, 0, nil, false
	}
	keys = map[string]int{}
	for _, n := range names {
		n = strings.TrimPrefix(strings.TrimPrefix(n, "first_stage_model."), "vae.")
		key, _, _ := strings.Cut(n, ".")
		keys[key]++
	}
	type kv struct {
		k string
		n int
	}
	var list []kv
	for k, n := range keys {
		list = append(list, kv{k, n})
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].n != list[j].n {
			return list[i].n > list[j].n
		}
		return list[i].k < list[j].k
	})
	for i, e := range list {
		if i == 6 {
			groups = append(groups, "…")
			break
		}
		groups = append(groups, fmt.Sprintf("%s ×%d", e.k, e.n))
	}
	return groups, len(names), keys, true
}

// explainMissing writes, for each component with missing tensors, which file
// it was checked against, what that file lacks and what it holds instead.
func explainMissing(b *strings.Builder, missing []missingGroup, modelPath string, s LoadSettings) {
	for _, g := range missing {
		role, path := fileForKind(g.Kind, modelPath, s)
		eg := g.Names
		if len(eg) > 3 {
			eg = []string{g.Names[0], g.Names[1], g.Names[len(g.Names)-1]}
		}
		if path == "" {
			fmt.Fprintf(b, "  - %s: %d tensors the model needs are missing (no separate file was given, so they should be inside the model file)\n", role, len(g.Names))
			fmt.Fprintf(b, "    e.g. %s\n", strings.Join(eg, ", "))
			continue
		}
		fmt.Fprintf(b, "  - %s %s: %d tensors the model needs are missing\n", role, filepath.Base(path), len(g.Names))
		fmt.Fprintf(b, "    e.g. %s\n", strings.Join(eg, ", "))
		groups, total, keys, ok := tensorGroups(path)
		if ok {
			fmt.Fprintf(b, "    that file holds %d tensors: %s\n", total, strings.Join(groups, ", "))
		}
		if why := sdmodel.PackedReason(path); why != "" {
			fmt.Fprintf(b, "    it is stored as %s, a ComfyUI-only pack stable-diffusion.cpp cannot map to the names it expects\n", why)
		}
		if strings.HasPrefix(role, "VAE") && sdmodel.ComponentRole(path, nil) == sdmodel.RoleAudioVAE {
			b.WriteString("    that is an audio VAE: --vae needs the video (or image) VAE published with the model\n")
		} else if strings.HasPrefix(role, "VAE") && ok {
			if keys["encoder"] == 0 && keys["decoder"] > 0 {
				b.WriteString("    it has a decoder but no encoder: this model also needs the encoder (image-to-video and first/last-frame modes encode their input frames)\n")
			} else {
				b.WriteString("    it is a different VAE from the one this model was built with, or uses a tensor layout stable-diffusion.cpp does not map\n")
			}
		}
	}
	b.WriteString("Use the file published with this model for each part above, in BF16/FP16, plain FP8 or GGUF.\n")
}

// wrongShapeRe matches the validation errors for a tensor that exists but has
// other dimensions than the runtime builds the model with:
// "Diffusion model tensor 'blocks.0.attn.qkv_proj.weight' has wrong shape in
// model metadata: got [5376, 21504, 1, 1], expected [1, 21504, 1, 1]".
var wrongShapeRe = regexp.MustCompile(`^\[ERROR\s*\]\s+\S+:\d+\s+-\s+(.+?) tensor '([^']+)' has wrong shape in model metadata: got (\[[^\]]*\]), expected (\[[^\]]*\])`)

// shapeMismatch is one tensor with unexpected dimensions.
type shapeMismatch struct{ Name, Got, Want string }

// shapeGroup is the shape mismatches of one component.
type shapeGroup struct {
	Kind  string
	Items []shapeMismatch
}

func parseShapeMismatches(text string) []shapeGroup {
	var out []shapeGroup
	idx := map[string]int{}
	for _, line := range strings.Split(text, "\n") {
		m := wrongShapeRe.FindStringSubmatch(strings.TrimSpace(line))
		if m == nil {
			continue
		}
		i, ok := idx[m[1]]
		if !ok {
			i = len(out)
			idx[m[1]] = i
			out = append(out, shapeGroup{Kind: m[1]})
		}
		out[i].Items = append(out[i].Items, shapeMismatch{Name: m[2], Got: m[3], Want: m[4]})
	}
	return out
}

func withoutShapeLines(text string) string {
	var kept []string
	for _, line := range strings.Split(text, "\n") {
		if !wrongShapeRe.MatchString(strings.TrimSpace(line)) {
			kept = append(kept, line)
		}
	}
	return strings.Join(kept, "\n")
}

// explainShapes writes, per component, how many tensors have other shapes
// than the runtime expects, with examples. When every one is in a vision
// tower (visual.*) it says so: the text encoder file carries a vision tower
// this runtime reads differently from how the file stores it.
func explainShapes(b *strings.Builder, groups []shapeGroup, modelPath string, s LoadSettings) {
	for _, g := range groups {
		role, path := fileForKind(g.Kind, modelPath, s)
		name := role
		if path != "" {
			name += " " + filepath.Base(path)
		}
		fmt.Fprintf(b, "  - %s: %d tensors have other shapes than this stable-diffusion.cpp build expects\n", name, len(g.Items))
		show := g.Items
		if len(show) > 2 {
			show = []shapeMismatch{g.Items[0], g.Items[len(g.Items)-1]}
		}
		for _, it := range show {
			fmt.Fprintf(b, "    %s: file has %s, runtime expects %s\n", it.Name, it.Got, it.Want)
		}
		vision := true
		for _, it := range g.Items {
			if !strings.Contains(it.Name, ".visual.") {
				vision = false
				break
			}
		}
		if vision {
			b.WriteString("    only the vision tower (visual.*) differs: the file stores it in another layout (a k-quantized, reshaped one) than this runtime reads. A text-only or Q8_0/F16 build of this encoder avoids it\n")
		}
	}
	b.WriteString("The files were made for a different stable-diffusion.cpp revision or layout than the runtime in use. Check the model card for the build it needs; a newer runtime from the Runtimes page may know the layout.\n")
}
