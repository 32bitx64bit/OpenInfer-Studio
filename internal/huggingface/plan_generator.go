package huggingface

import (
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/sdmodel"
)

// Generator component roles (PlanComponent.Role).
const (
	genModel       = "model"
	genVAE         = "vae"
	genTAESD       = "taesd"
	genT5          = "t5xxl"
	genClipL       = "clip_l"
	genClipG       = "clip_g"
	genLLM         = "llm"
	genTextEncoder = "text_encoder" // diffusers text_encoder*/ folder
	genClipVision  = "clip_vision"
	genTokenizer   = "tokenizer"
	genControlNet  = "controlnet"
	genUpscaler    = "esrgan"
	genLoRA        = "lora"
)

var genLabels = map[string]string{
	genModel:       "Diffusion model",
	genVAE:         "VAE",
	genTAESD:       "Tiny VAE (TAESD)",
	genT5:          "T5-XXL text encoder",
	genClipL:       "CLIP-L text encoder",
	genClipG:       "CLIP-G text encoder",
	genLLM:         "LLM text encoder",
	genTextEncoder: "Text encoder",
	genClipVision:  "CLIP vision encoder",
	genTokenizer:   "Tokenizer",
	genControlNet:  "ControlNet",
	genUpscaler:    "Upscaler (ESRGAN)",
	genLoRA:        "LoRA",
}

var genHints = map[string]string{
	genModel:      "The diffusion weights. Pick a precision: smaller files need less memory.",
	genVAE:        "Decodes the image. Small; keep it at 16-bit.",
	genTAESD:      "Optional fast preview decoder.",
	genT5:         "Text encoder for FLUX / SD3 / Wan style models.",
	genClipL:      "Text encoder used alongside the main model.",
	genClipG:      "Second text encoder for SDXL / SD3 style models.",
	genLLM:        "Language-model text encoder for Qwen-Image / Z-Image style models.",
	genClipVision: "Needed for image-to-video models.",
	genTokenizer:  "Tokenizer for the language-model text encoder.",
	genControlNet: "Optional. Steers generation from an extra image.",
	genUpscaler:   "Optional. Upscales finished images.",
	genLoRA:       "Optional. A style or subject adapter.",
}

var genShort = map[string]string{
	genVAE: "VAE", genTAESD: "TAESD", genT5: "T5", genClipL: "CLIP-L", genClipG: "CLIP-G",
	genLLM: "LLM encoder", genTextEncoder: "text encoder", genClipVision: "CLIP vision",
	genTokenizer: "tokenizer", genControlNet: "ControlNet", genUpscaler: "upscaler", genLoRA: "LoRA",
}

// genOrder is the display order of component roles.
var genOrder = map[string]int{
	genModel: 0, genVAE: 1, genTAESD: 2, genT5: 3, genClipL: 4, genClipG: 5, genLLM: 6,
	genTextEncoder: 7, genClipVision: 8, genTokenizer: 9, genControlNet: 10, genUpscaler: 11, genLoRA: 12,
}

// genTargetBits is what the default option of a role aims at: the big
// weights (diffusion model, large text encoders) at 8-bit, everything small
// at 16-bit.
func genTargetBits(role string) float64 {
	switch role {
	case genModel, genT5, genLLM, genTextEncoder:
		return 8
	}
	return 16
}

// genSelectedByDefault says whether a role is part of a normal download.
func genSelectedByDefault(role string) bool {
	switch role {
	case genTAESD, genControlNet, genUpscaler, genLoRA, genClipVision:
		return false
	}
	return true
}

// diffusersDirs are the component folders of a diffusers repository.
var diffusersDirRe = regexp.MustCompile(`^(unet|transformer(_\d+)?|prior|vae|text_encoder(_\d+)?|image_encoder|controlnet)$`)

var shardPartRe = regexp.MustCompile(`(?i)-(\d{5})-of-(\d{5})\.(safetensors|gguf)$`)

// genFile is one classified weight file of a generator repository.
type genFile struct {
	FileEntry
	role    string // gen* role
	key     string // component id
	label   string // component label
	diffDir string // diffusers component folder, "" for flat layouts
	prec    Precision
	stem    string // stemSansPrecision
	pickle  bool   // .bin / .ckpt / .pt / .pth
}

// buildGeneratorPlan lays out an image / video generator repository as
// components with one option per stored precision, instead of one download
// that carries every weight file in the repo.
func buildGeneratorPlan(files []FileEntry) Plan {
	plan := Plan{Kind: PlanGenerator}

	hasIndex := false
	for _, f := range files {
		if strings.EqualFold(path.Base(f.Path), "model_index.json") {
			hasIndex = true
			break
		}
	}

	var gfs []genFile
	for _, f := range files {
		if g, ok := classifyGeneratorFile(f, hasIndex); ok {
			gfs = append(gfs, g)
		}
	}

	byKey := map[string][]genFile{}
	var keys []string
	for _, g := range gfs {
		if _, ok := byKey[g.key]; !ok {
			keys = append(keys, g.key)
		}
		byKey[g.key] = append(byKey[g.key], g)
	}

	hasFlatModel := len(byKey[genModel]) > 0
	hasGGUFModel := false
	for _, g := range byKey[genModel] {
		if strings.HasSuffix(strings.ToLower(g.Path), ".gguf") {
			hasGGUFModel = true
		}
	}
	// An image-to-video-only repository needs the CLIP vision encoder; one
	// that also holds text-to-video models leaves the choice to the user.
	i2v := len(byKey[genModel]) > 0
	for _, g := range byKey[genModel] {
		if !looksImageToVideo(g.Path) {
			i2v = false
		}
	}

	var comps []PlanComponent
	for _, key := range keys {
		fs := byKey[key]
		c := generatorComponent(fs)
		if len(c.Options) == 0 {
			continue
		}
		c.Selected = genSelectedByDefault(c.Role)
		if c.Role == genClipVision && i2v {
			c.Selected = true
		}
		// The folders of a diffusers layout are a second way to get the same
		// parts. When single-file weights exist they are the ones to take.
		if fs[0].diffDir != "" && hasFlatModel {
			c.Selected = false
		}
		comps = append(comps, c)
	}
	sort.SliceStable(comps, func(i, j int) bool {
		if genOrder[comps[i].Role] != genOrder[comps[j].Role] {
			return genOrder[comps[i].Role] < genOrder[comps[j].Role]
		}
		// Single-file components before diffusers folders of the same role.
		di, dj := strings.HasPrefix(comps[i].ID, "d_"), strings.HasPrefix(comps[j].ID, "d_")
		if di != dj {
			return !di
		}
		return comps[i].ID < comps[j].ID
	})
	plan.Components = comps

	if len(comps) == 0 {
		plan.Notes = append(plan.Notes, "No loadable weight files found in this repository.")
		return plan
	}

	hasRole := func(roles ...string) bool {
		for _, c := range comps {
			for _, r := range roles {
				if c.Role == r {
					return true
				}
			}
		}
		return false
	}
	if hasGGUFModel && !hasRole(genVAE, genT5, genClipL, genClipG, genLLM, genTextEncoder) {
		plan.Notes = append(plan.Notes,
			"This repository only has the diffusion model. Most GGUF diffusion models also need a VAE and text encoder(s) from the original release, which are not included here.")
	}
	for _, c := range comps {
		if strings.HasPrefix(c.ID, "d_") {
			plan.Notes = append(plan.Notes,
				"This repository also ships a diffusers folder layout (unet/, transformer/, vae/, text_encoder/ …). Single weight files are the reliable way to load a model in Studio; the folder components may not load.")
			break
		}
	}
	return plan
}

// classifyGeneratorFile decides what role a repository file plays. Files
// that are not weights (and diffusers folders that hold none we use) are
// dropped.
func classifyGeneratorFile(f FileEntry, hasIndex bool) (genFile, bool) {
	lower := strings.ToLower(f.Path)
	base := path.Base(lower)
	isTokenizer := base == "tokenizer.json"
	if !isDiffusionWeight(f.Path) && !isTokenizer {
		return genFile{}, false
	}
	g := genFile{FileEntry: f, prec: PrecisionOf(f.Path), stem: stemSansPrecision(f.Path)}
	for _, ext := range []string{".bin", ".ckpt", ".pt", ".pth"} {
		if strings.HasSuffix(lower, ext) {
			g.pickle = true
		}
	}

	if hasIndex {
		if dir, _, ok := strings.Cut(f.Path, "/"); ok {
			dl := strings.ToLower(dir)
			switch {
			case diffusersDirRe.MatchString(dl):
				g.diffDir = dl
				g.key = "d_" + dl
				switch {
				case dl == "vae":
					g.role, g.label = genVAE, "VAE (diffusers folder)"
				case strings.HasPrefix(dl, "text_encoder"):
					g.role = genTextEncoder
					g.label = "Text encoder (" + dl + "/)"
				case dl == "image_encoder":
					g.role, g.label = genClipVision, "CLIP vision encoder (image_encoder/)"
				case dl == "controlnet":
					g.role, g.label = genControlNet, "ControlNet (controlnet/)"
				default:
					g.role, g.label = genModel, "Diffusion model ("+dl+"/)"
				}
				return g, true
			case dl == "scheduler", dl == "safety_checker", dl == "feature_extractor",
				strings.HasPrefix(dl, "tokenizer"):
				return genFile{}, false
			}
		}
	}

	role := ""
	switch sdmodel.ComponentRole(f.Path, nil) {
	case sdmodel.RoleVAE:
		role = genVAE
	case sdmodel.RoleTAESD:
		role = genTAESD
	case sdmodel.RoleLLM:
		role = genLLM
	case sdmodel.RoleT5XXL:
		role = genT5
	case sdmodel.RoleClipL:
		role = genClipL
	case sdmodel.RoleClipG:
		role = genClipG
	case sdmodel.RoleClipVision:
		role = genClipVision
	case sdmodel.RoleTokenizer:
		role = genTokenizer
	case sdmodel.RoleESRGAN:
		role = genUpscaler
	case sdmodel.RoleControlNet:
		role = genControlNet
	case sdmodel.RoleLoRA:
		role = genLoRA
	default:
		role = genModel
	}
	if isTokenizer {
		role = genTokenizer
	}
	g.role, g.key, g.label = role, role, genLabels[role]
	if role == genT5 && strings.Contains(base, "umt5") {
		g.label = "UMT5-XXL text encoder"
	}
	return g, true
}

// generatorComponent builds one component from its files.
func generatorComponent(fs []genFile) PlanComponent {
	first := fs[0]
	c := PlanComponent{
		ID: first.key, Role: first.role, Label: first.label, Hint: genHints[first.role],
		Short: genShort[first.role],
	}
	if c.Short != "" && first.diffDir != "" {
		c.Short += " folder"
	}

	// Pickle formats (.bin, .ckpt, .pt, .pth) are legacy duplicates of the
	// safetensors / GGUF weights; offer them only when nothing else is.
	hasSafe := false
	for _, g := range fs {
		if !g.pickle {
			hasSafe = true
		}
	}
	var keep []genFile
	for _, g := range fs {
		if hasSafe && g.pickle {
			continue
		}
		keep = append(keep, g)
	}

	// Shards of one weight set are a single option.
	type set struct {
		files []genFile
	}
	sets := map[string]*set{}
	var order []string
	for _, g := range keep {
		key := g.Path
		if m := shardPartRe.FindStringSubmatch(g.Path); m != nil {
			key = shardPartRe.ReplaceAllString(g.Path, "") + "|" + m[2]
		}
		if sets[key] == nil {
			sets[key] = &set{}
			order = append(order, key)
		}
		sets[key].files = append(sets[key].files, g)
	}

	stems := map[string]bool{}
	for _, g := range keep {
		stems[g.stem] = true
	}
	seen := map[string]bool{}
	for _, key := range order {
		s := sets[key]
		sort.Slice(s.files, func(i, j int) bool { return s.files[i].Path < s.files[j].Path })
		head := s.files[0]
		opt := PlanOption{
			// Option ids double as the download folder name: the file's own
			// name is enough (uniqueOptionIDs separates equal ones).
			ID:        first.role + "-" + strings.ToLower(safeGroupID(path.Base(head.Path))),
			Label:     head.prec.Label,
			Precision: head.prec.ID,
			Bits:      head.prec.effectiveBits(),
			Class:     head.prec.Class,
		}
		if first.diffDir != "" {
			opt.ID = first.key + "-" + strings.ToLower(safeGroupID(path.Base(head.Path)))
		}
		if opt.Label == "" {
			opt.Label = "Default"
		}
		if len(stems) > 1 {
			opt.Variant = head.stem
		}
		for _, g := range s.files {
			part := 0
			if m := shardPartRe.FindStringSubmatch(g.Path); m != nil {
				part, _ = strconv.Atoi(m[1])
			}
			pf := PlanFile{Path: g.Path, Size: g.Size, Part: part}
			if g.diffDir != "" {
				pf.Dest = g.Path
			}
			opt.Files = append(opt.Files, pf)
			opt.TotalBytes += g.Size
		}
		if len(s.files) > 1 {
			opt.Tags = append(opt.Tags, strconv.Itoa(len(s.files))+" parts")
		}
		if strings.HasSuffix(strings.ToLower(head.Path), ".gguf") {
			opt.Tags = append(opt.Tags, "GGUF")
		}
		opt.Warn = sdLoadWarning(head.prec, len(s.files))

		// The same weights listed twice (a mirrored folder) are one option.
		dup := opt.Variant + "|" + opt.Precision + "|" + strconv.FormatInt(opt.TotalBytes, 10)
		if seen[dup] {
			continue
		}
		seen[dup] = true
		c.Options = append(c.Options, opt)
	}
	sortOptions(c.Options)
	uniqueOptionIDs(c.Options)
	setDefault(&c, pickByBits(c.Options, genTargetBits(first.role)))
	return c
}

// sdLoadWarning explains why a precision is unlikely to load in
// stable-diffusion.cpp, or "" when it should. GGUF and plain float files
// load; packs that need per-tensor scales, integer packs and 4-bit formats
// outside GGUF are the ones its loader has been seen to reject.
func sdLoadWarning(p Precision, parts int) string {
	switch {
	case p.Class == Class4Bit:
		return "4-bit packs (NF4 / FP4 / SVDQuant) are not supported by stable-diffusion.cpp. A GGUF quantization is the 4-bit option that loads."
	case p.Class == ClassInt8:
		return "INT8 packs usually fail to load in stable-diffusion.cpp. A GGUF Q8_0 is the same size and loads."
	case p.Packed:
		return "Scaled FP8 needs per-tensor scales that stable-diffusion.cpp may not apply: it can fail to load or give poor output. Plain FP8, BF16/FP16 or GGUF are safer."
	case parts > 1:
		return "Split into " + strconv.Itoa(parts) + " files. stable-diffusion.cpp loads one weight file per component, so a multi-file set may not load."
	}
	return ""
}

// looksImageToVideo reports whether a file name marks an image-to-video
// model (which needs a CLIP vision encoder).
func looksImageToVideo(p string) bool {
	l := strings.ToLower(path.Base(p))
	for _, h := range []string{"i2v", "flf2v", "image-to-video", "image_to_video", "fun_inp", "fun-inp"} {
		if strings.Contains(l, h) {
			return true
		}
	}
	return false
}
