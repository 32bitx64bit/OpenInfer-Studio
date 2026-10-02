package huggingface

import (
	"path"
	"strconv"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/gguf"
)

// chatQuantPriority is the order a default model quantization is chosen in:
// the widely recommended 4-bit builds first, then the nearest neighbours.
var chatQuantPriority = []string{
	"q4_k_m", "ud-q4_k_xl", "oid-q4_k_xl", "q4_k_l", "q4_k_s", "iq4_xs", "q4_0",
	"q5_k_m", "q5_k_s", "q3_k_m", "q6_k", "q8_0",
}

// chatTargetBits aims the fallback default at a Q4_K_M-class build.
const chatTargetBits = 4.6

// buildChatPlan lays out a GGUF language-model repository: the model (one
// option per quantization or split set), an optional multimodal projector
// (one option per precision), and an optional speculative drafter.
func buildChatPlan(files []FileEntry, mods []string) Plan {
	plan := Plan{Kind: PlanChat}
	groups, projectors, drafts := GroupFiles(files)

	// GroupFiles turns a repo that holds only projectors or only drafters
	// into groups of their own; fold them back so they are components.
	var trunk []FileGroup
	for _, g := range groups {
		switch {
		case g.Draft:
			drafts = append(drafts, g.Files...)
		case g.Vision:
			projectors = append(projectors, g.Files...)
		default:
			trunk = append(trunk, g)
		}
	}

	var model *PlanComponent
	if len(trunk) > 0 {
		c := chatModelComponent(trunk)
		plan.Components = append(plan.Components, c)
		model = &plan.Components[len(plan.Components)-1]
	}
	first := len(plan.Components)
	if len(projectors) > 0 {
		plan.Components = append(plan.Components, chatProjectorComponent(projectors, mods))
	}
	if len(drafts) > 0 {
		plan.Components = append(plan.Components, chatDrafterComponent(drafts, model))
		if model == nil {
			plan.Components[len(plan.Components)-1].FollowPrecisionOf = ""
		}
	}
	// One repository holding several different models: a projector or drafter
	// belongs to one of them, and the library pairs a folder's projector with
	// every model in it. Nothing here can tell which file goes with which, so
	// leave them off until the user picks the ones named for their model.
	if names := distinctModels(trunk); len(names) > 1 && len(plan.Components) > first {
		for i := first; i < len(plan.Components); i++ {
			plan.Components[i].Selected = false
		}
		plan.Notes = append(plan.Notes, "This repository holds several models ("+strings.Join(names, ", ")+
			"). Projector and drafter files are not tied to one of them here: tick the ones named for the model you pick.")
	}
	if len(plan.Components) == 0 {
		plan.Notes = append(plan.Notes,
			"No GGUF files in this repository. For a safetensors language model, convert it to GGUF from the Quantize page. "+fileCensus(files))
	}
	return plan
}

func chatModelComponent(trunk []FileGroup) PlanComponent {
	c := PlanComponent{
		ID: "model", Role: "model", Label: "Model", Selected: true,
		Hint: "Pick a quantization. Smaller files use less memory and lose some quality.",
	}
	variants := map[string]bool{}
	for _, g := range trunk {
		variants[modelVariant(g.Files[0].Path)] = true
	}
	for _, g := range trunk {
		p := ggufPrecision(g.Quant)
		opt := PlanOption{
			ID: g.ID, Label: g.Label, Precision: p.ID, Bits: p.effectiveBits(), Class: p.Class,
			TotalBytes: g.TotalBytes, EstMemBytes: g.EstMemBytes,
		}
		if len(variants) > 1 {
			opt.Variant = modelVariant(g.Files[0].Path)
		}
		for _, f := range g.Files {
			opt.Files = append(opt.Files, PlanFile{Path: f.Path, Size: f.Size, Part: f.Part})
		}
		if g.Split {
			opt.Tags = append(opt.Tags, strconv.Itoa(g.Parts)+" parts")
		}
		switch {
		case strings.HasPrefix(g.Quant, "UD-"):
			opt.Tags = append(opt.Tags, "UD")
		case strings.HasPrefix(g.Quant, "OID-"):
			opt.Tags = append(opt.Tags, "OID")
		}
		c.Options = append(c.Options, opt)
	}
	// GroupFiles already ordered the groups by quantization rank, which is
	// the order people expect from a quant list; keep it.
	uniqueOptionIDs(c.Options)
	setDefault(&c, recommendChatQuant(c.Options))
	return c
}

// recommendChatQuant picks the default model option: a plain (non-MTP)
// build of the first quantization in chatQuantPriority the repo has, else
// the build nearest to Q4_K_M.
func recommendChatQuant(opts []PlanOption) int {
	plain := make([]int, 0, len(opts))
	for i, o := range opts {
		if !strings.Contains(o.Label, "MTP") {
			plain = append(plain, i)
		}
	}
	if len(plain) == 0 {
		for i := range opts {
			plain = append(plain, i)
		}
	}
	for _, want := range chatQuantPriority {
		for _, i := range plain {
			if opts[i].Precision == want {
				return i
			}
		}
	}
	sub := make([]PlanOption, len(plain))
	for n, i := range plain {
		sub[n] = opts[i]
	}
	return plain[pickByBits(sub, chatTargetBits)]
}

func chatProjectorComponent(projectors []GroupedFile, mods []string) PlanComponent {
	vision, audio := hasModality(mods, "vision"), hasModality(mods, "audio")
	c := PlanComponent{ID: "projector", Role: "projector", Short: "mmproj", Selected: true}
	switch {
	case vision && audio:
		c.Label = "Multimodal projector (vision + audio)"
		c.Hint = "Lets the model read images and, experimentally, hear audio."
	case audio:
		c.Label = "Audio projector"
		c.Hint = "Lets the model hear audio. Audio input is experimental."
		c.Experimental = true
	case vision:
		c.Label = "Vision projector"
		c.Hint = "Lets the model read images."
	default:
		c.Label = "Multimodal projector"
		c.Hint = "Lets the model read images or other media."
	}
	c.Hint += " llama.cpp loads one projector per model, so the options are alternative precisions."

	stems := map[string]bool{}
	for _, p := range projectors {
		stems[modelVariant(p.Path)] = true
	}
	for _, p := range projectors {
		pr := PrecisionOf(p.Path)
		opt := PlanOption{
			ID: "projector-" + safeGroupID(p.Path), Label: pr.Label, Precision: pr.ID,
			Bits: pr.effectiveBits(), Class: pr.Class, TotalBytes: p.Size,
			Files: []PlanFile{{Path: p.Path, Size: p.Size}},
		}
		if opt.Label == "" {
			opt.Label = "Default"
		}
		if len(stems) > 1 {
			opt.Variant = modelVariant(p.Path)
		}
		c.Options = append(c.Options, opt)
	}
	sortOptions(c.Options)
	uniqueOptionIDs(c.Options)
	// Projectors are small and quality-sensitive: aim at 16-bit.
	setDefault(&c, pickByBits(c.Options, 16))
	return c
}

func chatDrafterComponent(drafts []GroupedFile, model *PlanComponent) PlanComponent {
	c := PlanComponent{
		ID: "drafter", Role: "draft", Label: "Speculative drafter", Short: "drafter", Selected: true,
		Hint:              "Optional. Speeds up generation with identical output, at the cost of extra VRAM.",
		FollowPrecisionOf: "model",
	}
	draftVariants := map[string]bool{}
	for _, d := range drafts {
		draftVariants[modelVariant(d.Path)] = true
	}
	for _, d := range drafts {
		pr := ggufPrecision(d.Quant)
		mtp := ""
		if d.SpecType == string(gguf.SpecMTP) {
			mtp = "mtp-draft"
		}
		label := quantLabel(d.Quant, mtp, d.SpecType, d.Path)
		opt := PlanOption{
			ID: "drafter-" + safeGroupID(d.Path), Label: label, Precision: pr.ID,
			Bits: pr.effectiveBits(), Class: pr.Class, Kind: d.SpecType, TotalBytes: d.Size,
			Files: []PlanFile{{Path: d.Path, Size: d.Size}},
		}
		if len(draftVariants) > 1 {
			opt.Variant = modelVariant(d.Path)
		}
		c.Options = append(c.Options, opt)
	}
	sortOptions(c.Options)
	uniqueOptionIDs(c.Options)

	idx := -1
	if model != nil {
		if mi := optionIndex(model.Options, model.Default); mi >= 0 {
			// Same precision as the model's default, else the nearest to it.
			for i, o := range c.Options {
				if o.Precision != "" && o.Precision == model.Options[mi].Precision {
					idx = i
					break
				}
			}
			if idx < 0 {
				idx = pickByBits(c.Options, model.Options[mi].Bits)
			}
		}
	}
	if idx < 0 {
		idx = pickByBits(c.Options, chatTargetBits)
	}
	setDefault(&c, idx)
	return c
}

// modelVariant names which model a GGUF belongs to when a repository holds
// several: its file name without precision, with the folder in front when the
// folder names the model (4B/, 12B/) rather than a quantization.
func modelVariant(p string) string {
	stem := stemSansPrecision(p)
	if dir := path.Dir(p); dir != "." {
		last := path.Base(dir)
		bare := strings.TrimPrefix(strings.TrimPrefix(strings.ToUpper(last), "UD-"), "OID-")
		if _, isQuant := quantRanks[bare]; !isQuant {
			stem = last + "/" + stem
		}
	}
	return stem
}

// buildTags are name tokens that mark a build of one model (MTP, AMD tuning)
// rather than a different model.
var buildTags = map[string]bool{
	"mtp": true, "amd": true, "low": true, "high": true, "ultra": true,
	"fast": true, "sparse": true, "dense": true,
}

// distinctModels lists the different models among trunk groups, ignoring
// build tags and precisions. One name means a single model.
func distinctModels(trunk []FileGroup) []string {
	seen := map[string]bool{}
	var names []string
	for _, g := range trunk {
		name := dropTokens(modelVariant(g.Files[0].Path), func(l string) bool { return buildTags[l] })
		key := strings.ToLower(name)
		if !seen[key] {
			seen[key] = true
			names = append(names, name)
		}
	}
	return names
}
