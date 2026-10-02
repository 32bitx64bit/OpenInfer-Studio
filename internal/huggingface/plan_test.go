package huggingface

import (
	"strings"
	"testing"
)

func findComp(t *testing.T, p Plan, id string) PlanComponent {
	t.Helper()
	for _, c := range p.Components {
		if c.ID == id {
			return c
		}
	}
	var ids []string
	for _, c := range p.Components {
		ids = append(ids, c.ID)
	}
	t.Fatalf("component %q not in plan (have %v)", id, ids)
	return PlanComponent{}
}

func hasComp(p Plan, id string) bool {
	for _, c := range p.Components {
		if c.ID == id {
			return true
		}
	}
	return false
}

func optByLabel(t *testing.T, c PlanComponent, label string) PlanOption {
	t.Helper()
	for _, o := range c.Options {
		if o.Label == label {
			return o
		}
	}
	t.Fatalf("component %s has no option %q (have %v)", c.ID, label, optLabels(c))
	return PlanOption{}
}

func optLabels(c PlanComponent) []string {
	var out []string
	for _, o := range c.Options {
		out = append(out, o.Label)
	}
	return out
}

func defaultOpt(t *testing.T, c PlanComponent) PlanOption {
	t.Helper()
	i := optionIndex(c.Options, c.Default)
	if i < 0 {
		t.Fatalf("component %s default %q is not one of its options", c.ID, c.Default)
	}
	if !c.Options[i].Recommended {
		t.Errorf("component %s default option is not marked recommended", c.ID)
	}
	return c.Options[i]
}

func entries(pairs ...any) []FileEntry {
	var out []FileEntry
	for i := 0; i < len(pairs); i += 2 {
		out = append(out, FileEntry{Path: pairs[i].(string), Size: int64(pairs[i+1].(int))})
	}
	return out
}

// ---- chat -----------------------------------------------------------------

func TestChatPlanModelProjectorDefaults(t *testing.T) {
	p := buildChatPlan(entries(
		"Model-Q2_K.gguf", 2000, "Model-Q4_K_M.gguf", 4000, "Model-Q4_K_S.gguf", 3800,
		"Model-Q5_K_M.gguf", 5000, "Model-Q8_0.gguf", 8000, "Model-F16.gguf", 15000,
		"mmproj-Q8_0.gguf", 300, "mmproj-F16.gguf", 600, "mmproj-BF16.gguf", 600,
		"README.md", 10,
	), []string{"vision"})
	if p.Kind != PlanChat {
		t.Fatalf("kind = %s", p.Kind)
	}
	model := findComp(t, p, "model")
	if !model.Selected {
		t.Error("model must be included by default")
	}
	if got := defaultOpt(t, model); got.Precision != "q4_k_m" {
		t.Errorf("default model quant = %s, want q4_k_m", got.Precision)
	}
	if len(model.Options) != 6 {
		t.Errorf("model options = %v", optLabels(model))
	}

	proj := findComp(t, p, "projector")
	if !proj.Selected || proj.Experimental {
		t.Errorf("vision projector should be on by default and not experimental: %+v", proj)
	}
	if proj.Label != "Vision projector" {
		t.Errorf("projector label = %q", proj.Label)
	}
	// Three precisions are alternatives of one projector component.
	if len(proj.Options) != 3 {
		t.Fatalf("projector options = %v", optLabels(proj))
	}
	if got := defaultOpt(t, proj); got.Precision != "fp16" {
		t.Errorf("default projector = %s, want fp16 (the reference format, not bf16)", got.Precision)
	}
	if proj.Options[0].Precision != "q8_0" {
		t.Errorf("projector options should start at the smallest precision, got %v", optLabels(proj))
	}
}

func TestChatPlanDefaultQuantFallbacks(t *testing.T) {
	cases := []struct {
		name  string
		files []FileEntry
		want  string
	}{
		{"only full precision", entries("M-F16.gguf", 10), "fp16"},
		{"q8 beats f16 when no 4-bit", entries("M-Q8_0.gguf", 8, "M-F16.gguf", 16), "q8_0"},
		{"q5 when no q4", entries("M-Q5_K_M.gguf", 5, "M-Q8_0.gguf", 8, "M-Q3_K_M.gguf", 3), "q5_k_m"},
		{"unsloth dynamic when no plain q4", entries("M-UD-Q4_K_XL.gguf", 5, "M-Q8_0.gguf", 8), "ud-q4_k_xl"},
		{"nearest to q4 when nothing in the priority list", entries("M-IQ3_M.gguf", 3, "M-Q2_K_L.gguf", 2), "iq3_m"},
	}
	for _, c := range cases {
		p := buildChatPlan(c.files, nil)
		if got := defaultOpt(t, findComp(t, p, "model")); got.Precision != c.want {
			t.Errorf("%s: default = %s, want %s", c.name, got.Precision, c.want)
		}
	}
}

func TestChatPlanDefaultSkipsMTPBuilds(t *testing.T) {
	p := buildChatPlan(entries(
		"Qwen3.6-NEO-Q4_K_M.gguf", 4000, "Qwen3.6-NEO-MTP-Q4_K_M.gguf", 4100,
	), nil)
	m := findComp(t, p, "model")
	if len(m.Options) != 2 {
		t.Fatalf("options = %v", optLabels(m))
	}
	if got := defaultOpt(t, m); strings.Contains(got.Label, "MTP") {
		t.Errorf("default should be the plain build, got %q", got.Label)
	}
}

func TestChatPlanSplitSetIsOneOption(t *testing.T) {
	p := buildChatPlan(entries(
		"Big-Q4_K_M-00001-of-00003.gguf", 40, "Big-Q4_K_M-00002-of-00003.gguf", 40,
		"Big-Q4_K_M-00003-of-00003.gguf", 20, "Big-Q8_0.gguf", 150,
	), nil)
	m := findComp(t, p, "model")
	var set PlanOption
	for _, o := range m.Options {
		if len(o.Files) == 3 {
			set = o
		}
	}
	if len(set.Files) != 3 || set.TotalBytes != 100 {
		t.Fatalf("split set not folded into one option: %+v", m.Options)
	}
	hasParts := false
	for _, tag := range set.Tags {
		if tag == "3 parts" {
			hasParts = true
		}
	}
	if !hasParts {
		t.Errorf("split set should be tagged, got %v", set.Tags)
	}
}

func TestChatPlanAudioOnlyProjectorIsExperimental(t *testing.T) {
	files := entries("M-Q4_K_M.gguf", 4, "mmproj-ultravox-F16.gguf", 1)
	p := buildChatPlan(files, []string{"audio"})
	proj := findComp(t, p, "projector")
	if !proj.Experimental || proj.Label != "Audio projector" {
		t.Errorf("audio-only projector = %+v", proj)
	}
	both := findComp(t, buildChatPlan(files, []string{"vision", "audio"}), "projector")
	if both.Experimental || !strings.Contains(both.Label, "vision + audio") {
		t.Errorf("vision+audio projector = %+v", both)
	}
}

func TestChatPlanProjectorVariantsAreAlternatives(t *testing.T) {
	p := buildChatPlan(entries(
		"M-Q4_K_M.gguf", 4,
		"mmproj-vision-F16.gguf", 2, "mmproj-audio-F16.gguf", 1,
	), []string{"vision", "audio"})
	proj := findComp(t, p, "projector")
	if len(proj.Options) != 2 {
		t.Fatalf("options = %v", optLabels(proj))
	}
	for _, o := range proj.Options {
		if o.Variant == "" {
			t.Errorf("projectors from different builds need a variant to tell them apart: %+v", o)
		}
	}
}

func TestChatPlanDrafterFollowsModelPrecision(t *testing.T) {
	p := buildChatPlan(entries(
		"Muse-Q4_K_M.gguf", 4, "Muse-Q8_0.gguf", 8,
		"dflash-Muse-Q4_K_M.gguf", 1, "dflash-Muse-Q8_0.gguf", 2,
	), nil)
	d := findComp(t, p, "drafter")
	if d.FollowPrecisionOf != "model" || !d.Selected {
		t.Fatalf("drafter = %+v", d)
	}
	if got := defaultOpt(t, d); got.Precision != "q4_k_m" {
		t.Errorf("drafter default = %s, want the model's q4_k_m", got.Precision)
	}
	if got := defaultOpt(t, d); got.Kind == "" || !strings.Contains(got.Label, "DFlash") {
		t.Errorf("drafter option should name its speculative type: %+v", got)
	}
	// A drafter must never be offered as a model option.
	for _, o := range findComp(t, p, "model").Options {
		for _, f := range o.Files {
			if strings.Contains(f.Path, "dflash") {
				t.Errorf("drafter file leaked into model options: %s", f.Path)
			}
		}
	}
}

func TestChatPlanProjectorOnlyAndDrafterOnlyRepos(t *testing.T) {
	p := buildChatPlan(entries("mmproj-F16.gguf", 5, "mmproj-Q8_0.gguf", 3), []string{"vision"})
	if hasComp(p, "model") || len(findComp(t, p, "projector").Options) != 2 {
		t.Errorf("projector-only plan = %+v", p)
	}
	p = buildChatPlan(entries("dflash-Muse-Q4_K_M.gguf", 5), nil)
	d := findComp(t, p, "drafter")
	if hasComp(p, "model") || d.FollowPrecisionOf != "" || len(d.Options) != 1 {
		t.Errorf("drafter-only plan = %+v", p)
	}
}

func TestChatPlanNoGGUFExplains(t *testing.T) {
	p := buildChatPlan(entries("model.safetensors", 100, "config.json", 1), nil)
	if len(p.Components) != 0 || len(p.Notes) == 0 {
		t.Fatalf("plan = %+v", p)
	}
}

func TestChatPlanOptionIDsUniqueAndPathsKept(t *testing.T) {
	p := buildChatPlan(entries(
		"a/Model-Q4_K_M.gguf", 4, "b/Model-Q4_K_M.gguf", 4, "mmproj-F16.gguf", 1,
	), []string{"vision"})
	seen := map[string]bool{}
	for _, c := range p.Components {
		for _, o := range c.Options {
			if seen[c.ID+"/"+o.ID] {
				t.Errorf("duplicate option id %s in %s", o.ID, c.ID)
			}
			seen[c.ID+"/"+o.ID] = true
			if len(o.Files) == 0 {
				t.Errorf("option %s has no files", o.ID)
			}
		}
	}
}

// ---- generators -------------------------------------------------------------

func TestGeneratorPlanGGUFTransformerOnly(t *testing.T) {
	p := buildGeneratorPlan(entries(
		"flux1-dev-Q2_K.gguf", 4, "flux1-dev-Q4_0.gguf", 7, "flux1-dev-Q5_K_S.gguf", 8,
		"flux1-dev-Q8_0.gguf", 12, "flux1-dev-F16.gguf", 23, "README.md", 1,
	))
	if p.Kind != PlanGenerator || len(p.Components) != 1 {
		t.Fatalf("plan = %+v", p)
	}
	m := findComp(t, p, "model")
	if got := defaultOpt(t, m); got.Precision != "q8_0" {
		t.Errorf("default = %s, want q8_0 (the int8-class build)", got.Precision)
	}
	if m.Options[0].Precision != "q2_k" || m.Options[len(m.Options)-1].Precision != "fp16" {
		t.Errorf("options should run smallest to largest: %v", optLabels(m))
	}
	for _, o := range m.Options {
		if o.Warn != "" {
			t.Errorf("GGUF option %s should not warn: %s", o.Label, o.Warn)
		}
		if o.Variant != "" {
			t.Errorf("one model, no variant expected: %+v", o)
		}
	}
	// The repo has no VAE / text encoders: say so instead of leaving it out silently.
	if len(p.Notes) == 0 || !strings.Contains(p.Notes[0], "VAE") {
		t.Errorf("missing-companion note expected, got %v", p.Notes)
	}
}

func TestGeneratorPlanComfyRepackagedWan(t *testing.T) {
	p := buildGeneratorPlan(entries(
		"split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors", 28,
		"split_files/diffusion_models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors", 14,
		"split_files/diffusion_models/wan2.1_t2v_14B_fp8_scaled.safetensors", 14,
		"split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors", 3,
		"split_files/text_encoders/umt5_xxl_fp16.safetensors", 11,
		"split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6,
		"split_files/vae/wan_2.1_vae.safetensors", 1,
		"split_files/clip_vision/clip_vision_h.safetensors", 1,
	))
	m := findComp(t, p, "model")
	if len(m.Options) != 4 {
		t.Fatalf("model options = %v", optLabels(m))
	}
	variants := map[string]bool{}
	for _, o := range m.Options {
		variants[o.Variant] = true
	}
	if !variants["wan2.1_t2v_14B"] || !variants["wan2.1_t2v_1.3B"] {
		t.Errorf("14B and 1.3B are different models and need variant labels: %v", variants)
	}
	scaled := optByLabel(t, m, "FP8 scaled")
	if scaled.Warn == "" || !strings.Contains(scaled.Warn, "Scaled FP8") {
		t.Errorf("scaled fp8 must warn: %+v", scaled)
	}
	if plain := optByLabel(t, m, "FP8 e4m3fn"); plain.Warn != "" {
		t.Errorf("plain fp8 must not warn: %s", plain.Warn)
	}

	enc := findComp(t, p, "t5xxl")
	if enc.Label != "UMT5-XXL text encoder" || !enc.Selected {
		t.Errorf("text encoder = %+v", enc)
	}
	// The only 8-bit text-encoder build is the scaled pack, which may not
	// load: the default falls back to fp16 and the pack stays selectable.
	if got := defaultOpt(t, enc); got.Precision != "fp16" {
		t.Errorf("text encoder default = %s, want fp16", got.Precision)
	}
	if optByLabel(t, enc, "FP8 e4m3fn scaled").Warn == "" {
		t.Error("scaled text encoder should warn")
	}

	if v := findComp(t, p, "vae"); !v.Selected || len(v.Options) != 1 {
		t.Errorf("vae = %+v", v)
	}
	// A text-to-video repo does not need the CLIP vision encoder by default.
	if cv := findComp(t, p, "clip_vision"); cv.Selected {
		t.Error("clip_vision should be opt-in for a text-to-video model")
	}
	if len(p.Notes) != 0 {
		t.Errorf("no notes expected when the pipeline parts are present: %v", p.Notes)
	}
}

func TestGeneratorPlanImageToVideoOnlyIncludesClipVision(t *testing.T) {
	p := buildGeneratorPlan(entries(
		"wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors", 14,
		"wan2.1_i2v_720p_14B_fp8_e4m3fn.safetensors", 14,
		"clip_vision_h.safetensors", 1,
		"wan_2.1_vae.safetensors", 1,
	))
	if cv := findComp(t, p, "clip_vision"); !cv.Selected {
		t.Error("an image-to-video-only repo needs the CLIP vision encoder")
	}
}

func TestGeneratorPlanSD35PicksPlainFP8Encoder(t *testing.T) {
	p := buildGeneratorPlan(entries(
		"sd3.5_large_fp8_scaled.safetensors", 16,
		"text_encoders/clip_g.safetensors", 1, "text_encoders/clip_l.safetensors", 1,
		"text_encoders/t5xxl_fp16.safetensors", 10,
		"text_encoders/t5xxl_fp8_e4m3fn.safetensors", 5,
		"text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors", 5,
	))
	t5 := findComp(t, p, "t5xxl")
	if got := defaultOpt(t, t5); got.Precision != "fp8_e4m3fn" {
		t.Errorf("t5xxl default = %s, want plain fp8", got.Precision)
	}
	if len(t5.Options) != 3 {
		t.Errorf("t5xxl options = %v", optLabels(t5))
	}
	for _, id := range []string{"clip_g", "clip_l", "model"} {
		if !findComp(t, p, id).Selected {
			t.Errorf("%s should be included by default", id)
		}
	}
	// Only one option for the model and it is a scaled pack: it is still the
	// default (nothing else to pick) but carries the warning.
	m := findComp(t, p, "model")
	if len(m.Options) != 1 || m.Options[0].Warn == "" {
		t.Errorf("model = %+v", m)
	}
}

func TestGeneratorPlanIntegerPacksWarnAndLose(t *testing.T) {
	p := buildGeneratorPlan(entries(
		"flux-int8_convrot.safetensors", 12, "flux-bf16.safetensors", 24, "flux-nf4.safetensors", 7,
	))
	m := findComp(t, p, "model")
	if got := defaultOpt(t, m); got.Precision != "bf16" {
		t.Errorf("default = %s: packs that may not load must not win the default", got.Precision)
	}
	if !strings.Contains(optByLabel(t, m, "INT8").Warn, "Q8_0") {
		t.Errorf("int8 warning should point at GGUF Q8_0: %q", optByLabel(t, m, "INT8").Warn)
	}
	if optByLabel(t, m, "NF4").Warn == "" {
		t.Error("nf4 should warn")
	}
}

func TestGeneratorPlanDiffusersFoldersDoNotBundleEveryVariant(t *testing.T) {
	p := buildGeneratorPlan(entries(
		"model_index.json", 1,
		"sd_xl_base_1.0.safetensors", 7000,
		"unet/diffusion_pytorch_model.safetensors", 10000,
		"unet/diffusion_pytorch_model.fp16.safetensors", 5000,
		"unet/diffusion_pytorch_model.bin", 10000,
		"vae/diffusion_pytorch_model.safetensors", 330,
		"vae/diffusion_pytorch_model.fp16.safetensors", 170,
		"text_encoder/model.safetensors", 500, "text_encoder/model.fp16.safetensors", 250,
		"text_encoder_2/model.safetensors", 2700, "text_encoder_2/model.fp16.safetensors", 1400,
		"scheduler/scheduler_config.json", 1,
		"tokenizer/tokenizer.json", 1,
		"safety_checker/model.safetensors", 1200,
	))
	m := findComp(t, p, "model")
	if !m.Selected || len(m.Options) != 1 {
		t.Fatalf("top-level checkpoint should be the default model: %+v", m)
	}
	unet := findComp(t, p, "d_unet")
	if unet.Selected {
		t.Error("diffusers folders are off when a single-file checkpoint exists")
	}
	if len(unet.Options) != 2 {
		t.Errorf("unet options = %v (the .bin duplicate must be dropped)", optLabels(unet))
	}
	// Each folder defaults to one build, never all of them.
	for _, id := range []string{"d_unet", "d_vae", "d_text_encoder", "d_text_encoder_2"} {
		c := findComp(t, p, id)
		if got := defaultOpt(t, c); got.Precision != "fp16" && got.Label != "Default" {
			t.Errorf("%s default = %+v", id, got)
		}
	}
	if hasComp(p, "d_safety_checker") || hasComp(p, "d_scheduler") || hasComp(p, "d_tokenizer") {
		t.Error("non-weight folders must not become components")
	}
	foundNote := false
	for _, n := range p.Notes {
		if strings.Contains(n, "diffusers folder layout") {
			foundNote = true
		}
	}
	if !foundNote {
		t.Errorf("notes = %v", p.Notes)
	}
	// No plan option carries both the fp32 and fp16 file of one folder.
	for _, c := range p.Components {
		for _, o := range c.Options {
			if len(o.Files) != 1 {
				t.Errorf("%s/%s should be a single file, got %d", c.ID, o.ID, len(o.Files))
			}
		}
	}
}

func TestGeneratorPlanDiffusersOnlyIsSelectedAndShardsWarn(t *testing.T) {
	p := buildGeneratorPlan(entries(
		"model_index.json", 1,
		"transformer/diffusion_pytorch_model-00001-of-00003.safetensors", 10,
		"transformer/diffusion_pytorch_model-00002-of-00003.safetensors", 10,
		"transformer/diffusion_pytorch_model-00003-of-00003.safetensors", 5,
		"vae/diffusion_pytorch_model.safetensors", 1,
	))
	tr := findComp(t, p, "d_transformer")
	if !tr.Selected || len(tr.Options) != 1 {
		t.Fatalf("transformer = %+v", tr)
	}
	o := tr.Options[0]
	if len(o.Files) != 3 || o.TotalBytes != 25 || o.Warn == "" || !strings.Contains(o.Warn, "3 files") {
		t.Errorf("sharded set = %+v", o)
	}
	if o.Files[0].Part != 1 || o.Files[2].Part != 3 {
		t.Errorf("parts not ordered: %+v", o.Files)
	}
	if !findComp(t, p, "d_vae").Selected {
		t.Error("with no single-file weights the folders are the download")
	}
}

func TestGeneratorPlanPickleOnlyWhenNothingElse(t *testing.T) {
	p := buildGeneratorPlan(entries("model.ckpt", 4, "model.safetensors", 4, "legacy-only.ckpt", 4))
	m := findComp(t, p, "model")
	for _, o := range m.Options {
		for _, f := range o.Files {
			if strings.HasSuffix(f.Path, ".ckpt") {
				t.Errorf("pickle file offered next to safetensors: %s", f.Path)
			}
		}
	}
	p = buildGeneratorPlan(entries("legacy-only.ckpt", 4))
	if len(findComp(t, p, "model").Options) != 1 {
		t.Error("a repo with only a .ckpt must still offer it")
	}
}

func TestGeneratorPlanDedupesMirroredComponent(t *testing.T) {
	p := buildGeneratorPlan(entries(
		"vae/ae.safetensors", 300, "split_files/vae/ae.safetensors", 300, "flux1-dev-fp8.safetensors", 11,
	))
	if v := findComp(t, p, "vae"); len(v.Options) != 1 {
		t.Errorf("identical mirrored files should be one option: %v", optLabels(v))
	}
}

func TestGeneratorPlanOptionalExtrasOffByDefault(t *testing.T) {
	p := buildGeneratorPlan(entries(
		"flux1-dev-fp8.safetensors", 11,
		"loras/style-lora.safetensors", 1,
		"controlnet/canny.safetensors", 1,
		"upscale/4x-ultrasharp.safetensors", 1,
		"taesd/taesd.safetensors", 1,
	))
	if !findComp(t, p, "model").Selected {
		t.Error("model on")
	}
	for _, id := range []string{"lora", "controlnet", "esrgan", "taesd"} {
		if findComp(t, p, id).Selected {
			t.Errorf("%s is optional and should start off", id)
		}
	}
}

func TestGeneratorPlanOrdersComponents(t *testing.T) {
	p := buildGeneratorPlan(entries(
		"loras/x-lora.safetensors", 1, "ae.safetensors", 1, "flux1-dev-fp8.safetensors", 11,
		"clip_l.safetensors", 1, "t5xxl_fp16.safetensors", 9,
	))
	var roles []string
	for _, c := range p.Components {
		roles = append(roles, c.Role)
	}
	want := "model,vae,t5xxl,clip_l,lora"
	if strings.Join(roles, ",") != want {
		t.Errorf("order = %v, want %s", roles, want)
	}
}

func TestGeneratorPlanEmpty(t *testing.T) {
	p := buildGeneratorPlan(entries("README.md", 1, "model_index.json", 1))
	if len(p.Components) != 0 || len(p.Notes) == 0 {
		t.Errorf("plan = %+v", p)
	}
}

func TestBuildPlanDispatch(t *testing.T) {
	gen := BuildPlan(&RepoInfo{
		ID: "city96/FLUX.1-dev-gguf", PipelineTag: "text-to-image", Tags: []string{"gguf", "text-to-image"},
		Files: entries("flux1-dev-Q8_0.gguf", 12),
	})
	if gen.Kind != PlanGenerator {
		t.Errorf("flux gguf kind = %s", gen.Kind)
	}
	chat := BuildPlan(&RepoInfo{
		ID: "org/Model-GGUF", PipelineTag: "text-generation", Tags: []string{"gguf"},
		Files: entries("Model-Q4_K_M.gguf", 4),
	})
	if chat.Kind != PlanChat {
		t.Errorf("llm gguf kind = %s", chat.Kind)
	}
}

func TestPickByBitsPrefersLoadableAndAvoidsFP32(t *testing.T) {
	opts := []PlanOption{
		{ID: "a", Bits: 8, Warn: "x", TotalBytes: 1},
		{ID: "b", Bits: 16, TotalBytes: 2},
		{ID: "c", Bits: 32, TotalBytes: 4},
	}
	if got := opts[pickByBits(opts, 8)].ID; got != "b" {
		t.Errorf("picked %s: a warned option must lose to a sound one", got)
	}
	opts = []PlanOption{{ID: "x", Bits: 32, TotalBytes: 4}, {ID: "y", Bits: 16, TotalBytes: 2}}
	if got := opts[pickByBits(opts, 32)].ID; got != "y" {
		t.Errorf("picked %s: fp32 must not be chosen while a 16-bit option exists", got)
	}
	only := []PlanOption{{ID: "w", Bits: 8, Warn: "x"}}
	if pickByBits(only, 8) != 0 {
		t.Error("when everything warns, still pick something")
	}
	if pickByBits(nil, 8) != -1 {
		t.Error("no options → -1")
	}
}

func TestGeneratorPlanOptionIDsAreFolderSafeAndUnique(t *testing.T) {
	p := buildGeneratorPlan(entries(
		"split_files/diffusion_models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors", 14,
		"a/model.safetensors", 3, "b/model.safetensors", 4, // same name, different weights
		"split_files/vae/wan_2.1_vae.safetensors", 1,
	))
	for _, c := range p.Components {
		seen := map[string]bool{}
		for _, o := range c.Options {
			if strings.ContainsAny(o.ID, `/\ `) || o.ID != strings.ToLower(o.ID) {
				t.Errorf("%s option id %q is not a plain folder name", c.ID, o.ID)
			}
			if seen[o.ID] {
				t.Errorf("%s option id %q repeated", c.ID, o.ID)
			}
			seen[o.ID] = true
		}
	}
	if m := findComp(t, p, "model"); len(m.Options) != 3 {
		t.Errorf("same-named files with different sizes are different options: %v", optLabels(m))
	}
}
