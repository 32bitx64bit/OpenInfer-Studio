package huggingface

import "testing"

func TestGroupDiffusionFilesSingle(t *testing.T) {
	files := []FileEntry{
		{Path: "v1-5-pruned-emaonly.safetensors", Size: 100},
		{Path: "vae.safetensors", Size: 10},
		{Path: "README.md", Size: 5},
	}
	groups := GroupDiffusionFiles(files)
	if len(groups) != 1 {
		t.Fatalf("want 1 group (companion rides along), got %d (%+v)", len(groups), groups)
	}
	g := groups[0]
	if g.Kind != "checkpoint" {
		t.Fatalf("kind = %q, want checkpoint", g.Kind)
	}
	if len(g.Files) != 2 {
		t.Fatalf("checkpoint must carry its VAE companion, got %+v", g.Files)
	}
	if g.TotalBytes != 110 {
		t.Fatalf("total = %d, want 110", g.TotalBytes)
	}
	roles := map[string]string{}
	for _, f := range g.Files {
		roles[f.Path] = f.Role
	}
	if roles["v1-5-pruned-emaonly.safetensors"] != "checkpoint" {
		t.Fatalf("checkpoint role = %+v", roles)
	}
	if roles["vae.safetensors"] != "vae" {
		t.Fatalf("vae role = %+v", roles)
	}
}

func TestGroupDiffusionFilesBundle(t *testing.T) {
	files := []FileEntry{
		{Path: "model_index.json", Size: 10},
		{Path: "unet/diffusion_pytorch_model.safetensors", Size: 100},
		{Path: "vae/diffusion_pytorch_model.safetensors", Size: 20},
		{Path: "README.md", Size: 5},
	}
	groups := GroupDiffusionFiles(files)
	if len(groups) != 2 {
		t.Fatalf("want weights+config, got %+v", groups)
	}
	if groups[0].ID != "bundle-weights" || len(groups[0].Files) != 2 {
		t.Fatalf("weights group = %+v", groups[0])
	}
}

// A component sitting under text_encoders/ must never be offered as a
// separate "checkpoint" download — the role comes from the directory as
// well as the basename.
func TestDiffusionRoleUsesDirectory(t *testing.T) {
	cases := map[string]string{
		"text_encoders/qwen3vl_8b_bf16.safetensors":         "encoder",
		"vae/qwen_image_2.1_vae_bf16.safetensors":           "vae",
		"ckpt-qwen-image-2-1-q6_k/qwen-image-2.1-Q6_K.gguf": "checkpoint",
		"loras/detail-tweaker.safetensors":                  "lora",
	}
	for path, want := range cases {
		if got := diffusionRole(path); got != want {
			t.Errorf("diffusionRole(%q) = %q, want %q", path, got, want)
		}
	}
}

// The checkpoint download must carry exactly one VAE and one encoder —
// the most loadable variant (GGUF > bf16 > exotic quants) — not every
// alternative the repo offers. Optional extras stay in the components group.
func TestGroupDiffusionFilesPicksBestCompanions(t *testing.T) {
	files := []FileEntry{
		{Path: "ckpt/qwen-image-2.1-Q6_K.gguf", Size: 1000},
		{Path: "vae/qwen_image_2.1_vae_bf16.safetensors", Size: 10},
		{Path: "text_encoders/qwen3vl_8b_bf16.safetensors", Size: 200},
		{Path: "text_encoders/qwen3vl_8b_int8_convrot.safetensors", Size: 150},
		{Path: "loras/style.safetensors", Size: 5},
	}
	groups := GroupDiffusionFiles(files)
	var ck, comps *DiffusionGroup
	for i := range groups {
		switch groups[i].Kind {
		case "checkpoint":
			ck = &groups[i]
		case "component":
			comps = &groups[i]
		}
	}
	if ck == nil {
		t.Fatalf("no checkpoint group in %+v", groups)
	}
	byPath := map[string]string{}
	for _, f := range ck.Files {
		byPath[f.Path] = f.Role
	}
	if len(ck.Files) != 3 {
		t.Fatalf("checkpoint group must carry checkpoint + vae + one encoder, got %+v", ck.Files)
	}
	if byPath["text_encoders/qwen3vl_8b_bf16.safetensors"] != "encoder" {
		t.Fatalf("bf16 encoder should win, got %+v", byPath)
	}
	if _, bad := byPath["text_encoders/qwen3vl_8b_int8_convrot.safetensors"]; bad {
		t.Fatalf("int8_convrot must not be auto-attached: %+v", byPath)
	}
	if comps == nil {
		t.Fatalf("alternatives + LoRA must land in a components group")
	}
	rest := map[string]string{}
	for _, f := range comps.Files {
		rest[f.Path] = f.Role
	}
	if rest["loras/style.safetensors"] != "lora" {
		t.Fatalf("LoRA must stay optional, got %+v", rest)
	}
	if rest["text_encoders/qwen3vl_8b_int8_convrot.safetensors"] != "encoder" {
		t.Fatalf("unused encoder alternative must stay optional, got %+v", rest)
	}
}
