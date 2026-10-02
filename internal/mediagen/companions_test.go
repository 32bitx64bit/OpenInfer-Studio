package mediagen

import (
	"github.com/openinfer/openinfer-studio/internal/sdmodel"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestClassifyCompanion(t *testing.T) {
	cases := map[string]string{
		"vae/qwen_image_2.1_vae_bf16.safetensors":           CompanionVAE,
		"components/vae/qwen_image_vae.safetensors":         CompanionVAE,
		"text_encoders/qwen3vl_8b_int8_convrot.safetensors": CompanionLLM,
		"text_encoders/t5xxl_fp16.safetensors":              CompanionT5XXL,
		"clip_l.safetensors":                                CompanionClipL,
		"clip_g.safetensors":                                CompanionClipG,
		"clip_vision.safetensors":                           CompanionClipVision,
		"tokenizer.json":                                    CompanionTokenizer,
		"taesd_decoder.safetensors":                         CompanionTAESD,
		"4x-UltraSharp.pth":                                 CompanionUpscaler,
		"qwen-image-2.1-Q6_K.gguf":                          "",
		"README.md":                                         "",
	}
	for path, want := range cases {
		if got := classifyCompanion(path); got != want {
			t.Errorf("classifyCompanion(%q) = %q, want %q", path, got, want)
		}
	}
}

func TestApplyCompanionsPrefersExplicit(t *testing.T) {
	root := t.TempDir()
	vaeDir := filepath.Join(root, "components", "vae")
	encDir := filepath.Join(root, "text_encoders")
	if err := os.MkdirAll(vaeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(encDir, 0o755); err != nil {
		t.Fatal(err)
	}
	vae := filepath.Join(vaeDir, "qwen_image_vae.safetensors")
	enc := filepath.Join(encDir, "qwen3vl_8b.safetensors")
	primary := filepath.Join(root, "ckpt", "qwen-image-2.1-Q6_K.gguf")
	for _, p := range []string{vae, enc, primary} {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	s := DefaultLoadSettings()
	ApplyCompanions(&s, root, primary)
	if s.VAE != vae {
		t.Fatalf("VAE = %q, want %q", s.VAE, vae)
	}
	if s.LLM != enc {
		t.Fatalf("LLM = %q, want %q", s.LLM, enc)
	}

	// Explicit user choice wins over discovery.
	s2 := DefaultLoadSettings()
	s2.VAE = "/custom/vae.safetensors"
	ApplyCompanions(&s2, root, primary)
	if s2.VAE != "/custom/vae.safetensors" {
		t.Fatalf("explicit VAE overwritten: %q", s2.VAE)
	}
	if s2.LLM != enc {
		t.Fatalf("LLM = %q, want %q", s2.LLM, enc)
	}
}

func TestComponentReportMissing(t *testing.T) {
	root := t.TempDir()
	primary := filepath.Join(root, "ckpt", "qwen-image-2.1-Q6_K.gguf")
	if err := os.MkdirAll(filepath.Dir(primary), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(primary, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	vae := filepath.Join(root, "components", "vae", "qwen_image_vae.safetensors")
	if err := os.MkdirAll(filepath.Dir(vae), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(vae, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	repoFiles := []string{
		"qwen-image-2.1-UC-Q6_K.gguf", // primary upstream — not a companion
		"vae/qwen_image_2.1_vae_bf16.safetensors",
		"text_encoders/qwen3vl_8b_int8_convrot.safetensors",
	}
	sizes := map[string]int64{
		"vae/qwen_image_2.1_vae_bf16.safetensors":           675509688,
		"text_encoders/qwen3vl_8b_int8_convrot.safetensors": 9350000000,
	}
	report := ComponentReport(root, primary, repoFiles, sizes)
	if len(report) != 2 {
		t.Fatalf("report = %+v", report)
	}
	byRole := map[string]Companion{}
	for _, c := range report {
		byRole[c.Role] = c
	}
	if v := byRole[CompanionVAE]; v.Status != "ready" || v.Path != vae {
		t.Fatalf("vae entry = %+v", v)
	}
	if l := byRole[CompanionLLM]; l.Status != "missing" || l.RepoPath == "" || l.Size != 9350000000 {
		t.Fatalf("llm entry = %+v", l)
	}
	if n := len(MissingCompanions(report)); n != 1 {
		t.Fatalf("missing count = %d", n)
	}
}

func TestBuildServerArgsComponentMode(t *testing.T) {
	caps := ParseSDCapabilities(sdHelpSample + "\n         --llm FNAME  llm encoder\n         --diffusion-model FNAME  standalone\n")
	dir := t.TempDir()
	model := filepath.Join(dir, "m.gguf")
	llmPath := filepath.Join(dir, "qwen3vl.safetensors")
	vaePath := filepath.Join(dir, "vae.safetensors")
	for _, p := range []string{model, llmPath, vaePath} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	s := DefaultLoadSettings()
	s.LLM = llmPath
	s.VAE = vaePath
	args, err := BuildServerArgs(s, model, true, caps, sdHelpSample+" --llm --diffusion-model", "127.0.0.1", 9000)
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--diffusion-model "+model) {
		t.Fatalf("component mode flag missing: %q", joined)
	}
	if strings.Contains(joined, "--model "+model) {
		t.Fatalf("--model must not be used in component mode: %q", joined)
	}
	if !strings.Contains(joined, "--llm "+llmPath) {
		t.Fatalf("--llm companion missing: %q", joined)
	}
	if !strings.Contains(joined, "--vae "+vaePath) {
		t.Fatalf("--vae companion missing: %q", joined)
	}

	full, err := BuildServerArgs(s, model, false, caps, sdHelpSample+" --llm --diffusion-model", "127.0.0.1", 9000)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(full, " "), "--model "+model) {
		t.Fatalf("full checkpoints keep --model: %q", full)
	}
}

func TestCompanionPreferencePrefersLoadable(t *testing.T) {
	cases := []struct {
		a, b string // a should win over b
	}{
		{"text_encoders/qwen3vl_8b_bf16.safetensors", "text_encoders/qwen3vl_8b_int8_convrot.safetensors"},
		{"vae.safetensors", "vae_fp8_scaled.safetensors"},
		{"t5xxl.gguf", "t5xxl_fp16.safetensors"},
		{"clip_l_bf16.safetensors", "clip_l_q8_0.safetensors"},
	}
	for _, tc := range cases {
		if companionPreference(tc.a) <= companionPreference(tc.b) {
			t.Errorf("preference(%q)=%d should beat preference(%q)=%d",
				tc.a, companionPreference(tc.a), tc.b, companionPreference(tc.b))
		}
	}
}

func TestDiscoverCompanionsPrefersBF16OverInt8(t *testing.T) {
	root := t.TempDir()
	encDir := filepath.Join(root, "components", "text_encoders")
	if err := os.MkdirAll(encDir, 0o755); err != nil {
		t.Fatal(err)
	}
	int8 := filepath.Join(encDir, "qwen3vl_8b_int8_convrot.safetensors")
	bf16 := filepath.Join(encDir, "qwen3vl_8b_bf16.safetensors")
	primary := filepath.Join(root, "ckpt", "m.gguf")
	for _, p := range []string{int8, bf16, primary} {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got := DiscoverCompanions(root, primary)
	if got[CompanionLLM] != bf16 {
		t.Fatalf("llm companion = %q, want bf16 %q (int8_convrot breaks sd.cpp)", got[CompanionLLM], bf16)
	}
}

// A download that recorded what each file is beats guessing from its name.
func TestClassifyCompanionTrustsTheDownloadRecord(t *testing.T) {
	dir := t.TempDir()
	for _, f := range []string{"weights_a.safetensors", "weights_b.safetensors", "mystery_model.safetensors", "plain.safetensors"} {
		if err := os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := sdmodel.WriteManifest(dir, sdmodel.Manifest{Version: 1, Files: map[string]string{
		"weights_a.safetensors":     sdmodel.RoleVAE,
		"weights_b.safetensors":     sdmodel.RoleT5XXL,
		"mystery_model.safetensors": sdmodel.RoleModel,
	}}); err != nil {
		t.Fatal(err)
	}
	for file, want := range map[string]string{
		"weights_a.safetensors":     CompanionVAE,
		"weights_b.safetensors":     CompanionT5XXL,
		"mystery_model.safetensors": "", // the model itself is never a companion
		"plain.safetensors":         "", // not recorded, no hint: unchanged behaviour
	} {
		if got := classifyCompanion(filepath.Join(dir, file)); got != want {
			t.Errorf("classifyCompanion(%s) = %q, want %q", file, got, want)
		}
	}
	// The recorded companions are what discovery wires up.
	found := DiscoverCompanions(dir, filepath.Join(dir, "mystery_model.safetensors"))
	if found[CompanionVAE] == "" || found[CompanionT5XXL] == "" {
		t.Errorf("discovery ignored the record: %v", found)
	}
}
