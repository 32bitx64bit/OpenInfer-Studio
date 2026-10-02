package mediagen

import "testing"

func TestLoadOverridesSatisfiedBy(t *testing.T) {
	running := LoadSettings{VAE: "/m/vae.safetensors", LLM: "/m/qwen.gguf", LoraModelDir: "/loras"}
	cases := []struct {
		name string
		ov   LoadOverrides
		want bool
	}{
		{"empty overrides never force a restart", LoadOverrides{}, true},
		{"matching vae", LoadOverrides{VAE: "/m/vae.safetensors"}, true},
		{"different vae", LoadOverrides{VAE: "/m/other.safetensors"}, false},
		{"unset text encoder on the server", LoadOverrides{T5XXL: "/m/t5.gguf"}, false},
		{"all fields match", LoadOverrides{VAE: "/m/vae.safetensors", LLM: "/m/qwen.gguf", LoraModelDir: "/loras"}, true},
		{"lora dir differs", LoadOverrides{LoraModelDir: "/other"}, false},
	}
	for _, c := range cases {
		if got := c.ov.SatisfiedBy(running); got != c.want {
			t.Errorf("%s: SatisfiedBy = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestLoadOverridesApplyOnlyNonEmpty(t *testing.T) {
	s := LoadSettings{VAE: "/auto/vae", LLM: "/auto/llm", FlashAttention: true}
	LoadOverrides{VAE: "/graph/vae", LoraModelDir: "/loras"}.Apply(&s)
	if s.VAE != "/graph/vae" || s.LoraModelDir != "/loras" {
		t.Fatalf("overrides not applied: %+v", s)
	}
	if s.LLM != "/auto/llm" || !s.FlashAttention {
		t.Fatalf("untouched fields changed: %+v", s)
	}
}

func TestLoadOverridesIsZero(t *testing.T) {
	if !(LoadOverrides{}).IsZero() {
		t.Fatal("zero value must be zero")
	}
	if (LoadOverrides{ClipL: "x"}).IsZero() {
		t.Fatal("set field must not be zero")
	}
}
