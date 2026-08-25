package models

import "testing"

func TestDeriveAlias(t *testing.T) {
	cases := []struct {
		path, name, quant, want string
	}{
		// Solid general.name wins.
		{"/models/x/a.gguf", "MiniCPM V 4_6", "Q6_K", "MiniCPM V 4_6"},
		// Stub general.name ("Hf") → filename without quant.
		{
			"/home/u/.local/share/openinfer-studio/models/google--gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-e2b_q4_0-it-q4_0/gemma-4-E2B_q4_0-it.gguf",
			"Hf",
			"Q4_0",
			"gemma-4-E2B-it",
		},
		// Empty name → strip trailing quant from file.
		{"/m/repo/q4/MyModel-Q4_K_M.gguf", "", "Q4_K_M", "MyModel"},
		{"/m/Qwen3-8B-UD-Q4_K_XL.gguf", "", "UD-Q4_K_XL", "Qwen3-8B"},
		{"/m/Qwen3-8B-OID-Q4_K_XL.gguf", "", "OID-Q4_K_XL", "Qwen3-8B"},
		{"/m/Qwen3-8B-OID-IQ3_XXS_XL.gguf", "", "OID-IQ3_XXS_XL", "Qwen3-8B"},
		// Fall back to managed repo folder when filename is useless.
		{
			"/models/bartowski--Cool-Model-GGUF/q4/a.gguf",
			"Hf",
			"Q4_K_M",
			"Cool Model",
		},
		{"/m/x.gguf", "ab", "", "x"},
		// Local convert from an HF repo id stored as general.name.
		{
			"/home/u/.local/share/openinfer-studio/models/local--Blackfrost-AI-Muse-Glimmer-30B-Abliterated-BF16-BF16/files/Blackfrost-AI-Muse-Glimmer-30B-Abliterated-BF16-BF16.gguf",
			"Blackfrost-AI/Muse-Glimmer-30B-Abliterated-BF16",
			"BF16",
			"Muse-Glimmer-30B-Abliterated BF16",
		},
	}
	for _, tc := range cases {
		got := deriveAlias(tc.path, tc.name, tc.quant)
		if got != tc.want {
			t.Errorf("deriveAlias(%q, %q, %q) = %q, want %q", tc.path, tc.name, tc.quant, got, tc.want)
		}
	}
}

func TestQuantizedAlias(t *testing.T) {
	cases := []struct{ name, ftype, want string }{
		{"Muse Glimmer 30B Assistant", "Q4_K_M", "Muse Glimmer 30B Assistant Q4_K_M"},
		{"Muse Glimmer 30B Assistant F16", "Q4_K_M", "Muse Glimmer 30B Assistant Q4_K_M"},
		{"Muse Glimmer 30B Assistant Q4_K_M", "Q4_K_M", "Muse Glimmer 30B Assistant Q4_K_M"},
		{"Cool-Model-Q8_0", "Q4_K_S", "Cool-Model Q4_K_S"},
		{"MiniCPM V 4_6", "Q6_K", "MiniCPM V 4_6 Q6_K"},
		{"Name", "", "Name"},
		{"Muse Glimmer 30B Assistant", "OID-Q4_K_XL", "Muse Glimmer 30B Assistant OID-Q4_K_XL"},
		{"Muse Glimmer 30B Assistant Q4_K_M", "OID-Q4_K_XL", "Muse Glimmer 30B Assistant OID-Q4_K_XL"},
		{"Name", "OID-IQ3_XXS_XL", "Name OID-IQ3_XXS_XL"},
		{"Name OID-IQ3_XXS_XL", "OID-Q4_K_XL", "Name OID-Q4_K_XL"},
	}
	for _, tc := range cases {
		got := QuantizedAlias(tc.name, tc.ftype)
		if got != tc.want {
			t.Errorf("QuantizedAlias(%q, %q) = %q, want %q", tc.name, tc.ftype, got, tc.want)
		}
	}
}

func TestGoodAlias(t *testing.T) {
	if goodAlias("Hf") || goodAlias("ab") || goodAlias("model") {
		t.Fatal("stubs must be rejected")
	}
	if !goodAlias("Gemma 4") || !goodAlias("MiniCPM V 4_6") {
		t.Fatal("real names must be accepted")
	}
	if goodAlias("Blackfrost-AI/Muse-Glimmer-30B-Abliterated-BF16") {
		t.Fatal("HF repo ids must not be used as display names")
	}
	if got := DisplayNameFromRepo("Blackfrost-AI/Muse-Glimmer-30B-Abliterated-BF16"); got != "Muse-Glimmer-30B-Abliterated" {
		t.Fatalf("DisplayNameFromRepo = %q", got)
	}
}
