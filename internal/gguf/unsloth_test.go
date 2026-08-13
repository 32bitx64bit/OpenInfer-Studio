package gguf

import "testing"

func TestUnslothDynamicQuant(t *testing.T) {
	cases := map[string]string{
		"Qwen3-8B-UD-Q4_K_XL.gguf":                      "UD-Q4_K_XL",
		"/models/Llama-3.1-8B-Instruct-UD-IQ3_XXS.gguf": "UD-IQ3_XXS",
		"unsloth/Qwen2.5-7B-UD-Q2_K_XL.gguf":            "UD-Q2_K_XL",
		"Qwen3-8B-UD-Q8_K_XL.gguf":                      "UD-Q8_K_XL",
		"UD-Q5_K_XL/model.gguf":                         "UD-Q5_K_XL",
		"model-Q4_K_M.gguf":                             "",
		"Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf":  "",
		"unsloth-Q4_K_M.gguf":                           "",
		"STUD-Q4_K_M.gguf":                              "",
		"model-F16.gguf":                                "",
	}
	for in, want := range cases {
		if got := UnslothDynamicQuant(in, ""); got != want {
			t.Errorf("UnslothDynamicQuant(%q) = %q, want %q", in, got, want)
		}
	}
	if got := UnslothDynamicQuant("/x/model.gguf", "Qwen3-UD-Q6_K_XL"); got != "UD-Q6_K_XL" {
		t.Errorf("name overlay = %q", got)
	}
}
