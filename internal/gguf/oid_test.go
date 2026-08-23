package gguf

import "testing"

func TestOpenInferDynamicQuant(t *testing.T) {
	cases := map[string]string{
		"Qwen3-8B-OID-Q4_K_XL.gguf":                         "OID-Q4_K_XL",
		"/models/Llama-3.1-8B-Instruct-OID-Q3_K_XL.gguf":    "OID-Q3_K_XL",
		"Qwen3-8B-OID-Q5_K_XL.gguf":                         "OID-Q5_K_XL",
		"Qwen3-8B-OID-IQ3_XXS_XL.gguf":                      "OID-IQ3_XXS_XL",
		"/models/Llama-3.1-8B-Instruct-OID-IQ3_XXS_XL.gguf": "OID-IQ3_XXS_XL",
		"OID-IQ3_XXS_XL/model.gguf":                         "OID-IQ3_XXS_XL",
		"OID-Q4_K_XL/model.gguf":                            "OID-Q4_K_XL",
		"model-Q4_K_M.gguf":                                 "",
		"Qwen3-8B-UD-Q4_K_XL.gguf":                          "",
		"VOID-Q4_K_M.gguf":                                  "",
		"model-F16.gguf":                                    "",
	}
	for in, want := range cases {
		if got := OpenInferDynamicQuant(in, ""); got != want {
			t.Errorf("OpenInferDynamicQuant(%q) = %q, want %q", in, got, want)
		}
	}
	if got := OpenInferDynamicQuant("/x/model.gguf", "Qwen3-OID-Q6_K_XL"); got != "OID-Q6_K_XL" {
		t.Errorf("name overlay = %q", got)
	}
}

func TestOverlayDynamicQuant(t *testing.T) {
	if got := OverlayDynamicQuant("m-OID-IQ3_XXS_XL.gguf", "", "Q3_K_M"); got != "OID-IQ3_XXS_XL" {
		t.Errorf("iq3 overlay = %q", got)
	}
	if got := OverlayDynamicQuant("m-UD-Q4_K_XL.gguf", "", "Q4_K_M"); got != "UD-Q4_K_XL" {
		t.Errorf("ud overlay = %q", got)
	}
	if got := OverlayDynamicQuant("m-Q4_K_M.gguf", "", "Q4_K_M"); got != "Q4_K_M" {
		t.Errorf("passthrough = %q", got)
	}
}
