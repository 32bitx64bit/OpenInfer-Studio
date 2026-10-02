package huggingface

import "testing"

func TestPrecisionOf(t *testing.T) {
	cases := []struct {
		path   string
		id     string
		label  string
		class  string
		packed bool
	}{
		{"flux1-dev-fp8.safetensors", "fp8", "FP8", ClassFP8, false},
		{"t5xxl_fp8_e4m3fn.safetensors", "fp8_e4m3fn", "FP8 e4m3fn", ClassFP8, false},
		{"t5xxl_fp8_e4m3fn_scaled.safetensors", "fp8_e4m3fn_scaled", "FP8 e4m3fn scaled", ClassFP8, true},
		{"split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "fp8_e4m3fn_scaled", "FP8 e4m3fn scaled", ClassFP8, true},
		{"sd3.5_large_fp8_scaled.safetensors", "fp8_scaled", "FP8 scaled", ClassFP8, true},
		{"t5xxl_fp16.safetensors", "fp16", "FP16", ClassFull, false},
		{"wan2.1_t2v_14B_bf16.safetensors", "bf16", "BF16", ClassFull, false},
		{"unet/diffusion_pytorch_model.fp16.safetensors", "fp16", "FP16", ClassFull, false},
		{"v1-5-pruned.fp32.ckpt", "fp32", "FP32", ClassFull, false},
		{"flux1-dev-int8_convrot.safetensors", "int8", "INT8", ClassInt8, true},
		{"flux1-dev-nf4.safetensors", "nf4", "NF4", Class4Bit, true},
		{"svdq-int4_r32-flux.1-dev.safetensors", "int4", "INT4", Class4Bit, true},
		{"flux1-dev-Q8_0.gguf", "q8_0", "Q8_0", ClassQuant, false},
		{"flux1-dev-F16.gguf", "fp16", "F16", ClassFull, false},
		{"Q4_K_M/model.gguf", "q4_k_m", "Q4_K_M", ClassQuant, false},
		{"mmproj-BF16.gguf", "bf16", "BF16", ClassFull, false},
		{"Qwen3-8B-UD-Q4_K_XL.gguf", "ud-q4_k_xl", "UD-Q4_K_XL", ClassQuant, false},
		// Nothing stated in the name.
		{"diffusion_pytorch_model.safetensors", "", "", "", false},
		{"v1-5-pruned-emaonly.safetensors", "", "", "", false},
		{"ae.safetensors", "", "", "", false},
		// "flux1" / "sd3" must not read as float formats.
		{"flux1-schnell.safetensors", "", "", "", false},
		{"unet/model-00001-of-00003.safetensors", "", "", "", false},
		{"model.gguf", "", "", "", false},
	}
	for _, c := range cases {
		got := PrecisionOf(c.path)
		if got.ID != c.id || got.Label != c.label || got.Class != c.class || got.Packed != c.packed {
			t.Errorf("PrecisionOf(%q) = %+v, want id=%q label=%q class=%q packed=%v",
				c.path, got, c.id, c.label, c.class, c.packed)
		}
	}
}

func TestPrecisionBitsOrder(t *testing.T) {
	order := []string{
		"m-IQ2_XS.gguf", "m-Q2_K.gguf", "m-Q3_K_M.gguf", "m-Q4_K_M.gguf", "m-Q5_K_M.gguf",
		"m-Q6_K.gguf", "m-Q8_0.gguf", "m-F16.gguf", "m-F32.gguf",
	}
	prev := 0.0
	for _, p := range order {
		b := PrecisionOf(p).Bits
		if b <= prev {
			t.Errorf("%s bits %.2f is not above the previous %.2f", p, b, prev)
		}
		prev = b
	}
	// Unlisted dynamic variants fall back to digit + overhead, never 0.
	if b := PrecisionOf("m-UD-Q4_K_XXL.gguf").Bits; b < 4 || b > 5.5 {
		t.Errorf("unlisted Q4 variant bits = %.2f", b)
	}
	if b := PrecisionOf("flux1-dev-fp8.safetensors").Bits; b != 8 {
		t.Errorf("fp8 bits = %.2f", b)
	}
	if got := PrecisionOf("ae.safetensors"); got.effectiveBits() != 16 {
		t.Errorf("unstated precision should order as 16-bit, got %.2f", got.effectiveBits())
	}
}

func TestStemSansPrecision(t *testing.T) {
	cases := map[string]string{
		"flux1-dev-Q4_0.gguf":                                              "flux1-dev",
		"flux1-dev-fp8.safetensors":                                        "flux1-dev",
		"t5xxl_fp8_e4m3fn_scaled.safetensors":                              "t5xxl",
		"split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors": "umt5_xxl",
		"unet/diffusion_pytorch_model.fp16.safetensors":                    "diffusion_pytorch_model",
		"model-00001-of-00003.safetensors":                                 "model",
		"mmproj-model-f16.gguf":                                            "mmproj-model",
		"Qwen3-8B-UD-Q4_K_XL.gguf":                                         "Qwen3-8B",
		"wan2.1_t2v_14B_bf16.safetensors":                                  "wan2.1_t2v_14B",
		"a_fp8_b.safetensors":                                              "a_b",
		"ae.safetensors":                                                   "ae",
	}
	for in, want := range cases {
		if got := stemSansPrecision(in); got != want {
			t.Errorf("stemSansPrecision(%q) = %q, want %q", in, got, want)
		}
	}
	// Different precisions of one model share a stem; different models don't.
	if stemSansPrecision("flux1-dev-Q8_0.gguf") != stemSansPrecision("flux1-dev-F16.gguf") {
		t.Error("precisions of one model should share a stem")
	}
	if stemSansPrecision("flux1-dev-Q8_0.gguf") == stemSansPrecision("flux1-schnell-Q8_0.gguf") {
		t.Error("different models must not share a stem")
	}
}
