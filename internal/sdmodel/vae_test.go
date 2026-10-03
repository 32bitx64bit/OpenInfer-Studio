package sdmodel

import (
	"strings"
	"testing"
)

func TestVAEIncompatibilityUsesDecoderTensors(t *testing.T) {
	for _, prefix := range []string{"", "vae.", "first_stage_model."} {
		for _, file := range []string{"renamed.safetensors", "renamed.gguf", "ltx-2.5-video-vae-conv-bf16.safetensors"} {
			why := VAEIncompatibility(file, []string{prefix + "decoder.conv_in_x_t.weight", prefix + "decoder.diff_blocks.0.attn.qkv.weight"})
			if !strings.Contains(why, "diffusion-decoder") || !strings.Contains(why, "ltx-2.5-video-vae-conv-bf16.safetensors") {
				t.Errorf("%s %s: %q", prefix, file, why)
			}
		}
	}
	for _, names := range [][]string{nil, {"decoder.conv_in.conv.weight", "decoder.conv_out.conv.weight"}, {"decoder.diff_blocks.0.attn.qkv.weight"}} {
		if why := VAEIncompatibility("vae.safetensors", names); why != "" {
			t.Errorf("unknown or conv layout rejected: %v: %s", names, why)
		}
	}
	if why := VAEIncompatibility("ltx-2.5-audio-vae-bf16.safetensors", nil); !strings.Contains(why, "audio VAE") {
		t.Fatalf("audio VAE: %q", why)
	}
}

func TestVAEPreferenceConvBeforeDiffusionDecoder(t *testing.T) {
	conv := "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
	for _, other := range []string{"vae/ltx-2.5-video-vae-bf16.safetensors", "vae/ltx-2.5-video-vae-Q8_0.gguf"} {
		if Preference(conv) <= Preference(other) {
			t.Fatalf("%s must outrank %s", conv, other)
		}
	}
	if IsLTXConvVAEName("ltx-2.5-video-vae-conv-int8.safetensors") {
		t.Fatal("a packed quantization is not a supported conv alternative")
	}
}
