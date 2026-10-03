package mediagen

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

// These are the distinguishing tensor names in the VAE from the reported
// 6b30eb96 startup failure. Header-only fixtures require no model download.
var ltxDiffusionDecoderNames = []string{
	"encoder.conv_in.conv.weight", "encoder.conv_out.conv.weight",
	"decoder.conv_in.weight", "decoder.conv_out.weight",
	"decoder.conv_in_x_t.weight", "decoder.diff_blocks.0.attn.qkv.weight",
}

func TestPrepareLaunchRejectsIncompatibleVAE(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model.safetensors")
	bad := filepath.Join(dir, "ltx25_uncensored_video_vae.safetensors")
	good := filepath.Join(dir, "z_video_vae.safetensors")
	executable := filepath.Join(dir, "sd-server")
	writeSafetensors(t, model, []string{"transformer_blocks.0.attn1.to_q.weight"})
	writeSafetensors(t, bad, ltxDiffusionDecoderNames)
	if err := os.WriteFile(executable, nil, 0o755); err != nil {
		t.Fatal(err)
	}
	rt := &runtimes.Runtime{ExecutablePath: executable}
	for _, settings := range []LoadSettings{{}, {VAE: bad}, {RawArgs: "--vae " + bad}} {
		exe, args, _, _, _, _, err := PrepareLaunch(rt, sdHelpSample, model, settings)
		if err == nil || !strings.Contains(err.Error(), "diffusion-decoder") || exe != "" || len(args) != 0 {
			t.Fatalf("unsafe launch prepared: exe=%q args=%v err=%v", exe, args, err)
		}
	}
	report := ComponentReport(dir, model, nil, nil)
	if len(report) != 1 || report[0].Status != "incompatible" || !strings.Contains(report[0].Reason, "convolutional") {
		t.Fatalf("unusable component reported as ready: %+v", report)
	}
	if len(MissingCompanions(report)) != 0 {
		t.Fatal("must not offer to download the same incompatible file again")
	}

	writeSafetensors(t, good, []string{"decoder.conv_in.conv.weight", "decoder.conv_out.conv.weight"})
	for _, settings := range []LoadSettings{{}, {VAE: good}, {VAE: bad, RawArgs: "--vae " + good}} {
		_, args, _, _, _, _, err := PrepareLaunch(rt, sdHelpSample, model, settings)
		if err != nil || !strings.Contains(strings.Join(args, " "), good) {
			t.Fatalf("compatible alternative not used: args=%v err=%v", args, err)
		}
	}
	// Explicit choices must be explained, never silently replaced by discovery.
	_, _, _, _, _, _, err := PrepareLaunch(rt, sdHelpSample, model, LoadSettings{VAE: bad})
	if err == nil {
		t.Fatal("explicit incompatible VAE accepted despite compatible sibling")
	}
	report = ComponentReport(dir, model, nil, nil)
	if len(report) != 1 || report[0].Status != "ready" || report[0].Path != good {
		t.Fatalf("compatible alternative missing from report: %+v", report)
	}
}

func TestStartupFailureExplainsLTXDiffusionDecoder(t *testing.T) {
	vae := filepath.Join(t.TempDir(), "ltx25_uncensored_video_vae.safetensors")
	writeSafetensors(t, vae, ltxDiffusionDecoderNames)
	log := "[ERROR  ] model_manager.cpp:763 - VAE tensor 'first_stage_model.decoder.conv_in.conv.weight' not in model metadata\n"
	msg := startupFailure(log, "/model.log", "model.gguf", LoadSettings{VAE: vae})
	if !strings.Contains(msg, "diffusion-decoder") || !strings.Contains(msg, "ltx-2.5-video-vae-conv-bf16.safetensors") {
		t.Fatal(msg)
	}
}

func TestStartupFailureIgnoresPreviousAttempts(t *testing.T) {
	log := missingVAELog + "\n=== sd-server starting 2026-10-03T06:00:00Z model=/model.gguf ===\n[ERROR] out of memory\n"
	msg := startupFailure(log, "/model.log", "model.gguf", LoadSettings{})
	if !strings.Contains(msg, "out of memory") || strings.Contains(msg, "tensors the model needs are missing") {
		t.Fatal(msg)
	}
}
