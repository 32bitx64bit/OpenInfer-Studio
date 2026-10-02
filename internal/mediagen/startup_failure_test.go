package mediagen

import (
	"strings"
	"testing"
)

// The files from the reported launch.
func reportedSettings() (string, LoadSettings) {
	dir := "/home/u/models/Comfy-Org--MiniMax-H3/model-minimax_h3_fl2va_pruned_w6a8/"
	return dir + "minimax_h3_fl2va_pruned_w6a8.safetensors", LoadSettings{
		VAE: dir + "minimax_h3_video_vae_int8_convrot.safetensors",
		LLM: dir + "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
	}
}

const abortLog = `=== sd-server starting 2026-10-02T13:48:21Z model=/x/minimax_h3_fl2va_pruned_w6a8.safetensors ===
[INFO ] model.cpp:1011 - load tensors from /x/minimax_h3_fl2va_pruned_w6a8.safetensors
[INFO ] stable-diffusion.cpp:312 - Version: MiniMax-H3
[INFO ] stable-diffusion.cpp:340 - Using LLM text encoder qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
/src/ggml/src/ggml.c:1749: GGML_ASSERT(tensor_type_supported) failed
[New LWP 701841]
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
0x00007fef8a1af22b in ?? () from /usr/lib/libc.so.6
#0  0x00007fef8a1af22b in ?? () from /usr/lib/libc.so.6
#1  0x00007fef8a144b88 in raise () from /usr/lib/libc.so.6
#4  0x00007fef89dd5bc9 in Linear::init_params(ggml_context*) () from /x/libstable-diffusion.so
#9  0x00007fef89ac4d05 in GGMLBlock::init(ggml_context*) () from /x/libstable-diffusion.so
#9  0x00007fef89ac4d05 in GGMLBlock::init(ggml_context*) () from /x/libstable-diffusion.so
#19 0x0000555eb84778f9 in main ()
[Inferior 1 (process 694370) detached]
`

func TestStartupFailureExplainsPackedFilesInsteadOfTheBacktrace(t *testing.T) {
	model, s := reportedSettings()
	msg := startupFailure(abortLog, "/data/media/logs/model.log", model, s)
	for _, want := range []string{
		"aborted while loading weights stored in a format stable-diffusion.cpp cannot read",
		"diffusion model minimax_h3_fl2va_pruned_w6a8.safetensors: W6A8",
		"VAE minimax_h3_video_vae_int8_convrot.safetensors: INT8 with rotation (convrot)",
		"LLM text encoder qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors: NVFP4, AWQ",
		"GGML_ASSERT(tensor_type_supported) failed", // the line that says why survives
		"Full log: /data/media/logs/model.log",
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("message lacks %q:\n%s", want, msg)
		}
	}
	for _, noise := range []string{"#9 ", "GGMLBlock::init", "libthread_db", "[Inferior 1", "[New LWP"} {
		if strings.Contains(msg, noise) {
			t.Errorf("message still carries backtrace noise %q:\n%s", noise, msg)
		}
	}
}

func TestStartupFailureWithoutPacksIsJustTheCleanedTail(t *testing.T) {
	msg := startupFailure("[ERROR] could not open model file\n#0 0x1 in main ()\n", "/l.log", "/m/flux1-dev-fp8.safetensors", LoadSettings{LLM: "/m/t5xxl_fp16.safetensors"})
	if strings.Contains(msg, "cannot read") || !strings.Contains(msg, "could not open model file") || strings.Contains(msg, "#0") {
		t.Errorf("message = %s", msg)
	}
	// Packs alone do not explain an unrelated failure.
	model, s := reportedSettings()
	msg = startupFailure("[ERROR] out of memory allocating 12 GiB\n", "/l.log", model, s)
	if strings.Contains(msg, "cannot read") {
		t.Errorf("an out-of-memory error must not be blamed on the packs:\n%s", msg)
	}
}

func TestCleanLogTailKeepsRawTextWhenAllNoise(t *testing.T) {
	raw := "#0 0x1 in a ()\n#1 0x2 in b ()\n"
	if got := cleanLogTail(raw, 5); !strings.Contains(got, "#0") {
		t.Errorf("an all-noise log must come back raw, got %q", got)
	}
	long := strings.Repeat("line\n", 40) + "last\n"
	if got := cleanLogTail(long, 3); got != "line\nlast" && !strings.HasSuffix(got, "last") {
		t.Errorf("tail = %q", got)
	}
}

func TestPackedWarningsAtLaunch(t *testing.T) {
	model, s := reportedSettings()
	w := packedWarnings(model, s)
	if len(w) != 3 {
		t.Fatalf("warnings = %v", w)
	}
	if got := packedWarnings("/m/flux1-dev-Q8_0.gguf", LoadSettings{VAE: "/m/ae.safetensors"}); len(got) != 0 {
		t.Errorf("loadable files warned: %v", got)
	}
}
