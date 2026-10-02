package mediagen

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
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

// writeSafetensors writes a header-only safetensors file with the given tensor names.
func writeSafetensors(t *testing.T, path string, names []string) {
	t.Helper()
	header := map[string]any{}
	for _, n := range names {
		header[n] = map[string]any{"dtype": "BF16", "shape": []int{1}, "data_offsets": []int{0, 0}}
	}
	body, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	var n [8]byte
	binary.LittleEndian.PutUint64(n[:], uint64(len(body)))
	if err := os.WriteFile(path, append(n[:], body...), 0o644); err != nil {
		t.Fatal(err)
	}
}

// The log from the reported run: the VAE passed lacks the tensors the model's
// VAE needs.
const missingVAELog = `=== sd-server starting ===
[INFO ] model.cpp:1011 - load tensors from /x/model.safetensors
[ERROR] model_manager.cpp:763 - VAE tensor 'first_stage_model.encoder.down.5.block.1.norm1.bias' not in model metadata
[ERROR] model_manager.cpp:763 - VAE tensor 'first_stage_model.encoder.down.5.block.1.norm1.weight' not in model metadata
[ERROR] model_manager.cpp:763 - VAE tensor 'first_stage_model.encoder.down.5.block.1.norm1.weight' not in model metadata
[ERROR] model_manager.cpp:763 - VAE tensor 'first_stage_model.encoder.norm_out.bias' not in model metadata
[ERROR] model_manager.cpp:763 - VAE tensor 'first_stage_model.post_quant_conv.bias' not in model metadata
[ERROR] model_manager.cpp:763 - VAE tensor 'first_stage_model.quant_conv.weight' not in model metadata
[ERROR] diffusion_engine.cpp:1270 - model metadata validation failed
[ERROR] main.cpp:93 - new_sd_ctx_t failed
`

func TestParseMissingTensorsGroupsAndDedupes(t *testing.T) {
	g := parseMissingTensors(missingVAELog)
	if len(g) != 1 || g[0].Kind != "VAE" || len(g[0].Names) != 5 {
		t.Fatalf("groups = %+v", g)
	}
	mixed := missingVAELog + "[ERROR] model_manager.cpp:763 - LLM tensor 'model.layers.0.self_attn.q_proj.weight' not in model metadata\n"
	if g := parseMissingTensors(mixed); len(g) != 2 || g[1].Kind != "LLM" {
		t.Fatalf("mixed groups = %+v", g)
	}
	if len(parseMissingTensors("[ERROR] could not open model\n")) != 0 {
		t.Error("unrelated errors are not missing tensors")
	}
}

func TestStartupFailureNamesTheVAEAndWhatItHolds(t *testing.T) {
	dir := t.TempDir()
	decoderOnly := filepath.Join(dir, "some_vae_bf16.safetensors")
	var names []string
	for i := 0; i < 8; i++ {
		names = append(names, "decoder.up."+string(rune('0'+i))+".block.0.conv1.weight")
	}
	names = append(names, "post_quant_conv.weight", "post_quant_conv.bias")
	writeSafetensors(t, decoderOnly, names)

	msg := startupFailure(missingVAELog, "/l.log", "/m/model.safetensors", LoadSettings{VAE: decoderOnly})
	for _, want := range []string{
		"the files it was given do not match the model",
		"VAE some_vae_bf16.safetensors: 5 tensors the model needs are missing",
		"first_stage_model.encoder.down.5.block.1.norm1.bias",
		"that file holds 10 tensors: decoder ×8, post_quant_conv ×2",
		"it has a decoder but no encoder",
		"model metadata validation failed", // the rest of the tail still shows
		"Full log: /l.log",
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("message lacks %q:\n%s", want, msg)
		}
	}
	// The wall of repeated errors is summarized, not repeated.
	if strings.Count(msg, "not in model metadata") != 0 {
		t.Errorf("raw missing-tensor lines should be summarized:\n%s", msg)
	}

	// A VAE that has an encoder is a different VAE, not a decoder-only one.
	other := filepath.Join(dir, "other_vae.safetensors")
	writeSafetensors(t, other, []string{"encoder.conv_in.weight", "decoder.conv_in.weight", "encoder.down.0.block.0.conv1.weight"})
	msg = startupFailure(missingVAELog, "/l.log", "/m/model.safetensors", LoadSettings{VAE: other})
	if !strings.Contains(msg, "a different VAE from the one this model was built with") || strings.Contains(msg, "no encoder") {
		t.Errorf("message = %s", msg)
	}

	// A pack is named as the likely reason for names that do not map.
	pack := filepath.Join(dir, "video_vae_int8_convrot.safetensors")
	writeSafetensors(t, pack, []string{"decoder.conv_in.weight"})
	msg = startupFailure(missingVAELog, "/l.log", "/m/model.safetensors", LoadSettings{VAE: pack})
	if !strings.Contains(msg, "ComfyUI-only pack") {
		t.Errorf("a packed VAE should be called out:\n%s", msg)
	}

	// No separate VAE file: the tensors should have been in the model.
	msg = startupFailure(missingVAELog, "/l.log", "/m/model.safetensors", LoadSettings{})
	if !strings.Contains(msg, "no separate file was given") {
		t.Errorf("message = %s", msg)
	}
}

// Lines from the reported run with Abiray/MiniMax-H3-Pruned-GGUF: the audio
// VAE was passed as --vae, the text-encoder GGUF stores its vision tower in
// another layout, and the diffusion GGUF has other shapes than the runtime
// expects.
const prunedGGUFLog = `=== argv: --diffusion-model /m/MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf --vae /m/vae/minimax_h3_audio_vae_fp32.safetensors --llm /m/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf ===
[INFO   ] diffusion_engine.cpp:995  - Version: MiniMax-H3 
[ERROR  ] model_manager.cpp:793  - Conditioner model tensor 'text_encoders.llm.visual.blocks.0.attn.proj.weight' has wrong shape in model metadata: got [256, 5184, 1, 1], expected [1152, 1152, 1, 1]
[ERROR  ] model_manager.cpp:793  - Conditioner model tensor 'text_encoders.llm.visual.blocks.0.attn.qkv.weight' has wrong shape in model metadata: got [256, 15552, 1, 1], expected [1152, 3456, 1, 1]
[ERROR  ] model_manager.cpp:793  - Conditioner model tensor 'text_encoders.llm.visual.patch_embed.proj.weight' has wrong shape in model metadata: got [16, 16, 6, 1152], expected [16, 16, 2, 3456]
[ERROR  ] model_manager.cpp:793  - Diffusion model tensor 'model.diffusion_model.blocks.0.attn.qkv_proj.weight' has wrong shape in model metadata: got [5376, 21504, 1, 1], expected [1, 21504, 1, 1]
[ERROR  ] model_manager.cpp:793  - Diffusion model tensor 'model.diffusion_model.blocks.0.norm1.weight' has wrong shape in model metadata: got [5376, 1, 1, 1], expected [1, 1, 1, 1]
[ERROR  ] model_manager.cpp:793  - Diffusion model tensor 'model.diffusion_model.video_patch_proj.bias' has wrong shape in model metadata: got [5376, 1, 1, 1], expected [1, 1, 1, 1]
[ERROR  ] model_manager.cpp:763  - VAE tensor 'first_stage_model.decoder.mask_token' not in model metadata
[ERROR  ] model_manager.cpp:763  - VAE tensor 'first_stage_model.encoder.down.0.block.0.conv1.weight' not in model metadata
[ERROR  ] model_manager.cpp:763  - VAE tensor 'first_stage_model.quant_conv.weight' not in model metadata
[ERROR  ] diffusion_engine.cpp:1270 - model metadata validation failed
[ERROR  ] main.cpp:93   - new_sd_ctx_t failed
`

func TestStartupFailureOnTheReportedGGUFRun(t *testing.T) {
	dir := t.TempDir()
	vae := filepath.Join(dir, "minimax_h3_audio_vae_fp32.safetensors")
	writeSafetensors(t, vae, []string{"decoder.conv_in.weight", "decoder.up.0.block.0.conv1.weight", "encoder.conv_in.weight"})
	s := LoadSettings{VAE: vae, LLM: "/m/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"}
	msg := startupFailure(prunedGGUFLog, "/l.log", "/m/MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf", s)
	for _, want := range []string{
		"VAE minimax_h3_audio_vae_fp32.safetensors: 3 tensors the model needs are missing",
		"that is an audio VAE: --vae needs the video (or image) VAE",
		"LLM text encoder qwen3vl_32b_minimax_h3-Q4_K_M.gguf: 3 tensors have other shapes",
		"visual.blocks.0.attn.proj.weight: file has [256, 5184, 1, 1], runtime expects [1152, 1152, 1, 1]",
		"only the vision tower (visual.*) differs",
		"diffusion model MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf: 3 tensors have other shapes",
		"blocks.0.attn.qkv_proj.weight: file has [5376, 21504, 1, 1], runtime expects [1, 21504, 1, 1]",
		"model metadata validation failed",
		"Full log: /l.log",
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("message lacks %q:\n%s", want, msg)
		}
	}
	if strings.Contains(msg, "has wrong shape in model metadata") || strings.Contains(msg, "not in model metadata") {
		t.Errorf("the walls of errors should be summarized:\n%s", msg)
	}
	// The diffusion model's mismatch is not a vision-tower note.
	diffusionPart := msg[strings.Index(msg, "diffusion model MiniMax"):]
	if strings.Contains(diffusionPart[:strings.Index(diffusionPart, "The files were made")], "vision tower") {
		t.Errorf("vision-tower note attached to the diffusion model:\n%s", msg)
	}
}

func TestParseShapeMismatches(t *testing.T) {
	g := parseShapeMismatches(prunedGGUFLog)
	if len(g) != 2 || g[0].Kind != "Conditioner model" || len(g[0].Items) != 3 || g[1].Kind != "Diffusion model" {
		t.Fatalf("groups = %+v", g)
	}
	if got := g[1].Items[0]; got.Name != "model.diffusion_model.blocks.0.attn.qkv_proj.weight" || got.Got != "[5376, 21504, 1, 1]" || got.Want != "[1, 21504, 1, 1]" {
		t.Errorf("item = %+v", got)
	}
}
