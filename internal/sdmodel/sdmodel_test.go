package sdmodel

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeSafetensors writes a minimal valid safetensors file: an 8-byte LE
// header length followed by the JSON header. Zero-length tensor data is
// fine — only the header is ever read by this package.
func writeSafetensors(t *testing.T, path string, tensorNames []string) {
	t.Helper()
	header := map[string]any{}
	for _, n := range tensorNames {
		header[n] = map[string]any{
			"dtype": "F16", "shape": []int{1}, "data_offsets": []int{0, 0},
		}
	}
	header["__metadata__"] = map[string]any{"format": "pt"}
	body, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(body)))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.Write(lenBuf[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(body); err != nil {
		t.Fatal(err)
	}
}

func TestSafetensorsTensorNamesRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.safetensors")
	want := []string{"first_stage_model.encoder.conv_in.weight", "first_stage_model.decoder.conv_out.weight"}
	writeSafetensors(t, path, want)

	got, err := SafetensorsTensorNames(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
	seen := map[string]bool{}
	for _, n := range got {
		seen[n] = true
	}
	for _, w := range want {
		if !seen[w] {
			t.Errorf("missing tensor %q in %v", w, got)
		}
	}
	// __metadata__ must never appear as a tensor name.
	if seen["__metadata__"] {
		t.Fatal("__metadata__ leaked into tensor names")
	}
}

func TestSafetensorsTensorNamesRejectsGarbageLength(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "garbage.safetensors")
	// A file whose first 8 bytes decode as a huge/garbage header length
	// (this is what a raw non-safetensors byte blob looks like) must error,
	// not allocate ~unbounded memory.
	if err := os.WriteFile(path, []byte{2, 2, 2, 2, 2, 2, 2, 2}, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := SafetensorsTensorNames(path); err == nil {
		t.Fatal("expected error for garbage header length")
	}
}

func TestTensorNamesDispatch(t *testing.T) {
	dir := t.TempDir()
	st := filepath.Join(dir, "m.safetensors")
	writeSafetensors(t, st, []string{"vae.encoder.weight"})
	names, err := TensorNames(st)
	if err != nil || len(names) != 1 {
		t.Fatalf("TensorNames(.safetensors) = %v, %v", names, err)
	}

	unsupported := filepath.Join(dir, "m.ckpt")
	if err := os.WriteFile(unsupported, []byte{0}, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := TensorNames(unsupported); err == nil {
		t.Fatal("expected error for unsupported extension")
	}
}

func TestComponentRoleFromName(t *testing.T) {
	cases := map[string]string{
		"vae/qwen_image_2.1_vae_bf16.safetensors":           RoleVAE,
		"components/vae/qwen_image_vae.safetensors":         RoleVAE,
		"text_encoders/qwen3vl_8b_int8_convrot.safetensors": RoleLLM,
		"text_encoders/t5xxl_fp16.safetensors":              RoleT5XXL,
		"text_encoders/umt5_xxl_fp8_e4m3fn.safetensors":     RoleT5XXL, // Wan uses UMT5 via --t5xxl
		"clip_l.safetensors":                                RoleClipL,
		"clip_g.safetensors":                                RoleClipG,
		"clip_vision.safetensors":                           RoleClipVision,
		"tokenizer.json":                                    RoleTokenizer,
		"taesd_decoder.safetensors":                         RoleTAESD,
		"4x-UltraSharp.pth":                                 RoleESRGAN,
		"controlnet/diffusion_pytorch_model.safetensors":    RoleControlNet,
		"loras/add-detail-xl.safetensors":                   RoleLoRA,
		"qwen-image-2.1-Q6_K.gguf":                          "",
		"model-00001-of-00004.safetensors":                  "",
		// FLUX-family autoencoders are just called "ae".
		"ae.safetensors":                  RoleVAE,
		"split_files/vae/ae.sft":          RoleVAE,
		"flux1-dev.safetensors":           "",
		"image_encoder/model.safetensors": "",
	}
	for path, want := range cases {
		if got := ComponentRole(path, nil); got != want {
			t.Errorf("ComponentRole(%q) = %q, want %q", path, got, want)
		}
	}
}

func TestComponentRoleFromTensors(t *testing.T) {
	// Anonymous filename; only tensor evidence identifies the role.
	loraTensors := []string{"lora_unet.down_blocks.0.lora_down.weight", "lora_unet.down_blocks.0.lora_up.weight",
		"lora_unet.up_blocks.0.lora_down.weight", "lora_unet.up_blocks.0.lora_up.weight"}
	if got := ComponentRole("model.safetensors", loraTensors); got != RoleLoRA {
		t.Errorf("lora tensors: role = %q, want %q", got, RoleLoRA)
	}

	vaeTensors := []string{
		"first_stage_model.encoder.conv_in.weight", "first_stage_model.encoder.down.0.block.0.norm1.weight",
		"first_stage_model.decoder.conv_out.weight", "first_stage_model.decoder.up.0.block.0.norm1.weight",
	}
	if got := ComponentRole("model.safetensors", vaeTensors); got != RoleVAE {
		t.Errorf("vae tensors: role = %q, want %q", got, RoleVAE)
	}

	clipLTensors := make([]string, 0, 12)
	for i := 0; i < 12; i++ {
		clipLTensors = append(clipLTensors, "text_model.encoder.layers."+itoa(i)+".mlp.fc1.weight")
	}
	if got := ComponentRole("model.safetensors", clipLTensors); got != RoleClipL {
		t.Errorf("clip-l tensors (12 layers): role = %q, want %q", got, RoleClipL)
	}

	clipGTensors := make([]string, 0, 32)
	for i := 0; i < 32; i++ {
		clipGTensors = append(clipGTensors, "text_model.encoder.layers."+itoa(i)+".mlp.fc1.weight")
	}
	if got := ComponentRole("model.safetensors", clipGTensors); got != RoleClipG {
		t.Errorf("clip-g tensors (32 layers): role = %q, want %q", got, RoleClipG)
	}

	t5Tensors := []string{
		"encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
		"encoder.block.0.layer.1.DenseReluDense.wi.weight",
		"encoder.block.1.layer.1.DenseReluDense.wi.weight",
		"encoder.block.2.layer.1.DenseReluDense.wi.weight",
	}
	if got := ComponentRole("model.safetensors", t5Tensors); got != RoleT5XXL {
		t.Errorf("t5 tensors: role = %q, want %q", got, RoleT5XXL)
	}

	// A full SD1.x-style checkpoint (UNet + VAE) is not a "component" —
	// ComponentRole should return "" so it is imported as a checkpoint.
	fullCkpt := append(append([]string{}, vaeTensors...),
		"model.diffusion_model.time_embed.0.weight", "model.diffusion_model.time_embed.2.weight",
		"model.diffusion_model.input_blocks.0.0.weight", "model.diffusion_model.input_blocks.1.0.weight",
		"model.diffusion_model.middle_block.0.weight", "model.diffusion_model.middle_block.1.weight",
		"model.diffusion_model.output_blocks.0.0.weight", "model.diffusion_model.output_blocks.1.0.weight")
	if got := ComponentRole("v1-5-pruned-emaonly.safetensors", fullCkpt); got != "" {
		t.Errorf("full checkpoint tensors: role = %q, want \"\"", got)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	digits := []byte{}
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}

func TestDiffusionKindGGUFStyle(t *testing.T) {
	tensors := []string{
		"img_in.weight", "time_text_embed.timestep_embedder.linear_1.weight",
		"transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.img_mlp.proj.weight",
		"transformer_blocks.1.attn.to_q.weight", "transformer_blocks.1.img_mlp.proj.weight",
		"txt_in.weight", "norm_out.weight", "proj_out.weight",
	}
	if kind := DiffusionKind(tensors); kind != "image" {
		t.Fatalf("kind = %q, want image", kind)
	}
}

func TestDiffusionKindComfyUIPrefixStripped(t *testing.T) {
	base := []string{
		"img_in.weight", "time_text_embed.timestep_embedder.linear_1.weight",
		"transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.img_mlp.proj.weight",
		"transformer_blocks.1.attn.to_q.weight", "transformer_blocks.1.img_mlp.proj.weight",
		"txt_in.weight", "norm_out.weight", "proj_out.weight",
	}
	prefixed := make([]string, len(base))
	for i, n := range base {
		prefixed[i] = "model.diffusion_model." + n
	}
	if kind := DiffusionKind(prefixed); kind != "image" {
		t.Fatalf("kind = %q, want image (ComfyUI prefix should be stripped)", kind)
	}
}

func TestDiffusionKindUNetLayout(t *testing.T) {
	tensors := []string{
		"model.diffusion_model.time_embed.0.weight", "model.diffusion_model.time_embed.2.weight",
		"model.diffusion_model.input_blocks.0.0.weight", "model.diffusion_model.input_blocks.1.0.weight",
		"model.diffusion_model.middle_block.0.weight", "model.diffusion_model.middle_block.1.weight",
		"model.diffusion_model.output_blocks.0.0.weight", "model.diffusion_model.output_blocks.1.0.weight",
	}
	if kind := DiffusionKind(tensors); kind != "image" {
		t.Fatalf("kind = %q, want image (SD1.x/SDXL UNet layout)", kind)
	}
}

func TestDiffusionKindNoEvidence(t *testing.T) {
	tensors := []string{"token_embd.weight", "blk.0.attn_q.weight", "output.weight"}
	if kind := DiffusionKind(tensors); kind != "" {
		t.Fatalf("kind = %q, want \"\" for a plain LLM tensor set", kind)
	}
}

func TestIsFullCheckpoint(t *testing.T) {
	unetOnly := []string{
		"model.diffusion_model.time_embed.0.weight", "model.diffusion_model.input_blocks.0.0.weight",
		"model.diffusion_model.middle_block.0.weight", "model.diffusion_model.output_blocks.0.0.weight",
		"model.diffusion_model.input_blocks.1.0.weight", "model.diffusion_model.output_blocks.1.0.weight",
	}
	if IsFullCheckpoint(unetOnly) {
		t.Fatal("UNet-only weights must not read as a full checkpoint")
	}

	withVAE := append(append([]string{}, unetOnly...),
		"first_stage_model.encoder.conv_in.weight", "first_stage_model.decoder.conv_out.weight")
	if !IsFullCheckpoint(withVAE) {
		t.Fatal("UNet + VAE weights should read as a full checkpoint")
	}

	noDiffusion := []string{"first_stage_model.encoder.conv_in.weight"}
	if IsFullCheckpoint(noDiffusion) {
		t.Fatal("VAE-only weights (no diffusion transformer) must not read as a full checkpoint")
	}
}

func TestPreferenceRanksPlainFP8BetweenFloatsAndPacks(t *testing.T) {
	order := []string{
		"model-Q8_0.gguf",
		"t5xxl_fp16.safetensors",
		"t5xxl.safetensors",
		"t5xxl_fp8_e4m3fn.safetensors",
		"t5xxl_fp8_e4m3fn_scaled.safetensors",
	}
	for i := 0; i+1 < len(order); i++ {
		if Preference(order[i]) <= Preference(order[i+1]) {
			t.Errorf("Preference(%q)=%d should beat Preference(%q)=%d",
				order[i], Preference(order[i]), order[i+1], Preference(order[i+1]))
		}
	}
	if Preference("flux-int8_convrot.safetensors") >= Preference("flux-fp8.safetensors") {
		t.Error("int8 packs must rank under plain fp8")
	}
}

func TestPreferenceTreatsScaledAndFourBitPacksAsLast(t *testing.T) {
	plain := Preference("model-fp8.safetensors")
	for _, packed := range []string{
		"model-fp8-scaled.safetensors", "scaled_fp8_model.safetensors", "model_fp8_e4m3fn_scaled.safetensors",
		"flux-nf4.safetensors", "flux-fp4.safetensors", "svdq-int4-flux.safetensors", "flux-int8.safetensors",
	} {
		if Preference(packed) >= plain {
			t.Errorf("Preference(%q)=%d must rank under plain fp8 (%d)", packed, Preference(packed), plain)
		}
	}
	// "unscaled" and "scalediffusion" are not the word "scaled".
	if Preference("unscaled-model.safetensors") != 50 {
		t.Errorf("unscaled-model scored %d", Preference("unscaled-model.safetensors"))
	}
}

func TestVAENamesAreWordsNotSubstrings(t *testing.T) {
	cases := map[string]string{
		"sd_xl_base_1.0_0.9vae.safetensors":           "", // a checkpoint built with the 0.9 VAE
		"ClearVAE_V2.3.safetensors":                   RoleVAE,
		"vae-ft-mse-840000-ema-pruned.safetensors":    RoleVAE,
		"sdxl_vae.safetensors":                        RoleVAE,
		"flux.vae.safetensors":                        RoleVAE,
		"vae_1_0/diffusion_pytorch_model.safetensors": RoleVAE,
	}
	for path, want := range cases {
		if got := ComponentRole(path, nil); got != want {
			t.Errorf("ComponentRole(%q) = %q, want %q", path, got, want)
		}
	}
}

func TestManifestRoundTripMergeAndNestedLookup(t *testing.T) {
	dir := t.TempDir()
	if err := WriteManifest(dir, Manifest{Version: 1, Repo: "Comfy-Org/MiniMax-H3", DiffusionKind: "video",
		Files: map[string]string{"minimax_h3_bf16.safetensors": RoleModel, "text_encoders/enc.safetensors": RoleT5XXL}}); err != nil {
		t.Fatal(err)
	}
	// A later download into the same folder keeps what the first recorded.
	if err := WriteManifest(dir, Manifest{Version: 1, Files: map[string]string{"vae.safetensors": RoleVAE}}); err != nil {
		t.Fatal(err)
	}
	m := ReadManifest(dir)
	if m == nil || m.Repo != "Comfy-Org/MiniMax-H3" || len(m.Files) != 3 {
		t.Fatalf("manifest = %+v", m)
	}
	for path, want := range map[string]string{
		filepath.Join(dir, "minimax_h3_bf16.safetensors"):      RoleModel,
		filepath.Join(dir, "vae.safetensors"):                  RoleVAE,
		filepath.Join(dir, "text_encoders", "enc.safetensors"): RoleT5XXL, // found from a subfolder
	} {
		if got, ok := DeclaredRole(path); !ok || got != want {
			t.Errorf("DeclaredRole(%s) = %q,%v want %q", path, got, ok, want)
		}
	}
	if _, ok := DeclaredRole(filepath.Join(dir, "other.safetensors")); ok {
		t.Error("an unrecorded file has no declared role")
	}
	if DeclaredKind(filepath.Join(dir, "vae.safetensors")) != "video" {
		t.Error("the recorded modality should reach every file of the download")
	}
	// A repository that does both leaves the kind to the file name.
	both := t.TempDir()
	_ = WriteManifest(both, Manifest{Version: 1, DiffusionKind: "both", Files: map[string]string{"m.safetensors": RoleModel}})
	if DeclaredKind(filepath.Join(both, "m.safetensors")) != "" {
		t.Error("kind 'both' must not be reported as image or video")
	}
	// No temp files are left behind.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if e.Name() != ManifestName && e.Name() != "text_encoders" {
			t.Errorf("stray file %q", e.Name())
		}
	}
}

func TestManifestIgnoresGarbage(t *testing.T) {
	dir := t.TempDir()
	for name, body := range map[string]string{"junk": "not json", "wrongver": `{"version":2,"files":{"a":"model"}}`, "nofiles": `{"version":1}`} {
		d := filepath.Join(dir, name)
		_ = os.MkdirAll(d, 0o755)
		_ = os.WriteFile(filepath.Join(d, ManifestName), []byte(body), 0o644)
		if m := ReadManifest(d); m != nil {
			t.Errorf("%s: unusable manifest accepted: %+v", name, m)
		}
	}
}

func TestDiffusionKindRecognizesWanAndDiTLayouts(t *testing.T) {
	wan := []string{"patch_embedding.weight", "text_embedding.0.weight", "time_embedding.0.weight",
		"time_projection.1.weight", "blocks.0.self_attn.q.weight", "blocks.0.cross_attn.k.weight", "head.head.weight"}
	if got := DiffusionKind(wan); got != "video" {
		t.Errorf("Wan layout = %q, want video", got)
	}
	dit := []string{"x_embedder.proj.weight", "t_embedder.mlp.0.weight", "blocks.0.adaLN_modulation.1.weight", "final_layer.linear.weight"}
	if got := DiffusionKind(dit); got != "image" {
		t.Errorf("DiT layout = %q, want image", got)
	}
	var prefixed []string
	for i := 0; i < 10; i++ {
		prefixed = append(prefixed, "model.diffusion_model.mystery."+string(rune('a'+i))+".weight")
	}
	if got := DiffusionKind(prefixed); got == "" {
		t.Error("ComfyUI's model.diffusion_model. prefix is diffusion evidence")
	}
	// Language models and VAEs are not.
	llm := []string{"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "lm_head.weight"}
	if DiffusionKind(llm) != "" || !LooksLikeLLM(llm) {
		t.Error("an LLM must not read as a generator")
	}
	if LooksLikeLLM(wan) {
		t.Error("Wan tensors are not an LLM")
	}
}

func TestPackedNameAndReason(t *testing.T) {
	names := map[string]string{
		"minimax_h3_fl2va_pruned_w6a8.safetensors":      "W6A8",
		"minimax_h3_video_vae_int8_convrot.safetensors": "convrot",
		"qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors":  "NVFP4",
		"flux-nf4.safetensors":                          "NF4",
		"t5xxl_fp8_e4m3fn_scaled.safetensors":           "scaled FP8",
		"svdq-int4-flux.safetensors":                    "INT4",
	}
	for n, want := range names {
		if got := PackedName(n); !strings.Contains(got, want) {
			t.Errorf("PackedName(%q) = %q, want it to mention %q", n, got, want)
		}
	}
	for _, ok := range []string{"flux1-dev-fp8.safetensors", "t5xxl_fp16.safetensors", "wan2.1_t2v_14B_bf16.safetensors", "ae.safetensors", "flux1-dev-Q4_0.gguf"} {
		if got := PackedName(ok); got != "" {
			t.Errorf("PackedName(%q) = %q, want none", ok, got)
		}
	}

	// A pack is recognised from its scale tensors whatever it is called.
	dir := t.TempDir()
	scaled := filepath.Join(dir, "innocent_name.safetensors")
	writeSafetensors(t, scaled, []string{"blocks.0.attn.q.weight", "blocks.0.attn.q.weight_scale", "blocks.0.attn.q.comfy_quant"})
	if why := PackedReason(scaled); why == "" {
		t.Error("scale tensors identify a pack")
	}
	plain := filepath.Join(dir, "plain.safetensors")
	writeSafetensors(t, plain, []string{"blocks.0.attn.q.weight", "blocks.0.attn.q.bias"})
	if why := PackedReason(plain); why != "" {
		t.Errorf("plain weights flagged: %s", why)
	}
	// GGUF quantizations are native: never a pack, whatever the name says.
	if why := PackedReason(filepath.Join(dir, "model-int8-awq.gguf")); why != "" {
		t.Errorf("GGUF flagged: %s", why)
	}
	if why := PackedReason(filepath.Join(dir, "missing.safetensors")); why != "" {
		t.Errorf("an unreadable file is not called a pack: %s", why)
	}
}

func TestAudioVAEIsNotTheVAE(t *testing.T) {
	cases := map[string]string{
		"vae/minimax_h3_audio_vae_fp32.safetensors": RoleAudioVAE,
		"minimax_h3_audio_vae.safetensors":          RoleAudioVAE,
		"vae/minimax_h3_video_vae_fp32.safetensors": RoleVAE,
		"vae/audio_decoder.safetensors":             RoleAudioVAE, // in a vae/ folder, named for audio
		"audio_model_bf16.safetensors":              "",           // no VAE in it: not this rule
	}
	for path, want := range cases {
		if got := ComponentRole(path, nil); got != want {
			t.Errorf("ComponentRole(%q) = %q, want %q", path, got, want)
		}
	}
}
