package gguf

import "testing"

func TestDetectDiffusion(t *testing.T) {
	cases := []struct {
		name   string
		arch   string
		gguf   string
		path   string
		raw    map[string]any
		want   bool
		canvas uint32
	}{
		{
			name:   "arch diffusion-gemma",
			arch:   "diffusion-gemma",
			path:   "/models/foo.gguf",
			want:   true,
			canvas: 256,
		},
		{
			name:   "canvas kv",
			arch:   "llama",
			raw:    map[string]any{"diffusion.canvas_length": uint32(256)},
			path:   "/models/foo.gguf",
			want:   true,
			canvas: 256,
		},
		{
			name:   "filename diffusiongemma",
			arch:   "",
			path:   "/models/DiffusionGemma-26B-A4B-it-Q4_K_M.gguf",
			want:   true,
			canvas: 256,
		},
		{
			name:   "filename diffusion-gemma",
			arch:   "",
			path:   "/x/diffusion-gemma-q8.gguf",
			want:   true,
			canvas: 256,
		},
		{
			name: "normal gemma4",
			arch: "gemma4",
			path: "/models/gemma-4-12B-it-Q4_K_M.gguf",
			want: false,
		},
		{
			name: "mmproj ignored",
			arch: "",
			path: "/models/mmproj-diffusion-something.gguf",
			want: false,
		},
		{
			name:   "custom canvas",
			arch:   "diffusion-gemma",
			raw:    map[string]any{"diffusion.canvas_length": uint32(512)},
			path:   "/m.gguf",
			want:   true,
			canvas: 512,
		},
	}
	for _, c := range cases {
		got, canvas := DetectDiffusion(c.arch, c.gguf, c.path, c.raw)
		if got != c.want || (c.want && canvas != c.canvas) {
			t.Errorf("%s: got (%v,%d) want (%v,%d)", c.name, got, canvas, c.want, c.canvas)
		}
	}
}

func TestApplyDiffusionFlagsClearsDraftEmbed(t *testing.T) {
	md := &Metadata{
		Architecture:     "diffusion-gemma",
		SpeculativeDraft: true,
		IsEmbedding:      true,
		Raw:              map[string]any{"diffusion.canvas_length": uint32(256)},
	}
	md.ApplyDiffusionFlags("/x/diffusiongemma.gguf")
	if !md.IsDiffusion || md.CanvasLength != 256 {
		t.Fatalf("flags = %+v", md)
	}
	if md.SpeculativeDraft || md.IsEmbedding {
		t.Fatalf("draft/embed should be cleared: %+v", md)
	}
}

func TestDetectDiffusionTensorsQwenImage(t *testing.T) {
	// Qwen-Image is an SD-GGUF image checkpoint, not a block-diffusion LM:
	// DetectDiffusionTensors (IsDiffusion / DiffusionGemma chat path) must
	// NOT flag it, even though tensor evidence is present. IsSDCheckpoint
	// is the separate signal for "this is an sd.cpp image/video target".
	tensors := []string{
		"img_in.weight", "time_text_embed.timestep_embedder.linear_1.weight",
		"transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.img_mlp.proj.weight",
		"transformer_blocks.1.attn.to_q.weight", "transformer_blocks.1.img_mlp.proj.weight",
		"txt_in.weight", "norm_out.weight", "proj_out.weight",
	}
	isDiff, canvas := DetectDiffusionTensors("", "", "/m/qwen-image-2.1-Q6_K.gguf", nil, tensors)
	if isDiff || canvas != 0 {
		t.Fatalf("Qwen-Image SD-GGUF must not be flagged as a block-diffusion LM: isDiff=%v canvas=%d", isDiff, canvas)
	}
	if !IsSDCheckpoint("", "/m/qwen-image-2.1-Q6_K.gguf", tensors) {
		t.Fatal("Qwen-Image tensor signature not detected as an SD checkpoint")
	}
	if kind := SDKindFromTensors(tensors); kind != "image" {
		t.Fatalf("kind = %q, want image", kind)
	}
}

func TestDetectDiffusionStableDiffusionFilenameNotBlockLM(t *testing.T) {
	// A checkpoint filename containing the literal substring "diffusion"
	// ("stable-diffusion-xl…") must not trip the generic block-diffusion-LM
	// filename heuristic — it is an SD checkpoint, not DiffusionGemma.
	isDiff, _ := DetectDiffusion("", "", "/m/stable-diffusion-xl-base-1.0-Q4_K_M.gguf", nil)
	if isDiff {
		t.Fatal("stable-diffusion-xl filename misclassified as block-diffusion LM")
	}
	if !IsSDCheckpoint("", "/m/stable-diffusion-xl-base-1.0-Q4_K_M.gguf", nil) {
		t.Fatal("stable-diffusion-xl filename not recognized as an SD checkpoint")
	}
}

func TestIsSDCheckpointCollisionAvoidance(t *testing.T) {
	// Bare family tokens ("flux", "sdxl", "wan") without tensor evidence
	// are too collision-prone (an LLM repo mentioning "flux" once) and must
	// not match; multi-token checkpoints and tensor evidence should.
	if IsSDCheckpoint("", "/m/some-flux-finetune-chat.gguf", nil) {
		t.Fatal("bare 'flux' filename token should not match without tensor evidence")
	}
	if !IsSDCheckpoint("", "/m/qwen-image-2.1-Q6_K.gguf", nil) {
		t.Fatal("qwen-image filename should match without tensor evidence")
	}
}

func TestDetectDiffusionChatNotFlagged(t *testing.T) {
	chat := []string{
		"token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
		"blk.0.ffn_gate.weight", "output.weight", "output_norm.weight",
	}
	isDiff, _ := DetectDiffusionTensors("llama", "Llama-3.1-8B", "/m/llama.gguf", nil, chat)
	if isDiff {
		t.Fatal("chat GGUF flagged as diffusion")
	}
}
