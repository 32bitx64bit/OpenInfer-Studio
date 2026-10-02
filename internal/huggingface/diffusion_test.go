package huggingface

import "testing"

func TestDetectDiffusion(t *testing.T) {
	cases := []struct {
		name     string
		id, pipe string
		tags     []string
		files    []string
		want     string
	}{
		{
			name:  "SD1.5 pipeline tag",
			id:    "stable-diffusion-v1-5/stable-diffusion-v1-5",
			pipe:  "text-to-image",
			tags:  []string{"diffusers", "stable-diffusion", "text-to-image"},
			files: []string{"v1-5-pruned-emaonly.safetensors"},
			want:  DiffusionImage,
		},
		{
			name:  "SDXL tag set",
			id:    "stabilityai/stable-diffusion-xl-base-1.0",
			pipe:  "text-to-image",
			tags:  []string{"diffusers", "stable-diffusion-xl"},
			files: []string{"sd_xl_base_1.0.safetensors"},
			want:  DiffusionImage,
		},
		{
			name:  "FLUX family weights",
			id:    "black-forest-labs/FLUX.1-dev",
			pipe:  "text-to-image",
			tags:  []string{"diffusers"},
			files: []string{"flux1-dev.safetensors"},
			want:  DiffusionImage,
		},
		{
			name:  "Wan video bundle",
			id:    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
			pipe:  "text-to-video",
			tags:  []string{"diffusers"},
			files: []string{"model_index.json", "transformer/diffusion_pytorch_model.safetensors"},
			want:  DiffusionVideo,
		},
		{
			name:  "LTX family tag-less",
			id:    "Lightricks/LTX-2.3",
			pipe:  "",
			tags:  []string{"diffusers"},
			files: []string{"model_index.json", "transformer.safetensors"},
			want:  DiffusionVideo,
		},
		{
			name:  "plain LLM safetensors is not diffusion",
			id:    "meta-llama/Llama-3.1-8B",
			pipe:  "",
			tags:  []string{"transformers", "safetensors"},
			files: []string{"model.safetensors"},
			want:  DiffusionNone,
		},
		{
			name:  "LLM GGUF never diffusion",
			id:    "bartowski/Llama-3.2-3B-Instruct-GGUF",
			pipe:  "",
			tags:  []string{"gguf", "conversational"},
			files: []string{"Llama-3.2-3B-Instruct-Q4_K_M.gguf"},
			want:  DiffusionNone,
		},
		{
			name:  "draft-only package never diffusion",
			id:    "z-lab/Qwen3-4B-DFlash-GGUF",
			pipe:  "",
			tags:  []string{"gguf", "speculative-decoding"},
			files: []string{"Qwen3-4B-DFlash-Q4_K_M.gguf"},
			want:  DiffusionNone,
		},
		{
			name:  "GGUF-converted SD checkpoint",
			id:    "someone/sd15-q8-gguf",
			pipe:  "text-to-image",
			tags:  []string{"gguf"},
			files: []string{"v1-5-pruned-emaonly.q8_0.gguf"},
			want:  DiffusionImage,
		},
		{
			name:  "tag-less single-file checkpoint",
			id:    "runwayml/stable-diffusion-v1-5",
			pipe:  "",
			tags:  []string{},
			files: []string{"v1-5-pruned-emaonly.safetensors"},
			want:  DiffusionImage,
		},
		{
			// ComfyUI's org repackages diffusion weights with no pipeline tag,
			// no diffusers tag and no model_index.json.
			name:  "comfy-org repackaged video weights, untagged",
			id:    "Comfy-Org/MiniMax-H3",
			pipe:  "",
			tags:  []string{},
			files: []string{"split_files/diffusion_models/minimax_h3_bf16.safetensors", "split_files/vae/minimax_vae.safetensors"},
			want:  DiffusionVideo,
		},
		{
			name:  "comfy-org repackaged image weights",
			id:    "Comfy-Org/Lumina_Image_2.0_Repackaged",
			pipe:  "",
			tags:  []string{},
			files: []string{"split_files/diffusion_models/lumina_2_model_bf16.safetensors"},
			want:  DiffusionImage,
		},
		{
			name:  "comfyui tag with weights",
			id:    "someone/their-video-model",
			pipe:  "",
			tags:  []string{"comfyui"},
			files: []string{"their-video-model.safetensors"},
			want:  DiffusionImage,
		},
		{
			// The tags the repository actually carries: no pipeline tag, no
			// diffusers tag.
			name: "diffusion-single-file + comfyui tags only",
			id:   "someone-else/MiniMax-H3-fp8", pipe: "",
			tags:  []string{"diffusion-single-file", "comfyui", "license:minimax-h3-community-license-agreement"},
			files: []string{"minimax_h3_fp8.safetensors"},
			want:  DiffusionVideo,
		},
		{
			name:  "diffusion-single-file tag alone needs weights",
			id:    "someone/notes",
			pipe:  "",
			tags:  []string{"diffusion-single-file"},
			files: []string{"README.md"},
			want:  DiffusionNone,
		},
		{
			// MiniMax publishes language models too: the name alone says nothing.
			name:  "MiniMax language model GGUF is not a video generator",
			id:    "unsloth/MiniMax-M2-GGUF",
			pipe:  "text-generation",
			tags:  []string{"gguf"},
			files: []string{"MiniMax-M2-Q4_K_M.gguf"},
			want:  DiffusionNone,
		},
		{
			name:  "MiniMax language model safetensors is not a video generator",
			id:    "MiniMaxAI/MiniMax-M2",
			pipe:  "",
			tags:  []string{"transformers"},
			files: []string{"model-00001-of-00130.safetensors"},
			want:  DiffusionNone,
		},
		{
			name:  "comfy-org repo with no weights is not a generator",
			id:    "Comfy-Org/docs",
			pipe:  "",
			tags:  []string{},
			files: []string{"README.md"},
			want:  DiffusionNone,
		},
		{
			name:  "ckpt extension",
			id:    "someone/sd-classic",
			pipe:  "",
			tags:  []string{"stable-diffusion"},
			files: []string{"model.ckpt"},
			want:  DiffusionImage,
		},
	}
	for _, tc := range cases {
		if got := DetectDiffusion(tc.id, tc.pipe, tc.tags, tc.files); got != tc.want {
			t.Errorf("%s: DetectDiffusion(%q) = %q, want %q", tc.name, tc.id, got, tc.want)
		}
	}
}
