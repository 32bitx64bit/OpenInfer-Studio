package sdmodel

import (
	"path/filepath"
	"strconv"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/gguf"
)

// Pipeline component roles understood by sd.cpp, shared by the library
// scanner and the mediagen launcher so they agree on what a file is.
const (
	RoleVAE        = "vae"
	RoleTAESD      = "taesd"
	RoleLLM        = "llm"
	RoleT5XXL      = "t5xxl"
	RoleClipL      = "clip_l"
	RoleClipG      = "clip_g"
	RoleClipVision = "clip_vision"
	RoleTokenizer  = "tokenizer"
	RoleESRGAN     = "esrgan"
	RoleControlNet = "control_net"
	RoleLoRA       = "lora"
)

// ComponentRole classifies a diffusion pipeline weight file into an sd.cpp
// role, or "" when the file looks like a standalone/full checkpoint (or an
// unrelated file) rather than a pipeline component. Directory and basename
// hints are tried first (cheap, and authoritative for oddly-named tensors);
// tensorNames — from TensorNames, may be nil when unavailable — is used as
// a fallback so anonymously-named component files still classify correctly.
func ComponentRole(path string, tensorNames []string) string {
	base := strings.ToLower(filepath.Base(path))
	dir := strings.ToLower(filepath.ToSlash(filepath.Dir(path)))
	if role := roleFromName(base, dir); role != "" {
		return role
	}
	return roleFromTensors(tensorNames)
}

// roleFromName mirrors the location/basename heuristics sd.cpp downloaders
// and ComfyUI-style repos use for pipeline components. LoRA and tokenizer
// checks run first since they can otherwise collide with other tokens
// (a LoRA named "...-vae-fix-lora.safetensors" must not read as a VAE).
func roleFromName(base, dir string) string {
	switch {
	case base == "tokenizer.json" || strings.HasSuffix(base, "tokenizer.json"):
		return RoleTokenizer
	case strings.Contains(base, "lora") || strings.Contains(dir, "lora"):
		return RoleLoRA
	case strings.Contains(base, "taesd") || strings.HasSuffix(base, "tae.safetensors"):
		return RoleTAESD
	case strings.Contains(base, "clip_vision") || strings.Contains(base, "clip-vision") ||
		strings.Contains(dir, "clip_vision"):
		return RoleClipVision
	case strings.Contains(base, "clip_g") || strings.Contains(base, "clip-g"):
		return RoleClipG
	case strings.Contains(base, "clip_l") || strings.Contains(base, "clip-l"):
		return RoleClipL
	case strings.Contains(base, "t5xxl") || strings.Contains(base, "t5-xxl") ||
		strings.Contains(base, "t5_xxl") || strings.Contains(base, "umt5") ||
		strings.Contains(dir, "text_encoders/t5"):
		// Wan uses UMT5-XXL through --t5xxl, not --llm.
		return RoleT5XXL
	case strings.Contains(base, "esrgan") || strings.Contains(base, "upscal") ||
		strings.HasPrefix(base, "2x-") || strings.HasPrefix(base, "4x-") || strings.HasPrefix(base, "8x-"):
		return RoleESRGAN
	case strings.Contains(base, "controlnet") || strings.Contains(base, "control-net") ||
		strings.Contains(base, "control_net") || strings.Contains(dir, "controlnet") ||
		strings.Contains(dir, "control_net") || strings.Contains(dir, "control-net"):
		return RoleControlNet
	case hasVAEWord(base) || strings.Contains(dir, "/vae") ||
		strings.HasSuffix(dir, "vae") || strings.Contains(dir, "vae/") ||
		dirIsVAE(dir) ||
		strings.TrimSuffix(base, filepath.Ext(base)) == "ae":
		// "ae" is how FLUX-family releases name their autoencoder
		// (ae.safetensors); its tensors carry no vae./first_stage_model.
		// prefix, so the name is the only evidence.
		return RoleVAE
	case strings.Contains(base, "qwen3vl") || strings.Contains(base, "qwen2vl") ||
		strings.Contains(base, "qwen2.5") || strings.Contains(base, "qwen_2.5") ||
		strings.Contains(base, "qwen3-vl") || strings.Contains(base, "qwen3_vl") ||
		strings.Contains(base, "mistral") ||
		strings.Contains(dir, "text_encoder") || strings.Contains(base, "text_encoder") ||
		strings.Contains(base, "llm"):
		return RoleLLM
	}
	return ""
}

// hasVAEWord reports whether a file name names a VAE: a "vae" word (ae.vae,
// sdxl_vae, vae-ft-mse) or a name ending in "vae" after a letter (ClearVAE).
// A digit in front ("sd_xl_base_1.0_0.9vae") means a full checkpoint built
// with that VAE, not the VAE itself.
func hasVAEWord(base string) bool {
	for _, w := range strings.FieldsFunc(base, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9')
	}) {
		if w == "vae" {
			return true
		}
		if strings.HasSuffix(w, "vae") {
			if c := w[len(w)-4]; c >= 'a' && c <= 'z' {
				return true
			}
		}
	}
	return false
}

// dirIsVAE reports whether any folder of a path is a VAE folder: vae, or a
// numbered / named sibling such as vae_1_0 or vae_decoder.
func dirIsVAE(dir string) bool {
	for _, seg := range strings.Split(dir, "/") {
		if seg == "vae" || strings.HasPrefix(seg, "vae_") || strings.HasPrefix(seg, "vae-") {
			return true
		}
	}
	return false
}

// nameWords splits a lower-case file name into alphanumeric words.
func nameWords(base string) map[string]bool {
	out := map[string]bool{}
	for _, w := range strings.FieldsFunc(base, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
	}) {
		out[w] = true
	}
	return out
}

// roleFromTensors classifies a component from tensor-name evidence alone,
// for files whose name gives no hint. Thresholds require multiple matching
// tensors so a single incidentally-named weight cannot false-positive.
func roleFromTensors(tensorNames []string) string {
	if len(tensorNames) == 0 {
		return ""
	}
	lora, controlNet, t5, clipLayers, esrgan, vae := 0, 0, 0, 0, 0, 0
	maxClipLayer := -1
	for _, n := range tensorNames {
		lower := strings.ToLower(n)
		switch {
		case strings.Contains(lower, "lora_up") || strings.Contains(lower, "lora_down") ||
			strings.Contains(lower, "lora_a.") || strings.Contains(lower, "lora_b.") ||
			strings.HasSuffix(lower, ".lora_a") || strings.HasSuffix(lower, ".lora_b"):
			lora++
		case strings.HasPrefix(lower, "control_model.") || strings.Contains(lower, "input_hint_block"):
			controlNet++
		case strings.Contains(lower, "relative_attention_bias") ||
			(strings.HasPrefix(lower, "encoder.block.") && strings.Contains(lower, "dense")):
			t5++
		case strings.HasPrefix(lower, "text_model.encoder.layers."):
			clipLayers++
			if idx := layerIndex(lower, "text_model.encoder.layers."); idx > maxClipLayer {
				maxClipLayer = idx
			}
		case strings.Contains(lower, "rrdb") || strings.HasPrefix(lower, "conv_first.") || strings.HasPrefix(lower, "conv_body."):
			esrgan++
		case strings.HasPrefix(lower, "first_stage_model.") || strings.HasPrefix(lower, "vae."):
			vae++
		}
	}
	switch {
	case lora >= 2:
		return RoleLoRA
	case controlNet >= 2:
		return RoleControlNet
	case t5 >= 4:
		return RoleT5XXL
	case clipLayers >= 4:
		// CLIP-L (ViT-L/14) text towers run 12 layers; CLIP-G (bigG) runs
		// 32. 20 splits the difference without needing tensor shapes.
		if maxClipLayer >= 20 {
			return RoleClipG
		}
		return RoleClipL
	case esrgan >= 2:
		return RoleESRGAN
	case vae >= 4 && DiffusionKind(tensorNames) == "":
		// Standalone VAE: encoder/decoder weights with no diffusion
		// transformer signature alongside them (a full checkpoint has both
		// and is handled by IsFullCheckpoint, not this role classifier).
		return RoleVAE
	}
	return ""
}

func layerIndex(lower, prefix string) int {
	rest := strings.TrimPrefix(lower, prefix)
	part, _, _ := strings.Cut(rest, ".")
	n, err := strconv.Atoi(part)
	if err != nil {
		return -1
	}
	return n
}

// DiffusionKind returns "image", "video", or "" for a tensor-name slice,
// covering both GGUF SD conversions (reusing gguf.SDKindFromTensors) and
// safetensors-native layouts: a "model.diffusion_model." prefix (ComfyUI-
// style dumps) is stripped before applying the same DiT/UNet heuristics,
// and the SD1.x/SDXL CompVis UNet layout (input_blocks./output_blocks./
// middle_block.) is recognized directly since GGUF conversions don't use it.
func DiffusionKind(tensorNames []string) string {
	if len(tensorNames) == 0 {
		return ""
	}
	stripped := make([]string, len(tensorNames))
	for i, n := range tensorNames {
		stripped[i] = strings.TrimPrefix(n, "model.diffusion_model.")
	}
	if kind := gguf.SDKindFromTensors(stripped); kind != "" {
		return kind
	}
	unetHits := 0
	for _, n := range stripped {
		lower := strings.ToLower(n)
		if strings.HasPrefix(lower, "input_blocks.") || strings.HasPrefix(lower, "output_blocks.") ||
			strings.HasPrefix(lower, "middle_block.") || strings.HasPrefix(lower, "time_embed.") {
			unetHits++
		}
	}
	if unetHits >= 6 {
		return "image"
	}
	if kind := ditKind(stripped); kind != "" {
		return kind
	}
	// ComfyUI writes every diffusion model it saves under this prefix, and
	// nothing else uses it: enough for a transformer no signature knows yet.
	prefixed := 0
	for _, n := range tensorNames {
		if strings.HasPrefix(n, "model.diffusion_model.") {
			prefixed++
		}
	}
	if prefixed >= 8 {
		return "image"
	}
	return ""
}

// ditKind recognizes diffusion transformers that name their tensors
// blocks.N.* with patch / time / text embedders: Wan-style video models
// (patch_embedding + time_embedding + cross_attn) and DiT-style image models
// (x_embedder, t_embedder, adaLN modulation, final_layer).
func ditKind(names []string) string {
	var patch, timeEmb, cross, xEmb, tEmb, adaln, final int
	for _, n := range names {
		lower := strings.ToLower(n)
		switch {
		case strings.HasPrefix(lower, "patch_embedding."):
			patch++
		case strings.HasPrefix(lower, "time_embedding.") || strings.HasPrefix(lower, "time_projection."):
			timeEmb++
		case strings.HasPrefix(lower, "blocks.") && strings.Contains(lower, ".cross_attn."):
			cross++
		case strings.HasPrefix(lower, "x_embedder."):
			xEmb++
		case strings.HasPrefix(lower, "t_embedder."):
			tEmb++
		case strings.Contains(lower, "adaln_modulation"):
			adaln++
		case strings.HasPrefix(lower, "final_layer."):
			final++
		}
	}
	if patch > 0 && timeEmb > 0 && cross > 0 {
		return "video"
	}
	signals := 0
	for _, c := range []int{xEmb, tEmb, adaln, final} {
		if c > 0 {
			signals++
		}
	}
	if signals >= 3 {
		return "image"
	}
	return ""
}

// LooksLikeLLM reports whether tensor names are a language model's (decoder
// layers, token embeddings, an lm_head). Diffusion transformers and their
// VAEs have none of these; a diffusion pipeline's LLM text encoder does, and
// is a component, never the model.
func LooksLikeLLM(tensorNames []string) bool {
	for _, n := range tensorNames {
		lower := strings.ToLower(n)
		if strings.HasPrefix(lower, "model.layers.") || strings.HasPrefix(lower, "lm_head.") ||
			strings.Contains(lower, "embed_tokens.") || strings.HasPrefix(lower, "transformer.h.") ||
			strings.HasPrefix(lower, "gpt_neox.layers.") || strings.HasPrefix(lower, "language_model.") ||
			lower == "token_embd.weight" {
			return true
		}
	}
	return false
}

// IsFullCheckpoint reports whether a tensor-name slice looks like an
// all-in-one SD1.x/SDXL-style checkpoint: diffusion transformer/UNet
// weights AND embedded VAE weights ("first_stage_model." or "vae."
// prefixed) in the same file. Transformer-only checkpoints (ComfyUI-style
// GGUF dumps, split diffusers weights) return false and need --diffusion-
// model plus a separate --vae.
func IsFullCheckpoint(tensorNames []string) bool {
	if DiffusionKind(tensorNames) == "" {
		return false
	}
	for _, n := range tensorNames {
		lower := strings.ToLower(n)
		if strings.HasPrefix(lower, "first_stage_model.") || strings.HasPrefix(lower, "vae.") {
			return true
		}
	}
	return false
}

// Preference ranks how loadable a component file is for sd.cpp (higher is
// better). GGUF is native and always preferred, regardless of quant tokens
// in its name (a Qwen2.5-VL-7B-Instruct-Q8_0.gguf is still a perfectly
// loadable GGUF). Plain bf16/f16/f32 safetensors load fine, plain fp8 casts
// rank just below them. Exotic quant packs (ComfyUI int8_convrot,
// fp8_scaled, …) abort sd.cpp's Linear loader and rank last so a usable
// sibling wins whenever one exists.
//
// Shared by the mediagen launcher and the Hugging Face download grouper so
// both pick the same file when a repo offers several variants.
func Preference(path string) int {
	base := strings.ToLower(filepath.Base(path))
	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".gguf" {
		return 100
	}
	words := nameWords(base)
	if words["scaled"] || words["convrot"] || words["int8"] || words["w8a8"] ||
		words["nf4"] || words["fp4"] || words["nvfp4"] || words["int4"] || words["svdq"] {
		return 5
	}
	// "fp16" does not contain "f16", and ComfyUI repos spell it fp16.
	if strings.Contains(base, "bf16") || strings.Contains(base, "f16") ||
		strings.Contains(base, "fp16") || strings.Contains(base, "f32") ||
		strings.Contains(base, "fp32") {
		return 80
	}
	// Plain (unscaled) fp8 casts carry no side tensors, so they rank under
	// float files but above packs that need scales.
	if words["fp8"] || words["float8"] || words["e4m3"] || words["e4m3fn"] || words["e5m2"] {
		return 40
	}
	return 50
}
