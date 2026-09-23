package gguf

import (
	"path/filepath"
	"strings"
)

// DetectDiffusion reports whether a GGUF is a block-diffusion language
// model (DiffusionGemma) — as opposed to an autoregressive chat model.
//
// This is deliberately narrower than "is this an sd.cpp image/video
// checkpoint": Stable Diffusion / FLUX / Qwen-Image / Wan GGUF conversions
// are a separate concept (see IsSDCheckpoint) and never set IsDiffusion
// here, even though their filenames sometimes contain the literal substring
// "diffusion" ("stable-diffusion-xl…"). Callers that need "should this load
// as an sd.cpp image/video target" must use IsSDCheckpoint /
// models.IsDiffusionModel, never this flag.
//
// Signals (any one is enough):
//   - general.architecture starts with "diffusion" or contains "diffusion"
//   - diffusion.canvas_length (or <arch>.canvas_length) is present
//   - path/name contains diffusiongemma / diffusion-gemma, or a generic
//     "diffusion" token that is not also an SD-checkpoint signature
func DetectDiffusion(arch, name, path string, raw map[string]any) (isDiffusion bool, canvasLength uint32) {
	return DetectDiffusionTensors(arch, name, path, raw, nil)
}

// DetectDiffusionTensors is DetectDiffusion with optional tensor-name
// evidence. Pass the tensor table (names only are read) for files with no
// KV entries — e.g. abenzerps Qwen-Image GGUFs, which ship zero KV pairs.
// tensorNames is used only to keep SD-GGUF checkpoints (see IsSDCheckpoint)
// from tripping the generic block-diffusion-LM filename heuristic below —
// it is never itself a reason to set isDiffusion true.
func DetectDiffusionTensors(arch, name, path string, raw map[string]any, tensorNames []string) (isDiffusion bool, canvasLength uint32) {
	arch = strings.ToLower(strings.TrimSpace(arch))
	if arch != "" && (strings.HasPrefix(arch, "diffusion") || strings.Contains(arch, "diffusion")) {
		isDiffusion = true
	}

	if raw != nil {
		for _, key := range []string{
			"diffusion.canvas_length",
			arch + ".canvas_length",
		} {
			if key == ".canvas_length" {
				continue
			}
			if v, ok := raw[key]; ok {
				isDiffusion = true
				if n, ok := toUint32(v); ok && n > 0 {
					canvasLength = n
				}
			}
		}
	}

	blob := strings.ToLower(strings.TrimSpace(name) + " " + filepath.Base(path))
	// SD-GGUF image/video checkpoints are a separate concept from
	// block-diffusion LMs — see sdSignature/IsSDCheckpoint — but their
	// filenames often also contain the substring "diffusion"
	// ("stable-diffusion-xl…"), so that evidence is computed first and
	// suppresses the generic catch-all below.
	sdCheckpoint := sdSignature(blob, tensorNames)
	switch {
	case strings.Contains(blob, "diffusiongemma"),
		strings.Contains(blob, "diffusion-gemma"),
		strings.Contains(blob, "diffusion_gemma"):
		isDiffusion = true
	case !sdCheckpoint && strings.Contains(blob, "diffusion") &&
		!strings.Contains(blob, "mmproj") &&
		!strings.Contains(blob, "mm-proj"):
		// Generic diffusion LM filename (block-diffusion family).
		isDiffusion = true
	}

	if isDiffusion && canvasLength == 0 {
		canvasLength = 256 // DiffusionGemma default canvas
	}
	return isDiffusion, canvasLength
}

// IsSDCheckpoint reports whether a GGUF is a Stable Diffusion image/video
// generator checkpoint (FLUX / SD / Qwen-Image / Wan GGUF conversions),
// independent of DetectDiffusion's block-diffusion-LM flag. Signals: SD-GGUF
// tensor evidence (img/noise/time/text embedding, transformer_blocks,
// double/single stream blocks) when tensorNames is set, or strong filename
// family tokens (qwen-image, stable-diffusion, hunyuanvideo, …) otherwise.
// Callers deciding "is this an sd.cpp target" (the Library, mediagen) should
// use this — or models.IsDiffusionModel, which layers on sd_family/
// diffusion_kind metadata — never DetectDiffusion's IsDiffusion flag.
func IsSDCheckpoint(name, path string, tensorNames []string) bool {
	blob := strings.ToLower(strings.TrimSpace(name) + " " + filepath.Base(path))
	return sdSignature(blob, tensorNames)
}

// sdWeightKind reports whether a tensor-name slice looks like a diffusion
// image/video transformer ("image", "video", or "").
func sdWeightKind(tensorNames []string) string {
	imgHits, vidHits := 0, 0
	for _, n := range tensorNames {
		lower := strings.ToLower(n)
		switch {
		case strings.HasPrefix(lower, "transformer_blocks.") ||
			strings.HasPrefix(lower, "double_blocks.") ||
			strings.HasPrefix(lower, "single_blocks."):
			imgHits += 2
		case strings.Contains(lower, "img_in.") || strings.HasPrefix(lower, "img_in") ||
			strings.Contains(lower, "img_mlp.") || strings.Contains(lower, "img_attn."):
			imgHits += 2
		case lower == "img_emb.weight" || lower == "img_emb.bias" ||
			strings.Contains(lower, "time_text_embed.") ||
			strings.Contains(lower, "timestep_embedder") ||
			lower == "txt_in.weight" || lower == "txt_in.bias" ||
			strings.Contains(lower, "vector_extractor") ||
			strings.Contains(lower, "guidance_in."):
			imgHits++
		case strings.Contains(lower, "noise_pred.") || strings.Contains(lower, "noise_refiner."):
			imgHits++
		case strings.Contains(lower, "motion_module") ||
			strings.Contains(lower, "temporal_transformer") ||
			strings.Contains(lower, "temporal_attention") ||
			strings.Contains(lower, "wany_") ||
			strings.Contains(lower, "ltx"):
			vidHits += 2
		}
	}
	if vidHits >= 2 {
		return "video"
	}
	// A lone img_attn/img_mlp on a chat GGUF is implausible; transformer
	// blocks plus embedding evidence is the bar.
	if imgHits >= 6 {
		return "image"
	}
	return ""
}

// sdSignature reports whether a filename blob and/or tensor names identify
// an SD-GGUF checkpoint conversion.
func sdSignature(blob string, tensorNames []string) bool {
	kind := sdWeightKind(tensorNames)
	if kind != "" {
		return true
	}
	// No tensor evidence: only strong multi-token checkpoints count, so an
	// LLM repo that mentions "flux" once never matches.
	for _, fam := range []string{
		"stable-diffusion", "stable_diffusion", "qwen-image", "qwen_image",
		"hunyuanvideo", "hunyuan-video",
	} {
		if strings.Contains(blob, fam) {
			return true
		}
	}
	// "flux"/"sdxl"/"wan" alone are too collision-prone without tensors.
	return false
}

// SDKindFromTensors returns "image", "video", or "" for a tensor-name slice.
func SDKindFromTensors(tensorNames []string) string { return sdWeightKind(tensorNames) }

// ApplyDiffusionFlags sets IsDiffusion / CanvasLength on md.
// Safe to call after extract().
func (md *Metadata) ApplyDiffusionFlags(path string) {
	md.ApplyDiffusionFlagsTensors(path, nil)
}

// ApplyDiffusionFlagsTensors is ApplyDiffusionFlags with optional
// tensor-name evidence for KV-less SD-GGUF conversions.
func (md *Metadata) ApplyDiffusionFlagsTensors(path string, tensorNames []string) {
	isDiff, canvas := DetectDiffusionTensors(md.Architecture, md.Name, path, md.Raw, tensorNames)
	// IsDiffusion/CanvasLength are reserved for block-diffusion LMs
	// (DiffusionGemma, loaded through instances.Manager's visual-server
	// path). SD-GGUF image/video checkpoints never set this flag — the
	// Library routes those to the sd-server panel via modality/sd_family/
	// diffusion_kind in metadata_json (see models.IsDiffusionModel), which
	// callers must use instead of this flag.
	md.IsDiffusion = isDiff
	md.CanvasLength = canvas
	if isDiff {
		// Diffusion models are image/video or chat targets, never
		// embedders/drafts; clear those so Library/Chat do not hide them.
		md.SpeculativeDraft = false
		md.IsEmbedding = false
		md.IsReranker = false
		md.ClearMultimodal()
	}
}
