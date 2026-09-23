// Diffusion model detection for Hugging Face Discover.
//
// Stable Diffusion / FLUX / Wan / LTX style image/video generators live in
// safetensors / ckpt / pt single-file checkpoints or diffusers bundles
// (model_index.json), not GGUF. This file keeps that signal separate from
// the LLM audio/vision heuristics in modality.go so text models are never
// mislabeled as generators.
package huggingface

import (
	"path/filepath"
	"strings"
)

// DiffusionKind values returned by DetectDiffusion.
const (
	DiffusionNone  = ""
	DiffusionImage = "image"
	DiffusionVideo = "video"
	DiffusionBoth  = "both"
)

// Supported weight extensions for stable-diffusion.cpp:
// https://leejet.github.io/stable-diffusion.cpp (model docs list
// .ckpt / .safetensors / .gguf; server also accepts .pt / .pth).
func isDiffusionWeight(path string) bool {
	lower := strings.ToLower(path)
	for _, ext := range []string{".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin"} {
		if strings.HasSuffix(lower, ext) {
			return true
		}
	}
	return false
}

// imagePipelineTags are HF pipeline_tag values that produce still images.
var imagePipelineTags = map[string]bool{
	"text-to-image": true, "image-to-image": true,
	"unconditional-image-generation": true, "image-inpainting": true,
}

// videoPipelineTags are HF pipeline_tag values that produce video.
var videoPipelineTags = map[string]bool{
	"text-to-video": true, "image-to-video": true, "video-to-video": true,
}

// diffusionLibraryTags mark repos that use the diffusers ecosystem even when
// the pipeline_tag is missing or generic.
var diffusionLibraryTags = map[string]bool{
	"diffusers": true, "stable-diffusion": true, "stable-diffusion-xl": true,
	"flux": true, "text-to-image": true, "text-to-video": true,
	"image-to-video": true, "unconditional-image-generation": true,
}

// diffusionFamilyHints are repo-id / filename tokens for generator families.
// Checked as substrings on the lowercased id/blob; deliberately specific so
// LLM repos (llama, qwen, gemma, …) never match.
var diffusionFamilyHints = []string{
	"stable-diffusion", "stable_diffusion", "sdxl", "sd-xl", "sd1.", "sd-1.",
	"sd2.", "sd-2.", "sd3", "sd-3", "sdxl-turbo", "sd-turbo",
	"flux", "chroma", "qwen-image", "z-image", "ideogram", "krea",
	"wan2", "wan-2", "wanx", "ltx", "hunyuanvideo", "hunyuan-video",
	"minimax", "hailuo", "mochi", "cogvideo", "svd", "animatediff",
	"controlnet", "t2i-adapter",
}

// videoFamilyHints narrow a diffusion repo to video generation.
var videoFamilyHints = []string{
	"wan2", "wan-2", "wanx", "ltx", "hunyuanvideo", "hunyuan-video",
	"minimax", "hailuo", "mochi", "cogvideo", "text-to-video", "image-to-video",
	"video-to-video", "svd", "animatediff", "motion-module", "motion_module",
}

// DetectDiffusion reports whether a Hugging Face repo is an image/video
// generator for stable-diffusion.cpp and which modality it targets.
//
// Signals (any strong one is enough):
//   - pipeline_tag / tags naming an image/video generation task
//   - diffusers library tags with weight files present
//   - model_index.json (diffusers bundle marker)
//   - generator family tokens in the repo id + a supported weight file
//
// Returns DiffusionImage, DiffusionVideo, DiffusionBoth, or "" (not a
// generator). Draft-only GGUF sidecar packages and pure LLM repos return "".
func DetectDiffusion(repoID, pipelineTag string, tags []string, filePaths []string) string {
	lowerID := strings.ToLower(strings.TrimSpace(repoID))
	lowerPipe := strings.ToLower(strings.TrimSpace(pipelineTag))
	tagSet := map[string]bool{}
	for _, t := range tags {
		tagSet[strings.ToLower(strings.TrimSpace(t))] = true
	}

	// Never label speculative draft / speculator sidecar packages.
	if isDraftOnlyPackage(lowerID, filePaths) {
		return DiffusionNone
	}

	imagePipe := imagePipelineTags[lowerPipe]
	videoPipe := videoPipelineTags[lowerPipe]
	imageTag, videoTag := false, false
	for t := range tagSet {
		if imagePipelineTags[t] {
			imageTag = true
		}
		if videoPipelineTags[t] {
			videoTag = true
		}
	}
	libTag := false
	for t := range tagSet {
		if diffusionLibraryTags[t] {
			libTag = true
			break
		}
	}

	hasWeight := false
	hasIndex := false
	weightBlob := ""
	for _, p := range filePaths {
		base := strings.ToLower(filepath.Base(p))
		if base == "model_index.json" {
			hasIndex = true
		}
		if isDiffusionWeight(p) {
			// Skip GGUF LLM trunks: a .gguf weight only counts when the repo
			// already shows a diffusion signal (pipeline/tag/bundle/family).
			// Non-GGUF weights (.safetensors/.ckpt/…) count directly.
			if strings.HasSuffix(strings.ToLower(p), ".gguf") {
				weightBlob += " " + base
				continue
			}
			hasWeight = true
			weightBlob += " " + base
		}
	}
	// GGUF-converted diffusion checkpoints (sd.cpp --convert output) count
	// when a diffusion signal is present.
	hasGGUFWeight := false
	for _, p := range filePaths {
		if strings.HasSuffix(strings.ToLower(p), ".gguf") {
			hasGGUFWeight = true
			break
		}
	}

	familyHit := false
	for _, h := range diffusionFamilyHints {
		if strings.Contains(lowerID, h) || strings.Contains(weightBlob, strings.ReplaceAll(h, "-", "_")) ||
			strings.Contains(weightBlob, strings.ReplaceAll(h, "_", "-")) || strings.Contains(weightBlob, h) {
			familyHit = true
			break
		}
	}

	videoHint := false
	blob := lowerID + " " + weightBlob + " " + strings.Join(tags, " ") + " " + lowerPipe
	for _, h := range videoFamilyHints {
		if strings.Contains(blob, h) {
			videoHint = true
			break
		}
	}

	// Strong pipeline signals decide immediately.
	if imagePipe && videoPipe {
		return DiffusionBoth
	}
	if videoPipe {
		return DiffusionVideo
	}
	if imagePipe {
		if videoHint {
			return DiffusionBoth
		}
		return DiffusionImage
	}
	if imageTag && videoTag {
		return DiffusionBoth
	}
	if videoTag {
		return DiffusionVideo
	}
	if imageTag {
		// e.g. tag text-to-image on a video-capable repo (Wan t2i + t2v).
		if videoHint {
			return DiffusionBoth
		}
		return DiffusionImage
	}

	// Diffusers bundle marker with weights is a generator even without tags.
	if hasIndex && (hasWeight || hasGGUFWeight) {
		if videoHint {
			return DiffusionVideo
		}
		return DiffusionImage
	}

	// Library tags need weight evidence so transformer safetensors repos
	// (LLM .safetensors) are not mislabeled.
	if libTag && (hasWeight || hasIndex) {
		if videoHint {
			return DiffusionVideo
		}
		if imageTag {
			return DiffusionImage
		}
		// Bare "diffusers" library tag: image unless video tokens present.
		return DiffusionImage
	}

	// Family tokens + a supported weight file (covers tag-less single-file
	// checkpoints like v1-5-pruned-emaonly.safetensors).
	if familyHit && (hasWeight || hasGGUFWeight || hasIndex) {
		if videoHint {
			return DiffusionVideo
		}
		return DiffusionImage
	}

	return DiffusionNone
}

// IsDiffusionWeight reports whether a repo file can back an sd.cpp load.
func IsDiffusionWeight(path string) bool { return isDiffusionWeight(path) }

// DiffusionLabel returns a short UI label for a diffusion kind.
func DiffusionLabel(kind string) string {
	switch kind {
	case DiffusionImage:
		return "image"
	case DiffusionVideo:
		return "video"
	case DiffusionBoth:
		return "image+video"
	default:
		return ""
	}
}
