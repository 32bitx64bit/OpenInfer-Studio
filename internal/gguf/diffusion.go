package gguf

import (
	"path/filepath"
	"strings"
)

// DetectDiffusion reports whether a GGUF is a block-diffusion language model
// (DiffusionGemma and future diffusion-* architectures) rather than an
// autoregressive chat model.
//
// Signals (any one is enough):
//   - general.architecture starts with "diffusion" or contains "diffusion"
//   - diffusion.canvas_length (or <arch>.canvas_length) is present
//   - path/name contains diffusiongemma / diffusion-gemma / "diffusion"
func DetectDiffusion(arch, name, path string, raw map[string]any) (isDiffusion bool, canvasLength uint32) {
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
	switch {
	case strings.Contains(blob, "diffusiongemma"),
		strings.Contains(blob, "diffusion-gemma"),
		strings.Contains(blob, "diffusion_gemma"):
		isDiffusion = true
	case strings.Contains(blob, "diffusion") &&
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

// ApplyDiffusionFlags sets IsDiffusion / CanvasLength on md.
// Safe to call after extract().
func (md *Metadata) ApplyDiffusionFlags(path string) {
	isDiff, canvas := DetectDiffusion(md.Architecture, md.Name, path, md.Raw)
	md.IsDiffusion = isDiff
	md.CanvasLength = canvas
	if isDiff {
		// Diffusion LMs are chat targets, not embedders/drafts; clear those
		// so Library/Chat do not hide them.
		md.SpeculativeDraft = false
		md.IsEmbedding = false
		md.IsReranker = false
		md.ClearMultimodal()
	}
}
