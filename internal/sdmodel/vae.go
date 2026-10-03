package sdmodel

import (
	"path/filepath"
	"strings"
)

// IsLTXConvVAEName is a download-ranking hint for upstream's published conv
// variant. It does not replace tensor validation once the file is local.
func IsLTXConvVAEName(path string) bool {
	words := nameWords(strings.ToLower(filepath.Base(path)))
	ext := strings.ToLower(filepath.Ext(path))
	return words["ltx"] && words["video"] && words["vae"] && words["conv"] &&
		(ext == ".gguf" || ext == ".safetensors" && PackedName(path) == "")
}

// VAEIncompatibility returns a known reason this file cannot be used through
// sd.cpp's --vae. An empty result means unknown or no known conflict, not a
// guarantee that an arbitrary model/VAE pairing will work. Tensor evidence
// matters: a renamed or GGUF-converted diffusion decoder is still incompatible.
func VAEIncompatibility(path string, names []string) string {
	if ComponentRole(path, names) == RoleAudioVAE {
		return "that is an audio VAE: --vae needs the video (or image) VAE published with the model"
	}
	var noisyInput, diffusionBlocks bool
	for _, name := range names {
		name = strings.TrimPrefix(strings.TrimPrefix(name, "first_stage_model."), "vae.")
		noisyInput = noisyInput || name == "decoder.conv_in_x_t.weight"
		diffusionBlocks = diffusionBlocks || strings.HasPrefix(name, "decoder.diff_blocks.")
	}
	if noisyInput && diffusionBlocks {
		// Verified against stable-diffusion.cpp 3f8527a docs/ltx2.md.
		return "This is an LTX-2.5 diffusion-decoder VAE, which stable-diffusion.cpp does not implement. Use the convolutional video VAE ltx-2.5-video-vae-conv-bf16.safetensors from Lightricks/LTX-2.5 (vae/). Renaming or requantizing this file cannot change its decoder architecture."
	}
	return ""
}
