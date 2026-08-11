package instances

import "github.com/openinfer/openinfer-studio/internal/gguf"

func importDetectDiffusion(arch, name, path string, raw map[string]any) (bool, uint32) {
	return gguf.DetectDiffusion(arch, name, path, raw)
}
