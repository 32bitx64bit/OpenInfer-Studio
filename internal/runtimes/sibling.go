package runtimes

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

// Diffusion visual-server binary names, preferred first. Official llama.cpp
// archives do not ship these; Unsloth / PR builds place them next to llama-server.
var diffusionServerNames = []string{
	"llama-diffusion-gemma-visual-server",
	"llama-diffusion-visual-server",
}

// FindDiffusionServer looks for a block-diffusion HTTP visual server beside
// the registered llama-server executable. Returns an absolute path or an error
// explaining that a DiffusionGemma-capable runtime is required.
func FindDiffusionServer(llamaServerExe string) (string, error) {
	dir := filepath.Dir(llamaServerExe)
	for _, name := range diffusionServerNames {
		cand := filepath.Join(dir, name)
		if runtime.GOOS == "windows" {
			cand += ".exe"
		}
		if st, err := os.Stat(cand); err == nil && !st.IsDir() {
			return cand, nil
		}
		// Also try without forcing .exe when the stored name already has it.
		alt := filepath.Join(dir, name)
		if alt != cand {
			if st, err := os.Stat(alt); err == nil && !st.IsDir() {
				return alt, nil
			}
		}
	}
	return "", fmt.Errorf("runtime has no diffusion visual server next to %s (need llama-diffusion-gemma-visual-server from a DiffusionGemma llama.cpp build)", filepath.Base(llamaServerExe))
}
