package runtimes

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// FindSibling looks for name (and name.exe on Windows) in the same directory
// as llama-server. Official llama.cpp archives typically ship llama-quantize,
// llama-imatrix, llama-gguf-split, and llama-cli beside the server.
func FindSibling(llamaServerExe, name string) (string, error) {
	if llamaServerExe == "" || name == "" {
		return "", fmt.Errorf("empty sibling lookup")
	}
	dir := filepath.Dir(llamaServerExe)
	candidates := []string{name}
	if runtime.GOOS == "windows" && !strings.HasSuffix(strings.ToLower(name), ".exe") {
		candidates = []string{name + ".exe", name}
	}
	for _, n := range candidates {
		cand := filepath.Join(dir, n)
		if st, err := os.Stat(cand); err == nil && !st.IsDir() {
			return cand, nil
		}
	}
	return "", fmt.Errorf("runtime has no %s next to %s", name, filepath.Base(llamaServerExe))
}

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
	for _, name := range diffusionServerNames {
		if p, err := FindSibling(llamaServerExe, name); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("runtime has no diffusion visual server next to %s (need llama-diffusion-gemma-visual-server from a DiffusionGemma llama.cpp build)", filepath.Base(llamaServerExe))
}
