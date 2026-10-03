package mediagen

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

func TestHIPGraphWorkaroundUsesModelSignatureAndPreservesOtherBackends(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "arbitrary-name.gguf")
	writeLTXBroadcastGGUF(t, src)
	repaired := filepath.Join(dir, "repaired.gguf")
	if _, err := gguf.RepairLTXBroadcast(t.Context(), src, repaired, nil); err != nil {
		t.Fatal(err)
	}
	unrelated := filepath.Join(dir, "ltx25-name-is-not-proof.gguf")
	if err := os.WriteFile(unrelated, []byte("not an LTX tensor table"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, backend string
		args          []string
		want          bool
	}{
		{"HIP LTX", runtimes.BackendHIP, []string{"--diffusion-model", src}, true},
		{"repaired LTX", runtimes.BackendHIP, []string{"--diffusion-model", repaired}, true},
		{"CUDA LTX", runtimes.BackendCUDA, []string{"--diffusion-model", src}, false},
		{"CPU runtime", runtimes.BackendCPU, []string{"--diffusion-model", src}, false},
		{"CPU compute", runtimes.BackendHIP, []string{"--diffusion-model", src, "--backend", "cpu"}, false},
		{"last backend wins", runtimes.BackendHIP, []string{"--diffusion-model", src, "--backend", "cpu", "--backend", "ROCm0"}, true},
		{"LLM only", runtimes.BackendHIP, []string{"--llm", src}, false},
		{"misleading filename", runtimes.BackendHIP, []string{"--diffusion-model", unrelated}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			env := SDLaunchEnvironment(filepath.Join(dir, "sd-server"), tc.args, tc.backend)
			if got := env["GGML_CUDA_DISABLE_GRAPHS"] == "1"; got != tc.want {
				t.Fatalf("graph workaround = %v, want %v", got, tc.want)
			}
		})
	}
}
