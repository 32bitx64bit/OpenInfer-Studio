package mediagen

import (
	"strings"

	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

const ltxHIPGraphWarning = "GPU graph optimization is disabled for LTX on ROCm to avoid a HIP decoder crash. Generation still runs on the GPU; some operations may take longer."

func sdCPUBackend(args []string) bool {
	var backend string
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--backend" {
			backend = args[i+1]
		}
	}
	return backend == "cpu"
}

func ltxHIPGraphWorkaround(args []string, backend string) bool {
	if backend != runtimes.BackendHIP || sdCPUBackend(args) {
		return false
	}
	for _, i := range diffusionGGUFIndexes(args) {
		st, err := gguf.InspectLTXBroadcast(args[i])
		if err == nil && st.Detected {
			return true
		}
	}
	return false
}

// SDLaunchEnvironment is shared by the launcher and load preview. ggml's HIP
// backend supports GGML_CUDA_DISABLE_GRAPHS despite the CUDA name. LTX video
// decode can overflow ROCm's graph-instantiation stack on large graphs, even
// when the same work succeeds through ordinary GPU kernel dispatch.
func SDLaunchEnvironment(exe string, args []string, backend string) map[string]string {
	env := map[string]string{}
	for _, kv := range runtimes.LibPathEnv(exe) {
		if k, v, ok := strings.Cut(kv, "="); ok {
			env[k] = v
		}
	}
	if ltxHIPGraphWorkaround(args, backend) {
		env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
	}
	return env
}
