package mediagen

import (
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

// ROCm's free-memory query can omit VRAM used by the desktop. The runtime's
// default 512 MiB margin did not cover the observed discrepancy on an RX
// 7800 XT. Reserve 2 GiB for automatic HIP launches, while preserving every
// explicit budget (including 0) and runtimes without this advertised flag.
func applySDMemoryHeadroom(args []string, backend string, caps []string, help string) ([]string, string) {
	if backend != runtimes.BackendHIP || !SupportsSDFlag(caps, help, "--max-vram") {
		return args, ""
	}
	if sdCPUBackend(args) {
		return args, ""
	}
	for _, arg := range args {
		if arg == "--max-vram" {
			return args, ""
		}
	}
	return append(args, "--max-vram", "-2"), "Studio reserves 2 GiB of VRAM headroom for ROCm because its free-memory report can omit desktop allocations. Set Max VRAM explicitly to override this automatic reserve; 0 uses the runtime's full default budget."
}
