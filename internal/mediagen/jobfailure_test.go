package mediagen

import (
	"strings"
	"testing"
)

func TestExplainJobFailureIncludesGGMLIntermediateAllocation(t *testing.T) {
	log := "[INFO   ] video.cpp:1612 - generate_video 832x480x57\n" +
		"[ERROR  ] ggml - alloc_tensor_range: failed to allocate ROCm0 buffer of size 51118080\n" +
		"[ERROR  ] runner_cache.cpp:202 - ltxav failed to capture graph cut tensor: ggml_runner_cut:ltxav.transformer_blocks.2|vx|\n" +
		"[ERROR  ] ggml_runner.cpp:898 - ltxav segment 4/50 failed during output caching\n" +
		"[ERROR  ] diffusion_engine.cpp:2594 - diffusion model compute failed\n" +
		"[ERROR  ] video.cpp:1709 - sampling failed after 103.75s\n"
	got := explainJobFailure(log, "generate_video returned no results")
	for _, want := range []string{"GPU memory on ROCm0", "48.8 MiB", "832×480, 57 frames", "intermediate tensor cache", "lower Max VRAM budget"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in %q", want, got)
		}
	}
	if strings.Contains(got, "turned on") {
		t.Fatal("working-memory failure blindly recommends offloading weights")
	}
}

func TestExplainJobFailureDistinguishesHostAllocation(t *testing.T) {
	got := explainJobFailure("[ERROR  ] ggml - alloc_tensor_range: failed to allocate CPU buffer of size 1048576", "generation failed")
	if !strings.Contains(got, "system memory") || strings.Contains(got, "Max VRAM") {
		t.Fatalf("host allocation misdiagnosed: %q", got)
	}
}

func TestExplainJobFailureKeepsSpecificNonAllocationError(t *testing.T) {
	got := explainJobFailure("[ERROR  ] ggml - incompatible tensor shape", "unsupported sampler")
	if got != "unsupported sampler" {
		t.Fatalf("specific failure replaced: %q", got)
	}
}
