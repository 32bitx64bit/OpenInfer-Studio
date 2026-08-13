package quantize

import (
	"testing"

	"github.com/openinfer/openinfer-studio/internal/hardware"
	"github.com/openinfer/openinfer-studio/internal/models"
)

func TestRecommendPrefersQ8WhenVRAMIsHuge(t *testing.T) {
	src := models.Model{
		SizeBytes: 2 << 30, Parameters: 1_000_000_000, Quantization: "F16",
		ContextLength: 4096, Architecture: "llama",
	}
	hw := &hardware.Info{
		LogicalCores: 8, RAMAvailable: 64 << 30, RAMTotal: 64 << 30,
		CUDA: true, GPUs: []hardware.GPU{{Name: "gpu", VRAM: 48 << 30}},
	}
	rec := Recommend(src, hw, nil, nil)
	if rec.FType != "Q8_0" && rec.FType != "Q6_K" {
		t.Fatalf("expected near-lossless, got %s (%s)", rec.FType, rec.Reason)
	}
	if rec.Threads != 7 {
		t.Errorf("threads=%d", rec.Threads)
	}
}

func TestRecommendFallsBackWhenNothingFits(t *testing.T) {
	src := models.Model{SizeBytes: 80 << 30, Parameters: 70_000_000_000, Quantization: "F16", ContextLength: 8192}
	hw := &hardware.Info{LogicalCores: 4, RAMAvailable: 8 << 30, CUDA: true, GPUs: []hardware.GPU{{VRAM: 4 << 30}}}
	rec := Recommend(src, hw, nil, nil)
	if rec.FType == "" {
		t.Fatal("empty recommendation")
	}
}

func TestMoreAggressive(t *testing.T) {
	if moreAggressive("Q8_0") != "Q6_K" {
		t.Fatal(moreAggressive("Q8_0"))
	}
	if moreAggressive("Q4_K_M") != "Q4_K_S" {
		t.Fatal(moreAggressive("Q4_K_M"))
	}
}
