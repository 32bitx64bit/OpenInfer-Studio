package quantize

import (
	"testing"

	"github.com/openinfer/openinfer-studio/internal/hardware"
	"github.com/openinfer/openinfer-studio/internal/models"
)

func TestEstimateSizeFromParameters(t *testing.T) {
	src := models.Model{Parameters: 8_000_000_000, SizeBytes: 16 << 30, Quantization: "F16"}
	ft, _ := LookupFType("Q4_K_M")
	est := EstimateSize(src, ft)
	if est < 4<<30 || est > 6<<30 {
		t.Fatalf("8B Q4_K_M estimate %d bytes", est)
	}
}

func TestBuildPreviewBlockers(t *testing.T) {
	src := models.Model{Parameters: 8_000_000_000, SizeBytes: 16 << 30, Quantization: "Q4_K_M", Alias: "m"}
	ft, _ := LookupFType("IQ2_XXS")
	p := BuildPreview(EstimateInput{
		Source: src, FType: ft, DiskFree: 1024, Hardware: &hardware.Info{RAMAvailable: 1 << 20},
	}, Recommendation{FType: "Q4_K_M", Reason: "x", Threads: 3})
	if !p.IMatrixRequired {
		t.Fatal("IQ2 should require imatrix")
	}
	if p.HighPrecision {
		t.Fatal("Q4 source is not high precision")
	}
	if p.DiskOK {
		t.Fatal("tiny disk should block")
	}
	if len(p.Blockers) == 0 {
		t.Fatal("expected disk blocker")
	}
}
