package quantize

import (
	"strings"

	"github.com/openinfer/openinfer-studio/internal/hardware"
	"github.com/openinfer/openinfer-studio/internal/instances"
	"github.com/openinfer/openinfer-studio/internal/models"
)

// Recommendation is a hardware-aware default ftype plus extras.
type Recommendation struct {
	FType            string `json:"ftype"`
	Reason           string `json:"reason"`
	Threads          int    `json:"threads"`
	IMatrixGPULayers int    `json:"imatrix_gpu_layers"`
	QuantizeDraft    bool   `json:"quantize_draft"`
	DraftReason      string `json:"draft_reason,omitempty"`
	DraftFType       string `json:"draft_ftype,omitempty"`
}

func defaultThreads(hw *hardware.Info) int {
	n := 4
	if hw != nil && hw.LogicalCores > 1 {
		n = hw.LogicalCores - 1
	}
	if n < 1 {
		n = 1
	}
	return n
}

func hasGPU(hw *hardware.Info) bool {
	if hw == nil {
		return false
	}
	return hw.CUDA || hw.Metal || hw.Vulkan || hw.HIP || hw.SYCL || len(hw.GPUs) > 0
}

func vramOf(hw *hardware.Info) uint64 {
	if hw == nil {
		return 0
	}
	var n uint64
	for _, g := range hw.GPUs {
		n += g.VRAM
	}
	return n
}

func fitsWeights(hw *hardware.Info, src models.Model, ftypeName string, ctx int) bool {
	t, ok := LookupFType(ftypeName)
	if !ok {
		return false
	}
	estBytes := EstimateSize(src, t)
	proj := fileSize(src.ProjectorPath)
	weights := estBytes
	if proj > 0 && weights > proj {
		weights -= proj
	}
	fit := instances.EstimateMemory(estimateInputFromModel(src, weights, 0, hw, ctx))
	return fit.Fits
}

// Recommend picks a default ftype from detected hardware and the source model.
func Recommend(src models.Model, hw *hardware.Info, catalog []FType, draft *models.Model) Recommendation {
	rec := Recommendation{
		FType:   "Q4_K_M",
		Reason:  "Q4_K_M is the usual balance of size and quality for local inference.",
		Threads: defaultThreads(hw),
	}
	if hasGPU(hw) {
		vram := vramOf(hw)
		if vram > 0 && src.SizeBytes > 0 && uint64(src.SizeBytes) < vram/4 {
			rec.IMatrixGPULayers = 99
		} else if vram > 0 {
			rec.IMatrixGPULayers = 99
		}
	}

	advertised := func(name string) bool {
		if len(catalog) == 0 {
			return true
		}
		want := strings.ToUpper(name)
		for _, t := range catalog {
			if strings.EqualFold(t.Name, want) && t.Advertised && t.AliasOf == "" {
				return true
			}
		}
		return false
	}
	pick := func(name, reason string) bool {
		if advertised(name) && fitsWeights(hw, src, name, src.ContextLength) {
			rec.FType = name
			rec.Reason = reason
			return true
		}
		return false
	}

	switch {
	case pick("Q8_0", "This machine can likely run a Q8_0 of this model at default context — near-full quality."):
	case pick("Q6_K", "Q6_K should fit and stays close to full quality."):
	case pick("Q5_K_M", "Q5_K_M should fit with very good quality."):
	case pick("Q4_K_M", "Q4_K_M is the usual balance of size and quality for local inference."):
	case pick("IQ4_XS", "VRAM is tight. IQ4_XS with an importance matrix is a compact option that often tracks Q4 quality."):
	case pick("Q3_K_M", "VRAM is tight. Q3_K_M with an importance matrix may fit when Q4 does not."):
	default:
		if advertised("IQ4_XS") {
			rec.FType = "IQ4_XS"
			rec.Reason = "Even compact quants may not fully offload this model. IQ4_XS reduces weights; consider lower context or CPU offload."
		} else {
			rec.Reason = "Could not confirm a fit. Q4_K_M is still a reasonable starting point; check the estimate before starting."
		}
	}

	if draft != nil && draft.SizeBytes > 0 && hasGPU(hw) {
		q8t, _ := LookupFType(rec.FType)
		mainEst := EstimateSize(src, q8t)
		fitWithDraft := instances.EstimateMemory(estimateInputFromModel(src, mainEst, draft.SizeBytes, hw, src.ContextLength))
		smaller := draft.SizeBytes / 2
		if smaller < 1<<20 {
			smaller = draft.SizeBytes
		}
		fitWithout := instances.EstimateMemory(estimateInputFromModel(src, mainEst, smaller, hw, src.ContextLength))
		if !fitWithDraft.Fits && fitWithout.Fits {
			rec.QuantizeDraft = true
			rec.DraftFType = moreAggressive(rec.FType)
			rec.DraftReason = "Quantizing the speculative draft would free enough memory to improve offload of the main model."
		}
	}
	return rec
}

func moreAggressive(ftype string) string {
	switch CanonicalFType(ftype) {
	case "Q8_0":
		return "Q6_K"
	case "Q6_K", "Q5_K_M", "Q5_K_S":
		return "Q4_K_M"
	case "Q4_K_M":
		return "Q4_K_S"
	default:
		return "Q4_K_S"
	}
}
