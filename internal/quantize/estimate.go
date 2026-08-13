package quantize

import (
	"strings"

	"github.com/openinfer/openinfer-studio/internal/hardware"
	"github.com/openinfer/openinfer-studio/internal/instances"
	"github.com/openinfer/openinfer-studio/internal/models"
)

// EstimateInput is everything Preview needs besides the ftype catalog entry.
type EstimateInput struct {
	Source        models.Model
	FType         FType
	Hardware      *hardware.Info
	DiskFree      uint64
	DraftBytes    int64
	ProjectorKeep bool
	Context       int
}

// Preview is the synchronous estimate returned by POST /quantize/preview.
type Preview struct {
	EstimatedBytes     int64              `json:"estimated_bytes"`
	SourceBytes        int64              `json:"source_bytes"`
	DeltaBytes         int64              `json:"delta_bytes"`
	QuantizeRAMBytes   int64              `json:"quantize_ram_bytes"`
	DiskFreeBytes      uint64             `json:"disk_free_bytes"`
	DiskOK             bool               `json:"disk_ok"`
	RAMOK              bool               `json:"ram_ok"`
	Fit                instances.Estimate `json:"fit"`
	Warnings           []string           `json:"warnings"`
	Blockers           []string           `json:"blockers"`
	Recommended        string             `json:"recommended_ftype"`
	RecommendReason    string             `json:"recommend_reason"`
	FType              FType              `json:"ftype"`
	ThreadsDefault     int                `json:"threads_default"`
	IMatrixRequired    bool               `json:"imatrix_required"`
	IMatrixRecommended bool               `json:"imatrix_recommended"`
	HighPrecision      bool               `json:"high_precision_source"`
}

func sourceBPW(quant string) float64 {
	if t, ok := LookupFType(quant); ok && t.BPW > 0 {
		return t.BPW
	}
	return 16
}

// EstimateSize projects output bytes from parameter count or file-size scaling.
func EstimateSize(src models.Model, ftype FType) int64 {
	if ftype.Name == "COPY" {
		return src.SizeBytes
	}
	bpw := ftype.BPW
	if bpw <= 0 {
		bpw = 4.89
	}
	if src.Parameters > 0 {
		est := int64(float64(src.Parameters) * bpw / 8)
		// Tokenizer / metadata overhead: ~2–4% on typical models.
		return est + est/25
	}
	if src.SizeBytes > 0 {
		s := sourceBPW(src.Quantization)
		if s <= 0 {
			s = 16
		}
		return int64(float64(src.SizeBytes) * (bpw / s))
	}
	return 0
}

func estimateInputFromModel(m models.Model, weights int64, draft int64, hw *hardware.Info, ctx int) instances.EstimateInput {
	var meta struct {
		BlockCount            uint32   `json:"block_count"`
		HeadCountKV           uint32   `json:"head_count_kv"`
		HeadCountKVLayers     []uint32 `json:"head_count_kv_layers"`
		HeadDim               uint32   `json:"head_dim"`
		ValueDim              uint32   `json:"value_dim"`
		HeadDimSWA            uint32   `json:"head_dim_swa"`
		ValueDimSWA           uint32   `json:"value_dim_swa"`
		SlidingWindow         uint32   `json:"sliding_window"`
		SlidingWindowPattern  []bool   `json:"sliding_window_pattern"`
		SharedKVLayers        uint32   `json:"shared_kv_layers"`
		FullAttentionInterval uint32   `json:"full_attention_interval"`
		SSMStateSize          uint32   `json:"ssm_state_size"`
		SSMInnerSize          uint32   `json:"ssm_inner_size"`
		EmbeddingLength       uint32   `json:"embedding_length"`
		HasVision             bool     `json:"has_vision"`
		HasAudio              bool     `json:"has_audio"`
	}
	_ = jsonUnmarshal(m.Metadata, &meta)
	var vram uint64
	var ram uint64
	gpu := false
	if hw != nil {
		for _, g := range hw.GPUs {
			vram += g.VRAM
		}
		ram = hw.RAMAvailable
		gpu = hw.CUDA || hw.Metal || hw.Vulkan || hw.HIP || hw.SYCL || len(hw.GPUs) > 0
	}
	proj := int64(0)
	if m.ProjectorPath != "" {
		proj = fileSize(m.ProjectorPath)
	}
	off := meta.BlockCount
	if off == 0 && gpu {
		off = 999
	}
	if !gpu {
		off = 0
	}
	if ctx <= 0 {
		ctx = m.ContextLength
	}
	return instances.EstimateInput{
		Weights: weights, DraftWeights: draft, Projector: proj,
		Layers: meta.BlockCount, KVHeads: meta.HeadCountKV,
		HeadCountKVLayers: meta.HeadCountKVLayers,
		HeadDim:           meta.HeadDim, ValueDim: meta.ValueDim,
		HeadDimSWA: meta.HeadDimSWA, ValueDimSWA: meta.ValueDimSWA,
		SlidingWindow: meta.SlidingWindow, SlidingWindowPattern: meta.SlidingWindowPattern,
		SharedKVLayers: meta.SharedKVLayers, FullAttentionInterval: meta.FullAttentionInterval,
		SSMStateSize: meta.SSMStateSize, SSMInnerSize: meta.SSMInnerSize,
		EmbeddingLength: meta.EmbeddingLength,
		Ctx:             ctx, ModelContext: m.ContextLength,
		GPUOffload: gpu, OffloadedLayers: off,
		VRAM: vram, RAM: ram,
		HasVision: meta.HasVision, HasAudio: meta.HasAudio,
		FlashAttention: "auto",
	}
}

func BuildPreview(in EstimateInput, rec Recommendation) Preview {
	p := Preview{
		SourceBytes:        in.Source.SizeBytes,
		FType:              in.FType,
		Recommended:        rec.FType,
		RecommendReason:    rec.Reason,
		ThreadsDefault:     rec.Threads,
		IMatrixRequired:    RequiresIMatrix(in.FType.Name),
		IMatrixRecommended: RecommendsIMatrix(in.FType.Name),
		HighPrecision:      HighPrecision(in.Source.Quantization),
		DiskFreeBytes:      in.DiskFree,
		QuantizeRAMBytes:   in.Source.SizeBytes,
	}
	p.EstimatedBytes = EstimateSize(in.Source, in.FType)
	p.DeltaBytes = p.EstimatedBytes - p.SourceBytes
	need := uint64(p.EstimatedBytes) + uint64(p.EstimatedBytes)/10
	p.DiskOK = in.DiskFree == 0 || in.DiskFree >= need
	if in.Hardware != nil && in.Hardware.RAMAvailable > 0 {
		p.RAMOK = uint64(in.Source.SizeBytes) <= in.Hardware.RAMAvailable
	} else {
		p.RAMOK = true
	}
	weights := p.EstimatedBytes
	if in.Source.ProjectorPath != "" {
		ps := fileSize(in.Source.ProjectorPath)
		if weights > ps {
			weights -= ps
		}
	}
	p.Fit = instances.EstimateMemory(estimateInputFromModel(in.Source, weights, in.DraftBytes, in.Hardware, in.Context))

	if !p.HighPrecision {
		p.Warnings = append(p.Warnings, "Source is already quantized. Requantizing from "+in.Source.Quantization+" can severely reduce quality. Use a F16/BF16/F32/Q8 source, or enable requantize in Advanced.")
	}
	if p.IMatrixRequired {
		p.Warnings = append(p.Warnings, in.FType.Name+" requires an importance matrix. Generate or select one before starting.")
	} else if p.IMatrixRecommended {
		p.Warnings = append(p.Warnings, in.FType.Name+" benefits from an importance matrix for better quality at this size.")
	}
	if in.FType.Experimental {
		p.Warnings = append(p.Warnings, in.FType.Name+" is experimental and may produce poor results.")
	}
	if !p.DiskOK {
		p.Blockers = append(p.Blockers, "Not enough free disk space for the estimated output.")
	}
	if !p.RAMOK {
		p.Warnings = append(p.Warnings, "Quantization typically loads the source into RAM. This machine may not have enough free memory.")
	}
	if strings.Contains(strings.ToLower(in.Source.Architecture), "moe") && in.FType.MoEOnly {
		// ok
	} else if in.FType.MoEOnly {
		p.Warnings = append(p.Warnings, "MXFP4_MOE is intended for Mixture-of-Experts models.")
	}
	return p
}
