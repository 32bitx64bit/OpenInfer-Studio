package pipeline

import (
	"math"

	"quantlab/core"
	"quantlab/tensorbank"
)

// EvalScratchConfig describes the evaluation data that can coexist with
// quantization artifacts. VocabSize zero means the source did not expose it.
type EvalScratchConfig struct {
	CtxSize           int
	Chunks            int
	VocabSize         uint64
	SourceRepairBytes uint64
	// LogitsFiles is the maximum number of baseline captures that coexist,
	// not the total number of corpora evaluated. Domain holdouts reuse one
	// scratch path sequentially. Zero means one for backward compatibility.
	LogitsFiles int
	// GenerateIMatrix reserves space for a newly generated importance matrix.
	// ReusedImatrixBytes accounts for a known existing matrix without treating
	// it as a second generated artifact.
	GenerateIMatrix    bool
	ReusedImatrixBytes uint64
	// AnchorWorkers is the maximum number of trim+quantize jobs that may
	// coexist. Zero uses the pipeline maximum (2). Every worker holds both a
	// high-precision source subset and its quantized output near completion.
	AnchorWorkers int
	// TransformCopies reserves full payload rewrites. Zero means one copy
	// when the effort profile enables scale-fold or in-place reconstruct
	// (a single job-private GGUF, not two). Explicit values are used as-is.
	TransformCopies int
}

// EstimateScratch returns a conservative peak disk reservation for a run.
//
// Anchor scratch uses the streaming variant bank model with a bounded worker
// pool: each dtype trims the source to its keep-set, --pure quantizes that
// subset, and places a trimmed variant. Later evaluation retains
// candidates and baseline logits, but those stages do not overlap the
// in-flight anchor workers. EstimateScratch therefore returns the maximum
// stage peak rather than summing mutually exclusive peaks.
//
// Anchor and artifact sizes use the bank's exact tensor geometry and GGUF
// metadata/alignment layout.
func EstimateScratch(bank *core.TensorBank, effort Effort, budget uint64, eval EvalScratchConfig) uint64 {
	if bank == nil || len(bank.Tensors) == 0 {
		return saturatingAdd(budget, budget, budget, eval.SourceRepairBytes, estimateIMatrix(nil, eval), 256<<20)
	}
	p, err := EffortFor(effort)
	if err != nil {
		p, _ = EffortFor(EffortProfiled)
	}
	dtypes := estimateDTypes(p)
	var maxAnchor uint64
	for _, d := range dtypes {
		n := estimatedArtifact(bank, d)
		if n > maxAnchor {
			maxAnchor = n
		}
	}
	// A solved candidate is bounded by budget only at the payload level. The
	// largest feasible single-dtype artifact is a safe bound for metadata and
	// any policy-preserved float tensors.
	candidate := maxAnchor
	if budget > candidate {
		candidate = budget + tensorbank.OverheadReserve(bank)
	}
	if eval.Chunks <= 0 {
		eval.Chunks = p.EvalChunks
	}
	workers := eval.AnchorWorkers
	if workers <= 0 || workers > 2 {
		workers = 2
	}
	// Streaming variant bank with up to two parallel subset-anchor jobs.
	// stageQuantize starts the largest dtype first; each job trims to a
	// high-precision subset, then --pure quantizes it. Near completion a
	// worker owns both files. A source subset can be much larger than its
	// quantized anchor, so both terms must be reserved.
	// Accumulated trimmed variants ≈ output ≤ candidate. Per-variant
	// metadata/alignment slack is OverheadReserve.
	var quantizeWorking uint64
	if maxAnchor > 0 {
		sourceArtifact := estimatedSourceArtifact(bank)
		for i := 0; i < workers; i++ {
			quantizeWorking = saturatingAdd(quantizeWorking, sourceArtifact, maxAnchor)
		}
		quantizeWorking = saturatingAdd(quantizeWorking, candidate, tensorbank.OverheadReserve(bank))
	}

	// A job-private source clone (fold + Hadamard/CSK in place) coexists
	// with the library file. Lifecycle cleanup keeps only that extra copy.
	var transforms uint64
	sourceArtifact := estimatedSourceArtifact(bank)
	for i := 0; i < estimateTransformCopies(p, eval); i++ {
		transforms = saturatingAdd(transforms, sourceArtifact)
	}
	common := saturatingAdd(eval.SourceRepairBytes, estimateIMatrix(bank, eval), transforms, 2<<20)

	// Domain holdouts overwrite one shared logits file. Evaluation retains
	// the main validation baseline plus that shared domain file.
	logitsFiles := eval.LogitsFiles
	if logitsFiles <= 0 {
		logitsFiles = 1
	}
	logitsOne := estimateLogits(eval)
	var logits uint64
	for i := 0; i < logitsFiles; i++ {
		logits = saturatingAdd(logits, logitsOne)
	}

	// Bound proof (estimate ≥ true peak at all times). Let output be the
	// assembled manifest bytes; output ≤ maxAnchor ≤ candidate.
	//  - Quantize: workers source subsets + workers quantized outputs +
	//    accumulated trimmed variants.
	//  - Evaluation: variants + candidate + concurrent logits.
	// Source repair, generated imatrix, and the job-private payload clone
	// persist across stages and are included in each peak through common.
	quantizePeak := saturatingAdd(common, quantizeWorking)
	evalPeak := saturatingAdd(common, candidate, candidate, logits)
	if evalPeak > quantizePeak {
		return evalPeak
	}
	return quantizePeak
}

func estimateTransformCopies(p EffortProfile, eval EvalScratchConfig) int {
	if eval.TransformCopies > 0 {
		return eval.TransformCopies
	}
	if p.ScaleFold || p.InPlaceReconstruct {
		return 1
	}
	return 0
}

func estimatedSourceArtifact(bank *core.TensorBank) uint64 {
	if bank == nil {
		return 0
	}
	options := make([]core.TensorOption, 0, len(bank.Tensors))
	for _, t := range bank.Tensors {
		options = append(options, core.TensorOption{TensorName: t.Name, Target: t.DType, Bytes: t.Length})
	}
	if n, ok := tensorbank.PlannedArtifactSize(bank, &core.SelectionManifest{Options: options}); ok {
		return n
	}
	return saturatingAdd(bank.TotalBytes(), tensorbank.OverheadReserve(bank), 1<<20)
}

func estimateIMatrix(bank *core.TensorBank, eval EvalScratchConfig) uint64 {
	if !eval.GenerateIMatrix {
		return eval.ReusedImatrixBytes
	}
	// llama-imatrix accumulates width-dependent statistics. A square F32
	// workspace for the widest input gives an architecture-aware lower bound;
	// retain a substantial floor and payload-derived bound for partial/unknown
	// tensor metadata.
	var width uint64
	var payload uint64
	if bank != nil {
		payload = bank.TotalBytes()
		for _, t := range bank.Tensors {
			if len(t.Shape) > 0 && t.Shape[0] > width {
				width = t.Shape[0]
			}
		}
	}
	var geometry uint64
	if width > 0 && width <= math.MaxUint64/width/4 {
		geometry = width * width * 4
	}
	derived := payload / 32
	reserve := uint64(256 << 20)
	if geometry > reserve {
		reserve = geometry
	}
	if derived > reserve {
		reserve = derived
	}
	return reserve
}

func estimateDTypes(p EffortProfile) []core.DType {
	set := map[core.DType]bool{}
	add := func(ds []core.DType) {
		for _, d := range ds {
			d = d.BaseTensorType()
			if d.IsQuant() {
				if _, ok := d.Geometry(); ok {
					set[d] = true
				}
			}
		}
	}
	if len(p.Candidates) == 0 {
		add(core.QuantDTypes)
	} else {
		add(p.Candidates)
	}
	out := make([]core.DType, 0, len(set))
	for d := range set {
		out = append(out, d)
	}
	return out
}

func estimatedArtifact(bank *core.TensorBank, d core.DType) uint64 {
	options := make([]core.TensorOption, 0, len(bank.Tensors))
	for _, t := range bank.Tensors {
		target, bytes := t.DType, t.Length
		if t.Quantizable() {
			target = d
			if n, ok := target.ExactBytes(t.Elements); ok {
				bytes = n
			}
		}
		options = append(options, core.TensorOption{TensorName: t.Name, Target: target, Bytes: bytes})
	}
	m := &core.SelectionManifest{Options: options}
	if n, ok := tensorbank.PlannedArtifactSize(bank, m); ok {
		return n
	}
	var payload uint64
	for _, o := range options {
		payload = saturatingAdd(payload, o.Bytes)
	}
	return saturatingAdd(payload, tensorbank.OverheadReserve(bank), 1<<20)
}

func estimateLogits(c EvalScratchConfig) uint64 {
	if c.CtxSize > 0 && c.Chunks > 0 && c.VocabSize > 0 {
		n := uint64(c.CtxSize) * uint64(c.Chunks)
		if n <= math.MaxUint64/c.VocabSize/4 {
			n *= c.VocabSize * 4 // float32 logits
			if n > 64<<20 {
				return n
			}
		}
	}
	return 256 << 20
}

func saturatingAdd(values ...uint64) uint64 {
	var total uint64
	for _, n := range values {
		if math.MaxUint64-total < n {
			return math.MaxUint64
		}
		total += n
	}
	return total
}
