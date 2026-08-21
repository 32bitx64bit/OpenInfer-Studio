package pipeline

import (
	"testing"

	"quantlab/core"
	"quantlab/tensorbank"
)

func estimateTestBank(kv, alignment uint64) *core.TensorBank {
	return &core.TensorBank{
		SourcePath: "source.gguf", KVMetadataLen: kv, Alignment: alignment,
		Tensors: []core.TensorDesc{
			{Name: "blk.0.attn.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Elements: 256 * 256, Length: 256 * 256 * 2},
			{Name: "blk.0.norm.weight", DType: core.DTypeF32, Shape: []uint64{256}, Elements: 256, Length: 256 * 4},
		},
	}
}

func TestEstimateScratchEffortAndRepair(t *testing.T) {
	bank := estimateTestBank(128, 32)
	eval := EvalScratchConfig{CtxSize: 512, VocabSize: 1 << 20}
	fast := EstimateScratch(bank, EffortFast, 64<<20, eval)
	profiled := EstimateScratch(bank, EffortProfiled, 64<<20, eval)
	deep := EstimateScratch(bank, EffortDeep, 64<<20, eval)
	if !(fast < profiled && profiled < deep) {
		t.Fatalf("effort scratch fast/profiled/deep = %d/%d/%d", fast, profiled, deep)
	}
	withRepair := EstimateScratch(bank, EffortFast, 64<<20, EvalScratchConfig{SourceRepairBytes: 12345})
	withoutRepair := EstimateScratch(bank, EffortFast, 64<<20, EvalScratchConfig{})
	if withRepair-withoutRepair != 12345 {
		t.Fatalf("repair reserve = %d, want 12345", withRepair-withoutRepair)
	}
}

func TestEstimateScratchAccountsForMetadataAndAlignment(t *testing.T) {
	plain := EstimateScratch(estimateTestBank(1, 32), EffortFast, 0, EvalScratchConfig{})
	heavy := EstimateScratch(estimateTestBank(1<<20, 4096), EffortFast, 0, EvalScratchConfig{})
	if heavy <= plain {
		t.Fatalf("metadata/alignment estimate %d <= plain %d", heavy, plain)
	}
}

func TestEstimateScratchAccountsForConcurrentHoldoutLogits(t *testing.T) {
	bank := estimateTestBank(128, 32)
	cfg := EvalScratchConfig{CtxSize: 512, Chunks: 4, VocabSize: 32000, LogitsFiles: 1}
	one := EstimateScratch(bank, EffortFast, 64<<20, cfg)
	cfg.LogitsFiles = 3
	three := EstimateScratch(bank, EffortFast, 64<<20, cfg)
	want := uint64(2) * estimateLogits(cfg)
	if three-one != want {
		t.Fatalf("additional logits reserve = %d, want %d", three-one, want)
	}
}

func TestEstimateScratchReservesGeneratedIMatrixButNotNewOneForReuse(t *testing.T) {
	bank := estimateTestBank(128, 32)
	plain := EstimateScratch(bank, EffortFast, 64<<20, EvalScratchConfig{})
	generated := EstimateScratch(bank, EffortFast, 64<<20, EvalScratchConfig{GenerateIMatrix: true})
	reused := EstimateScratch(bank, EffortFast, 64<<20, EvalScratchConfig{ReusedImatrixBytes: 4096})
	if generated-plain < 256<<20 {
		t.Fatalf("generated imatrix reserve = %d, want at least 256 MiB", generated-plain)
	}
	if reused-plain != 4096 {
		t.Fatalf("reused imatrix reserve = %d, want known file size", reused-plain)
	}
}

// realisticEstimateBank models a multi-layer model with multiple quantizable
// tensors so that 4 distinct dtypes produce visibly different anchor sizes.
func realisticEstimateBank() *core.TensorBank {
	var tensors []core.TensorDesc
	for blk := 0; blk < 8; blk++ {
		tensors = append(tensors,
			core.TensorDesc{
				Name: "blk." + itoa(blk) + ".attn.weight", DType: core.DTypeF16,
				Shape: []uint64{256, 256}, Elements: 256 * 256, Length: 256 * 256 * 2,
			},
			core.TensorDesc{
				Name: "blk." + itoa(blk) + ".ffn.weight", DType: core.DTypeF16,
				Shape: []uint64{512, 256}, Elements: 512 * 256, Length: 512 * 256 * 2,
			},
		)
	}
	tensors = append(tensors, core.TensorDesc{
		Name: "output.norm", DType: core.DTypeF32,
		Shape: []uint64{256}, Elements: 256, Length: 256 * 4,
	})
	return &core.TensorBank{
		SourcePath: "source.gguf", KVMetadataLen: 512, Alignment: 32, Tensors: tensors,
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [8]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// largeEstimateBank models a ~57 GiB BF16 source as pure tensor metadata (no
// payload is materialized): 8 quantizable 2D weight tensors of 3.84e9 elements
// each, plus a small F32 norm. The geometry produces realistic per-dtype
// full-anchor sizes from core.ExactBytes (Q3_K≈12.3 GiB, Q5_K≈19.7 GiB,
// Q6_K≈23.5 GiB, Q8_0≈30.4 GiB), so anchors dominate the eval/logits reserves
// and the streaming-variant-bank reduction is observable. A TensorBank holds
// only descriptors, so a 57 GiB-scale bank costs a few KiB of memory.
func largeEstimateBank() *core.TensorBank {
	const cols = uint64(15000000)
	const per = 256 * cols // 3.84e9 elements per tensor
	var tensors []core.TensorDesc
	for blk := 0; blk < 8; blk++ {
		tensors = append(tensors, core.TensorDesc{
			Name: "blk." + itoa(blk) + ".attn.weight", DType: core.DTypeBF16,
			Shape: []uint64{256, cols}, Elements: per, Length: per * 2,
		})
	}
	tensors = append(tensors, core.TensorDesc{
		Name: "output.norm", DType: core.DTypeF32,
		Shape: []uint64{256}, Elements: 256, Length: 256 * 4,
	})
	return &core.TensorBank{SourcePath: "source.gguf", KVMetadataLen: 512, Alignment: 32, Tensors: tensors}
}

// oldEstimateScratch reproduces the previous per-dtype-full-anchor formula
// for before/after comparison.
func oldEstimateScratch(bank *core.TensorBank, effort Effort, budget uint64, eval EvalScratchConfig) uint64 {
	if bank == nil || len(bank.Tensors) == 0 {
		return saturatingAdd(budget, budget, budget, eval.SourceRepairBytes, estimateIMatrix(nil, eval), 256<<20)
	}
	p, err := EffortFor(effort)
	if err != nil {
		p, _ = EffortFor(EffortProfiled)
	}
	dtypes := estimateDTypes(p)
	var anchors, largest uint64
	for _, d := range dtypes {
		n := estimatedArtifact(bank, d)
		anchors = saturatingAdd(anchors, n)
		if n > largest {
			largest = n
		}
	}
	candidate := largest
	if budget > candidate {
		candidate = budget + tensorbank.OverheadReserve(bank)
	}
	if eval.Chunks <= 0 {
		eval.Chunks = p.EvalChunks
	}
	return saturatingAdd(anchors, candidate, candidate, eval.SourceRepairBytes, estimateIMatrix(bank, eval), estimateLogits(eval), 2<<20)
}

// TestEstimateScratchStreamingVariantBankReduction proves the streaming
// variant bank estimator remains below the previous per-dtype-full-anchor
// formula while honestly reserving both high-precision source subsets and
// their quantized outputs for two in-flight workers.
//
// EffortProfiled explores the full candidate lattice (~19 distinct base dtypes
// via estimateDTypes). The old formula reserved one full-model anchor per
// dtype (their sum), while the parallel streaming flow holds at most two
// source subsets, two quantized outputs, and accumulated trimmed variants.
func TestEstimateScratchStreamingVariantBankReduction(t *testing.T) {
	bank := largeEstimateBank()
	eval := EvalScratchConfig{CtxSize: 512, Chunks: 4, VocabSize: 32000}
	budget := uint64(0) // target-bpw=0: candidate = largest anchor
	old := oldEstimateScratch(bank, EffortProfiled, budget, eval)
	newEst := EstimateScratch(bank, EffortProfiled, budget, eval)
	if old == 0 {
		t.Fatal("old estimate is zero")
	}
	reduction := float64(old-newEst) / float64(old)
	if reduction <= 0 {
		t.Fatalf("scratch estimate did not improve (old=%d new=%d)", old, newEst)
	}
	t.Logf("streaming variant bank: old=%d new=%d reduction=%.1f%%", old, newEst, reduction*100)
}

// TestEstimateScratchPeakFormula asserts the estimate takes the maximum of
// the quantize and search/evaluation stage peaks rather than summing them.
func TestEstimateScratchPeakFormula(t *testing.T) {
	bank := realisticEstimateBank()
	eval := EvalScratchConfig{CtxSize: 512, Chunks: 4, VocabSize: 32000}
	budget := uint64(0)
	got := EstimateScratch(bank, EffortProfiled, budget, eval)

	p, _ := EffortFor(EffortProfiled)
	dtypes := estimateDTypes(p)
	var maxAnchor uint64
	for _, d := range dtypes {
		n := estimatedArtifact(bank, d)
		if n > maxAnchor {
			maxAnchor = n
		}
	}
	candidate := maxAnchor
	overhead := tensorbank.OverheadReserve(bank)
	sourceArtifact := estimatedSourceArtifact(bank)
	// Each worker owns a source subset and quantized output near completion.
	quantizeWorking := saturatingAdd(sourceArtifact, sourceArtifact, maxAnchor, maxAnchor, candidate, overhead)
	common := saturatingAdd(eval.SourceRepairBytes, estimateIMatrix(bank, eval), 2<<20)
	if p.ScaleFold || p.InPlaceReconstruct {
		common = saturatingAdd(common, sourceArtifact)
	}
	quantizePeak := saturatingAdd(common, quantizeWorking)
	logitsOne := estimateLogits(eval)
	evalPeak := saturatingAdd(common, candidate, candidate, logitsOne)
	want := quantizePeak
	if evalPeak > want {
		want = evalPeak
	}
	if got != want {
		t.Fatalf("peak = %d, want %d (source=%d maxAnchor=%d candidate=%d overhead=%d)", got, want, sourceArtifact, maxAnchor, candidate, overhead)
	}
}

// TestEstimateScratchBeforeAfter57GBCase computes the before/after scratch
// the estimator produces for the 57 GiB BF16 / 4-anchor case from the problem
// statement, using a ~57 GiB metadata bank whose geometry yields the stated
// per-dtype full-anchor sizes. The new estimate includes two full-size source
// subset bounds while avoiding a sum of mutually exclusive stage peaks.
func TestEstimateScratchBeforeAfter57GBCase(t *testing.T) {
	const gi = float64(1 << 30)
	bank := largeEstimateBank()
	eval := EvalScratchConfig{CtxSize: 2048, Chunks: 8, VocabSize: 128256}
	budget := uint64(0)
	old := oldEstimateScratch(bank, EffortDeep, budget, eval)
	newEst := EstimateScratch(bank, EffortDeep, budget, eval)
	if old == 0 || newEst == 0 {
		t.Fatal("zero estimate")
	}
	reduction := float64(old-newEst) / float64(old)
	for _, d := range []core.DType{core.DTypeQ3_K, core.DTypeQ5_K_T, core.DTypeQ6_K, core.DTypeQ8_0} {
		t.Logf("  anchor %s = %.2f GiB", d, float64(estimatedArtifact(bank, d))/gi)
	}
	t.Logf("57GB case: old=%.1f GiB new=%.1f GiB reduction=%.1f%%", float64(old)/gi, float64(newEst)/gi, reduction*100)
	if reduction <= 0 {
		t.Fatalf("estimate did not improve (old=%d new=%d)", old, newEst)
	}
}

// TestEstimateScratch57GBFourAnchorCase asserts the honest streaming
// relationship for the concrete 57 GiB / 4-dtype manifest (Q3_K, Q5_K, Q6_K,
// Q8_0; 13.5 GiB output) under the parallel source-subset bound.
//
// The estimator cannot know the solved manifest ahead of time, so it bounds
// each source subset by the source artifact and each quantized output by the
// largest anchor.
func TestEstimateScratch57GBFourAnchorCase(t *testing.T) {
	const gi = float64(1 << 30)
	bank := largeEstimateBank()
	four := []core.DType{core.DTypeQ3_K, core.DTypeQ5_K_T, core.DTypeQ6_K, core.DTypeQ8_0}
	var sumAnchors, maxAnchor uint64
	for _, d := range four {
		n := estimatedArtifact(bank, d)
		t.Logf("  anchor %s = %.2f GiB", d, float64(n)/gi)
		sumAnchors += n
		if n > maxAnchor {
			maxAnchor = n
		}
	}
	output := uint64(13.5 * (1 << 30))
	sourceArtifact := estimatedSourceArtifact(bank)
	if output > maxAnchor {
		t.Fatalf("output %.1f GiB > maxAnchor %.1f GiB; case is inconsistent", float64(output)/gi, float64(maxAnchor)/gi)
	}
	serialPeak := sourceArtifact + maxAnchor + output
	parallelPeak := 2*sourceArtifact + 2*maxAnchor + output
	if parallelPeak < serialPeak {
		t.Fatalf("parallel peak %.1f GiB < serial peak %.1f GiB", float64(parallelPeak)/gi, float64(serialPeak)/gi)
	}
	// Estimator bound (candidate = maxAnchor for budget 0): quantize holds
	// two source/output worker pairs plus variants; search can hold variants,
	// a base candidate, a sparse promotion shard, and an evaluated candidate.
	quantizeBound := 2*sourceArtifact + 3*maxAnchor
	searchBound := 4 * maxAnchor
	estimatorBound := quantizeBound
	if searchBound > estimatorBound {
		estimatorBound = searchBound
	}
	if estimatorBound < parallelPeak {
		t.Fatalf("estimator bound %.1f GiB < true parallel peak %.1f GiB", float64(estimatorBound)/gi, float64(parallelPeak)/gi)
	}
	t.Logf("4-anchor case: serial=%.1f GiB parallel=%.1f GiB estimatorBound=%.1f GiB oldAnchorSum=%.1f GiB",
		float64(serialPeak)/gi, float64(parallelPeak)/gi, float64(estimatorBound)/gi, float64(sumAnchors)/gi)
}

func serialStreamingScratch(bank *core.TensorBank, effort Effort, budget uint64, eval EvalScratchConfig) uint64 {
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
	candidate := maxAnchor
	if budget > candidate {
		candidate = budget + tensorbank.OverheadReserve(bank)
	}
	if eval.Chunks <= 0 {
		eval.Chunks = p.EvalChunks
	}
	var anchors uint64
	if maxAnchor > 0 {
		anchors = saturatingAdd(estimatedSourceArtifact(bank), maxAnchor, candidate, tensorbank.OverheadReserve(bank))
	}
	common := saturatingAdd(eval.SourceRepairBytes, estimateIMatrix(bank, eval), 2<<20)
	if p.ScaleFold || p.InPlaceReconstruct {
		common = saturatingAdd(common, estimatedSourceArtifact(bank))
	}
	quantizePeak := saturatingAdd(common, anchors)
	logitsFiles := eval.LogitsFiles
	if logitsFiles <= 0 {
		logitsFiles = 1
	}
	logitsOne := estimateLogits(eval)
	var logits uint64
	for i := 0; i < logitsFiles; i++ {
		logits = saturatingAdd(logits, logitsOne)
	}
	evalPeak := saturatingAdd(common, candidate, candidate, logits)
	if evalPeak > quantizePeak {
		return evalPeak
	}
	return quantizePeak
}

func TestEstimateScratchParallelPeakAtLeastSerial(t *testing.T) {
	bank := realisticEstimateBank()
	eval := EvalScratchConfig{CtxSize: 512, Chunks: 4, VocabSize: 32000}
	got := EstimateScratch(bank, EffortProfiled, 0, eval)
	serial := serialStreamingScratch(bank, EffortProfiled, 0, eval)
	if got < serial {
		t.Fatalf("parallel peak %d < serial peak %d", got, serial)
	}
	if got == ^uint64(0) {
		t.Fatal("estimate overflowed to MaxUint64")
	}
	large := EstimateScratch(largeEstimateBank(), EffortDeep, 0, eval)
	if large < serialStreamingScratch(largeEstimateBank(), EffortDeep, 0, eval) {
		t.Fatal("large-bank parallel peak < serial")
	}
}

func TestEstimateScratchDoesNotOverflow(t *testing.T) {
	bank := estimateTestBank(128, 32)
	got := EstimateScratch(bank, EffortFast, ^uint64(0), EvalScratchConfig{SourceRepairBytes: ^uint64(0)})
	if got != ^uint64(0) {
		t.Fatalf("saturating estimate = %d, want MaxUint64", got)
	}
}

func TestEstimateScratchDeepLargeVocabDoesNotSumExclusivePeaks(t *testing.T) {
	const gi = uint64(1 << 30)
	bank := largeEstimateBank()
	source := estimatedSourceArtifact(bank)
	cfg := EvalScratchConfig{
		CtxSize:           4096,
		Chunks:            8,
		VocabSize:         248320,
		LogitsFiles:       3,
		SourceRepairBytes: source + source/10,
	}
	got := EstimateScratch(bank, EffortDeep, uint64(13.5*float64(gi)), cfg)
	// 400 GiB is well above the honest stage peak (quantize workers + one
	// job-private payload clone) and well below summing the exclusive
	// quantize and evaluation peaks.
	if got >= 400*gi {
		t.Fatalf("deep large-vocab peak still sums exclusive stages: %.1f GiB", float64(got)/float64(gi))
	}
	t.Logf("deep large-vocab stage peak: %.1f GiB", float64(got)/float64(gi))
}

func TestSplitThreads(t *testing.T) {
	if splitThreads(8, 2) != 4 {
		t.Fatalf("splitThreads(8,2) = %d, want 4", splitThreads(8, 2))
	}
	if splitThreads(1, 2) != 1 {
		t.Fatalf("splitThreads(1,2) = %d, want 1", splitThreads(1, 2))
	}
	if splitThreads(0, 2) != 0 {
		t.Fatalf("splitThreads(0,2) = %d, want 0", splitThreads(0, 2))
	}
	if splitThreads(8, 1) != 8 {
		t.Fatalf("splitThreads(8,1) = %d, want 8", splitThreads(8, 1))
	}
}
