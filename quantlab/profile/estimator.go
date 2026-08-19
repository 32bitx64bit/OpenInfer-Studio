package profile

import (
	"math"
	"sort"
	"strings"

	"quantlab/core"
	"quantlab/graph"
)

// ImatrixStats holds importance statistics for one tensor, distilled from
// an importance matrix. The aggregates feed the heuristic estimator; Values
// (when retained) carries the full per-(row, 256-chunk) mean activation
// powers in row-major order and feeds the exact weighted-error estimator
// (see exact.go).
type ImatrixStats struct {
	Mean      float64 `json:"mean"`
	Max       float64 `json:"max"`
	P50       float64 `json:"p50"`
	P95       float64 `json:"p95"`
	Variance  float64 `json:"variance"`
	Spikiness float64 `json:"spikiness"` // max(Max/Mean, P95/P50), clamped to [0, 64]
	Samples   uint64  `json:"samples"`
	// Entropy is the normalized entropy of the channel-importance
	// distribution in [0,1]: low means importance concentrated on few
	// channels (outlier-driven), high means spread evenly.
	Entropy float64 `json:"entropy,omitempty"`
	// EffRank is the participation ratio of the channel-importance
	// distribution: (sum v)^2 / sum v^2 over the per-row means, in [1, rows].
	EffRank float64 `json:"effRank,omitempty"`
	// Values is the full per-(row, chunk) mean activation power vector in
	// row-major order, bounded by maxImatrixVector. Nil when the vector
	// exceeded the retention budget or the source lacked per-block data.
	Values []float32 `json:"values,omitempty"`
}

// FallbackEstimator is the deterministic heuristic sensitivity estimator.
//
// HEURISTIC: every loss it produces is an estimate, not a measurement. The
// estimate combines:
//   - a base severity per dtype derived from exact block-geometry bpw,
//   - a small IQ vs K-quant codebook multiplier (IQ is cheaper on spiky
//     tensors; K-quants are cheaper on regular ones),
//   - a tensor-role factor from name classification (embedding/output,
//     attn_v / attn_out vs harvested ffn_up / MoE experts),
//   - a first/last-block bump and a depth factor from the parsed layer index,
//   - an imatrix factor when aggregate statistics are supplied (mean
//     importance and spikiness),
//   - an uncertainty score exposed as Confidence (lower when no imatrix
//     evidence exists), which the solver converts into a conservative loss
//     inflation via the confidence penalty.
//
// Given identical inputs the estimate is bit-for-bit reproducible.
type FallbackEstimator struct {
	imatrix     map[string]ImatrixStats
	imatrixMean float64
	hasImatrix  bool
	maxLayer    int // last blk.N in the bank; -1 = unknown (blk.0/1 still count as first)
	// exact holds precomputed importance-weighted SSE per (tensor, dtype)
	// from BuildExactLossTable; nil disables the exact path.
	exact map[string]map[core.DType]float64
	// calib corrects heuristic losses toward measured behavior; nil
	// disables calibration.
	calib *Calibration
	// roleScale scales the role factor; 0 means 1 (untouched).
	roleScale float64
	// targetBPW, when > 0, lets role() harvest FFN harder at Q3-ish budgets.
	targetBPW float64
	// graphTensors / residualDim feed shape-and-stats bit policy.
	graphTensors []graph.Tensor
	residualDim  int
}

// HasImportance reports whether imatrix statistics exist for name.
// llama.cpp IQ kernels (IQ2_S and similar) abort with
// GGML_ASSERT(imatrix != NULL) when a tensor has no row, so the solver must
// not offer RequiresImatrix dtypes without a hit. Alternate .weight suffix
// matching covers llama-imatrix GGUF keys that omit or include it.
func (e *FallbackEstimator) HasImportance(name string) bool {
	_, ok := e.stats(name)
	return ok
}

func (e *FallbackEstimator) stats(name string) (ImatrixStats, bool) {
	if e == nil || !e.hasImatrix {
		return ImatrixStats{}, false
	}
	if st, ok := e.imatrix[name]; ok {
		return st, true
	}
	alt := name
	if strings.HasSuffix(name, ".weight") {
		alt = strings.TrimSuffix(name, ".weight")
	} else {
		alt = name + ".weight"
	}
	st, ok := e.imatrix[alt]
	return st, ok
}

// SetMaxLayer records the highest transformer-block index in the bank so
// first/last sensitivity bumps can apply. Negative means unknown.
func (e *FallbackEstimator) SetMaxLayer(n int) {
	if e == nil {
		return
	}
	e.maxLayer = n
}

// BindBank sets maxLayer from the bank's tensor names so first/last-block
// priors match Solve. No-op when e or bank is nil.
func (e *FallbackEstimator) BindBank(bank *core.TensorBank) {
	if e == nil || bank == nil {
		return
	}
	names := make([]string, len(bank.Tensors))
	for i, t := range bank.Tensors {
		names[i] = t.Name
	}
	e.SetMaxLayer(maxLayerIndex(names))
	e.graphTensors = toGraph(bank)
	e.residualDim = graph.Residual(e.graphTensors)
}

// SetTargetBPW records the compression target so role() can harvest FFN
// harder at Q3-ish budgets. Zero or negative leaves the role table unchanged.
func (e *FallbackEstimator) SetTargetBPW(bpw float64) {
	if e == nil {
		return
	}
	e.targetBPW = bpw
}

// NewFallbackEstimator builds an estimator over optional imatrix statistics.
func NewFallbackEstimator(imatrix map[string]ImatrixStats) *FallbackEstimator {
	e := &FallbackEstimator{imatrix: imatrix, maxLayer: -1}
	if len(imatrix) > 0 {
		e.hasImatrix = true
		// Deterministic accumulation order.
		keys := make([]string, 0, len(imatrix))
		for k := range imatrix {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		var sum float64
		for _, k := range keys {
			sum += imatrix[k].Mean
		}
		e.imatrixMean = sum / float64(len(keys))
	}
	return e
}

// baseSeverity is the heuristic relative distortion of a dtype at unit
// sensitivity: quadratic in the Q8_0/bpw ratio.
func baseSeverity(d core.DType) float64 {
	bpw, ok := d.BitsPerWeight()
	if !ok || bpw <= 0 {
		return 1
	}
	r := 8.5 / bpw
	return 0.1 * r * r
}

// codebookFactor is a small per-family multiplier so Pareto selection can
// prefer IQ rungs on spiky / concentrated tensors and K-quants on regular
// ones at similar bpw (CBMK-G), without pretending bits-per-weight is
// quality. HEURISTIC.
func codebookFactor(d core.DType, spikiness, entropy float64) float64 {
	s := spikiness
	if s < 1 {
		s = 1
	}
	// 0 at Spikiness=1 (flat), 1 at Spikiness>=4.
	spikeNorm := math.Min(1, (s-1)/3)
	conc := 0.0
	if entropy >= 0 && entropy <= 1 {
		conc = 1 - entropy
	}
	// Concentrated importance (low entropy) is the Gini/outlier cue for IQ.
	iqPref := math.Max(spikeNorm, conc)
	if d.RequiresImatrix() {
		// IQ: slight penalty when flat, discount when spiky/concentrated.
		return 1.06 - 0.24*iqPref
	}
	// K-quant / legacy: slight preference when regular, mild penalty when spiky.
	return 0.96 + 0.12*iqPref
}

// depthFactor parses "blk.N." and decays sensitivity with depth; tensors
// without a layer index get the mid-depth factor. HEURISTIC.
func depthFactor(name string) float64 {
	depth := layerIndex(name)
	if depth < 0 {
		return 1 + 0.25*math.Exp(-2.0)
	}
	return 1 + 0.25*math.Exp(-0.5*float64(depth))
}

// imatrixFactor scales by relative mean importance, clamped to [0.25, 4],
// then by a spikiness extra in [1, 3].
func (e *FallbackEstimator) imatrixFactor(name string) float64 {
	if !e.hasImatrix {
		return 1
	}
	st, ok := e.stats(name)
	if !ok || e.imatrixMean <= 0 {
		return 1
	}
	f := st.Mean / e.imatrixMean
	f = math.Max(0.25, math.Min(4.0, f))
	spike := st.Spikiness
	if spike < 1 {
		spike = 1
	}
	extra := math.Min(3.0, spike)
	return f * extra
}

func (e *FallbackEstimator) spikiness(name string) float64 {
	st, ok := e.stats(name)
	if !ok || st.Spikiness <= 0 {
		return 1
	}
	return st.Spikiness
}

func (e *FallbackEstimator) entropy(name string) float64 {
	st, ok := e.stats(name)
	if !ok || st.Entropy <= 0 {
		return 1
	}
	return st.Entropy
}

// lossScale converts a per-weight heuristic into a tensor-total loss so the
// solver's gain (Δloss/Δbytes) is independent of tensor size. Without this,
// large tensors (fused QKV, linear-attn, wide FFN) are starved of bits on
// every architecture. Zero-element tensors scale as 1.
func lossScale(elements uint64) float64 {
	if elements == 0 {
		return 1
	}
	return float64(elements)
}

// Estimate returns the heuristic loss and a confidence in (0,1). Confidence
// is 0.6–0.8 when imatrix statistics back the tensor (scaled by sample
// count), 0.5 when an imatrix exists but lacks this tensor, and 0.3 with no
// imatrix at all. HEURISTIC — see the package and type docs.
func (e *FallbackEstimator) Estimate(t core.TensorDesc, target core.DType) (loss, confidence float64) {
	if sse, weighted, ok := e.exactSSE(t, target); ok {
		// Relative weighted MSE per weight replaces the analytic
		// severity*codebook terms; the functional priors stay because the
		// reconstruction error cannot see them. Weighted sums normalize by
		// the tensor's mean importance so the result is a per-weight
		// relative distortion comparable with baseSeverity.
		norm := float64(t.Elements)
		if weighted {
			if m := impMean(e, t.Name); m > 0 {
				norm *= m
			}
		}
		rel := sse / norm
		if rel < 0 || math.IsNaN(rel) {
			rel = 0
		}
		loss = exactUnitAlign * rel * e.role(t.Name) *
			depthFactor(t.Name) * e.imatrixFactor(t.Name) *
			lossScale(t.Elements)
		confidence = 0.8
		if weighted {
			confidence = 0.9
			if st, ok := e.stats(t.Name); ok {
				confidence = math.Max(confidence, 0.6+0.2*math.Min(1, float64(st.Samples)/4096.0))
			}
		}
		return loss, confidence
	}
	loss, confidence = e.heuristic(t, target)
	loss = e.calibrate(t, target, loss)
	return loss, confidence
}

// heuristic is the uncalibrated analytic estimate; CalFeatures must call
// this (not Estimate) so fitting never recurs through the calibration.
func (e *FallbackEstimator) heuristic(t core.TensorDesc, target core.DType) (loss, confidence float64) {
	spike := e.spikiness(t.Name)
	loss = baseSeverity(target) * codebookFactor(target, spike, e.entropy(t.Name)) * e.role(t.Name) *
		e.geometryFactor(t) * depthFactor(t.Name) * e.imatrixFactor(t.Name) *
		lossScale(t.Elements)
	switch {
	case !e.hasImatrix:
		confidence = 0.3
	default:
		st, ok := e.stats(t.Name)
		if !ok {
			confidence = 0.5
		} else {
			confidence = 0.6 + 0.2*math.Min(1, float64(st.Samples)/4096.0)
		}
	}
	return loss, confidence
}

// SetRoleScale scales the role prior multipliers: 1 (or 0, the unset
// default) keeps them, a negative value flattens them entirely, any other
// positive scale compresses the prior toward 1.
func (e *FallbackEstimator) SetRoleScale(scale float64) {
	if e == nil || scale == 0 {
		return
	}
	e.roleScale = scale
}

// role applies the role prior with the configured scale: 1 + (f-1)*scale.
// A negative scale (explicit flatten) removes the prior entirely.
func (e *FallbackEstimator) role(name string) float64 {
	if e != nil && e.roleScale < 0 {
		return 1
	}
	f := roleFactor(name, e.maxLayer)
	if e != nil && e.targetBPW > 0 {
		f *= roleBPWScale(name, e.targetBPW)
	}
	if e != nil && e.roleScale > 0 && e.roleScale != 1 {
		f = 1 + (f-1)*e.roleScale
	}
	return f
}

// exactSSE returns the precomputed importance-weighted SSE for (t, target)
// and whether it was importance-weighted (vs uniform).
func (e *FallbackEstimator) exactSSE(t core.TensorDesc, target core.DType) (sse float64, weighted, ok bool) {
	if e == nil || e.exact == nil {
		return 0, false, false
	}
	m, found := e.exact[t.Name]
	if !found {
		return 0, false, false
	}
	v, found := m[target]
	if !found {
		return 0, false, false
	}
	_, weighted = lookupImatrixVec(e, t.Name)
	return v, weighted, true
}

func lookupImatrixVec(e *FallbackEstimator, name string) ([]float32, bool) {
	if st, ok := e.stats(name); ok && st.Values != nil {
		return st.Values, true
	}
	return nil, false
}

// impMean resolves the tensor's mean importance for normalization.
func impMean(e *FallbackEstimator, name string) float64 {
	if st, ok := e.stats(name); ok && st.Mean > 0 {
		return st.Mean
	}
	return 1
}
