package profile

import (
	"fmt"
	"sort"

	"quantlab/anchor"
	"quantlab/core"
)

// ScoredOption is a legal per-tensor candidate with exact bytes, an effective
// loss (measured or heuristic, confidence-adjusted, soft priors folded in),
// and its evidence classification.
type ScoredOption struct {
	core.TensorOption
	Loss       float64
	Evidence   EvidenceKind
	Confidence float64
}

// EnumerateOptions computes the exact legal option set for one tensor:
//
//   - Structurally preserved tensors (norms, non-quantizable) yield exactly
//     one option: keep the current float dtype at its exact byte cost.
//   - Otherwise every candidate dtype with known geometry and a compatible
//     block alignment (legal shape) yields an option with exact bytes from
//     core.DType.ExactBytes; options below a hard anchor floor are dropped.
//     Imatrix spikes are not a floor: MagR already pins a few outlier
//     weights, and the estimator prefers IQ on concentrated importance.
//   - Each option's loss comes from measured cache evidence when present
//     (overriding the heuristic), else from the fallback estimator, inflated
//     by the confidence penalty and increased by any soft-prior violation.
//
// The result is Pareto-pruned and sorted by ascending bytes.
//
// NOTE: core.OptionsFor is the lightweight counterpart of this function (no
// loss scoring, anchors, floors, or pruning). The two are kept in sync
// deliberately: both derive option geometry from core.DType.ExactBytes and
// the same block-alignment rule. Change one, change the other.
func EnumerateOptions(t core.TensorDesc, candidates []core.DType, set *anchor.Set,
	cache *Cache, est *FallbackEstimator, confPenalty float64) ([]ScoredOption, error) {

	if err := t.Validate(); err != nil {
		return nil, err
	}
	if est == nil {
		est = NewFallbackEstimator(nil)
	}
	if set == nil {
		set = &anchor.Set{}
	}

	// Structural preservation: keep current storage, exactly one option.
	if set.Preserved(t) {
		b, ok := t.DType.ExactBytes(t.Elements)
		if !ok {
			return nil, fmt.Errorf("profile: tensor %q: no geometry for %q", t.Name, t.DType)
		}
		return []ScoredOption{{
			TensorOption: core.TensorOption{TensorName: t.Name, Target: t.DType, Bytes: b},
			Evidence:     EvidenceHeuristic, Confidence: 1,
		}}, nil
	}

	floor, hasFloor := set.Floor(t.Name)
	opts := make([]ScoredOption, 0, len(candidates))
	for _, d := range candidates {
		if !d.IsQuant() {
			continue
		}
		g, ok := d.BaseTensorType().Geometry()
		if !ok {
			continue
		}
		// Legal shape: the contiguous dimension must be block-aligned.
		if t.Shape[0]%g.BlockSize != 0 {
			continue
		}
		if hasFloor && anchor.Rank(d) > anchor.Rank(floor) {
			continue
		}
		if d.RequiresImatrix() && !est.HasImportance(t.Name) {
			continue
		}
		b, _ := d.ExactBytes(t.Elements)

		l, c := est.Estimate(t, d)
		loss, ev, conf := l, EvidenceHeuristic, c
		scale := lossScale(t.Elements)
		if cl, hit := cache.Get(t.Name, d); hit && cl.Confidence >= conf {
			// Cache stores per-weight / attributed-KLD units; scale to match
			// Estimate so measured and heuristic options share a size-neutral
			// gain (Δloss/Δbpw) across architectures. The cached entry wins
			// only when its confidence at least matches the estimator's, so
			// fresh exact estimates outrank stale search-derived shares on
			// reruns of models whose code or evidence improved.
			loss, ev, conf = cl.Loss*scale, EvidenceMeasured, cl.Confidence
		}
		// Confidence penalty: weak evidence pays a conservative surcharge.
		loss *= 1 + confPenalty*(1-conf)
		// Soft priors shape the landscape; they never forbid an option.
		// Rank-distance priors are O(1); scale them or they vanish against
		// element-weighted heuristic/measured loss.
		loss += set.PriorLoss(t, d) * scale

		opts = append(opts, ScoredOption{
			TensorOption: core.TensorOption{TensorName: t.Name, Target: d, Bytes: b, PriorLoss: loss},
			Loss:         loss,
			Evidence:     ev,
			Confidence:   conf,
		})
	}
	if len(opts) == 0 {
		return nil, fmt.Errorf("profile: tensor %q: no legal quant option (floor %q, candidates %v)",
			t.Name, floor, candidates)
	}
	return ParetoPrune(opts), nil
}

// ParetoPrune removes dominated options: an option is dominated when another
// has both <= bytes and <= loss (with at least one strict). The surviving
// frontier is sorted by ascending bytes with strictly descending loss.
// Ties are broken deterministically by target name. Options that share a
// byte cost are resolved by the sort itself (lowest loss first), so the
// later, worse same-byte option is always dominated by the earlier keep.
func ParetoPrune(opts []ScoredOption) []ScoredOption {
	sorted := append([]ScoredOption(nil), opts...)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].Bytes != sorted[j].Bytes {
			return sorted[i].Bytes < sorted[j].Bytes
		}
		if sorted[i].Loss != sorted[j].Loss {
			return sorted[i].Loss < sorted[j].Loss
		}
		return sorted[i].Target < sorted[j].Target
	})
	out := sorted[:0]
	bestLoss := -1.0
	for _, o := range sorted {
		if bestLoss >= 0 && o.Loss >= bestLoss {
			continue // dominated by a cheaper option with <= loss
		}
		out = append(out, o)
		bestLoss = o.Loss
	}
	return out
}
