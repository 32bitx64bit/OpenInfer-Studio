package profile

import (
	"math"
	"sort"
	"strconv"
	"strings"

	"quantlab/core"
)

// ExpertUtilAlpha scales per-expert importance by utilization^alpha when
// joining per-expert imatrix entries into one fused expert-stack entry.
// Utilization comes from per-expert activation-sample counts; alpha < 1
// keeps rare experts in play while letting heavily routed experts dominate
// within-tensor importance weighting, per router-traffic evidence.
const ExpertUtilAlpha = 0.5

// JoinExpertImatrix synthesizes imatrix entries for fused MoE expert-stack
// tensors (e.g. blk.N.ffn_up_exps.weight) from per-expert entries
// (e.g. blk.N.experts.J.ffn_up) when the file has no direct row for the
// fused name. Per-expert values are normalized to unit mean so a heavily
// activated expert does not dominate by absolute scale, then weighted by
// utilization^ExpertUtilAlpha and concatenated in expert order; the bank's
// fused tensor's row order is per-expert row blocks, matching llama.cpp's
// expert-stack layout. Direct hits (including .weight alternates) win
// over synthesized entries. The input map is not mutated.
func JoinExpertImatrix(imatrix map[string]ImatrixStats, bank *core.TensorBank) map[string]ImatrixStats {
	if bank == nil || len(imatrix) == 0 {
		return imatrix
	}
	out := make(map[string]ImatrixStats, len(imatrix)+4)
	for k, v := range imatrix {
		out[k] = v
	}
	// Index normalized bases of per-expert entries.
	type expEntry struct {
		st  ImatrixStats
		idx int
	}
	byBase := map[string][]expEntry{}
	for name, st := range imatrix {
		base, idx, ok := stripExpertSegment(name)
		if !ok {
			continue
		}
		byBase[base] = append(byBase[base], expEntry{st: st, idx: idx})
	}
	if len(byBase) == 0 {
		return out
	}
	for _, t := range bank.Tensors {
		if !isFusedExpertStack(t.Name) {
			continue
		}
		if _, ok := lookupImatrixDefault(out, t.Name); ok {
			continue // direct hit wins
		}
		base := fusedBase(t.Name)
		entries, ok := byBase[base]
		if !ok || len(entries) == 0 {
			continue
		}
		sort.Slice(entries, func(i, j int) bool {
			if entries[i].idx != entries[j].idx {
				return entries[i].idx < entries[j].idx
			}
			return entries[i].st.Samples < entries[j].st.Samples
		})
		var maxSamples uint64
		for _, e := range entries {
			if e.st.Samples > maxSamples {
				maxSamples = e.st.Samples
			}
		}
		var joined []float32
		var totalSamples uint64
		for _, e := range entries {
			util := 1.0
			if maxSamples > 0 {
				util = math.Pow(float64(e.st.Samples)/float64(maxSamples), ExpertUtilAlpha)
			}
			mean := e.st.Mean
			if mean <= 0 {
				mean = 1
			}
			if len(e.st.Values) == 0 {
				continue
			}
			for _, v := range e.st.Values {
				joined = append(joined, float32(float64(v)/mean*util))
			}
			totalSamples += e.st.Samples
		}
		if len(joined) == 0 {
			continue
		}
		st := aggregateImatrix(joined, nil, len(joined) <= maxImatrixVector)
		if totalSamples > 0 {
			st.Samples = totalSamples
		}
		out[t.Name] = st
	}
	return out
}

// isFusedExpertStack reports whether name looks like a fused expert stack.
func isFusedExpertStack(name string) bool {
	n := strings.ToLower(strings.TrimSuffix(name, ".weight"))
	return strings.Contains(n, "_exps")
}

// fusedBase strips the _exps decoration for per-expert entry matching:
// "blk.3.ffn_up_exps.weight" -> "blk.3.ffn_up".
func fusedBase(name string) string {
	n := strings.TrimSuffix(name, ".weight")
	n = strings.TrimSuffix(n, "_exps")
	return n
}

// stripExpertSegment removes an "experts.J" / "expert.J" segment from an
// imatrix entry name, returning the base and expert index:
// "blk.0.experts.17.ffn_up" -> ("blk.0.ffn_up", 17, true).
func stripExpertSegment(name string) (string, int, bool) {
	segs := strings.Split(name, ".")
	var out []string
	idx := -1
	for i := 0; i < len(segs); i++ {
		s := segs[i]
		if (s == "experts" || s == "expert") && i+1 < len(segs) {
			if n, err := strconv.Atoi(segs[i+1]); err == nil {
				if idx < 0 {
					idx = n
				}
				i++
				continue
			}
		}
		out = append(out, s)
	}
	if idx < 0 {
		return "", 0, false
	}
	return strings.Join(out, "."), idx, true
}

// lookupImatrixDefault mirrors the name-alternation used by
// FallbackEstimator.stats without the estimator.
func lookupImatrixDefault(m map[string]ImatrixStats, name string) (ImatrixStats, bool) {
	if st, ok := m[name]; ok {
		return st, true
	}
	alt := name
	if strings.HasSuffix(name, ".weight") {
		alt = strings.TrimSuffix(name, ".weight")
	} else {
		alt = name + ".weight"
	}
	if st, ok := m[alt]; ok {
		return st, true
	}
	return ImatrixStats{}, false
}
