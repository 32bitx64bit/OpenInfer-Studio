package profile

import (
	"quantlab/core"
	"quantlab/graph"
)

func toGraph(bank *core.TensorBank) []graph.Tensor {
	if bank == nil {
		return nil
	}
	out := make([]graph.Tensor, len(bank.Tensors))
	for i, t := range bank.Tensors {
		out[i] = graph.Tensor{Name: t.Name, Shape: t.Shape}
	}
	return out
}

// geometryFactor is a small shape-and-stats prior on top of name roles.
// Unique-large-axis (embed/head), GQA-short, rank-3 expert stacks, and
// energy/byte from imatrix. Name tables remain the fallback. This is not
// the quality lever for a 3.5 bpw hybrid; search KLD is.
func (e *FallbackEstimator) geometryFactor(t core.TensorDesc) float64 {
	if e == nil {
		return 1
	}
	ts := e.graphTensors
	if len(ts) == 0 {
		return 1
	}
	gt := graph.Tensor{Name: t.Name, Shape: t.Shape}
	f := 1.0
	if graph.UniqueLargeAxis(gt, ts) {
		f *= 1.22
	}
	if graph.GQAShort(gt, e.residualDim) {
		f *= 1.12
	}
	if len(t.Shape) == 3 {
		f *= 0.97
		if st, ok := e.stats(t.Name); ok && st.Mean > 0 && len(st.Values) > 0 {
			// Hot-slice: max 256-chunk vs mean. Fused dtype tracks the expert that hurts.
			mx := 0.0
			for _, v := range st.Values {
				if float64(v) > mx {
					mx = float64(v)
				}
			}
			if mx > st.Mean*1.5 {
				f *= 1.08
			}
		}
	}
	if graph.ConvLike(gt) {
		f *= 1.3
	}
	if st, ok := e.stats(t.Name); ok && t.Elements > 0 && st.Mean > 0 {
		// Energy per byte: spiky / high-mean tensors keep more bits.
		f *= 1 + 0.05*mathMin(st.Spikiness/8, 1)
	}
	if len(t.Shape) == 1 {
		f *= 1.4 // keep 1-D closer to float via higher loss of quant
	}
	return f
}

func mathMin(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
