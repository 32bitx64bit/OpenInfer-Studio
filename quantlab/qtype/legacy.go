package qtype

import (
	"math"
)

// scaleGrid multipliers around the reference scale: the deterministic
// importance-weighted search set used by the IQ nonlinear grids.
var scaleGrid = [...]float64{0.9, 1.0, 1.1}

func weightedSSE(src, rec []float32, imp []float32) float64 {
	if imp == nil {
		var s float64
		for i, v := range src {
			e := float64(v) - float64(rec[i])
			s += e * e
		}
		return s
	}
	var s float64
	for i, v := range src {
		e := float64(v) - float64(rec[i])
		s += float64(imp[i]) * e * e
	}
	return s
}

// clampRound is nearest_int with clamping: round-half-away-from-zero,
// matching the ggml reference quantizers.
func clampRound(x float64, lo, hi int) int {
	var v int64
	if x >= 0 {
		v = int64(x + 0.5)
	} else {
		v = -int64(-x + 0.5)
	}
	if v < int64(lo) {
		return lo
	}
	if v > int64(hi) {
		return hi
	}
	return int(v)
}

// Legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 reference codecs.
//
// Q8_0 mirrors quantize_row_q8_0_ref exactly and ignores importance (real
// llama-quantize uses plain RTN for Q8_0 even with an imatrix). Q4_0/Q5_0
// use makeQXQuants (rmse_type 1) and Q4_1/Q5_1 use makeQKX3Quants with
// importance-derived weights weight[j] = imp[j]*sqrt(sigma2 + x[j]^2) where
// sigma2 is the mean square of the whole row — matching llama-quantize's
// quantize_row_q*_impl. With imp == nil the simple _ref min/max quantizers
// are reproduced instead. Row-aware entry points carry the row length so
// sigma2 spans the right window.

// rowSigma2 returns the mean square of one row of weights.
func rowSigma2(src []float32) float64 {
	if len(src) == 0 {
		return 0
	}
	var s float64
	for _, v := range src {
		s += float64(v) * float64(v)
	}
	return s / float64(len(src))
}

// legacyWeights builds the per-element fit weight
// imp[j]*sqrt(sigma2 + x[j]^2) used by the *_impl quantizers.
func legacyWeights(src, imp []float32, sigma2 float64) []float32 {
	w := make([]float32, len(src))
	for j, x := range src {
		w[j] = imp[j] * float32(math.Sqrt(sigma2+float64(x)*float64(x)))
	}
	return w
}

// q8_0rt mirrors quantize_row_q8_0_ref and ignores importance.
func q8_0rt(src, imp []float32, ws *Workspace) float64 {
	_ = imp
	_ = ws
	amax := 0.0
	for _, v := range src {
		if a := math.Abs(float64(v)); a > amax {
			amax = a
		}
	}
	if amax < groupMaxEps {
		for i := range src {
			src[i] = 0
		}
		return 0
	}
	d := amax / 127
	id := 0.0
	if d != 0 {
		id = 1 / d
	}
	df := float64(f16rt(float32(d)))
	var sse float64
	for i, v := range src {
		q := clampRound(float64(v)*id, -127, 127)
		rec := df * float64(q)
		src[i] = float32(rec)
		e := float64(v) - rec
		sse += e * e
	}
	return sse
}

// q4_0Ref mirrors quantize_row_q4_0_ref: one min/max scale per block.
func q4_0Ref(src []float32) float64 {
	amax, max := 0.0, 0.0
	for _, v := range src {
		if a := math.Abs(float64(v)); a > amax {
			amax, max = a, float64(v)
		}
	}
	if amax < groupMaxEps {
		for i := range src {
			src[i] = 0
		}
		return 0
	}
	d := max / -8
	id := 0.0
	if d != 0 {
		id = 1 / d
	}
	df := float64(f16rt(float32(d)))
	var sse float64
	for i, v := range src {
		n := int(float64(v)*id + 8.5)
		if n < 0 {
			n = 0
		}
		if n > 15 {
			n = 15
		}
		rec := df * float64(n-8)
		src[i] = float32(rec)
		e := float64(v) - rec
		sse += e * e
	}
	return sse
}

// q5_0Ref mirrors quantize_row_q5_0_ref.
func q5_0Ref(src []float32) float64 {
	amax, max := 0.0, 0.0
	for _, v := range src {
		if a := math.Abs(float64(v)); a > amax {
			amax, max = a, float64(v)
		}
	}
	if amax < groupMaxEps {
		for i := range src {
			src[i] = 0
		}
		return 0
	}
	d := max / -16
	id := 0.0
	if d != 0 {
		id = 1 / d
	}
	df := float64(f16rt(float32(d)))
	var sse float64
	for i, v := range src {
		n := int(float64(v)*id + 16.5)
		if n < 0 {
			n = 0
		}
		if n > 31 {
			n = 31
		}
		rec := df * float64(n-16)
		src[i] = float32(rec)
		e := float64(v) - rec
		sse += e * e
	}
	return sse
}

// q4_1Ref mirrors quantize_row_q4_1_ref: min/max affine per block.
func q4_1Ref(src []float32) float64 {
	mn, mx := math.Inf(1), math.Inf(-1)
	for _, v := range src {
		fv := float64(v)
		if fv < mn {
			mn = fv
		}
		if fv > mx {
			mx = fv
		}
	}
	if mn > 0 {
		mn = 0
	}
	if !(mx-mn > 0) {
		for i := range src {
			src[i] = float32(mn)
		}
		return 0
	}
	d := (mx - mn) / 15
	id := 0.0
	if d != 0 {
		id = 1 / d
	}
	df, mf := float64(f16rt(float32(d))), float64(f16rt(float32(mn)))
	var sse float64
	for i, v := range src {
		n := int((float64(v)-mn)*id + 0.5)
		if n < 0 {
			n = 0
		}
		if n > 15 {
			n = 15
		}
		rec := df*float64(n) + mf
		src[i] = float32(rec)
		e := float64(v) - rec
		sse += e * e
	}
	return sse
}

// q5_1Ref mirrors quantize_row_q5_1_ref.
func q5_1Ref(src []float32) float64 {
	mn, mx := math.Inf(1), math.Inf(-1)
	for _, v := range src {
		fv := float64(v)
		if fv < mn {
			mn = fv
		}
		if fv > mx {
			mx = fv
		}
	}
	if mn > 0 {
		mn = 0
	}
	if !(mx-mn > 0) {
		for i := range src {
			src[i] = float32(mn)
		}
		return 0
	}
	d := (mx - mn) / 31
	id := 0.0
	if d != 0 {
		id = 1 / d
	}
	df, mf := float64(f16rt(float32(d))), float64(f16rt(float32(mn)))
	var sse float64
	for i, v := range src {
		n := int((float64(v)-mn)*id + 0.5)
		if n < 0 {
			n = 0
		}
		if n > 31 {
			n = 31
		}
		rec := df*float64(n) + mf
		src[i] = float32(rec)
		e := float64(v) - rec
		sse += e * e
	}
	return sse
}

// blockSigma2 returns the row mean-square for the current block: the
// workspace's row-level value when available, else the block's own mean
// square (the correct window when a caller processes one block at a time).
func blockSigma2(src []float32, ws *Workspace) float64 {
	if ws != nil && ws.rowSigma2 > 0 {
		return ws.rowSigma2
	}
	return rowSigma2(src)
}

// q4_0rt: d (f16) + nibbles; reconstruction d*(n-8), n in [0,15].
func q4_0rt(src, imp []float32, ws *Workspace) float64 {
	if imp == nil {
		return q4_0Ref(src)
	}
	w := legacyWeights(src, imp, blockSigma2(src, ws))
	scale, L := makeQXQuants(src, 8, w)
	d := float64(f16rt(float32(scale)))
	var sse float64
	for i, v := range src {
		rec := d * float64(L[i]-8)
		src[i] = float32(rec)
		e := float64(v) - rec
		sse += float64(imp[i]) * e * e
	}
	return sse
}

// q5_0rt: d (f16) + qh bits, nibbles; reconstruction d*((n)-16),
// level in [-16,15].
func q5_0rt(src, imp []float32, ws *Workspace) float64 {
	if imp == nil {
		return q5_0Ref(src)
	}
	w := legacyWeights(src, imp, blockSigma2(src, ws))
	scale, L := makeQXQuants(src, 16, w)
	d := float64(f16rt(float32(scale)))
	var sse float64
	for i, v := range src {
		rec := d * float64(L[i]-16)
		src[i] = float32(rec)
		e := float64(v) - rec
		sse += float64(imp[i]) * e * e
	}
	return sse
}

// q4_1rt: d, m (f16) + nibbles; reconstruction n*d + m, n in [0,15].
func q4_1rt(src, imp []float32, ws *Workspace) float64 {
	if imp == nil {
		return q4_1Ref(src)
	}
	w := legacyWeights(src, imp, blockSigma2(src, ws))
	scale, min, L := makeQKX3Quants(src, w, 15, -0.9, 0.05, 36, false)
	df, mf := float64(f16rt(float32(scale))), float64(f16rt(float32(min)))
	var sse float64
	for i, v := range src {
		rec := df*float64(L[i]) + mf
		src[i] = float32(rec)
		e := float64(v) - rec
		sse += float64(imp[i]) * e * e
	}
	return sse
}

// q5_1rt: d, m (f16), qh, nibbles; reconstruction d*n + m.
func q5_1rt(src, imp []float32, ws *Workspace) float64 {
	if imp == nil {
		return q5_1Ref(src)
	}
	w := legacyWeights(src, imp, blockSigma2(src, ws))
	scale, min, L := makeQKX3Quants(src, w, 31, -0.9, 0.05, 36, false)
	df, mf := float64(f16rt(float32(scale))), float64(f16rt(float32(min)))
	var sse float64
	for i, v := range src {
		rec := df*float64(L[i]) + mf
		src[i] = float32(rec)
		e := float64(v) - rec
		sse += float64(imp[i]) * e * e
	}
	return sse
}
