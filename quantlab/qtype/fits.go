package qtype

import "math"

// This file ports the weighted scale/min fitting routines from
// ggml-quants.c (make_qx_quants, make_q3_quants, make_qkx2_quants,
// make_qkx3_quants, make_qp_quants) so the reference quantizers match what
// llama-quantize actually writes. The previous reference used min/max block
// scales plus a coarse shrink grid, which overstated K-quant reconstruction
// error by ~1.9x on real imatrix-weighted tensors and biased the solver
// toward IQ formats.

const groupMaxEps = 1e-15

func nearestInt(f float64) int { return int(math.Round(f)) }

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// makeQXQuants ports make_qx_quants with rmse_type 1: a signed block scale
// fitted by weighted least squares over a +/-nmax level grid, refined over
// scale candidates. qw is the per-element fit weight; nil selects x*x. L
// receives biased levels (0..2*nmax-1) and the returned scale is signed.
func makeQXQuants(x []float32, nmax int, qw []float32) (float64, []int) {
	n := len(x)
	L := make([]int, n)
	amax, max := 0.0, 0.0
	for _, v := range x {
		ax := math.Abs(float64(v))
		if ax > amax {
			amax, max = ax, float64(v)
		}
	}
	if amax < groupMaxEps || max == 0 {
		return 0, L
	}
	weight := func(i int) float64 {
		if qw != nil {
			return float64(qw[i])
		}
		return float64(x[i]) * float64(x[i])
	}
	iscale := -float64(nmax) / max
	var sumlx, suml2 float64
	for i := range x {
		l := clampInt(nearestInt(iscale*float64(x[i])), -nmax, nmax-1)
		L[i] = l + nmax
		w := weight(i)
		sumlx += w * float64(x[i]) * float64(l)
		suml2 += w * float64(l) * float64(l)
	}
	scale := 0.0
	if suml2 != 0 {
		scale = sumlx / suml2
	}
	best := scale * sumlx
	for is := -9; is <= 9; is++ {
		if is == 0 {
			continue
		}
		iscale = -(float64(nmax) + 0.1*float64(is)) / max
		sumlx, suml2 = 0, 0
		for i := range x {
			l := clampInt(nearestInt(iscale*float64(x[i])), -nmax, nmax-1)
			w := weight(i)
			sumlx += w * float64(x[i]) * float64(l)
			suml2 += w * float64(l) * float64(l)
		}
		if suml2 > 0 && sumlx*sumlx > best*suml2 {
			for i := range x {
				l := clampInt(nearestInt(iscale*float64(x[i])), -nmax, nmax-1)
				L[i] = l + nmax
			}
			scale = sumlx / suml2
			best = scale * sumlx
		}
	}
	return scale, L
}

// makeQ3Quants ports make_q3_quants with do_rmse=true: the unsigned Q3_K
// reference fit used when no imatrix is available. L receives biased levels.
func makeQ3Quants(x []float32, nmax int) (float64, []int) {
	n := len(x)
	L := make([]int, n)
	amax, max := 0.0, 0.0
	for _, v := range x {
		ax := math.Abs(float64(v))
		if ax > amax {
			amax, max = ax, float64(v)
		}
	}
	if amax < groupMaxEps || max == 0 {
		return 0, L
	}
	iscale := -float64(nmax) / max
	var sumlx, suml2 float64
	for i := range x {
		l := clampInt(nearestInt(iscale*float64(x[i])), -nmax, nmax-1)
		L[i] = l
		w := float64(x[i]) * float64(x[i])
		sumlx += w * float64(x[i]) * float64(l)
		suml2 += w * float64(l) * float64(l)
	}
	for itry := 0; itry < 5; itry++ {
		changed := 0
		for i := range x {
			w := float64(x[i]) * float64(x[i])
			slx := sumlx - w*float64(x[i])*float64(L[i])
			if slx > 0 {
				sl2 := suml2 - w*float64(L[i])*float64(L[i])
				nl := clampInt(nearestInt(float64(x[i])*sl2/slx), -nmax, nmax-1)
				if nl != L[i] {
					slx += w * float64(x[i]) * float64(nl)
					sl2 += w * float64(nl) * float64(nl)
					if sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2 {
						L[i] = nl
						sumlx, suml2 = slx, sl2
						changed++
					}
				}
			}
		}
		if changed == 0 {
			break
		}
	}
	scale := 0.0
	if suml2 > 0 {
		scale = sumlx / suml2
	}
	for i := range L {
		L[i] += nmax
	}
	return scale, L
}

// makeQKX2Quants ports make_qkx2_quants: an unsigned min/scale pair fitted
// by weighted least squares. It returns the scale and positive min; L holds
// levels in 0..nmax.
func makeQKX2Quants(x, weights []float32, nmax int, rmin, rdelta float64, nstep int, useMAD bool) (float64, float64, []int) {
	n := len(x)
	L := make([]int, n)
	mn, mx := float64(x[0]), float64(x[0])
	sumW := float64(weights[0])
	sumX := sumW * float64(x[0])
	for i := 1; i < n; i++ {
		if float64(x[i]) < mn {
			mn = float64(x[i])
		}
		if float64(x[i]) > mx {
			mx = float64(x[i])
		}
		w := float64(weights[i])
		sumW += w
		sumX += w * float64(x[i])
	}
	if mn > 0 {
		mn = 0
	}
	if mx == mn {
		return 0, -mn, L
	}
	iscale := float64(nmax) / (mx - mn)
	scale := 1 / iscale
	best := 0.0
	err := func(d, xv float64) float64 {
		diff := d - xv
		if useMAD {
			return math.Abs(diff)
		}
		return diff * diff
	}
	for i := range x {
		l := clampInt(nearestInt(iscale*(float64(x[i])-mn)), 0, nmax)
		L[i] = l
		best += float64(weights[i]) * err(scale*float64(l)+mn, float64(x[i]))
	}
	if nstep < 1 {
		return scale, -mn, L
	}
	Laux := make([]int, n)
	for is := 0; is <= nstep; is++ {
		iscale = (rmin + rdelta*float64(is) + float64(nmax)) / (mx - mn)
		var sumL, sumL2, sumXL float64
		for i := range x {
			l := clampInt(nearestInt(iscale*(float64(x[i])-mn)), 0, nmax)
			Laux[i] = l
			w := float64(weights[i])
			sumL += w * float64(l)
			sumL2 += w * float64(l) * float64(l)
			sumXL += w * float64(l) * float64(x[i])
		}
		D := sumW*sumL2 - sumL*sumL
		if D > 0 {
			thisScale := (sumW*sumXL - sumX*sumL) / D
			thisMin := (sumL2*sumX - sumL*sumXL) / D
			if thisMin > 0 {
				thisMin = 0
				thisScale = sumXL / sumL2
			}
			cur := 0.0
			for i := range x {
				cur += float64(weights[i]) * err(thisScale*float64(Laux[i])+thisMin, float64(x[i]))
			}
			if cur < best {
				copy(L, Laux)
				best = cur
				scale = thisScale
				mn = thisMin
			}
		}
	}
	return scale, -mn, L
}

// makeQKX3Quants ports make_qkx3_quants; nil weights fall back to x*x like
// the C routine. It returns the scale and positive min; L holds 0..nmax.
func makeQKX3Quants(x, weights []float32, nmax int, rmin, rdelta float64, nstep int, useMAD bool) (float64, float64, []int) {
	n := len(x)
	L := make([]int, n)
	wAt := func(i int) float64 {
		if weights != nil {
			return float64(weights[i])
		}
		return float64(x[i]) * float64(x[i])
	}
	mn, mx := float64(x[0]), float64(x[0])
	sumW := wAt(0)
	sumX := sumW * float64(x[0])
	for i := 1; i < n; i++ {
		if float64(x[i]) < mn {
			mn = float64(x[i])
		}
		if float64(x[i]) > mx {
			mx = float64(x[i])
		}
		w := wAt(i)
		sumW += w
		sumX += w * float64(x[i])
	}
	if mn > 0 {
		mn = 0
	}
	if mx <= mn {
		return 0, -mn, L
	}
	err := func(d, xv float64) float64 {
		diff := d - xv
		if useMAD {
			return math.Abs(diff)
		}
		return diff * diff
	}
	iscale := float64(nmax) / (mx - mn)
	scale := 1 / iscale
	best := 0.0
	for i := range x {
		l := clampInt(nearestInt(iscale*(float64(x[i])-mn)), 0, nmax)
		L[i] = l
		best += wAt(i) * err(scale*float64(l)+mn, float64(x[i]))
	}
	if nstep < 1 {
		return scale, -mn, L
	}
	Laux := make([]int, n)
	for is := 0; is <= nstep; is++ {
		iscale = (rmin + rdelta*float64(is) + float64(nmax)) / (mx - mn)
		var sumL, sumL2, sumXL float64
		for i := range x {
			l := clampInt(nearestInt(iscale*(float64(x[i])-mn)), 0, nmax)
			Laux[i] = l
			w := wAt(i)
			sumL += w * float64(l)
			sumL2 += w * float64(l) * float64(l)
			sumXL += w * float64(l) * float64(x[i])
		}
		D := sumW*sumL2 - sumL*sumL
		if D > 0 {
			thisScale := (sumW*sumXL - sumX*sumL) / D
			thisMin := (sumL2*sumX - sumL*sumXL) / D
			if thisMin > 0 {
				thisMin = 0
				thisScale = sumXL / sumL2
			}
			cur := 0.0
			for i := range x {
				cur += wAt(i) * err(thisScale*float64(Laux[i])+thisMin, float64(x[i]))
			}
			if cur < best {
				copy(L, Laux)
				best = cur
				scale = thisScale
				mn = thisMin
			}
		}
	}
	return scale, -mn, L
}

// makeQPQuants ports make_qp_quants: fits a shared positive scale for a
// vector of sub-block scales/mins, refining the level assignment. Returns
// the scale and levels in 0..nmax.
func makeQPQuants(x []float64, nmax int, qw []float64) (float64, []int) {
	n := len(x)
	L := make([]int, n)
	mx := 0.0
	for _, v := range x {
		if v > mx {
			mx = v
		}
	}
	if mx < groupMaxEps {
		return 0, L
	}
	iscale := float64(nmax) / mx
	for i := range x {
		L[i] = nearestInt(iscale * x[i])
	}
	scale := 1 / iscale
	best := 0.0
	for i := range x {
		d := x[i] - scale*float64(L[i])
		best += qw[i] * d * d
	}
	for is := -4; is <= 4; is++ {
		if is == 0 {
			continue
		}
		iscaleIs := (0.1*float64(is) + float64(nmax)) / mx
		scaleIs := 1 / iscaleIs
		mse := 0.0
		for i := range x {
			l := clampInt(nearestInt(iscaleIs*x[i]), 0, nmax)
			d := x[i] - scaleIs*float64(l)
			mse += qw[i] * d * d
		}
		if mse < best {
			best = mse
			iscale = iscaleIs
		}
	}
	var sumLX, sumL2 float64
	for i := range x {
		l := clampInt(nearestInt(iscale*x[i]), 0, nmax)
		L[i] = l
		sumLX += qw[i] * x[i] * float64(l)
		sumL2 += qw[i] * float64(l) * float64(l)
	}
	for itry := 0; itry < 5; itry++ {
		changed := 0
		for i := range x {
			slx := sumLX - qw[i]*x[i]*float64(L[i])
			sl2 := sumL2 - qw[i]*float64(L[i])*float64(L[i])
			if slx > 0 && sl2 > 0 {
				nl := clampInt(nearestInt(x[i]*sl2/slx), 0, nmax)
				if nl != L[i] {
					slx += qw[i] * x[i] * float64(nl)
					sl2 += qw[i] * float64(nl) * float64(nl)
					if slx*slx*sumL2 > sumLX*sumLX*sl2 {
						L[i] = nl
						sumLX, sumL2 = slx, sl2
						changed++
					}
				}
			}
		}
		if changed == 0 {
			break
		}
	}
	if sumL2 > 0 {
		return sumLX / sumL2, L
	}
	return 0, L
}
