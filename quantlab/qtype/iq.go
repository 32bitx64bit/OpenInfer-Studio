package qtype

import (
	"math"
)

// Nonlinear IQ lookup tables from ggml-quants.c / ggml-common.h.
//
// IQ4_NL / IQ4_XS reconstruction matches llama.cpp packing: int8 NL levels
// times an f16 (and for IQ4_XS, 6-bit) scale, not a free-float scale. IQ2/IQ3
// still use the geometric NL grids; their superblock scale is snapped to f16
// so the exact-loss table does not assume a scale llama-quantize cannot store.
var (
	kvaluesIQ4NL = [16]int8{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}
	kIQ3NL       = [...]float64{-1.0, -0.6292, -0.3536, -0.1150, 0.1150, 0.3536, 0.6292, 1.0}
	kIQ2NL       = [...]float64{-1.0, -0.3313, 0.3313, 1.0}
)

const groupMaxEPS = 1e-15

func nearestLevel(v, scale float64, levels []float64) float64 {
	if scale == 0 {
		return 0
	}
	x := v / scale
	best, bestD := levels[0], math.Abs(x-levels[0])
	for _, l := range levels[1:] {
		if d := math.Abs(x - l); d < bestD {
			best, bestD = l, d
		}
	}
	return best * scale
}

func bestIndexIQ4(x float64) int {
	if x <= float64(kvaluesIQ4NL[0]) {
		return 0
	}
	if x >= float64(kvaluesIQ4NL[15]) {
		return 15
	}
	ml, mu := 0, 15
	for mu-ml > 1 {
		mav := (ml + mu) / 2
		if x < float64(kvaluesIQ4NL[mav]) {
			mu = mav
		} else {
			ml = mav
		}
	}
	if x-float64(kvaluesIQ4NL[mu-1]) < float64(kvaluesIQ4NL[mu])-x {
		return mu - 1
	}
	return mu
}

func iq4weight(src, imp []float32, sigma2 float64) []float32 {
	w := make([]float32, len(src))
	if imp != nil {
		for j, x := range src {
			w[j] = imp[j] * float32(math.Sqrt(sigma2+float64(x)*float64(x)))
		}
		return w
	}
	for j, x := range src {
		w[j] = x * x
	}
	return w
}

func iq4sigma2(src []float32) float64 {
	var sum float64
	for _, x := range src {
		sum += float64(x) * float64(x)
	}
	return sum * 2 / float64(len(src))
}

func iq4blockScale(xb, weight []float32, ntry int) (scale float64, ok bool) {
	amax, max := 0.0, 0.0
	for _, v := range xb {
		ax := math.Abs(float64(v))
		if ax > amax {
			amax = ax
			max = float64(v)
		}
	}
	if amax < groupMaxEPS {
		return 0, false
	}
	values0 := float64(kvaluesIQ4NL[0])
	d := max / values0
	if ntry > 0 {
		d = -max / values0
	}
	id := 0.0
	if d != 0 {
		id = 1 / d
	}
	sumqx, sumq2 := 0.0, 0.0
	for j, v := range xb {
		l := bestIndexIQ4(id * float64(v))
		q := float64(kvaluesIQ4NL[l])
		w := float64(weight[j])
		sumqx += w * q * float64(v)
		sumq2 += w * q * q
	}
	if sumq2 > 0 {
		d = sumqx / sumq2
	} else {
		d = 0
	}
	best := d * sumqx
	for itry := -ntry; itry <= ntry; itry++ {
		id = (float64(itry) + values0) / max
		sumqx, sumq2 = 0, 0
		for j, v := range xb {
			l := bestIndexIQ4(id * float64(v))
			q := float64(kvaluesIQ4NL[l])
			w := float64(weight[j])
			sumqx += w * q * float64(v)
			sumq2 += w * q * q
		}
		if sumq2 > 0 && sumqx*sumqx > best*sumq2 {
			d = sumqx / sumq2
			best = d * sumqx
		}
	}
	return d, true
}

func iq4reconstruct(src []float32, d float64, idx []int) float64 {
	var sse float64
	for j, v := range src {
		q := d * float64(kvaluesIQ4NL[idx[j]])
		src[j] = float32(q)
		e := float64(v) - q
		sse += e * e
	}
	return sse
}

// iq4nlrt matches llama.cpp quantize_row_iq4_nl with ntry=7 (the imatrix path).
func iq4nlrt(src, imp []float32, ws *Workspace) float64 {
	_ = ws
	sigma2 := iq4sigma2(src)
	weight := iq4weight(src, imp, sigma2)
	d, ok := iq4blockScale(src, weight, 7)
	if !ok {
		for i := range src {
			src[i] = 0
		}
		return 0
	}
	d = float64(f16rt(float32(d)))
	idx := make([]int, len(src))
	id := 0.0
	if d != 0 {
		id = 1 / d
	}
	for j, v := range src {
		idx[j] = bestIndexIQ4(id * float64(v))
	}
	return iq4reconstruct(src, d, idx)
}

// iq4xsrt matches llama.cpp IQ4_XS: f16 superblock scale and 6-bit per-32 scales.
func iq4xsrt(src, imp []float32, ws *Workspace) float64 {
	_ = ws
	const block = 32
	nb := len(src) / block
	if nb == 0 {
		return 0
	}
	sigma2 := iq4sigma2(src)
	scales := make([]float64, nb)
	maxScale, amaxScale := 0.0, 0.0
	for ib := 0; ib < nb; ib++ {
		off := ib * block
		xb := src[off : off+block]
		w := iq4weight(xb, impSlice(imp, off, block), sigma2)
		d, ok := iq4blockScale(xb, w, 7)
		if !ok {
			continue
		}
		scales[ib] = d
		if ad := math.Abs(d); ad > amaxScale {
			amaxScale = ad
			maxScale = d
		}
	}
	d := float64(f16rt(float32(-maxScale / 32)))
	id := 0.0
	if d != 0 {
		id = 1 / d
	}
	var sse float64
	idx := make([]int, block)
	orig := append([]float32(nil), src...)
	for ib := 0; ib < nb; ib++ {
		l := int(math.Round(id * scales[ib]))
		if l < -32 {
			l = -32
		}
		if l > 31 {
			l = 31
		}
		dl := d * float64(l)
		idl := 0.0
		if dl != 0 {
			idl = 1 / dl
		}
		off := ib * block
		for j := 0; j < block; j++ {
			idx[j] = bestIndexIQ4(idl * float64(orig[off+j]))
		}
		blk := src[off : off+block]
		copy(blk, orig[off:off+block])
		sse += iq4reconstruct(blk, dl, idx)
	}
	return sse
}

// iqNLBlock quantizes one contiguous slice onto a nonlinear grid with a
// single scale, then snaps that scale to f16 (the packed superblock d).
func iqNLBlock(src, out, imp []float32, levels []float64, ws *Workspace) float64 {
	amax := 0.0
	for _, v := range src {
		if a := math.Abs(float64(v)); a > amax {
			amax = a
		}
	}
	if amax == 0 {
		for i := range out {
			out[i] = 0
		}
		return 0
	}
	best := math.Inf(1)
	bestRec, trial := iqScratch(len(out), ws)
	var bestS float64
	for _, f := range scaleGrid {
		s := amax * f
		for i, v := range src {
			trial[i] = float32(nearestLevel(float64(v), s, levels))
		}
		if e := weightedSSE(src, trial, imp); e < best {
			best = e
			bestS = s
			copy(bestRec, trial)
		}
	}
	s := float64(f16rt(float32(bestS)))
	if s != bestS && s != 0 {
		for i, v := range src {
			bestRec[i] = float32(nearestLevel(float64(v), s, levels))
		}
		best = weightedSSE(src, bestRec, imp)
	}
	copy(out, bestRec)
	return best
}

func iqSuperblock(src, imp []float32, levels []float64, subLen int, ws *Workspace) float64 {
	var sse float64
	for off := 0; off < len(src); off += subLen {
		end := off + subLen
		if end > len(src) {
			end = len(src)
		}
		sse += iqNLBlock(src[off:end], src[off:end], impSlice(imp, off, end-off), levels, ws)
	}
	return sse
}

func iq3srt(src, imp []float32, ws *Workspace) float64 {
	return iqSuperblock(src, imp, kIQ3NL[:], 16, ws)
}

func iq3xxsrt(src, imp []float32, ws *Workspace) float64 {
	return iqSuperblock(src, imp, kIQ3NL[:], 32, ws)
}

func iq2srt(src, imp []float32, ws *Workspace) float64 {
	return iqSuperblock(src, imp, kIQ2NL[:], 16, ws)
}

func iq2xsrt(src, imp []float32, ws *Workspace) float64 {
	return iqSuperblock(src, imp, kIQ2NL[:], 32, ws)
}

func iq2xxsrt(src, imp []float32, ws *Workspace) float64 {
	return iqNLBlock(src, src, imp, kIQ2NL[:], ws)
}

func iqScratch(n int, ws *Workspace) (best, trial []float32) {
	if ws != nil {
		ws.ensure(n)
		return ws.best[:n], ws.trial[:n]
	}
	return make([]float32, n), make([]float32, n)
}
