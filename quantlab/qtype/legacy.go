package qtype

import (
	"math"
)

// scaleGrid multipliers around the reference scale: the deterministic
// importance-weighted search set.
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

// searchSymmetric finds the scale d minimizing importance-weighted error
// over the grid, reconstructing into rec with fn(d). Reconstruction values
// lie on d*level for integer levels within +/-maxLevel.
func searchSymmetric(src, rec, imp []float32, maxLevel float64, ws *Workspace, fn func(d float64, out []float32)) float64 {
	amax := 0.0
	for _, v := range src {
		if a := math.Abs(float64(v)); a > amax {
			amax = a
		}
	}
	if amax == 0 {
		fn(0, rec)
		return weightedSSE(src, rec, imp)
	}
	base := amax / maxLevel
	best := math.Inf(1)
	bestD := base
	var scratch []float32
	if ws != nil {
		scratch = ws.scratch[:len(rec)]
	} else {
		scratch = make([]float32, len(rec))
	}
	for _, f := range scaleGrid {
		d := base * f
		if d <= 0 {
			continue
		}
		fn(d, scratch)
		if s := weightedSSE(src, scratch, imp); s < best {
			best = s
			bestD = d
		}
	}
	fn(bestD, rec)
	return best
}

// searchAffine finds (d, m) minimizing importance-weighted error for the
// grid m + d*level, level in [0, 2^bits-1]. m candidates sit at or below
// the block minimum so the level range covers the data.
func searchAffine(src, rec, imp []float32, bits int, ws *Workspace, fn func(d, m float64, out []float32)) float64 {
	mn, mx := math.Inf(1), math.Inf(-1)
	for _, v := range src {
		if float64(v) < mn {
			mn = float64(v)
		}
		if float64(v) > mx {
			mx = float64(v)
		}
	}
	levels := math.Pow(2, float64(bits)) - 1
	if !(mx-mn > 0) || levels == 0 {
		fn(0, mn, rec)
		for i := range rec {
			rec[i] = float32(mn)
		}
		return weightedSSE(src, rec, imp)
	}
	baseD := (mx - mn) / levels
	best := math.Inf(1)
	bestD, bestM := baseD, mn
	var scratch []float32
	if ws != nil {
		scratch = ws.scratch[:len(rec)]
	} else {
		scratch = make([]float32, len(rec))
	}
	for _, f := range scaleGrid {
		d := baseD * f
		if d <= 0 {
			continue
		}
		for _, mo := range [...]float64{0, 0.5, 1.0} {
			m := mn - mo*d*0.5
			fn(d, m, scratch)
			if s := weightedSSE(src, scratch, imp); s < best {
				best = s
				bestD, bestM = d, m
			}
		}
	}
	fn(bestD, bestM, rec)
	return best
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

// q8_0rt: d (f16) + 32 int8 levels; reconstruction d*q, q in [-127,127].
func q8_0rt(src, imp []float32, ws *Workspace) float64 {
	var rec [32]float32
	sse := searchSymmetric(src[:], rec[:], imp, 127, ws, func(d float64, out []float32) {
		df := f16rt(float32(d))
		id := 0.0
		if d != 0 {
			id = 1 / float64(df)
		}
		for i, v := range src {
			q := clampRound(float64(v)*id, -127, 127)
			out[i] = float32(df * float32(q))
		}
	})
	copy(src, rec[:])
	return sse
}

// q4_0rt: d (f16) + nibbles; reconstruction d*(n-8), n in [0,15].
// Dequant reads qs[j]&0xF -> y[j], qs[j]>>4 -> y[j+16].
func q4_0rt(src, imp []float32, ws *Workspace) float64 {
	var rec [32]float32
	sse := searchSymmetric(src[:], rec[:], imp, 8, ws, func(d float64, out []float32) {
		df := f16rt(float32(d))
		id := 0.0
		if d != 0 {
			id = 1 / float64(df)
		}
		var qs [16]byte
		for j := 0; j < 16; j++ {
			x0 := float64(src[j]) * id
			x1 := float64(src[j+16]) * id
			n0 := clampRound(x0+8, 0, 15)
			n1 := clampRound(x1+8, 0, 15)
			qs[j] = byte(n0) | byte(n1)<<4
		}
		for j := 0; j < 16; j++ {
			out[j] = float32(df) * float32(int(qs[j]&0xF)-8)
			out[j+16] = float32(df) * float32(int(qs[j]>>4)-8)
		}
	})
	copy(src, rec[:])
	return sse
}

// q4_1rt: d, m (f16) + nibbles; reconstruction n*d + m, n in [0,15].
func q4_1rt(src, imp []float32, ws *Workspace) float64 {
	var rec [32]float32
	sse := searchAffine(src[:], rec[:], imp, 4, ws, func(d, m float64, out []float32) {
		df, mf := f16rt(float32(d)), f16rt(float32(m))
		id := 0.0
		if d != 0 {
			id = 1 / float64(df)
		}
		var qs [16]byte
		for j := 0; j < 16; j++ {
			n0 := clampRound((float64(src[j])-float64(mf))*id, 0, 15)
			n1 := clampRound((float64(src[j+16])-float64(mf))*id, 0, 15)
			qs[j] = byte(n0) | byte(n1)<<4
		}
		for j := 0; j < 16; j++ {
			out[j] = float32(df)*float32(qs[j]&0xF) + mf
			out[j+16] = float32(df)*float32(qs[j]>>4) + mf
		}
	})
	copy(src, rec[:])
	return sse
}

// q5_0rt: d (f16), qh bits, nibbles; reconstruction d*((n|hb<<4)-16),
// level in [-16,15].
func q5_0rt(src, imp []float32, ws *Workspace) float64 {
	var rec [32]float32
	sse := searchSymmetric(src[:], rec[:], imp, 16, ws, func(d float64, out []float32) {
		df := f16rt(float32(d))
		id := 0.0
		if d != 0 {
			id = 1 / float64(df)
		}
		var qs [16]byte
		var qh2 [4]byte
		for j := 0; j < 16; j++ {
			n0 := clampRound(float64(src[j])*id+16, 0, 31)
			n1 := clampRound(float64(src[j+16])*id+16, 0, 31)
			qs[j] = byte(n0&0xF) | byte(n1&0xF)<<4
			if n0&0x10 != 0 {
				qh2[j/8] |= 1 << (j % 8)
			}
			if n1&0x10 != 0 {
				qh2[(j+16)/8] |= 1 << ((j + 16) % 8)
			}
		}
		qhBits := uint32(qh2[0]) | uint32(qh2[1])<<8 | uint32(qh2[2])<<16 | uint32(qh2[3])<<24
		for j := 0; j < 16; j++ {
			xh0 := uint8((qhBits>>uint(j+0))<<4) & 0x10
			xh1 := uint8((qhBits >> uint(j+12))) & 0x10
			out[j] = float32(df) * float32(int32((qs[j]&0xF)|xh0)-16)
			out[j+16] = float32(df) * float32(int32((qs[j]>>4)|xh1)-16)
		}
	})
	copy(src, rec[:])
	return sse
}

// q5_1rt: d, m (f16), qh, nibbles; reconstruction d*(n|hb<<4) + m.
func q5_1rt(src, imp []float32, ws *Workspace) float64 {
	var rec [32]float32
	sse := searchAffine(src[:], rec[:], imp, 5, ws, func(d, m float64, out []float32) {
		df, mf := f16rt(float32(d)), f16rt(float32(m))
		id := 0.0
		if d != 0 {
			id = 1 / float64(df)
		}
		var qs [16]byte
		var qh2 [4]byte
		for j := 0; j < 16; j++ {
			n0 := clampRound((float64(src[j])-float64(mf))*id, 0, 31)
			n1 := clampRound((float64(src[j+16])-float64(mf))*id, 0, 31)
			qs[j] = byte(n0&0xF) | byte(n1&0xF)<<4
			if n0&0x10 != 0 {
				qh2[j/8] |= 1 << (j % 8)
			}
			if n1&0x10 != 0 {
				qh2[(j+16)/8] |= 1 << ((j + 16) % 8)
			}
		}
		qhBits := uint32(qh2[0]) | uint32(qh2[1])<<8 | uint32(qh2[2])<<16 | uint32(qh2[3])<<24
		for j := 0; j < 16; j++ {
			xh0 := uint8((qhBits>>uint(j+0))<<4) & 0x10
			xh1 := uint8((qhBits >> uint(j+12))) & 0x10
			out[j] = float32(df)*float32((qs[j]&0xF)|xh0) + mf
			out[j+16] = float32(df)*float32((qs[j]>>4)|xh1) + mf
		}
	})
	copy(src, rec[:])
	return sse
}
