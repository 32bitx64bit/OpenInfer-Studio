package qtype

import (
	"math"
)

// K-quant reference implementations mirroring ggml-quants.c block layouts
// and dequantization semantics exactly (QK_K = 256):
//
//   Q2_K { d f16; dmin f16; scales[16]; qs[64] }            84 bytes
//   Q3_K { hmask[32]; qs[64]; scales[12]; d f16 }           110 bytes
//   Q4_K { d f16; dmin f16; scales[12]; qs[128] }           144 bytes
//   Q5_K { d f16; dmin f16; scales[12]; qh[32]; qs[128] }   176 bytes
//   Q6_K { ql[128]; qh[64]; scales int8[16]; d f16 }        210 bytes

const qkK = 256

func f16(v float32) float64 { return float64(f16rt(v)) }

// bestShrink tries a set of global shrink factors applied to the
// super-block scales, returning the reconstruction with the lowest
// importance-weighted SSE. fn rebuilds and reconstructs the block given
// the shrink factor and writes the reconstruction into out.
func bestShrink(src, out, imp []float32, ws *Workspace, fn func(shrink float64, out []float32)) float64 {
	best := math.Inf(1)
	var bestRec, scratch []float32
	if ws != nil {
		bestRec = ws.best[:len(out)]
		scratch = ws.scratch[:len(out)]
	} else {
		bestRec = make([]float32, len(out))
		scratch = make([]float32, len(out))
	}
	for _, f := range scaleGrid {
		fn(f, scratch)
		if s := weightedSSE(src, scratch, imp); s < best {
			best = s
			copy(bestRec, scratch)
		}
	}
	copy(out, bestRec)
	return best
}

// q2Krt: 16 sub-blocks of 16; value = d*(sc&0xF)*q - dmin*(sc>>4), q in [0,3].
func q2Krt(src, imp []float32, ws *Workspace) float64 {
	rec := make([]float32, qkK)
	sse := bestShrink(src, rec, imp, ws, func(shrink float64, out []float32) {
		var scales [16]float64
		var mins [16]float64
		maxScale, maxMin := 0.0, 0.0
		for j := 0; j < 16; j++ {
			x := src[16*j : 16*j+16]
			mn, mx := math.Inf(1), math.Inf(-1)
			for _, v := range x {
				mn = math.Min(mn, float64(v))
				mx = math.Max(mx, float64(v))
			}
			if mn > 0 {
				mn = 0
			}
			m := -mn
			s := (mx + m) / 3
			scales[j], mins[j] = s, m
			maxScale = math.Max(maxScale, s)
			maxMin = math.Max(maxMin, m)
		}
		var d, dmin float64
		var sc [16]byte
		if maxScale > 0 {
			d = f16(float32(maxScale * shrink / 15))
			for j := 0; j < 16; j++ {
				sc[j] = byte(clampRound(scales[j]/d, 0, 15))
			}
		}
		if maxMin > 0 {
			dmin = f16(float32(maxMin / 15))
			for j := 0; j < 16; j++ {
				sc[j] |= byte(clampRound(mins[j]/dmin, 0, 15)) << 4
			}
		}
		var L [qkK]int
		for j := 0; j < 16; j++ {
			dl := d * float64(sc[j]&0xF)
			ml := dmin * float64(sc[j]>>4)
			if dl == 0 {
				continue
			}
			for i := 0; i < 16; i++ {
				L[16*j+i] = clampRound((float64(src[16*j+i])+ml)/dl, 0, 3)
			}
		}
		var qs [64]byte
		for j := 0; j < qkK; j += 128 {
			for l := 0; l < 32; l++ {
				qs[j/4+l] = byte(L[j+l] | L[j+l+32]<<2 | L[j+l+64]<<4 | L[j+l+96]<<6)
			}
		}
		// Dequantize per the reference layout.
		is := 0
		for n := 0; n < qkK; n += 128 {
			shift := 0
			for j := 0; j < 4; j++ {
				scb := sc[is]
				dl := d * float64(scb&0xF)
				ml := dmin * float64(scb>>4)
				is++
				for l := 0; l < 16; l++ {
					out[n+32*j+l] = float32(dl*float64((qs[n/4+l]>>shift)&3) - ml)
				}
				scb = sc[is]
				dl = d * float64(scb&0xF)
				ml = dmin * float64(scb>>4)
				is++
				for l := 0; l < 16; l++ {
					out[n+32*j+16+l] = float32(dl*float64((qs[n/4+l+16]>>shift)&3) - ml)
				}
				shift += 2
			}
		}
	})
	copy(src, rec)
	return sse
}

// q3Krt: 16 sub-blocks of 16; value = d*(sc-32)*(q - hbit?0:4),
// levels in [-4,3], sc 6-bit packed into scales[12].
//
// Element e (of 256) maps to: qs byte 32*(e/128) + e%32, bits 2*((e%128)/32);
// hmask byte e%32, bit 4*(e/128) + (e%128)/32. Scale k (= e/16) unpacks as
// (k<8 ? scales[k]&0xF : scales[k-8]>>4) | ((scales[8+k%4]>>2*(k/4))&3)<<4.
func q3Krt(src, imp []float32, ws *Workspace) float64 {
	rec := make([]float32, qkK)
	sse := bestShrink(src, rec, imp, ws, func(shrink float64, out []float32) {
		var scales [16]float64
		maxAbs, maxSigned := 0.0, 0.0
		for j := 0; j < 16; j++ {
			x := src[16*j : 16*j+16]
			amax, signed := 0.0, 0.0
			for _, v := range x {
				if a := math.Abs(float64(v)); a > amax {
					amax, signed = a, float64(v)
				}
			}
			s := 0.0
			if amax > 0 {
				s = signed / 3.5
			}
			scales[j] = s
			if a := math.Abs(s); a > maxAbs {
				maxAbs, maxSigned = a, s
			}
		}
		var d float64
		var sc6 [16]int
		var packed [12]byte
		if maxAbs > 0 {
			d = f16(float32(maxSigned * shrink / -32))
			if d != 0 {
				for j := 0; j < 16; j++ {
					l := clampRound(scales[j]/d, -32, 31)
					sc6[j] = l + 32
					if j < 8 {
						packed[j] |= byte(sc6[j] & 0xF)
					} else {
						packed[j-8] |= byte(sc6[j]&0xF) << 4
					}
				}
				for j := 0; j < 16; j++ {
					packed[j%4+8] |= byte((sc6[j]>>4)&3) << (2 * (j / 4))
				}
			}
		}
		var L [qkK]int
		for j := 0; j < 16; j++ {
			dl := d * float64(sc6[j]-32)
			if dl == 0 {
				continue
			}
			for i := 0; i < 16; i++ {
				L[16*j+i] = clampRound(float64(src[16*j+i])/dl, -4, 3)
			}
		}
		var qs [64]byte
		var hm [32]byte
		for e := 0; e < qkK; e++ {
			qb := L[e]
			if qb < 0 {
				qb += 4
			} else {
				hm[e%32] |= 1 << uint(4*(e/128)+(e%128)/32)
			}
			qs[32*(e/128)+e%32] |= byte(qb&3) << (2 * ((e % 128) / 32))
		}
		for e := 0; e < qkK; e++ {
			sc := d * float64(sc6[e/16]-32)
			q := int((qs[32*(e/128)+e%32] >> (2 * ((e % 128) / 32))) & 3)
			if hm[e%32]&(1<<uint(4*(e/128)+(e%128)/32)) == 0 {
				q -= 4
			}
			out[e] = float32(sc * float64(q))
		}
	})
	copy(src, rec)
	return sse
}

// q4Krt: 8 sub-blocks of 32; value = d*sc*n - dmin*m, n in [0,15],
// (sc, m) 6-bit pairs packed into scales[12] via the get_scale_min_k4
// inverse layout.
func q4Krt(src, imp []float32, ws *Workspace) float64 {
	rec := make([]float32, qkK)
	sse := bestShrink(src, rec, imp, ws, func(shrink float64, out []float32) {
		var scales [8]float64
		var mins [8]float64
		maxScale, maxMin := 0.0, 0.0
		for j := 0; j < 8; j++ {
			x := src[32*j : 32*j+32]
			mn, mx := math.Inf(1), math.Inf(-1)
			for _, v := range x {
				mn = math.Min(mn, float64(v))
				mx = math.Max(mx, float64(v))
			}
			if mn > 0 {
				mn = 0
			}
			m := -mn
			s := (mx + m) / 15
			scales[j], mins[j] = s, m
			maxScale = math.Max(maxScale, s)
			maxMin = math.Max(maxMin, m)
		}
		var d, dmin float64
		var packed [12]byte
		if maxScale > 0 {
			d = f16(float32(maxScale * shrink / 63))
		}
		if maxMin > 0 {
			dmin = f16(float32(maxMin / 63))
		}
		for j := 0; j < 8; j++ {
			ls := 0
			lm := 0
			if d > 0 {
				ls = clampRound(scales[j]/d, 0, 63)
			}
			if dmin > 0 {
				lm = clampRound(mins[j]/dmin, 0, 63)
			}
			if j < 4 {
				packed[j] = byte(ls)
				packed[j+4] = byte(lm)
			} else {
				packed[j+4] = byte(ls&0xF) | byte(lm&0xF)<<4
				packed[j-4] |= byte(ls>>4) << 6
				packed[j] |= byte(lm>>4) << 6
			}
		}
		var L [qkK]int
		for j := 0; j < 8; j++ {
			sc, m := scaleMinK4(packed[:], j)
			d1 := d * float64(sc)
			m1 := dmin * float64(m)
			if d1 == 0 {
				continue
			}
			for i := 0; i < 32; i++ {
				L[32*j+i] = clampRound((float64(src[32*j+i])+m1)/d1, 0, 15)
			}
		}
		var qs [128]byte
		for g := 0; g < 4; g++ {
			for l := 0; l < 32; l++ {
				qs[32*g+l] = byte(L[64*g+l] | L[64*g+l+32]<<4)
			}
		}
		// Dequantize per the reference layout.
		is := 0
		q := qs[:]
		for j := 0; j < qkK; j += 64 {
			sc, m := scaleMinK4(packed[:], is)
			d1 := d * float64(sc)
			m1 := dmin * float64(m)
			sc, m = scaleMinK4(packed[:], is+1)
			d2 := d * float64(sc)
			m2 := dmin * float64(m)
			is += 2
			for l := 0; l < 32; l++ {
				out[j+l] = float32(d1*float64(q[l]&0xF) - m1)
				out[j+l+32] = float32(d2*float64(q[l]>>4) - m2)
			}
			q = q[32:]
		}
	})
	copy(src, rec)
	return sse
}

// scaleMinK4 is get_scale_min_k4: 6-bit (scale, min) j unpacked from the
// 12-byte packed scales.
func scaleMinK4(q []byte, j int) (int, int) {
	if j < 4 {
		return int(q[j] & 63), int(q[j+4] & 63)
	}
	return int(q[j+4]&0xF) | int(q[j-4]>>6)<<4, int(q[j+4]>>4) | int(q[j]>>6)<<4
}

// q5Krt: like Q4_K with 5-bit levels; the 5th bit lives in qh.
func q5Krt(src, imp []float32, ws *Workspace) float64 {
	rec := make([]float32, qkK)
	sse := bestShrink(src, rec, imp, ws, func(shrink float64, out []float32) {
		var scales [8]float64
		var mins [8]float64
		maxScale, maxMin := 0.0, 0.0
		for j := 0; j < 8; j++ {
			x := src[32*j : 32*j+32]
			mn, mx := math.Inf(1), math.Inf(-1)
			for _, v := range x {
				mn = math.Min(mn, float64(v))
				mx = math.Max(mx, float64(v))
			}
			if mn > 0 {
				mn = 0
			}
			m := -mn
			s := (mx + m) / 31
			scales[j], mins[j] = s, m
			maxScale = math.Max(maxScale, s)
			maxMin = math.Max(maxMin, m)
		}
		var d, dmin float64
		var packed [12]byte
		if maxScale > 0 {
			d = f16(float32(maxScale * shrink / 63))
		}
		if maxMin > 0 {
			dmin = f16(float32(maxMin / 63))
		}
		for j := 0; j < 8; j++ {
			ls := 0
			lm := 0
			if d > 0 {
				ls = clampRound(scales[j]/d, 0, 63)
			}
			if dmin > 0 {
				lm = clampRound(mins[j]/dmin, 0, 63)
			}
			if j < 4 {
				packed[j] = byte(ls)
				packed[j+4] = byte(lm)
			} else {
				packed[j+4] = byte(ls&0xF) | byte(lm&0xF)<<4
				packed[j-4] |= byte(ls>>4) << 6
				packed[j] |= byte(lm>>4) << 6
			}
		}
		var L [qkK]int
		for j := 0; j < 8; j++ {
			sc, m := scaleMinK4(packed[:], j)
			d1 := d * float64(sc)
			m1 := dmin * float64(m)
			if d1 == 0 {
				continue
			}
			for i := 0; i < 32; i++ {
				L[32*j+i] = clampRound((float64(src[32*j+i])+m1)/d1, 0, 31)
			}
		}
		var ql [128]byte
		var qh [32]byte
		for g := 0; g < 4; g++ {
			u1 := 1 << (2 * uint(g))
			u2 := 2 << (2 * uint(g))
			for l := 0; l < 32; l++ {
				n1 := L[64*g+l]
				n2 := L[64*g+l+32]
				ql[32*g+l] = byte(n1&0xF) | byte(n2&0xF)<<4
				if n1&0x10 != 0 {
					qh[l] |= byte(u1)
				}
				if n2&0x10 != 0 {
					qh[l] |= byte(u2)
				}
			}
		}
		is := 0
		for g := 0; g < 4; g++ {
			u1 := 1 << (2 * uint(g))
			u2 := 2 << (2 * uint(g))
			sc, m := scaleMinK4(packed[:], is)
			d1 := d * float64(sc)
			m1 := dmin * float64(m)
			sc, m = scaleMinK4(packed[:], is+1)
			d2 := d * float64(sc)
			m2 := dmin * float64(m)
			is += 2
			for l := 0; l < 32; l++ {
				hb1 := 0
				if qh[l]&byte(u1) != 0 {
					hb1 = 16
				}
				hb2 := 0
				if qh[l]&byte(u2) != 0 {
					hb2 = 16
				}
				out[64*g+l] = float32(d1*float64(int(ql[32*g+l]&0xF)+hb1) - m1)
				out[64*g+l+32] = float32(d2*float64(int(ql[32*g+l]>>4)+hb2) - m2)
			}
		}
	})
	copy(src, rec)
	return sse
}

// q6Krt: 16 sub-blocks of 16; value = d*sc*q, q in [-32,31], sc int8.
func q6Krt(src, imp []float32, ws *Workspace) float64 {
	rec := make([]float32, qkK)
	sse := bestShrink(src, rec, imp, ws, func(shrink float64, out []float32) {
		var scales [16]float64
		maxAbs, maxSigned := 0.0, 0.0
		for j := 0; j < 16; j++ {
			x := src[16*j : 16*j+16]
			amax, signed := 0.0, 0.0
			for _, v := range x {
				if a := math.Abs(float64(v)); a > amax {
					amax, signed = a, float64(v)
				}
			}
			s := 0.0
			if amax > 0 {
				s = signed / 31.5
			}
			scales[j] = s
			if a := math.Abs(s); a > maxAbs {
				maxAbs, maxSigned = a, s
			}
		}
		var d float64
		var sc [16]int8
		if maxAbs > 0 {
			d = f16(float32(maxSigned * shrink / -128))
			if d != 0 {
				for j := 0; j < 16; j++ {
					l := clampRound(scales[j]/d, -128, 127)
					sc[j] = int8(l)
				}
			}
		}
		var L [qkK]int
		for j := 0; j < 16; j++ {
			dl := d * float64(sc[j])
			if dl == 0 {
				continue
			}
			for i := 0; i < 16; i++ {
				L[16*j+i] = clampRound(float64(src[16*j+i])/dl, -32, 31)
			}
		}
		var ql [128]byte
		var qh [64]byte
		for c := 0; c < 2; c++ {
			for l := 0; l < 32; l++ {
				q1 := L[c*128+l] + 32
				q2 := L[c*128+l+32] + 32
				q3 := L[c*128+l+64] + 32
				q4 := L[c*128+l+96] + 32
				ql[c*64+l] = byte(q1&0xF) | byte(q3&0xF)<<4
				ql[c*64+l+32] = byte(q2&0xF) | byte(q4&0xF)<<4
				qh[c*32+l] = byte(q1>>4)&3 | byte(q2>>4)&3<<2 | byte(q3>>4)&3<<4 | byte(q4>>4)&3<<6
			}
		}
		for c := 0; c < 2; c++ {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((ql[c*64+l]&0xF)|(((qh[c*32+l]>>0)&3)<<4)) - 32
				q2 := int8((ql[c*64+l+32]&0xF)|(((qh[c*32+l]>>2)&3)<<4)) - 32
				q3 := int8((ql[c*64+l]>>4)|(((qh[c*32+l]>>4)&3)<<4)) - 32
				q4 := int8((ql[c*64+l+32]>>4)|(((qh[c*32+l]>>6)&3)<<4)) - 32
				out[c*128+l] = float32(d * float64(sc[c*8+is+0]) * float64(q1))
				out[c*128+l+32] = float32(d * float64(sc[c*8+is+2]) * float64(q2))
				out[c*128+l+64] = float32(d * float64(sc[c*8+is+4]) * float64(q3))
				out[c*128+l+96] = float32(d * float64(sc[c*8+is+6]) * float64(q4))
			}
		}
	})
	copy(src, rec)
	return sse
}
