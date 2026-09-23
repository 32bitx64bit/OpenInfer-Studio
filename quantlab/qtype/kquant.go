package qtype

import (
	"math"
)

// K-quant reference implementations mirroring ggml-quants.c block layouts
// and dequantization semantics exactly (QK_K = 256):
//
//   Q2_K { scales[16]; qs[64]; d f16; dmin f16 }            84 bytes
//   Q3_K { hmask[32]; qs[64]; scales[12]; d f16 }           110 bytes
//   Q4_K { d f16; dmin f16; scales[12]; qs[128] }           144 bytes
//   Q5_K { d f16; dmin f16; scales[12]; qh[32]; qs[128] }   176 bytes
//   Q6_K { ql[128]; qh[64]; scales int8[16]; d f16 }        210 bytes
//
// Scale/min fitting mirrors the *_impl (importance-matrix) routines in
// ggml-quants.c, with the *_ref weights used when no importance vector is
// supplied. See fits.go. The encode*Block functions are the single source of
// truth for both the round-trip reference and Pack.

const qkK = 256

func f16(v float32) float64 { return float64(f16rt(v)) }

func q2Krt(src, imp []float32, ws *Workspace) float64 {
	var dst [84]byte
	rec := make([]float32, qkK)
	encodeQ2KBlockInto(src, imp, dst[:], rec)
	sse := weightedSSE(src, rec, imp)
	copy(src, rec)
	return sse
}

// encodeQ2KBlockInto: 16 sub-blocks of 16; value = d*(sc&0xF)*q - dmin*(sc>>4).
func encodeQ2KBlockInto(src, imp []float32, dst []byte, rec []float32) {
	var sumx2 float64
	for _, v := range src {
		sumx2 += float64(v) * float64(v)
	}
	sigma2 := sumx2 / qkK
	var scales, mins, sw [16]float64
	var L [qkK]int
	for j := 0; j < 16; j++ {
		x := src[16*j : 16*j+16]
		var weights [16]float32
		var lb []int
		if imp != nil {
			for l := 0; l < 16; l++ {
				xv := float64(x[l])
				weights[l] = imp[16*j+l] * float32(math.Sqrt(sigma2+xv*xv))
			}
			scales[j], mins[j], lb = makeQKX3Quants(x, weights[:], 3, -0.9, 0.05, 36, false)
		} else {
			for l := 0; l < 16; l++ {
				weights[l] = float32(math.Abs(float64(x[l])))
			}
			scales[j], mins[j], lb = makeQKX2Quants(x, weights[:], 3, -0.5, 0.1, 15, true)
		}
		copy(L[16*j:16*j+16], lb)
		for l := 0; l < 16; l++ {
			sw[j] += float64(weights[l])
		}
	}
	dBlock, Ls := makeQPQuants(scales[:], 15, sw[:])
	mBlock, Lm := makeQPQuants(mins[:], 15, sw[:])
	d := f16(float32(dBlock))
	dmin := f16(float32(mBlock))
	var sc [16]byte
	for j := 0; j < 16; j++ {
		sc[j] = byte(Ls[j]) | byte(Lm[j])<<4
	}
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
	// block_q2_K layout: scales[16] (0-15), qs[64] (16-79), d (80-81),
	// dmin (82-83).
	copy(dst[0:16], sc[:])
	for j := 0; j < qkK; j += 128 {
		for l := 0; l < 32; l++ {
			dst[16+j/4+l] = byte(L[j+l] | L[j+l+32]<<2 | L[j+l+64]<<4 | L[j+l+96]<<6)
		}
	}
	putF16(dst[80:82], float32(d))
	putF16(dst[82:84], float32(dmin))
	qs := dst[16:80]
	is := 0
	for n := 0; n < qkK; n += 128 {
		shift := 0
		for j := 0; j < 4; j++ {
			scb := sc[is]
			dl := d * float64(scb&0xF)
			ml := dmin * float64(scb>>4)
			is++
			for l := 0; l < 16; l++ {
				rec[n+32*j+l] = float32(dl*float64((qs[n/4+l]>>shift)&3) - ml)
			}
			scb = sc[is]
			dl = d * float64(scb&0xF)
			ml = dmin * float64(scb>>4)
			is++
			for l := 0; l < 16; l++ {
				rec[n+32*j+16+l] = float32(dl*float64((qs[n/4+l+16]>>shift)&3) - ml)
			}
			shift += 2
		}
	}
}

func q3Krt(src, imp []float32, ws *Workspace) float64 {
	var dst [110]byte
	rec := make([]float32, qkK)
	encodeQ3KBlockInto(src, imp, dst[:], rec)
	sse := weightedSSE(src, rec, imp)
	copy(src, rec)
	return sse
}

// encodeQ3KBlockInto: 16 sub-blocks of 16; value = d*(sc-32)*(q - hbit?0:4),
// levels in [-4,3], sc 6-bit packed into scales[12].
func encodeQ3KBlockInto(src, imp []float32, dst []byte, rec []float32) {
	var sumsq2 float64
	for _, v := range src {
		sumsq2 += float64(v) * float64(v)
	}
	sigma2 := 2 * sumsq2 / qkK
	var scales [16]float64
	var sc6 [16]int
	var packed [12]byte
	var d float64
	if imp != nil {
		var sw [16]float64
		scale32 := make([]float32, 16)
		for j := 0; j < 16; j++ {
			x := src[16*j : 16*j+16]
			var weight [16]float32
			for l := 0; l < 16; l++ {
				xv := float64(x[l])
				weight[l] = imp[16*j+l] * float32(math.Sqrt(sigma2+xv*xv))
				sw[j] += float64(weight[l])
			}
			scales[j], _ = makeQXQuants(x, 4, weight[:])
			scale32[j] = float32(scales[j])
		}
		sw32 := make([]float32, 16)
		for j := range sw32 {
			sw32[j] = float32(sw[j])
		}
		dBlock, Ls := makeQXQuants(scale32, 32, sw32)
		d = f16(float32(dBlock))
		for j := 0; j < 16; j++ {
			sc6[j] = clampInt(Ls[j], 0, 63)
		}
	} else {
		maxAbs, maxSigned := 0.0, 0.0
		for j := 0; j < 16; j++ {
			scales[j], _ = makeQ3Quants(src[16*j:16*j+16], 4)
			if a := math.Abs(scales[j]); a > maxAbs {
				maxAbs, maxSigned = a, scales[j]
			}
		}
		if maxAbs > 0 {
			iscale := -32 / maxSigned
			d = f16(float32(1 / iscale))
			if d != 0 {
				for j := 0; j < 16; j++ {
					sc6[j] = clampInt(nearestInt(iscale*scales[j]), -32, 31) + 32
				}
			}
		}
	}
	for j := 0; j < 16; j++ {
		if j < 8 {
			packed[j] |= byte(sc6[j] & 0xF)
		} else {
			packed[j-8] |= byte(sc6[j]&0xF) << 4
		}
	}
	for j := 0; j < 16; j++ {
		packed[j%4+8] |= byte((sc6[j]>>4)&3) << (2 * (j / 4))
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
	copy(dst[0:32], hm[:])
	copy(dst[32:96], qs[:])
	copy(dst[96:108], packed[:])
	putF16(dst[108:110], float32(d))
	for e := 0; e < qkK; e++ {
		sc := d * float64(sc6[e/16]-32)
		q := int((qs[32*(e/128)+e%32] >> (2 * ((e % 128) / 32))) & 3)
		if hm[e%32]&(1<<uint(4*(e/128)+(e%128)/32)) == 0 {
			q -= 4
		}
		rec[e] = float32(sc * float64(q))
	}
}

func q4Krt(src, imp []float32, ws *Workspace) float64 {
	var dst [144]byte
	rec := make([]float32, qkK)
	encodeQ4KBlockInto(src, imp, dst[:], rec)
	sse := weightedSSE(src, rec, imp)
	copy(src, rec)
	return sse
}

// encodeQ4KBlockInto: 8 sub-blocks of 32; value = d*sc*n - dmin*m, n in [0,15],
// (sc, m) 6-bit pairs packed into scales[12].
func encodeQ4KBlockInto(src, imp []float32, dst []byte, rec []float32) {
	var sumX2 float64
	for _, v := range src {
		sumX2 += float64(v) * float64(v)
	}
	sigma2 := 2 * sumX2 / qkK
	avX := math.Sqrt(sigma2)
	var scales, mins, sw [8]float64
	var L [qkK]int
	for j := 0; j < 8; j++ {
		x := src[32*j : 32*j+32]
		var weights [32]float32
		var lb []int
		if imp != nil {
			for l := 0; l < 32; l++ {
				xv := float64(x[l])
				weights[l] = imp[32*j+l] * float32(math.Sqrt(sigma2+xv*xv))
			}
			scales[j], mins[j], lb = makeQKX3Quants(x, weights[:], 15, -0.9, 0.05, 36, false)
		} else {
			for l := 0; l < 32; l++ {
				weights[l] = float32(avX + math.Abs(float64(x[l])))
			}
			scales[j], mins[j], lb = makeQKX2Quants(x, weights[:], 15, -1.0, 0.1, 20, false)
		}
		copy(L[32*j:32*j+32], lb)
		for l := 0; l < 32; l++ {
			sw[j] += float64(weights[l])
		}
	}
	dBlock, Ls := makeQPQuants(scales[:], 63, sw[:])
	mBlock, Lm := makeQPQuants(mins[:], 63, sw[:])
	d := f16(float32(dBlock))
	dmin := f16(float32(mBlock))
	var packed [12]byte
	for j := 0; j < 8; j++ {
		ls, lm := Ls[j], Lm[j]
		if j < 4 {
			packed[j] = byte(ls)
			packed[j+4] = byte(lm)
		} else {
			packed[j+4] = byte(ls&0xF) | byte(lm&0xF)<<4
			packed[j-4] |= byte(ls>>4) << 6
			packed[j] |= byte(lm>>4) << 6
		}
	}
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
	putF16(dst[0:2], float32(d))
	putF16(dst[2:4], float32(dmin))
	copy(dst[4:16], packed[:])
	for g := 0; g < 4; g++ {
		for l := 0; l < 32; l++ {
			dst[16+32*g+l] = byte(L[64*g+l] | L[64*g+l+32]<<4)
		}
	}
	is := 0
	q := dst[16:144]
	for j := 0; j < qkK; j += 64 {
		sc, m := scaleMinK4(packed[:], is)
		d1 := d * float64(sc)
		m1 := dmin * float64(m)
		sc, m = scaleMinK4(packed[:], is+1)
		d2 := d * float64(sc)
		m2 := dmin * float64(m)
		is += 2
		for l := 0; l < 32; l++ {
			rec[j+l] = float32(d1*float64(q[l]&0xF) - m1)
			rec[j+l+32] = float32(d2*float64(q[l]>>4) - m2)
		}
		q = q[32:]
	}
}

// scaleMinK4 is get_scale_min_k4: 6-bit (scale, min) j unpacked from the
// 12-byte packed scales.
func scaleMinK4(q []byte, j int) (int, int) {
	if j < 4 {
		return int(q[j] & 63), int(q[j+4] & 63)
	}
	return int(q[j+4]&0xF) | int(q[j-4]>>6)<<4, int(q[j+4]>>4) | int(q[j]>>6)<<4
}

func q5Krt(src, imp []float32, ws *Workspace) float64 {
	var dst [176]byte
	rec := make([]float32, qkK)
	encodeQ5KBlockInto(src, imp, dst[:], rec)
	sse := weightedSSE(src, rec, imp)
	copy(src, rec)
	return sse
}

// encodeQ5KBlockInto: like Q4_K with 5-bit levels; the 5th bit lives in qh.
func encodeQ5KBlockInto(src, imp []float32, dst []byte, rec []float32) {
	var sumX2 float64
	for _, v := range src {
		sumX2 += float64(v) * float64(v)
	}
	sigma2 := 2 * sumX2 / qkK
	avX := math.Sqrt(sigma2)
	var scales, mins, sw [8]float64
	var L [qkK]int
	for j := 0; j < 8; j++ {
		x := src[32*j : 32*j+32]
		var weights [32]float32
		var lb []int
		if imp != nil {
			for l := 0; l < 32; l++ {
				xv := float64(x[l])
				weights[l] = imp[32*j+l] * float32(math.Sqrt(sigma2+xv*xv))
			}
			scales[j], mins[j], lb = makeQKX3Quants(x, weights[:], 31, -0.9, 0.05, 36, false)
		} else {
			for l := 0; l < 32; l++ {
				weights[l] = float32(avX + math.Abs(float64(x[l])))
			}
			scales[j], mins[j], lb = makeQKX2Quants(x, weights[:], 31, -0.5, 0.1, 15, false)
		}
		copy(L[32*j:32*j+32], lb)
		for l := 0; l < 32; l++ {
			sw[j] += float64(weights[l])
		}
	}
	dBlock, Ls := makeQPQuants(scales[:], 63, sw[:])
	mBlock, Lm := makeQPQuants(mins[:], 63, sw[:])
	d := f16(float32(dBlock))
	dmin := f16(float32(mBlock))
	var packed [12]byte
	for j := 0; j < 8; j++ {
		ls, lm := Ls[j], Lm[j]
		if j < 4 {
			packed[j] = byte(ls)
			packed[j+4] = byte(lm)
		} else {
			packed[j+4] = byte(ls&0xF) | byte(lm&0xF)<<4
			packed[j-4] |= byte(ls>>4) << 6
			packed[j] |= byte(lm>>4) << 6
		}
	}
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
	putF16(dst[0:2], float32(d))
	putF16(dst[2:4], float32(dmin))
	copy(dst[4:16], packed[:])
	ql := dst[48:176]
	qh := dst[16:48]
	for g := 0; g < 4; g++ {
		u1 := byte(1 << (2 * uint(g)))
		u2 := byte(2 << (2 * uint(g)))
		for l := 0; l < 32; l++ {
			n1 := L[64*g+l]
			n2 := L[64*g+l+32]
			ql[32*g+l] = byte(n1&0xF) | byte(n2&0xF)<<4
			if n1&0x10 != 0 {
				qh[l] |= u1
			}
			if n2&0x10 != 0 {
				qh[l] |= u2
			}
		}
	}
	for j := 0; j < 8; j++ {
		sc, m := scaleMinK4(packed[:], j)
		d1 := d * float64(sc)
		m1 := dmin * float64(m)
		for i := 0; i < 32; i++ {
			rec[32*j+i] = float32(d1*float64(L[32*j+i]) - m1)
		}
	}
}

func q6Krt(src, imp []float32, ws *Workspace) float64 {
	var dst [210]byte
	rec := make([]float32, qkK)
	encodeQ6KBlockInto(src, imp, dst[:], rec)
	sse := weightedSSE(src, rec, imp)
	copy(src, rec)
	return sse
}

// encodeQ6KBlockInto: 16 sub-blocks of 16; value = d*sc*q, q in [-32,31].
func encodeQ6KBlockInto(src, imp []float32, dst []byte, rec []float32) {
	var scales [16]float64
	maxAbs, maxSigned := 0.0, 0.0
	for j := 0; j < 16; j++ {
		var qw []float32
		if imp != nil {
			qw = imp[16*j : 16*j+16]
		}
		scales[j], _ = makeQXQuants(src[16*j:16*j+16], 32, qw)
		if a := math.Abs(scales[j]); a > maxAbs {
			maxAbs, maxSigned = a, scales[j]
		}
	}
	var d float64
	var sc [16]int8
	var L [qkK]int
	if maxAbs >= groupMaxEps {
		iscale := -128 / maxSigned
		d = f16(float32(1 / iscale))
		for j := 0; j < 16; j++ {
			l := nearestInt(iscale * scales[j])
			if l > 127 {
				l = 127
			}
			sc[j] = int8(l)
		}
		for j := 0; j < 16; j++ {
			dl := d * float64(sc[j])
			if dl == 0 {
				continue
			}
			for i := 0; i < 16; i++ {
				L[16*j+i] = clampRound(float64(src[16*j+i])/dl, -32, 31)
			}
		}
	}
	ql := dst[0:128]
	qh := dst[128:192]
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
	copy(dst[192:208], int8Bytes(sc[:]))
	putF16(dst[208:210], float32(d))
	for c := 0; c < 2; c++ {
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := int8((ql[c*64+l]&0xF)|(((qh[c*32+l]>>0)&3)<<4)) - 32
			q2 := int8((ql[c*64+l+32]&0xF)|(((qh[c*32+l]>>2)&3)<<4)) - 32
			q3 := int8((ql[c*64+l]>>4)|(((qh[c*32+l]>>4)&3)<<4)) - 32
			q4 := int8((ql[c*64+l+32]>>4)|(((qh[c*32+l]>>6)&3)<<4)) - 32
			rec[c*128+l] = float32(d * float64(sc[c*8+is+0]) * float64(q1))
			rec[c*128+l+32] = float32(d * float64(sc[c*8+is+2]) * float64(q2))
			rec[c*128+l+64] = float32(d * float64(sc[c*8+is+4]) * float64(q3))
			rec[c*128+l+96] = float32(d * float64(sc[c*8+is+6]) * float64(q4))
		}
	}
}
