package qtype

import (
	"encoding/binary"
	"fmt"
	"math"

	"quantlab/core"
)

// PackSupported reports whether d can emit legal ggml superblocks.
func PackSupported(d core.DType) bool {
	switch d.BaseTensorType() {
	case core.DTypeQ8_0, core.DTypeQ4_0, core.DTypeQ4_1, core.DTypeQ5_0, core.DTypeQ5_1,
		core.DTypeQ2_K, core.DTypeQ3_K, core.DTypeQ4_K_T, core.DTypeQ5_K_T, core.DTypeQ6_K:
		return true
		// Q6_K is included only with the ggml ql/qh interleave (see packQ6K).
	}
	return false
}

// Pack quantizes src into ggml-packed superblocks and returns packed bytes
// plus the reconstruction (same length as src).
func Pack(d core.DType, src, imp []float32) ([]byte, []float32, error) {
	return PackOpts(d, src, imp, PackOptions{})
}

// PackOptions controls row-aware packing.
type PackOptions struct {
	// Viterbi runs a short DP over shrink factors on consecutive K-quant
	// superblocks along each row (RowLen elements). Ignored for 32-wide types.
	Viterbi bool
	// RowLen is the contiguous row length (GGUF ne0). Zero treats src as one row.
	RowLen int
	// Shrink, when > 0, forces a single shrink factor (no grid, no Viterbi).
	Shrink float64
}

var viterbiShrinks = [...]float64{0.9, 1.0, 1.1}

// PackOpts is Pack with Viterbi / shrink overrides.
func PackOpts(d core.DType, src, imp []float32, opt PackOptions) ([]byte, []float32, error) {
	if !PackSupported(d) {
		return nil, nil, fmt.Errorf("qtype: pack does not support %s", d)
	}
	bs := BlockSize(d)
	ts := TypeSize(d)
	if bs == 0 || ts == 0 || len(src)%bs != 0 {
		return nil, nil, fmt.Errorf("qtype: pack %s: %d elements, block %d", d, len(src), bs)
	}
	if imp != nil && len(imp) != len(src) {
		return nil, nil, fmt.Errorf("qtype: pack importance length mismatch")
	}
	row := opt.RowLen
	if row <= 0 {
		row = len(src)
	}
	if len(src)%row != 0 || row%bs != 0 {
		return nil, nil, fmt.Errorf("qtype: pack row length %d incompatible with %d elems / block %d", row, len(src), bs)
	}
	rec := make([]float32, len(src))
	packed := make([]byte, (len(src)/bs)*ts)
	nBlocksRow := row / bs
	useVit := opt.Viterbi && bs == qkK && opt.Shrink == 0 && nBlocksRow > 1
	for rowOff := 0; rowOff < len(src); rowOff += row {
		if useVit {
			packRowViterbi(d, src[rowOff:rowOff+row], impSlice(imp, rowOff, row),
				packed[(rowOff/bs)*ts:], rec[rowOff:rowOff+row])
			continue
		}
		for b := 0; b < nBlocksRow; b++ {
			off := rowOff + b*bs
			po := (off / bs) * ts
			packOne(d, src[off:off+bs], impSlice(imp, off, bs), packed[po:po+ts], rec[off:off+bs], opt.Shrink)
		}
	}
	return packed, rec, nil
}

func packOne(d core.DType, src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	if shrink <= 0 {
		best := math.Inf(1)
		tmp := make([]byte, len(dst))
		trial := make([]float32, len(rec))
		for _, f := range scaleGrid {
			sse := packOneShrink(d, src, imp, tmp, trial, f)
			if sse < best {
				best = sse
				copy(dst, tmp)
				copy(rec, trial)
			}
		}
		return best
	}
	return packOneShrink(d, src, imp, dst, rec, shrink)
}

func packOneShrink(d core.DType, src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	switch d.BaseTensorType() {
	case core.DTypeQ8_0:
		return packQ8_0(src, imp, dst, rec, shrink)
	case core.DTypeQ4_0:
		return packQ4_0(src, imp, dst, rec, shrink)
	case core.DTypeQ4_1:
		return packQ4_1(src, imp, dst, rec, shrink)
	case core.DTypeQ5_0:
		return packQ5_0(src, imp, dst, rec, shrink)
	case core.DTypeQ5_1:
		return packQ5_1(src, imp, dst, rec, shrink)
	case core.DTypeQ2_K:
		return packQ2K(src, imp, dst, rec, shrink)
	case core.DTypeQ3_K:
		return packQ3K(src, imp, dst, rec, shrink)
	case core.DTypeQ4_K_T:
		return packQ4K(src, imp, dst, rec, shrink)
	case core.DTypeQ5_K_T:
		return packQ5K(src, imp, dst, rec, shrink)
	case core.DTypeQ6_K:
		return packQ6K(src, imp, dst, rec, shrink)
	}
	return math.Inf(1)
}

func packRowViterbi(d core.DType, row, imp []float32, packed []byte, rec []float32) {
	bs := BlockSize(d)
	ts := TypeSize(d)
	n := len(row) / bs
	const ns = 3
	type cell struct {
		cost float64
		prev int
	}
	dp := make([][ns]cell, n)
	store := make([][][]byte, n)     // [block][state] packed
	storeR := make([][][]float32, n) // [block][state] rec
	for t := 0; t < n; t++ {
		store[t] = make([][]byte, ns)
		storeR[t] = make([][]float32, ns)
		src := row[t*bs : (t+1)*bs]
		im := impSlice(imp, t*bs, bs)
		for s, sh := range viterbiShrinks {
			pb := make([]byte, ts)
			pr := make([]float32, bs)
			sse := packOneShrink(d, src, im, pb, pr, sh)
			store[t][s] = pb
			storeR[t][s] = pr
			best := math.Inf(1)
			prev := 0
			if t == 0 {
				best = sse
			} else {
				for p := 0; p < ns; p++ {
					pen := 0.02 * math.Abs(viterbiShrinks[s]-viterbiShrinks[p])
					c := dp[t-1][p].cost + sse + pen
					if c < best {
						best = c
						prev = p
					}
				}
			}
			dp[t][s] = cell{cost: best, prev: prev}
		}
	}
	bestS := 0
	for s := 1; s < ns; s++ {
		if dp[n-1][s].cost < dp[n-1][bestS].cost {
			bestS = s
		}
	}
	path := make([]int, n)
	path[n-1] = bestS
	for t := n - 1; t > 0; t-- {
		path[t-1] = dp[t][path[t]].prev
	}
	for t := 0; t < n; t++ {
		s := path[t]
		copy(packed[t*ts:(t+1)*ts], store[t][s])
		copy(rec[t*bs:(t+1)*bs], storeR[t][s])
	}
}

func putF16(dst []byte, v float32) {
	binary.LittleEndian.PutUint16(dst, F16Bits(v))
}

func packQ8_0(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	amax := 0.0
	for _, v := range src {
		if a := math.Abs(float64(v)); a > amax {
			amax = a
		}
	}
	d := f16rt(float32(amax / 127 * shrink))
	id := 0.0
	if d != 0 {
		id = 1 / float64(d)
	}
	putF16(dst[0:2], d)
	for i, v := range src {
		q := clampRound(float64(v)*id, -127, 127)
		dst[2+i] = byte(q)
		rec[i] = d * float32(q)
	}
	return weightedSSE(src, rec, imp)
}

func packQ4_0(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	amax := 0.0
	for _, v := range src {
		if a := math.Abs(float64(v)); a > amax {
			amax = a
		}
	}
	d := f16rt(float32(amax / 8 * shrink))
	id := 0.0
	if d != 0 {
		id = 1 / float64(d)
	}
	putF16(dst[0:2], d)
	for j := 0; j < 16; j++ {
		n0 := clampRound(float64(src[j])*id+8, 0, 15)
		n1 := clampRound(float64(src[j+16])*id+8, 0, 15)
		dst[2+j] = byte(n0) | byte(n1)<<4
		rec[j] = d * float32(n0-8)
		rec[j+16] = d * float32(n1-8)
	}
	return weightedSSE(src, rec, imp)
}

func packQ4_1(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	mn, mx := math.Inf(1), math.Inf(-1)
	for _, v := range src {
		if float64(v) < mn {
			mn = float64(v)
		}
		if float64(v) > mx {
			mx = float64(v)
		}
	}
	d := f16rt(float32((mx - mn) / 15 * shrink))
	m := f16rt(float32(mn))
	id := 0.0
	if d != 0 {
		id = 1 / float64(d)
	}
	putF16(dst[0:2], d)
	putF16(dst[2:4], m)
	for j := 0; j < 16; j++ {
		n0 := clampRound((float64(src[j])-float64(m))*id, 0, 15)
		n1 := clampRound((float64(src[j+16])-float64(m))*id, 0, 15)
		dst[4+j] = byte(n0) | byte(n1)<<4
		rec[j] = d*float32(n0) + m
		rec[j+16] = d*float32(n1) + m
	}
	return weightedSSE(src, rec, imp)
}

func packQ5_0(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	amax := 0.0
	for _, v := range src {
		if a := math.Abs(float64(v)); a > amax {
			amax = a
		}
	}
	d := f16rt(float32(amax / 16 * shrink))
	id := 0.0
	if d != 0 {
		id = 1 / float64(d)
	}
	putF16(dst[0:2], d)
	var qh uint32
	for j := 0; j < 16; j++ {
		n0 := clampRound(float64(src[j])*id+16, 0, 31)
		n1 := clampRound(float64(src[j+16])*id+16, 0, 31)
		dst[6+j] = byte(n0&0xF) | byte(n1&0xF)<<4
		if n0&0x10 != 0 {
			qh |= 1 << uint(j)
		}
		if n1&0x10 != 0 {
			qh |= 1 << uint(j+16)
		}
		rec[j] = d * float32(n0-16)
		rec[j+16] = d * float32(n1-16)
	}
	binary.LittleEndian.PutUint32(dst[2:6], qh)
	return weightedSSE(src, rec, imp)
}

func packQ5_1(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	mn, mx := math.Inf(1), math.Inf(-1)
	for _, v := range src {
		if float64(v) < mn {
			mn = float64(v)
		}
		if float64(v) > mx {
			mx = float64(v)
		}
	}
	d := f16rt(float32((mx - mn) / 31 * shrink))
	m := f16rt(float32(mn))
	id := 0.0
	if d != 0 {
		id = 1 / float64(d)
	}
	putF16(dst[0:2], d)
	putF16(dst[2:4], m)
	var qh uint32
	for j := 0; j < 16; j++ {
		n0 := clampRound((float64(src[j])-float64(m))*id, 0, 31)
		n1 := clampRound((float64(src[j+16])-float64(m))*id, 0, 31)
		dst[8+j] = byte(n0&0xF) | byte(n1&0xF)<<4
		if n0&0x10 != 0 {
			qh |= 1 << uint(j)
		}
		if n1&0x10 != 0 {
			qh |= 1 << uint(j+16)
		}
		rec[j] = d*float32(n0) + m
		rec[j+16] = d*float32(n1) + m
	}
	binary.LittleEndian.PutUint32(dst[4:8], qh)
	return weightedSSE(src, rec, imp)
}

func packQ2K(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	var scales, mins [16]float64
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
			if d != 0 {
				sc[j] = byte(clampRound(scales[j]/d, 0, 15))
			}
		}
	}
	if maxMin > 0 {
		dmin = f16(float32(maxMin / 15))
		for j := 0; j < 16; j++ {
			if dmin != 0 {
				sc[j] |= byte(clampRound(mins[j]/dmin, 0, 15)) << 4
			}
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
	putF16(dst[0:2], float32(d))
	putF16(dst[2:4], float32(dmin))
	copy(dst[4:20], sc[:])
	for j := 0; j < qkK; j += 128 {
		for l := 0; l < 32; l++ {
			dst[20+j/4+l] = byte(L[j+l] | L[j+l+32]<<2 | L[j+l+64]<<4 | L[j+l+96]<<6)
		}
	}
	// Dequant into rec using the same layout as q2Krt.
	qs := dst[20:84]
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
	return weightedSSE(src, rec, imp)
}

func packQ3K(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
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
	return weightedSSE(src, rec, imp)
}

func packQ4K(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	var scales, mins [8]float64
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
		ls, lm := 0, 0
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
	return weightedSSE(src, rec, imp)
}

func packQ5K(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	var scales, mins [8]float64
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
		ls, lm := 0, 0
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
	putF16(dst[0:2], float32(d))
	putF16(dst[2:4], float32(dmin))
	copy(dst[4:16], packed[:])
	ql := dst[48:176]
	qh := dst[16:48]
	for i := range qh {
		qh[i] = 0
	}
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
	return weightedSSE(src, rec, imp)
}

func packQ6K(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	// ggml block_q6_K: ql[128], qh[64], scales int8[16], d f16 — same
	// nibble interleave as q6Krt. Do not ship a flattened ql[i]=L[i] layout.
	var scales [16]float64
	maxAbs, maxSigned := 0.0, 0.0
	for j := 0; j < 16; j++ {
		amax, signed := 0.0, 0.0
		for i := 0; i < 16; i++ {
			v := float64(src[16*j+i])
			if a := math.Abs(v); a > amax {
				amax, signed = a, v
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
	d := 0.0
	var sc [16]int8
	if maxAbs > 0 {
		d = f16(float32(maxSigned * shrink / -128))
		if d != 0 {
			for j := 0; j < 16; j++ {
				sc[j] = int8(clampRound(scales[j]/d, -128, 127))
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
	ql := dst[0:128]
	qh := dst[128:192]
	for i := range ql {
		ql[i] = 0
	}
	for i := range qh {
		qh[i] = 0
	}
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
	return weightedSSE(src, rec, imp)
}

func int8Bytes(sc []int8) []byte {
	out := make([]byte, len(sc))
	for i, v := range sc {
		out[i] = byte(v)
	}
	return out
}
