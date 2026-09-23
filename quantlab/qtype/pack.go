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
	// Viterbi is retained for compatibility. It historically ran a DP over
	// per-superblock shrink factors; K packers now use llama.cpp's
	// least-squares fit, which has no shrink factor, so it is a no-op.
	Viterbi bool
	// RowLen is the contiguous row length (GGUF ne0). Zero treats src as one row.
	RowLen int
	// Shrink is retained for compatibility with non-K packers; K packers
	// use llama.cpp's least-squares fit and ignore it.
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
	// K packers port llama.cpp's least-squares fit and no longer have a
	// shrink factor to search, so the Viterbi DP degenerates; keep the
	// direct path for them.
	useVit := opt.Viterbi && bs == qkK && opt.Shrink == 0 && nBlocksRow > 1 && !isKQuant(d.BaseTensorType())
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

func isKQuant(d core.DType) bool {
	switch d {
	case core.DTypeQ2_K, core.DTypeQ3_K, core.DTypeQ4_K_T, core.DTypeQ5_K_T, core.DTypeQ6_K:
		return true
	}
	return false
}

func packOne(d core.DType, src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	if shrink <= 0 && isKQuant(d.BaseTensorType()) {
		return packOneShrink(d, src, imp, dst, rec, 0)
	}
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
	encodeQ2KBlockInto(src, imp, dst, rec)
	return weightedSSE(src, rec, imp)
}

func packQ3K(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	encodeQ3KBlockInto(src, imp, dst, rec)
	return weightedSSE(src, rec, imp)
}

func packQ4K(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	encodeQ4KBlockInto(src, imp, dst, rec)
	return weightedSSE(src, rec, imp)
}

func packQ5K(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	encodeQ5KBlockInto(src, imp, dst, rec)
	return weightedSSE(src, rec, imp)
}

func packQ6K(src, imp []float32, dst []byte, rec []float32, shrink float64) float64 {
	encodeQ6KBlockInto(src, imp, dst, rec)
	return weightedSSE(src, rec, imp)
}

func int8Bytes(sc []int8) []byte {
	out := make([]byte, len(sc))
	for i, v := range sc {
		out[i] = byte(v)
	}
	return out
}
