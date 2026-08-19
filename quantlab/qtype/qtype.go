// Package qtype provides format-faithful reference implementations of the
// GGUF per-tensor block quantization formats: quantize a block of weights
// into the on-disk layout, dequantize it back, and measure the
// importance-weighted squared error the format induces.
//
// The layouts and dequantization semantics mirror ggml-quants.c. K/legacy
// types choose per-block scales by a small deterministic importance-weighted
// grid search rather than llama.cpp's iterative refinement. IQ4_NL and IQ4_XS
// round-trip through the packed f16 / 6-bit scale and int8 NL table so the
// exact-loss table matches llama-quantize reconstruction for those formats.
//
// Only per-tensor base types are supported; recipe labels are not.
package qtype

import (
	"fmt"
	"math"

	"quantlab/core"
)

// BlockSize returns the elements per quantization block of d.
func BlockSize(d core.DType) int {
	switch d.BaseTensorType() {
	case core.DTypeQ8_0, core.DTypeQ8_1, core.DTypeQ4_0, core.DTypeQ4_1,
		core.DTypeQ5_0, core.DTypeQ5_1, core.DTypeIQ4_NL, core.DTypeIQ4_XS:
		return 32
	case core.DTypeQ2_K, core.DTypeQ3_K, core.DTypeQ4_K_T, core.DTypeQ5_K_T,
		core.DTypeQ6_K, core.DTypeQ8_K,
		core.DTypeIQ3_S, core.DTypeIQ3_XXS, core.DTypeIQ2_S, core.DTypeIQ2_XS,
		core.DTypeIQ2_XXS:
		return 256
	}
	if d.IsFloat() {
		return 1
	}
	return 0
}

// TypeSize returns the on-disk byte size of one block of d (equal to
// core.BlockGeometry for quant types; float types round to their storage).
func TypeSize(d core.DType) int {
	if g, ok := d.Geometry(); ok {
		return int(g.TypeSize)
	}
	return 0
}

// Supported reports whether d has a reference quantizer implementation.
func Supported(d core.DType) bool {
	switch d.BaseTensorType() {
	case core.DTypeF16, core.DTypeBF16, core.DTypeQ8_0,
		core.DTypeQ4_0, core.DTypeQ4_1, core.DTypeQ5_0, core.DTypeQ5_1,
		core.DTypeQ2_K, core.DTypeQ3_K, core.DTypeQ4_K_T, core.DTypeQ5_K_T,
		core.DTypeQ6_K,
		core.DTypeIQ4_NL, core.DTypeIQ4_XS, core.DTypeIQ3_S, core.DTypeIQ3_XXS,
		core.DTypeIQ2_S, core.DTypeIQ2_XS, core.DTypeIQ2_XXS:
		return true
	}
	return false
}

// SupportedTypes lists every dtype with a reference implementation,
// highest fidelity first.
func SupportedTypes() []core.DType {
	return []core.DType{
		core.DTypeQ8_0,
		core.DTypeQ6_K,
		core.DTypeQ5_K_T,
		core.DTypeIQ4_NL,
		core.DTypeQ4_K_T,
		core.DTypeQ5_1, core.DTypeQ5_0,
		core.DTypeIQ4_XS,
		core.DTypeQ4_1, core.DTypeQ4_0,
		core.DTypeIQ3_S,
		core.DTypeQ3_K,
		core.DTypeIQ3_XXS,
		core.DTypeIQ2_S,
		core.DTypeQ2_K,
		core.DTypeIQ2_XS,
		core.DTypeIQ2_XXS,
	}
}

// QuantizeDequant stores src as dtype d and writes the dequantized
// reconstruction back over src, in place. imp, when non-nil, carries
// per-element importance weights (len == len(src)) that guide the scale
// search. It reports the sum of squared reconstruction error (unweighted).
// The slice length must be a multiple of d's block size.
func QuantizeDequant(d core.DType, src, imp []float32) (float64, error) {
	ws := NewWorkspace(d)
	defer ws.Release()
	return quantizeDequant(d, src, imp, ws)
}

// QuantizeDequantWS is QuantizeDequant with a caller-owned reusable
// workspace (safe across dtypes; it grows as needed).
func QuantizeDequantWS(d core.DType, src, imp []float32, ws *Workspace) (float64, error) {
	return quantizeDequant(d, src, imp, ws)
}

func quantizeDequant(d core.DType, src, imp []float32, ws *Workspace) (float64, error) {
	bs := BlockSize(d)
	if bs == 0 {
		return 0, fmt.Errorf("qtype: no reference quantizer for %s", d)
	}
	if len(src)%bs != 0 {
		return 0, fmt.Errorf("qtype: %d elements not a multiple of %s block size %d", len(src), d, bs)
	}
	if imp != nil && len(imp) != len(src) {
		return 0, fmt.Errorf("qtype: importance length %d != elements %d", len(imp), len(src))
	}
	if ws != nil {
		ws.ensure(bs)
	}
	var sse float64
	for off := 0; off < len(src); off += bs {
		sse += blockRoundTripWS(d, src[off:off+bs], impSlice(imp, off, bs), ws)
	}
	return sse, nil
}

func impSlice(imp []float32, off, n int) []float32 {
	if imp == nil {
		return nil
	}
	return imp[off : off+n]
}

// WeightedError returns the importance-weighted sum of squared error of
// storing src as dtype d, without mutating src. imp must match src length.
func WeightedError(d core.DType, src, imp []float32) (float64, error) {
	if imp == nil {
		imp = ones(len(src))
	}
	bs := BlockSize(d)
	if bs == 0 {
		return 0, fmt.Errorf("qtype: no reference quantizer for %s", d)
	}
	if len(src)%bs != 0 || len(imp) != len(src) {
		return 0, fmt.Errorf("qtype: bad block alignment for %s", d)
	}
	ws := NewWorkspace(d)
	defer ws.Release()
	work := ws.block(d)
	var sum float64
	for off := 0; off < len(src); off += bs {
		copy(work, src[off:off+bs])
		w := imp[off : off+bs]
		blockRoundTripWS(d, work, w, ws)
		for i, q := range work {
			e := float64(src[off+i]) - float64(q)
			sum += float64(w[i]) * e * e
		}
	}
	return sum, nil
}

func ones(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

// Workspace holds reusable scratch buffers for one goroutine's block
// round-trips, amortizing allocations across a whole tensor scan.
type Workspace struct {
	scratch []float32
	best    []float32
	trial   []float32
	pool    *workspacePool
}

type workspacePool struct {
	ch chan *Workspace
}

var wsPool = &workspacePool{ch: make(chan *Workspace, 64)}

// NewWorkspace borrows a workspace from the shared pool.
func NewWorkspace(d core.DType) *Workspace {
	bs := BlockSize(d)
	if bs == 0 {
		bs = 1
	}
	select {
	case w := <-wsPool.ch:
		w.ensure(bs)
		return w
	default:
		w := &Workspace{}
		w.ensure(bs)
		return w
	}
}

func (w *Workspace) ensure(bs int) {
	if cap(w.scratch) < bs || cap(w.best) < bs || cap(w.trial) < bs {
		w.scratch = make([]float32, bs)
		w.best = make([]float32, bs)
		w.trial = make([]float32, bs)
	}
	w.scratch = w.scratch[:bs]
	w.best = w.best[:bs]
	w.trial = w.trial[:bs]
}

func (w *Workspace) block(d core.DType) []float32 {
	w.ensure(BlockSize(d))
	return w.scratch
}

// Release returns the workspace to the shared pool.
func (w *Workspace) Release() {
	if w == nil {
		return
	}
	select {
	case wsPool.ch <- w:
	default:
	}
}

// blockRoundTrip quantizes then dequantizes one block in place (src holds
// the weights on entry and the reconstruction on exit) and returns the
// unweighted sum of squared error.
func blockRoundTrip(d core.DType, src, imp []float32) float64 {
	return blockRoundTripWS(d, src, imp, nil)
}

func blockRoundTripWS(d core.DType, src, imp []float32, ws *Workspace) float64 {
	switch d.BaseTensorType() {
	case core.DTypeF16:
		var sse float64
		for i, v := range src {
			q := f16rt(v)
			src[i] = q
			e := float64(v) - float64(q)
			sse += e * e
		}
		return sse
	case core.DTypeBF16:
		var sse float64
		for i, v := range src {
			q := bf16round(v)
			src[i] = q
			e := float64(v) - float64(q)
			sse += e * e
		}
		return sse
	case core.DTypeQ8_0:
		return q8_0rt(src, imp, ws)
	case core.DTypeQ4_0:
		return q4_0rt(src, imp, ws)
	case core.DTypeQ4_1:
		return q4_1rt(src, imp, ws)
	case core.DTypeQ5_0:
		return q5_0rt(src, imp, ws)
	case core.DTypeQ5_1:
		return q5_1rt(src, imp, ws)
	case core.DTypeQ2_K:
		return q2Krt(src, imp, ws)
	case core.DTypeQ3_K:
		return q3Krt(src, imp, ws)
	case core.DTypeQ4_K_T:
		return q4Krt(src, imp, ws)
	case core.DTypeQ5_K_T:
		return q5Krt(src, imp, ws)
	case core.DTypeQ6_K:
		return q6Krt(src, imp, ws)
	case core.DTypeIQ4_NL:
		return iq4nlrt(src, imp, ws)
	case core.DTypeIQ4_XS:
		return iq4xsrt(src, imp, ws)
	case core.DTypeIQ3_S:
		return iq3srt(src, imp, ws)
	case core.DTypeIQ3_XXS:
		return iq3xxsrt(src, imp, ws)
	case core.DTypeIQ2_S:
		return iq2srt(src, imp, ws)
	case core.DTypeIQ2_XS:
		return iq2xsrt(src, imp, ws)
	case core.DTypeIQ2_XXS:
		return iq2xxsrt(src, imp, ws)
	}
	return 0
}

// F16Bits converts v to IEEE 754 half precision bits (round to nearest
// even, matching GGML_CPU_FP32_TO_FP16).
func F16Bits(v float32) uint16 {
	b := math.Float32bits(v)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23) & 0xff
	man := b & 0x7fffff
	if exp == 0xff {
		return sign | 0x7c00 // inf/nan
	}
	e := exp - 127
	if e > 15 {
		return sign | 0x7c00 // overflow -> inf
	}
	if e >= -14 {
		m := man >> 13
		round := (man>>12)&1 != 0 && (man&0xfff != 0 || m&1 != 0)
		h := uint16((e+15)<<10) | uint16(m)
		if round {
			h++
		}
		return sign | h
	}
	if e >= -25 {
		shift := uint32(-e - 1)
		m := man | 0x800000
		round := (m>>(shift-1))&1 != 0 && (m&((1<<(shift-1))-1) != 0 || (m>>shift)&1 != 0)
		h := uint16(m >> shift)
		if round {
			h++
		}
		return sign | h
	}
	return sign
}

// F16ToF32 expands IEEE 754 half bits to float32.
func F16ToF32(h uint16) float32 { return f16tof32(h) }

// f16tof32 expands IEEE 754 half bits to float32.
func f16tof32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	man := uint32(h & 0x3ff)
	if exp == 0 {
		if man == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(127 - 15 + 1)
		for man&0x400 == 0 {
			man <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (man&0x3ff)<<13)
	}
	if exp == 0x1f {
		return math.Float32frombits(sign | 0x7f800000 | man<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | man<<13)
}

func bf16round(v float32) float32 {
	b := math.Float32bits(v)
	round := (b>>15)&1 != 0 && (b&0x7fff != 0)
	b >>= 16
	if round {
		b++
	}
	return math.Float32frombits(b << 16)
}

// fp16 round-trip used by K-quant scale storage.
func f16rt(v float32) float32 {
	return f16tof32(F16Bits(v))
}
