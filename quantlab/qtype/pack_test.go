package qtype

import (
	"math/rand"
	"testing"

	"quantlab/core"
)

func TestPackQ4_0MatchesRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	src := make([]float32, 32)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
	}
	orig := append([]float32(nil), src...)
	packed, rec, err := Pack(core.DTypeQ4_0, src, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(packed) != TypeSize(core.DTypeQ4_0) {
		t.Fatalf("packed %d bytes, want %d", len(packed), TypeSize(core.DTypeQ4_0))
	}
	work := append([]float32(nil), orig...)
	if _, err := QuantizeDequant(core.DTypeQ4_0, work, nil); err != nil {
		t.Fatal(err)
	}
	var a, b float64
	for i := range rec {
		e1 := float64(orig[i] - rec[i])
		e2 := float64(orig[i] - work[i])
		a += e1 * e1
		b += e2 * e2
	}
	if a > b*1.05+1e-6 {
		t.Fatalf("pack SSE %g > roundtrip %g", a, b)
	}
}

func TestViterbiNotWorseThanIndependent(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const n = 256 * 4
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
		if i >= 256 && i < 512 {
			src[i] *= 6
		}
	}
	_, recN, err := PackOpts(core.DTypeQ4_K_T, src, nil, PackOptions{RowLen: n})
	if err != nil {
		t.Fatal(err)
	}
	_, recV, err := PackOpts(core.DTypeQ4_K_T, src, nil, PackOptions{RowLen: n, Viterbi: true})
	if err != nil {
		t.Fatal(err)
	}
	sse := func(rec []float32) float64 {
		var s float64
		for i := range src {
			e := float64(src[i] - rec[i])
			s += e * e
		}
		return s
	}
	if sse(recV) > sse(recN)*1.02+1e-6 {
		t.Fatalf("viterbi SSE %g worse than independent %g", sse(recV), sse(recN))
	}
}

func TestPackQ6KMatchesRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	src := make([]float32, 256)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
	}
	orig := append([]float32(nil), src...)
	packed, rec, err := Pack(core.DTypeQ6_K, src, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(packed) != TypeSize(core.DTypeQ6_K) {
		t.Fatalf("packed %d bytes, want %d", len(packed), TypeSize(core.DTypeQ6_K))
	}
	work := append([]float32(nil), orig...)
	if _, err := QuantizeDequant(core.DTypeQ6_K, work, nil); err != nil {
		t.Fatal(err)
	}
	var a, b float64
	for i := range rec {
		e1 := float64(orig[i] - rec[i])
		e2 := float64(orig[i] - work[i])
		a += e1 * e1
		b += e2 * e2
	}
	if a > b*1.05+1e-6 {
		t.Fatalf("pack SSE %g > roundtrip %g", a, b)
	}
}

func TestPackQ3KMatchesRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	src := make([]float32, 256)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
	}
	orig := append([]float32(nil), src...)
	_, rec, err := Pack(core.DTypeQ3_K, src, nil)
	if err != nil {
		t.Fatal(err)
	}
	work := append([]float32(nil), orig...)
	if _, err := QuantizeDequant(core.DTypeQ3_K, work, nil); err != nil {
		t.Fatal(err)
	}
	var a, b float64
	for i := range rec {
		e1 := float64(orig[i] - rec[i])
		e2 := float64(orig[i] - work[i])
		a += e1 * e1
		b += e2 * e2
	}
	if a > b*1.05+1e-6 {
		t.Fatalf("pack SSE %g > roundtrip %g", a, b)
	}
}

// decodeQ2KBlockGGML is an independent decoder written from ggml's
// dequantize_row_q2_K field order: scales[16] at bytes 0-15, qs[64] at
// 16-79, f16 d at 80-81, f16 dmin at 82-83.
func decodeQ2KBlockGGML(dst []byte) []float32 {
	d := f16tof32(uint16(dst[80]) | uint16(dst[81])<<8)
	dmin := f16tof32(uint16(dst[82]) | uint16(dst[83])<<8)
	out := make([]float32, 256)
	pos := 0
	is := 0
	for n := 0; n < 256; n += 128 {
		q := dst[16+n/4:]
		shift := 0
		for j := 0; j < 4; j++ {
			sc := dst[is]
			is++
			dl := d * float32(sc&0xF)
			ml := dmin * float32(sc>>4)
			for l := 0; l < 16; l++ {
				out[pos] = dl*float32((q[l]>>shift)&3) - ml
				pos++
			}
			sc = dst[is]
			is++
			dl = d * float32(sc&0xF)
			ml = dmin * float32(sc>>4)
			for l := 0; l < 16; l++ {
				out[pos] = dl*float32((q[l+16]>>shift)&3) - ml
				pos++
			}
			shift += 2
		}
	}
	return out
}

func TestPackQ2KMatchesRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	src := make([]float32, 256)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
	}
	orig := append([]float32(nil), src...)
	packed, rec, err := Pack(core.DTypeQ2_K, src, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(packed) != TypeSize(core.DTypeQ2_K) {
		t.Fatalf("packed %d bytes, want %d", len(packed), TypeSize(core.DTypeQ2_K))
	}
	work := append([]float32(nil), orig...)
	if _, err := QuantizeDequant(core.DTypeQ2_K, work, nil); err != nil {
		t.Fatal(err)
	}
	for i := range rec {
		if rec[i] != work[i] {
			t.Fatalf("pack rec[%d] = %v, roundtrip %v", i, rec[i], work[i])
		}
	}
	// Independent ggml-layout decoder must reproduce the same values.
	dec := decodeQ2KBlockGGML(packed)
	for i := range dec {
		e := dec[i] - rec[i]
		if e > 1e-6 || e < -1e-6 {
			t.Fatalf("ggml decode[%d] = %v, pack rec %v", i, dec[i], rec[i])
		}
	}
}

func TestPackQ2KLayoutFields(t *testing.T) {
	// Non-uniform block so d/dmin/scales/qs all carry signal.
	src := make([]float32, 256)
	for i := range src {
		src[i] = float32(i%17) - 8 + 0.35*float32(i%3)
	}
	packed, rec, err := Pack(core.DTypeQ2_K, src, nil)
	if err != nil {
		t.Fatal(err)
	}
	dec := decodeQ2KBlockGGML(packed)
	for i := range dec {
		e := dec[i] - rec[i]
		if e > 1e-5 || e < -1e-5 {
			t.Fatalf("decode[%d] = %v, want %v", i, dec[i], rec[i])
		}
	}
}
