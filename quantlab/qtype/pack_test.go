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
