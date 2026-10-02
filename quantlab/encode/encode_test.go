package encode

import (
	"math/rand"
	"testing"

	"quantlab/core"
	"quantlab/profile"
	"quantlab/qtype"
)

func TestGPTQBeatsRTNOnSyntheticWx(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const ne0, rows, p = 64, 32, 48
	w := make([]float32, ne0*rows)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	sigma := make([]float32, ne0)
	for i := range sigma {
		sigma[i] = 1
	}
	sk := profile.MakeSketches(ne0, p, sigma, 11)
	imp := make([]float32, len(w))
	for i := range imp {
		imp[i] = 1
	}

	sse := func(rec []float32) float64 {
		var s float64
		for _, x := range sk {
			for r := 0; r < rows; r++ {
				var y, yq float64
				for c := 0; c < ne0; c++ {
					y += float64(w[r*ne0+c]) * float64(x[c])
					yq += float64(rec[r*ne0+c]) * float64(x[c])
				}
				e := y - yq
				s += e * e
			}
		}
		return s
	}

	rtn := append([]float32(nil), w...)
	_, recRTN, err := qtype.Pack(core.DTypeQ4_0, rtn, imp)
	if err != nil {
		t.Fatal(err)
	}

	gptq := append([]float32(nil), w...)
	gptqCompensate(gptq, ne0, sk, imp, core.DTypeQ4_0)
	_, recGPTQ, err := qtype.Pack(core.DTypeQ4_0, gptq, imp)
	if err != nil {
		t.Fatal(err)
	}
	if sse(recGPTQ) > sse(recRTN)*0.999 {
		t.Fatalf("GPTQ Wx SSE %g not better than RTN %g", sse(recGPTQ), sse(recRTN))
	}
}

func TestSupportedExcludesIQ(t *testing.T) {
	if !Supported(core.DTypeQ4_K_T) || !Supported(core.DTypeQ4_0) {
		t.Fatal("expected packed K/legacy support")
	}
	if Supported(core.DTypeIQ2_XXS) || Supported(core.DTypeIQ3_S) {
		t.Fatal("IQ must stay on llama-quantize")
	}
}

func TestImportanceUsesExpertLayout(t *testing.T) {
	im := map[string]profile.ImatrixStats{"blk.0.ffn_down_exps.weight": {Values: []float32{1, 2, 10, 20}}}
	got := importance(im, "blk.0.ffn_down_exps.weight", 8, 2) // 2 experts x 2 rows x ne0=2
	want := []float32{1, 2, 1, 2, 10, 20, 10, 20}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("importance = %v, want %v", got, want)
		}
	}
}

func TestGPTQWithoutSketchesIsDiagonalOnly(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const ne0 = 64
	w := make([]float32, ne0*4)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	imp := make([]float32, len(w))
	for i := range imp {
		imp[i] = 1
	}
	got := append([]float32(nil), w...)
	gptqCompensate(got, ne0, nil, imp, core.DTypeQ4_0)
	for i := range w {
		if got[i] != w[i] {
			t.Fatalf("nil sketches must not redistribute error (elem %d changed)", i)
		}
	}
}
