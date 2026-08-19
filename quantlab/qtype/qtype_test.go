package qtype

import (
	"math"
	"math/rand"
	"testing"

	"quantlab/core"
)

func TestF16RoundTrip(t *testing.T) {
	cases := []struct {
		in   float32
		want uint16
	}{
		{0, 0x0000},
		{1, 0x3c00},
		{-1, 0xbc00},
		{0.5, 0x3800},
		{65504, 0x7bff},
		{65520, 0x7c00},  // overflow -> inf
		{-65520, 0xfc00}, // overflow -> -inf
		{5.960464477539063e-08, 0x0001},
		{2.9802322e-08, 0x0000}, // exact tie rounds to even (zero)
	}
	for _, c := range cases {
		if got := F16Bits(c.in); got != c.want {
			t.Errorf("F16Bits(%v) = %#04x, want %#04x", c.in, got, c.want)
		}
	}
}

func TestBlockSizes(t *testing.T) {
	for _, d := range SupportedTypes() {
		g, ok := d.Geometry()
		if !ok {
			t.Fatalf("%s: no geometry", d)
		}
		if BlockSize(d) != int(g.BlockSize) || TypeSize(d) != int(g.TypeSize) {
			t.Errorf("%s: block/type size mismatch with geometry", d)
		}
	}
}

// relErr bounds: reconstruction error must stay within a small multiple of
// the level spacing implied by the format's bits per weight.
func TestRoundTripErrorBounds(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	block := make([]float32, 256)
	for i := range block {
		block[i] = float32(rng.NormFloat64())
	}
	bounds := map[core.DType]float64{
		core.DTypeQ8_0:   1e-3,
		core.DTypeQ6_K:   2e-3,
		core.DTypeQ5_K_T: 4e-3,
		core.DTypeQ4_K_T: 8e-3,
		core.DTypeQ5_1:   4e-3,
		core.DTypeQ5_0:   4e-3,
		core.DTypeQ4_1:   8e-3,
		core.DTypeQ4_0:   8e-3,
		core.DTypeQ3_K:   5e-2,
		core.DTypeQ2_K:   1.2e-1,
		core.DTypeIQ4_NL: 8e-3,
		core.DTypeIQ3_S:  5e-2,
		core.DTypeIQ2_S:  1.5e-1,
	}
	for d, bound := range bounds {
		work := append([]float32(nil), block...)
		sse, err := QuantizeDequant(d, work, nil)
		if err != nil {
			t.Fatalf("%s: %v", d, err)
		}
		mse := sse / float64(len(block))
		if mse > bound {
			t.Errorf("%s: mse %.6f exceeds bound %.6f", d, mse, bound)
		}
		// Monotone sanity: lower bpw formats have higher error on the same
		// data (checked pairwise over the ordered ladder below).
	}
	ladder := []core.DType{
		core.DTypeQ8_0, core.DTypeQ6_K, core.DTypeQ5_K_T, core.DTypeQ4_K_T,
		core.DTypeQ3_K, core.DTypeQ2_K,
	}
	prev := math.Inf(-1)
	for _, d := range ladder {
		work := append([]float32(nil), block...)
		sse, _ := QuantizeDequant(d, work, nil)
		mse := sse / float64(len(block))
		if mse < prev {
			t.Errorf("%s: mse %.6f lower than higher-fidelity neighbor %.6f", d, mse, prev)
		}
		prev = mse
	}
}

// The importance-weighted scale search must reduce error on high-weight
// elements relative to the unweighted search on the same block.
func TestWeightedErrorSkew(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	block := make([]float32, 256)
	for i := range block {
		block[i] = float32(rng.NormFloat64())
	}
	imp := make([]float32, 256)
	for i := range imp {
		if i%32 < 16 {
			imp[i] = 100
		} else {
			imp[i] = 1
		}
	}
	for _, d := range []core.DType{core.DTypeQ4_0, core.DTypeQ8_0, core.DTypeQ4_K_T, core.DTypeQ6_K} {
		var hiNil, hiImp float64
		for weighted := range []int{0, 1} {
			work := append([]float32(nil), block...)
			var w []float32
			if weighted == 1 {
				w = imp
			}
			bs := BlockSize(d)
			for off := 0; off < len(work); off += bs {
				blockRoundTrip(d, work[off:off+bs], impSlice(w, off, bs))
				for i := off; i < off+bs; i++ {
					e := float64(block[i]) - float64(work[i])
					if i%32 < 16 {
						if weighted == 1 {
							hiImp += e * e
						} else {
							hiNil += e * e
						}
					}
				}
			}
		}
		if hiImp > hiNil*1.001 {
			t.Errorf("%s: weighted search high-imp error %.4f exceeds unweighted %.4f", d, hiImp, hiNil)
		}
	}
}

func TestWeightedErrorDoesNotMutate(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	block := make([]float32, 256)
	imp := make([]float32, 256)
	for i := range block {
		block[i] = float32(rng.NormFloat64())
		imp[i] = float32(1 + rng.Float64())
	}
	orig := append([]float32(nil), block...)
	for _, d := range SupportedTypes() {
		if _, err := WeightedError(d, block, imp); err != nil {
			t.Fatalf("%s: %v", d, err)
		}
	}
	for i := range block {
		if block[i] != orig[i] {
			t.Fatalf("WeightedError mutated src at %d", i)
		}
	}
}

// Q4_0 with d = 1 and integer levels in [-8, 7] reconstructs exactly.
func TestQ4_0GridAligned(t *testing.T) {
	block := make([]float32, 32)
	for i := range block {
		block[i] = float32(i%16 - 8)
	}
	work := append([]float32(nil), block...)
	sse, err := QuantizeDequant(core.DTypeQ4_0, work, nil)
	if err != nil {
		t.Fatal(err)
	}
	if sse > 1e-9 {
		t.Errorf("grid-aligned block has sse %g, want ~0", sse)
	}
}

func TestScaleMinK4RoundTrip(t *testing.T) {
	for sc := 0; sc < 64; sc++ {
		for m := 0; m < 64; m++ {
			var packed [12]byte
			j := sc // pack index j<4 path
			_ = j
			// pack (sc, m) at index 0 and 5 to cover both layout branches
			for _, idx := range []int{0, 5} {
				s, mm := sc, m
				if idx < 4 {
					packed[idx] = byte(s)
					packed[idx+4] = byte(mm)
				} else {
					packed[idx+4] = byte(s&0xF) | byte(mm&0xF)<<4
					packed[idx-4] |= byte(s>>4) << 6
					packed[idx] |= byte(mm>>4) << 6
				}
			}
			for _, idx := range []int{0, 5} {
				s, mm := scaleMinK4(packed[:], idx)
				if s != sc || mm != m {
					t.Fatalf("scaleMinK4(%d,%d) = (%d,%d)", sc, m, s, mm)
				}
			}
			// reset for next iteration
			packed = [12]byte{}
			_ = packed
		}
	}
}

func TestUnsupportedType(t *testing.T) {
	if _, err := QuantizeDequant(core.DTypeIQ1_S, make([]float32, 256), nil); err == nil {
		t.Fatal("expected error for unsupported dtype")
	}
	if Supported(core.DTypeIQ1_S) {
		t.Fatal("IQ1_S must not be Supported")
	}
}

func TestIQ3SBeatsQ3KOnSpikyBlock(t *testing.T) {
	block := make([]float32, 256)
	imp := make([]float32, 256)
	for i := range block {
		block[i] = 0.05
		imp[i] = 1
	}
	for i := 0; i < 8; i++ {
		block[i*32] = 3
		imp[i*32] = 20
	}
	q3 := append([]float32(nil), block...)
	iq := append([]float32(nil), block...)
	e3, err := QuantizeDequant(core.DTypeQ3_K, q3, imp)
	if err != nil {
		t.Fatal(err)
	}
	ei, err := QuantizeDequant(core.DTypeIQ3_S, iq, imp)
	if err != nil {
		t.Fatal(err)
	}
	var w3, wi float64
	for i := range block {
		d3 := float64(block[i]) - float64(q3[i])
		di := float64(block[i]) - float64(iq[i])
		w3 += float64(imp[i]) * d3 * d3
		wi += float64(imp[i]) * di * di
	}
	if wi > w3 {
		t.Errorf("IQ3_S weighted sse %v (unweighted %v) should beat Q3_K %v (%v) on outlier block", wi, ei, w3, e3)
	}
}

func BenchmarkQ6KBlock(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	block := make([]float32, 256)
	for i := range block {
		block[i] = float32(rng.NormFloat64())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blockRoundTrip(core.DTypeQ6_K, block, nil)
	}
}
