package profile

import (
	"math"
	"math/rand"
	"testing"

	"quantlab/anchor"
	"quantlab/core"
)

func gaussRows(rows, ne0 int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	v := make([]float32, rows*ne0)
	for i := range v {
		v[i] = float32(rng.NormFloat64() * 0.02)
	}
	return v
}

func TestEmbeddingRowErrorsOrdersRungs(t *testing.T) {
	const ne0 = 256
	vals := gaussRows(512, ne0, 1)
	// One spiky row: a few huge entries starve the block scale.
	for c := 0; c < 4; c++ {
		vals[7*ne0+c*64] = 1.5
	}
	// An all-zero row is skipped, not counted as error 0 or NaN.
	for c := 0; c < ne0; c++ {
		vals[9*ne0+c] = 0
	}
	stats := EmbeddingRowErrors(vals, ne0, EmbedRowDTypes)
	prev := 0.0
	for _, d := range []core.DType{core.DTypeQ8_0, core.DTypeQ6_K, core.DTypeQ5_K_T, core.DTypeQ4_K_T, core.DTypeQ3_K, core.DTypeQ2_K} {
		st, ok := stats[d]
		if !ok {
			t.Fatalf("no stats for %s", d)
		}
		if st.Rows != 511 {
			t.Errorf("%s rows = %d, want 511 (zero row skipped)", d, st.Rows)
		}
		if !(st.Median > prev) {
			t.Errorf("%s median %v not above higher-fidelity rung's %v", d, st.Median, prev)
		}
		if st.P999 < st.Median || st.Max < st.P999 {
			t.Errorf("%s percentiles out of order: %+v", d, st)
		}
		prev = st.Median
	}
}

func TestYardstickErrorInterpolates(t *testing.T) {
	stats := map[core.DType]RowErrorStats{
		core.DTypeQ4_K_T: {Median: 0.08}, // 4.5 bpw
		core.DTypeQ5_K_T: {Median: 0.04}, // 5.5 bpw
	}
	y, ok := YardstickError(5.0, stats)
	if !ok || math.Abs(y-math.Sqrt(0.08*0.04)) > 1e-12 {
		t.Errorf("yardstick at 5.0 = %v, want geometric midpoint %v", y, math.Sqrt(0.08*0.04))
	}
	if y, _ := YardstickError(9, stats); y != 0.04 {
		t.Errorf("above the top rung = %v, want clamp 0.04", y)
	}
	if y, _ := YardstickError(2, stats); y != 0.08 {
		t.Errorf("below the bottom rung = %v, want clamp 0.08", y)
	}
}

func TestChooseEmbeddingFloor(t *testing.T) {
	stats := map[core.DType]RowErrorStats{
		core.DTypeQ8_0:   {Median: 0.004, P999: 0.006},
		core.DTypeQ6_K:   {Median: 0.015, P999: 0.022},
		core.DTypeQ5_K_T: {Median: 0.03, P999: 0.045},
		core.DTypeQ4_K_T: {Median: 0.06, P999: 0.09},
	}
	// Target-rate typical error 0.05: Q5_K's worst rows (0.045) qualify.
	if d, ok := ChooseEmbeddingFloor(stats, 0.05, core.DTypeQ6_K, anchor.Rank); !ok || d != core.DTypeQ5_K_T {
		t.Errorf("floor = %s %v, want Q5_K", d, ok)
	}
	// Never above the policy floor, even when only Q8_0 would qualify.
	if d, ok := ChooseEmbeddingFloor(stats, 0.007, core.DTypeQ6_K, anchor.Rank); ok {
		t.Errorf("floor = %s, want no relaxation above the Q6_K policy", d)
	}
	// Spiky table: worst rows fail everywhere below the policy.
	spiky := map[core.DType]RowErrorStats{
		core.DTypeQ6_K:   {Median: 0.015, P999: 0.4},
		core.DTypeQ5_K_T: {Median: 0.03, P999: 0.8},
	}
	if _, ok := ChooseEmbeddingFloor(spiky, 0.05, core.DTypeQ6_K, anchor.Rank); ok {
		t.Error("spiky rows must keep the policy floor")
	}
}

func TestEmbeddingFloorEndToEndOnGaussianRows(t *testing.T) {
	const ne0 = 256
	stats := EmbeddingRowErrors(gaussRows(2048, ne0, 3), ne0, EmbedRowDTypes)
	y, ok := YardstickError(4.5, stats)
	if !ok {
		t.Fatal("no yardstick")
	}
	d, ok := ChooseEmbeddingFloor(stats, y, core.DTypeQ6_K, anchor.Rank)
	// Well-behaved rows at a Q4-class target: a Q5_K table keeps every
	// token at or under a typical Q4_K row's noise.
	if !ok || anchor.Rank(d) <= anchor.Rank(core.DTypeQ6_K) {
		t.Errorf("gaussian rows at 4.5 bpw: floor = %s %v, want cheaper than Q6_K", d, ok)
	}
}
