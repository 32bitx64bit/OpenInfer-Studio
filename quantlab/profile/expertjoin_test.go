package profile

import (
	"math"
	"testing"

	"quantlab/core"
)

func TestStripExpertSegment(t *testing.T) {
	cases := []struct {
		in   string
		base string
		idx  int
		ok   bool
	}{
		{"blk.0.experts.17.ffn_up", "blk.0.ffn_up", 17, true},
		{"blk.0.expert.3.ffn_down.weight", "blk.0.ffn_down.weight", 3, true},
		{"blk.0.mlp.experts.2.gate_proj", "blk.0.mlp.gate_proj", 2, true},
		{"blk.0.ffn_up", "", 0, false},
		{"blk.0.experts.ffn_up", "", 0, false},
	}
	for _, c := range cases {
		base, idx, ok := stripExpertSegment(c.in)
		if base != c.base || idx != c.idx || ok != c.ok {
			t.Errorf("stripExpertSegment(%q) = (%q, %d, %v), want (%q, %d, %v)",
				c.in, base, idx, ok, c.base, c.idx, c.ok)
		}
	}
}

func TestJoinExpertImatrix(t *testing.T) {
	bank := &core.TensorBank{SourcePath: "/x", ModelID: "m", Tensors: []core.TensorDesc{
		{Name: "blk.0.ffn_up_exps.weight", DType: core.DTypeF32,
			Shape: []uint64{256, 4, 2}, Elements: 256 * 4 * 2, Length: 256 * 4 * 2 * 4},
	}}
	// expert 7: hot (many samples), expert 2: cold; both with distinct means.
	imatrix := map[string]ImatrixStats{
		"blk.0.experts.7.ffn_up": {
			Mean: 10, Samples: 1000, Values: []float32{10, 10, 10, 10},
		},
		"blk.0.experts.2.ffn_up": {
			Mean: 1, Samples: 100, Values: []float32{1, 1, 1, 1},
		},
	}
	joined := JoinExpertImatrix(imatrix, bank)
	if _, dup := joined["blk.0.experts.7.ffn_up"]; !dup {
		t.Fatal("original entries must survive")
	}
	st, ok := joined["blk.0.ffn_up_exps.weight"]
	if !ok {
		t.Fatalf("no synthesized fused entry: %v", joined)
	}
	// Utilization weighting: hot expert (util 1) contributes ~1/unit-mean;
	// cold expert contributes (0.1)^0.5 ≈ 0.316 ordered by expert index.
	if len(st.Values) != 8 {
		t.Fatalf("joined values len %d, want 8", len(st.Values))
	}
	if abs(st.Values[0]-float32(math.Pow(0.1, 0.5))) > 0.02 {
		t.Errorf("cold expert mean weight %v, want ~0.316", st.Values[0])
	}
	if abs(st.Values[4]-1) > 0.02 {
		t.Errorf("hot expert mean weight %v, want ~1", st.Values[4])
	}
	if st.Samples != 1100 {
		t.Errorf("samples %d, want 1100", st.Samples)
	}
	// Estimator must now see importance under the fused name too.
	est := NewFallbackEstimator(joined)
	if !est.HasImportance("blk.0.ffn_up_exps.weight") {
		t.Error("estimator does not see the fused entry")
	}
}

func TestJoinExpertImatrixDirectHitWins(t *testing.T) {
	bank := &core.TensorBank{SourcePath: "/x", ModelID: "m", Tensors: []core.TensorDesc{
		{Name: "blk.0.ffn_up_exps.weight", DType: core.DTypeF32,
			Shape: []uint64{256, 4, 2}, Elements: 256 * 4 * 2, Length: 256 * 4 * 2 * 4},
	}}
	direct := ImatrixStats{Mean: 3, Samples: 42, Values: []float32{3}}
	imatrix := map[string]ImatrixStats{
		"blk.0.ffn_up_exps":      direct,
		"blk.0.experts.7.ffn_up": {Mean: 10, Samples: 1000, Values: []float32{10}},
	}
	got := JoinExpertImatrix(imatrix, bank)
	if got["blk.0.ffn_up_exps"].Samples != 42 {
		t.Errorf("direct hit must survive; samples %d, want 42", got["blk.0.ffn_up_exps"].Samples)
	}
	if _, synthesized := got["blk.0.ffn_up_exps.weight"]; synthesized {
		t.Error("synthesis must not overwrite the direct hit (.weight alternate resolves it)")
	}
	// The estimator resolves the direct hit through its .weight alternate.
	est := NewFallbackEstimator(got)
	if !est.HasImportance("blk.0.ffn_up_exps.weight") {
		t.Error("estimator must resolve the direct hit via alternate name")
	}
}

func abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
