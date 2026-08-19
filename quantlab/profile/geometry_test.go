package profile

import (
	"strconv"
	"testing"

	"quantlab/anchor"
	"quantlab/core"
)

func TestGeometryUniqueLargeAxisRaisesLoss(t *testing.T) {
	bank := &core.TensorBank{
		SourcePath: "/m.gguf",
		Tensors: []core.TensorDesc{
			{Name: "blk.0.attn_norm.weight", DType: core.DTypeF32, Shape: []uint64{256}, Length: 1024, Elements: 256},
			{Name: "token_embd.weight", DType: core.DTypeF16, Shape: []uint64{256, 32000}, Length: 16384000, Elements: 8192000},
			{Name: "blk.0.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
		},
	}
	est := NewFallbackEstimator(nil)
	est.BindBank(bank)
	emb := bank.Tensors[1]
	q := bank.Tensors[2]
	le, _ := est.Estimate(emb, core.DTypeQ4_K_T)
	lq, _ := est.Estimate(q, core.DTypeQ4_K_T)
	if le/float64(emb.Elements) <= lq/float64(q.Elements) {
		t.Fatalf("unique-large-axis per-element loss %g not above regular %g", le/float64(emb.Elements), lq/float64(q.Elements))
	}
}

// magrLikeStats is an imatrix aggregate that used to trip a whole-tensor Q6
// hard floor (Max/Mean = 64, Spikiness = 32). MagR already pins a handful of
// outlier weights inside the tensor; a dtype floor on the rest made 3.5 bpw
// Dynamic jobs infeasible on Qwen3.5-class hybrids.
func magrLikeStats() ImatrixStats {
	return ImatrixStats{Mean: 1, Max: 64, P50: 1, P95: 32, Spikiness: 32, Samples: 4096}
}

func TestSpikyImatrixDoesNotHardFloorQ6(t *testing.T) {
	tns := core.TensorDesc{
		Name: "blk.0.ffn_down.weight", DType: core.DTypeF16,
		Shape: []uint64{256, 256}, Length: 131072, Elements: 65536,
	}
	est := NewFallbackEstimator(map[string]ImatrixStats{tns.Name: magrLikeStats()})
	opts, err := EnumerateOptions(tns, []core.DType{core.DTypeQ8_0, core.DTypeQ6_K, core.DTypeQ4_K_T, core.DTypeQ2_K}, &anchor.Set{}, nil, est, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	hasQ2 := false
	for _, o := range opts {
		if o.Target == core.DTypeQ2_K {
			hasQ2 = true
		}
	}
	if !hasQ2 {
		t.Fatal("MagR-like imatrix spike must still enumerate Q2; MagR pins outliers, it does not Q6 the tensor")
	}
}

func TestSpikyImatrixQ3BudgetStaysFeasible(t *testing.T) {
	// 48 FFN downs at Qwen3.5-27B ffn_down shape. A Q6 hard floor on these
	// alone exceeds a 3.5 bpw budget; Q2_K does not.
	const layers = 48
	bank := &core.TensorBank{SourcePath: "/m.gguf", ModelID: "qwen35-ffn"}
	imatrix := map[string]ImatrixStats{}
	for i := 0; i < layers; i++ {
		td := weightTD("blk."+strconv.Itoa(i)+".ffn_down.weight", 17408, 5120)
		bank.Tensors = append(bank.Tensors, td)
		imatrix[td.Name] = magrLikeStats()
	}
	set, err := anchor.Derive(bank, nil, anchor.PolicyForBPW(3.5))
	if err != nil {
		t.Fatal(err)
	}
	cands := []core.DType{core.DTypeQ8_0, core.DTypeQ6_K, core.DTypeQ4_K_T, core.DTypeQ2_K, core.DTypeIQ2_XXS}
	if _, err := Solve(Request{
		Bank: bank, Anchors: set, Candidates: cands,
		TargetBPW: 3.5, Imatrix: imatrix,
	}); err != nil {
		t.Fatalf("3.5 bpw must stay feasible with MagR-like imatrix spikes: %v", err)
	}
}
