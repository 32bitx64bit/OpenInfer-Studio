package profile

import (
	"testing"
	"time"

	"quantlab/anchor"
	"quantlab/core"
	"quantlab/kld"
)

func weightTD(name string, rows, cols uint64) core.TensorDesc {
	elems := rows * cols
	b, ok := core.DTypeF16.ExactBytes(elems)
	if !ok {
		panic(name)
	}
	return core.TensorDesc{Name: name, DType: core.DTypeF16, Shape: []uint64{rows, cols}, Length: b, Elements: elems}
}

func TestRoleFactorValues(t *testing.T) {
	const maxL = 5
	cases := []struct {
		name string
		want float64
	}{
		{"token_embd.weight", 1.5},
		{"output.weight", 1.5},
		{"blk.3.attn_v.weight", 1.35},
		{"blk.3.attn_output.weight", 1.35},
		{"blk.3.ffn_down.weight", 1.05},
		{"blk.3.ffn_down_exps.weight", 0.9},
		{"blk.3.attn_q.weight", 1.05},
		{"blk.3.attn_gate.weight", 1.05},
		{"blk.3.ffn_up.weight", 0.85},
		{"blk.3.ffn_gate.weight", 0.85},
		{"blk.3.ffn_up_exps.weight", 0.75},
		{"model.layers.3.mlp.experts.2.gate_proj.weight", 0.75},
		{"blk.3.ssm_out.weight", 1.05},
		{"blk.3.ssm_alpha.weight", 1.05},
		{"blk.3.attn_qkv.weight", 1.1},
	}
	for _, tc := range cases {
		got := roleFactor(tc.name, maxL)
		if got != tc.want {
			t.Errorf("roleFactor(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
	emb := roleFactor("token_embd.weight", maxL)
	v := roleFactor("blk.3.attn_v.weight", maxL)
	down := roleFactor("blk.3.ffn_down.weight", maxL)
	up := roleFactor("blk.3.ffn_up.weight", maxL)
	q := roleFactor("blk.3.attn_q.weight", maxL)
	if !(emb > v && v > down && down > up) {
		t.Errorf("keeper order embed %v > attn_v %v > ffn_down %v > ffn_up %v", emb, v, down, up)
	}
	if q >= v {
		t.Errorf("attn_q factor %v must be below attn_v %v", q, v)
	}
	if roleFactor("blk.3.ffn_down_exps.weight", maxL) >= down {
		t.Errorf("expert ffn_down must be below dense ffn_down %v", down)
	}
	if roleFactor("blk.3.ffn_up_exps.weight", maxL) >= up {
		t.Errorf("expert ffn_up must be below dense ffn_up %v", up)
	}
	first := roleFactor("blk.0.ffn_down.weight", maxL)
	second := roleFactor("blk.1.ffn_down.weight", maxL)
	mid := roleFactor("blk.3.ffn_down.weight", maxL)
	secondLast := roleFactor("blk.4.ffn_down.weight", maxL)
	last := roleFactor("blk.5.ffn_down.weight", maxL)
	if first <= mid || second <= mid || last <= mid || secondLast <= mid {
		t.Errorf("first-two/last-two bump missing: 0=%v 1=%v 3=%v 4=%v 5=%v", first, second, mid, secondLast, last)
	}
}

func TestBindBankAppliesLastBlockBump(t *testing.T) {
	mid := weightTD("blk.3.ffn_down.weight", 64, 64)
	last := weightTD("blk.5.ffn_down.weight", 64, 64)
	bank := &core.TensorBank{Tensors: []core.TensorDesc{mid, last}}
	est := NewFallbackEstimator(nil)
	before, _ := est.Estimate(last, core.DTypeQ4_K_T)
	est.BindBank(bank)
	afterMid, _ := est.Estimate(mid, core.DTypeQ4_K_T)
	afterLast, _ := est.Estimate(last, core.DTypeQ4_K_T)
	if afterLast <= afterMid {
		t.Fatalf("BindBank last-block loss %v should exceed mid %v", afterLast, afterMid)
	}
	if afterLast <= before {
		t.Fatalf("BindBank should raise last-block loss: before %v after %v", before, afterLast)
	}
}

func TestEstimateScalesWithElements(t *testing.T) {
	small := weightTD("blk.1.ffn_up.weight", 64, 64)
	large := weightTD("blk.1.ffn_up.weight", 64, 64*8)
	est := NewFallbackEstimator(nil)
	ls, _ := est.Estimate(small, core.DTypeQ4_K_T)
	ll, _ := est.Estimate(large, core.DTypeQ4_K_T)
	if ls <= 0 || ll/ls < 7.9 || ll/ls > 8.1 {
		t.Fatalf("loss ratio %v/%v = %v, want ~8 (element scale)", ll, ls, ll/ls)
	}
}

func TestSolveDoesNotStarveLargeTensors(t *testing.T) {
	// Architecture-agnostic RD: a 32× larger fused projection must receive
	// bits at a 3.5-ish bpw budget instead of remaining at Q2 while a small
	// attn_v soaks the allocation. Name class must not override size.
	small := weightTD("blk.2.attn_v.weight", 256, 256)
	huge := weightTD("blk.2.attn_qkv.weight", 256, 256*32)
	bank := &core.TensorBank{SourcePath: "/m.gguf", ModelID: "size", Tensors: []core.TensorDesc{small, huge}}
	set, err := anchor.Derive(bank, nil, anchor.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	bQ2s, _ := core.DTypeQ2_K.ExactBytes(small.Elements)
	bQ4h, _ := core.DTypeQ4_K_T.ExactBytes(huge.Elements)
	res, err := Solve(Request{
		Bank: bank, Anchors: set,
		Candidates:  []core.DType{core.DTypeQ6_K, core.DTypeQ4_K_T, core.DTypeQ2_K},
		BudgetBytes: bQ4h + bQ2s + 256,
	})
	if err != nil {
		t.Fatal(err)
	}
	gotS, gotH := targetOf(t, res, small.Name), targetOf(t, res, huge.Name)
	if gotH == core.DTypeQ2_K {
		t.Fatalf("starved large tensor at Q2_K; small=%v huge=%v", gotS, gotH)
	}
}

func TestIQPreferredOnSpikyPareto(t *testing.T) {
	vals := make([]float32, 32)
	vals[31] = 50
	st := aggregateImatrix(vals, nil, true)
	td := weightTD("blk.1.ffn_down.weight", 256, 256)
	est := NewFallbackEstimator(map[string]ImatrixStats{td.Name: st})
	cands := []core.DType{core.DTypeQ4_K_T, core.DTypeQ4_0, core.DTypeIQ4_XS, core.DTypeIQ4_NL}
	opts, err := EnumerateOptions(td, cands, nil, nil, est, DefaultConfidencePenalty)
	if err != nil {
		t.Fatal(err)
	}
	hasIQ4XS, hasQ40 := false, false
	for _, o := range opts {
		if o.Target == core.DTypeIQ4_XS {
			hasIQ4XS = true
		}
		if o.Target == core.DTypeQ4_0 {
			hasQ40 = true
		}
	}
	if !hasIQ4XS {
		t.Fatal("IQ4_XS dropped on spiky tensor")
	}
	if hasQ40 {
		t.Fatal("Q4_0 survived Pareto on spiky tensor; IQ should dominate at similar bpw")
	}
}

func TestIQPreferredOnLowEntropy(t *testing.T) {
	vals := make([]float32, 64)
	for i := range vals {
		vals[i] = 0.05
	}
	vals[0] = 40
	st := aggregateImatrix(vals, nil, true)
	if st.Entropy > 0.7 {
		t.Fatalf("entropy %v, want concentrated", st.Entropy)
	}
	td := weightTD("blk.2.ffn_down.weight", 256, 256)
	est := NewFallbackEstimator(map[string]ImatrixStats{td.Name: st})
	lK, _ := est.Estimate(td, core.DTypeQ4_K_T)
	lIQ, _ := est.Estimate(td, core.DTypeIQ4_NL)
	if !(lIQ < lK) {
		t.Fatalf("concentrated tensor: IQ4_NL loss %v should be below Q4_K %v", lIQ, lK)
	}
}

func TestKQuantPreferredOnFlatWhenBytesAllow(t *testing.T) {
	flat := aggregateImatrix([]float32{1, 1, 1, 1, 1, 1, 1, 1}, nil, false)
	td := weightTD("blk.1.ffn_up.weight", 256, 256)
	est := NewFallbackEstimator(map[string]ImatrixStats{td.Name: flat})
	lK, _ := est.Estimate(td, core.DTypeQ4_K_T)
	lIQ, _ := est.Estimate(td, core.DTypeIQ4_NL)
	if !(lK < lIQ) {
		t.Fatalf("flat tensor: Q4_K loss %v not below IQ4_NL %v", lK, lIQ)
	}

	bank := &core.TensorBank{SourcePath: "/m.gguf", ModelID: "flat", Tensors: []core.TensorDesc{td}}
	set, err := anchor.Derive(bank, nil, anchor.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	bQ4, _ := core.DTypeQ4_K_T.ExactBytes(td.Elements)
	res, err := Solve(Request{
		Bank: bank, Anchors: set,
		Candidates:  []core.DType{core.DTypeQ4_K_T, core.DTypeIQ2_S, core.DTypeQ2_K},
		Imatrix:     map[string]ImatrixStats{td.Name: flat},
		BudgetBytes: bQ4 + 1024,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := targetOf(t, res, td.Name); got != core.DTypeQ4_K_T {
		t.Errorf("flat tensor at comfortable budget = %v, want Q4_K not IQ2", got)
	}
}

func TestSolvePrefersAttnVOverAttnQTightBudget(t *testing.T) {
	q := weightTD("blk.1.attn_q.weight", 256, 256)
	v := weightTD("blk.1.attn_v.weight", 256, 256)
	bank := &core.TensorBank{SourcePath: "/m.gguf", ModelID: "qv", Tensors: []core.TensorDesc{q, v}}
	set, err := anchor.Derive(bank, nil, anchor.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := set.Floor(q.Name); ok {
		t.Fatal("attn_q hard floor")
	}
	if _, ok := set.Floor(v.Name); ok {
		t.Fatal("attn_v hard floor")
	}
	st := ImatrixStats{Mean: 1, Max: 1, P50: 1, P95: 1, Spikiness: 1, Samples: 4096}
	bQ2, _ := core.DTypeQ2_K.ExactBytes(q.Elements)
	bQ4, _ := core.DTypeQ4_K_T.ExactBytes(v.Elements)
	res, err := Solve(Request{
		Bank: bank, Anchors: set,
		Candidates:  []core.DType{core.DTypeQ6_K, core.DTypeQ4_K_T, core.DTypeQ2_K, core.DTypeIQ2_S},
		Imatrix:     map[string]ImatrixStats{q.Name: st, v.Name: st},
		BudgetBytes: bQ4 + bQ2 + 256,
	})
	if err != nil {
		t.Fatalf("tight budget infeasible (ValuePrior must stay soft): %v", err)
	}
	gotQ, gotV := targetOf(t, res, q.Name), targetOf(t, res, v.Name)
	if gotQ != core.DTypeQ2_K && gotQ != core.DTypeIQ2_S {
		t.Errorf("attn_q = %v, want Q2_K or IQ2_S under tight budget", gotQ)
	}
	if anchor.Rank(gotV) >= anchor.Rank(gotQ) {
		t.Errorf("attn_v = %v not higher fidelity than attn_q = %v", gotV, gotQ)
	}
}

func TestSolveSwiGLUCoupling(t *testing.T) {
	gate := weightTD("blk.0.ffn_gate.weight", 256, 256)
	up := weightTD("blk.0.ffn_up.weight", 256, 256)
	bank := &core.TensorBank{SourcePath: "/m.gguf", ModelID: "swiglu", Tensors: []core.TensorDesc{gate, up}}
	set, err := anchor.Derive(bank, nil, anchor.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	// Non-convex / asymmetric curves so greedy leaves them mismatched.
	cache := measuredCache(t, map[string]map[core.DType]float64{
		gate.Name: {core.DTypeQ8_0: 1, core.DTypeQ6_K: 2, core.DTypeQ4_K_T: 80, core.DTypeQ2_K: 100},
		up.Name:   {core.DTypeQ8_0: 1, core.DTypeQ6_K: 1.1, core.DTypeQ4_K_T: 1.2, core.DTypeQ2_K: 1.3},
	})
	bQ6, _ := core.DTypeQ6_K.ExactBytes(gate.Elements)
	bQ2, _ := core.DTypeQ2_K.ExactBytes(up.Elements)
	bQ4, _ := core.DTypeQ4_K_T.ExactBytes(gate.Elements)
	res, err := Solve(Request{
		Bank: bank, Anchors: set, Candidates: testCands, Cache: cache,
		BudgetBytes: bQ6 + bQ2 + 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	g, u := targetOf(t, res, gate.Name), targetOf(t, res, up.Name)
	if g != u {
		t.Fatalf("SwiGLU gate=%v up=%v, want matched dtype", g, u)
	}
	if g != core.DTypeQ4_K_T && g != core.DTypeQ6_K && g != core.DTypeQ2_K {
		t.Errorf("coupled dtype %v not Q2/Q4/Q6", g)
	}
	// Q4+Q4 fits the Q6+Q2 budget; coupling should not pick an unmatched pair.
	_ = bQ4
}

func TestSolve2OptMovesBitsToFFNDown(t *testing.T) {
	// Equal-size tensors, non-convex ffn_down curve so greedy over-spends on
	// attn_q's cheap first rung and starves ffn_down's valuable second rung.
	q := weightTD("blk.1.attn_q.weight", 256, 256)
	d := weightTD("blk.1.ffn_down.weight", 256, 256)
	bank := &core.TensorBank{SourcePath: "/m.gguf", ModelID: "2opt", Tensors: []core.TensorDesc{q, d}}
	set, err := anchor.Derive(bank, nil, anchor.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	cache := measuredCache(t, map[string]map[core.DType]float64{
		q.Name: {core.DTypeQ8_0: 5, core.DTypeQ6_K: 8, core.DTypeQ4_K_T: 10, core.DTypeQ2_K: 40},
		d.Name: {core.DTypeQ8_0: 4, core.DTypeQ6_K: 10, core.DTypeQ4_K_T: 95, core.DTypeQ2_K: 100},
	})
	bQ6, _ := core.DTypeQ6_K.ExactBytes(q.Elements)
	bQ2, _ := core.DTypeQ2_K.ExactBytes(d.Elements)
	res, err := Solve(Request{
		Bank: bank, Anchors: set, Candidates: testCands, Cache: cache,
		BudgetBytes: bQ6 + bQ2,
	})
	if err != nil {
		t.Fatal(err)
	}
	gotQ, gotD := targetOf(t, res, q.Name), targetOf(t, res, d.Name)
	if anchor.Rank(gotD) > anchor.Rank(gotQ) {
		t.Errorf("2-opt did not move bits toward ffn_down: attn_q=%v ffn_down=%v", gotQ, gotD)
	}
}

func TestIngestKLDHistoryWritesMeasuredCache(t *testing.T) {
	c := NewCache("m", "sha")
	at := time.Date(2026, 8, 18, 0, 0, 0, 0, time.UTC)
	groups := []core.MoveGroup{{
		ID: "g1",
		Moves: []core.Move{
			{TensorName: "blk.0.attn_v.weight", From: core.DTypeQ2_K, To: core.DTypeQ4_K_T},
			{TensorName: "blk.0.ffn_down.weight", From: core.DTypeQ2_K, To: core.DTypeQ4_K_T},
		},
	}}
	hist := []kld.Step{
		{Kind: "baseline", KLD: 1.0},
		{Kind: "scan"},
		{Kind: "eval-error", Error: "boom"},
		{Kind: "solo", Accepted: true, KLD: 0.4, GroupIDs: []string{"g1"}},
	}
	n, err := IngestKLDHistory(c, hist, groups, "run-1", at)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("ingested %d, want 2", n)
	}
	cl, ok := c.Get("blk.0.attn_v.weight", core.DTypeQ4_K_T)
	if !ok {
		t.Fatal("missing attn_v entry")
	}
	if cl.Evidence != EvidenceMeasured || cl.Prov == nil || cl.Prov.Tool != "kld-search" {
		t.Fatalf("provenance: %+v", cl)
	}
	if cl.Confidence < 0.6 || cl.Confidence > 0.8 {
		t.Fatalf("confidence %v, want 0.6–0.8", cl.Confidence)
	}
	if cl.Loss != 0.3 { // (1.0-0.4)/2
		t.Fatalf("attributed loss %v, want 0.3", cl.Loss)
	}
	if _, err := IngestKLDHistory(c, nil, nil, "run-1", at); err != nil {
		t.Fatal(err)
	}
}

func TestCacheStoresSearchConfidence(t *testing.T) {
	c := NewCache("m", "s")
	prov := &core.Provenance{Tool: "kld-search", RunID: "r", MeasuredAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
	if err := c.Put(CandidateLoss{
		TensorName: "t", Target: core.DTypeQ4_K_T, Loss: 0.02,
		Evidence: EvidenceMeasured, Confidence: 0.75, Prov: prov,
	}); err != nil {
		t.Fatal(err)
	}
	cl, ok := c.Get("t", core.DTypeQ4_K_T)
	if !ok || cl.Confidence != 0.75 {
		t.Fatalf("got %+v ok=%v", cl, ok)
	}
}
