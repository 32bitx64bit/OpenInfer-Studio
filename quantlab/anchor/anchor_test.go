package anchor

import (
	"testing"

	"quantlab/core"
)

func testBank() *core.TensorBank {
	return &core.TensorBank{
		SourcePath: "/m.gguf",
		Tensors: []core.TensorDesc{
			{Name: "token_embd.weight", DType: core.DTypeF16, Shape: []uint64{256, 4}, Length: 2048, Elements: 1024},
			{Name: "blk.0.ffn_norm.weight", DType: core.DTypeF32, Shape: []uint64{4}, Length: 16, Elements: 4},
			{Name: "blk.0.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{256, 4}, Length: 2048, Elements: 1024},
			{Name: "blk.0.attn_v.weight", DType: core.DTypeF16, Shape: []uint64{256, 4}, Length: 2048, Elements: 1024},
			{Name: "blk.0.ffn_down.weight", DType: core.DTypeF16, Shape: []uint64{256, 4}, Length: 2048, Elements: 1024},
			{Name: "blk.0.ffn_up.weight", DType: core.DTypeF16, Shape: []uint64{256, 4}, Length: 2048, Elements: 1024},
		},
	}
}

func TestNormsStructurallyPreservedNoFloor(t *testing.T) {
	s, err := Derive(testBank(), nil, Policy{})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Floor("blk.0.ffn_norm.weight"); ok {
		t.Error("norm received a hard floor; must be structurally preserved instead")
	}
	norm, _ := testBank().Find("blk.0.ffn_norm.weight")
	if !s.Preserved(norm) {
		t.Error("norm not preserved")
	}
	// A quantizable tensor matching a norm pattern is preserved too.
	weird := core.TensorDesc{Name: "blk.0.attn_norm.weight", DType: core.DTypeF16, Shape: []uint64{256, 4}, Length: 2048, Elements: 1024}
	if !s.Preserved(weird) {
		t.Error("2D norm-pattern tensor not preserved")
	}
	if _, ok := s.Floor("token_embd.weight"); ok {
		t.Error("embedding received a hard floor; must be a soft prior")
	}
	if _, ok := s.Floor("blk.0.attn_q.weight"); ok {
		t.Error("attention received a hard floor; must be a soft prior")
	}
	if _, ok := s.Floor("blk.0.attn_v.weight"); ok {
		t.Error("attn_v received a hard floor; ValuePrior must stay soft")
	}
	if _, ok := s.Floor("blk.0.ffn_down.weight"); ok {
		t.Error("ffn_down received a hard floor; ValuePrior must stay soft")
	}
}

func TestSoftPriorsPenalizeBelowPreference(t *testing.T) {
	s, err := Derive(testBank(), nil, Policy{})
	if err != nil {
		t.Fatal(err)
	}
	embd, _ := testBank().Find("token_embd.weight")
	attn, _ := testBank().Find("blk.0.attn_q.weight")
	attnV, _ := testBank().Find("blk.0.attn_v.weight")
	ffnDown, _ := testBank().Find("blk.0.ffn_down.weight")
	ffnUp, _ := testBank().Find("blk.0.ffn_up.weight")

	if p := s.PriorLoss(embd, core.DTypeQ4_K_T); p <= 0 {
		t.Errorf("embedding at Q4_K below Q6_K prior: penalty = %v, want > 0", p)
	}
	if p := s.PriorLoss(embd, core.DTypeQ8_0); p != 0 {
		t.Errorf("embedding above prior penalized: %v", p)
	}
	if p := s.PriorLoss(embd, core.DTypeF16); p != 0 {
		t.Errorf("float target penalized: %v", p)
	}
	// Attention prior is weaker than embedding prior for the same distance.
	pe := s.PriorLoss(embd, core.DTypeQ4_K_T)
	pa := s.PriorLoss(attn, core.DTypeQ4_K_T)
	if pa <= 0 || pa >= pe {
		t.Errorf("attention penalty %v should be positive but below embedding %v", pa, pe)
	}
	pv := s.PriorLoss(attnV, core.DTypeQ4_K_T)
	if pv <= pa {
		t.Errorf("attn_v value prior %v should exceed attn_q attention prior %v", pv, pa)
	}
	if p := s.PriorLoss(ffnDown, core.DTypeQ4_K_T); p <= 0 {
		t.Errorf("ffn_down value prior = %v, want > 0", p)
	}
	if p := s.PriorLoss(ffnUp, core.DTypeQ4_K_T); p != 0 {
		t.Errorf("ffn_up harvest tensor penalized: %v", p)
	}
	// Norms are preserved: no prior even if a pattern overlapped.
	norm, _ := testBank().Find("blk.0.ffn_norm.weight")
	if p := s.PriorLoss(norm, core.DTypeQ4_K_T); p != 0 {
		t.Errorf("preserved norm penalized: %v", p)
	}
}

func TestOnlyExplicitAndCalibrationAreHard(t *testing.T) {
	s, err := Derive(testBank(), []core.Anchor{
		{Kind: core.AnchorExplicit, Name: "blk.0.ffn_down.weight", MinDType: core.DTypeQ6_K},
		{Kind: core.AnchorCalibration, Name: "blk.0.attn_q.weight", MinDType: core.DTypeQ8_0},
		{Kind: core.AnchorEmbedding, Name: "token_embd.weight", MinDType: core.DTypeQ8_0},
	}, Policy{})
	if err != nil {
		t.Fatal(err)
	}
	if f, ok := s.Floor("blk.0.ffn_down.weight"); !ok || f != core.DTypeQ6_K {
		t.Errorf("explicit floor = %v,%v", f, ok)
	}
	if f, ok := s.Floor("blk.0.attn_q.weight"); !ok || f != core.DTypeQ8_0 {
		t.Errorf("calibration floor = %v,%v", f, ok)
	}
	// User-supplied embedding-kind anchor is demoted to a prior, not a floor.
	if _, ok := s.Floor("token_embd.weight"); ok {
		t.Error("embedding-kind anchor became a hard floor")
	}
}

func TestCheckViolations(t *testing.T) {
	s, err := Derive(testBank(), []core.Anchor{
		{Kind: core.AnchorExplicit, Name: "blk.0.ffn_down.weight", MinDType: core.DTypeQ6_K},
	}, Policy{})
	if err != nil {
		t.Fatal(err)
	}
	p := &core.Profile{
		ID: "p",
		Assignments: []core.QuantAssignment{
			{TensorName: "token_embd.weight", Target: core.DTypeQ4_K_T, BitsPerWeight: 4.5},     // soft prior only: no violation
			{TensorName: "blk.0.ffn_down.weight", Target: core.DTypeQ4_K_T, BitsPerWeight: 4.5}, // below hard floor
		},
	}
	v := s.Check(p)
	if len(v) != 1 || v[0].Tensor != "blk.0.ffn_down.weight" {
		t.Fatalf("expected exactly one violation on ffn_down, got %+v", v)
	}
	p.Assignments[1].Target = core.DTypeF16
	if v := s.Check(p); len(v) != 0 {
		t.Fatalf("float target flagged: %+v", v)
	}
}

func TestDeriveRejectsBadExplicit(t *testing.T) {
	if _, err := Derive(testBank(), []core.Anchor{{Kind: "bogus"}}, Policy{}); err == nil {
		t.Fatal("invalid explicit anchor accepted")
	}
	if _, err := Derive(nil, nil, Policy{}); err == nil {
		t.Fatal("nil bank accepted")
	}
}

// TestRankFidelityOrder pins the fidelity semantics of Rank: the order is by
// exact bits-per-weight descending (floats above all quants), with equal-bpw
// dtypes ordered deterministically by name, and recipe labels sharing their
// base type's bpw.
func TestRankFidelityOrder(t *testing.T) {
	if Rank(core.DTypeF16) != -1 || Rank(core.DTypeF32) != -1 {
		t.Fatal("float dtypes must rank above all quants")
	}
	// Strictly decreasing fidelity by exact bpw.
	bpw := func(d core.DType) float64 { v, _ := d.BitsPerWeight(); return v }
	chain := []core.DType{core.DTypeQ8_0, core.DTypeQ6_K, core.DTypeQ5_1, core.DTypeQ5_0, core.DTypeQ4_0, core.DTypeQ2_K, core.DTypeIQ1_S}
	for i := 0; i+1 < len(chain); i++ {
		hi, lo := chain[i], chain[i+1]
		if bpw(hi) <= bpw(lo) {
			t.Fatalf("test chain not bpw-ordered: %s %.3f vs %s %.3f", hi, bpw(hi), lo, bpw(lo))
		}
		if Rank(hi) >= Rank(lo) {
			t.Errorf("Rank(%s)=%d not higher fidelity than Rank(%s)=%d", hi, Rank(hi), lo, Rank(lo))
		}
	}
	// Equal-bpw ties (Q4_K and IQ4_NL are both 4.5 bpw) resolve by name, so
	// the order is total and deterministic.
	if bpw(core.DTypeQ4_K_T) != bpw(core.DTypeIQ4_NL) {
		t.Fatal("test premise broken: Q4_K and IQ4_NL bpw differ")
	}
	if Rank(core.DTypeIQ4_NL) >= Rank(core.DTypeQ4_K_T) {
		t.Error("equal-bpw tie not broken by name ascending")
	}
	// A recipe label ranks at its base type's bpw, adjacent to it.
	if Rank(core.DTypeQ4_K_M) < Rank(core.DTypeQ4_K_T)-1 || Rank(core.DTypeQ4_K_M) > Rank(core.DTypeQ4_K_T)+1 {
		t.Errorf("recipe label Q4_K_M rank %d not adjacent to base Q4_K rank %d",
			Rank(core.DTypeQ4_K_M), Rank(core.DTypeQ4_K_T))
	}
	// Invalid (non-quant) dtypes rank as non-quant, below no floor: they are
	// never legal assignment targets, so they never reach floor comparisons.
	if Rank("Q9_ZZ") != -1 {
		t.Error("invalid dtype not ranked as non-quant")
	}
}

func hybridBank() *core.TensorBank {
	td := func(name string, dt core.DType, shape []uint64) core.TensorDesc {
		var elems uint64 = 1
		for _, d := range shape {
			elems *= d
		}
		b, ok := dt.ExactBytes(elems)
		if !ok {
			panic(name)
		}
		return core.TensorDesc{Name: name, DType: dt, Shape: shape, Length: b, Elements: elems}
	}
	return &core.TensorBank{
		SourcePath: "/hybrid.gguf",
		Tensors: []core.TensorDesc{
			td("token_embd.weight", core.DTypeF16, []uint64{256, 4}),
			td("blk.0.ssm_out.weight", core.DTypeF16, []uint64{256, 4}),
			td("blk.0.ssm_a", core.DTypeF32, []uint64{64}),
			td("blk.0.ssm_dt.bias", core.DTypeF32, []uint64{64}),
			td("blk.0.ssm_conv1d.weight", core.DTypeF32, []uint64{256, 4}),
			td("blk.0.attn_qkv.weight", core.DTypeF16, []uint64{256, 4}),
			td("blk.0.ffn_gate_inp.weight", core.DTypeF16, []uint64{256, 8}),
			td("blk.0.mystery.weight", core.DTypeF16, []uint64{256, 4}),
			td("blk.1.attn_q.weight", core.DTypeF16, []uint64{256, 4}),
		},
	}
}

func TestDeriveLayoutNativeSSMVsAttn(t *testing.T) {
	bank := hybridBank()
	s, err := Derive(bank, nil, Policy{})
	if err != nil {
		t.Fatal(err)
	}
	must := func(name string) core.TensorDesc {
		t.Helper()
		td, ok := bank.Find(name)
		if !ok {
			t.Fatalf("missing %s", name)
		}
		return td
	}
	ssmOut := must("blk.0.ssm_out.weight")
	attnQ := must("blk.1.attn_q.weight")
	qkv := must("blk.0.attn_qkv.weight")
	conv := must("blk.0.ssm_conv1d.weight")
	aLog := must("blk.0.ssm_a")
	dt := must("blk.0.ssm_dt.bias")
	mystery := must("blk.0.mystery.weight")
	router := must("blk.0.ffn_gate_inp.weight")

	if s.Preserved(ssmOut) {
		t.Error("ssm_out must stay quantizable; convert stores it as a 2D weight")
	}
	if p := s.PriorLoss(ssmOut, core.DTypeQ4_K_T); p <= 0 {
		t.Errorf("ssm_out attention prior = %v, want > 0", p)
	}
	if p := s.PriorLoss(qkv, core.DTypeQ4_K_T); p <= 0 {
		t.Errorf("linear-attn attn_qkv prior = %v, want > 0", p)
	}
	if p := s.PriorLoss(attnQ, core.DTypeQ4_K_T); p <= 0 {
		t.Errorf("attn_q attention prior = %v, want > 0", p)
	}
	if !s.Preserved(conv) {
		t.Error("ssm_conv1d (2D F32, block-aligned) must be preserved; convert stores conv1d as F32")
	}
	if !s.Preserved(aLog) {
		t.Error("ssm_a (A_log) must be preserved F32")
	}
	if !s.Preserved(dt) {
		t.Error("ssm_dt must be preserved F32")
	}
	if !s.Preserved(mystery) {
		t.Error("unknown tensor must fail closed (not silently quantized)")
	}
	if f, ok := s.Floor("blk.0.ffn_gate_inp.weight"); !ok || f != core.DTypeQ5_K_T {
		t.Errorf("moe router floor = %v,%v, want Q5_K", f, ok)
	}
	if s.Preserved(router) {
		t.Error("quantizable moe router should be floored, not preserved")
	}
	if _, ok := s.Floor("blk.1.attn_q.weight"); ok {
		t.Error("attn_q received a hard floor; must be a soft prior")
	}
	if _, ok := s.Floor("blk.0.ssm_out.weight"); ok {
		t.Error("ssm_out received a hard floor")
	}
}

func TestDeriveRejectsInvalidRouterFloor(t *testing.T) {
	if _, err := Derive(testBank(), nil, Policy{RouterFloor: "NOPE"}); err == nil {
		t.Fatal("invalid router floor accepted")
	}
}

func TestDefaultPolicyValuePriorSoft(t *testing.T) {
	p := DefaultPolicy()
	if p.ValuePrior != core.DTypeQ6_K || p.AttentionPrior != core.DTypeQ5_K_T {
		t.Fatalf("priors: value=%s attn=%s", p.ValuePrior, p.AttentionPrior)
	}
	if p.RouterFloor != core.DTypeQ5_K_T {
		t.Fatalf("router floor %s", p.RouterFloor)
	}
	if p.ValueWeight != p.EmbeddingWeight {
		t.Fatalf("value weight %v != embedding %v", p.ValueWeight, p.EmbeddingWeight)
	}
	if p.DownPrior != core.DTypeQ6_K || p.DownWeight != p.ValueWeight {
		t.Fatalf("historical down prior %s weight %v, want Q6_K at value weight", p.DownPrior, p.DownWeight)
	}
}

func TestPolicyForBPW(t *testing.T) {
	hist := PolicyForBPW(0)
	def := DefaultPolicy()
	if hist.ValuePrior != def.ValuePrior || hist.DownPrior != def.DownPrior ||
		hist.DownWeight != def.DownWeight || hist.ExpertDownWeight != def.ExpertDownWeight {
		t.Fatalf("bpw<=0 changed historical policy: %+v vs %+v", hist, def)
	}
	if hist.EmbeddingPrior != core.DTypeQ6_K || hist.AttentionPrior != core.DTypeQ5_K_T {
		t.Fatalf("embed/attn priors drifted: embed=%s attn=%s", hist.EmbeddingPrior, hist.AttentionPrior)
	}

	low := PolicyForBPW(3.5)
	if low.ValuePrior != core.DTypeQ6_K {
		t.Fatalf("V prior at 3.5 bpw = %s, want Q6_K", low.ValuePrior)
	}
	if low.EmbeddingPrior != core.DTypeQ6_K || low.AttentionPrior != core.DTypeQ5_K_T {
		t.Fatalf("embed/attn priors at 3.5 bpw: embed=%s attn=%s", low.EmbeddingPrior, low.AttentionPrior)
	}
	if low.DownPrior == core.DTypeQ6_K {
		t.Fatal("3.5 bpw still puts Q6_K on every ffn_down")
	}
	if low.DownPrior != core.DTypeQ4_K_T && low.DownPrior != core.DTypeQ5_K_T {
		t.Fatalf("3.5 bpw down prior %s, want Q4_K or Q5_K", low.DownPrior)
	}
	if low.DownWeight >= low.ValueWeight {
		t.Fatalf("down weight %v must be weaker than V %v", low.DownWeight, low.ValueWeight)
	}
	if low.ExpertDownWeight >= low.DownWeight {
		t.Fatalf("expert down weight %v must be weaker than dense down %v", low.ExpertDownWeight, low.DownWeight)
	}

	high := PolicyForBPW(4.5)
	if high.DownPrior != core.DTypeQ5_K_T && high.DownPrior != core.DTypeQ6_K {
		t.Fatalf("4.5 bpw down prior %s, want Q5_K or Q6_K", high.DownPrior)
	}
	if high.DownWeight >= high.ValueWeight {
		t.Fatalf("4.5 bpw down weight %v not weaker than V %v", high.DownWeight, high.ValueWeight)
	}
}

func TestDeriveSplitsValueAndDownPriors(t *testing.T) {
	bank := testBank()
	s, err := Derive(bank, nil, PolicyForBPW(3.5))
	if err != nil {
		t.Fatal(err)
	}
	attnV, _ := bank.Find("blk.0.attn_v.weight")
	ffnDown, _ := bank.Find("blk.0.ffn_down.weight")
	attnQ, _ := bank.Find("blk.0.attn_q.weight")
	ffnUp, _ := bank.Find("blk.0.ffn_up.weight")

	if _, ok := s.Floor(attnV.Name); ok {
		t.Error("attn_v hard-pinned; ValuePrior must stay soft")
	}
	if _, ok := s.Floor(ffnDown.Name); ok {
		t.Error("ffn_down hard-pinned; DownPrior must stay soft")
	}
	if p := s.PriorLoss(attnV, core.DTypeQ4_K_T); p <= 0 {
		t.Errorf("attn_v below Q6_K V prior: penalty = %v", p)
	}
	// Q4_K matches the 3.5-bpw down prior, so ffn_down must not be pulled toward Q6.
	if p := s.PriorLoss(ffnDown, core.DTypeQ4_K_T); p != 0 {
		t.Errorf("ffn_down at Q4_K penalized under 3.5 policy: %v", p)
	}
	pv := s.PriorLoss(attnV, core.DTypeQ3_K)
	pd := s.PriorLoss(ffnDown, core.DTypeQ3_K)
	if pv <= pd {
		t.Errorf("attn_v Q3 penalty %v should exceed ffn_down %v", pv, pd)
	}
	if p := s.PriorLoss(attnQ, core.DTypeQ4_K_T); p <= 0 {
		t.Errorf("attn_q attention prior = %v, want > 0", p)
	}
	if p := s.PriorLoss(ffnUp, core.DTypeQ4_K_T); p != 0 {
		t.Errorf("ffn_up harvest tensor penalized: %v", p)
	}
}

func TestFusedQKVNotValuePrior(t *testing.T) {
	if isAttnV("blk.0.attn_qkv.weight") {
		t.Fatal("fused qkv must not be classified as V")
	}
	if isAttnV("blk.0.attn_q.weight") {
		t.Fatal("attn_q is not V")
	}
	if !isAttnV("blk.0.attn_v.weight") || !isAttnV("layers.2.self_attn.v_proj.weight") {
		t.Fatal("attn_v / v_proj aliases not recognized")
	}
	if !isFFNDown("blk.0.ffn_down.weight") || !isFFNDown("layers.2.mlp.down_proj.weight") {
		t.Fatal("ffn_down / down_proj aliases not recognized")
	}
	if isFFNDown("blk.0.attn_v.weight") {
		t.Fatal("attn_v classified as ffn_down")
	}

	bank := hybridBank()
	s, err := Derive(bank, nil, Policy{})
	if err != nil {
		t.Fatal(err)
	}
	qkv, _ := bank.Find("blk.0.attn_qkv.weight")
	attnQ, _ := bank.Find("blk.1.attn_q.weight")
	pq := s.PriorLoss(qkv, core.DTypeQ4_K_T)
	pa := s.PriorLoss(attnQ, core.DTypeQ4_K_T)
	if pq != pa {
		t.Errorf("fused qkv prior %v != attn_q %v (qkv must not get V prior)", pq, pa)
	}
}

func TestExpertDownPriorWeaker(t *testing.T) {
	td := func(name string) core.TensorDesc {
		return core.TensorDesc{Name: name, DType: core.DTypeF16, Shape: []uint64{256, 4}, Length: 2048, Elements: 1024}
	}
	bank := &core.TensorBank{
		SourcePath: "/moe.gguf",
		Tensors: []core.TensorDesc{
			td("blk.1.ffn_down.weight"),
			td("blk.1.ffn_down_exps.weight"),
		},
	}
	s, err := Derive(bank, nil, PolicyForBPW(3.5))
	if err != nil {
		t.Fatal(err)
	}
	dense, _ := bank.Find("blk.1.ffn_down.weight")
	expert, _ := bank.Find("blk.1.ffn_down_exps.weight")
	pd := s.PriorLoss(dense, core.DTypeQ3_K)
	pe := s.PriorLoss(expert, core.DTypeQ3_K)
	if pe <= 0 {
		t.Errorf("expert down prior = %v, want > 0 at Q3", pe)
	}
	if pe >= pd {
		t.Errorf("expert down penalty %v should be below dense down %v", pe, pd)
	}
}
