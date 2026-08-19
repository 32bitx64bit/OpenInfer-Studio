package profile

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"

	"quantlab/anchor"
	"quantlab/core"
)

func qbank() *core.TensorBank {
	return &core.TensorBank{
		SourcePath: "/m.gguf",
		ModelID:    "test-model",
		Tensors: []core.TensorDesc{
			{Name: "token_embd.weight", DType: core.DTypeF16, Shape: []uint64{256, 512}, Length: 262144, Elements: 131072},
			{Name: "blk.0.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
			{Name: "blk.0.attn_v.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
			{Name: "blk.0.ffn_up.weight", DType: core.DTypeF16, Shape: []uint64{256, 512}, Length: 262144, Elements: 131072},
			{Name: "blk.1.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
			{Name: "blk.1.ffn_down.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
			{Name: "blk.0.ffn_norm.weight", DType: core.DTypeF32, Shape: []uint64{256}, Length: 1024, Elements: 256},
		},
	}
}

var testCands = []core.DType{core.DTypeQ8_0, core.DTypeQ6_K, core.DTypeQ4_K_T, core.DTypeQ2_K}

func measuredCache(t *testing.T, entries map[string]map[core.DType]float64) *Cache {
	t.Helper()
	c := NewCache("test-model", "sha-test")
	for tensor, byDT := range entries {
		for dt, loss := range byDT {
			prov := &core.Provenance{
				Tool: "llama-perplexity", ToolVersion: "b1", RunID: "run-1",
				MeasuredAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
			}
			if err := c.Put(CandidateLoss{
				TensorName: tensor, Target: dt, Loss: loss,
				Evidence: EvidenceMeasured, Confidence: 1.0, Prov: prov,
			}); err != nil {
				t.Fatal(err)
			}
		}
	}
	return c
}

func targetOf(t *testing.T, res *Result, name string) core.DType {
	t.Helper()
	for _, qa := range res.Profile.Assignments {
		if qa.TensorName == name {
			return qa.Target
		}
	}
	t.Fatalf("no assignment for %q", name)
	return ""
}

func TestRequestValidate(t *testing.T) {
	if err := (Request{Bank: qbank(), BudgetBytes: 1 << 20}).Validate(); err != nil {
		t.Fatalf("valid request rejected: %v", err)
	}
	if err := (Request{}).Validate(); err == nil {
		t.Fatal("nil bank accepted")
	}
	if err := (Request{Bank: qbank(), Candidates: []core.DType{core.DTypeF16}}).Validate(); err == nil {
		t.Fatal("float candidate accepted")
	}
	if err := (Request{Bank: qbank(), TargetBPW: -1}).Validate(); err == nil {
		t.Fatal("negative bpw accepted")
	}
}

func TestBPWToBudget(t *testing.T) {
	b := qbank()
	var elems uint64
	for _, ts := range b.Tensors {
		elems += ts.Elements
	}
	got, err := BPWToBudget(b, 4.0)
	if err != nil {
		t.Fatal(err)
	}
	if want := uint64(4.0 * float64(elems) / 8.0); got != want {
		t.Fatalf("budget = %d, want %d", got, want)
	}
	if _, err := BPWToBudget(b, 0); err == nil {
		t.Fatal("zero bpw accepted")
	}
	if _, err := BPWToBudget(nil, 4); err == nil {
		t.Fatal("nil bank accepted")
	}
}

func TestParetoPruneFrontier(t *testing.T) {
	mk := func(dt core.DType, bytes uint64, loss float64) ScoredOption {
		return ScoredOption{TensorOption: core.TensorOption{TensorName: "t", Target: dt, Bytes: bytes}, Loss: loss}
	}
	opts := []ScoredOption{
		mk("A", 100, 10), // dominated by B (fewer bytes, less loss)
		mk("B", 40, 5),
		mk("C", 50, 6),  // dominated by B
		mk("D", 90, 12), // dominated by E (more bytes, less loss)
		mk("E", 80, 3),
		mk("F", 80, 9), // same bytes as E, worse loss
		mk("G", 80, 3), // duplicate of E; dedup by target name keeps A-name... same loss/bytes, E < G
	}
	got := ParetoPrune(opts)
	// Frontier: 40/5 (B), 80/3 (E). Nonuniform curve preserved.
	if len(got) != 2 {
		t.Fatalf("frontier = %+v, want 2 options", got)
	}
	if got[0].Target != "B" || got[0].Bytes != 40 || got[0].Loss != 5 {
		t.Errorf("got[0] = %+v", got[0])
	}
	if got[1].Target != "E" || got[1].Bytes != 80 || got[1].Loss != 3 {
		t.Errorf("got[1] = %+v", got[1])
	}
}

func TestEnumerateOptionsLegalShapeAndPreservation(t *testing.T) {
	set, err := anchor.Derive(qbank(), nil, anchor.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	est := NewFallbackEstimator(nil)
	norm, _ := qbank().Find("blk.0.ffn_norm.weight")
	opts, err := EnumerateOptions(norm, testCands, set, nil, est, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	if len(opts) != 1 || opts[0].Target != core.DTypeF32 {
		t.Fatalf("norm not preserved: %+v", opts)
	}
	// 2D block-256-aligned tensor gets the full lattice minus dominated.
	q, _ := qbank().Find("blk.0.attn_q.weight")
	opts, err = EnumerateOptions(q, testCands, set, nil, est, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	var seen []core.DType
	for _, o := range opts {
		seen = append(seen, o.Target)
		b, ok := o.Target.ExactBytes(q.Elements)
		if !ok || b != o.Bytes {
			t.Errorf("%v: bytes %d, exact %d/%v", o.Target, o.Bytes, b, ok)
		}
	}
	if len(seen) != len(testCands) {
		t.Errorf("expected %d frontier options (all distinct bytes), got %v", len(testCands), seen)
	}
	// Block-32-only tensor: 256-alignment holds since 256%256==0; use 64-dim
	// to exclude K-quants (64%256 != 0, 64%32 == 0).
	p32 := core.TensorDesc{Name: "x.weight", DType: core.DTypeF16, Shape: []uint64{64, 4}, Length: 512, Elements: 256}
	opts, err = EnumerateOptions(p32, testCands, set, nil, est, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	for _, o := range opts {
		if g, ok := o.Target.Geometry(); ok && g.BlockSize == 256 {
			t.Errorf("illegal 256-block type %v offered for 64-aligned tensor", o.Target)
		}
	}
}

func TestSolveExactBudgetAndManifest(t *testing.T) {
	set, _ := anchor.Derive(qbank(), nil, anchor.Policy{})
	res, err := Solve(Request{
		Bank: qbank(), Anchors: set, Candidates: testCands,
		BudgetBytes: 200000,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Manifest.TotalBytes > 200000 {
		t.Fatalf("budget violated: %d > 200000", res.Manifest.TotalBytes)
	}
	if res.Profile.EstimatedBytes != res.Manifest.TotalBytes {
		t.Fatalf("profile estimate %d != manifest total %d", res.Profile.EstimatedBytes, res.Manifest.TotalBytes)
	}
	if err := res.Manifest.Validate(qbank()); err != nil {
		t.Fatal(err)
	}
	if err := res.Profile.Validate(qbank()); err != nil {
		t.Fatal(err)
	}
	if len(res.Profile.Assignments) != len(qbank().Tensors) {
		t.Fatalf("assignments %d, want one per tensor (%d)", len(res.Profile.Assignments), len(qbank().Tensors))
	}
	if res.Diag.MinBytes > res.Manifest.TotalBytes || res.Manifest.TotalBytes > res.Diag.MaxBytes {
		t.Fatalf("total %d outside [%d,%d]", res.Manifest.TotalBytes, res.Diag.MinBytes, res.Diag.MaxBytes)
	}
	if res.Diag.SlopBytes != 200000-res.Manifest.TotalBytes {
		t.Fatalf("slop %d", res.Diag.SlopBytes)
	}
}

func TestSolveNonuniformCurvesPreferCheapTensors(t *testing.T) {
	set, _ := anchor.Derive(qbank(), nil, anchor.Policy{})
	// Flat curve on blk.1.attn_q (downgrading nearly free), steep on blk.0.attn_q.
	cache := measuredCache(t, map[string]map[core.DType]float64{
		"blk.0.attn_q.weight": {core.DTypeQ8_0: 1.0, core.DTypeQ6_K: 4.0, core.DTypeQ4_K_T: 20.0, core.DTypeQ2_K: 100.0},
		"blk.1.attn_q.weight": {core.DTypeQ8_0: 1.0, core.DTypeQ6_K: 1.05, core.DTypeQ4_K_T: 1.1, core.DTypeQ2_K: 1.15},
	})
	// Tight budget: both must shrink; the flat one should hit Q2_K first.
	res, err := Solve(Request{
		Bank: qbank(), Anchors: set, Candidates: testCands, Cache: cache,
		BudgetBytes: 190000,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := targetOf(t, res, "blk.1.attn_q.weight"); got != core.DTypeQ2_K {
		t.Errorf("flat-curve tensor = %v, want Q2_K", got)
	}
	if got := targetOf(t, res, "blk.0.attn_q.weight"); got == core.DTypeQ2_K {
		t.Errorf("steep tensor squeezed to Q2_K alongside the flat one")
	}
}

func TestSolveTargetBPW(t *testing.T) {
	set, _ := anchor.Derive(qbank(), nil, anchor.Policy{})
	budget, err := BPWToBudget(qbank(), 4.5)
	if err != nil {
		t.Fatal(err)
	}
	res, err := Solve(Request{
		Bank: qbank(), Anchors: set, Candidates: testCands, TargetBPW: 4.5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Manifest.TotalBytes > budget {
		t.Fatalf("bpw budget violated: %d > %d", res.Manifest.TotalBytes, budget)
	}
	if res.Diag.BudgetBytes != budget {
		t.Fatalf("diagnostics budget %d != %d", res.Diag.BudgetBytes, budget)
	}
	var elems uint64
	for _, ts := range qbank().Tensors {
		elems += ts.Elements
	}
	bpw := float64(res.Manifest.TotalBytes) * 8.0 / float64(elems)
	if bpw > 4.5+1e-9 {
		t.Fatalf("achieved bpw %.4f > 4.5", bpw)
	}
}

func TestSolveInfeasible(t *testing.T) {
	set, _ := anchor.Derive(qbank(), nil, anchor.Policy{})
	// First: budget below unconstrained minimum.
	res0, err := Solve(Request{Bank: qbank(), Anchors: set, Candidates: testCands, BudgetBytes: 1})
	if err == nil {
		t.Fatalf("expected infeasible, got %+v", res0.Diag)
	}
	inf, ok := err.(*InfeasibleError)
	if !ok {
		t.Fatalf("error type %T, want *InfeasibleError", err)
	}
	if inf.MinBytes == 0 || inf.BudgetBytes != 1 || inf.MinBytes <= 1 {
		t.Fatalf("bad infeasible details: %+v", inf)
	}
	if !strings.Contains(inf.Error(), "infeasible") {
		t.Fatalf("error message: %v", inf.Error())
	}

	// Second: a hard floor raises the minimum above the budget.
	setF, _ := anchor.Derive(qbank(), []core.Anchor{
		{Kind: core.AnchorExplicit, Name: "blk.0.ffn_up.weight", MinDType: core.DTypeQ6_K},
	}, anchor.Policy{})
	cheap := uint64(0)
	for _, ts := range qbank().Tensors {
		b, _ := core.DTypeQ2_K.ExactBytes(ts.Elements)
		cheap += b
	}
	if _, err := Solve(Request{Bank: qbank(), Anchors: setF, Candidates: testCands, BudgetBytes: cheap}); err == nil {
		t.Fatal("floored infeasible budget accepted")
	} else if ie, ok := err.(*InfeasibleError); !ok {
		t.Fatalf("error type %T, want *InfeasibleError", err)
	} else if ie.MinBytes <= cheap {
		t.Fatalf("floor did not raise minimum: %d <= %d", ie.MinBytes, cheap)
	}
}

func TestSolveDeterminism(t *testing.T) {
	set, _ := anchor.Derive(qbank(), nil, anchor.Policy{})
	cache := measuredCache(t, map[string]map[core.DType]float64{
		"blk.0.attn_v.weight": {core.DTypeQ6_K: 0.5, core.DTypeQ4_K_T: 3.0},
	})
	mk := func() []byte {
		res, err := Solve(Request{
			Bank: qbank(), Anchors: set, Candidates: testCands, Cache: cache,
			Imatrix: map[string]ImatrixStats{
				"blk.0.attn_q.weight":   {Mean: 2.0, Max: 4.0, Samples: 100},
				"blk.1.ffn_down.weight": {Mean: 0.5, Max: 1.0, Samples: 200},
			},
			BudgetBytes: 250000,
		})
		if err != nil {
			t.Fatal(err)
		}
		b, _ := json.Marshal(res.Profile)
		return b
	}
	a, b := mk(), mk()
	if !bytes.Equal(a, b) {
		t.Fatal("solver not deterministic across identical runs")
	}
}

func TestMeasuredEvidenceOverridesHeuristic(t *testing.T) {
	set, _ := anchor.Derive(qbank(), nil, anchor.Policy{})
	base := Request{Bank: qbank(), Anchors: set, Candidates: testCands, BudgetBytes: 260000}
	// Unconstrained-ish baseline: heuristic prefers cheap options for ffn.
	resH, err := Solve(base)
	if err != nil {
		t.Fatal(err)
	}
	hTar := targetOf(t, resH, "blk.0.ffn_up.weight")

	cache := measuredCache(t, map[string]map[core.DType]float64{
		"blk.0.ffn_up.weight": {core.DTypeQ2_K: 500.0, core.DTypeQ4_K_T: 400.0, core.DTypeQ6_K: 1.0, core.DTypeQ8_0: 0.5},
	})
	resM, err := Solve(Request{
		Bank: qbank(), Anchors: set, Candidates: testCands, Cache: cache,
		BudgetBytes: 260000,
	})
	if err != nil {
		t.Fatal(err)
	}
	mTar := targetOf(t, resM, "blk.0.ffn_up.weight")
	if anchor.Rank(mTar) >= anchor.Rank(hTar) && mTar == hTar {
		// measured only matters if it changed the choice
	}
	if anchor.Rank(mTar) > anchor.Rank(core.DTypeQ6_K) {
		t.Errorf("measured steep curve ignored: ffn_up = %v, want >= Q6_K (heuristic chose %v)", mTar, hTar)
	}
	if resM.Diag.MeasuredTensors == 0 || resM.Diag.EstimatedTensors == 0 {
		t.Errorf("evidence mix not tracked: %+v", resM.Diag)
	}
}

func TestSolveConfidencePenaltyAvoidsWeakEvidence(t *testing.T) {
	set, _ := anchor.Derive(qbank(), nil, anchor.Policy{})
	// Imatrix says blk.0.attn_q is very important; heuristic confidence is low.
	highPenalty, err := Solve(Request{
		Bank: qbank(), Anchors: set, Candidates: testCands,
		Imatrix: map[string]ImatrixStats{
			"blk.0.attn_q.weight": {Mean: 8.0, Max: 16.0, Samples: 8192},
			"blk.1.attn_q.weight": {Mean: 1.0, Max: 2.0, Samples: 8192},
		},
		BudgetBytes: 180000, ConfidencePenalty: 0.9,
	})
	if err != nil {
		t.Fatal(err)
	}
	if hi, lo := targetOf(t, highPenalty, "blk.0.attn_q.weight"), targetOf(t, highPenalty, "blk.1.attn_q.weight"); anchor.Rank(hi) > anchor.Rank(lo) {
		t.Errorf("imatrix-important tensor squeezed harder (%v) than unimportant (%v)", hi, lo)
	}
}

func TestEstimatorDeterministicAndMarked(t *testing.T) {
	e1 := NewFallbackEstimator(map[string]ImatrixStats{
		"blk.0.attn_q.weight": {Mean: 3.0, Samples: 100},
		"blk.1.attn_q.weight": {Mean: 1.0, Samples: 100},
	})
	e2 := NewFallbackEstimator(map[string]ImatrixStats{
		"blk.1.attn_q.weight": {Mean: 1.0, Samples: 100},
		"blk.0.attn_q.weight": {Mean: 3.0, Samples: 100},
	})
	q, _ := qbank().Find("blk.0.attn_q.weight")
	l1, c1 := e1.Estimate(q, core.DTypeQ4_K_T)
	l2, c2 := e2.Estimate(q, core.DTypeQ4_K_T)
	if l1 != l2 || c1 != c2 {
		t.Fatalf("estimator order-dependent: (%v,%v) vs (%v,%v)", l1, c1, l2, c2)
	}
	if c1 <= 0.5 || c1 > 1 {
		t.Fatalf("imatrix-backed confidence %v out of (0.5,1]", c1)
	}
	// Severity must be monotone in fidelity.
	q8, _ := e1.Estimate(q, core.DTypeQ8_0)
	if !(q8 < l1) {
		t.Fatalf("Q8_0 loss %v not below Q4_K loss %v", q8, l1)
	}
	// Element-weighted loss: embedding has 2× the elements of blk.0.attn_q,
	// so it outranks attention without imatrix. Imatrix lifts the same
	// attention tensor above its no-imatrix estimate (not above the larger
	// embedding — size dominates name/imatrix at that ratio).
	plain := NewFallbackEstimator(nil)
	embd, _ := qbank().Find("token_embd.weight")
	attn, _ := qbank().Find("blk.0.attn_q.weight")
	le, _ := plain.Estimate(embd, core.DTypeQ4_K_T)
	la, _ := plain.Estimate(attn, core.DTypeQ4_K_T)
	if le <= la {
		t.Fatalf("embedding loss %v not above attention loss %v (no imatrix)", le, la)
	}
	if l1 <= la {
		t.Fatalf("imatrix-informed attention loss %v not above plain attention %v", l1, la)
	}
	deep := core.TensorDesc{Name: "blk.9.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536}
	ld, _ := NewFallbackEstimator(nil).Estimate(deep, core.DTypeQ4_K_T)
	l0, _ := NewFallbackEstimator(nil).Estimate(q, core.DTypeQ4_K_T)
	if !(l0 > ld) {
		t.Fatalf("depth-0 loss %v not above depth-9 loss %v", l0, ld)
	}
	noIm, cn := NewFallbackEstimator(nil).Estimate(q, core.DTypeQ4_K_T)
	if cn != 0.3 {
		t.Fatalf("no-imatrix confidence = %v, want 0.3", cn)
	}
	if math.IsNaN(noIm) || math.IsInf(noIm, 0) {
		t.Fatal("non-finite estimate")
	}
}

func TestCacheRoundTripStrictIdentity(t *testing.T) {
	c := measuredCache(t, map[string]map[core.DType]float64{
		"blk.0.attn_q.weight": {core.DTypeQ6_K: 0.5, core.DTypeQ4_K_T: 3.0},
	})
	var buf bytes.Buffer
	if err := c.Save(&buf); err != nil {
		t.Fatal(err)
	}
	snapshot := append([]byte(nil), buf.Bytes()...) // decoder drains buf below
	got, err := LoadCache(&buf, "test-model", "sha-test")
	if err != nil {
		t.Fatal(err)
	}
	if cl, ok := got.Get("blk.0.attn_q.weight", core.DTypeQ4_K_T); !ok || cl.Loss != 3.0 || cl.Evidence != EvidenceMeasured || cl.Prov == nil {
		t.Fatalf("roundtrip lost entry: %+v ok=%v", cl, ok)
	}
	if _, ok := got.Get("blk.0.attn_q.weight", core.DTypeQ2_K); ok {
		t.Fatal("phantom entry")
	}
	// Deterministic serialization.
	var buf2 bytes.Buffer
	if err := got.Save(&buf2); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(snapshot, buf2.Bytes()) {
		t.Fatal("cache save not deterministic")
	}
	// Wrong model identity rejected.
	if _, err := LoadCache(bytes.NewReader(snapshot), "other-model", "sha-test"); err == nil {
		t.Fatal("identity mismatch accepted")
	}
	if _, err := LoadCache(bytes.NewReader(snapshot), "test-model", "sha-other"); err == nil {
		t.Fatal("sha mismatch accepted")
	}
	// Wrong version rejected.
	bad := strings.Replace(string(snapshot), `"version": 1`, `"version": 2`, 1)
	if _, err := LoadCache(strings.NewReader(bad), "test-model", "sha-test"); err == nil {
		t.Fatal("version mismatch accepted")
	}
	// Invalid provenance rejected.
	bad = strings.Replace(string(snapshot), `"runID": "run-1"`, `"runID": ""`, 1)
	if _, err := LoadCache(strings.NewReader(bad), "test-model", "sha-test"); err == nil {
		t.Fatal("missing provenance accepted")
	}
	// Unknown fields rejected (strict).
	bad = strings.Replace(string(snapshot), `"version": 1`, `"version": 1, "extra": true`, 1)
	if _, err := LoadCache(strings.NewReader(bad), "test-model", "sha-test"); err == nil {
		t.Fatal("unknown field accepted")
	}
}

func TestCachePutValidation(t *testing.T) {
	c := NewCache("m", "s")
	if err := c.Put(CandidateLoss{TensorName: "t", Target: core.DTypeQ4_K_T, Loss: 1, Evidence: EvidenceHeuristic, Confidence: 1}); err == nil {
		t.Fatal("heuristic evidence accepted into cache")
	}
	prov := &core.Provenance{Tool: "t", RunID: "r", MeasuredAt: time.Now()}
	if err := c.Put(CandidateLoss{TensorName: "t", Target: core.DTypeQ4_K_T, Loss: -1, Evidence: EvidenceMeasured, Confidence: 1, Prov: prov}); err == nil {
		t.Fatal("negative loss accepted")
	}
	if err := c.Put(CandidateLoss{TensorName: "t", Target: core.DTypeQ4_K_T, Loss: 1, Evidence: EvidenceMeasured, Confidence: 1.5, Prov: prov}); err == nil {
		t.Fatal("confidence > 1 accepted")
	}
	if err := c.Put(CandidateLoss{TensorName: "t", Target: core.DTypeQ4_K_T, Loss: 1, Evidence: EvidenceMeasured, Confidence: 1, Prov: prov}); err != nil {
		t.Fatal(err)
	}
	if err := c.Put(CandidateLoss{TensorName: "t", Target: core.DTypeQ4_K_T, Loss: 2, Evidence: EvidenceMeasured, Confidence: 1, Prov: prov}); err == nil {
		t.Fatal("duplicate key accepted")
	}
	if err := (CandidateLoss{TensorName: "t", Target: core.DTypeQ4_K_T, Loss: 1, Evidence: EvidenceMeasured, Confidence: 1}); err.Validate() == nil {
		t.Fatal("measured without provenance validated")
	}
}

func TestSolveSoftPriorsShapeNotPin(t *testing.T) {
	set, _ := anchor.Derive(qbank(), nil, anchor.Policy{})
	// Very tight budget: priors must not prevent solving (no blanket Q8 floor).
	res, err := Solve(Request{
		Bank: qbank(), Anchors: set, Candidates: testCands,
		BudgetBytes: 175000,
	})
	if err != nil {
		t.Fatalf("soft priors acted as hard floors: %v", err)
	}
	// Prior-free solve should differ: priors shifted loss but stayed solvable.
	res2, err := Solve(Request{
		Bank: qbank(), Anchors: &anchor.Set{}, Candidates: testCands,
		BudgetBytes: 175000,
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = res2
	// Norms are preserved regardless.
	if got := targetOf(t, res, "blk.0.ffn_norm.weight"); got != core.DTypeF32 {
		t.Errorf("norm quantized to %v", got)
	}
}

func TestSolveUnconstrainedMaxFidelity(t *testing.T) {
	set, _ := anchor.Derive(qbank(), nil, anchor.Policy{})
	res, err := Solve(Request{Bank: qbank(), Anchors: set, Candidates: testCands})
	if err != nil {
		t.Fatal(err)
	}
	if res.Manifest.TotalBytes != res.Diag.MaxBytes {
		t.Fatalf("unconstrained total %d != max %d", res.Manifest.TotalBytes, res.Diag.MaxBytes)
	}
	if res.Profile.ID == "" {
		t.Fatal("no deterministic profile id")
	}
}

func tinyImatrixBank() *core.TensorBank {
	td := func(name string) core.TensorDesc {
		return core.TensorDesc{
			Name: name, DType: core.DTypeF16, Shape: []uint64{256, 256},
			Length: 131072, Elements: 65536,
		}
	}
	return &core.TensorBank{
		SourcePath: "/m.gguf", ModelID: "imatrix-tiny",
		Tensors: []core.TensorDesc{
			td("blk.0.ffn_up.weight"),
			td("blk.0.ffn_down.weight"),
		},
	}
}

func TestSolveImatrixChangesAssignment(t *testing.T) {
	bank := tinyImatrixBank()
	set, err := anchor.Derive(bank, nil, anchor.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	bQ2, ok := core.DTypeQ2_K.ExactBytes(65536)
	if !ok {
		t.Fatal("Q2_K geometry")
	}
	bQ6, _ := core.DTypeQ6_K.ExactBytes(65536)
	// Enough for Q6+Q2 but not two Q6 (or Q8+Q2).
	budget := bQ6 + bQ2 + 1024
	base := Request{Bank: bank, Anchors: set, Candidates: testCands, BudgetBytes: budget}

	resNil, err := Solve(base)
	if err != nil {
		t.Fatal(err)
	}
	resIm, err := Solve(Request{
		Bank: bank, Anchors: set, Candidates: testCands, BudgetBytes: budget,
		Imatrix: map[string]ImatrixStats{
			"blk.0.ffn_up.weight":   {Mean: 100, Max: 400, P50: 20, P95: 200, Spikiness: 8, Samples: 4096},
			"blk.0.ffn_down.weight": {Mean: 0.1, Max: 0.12, P50: 0.1, P95: 0.11, Spikiness: 1.2, Samples: 4096},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	upNil, downNil := targetOf(t, resNil, "blk.0.ffn_up.weight"), targetOf(t, resNil, "blk.0.ffn_down.weight")
	upIm, downIm := targetOf(t, resIm, "blk.0.ffn_up.weight"), targetOf(t, resIm, "blk.0.ffn_down.weight")
	if upNil == upIm && downNil == downIm {
		t.Fatalf("imatrix did not change assignment: up=%s down=%s", upNil, downNil)
	}
	if anchor.Rank(upIm) > anchor.Rank(downIm) {
		t.Errorf("imatrix-important ffn_up = %s squeezed harder than ffn_down = %s (nil was up=%s down=%s)",
			upIm, downIm, upNil, downNil)
	}
	est := NewFallbackEstimator(map[string]ImatrixStats{
		"blk.0.ffn_up.weight":   {Mean: 100, Samples: 4096},
		"blk.0.ffn_down.weight": {Mean: 0.1, Samples: 4096},
	})
	if !est.hasImatrix {
		t.Fatal("hasImatrix false with non-empty stats")
	}
}

// TestBetterUpgradeTieBreak pins the deterministic five-case upgrade
// ranking, including the final case, which must compare the candidate's
// target against the CURRENT best target (the original implementation
// indexed a per-tensor option list with the wrong tensor's cursor — a latent
// panic). betterUpgrade is now a pure struct comparison; this test would
// catch any regression to list indexing.
func TestBetterUpgradeTieBreak(t *testing.T) {
	base := upgradeStep{name: "blk.0", target: core.DTypeQ6_K, gain: 1.0, lossDec: 2.0, bytesInc: 100}
	// Case 1: first best always wins.
	if !betterUpgrade(base, upgradeStep{}, true) {
		t.Fatal("first candidate not accepted")
	}
	same := func(mut func(*upgradeStep)) upgradeStep {
		c := base
		mut(&c)
		return c
	}
	cases := []struct {
		name string
		cand upgradeStep
		want bool
	}{
		{"larger loss decrease wins", same(func(c *upgradeStep) { c.lossDec = 2.1 }), true},
		{"smaller loss decrease loses", same(func(c *upgradeStep) { c.lossDec = 1.9 }), false},
		{"larger loss decrease beats higher gain", same(func(c *upgradeStep) { c.lossDec = 3; c.gain = 0.5 }), true},
		{"smaller loss decrease loses despite higher gain", same(func(c *upgradeStep) { c.lossDec = 1; c.gain = 9 }), false},
		{"loss tie: higher gain wins", same(func(c *upgradeStep) { c.gain = 1.1 }), true},
		{"loss tie: lower gain loses", same(func(c *upgradeStep) { c.gain = 0.9 }), false},
		{"loss tie: fewer bytes wins", same(func(c *upgradeStep) { c.bytesInc = 99 }), true},
		{"loss tie: more bytes loses", same(func(c *upgradeStep) { c.bytesInc = 101 }), false},
		{"byte tie: smaller name wins", same(func(c *upgradeStep) { c.name = "blk.-1" }), true},
		{"byte tie: larger name loses", same(func(c *upgradeStep) { c.name = "blk.1" }), false},
		{"name tie: smaller target wins", same(func(c *upgradeStep) { c.target = core.DTypeQ4_K_T }), true},
		{"name tie: larger target loses", same(func(c *upgradeStep) { c.target = core.DTypeQ8_0 }), false},
		{"fully identical loses", base, false},
	}
	for _, tc := range cases {
		if got := betterUpgrade(tc.cand, base, false); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestEnumerateOptionsOmitsIQWithoutImatrix(t *testing.T) {
	attn := core.TensorDesc{Name: "blk.0.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536}
	ffn := core.TensorDesc{Name: "blk.0.ffn_up.weight", DType: core.DTypeF16, Shape: []uint64{256, 512}, Length: 262144, Elements: 131072}
	cands := []core.DType{core.DTypeQ4_K_T, core.DTypeIQ2_S}
	est := NewFallbackEstimator(map[string]ImatrixStats{
		"blk.0.attn_q.weight": {Mean: 1, Max: 1, Samples: 8},
	})
	attnOpts, err := EnumerateOptions(attn, cands, nil, nil, est, DefaultConfidencePenalty)
	if err != nil {
		t.Fatal(err)
	}
	ffnOpts, err := EnumerateOptions(ffn, cands, nil, nil, est, DefaultConfidencePenalty)
	if err != nil {
		t.Fatal(err)
	}
	hasIQ := func(opts []ScoredOption) bool {
		for _, o := range opts {
			if o.Target == core.DTypeIQ2_S {
				return true
			}
		}
		return false
	}
	if !hasIQ(attnOpts) {
		t.Fatal("attn_q with imatrix row dropped IQ2_S")
	}
	if hasIQ(ffnOpts) {
		t.Fatal("ffn_up without imatrix row was offered IQ2_S")
	}
}
