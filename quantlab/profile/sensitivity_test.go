package profile

import (
	"math"
	"testing"

	"quantlab/anchor"
	"quantlab/core"
)

func TestGroupByRole(t *testing.T) {
	bank := qbank()
	groups := GroupByRole(bank, 0, nil)
	want := []string{"token_embd", "attn_q", "attn_v", "ffn_up", "ffn_down"}
	if len(groups) != len(want) {
		t.Fatalf("groups = %+v", groups)
	}
	for i, g := range groups {
		if g.Role != want[i] {
			t.Errorf("group %d = %q, want %q", i, g.Role, want[i])
		}
	}
	if q := groups[1]; len(q.Tensors) != 2 || q.Elements != 2*65536 {
		t.Errorf("attn_q group = %+v", q)
	}
	// Norms are not quantizable and never form a group.
	for _, g := range groups {
		if g.Role == "ffn_norm" {
			t.Error("norm grouped")
		}
	}
	// Capping merges the smallest roles into "other".
	capped := GroupByRole(bank, 3, nil)
	if len(capped) != 3 || capped[2].Role != "other" {
		t.Fatalf("capped = %+v", capped)
	}
	if capped[0].Role != "token_embd" || capped[1].Role != "attn_q" {
		t.Errorf("kept the wrong roles: %+v", capped)
	}
	if capped[2].Elements != 65536+131072+65536 {
		t.Errorf("other elements = %d", capped[2].Elements)
	}
}

// synthRow builds a monotone exact-loss row: error shrinks geometrically
// with bits per weight, scaled by scale.
func synthRow(t core.TensorDesc, cands []core.DType, scale float64) map[core.DType]float64 {
	row := map[core.DType]float64{}
	for _, d := range cands {
		b, ok := d.ExactBytes(t.Elements)
		if !ok {
			continue
		}
		bpw := float64(b) * 8 / float64(t.Elements)
		row[d] = scale * math.Pow(2, -1.6*bpw)
	}
	return row
}

func TestSensitivityLossIsRungRelativeAndDepthFlat(t *testing.T) {
	bank := qbank()
	q0, _ := bank.Find("blk.0.attn_q.weight")
	q1, _ := bank.Find("blk.1.attn_q.weight")
	cands := []core.DType{core.DTypeQ8_0, core.DTypeQ6_K, core.DTypeQ4_K_T, core.DTypeQ3_K, core.DTypeQ2_K}
	// Layer 1 has a 50x larger raw wSSE scale (activation power grows with
	// depth); the calibrated model must not let that dominate.
	row0 := synthRow(q0, cands, 1)
	row1 := synthRow(q1, cands, 50)
	sens := &Sensitivity{Roles: map[string]RoleSensitivity{
		"attn_q": {Role: "attn_q", ProbeDType: core.DTypeQ3_K, KLD: 0.2, Elements: q0.Elements + q1.Elements},
	}}
	if err := sens.Validate(); err != nil {
		t.Fatal(err)
	}
	l0, ok0 := sens.Loss(q0, core.DTypeQ3_K, row0)
	l1, ok1 := sens.Loss(q1, core.DTypeQ3_K, row1)
	if !ok0 || !ok1 {
		t.Fatal("loss unavailable")
	}
	if math.Abs(l0-0.1) > 1e-12 || math.Abs(l1-0.1) > 1e-12 {
		t.Errorf("probe-rung shares = %v, %v; want 0.1 each (depth-flat)", l0, l1)
	}
	// Other rungs scale with the tensor's own wSSE ratio.
	l8, _ := sens.Loss(q0, core.DTypeQ8_0, row0)
	if want := 0.1 * row0[core.DTypeQ8_0] / row0[core.DTypeQ3_K]; math.Abs(l8-want) > 1e-12 {
		t.Errorf("Q8_0 loss = %v, want %v", l8, want)
	}
	// Unknown role or dtype: not calibrated.
	if _, ok := sens.Loss(core.TensorDesc{Name: "blk.0.ffn_up.weight", Elements: 10}, core.DTypeQ3_K, row0); ok {
		t.Error("uncalibrated role returned a loss")
	}
	if _, ok := sens.Loss(q0, core.DTypeIQ2_XS, row0); ok {
		t.Error("dtype missing from the row returned a loss")
	}
	// Probe rung absent from the row: the closest-bytes rung is the reference.
	noQ3 := map[core.DType]float64{}
	for d, v := range row0 {
		if d != core.DTypeQ3_K {
			noQ3[d] = v
		}
	}
	// Q2_K (2.625 bpw) is nearer to Q3_K (3.4375) than Q4_K (4.5), so it
	// carries the probe share.
	l, ok := sens.Loss(q0, core.DTypeQ2_K, noQ3)
	if !ok || math.Abs(l-0.1) > 1e-12 {
		t.Errorf("fallback reference: Q2_K loss = %v ok=%v, want 0.1 (Q2_K is the nearest rung)", l, ok)
	}
}

func TestProbeDTypeFor(t *testing.T) {
	bank := &core.TensorBank{ModelID: "m", SourcePath: "/m", Tensors: []core.TensorDesc{
		{Name: "blk.0.a.weight", DType: core.DTypeF16, Shape: []uint64{256, 8}, Length: 4096, Elements: 2048},
		{Name: "blk.0.b.weight", DType: core.DTypeF16, Shape: []uint64{96, 8}, Length: 1536, Elements: 768},
	}}
	if d, ok := ProbeDTypeFor(bank, []string{"blk.0.a.weight"}, DefaultProbeDTypes, nil); !ok || d != core.DTypeQ3_K {
		t.Errorf("256-aligned role: %v %v", d, ok)
	}
	// A 32-only member forces the 32-block fallback for the whole role.
	if d, ok := ProbeDTypeFor(bank, []string{"blk.0.a.weight", "blk.0.b.weight"}, DefaultProbeDTypes, nil); !ok || d != core.DTypeQ4_0 {
		t.Errorf("mixed role: %v %v", d, ok)
	}
	// The exact table must cover the rung.
	exact := map[string]map[core.DType]float64{"blk.0.a.weight": {core.DTypeQ4_0: 1}}
	if d, ok := ProbeDTypeFor(bank, []string{"blk.0.a.weight"}, DefaultProbeDTypes, exact); !ok || d != core.DTypeQ4_0 {
		t.Errorf("table-limited role: %v %v", d, ok)
	}
}

func TestLowerHull(t *testing.T) {
	opts := []ScoredOption{
		{TensorOption: core.TensorOption{Bytes: 100}, Loss: 10},
		{TensorOption: core.TensorOption{Bytes: 200}, Loss: 9}, // above the chord 10 -> 1: dropped
		{TensorOption: core.TensorOption{Bytes: 300}, Loss: 1},
		{TensorOption: core.TensorOption{Bytes: 400}, Loss: 0.5},
	}
	got := lowerHull(opts)
	want := []int{0, 2, 3}
	if len(got) != len(want) {
		t.Fatalf("hull = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("hull = %v, want %v", got, want)
		}
	}
}

// calibratedFixture is a two-role bank: attention (probed as sensitive)
// and FFN (probed as insensitive), identical sizes, plus a preserved norm.
func calibratedFixture(kldAttn, kldFFN float64) (*core.TensorBank, []core.DType, map[string]map[core.DType]float64, *Sensitivity) {
	bank := &core.TensorBank{SourcePath: "/m.gguf", ModelID: "cal", Tensors: []core.TensorDesc{
		{Name: "blk.0.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
		{Name: "blk.1.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
		{Name: "blk.0.ffn_up.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
		{Name: "blk.1.ffn_up.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
		{Name: "blk.0.ffn_norm.weight", DType: core.DTypeF32, Shape: []uint64{256}, Length: 1024, Elements: 256},
	}}
	cands := []core.DType{core.DTypeQ8_0, core.DTypeQ6_K, core.DTypeQ4_K_T, core.DTypeQ3_K, core.DTypeQ2_K}
	exact := map[string]map[core.DType]float64{}
	for _, td := range bank.Tensors {
		if td.Quantizable() {
			exact[td.Name] = synthRow(td, cands, 1)
		}
	}
	sens := &Sensitivity{Background: 0.001, Roles: map[string]RoleSensitivity{
		"attn_q": {Role: "attn_q", ProbeDType: core.DTypeQ3_K, KLD: kldAttn, Elements: 2 * 65536},
		"ffn_up": {Role: "ffn_up", ProbeDType: core.DTypeQ3_K, KLD: kldFFN, Elements: 2 * 65536},
	}}
	return bank, cands, exact, sens
}

func TestSolveCalibratedSpendsOnSensitiveRole(t *testing.T) {
	bank, cands, exact, sens := calibratedFixture(1.0, 0.02)
	// Budget: enough for roughly Q6_K on two tensors and Q3_K on two.
	q6, _ := core.DTypeQ6_K.ExactBytes(65536)
	q3, _ := core.DTypeQ3_K.ExactBytes(65536)
	budget := 2*q6 + 2*q3 + 1024
	res, err := Solve(Request{Bank: bank, Candidates: cands, BudgetBytes: budget, ExactLoss: exact, Sensitivity: sens})
	if err != nil {
		t.Fatal(err)
	}
	if res.Profile.EstimatedBytes > budget {
		t.Fatalf("over budget: %d > %d", res.Profile.EstimatedBytes, budget)
	}
	for _, name := range []string{"blk.0.attn_q.weight", "blk.1.attn_q.weight"} {
		if got := targetOf(t, res, name); anchor.Rank(got) > anchor.Rank(core.DTypeQ6_K) {
			t.Errorf("%s = %s, want >= Q6_K", name, got)
		}
	}
	for _, name := range []string{"blk.0.ffn_up.weight", "blk.1.ffn_up.weight"} {
		if got := targetOf(t, res, name); anchor.Rank(got) < anchor.Rank(core.DTypeQ4_K_T) {
			t.Errorf("%s = %s, want <= Q4_K", name, got)
		}
	}
	if got := targetOf(t, res, "blk.0.ffn_norm.weight"); got != core.DTypeF32 {
		t.Errorf("norm = %s, want preserved F32", got)
	}
	if res.Diag.MeasuredTensors != 4 {
		t.Errorf("measured tensors = %d, want 4", res.Diag.MeasuredTensors)
	}
	// Brute-force optimum over the same option sets: the Lagrangian sweep
	// plus refill must match it on this small convex instance.
	best := bruteForceCalibrated(t, bank, cands, exact, sens, budget)
	if res.Diag.TotalLoss > best*(1+1e-9) {
		t.Errorf("total loss %v exceeds brute-force optimum %v", res.Diag.TotalLoss, best)
	}
}

func bruteForceCalibrated(t *testing.T, bank *core.TensorBank, cands []core.DType,
	exact map[string]map[core.DType]float64, sens *Sensitivity, budget uint64) float64 {
	t.Helper()
	est := NewFallbackEstimator(nil)
	est.BindBank(bank)
	var fronts [][]ScoredOption
	for _, td := range bank.Tensors {
		opts, err := enumerateCalibrated(td, cands, &anchor.Set{}, est, sens, exact[td.Name])
		if err != nil {
			t.Fatal(err)
		}
		fronts = append(fronts, opts)
	}
	best := math.Inf(1)
	var rec func(i int, bytes uint64, loss float64)
	rec = func(i int, bytes uint64, loss float64) {
		if bytes > budget {
			return
		}
		if i == len(fronts) {
			if loss < best {
				best = loss
			}
			return
		}
		for _, o := range fronts[i] {
			rec(i+1, bytes+o.Bytes, loss+o.Loss)
		}
	}
	rec(0, 0, 0)
	return best
}

func TestSolveCalibratedDropsPolicyFloorsForProbedRoles(t *testing.T) {
	bank := &core.TensorBank{SourcePath: "/m.gguf", ModelID: "cal", Tensors: []core.TensorDesc{
		{Name: "output.weight", DType: core.DTypeF16, Shape: []uint64{256, 1024}, Length: 524288, Elements: 262144},
		{Name: "blk.0.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
	}}
	cands := []core.DType{core.DTypeQ8_0, core.DTypeQ6_K, core.DTypeQ4_K_T, core.DTypeQ3_K}
	exact := map[string]map[core.DType]float64{}
	for _, td := range bank.Tensors {
		exact[td.Name] = synthRow(td, cands, 1)
	}
	set, err := anchor.Derive(bank, nil, anchor.PolicyForBPW(4.5))
	if err != nil {
		t.Fatal(err)
	}
	if floor, ok := set.Floor("output.weight"); !ok || floor != core.DTypeQ6_K {
		t.Fatalf("fixture: expected a Q6_K output floor, got %v %v", floor, ok)
	}
	// Output measured as nearly insensitive, attention as very sensitive.
	sens := &Sensitivity{Roles: map[string]RoleSensitivity{
		"output": {Role: "output", ProbeDType: core.DTypeQ3_K, KLD: 0.001, Elements: 262144},
		"attn_q": {Role: "attn_q", ProbeDType: core.DTypeQ3_K, KLD: 1.0, Elements: 65536},
	}}
	q4o, _ := core.DTypeQ4_K_T.ExactBytes(262144)
	q8a, _ := core.DTypeQ8_0.ExactBytes(65536)
	res, err := Solve(Request{Bank: bank, Anchors: set, Candidates: cands, BudgetBytes: q4o + q8a, ExactLoss: exact, Sensitivity: sens})
	if err != nil {
		t.Fatal(err)
	}
	if got := targetOf(t, res, "output.weight"); anchor.Rank(got) <= anchor.Rank(core.DTypeQ6_K) {
		t.Errorf("output = %s; the measured-insensitive head should drop below the Q6_K policy floor", got)
	}
	if got := targetOf(t, res, "blk.0.attn_q.weight"); got != core.DTypeQ8_0 {
		t.Errorf("attn_q = %s, want Q8_0", got)
	}
	for _, a := range res.Profile.Anchors {
		if a.Matches("output.weight") {
			t.Errorf("profile still records a floor on the probed output head: %+v", a)
		}
	}
	// The default (uncalibrated) path keeps honoring the floor.
	res, err = Solve(Request{Bank: bank, Anchors: set, Candidates: cands, BudgetBytes: q4o + q8a + 1<<20, ExactLoss: exact})
	if err != nil {
		t.Fatal(err)
	}
	if got := targetOf(t, res, "output.weight"); anchor.Rank(got) > anchor.Rank(core.DTypeQ6_K) {
		t.Errorf("uncalibrated output = %s violates the policy floor", got)
	}
}

func TestSolveCalibratedDegenerateRows(t *testing.T) {
	bank, cands, exact, sens := calibratedFixture(1.0, 0.5)
	// All-zero weights: every rung is lossless, so the cheapest wins.
	zero := exact["blk.0.ffn_up.weight"]
	for d := range zero {
		zero[d] = 0
	}
	// Corrupt weights: a non-finite row pins the tensor to top fidelity.
	bad := exact["blk.1.ffn_up.weight"]
	for d := range bad {
		bad[d] = math.NaN()
	}
	res, err := Solve(Request{Bank: bank, Candidates: cands, BudgetBytes: 0, ExactLoss: exact, Sensitivity: sens})
	if err != nil {
		t.Fatal(err)
	}
	if got := targetOf(t, res, "blk.0.ffn_up.weight"); got != core.DTypeQ2_K {
		t.Errorf("zero-error tensor = %s, want the cheapest rung Q2_K", got)
	}
	if got := targetOf(t, res, "blk.1.ffn_up.weight"); got != core.DTypeQ8_0 {
		t.Errorf("non-finite row = %s, want pinned Q8_0", got)
	}
}

func TestSolveCalibratedPinsUnprobedRoles(t *testing.T) {
	bank, cands, exact, sens := calibratedFixture(1.0, 0.5)
	// A tiny router-like tensor whose role was not probed.
	router := core.TensorDesc{Name: "blk.0.ffn_gate_inp.weight", DType: core.DTypeF16, Shape: []uint64{256, 8}, Length: 4096, Elements: 2048}
	bank.Tensors = append(bank.Tensors, router)
	exact[router.Name] = synthRow(router, cands, 1)
	sens.Pinned = []string{"ffn_gate_inp"}
	res, err := Solve(Request{Bank: bank, Candidates: cands, BudgetBytes: 0, ExactLoss: exact, Sensitivity: sens})
	if err != nil {
		t.Fatal(err)
	}
	if got := targetOf(t, res, router.Name); got != core.DTypeQ8_0 {
		t.Errorf("pinned role = %s, want the highest-fidelity rung Q8_0", got)
	}
	// Calibrated solve requires the exact table.
	if _, err := Solve(Request{Bank: bank, Candidates: cands, Sensitivity: sens}); err == nil {
		t.Error("expected an error without an exact loss table")
	}
	// Infeasible budgets still report the envelope.
	if _, err := Solve(Request{Bank: bank, Candidates: cands, BudgetBytes: 1, ExactLoss: exact, Sensitivity: sens}); err == nil {
		t.Error("expected infeasible error")
	} else if _, ok := err.(*InfeasibleError); !ok {
		t.Errorf("error type = %T", err)
	}
}
