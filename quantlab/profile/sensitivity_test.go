package profile

import (
	"fmt"
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
		opts, err := enumerateCalibrated(td, cands, &anchor.Set{}, est, sens, exact[td.Name], true)
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

func TestSolveCalibratedKeepsPolicyFloors(t *testing.T) {
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
	// Output measured as nearly insensitive, attention as very sensitive:
	// the probe must not be able to harvest the protected head.
	sens := &Sensitivity{Roles: map[string]RoleSensitivity{
		"output": {Role: "output", ProbeDType: core.DTypeQ3_K, KLD: 0.001, Elements: 262144},
		"attn_q": {Role: "attn_q", ProbeDType: core.DTypeQ3_K, KLD: 1.0, Elements: 65536},
	}}
	q6o, _ := core.DTypeQ6_K.ExactBytes(262144)
	q8a, _ := core.DTypeQ8_0.ExactBytes(65536)
	q4o, _ := core.DTypeQ4_K_T.ExactBytes(262144)
	res, err := Solve(Request{Bank: bank, Anchors: set, Candidates: cands, BudgetBytes: q6o + q8a, ExactLoss: exact, Sensitivity: sens})
	if err != nil {
		t.Fatal(err)
	}
	if got := targetOf(t, res, "output.weight"); anchor.Rank(got) > anchor.Rank(core.DTypeQ6_K) {
		t.Errorf("output = %s violates the Q6_K policy floor", got)
	}
	if got := targetOf(t, res, "blk.0.attn_q.weight"); got != core.DTypeQ8_0 {
		t.Errorf("attn_q = %s, want Q8_0", got)
	}
	kept := false
	for _, a := range res.Profile.Anchors {
		if a.Matches("output.weight") {
			kept = true
		}
	}
	if !kept {
		t.Error("profile dropped the policy floor on the probed output head")
	}
	// A budget that cannot cover the floor is infeasible, not silently
	// harvested below it.
	if _, err := Solve(Request{Bank: bank, Anchors: set, Candidates: cands, BudgetBytes: q4o + q8a, ExactLoss: exact, Sensitivity: sens}); err == nil {
		t.Error("expected infeasible budget below the policy floor")
	} else if _, ok := err.(*InfeasibleError); !ok {
		t.Errorf("error type = %T", err)
	}
	// The default (uncalibrated) path keeps honoring the floor too.
	res, err = Solve(Request{Bank: bank, Anchors: set, Candidates: cands, BudgetBytes: q6o + q8a, ExactLoss: exact})
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

func TestDepthBuckets(t *testing.T) {
	if DepthBuckets(7) != nil {
		t.Error("7 layers should have no buckets")
	}
	b := DepthBuckets(8)
	if len(b) != 6 {
		t.Fatalf("8 layers: %d buckets, want 6", len(b))
	}
	if b[0] != [2]int{0, 0} {
		t.Errorf("first bucket = %v, want [0 0]", b[0])
	}
	if b[5] != [2]int{7, 7} {
		t.Errorf("last bucket = %v, want [7 7]", b[5])
	}
	// Middle buckets cover 1..6 contiguously.
	pos := 1
	for _, bk := range b[1:5] {
		if bk[0] != pos {
			t.Errorf("bucket start %d, want %d", bk[0], pos)
		}
		pos = bk[1] + 1
	}
	if pos != 7 {
		t.Errorf("middle buckets end at %d, want 7", pos)
	}
}

func TestComputeDepthModelShares(t *testing.T) {
	bank := &core.TensorBank{ModelID: "m", Tensors: []core.TensorDesc{
		{Name: "blk.0.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{64, 64}, Elements: 4096},
		{Name: "blk.4.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{64, 64}, Elements: 4096},
		{Name: "blk.7.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{64, 64}, Elements: 4096},
	}}
	sens := map[string]RoleSensitivity{
		"attn_q": {Role: "attn_q", KLD: 0.30, Elements: 12288, ProbeDType: core.DTypeQ3_K},
	}
	buckets := DepthBuckets(8)
	measured := map[string]float64{
		"depth-0-0": 0.10, // first layer is 2× the average
		"depth-3-4": 0.05, // middle layers average
		"depth-7-7": 0.15, // last layer is 3× the average
	}
	dm := ComputeDepthModel(bank, buckets, sens, measured, 0.002)
	if dm == nil {
		t.Fatal("nil depth model")
	}
	if err := (&Sensitivity{Roles: sens, Depth: dm}).Validate(); err != nil {
		t.Fatal(err)
	}
	// Share normalization: role total is preserved.
	total := 0.0
	for _, v := range dm.Shares {
		total += v
	}
	if diff := total - 0.30; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("share total %v, want 0.30", total)
	}
	// First/last layers get higher-or-equal share than middle of same role.
	s0 := dm.Shares["blk.0.attn_q.weight"]
	sMid := dm.Shares["blk.4.attn_q.weight"]
	s7 := dm.Shares["blk.7.attn_q.weight"]
	if s0 < sMid {
		t.Errorf("layer-0 share %v < mid %v", s0, sMid)
	}
	if s7 < sMid {
		t.Errorf("layer-7 share %v < mid %v", s7, sMid)
	}
	for _, b := range dm.Buckets {
		if b.Factor < DepthFactorMin || b.Factor > DepthFactorMax {
			t.Errorf("bucket %d-%d factor %v outside [%v,%v]", b.First, b.Last, b.Factor, DepthFactorMin, DepthFactorMax)
		}
	}
}

// depthBank is a 10-layer bank with one attention and one FFN tensor per
// layer, equal sizes.
func depthBank() *core.TensorBank {
	bank := &core.TensorBank{ModelID: "m"}
	for l := 0; l < 10; l++ {
		for _, stem := range []string{"attn_q", "ffn_up"} {
			bank.Tensors = append(bank.Tensors, core.TensorDesc{
				Name: fmt.Sprintf("blk.%d.%s.weight", l, stem), DType: core.DTypeF16,
				Shape: []uint64{256, 16}, Elements: 4096,
			})
		}
	}
	return bank
}

func TestDepthFamily(t *testing.T) {
	cases := map[string]string{
		"blk.3.attn_q.weight":        DepthFamilyMix,
		"blk.3.attn_output.weight":   DepthFamilyMix,
		"blk.3.ssm_out.weight":       DepthFamilyMix,
		"blk.3.ffn_down.weight":      DepthFamilyFFN,
		"blk.3.ffn_gate_exps.weight": DepthFamilyFFN,
		"blk.3.ffn_up_shexp.weight":  DepthFamilyFFN,
		"model.layers.3.mlp.up_proj": DepthFamilyFFN,
	}
	for name, want := range cases {
		if got := DepthFamily(name); got != want {
			t.Errorf("DepthFamily(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestComputeDepthModelInterpolatesAndReproducesBuckets(t *testing.T) {
	bank := depthBank()
	sens := map[string]RoleSensitivity{
		"attn_q": {Role: "attn_q", KLD: 1.0, Elements: 40960, ProbeDType: core.DTypeQ3_K},
		"ffn_up": {Role: "ffn_up", KLD: 1.0, Elements: 40960, ProbeDType: core.DTypeQ3_K},
	}
	buckets := DepthBuckets(10) // {0} {1-2} {3-4} {5-6} {7-8} {9}
	// Attention is front-loaded, FFN back-loaded: one shared factor per
	// bucket cannot express both.
	measured := map[string]float64{}
	attn := []float64{0.4, 0.4, 0.2, 0.1, 0.1, 0.05}
	ffn := []float64{0.05, 0.1, 0.1, 0.2, 0.4, 0.4}
	for i, b := range buckets {
		measured[DepthKey(DepthFamilyMix, b)] = attn[i]
		measured[DepthKey(DepthFamilyFFN, b)] = ffn[i]
	}
	dm := ComputeDepthModel(bank, buckets, sens, measured, 0)
	if dm == nil {
		t.Fatal("nil depth model")
	}
	share := func(l int, stem string) float64 { return dm.Shares[fmt.Sprintf("blk.%d.%s.weight", l, stem)] }
	if !(share(0, "attn_q") > share(9, "attn_q")) || !(share(9, "ffn_up") > share(0, "ffn_up")) {
		t.Errorf("families not separated: attn0=%v attn9=%v ffn0=%v ffn9=%v",
			share(0, "attn_q"), share(9, "attn_q"), share(0, "ffn_up"), share(9, "ffn_up"))
	}
	// Smooth: within bucket {3-4} the two layers differ (interpolated, not
	// a step), and the profile is monotone where the knots are.
	if share(3, "attn_q") == share(4, "attn_q") {
		t.Error("layers 3 and 4 share a factor; expected interpolation")
	}
	// Monotone measurements give a monotone profile (no invented dips).
	for l := 0; l < 9; l++ {
		if share(l+1, "ffn_up") < share(l, "ffn_up")*(1-1e-9) {
			t.Errorf("ffn profile not monotone at %d: %v -> %v", l, share(l, "ffn_up"), share(l+1, "ffn_up"))
		}
	}
	// The fit reproduces each bucket's measured/predicted ratio in total:
	// Σ_{t∈b} pred_t·ρ(l_t) ≈ measured_b, i.e. bucket shares are
	// proportional to measurements once each role is renormalized.
	var sumMeasured float64
	for _, v := range attn {
		sumMeasured += v
	}
	for i, b := range buckets {
		var got float64
		for l := b[0]; l <= b[1]; l++ {
			got += share(l, "attn_q")
		}
		want := attn[i] / sumMeasured * 1.0
		// Interpolation smooths across bucket edges, so interior totals
		// track measurements approximately; single-layer edges exactly.
		tol := 0.15 * want
		if b[0] == b[1] {
			tol = 1e-9 + 0.02*want
		}
		if math.Abs(got-want) > tol {
			t.Errorf("attn bucket %v share %.4f, want ≈ %.4f", b, got, want)
		}
	}
	for _, rb := range dm.Buckets {
		if rb.Family == "" {
			t.Errorf("family-split buckets must carry their family: %+v", rb)
		}
	}
}

func TestComputeDepthModelCombinedFallback(t *testing.T) {
	bank := depthBank()
	sens := map[string]RoleSensitivity{
		"attn_q": {Role: "attn_q", KLD: 1.0, Elements: 40960, ProbeDType: core.DTypeQ3_K},
		"ffn_up": {Role: "ffn_up", KLD: 1.0, Elements: 40960, ProbeDType: core.DTypeQ3_K},
	}
	buckets := DepthBuckets(10)
	measured := map[string]float64{}
	for i, b := range buckets {
		measured[DepthKey("", b)] = []float64{0.8, 0.3, 0.2, 0.2, 0.3, 0.8}[i]
	}
	dm := ComputeDepthModel(bank, buckets, sens, measured, 0)
	if dm == nil {
		t.Fatal("nil depth model")
	}
	// Combined probes give both families the same profile.
	for l := 0; l < 10; l++ {
		a := dm.Shares[fmt.Sprintf("blk.%d.attn_q.weight", l)]
		f := dm.Shares[fmt.Sprintf("blk.%d.ffn_up.weight", l)]
		if math.Abs(a-f) > 1e-12 {
			t.Errorf("layer %d: attn %v != ffn %v under a combined probe", l, a, f)
		}
	}
}

func TestComputeDepthModelFlagsClamp(t *testing.T) {
	bank := depthBank()
	sens := map[string]RoleSensitivity{
		"attn_q": {Role: "attn_q", KLD: 0.01, Elements: 40960, ProbeDType: core.DTypeQ3_K},
	}
	b := DepthBuckets(10)[0]
	dm := ComputeDepthModel(bank, DepthBuckets(10), sens, map[string]float64{DepthKey(DepthFamilyMix, b): 50}, 0)
	if dm == nil || len(dm.Buckets) != 1 || !dm.Buckets[0].Clamped || dm.Buckets[0].Factor != DepthFactorMax {
		t.Fatalf("expected one clamped bucket at %v, got %+v", DepthFactorMax, dm)
	}
}

func TestFitRungExponent(t *testing.T) {
	// KLD quadruples while wSSE doubles: b = 2.
	if b, ok := FitRungExponent(0.1, 1, 0.4, 2); !ok || math.Abs(b-2) > 1e-12 {
		t.Errorf("b = %v %v, want 2", b, ok)
	}
	// Linear regime.
	if b, ok := FitRungExponent(0.1, 1, 0.3, 3); !ok || math.Abs(b-1) > 1e-12 {
		t.Errorf("b = %v %v, want 1", b, ok)
	}
	// Clamped to the guard.
	if b, ok := FitRungExponent(0.1, 1, 10, 2); !ok || b != MaxRungExponent {
		t.Errorf("b = %v %v, want clamp %v", b, ok, MaxRungExponent)
	}
	// Rungs too close in wSSE, or unusable points.
	if _, ok := FitRungExponent(0.1, 1, 0.12, 1.2); ok {
		t.Error("expected no fit for close rungs")
	}
	if _, ok := FitRungExponent(0, 1, 0.4, 2); ok {
		t.Error("expected no fit for a zero KLD")
	}
}

func TestSensitivityLossAppliesExponent(t *testing.T) {
	tensor := core.TensorDesc{Name: "blk.0.attn_q.weight", DType: core.DTypeF16, Shape: []uint64{256, 16}, Elements: 4096}
	row := map[core.DType]float64{core.DTypeQ3_K: 4, core.DTypeQ4_K_T: 1, core.DTypeQ2_K: 16}
	mk := func(b float64) *Sensitivity {
		return &Sensitivity{Roles: map[string]RoleSensitivity{
			"attn_q": {Role: "attn_q", ProbeDType: core.DTypeQ3_K, KLD: 0.2, Elements: 4096, Exponent: b},
		}}
	}
	lin, _ := mk(0).Loss(tensor, core.DTypeQ4_K_T, row)
	sq, _ := mk(2).Loss(tensor, core.DTypeQ4_K_T, row)
	atProbe, _ := mk(2).Loss(tensor, core.DTypeQ3_K, row)
	if math.Abs(lin-0.05) > 1e-12 || math.Abs(sq-0.0125) > 1e-12 {
		t.Errorf("Q4_K loss linear %v (want 0.05), b=2 %v (want 0.0125)", lin, sq)
	}
	if math.Abs(atProbe-0.2) > 1e-12 {
		t.Errorf("probe-rung loss %v must equal the measured 0.2 for any b", atProbe)
	}
	if err := mk(3).Validate(); err == nil {
		t.Error("exponent outside the guard must fail validation")
	}
}

func TestSecondProbeDTypes(t *testing.T) {
	if SecondProbeDTypes(0) != nil {
		t.Error("unset target must not plan a second rung")
	}
	for bpw, want := range map[float64]core.DType{2.2: core.DTypeIQ2_XS, 3.5: core.DTypeQ4_K_T, 5.0: core.DTypeQ5_K_T, 7.0: core.DTypeQ6_K} {
		if got := SecondProbeDTypes(bpw); len(got) == 0 || got[0] != want {
			t.Errorf("SecondProbeDTypes(%v) = %v, want first %s", bpw, got, want)
		}
	}
}

func TestSolveCalibratedPricesSmallUnprobedRoles(t *testing.T) {
	bank, cands, exact, sens := calibratedFixture(1.0, 0.5)
	for role, r := range sens.Roles {
		var sum float64
		for _, td := range bank.Tensors {
			if RoleKey(td.Name) == role {
				sum += exact[td.Name][core.DTypeQ3_K]
				r.Tensors = append(r.Tensors, td.Name)
			}
		}
		r.SumWSSE = sum
		sens.Roles[role] = r
	}
	small := core.TensorDesc{Name: "blk.0.attn_k_b.weight", DType: core.DTypeF16, Shape: []uint64{256, 8}, Length: 4096, Elements: 2048}
	router := core.TensorDesc{Name: "blk.0.ffn_gate_inp.weight", DType: core.DTypeF16, Shape: []uint64{256, 8}, Length: 4096, Elements: 2048}
	bank.Tensors = append(bank.Tensors, small, router)
	// The small tensor's own error is negligible at every rung.
	exact[small.Name] = synthRow(small, cands, 1e-6)
	exact[router.Name] = synthRow(router, cands, 1e-6)
	sens.Pinned = []string{"attn_k_b", "ffn_gate_inp"}
	est := NewFallbackEstimator(nil)
	est.BindBank(bank)
	priced, err := enumerateCalibrated(small, cands, &anchor.Set{}, est, sens, exact[small.Name], true)
	if err != nil {
		t.Fatal(err)
	}
	if len(priced) < 2 {
		t.Fatalf("priced frontier = %+v, want several rungs", priced)
	}
	rate, ok := sens.PinnedRate(small.Name)
	if !ok {
		t.Fatal("no pinned rate")
	}
	// The mix family's most sensitive rate is attention's.
	if want := sens.Roles["attn_q"].KLD / sens.Roles["attn_q"].SumWSSE; math.Abs(rate-want) > 1e-12*want {
		t.Errorf("rate = %v, want attn_q's %v", rate, want)
	}
	for _, o := range priced {
		if want := rate * exact[small.Name][o.Target]; math.Abs(o.Loss-want) > 1e-15+1e-12*want {
			t.Errorf("%s loss %v, want %v", o.Target, o.Loss, want)
		}
	}
	pinned, err := enumerateCalibrated(small, cands, &anchor.Set{}, est, sens, exact[small.Name], false)
	if err != nil || len(pinned) != 1 || pinned[0].Target != core.DTypeQ8_0 {
		t.Errorf("PinUnprobed frontier = %+v %v, want Q8_0 only", pinned, err)
	}
	rt, err := enumerateCalibrated(router, cands, &anchor.Set{}, est, sens, exact[router.Name], true)
	if err != nil || len(rt) != 1 || rt[0].Target != core.DTypeQ8_0 {
		t.Errorf("router frontier = %+v %v, want pinned Q8_0", rt, err)
	}
	// A budget-constrained solve then trades the priced tensor like any
	// other: with only Q2_K-level bytes for it, it is not forced to Q8_0.
	var budget uint64
	for _, td := range bank.Tensors {
		switch {
		case td.Name == router.Name:
			b, _ := core.DTypeQ8_0.ExactBytes(td.Elements)
			budget += b
		case td.Quantizable():
			b, _ := core.DTypeQ2_K.ExactBytes(td.Elements)
			budget += b
		default:
			budget += td.Length
		}
	}
	res, err := Solve(Request{Bank: bank, Candidates: cands, BudgetBytes: budget, ExactLoss: exact, Sensitivity: sens})
	if err != nil {
		t.Fatal(err)
	}
	if got := targetOf(t, res, small.Name); got == core.DTypeQ8_0 {
		t.Errorf("priced small role kept Q8_0 under a Q2_K-level budget")
	}
	if _, err := Solve(Request{Bank: bank, Candidates: cands, BudgetBytes: budget, ExactLoss: exact, Sensitivity: sens, PinUnprobed: true}); err == nil {
		t.Error("PinUnprobed: expected infeasible at a budget that only fits the small role below Q8_0")
	}
}

func TestSensitivityResolvesMergedRole(t *testing.T) {
	sens := &Sensitivity{Roles: map[string]RoleSensitivity{
		"other": {Role: "other", ProbeDType: core.DTypeQ3_K, KLD: 0.1, Elements: 8192,
			Tensors: []string{"blk.0.ssm_x.weight", "blk.1.ssm_x.weight"}},
	}}
	if key, ok := sens.RoleOf("blk.1.ssm_x.weight"); !ok || key != "other" {
		t.Errorf("RoleOf merged member = %q %v, want other", key, ok)
	}
	if !sens.Calibrated("blk.0.ssm_x.weight") {
		t.Error("merged-role member must count as calibrated")
	}
	td := core.TensorDesc{Name: "blk.0.ssm_x.weight", DType: core.DTypeF16, Shape: []uint64{256, 16}, Elements: 4096}
	if l, ok := sens.Loss(td, core.DTypeQ3_K, map[core.DType]float64{core.DTypeQ3_K: 2}); !ok || math.Abs(l-0.05) > 1e-12 {
		t.Errorf("merged-role loss = %v %v, want 0.05", l, ok)
	}
}

// Large single-layer edge factors must not leak into the interior: with a
// uniform interior, every interior layer keeps the same weight.
func TestComputeDepthModelEdgesDoNotCarveInterior(t *testing.T) {
	for _, n := range []int{16, 64} {
		bank := &core.TensorBank{ModelID: "m"}
		for l := 0; l < n; l++ {
			bank.Tensors = append(bank.Tensors, core.TensorDesc{
				Name: fmt.Sprintf("blk.%d.attn_q.weight", l), DType: core.DTypeF16,
				Shape: []uint64{256, 16}, Elements: 4096,
			})
		}
		sens := map[string]RoleSensitivity{
			"attn_q": {Role: "attn_q", KLD: 1.0, Elements: uint64(4096 * n), ProbeDType: core.DTypeQ3_K},
		}
		buckets := DepthBuckets(n)
		per := 1.0 / float64(n)
		measured := map[string]float64{}
		for _, b := range buckets {
			layers := float64(b[1] - b[0] + 1)
			f := 0.83
			if b[0] == b[1] {
				f = 5
			}
			measured[DepthKey(DepthFamilyMix, b)] = f * per * layers
		}
		dm := ComputeDepthModel(bank, buckets, sens, measured, 0)
		if dm == nil {
			t.Fatal("nil depth model")
		}
		for _, b := range dm.Buckets {
			if b.Clamped {
				t.Errorf("n=%d: bucket %d-%d clamped", n, b.First, b.Last)
			}
		}
		ref := dm.Shares["blk.1.attn_q.weight"]
		for l := 1; l < n-1; l++ {
			got := dm.Shares[fmt.Sprintf("blk.%d.attn_q.weight", l)]
			if math.Abs(got-ref) > 1e-12*ref {
				t.Fatalf("n=%d: interior layer %d share %v != layer 1 %v (edge leaked)", n, l, got, ref)
			}
		}
		edge := dm.Shares["blk.0.attn_q.weight"]
		if r := edge / ref; math.Abs(r-5/0.83) > 1e-9 {
			t.Errorf("n=%d: edge/interior = %v, want %v", n, r, 5/0.83)
		}
	}
}

func TestPinnedRateIgnoresHeadForLayerTensors(t *testing.T) {
	sens := &Sensitivity{Roles: map[string]RoleSensitivity{
		"output": {Role: "output", ProbeDType: core.DTypeQ3_K, KLD: 1, SumWSSE: 1, Elements: 1, Tensors: []string{"output.weight"}},
		"attn_q": {Role: "attn_q", ProbeDType: core.DTypeQ3_K, KLD: 1, SumWSSE: 100, Elements: 1, Tensors: []string{"blk.0.attn_q.weight"}},
	}}
	if r, ok := sens.PinnedRate("blk.0.attn_k_b.weight"); !ok || r != 0.01 {
		t.Errorf("layer tensor rate = %v %v, want attn_q's 0.01 (not the head's 1)", r, ok)
	}
	if r, ok := sens.PinnedRate("some_global.weight"); !ok || r != 1 {
		t.Errorf("global tensor rate = %v %v, want the max over all roles", r, ok)
	}
}
