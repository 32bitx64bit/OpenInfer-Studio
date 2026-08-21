package pipeline

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"quantlab/core"
	"quantlab/state"
)

func TestEffortForPresets(t *testing.T) {
	fast, err := EffortFor(EffortFast)
	if err != nil {
		t.Fatal(err)
	}
	if fast.Reconstruct || fast.InPlaceReconstruct || fast.ScaleFold || fast.ProbeKLD || fast.SolverFTI {
		t.Errorf("fast reconstruct/inplace/fold/probeKLD/solverFTI = %v/%v/%v/%v/%v, want false",
			fast.Reconstruct, fast.InPlaceReconstruct, fast.ScaleFold, fast.ProbeKLD, fast.SolverFTI)
	}
	if fast.EvalChunks != 2 {
		t.Errorf("fast chunks = %d, want 2", fast.EvalChunks)
	}
	if fast.EvalCtx != 512 {
		t.Errorf("fast ctx = %d, want 512", fast.EvalCtx)
	}
	wantRecipes := []core.DType{core.DTypeQ3_K_M, core.DTypeQ4_K_M}
	if len(fast.AnchorRecipes) != 2 || fast.AnchorRecipes[0] != wantRecipes[0] || fast.AnchorRecipes[1] != wantRecipes[1] {
		t.Errorf("fast recipes = %v", fast.AnchorRecipes)
	}
	wantLattice := []core.DType{
		core.DTypeQ8_0, core.DTypeQ6_K, core.DTypeQ5_K_T, core.DTypeQ5_1,
		core.DTypeIQ4_NL, core.DTypeQ4_K_T, core.DTypeQ4_1, core.DTypeIQ4_XS,
		core.DTypeQ4_0, core.DTypeIQ3_S, core.DTypeQ3_K, core.DTypeQ2_K,
		core.DTypeIQ2_XS, core.DTypeIQ2_XXS,
	}
	if len(fast.Candidates) != len(wantLattice) {
		t.Fatalf("fast lattice = %v", fast.Candidates)
	}
	for i, d := range wantLattice {
		if fast.Candidates[i] != d {
			t.Fatalf("fast lattice[%d] = %s, want %s", i, fast.Candidates[i], d)
		}
	}

	for _, e := range []Effort{"", EffortProfiled} {
		p, err := EffortFor(e)
		if err != nil {
			t.Fatal(err)
		}
		if p.EvalChunks != 4 {
			t.Errorf("profiled (%q) chunks = %d, want 4", e, p.EvalChunks)
		}
		if p.EvalCtx != 2048 {
			t.Errorf("profiled (%q) ctx = %d, want 2048", e, p.EvalCtx)
		}
		if len(p.Candidates) != 0 {
			t.Errorf("profiled lattice = %v, want full (nil)", p.Candidates)
		}
		if p.Reconstruct {
			t.Errorf("profiled reconstruct = true, want false (Extra/CLI only)")
		}
		if !p.ScaleFold || !p.InPlaceReconstruct {
			t.Errorf("profiled fold/inplace = %v/%v, want true", p.ScaleFold, p.InPlaceReconstruct)
		}
		if !p.ProbeKLD || !p.SolverFTI {
			t.Errorf("profiled probeKLD/solverFTI = %v/%v, want true", p.ProbeKLD, p.SolverFTI)
		}
		if len(p.AnchorRecipes) != 3 || p.AnchorRecipes[0] != core.DTypeQ3_K_M ||
			p.AnchorRecipes[1] != core.DTypeQ3_K_L || p.AnchorRecipes[2] != core.DTypeQ4_K_M {
			t.Errorf("profiled recipes = %v", p.AnchorRecipes)
		}
	}

	deep, err := EffortFor(EffortDeep)
	if err != nil {
		t.Fatal(err)
	}
	if deep.EvalChunks != 8 {
		t.Errorf("deep chunks = %d, want 8", deep.EvalChunks)
	}
	if deep.EvalCtx != 4096 {
		t.Errorf("deep ctx = %d, want 4096", deep.EvalCtx)
	}
	if len(deep.Candidates) != 0 {
		t.Errorf("deep lattice = %v, want full (nil)", deep.Candidates)
	}
	if deep.Reconstruct {
		t.Errorf("deep reconstruct = true, want false (Extra/CLI only)")
	}
	if !deep.ScaleFold || !deep.InPlaceReconstruct {
		t.Errorf("deep fold/inplace = %v/%v, want true", deep.ScaleFold, deep.InPlaceReconstruct)
	}
	if !deep.ProbeKLD || !deep.SolverFTI {
		t.Errorf("deep probeKLD/solverFTI = %v/%v, want true", deep.ProbeKLD, deep.SolverFTI)
	}
	wantDeep := []core.DType{core.DTypeQ3_K_M, core.DTypeQ3_K_L, core.DTypeQ4_K_M, core.DType("Q4_K_L")}
	if len(deep.AnchorRecipes) != len(wantDeep) {
		t.Fatalf("deep recipes = %v", deep.AnchorRecipes)
	}
	for i, d := range wantDeep {
		if deep.AnchorRecipes[i] != d {
			t.Fatalf("deep recipes[%d] = %s, want %s", i, deep.AnchorRecipes[i], d)
		}
	}

	// Every preset shares the same high-precision reference gates (Q5 /
	// unset target): mean-KLD <= 0.15, p95 <= 1.0. Runtime defaults scale
	// with TargetBPW; see TestQualityGateThresholds.
	for _, e := range []Effort{EffortFast, EffortProfiled, EffortDeep} {
		p, _ := EffortFor(e)
		if len(p.Gates) != 2 {
			t.Fatalf("%s gates = %+v", e, p.Gates)
		}
		if p.Gates[0].Metric != core.MetricKLD || p.Gates[0].MaxDelta != 0.15 {
			t.Errorf("%s mean gate = %+v", e, p.Gates[0])
		}
		if p.Gates[1].Metric != core.MetricP95KLD || p.Gates[1].MaxAbsolute != 1.0 {
			t.Errorf("%s p95 gate = %+v", e, p.Gates[1])
		}
		for _, g := range p.Gates {
			if err := g.Validate(); err != nil {
				t.Errorf("%s gate invalid: %v", e, err)
			}
		}
	}

	if _, err := EffortFor("ludicrous"); err == nil {
		t.Error("unknown effort accepted")
	}
}

func TestPlanCtxSizeFromEffort(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.planEffort("ctxprof", "profiled", func(o *PlanOptions) { o.CtxSize = 0 })
	if r.Config.CtxSize != 2048 {
		t.Fatalf("profiled ctx = %d, want 2048", r.Config.CtxSize)
	}
	r2 := f.planEffort("ctxdeep", "deep", func(o *PlanOptions) { o.CtxSize = 0 })
	if r2.Config.CtxSize != 4096 {
		t.Fatalf("deep ctx = %d, want 4096", r2.Config.CtxSize)
	}
	r3 := f.planEffort("ctxfast", "fast", func(o *PlanOptions) { o.CtxSize = 0 })
	if r3.Config.CtxSize != 512 {
		t.Fatalf("fast ctx = %d, want 512", r3.Config.CtxSize)
	}
	r4 := f.planEffort("ctxexp", "deep", nil) // fixture default CtxSize 512
	if r4.Config.CtxSize != 512 {
		t.Fatalf("explicit ctx lost: %d", r4.Config.CtxSize)
	}
}

// planEffort is a fixture plan with an effort preset and optional explicit
// overrides.
func (f *fixture) planEffort(runID, effort string, tweak func(*PlanOptions)) *state.Run {
	f.t.Helper()
	opts := PlanOptions{
		SourcePath:      f.src,
		StateDir:        f.stateDir,
		OutputDir:       f.outDir,
		CalibrationDir:  f.calibDir,
		LlamaQuantize:   f.src,
		LlamaPerplexity: f.src,
		BudgetBytes:     f.budget,
		Threads:         2,
		CtxSize:         512,
		Effort:          effort,
		RunID:           runID,
		Now:             time.Now(),
		Stdout:          io.Discard,
	}
	if tweak != nil {
		tweak(&opts)
	}
	r, err := Plan(opts)
	if err != nil {
		f.t.Fatal(err)
	}
	return r
}

func TestPlanEffortFastKnobs(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.planEffort("fasty", "fast", nil)
	if r.Config.Effort != state.EffortFast {
		t.Fatalf("stored effort = %q", r.Config.Effort)
	}
	if r.Config.SearchEnabled || r.Config.MaxSearchIterations != 0 {
		t.Fatalf("fast search config = %v/%d", r.Config.SearchEnabled, r.Config.MaxSearchIterations)
	}
	if len(r.Config.Gates) != 0 || r.Config.GatesOptOut {
		t.Fatalf("fast gates = %+v optOut=%v (defaults resolve at engine time)", r.Config.Gates, r.Config.GatesOptOut)
	}
	e := f.engine(r)
	if e.Extra.Chunks != 2 {
		t.Fatalf("fast eval chunks = %d, want 2", e.Extra.Chunks)
	}
}

func TestPlanProfiledAndDeepDisableSearch(t *testing.T) {
	f := newFixture(t, 150000)
	for _, effort := range []string{"profiled", "deep"} {
		r := f.planEffort("nosearch-"+effort, effort, nil)
		if r.Config.SearchEnabled || r.Config.MaxSearchIterations != 0 {
			t.Fatalf("%s search config = %v/%d", effort, r.Config.SearchEnabled, r.Config.MaxSearchIterations)
		}
	}
}

func TestPlanExplicitFlagOverrides(t *testing.T) {
	f := newFixture(t, 150000)
	g, err := ParseGates("mean-kld=0.9")
	if err != nil {
		t.Fatal(err)
	}
	r := f.planEffort("override", "fast", func(o *PlanOptions) {
		o.Chunks = 9
		o.ChunksSet = true
		o.Gates = g
	})
	if len(r.Config.Gates) != 1 || r.Config.Gates[0].MaxDelta != 0.9 {
		t.Fatalf("explicit gates lost: %+v", r.Config.Gates)
	}
	e := f.engine(r)
	if e.Extra.Chunks != 9 {
		t.Fatalf("explicit chunks lost: %d", e.Extra.Chunks)
	}
	// Explicit zero chunks overrides the preset with "unlimited".
	r2 := f.planEffort("override0", "deep", func(o *PlanOptions) {
		o.Chunks = 0
		o.ChunksSet = true
	})
	e2 := f.engine(r2)
	if e2.Extra.Chunks != 0 {
		t.Fatalf("explicit chunks=0 lost: %d", e2.Extra.Chunks)
	}
}

// TestResumeStoredEffortZeroFlags: a reloaded checkpoint runs under its
// stored effort with no flags at all.
func TestResumeStoredEffortZeroFlags(t *testing.T) {
	f := newFixture(t, 150000)
	f.planEffort("fastrun", "fast", nil)
	store := state.Store{Dir: f.stateDir}
	r, err := store.Load("fastrun")
	if err != nil {
		t.Fatal(err)
	}
	if r.Config.Effort != state.EffortFast || r.Config.SearchEnabled {
		t.Fatalf("stored config = effort %q search %v", r.Config.Effort, r.Config.SearchEnabled)
	}
	e := f.engine(r)
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	done, err := store.Load("fastrun")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := done.NextStage(); ok {
		t.Fatalf("run incomplete: %v", done.Completed)
	}
	// Search was skipped by the preset: completed with no artifact.
	if a := done.Artifacts[core.StageSearch]; a != "" {
		t.Fatalf("fast run produced a search artifact: %s", a)
	}
	rep := f.readReport("fastrun")
	if rep.GatesConfigured != 2 || !rep.GatesPass {
		t.Fatalf("default gates not applied on zero-flag resume: %+v pass=%v", rep.Gates, rep.GatesPass)
	}
}

func TestProfiledAssembleClonesPrivatePayload(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.planEffort("priv", "profiled", nil)
	e := f.engine(r)
	e.StageLimit = 1
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	store := state.Store{Dir: f.stateDir}
	done, err := store.Load("priv")
	if err != nil {
		t.Fatal(err)
	}
	e2 := f.engine(done)
	if e2.Extra.PrivateSourcePath == "" {
		t.Fatal("profiled assemble did not record a job-private payload")
	}
	if samePath(e2.Extra.PrivateSourcePath, done.Config.SourcePath) {
		t.Fatal("private payload path is the library source")
	}
	if _, err := os.Stat(e2.Extra.PrivateSourcePath); err != nil {
		t.Fatalf("private payload missing: %v", err)
	}
	src, err := os.ReadFile(done.Config.SourcePath)
	if err != nil {
		t.Fatal(err)
	}
	clone, err := os.ReadFile(e2.Extra.PrivateSourcePath)
	if err != nil {
		t.Fatal(err)
	}
	if len(clone) != len(src) {
		t.Fatalf("clone size %d != source %d", len(clone), len(src))
	}
}

// TestGateDefaultsFromEffort: an empty gate list (no opt-out) resolves to the
// effort profile defaults and is enforced fail-closed.
func TestGateDefaultsFromEffort(t *testing.T) {
	f := newFixture(t, 150000)
	f.runner.candKLD = 0.2 // mean 0.2 > 0.15 default, p95 0.4 <= 1.0
	r := f.plan("gatedef", "")
	if err := f.engine(r).Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	rep := f.readReport("gatedef")
	if rep.GatesConfigured != 2 {
		t.Fatalf("gatesConfigured = %d, want 2 (effort defaults)", rep.GatesConfigured)
	}
	if rep.GatesPass {
		t.Fatal("gates passed despite mean-KLD 0.2 > 0.15 default")
	}
	mean, p95 := rep.Gates[0], rep.Gates[1]
	if !mean.Measured || mean.Pass || mean.MaxDelta != 0.15 {
		t.Errorf("mean gate = %+v", mean)
	}
	if !p95.Measured || !p95.Pass || p95.MaxAbsolute != 1.0 {
		t.Errorf("p95 gate = %+v", p95)
	}
}

// TestDeepFinalValidation: when evaluate already recorded KLD+p95, deep emit
// skips the extra pass and still enforces gates against BestProfileID.
func TestDeepFinalValidation(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.planEffort("deepv", "deep", nil)
	e := f.engine(r)
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	store := state.Store{Dir: f.stateDir}
	done, err := store.Load("deepv")
	if err != nil {
		t.Fatal(err)
	}
	finalID := done.BestProfileID + "+final"
	if hasMeasurementPub(done, finalID, core.MetricKLD) {
		t.Fatal("deep extra pass ran despite evaluate KLD+p95 already recorded")
	}
	if !hasMeasurementPub(done, done.BestProfileID, core.MetricKLD) {
		t.Fatal("best profile KLD missing")
	}
	if !hasMeasurementPub(done, done.BestProfileID, core.MetricP95KLD) {
		t.Fatal("best profile p95 missing")
	}
	rep := f.readReport("deepv")
	if rep.GatesConfigured != 2 || !rep.GatesPass {
		t.Fatalf("deep report gates = %+v pass=%v", rep.Gates, rep.GatesPass)
	}
	for _, g := range rep.Gates {
		if !g.Measured || g.Value != 0.0125 && g.Value != 0.025 {
			t.Errorf("gate not evaluated against best profile: %+v", g)
		}
	}

	// Idempotency: forget the emit completion and resume; the skipped extra
	// pass must stay skipped.
	pplRuns := f.runner.pplRuns
	r2, err := store.Load("deepv")
	if err != nil {
		t.Fatal(err)
	}
	r2.Completed = r2.Completed[:len(r2.Completed)-1] // drop emit
	if err := store.Save(r2); err != nil {
		t.Fatal(err)
	}
	if err := f.engine(r2).Resume(context.Background()); err != nil {
		t.Fatalf("resume after deep validation failed: %v", err)
	}
	if f.runner.pplRuns != pplRuns {
		t.Fatalf("deep final validation re-ran on resume: %d -> %d", pplRuns, f.runner.pplRuns)
	}
	done2, err := store.Load("deepv")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := done2.NextStage(); ok {
		t.Fatalf("run incomplete after healing resume: %v", done2.Completed)
	}
}

// TestDeepFinalValidationGateFailure: a candidate breaching the default
// gates is still published; the report records GatesPass=false.
func TestDeepFinalValidationGateFailure(t *testing.T) {
	f := newFixture(t, 150000)
	f.runner.candKLD = 0.2 // mean 0.2 > 0.15 default gate
	r := f.planEffort("deepfail", "deep", nil)
	if err := f.engine(r).Resume(context.Background()); err != nil {
		t.Fatalf("deep validation must still emit: %v", err)
	}
	rep := f.readReport("deepfail")
	if rep.GatesPass {
		t.Fatal("gates passed despite mean-KLD 0.2 > 0.15")
	}
	for _, name := range []string{"tiny-deepfail.gguf", "tiny-deepfail.oid-plan.json", "tiny-deepfail.report.json"} {
		if _, statErr := os.Stat(filepath.Join(f.outDir, name)); statErr != nil {
			t.Errorf("%s missing after failed validation: %v", name, statErr)
		}
	}
}

func TestQualityGateThresholds(t *testing.T) {
	cases := []struct {
		bpw       float64
		mean, p95 float64
	}{
		{0, 0.15, 1.0},
		{-1, 0.15, 1.0},
		{5.5, 0.15, 1.0},
		{4.5, 0.40, 3.0},
		{3.5, 2.00, 16.0},
		{2.5, 4.00, 32.0},
		{8.0, 0.05, 0.40},
		{1.5, 8.00, 64.0},
		{1.0, 8.00, 64.0},
	}
	for _, tc := range cases {
		mean, p95 := QualityGateThresholds(tc.bpw)
		if mean != tc.mean || p95 != tc.p95 {
			t.Errorf("bpw %.2f: mean/p95 = %.4f/%.4f, want %.4f/%.4f",
				tc.bpw, mean, p95, tc.mean, tc.p95)
		}
	}
	midMean, midP95 := QualityGateThresholds(4.0)
	if midMean <= 0.40 || midMean >= 2.00 || midP95 <= 3.0 || midP95 >= 16.0 {
		t.Errorf("4.0 bpw interpolated to %.4f / %.4f, want between Q4 and Q3 knots", midMean, midP95)
	}
}

func TestResolvedGatesScaleWithTargetBPW(t *testing.T) {
	e := &Engine{Run: &state.Run{Config: state.RunConfig{TargetBPW: 3.5}}}
	gates := e.resolvedGates()
	if len(gates) != 2 {
		t.Fatalf("q3 gates = %+v", gates)
	}
	if gates[0].Metric != core.MetricKLD || gates[0].MaxDelta != 2.0 {
		t.Errorf("q3 mean gate = %+v, want maxDelta 2.0", gates[0])
	}
	if gates[1].Metric != core.MetricP95KLD || gates[1].MaxAbsolute != 16.0 {
		t.Errorf("q3 p95 gate = %+v, want maxAbsolute 16.0", gates[1])
	}

	unset := (&Engine{Run: &state.Run{Config: state.RunConfig{}}}).resolvedGates()
	if unset[0].MaxDelta != 0.15 || unset[1].MaxAbsolute != 1.0 {
		t.Errorf("unset-target gates = %+v, want 0.15 / 1.0", unset)
	}
}

func TestResolvedGatesScaleDomainHoldout(t *testing.T) {
	dir := t.TempDir()
	code := filepath.Join(dir, "evaluation-code.txt")
	prose := filepath.Join(dir, "evaluation-prose.txt")
	payload := bytes.Repeat([]byte("x"), int(minPerplexityCorpusBytes(4096)))
	for _, path := range []string{code, prose} {
		if err := os.WriteFile(path, payload, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	e := &Engine{Run: &state.Run{Config: state.RunConfig{
		TargetBPW:         3.5,
		CtxSize:           4096,
		EvalCorpus:        filepath.Join(dir, "evaluation.txt"),
		DomainEvalCorpora: map[string]string{"code": code, "prose": prose},
	}}}
	gates := e.resolvedGates()
	if len(gates) != 3 {
		t.Fatalf("q3+domains gates = %+v", gates)
	}
	dom := gates[2]
	want := worstDomainGateDelta(2.0)
	if dom.Metric != core.MetricWorstDomainKLD || dom.MaxDelta != want {
		t.Errorf("q3 domain gate = %+v, want maxDelta %v", dom, want)
	}
}

func TestResolvedGatesOmitUnusableDomainHoldouts(t *testing.T) {
	dir := t.TempDir()
	code := filepath.Join(dir, "evaluation-code.txt")
	prose := filepath.Join(dir, "evaluation-prose.txt")
	for _, path := range []string{code, prose} {
		if err := os.WriteFile(path, []byte("tiny"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	e := &Engine{Run: &state.Run{Config: state.RunConfig{
		TargetBPW:         3.5,
		CtxSize:           4096,
		EvalCorpus:        filepath.Join(dir, "evaluation.txt"),
		DomainEvalCorpora: map[string]string{"code": code, "prose": prose},
	}}}
	gates := e.resolvedGates()
	if len(gates) != 2 {
		t.Fatalf("unusable domain holdouts still added a worst-domain gate: %+v", gates)
	}
}

func TestQ3DefaultGatesAcceptTypicalDivergence(t *testing.T) {
	f := newFixture(t, 150000)
	f.runner.candKLD = 1.23 // ~current Deep Q3; fake p95 is 2.46
	r := f.planEffort("q3ok", "fast", func(o *PlanOptions) { o.TargetBPW = 3.5 })
	if err := f.engine(r).Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	rep := f.readReport("q3ok")
	if !rep.GatesPass {
		t.Fatalf("typical Q3 KLD failed Q3 gates: %+v", rep.Gates)
	}
	if rep.Gates[0].MaxDelta != 2.0 || rep.Gates[1].MaxAbsolute != 16.0 {
		t.Errorf("q3 report gates = %+v", rep.Gates)
	}
}

func TestQ3DefaultGatesRejectBrokenDivergence(t *testing.T) {
	f := newFixture(t, 150000)
	f.runner.candKLD = 3.3 // sensitivity-probe dump, not a finished Q3
	r := f.planEffort("q3bad", "fast", func(o *PlanOptions) { o.TargetBPW = 3.5 })
	if err := f.engine(r).Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	rep := f.readReport("q3bad")
	if rep.GatesPass {
		t.Fatal("broken 3.3 mean KLD passed Q3 gates")
	}
	if rep.Gates[0].Pass {
		t.Errorf("mean gate passed: %+v", rep.Gates[0])
	}
}
