package pipeline

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"quantlab/anchor"
	"quantlab/core"
	"quantlab/orchestrate"
	"quantlab/tensorbank"
)

// mixKLD models a candidate whose attention damage compounds with FFN
// damage: isolated probes (everything else Q8_0) see attention as cheap,
// but inside an aggressive mix its marginal cost is several times larger.
// That is exactly the gap in-context refinement measures and corrects.
func mixKLD(t *testing.T) func(path string) (float64, bool) {
	return func(path string) (float64, bool) {
		s, err := tensorbank.OpenSource(path)
		if err != nil {
			return 0, false
		}
		defer s.Close()
		f, err := tensorbank.Parse(s)
		if err != nil {
			return 0, false
		}
		errOf := func(d core.DType) float64 {
			bpw, ok := d.BitsPerWeight()
			if !ok || d.IsFloat() {
				return 0
			}
			return math.Pow(2, -1.2*bpw)
		}
		var attn, ffn float64
		for _, ti := range f.Tensors {
			switch {
			case strings.Contains(ti.Name, "attn_q"):
				attn += errOf(ti.DType)
			case strings.Contains(ti.Name, "ffn_down"):
				ffn += errOf(ti.DType)
			}
		}
		return 0.001 + 0.1*attn + 8*ffn + 150*attn*ffn, true
	}
}

func refineFixture(t *testing.T, withSearch bool) (*fixture, *Engine) {
	t.Helper()
	f := newFixture(t, 520000)
	var tensors []gtensor
	for l := 0; l < 4; l++ {
		tensors = append(tensors,
			gtensor{fmt.Sprintf("blk.%d.attn_q.weight", l), core.DTypeF16, []uint64{256, 256}},
			gtensor{fmt.Sprintf("blk.%d.ffn_down.weight", l), core.DTypeF16, []uint64{512, 256}},
			gtensor{fmt.Sprintf("blk.%d.attn_norm.weight", l), core.DTypeF32, []uint64{256}})
	}
	if err := writeGGUF(f.src, tensors); err != nil {
		t.Fatal(err)
	}
	finiteF16Payloads(t, f.src)
	f.runner.kldForModel = mixKLD(t)
	f.runner.noP95 = false
	r := f.planEffort("refine", "profiled", func(o *PlanOptions) {
		o.Gates = nil
		o.GatesOptOut = true
	})
	searchPath := filepath.Join(filepath.Dir(r.Config.EvalCorpus), "search.txt")
	if withSearch {
		if err := os.WriteFile(searchPath, []byte("search holdout text for tuning only\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	} else {
		// Plan reserves a holdout when refinement is on; simulate a
		// calibration dir that has none.
		os.Remove(searchPath)
		if r.Config.SearchCorpus != "" {
			os.Remove(r.Config.SearchCorpus)
		}
		r.Config.SearchCorpus = ""
	}
	e := f.engine(r)
	var log bytes.Buffer
	e.Out = &log
	t.Cleanup(func() {
		if t.Failed() {
			t.Log(log.String())
		}
	})
	return f, e
}

func TestRefineMeasuresInContextAndKeepsOnlyImprovements(t *testing.T) {
	f, e := refineFixture(t, true)
	e.StageLimit = 6 // through search
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(e.refinePath())
	if err != nil {
		t.Fatalf("refine.json: %v", err)
	}
	var st refineState
	if err := json.Unmarshal(data, &st); err != nil {
		t.Fatal(err)
	}
	if !st.Done || len(st.Rounds) == 0 {
		t.Fatalf("refine state = %+v", st)
	}
	r0 := st.Rounds[0]
	for i, rr := range st.Rounds {
		t.Logf("round %d: base %s %.5f lambdas %v def %.2f resolved %s %.5f accepted %v %s",
			i+1, rr.BaseProfile, rr.BaseValue, rr.Lambdas, rr.DefaultScale, rr.Resolved, rr.ResolvedVal, rr.Accepted, rr.Note)
	}
	if len(r0.Variants) == 0 {
		t.Fatalf("no in-context variants measured: %+v", r0)
	}
	// The compounding interaction makes attention costlier in the mix
	// than its isolated probe predicted, relative to the FFN.
	if r0.Lambdas["attn_q"] <= r0.Lambdas["ffn_down"] && r0.Lambdas["ffn_down"] > 0 {
		t.Errorf("lambdas = %v: want attn_q corrected above ffn_down", r0.Lambdas)
	}
	// Refinement chose on the search holdout only.
	evalAbs, _ := filepath.Abs(e.Run.Config.EvalCorpus)
	refineEvals := 0
	for i, c := range f.runner.corpora {
		if !strings.Contains(c, "search.txt") {
			continue
		}
		refineEvals++
		_ = i
	}
	if refineEvals == 0 {
		t.Error("no evaluation ran on the search holdout")
	}
	if r0.Accepted {
		if e.Run.BestProfileID != st.Final || e.Run.Manifest.ProfileID != st.Final {
			t.Errorf("accepted %s but run holds best=%s manifest=%s", st.Final, e.Run.BestProfileID, e.Run.Manifest.ProfileID)
		}
		if got := e.Run.Artifacts[core.StageQuantize]; got != e.refinedCandidatePath(st.Final) {
			t.Errorf("candidate artifact = %s, want the refined candidate", got)
		}
		caps, err := e.caps(context.Background(), orchestrate.ToolPerplexity)
		if err != nil {
			t.Fatal(err)
		}
		evalCfg := orchestrate.EvalConfig{CorpusPath: evalAbs, CtxSize: e.Run.Config.CtxSize,
			Chunks: e.Extra.Chunks, Threads: e.Run.Config.Threads, NGPULayers: -1}
		if _, ok := e.measurementForEval(st.Final, core.MetricKLD, evalCfg, caps); !ok {
			if _, ok := e.measurement(st.Final, core.MetricKLD); !ok {
				t.Fatal("accepted profile was not scored")
			}
			t.Error("accepted profile was scored, but not on the evaluation corpus")
		}
		if !(r0.ResolvedVal < r0.BaseValue) {
			t.Errorf("accepted a non-improvement: %v vs %v", r0.ResolvedVal, r0.BaseValue)
		}
		if e.Run.Manifest.TotalBytes > e.solveBudget() {
			t.Errorf("refined manifest %d bytes over budget %d", e.Run.Manifest.TotalBytes, e.solveBudget())
		}
	} else if e.Run.BestProfileID != r0.BaseProfile {
		t.Errorf("rejected refinement but best changed to %s", e.Run.BestProfileID)
	}

	// Resume re-running search must not remeasure anything.
	before := len(f.runner.evaluated)
	e.Run.Completed = e.Run.Completed[:len(e.Run.Completed)-1]
	e2 := f.engine(e.Run)
	e2.StageLimit = 1
	if err := e2.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := len(f.runner.evaluated); got != before {
		t.Errorf("resumed search re-ran %d evaluations", got-before)
	}
}

func TestRefineSkipsWithoutSearchHoldout(t *testing.T) {
	_, e := refineFixture(t, false)
	e.StageLimit = 6
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(e.refinePath()); !os.IsNotExist(err) {
		t.Errorf("refine ran without a distinct search holdout (state %v)", err)
	}
}

func TestNoRefineOptOut(t *testing.T) {
	_, e := refineFixture(t, true)
	e.Extra.NoRefine = true
	e.StageLimit = 6
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(e.refinePath()); !os.IsNotExist(err) {
		t.Error("NoRefine still ran refinement")
	}
}

// TestRefineInstallsAcceptedProfile drives the install path: a recorded,
// accepted round is re-solved, assembled, checkpointed and scored on the
// evaluation corpus; emit then publishes the refined artifact.
func TestRefineInstallsAcceptedProfile(t *testing.T) {
	f, e := refineFixture(t, true)
	e.StageLimit = 5 // through evaluate
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	solved, err := e.loadSolveArtifact()
	if err != nil {
		t.Fatal(err)
	}
	set, err := e.deriveAnchors(e.Run.Bank)
	if err != nil {
		t.Fatal(err)
	}
	table := e.loadExactLoss(e.Run.Bank)
	lambdas := map[string]float64{"attn_q": refineLambdaMax, "ffn_down": refineLambdaMin}
	next, err := e.refineSolve(e.Run.Bank, set, solved.Sensitivity.Scaled(lambdas, 1), table, solved.Candidates)
	if err != nil {
		t.Fatal(err)
	}
	base := e.Run.BestProfileID
	if next.ID == base {
		t.Fatal("fixture: a 16x attention/FFN correction should move bytes")
	}
	evalCfg, ok := e.refineEvalConfig()
	if !ok {
		t.Fatal("no search holdout")
	}
	sig, err := e.refineSignature(solved, evalCfg)
	if err != nil {
		t.Fatal(err)
	}
	st := &refineState{Version: refineStateVersion, Signature: sig, Measured: map[string]refineMeasure{},
		Done: true, Final: next.ID,
		Rounds: []refineRound{{BaseProfile: base, Lambdas: lambdas, DefaultScale: 1, Resolved: next.ID, Accepted: true}}}
	if err := e.writeJSON(e.refinePath(), st); err != nil {
		t.Fatal(err)
	}
	e2 := f.engine(e.Run)
	e2.StageLimit = 2 // search, emit
	if err := e2.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if e2.Run.BestProfileID != next.ID || e2.Run.Manifest.ProfileID != next.ID {
		t.Fatalf("installed best=%s manifest=%s, want %s", e2.Run.BestProfileID, e2.Run.Manifest.ProfileID, next.ID)
	}
	if _, ok := e2.measurement(next.ID, core.MetricKLD); !ok {
		t.Error("installed profile was not scored")
	}
	// The refined plan spends more on attention than the solved one.
	rank := func(m *core.SelectionManifest, name string) int {
		for _, o := range m.Options {
			if o.TensorName == name {
				return anchor.Rank(o.Target)
			}
		}
		return -2
	}
	if !(rank(e2.Run.Manifest, "blk.0.attn_q.weight") < rank(solved.Manifest, "blk.0.attn_q.weight")) &&
		!(rank(e2.Run.Manifest, "blk.0.ffn_down.weight") > rank(solved.Manifest, "blk.0.ffn_down.weight")) {
		t.Errorf("refined plan did not shift bytes toward attention")
	}
	if _, more := e2.Run.NextStage(); more {
		t.Fatal("run not complete after emit")
	}
	if _, err := os.Stat(e2.refinedCandidatePath(next.ID)); !os.IsNotExist(err) {
		t.Error("refined candidate scratch left after emit")
	}
}

func TestPlanReservesSearchHoldoutForRefinement(t *testing.T) {
	f := newFixture(t, 520000)
	var docs strings.Builder
	for i := 0; i < 200; i++ {
		fmt.Fprintf(&docs, "record %d: the quick brown fox %d jumps over the lazy dog.\n\n", i, i*7)
	}
	if err := os.WriteFile(filepath.Join(f.calibDir, "docs.txt"), []byte(docs.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	r := f.planEffort("hold", "profiled", nil)
	e := f.engine(r)
	if _, ok := e.searchCorpusPath(); !ok {
		t.Fatalf("profiled plan built no search holdout (search=%q eval=%q)", r.Config.SearchCorpus, r.Config.EvalCorpus)
	}

	// Too few distinct records for a holdout: the plan still succeeds
	// without one (historical split), and refinement will skip.
	g := newFixture(t, 520000)
	if err := os.WriteFile(filepath.Join(g.calibDir, "docs.txt"),
		[]byte(strings.Repeat("alpha beta gamma delta.\n\nepsilon zeta eta theta.\n\n", 40)), 0o644); err != nil {
		t.Fatal(err)
	}
	r2 := g.planEffort("tiny", "profiled", nil)
	if _, ok := g.engine(r2).searchCorpusPath(); ok {
		t.Error("two distinct records still produced a holdout; fallback not exercised")
	}
}

// acceptedRefineState records one accepted round whose re-solve moves bytes
// toward attention, after the run has completed evaluate.
func acceptedRefineState(t *testing.T, e *Engine, done bool) string {
	t.Helper()
	solved, err := e.loadSolveArtifact()
	if err != nil {
		t.Fatal(err)
	}
	set, err := e.deriveAnchors(e.Run.Bank)
	if err != nil {
		t.Fatal(err)
	}
	lambdas := map[string]float64{"attn_q": refineLambdaMax, "ffn_down": refineLambdaMin}
	next, err := e.refineSolve(e.Run.Bank, set, solved.Sensitivity.Scaled(lambdas, 1), e.loadExactLoss(e.Run.Bank), solved.Candidates)
	if err != nil {
		t.Fatal(err)
	}
	if next.ID == e.Run.BestProfileID {
		t.Fatal("fixture: correction should move bytes")
	}
	evalCfg, _ := e.refineEvalConfig()
	sig, err := e.refineSignature(solved, evalCfg)
	if err != nil {
		t.Fatal(err)
	}
	st := &refineState{Version: refineStateVersion, Signature: sig, Measured: map[string]refineMeasure{},
		Rounds: []refineRound{{BaseProfile: e.Run.BestProfileID, Lambdas: lambdas, DefaultScale: 1, Resolved: next.ID, Accepted: true}}}
	if done {
		st.Done, st.Final = true, next.ID
	}
	if err := e.writeJSON(e.refinePath(), st); err != nil {
		t.Fatal(err)
	}
	return next.ID
}

// A failure while scoring an installed refined profile must not fail open:
// the search stage stays incomplete and a resume finishes the scoring.
func TestRefineInstalledScoringFailurePropagates(t *testing.T) {
	f, e := refineFixture(t, true)
	e.StageLimit = 5
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	nextID := acceptedRefineState(t, e, true)
	f.runner.evalErr = func(model, corpus string) error {
		if strings.Contains(model, "candidate-refined") && strings.Contains(corpus, "evaluation") {
			return fmt.Errorf("llama-perplexity crashed")
		}
		return nil
	}
	e2 := f.engine(e.Run)
	e2.StageLimit = 1
	if err := e2.Resume(context.Background()); err == nil {
		t.Fatal("scoring failure after install was swallowed")
	}
	store := e2.Store
	saved, err := store.Load(e2.Run.RunID)
	if err != nil {
		t.Fatal(err)
	}
	if saved.BestProfileID != nextID {
		t.Fatalf("checkpoint best = %s, want installed %s", saved.BestProfileID, nextID)
	}
	if next, _ := saved.NextStage(); next != core.StageSearch {
		t.Fatalf("next stage = %s, want search to be retried", next)
	}
	f.runner.evalErr = nil
	e3 := f.engine(saved)
	e3.StageLimit = 1
	if err := e3.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, ok := e3.measurement(nextID, core.MetricKLD); !ok {
		t.Error("installed profile still unscored after resume")
	}
}

// Resuming between rounds continues from the last accepted re-solve, not
// from the manifest (which holds the pre-refinement profile until the end).
func TestRefineResumeBetweenRoundsContinuesFromAccepted(t *testing.T) {
	f, e := refineFixture(t, true)
	e.StageLimit = 5
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	nextID := acceptedRefineState(t, e, false) // 1 round recorded, not done
	e2 := f.engine(e.Run)
	e2.StageLimit = 1
	if err := e2.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if e2.Run.BestProfileID != nextID {
		t.Fatalf("best = %s, want the accepted re-solve %s", e2.Run.BestProfileID, nextID)
	}
}
