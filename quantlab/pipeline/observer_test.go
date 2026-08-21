package pipeline

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"quantlab/core"
	"quantlab/state"
)

// obsEvent records one observer callback in order.
type obsEvent struct {
	kind     string // started | progress | completed | measurement
	stage    core.Stage
	progress float64
	message  string
	artifact string
	m        core.Measurement
}

type recordingObserver struct {
	events []obsEvent
}

func (o *recordingObserver) StageStarted(s core.Stage) {
	o.events = append(o.events, obsEvent{kind: "started", stage: s})
}

func (o *recordingObserver) StageProgress(s core.Stage, p float64, msg string) {
	o.events = append(o.events, obsEvent{kind: "progress", stage: s, progress: p, message: msg})
}

func (o *recordingObserver) StageCompleted(s core.Stage, artifact string) {
	o.events = append(o.events, obsEvent{kind: "completed", stage: s, artifact: artifact})
}

func (o *recordingObserver) Measurement(m core.Measurement) {
	o.events = append(o.events, obsEvent{kind: "measurement", m: m})
}

func (o *recordingObserver) filter(kind string) []obsEvent {
	var out []obsEvent
	for _, ev := range o.events {
		if ev.kind == kind {
			out = append(out, ev)
		}
	}
	return out
}

// TestObserverEventOrdering drives a full remaining-stage run and verifies
// the event contract: stages start and complete in canonical order with
// artifacts, evaluate measurements stream, and progress values stay in [0,1].
func TestObserverEventOrdering(t *testing.T) {
	f := newFixture(t, 95000)
	r := f.plan("obs", "mean-kld=1.0")
	store := state.Store{Dir: f.stateDir}
	e := f.engine(r)
	e.StageLimit = 3
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	r1, err := store.Load("obs")
	if err != nil {
		t.Fatal(err)
	}
	r1.Config.BudgetBytes = r1.Manifest.TotalBytes + 9000
	if err := store.Save(r1); err != nil {
		t.Fatal(err)
	}

	obs := &recordingObserver{}
	e = f.engine(r1)
	e.Obs = obs
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}

	// Stage boundaries: exactly the remaining stages, in canonical order,
	// each started before it completes.
	started := obs.filter("started")
	completed := obs.filter("completed")
	want := []core.Stage{core.StageQuantize, core.StageEvaluate, core.StageSearch, core.StageEmit}
	if len(started) != len(want) || len(completed) != len(want) {
		t.Fatalf("started=%v completed=%v", started, completed)
	}
	pos := map[core.Stage]int{}
	for i, ev := range obs.events {
		if ev.kind == "started" {
			pos[ev.stage] = i
		}
	}
	for i, s := range want {
		if started[i].stage != s {
			t.Fatalf("started[%d] = %s, want %s", i, started[i].stage, s)
		}
		if completed[i].stage != s {
			t.Fatalf("completed[%d] = %s, want %s", i, completed[i].stage, s)
		}
		if pos[s] > indexOf(obs.events, "completed", s) {
			t.Fatalf("stage %s completed before it started", s)
		}
	}
	// Artifacts ride the completed events. Search is a no-op skip for
	// checkpoint compatibility and has no artifact.
	for _, s := range []core.Stage{core.StageQuantize, core.StageEvaluate, core.StageEmit} {
		ev := completed[indexOfCompleted(completed, s)]
		if ev.artifact == "" {
			t.Errorf("completed(%s) carried no artifact", s)
		}
	}

	// Measurement stream: baseline ppl and candidate metrics from evaluate.
	ms := obs.filter("measurement")
	if len(ms) < 3 {
		t.Fatalf("measurement events = %d, want >= 3", len(ms))
	}
	var sawBaseline, sawCandKLD bool
	for _, ev := range ms {
		if ev.m.ProfileID == "baseline" && ev.m.Metric == core.MetricPerplexity {
			sawBaseline = true
		}
		if ev.m.Metric == core.MetricKLD {
			sawCandKLD = true
		}
	}
	if !sawBaseline || !sawCandKLD {
		t.Errorf("missing evaluate-stage measurement events: baseline=%v kld=%v", sawBaseline, sawCandKLD)
	}

	// Progress events are bounded fractions.
	for _, ev := range obs.filter("progress") {
		if ev.progress < 0 || ev.progress > 1 {
			t.Errorf("progress out of range: %+v", ev)
		}
		if ev.message == "" {
			t.Errorf("progress without message: %+v", ev)
		}
	}
}

func indexOf(events []obsEvent, kind string, s core.Stage) int {
	for i, ev := range events {
		if ev.kind == kind && ev.stage == s {
			return i
		}
	}
	return -1
}

func indexOfCompleted(completed []obsEvent, s core.Stage) int {
	for i, ev := range completed {
		if ev.stage == s {
			return i
		}
	}
	return -1
}

// TestObserverNilCompatibility: an engine without an observer behaves
// exactly as before (the whole existing suite runs nil; this pins it).
func TestObserverNilCompatibility(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.plan("nilobs", "mean-kld=0.5")
	e := f.engine(r)
	if e.Obs != nil {
		t.Fatal("engine observer defaults to non-nil")
	}
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, ok := r.NextStage(); ok {
		t.Fatalf("run incomplete: %v", r.Completed)
	}
}

type durationObserver struct {
	recordingObserver
	stage    core.Stage
	duration time.Duration
	calls    int
}

func (o *durationObserver) StageDuration(stage core.Stage, d time.Duration) {
	o.stage = stage
	o.duration = d
	o.calls++
}

func TestWrapTimerNil(t *testing.T) {
	if WrapTimer(nil) != nil {
		t.Fatal("WrapTimer(nil) should stay nil")
	}
}

func TestWrapTimerForwardsStageDuration(t *testing.T) {
	inner := &durationObserver{}
	obs := WrapTimer(inner)
	obs.StageStarted(core.StageQuantize)
	time.Sleep(2 * time.Millisecond)
	obs.StageCompleted(core.StageQuantize, "candidate.gguf")
	if inner.calls != 1 {
		t.Fatalf("StageDuration calls = %d, want 1", inner.calls)
	}
	if inner.stage != core.StageQuantize {
		t.Fatalf("stage = %s, want %s", inner.stage, core.StageQuantize)
	}
	if inner.duration < 0 {
		t.Fatalf("duration %v", inner.duration)
	}
	if len(inner.filter("started")) != 1 || len(inner.filter("completed")) != 1 {
		t.Fatalf("did not forward start/complete: %+v", inner.events)
	}
	if inner.filter("completed")[0].artifact != "candidate.gguf" {
		t.Fatalf("artifact = %q", inner.filter("completed")[0].artifact)
	}
}

func TestWrapTimerIdempotent(t *testing.T) {
	inner := &durationObserver{}
	once := WrapTimer(inner)
	twice := WrapTimer(once)
	if once != twice {
		t.Fatal("WrapTimer double-wraps an already timed observer")
	}
}

func TestWrapTimerWithoutStageTimer(t *testing.T) {
	inner := &recordingObserver{}
	obs := WrapTimer(inner)
	obs.StageStarted(core.StageEmit)
	obs.StageProgress(core.StageEmit, 0.5, "emit")
	obs.StageCompleted(core.StageEmit, "out.gguf")
	obs.Measurement(core.Measurement{ProfileID: "p"})
	if len(inner.filter("started")) != 1 || len(inner.filter("completed")) != 1 {
		t.Fatalf("forwarding broken: %+v", inner.events)
	}
	if len(inner.filter("progress")) != 1 || len(inner.filter("measurement")) != 1 {
		t.Fatalf("progress/measurement not forwarded: %+v", inner.events)
	}
}

func TestSubsetFirstTrimsAllDTypes(t *testing.T) {
	cases := []struct {
		name   string
		dt     core.DType
		keep   map[string]struct{}
		needIM bool
	}{
		{"IQ2_S", core.DTypeIQ2_S, map[string]struct{}{"blk.0.attn_q.weight": {}}, true},
		{"Q4_K", core.DTypeQ4_K_T, map[string]struct{}{"blk.0.attn_q.weight": {}}, false},
		{"Q8_0", core.DTypeQ8_0, map[string]struct{}{"blk.0.attn_q.weight": {}, "blk.0.ffn_down.weight": {}}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFixture(t, 150000)
			r := f.plan("subset-"+tc.name, "mean-kld=1.0")
			if tc.needIM {
				im := filepath.Join(f.dir, "im.bin")
				if err := os.WriteFile(im, []byte("imatrix"), 0o644); err != nil {
					t.Fatal(err)
				}
				r.Config.ImatrixPath = im
			}
			store := state.Store{Dir: f.stateDir}
			if err := store.Save(r); err != nil {
				t.Fatal(err)
			}
			e := f.rawEngine(r)
			e.StageLimit = 1
			if err := e.Resume(context.Background()); err != nil {
				t.Fatal(err)
			}
			r1, err := store.Load("subset-" + tc.name)
			if err != nil {
				t.Fatal(err)
			}
			if tc.needIM {
				r1.Config.ImatrixPath = r.Config.ImatrixPath
			}
			var opts []core.TensorOption
			for name := range tc.keep {
				elems := uint64(256 * 256)
				if name == "blk.0.ffn_down.weight" {
					elems = 512 * 256
				}
				b, ok := tc.dt.ExactBytes(elems)
				if !ok {
					t.Fatal("geometry")
				}
				opts = append(opts, core.TensorOption{TensorName: name, Target: tc.dt, Bytes: b})
			}
			r1.Manifest = &core.SelectionManifest{ProfileID: "subset-" + tc.name, Options: opts}
			if err := store.Save(r1); err != nil {
				t.Fatal(err)
			}
			srcTensors := countGGUFTensors(t, f.src)
			if srcTensors < 2 {
				t.Fatalf("source tensor count %d, want a full model", srcTensors)
			}
			e = f.rawEngine(r1)
			keepFor := func(core.DType) (map[string]struct{}, error) { return tc.keep, nil }
			if err := e.runTrimmedAnchorJobs(context.Background(), []core.DType{tc.dt}, keepFor, e.variantsDir(), "meta.json"); err != nil {
				t.Fatal(err)
			}
			if f.runner.lastQuantType != tc.dt.PureFType() {
				t.Fatalf("quantized %s, want %s", f.runner.lastQuantType, tc.dt.PureFType())
			}
			if f.runner.lastQuantInTensors != len(tc.keep) {
				t.Fatalf("%s quantize saw %d tensors, want %d (trimmed subset, not the %d-tensor source)",
					tc.dt, f.runner.lastQuantInTensors, len(tc.keep), srcTensors)
			}
		})
	}
}

func TestEmptyKeepSetErrors(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.plan("emptykeep", "mean-kld=1.0")
	store := state.Store{Dir: f.stateDir}
	e := f.rawEngine(r)
	e.StageLimit = 1
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	r1, err := store.Load("emptykeep")
	if err != nil {
		t.Fatal(err)
	}
	r1.Manifest = &core.SelectionManifest{ProfileID: "emptykeep"}
	if err := store.Save(r1); err != nil {
		t.Fatal(err)
	}
	e = f.rawEngine(r1)
	keepFor := func(core.DType) (map[string]struct{}, error) { return map[string]struct{}{}, nil }
	err = e.runTrimmedAnchorJobs(context.Background(), []core.DType{core.DTypeQ8_0}, keepFor, e.variantsDir(), "meta.json")
	if err == nil || !strings.Contains(err.Error(), "no tensors to keep") {
		t.Fatalf("want empty keep error, got %v", err)
	}
}
