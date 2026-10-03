package workflow

import (
	"context"
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

type cacheRunner struct {
	*fakeRunner
	identity string
}

func (f *cacheRunner) WorkflowCacheIdentity(string, *mediagen.LoadOverrides) (string, error) {
	return f.identity, nil
}

func TestHistorySurvivesRestartAndFreezesGraph(t *testing.T) {
	store := newStore(t)
	runner := &cacheRunner{fakeRunner: &fakeRunner{dir: t.TempDir()}, identity: "runtime-a"}
	e := NewPersistentExecutor(runner, nil, store.db)
	g := smallTxt2img()
	g.Nodes[4].Params["seed"] = -1
	v, err := e.SubmitWithOptions(planFor(t, g), RunOptions{Graph: &g, WorkflowID: "workflow-a"})
	if err != nil {
		t.Fatal(err)
	}
	g.Nodes[1].Params["text"] = "edited after submission"
	v = waitRun(t, e, v.ID, RunComplete)
	if v.Graph == nil || v.Graph.Nodes[1].Params["text"] == "edited after submission" || v.Nodes["n5"].Seed == nil || *v.Nodes["n5"].Seed < 1 {
		t.Fatalf("snapshot lost graph or seed: %+v", v)
	}
	if v.SourceGraph == nil || v.SourceGraph.Nodes[4].Params["seed"] != float64(-1) || v.Graph.Nodes[4].Params["seed"] != float64(*v.Nodes["n5"].Seed) {
		t.Fatal("source graph must retain the submitted seed while graph records the resolved seed")
	}
	// Callers may inspect or edit returned snapshots without mutating history.
	*v.Nodes["n5"].Seed = -1
	v.Nodes["n5"].Outputs[0] = "changed.png"
	frozen, _ := e.Get(v.ID)
	if *frozen.Nodes["n5"].Seed < 1 || frozen.Nodes["n5"].Outputs[0] == "changed.png" {
		t.Fatal("returned node data aliases the executor")
	}
	e.Close()
	e2 := NewPersistentExecutor(runner, nil, store.db)
	defer e2.Close()
	got, ok := e2.Get(v.ID)
	if !ok || got.State != RunComplete || got.WorkflowID != "workflow-a" || got.Nodes["n5"].Seed == nil {
		t.Fatalf("restart history: %+v", got)
	}
	h, err := e2.History()
	if err != nil || len(h) != 1 {
		t.Fatalf("history = %v, %v", h, err)
	}
}

func TestCacheDoesNotAttributeOutputToChangedInputsOrRuntime(t *testing.T) {
	for _, changeRuntime := range []bool{false, true} {
		t.Run(map[bool]string{false: "source", true: "runtime"}[changeRuntime], func(t *testing.T) {
			store := newStore(t)
			runner := &cacheRunner{fakeRunner: &fakeRunner{dir: t.TempDir()}, identity: "runtime-a"}
			source := filepath.Join(runner.dir, "source.png")
			if err := writeImage(runner.dir, "source.png", image.NewGray(image.Rect(0, 0, 32, 32))); err != nil {
				t.Fatal(err)
			}
			g := smallTxt2img()
			plan := planFor(t, g)
			plan.Stages[0].Generate.Init = &ImageRef{Path: source}
			plan.Stages[0].Generate.Params.InitImagePath = source
			runner.hook = func(context.Context, mediagen.GenerateParams) error {
				if changeRuntime {
					runner.identity = "runtime-b"
					return nil
				}
				return os.WriteFile(source, []byte("changed after reading"), 0600)
			}
			e := NewPersistentExecutor(runner, nil, store.db)
			defer e.Close()
			v, err := e.Submit(plan)
			if err != nil {
				t.Fatal(err)
			}
			waitRun(t, e, v.ID, RunComplete)
			var entries int
			if err := store.db.QueryRow(`SELECT count(*) FROM workflow_cache`).Scan(&entries); err != nil || entries != 0 {
				t.Fatalf("result cached under changed identity: entries=%d err=%v", entries, err)
			}
		})
	}
}

func TestCacheReusesGenerationInvalidatesRuntimeAndMissingFiles(t *testing.T) {
	store := newStore(t)
	runner := &cacheRunner{fakeRunner: &fakeRunner{dir: t.TempDir()}, identity: "runtime-a"}
	e := NewPersistentExecutor(runner, nil, store.db)
	defer e.Close()
	submit := func(force bool) RunView {
		v, err := e.SubmitWithOptions(planFor(t, smallTxt2img()), RunOptions{Force: force})
		if err != nil {
			t.Fatal(err)
		}
		return waitRun(t, e, v.ID, RunComplete)
	}
	a := submit(false)
	b := submit(false)
	if !b.Nodes["n5"].Cached || b.Nodes["n6"].Cached || runner.n != 1 {
		t.Fatalf("expected generation reuse and a fresh save: %+v, calls=%d", b.Nodes, runner.n)
	}
	if a.Nodes["n6"].Outputs[0] == b.Nodes["n6"].Outputs[0] {
		t.Fatal("saving must create independent outputs per run")
	}
	_ = os.Remove(filepath.Join(runner.dir, filepath.FromSlash(b.Nodes["n5"].Outputs[0])))
	if c := submit(false); c.Nodes["n5"].Cached || runner.n != 2 {
		t.Fatal("missing cached output must regenerate")
	}
	runner.identity = "runtime-b"
	if c := submit(false); c.Nodes["n5"].Cached || runner.n != 3 {
		t.Fatal("runtime change must invalidate generation")
	}
	if c := submit(true); c.Nodes["n5"].Cached || runner.n != 4 {
		t.Fatal("force must bypass cache")
	}
}

func TestRandomStagesNeverReuse(t *testing.T) {
	store := newStore(t)
	runner := &cacheRunner{fakeRunner: &fakeRunner{dir: t.TempDir()}, identity: "runtime-a"}
	e := NewPersistentExecutor(runner, nil, store.db)
	defer e.Close()
	g := smallTxt2img()
	g.Nodes[4].Params["seed"] = -1
	for range 2 {
		v, err := e.Submit(planFor(t, g))
		if err != nil {
			t.Fatal(err)
		}
		v = waitRun(t, e, v.ID, RunComplete)
		if v.Nodes["n5"].Cached {
			t.Fatal("random output reused")
		}
	}
	if runner.n != 2 {
		t.Fatal("random stage did not generate twice")
	}
}

func TestResolvedSeedIsDurableBeforeInferenceCompletes(t *testing.T) {
	store := newStore(t)
	runner := &fakeRunner{dir: t.TempDir()}
	started := make(chan struct{})
	runner.hook = func(ctx context.Context, _ mediagen.GenerateParams) error {
		close(started)
		<-ctx.Done()
		return ctx.Err()
	}
	e := NewPersistentExecutor(runner, nil, store.db)
	defer e.Close()
	g := smallTxt2img()
	g.Nodes[4].Params["seed"] = -1
	v, err := e.SubmitWithOptions(planFor(t, g), RunOptions{Graph: &g})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	durable, err := e.storedRun(v.ID)
	if err != nil || durable.Nodes["n5"].Seed == nil || *durable.Nodes["n5"].Seed < 1 || durable.Graph == nil || durable.SourceGraph.Nodes[4].Params["seed"] != float64(-1) {
		t.Fatalf("running seed was not saved: %+v err=%v", durable, err)
	}
}

func TestRestartRecordsInterruptedRuns(t *testing.T) {
	store := newStore(t)
	raw := `{"id":"interrupted","state":"running","created_at":"2026-01-01","nodes":{"n1":{"state":"running"},"n2":{"state":"done"}}}`
	if _, err := store.db.Exec(`INSERT INTO workflow_runs VALUES ('interrupted','running','2026-01-01',?)`, raw); err != nil {
		t.Fatal(err)
	}
	e := NewPersistentExecutor(&fakeRunner{dir: t.TempDir()}, nil, store.db)
	defer e.Close()
	v, ok := e.Get("interrupted")
	if !ok || v.State != RunCanceled || v.Nodes["n1"].State != NodeCanceled || v.Nodes["n2"].State != NodeDone {
		t.Fatalf("recovery = %+v", v)
	}
}

func TestCloseWaitsForActiveRunAndRejectsSubmissions(t *testing.T) {
	runner := &fakeRunner{dir: t.TempDir()}
	entered := make(chan struct{})
	runner.hook = func(ctx context.Context, _ mediagen.GenerateParams) error {
		close(entered)
		<-ctx.Done()
		return ctx.Err()
	}
	e := NewExecutor(runner, nil, nil)
	v, err := e.Submit(planFor(t, smallTxt2img()))
	if err != nil {
		t.Fatal(err)
	}
	<-entered
	e.Close()
	e.Close()
	got, _ := e.Get(v.ID)
	if got.State != RunCanceled {
		t.Fatalf("shutdown left %s", got.State)
	}
	if _, err := e.Submit(planFor(t, smallTxt2img())); err == nil {
		t.Fatal("closed executor accepted a run")
	}
}
