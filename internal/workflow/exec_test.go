package workflow

import (
	"context"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

type sinkEvent struct {
	name    string
	payload map[string]any
}

type fakeSink struct {
	mu     sync.Mutex
	events []sinkEvent
}

func (s *fakeSink) Publish(event string, payload any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m, _ := payload.(map[string]any)
	s.events = append(s.events, sinkEvent{event, m})
}

func (s *fakeSink) names() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []string
	for _, e := range s.events {
		out = append(out, e.name)
	}
	return out
}

// fakeRunner stands in for the media manager: each generation writes a real
// png of the requested size into the media dir and returns it as a job.
type fakeRunner struct {
	dir string

	mu    sync.Mutex
	calls []mediagen.GenerateParams
	ovs   []mediagen.LoadOverrides
	n     int
	hook  func(ctx context.Context, p mediagen.GenerateParams) error // optional: block or fail
}

func (f *fakeRunner) MediaDir() string { return f.dir }

func (f *fakeRunner) RunStage(ctx context.Context, modelID string, p mediagen.GenerateParams, ov *mediagen.LoadOverrides) (*mediagen.Job, error) {
	f.mu.Lock()
	f.n++
	n := f.n
	f.calls = append(f.calls, p)
	if ov != nil {
		f.ovs = append(f.ovs, *ov)
	}
	hook := f.hook
	f.mu.Unlock()
	if hook != nil {
		if err := hook(ctx, p); err != nil {
			return nil, err
		}
	}
	w, h := p.Width, p.Height
	if w == 0 {
		w, h = 8, 8
	}
	rel := fmt.Sprintf("images/job-%d-0.png", n)
	abs := filepath.Join(f.dir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return nil, err
	}
	file, err := os.Create(abs)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	if err := png.Encode(file, image.NewNRGBA(image.Rect(0, 0, w, h))); err != nil {
		return nil, err
	}
	return &mediagen.Job{ID: fmt.Sprintf("job-%d", n), State: mediagen.StateComplete, OutputPaths: []string{rel}}, nil
}

func newExec(t *testing.T) (*Executor, *fakeRunner, *fakeSink) {
	t.Helper()
	runner := &fakeRunner{dir: t.TempDir()}
	sink := &fakeSink{}
	e := NewExecutor(runner, sink, nil)
	t.Cleanup(e.Close)
	return e, runner, sink
}

func waitRun(t *testing.T, e *Executor, id string, want ...string) RunView {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		v, ok := e.Get(id)
		if !ok {
			t.Fatalf("run %s vanished", id)
		}
		for _, w := range want {
			if v.State == w {
				return v
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	v, _ := e.Get(id)
	t.Fatalf("run %s never reached %v; state %q nodes %+v", id, want, v.State, v.Nodes)
	return RunView{}
}

func smallTxt2img() Graph {
	g := txt2img()
	g.Nodes[3].Params = map[string]any{"width": 64, "height": 80, "batch": 1}
	g.Nodes[4].Params = map[string]any{"seed": 11, "steps": 4}
	g.Nodes[5].Params = map[string]any{"prefix": "my shot/"}
	return g
}

func planFor(t *testing.T, g Graph) *Plan {
	t.Helper()
	return mustPlan(t, g, newFakeEnv(), Options{})
}

func TestRunTxt2ImgEndToEnd(t *testing.T) {
	e, runner, sink := newExec(t)
	plan := planFor(t, smallTxt2img())

	view, err := e.Submit(plan)
	if err != nil {
		t.Fatal(err)
	}
	if view.State != RunQueued || view.Nodes["n5"].State != NodePending || view.Nodes["n1"].State != NodePending {
		t.Fatalf("initial view = %+v", view)
	}
	done := waitRun(t, e, view.ID, RunComplete, RunFailed)
	if done.State != RunComplete {
		t.Fatalf("run = %+v", done)
	}
	for id, n := range done.Nodes {
		if n.State != NodeDone {
			t.Errorf("node %s = %q, want done", id, n.State)
		}
	}
	gen := done.Nodes["n5"]
	if len(gen.Outputs) != 1 || gen.JobID != "job-1" || gen.Message != "seed 11" || !strings.HasPrefix(gen.FileURLs[0], "file://") {
		t.Fatalf("generate node = %+v", gen)
	}
	saved := done.Nodes["n6"]
	if len(saved.Outputs) != 1 || !strings.HasPrefix(saved.Outputs[0], "images/my_shot_"+view.ID[:8]) {
		t.Fatalf("save node outputs = %v, want sanitised prefix", saved.Outputs)
	}
	if _, err := os.Stat(filepath.Join(runner.dir, filepath.FromSlash(saved.Outputs[0]))); err != nil {
		t.Fatalf("saved file missing: %v", err)
	}
	if runner.calls[0].Seed != 11 || runner.calls[0].Width != 64 {
		t.Fatalf("params = %+v", runner.calls[0])
	}

	names := sink.names()
	if names[0] != "workflow.run_queued" || names[1] != "workflow.run_started" || names[len(names)-1] != "workflow.run_finished" {
		t.Fatalf("event order = %v", names)
	}
}

func TestRandomSeedIsResolvedBeforeSubmitting(t *testing.T) {
	e, runner, _ := newExec(t)
	g := smallTxt2img()
	g.Nodes[4].Params["seed"] = -1
	v, _ := e.Submit(planFor(t, g))
	done := waitRun(t, e, v.ID, RunComplete, RunFailed)
	if got := runner.calls[0].Seed; got <= 0 {
		t.Fatalf("seed sent = %d, want a concrete positive seed", got)
	}
	if want := fmt.Sprintf("seed %d", runner.calls[0].Seed); done.Nodes["n5"].Message != want {
		t.Fatalf("node message = %q, want %q", done.Nodes["n5"].Message, want)
	}
}

func TestHiresFixChainsStageOutputs(t *testing.T) {
	e, runner, _ := newExec(t)
	g := hiresFix()
	g.Nodes[3].Params = map[string]any{"width": 64, "height": 80, "batch": 1}
	g.Nodes[4].Params = map[string]any{"seed": 5, "steps": 4}
	g.Nodes[6].Params = map[string]any{"seed": 5, "steps": 4, "strength": 0.4}

	v, err := e.Submit(planFor(t, g))
	if err != nil {
		t.Fatal(err)
	}
	done := waitRun(t, e, v.ID, RunComplete, RunFailed)
	if done.State != RunComplete {
		t.Fatalf("run = %+v", done)
	}
	if len(runner.calls) != 2 {
		t.Fatalf("generate calls = %d, want 2", len(runner.calls))
	}
	refine := runner.calls[1]
	if refine.Width != 96 || refine.Height != 120 || refine.Strength != 0.4 {
		t.Fatalf("refine params = %+v", refine)
	}
	if !strings.Contains(refine.InitImagePath, "wf-"+v.ID[:8]+"-s2.png") {
		t.Fatalf("refine init image = %q, want the resize output", refine.InitImagePath)
	}
	f, err := os.Open(refine.InitImagePath)
	if err != nil {
		t.Fatalf("resize output missing: %v", err)
	}
	defer f.Close()
	cfg, err := png.DecodeConfig(f)
	if err != nil || cfg.Width != 96 || cfg.Height != 120 {
		t.Fatalf("resized image = %+v, %v; want 96 x 120", cfg, err)
	}
	if len(runner.ovs) != 2 {
		t.Fatalf("overrides passed = %d, want one per generate", len(runner.ovs))
	}
}

func TestFailedStageStopsTheRunAndMarksTheNode(t *testing.T) {
	e, runner, _ := newExec(t)
	runner.hook = func(context.Context, mediagen.GenerateParams) error { return fmt.Errorf("out of memory") }
	v, _ := e.Submit(planFor(t, smallTxt2img()))
	done := waitRun(t, e, v.ID, RunFailed, RunComplete)
	if done.State != RunFailed || done.Error != "out of memory" {
		t.Fatalf("run = %+v", done)
	}
	if n := done.Nodes["n5"]; n.State != NodeFailed || n.Message != "out of memory" {
		t.Fatalf("sample node = %+v", n)
	}
	if done.Nodes["n1"].State != NodePending || done.Nodes["n6"].State != NodePending {
		t.Fatalf("loaders and the unreached Save must read pending, got %+v", done.Nodes)
	}
}

func TestCancelRunningStopsTheGeneration(t *testing.T) {
	e, runner, _ := newExec(t)
	started := make(chan struct{})
	runner.hook = func(ctx context.Context, _ mediagen.GenerateParams) error {
		close(started)
		<-ctx.Done()
		return ctx.Err()
	}
	v, _ := e.Submit(planFor(t, smallTxt2img()))
	<-started
	if err := e.Cancel(v.ID); err != nil {
		t.Fatal(err)
	}
	done := waitRun(t, e, v.ID, RunCanceled)
	if done.Nodes["n5"].State != NodeCanceled || done.Nodes["n6"].State != NodeCanceled {
		t.Fatalf("nodes = %+v", done.Nodes)
	}
}

func TestRunsExecuteFirstInFirstOutAndQueuedRunsCancelCleanly(t *testing.T) {
	e, runner, sink := newExec(t)
	release := make(chan struct{})
	runner.hook = func(ctx context.Context, p mediagen.GenerateParams) error {
		if p.Seed == 11 { // the first run blocks until released
			select {
			case <-release:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		return nil
	}
	first, _ := e.Submit(planFor(t, smallTxt2img()))
	g2 := smallTxt2img()
	g2.Nodes[4].Params["seed"] = 22
	second, _ := e.Submit(planFor(t, g2))
	g3 := smallTxt2img()
	g3.Nodes[4].Params["seed"] = 33
	third, _ := e.Submit(planFor(t, g3))

	if v, _ := e.Get(second.ID); v.State != RunQueued {
		t.Fatalf("second run = %q while the first is running, want queued", v.State)
	}
	if err := e.Cancel(third.ID); err != nil {
		t.Fatal(err)
	}
	if v, _ := e.Get(third.ID); v.State != RunCanceled || v.Nodes["n5"].State != NodeCanceled {
		t.Fatalf("canceled queued run = %+v", v)
	}
	close(release)

	waitRun(t, e, first.ID, RunComplete)
	waitRun(t, e, second.ID, RunComplete)
	seeds := []int64{}
	for _, c := range runner.calls {
		seeds = append(seeds, c.Seed)
	}
	if fmt.Sprint(seeds) != "[11 22]" {
		t.Fatalf("generations ran with seeds %v, want [11 22] (third was canceled while queued)", seeds)
	}
	finishes := 0
	for _, n := range sink.names() {
		if n == "workflow.run_finished" {
			finishes++
		}
	}
	if finishes != 3 {
		t.Fatalf("run_finished events = %d, want 3", finishes)
	}
}

func TestQueueRejectsWhenFull(t *testing.T) {
	e, runner, _ := newExec(t)
	block := make(chan struct{})
	runner.hook = func(ctx context.Context, _ mediagen.GenerateParams) error {
		select {
		case <-block:
		case <-ctx.Done():
		}
		return nil
	}
	defer close(block)
	plan := planFor(t, smallTxt2img())
	var err error
	for i := 0; i < maxQueuedRuns+3 && err == nil; i++ {
		_, err = e.Submit(plan)
	}
	if err != ErrQueueFull {
		t.Fatalf("err = %v, want ErrQueueFull", err)
	}
}

func TestSubmitRejectsEmptyPlans(t *testing.T) {
	e, _, _ := newExec(t)
	if _, err := e.Submit(nil); err == nil {
		t.Fatal("nil plan must be rejected")
	}
	if _, err := e.Submit(&Plan{}); err == nil {
		t.Fatal("a plan with no stages must be rejected")
	}
	if err := e.Cancel("nope"); err != ErrRunNotFound {
		t.Fatalf("Cancel unknown = %v", err)
	}
}

func TestSavePrefixSanitising(t *testing.T) {
	for in, want := range map[string]string{
		"portrait_":             "portrait_",
		"my shot/../x":          "my_shot_.._x",
		"":                      "image_",
		"...":                   "image_",
		strings.Repeat("a", 90): strings.Repeat("a", 64),
	} {
		if got := savePrefix(in, "image_"); got != want {
			t.Errorf("savePrefix(%q) = %q, want %q", in, got, want)
		}
	}
}
