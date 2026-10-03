package workflow

import (
	"context"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

type observedRunner struct {
	*fakeRunner
	observed chan struct{}
	release  chan struct{}
	callback func(mediagen.Progress)
}

func (r *observedRunner) RunStageWithProgress(ctx context.Context, id string, p mediagen.GenerateParams, ov *mediagen.LoadOverrides, onProgress func(mediagen.Progress)) (*mediagen.Job, error) {
	r.callback = onProgress
	responding := true
	onProgress(mediagen.Progress{JobID: "job-1", Phase: "sampling", Message: "Sampling: step 2/4", Current: 2, Total: 4, Unit: "steps", ServerResponding: &responding})
	close(r.observed)
	select {
	case <-r.release:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return r.fakeRunner.RunStage(ctx, id, p, ov)
}

func TestGenerationProgressReachesNodeSnapshotsAndEvents(t *testing.T) {
	runner := &observedRunner{fakeRunner: &fakeRunner{dir: t.TempDir()}, observed: make(chan struct{}), release: make(chan struct{})}
	sink := &fakeSink{}
	e := NewExecutor(runner, sink, nil)
	t.Cleanup(e.Close)
	plan := planFor(t, smallTxt2img())
	v, err := e.Submit(plan)
	if err != nil {
		t.Fatal(err)
	}
	<-runner.observed
	v, _ = e.Get(v.ID)
	node := plan.Stages[0].NodeID
	ns := v.Nodes[node]
	if ns.Progress == nil || ns.Progress.Current != 2 || ns.Progress.Total != 4 || ns.JobID != "job-1" || ns.Message != "Sampling: step 2/4" {
		t.Fatalf("running node has no usable progress: %+v", ns)
	}
	sink.mu.Lock()
	found := false
	for _, event := range sink.events {
		if event.name == "workflow.node_state" && event.payload["node_id"] == node && event.payload["progress"] != nil {
			found = true
		}
	}
	sink.mu.Unlock()
	if !found {
		t.Fatal("live progress event was lost")
	}
	close(runner.release)
	v = waitRun(t, e, v.ID, RunComplete, RunFailed)
	if v.State != RunComplete {
		t.Fatalf("generation failed: %+v", v)
	}
	runner.callback(mediagen.Progress{Message: "late progress"})
	v, _ = e.Get(v.ID)
	if v.Nodes[node].State != NodeDone || v.Nodes[node].Message == "late progress" {
		t.Fatal("late callback resurrected a finished node")
	}
}
