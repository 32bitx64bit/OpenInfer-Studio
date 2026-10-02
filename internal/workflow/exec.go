package workflow

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

// Run states.
const (
	RunQueued   = "queued"
	RunRunning  = "running"
	RunComplete = "complete"
	RunFailed   = "failed"
	RunCanceled = "canceled"
)

// Node states within a run.
const (
	NodePending  = "pending"
	NodeRunning  = "running"
	NodeDone     = "done"
	NodeFailed   = "failed"
	NodeCanceled = "canceled"
)

const (
	maxQueuedRuns = 20
	keepRuns      = 30
	maxResizePix  = 64_000_000 // decode guard: 64 megapixels
)

// ErrRunNotFound is returned for an unknown run id.
var ErrRunNotFound = errors.New("workflow run not found")

// ErrQueueFull is returned when too many runs are already waiting.
var ErrQueueFull = errors.New("too many runs are queued; wait for one to finish")

// EventSink receives progress events (the API hub).
type EventSink interface {
	Publish(event string, payload any)
}

// StageRunner runs one generation to completion and says where media lives.
// *mediagen.Manager satisfies it.
type StageRunner interface {
	RunStage(ctx context.Context, modelID string, p mediagen.GenerateParams, ov *mediagen.LoadOverrides) (*mediagen.Job, error)
	MediaDir() string
}

// NodeState is one node's progress within a run.
type NodeState struct {
	State    string   `json:"state"`
	Message  string   `json:"message,omitempty"`
	Outputs  []string `json:"outputs,omitempty"`   // media-root-relative
	FileURLs []string `json:"file_urls,omitempty"` // local file:// URLs for previews
	JobID    string   `json:"job_id,omitempty"`
	Millis   int64    `json:"ms,omitempty"`
}

// RunView is a copy of a run's state, safe to serialise.
type RunView struct {
	ID         string               `json:"id"`
	State      string               `json:"state"`
	Error      string               `json:"error,omitempty"`
	Nodes      map[string]NodeState `json:"nodes"`
	CreatedAt  string               `json:"created_at"`
	StartedAt  string               `json:"started_at,omitempty"`
	FinishedAt string               `json:"finished_at,omitempty"`
}

type run struct {
	view    RunView
	plan    *Plan
	ctx     context.Context
	cancel  context.CancelFunc
	outputs map[string][]string // stage id -> media-root-relative outputs
}

// Executor runs plans one at a time, first in first out, like a render
// queue: one GPU, one run. Runs live in memory (outputs persist as media
// files and media_jobs rows); the last few are kept so the UI can recover
// after a reconnect.
type Executor struct {
	runner StageRunner
	events EventSink
	log    *slog.Logger

	mu    sync.Mutex
	runs  map[string]*run
	order []string
	queue chan *run
	quit  chan struct{}
}

// NewExecutor starts the single worker. Call Close to stop it.
func NewExecutor(runner StageRunner, events EventSink, log *slog.Logger) *Executor {
	if log == nil {
		log = slog.Default()
	}
	e := &Executor{
		runner: runner, events: events, log: log,
		runs: map[string]*run{}, queue: make(chan *run, maxQueuedRuns), quit: make(chan struct{}),
	}
	go e.loop()
	return e
}

// Close stops the worker and cancels the active run.
func (e *Executor) Close() {
	close(e.quit)
	e.mu.Lock()
	for _, r := range e.runs {
		r.cancel()
	}
	e.mu.Unlock()
}

func (e *Executor) publish(event string, payload map[string]any) {
	if e.events != nil {
		e.events.Publish(event, payload)
	}
}

func ts() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// Submit queues a plan. It returns immediately; progress arrives as
// workflow.* events and through Get.
func (e *Executor) Submit(plan *Plan) (RunView, error) {
	if plan == nil || len(plan.Stages) == 0 {
		return RunView{}, errors.New("nothing to run")
	}
	ctx, cancel := context.WithCancel(context.Background())
	r := &run{plan: plan, ctx: ctx, cancel: cancel, outputs: map[string][]string{}}
	r.view = RunView{ID: uuid.NewString(), State: RunQueued, Nodes: map[string]NodeState{}, CreatedAt: ts()}
	for _, st := range plan.Stages {
		r.view.Nodes[st.NodeID] = NodeState{State: NodePending}
		for _, id := range st.Absorbs {
			r.view.Nodes[id] = NodeState{State: NodePending}
		}
	}

	e.mu.Lock()
	select {
	case e.queue <- r:
	default:
		e.mu.Unlock()
		cancel()
		return RunView{}, ErrQueueFull
	}
	e.runs[r.view.ID] = r
	e.order = append(e.order, r.view.ID)
	e.evictLocked()
	view := r.snapshotLocked()
	e.mu.Unlock()

	e.publish("workflow.run_queued", map[string]any{"run_id": view.ID, "nodes": view.Nodes})
	return view, nil
}

// evictLocked drops the oldest finished runs beyond keepRuns.
func (e *Executor) evictLocked() {
	for len(e.order) > keepRuns {
		id := e.order[0]
		r := e.runs[id]
		if r != nil && (r.view.State == RunQueued || r.view.State == RunRunning) {
			return // never evict live runs
		}
		delete(e.runs, id)
		e.order = e.order[1:]
	}
}

func (r *run) snapshotLocked() RunView {
	v := r.view
	v.Nodes = make(map[string]NodeState, len(r.view.Nodes))
	for k, n := range r.view.Nodes {
		v.Nodes[k] = n
	}
	return v
}

// Get returns a run's current state.
func (e *Executor) Get(id string) (RunView, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	r, ok := e.runs[id]
	if !ok {
		return RunView{}, false
	}
	return r.snapshotLocked(), true
}

// List returns recent runs, newest first.
func (e *Executor) List() []RunView {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]RunView, 0, len(e.order))
	for i := len(e.order) - 1; i >= 0; i-- {
		if r := e.runs[e.order[i]]; r != nil {
			out = append(out, r.snapshotLocked())
		}
	}
	return out
}

// Cancel stops a queued or running run. A running run's GPU job is canceled
// too (through the media manager's own cancel path).
func (e *Executor) Cancel(id string) error {
	e.mu.Lock()
	r, ok := e.runs[id]
	if !ok {
		e.mu.Unlock()
		return ErrRunNotFound
	}
	queued := r.view.State == RunQueued
	if queued {
		r.view.State = RunCanceled
		r.view.FinishedAt = ts()
		for k, n := range r.view.Nodes {
			n.State = NodeCanceled
			r.view.Nodes[k] = n
		}
	}
	e.mu.Unlock()
	r.cancel()
	if queued {
		e.publish("workflow.run_finished", map[string]any{"run_id": id, "state": RunCanceled})
	}
	return nil
}

func (e *Executor) loop() {
	for {
		select {
		case <-e.quit:
			return
		case r := <-e.queue:
			e.mu.Lock()
			skip := r.view.State != RunQueued
			e.mu.Unlock()
			if !skip {
				e.execute(r)
			}
		}
	}
}

func (e *Executor) setNode(r *run, nodeID string, ns NodeState) {
	e.mu.Lock()
	r.view.Nodes[nodeID] = ns
	runID := r.view.ID
	e.mu.Unlock()
	payload := map[string]any{"run_id": runID, "node_id": nodeID, "state": ns.State}
	if ns.Message != "" {
		payload["message"] = ns.Message
	}
	if len(ns.Outputs) > 0 {
		payload["outputs"] = ns.Outputs
		payload["file_urls"] = ns.FileURLs
	}
	if ns.JobID != "" {
		payload["job_id"] = ns.JobID
	}
	if ns.Millis > 0 {
		payload["ms"] = ns.Millis
	}
	e.publish("workflow.node_state", payload)
}

func (e *Executor) nodeState(r *run, id string) string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return r.view.Nodes[id].State
}

func (e *Executor) finish(r *run, state, msg string) {
	e.mu.Lock()
	r.view.State = state
	r.view.Error = msg
	r.view.FinishedAt = ts()
	id := r.view.ID
	e.mu.Unlock()
	payload := map[string]any{"run_id": id, "state": state}
	if msg != "" {
		payload["error"] = msg
	}
	e.publish("workflow.run_finished", payload)
}

func (e *Executor) execute(r *run) {
	e.mu.Lock()
	r.view.State = RunRunning
	r.view.StartedAt = ts()
	id := r.view.ID
	e.mu.Unlock()
	e.publish("workflow.run_started", map[string]any{"run_id": id})

	servers := map[string]ServerNeed{}
	for _, s := range r.plan.Servers {
		servers[s.Signature] = s
	}
	prefix := id[:8]

	for _, st := range r.plan.Stages {
		if r.ctx.Err() != nil {
			e.abandon(r, st, NodeCanceled, "")
			e.finish(r, RunCanceled, "")
			return
		}
		for _, a := range st.Absorbs {
			if e.nodeState(r, a) == NodePending {
				e.setNode(r, a, NodeState{State: NodeRunning})
			}
		}
		e.setNode(r, st.NodeID, NodeState{State: NodeRunning})

		start := time.Now()
		outs, jobID, msg, err := e.runStage(r, st, servers, prefix)
		elapsed := time.Since(start).Milliseconds()
		if err != nil {
			if r.ctx.Err() != nil {
				e.abandon(r, st, NodeCanceled, "")
				e.finish(r, RunCanceled, "")
				return
			}
			e.log.Warn("workflow stage failed", "run", id, "stage", st.ID, "kind", st.Kind, "err", err)
			e.abandon(r, st, NodeFailed, err.Error())
			e.finish(r, RunFailed, err.Error())
			return
		}
		e.mu.Lock()
		r.outputs[st.ID] = outs
		e.mu.Unlock()
		for _, a := range st.Absorbs {
			if e.nodeState(r, a) == NodeRunning {
				e.setNode(r, a, NodeState{State: NodeDone})
			}
		}
		e.setNode(r, st.NodeID, NodeState{
			State: NodeDone, Message: msg, Outputs: outs, FileURLs: e.fileURLs(outs), JobID: jobID, Millis: elapsed,
		})
	}
	e.finish(r, RunComplete, "")
}

// abandon marks the stage's node (and any absorbed nodes still mid-flight)
// after a failure or cancel, and the nodes of every stage not reached yet.
func (e *Executor) abandon(r *run, at Stage, state, msg string) {
	if e.nodeState(r, at.NodeID) != NodeDone {
		e.setNode(r, at.NodeID, NodeState{State: state, Message: msg})
	}
	for _, a := range at.Absorbs {
		if e.nodeState(r, a) == NodeRunning {
			e.setNode(r, a, NodeState{State: NodePending})
		}
	}
	reached := false
	for _, st := range r.plan.Stages {
		if st.ID == at.ID {
			reached = true
			continue
		}
		if reached && state == NodeCanceled && e.nodeState(r, st.NodeID) == NodePending {
			e.setNode(r, st.NodeID, NodeState{State: NodeCanceled})
		}
	}
}

func (e *Executor) runStage(r *run, st Stage, servers map[string]ServerNeed, prefix string) (outs []string, jobID, msg string, err error) {
	switch st.Kind {
	case StageGenerate:
		return e.runGenerate(r, st, servers)
	case StageResize:
		outs, err = e.runResize(r, st, prefix)
		return outs, "", "", err
	case StageSave:
		outs, err = e.runSave(r, st, prefix)
		return outs, "", "", err
	}
	return nil, "", "", fmt.Errorf("unknown stage kind %q", st.Kind)
}

func (e *Executor) runGenerate(r *run, st Stage, servers map[string]ServerNeed) ([]string, string, string, error) {
	gs := st.Generate
	need, ok := servers[gs.Server]
	if !ok {
		return nil, "", "", fmt.Errorf("plan references unknown server %q", gs.Server)
	}
	p := gs.Params
	if p.Seed < 0 {
		// Resolve "random" here so the run is reproducible and the seed that
		// made an image is visible on the node and in the job row.
		p.Seed = 1 + rand.Int64N(1<<31-1)
	}
	if gs.Init != nil {
		path, err := e.resolveImage(r, *gs.Init)
		if err != nil {
			return nil, "", "", err
		}
		p.InitImagePath = path
	}
	ov := need.Overrides
	job, err := e.runner.RunStage(r.ctx, need.ModelID, p, &ov)
	if err != nil {
		return nil, "", "", err
	}
	if len(job.OutputPaths) == 0 {
		return nil, job.ID, "", errors.New("generation finished without an output")
	}
	return job.OutputPaths, job.ID, fmt.Sprintf("seed %d", p.Seed), nil
}

// resolveImage turns an ImageRef into an absolute file path.
func (e *Executor) resolveImage(r *run, ref ImageRef) (string, error) {
	if ref.Path != "" {
		return ref.Path, nil
	}
	e.mu.Lock()
	outs := r.outputs[ref.Stage]
	e.mu.Unlock()
	if len(outs) == 0 {
		return "", fmt.Errorf("stage %s produced no image", ref.Stage)
	}
	return filepath.Join(e.runner.MediaDir(), filepath.FromSlash(outs[0])), nil
}

func (e *Executor) runResize(r *run, st Stage, prefix string) ([]string, error) {
	rz := st.Resize
	src, err := e.resolveImage(r, rz.Src)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(src)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	cfg, _, err := image.DecodeConfig(f)
	if err != nil {
		return nil, fmt.Errorf("cannot read %s as an image (Resize supports png, jpeg and gif; set the Sample's output format to png): %w", filepath.Base(src), err)
	}
	if cfg.Width*cfg.Height > maxResizePix {
		return nil, fmt.Errorf("image is %d x %d; Resize is limited to %d megapixels", cfg.Width, cfg.Height, maxResizePix/1_000_000)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	out := resizeImage(img, rz.Width, rz.Height, rz.Filter)

	rel := filepath.ToSlash(filepath.Join("images", fmt.Sprintf("wf-%s-%s.png", prefix, st.ID)))
	dest := filepath.Join(e.runner.MediaDir(), filepath.FromSlash(rel))
	tmp := dest + ".writing"
	w, err := os.Create(tmp)
	if err != nil {
		return nil, err
	}
	if err := png.Encode(w, out); err != nil {
		w.Close()
		os.Remove(tmp)
		return nil, err
	}
	if err := w.Close(); err != nil {
		os.Remove(tmp)
		return nil, err
	}
	if err := os.Rename(tmp, dest); err != nil {
		os.Remove(tmp)
		return nil, err
	}
	return []string{rel}, nil
}

// savePrefix keeps a user-typed prefix to safe file-name characters.
func savePrefix(p, fallback string) string {
	var b strings.Builder
	for _, c := range p {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '-', c == '_', c == '.':
			b.WriteRune(c)
		default:
			b.WriteByte('_')
		}
	}
	s := strings.Trim(b.String(), ".")
	if s == "" {
		s = fallback
	}
	if len(s) > 64 {
		s = s[:64]
	}
	return s
}

func (e *Executor) runSave(r *run, st Stage, prefix string) ([]string, error) {
	sv := st.Save
	var sources []string
	if sv.Src.Path != "" {
		sources = []string{sv.Src.Path}
	} else {
		e.mu.Lock()
		rels := append([]string(nil), r.outputs[sv.Src.Stage]...)
		e.mu.Unlock()
		for _, rel := range rels {
			sources = append(sources, filepath.Join(e.runner.MediaDir(), filepath.FromSlash(rel)))
		}
	}
	if len(sources) == 0 {
		return nil, errors.New("nothing to save")
	}
	sub, fallback := "images", "image_"
	if sv.Kind == "video" {
		sub, fallback = "videos", "video_"
	}
	name := savePrefix(sv.Prefix, fallback)
	var outs []string
	for i, src := range sources {
		rel := filepath.ToSlash(filepath.Join(sub, fmt.Sprintf("%s%s-%d%s", name, prefix, i, strings.ToLower(filepath.Ext(src)))))
		if err := copyFileAtomic(src, filepath.Join(e.runner.MediaDir(), filepath.FromSlash(rel))); err != nil {
			return nil, err
		}
		outs = append(outs, rel)
	}
	return outs, nil
}

func copyFileAtomic(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp := dst + ".copying"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dst)
}

// fileURLs builds local file:// URLs the QML Image element can load.
func (e *Executor) fileURLs(rels []string) []string {
	root := e.runner.MediaDir()
	out := make([]string, 0, len(rels))
	for _, rel := range rels {
		abs := filepath.ToSlash(filepath.Join(root, filepath.FromSlash(rel)))
		if runtime.GOOS == "windows" && !strings.HasPrefix(abs, "/") {
			abs = "/" + abs
		}
		out = append(out, "file://"+(&url.URL{Path: abs}).EscapedPath())
	}
	return out
}
