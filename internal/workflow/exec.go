package workflow

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"log/slog"
	"math"
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

type progressStageRunner interface {
	RunStageWithProgress(context.Context, string, mediagen.GenerateParams, *mediagen.LoadOverrides, func(mediagen.Progress)) (*mediagen.Job, error)
}

// NodeState is one node's progress within a run.
type NodeState struct {
	Cached   bool               `json:"cached,omitempty"`
	Seed     *int64             `json:"seed,omitempty"`
	Progress *mediagen.Progress `json:"progress,omitempty"`
	State    string             `json:"state"`
	Message  string             `json:"message,omitempty"`
	Outputs  []string           `json:"outputs,omitempty"`   // media-root-relative
	FileURLs []string           `json:"file_urls,omitempty"` // local file:// URLs for previews
	JobID    string             `json:"job_id,omitempty"`
	Millis   int64              `json:"ms,omitempty"`
}

// RunView is a copy of a run's state, safe to serialise.
type RunView struct {
	SourceGraph *Graph               `json:"source_graph,omitempty"`
	Graph       *Graph               `json:"graph,omitempty"`
	WorkflowID  string               `json:"workflow_id,omitempty"`
	Only        string               `json:"only,omitempty"`
	ID          string               `json:"id"`
	State       string               `json:"state"`
	Error       string               `json:"error,omitempty"`
	Nodes       map[string]NodeState `json:"nodes"`
	CreatedAt   string               `json:"created_at"`
	StartedAt   string               `json:"started_at,omitempty"`
	FinishedAt  string               `json:"finished_at,omitempty"`
}

type run struct {
	view    RunView
	plan    *Plan
	ctx     context.Context
	cancel  context.CancelFunc
	force   bool
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

	mu        sync.Mutex
	runs      map[string]*run
	order     []string
	queue     chan *run
	quit      chan struct{}
	done      chan struct{}
	closeOnce sync.Once
	db        *sql.DB
	initErr   error
}

// NewExecutor starts the single worker. Call Close to stop it.
func NewExecutor(runner StageRunner, events EventSink, log *slog.Logger) *Executor {
	if log == nil {
		log = slog.Default()
	}
	e := &Executor{
		runner: runner, events: events, log: log,
		runs: map[string]*run{}, queue: make(chan *run, maxQueuedRuns), quit: make(chan struct{}), done: make(chan struct{}),
	}
	go e.loop()
	return e
}

// Close stops the worker and cancels the active run.
func (e *Executor) Close() {
	e.closeOnce.Do(func() {
		close(e.quit)
		e.mu.Lock()
		for _, r := range e.runs {
			r.cancel()
			if r.view.State == RunQueued {
				r.view.State, r.view.FinishedAt = RunCanceled, ts()
				for id, ns := range r.view.Nodes {
					ns.State = NodeCanceled
					r.view.Nodes[id] = ns
				}
				if err := e.persistLocked(r); err != nil {
					e.log.Error("saving canceled workflow", "err", err)
				}
			}
		}
		e.mu.Unlock()
		<-e.done
	})
}

// NewPersistentExecutor adds history and result reuse to the same serialized queue.
func NewPersistentExecutor(runner StageRunner, events EventSink, db *sql.DB) *Executor {
	e := NewExecutor(runner, events, nil)
	e.db = db
	e.initErr = e.recoverRuns()
	return e
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
	return e.SubmitWithOptions(plan, RunOptions{})
}

func (e *Executor) SubmitWithOptions(plan *Plan, opts RunOptions) (RunView, error) {
	if e.initErr != nil {
		return RunView{}, e.initErr
	}
	select {
	case <-e.quit:
		return RunView{}, errors.New("workflow executor stopped")
	default:
	}
	if plan == nil || len(plan.Stages) == 0 {
		return RunView{}, errors.New("nothing to run")
	}
	raw, err := json.Marshal(plan)
	if err != nil {
		return RunView{}, err
	}
	var snapshot Plan
	if err := json.Unmarshal(raw, &snapshot); err != nil {
		return RunView{}, err
	}
	plan = &snapshot
	var graph *Graph
	if opts.Graph != nil {
		raw, err := json.Marshal(opts.Graph)
		if err != nil {
			return RunView{}, err
		}
		graph = &Graph{}
		if err := json.Unmarshal(raw, graph); err != nil {
			return RunView{}, err
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	r := &run{plan: plan, ctx: ctx, cancel: cancel, force: opts.Force, outputs: map[string][]string{}}
	r.view = RunView{ID: uuid.NewString(), State: RunQueued, Nodes: map[string]NodeState{}, CreatedAt: ts(), Graph: graph, WorkflowID: opts.WorkflowID, Only: opts.Only}
	if graph != nil {
		raw, _ := json.Marshal(graph)
		r.view.SourceGraph = &Graph{}
		_ = json.Unmarshal(raw, r.view.SourceGraph)
	}
	for _, st := range plan.Stages {
		r.view.Nodes[st.NodeID] = NodeState{State: NodePending}
		for _, id := range st.Absorbs {
			r.view.Nodes[id] = NodeState{State: NodePending}
		}
	}

	e.mu.Lock()
	select {
	case <-e.quit:
		e.mu.Unlock()
		cancel()
		return RunView{}, errors.New("workflow executor stopped")
	default:
	}
	if len(e.queue) >= cap(e.queue) {
		e.mu.Unlock()
		cancel()
		return RunView{}, ErrQueueFull
	}
	if err := e.persistLocked(r); err != nil {
		e.mu.Unlock()
		cancel()
		return RunView{}, err
	}
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
	if r.view.Graph != nil {
		raw, _ := json.Marshal(r.view.Graph)
		v.Graph = &Graph{}
		_ = json.Unmarshal(raw, v.Graph)
	}
	if r.view.SourceGraph != nil {
		raw, _ := json.Marshal(r.view.SourceGraph)
		v.SourceGraph = &Graph{}
		_ = json.Unmarshal(raw, v.SourceGraph)
	}
	v.Nodes = make(map[string]NodeState, len(r.view.Nodes))
	for k, n := range r.view.Nodes {
		n.Outputs = append([]string(nil), n.Outputs...)
		n.FileURLs = append([]string(nil), n.FileURLs...)
		if n.Seed != nil {
			seed := *n.Seed
			n.Seed = &seed
		}
		if n.Progress != nil {
			progress := *n.Progress
			if progress.ServerResponding != nil {
				responding := *progress.ServerResponding
				progress.ServerResponding = &responding
			}
			n.Progress = &progress
		}
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
		v, err := e.storedRun(id)
		return v, err == nil
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
		if _, err := e.storedRun(id); err == nil {
			return nil
		}
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
	if queued {
		if err := e.persistLocked(r); err != nil {
			e.mu.Unlock()
			r.cancel()
			return err
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
	defer close(e.done)
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
	if ns.Progress != nil {
		payload["progress"] = ns.Progress
	}
	if ns.Cached {
		payload["cached"] = true
	}
	if ns.Seed != nil {
		payload["seed"] = *ns.Seed
	}
	if ns.State == NodeDone || ns.State == NodeFailed || ns.State == NodeCanceled {
		e.persist(r)
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
	if err := e.persistLocked(r); err != nil {
		e.log.Error("saving finished workflow", "err", err)
	}
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
	e.persist(r)
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
		inputIdentity := e.cacheInputIdentity(r, st)
		key := e.cacheKey(r, st, servers)
		entry, hit := e.cached(key)
		if r.force {
			hit = false
		}
		var outs []string
		var jobID, msg string
		var err error
		if hit {
			outs, jobID, msg = entry.Outputs, entry.JobID, entry.Message
		} else {
			outs, jobID, msg, err = e.runStage(r, st, servers, prefix)
		}
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
		e.mu.Lock()
		seed := r.view.Nodes[st.NodeID].Seed
		e.mu.Unlock()
		if hit {
			seed = entry.Seed
			e.recordSeed(r, st.NodeID, seed)
		}
		if !hit {
			// A source or running configuration changed during execution: this
			// result cannot be attributed to the new identity. A newly loaded
			// server may establish its first identity only after generating.
			resolvedKey := e.cacheKey(r, st, servers)
			if inputIdentity != "" && inputIdentity == e.cacheInputIdentity(r, st) && (key == "" || key == resolvedKey) {
				e.putCache(resolvedKey, cacheEntry{Outputs: outs, JobID: jobID, Message: msg, Seed: seed})
			}
		}
		e.setNode(r, st.NodeID, NodeState{
			Cached: hit, Seed: seed,
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
	case StageUpscale:
		return e.runUpscale(r, st, servers)
	case StageImage:
		outs, err = e.runImage(r, st, prefix)
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
	p.CFGScaleExplicit = true
	p.GuidanceExplicit = true
	if p.Seed < 0 {
		// Resolve "random" here so the run is reproducible and the seed that
		// made an image is visible on the node and in the job row.
		p.Seed = 1 + rand.Int64N(1<<31-1)
	}
	e.recordSeed(r, st.NodeID, &p.Seed)
	if gs.Init != nil {
		path, err := e.resolveImage(r, *gs.Init)
		if err != nil {
			return nil, "", "", err
		}
		p.InitImagePath = path
	}
	for _, input := range []struct {
		ref *ImageRef
		dst *string
	}{{gs.Mask, &p.MaskImagePath}, {gs.Control, &p.ControlImagePath}} {
		if input.ref != nil {
			path, err := e.resolveImage(r, *input.ref)
			if err != nil {
				return nil, "", "", err
			}
			*input.dst = path
		}
	}
	if gs.Reference != nil {
		path, err := e.resolveImage(r, *gs.Reference)
		if err != nil {
			return nil, "", "", err
		}
		p.RefImagePaths = []string{path}
	}
	if p.InitImagePath != "" && (p.Width == 0 || p.Height == 0) {
		img, err := readImage(p.InitImagePath)
		if err != nil {
			return nil, "", "", err
		}
		p.Width, p.Height = img.Bounds().Dx(), img.Bounds().Dy()
		if p.Width > 4096 || p.Height > 4096 {
			return nil, "", "", fmt.Errorf("starting image exceeds 4096 × 4096; resize it before sampling")
		}
	}

	ov := need.Overrides
	var job *mediagen.Job
	var err error
	if runner, ok := e.runner.(progressStageRunner); ok {
		job, err = runner.RunStageWithProgress(r.ctx, need.ModelID, p, &ov, func(progress mediagen.Progress) {
			e.mu.Lock()
			ns := r.view.Nodes[st.NodeID]
			if ns.State != NodeRunning || r.view.State != RunRunning || r.ctx.Err() != nil {
				e.mu.Unlock()
				return
			}
			ns.Progress, ns.Message, ns.JobID = &progress, progress.Message, progress.JobID
			r.view.Nodes[st.NodeID] = ns
			e.mu.Unlock()
			e.publish("workflow.node_state", map[string]any{"run_id": r.view.ID, "node_id": st.NodeID,
				"state": NodeRunning, "job_id": progress.JobID, "message": progress.Message, "progress": progress})
		})
	} else {
		job, err = e.runner.RunStage(r.ctx, need.ModelID, p, &ov)
	}
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
	if ref.Index < 0 || ref.Index >= len(outs) {
		return "", fmt.Errorf("stage %s has no image at index %d", ref.Stage, ref.Index)
	}
	return filepath.Join(e.runner.MediaDir(), filepath.FromSlash(outs[ref.Index])), nil
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
	width, height := rz.Width, rz.Height
	if width == 0 && height == 0 && rz.Mode != "size" {
		width = int(math.Round(float64(cfg.Width) * rz.Scale))
		height = int(math.Round(float64(cfg.Height) * rz.Scale))
	}
	if width < 16 || height < 16 || width > 4096 || height > 4096 {
		return nil, fmt.Errorf("Resize would produce %d × %d; the limit is 16 to 4096 on each side", width, height)
	}
	if err := r.ctx.Err(); err != nil {
		return nil, err
	}
	out := resizeImage(img, width, height, rz.Filter)

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
		rel := filepath.ToSlash(filepath.Join(sub, fmt.Sprintf("%s%s-%s-%d%s", name, prefix, st.ID, i, strings.ToLower(filepath.Ext(src)))))
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
