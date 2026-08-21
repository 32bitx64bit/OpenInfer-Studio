package pipeline

import (
	"sync"
	"time"

	"quantlab/core"
	"quantlab/tensorbank"
)

// Observer receives run lifecycle events from Engine.Resume: stage
// boundaries, throttled byte progress during quantize/assembly, and every
// measurement as it is recorded. A nil Observer preserves the previous
// silent behavior.
type Observer interface {
	StageStarted(stage core.Stage)
	StageProgress(stage core.Stage, progress float64, message string)
	StageCompleted(stage core.Stage, artifact string)
	Measurement(m core.Measurement)
}

// StageTimer is an optional observer extension. WrapTimer records per-stage
// wall time and, when the inner observer implements StageTimer, forwards
// StageDuration. The Observer interface itself is unchanged.
type StageTimer interface {
	StageDuration(stage core.Stage, d time.Duration)
}

type timedObs struct {
	Observer
	mu sync.Mutex
	t0 map[core.Stage]time.Time
}

// WrapTimer returns an Observer that records stage durations. If obs
// implements StageTimer, durations are forwarded. A nil obs stays nil.
// WrapTimer is idempotent: wrapping an already-timed observer returns it.
func WrapTimer(obs Observer) Observer {
	if obs == nil {
		return nil
	}
	if _, ok := obs.(*timedObs); ok {
		return obs
	}
	return &timedObs{Observer: obs, t0: make(map[core.Stage]time.Time)}
}

func (t *timedObs) StageStarted(stage core.Stage) {
	t.mu.Lock()
	t.t0[stage] = time.Now()
	t.mu.Unlock()
	t.Observer.StageStarted(stage)
}

func (t *timedObs) StageCompleted(stage core.Stage, artifact string) {
	t.mu.Lock()
	start, ok := t.t0[stage]
	delete(t.t0, stage)
	t.mu.Unlock()
	if ok {
		if st, ok := t.Observer.(StageTimer); ok {
			st.StageDuration(stage, time.Since(start))
		}
	}
	t.Observer.StageCompleted(stage, artifact)
}

func (e *Engine) obsStarted(s core.Stage) {
	if e.Obs != nil {
		e.Obs.StageStarted(s)
	}
}

func (e *Engine) obsCompleted(s core.Stage) {
	if e.Obs != nil {
		e.Obs.StageCompleted(s, e.Run.Artifacts[s])
	}
}

func (e *Engine) obsProgress(stage core.Stage, progress float64, message string) {
	if e.Obs != nil {
		e.Obs.StageProgress(stage, progress, message)
	}
}

func (e *Engine) obsMeasurement(m core.Measurement) {
	if e.Obs != nil {
		e.Obs.Measurement(m)
	}
}

// progressMinInterval bounds StageProgress callbacks from byte-progress
// reporters; completion (1.0) is always reported.
const progressMinInterval = 100 * time.Millisecond

// throttledProgress adapts a tensorbank.ProgressFunc to StageProgress,
// coalescing bursts to one event per progressMinInterval.
type throttledProgress struct {
	stage   core.Stage
	message string
	obs     Observer
	now     func() time.Time

	mu   sync.Mutex
	last time.Time
	done bool
}

func (p *throttledProgress) report(copied, total uint64) {
	frac := 0.0
	if total > 0 {
		frac = float64(copied) / float64(total)
		if frac > 1 {
			frac = 1
		}
	}
	complete := total > 0 && copied >= total
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.done {
		return
	}
	if now := p.now(); !complete && !p.last.IsZero() && now.Sub(p.last) < progressMinInterval {
		return
	} else {
		p.last = now
	}
	if complete {
		p.done = true
	}
	p.obs.StageProgress(p.stage, frac, p.message)
}

// progressFunc returns a throttled byte-progress reporter for an assembly
// stage, or nil when no observer is attached (silent behavior).
func (e *Engine) progressFunc(stage core.Stage, message string) tensorbank.ProgressFunc {
	if e.Obs == nil {
		return nil
	}
	p := &throttledProgress{stage: stage, message: message, obs: e.Obs, now: time.Now}
	return p.report
}
