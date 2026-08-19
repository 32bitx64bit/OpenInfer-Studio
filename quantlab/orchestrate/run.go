package orchestrate

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// DefaultMaxOutputBytes bounds each of stdout and stderr captured by OSRunner.
const DefaultMaxOutputBytes int64 = 4 << 20

// ExitError reports a process that ran to completion but did not exit 0, or
// was terminated by a signal. StderrTail carries the end of captured stderr
// for diagnostics.
type ExitError struct {
	Path       string
	ExitCode   int // -1 when terminated by a signal
	Signal     string
	StderrTail string
}

// IdleTimeoutError reports a process killed after producing no stdout/stderr
// activity for the configured interval.
type IdleTimeoutError struct {
	Path string
	Idle time.Duration
}

func (e *IdleTimeoutError) Error() string {
	return fmt.Sprintf("orchestrate: %s produced no output for %s", e.Path, e.Idle)
}

type OutputEvent struct {
	Tool   Tool
	Stream string
	Line   string
}

func (e *ExitError) Error() string {
	if e.ExitCode >= 0 {
		return fmt.Sprintf("orchestrate: %s exited %d: %s", e.Path, e.ExitCode, e.StderrTail)
	}
	return fmt.Sprintf("orchestrate: %s killed by %s: %s", e.Path, e.Signal, e.StderrTail)
}

// limitedBuffer captures at most max bytes, counting any discarded tail.
type limitedBuffer struct {
	buf       bytes.Buffer
	max       int64
	truncated int64
}

type activityWriter struct {
	dst      *limitedBuffer
	tool     Tool
	stream   string
	callback func(OutputEvent)
	lastUnix *atomic.Int64
	mu       sync.Mutex
	pending  string
}

func (w *activityWriter) Write(p []byte) (int, error) {
	w.lastUnix.Store(time.Now().UnixNano())
	n, err := w.dst.Write(p)
	if w.callback == nil || len(p) == 0 {
		return n, err
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.pending += string(p)
	if len(w.pending) > 64<<10 && !strings.Contains(w.pending, "\n") {
		w.callback(OutputEvent{Tool: w.tool, Stream: w.stream, Line: w.pending[:64<<10]})
		w.pending = w.pending[64<<10:]
	}
	for {
		i := strings.IndexByte(w.pending, '\n')
		if i < 0 {
			break
		}
		line := strings.TrimRight(w.pending[:i], "\r")
		w.pending = w.pending[i+1:]
		w.callback(OutputEvent{Tool: w.tool, Stream: w.stream, Line: line})
	}
	return n, err
}

func (w *activityWriter) Flush() {
	if w.callback == nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.pending != "" {
		w.callback(OutputEvent{Tool: w.tool, Stream: w.stream, Line: strings.TrimRight(w.pending, "\r")})
		w.pending = ""
	}
}

func (l *limitedBuffer) Write(p []byte) (int, error) {
	if l.max > 0 {
		remaining := l.max - int64(l.buf.Len())
		if remaining <= 0 {
			l.truncated += int64(len(p))
			return len(p), nil
		}
		if int64(len(p)) > remaining {
			l.buf.Write(p[:remaining])
			l.truncated += int64(len(p)) - remaining
			return len(p), nil
		}
	}
	l.buf.Write(p)
	return len(p), nil
}

// tail returns the last n bytes of the captured output as a string.
func tail(s string, n int) string {
	if len(s) > n {
		return "..." + s[len(s)-n:]
	}
	return s
}

// OSRunner executes invocations with the OS process interface: argv-only
// exec.CommandContext (never a shell), explicit environment and working
// directory, bounded output capture, context cancellation, and process-group
// cleanup so grandchildren cannot outlive a cancelled run.
type OSRunner struct {
	// Env is the exact process environment. It is used when the Invocation
	// does not carry its own Env. Nil means "refuse": callers must pass an
	// explicit environment (possibly empty) so nothing leaks implicitly.
	Env []string
	// WorkDir is the default working directory; Invocation.WorkDir wins.
	WorkDir string
	// MaxOutputBytes caps stdout and stderr capture each; 0 uses
	// DefaultMaxOutputBytes.
	MaxOutputBytes int64
	// KillGrace is how long to wait for a killed process to reap before
	// reporting failure; 0 uses a sane default.
	KillGrace time.Duration
	// IdleTimeout kills a tool that emits no stdout or stderr for this long.
	// Zero disables idle supervision.
	IdleTimeout time.Duration
	// OnOutput receives complete stdout/stderr lines as they arrive.
	OnOutput func(OutputEvent)
}

func (r OSRunner) maxOut() int64 {
	if r.MaxOutputBytes > 0 {
		return r.MaxOutputBytes
	}
	return DefaultMaxOutputBytes
}

// Run executes iv to completion, cancellation, or failure. The returned
// Result is always populated with whatever output was captured, even on
// error. A nonzero exit yields a non-nil *ExitError; context cancellation
// yields the ctx error after the process group has been killed.
func (r OSRunner) Run(ctx context.Context, iv Invocation) (Result, error) {
	if err := iv.Validate(); err != nil {
		return Result{}, err
	}
	env := iv.Env
	if env == nil {
		env = r.Env
	}
	if env == nil {
		return Result{}, fmt.Errorf("orchestrate: no explicit environment for %s", iv.Tool)
	}
	dir := iv.WorkDir
	if dir == "" {
		dir = r.WorkDir
	}

	runCtx := ctx
	var cancel context.CancelFunc
	var idleExceeded atomic.Bool
	var idleDone chan struct{}
	var idleStopped chan struct{}
	var lastOutput atomic.Int64
	if r.IdleTimeout > 0 {
		runCtx, cancel = context.WithCancel(ctx)
		defer cancel()
		lastOutput.Store(time.Now().UnixNano())
		idleDone = make(chan struct{})
		idleStopped = make(chan struct{})
		interval := r.IdleTimeout / 4
		if interval < 10*time.Millisecond {
			interval = 10 * time.Millisecond
		}
		go func() {
			defer close(idleStopped)
			ticker := time.NewTicker(interval)
			defer ticker.Stop()
			for {
				select {
				case <-ticker.C:
					last := time.Unix(0, lastOutput.Load())
					if time.Since(last) >= r.IdleTimeout {
						idleExceeded.Store(true)
						cancel()
						return
					}
				case <-idleDone:
					return
				case <-ctx.Done():
					return
				}
			}
		}()
	}

	cmd := exec.CommandContext(runCtx, iv.Path, iv.Argv...)
	cmd.Env = env
	cmd.Dir = dir
	// Put the child in its own process group / job unit and, on cancel,
	// kill the whole tree rather than just the direct child.
	configureProcGroup(cmd)
	grace := r.KillGrace
	if grace <= 0 {
		grace = 5 * time.Second
	}
	cmd.Cancel = func() error { return killProcTree(cmd) }
	cmd.WaitDelay = grace

	stdout := &limitedBuffer{max: r.maxOut()}
	stderr := &limitedBuffer{max: r.maxOut()}
	stdoutWriter := &activityWriter{
		dst: stdout, tool: iv.Tool, stream: "stdout",
		callback: r.OnOutput, lastUnix: &lastOutput,
	}
	stderrWriter := &activityWriter{
		dst: stderr, tool: iv.Tool, stream: "stderr",
		callback: r.OnOutput, lastUnix: &lastOutput,
	}
	cmd.Stdout = stdoutWriter
	cmd.Stderr = stderrWriter

	start := time.Now()
	runErr := cmd.Run()
	if idleDone != nil {
		close(idleDone)
		<-idleStopped
	}
	stdoutWriter.Flush()
	stderrWriter.Flush()
	res := Result{
		ExitCode:        0,
		Stdout:          stdout.buf.String(),
		Stderr:          stderr.buf.String(),
		StdoutTruncated: stdout.truncated > 0,
		StderrTruncated: stderr.truncated > 0,
		Duration:        time.Since(start),
	}
	if ctx.Err() != nil {
		return res, ctx.Err()
	}
	if idleExceeded.Load() {
		return res, &IdleTimeoutError{Path: iv.Path, Idle: r.IdleTimeout}
	}
	if runErr == nil {
		return res, nil
	}
	var ee *exec.ExitError
	if errors.As(runErr, &ee) {
		ex := &ExitError{Path: iv.Path, ExitCode: ee.ExitCode(), StderrTail: tail(res.Stderr, 512)}
		if ee.ExitCode() < 0 {
			if ws, ok := ee.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
				ex.Signal = ws.Signal().String()
			}
		}
		res.ExitCode = ee.ExitCode()
		return res, ex
	}
	return res, fmt.Errorf("orchestrate: start %s: %w", iv.Path, runErr)
}

// RunOK is Run plus a hard requirement on exit code 0; the *ExitError
// diagnostic is returned otherwise.
func (r OSRunner) RunOK(ctx context.Context, iv Invocation) (Result, error) {
	res, err := r.Run(ctx, iv)
	if err != nil {
		return res, err
	}
	if res.ExitCode != 0 {
		return res, &ExitError{Path: iv.Path, ExitCode: res.ExitCode, StderrTail: tail(res.Stderr, 512)}
	}
	return res, nil
}

// CominedOutputText returns stdout+stderr merged in the order they were
// captured (stdout first), for parsers that accept either stream.
func CombinedOutput(res Result) string {
	if res.Stderr == "" {
		return res.Stdout
	}
	if res.Stdout == "" {
		return res.Stderr
	}
	return strings.TrimRight(res.Stdout, "\n") + "\n" + res.Stderr
}
