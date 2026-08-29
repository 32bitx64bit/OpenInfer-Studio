// Package orchestrate defines the contract for driving external tools
// (llama-quantize, perplexity harness, sampling/KLD probes). All execution is
// argv-only: no shell strings anywhere.
package orchestrate

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"quantlab/core"
)

// Tool enumerates the external binaries the pipeline may invoke.
type Tool string

const (
	ToolLlamaQuantize Tool = "llama-quantize"
	ToolPerplexity    Tool = "llama-perplexity"
)

// Invocation is a fully-resolved, argv-only command specification.
type Invocation struct {
	Tool Tool     `json:"tool"`
	Path string   `json:"path"` // absolute path to the binary
	Argv []string `json:"argv"` // arguments only; never includes Path
	Env  []string `json:"env,omitempty"`
	// WorkDir, when non-empty, is the process working directory.
	WorkDir string `json:"workDir,omitempty"`
}

// Validate enforces the argv-only and absolute-path invariants.
func (iv Invocation) Validate() error {
	if iv.Tool != ToolLlamaQuantize && iv.Tool != ToolPerplexity {
		return fmt.Errorf("orchestrate: unknown tool %q", iv.Tool)
	}
	if iv.Path == "" || !filepath.IsAbs(iv.Path) {
		return fmt.Errorf("orchestrate: %s path must be absolute, got %q", iv.Tool, iv.Path)
	}
	if strings.ContainsAny(iv.Path, ";&|`$") {
		return fmt.Errorf("orchestrate: %s path contains shell metacharacters", iv.Tool)
	}
	for _, a := range iv.Argv {
		if strings.ContainsAny(a, "\x00") {
			return fmt.Errorf("orchestrate: %s argv contains NUL", iv.Tool)
		}
	}
	return nil
}

// Result captures one completed invocation.
type Result struct {
	ExitCode int    `json:"exitCode"`
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	// StdoutTruncated / StderrTruncated report that capture hit the
	// configured byte bound and tail output was discarded.
	StdoutTruncated bool          `json:"stdoutTruncated,omitempty"`
	StderrTruncated bool          `json:"stderrTruncated,omitempty"`
	Duration        time.Duration `json:"duration,omitempty"`
}

// Runner executes invocations. The production implementation uses
// exec.CommandContext with argv only; tests substitute a fake.
//
// Process-tree cleanup contract: on Unix the child is placed in its own
// process group and cancellation kills the whole group, so grandchildren
// cannot outlive a cancelled run. On Windows, with the standard library
// alone (no golang.org/x/sys), only the direct child is killed:
// CREATE_NEW_PROCESS_GROUP addresses the child's console group, and
// grandchildren spawned by the tool may survive cancellation. The upgrade
// path is a Windows Job Object (via golang.org/x/sys/windows) assigned to
// the child with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; this module stays
// stdlib-only, so the limitation is documented rather than fixed here.
type Runner interface {
	Run(ctx context.Context, iv Invocation) (Result, error)
}

// QuantizeJob describes one llama-quantize execution derived from a profile.
type QuantizeJob struct {
	ProfileID string `json:"profileID"`
	InPath    string `json:"inPath"`
	OutPath   string `json:"outPath"`
	// Type is the primary dtype argument passed to llama-quantize; per-tensor
	// overrides from the profile are materialized separately by the delegated
	// mixed-quant path.
	Type core.DType `json:"type"`
	// Threads, when > 0, sets the worker count.
	Threads int `json:"threads,omitempty"`
}

func (j QuantizeJob) Validate() error {
	if j.ProfileID == "" || j.InPath == "" || j.OutPath == "" {
		return fmt.Errorf("orchestrate: quantize job missing id/paths")
	}
	if j.InPath == j.OutPath {
		return fmt.Errorf("orchestrate: quantize in/out paths must differ")
	}
	if !j.Type.IsQuant() {
		return fmt.Errorf("orchestrate: quantize type %q is not a quant dtype", j.Type)
	}
	return nil
}

// Invocation builds the llama-quantize argv:
//
//	llama-quantize [--threads N] <in> <out> <type>
//
// Argument order and spelling are fixed here so delegated implementations and
// fakes share one contract.
func (j QuantizeJob) Invocation(binaryPath string) (Invocation, error) {
	if err := j.Validate(); err != nil {
		return Invocation{}, err
	}
	argv := []string{}
	if j.Threads > 0 {
		argv = append(argv, "--threads", fmt.Sprintf("%d", j.Threads))
	}
	argv = append(argv, j.InPath, j.OutPath, string(j.Type))
	iv := Invocation{Tool: ToolLlamaQuantize, Path: binaryPath, Argv: argv}
	if err := iv.Validate(); err != nil {
		return Invocation{}, err
	}
	return iv, nil
}

// PerplexityJob describes one llama-perplexity evaluation.
type PerplexityJob struct {
	ModelPath string `json:"modelPath"`
	TextPath  string `json:"textPath"` // evaluation corpus
	CtxSize   int    `json:"ctxSize"`
}

func (j PerplexityJob) Validate() error {
	if j.ModelPath == "" || j.TextPath == "" {
		return fmt.Errorf("orchestrate: perplexity job missing paths")
	}
	if j.CtxSize <= 0 {
		return fmt.Errorf("orchestrate: perplexity ctx size must be > 0")
	}
	return nil
}

// Invocation builds the llama-perplexity argv:
//
//	llama-perplexity -m <model> -f <text> -c <ctx>
func (j PerplexityJob) Invocation(binaryPath string) (Invocation, error) {
	if err := j.Validate(); err != nil {
		return Invocation{}, err
	}
	iv := Invocation{
		Tool: ToolPerplexity,
		Path: binaryPath,
		Argv: []string{"-m", j.ModelPath, "-f", j.TextPath, "-c", fmt.Sprintf("%d", j.CtxSize)},
	}
	if err := iv.Validate(); err != nil {
		return Invocation{}, err
	}
	return iv, nil
}
