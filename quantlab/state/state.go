// Package state provides crash-resumable pipeline state. A run is a JSON
// document versioned by SchemaVersion; completed stages are never re-executed,
// and the store writes atomically (temp file + rename) so an interrupted run
// always leaves a loadable checkpoint. The persisted RunConfig is complete:
// a run resumes with zero CLI arguments.
package state

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"quantlab/core"
	"quantlab/kld"
)

// MaxRunIDLen bounds run IDs.
const MaxRunIDLen = 64

// ValidRunID enforces the run-ID safety contract: run IDs are interpolated
// into checkpoint and artifact file names, so they are restricted to
// [a-zA-Z0-9._-] with no path separators, no whitespace, and a length cap.
func ValidRunID(runID string) error {
	if runID == "" {
		return fmt.Errorf("state: empty run id")
	}
	if runID == "." || runID == ".." {
		return fmt.Errorf("state: run id %q is a reserved path element", runID)
	}
	if len(runID) > MaxRunIDLen {
		return fmt.Errorf("state: run id %q exceeds %d characters", runID, MaxRunIDLen)
	}
	for _, r := range runID {
		switch {
		case r >= 'a' && r <= 'z',
			r >= 'A' && r <= 'Z',
			r >= '0' && r <= '9',
			r == '.', r == '_', r == '-':
		default:
			return fmt.Errorf("state: run id %q contains forbidden character %q (allowed: letters, digits, '.', '_', '-')",
				runID, string(r))
		}
	}
	return nil
}

// SchemaVersion is bumped on any incompatible layout change.
const SchemaVersion = 2

// Effort selects a preset search/evaluation effort profile. The profile
// values themselves live in pipeline.EffortFor; state owns the type so
// RunConfig can persist and validate it without an import cycle.
type Effort string

const (
	EffortFast     Effort = "fast"
	EffortProfiled Effort = "profiled"
	EffortDeep     Effort = "deep"
)

// Validate rejects unknown effort names; empty is valid and means profiled.
func (e Effort) Validate() error {
	switch e {
	case "", EffortFast, EffortProfiled, EffortDeep:
		return nil
	}
	return fmt.Errorf("state: unknown effort %q (want fast, profiled, or deep)", e)
}

// ToolPaths locates the external binaries a run drives.
type ToolPaths struct {
	LlamaQuantize   string `json:"llamaQuantize"`
	LlamaPerplexity string `json:"llamaPerplexity"`
	// LlamaIatrix is the llama-imatrix binary, required when any IQ dtype or
	// ImatrixPath output generation is configured.
	LlamaImatrix string `json:"llamaImatrix,omitempty"`
}

func (t ToolPaths) Validate() error {
	for name, p := range map[string]string{
		"llamaQuantize":   t.LlamaQuantize,
		"llamaPerplexity": t.LlamaPerplexity,
	} {
		if p == "" {
			return fmt.Errorf("state: tool path %s is required", name)
		}
		if !filepath.IsAbs(p) {
			return fmt.Errorf("state: tool path %s must be absolute, got %q", name, p)
		}
	}
	if t.LlamaImatrix != "" && !filepath.IsAbs(t.LlamaImatrix) {
		return fmt.Errorf("state: llamaImatrix must be absolute")
	}
	return nil
}

// RunConfig is everything a run needs to execute or resume without CLI
// arguments.
type RunConfig struct {
	SourcePath string    `json:"sourcePath"` // input GGUF
	OutputDir  string    `json:"outputDir"`  // final artifacts
	WorkDir    string    `json:"workDir"`    // intermediates, candidate GGUFs
	Tools      ToolPaths `json:"tools"`

	// ImatrixPath, when set, is a precomputed importance matrix consumed by
	// IQ quant targets.
	ImatrixPath string `json:"imatrixPath,omitempty"`
	// CalibrationCorpus is the text used to build an imatrix or sensitivity
	// priors; EvalCorpus is the held-out text for perplexity/KLD scoring.
	CalibrationCorpus string `json:"calibrationCorpus,omitempty"`
	// SearchCorpus is a tuning-only holdout. New search-enabled runs require
	// it to be distinct from EvalCorpus; empty is retained only so older
	// checkpoints can load and fail with an actionable runtime error.
	SearchCorpus string `json:"searchCorpus,omitempty"`
	EvalCorpus   string `json:"evalCorpus"`
	// DomainEvalCorpora freezes the expected final-validation domain set at
	// plan time so a changed manifest cannot silently shrink gate coverage.
	DomainEvalCorpora map[string]string `json:"domainEvalCorpora,omitempty"`

	// BudgetBytes is the hard size ceiling for the final emitted ARTIFACT:
	// the complete output GGUF file, including its header, KV metadata,
	// tensor-info section, and alignment padding — not just tensor payload
	// bytes. The solver and search account payload bytes against
	// BudgetBytes minus a metadata/padding reserve, and emit hard-fails
	// before publishing anything larger than BudgetBytes.
	BudgetBytes uint64 `json:"budgetBytes"`
	// TargetBPW, when > 0, is the soft average bits-per-weight goal the
	// solver optimizes toward within BudgetBytes.
	TargetBPW float64 `json:"targetBPW,omitempty"`

	Threads int `json:"threads"`
	CtxSize int `json:"ctxSize"` // evaluation context size

	// Gates are acceptance thresholds the final candidate must pass. When
	// empty (and GatesOptOut is false), the engine applies TargetBPW-scaled
	// default gates instead of trivially passing. Unset TargetBPW keeps the
	// Q5 reference (mean-KLD 0.15, p95-KLD 1.0).
	Gates []core.QualityGate `json:"gates,omitempty"`
	// GatesOptOut records an explicit "-gates none": no gates are configured
	// and no effort-profile defaults are applied.
	GatesOptOut bool `json:"gatesOptOut,omitempty"`
	// Effort selects the preset profile supplying defaults for knobs left
	// unset (search bounds, eval chunks, gates, kld engine tuning). Empty
	// means profiled.
	Effort Effort `json:"effort,omitempty"`

	// SearchEnabled is retained for old checkpoints. Plan never sets it.
	SearchEnabled bool `json:"searchEnabled"`
	// MaxSearchIterations bounds the search when enabled.
	MaxSearchIterations int `json:"maxSearchIterations,omitempty"`
}

func (c RunConfig) Validate() error {
	for name, p := range map[string]string{
		"sourcePath": c.SourcePath,
		"outputDir":  c.OutputDir,
		"workDir":    c.WorkDir,
		"evalCorpus": c.EvalCorpus,
	} {
		if p == "" {
			return fmt.Errorf("state: config %s is required", name)
		}
		if !filepath.IsAbs(p) {
			return fmt.Errorf("state: config %s must be absolute, got %q", name, p)
		}
	}
	if err := c.Tools.Validate(); err != nil {
		return err
	}
	if c.ImatrixPath != "" && !filepath.IsAbs(c.ImatrixPath) {
		return fmt.Errorf("state: imatrixPath must be absolute")
	}
	if c.CalibrationCorpus != "" && !filepath.IsAbs(c.CalibrationCorpus) {
		return fmt.Errorf("state: calibrationCorpus must be absolute")
	}
	if c.SearchCorpus != "" && !filepath.IsAbs(c.SearchCorpus) {
		return fmt.Errorf("state: searchCorpus must be absolute")
	}
	for domain, path := range c.DomainEvalCorpora {
		if domain == "" || path == "" || !filepath.IsAbs(path) {
			return fmt.Errorf("state: invalid domain evaluation corpus %q=%q", domain, path)
		}
	}
	if c.BudgetBytes == 0 {
		return fmt.Errorf("state: budgetBytes must be > 0")
	}
	if c.TargetBPW < 0 {
		return fmt.Errorf("state: targetBPW must be >= 0")
	}
	if c.Threads <= 0 {
		return fmt.Errorf("state: threads must be > 0")
	}
	if c.CtxSize <= 0 {
		return fmt.Errorf("state: ctxSize must be > 0")
	}
	for _, g := range c.Gates {
		if err := g.Validate(); err != nil {
			return err
		}
	}
	if err := c.Effort.Validate(); err != nil {
		return err
	}
	if c.SearchEnabled && c.MaxSearchIterations <= 0 {
		return fmt.Errorf("state: searchEnabled requires maxSearchIterations > 0")
	}
	if c.SearchEnabled && c.SearchCorpus != "" && filepath.Clean(c.SearchCorpus) == filepath.Clean(c.EvalCorpus) {
		return fmt.Errorf("state: searchCorpus must be distinct from evalCorpus")
	}
	return nil
}

// Run is the complete resumable state of one optimization run.
type Run struct {
	Version   int       `json:"version"`
	RunID     string    `json:"runID"`
	Config    RunConfig `json:"config"`
	CreatedAt time.Time `json:"createdAt"`
	UpdatedAt time.Time `json:"updatedAt"`
	// Completed lists finished stages in order; resume starts at the first
	// stage in core.StageOrder not present here.
	Completed []core.Stage `json:"completed"`
	// Artifacts maps stage -> primary output path (bank manifest, profile
	// JSON, quantized GGUF, eval report).
	Artifacts map[core.Stage]string `json:"artifacts,omitempty"`
	// Bank, once assembled, is persisted so later stages never re-parse GGUF.
	Bank *core.TensorBank `json:"bank,omitempty"`
	// Manifest is the frozen per-tensor selection from the solve stage.
	Manifest *core.SelectionManifest `json:"manifest,omitempty"`
	// Measurements accumulates scored evaluations with provenance.
	Measurements []core.Measurement `json:"measurements,omitempty"`
	// Evals accumulates legacy lightweight metrics for candidate profiles.
	Evals []core.EvalResult `json:"evals,omitempty"`
	// SearchHistory records accepted/rejected kld search moves.
	SearchHistory []kld.Step `json:"searchHistory,omitempty"`
	// MoveGroups records committed/reverted grouped moves.
	MoveGroups []core.MoveGroup `json:"moveGroups,omitempty"`
	// BestProfileID is the current incumbent.
	BestProfileID string `json:"bestProfileID,omitempty"`
}

// NewRun creates a run positioned before StageAssemble. Passing a config
// validates it immediately; a run created without one (the current CLI
// skeleton does this) cannot complete stages or resume — see Validate.
// Implementation agents should always pass a config.
func NewRun(runID string, now time.Time, configs ...RunConfig) (*Run, error) {
	if err := ValidRunID(runID); err != nil {
		return nil, err
	}
	if len(configs) > 1 {
		return nil, fmt.Errorf("state: at most one config")
	}
	r := &Run{
		Version:   SchemaVersion,
		RunID:     runID,
		CreatedAt: now.UTC(),
		UpdatedAt: now.UTC(),
		Artifacts: map[core.Stage]string{},
	}
	if len(configs) == 1 {
		if err := configs[0].Validate(); err != nil {
			return nil, err
		}
		r.Config = configs[0]
	}
	return r, nil
}

func (r *Run) Validate() error {
	if r.Version != SchemaVersion {
		return fmt.Errorf("state: unsupported version %d", r.Version)
	}
	if err := ValidRunID(r.RunID); err != nil {
		return err
	}
	// Config is validated whenever present, and is mandatory once the run has
	// executed any stage: a partially-executed checkpoint must carry
	// everything resume needs.
	if r.Config.SourcePath != "" || len(r.Completed) > 0 {
		if err := r.Config.Validate(); err != nil {
			return err
		}
	}
	last := -1
	for _, s := range r.Completed {
		if !s.Valid() {
			return fmt.Errorf("state: invalid stage %q", s)
		}
		idx := core.StageIndex(s)
		if idx <= last {
			return fmt.Errorf("state: stages out of order at %q", s)
		}
		last = idx
	}
	for _, m := range r.Measurements {
		if err := m.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// NextStage returns the first unfinished stage, or false when the run is done.
func (r *Run) NextStage() (core.Stage, bool) {
	done := make(map[core.Stage]bool, len(r.Completed))
	for _, s := range r.Completed {
		done[s] = true
	}
	for _, s := range core.StageOrder {
		if !done[s] {
			return s, true
		}
	}
	return "", false
}

// MarkComplete records a finished stage and its primary artifact. Stages must
// complete in canonical order.
func (r *Run) MarkComplete(s core.Stage, artifactPath string, now time.Time) error {
	next, ok := r.NextStage()
	if !ok {
		return fmt.Errorf("state: run already complete")
	}
	if s != next {
		return fmt.Errorf("state: cannot complete %q before %q", s, next)
	}
	if err := r.Config.Validate(); err != nil {
		return fmt.Errorf("state: cannot complete stages without a valid config: %w", err)
	}
	r.Completed = append(r.Completed, s)
	if artifactPath != "" {
		// A checkpoint saved before any stage completed serializes no
		// Artifacts map (omitempty), so a reloaded run can carry nil.
		// Initialize lazily so recording the first artifact is always safe.
		if r.Artifacts == nil {
			r.Artifacts = make(map[core.Stage]string)
		}
		r.Artifacts[s] = artifactPath
	}
	r.UpdatedAt = now.UTC()
	return nil
}

// Store persists runs as single JSON files, atomically replaced on save.
type Store struct {
	Dir string
}

// Path returns the checkpoint file for a run id. The id is validated first:
// an unsafe id (path separators, traversal, metacharacters) is rejected
// rather than interpolated into a filesystem path.
func (s Store) Path(runID string) (string, error) {
	if err := ValidRunID(runID); err != nil {
		return "", err
	}
	return filepath.Join(s.Dir, runID+".json"), nil
}

// Save validates and atomically writes the run (temp file + rename).
func (s Store) Save(r *Run) error {
	if err := r.Validate(); err != nil {
		return err
	}
	if s.Dir == "" {
		return fmt.Errorf("state: store dir not set")
	}
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return fmt.Errorf("state: mkdir: %w", err)
	}
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("state: marshal: %w", err)
	}
	path, err := s.Path(r.RunID)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("state: write tmp: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("state: rename: %w", err)
	}
	return nil
}

// Load reads and validates a checkpoint. The returned run carries its full
// RunConfig, so callers resume without any CLI arguments. Artifacts is always
// non-nil on the returned run even when the checkpoint omitted it (an empty
// map serializes to nothing under omitempty), so MarkComplete is safe on
// every reload path. Run IDs are validated before any filesystem path is
// derived from them (path-traversal rejection).
func (s Store) Load(runID string) (*Run, error) {
	path, err := s.Path(runID)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("state: load: %w", err)
	}
	var r Run
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("state: unmarshal: %w", err)
	}
	if err := r.Validate(); err != nil {
		return nil, err
	}
	if r.Artifacts == nil {
		r.Artifacts = make(map[core.Stage]string)
	}
	return &r, nil
}
