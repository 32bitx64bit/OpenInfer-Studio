// Package pipeline wires the quantlab library stages into the resumable
// end-to-end optimizer driven by the quantlab CLI. Every stage boundary is
// checkpointed through state.Run.MarkComplete with its primary artifact, all
// external process execution is argv-only via orchestrate.Runner, and every
// measurement is recorded with provenance so a resumed run never remeasures a
// candidate that is already recorded in the checkpoint.
package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"quantlab/anchor"
	"quantlab/core"
	"quantlab/orchestrate"
	"quantlab/profile"
	"quantlab/state"
	"quantlab/tensorbank"
)

// OSRunner returns the production runner: argv-only exec.CommandContext with
// an explicit environment, bounded output capture, and process-group cleanup.
func OSRunner() orchestrate.OSRunner {
	return orchestrate.OSRunner{Env: os.Environ(), IdleTimeout: time.Hour}
}

// ExtraConfig carries run options that have no field on state.RunConfig. It
// is persisted as a sidecar JSON next to the checkpoint so resume still needs
// zero CLI flags.
type ExtraConfig struct {
	Chunks int `json:"chunks,omitempty"`
	// ExactEstimatorOff disables the solve-time exact loss table.
	ExactEstimatorOff bool `json:"exactEstimatorOff,omitempty"`
	// ScaleFold forces the AWQ-style scale-folding stage on. NoScaleFold
	// opts out. Profiled/deep enable folding via the effort profile.
	ScaleFold   bool `json:"scaleFold,omitempty"`
	NoScaleFold bool `json:"noScaleFold,omitempty"`
	// Hadamard / CSK / FTI: force-on extras (CLI/Reconstruct). No* opts out.
	// ProbeKLD: force-on; profiled/deep enable it via the effort profile.
	Hadamard   bool `json:"hadamard,omitempty"`
	NoHadamard bool `json:"noHadamard,omitempty"`
	CSK        bool `json:"csk,omitempty"`
	NoCSK      bool `json:"noCSK,omitempty"`
	// CSKMaxWorkingSetBytes bounds live reconstruction memory. Studio sets a
	// host-aware value; zero uses reconstruct's conservative default.
	CSKMaxWorkingSetBytes uint64 `json:"cskMaxWorkingSetBytes,omitempty"`
	FTI                   bool   `json:"fti,omitempty"`
	NoFTI                 bool   `json:"noFTI,omitempty"`
	ProbeKLD              bool   `json:"probeKLD,omitempty"`
	NoProbeKLD            bool   `json:"noProbeKLD,omitempty"`
	NoPermute             bool   `json:"noPermute,omitempty"`
	NoMagR                bool   `json:"noMagR,omitempty"`
	NoLWC                 bool   `json:"noLWC,omitempty"`
	NoFreqVQ              bool   `json:"noFreqVQ,omitempty"`
	NoGPTQ                bool   `json:"noGPTQ,omitempty"`
	NoViterbi             bool   `json:"noViterbi,omitempty"`
	NoExpertCentroid      bool   `json:"noExpertCentroid,omitempty"`
	// Encode / FreqVQ / ExpertCentroid are opt-in. They are not default-on
	// for profiled/deep: GPTQ does not pack IQ (the bulk of a 3.5 bpw
	// hybrid), and FreqVQ/centroid are skip-safe extras, not the quality lever.
	Encode         bool `json:"encode,omitempty"`
	FreqVQ         bool `json:"freqVQ,omitempty"`
	ExpertCentroid bool `json:"expertCentroid,omitempty"`
	// FoldedSourcePath / FoldedImatrixPath record the fold redirect once
	// applied (persisted sidecar; resume-safe).
	FoldedSourcePath  string `json:"foldedSourcePath,omitempty"`
	FoldedImatrixPath string `json:"foldedImatrixPath,omitempty"`
	// PrivateSourcePath is a job-owned clone of the library GGUF so fold and
	// reconstruct can rewrite in place without mutating the user's file.
	PrivateSourcePath string `json:"privateSourcePath,omitempty"`
	// FTIImatrixPath is the sharpened imatrix GGUF.
	FTIImatrixPath string `json:"ftiImatrixPath,omitempty"`
	// ReconstructedSourcePath / ReconstructedImatrixPath record the
	// Hadamard/CSK rewrite (persisted sidecar; resume-safe).
	ReconstructedSourcePath  string `json:"reconstructedSourcePath,omitempty"`
	ReconstructedImatrixPath string `json:"reconstructedImatrixPath,omitempty"`
	// PayloadSHA is the hex digest of payloadSource() after fold/reconstruct.
	// SelectionManifest.SourceSHA must match this (not TensorBank.SHA256)
	// so tensorbank.Build accepts the rewritten primary GGUF.
	PayloadSHA     string `json:"payloadSHA,omitempty"`
	PayloadSHAPath string `json:"payloadSHAPath,omitempty"`
	// MTPStrippedPath is the payload GGUF with fused NextN/MTP tensors
	// removed (Q2-class jobs). Highest-priority payloadSource when present.
	MTPStrippedPath string `json:"mtpStrippedPath,omitempty"`
}

// Engine executes the remaining stages of a loaded run.
type Engine struct {
	Store  state.Store
	Run    *state.Run
	Runner orchestrate.Runner
	Out    io.Writer
	// DryRun plans every remaining stage (including quantize --dry-run where
	// the binary advertises it) without executing tools or writing artifacts,
	// and never persists stage completions.
	DryRun bool
	// StageLimit bounds the number of stages executed by one Resume call.
	StageLimit int
	// Extra holds sidecar options loaded by NewEngine.
	Extra ExtraConfig
	// Obs, when non-nil, receives stage-boundary, byte-progress, and
	// measurement events. Nil preserves silent behavior.
	Obs Observer

	now   func() time.Time
	stage core.Stage // stage currently executing (for progress attribution)
	capsQ *orchestrate.Capabilities
	capsP *orchestrate.Capabilities

	// lossCache is the in-memory measured-loss cache loaded at solve.
	// Nil when no cache is present.
	lossCache *profile.Cache

	hashMu     sync.Mutex
	fileHashes map[string]cachedFileHash
}

type cachedFileHash struct {
	size    int64
	modTime int64
	sha     string
}

// NewEngine wires an engine over a loaded run, loading the sidecar extra
// config when present.
func NewEngine(store state.Store, r *state.Run, runner orchestrate.Runner, out io.Writer) (*Engine, error) {
	if runner == nil {
		return nil, fmt.Errorf("pipeline: nil runner")
	}
	e := &Engine{Store: store, Run: r, Runner: runner, Out: out, now: time.Now}
	if err := e.loadExtra(); err != nil {
		return nil, err
	}
	return e, nil
}

func (e *Engine) extraPath() string {
	return filepath.Join(e.Store.Dir, e.Run.RunID+".extra.json")
}

func (e *Engine) loadExtra() error {
	data, err := os.ReadFile(e.extraPath())
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("pipeline: read extra config: %w", err)
	}
	if err := json.Unmarshal(data, &e.Extra); err != nil {
		return fmt.Errorf("pipeline: decode extra config: %w", err)
	}
	return nil
}

func (e *Engine) saveExtra() error {
	data, err := json.MarshalIndent(&e.Extra, "", "  ")
	if err != nil {
		return err
	}
	tmp := e.extraPath() + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, e.extraPath())
}

// runOK executes an invocation through the Runner interface and requires
// exit code 0 (the Runner interface itself reports nonzero exits as errors
// or via Result.ExitCode).
func runOK(ctx context.Context, r orchestrate.Runner, iv orchestrate.Invocation) (orchestrate.Result, error) {
	res, err := r.Run(ctx, iv)
	if err != nil {
		return res, err
	}
	if res.ExitCode != 0 {
		return res, fmt.Errorf("%s exited %d", iv.Path, res.ExitCode)
	}
	return res, nil
}

func (e *Engine) printf(format string, args ...any) {
	w := e.Out
	if w == nil {
		w = os.Stdout
	}
	fmt.Fprintf(w, format, args...)
}

// Resume executes remaining stages in core.StageOrder, checkpointing after
// each stage boundary. An interruption at any point leaves a loadable
// checkpoint that resumes cleanly.
func (e *Engine) Resume(ctx context.Context) error {
	// A checkpointed bank is only valid for the exact source it was assembled
	// from. Check before consulting completed stages or invoking any tool.
	if e.Run.Bank != nil && e.Run.Bank.SHA256 != "" {
		s, err := tensorbank.OpenSource(e.Run.Config.SourcePath)
		if err != nil {
			return fmt.Errorf("pipeline: resume source: %w", err)
		}
		sha, hashErr := s.SHA256()
		_ = s.Close()
		if hashErr != nil {
			return fmt.Errorf("pipeline: hash resume source: %w", hashErr)
		}
		if sha != e.Run.Bank.SHA256 {
			return fmt.Errorf("pipeline: source SHA256 changed since assemble (checkpoint %s, current %s); checkpoints were left intact", e.Run.Bank.SHA256, sha)
		}
	}
	executed := 0
	local := map[core.Stage]bool{}
	for _, s := range e.Run.Completed {
		local[s] = true
	}
	for _, next := range core.StageOrder {
		if local[next] {
			continue
		}
		if e.StageLimit > 0 && executed >= e.StageLimit {
			e.printf("run %s: stage limit reached (next: %s)\n", e.Run.RunID, next)
			return nil
		}
		e.printf("run %s: stage %s\n", e.Run.RunID, next)
		e.stage = next
		e.obsStarted(next)
		if err := e.executeStage(ctx, next); err != nil {
			return fmt.Errorf("stage %s: %w", next, err)
		}
		e.obsCompleted(next)
		local[next] = true
		if !e.DryRun {
			if err := e.Store.Save(e.Run); err != nil {
				return fmt.Errorf("stage %s: checkpoint save: %w", next, err)
			}
		}
		executed++
	}
	// A crash between MarkComplete(emit) and scratch cleanup leaves a fully
	// complete run with leftover scratch; clean idempotently here.
	if !e.DryRun {
		e.cleanupScratch()
	}
	e.printf("run %s: all stages complete\n", e.Run.RunID)
	return nil
}

// cleanupScratch removes intermediate working artifacts (anchor GGUFs,
// baseline logits, search scratch). Idempotent; used by emit after the
// checkpoint records completion so an interrupted cleanup still finishes on
// the next resume.
func (e *Engine) cleanupScratch() {
	os.RemoveAll(e.anchorDir())
	if matches, err := filepath.Glob(filepath.Join(e.workDir(), "baseline-logits*.bin*")); err == nil {
		for _, path := range matches {
			os.Remove(path)
		}
	}
	os.RemoveAll(e.searchDir())
	os.Remove(filepath.Join(e.workDir(), "search-checkpoint.json"))
	os.Remove(filepath.Join(e.workDir(), "search-final.json"))
	for _, path := range []string{
		filepath.Join(e.workDir(), "candidate.gguf"),
		filepath.Join(e.workDir(), "final.gguf"),
		e.Extra.FoldedSourcePath,
		e.Extra.FoldedImatrixPath,
		e.Extra.FTIImatrixPath,
		e.Extra.ReconstructedSourcePath,
		e.Extra.ReconstructedImatrixPath,
		e.Extra.MTPStrippedPath,
		e.Extra.PrivateSourcePath,
	} {
		if generatedUnder(e.workDir(), path) {
			os.Remove(path)
		}
	}
}

func (e *Engine) executeStage(ctx context.Context, s core.Stage) error {
	switch s {
	case core.StageAssemble:
		return e.stageAssemble(ctx)
	case core.StageAnchor:
		return e.stageAnchor(ctx)
	case core.StageSolve:
		return e.stageSolve(ctx)
	case core.StageQuantize:
		if err := e.stageQuantize(ctx); err != nil {
			return err
		}
		return e.persistLossCache()
	case core.StageEvaluate:
		return e.stageEvaluate(ctx)
	case core.StageSearch:
		return e.stageSearch(ctx)
	case core.StageEmit:
		if err := e.stageEmit(ctx); err != nil {
			return err
		}
		return e.clearRunLossCache()
	}
	return fmt.Errorf("pipeline: unknown stage %q", s)
}

// complete records a finished stage; in dry-run the checkpoint is left
// untouched so a later real resume re-executes the stage.
func (e *Engine) complete(s core.Stage, artifact string) error {
	if e.DryRun {
		return nil
	}
	return e.Run.MarkComplete(s, artifact, e.now())
}

// caps probes (once) the advertised capabilities of a tool binary.
func (e *Engine) caps(ctx context.Context, tool orchestrate.Tool) (*orchestrate.Capabilities, error) {
	switch tool {
	case orchestrate.ToolLlamaQuantize:
		if e.capsQ != nil {
			return e.capsQ, nil
		}
		c, err := orchestrate.ProbeCapabilities(ctx, e.Runner, tool, e.Run.Config.Tools.LlamaQuantize)
		if err != nil {
			return nil, err
		}
		e.capsQ = &c
		return e.capsQ, nil
	case orchestrate.ToolPerplexity:
		if e.capsP != nil {
			return e.capsP, nil
		}
		c, err := orchestrate.ProbeCapabilities(ctx, e.Runner, tool, e.Run.Config.Tools.LlamaPerplexity)
		if err != nil {
			return nil, err
		}
		e.capsP = &c
		return e.capsP, nil
	}
	return nil, fmt.Errorf("pipeline: unknown tool %q", tool)
}

func (e *Engine) workDir() string    { return e.Run.Config.WorkDir }
func (e *Engine) anchorDir() string  { return filepath.Join(e.workDir(), "anchors") }
func (e *Engine) searchDir() string  { return filepath.Join(e.workDir(), "search") }
func (e *Engine) logitsPath() string { return filepath.Join(e.workDir(), "baseline-logits.bin") }

// payloadBudget converts the artifact BudgetBytes into the payload-byte
// budget the solver and search operate under, by deducting a conservative
// overhead reserve (header, KV metadata, tensor infos, alignment padding).
// It never returns 0-for-unlimited: a budget too small to cover overhead
// yields 1, the tightest payload budget.
func (e *Engine) payloadBudget() uint64 {
	budget := e.Run.Config.BudgetBytes
	reserve := tensorbank.OverheadReserve(e.Run.Bank)
	if reserve >= budget {
		return 1
	}
	return budget - reserve
}

// solveBudget is the payload ceiling the rate-distortion solver may fill.
func (e *Engine) solveBudget() uint64 {
	return e.payloadBudget()
}

func (e *Engine) stageSearch(ctx context.Context) error {
	_ = ctx
	if e.DryRun {
		e.printf("plan: search skipped\n")
		return e.complete(core.StageSearch, "")
	}
	e.printf("  search: skipped\n")
	if err := e.ingestSearchLossCache(); err != nil {
		return err
	}
	if err := e.fitCalibration(); err != nil {
		return err
	}
	return e.complete(core.StageSearch, "")
}

func (e *Engine) writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// stageAssemble builds the TensorBank from the source GGUF. The bank is
// persisted inside the checkpoint so later stages never re-parse the source.
func (e *Engine) stageAssemble(ctx context.Context) error {
	src := e.Run.Config.SourcePath
	s, err := tensorbank.OpenSource(src)
	if err != nil {
		return err
	}
	defer s.Close()
	if e.DryRun {
		pf, err := tensorbank.Parse(s)
		if err != nil {
			return err
		}
		bank := &core.TensorBank{
			SourcePath:    src,
			ModelID:       pf.ModelID,
			Alignment:     uint64(pf.Alignment),
			KVMetadataLen: uint64(len(pf.KVBytes)),
			Arch:          pf.Architecture,
		}
		for _, t := range pf.Tensors {
			bank.Tensors = append(bank.Tensors, core.TensorDesc{
				Name: t.Name, DType: t.DType, Shape: t.Shape,
				Offset: t.RelOffset, Length: t.Length, Elements: t.Elements,
			})
		}
		if err := bank.Validate(); err != nil {
			return err
		}
		e.Run.Bank = bank
		e.printf("plan: assemble bank from %s\n", src)
		if err := e.stageFTI(ctx); err != nil {
			return err
		}
		if err := e.ensurePrivatePayload(); err != nil {
			return err
		}
		if err := e.stageFold(ctx); err != nil {
			return err
		}
		if err := e.stageReconstruct(ctx); err != nil {
			return err
		}
		if err := e.stageDropMTP(ctx); err != nil {
			return err
		}
		return e.complete(core.StageAssemble, "")
	}
	bank, err := tensorbank.NewAssembler().Assemble(s, src, "")
	if err != nil {
		return err
	}
	e.Run.Bank = bank
	if err := e.stageFTI(ctx); err != nil {
		return err
	}
	if err := e.ensurePrivatePayload(); err != nil {
		return err
	}
	if err := e.stageFold(ctx); err != nil {
		return err
	}
	if err := e.stageReconstruct(ctx); err != nil {
		return err
	}
	if err := e.stageDropMTP(ctx); err != nil {
		return err
	}
	if !e.DryRun {
		if _, err := e.payloadIdentitySHA(); err != nil {
			return err
		}
		if e.Extra.PayloadSHA != "" && e.Run.Bank != nil && e.Extra.PayloadSHA != e.Run.Bank.SHA256 {
			e.printf("  payload: rewritten source sha %s\n", e.Extra.PayloadSHA[:16])
		}
	}
	artifact := filepath.Join(e.workDir(), "bank.json")
	if err := e.writeJSON(artifact, e.Run.Bank); err != nil {
		return err
	}
	e.printf("  bank: %d tensors, %d bytes payload, sha %s\n",
		len(e.Run.Bank.Tensors), e.Run.Bank.TotalBytes(), e.Run.Bank.SHA256[:16])
	return e.complete(core.StageAssemble, artifact)
}

// stageAnchor derives the anchor set (hard floors + soft priors) for the
// assembled bank. The derivation is deterministic, so later stages re-derive
// rather than depend on a persisted set; the artifact is the audit record.
func (e *Engine) stageAnchor(ctx context.Context) error {
	set, err := anchor.Derive(e.Run.Bank, nil, anchor.PolicyForBPW(e.Run.Config.TargetBPW))
	if err != nil {
		return err
	}
	if e.DryRun {
		e.printf("plan: %d hard floors, %d soft priors\n", len(set.Hard), len(set.Priors))
		return e.complete(core.StageAnchor, "")
	}
	artifact := filepath.Join(e.workDir(), "anchors.json")
	if err := e.writeJSON(artifact, set); err != nil {
		return err
	}
	e.printf("  anchors: %d hard floors, %d soft priors\n", len(set.Hard), len(set.Priors))
	return e.complete(core.StageAnchor, artifact)
}

// candidateDTypes restricts the solver lattice to explicit per-tensor base
// types (recipe labels would produce mixed upgrade rules the streaming
// assembler cannot source from a single anchor file), plus IQ types only when
// an importance matrix is configured. A non-nil effort-profile lattice
// further intersects the result.
func (e *Engine) candidateDTypes() []core.DType {
	hasImatrix := e.imatrixPath() != ""
	var out []core.DType
	for _, d := range core.QuantDTypes {
		if d.IsRecipeLabel() {
			continue
		}
		if d.RequiresImatrix() && !hasImatrix {
			continue
		}
		if _, ok := d.Geometry(); !ok {
			continue
		}
		out = append(out, d)
	}
	allowed := e.effortProfile().Candidates
	if len(allowed) == 0 {
		return out
	}
	set := make(map[core.DType]bool, len(allowed))
	for _, d := range allowed {
		set[d.BaseTensorType()] = true
	}
	var filt []core.DType
	for _, d := range out {
		if set[d] {
			filt = append(filt, d)
		}
	}
	return filt
}

// stageSolve runs the rate-distortion solver and freezes the selection
// manifest that quantize assembles.
func (e *Engine) stageSolve(ctx context.Context) error {
	cfg := e.Run.Config
	bank := e.Run.Bank
	set, err := anchor.Derive(bank, nil, anchor.PolicyForBPW(cfg.TargetBPW))
	if err != nil {
		return err
	}
	cache := e.loadLossCache()
	e.lossCache = cache
	var imatrix map[string]profile.ImatrixStats
	if ip := e.imatrixPath(); ip != "" {
		stats, err := e.loadSolverImatrix(bank)
		if err != nil {
			return fmt.Errorf("pipeline: load imatrix: %w", err)
		}
		imatrix = stats
		if len(stats) == 0 {
			e.printf("  imatrix: no per-tensor stats in %s (solver uses heuristic)\n", ip)
		} else {
			e.printf("  imatrix: %d tensors from %s\n", len(stats), ip)
		}
	}
	// BudgetBytes caps the FINAL artifact (metadata + payload). The solver
	// plans payload bytes only, so deduct a conservative overhead reserve;
	// emit later hard-checks the exact artifact size.
	req := profile.Request{
		Bank:        bank,
		Anchors:     set,
		Candidates:  e.candidateDTypes(),
		BudgetBytes: e.solveBudget(),
		TargetBPW:   cfg.TargetBPW,
		Cache:       cache,
		Imatrix:     imatrix,
		Calibration: e.loadCalibration(bank),
	}
	if e.exactEstimatorEnabled() {
		if !e.DryRun {
			tableBank := bank
			if fp := e.payloadSource(); fp != tableBank.SourcePath {
				sb := *tableBank
				sb.SourcePath = fp
				tableBank = &sb
			}
			existing := e.loadExactLoss(bank)
			partial := make(map[string]map[core.DType]float64, len(existing))
			for name, losses := range existing {
				partial[name] = make(map[core.DType]float64, len(losses))
				for d, v := range losses {
					partial[name][d] = v
				}
			}
			signature, err := e.exactLossSignature(bank)
			if err != nil {
				return fmt.Errorf("pipeline: exact loss identity: %w", err)
			}
			var partialMu sync.Mutex
			table, err := profile.BuildExactLossTableCfg(tableBank, req.Candidates, imatrix,
				func(done, total int64) {
					if total <= 0 {
						return
					}
					frac := float64(done) / float64(total)
					if frac > 1 {
						frac = 1
					}
					e.obsProgress(core.StageSolve, frac, "computing exact loss table")
				}, profile.ExactConfig{
					ProbeKLD: e.probeKLDEnabled(),
					Context:  ctx,
					Existing: existing,
					OnTensor: func(name string, losses map[core.DType]float64) error {
						partialMu.Lock()
						defer partialMu.Unlock()
						copyLosses := make(map[core.DType]float64, len(losses))
						for d, v := range losses {
							copyLosses[d] = v
						}
						partial[name] = copyLosses
						return e.saveExactLossWithSignature(bank, signature, partial)
					},
				})
			if err != nil {
				return fmt.Errorf("pipeline: exact loss table: %w", err)
			}
			covered := 0
			for _, m := range table {
				if len(m) > 0 {
					covered++
				}
			}
			req.ExactLoss = table
			if err := e.saveExactLossWithSignature(bank, signature, table); err != nil {
				return fmt.Errorf("pipeline: persist exact loss table: %w", err)
			}
			e.printf("  exact loss table: %d tensors covered\n", covered)
		} else {
			e.printf("plan: exact loss table over %d candidate dtypes\n", len(req.Candidates))
		}
	}
	res, err := profile.Solve(req)
	if err != nil {
		return err
	}
	e.printf("  profile %s: %d assignments, %d bytes (budget %d, slop %d)\n",
		res.Profile.ID, len(res.Profile.Assignments), res.Profile.EstimatedBytes,
		res.Diag.BudgetBytes, res.Diag.SlopBytes)
	e.Run.Manifest = res.Manifest
	if err := e.bindManifestSource(e.Run.Manifest); err != nil {
		return err
	}
	e.Run.BestProfileID = res.Profile.ID
	if e.DryRun {
		return e.complete(core.StageSolve, "")
	}
	artifact := filepath.Join(e.workDir(), "profile.json")
	if err := e.writeJSON(artifact, struct {
		Profile     *core.Profile           `json:"profile"`
		Manifest    *core.SelectionManifest `json:"manifest"`
		Diagnostics profile.Diagnostics     `json:"diagnostics"`
	}{
		Profile:     res.Profile,
		Manifest:    res.Manifest,
		Diagnostics: res.Diag,
	}); err != nil {
		return err
	}
	return e.complete(core.StageSolve, artifact)
}

// loadLossCache loads the optional measured-loss cache from the work dir,
// falling back to the store sidecar (runID.loss-cache.json) so a resumed
// job still sees search-written entries after scratch cleanup. A missing
// file yields nil; identity mismatch also yields nil so a stale cache never
// silently leaks, and never fails the run.
func (e *Engine) loadLossCache() *profile.Cache {
	if e.Run == nil || e.Run.Bank == nil {
		return nil
	}
	bank := e.Run.Bank
	try := func(path string) *profile.Cache {
		if path == "" {
			return nil
		}
		f, err := os.Open(path)
		if err != nil {
			return nil
		}
		defer f.Close()
		c, err := profile.LoadCache(f, bank.ModelID, bank.SHA256)
		if err != nil {
			return nil
		}
		return c
	}
	if c := try(filepath.Join(e.workDir(), "loss-cache.json")); c != nil {
		return c
	}
	if e.Store.Dir != "" && e.Run.RunID != "" {
		return try(filepath.Join(e.Store.Dir, e.Run.RunID+".loss-cache.json"))
	}
	return nil
}

// hasMeasurement reports whether a (profileID, metric) measurement is already
// recorded; this is the resume-without-remeasure contract.
func (e *Engine) hasMeasurement(profileID string, metric core.MetricKind) bool {
	_, ok := e.measurement(profileID, metric)
	return ok
}

// exactEstimatorEnabled reports whether the solve-time exact loss table
// runs for this effort profile and config.
func (e *Engine) exactEstimatorEnabled() bool {
	if e.Extra.ExactEstimatorOff {
		return false
	}
	return e.effortProfile().ExactEstimator
}

// calibrationPath is where the fitted calibration store lives (store
// sidecar, surviving scratch cleanup).
func (e *Engine) calibrationPath() string {
	if e.Run == nil || e.Run.RunID == "" || e.Store.Dir == "" {
		return ""
	}
	return filepath.Join(e.Store.Dir, e.Run.RunID+".calibration.json")
}

// loadCalibration resolves the store's hierarchical levels for this model.
// Missing/corrupt/identity-mismatched stores yield nil (fail-open, like the
// loss cache: a run never dies because priors are absent).
func (e *Engine) loadCalibration(bank *core.TensorBank) *profile.Calibration {
	p := e.calibrationPath()
	if p == "" || bank == nil {
		return nil
	}
	f, err := os.Open(p)
	if err != nil {
		return nil
	}
	defer f.Close()
	s, err := profile.LoadCalibration(f, bank.ModelID, bank.SHA256)
	if err != nil || s == nil {
		return nil
	}
	return s.Resolve(s.Arch)
}

func (e *Engine) measurement(profileID string, metric core.MetricKind) (core.Measurement, bool) {
	for i := len(e.Run.Measurements) - 1; i >= 0; i-- {
		m := e.Run.Measurements[i]
		if m.ProfileID == profileID && m.Metric == metric {
			return m, true
		}
	}
	return core.Measurement{}, false
}

func (e *Engine) recordMeasurement(m core.Measurement) error {
	if err := m.Validate(); err != nil {
		return err
	}
	e.Run.Measurements = append(e.Run.Measurements, m)
	e.obsMeasurement(m)
	return nil
}

// ParseGates parses a -gates flag value: comma-separated metric thresholds.
//
//	mean-kld=<max>             gate on mean KLD
//	p95-kld=<max>              gate on 95th-percentile KLD
//	max-kld=<max>              gate on maximum per-token KLD
//	rms-delta-p=<max>          gate on RMS probability drift
//	top1-disagreement=<max>    gate on argmax disagreement fraction
//	perplexity-delta=<max>     gate on perplexity regression
func ParseGates(spec string) ([]core.QualityGate, error) {
	if strings.TrimSpace(spec) == "" {
		return nil, nil
	}
	var gates []core.QualityGate
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		k, v, ok := strings.Cut(part, "=")
		if !ok {
			return nil, fmt.Errorf("pipeline: gate %q: want metric=value", part)
		}
		val, err := strconv.ParseFloat(v, 64)
		if err != nil || val < 0 {
			return nil, fmt.Errorf("pipeline: gate %q: invalid threshold %q", part, v)
		}
		switch k {
		case "mean-kld":
			gates = append(gates, core.QualityGate{Metric: core.MetricKLD, MaxDelta: val})
		case "p95-kld":
			// Absolute cap on the p95 KLD measurement; MaxDelta is disabled
			// (always satisfied) so only the absolute bound applies.
			gates = append(gates, core.QualityGate{Metric: core.MetricP95KLD, MaxAbsolute: val, MaxDelta: math.MaxFloat64})
		case "cvar-kld":
			gates = append(gates, core.QualityGate{Metric: core.MetricCVaRKLD, MaxAbsolute: val, MaxDelta: math.MaxFloat64})
		case "worst-domain-kld":
			gates = append(gates, core.QualityGate{Metric: core.MetricWorstDomainKLD, MaxDelta: val})
		case "max-kld":
			gates = append(gates, core.QualityGate{Metric: core.MetricMaxKLD, MaxDelta: val})
		case "rms-delta-p":
			gates = append(gates, core.QualityGate{Metric: core.MetricRMSDeltaP, MaxDelta: val})
		case "top1-disagreement":
			gates = append(gates, core.QualityGate{Metric: core.MetricTop1Disagreement, MaxDelta: val})
		case "perplexity-delta":
			gates = append(gates, core.QualityGate{Metric: core.MetricPerplexity, MaxDelta: val})
		default:
			return nil, fmt.Errorf("pipeline: gate %q: unknown metric %q", part, k)
		}
	}
	for _, g := range gates {
		if err := g.Validate(); err != nil {
			return nil, err
		}
	}
	return gates, nil
}
