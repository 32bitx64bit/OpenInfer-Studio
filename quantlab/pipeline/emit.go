package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"quantlab/core"
	"quantlab/orchestrate"
	"quantlab/state"
	"quantlab/tensorbank"
)

// RecipeSidecar is the .oid-plan.json recipe written next to the final GGUF:
// everything needed to reproduce the artifact.
type RecipeSidecar struct {
	RunID       string              `json:"runID"`
	ProfileID   string              `json:"profileID"`
	SourcePath  string              `json:"sourcePath"`
	SourceSHA   string              `json:"sourceSHA,omitempty"`
	BudgetBytes uint64              `json:"budgetBytes"`
	TotalBytes  uint64              `json:"totalBytes"`
	Assignments []core.TensorOption `json:"assignments"`
	Tools       state.ToolPaths     `json:"tools"`
	CreatedAt   time.Time           `json:"createdAt"`
}

// GateReport is one acceptance-gate outcome in the run report.
type GateReport struct {
	Metric      core.MetricKind `json:"metric"`
	MaxDelta    float64         `json:"maxDelta"`
	MaxAbsolute float64         `json:"maxAbsolute,omitempty"`
	Measured    bool            `json:"measured"`
	Value       float64         `json:"value,omitempty"`
	Delta       float64         `json:"delta,omitempty"`
	Pass        bool            `json:"pass"`
}

// RunReport is the final JSON summary of a completed run.
type RunReport struct {
	RunID       string       `json:"runID"`
	CreatedAt   time.Time    `json:"createdAt"`
	FinishedAt  time.Time    `json:"finishedAt"`
	Source      reportFile   `json:"source"`
	Output      reportFile   `json:"output"`
	BudgetBytes uint64       `json:"budgetBytes"`
	ProfileID   string       `json:"profileID"`
	Gates       []GateReport `json:"gates,omitempty"`
	GatesPass   bool         `json:"gatesPass"`
	// GatesConfigured reports how many gates the run was configured with.
	// It is 0 only on explicit opt-out ("-gates none"); otherwise unset
	// gates resolve to the effort profile defaults. GatesPass is vacuously
	// true when it is 0.
	GatesConfigured int                `json:"gatesConfigured"`
	Measurements    []core.Measurement `json:"measurements"`
	Search          searchSummary      `json:"search"`
}

// searchSummary is retained on the run report so old checkpoints that still
// carry a StageSearch artifact decode. New runs leave it zero.
type searchSummary struct {
	BestProfileID string           `json:"bestProfileID"`
	Steps         int              `json:"steps"`
	Evaluations   int              `json:"evaluations"`
	Attempts      int              `json:"attempts"`
	Budget        int              `json:"evaluationBudget,omitempty"`
	BudgetReached bool             `json:"evaluationBudgetReached,omitempty"`
	FreshMeasured int              `json:"freshMeasured"`
	Pruned        []string         `json:"pruned"`
	Groups        []core.MoveGroup `json:"groups"`
}

type reportFile struct {
	Path           string `json:"path"`
	Bytes          int64  `json:"bytes"`
	EstimatedBytes uint64 `json:"estimatedBytes,omitempty"`
}

// stageEmit publishes the final GGUF with its recipe sidecar and run report,
// then cleans up scratch artifacts. It is idempotent and crash-safe:
//
//   - Sidecar and report are written BEFORE the artifact move (both to
//     deterministic paths, atomically via tmp+rename), so a crash at any
//     point leaves them either absent or complete.
//   - The move is verifiable: if the destination already exists (a crash
//     after the rename) with a size matching the exact planned artifact
//     size, the move is treated as done and never repeated.
//   - The artifact size is hard-checked against BudgetBytes (the FINAL
//     artifact, metadata included) before anything is published.
//   - Quality gates are evaluated into the run report; a failing gate does
//     not block publish (GatesPass records the outcome).
//   - Scratch cleanup runs only AFTER MarkComplete succeeds; a crash between
//     completion and cleanup is healed by the next resume.
//
// A `quantlab resume` after a crash at ANY point inside this stage therefore
// completes without manual surgery.
func (e *Engine) stageEmit(ctx context.Context) error {
	cfg := e.Run.Config
	final := e.Run.Artifacts[core.StageQuantize]
	base := strings.TrimSuffix(filepath.Base(cfg.SourcePath), filepath.Ext(cfg.SourcePath))
	outName := base + "-" + e.Run.RunID + ".gguf"
	dest := filepath.Join(cfg.OutputDir, outName)
	reportPath := filepath.Join(cfg.OutputDir, base+"-"+e.Run.RunID+".report.json")
	sidecarPath := strings.TrimSuffix(dest, ".gguf") + ".oid-plan.json"

	if e.DryRun {
		e.printf("plan: emit model %s\n", dest)
		e.printf("plan: emit recipe %s\n", sidecarPath)
		e.printf("plan: emit report %s\n", reportPath)
		return e.complete(core.StageEmit, "")
	}
	if final == "" {
		return fmt.Errorf("pipeline: no final artifact to emit")
	}

	// Resolve the source of the artifact bytes: normally the staged final
	// GGUF, but after a crash-after-move only the destination exists.
	emitSrc := final
	finalStat, finalErr := os.Stat(final)
	if finalErr != nil {
		if !os.IsNotExist(finalErr) {
			return finalErr
		}
		// Crash between the rename and the stage checkpoint: the move
		// already happened. Verify the destination before trusting it.
		dstStat, dstErr := os.Stat(dest)
		if dstErr != nil {
			return fmt.Errorf("pipeline: emit crashed: neither staged artifact %s nor destination %s exists", final, dest)
		}
		if err := e.verifyEmitted(dest, uint64(dstStat.Size())); err != nil {
			return fmt.Errorf("pipeline: emit crashed and destination is not recoverable: %w", err)
		}
		emitSrc = dest
		finalStat = dstStat
	}

	// BudgetBytes caps the complete final artifact. Hard-fail before any
	// mutation/publishing.
	if uint64(finalStat.Size()) > cfg.BudgetBytes {
		return fmt.Errorf("pipeline: final artifact %s is %d bytes, exceeding budget %d (budget covers the complete file: header, metadata, tensor info, padding, payload)",
			emitSrc, finalStat.Size(), cfg.BudgetBytes)
	}

	// Deep-effort verification: one full-corpus KLD evaluation of the final
	// artifact. Gate outcomes land on the report; they do not abort publish.
	if err := e.finalValidation(ctx, emitSrc); err != nil {
		return err
	}

	sidecar := RecipeSidecar{
		RunID:       e.Run.RunID,
		ProfileID:   e.Run.Manifest.ProfileID,
		SourcePath:  cfg.SourcePath,
		SourceSHA:   e.Run.Manifest.SourceSHA,
		BudgetBytes: cfg.BudgetBytes,
		TotalBytes:  e.Run.Manifest.TotalBytes,
		Assignments: e.Run.Manifest.Options,
		Tools:       cfg.Tools,
		CreatedAt:   e.now(),
	}
	if err := e.writeJSON(sidecarPath, &sidecar); err != nil {
		return err
	}

	caps, err := e.caps(ctx, orchestrate.ToolPerplexity)
	if err != nil {
		return err
	}
	report, err := e.buildReport(emitSrc, finalStat.Size(), caps)
	if err != nil {
		return err
	}
	if err := e.writeJSON(reportPath, &report); err != nil {
		return err
	}

	// The move itself. Already-done (crash after rename) is detected above;
	// re-running it is harmless anyway (os.Rename overwrites atomically).
	if emitSrc != dest {
		if err := moveFile(emitSrc, dest); err != nil {
			return err
		}
	}
	if err := e.complete(core.StageEmit, dest); err != nil {
		return err
	}

	// Scratch cleanup only after the checkpoint records completion, so a
	// crash before MarkComplete never destroys resumability.
	e.cleanupScratch()

	e.printf("  emitted %s (%d bytes, budget %d, gates pass=%v)\n",
		dest, report.Output.Bytes, cfg.BudgetBytes, report.GatesPass)
	return nil
}

// verifyEmitted checks that a pre-existing destination is exactly the
// artifact this run planned: size must match the exact planned artifact size
// when computable, and never exceed the budget.
func (e *Engine) verifyEmitted(dest string, size uint64) error {
	if size > e.Run.Config.BudgetBytes {
		return fmt.Errorf("%s is %d bytes, over budget %d", dest, size, e.Run.Config.BudgetBytes)
	}
	if e.Run.Bank != nil && e.Run.Manifest != nil {
		if planned, ok := tensorbank.PlannedArtifactSize(e.Run.Bank, e.Run.Manifest); ok && planned != size {
			return fmt.Errorf("%s size %d does not match planned artifact size %d", dest, size, planned)
		}
	}
	return nil
}

// finalValidationID is the measurement profile ID for the deep-effort
// full-corpus verification pass.
func finalValidationID(profileID string) string { return profileID + "+final" }

// finalValidation implements the deep effort preset's final verification:
// one additional KLD evaluation of the final artifact using the same
// EvalConfig chunks as evaluate (e.Extra.Chunks). Logits were captured at
// Extra.Chunks; Chunks=0 (unlimited) against those logits is not comparable.
//
// When search accepted nothing and BestProfileID already has evaluate
// KLD+p95, the extra pass is skipped — it would remeasure the same artifact
// at the same chunk count. Gate outcomes are recorded against BestProfileID
// (or +final when the extra pass did run).
//
// Extra-pass measurements are recorded under finalValidationID and
// checkpointed immediately, so a crash anywhere later in emit resumes
// without re-running the pass, and a completed emit never re-executes.
// Gate failures are recorded on the report (GatesPass=false) but do not
// abort publish: a multi-hour run still yields a GGUF.
func (e *Engine) finalValidation(ctx context.Context, modelPath string) error {
	if Effort(e.Run.Config.Effort) != EffortDeep {
		return nil
	}
	cfg := e.Run.Config
	bestID := e.Run.BestProfileID
	id := finalValidationID(bestID)
	caps, err := e.caps(ctx, orchestrate.ToolPerplexity)
	if err != nil {
		return err
	}
	evalCfg := orchestrate.EvalConfig{
		CorpusPath: cfg.EvalCorpus,
		CtxSize:    cfg.CtxSize,
		Chunks:     e.Extra.Chunks,
		Threads:    cfg.Threads,
		NGPULayers: -1,
	}
	_, hasKLD := e.measurementForEval(bestID, core.MetricKLD, evalCfg, caps)
	_, hasP95 := e.measurementForEval(bestID, core.MetricP95KLD, evalCfg, caps)
	_, hasFinal := e.measurementForEval(id, core.MetricKLD, evalCfg, caps)
	skipExtra := hasKLD && hasP95
	if !skipExtra && !hasFinal {
		_, hasBaseline := e.measurementForEval("baseline", core.MetricPerplexity, evalCfg, caps)
		if !e.recordedLogits(e.logitsPath(), evalCfg, caps) || !hasBaseline {
			metrics, prov, err := e.captureBaselineLogits(ctx, evalCfg, caps,
				e.logitsPath(), "deep final validation baseline")
			if err != nil {
				return err
			}
			if !metrics.HasPPL {
				return fmt.Errorf("pipeline: deep final validation baseline produced no perplexity")
			}
			if err := e.recordMeasurement(core.Measurement{
				ProfileID: "baseline", Metric: core.MetricPerplexity,
				Value: metrics.Perplexity, Baseline: metrics.Perplexity, Prov: prov,
			}); err != nil {
				return err
			}
		}
		m, err := e.evalModel(ctx, evalCfg, caps, modelPath, e.logitsPath())
		if err != nil {
			return fmt.Errorf("pipeline: deep final validation: %w", err)
		}
		if !m.HasMeanKLD {
			return fmt.Errorf("pipeline: deep final validation produced no KLD")
		}
		prov, err := e.newEvalProvenance(evalCfg, caps)
		if err != nil {
			return err
		}
		if err := e.recordMeasurement(core.Measurement{
			ProfileID: id,
			Metric:    core.MetricKLD,
			Value:     m.MeanKLD,
			Baseline:  0,
			Delta:     m.MeanKLD,
			Prov:      prov,
		}); err != nil {
			return err
		}
		if m.HasP95 {
			if err := e.recordMeasurement(core.Measurement{
				ProfileID: id,
				Metric:    core.MetricP95KLD,
				Value:     m.P95KLD,
				Baseline:  0,
				Delta:     m.P95KLD,
				Prov:      prov,
			}); err != nil {
				return err
			}
		}
		for _, aux := range []struct {
			metric  core.MetricKind
			value   float64
			present bool
		}{
			{core.MetricMaxKLD, m.MaxKLD, m.HasMax},
			{core.MetricCVaRKLD, m.CVaRKLD, m.HasCVaR},
			{core.MetricRMSDeltaP, m.RMSDeltaP, m.HasRMS},
			{core.MetricTop1Disagreement, 1 - m.SameTop, m.HasSameTop},
		} {
			if aux.present {
				if err := e.recordMeasurement(core.Measurement{
					ProfileID: id, Metric: aux.metric, Value: aux.value,
					Baseline: 0, Delta: aux.value, Prov: prov,
				}); err != nil {
					return err
				}
			}
		}
		e.printf("  deep validation: mean KLD %.6f (p95 %.6f)\n", m.MeanKLD, m.P95KLD)
		// Checkpoint the pass immediately: a crash before emit completes
		// must not re-run it on resume.
		if err := e.Store.Save(e.Run); err != nil {
			return fmt.Errorf("pipeline: deep final validation checkpoint: %w", err)
		}
	}
	for _, g := range e.resolvedGates() {
		m, ok := e.gateMeasurement(g.Metric, caps)
		if !ok || !g.Passes(m) {
			e.printf("  deep validation: gate %s failed (measured=%v); publishing anyway\n", g.Metric, ok)
		}
	}
	return nil
}

// gateMeasurement returns the measurement a gate is evaluated against: the
// deep final-validation measurement when present, else the best profile's.
// The virtual worst-domain metric resolves to the maximum recorded
// per-domain KLD.
func (e *Engine) gateMeasurement(metric core.MetricKind, caps *orchestrate.Capabilities) (core.Measurement, bool) {
	evalCfg := orchestrate.EvalConfig{
		CorpusPath: e.Run.Config.EvalCorpus,
		CtxSize:    e.Run.Config.CtxSize,
		Chunks:     e.Extra.Chunks,
		Threads:    e.Run.Config.Threads,
		NGPULayers: -1,
	}
	if metric == core.MetricWorstDomainKLD {
		domains, err := e.evalableDomainEvalPaths()
		if err != nil || len(domains) == 0 {
			return core.Measurement{}, false
		}
		var max core.Measurement
		haveMax := false
		for domain, corpus := range domains {
			domainCfg := evalCfg
			domainCfg.CorpusPath = corpus
			m, ok := e.measurementForEval(e.Run.BestProfileID, core.DomainMetric(domain), domainCfg, caps)
			if !ok {
				return core.Measurement{}, false
			}
			if !haveMax || m.Delta > max.Delta {
				max = m
				haveMax = true
			}
		}
		max.Metric = core.MetricWorstDomainKLD
		return max, true
	}
	if m, ok := e.measurementForEval(finalValidationID(e.Run.BestProfileID), metric, evalCfg, caps); ok {
		return m, true
	}
	if m, ok := e.measurementForEval(e.Run.BestProfileID, metric, evalCfg, caps); ok {
		return m, true
	}
	return core.Measurement{}, false
}

func (e *Engine) buildReport(path string, size int64, caps *orchestrate.Capabilities) (RunReport, error) {
	cfg := e.Run.Config
	report := RunReport{
		RunID:        e.Run.RunID,
		CreatedAt:    e.Run.CreatedAt,
		FinishedAt:   e.now(),
		BudgetBytes:  cfg.BudgetBytes,
		ProfileID:    e.Run.BestProfileID,
		Measurements: e.Run.Measurements,
		Search:       searchSummary{Steps: len(e.Run.SearchHistory), Groups: e.Run.MoveGroups},
	}
	if st, err := os.Stat(cfg.SourcePath); err == nil {
		report.Source = reportFile{Path: cfg.SourcePath, Bytes: st.Size()}
	}
	report.Output = reportFile{Path: path, Bytes: size}
	if e.Run.Manifest != nil {
		report.Output.EstimatedBytes = e.Run.Manifest.TotalBytes
	}
	if sumPath := e.Run.Artifacts[core.StageSearch]; sumPath != "" {
		if data, err := os.ReadFile(sumPath); err == nil {
			var sum searchSummary
			if err := json.Unmarshal(data, &sum); err == nil {
				report.Search = sum
			}
		}
	}
	gates := e.resolvedGates()
	report.GatesPass = true
	for _, g := range gates {
		gr := GateReport{Metric: g.Metric, MaxDelta: g.MaxDelta, MaxAbsolute: g.MaxAbsolute}
		if m, ok := e.gateMeasurement(g.Metric, caps); ok {
			gr.Measured, gr.Value, gr.Delta, gr.Pass = true, m.Value, m.Delta, g.Passes(m)
		} else if g.Metric == core.MetricWorstDomainKLD {
			// Domain holdouts run during evaluate. A skipped search never
			// measured them; that is not a failed Q3.
			continue
		}
		// Fail-closed: a required gate whose measurement is missing never passes.
		if !gr.Pass {
			report.GatesPass = false
		}
		report.Gates = append(report.Gates, gr)
	}
	report.GatesConfigured = len(report.Gates)
	return report, nil
}

// moveFile renames src to dst, falling back to a copy when the two sit on
// different filesystems.
func moveFile(src, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	if err := os.Rename(src, dst); err == nil {
		return nil
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp := dst + ".tmp"
	out, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Sync(); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, dst); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Remove(src)
}

// PrintStatus renders a human-readable stage/progress/measurements summary.
func PrintStatus(w io.Writer, r *state.Run) {
	fmt.Fprintf(w, "run %s (schema v%d)\n", r.RunID, r.Version)
	fmt.Fprintf(w, "  created %s, updated %s\n", r.CreatedAt.Format(time.RFC3339), r.UpdatedAt.Format(time.RFC3339))
	c := r.Config
	fmt.Fprintf(w, "  src %s\n", c.SourcePath)
	fmt.Fprintf(w, "  budget %d bytes", c.BudgetBytes)
	if c.TargetBPW > 0 {
		fmt.Fprintf(w, " (target bpw %.3f)", c.TargetBPW)
	}
	fmt.Fprintf(w, ", threads %d, ctx %d\n", c.Threads, c.CtxSize)
	if c.SearchEnabled {
		fmt.Fprintf(w, "  search: enabled, max %d iterations\n", c.MaxSearchIterations)
	} else {
		fmt.Fprintf(w, "  search: disabled\n")
	}
	done := map[core.Stage]bool{}
	for _, s := range r.Completed {
		done[s] = true
	}
	for _, s := range core.StageOrder {
		mark := " "
		if done[s] {
			mark = "x"
		}
		line := fmt.Sprintf("  [%s] %s", mark, s)
		if a := r.Artifacts[s]; a != "" {
			line += " -> " + a
		}
		fmt.Fprintln(w, line)
	}
	if next, ok := r.NextStage(); ok {
		fmt.Fprintf(w, "  next stage: %s (%d/%d complete)\n", next, len(r.Completed), len(core.StageOrder))
	} else {
		fmt.Fprintf(w, "  complete (%d/%d)\n", len(r.Completed), len(core.StageOrder))
	}
	if r.Bank != nil {
		fmt.Fprintf(w, "  bank: %d tensors, %d bytes\n", len(r.Bank.Tensors), r.Bank.TotalBytes())
	}
	if r.Manifest != nil {
		fmt.Fprintf(w, "  manifest: profile %s, %d options, %d bytes\n",
			r.Manifest.ProfileID, len(r.Manifest.Options), r.Manifest.TotalBytes)
	}
	if len(r.Measurements) > 0 {
		fmt.Fprintf(w, "  measurements: %d\n", len(r.Measurements))
		for _, m := range r.Measurements {
			fmt.Fprintf(w, "    %s %s = %.6f (baseline %.6f, delta %.6f)\n",
				m.ProfileID, m.Metric, m.Value, m.Baseline, m.Delta)
		}
	}
	if len(r.SearchHistory) > 0 {
		fmt.Fprintf(w, "  search history: %d steps (best %s)\n", len(r.SearchHistory), r.BestProfileID)
	}
}
