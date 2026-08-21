package pipeline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"quantlab/core"
	"quantlab/orchestrate"
)

// stageEvaluate measures the baseline once (saving its logits) and the
// candidate against those logits, recording measurements with full
// provenance. Already-recorded measurements are never repeated on resume.
func (e *Engine) stageEvaluate(ctx context.Context) error {
	cfg := e.Run.Config
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
	logits := e.logitsPath()

	if e.DryRun {
		baseIv, err := orchestrate.PlanBaselineEval(evalCfg, cfg.SourcePath, logits, caps, cfg.Tools.LlamaPerplexity)
		if err != nil {
			return err
		}
		e.printf("plan: %s %s\n", baseIv.Path, argvString(baseIv.Argv))
		cand := filepath.Join(e.workDir(), "candidate.gguf")
		candIv, err := orchestrate.PlanCandidateEval(evalCfg, cand, logits, caps, cfg.Tools.LlamaPerplexity)
		if err != nil {
			return err
		}
		e.printf("plan: %s %s\n", candIv.Path, argvString(candIv.Argv))
		return e.complete(core.StageEvaluate, "")
	}

	_, hasBaseline := e.measurementForEval("baseline", core.MetricPerplexity, evalCfg, caps)
	if (!e.recordedLogits(logits, evalCfg, caps) || !hasBaseline) &&
		!e.hasAnyMeasurementForEval(core.MetricKLD, evalCfg, caps) {
		m, prov, err := e.captureBaselineLogits(ctx, evalCfg, caps, logits, "baseline eval")
		if err != nil {
			return err
		}
		if !m.HasPPL {
			return fmt.Errorf("baseline eval: no perplexity in tool output")
		}
		meas := core.Measurement{
			ProfileID: "baseline",
			Metric:    core.MetricPerplexity,
			Value:     m.Perplexity,
			Baseline:  m.Perplexity,
			Prov:      prov,
		}
		if err := e.recordMeasurement(meas); err != nil {
			return err
		}
		e.printf("  baseline: ppl %.4f\n", m.Perplexity)
	}

	profID := e.Run.BestProfileID
	if profID == "" {
		return fmt.Errorf("pipeline: no candidate profile recorded")
	}
	if _, ok := e.measurementForEval(profID, core.MetricKLD, evalCfg, caps); !ok {
		cand := e.Run.Artifacts[core.StageQuantize]
		if cand == "" {
			return fmt.Errorf("pipeline: no candidate artifact from quantize stage")
		}
		m, err := e.evalModel(ctx, evalCfg, caps, cand, logits)
		if err != nil {
			return err
		}
		if !m.HasMeanKLD {
			return fmt.Errorf("pipeline: candidate %s produced no KLD", profID)
		}
		prov, err := e.newEvalProvenance(evalCfg, caps)
		if err != nil {
			return err
		}
		if err := e.recordMeasurement(core.Measurement{
			ProfileID: profID,
			Metric:    core.MetricKLD,
			Value:     m.MeanKLD,
			Baseline:  0,
			Delta:     m.MeanKLD,
			Prov:      prov,
		}); err != nil {
			return err
		}
		// p95 KLD is a first-class gated metric: absolute divergence of
		// the candidate vs the source baseline (Baseline = 0).
		if m.HasP95 {
			if err := e.recordMeasurement(core.Measurement{
				ProfileID: profID,
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
					ProfileID: profID, Metric: aux.metric, Value: aux.value,
					Baseline: 0, Delta: aux.value, Prov: prov,
				}); err != nil {
					return err
				}
			}
		}
		e.printf("  candidate %s: mean KLD %.6f (p95 %.6f)\n", profID, m.MeanKLD, m.P95KLD)
		if m.HasPPL {
			basePPL := m.Perplexity
			if b, ok := e.measurementForEval("baseline", core.MetricPerplexity, evalCfg, caps); ok {
				basePPL = b.Value
			}
			if err := e.recordMeasurement(core.Measurement{
				ProfileID: profID,
				Metric:    core.MetricPerplexity,
				Value:     m.Perplexity,
				Baseline:  basePPL,
				Delta:     m.Perplexity - basePPL,
				Prov:      prov,
			}); err != nil {
				return err
			}
		}
		e.recordAux(profID, m)
	}

	// Per-domain stratified evaluation: when the calibration manifest
	// carries evaluation-<domain>.txt corpora, measure the candidate's mean
	// KLD per domain so the worst-domain gate and audit reports can see
	// domain-specific degradation a single mixed corpus would hide.
	if !e.DryRun {
		domains, err := e.evalableDomainEvalPaths()
		if err != nil {
			return err
		}
		e.logSkippedDomainHoldouts(domains)
		names := make([]string, 0, len(domains))
		for name := range domains {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, dom := range names {
			corpus := domains[dom]
			dm := core.DomainMetric(dom)
			evalCfgDom := evalCfg
			evalCfgDom.CorpusPath = corpus
			if _, ok := e.measurementForEval(profID, dm, evalCfgDom, caps); ok {
				continue
			}
			logits := e.logitsDomainPath(dom)
			if !e.recordedLogits(logits, evalCfgDom, caps) {
				metrics, _, err := e.captureBaselineLogits(ctx, evalCfgDom, caps, logits, "baseline eval ("+dom+")")
				if err != nil {
					if errors.Is(err, errEvalCorpusTooShort) {
						e.printf("  domain %s: skip holdout (%v)\n", dom, err)
						continue
					}
					return err
				}
				if !metrics.HasPPL {
					return fmt.Errorf("baseline eval (%s): no perplexity in tool output", dom)
				}
			}
			cand := e.Run.Artifacts[core.StageQuantize]
			if cand == "" {
				return fmt.Errorf("pipeline: no candidate artifact from quantize stage")
			}
			m, err := e.evalModel(ctx, evalCfgDom, caps, cand, logits)
			if err != nil {
				if errors.Is(err, errEvalCorpusTooShort) {
					e.printf("  domain %s: skip holdout (%v)\n", dom, err)
					continue
				}
				return err
			}
			if !m.HasMeanKLD {
				return fmt.Errorf("pipeline: domain %s eval produced no KLD", dom)
			}
			prov, err := e.newEvalProvenance(evalCfgDom, caps)
			if err != nil {
				return err
			}
			if err := e.recordMeasurement(core.Measurement{
				ProfileID: profID,
				Metric:    dm,
				Value:     m.MeanKLD,
				Baseline:  0,
				Delta:     m.MeanKLD,
				Prov:      prov,
			}); err != nil {
				return err
			}
			if err := e.Store.Save(e.Run); err != nil {
				return fmt.Errorf("pipeline: checkpoint domain %s evaluation: %w", dom, err)
			}
			e.printf("  domain %s: mean KLD %.6f\n", dom, m.MeanKLD)
		}
	}

	artifact := filepath.Join(e.workDir(), "measurements.json")
	if err := e.writeJSON(artifact, e.Run.Measurements); err != nil {
		return err
	}
	return e.complete(core.StageEvaluate, artifact)
}

type logitsCheckpoint struct {
	Bytes int64           `json:"bytes"`
	Prov  core.Provenance `json:"prov"`
}

func (e *Engine) recordedLogits(path string, cfg orchestrate.EvalConfig, caps *orchestrate.Capabilities) bool {
	data, err := os.ReadFile(path + ".complete.json")
	if err != nil {
		return false
	}
	var checkpoint logitsCheckpoint
	if json.Unmarshal(data, &checkpoint) != nil || checkpoint.Bytes <= 0 {
		return false
	}
	st, err := os.Stat(path)
	if err != nil || st.IsDir() || st.Size() != checkpoint.Bytes {
		return false
	}
	want, err := e.newEvalProvenance(cfg, caps)
	return err == nil && sameEvalProvenance(checkpoint.Prov, want)
}

func (e *Engine) captureBaselineLogits(ctx context.Context, cfg orchestrate.EvalConfig,
	caps *orchestrate.Capabilities, path, label string) (orchestrate.EvalMetrics, core.Provenance, error) {
	tmp := path + ".partial"
	_ = os.Remove(tmp)
	defer os.Remove(tmp)
	iv, err := orchestrate.PlanBaselineEval(cfg, e.Run.Config.SourcePath, tmp, caps, e.Run.Config.Tools.LlamaPerplexity)
	if err != nil {
		return orchestrate.EvalMetrics{}, core.Provenance{}, err
	}
	res, err := runOK(ctx, e.Runner, iv)
	out := orchestrate.CombinedOutput(res)
	if evalOutputCorpusTooShort(out) {
		return orchestrate.EvalMetrics{}, core.Provenance{}, fmt.Errorf("%s: %w", label, errEvalCorpusTooShort)
	}
	if err != nil {
		return orchestrate.EvalMetrics{}, core.Provenance{}, fmt.Errorf("%s: %w", label, err)
	}
	metrics, err := orchestrate.ParseEvalMetrics(out)
	if err != nil {
		return orchestrate.EvalMetrics{}, core.Provenance{}, fmt.Errorf("%s: %w", label, err)
	}
	st, err := os.Stat(tmp)
	if err != nil || st.IsDir() || st.Size() == 0 {
		return orchestrate.EvalMetrics{}, core.Provenance{}, fmt.Errorf("%s: tool produced no baseline logits", label)
	}
	_ = os.Remove(path)
	if err := os.Rename(tmp, path); err != nil {
		return orchestrate.EvalMetrics{}, core.Provenance{}, fmt.Errorf("%s: commit baseline logits: %w", label, err)
	}
	prov, err := e.newEvalProvenance(cfg, caps)
	if err != nil {
		return orchestrate.EvalMetrics{}, core.Provenance{}, err
	}
	if err := e.writeJSON(path+".complete.json", &logitsCheckpoint{Bytes: st.Size(), Prov: prov}); err != nil {
		return orchestrate.EvalMetrics{}, core.Provenance{}, fmt.Errorf("%s: checkpoint baseline logits: %w", label, err)
	}
	return metrics, prov, nil
}

// evalModel runs one perplexity/KLD evaluation of modelPath against the
// shared baseline logits.
func (e *Engine) evalModel(ctx context.Context, cfg orchestrate.EvalConfig, caps *orchestrate.Capabilities,
	modelPath, logits string) (orchestrate.EvalMetrics, error) {
	iv, err := orchestrate.PlanCandidateEval(cfg, modelPath, logits, caps, e.Run.Config.Tools.LlamaPerplexity)
	if err != nil {
		return orchestrate.EvalMetrics{}, err
	}
	res, err := runOK(ctx, e.Runner, iv)
	out := orchestrate.CombinedOutput(res)
	if evalOutputCorpusTooShort(out) {
		return orchestrate.EvalMetrics{}, errEvalCorpusTooShort
	}
	if err != nil {
		return orchestrate.EvalMetrics{}, err
	}
	m, err := orchestrate.ParseEvalMetrics(out)
	if err != nil {
		return orchestrate.EvalMetrics{}, err
	}
	return m, nil
}

func (e *Engine) newEvalProvenance(cfg orchestrate.EvalConfig, caps *orchestrate.Capabilities) (core.Provenance, error) {
	binarySHA, err := e.cachedFileSHA(e.Run.Config.Tools.LlamaPerplexity)
	if err != nil {
		return core.Provenance{}, err
	}
	producerSHA, err := e.cachedFileSHA(e.Run.Config.Tools.LlamaQuantize)
	if err != nil {
		return core.Provenance{}, err
	}
	corpusSHA, err := e.cachedFileSHA(cfg.CorpusPath)
	if err != nil {
		return core.Provenance{}, err
	}
	prov := core.Provenance{
		Tool: "llama-perplexity", ToolVersion: caps.Version,
		BinarySHA256: binarySHA, ProducerSHA256: producerSHA,
		RunID: e.Run.RunID, CorpusSHA: corpusSHA, MeasuredAt: e.now(),
	}
	if imatrix := e.imatrixPath(); imatrix != "" {
		prov.ImatrixSHA, err = e.cachedFileSHA(imatrix)
		if err != nil {
			return core.Provenance{}, err
		}
	}
	prov.SourceSHA = e.Extra.PayloadSHA
	if prov.SourceSHA == "" && e.Run.Bank != nil {
		prov.SourceSHA = e.Run.Bank.SHA256
	}
	if prov.SourceSHA == "" {
		prov.SourceSHA, err = e.cachedFileSHA(e.Run.Config.SourcePath)
		if err != nil {
			return core.Provenance{}, fmt.Errorf("pipeline: hash evaluation source: %w", err)
		}
	}
	prov.EvalContext = cfg.CtxSize
	prov.EvalChunks = cfg.Chunks
	prov.EvalThreads = cfg.Threads
	prov.EvalNGPULayers = cfg.NGPULayers
	prov.EvalSeed = cfg.Seed
	prov.EvalConfigured = true
	return prov, nil
}

func (e *Engine) cachedFileSHA(path string) (string, error) {
	st, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	key := filepath.Clean(path)
	e.hashMu.Lock()
	cached, ok := e.fileHashes[key]
	e.hashMu.Unlock()
	modTime := st.ModTime().UnixNano()
	if ok && cached.size == st.Size() && cached.modTime == modTime {
		return cached.sha, nil
	}
	sha, err := orchestrate.HashFile(path)
	if err != nil {
		return "", err
	}
	e.hashMu.Lock()
	if e.fileHashes == nil {
		e.fileHashes = make(map[string]cachedFileHash)
	}
	e.fileHashes[key] = cachedFileHash{size: st.Size(), modTime: modTime, sha: sha}
	e.hashMu.Unlock()
	return sha, nil
}

func sameEvalProvenance(got, want core.Provenance) bool {
	return got.Tool == want.Tool &&
		got.ToolVersion == want.ToolVersion &&
		got.BinarySHA256 != "" && got.BinarySHA256 == want.BinarySHA256 &&
		got.ProducerSHA256 != "" && got.ProducerSHA256 == want.ProducerSHA256 &&
		got.RunID == want.RunID &&
		got.SourceSHA != "" && got.SourceSHA == want.SourceSHA &&
		got.CorpusSHA == want.CorpusSHA &&
		got.ImatrixSHA == want.ImatrixSHA &&
		got.EvalConfigured &&
		got.EvalContext == want.EvalContext &&
		got.EvalChunks == want.EvalChunks &&
		got.EvalThreads == want.EvalThreads &&
		got.EvalNGPULayers == want.EvalNGPULayers &&
		got.EvalSeed == want.EvalSeed
}

// recordAux stores auxiliary metrics (max KLD, RMS delta-p, same-top) in
// the legacy eval list; gated metrics live in Measurements only.
func (e *Engine) recordAux(profileID string, m orchestrate.EvalMetrics) {
	add := func(metric core.MetricKind, v float64) {
		e.Run.Evals = append(e.Run.Evals, core.EvalResult{ProfileID: profileID, Metric: metric, Value: v})
	}
	if m.HasMax {
		add(core.MetricMaxKLD, m.MaxKLD)
	}
	if m.HasCVaR {
		add(core.MetricCVaRKLD, m.CVaRKLD)
	}
	if m.HasRMS {
		add(core.MetricRMSDeltaP, m.RMSDeltaP)
	}
	if m.HasSameTop {
		add(core.MetricTop1Disagreement, 1-m.SameTop)
	}
}

func (e *Engine) measurementForEval(profileID string, metric core.MetricKind,
	cfg orchestrate.EvalConfig, caps *orchestrate.Capabilities) (core.Measurement, bool) {
	want, err := e.newEvalProvenance(cfg, caps)
	if err != nil {
		return core.Measurement{}, false
	}
	for i := len(e.Run.Measurements) - 1; i >= 0; i-- {
		m := e.Run.Measurements[i]
		if m.ProfileID == profileID && m.Metric == metric && sameEvalProvenance(m.Prov, want) {
			return m, true
		}
	}
	return core.Measurement{}, false
}

func (e *Engine) hasAnyMeasurementForEval(metric core.MetricKind,
	cfg orchestrate.EvalConfig, caps *orchestrate.Capabilities) bool {
	want, err := e.newEvalProvenance(cfg, caps)
	if err != nil {
		return false
	}
	for _, m := range e.Run.Measurements {
		if m.Metric == metric && sameEvalProvenance(m.Prov, want) {
			return true
		}
	}
	return false
}

func (e *Engine) recordEvalMetricSet(profileID string, m orchestrate.EvalMetrics, prov core.Provenance) error {
	add := func(metric core.MetricKind, value, baseline float64) error {
		return e.recordMeasurement(core.Measurement{
			ProfileID: profileID, Metric: metric, Value: value,
			Baseline: baseline, Delta: value - baseline, Prov: prov,
		})
	}
	if err := add(core.MetricKLD, m.MeanKLD, 0); err != nil {
		return err
	}
	for _, aux := range []struct {
		metric  core.MetricKind
		value   float64
		present bool
	}{
		{core.MetricP95KLD, m.P95KLD, m.HasP95},
		{core.MetricCVaRKLD, m.CVaRKLD, m.HasCVaR},
		{core.MetricMaxKLD, m.MaxKLD, m.HasMax},
		{core.MetricRMSDeltaP, m.RMSDeltaP, m.HasRMS},
		{core.MetricTop1Disagreement, 1 - m.SameTop, m.HasSameTop},
	} {
		if aux.present {
			if err := add(aux.metric, aux.value, 0); err != nil {
				return err
			}
		}
	}
	if m.HasPPL {
		baseline := m.Perplexity
		for i := len(e.Run.Measurements) - 1; i >= 0; i-- {
			b := e.Run.Measurements[i]
			if b.ProfileID == "baseline" && b.Metric == core.MetricPerplexity &&
				sameEvalProvenance(b.Prov, prov) {
				baseline = b.Value
				break
			}
		}
		if err := add(core.MetricPerplexity, m.Perplexity, baseline); err != nil {
			return err
		}
	}
	return nil
}
