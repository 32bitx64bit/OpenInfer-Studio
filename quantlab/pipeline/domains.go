package pipeline

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// errEvalCorpusTooShort is returned when llama-perplexity refuses a holdout
// because it tokenizes to fewer than 2×n_ctx tokens. Domain slices of a
// mixed calibration corpus often miss that floor at ctx=4096; the job must
// still publish the already-assembled GGUF.
var errEvalCorpusTooShort = errors.New("pipeline: evaluation corpus is too short for this context size")

// minPerplexityCorpusBytes is a conservative skip threshold. llama-perplexity
// requires ≥2×n_ctx tokens; 4 bytes/token keeps a 22 KiB code holdout from
// aborting a multi-hour Deep run while still admitting a 200 KiB long-context
// file.
func minPerplexityCorpusBytes(ctxSize int) int64 {
	if ctxSize <= 0 {
		ctxSize = 512
	}
	return int64(ctxSize) * 2 * 4
}

func domainHoldoutTooShort(path string, ctxSize int) bool {
	st, err := os.Stat(path)
	if err != nil {
		return false
	}
	return st.Size() < minPerplexityCorpusBytes(ctxSize)
}

func evalOutputCorpusTooShort(output string) bool {
	return strings.Contains(output, "you need at least") &&
		strings.Contains(output, "tokens to evaluate perplexity")
}

const searchBaselineProfileID = "baseline-search"

// domainEvalPaths returns per-domain evaluation corpus files registered in
// the calibration manifest (calibrate writes evaluation-<domain>.txt when a
// mixture has more than one domain), keyed by domain name.
func domainEvalPaths(exprusDir string) (map[string]string, error) {
	manifest := filepath.Join(exprusDir, "manifest.json")
	data, err := os.ReadFile(manifest)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var m struct {
		Outputs map[string]string `json:"outputs"`
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("pipeline: decode calibration manifest: %w", err)
	}
	out := map[string]string{}
	for name := range m.Outputs {
		if !strings.HasPrefix(name, "evaluation-") || !strings.HasSuffix(name, ".txt") {
			continue
		}
		dom := strings.TrimSuffix(strings.TrimPrefix(name, "evaluation-"), ".txt")
		path := filepath.Join(exprusDir, name)
		st, err := os.Stat(path)
		if err != nil || st.IsDir() || st.Size() == 0 {
			if err == nil {
				err = fmt.Errorf("empty or non-file corpus")
			}
			return nil, fmt.Errorf("pipeline: registered domain holdout %s is unavailable: %w", name, err)
		}
		out[dom] = path
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

// evalableDomainEvalPaths is the subset of registered holdouts large enough
// for llama-perplexity at this run's context size. Missing files still error;
// undersized files are omitted so gates and eval loops do not fail-closed on
// a calibration split that cannot possibly score.
func (e *Engine) evalableDomainEvalPaths() (map[string]string, error) {
	all, err := e.expectedDomainEvalPaths()
	if err != nil {
		return nil, err
	}
	ctxSize := 0
	if e.Run != nil {
		ctxSize = e.Run.Config.CtxSize
	}
	out := make(map[string]string, len(all))
	for domain, path := range all {
		if domainHoldoutTooShort(path, ctxSize) {
			continue
		}
		out[domain] = path
	}
	return out, nil
}

func (e *Engine) logSkippedDomainHoldouts(evalable map[string]string) {
	all, err := e.expectedDomainEvalPaths()
	if err != nil || len(all) == 0 {
		return
	}
	var skipped []string
	for domain := range all {
		if _, ok := evalable[domain]; !ok {
			skipped = append(skipped, domain)
		}
	}
	if len(skipped) == 0 {
		return
	}
	sort.Strings(skipped)
	ctxSize := 0
	if e.Run != nil {
		ctxSize = e.Run.Config.CtxSize
	}
	e.printf("  skipping %d domain holdout(s) too short for ctx %d: %s\n",
		len(skipped), ctxSize, strings.Join(skipped, ", "))
}

func (e *Engine) expectedDomainEvalPaths() (map[string]string, error) {
	if len(e.Run.Config.DomainEvalCorpora) == 0 {
		return domainEvalPaths(filepath.Dir(e.Run.Config.EvalCorpus))
	}
	out := make(map[string]string, len(e.Run.Config.DomainEvalCorpora))
	for domain, path := range e.Run.Config.DomainEvalCorpora {
		st, err := os.Stat(path)
		if err != nil || st.IsDir() || st.Size() == 0 {
			if err == nil {
				err = fmt.Errorf("empty or non-file corpus")
			}
			return nil, fmt.Errorf("pipeline: expected domain holdout %s is unavailable: %w", domain, err)
		}
		out[domain] = path
	}
	return out, nil
}

// logitsDomainPath is the shared domain baseline-logits scratch path. Domain
// evaluation is sequential; captureBaselineLogits replaces this file whenever
// its corpus provenance changes, so seven holdouts never retain seven
// full-vocabulary logits files at once.
func (e *Engine) logitsDomainPath(_ string) string {
	head := strings.TrimSuffix(e.logitsPath(), ".bin")
	return head + "-domain.bin"
}

// searchCorpusPath resolves the tuning-only holdout. It is deliberately
// distinct from evaluation.txt, which remains untouched until final scoring.
func (e *Engine) searchCorpusPath() (string, bool) {
	path := e.Run.Config.SearchCorpus
	if path == "" {
		// Backward-compatible discovery for checkpoints created before
		// SearchCorpus was persisted.
		path = filepath.Join(filepath.Dir(e.Run.Config.EvalCorpus), "search.txt")
	}
	st, err := os.Stat(path)
	if err != nil || st.IsDir() || st.Size() == 0 {
		return path, false
	}
	evalAbs, _ := filepath.Abs(e.Run.Config.EvalCorpus)
	searchAbs, _ := filepath.Abs(path)
	return path, searchAbs != evalAbs
}

func (e *Engine) searchLogitsPath() string {
	head := strings.TrimSuffix(e.logitsPath(), ".bin")
	return head + "-search.bin"
}
