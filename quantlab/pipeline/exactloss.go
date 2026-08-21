package pipeline

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"quantlab/core"
)

// exactLossStore persists the solve-time exact loss table so the search
// stage reuses it for seed re-solves without re-streaming the source.
type exactLossStore struct {
	Version   int                               `json:"version"`
	ModelID   string                            `json:"modelID"`
	ModelSHA  string                            `json:"modelSHA"`
	Signature string                            `json:"signature"`
	Entries   map[string]map[core.DType]float64 `json:"entries"`
}

func (e *Engine) exactLossWorkPath() string {
	if e.Run == nil {
		return ""
	}
	return filepath.Join(e.workDir(), "exact-loss.json")
}

func (e *Engine) saveExactLossWithSignature(bank *core.TensorBank, signature string, table map[string]map[core.DType]float64) error {
	if bank == nil || len(table) == 0 {
		return nil
	}
	p := e.exactLossWorkPath()
	if p == "" {
		return nil
	}
	// Validate: drop non-finite values before they reach the solver or the
	// JSON encoder (NaN/Inf can arise from corrupt source weights).
	clean := map[string]map[core.DType]float64{}
	for name, m := range table {
		for d, v := range m {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				continue
			}
			if clean[name] == nil {
				clean[name] = map[core.DType]float64{}
			}
			clean[name][d] = v
		}
	}
	return e.writeJSON(p, &exactLossStore{
		Version: 2, ModelID: bank.ModelID, ModelSHA: bank.SHA256,
		Signature: signature, Entries: clean,
	})
}

// loadExactLoss reads a previously persisted table; identity mismatch or
// any decode error yields nil (fail-open).
func (e *Engine) loadExactLoss(bank *core.TensorBank) map[string]map[core.DType]float64 {
	if bank == nil {
		return nil
	}
	p := e.exactLossWorkPath()
	if p == "" {
		return nil
	}
	data, err := os.ReadFile(p)
	if err != nil {
		return nil
	}
	var s exactLossStore
	if err := json.Unmarshal(data, &s); err != nil {
		return nil
	}
	if s.ModelID != bank.ModelID || (bank.SHA256 != "" && s.ModelSHA != "" && s.ModelSHA != bank.SHA256) {
		return nil
	}
	signature, err := e.exactLossSignature(bank)
	if err != nil || s.Version < 2 || s.Signature == "" || s.Signature != signature {
		return nil
	}
	return s.Entries
}

func (e *Engine) exactLossSignature(bank *core.TensorBank) (string, error) {
	if bank == nil {
		return "", fmt.Errorf("pipeline: exact-loss signature needs a tensor bank")
	}
	h := sha256.New()
	_, _ = fmt.Fprintf(h, "v2\x00%s\x00%s\x00probe=%t\x00solverfti=%t\x00sketches=v1\x00",
		bank.ModelID, bank.SHA256, e.probeKLDEnabled(), e.solverFTIEnabled())
	payloadSHA, err := e.payloadIdentitySHA()
	if err != nil {
		return "", err
	}
	_, _ = fmt.Fprintf(h, "payload=%s\x00", payloadSHA)
	for _, d := range e.candidateDTypes() {
		_, _ = fmt.Fprintf(h, "%s\x00", d.BaseTensorType())
	}
	if p := e.imatrixPath(); p != "" {
		f, err := os.Open(p)
		if err != nil {
			return "", err
		}
		_, copyErr := io.Copy(h, f)
		closeErr := f.Close()
		if copyErr != nil {
			return "", copyErr
		}
		if closeErr != nil {
			return "", closeErr
		}
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}
