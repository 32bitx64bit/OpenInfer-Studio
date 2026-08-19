package orchestrate

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"time"

	"quantlab/core"
)

// RunProvenance is the full audit record for one tool execution: the core
// provenance contract plus the tool binary's own content hash.
type RunProvenance struct {
	core.Provenance
}

// HashFile returns the hex SHA-256 of the file at path.
func HashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// NewRunProvenance builds a validated provenance record for a run: binary
// hash + version, corpus and (optional) imatrix digests, run id, and time.
// corpusPath and imatrixPath may be empty to omit the respective digest;
// binaryPath and runID are required.
func NewRunProvenance(tool Tool, binaryPath, version, runID, corpusPath, imatrixPath string, at time.Time) (RunProvenance, error) {
	if binaryPath == "" || runID == "" {
		return RunProvenance{}, fmt.Errorf("orchestrate: provenance needs binary path and run id")
	}
	binSHA, err := HashFile(binaryPath)
	if err != nil {
		return RunProvenance{}, fmt.Errorf("orchestrate: hash binary: %w", err)
	}
	rp := RunProvenance{
		Provenance: core.Provenance{
			Tool:         string(tool),
			ToolVersion:  version,
			BinarySHA256: binSHA,
			RunID:        runID,
			MeasuredAt:   at,
		},
	}
	if corpusPath != "" {
		sha, err := HashFile(corpusPath)
		if err != nil {
			return RunProvenance{}, fmt.Errorf("orchestrate: hash corpus: %w", err)
		}
		rp.CorpusSHA = sha
	}
	if imatrixPath != "" {
		sha, err := HashFile(imatrixPath)
		if err != nil {
			return RunProvenance{}, fmt.Errorf("orchestrate: hash imatrix: %w", err)
		}
		rp.ImatrixSHA = sha
	}
	if rp.MeasuredAt.IsZero() {
		rp.MeasuredAt = time.Now()
	}
	if err := rp.Provenance.Validate(); err != nil {
		return RunProvenance{}, err
	}
	return rp, nil
}
