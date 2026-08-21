package pipeline

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"quantlab/core"
	"quantlab/profile"
	"quantlab/scalefold"
	"quantlab/tensorbank"
)

// payloadSource is the GGUF path that concrete weight payloads are read
// from: MTP-stripped when present, else reconstructed (Hadamard/CSK), else
// folded, else the configured source. Structural reads (anchors, parses,
// baseline logits) keep using the configured source; only payload producers
// follow this redirect.
func (e *Engine) payloadSource() string {
	if p := e.Extra.MTPStrippedPath; p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return e.weightPayloadSource()
}

// weightPayloadSource is reconstructed, else folded, else the configured
// source — the GGUF that still carries fused MTP heads when present.
func (e *Engine) weightPayloadSource() string {
	if p := e.Extra.ReconstructedSourcePath; p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	if p := e.Extra.FoldedSourcePath; p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	if p := e.Extra.PrivateSourcePath; p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return e.Run.Config.SourcePath
}

// imatrixPath is the imatrix GGUF aligned with the latest reconstruct
// transform: reconstructed > folded > FTI > configured.
func (e *Engine) imatrixPath() string {
	if p := e.Extra.ReconstructedImatrixPath; p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	if p := e.Extra.FoldedImatrixPath; p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	if p := e.Extra.FTIImatrixPath; p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return e.Run.Config.ImatrixPath
}

func samePath(a, b string) bool {
	if a == "" || b == "" {
		return a == b
	}
	if a == b {
		return true
	}
	aa, err1 := filepath.Abs(a)
	bb, err2 := filepath.Abs(b)
	return err1 == nil && err2 == nil && aa == bb
}

// payloadIdentitySHA is the file hash tensorbank.Build compares against
// SelectionManifest.SourceSHA. After scale-fold / reconstruct the primary
// GGUF is a rewrite, so this is not TensorBank.SHA256 (that stays the
// original source identity for resume).
func (e *Engine) payloadIdentitySHA() (string, error) {
	path := e.payloadSource()
	inPlace := e.Extra.ReconstructedSourcePath != "" && samePath(path, e.Extra.ReconstructedSourcePath)
	if !inPlace && e.Run != nil && samePath(path, e.Run.Config.SourcePath) {
		if e.Run.Bank != nil && e.Run.Bank.SHA256 != "" {
			return e.Run.Bank.SHA256, nil
		}
	}
	if e.Extra.PayloadSHA != "" && samePath(e.Extra.PayloadSHAPath, path) {
		return e.Extra.PayloadSHA, nil
	}
	s, err := tensorbank.OpenSource(path)
	if err != nil {
		return "", fmt.Errorf("pipeline: open payload for SHA: %w", err)
	}
	sha, err := s.SHA256()
	_ = s.Close()
	if err != nil {
		return "", fmt.Errorf("pipeline: hash payload: %w", err)
	}
	e.Extra.PayloadSHA = sha
	e.Extra.PayloadSHAPath = path
	if err := e.saveExtra(); err != nil {
		return "", err
	}
	return sha, nil
}

// bindManifestSource sets manifest.SourceSHA to the payload GGUF identity
// so assembly of a folded/reconstructed primary does not fail the original-
// source SHA check.
func (e *Engine) bindManifestSource(m *core.SelectionManifest) error {
	if m == nil {
		return nil
	}
	sha, err := e.payloadIdentitySHA()
	if err != nil {
		return err
	}
	m.SourceSHA = sha
	return nil
}

func (e *Engine) manifestFor(p *core.Profile) (*core.SelectionManifest, error) {
	m, err := core.ManifestFor(p, e.Run.Bank)
	if err != nil {
		return nil, err
	}
	if err := e.bindManifestSource(m); err != nil {
		return nil, err
	}
	return m, nil
}

// foldState is the persisted fold audit artifact.
type foldState struct {
	Clusters      []scalefold.Cluster `json:"clusters"`
	Folded        string              `json:"folded"`
	FoldedImatrix string              `json:"foldedImatrix,omitempty"`
	SourceSHA     string              `json:"sourceSHA"`
}

// stageFold applies AWQ-style equivalent scaling when enabled: discover
// norm/consumer clusters, choose the fold exponent per cluster by measured
// importance-weighted error at the probe dtype, emit a folded GGUF and a
// matching folded imatrix, and redirect payload paths. Disabled, missing
// imatrix, or an already-readied fold all no-op. Runs at the end of the
// assemble stage (before anchors, solve, quantize).
func (e *Engine) stageFold(ctx context.Context) error {
	if !e.foldEnabled() || e.Run.Bank == nil {
		return nil
	}
	if e.imatrixPath() == "" {
		e.printf("  scale-fold: no imatrix configured; skipped\n")
		return nil
	}
	if e.Extra.FoldedSourcePath != "" {
		if _, err := os.Stat(e.Extra.FoldedSourcePath); err == nil {
			return nil // already folded (resume)
		}
	}
	probe := core.DTypeQ4_K_T
	if e.DryRun {
		clusters := scalefold.Discover(e.Run.Bank)
		e.printf("plan: scale-fold, %d clusters discovered\n", len(clusters))
		return nil
	}
	clusters := scalefold.Discover(e.Run.Bank)
	if len(clusters) == 0 {
		e.printf("  scale-fold: no eligible norm/consumer clusters; skipped\n")
		return nil
	}
	srcPath := e.payloadSource()
	src, err := tensorbank.OpenSource(srcPath)
	if err != nil {
		return fmt.Errorf("scale-fold: open source: %w", err)
	}
	defer src.Close()
	rawImatrix, err := profile.LoadImatrix(e.imatrixPath())
	if err != nil {
		e.printf("  scale-fold: skipped (%v)\n", err)
		return nil
	}
	if len(rawImatrix) == 0 {
		e.printf("  scale-fold: no per-tensor stats; skipped\n")
		return nil
	}
	clusters, err = scalefold.ChooseAlpha(src, clusters, rawImatrix, probe)
	if err != nil {
		return fmt.Errorf("scale-fold: alpha selection: %w", err)
	}
	applied := 0
	for _, cl := range clusters {
		if cl.Skipped == "" && cl.Alpha > 0 && len(cl.Scales) > 0 {
			applied++
		}
	}
	if applied == 0 {
		e.printf("  scale-fold: no cluster improved by folding (%d skipped)\n", len(clusters))
		return nil
	}
	foldedPath := srcPath
	if !e.jobOwnsPath(srcPath) {
		foldedPath = filepath.Join(e.workDir(), "folded.gguf")
	}
	if err := scalefold.Apply(src, clusters, foldedPath); err != nil {
		return fmt.Errorf("scale-fold: apply: %w", err)
	}
	foldedImatrix := ""
	impSrc, err := tensorbank.OpenSource(e.imatrixPath())
	if err != nil {
		return fmt.Errorf("scale-fold: open imatrix: %w", err)
	}
	foldedImatrix = filepath.Join(e.workDir(), "folded-imatrix.gguf")
	if err := scalefold.ApplyImatrix(impSrc, clusters, foldedImatrix); err != nil {
		impSrc.Close()
		return fmt.Errorf("scale-fold: apply imatrix: %w", err)
	}
	impSrc.Close()
	e.Extra.FoldedSourcePath = foldedPath
	e.Extra.FoldedImatrixPath = foldedImatrix
	e.Extra.PayloadSHA = ""
	e.Extra.PayloadSHAPath = ""
	if err := e.saveExtra(); err != nil {
		return fmt.Errorf("scale-fold: persist extra config: %w", err)
	}
	st := foldState{
		Clusters:      clusters,
		Folded:        foldedPath,
		FoldedImatrix: foldedImatrix,
		SourceSHA:     e.Run.Bank.SHA256,
	}
	if err := e.writeJSON(filepath.Join(e.workDir(), "scale-fold.json"), &st); err != nil {
		return err
	}
	e.printf("  scale-fold: %d/%d clusters folded -> %s\n", applied, len(clusters), foldedPath)
	return nil
}
