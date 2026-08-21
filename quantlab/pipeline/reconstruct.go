package pipeline

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"quantlab/core"
	"quantlab/profile"
	"quantlab/reconstruct"
	"quantlab/tensorbank"
)

func (e *Engine) reconstructOn() bool {
	return e.effortProfile().Reconstruct
}

func (e *Engine) foldEnabled() bool {
	if e.Extra.NoScaleFold {
		return false
	}
	if e.Extra.ScaleFold {
		return true
	}
	return e.effortProfile().ScaleFold
}

func (e *Engine) hadamardEnabled() bool {
	if e.Extra.NoHadamard {
		return false
	}
	if e.Extra.Hadamard {
		return true
	}
	return e.reconstructOn() || e.effortProfile().InPlaceReconstruct
}

func (e *Engine) cskEnabled() bool {
	if e.Extra.NoCSK {
		return false
	}
	if e.Extra.CSK {
		return true
	}
	return e.reconstructOn() || e.effortProfile().InPlaceReconstruct
}

func (e *Engine) permuteEnabled() bool {
	if e.Extra.NoPermute {
		return false
	}
	return e.reconstructOn()
}

func (e *Engine) magrEnabled() bool {
	if e.Extra.NoMagR {
		return false
	}
	return e.reconstructOn()
}

func (e *Engine) lwcEnabled() bool {
	if e.Extra.NoLWC {
		return false
	}
	return e.reconstructOn()
}

func (e *Engine) freqVQEnabled() bool {
	if e.Extra.NoFreqVQ {
		return false
	}
	return e.Extra.FreqVQ
}

func (e *Engine) centroidEnabled() bool {
	if e.Extra.NoExpertCentroid {
		return false
	}
	return e.Extra.ExpertCentroid
}

func (e *Engine) reconstructEnabled() bool {
	return e.hadamardEnabled() || e.cskEnabled() || e.permuteEnabled() ||
		e.magrEnabled() || e.lwcEnabled() || e.freqVQEnabled() || e.centroidEnabled()
}

func (e *Engine) encodeEnabled() bool {
	return e.Extra.Encode
}

func (e *Engine) gptqEnabled() bool {
	return e.encodeEnabled() && !e.Extra.NoGPTQ
}

func (e *Engine) viterbiEnabled() bool {
	return e.encodeEnabled() && !e.Extra.NoViterbi
}

func (e *Engine) ftiEnabled() bool {
	if e.Extra.NoFTI {
		return false
	}
	if e.Extra.FTI {
		return true
	}
	return e.reconstructOn()
}

func (e *Engine) solverFTIEnabled() bool {
	if e.Extra.NoFTI {
		return false
	}
	return e.effortProfile().SolverFTI
}

// applySolverFTI sharpens in-memory imatrix stats for exact-loss and Solve.
// llama-quantize still consumes the original measured matrix. Skip when a
// sharpened GGUF is already the active imatrix (CLI -fti / Reconstruct).
func (e *Engine) applySolverFTI(stats map[string]profile.ImatrixStats) {
	if !e.solverFTIEnabled() || len(stats) == 0 {
		return
	}
	if p := e.Extra.FTIImatrixPath; p != "" && samePath(e.imatrixPath(), p) {
		return
	}
	profile.SharpenStats(stats, profile.DefaultFTIPower)
}

func (e *Engine) loadSolverImatrix(bank *core.TensorBank) (map[string]profile.ImatrixStats, error) {
	ip := e.imatrixPath()
	if ip == "" {
		return nil, nil
	}
	stats, err := profile.LoadImatrix(ip)
	if err != nil {
		return nil, err
	}
	joined := profile.JoinExpertImatrix(stats, bank)
	e.applySolverFTI(joined)
	return joined, nil
}

func (e *Engine) probeKLDEnabled() bool {
	if e.Extra.NoProbeKLD {
		return false
	}
	if e.Extra.ProbeKLD {
		return true
	}
	return e.effortProfile().ProbeKLD
}

func (e *Engine) stageFTI(ctx context.Context) error {
	if !e.ftiEnabled() {
		return nil
	}
	src := e.Run.Config.ImatrixPath
	if src == "" {
		e.printf("  fti: no imatrix configured; skipped\n")
		return nil
	}
	if e.Extra.FTIImatrixPath != "" {
		if _, err := os.Stat(e.Extra.FTIImatrixPath); err == nil {
			return nil
		}
	}
	if e.DryRun {
		e.printf("plan: FTI-sharpened imatrix GGUF\n")
		return nil
	}
	dst := filepath.Join(e.workDir(), "fti-imatrix.gguf")
	if err := os.MkdirAll(e.workDir(), 0o755); err != nil {
		return err
	}
	if err := profile.WriteSharpenedImatrix(src, dst, profile.DefaultFTIPower); err != nil {
		e.printf("  fti: skipped (%v)\n", err)
		return nil
	}
	e.Extra.FTIImatrixPath = dst
	if err := e.saveExtra(); err != nil {
		return fmt.Errorf("fti: persist extra config: %w", err)
	}
	e.printf("  fti: sharpened imatrix -> %s\n", dst)
	return nil
}

// ensurePrivatePayload copies the library GGUF into the work dir when fold
// or reconstruct needs a writable source. Prefers a filesystem clone so the
// extra file is cheap until in-place rewrites dirty pages.
func (e *Engine) ensurePrivatePayload() error {
	if !e.foldEnabled() && !e.reconstructEnabled() {
		return nil
	}
	src := e.weightPayloadSource()
	if e.jobOwnsPath(src) {
		return nil
	}
	if e.DryRun {
		e.printf("plan: clone source to job-private payload\n")
		return nil
	}
	if p := e.Extra.PrivateSourcePath; p != "" {
		if _, err := os.Stat(p); err == nil {
			return nil
		}
	}
	if err := os.MkdirAll(e.workDir(), 0o755); err != nil {
		return err
	}
	dst := filepath.Join(e.workDir(), "payload.gguf")
	if err := tensorbank.CloneFile(src, dst); err != nil {
		return fmt.Errorf("pipeline: clone job-private source: %w", err)
	}
	e.Extra.PrivateSourcePath = dst
	if err := e.saveExtra(); err != nil {
		return fmt.Errorf("pipeline: persist private payload: %w", err)
	}
	e.printf("  payload: job-private clone -> %s\n", dst)
	return nil
}

func (e *Engine) stageReconstruct(ctx context.Context) error {
	if !e.reconstructEnabled() || e.Run.Bank == nil {
		return nil
	}
	if e.Extra.ReconstructedSourcePath != "" {
		if _, err := os.Stat(e.Extra.ReconstructedSourcePath); err == nil {
			return nil
		}
	}
	if e.DryRun {
		e.printf("plan: reconstruct GGUF (permute=%v hadamard=%v magr=%v lwc=%v csk=%v freqvq=%v centroid=%v)\n",
			e.permuteEnabled(), e.hadamardEnabled(), e.magrEnabled(), e.lwcEnabled(),
			e.cskEnabled(), e.freqVQEnabled(), e.centroidEnabled())
		return nil
	}
	srcPath := e.payloadSource()
	src, err := tensorbank.OpenSource(srcPath)
	if err != nil {
		return fmt.Errorf("reconstruct: open source: %w", err)
	}
	defer src.Close()
	var imatrix map[string]profile.ImatrixStats
	if ip := e.imatrixPath(); ip != "" && (e.cskEnabled() || e.magrEnabled() || e.lwcEnabled() || e.permuteEnabled()) {
		stats, err := profile.LoadImatrix(ip)
		if err != nil {
			e.printf("  reconstruct: csk skipped (%v)\n", err)
		} else {
			imatrix = stats
		}
	}
	outPath := srcPath
	if !e.jobOwnsPath(srcPath) {
		e.printf("  reconstruct: skip Hadamard/CSK (source %s is not a job-private GGUF; in-place rewrite would mutate the library file)\n", srcPath)
		return nil
	}
	if err := os.MkdirAll(e.workDir(), 0o755); err != nil {
		return err
	}
	flag := filepath.Join(e.workDir(), "reconstruct.inprogress")
	if _, err := os.Stat(flag); err == nil {
		return fmt.Errorf("pipeline: in-place reconstruct was interrupted; restore the job-private source and delete %s", flag)
	}
	if err := os.WriteFile(flag, []byte("in-place\n"), 0o600); err != nil {
		return fmt.Errorf("reconstruct: write in-progress marker: %w", err)
	}
	res, err := reconstruct.Apply(src, outPath, reconstruct.Options{
		Permute:            e.permuteEnabled(),
		Hadamard:           e.hadamardEnabled(),
		MagR:               e.magrEnabled(),
		LWC:                e.lwcEnabled(),
		CSK:                e.cskEnabled(),
		FreqVQ:             e.freqVQEnabled(),
		ExpertCentroid:     e.centroidEnabled(),
		Imatrix:            imatrix,
		Context:            ctx,
		MaxWorkingSetBytes: e.Extra.CSKMaxWorkingSetBytes,
		Progress: func(p reconstruct.Progress) {
			if p.Layers <= 0 {
				return
			}
			layerFraction := 0.0
			if p.Total > 0 {
				layerFraction = float64(p.Current) / float64(p.Total)
			}
			fraction := (float64(p.Layer-1) + layerFraction) / float64(p.Layers)
			e.obsProgress(core.StageAssemble, fraction,
				fmt.Sprintf("CSK layer %d/%d: %s", p.Layer, p.Layers, p.Detail))
		},
	})
	if err != nil {
		return fmt.Errorf("reconstruct: %w", err)
	}
	if !res.Written {
		_ = os.Remove(flag)
		if res.SkipHadamard != "" {
			e.printf("  reconstruct: hadamard skipped (%s)\n", res.SkipHadamard)
		}
		if res.SkipCSK != "" {
			e.printf("  reconstruct: csk skipped (%s)\n", res.SkipCSK)
		}
		return nil
	}
	e.Extra.ReconstructedSourcePath = outPath
	e.Extra.PayloadSHA = ""
	e.Extra.PayloadSHAPath = ""
	if ip := e.imatrixPath(); ip != "" && res.Hadamard {
		impSrc, err := tensorbank.OpenSource(ip)
		if err != nil {
			return fmt.Errorf("reconstruct: open imatrix: %w", err)
		}
		impOut := filepath.Join(e.workDir(), "reconstructed-imatrix.gguf")
		if err := reconstruct.ApplyImatrix(impSrc, impOut); err != nil {
			impSrc.Close()
			return fmt.Errorf("reconstruct: imatrix: %w", err)
		}
		impSrc.Close()
		e.Extra.ReconstructedImatrixPath = impOut
	}
	// Once reconstruction has a durable replacement, prior model-sized
	// transform copies are no longer needed for resume or downstream payload
	// assembly. Keep an imatrix predecessor only when CSK ran without a
	// Hadamard imatrix rewrite.
	if p := e.Extra.FoldedSourcePath; generatedUnder(e.workDir(), p) {
		if samePath(p, outPath) {
			e.Extra.FoldedSourcePath = ""
		} else if err := os.Remove(p); err == nil || os.IsNotExist(err) {
			e.Extra.FoldedSourcePath = ""
		}
	}
	if e.Extra.ReconstructedImatrixPath != "" {
		for _, old := range []*string{&e.Extra.FoldedImatrixPath, &e.Extra.FTIImatrixPath} {
			if generatedUnder(e.workDir(), *old) {
				if err := os.Remove(*old); err == nil || os.IsNotExist(err) {
					*old = ""
				}
			}
		}
	}
	if err := e.saveExtra(); err != nil {
		return fmt.Errorf("reconstruct: persist extra config: %w", err)
	}
	_ = os.Remove(flag)
	e.printf("  reconstruct: hadamard=%v csk_layers=%d in_place=%v -> %s\n", res.Hadamard, res.CSKLayers, res.InPlace, outPath)
	return nil
}

func (e *Engine) jobOwnsPath(path string) bool {
	if path == "" {
		return false
	}
	wd := e.workDir()
	if generatedUnder(wd, path) {
		return true
	}
	// Studio job dir is <job>/ql/work; SSM-repaired sources live in <job>/.
	if filepath.Base(wd) == "work" && filepath.Base(filepath.Dir(wd)) == "ql" {
		return generatedUnder(filepath.Dir(filepath.Dir(wd)), path)
	}
	return false
}

func generatedUnder(dir, path string) bool {
	if dir == "" || path == "" {
		return false
	}
	rel, err := filepath.Rel(dir, path)
	if err != nil || rel == "." || rel == ".." || filepath.IsAbs(rel) {
		return false
	}
	return len(rel) < 3 || rel[:3] != ".."+string(filepath.Separator)
}
