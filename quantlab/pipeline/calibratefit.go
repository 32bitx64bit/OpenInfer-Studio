package pipeline

import (
	"bytes"
	"os"
	"path/filepath"

	"quantlab/profile"
)

// fitCalibration refits the model-level calibration from the measured loss
// cache the search ingest just grew, and persists it to the store sidecar
// so later solves and later runs of the same model start closer to truth.
// Errors are logged, never fatal: calibration improves estimates but no
// gate or budget depends on it.
func (e *Engine) fitCalibration() error {
	if e.DryRun || e.Run == nil || e.Run.Bank == nil {
		return nil
	}
	cache := e.lossCache
	if cache == nil || len(cache.Entries) < 12 {
		return nil
	}
	var imatrix map[string]profile.ImatrixStats
	if p := e.imatrixPath(); p != "" {
		if stats, err := profile.LoadImatrix(p); err == nil {
			imatrix = profile.JoinExpertImatrix(stats, e.Run.Bank)
		}
	}
	est := profile.NewFallbackEstimator(imatrix)
	est.BindBank(e.Run.Bank)
	cal := profile.FitCalibration(e.Run.Bank, cache, est, "quantlab/"+e.Run.RunID)
	if cal == nil {
		return nil
	}
	store := profile.NewCalibrationStore(e.Run.Bank.ModelID, e.Run.Bank.SHA256, e.Run.Bank.Arch)
	store.Levels[profile.LevelModel] = cal
	store.Samples[profile.LevelModel] = cal.Samples
	p := e.calibrationPath()
	if p == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		e.printf("  calibration: save skipped: %v\n", err)
		return nil
	}
	var buf bytes.Buffer
	if err := profile.SaveCalibration(&buf, store); err != nil {
		e.printf("  calibration: save skipped: %v\n", err)
		return nil
	}
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, buf.Bytes(), 0o644); err != nil {
		e.printf("  calibration: save skipped: %v\n", err)
		return nil
	}
	if err := os.Rename(tmp, p); err != nil {
		e.printf("  calibration: save skipped: %v\n", err)
		return nil
	}
	e.printf("  calibration: fitted %d samples (R2 %.3f)\n", cal.Samples, cal.R2)
	return nil
}
