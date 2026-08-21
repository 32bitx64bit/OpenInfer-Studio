package pipeline

import (
	"bytes"
	"os"
	"path/filepath"
	"time"

	"quantlab/profile"
)

func (e *Engine) lossCacheWorkPath() string {
	if e.Run == nil {
		return ""
	}
	return filepath.Join(e.workDir(), "loss-cache.json")
}

func (e *Engine) lossCacheStorePath() string {
	if e.Run == nil || e.Run.RunID == "" || e.Store.Dir == "" {
		return ""
	}
	return filepath.Join(e.Store.Dir, e.Run.RunID+".loss-cache.json")
}

func (e *Engine) stamp() time.Time {
	if e.now != nil {
		return e.now()
	}
	return time.Now()
}

// persistLossCache writes the in-memory cache to workDir and the store
// sidecar. No-op when nothing has been loaded or ingested.
func (e *Engine) persistLossCache() error {
	if e.DryRun || e.lossCache == nil || len(e.lossCache.Entries) == 0 {
		return nil
	}
	if err := e.saveLossCache(e.lossCache); err != nil {
		e.printf("  loss-cache: save skipped: %v\n", err)
	}
	return nil
}

func (e *Engine) saveLossCache(c *profile.Cache) error {
	if c == nil {
		return nil
	}
	write := func(path string) error {
		if path == "" {
			return nil
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return err
		}
		var buf bytes.Buffer
		if err := c.Save(&buf); err != nil {
			return err
		}
		tmp := path + ".tmp"
		if err := os.WriteFile(tmp, buf.Bytes(), 0o644); err != nil {
			return err
		}
		return os.Rename(tmp, path)
	}
	if err := write(e.lossCacheWorkPath()); err != nil {
		return err
	}
	return write(e.lossCacheStorePath())
}

// ingestSearchLossCache attributes accepted KLD search steps into the measured
// loss cache. Empty SearchHistory is a no-op. Errors are logged, never fatal.
func (e *Engine) ingestSearchLossCache() error {
	if e.Run == nil || len(e.Run.SearchHistory) == 0 {
		return nil
	}
	if e.DryRun {
		return nil
	}
	bank := e.Run.Bank
	if bank == nil {
		return nil
	}
	c := e.lossCache
	if c == nil {
		c = profile.NewCache(bank.ModelID, bank.SHA256)
	}
	n, err := profile.IngestKLDHistory(c, e.Run.SearchHistory, e.Run.MoveGroups, e.Run.RunID, e.stamp())
	if err != nil {
		e.printf("  loss-cache: ingest skipped: %v\n", err)
		return nil
	}
	e.lossCache = c
	if n == 0 {
		return nil
	}
	if err := e.saveLossCache(c); err != nil {
		e.printf("  loss-cache: save skipped: %v\n", err)
		return nil
	}
	e.printf("  loss-cache: ingested %d search entries\n", n)
	return nil
}

// clearRunLossCache deletes the run's measured-loss cache (work copy and
// store sidecar) once the run has emitted successfully. Mid-run resumes
// still find it; a completed run leaves nothing behind to replay into the
// decide stage of future runs. Deletion errors are logged, never fatal.
func (e *Engine) clearRunLossCache() error {
	if e.DryRun {
		return nil
	}
	cleared := false
	for _, p := range []string{e.lossCacheWorkPath(), e.lossCacheStorePath()} {
		if p == "" {
			continue
		}
		if err := os.Remove(p); err == nil {
			cleared = true
		} else if !os.IsNotExist(err) {
			e.printf("  loss-cache: cleanup skipped: %v\n", err)
		}
	}
	if cleared {
		e.lossCache = nil
	}
	return nil
}
