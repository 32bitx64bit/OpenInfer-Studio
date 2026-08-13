// Package config locates and creates the application data directories
// (database, runtimes, models, downloads, caches, logs, presets, temp).
package config

import (
	"fmt"
	"os"
	"path/filepath"
)

const appDirName = "openinfer-studio"

// Layout describes the application-managed directory tree. All paths are
// absolute after Open() succeeds.
type Layout struct {
	ConfigDir string // platform config dir (settings export, manual files)
	DataDir   string // platform data dir root
	CacheDir  string // platform cache dir

	Database    string // <data>/database
	Runtimes    string // <data>/runtimes
	Models      string // <data>/models
	Downloads   string // <data>/downloads
	Partial     string // <data>/downloads/partial
	Active      string // <data>/downloads/active
	HFCache     string // <data>/cache/huggingface
	MetaCache   string // <data>/cache/metadata
	AppLogs     string // <data>/logs/application
	InstLogs    string // <data>/logs/instances
	Presets     string // <data>/presets
	Sessions    string // <data>/sessions
	Temp        string // <data>/temp
	Attachments string // <data>/attachments (chat audio/files)

	QuantJobs        string // <data>/quant/jobs
	QuantIMatrices   string // <data>/quant/imatrices
	QuantCalibration string // <data>/quant/calibration
	QuantLogs        string // <data>/logs/quant
}

// Open resolves the layout, creating every directory. If dataOverride is
// non-empty it replaces the platform data dir (used by tests and portable
// mode).
func Open(dataOverride string) (*Layout, error) {
	cfgDir, err := os.UserConfigDir()
	if err != nil {
		return nil, fmt.Errorf("locating config dir: %w", err)
	}
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return nil, fmt.Errorf("locating cache dir: %w", err)
	}
	dataDir := dataOverride
	if dataDir == "" {
		dataDir, err = platformDataDir()
		if err != nil {
			return nil, fmt.Errorf("locating data dir: %w", err)
		}
	}

	l := &Layout{
		ConfigDir: filepath.Join(cfgDir, appDirName),
		DataDir:   dataDir,
		CacheDir:  filepath.Join(cacheDir, appDirName),
	}
	l.Database = filepath.Join(dataDir, "database")
	l.Runtimes = filepath.Join(dataDir, "runtimes")
	l.Models = filepath.Join(dataDir, "models")
	l.Downloads = filepath.Join(dataDir, "downloads")
	l.Partial = filepath.Join(l.Downloads, "partial")
	l.Active = filepath.Join(l.Downloads, "active")
	l.HFCache = filepath.Join(dataDir, "cache", "huggingface")
	l.MetaCache = filepath.Join(dataDir, "cache", "metadata")
	l.AppLogs = filepath.Join(dataDir, "logs", "application")
	l.InstLogs = filepath.Join(dataDir, "logs", "instances")
	l.Presets = filepath.Join(dataDir, "presets")
	l.Sessions = filepath.Join(dataDir, "sessions")
	l.Temp = filepath.Join(dataDir, "temp")
	l.Attachments = filepath.Join(dataDir, "attachments")
	l.QuantJobs = filepath.Join(dataDir, "quant", "jobs")
	l.QuantIMatrices = filepath.Join(dataDir, "quant", "imatrices")
	l.QuantCalibration = filepath.Join(dataDir, "quant", "calibration")
	l.QuantLogs = filepath.Join(dataDir, "logs", "quant")

	for _, d := range []string{
		l.ConfigDir, l.DataDir, l.CacheDir, l.Database, l.Runtimes, l.Models,
		l.Partial, l.Active, l.HFCache, l.MetaCache, l.AppLogs, l.InstLogs,
		l.Presets, l.Sessions, l.Temp, l.Attachments,
		l.QuantJobs, l.QuantIMatrices, l.QuantCalibration, l.QuantLogs,
	} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return nil, fmt.Errorf("creating %s: %w", d, err)
		}
	}
	return l, nil
}

// TempDir returns a fresh unique directory inside the managed temp tree.
func (l *Layout) TempDir(prefix string) (string, error) {
	return os.MkdirTemp(l.Temp, prefix)
}
