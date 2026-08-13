// Package runtimes manages installed llama.cpp builds: release discovery,
// hardware-aware asset selection, verified install, manifests, capabilities.
// Installed builds are immutable; updates stage then replace atomically.
package runtimes

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/openinfer/openinfer-studio/internal/downloads"
)

// Manifest is stored as manifest.json inside every installed runtime dir.
type Manifest struct {
	RuntimeID           string         `json:"runtime_id"`
	Source              string         `json:"source"` // official-release|custom-import
	ReleaseID           string         `json:"release_id"`
	Build               string         `json:"build"`
	Commit              string         `json:"commit"`
	Platform            string         `json:"platform"`
	Architecture        string         `json:"architecture"`
	Backend             string         `json:"backend"`
	DownloadURLIdentity string         `json:"download_url_identity"`
	ArchiveSHA256       string         `json:"archive_sha256"`
	InstalledAt         string         `json:"installed_at"`
	ExecutablePath      string         `json:"executable_path"`
	VersionOutput       string         `json:"version_output"`
	Capabilities        map[string]any `json:"capabilities"`
}

// Runtime is the UI view of an installed build.
type Runtime struct {
	ID             string   `json:"id"`
	Source         string   `json:"source"`
	ReleaseID      string   `json:"release_id"`
	Build          string   `json:"build"`
	Commit         string   `json:"commit"`
	Platform       string   `json:"platform"`
	Architecture   string   `json:"architecture"`
	Backend        string   `json:"backend"`
	DownloadURL    string   `json:"download_url"`
	ArchiveSHA256  string   `json:"archive_sha256"`
	InstalledAt    string   `json:"installed_at"`
	InstallDir     string   `json:"install_dir"`
	ExecutablePath string   `json:"executable_path"`
	VersionOutput  string   `json:"version_output"`
	HelpOutput     string   `json:"help_output,omitempty"`
	Preferred      bool     `json:"preferred"`
	Healthy        bool     `json:"healthy"`
	Capabilities   []string `json:"capabilities"`
	UsedByModels   []string `json:"used_by_models"`
}

type EventSink interface {
	Publish(event string, payload any)
}

type Manager struct {
	db     *sql.DB
	dir    string // <data>/runtimes
	dl     *downloads.Manager
	events EventSink
	log    *slog.Logger
	feed   *ReleaseFeed
}

func NewManager(db *sql.DB, dir string, dl *downloads.Manager, events EventSink, log *slog.Logger) *Manager {
	return &Manager{db: db, dir: dir, dl: dl, events: events, log: log, feed: NewReleaseFeed()}
}

// SetFeed overrides the release feed (tests).
func (m *Manager) SetFeed(f *ReleaseFeed) { m.feed = f }

// Feed returns the release feed (for the check-for-updates endpoint).
func (m *Manager) Feed() *ReleaseFeed { return m.feed }

var now = func() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// List returns all installed runtimes with capabilities and pinning info.
func (m *Manager) List() ([]Runtime, error) {
	rows, err := m.db.Query(`SELECT id,source,release_id,build,commit_hash,platform,architecture,backend,
		download_url,archive_sha256,installed_at,install_dir,executable_path,version_output,preferred,healthy
		FROM runtimes ORDER BY installed_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Runtime
	for rows.Next() {
		var r Runtime
		var pref, healthy int
		if err := rows.Scan(&r.ID, &r.Source, &r.ReleaseID, &r.Build, &r.Commit, &r.Platform,
			&r.Architecture, &r.Backend, &r.DownloadURL, &r.ArchiveSHA256, &r.InstalledAt,
			&r.InstallDir, &r.ExecutablePath, &r.VersionOutput, &pref, &healthy); err != nil {
			return nil, err
		}
		r.Preferred = pref == 1
		r.Healthy = healthy == 1
		// Correct custom imports that were recorded as cpu before backend
		// detection existed. Only upgrade from cpu when evidence finds a GPU
		// backend — never downgrade a stored label without path hints.
		if r.Source == "custom-import" && r.Backend == BackendCPU {
			if detected := DetectBackend(r.VersionOutput, nil, r.InstallDir, filepath.Dir(r.ExecutablePath)); detected != BackendCPU {
				if _, err := m.db.Exec(`UPDATE runtimes SET backend = ? WHERE id = ?`, detected, r.ID); err == nil {
					r.Backend = detected
				}
			}
		}
		r.Capabilities, _ = m.capabilities(r.ID)
		if r.Capabilities == nil {
			r.Capabilities = []string{}
		}
		r.UsedByModels, _ = m.usedBy(r.ID)
		if r.UsedByModels == nil {
			r.UsedByModels = []string{}
		}
		out = append(out, r)
	}
	return out, nil
}

func (m *Manager) capabilities(id string) ([]string, error) {
	rows, err := m.db.Query(`SELECT capability FROM runtime_capabilities WHERE runtime_id = ? AND supported = 1`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var c string
		if rows.Scan(&c) == nil {
			out = append(out, c)
		}
	}
	return out, nil
}

func (m *Manager) usedBy(id string) ([]string, error) {
	rows, err := m.db.Query(`SELECT COALESCE(NULLIF(alias,''), id) FROM models WHERE pinned_runtime = ?`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var s string
		if rows.Scan(&s) == nil {
			out = append(out, s)
		}
	}
	return out, nil
}

// Get returns one runtime by ID.
func (m *Manager) Get(id string) (*Runtime, error) {
	all, err := m.List()
	if err != nil {
		return nil, err
	}
	for i := range all {
		if all[i].ID == id {
			return &all[i], nil
		}
	}
	return nil, fmt.Errorf("runtime %s not found", id)
}

// HelpOutput loads the raw --help text for a runtime (kept on disk, not DB).
func (m *Manager) HelpOutput(id string) (string, error) {
	r, err := m.Get(id)
	if err != nil {
		return "", err
	}
	b, err := os.ReadFile(filepath.Join(r.InstallDir, "help.txt"))
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// Install stages, verifies and atomically installs an official release
// asset. progress events are emitted via the downloads manager.
func (m *Manager) Install(ctx context.Context, rel Release, match AssetMatch) (string, error) {
	staging, err := os.MkdirTemp(filepath.Dir(m.dir), ".staging-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(staging)

	archivePath := filepath.Join(staging, match.Asset.Name)
	id, err := m.dl.Enqueue("runtime", "llama.cpp "+rel.Tag+" "+match.Backend, staging,
		[]downloads.FileSpec{{URL: match.Asset.DownloadURL, DestPath: archivePath, Size: match.Asset.Size}},
		map[string]any{"release": rel.Tag, "backend": match.Backend})
	if err != nil {
		return "", fmt.Errorf("enqueue runtime download: %w", err)
	}
	if _, err := m.dl.WaitComplete(ctx, id); err != nil {
		return "", fmt.Errorf("runtime download failed: %w", err)
	}

	sum, err := sha256File(archivePath)
	if err != nil {
		return "", err
	}

	extractDir := filepath.Join(staging, "extracted")
	if err := os.MkdirAll(extractDir, 0o755); err != nil {
		return "", err
	}
	if _, err := ExtractArchive(archivePath, extractDir); err != nil {
		return "", fmt.Errorf("extracting runtime: %w", err)
	}

	exe, err := findServerExecutable(extractDir)
	if err != nil {
		return "", err
	}
	if runtime.GOOS != "windows" {
		_ = os.Chmod(exe, 0o755)
	}

	versionOut, helpOut, err := probeRuntime(exe)
	if err != nil {
		return "", fmt.Errorf("runtime smoke test failed: %w", err)
	}
	caps := ParseCapabilities(helpOut)

	runtimeID := fmt.Sprintf("%s-%s-%s-%s", rel.Tag, runtime.GOOS, runtime.GOARCH, match.Backend)
	// Ensure uniqueness if the same build is reinstalled.
	if _, err := m.Get(runtimeID); err == nil {
		runtimeID = runtimeID + "-" + uuid.NewString()[:8]
	}
	finalDir := filepath.Join(m.dir, runtimeID)

	man := Manifest{
		RuntimeID: runtimeID, Source: "official-release", ReleaseID: rel.Tag,
		Build: rel.Tag, Platform: runtime.GOOS, Architecture: runtime.GOARCH,
		Backend: match.Backend, DownloadURLIdentity: match.Asset.DownloadURL,
		ArchiveSHA256: sum, InstalledAt: now(),
		ExecutablePath: exe, VersionOutput: versionOut,
		Capabilities: map[string]any{"flags": caps},
	}

	// Record the executable path relative to the final directory.
	relExe, err := filepath.Rel(extractDir, exe)
	if err == nil {
		man.ExecutablePath = filepath.Join(finalDir, relExe)
	}

	// Finalize layout: manifest + help text next to the binaries.
	if err := os.WriteFile(filepath.Join(extractDir, "help.txt"), []byte(helpOut), 0o644); err != nil {
		return "", err
	}
	manBytes, _ := json.MarshalIndent(man, "", "  ")
	if err := os.WriteFile(filepath.Join(extractDir, "manifest.json"), manBytes, 0o644); err != nil {
		return "", err
	}

	// Atomic move into the immutable final directory.
	if err := os.MkdirAll(m.dir, 0o755); err != nil {
		return "", err
	}
	if err := os.Rename(extractDir, finalDir); err != nil {
		return "", fmt.Errorf("installing runtime into place: %w", err)
	}

	if err := m.record(man, helpOut, caps, finalDir); err != nil {
		os.RemoveAll(finalDir)
		return "", err
	}
	m.log.Info("runtime installed", "id", runtimeID, "backend", match.Backend)
	if m.events != nil {
		m.events.Publish("runtime.installed", map[string]any{"id": runtimeID, "backend": match.Backend})
	}
	return runtimeID, nil
}

func (m *Manager) record(man Manifest, helpOut string, caps []string, dir string) error {
	tx, err := m.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`INSERT INTO runtimes
		(id,source,release_id,build,commit_hash,platform,architecture,backend,download_url,
		 archive_sha256,installed_at,install_dir,executable_path,version_output,preferred,healthy)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)`,
		man.RuntimeID, man.Source, man.ReleaseID, man.Build, man.Commit, man.Platform,
		man.Architecture, man.Backend, man.DownloadURLIdentity, man.ArchiveSHA256,
		man.InstalledAt, dir, man.ExecutablePath, man.VersionOutput, 0); err != nil {
		return err
	}
	for _, c := range caps {
		if _, err := tx.Exec(`INSERT OR REPLACE INTO runtime_capabilities(runtime_id,capability,supported)
			VALUES (?,?,1)`, man.RuntimeID, c); err != nil {
			return err
		}
	}
	// First installed runtime becomes preferred automatically.
	var n int
	if err := tx.QueryRow(`SELECT COUNT(1) FROM runtimes`).Scan(&n); err == nil && n == 1 {
		_, _ = tx.Exec(`UPDATE runtimes SET preferred = 1 WHERE id = ?`, man.RuntimeID)
	}
	return tx.Commit()
}

func isArchivePath(path string) bool {
	lower := strings.ToLower(path)
	return strings.HasSuffix(lower, ".zip") ||
		strings.HasSuffix(lower, ".tar.gz") ||
		strings.HasSuffix(lower, ".tgz")
}

// ImportCustom registers a user-provided llama-server executable or a
// prebuilt archive (.zip / .tar.gz / .tgz). Bare executables stay in place;
// archives are extracted into the managed runtimes directory.
func (m *Manager) ImportCustom(path string) (string, error) {
	st, err := os.Stat(path)
	if err != nil || st.IsDir() {
		return "", fmt.Errorf("invalid path %q", path)
	}
	if isArchivePath(path) {
		return m.importCustomArchive(path)
	}
	return m.importCustomExecutable(path)
}

func (m *Manager) importCustomExecutable(exePath string) (string, error) {
	versionOut, helpOut, err := probeRuntime(exePath)
	if err != nil {
		return "", fmt.Errorf("custom runtime failed smoke test: %w", err)
	}
	caps := ParseCapabilities(helpOut)
	backend := DetectBackend(versionOut, []string{exePath}, filepath.Dir(exePath))
	id := "custom-" + uuid.NewString()[:8]
	man := Manifest{
		RuntimeID: id, Source: "custom-import", Platform: runtime.GOOS,
		Architecture: runtime.GOARCH, Backend: backend, InstalledAt: now(),
		ExecutablePath: exePath, VersionOutput: versionOut,
		Capabilities: map[string]any{"flags": caps},
	}
	// Persist help text alongside our records without touching the user's dir.
	recDir := filepath.Join(m.dir, id)
	if err := os.MkdirAll(recDir, 0o755); err != nil {
		return "", err
	}
	_ = os.WriteFile(filepath.Join(recDir, "help.txt"), []byte(helpOut), 0o644)
	manBytes, _ := json.MarshalIndent(man, "", "  ")
	_ = os.WriteFile(filepath.Join(recDir, "manifest.json"), manBytes, 0o644)
	if err := m.record(man, helpOut, caps, recDir); err != nil {
		return "", err
	}
	return id, nil
}

func (m *Manager) importCustomArchive(archivePath string) (string, error) {
	staging, err := os.MkdirTemp(filepath.Dir(m.dir), ".staging-import-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(staging)

	extractDir := filepath.Join(staging, "extracted")
	if err := os.MkdirAll(extractDir, 0o755); err != nil {
		return "", err
	}
	if _, err := ExtractArchive(archivePath, extractDir); err != nil {
		return "", fmt.Errorf("extracting archive: %w", err)
	}

	exe, err := findServerExecutable(extractDir)
	if err != nil {
		return "", err
	}
	if runtime.GOOS != "windows" {
		_ = os.Chmod(exe, 0o755)
	}

	versionOut, helpOut, err := probeRuntime(exe)
	if err != nil {
		return "", fmt.Errorf("custom runtime failed smoke test: %w", err)
	}
	caps := ParseCapabilities(helpOut)

	id := "custom-" + uuid.NewString()[:8]
	finalDir := filepath.Join(m.dir, id)
	relExe, err := filepath.Rel(extractDir, exe)
	if err != nil {
		return "", err
	}
	backend := DetectBackend(versionOut, []string{archivePath, filepath.Base(archivePath)}, extractDir)
	man := Manifest{
		RuntimeID: id, Source: "custom-import", Platform: runtime.GOOS,
		Architecture: runtime.GOARCH, Backend: backend, InstalledAt: now(),
		ExecutablePath: filepath.Join(finalDir, relExe), VersionOutput: versionOut,
		Capabilities: map[string]any{"flags": caps},
	}

	if err := os.WriteFile(filepath.Join(extractDir, "help.txt"), []byte(helpOut), 0o644); err != nil {
		return "", err
	}
	manBytes, _ := json.MarshalIndent(man, "", "  ")
	if err := os.WriteFile(filepath.Join(extractDir, "manifest.json"), manBytes, 0o644); err != nil {
		return "", err
	}

	if err := os.MkdirAll(m.dir, 0o755); err != nil {
		return "", err
	}
	if err := os.Rename(extractDir, finalDir); err != nil {
		return "", fmt.Errorf("installing imported runtime: %w", err)
	}
	if err := m.record(man, helpOut, caps, finalDir); err != nil {
		os.RemoveAll(finalDir)
		return "", err
	}
	return id, nil
}

// Remove deletes an installed runtime. Runtimes pinned by models are refused.
func (m *Manager) Remove(id string) error {
	used, err := m.usedBy(id)
	if err != nil {
		return err
	}
	if len(used) > 0 {
		return fmt.Errorf("runtime is pinned by models: %s", strings.Join(used, ", "))
	}
	r, err := m.Get(id)
	if err != nil {
		return err
	}
	if r.Source == "official-release" {
		// Only remove directories inside the managed runtime root.
		cleanDir := filepath.Clean(r.InstallDir)
		if filepath.Dir(cleanDir) != filepath.Clean(m.dir) {
			return fmt.Errorf("refusing to delete outside runtime root: %s", cleanDir)
		}
		if err := os.RemoveAll(cleanDir); err != nil {
			return err
		}
	} else {
		// Record dir (bare exe) or extracted tree (archive import); user's original path untouched.
		os.RemoveAll(r.InstallDir)
	}
	_, err = m.db.Exec(`DELETE FROM runtimes WHERE id = ?`, id)
	return err
}

// SetPreferred marks one runtime as the global default.
func (m *Manager) SetPreferred(id string) error {
	tx, err := m.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE runtimes SET preferred = 0`); err != nil {
		return err
	}
	res, err := tx.Exec(`UPDATE runtimes SET preferred = 1 WHERE id = ?`, id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return fmt.Errorf("runtime %s not found", id)
	}
	return tx.Commit()
}

// Preferred returns the globally preferred runtime, or nil.
func (m *Manager) Preferred() (*Runtime, error) {
	all, err := m.List()
	if err != nil {
		return nil, err
	}
	for i := range all {
		if all[i].Preferred {
			return &all[i], nil
		}
	}
	return nil, nil
}

// HealthCheck reruns the smoke test and records the outcome.
func (m *Manager) HealthCheck(id string) (bool, string, error) {
	r, err := m.Get(id)
	if err != nil {
		return false, "", err
	}
	versionOut, _, err := probeRuntime(r.ExecutablePath)
	ok := err == nil
	h := 0
	if ok {
		h = 1
	}
	_, _ = m.db.Exec(`UPDATE runtimes SET healthy = ? WHERE id = ?`, h, id)
	if err != nil {
		return false, "", err
	}
	// Refresh sibling-tool catalogs while we're probing.
	_, _ = m.Tools(id)
	return true, versionOut, nil
}

// findServerExecutable locates llama-server inside an extracted tree.
func findServerExecutable(root string) (string, error) {
	want := "llama-server"
	if runtime.GOOS == "windows" {
		want = "llama-server.exe"
	}
	var found string
	err := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.EqualFold(d.Name(), want) {
			found = p
			return io.EOF // stop walk early
		}
		return nil
	})
	if err != nil && !errors.Is(err, io.EOF) {
		return "", err
	}
	if found == "" {
		return "", fmt.Errorf("archive does not contain a llama-server executable")
	}
	return found, nil
}

// LibPathEnv returns environment entries putting the executable's own
// directory (and lib/ subdir) on the shared-library search path. Official
// llama.cpp archives ship libllama-*.so / DLLs next to the binaries, and the
// dynamic loader does not search there by default on Linux/macOS.
func LibPathEnv(exe string) []string {
	dir := filepath.Dir(exe)
	dirs := dir
	if fi, err := os.Stat(filepath.Join(dir, "lib")); err == nil && fi.IsDir() {
		dirs += string(os.PathListSeparator) + filepath.Join(dir, "lib")
	}
	merge := func(key string) string {
		if cur := os.Getenv(key); cur != "" {
			return key + "=" + dirs + string(os.PathListSeparator) + cur
		}
		return key + "=" + dirs
	}
	return []string{merge("LD_LIBRARY_PATH"), merge("DYLD_LIBRARY_PATH"), merge("DYLD_FALLBACK_LIBRARY_PATH")}
}

// probeRuntime runs `llama-server --version` and `--help` with a timeout.
func probeRuntime(exe string) (version, help string, err error) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	vcmd := exec.CommandContext(ctx, exe, "--version")
	vcmd.Dir = filepath.Dir(exe)
	vcmd.Env = append(os.Environ(), LibPathEnv(exe)...)
	vout, err := vcmd.CombinedOutput()
	if err != nil {
		return "", "", fmt.Errorf("--version: %w: %s", err, truncate(string(vout), 512))
	}
	help, herr := ProbeHelp(exe)
	if herr != nil {
		return string(vout), "", herr
	}
	return string(vout), help, nil
}

// ProbeHelp runs `exe --help` with the same library-path and timeout rules as
// llama-server probing. Some builds exit non-zero for --help; output is still
// accepted when present.
func ProbeHelp(exe string) (string, error) {
	hctx, hcancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer hcancel()
	hcmd := exec.CommandContext(hctx, exe, "--help")
	hcmd.Dir = filepath.Dir(exe)
	hcmd.Env = append(os.Environ(), LibPathEnv(exe)...)
	hout, herr := hcmd.CombinedOutput()
	if herr != nil && len(hout) == 0 {
		return "", fmt.Errorf("--help: %w", herr)
	}
	return string(hout), nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func sha256File(path string) (string, error) {
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
