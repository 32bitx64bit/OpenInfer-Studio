// Package downloads is the persistent queue for model and runtime downloads:
// HTTP range resume, pause/cancel/retry, disk preflight, SHA-256 verify, and
// atomic completion. Partial files are never treated as finished output.
package downloads

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
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// EventSink decouples the manager from the API hub.
type EventSink interface {
	Publish(event string, payload any)
}

// FileSpec describes one file of a (possibly multi-file) download.
type FileSpec struct {
	URL      string `json:"url"`
	DestPath string `json:"dest_path"`
	Size     int64  `json:"size"`
	SHA256   string `json:"sha256,omitempty"`
}

// Progress is the event payload for download.progress.
type Progress struct {
	ID         string  `json:"id"`
	State      string  `json:"state"`
	DoneBytes  int64   `json:"done_bytes"`
	TotalBytes int64   `json:"total_bytes"`
	SpeedBPS   float64 `json:"speed_bps"`
	ETAsec     int64   `json:"eta_seconds"`
	Error      string  `json:"error,omitempty"`
}

// DiskSpaceFunc reports free bytes for a directory (injected for tests).
type DiskSpaceFunc func(dir string) uint64

type Manager struct {
	db       *sql.DB
	partial  string
	events   EventSink
	log      *slog.Logger
	http     *http.Client
	diskFree DiskSpaceFunc

	mu      sync.Mutex
	limit   int // simultaneous download jobs
	parts   int // parallel Range connections per large file
	running int
	cancels map[string]context.CancelFunc
	authHdr func() string // optional Authorization header provider (HF token)
}

func NewManager(db *sql.DB, partialDir string, events EventSink, log *slog.Logger, disk DiskSpaceFunc) *Manager {
	return &Manager{
		db: db, partial: partialDir, events: events, log: log, diskFree: disk,
		http: &http.Client{
			Timeout: 0, // no global timeout; per-request contexts govern
			Transport: &http.Transport{
				Proxy: http.ProxyFromEnvironment,
				DialContext: (&net.Dialer{
					Timeout:   30 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,
				ForceAttemptHTTP2:     true,
				MaxIdleConns:          100,
				MaxIdleConnsPerHost:   32,
				MaxConnsPerHost:       32,
				IdleConnTimeout:       90 * time.Second,
				TLSHandshakeTimeout:   15 * time.Second,
				ResponseHeaderTimeout: 60 * time.Second,
				ExpectContinueTimeout: 1 * time.Second,
			},
		},
		limit:   2,
		parts:   8,
		cancels: map[string]context.CancelFunc{},
	}
}

// SetConcurrency changes the number of simultaneous active downloads.
func (m *Manager) SetConcurrency(n int) {
	if n < 1 {
		n = 1
	}
	if n > 8 {
		n = 8
	}
	m.mu.Lock()
	m.limit = n
	m.mu.Unlock()
	go m.pump()
}

// SetConnections changes how many parallel Range connections are used for
// each large file (Hugging Face CDN often caps a single stream).
func (m *Manager) SetConnections(n int) {
	if n < 1 {
		n = 1
	}
	if n > 16 {
		n = 16
	}
	m.mu.Lock()
	m.parts = n
	m.mu.Unlock()
}

func (m *Manager) connections() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.parts
}

// SetAuthHeader injects a provider for an Authorization header on requests.
func (m *Manager) SetAuthHeader(fn func() string) { m.authHdr = fn }

// Item is the queue view returned to the UI.
type Item struct {
	ID         string          `json:"id"`
	Kind       string          `json:"kind"`
	Label      string          `json:"label"`
	State      string          `json:"state"`
	QueuePos   int             `json:"queue_pos"`
	TotalBytes int64           `json:"total_bytes"`
	DoneBytes  int64           `json:"done_bytes"`
	DestDir    string          `json:"dest_dir"`
	Error      string          `json:"error"`
	Files      []ItemFile      `json:"files"`
	Meta       json.RawMessage `json:"meta"`
}

type ItemFile struct {
	URL        string `json:"url"`
	DestPath   string `json:"dest_path"`
	TotalBytes int64  `json:"total_bytes"`
	DoneBytes  int64  `json:"done_bytes"`
	State      string `json:"state"`
	Resumable  int    `json:"resumable"`
}

var now = func() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// Enqueue registers a new multi-file download and starts the queue.
func (m *Manager) Enqueue(kind, label, destDir string, files []FileSpec, meta any) (string, error) {
	if len(files) == 0 {
		return "", errors.New("download has no files")
	}
	id := uuid.NewString()
	metaJSON, _ := json.Marshal(meta)
	tx, err := m.db.Begin()
	if err != nil {
		return "", err
	}
	defer tx.Rollback()
	var maxPos int
	_ = tx.QueryRow(`SELECT COALESCE(MAX(queue_pos),0) FROM downloads`).Scan(&maxPos)
	var total int64
	for _, f := range files {
		total += f.Size
	}
	if _, err := tx.Exec(`INSERT INTO downloads
		(id,kind,label,state,queue_pos,total_bytes,done_bytes,dest_dir,meta_json,created_at,updated_at)
		VALUES (?,?,?,?,?,?,0,?,?,?,?)`,
		id, kind, label, "queued", maxPos+1, total, destDir, string(metaJSON), now(), now()); err != nil {
		return "", err
	}
	for _, f := range files {
		partial := filepath.Join(m.partial, id+"-"+filepath.Base(f.DestPath)+".part")
		if _, err := tx.Exec(`INSERT INTO download_files
			(id,download_id,url,dest_path,partial_path,total_bytes,done_bytes,sha256,state)
			VALUES (?,?,?,?,?,?,0,?,'queued')`,
			uuid.NewString(), id, f.URL, f.DestPath, partial, f.Size, f.SHA256); err != nil {
			return "", err
		}
	}
	if err := tx.Commit(); err != nil {
		return "", err
	}
	m.log.Info("download enqueued", "id", id, "label", label, "files", len(files))
	go m.pump()
	return id, nil
}

// List returns the queue ordered by position.
func (m *Manager) List() ([]Item, error) {
	rows, err := m.db.Query(`SELECT id,kind,label,state,queue_pos,total_bytes,done_bytes,dest_dir,error,meta_json
		FROM downloads ORDER BY queue_pos`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Item
	ids := []string{}
	for rows.Next() {
		var it Item
		var meta string
		if err := rows.Scan(&it.ID, &it.Kind, &it.Label, &it.State, &it.QueuePos,
			&it.TotalBytes, &it.DoneBytes, &it.DestDir, &it.Error, &meta); err != nil {
			return nil, err
		}
		it.Meta = json.RawMessage(meta)
		out = append(out, it)
		ids = append(ids, it.ID)
	}
	for i, id := range ids {
		fs, err := m.filesOf(id)
		if err != nil {
			return nil, err
		}
		out[i].Files = fs
	}
	return out, nil
}

func (m *Manager) filesOf(id string) ([]ItemFile, error) {
	rows, err := m.db.Query(`SELECT url,dest_path,total_bytes,done_bytes,state,resumable
		FROM download_files WHERE download_id = ? ORDER BY rowid`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ItemFile
	for rows.Next() {
		var f ItemFile
		if err := rows.Scan(&f.URL, &f.DestPath, &f.TotalBytes, &f.DoneBytes, &f.State, &f.Resumable); err != nil {
			return nil, err
		}
		out = append(out, f)
	}
	return out, nil
}

// Pause suspends an active download (partial files are kept for resume).
func (m *Manager) Pause(id string) error {
	m.mu.Lock()
	cancel, ok := m.cancels[id]
	m.mu.Unlock()
	if ok {
		cancel()
	}
	return m.setState(id, "paused", "")
}

// Resume re-queues a paused/failed/canceled download.
func (m *Manager) Resume(id string) error {
	if err := m.setState(id, "queued", ""); err != nil {
		return err
	}
	go m.pump()
	return nil
}

// Cancel aborts and removes partial files.
func (m *Manager) Cancel(id string) error {
	m.mu.Lock()
	cancel, ok := m.cancels[id]
	m.mu.Unlock()
	if ok {
		cancel()
	}
	m.cleanupPartials(id)
	return m.setState(id, "canceled", "")
}

// Retry clears the error and re-queues (keeping partial files).
func (m *Manager) Retry(id string) error { return m.Resume(id) }

// Reorder moves a queued item to a new queue position.
func (m *Manager) Reorder(id string, newPos int) error {
	_, err := m.db.Exec(`UPDATE downloads SET queue_pos = ?, updated_at = ? WHERE id = ? AND state = 'queued'`,
		newPos, now(), id)
	return err
}

// Delete removes the record and any partial files.
func (m *Manager) Delete(id string) error {
	_ = m.Cancel(id)
	_, err := m.db.Exec(`DELETE FROM downloads WHERE id = ?`, id)
	return err
}

func (m *Manager) setState(id, state, errStr string) error {
	_, err := m.db.Exec(`UPDATE downloads SET state = ?, error = ?, updated_at = ? WHERE id = ?`,
		state, errStr, now(), id)
	if err == nil && m.events != nil {
		m.events.Publish("download.state_changed", map[string]any{"id": id, "state": state, "error": errStr})
	}
	return err
}

func (m *Manager) cleanupPartials(id string) {
	rows, err := m.db.Query(`SELECT partial_path FROM download_files WHERE download_id = ?`, id)
	if err != nil {
		return
	}
	defer rows.Close()
	for rows.Next() {
		var p string
		if rows.Scan(&p) == nil && p != "" {
			os.Remove(p)
		}
	}
}

// WaitComplete blocks until the download reaches a terminal state and
// returns that state plus the final error text, if any.
func (m *Manager) WaitComplete(ctx context.Context, id string) (string, error) {
	tick := time.NewTicker(200 * time.Millisecond)
	defer tick.Stop()
	for {
		var state, errStr string
		if err := m.db.QueryRow(`SELECT state, error FROM downloads WHERE id = ?`, id).
			Scan(&state, &errStr); err != nil {
			return "", err
		}
		switch state {
		case "complete":
			return state, nil
		case "failed":
			return state, errors.New(errStr)
		case "canceled":
			return state, errors.New("download canceled")
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-tick.C:
		}
	}
}

// RecoverAfterRestart marks downloads that were active when the app last
// exited as paused; the user (or pump) can resume them.
func (m *Manager) RecoverAfterRestart() error {
	_, err := m.db.Exec(`UPDATE downloads SET state = 'paused', updated_at = ? WHERE state = 'active'`, now())
	return err
}

// pump starts queued downloads up to the concurrency limit.
func (m *Manager) pump() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.limit < 1 || m.running >= m.limit {
		return
	}
	rows, err := m.db.Query(`SELECT id FROM downloads WHERE state = 'queued' ORDER BY queue_pos LIMIT ?`,
		m.limit-m.running)
	if err != nil {
		return
	}
	var ids []string
	for rows.Next() {
		var id string
		if rows.Scan(&id) == nil {
			ids = append(ids, id)
		}
	}
	rows.Close()
	for _, id := range ids {
		// Claim atomically so a delayed pump cannot start the same item twice.
		res, err := m.db.Exec(`UPDATE downloads SET state = 'active', error = '', updated_at = ? WHERE id = ? AND state = 'queued'`,
			now(), id)
		if err != nil {
			continue
		}
		n, _ := res.RowsAffected()
		if n == 0 {
			continue
		}
		if m.events != nil {
			m.events.Publish("download.state_changed", map[string]any{"id": id, "state": "active", "error": ""})
		}
		ctx, cancel := context.WithCancel(context.Background())
		m.cancels[id] = cancel
		m.running++
		go func(id string) {
			m.run(ctx, id)
			m.mu.Lock()
			delete(m.cancels, id)
			m.running--
			m.mu.Unlock()
			go m.pump()
		}(id)
	}
}

// run executes all files of one download sequentially.
// Caller (pump) has already claimed the row as active.
func (m *Manager) run(ctx context.Context, id string) {
	files, err := m.filesOf(id)
	if err != nil {
		m.fail(id, err)
		return
	}
	// Disk preflight: remaining bytes vs free space.
	var remaining int64
	for _, f := range files {
		remaining += f.TotalBytes - f.DoneBytes
	}
	if free := m.diskFree(filepath.Dir(files[0].DestPath)); free > 0 && uint64(remaining) > free {
		m.fail(id, fmt.Errorf("insufficient disk space: need ~%d bytes, have %d", remaining, free))
		return
	}

	for _, f := range files {
		if f.State == "complete" {
			continue
		}
		if err := m.downloadFile(ctx, id, f.DestPath); err != nil {
			if errors.Is(err, context.Canceled) {
				// Pause/Cancel already set the state.
				return
			}
			m.fail(id, err)
			return
		}
	}
	// All files complete and verified.
	_ = m.setState(id, "complete", "")
	m.log.Info("download complete", "id", id)
}

func (m *Manager) fail(id string, err error) {
	m.log.Error("download failed", "id", id, "err", err)
	_ = m.setState(id, "failed", err.Error())
}

type fileRow struct {
	url, dest, partial, sha string
	size, done              int64
	resumable               int
}

func (m *Manager) fileRow(id, dest string) (*fileRow, error) {
	var r fileRow
	err := m.db.QueryRow(`SELECT url,dest_path,partial_path,total_bytes,done_bytes,sha256,resumable
		FROM download_files WHERE download_id = ? AND dest_path = ?`, id, dest).
		Scan(&r.url, &r.dest, &r.partial, &r.size, &r.done, &r.sha, &r.resumable)
	return &r, err
}

// downloadFile fetches one file with range resumption and verification.
// Large files use parallel Range connections when the server supports them.
func (m *Manager) downloadFile(ctx context.Context, id, dest string) error {
	fr, err := m.fileRow(id, dest)
	if err != nil {
		return err
	}
	// Resume offset = size of existing partial (authoritative over DB).
	var offset int64
	if st, err := os.Stat(fr.partial); err == nil {
		offset = st.Size()
		if fr.size > 0 && offset > fr.size {
			os.Remove(fr.partial)
			offset = 0
		}
	}
	if fr.size > 0 && offset == fr.size {
		return m.finalize(id, fr)
	}

	if m.shouldMultipart(fr, offset) {
		if err := m.downloadMultipart(ctx, id, fr); err != nil {
			if errors.Is(err, errNoRanges) {
				m.log.Info("server rejected multipart ranges; falling back to single stream", "url", fr.url)
				cleanupPartFiles(fr.partial)
				return m.downloadSingle(ctx, id, fr, 0)
			}
			return err
		}
		return nil
	}
	return m.downloadSingle(ctx, id, fr, offset)
}

func (m *Manager) newRequest(ctx context.Context, url string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if m.authHdr != nil {
		if h := m.authHdr(); h != "" {
			req.Header.Set("Authorization", h)
		}
	}
	req.Header.Set("User-Agent", "openinfer-studio/0.1")
	return req, nil
}

func (m *Manager) downloadSingle(ctx context.Context, id string, fr *fileRow, offset int64) error {
	req, err := m.newRequest(ctx, fr.url)
	if err != nil {
		return err
	}
	if offset > 0 && fr.resumable != 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}

	resp, err := m.http.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	appendMode := false
	switch {
	case resp.StatusCode == http.StatusPartialContent && offset > 0:
		appendMode = true
		if fr.resumable == -1 {
			_, _ = m.db.Exec(`UPDATE download_files SET resumable = 1 WHERE download_id = ? AND dest_path = ?`, id, fr.dest)
		}
	case resp.StatusCode == http.StatusOK:
		if offset > 0 {
			m.log.Info("server does not support ranges; restarting file", "url", fr.url)
			_, _ = m.db.Exec(`UPDATE download_files SET resumable = 0, done_bytes = 0 WHERE download_id = ? AND dest_path = ?`, id, fr.dest)
			offset = 0
		}
	case resp.StatusCode == http.StatusRequestedRangeNotSatisfiable:
		os.Remove(fr.partial)
		return m.downloadSingle(ctx, id, fr, 0)
	default:
		return fmt.Errorf("HTTP %d for %s", resp.StatusCode, fr.url)
	}

	if err := os.MkdirAll(filepath.Dir(fr.partial), 0o755); err != nil {
		return err
	}
	flag := os.O_CREATE | os.O_WRONLY
	if appendMode {
		flag |= os.O_APPEND
	} else {
		flag |= os.O_TRUNC
	}
	out, err := os.OpenFile(fr.partial, flag, 0o644)
	if err != nil {
		return err
	}
	if appendMode {
		if st, err := out.Stat(); err == nil && st.Size() != offset {
			out.Close()
			return fmt.Errorf("partial changed underfoot: size %d, expected offset %d", st.Size(), offset)
		}
	}

	total := fr.size
	if total == 0 {
		if cl := resp.Header.Get("Content-Length"); cl != "" {
			if n, err := strconv.ParseInt(cl, 10, 64); err == nil {
				total = n + offset
				_, _ = m.db.Exec(`UPDATE download_files SET total_bytes = ? WHERE download_id = ? AND dest_path = ?`,
					total, id, fr.dest)
			}
		}
	}

	buf := make([]byte, 1<<20)
	written := offset
	lastEmit := time.Now()
	lastBytes := offset
	var speed float64
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := out.Write(buf[:n]); werr != nil {
				out.Close()
				return werr
			}
			written += int64(n)
			if time.Since(lastEmit) > 500*time.Millisecond {
				elapsed := time.Since(lastEmit).Seconds()
				inst := float64(written-lastBytes) / elapsed
				speed = 0.7*speed + 0.3*inst
				lastEmit = time.Now()
				lastBytes = written
				m.persistProgress(id, fr.dest, written)
				m.emitProgress(id, written, total, speed)
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			out.Close()
			m.persistProgress(id, fr.dest, written)
			return fmt.Errorf("interrupted: %w", rerr)
		}
	}
	out.Close()
	m.persistProgress(id, fr.dest, written)
	if st, err := os.Stat(fr.partial); err == nil && st.Size() != written {
		return fmt.Errorf("partial size mismatch: counter %d, on disk %d", written, st.Size())
	}
	if fr.size > 0 && written != fr.size {
		return fmt.Errorf("size mismatch: expected %d bytes, got %d", fr.size, written)
	}
	return m.finalize(id, fr)
}

// finalize verifies the checksum (when known) and atomically moves the
// partial file to its destination.
func (m *Manager) finalize(id string, fr *fileRow) error {
	if fr.sha != "" {
		sum, err := sha256File(fr.partial)
		if err != nil {
			return err
		}
		if !strings.EqualFold(sum, fr.sha) {
			os.Remove(fr.partial)
			_, _ = m.db.Exec(`UPDATE download_files SET done_bytes = 0 WHERE download_id = ? AND dest_path = ?`, id, fr.dest)
			return fmt.Errorf("sha256 mismatch for %s (expected %s, got %s)", filepath.Base(fr.dest), fr.sha, sum)
		}
	}
	if err := os.MkdirAll(filepath.Dir(fr.dest), 0o755); err != nil {
		return err
	}
	if err := os.Rename(fr.partial, fr.dest); err != nil {
		// Cross-device fallback: copy then remove.
		if err := copyFile(fr.partial, fr.dest); err != nil {
			return err
		}
		os.Remove(fr.partial)
	}
	_, err := m.db.Exec(`UPDATE download_files SET state = 'complete' WHERE download_id = ? AND dest_path = ?`, id, fr.dest)
	return err
}

func (m *Manager) persistProgress(id, dest string, done int64) {
	tx, err := m.db.Begin()
	if err != nil {
		return
	}
	defer tx.Rollback()
	_, _ = tx.Exec(`UPDATE download_files SET done_bytes = ? WHERE download_id = ? AND dest_path = ?`, done, id, dest)
	_, _ = tx.Exec(`UPDATE downloads SET done_bytes =
		(SELECT COALESCE(SUM(done_bytes),0) FROM download_files WHERE download_id = ?), updated_at = ? WHERE id = ?`,
		id, now(), id)
	_ = tx.Commit()
}

func (m *Manager) emitProgress(id string, done, total int64, speed float64) {
	if m.events == nil {
		return
	}
	var eta int64
	if speed > 0 && total > done {
		eta = int64(float64(total-done) / speed)
	}
	m.events.Publish("download.progress", Progress{
		ID: id, State: "active", DoneBytes: done, TotalBytes: total,
		SpeedBPS: speed, ETAsec: eta,
	})
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

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp := dst + ".copying"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}
