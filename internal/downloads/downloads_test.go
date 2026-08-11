package downloads

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/migrations"
	_ "modernc.org/sqlite"
)

type nullSink struct{}

func (nullSink) Publish(string, any) {}

func testManager(t *testing.T) (*Manager, *sql.DB, string) {
	t.Helper()
	dir := t.TempDir()
	db, err := database.Open(dir, migrations.FS)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	partial := filepath.Join(dir, "partial")
	os.MkdirAll(partial, 0o755)
	m := NewManager(db.DB, partial, nullSink{}, slog.New(slog.NewTextHandler(io.Discard, nil)),
		func(string) uint64 { return 1 << 40 })
	return m, db.DB, dir
}

// rangeServer serves a body with Range support and controllable failures.
type rangeServer struct {
	body    []byte
	noRange bool
	// dropAfter: if >0, connection drops after that many bytes (first request).
	dropAfter int64
	requests  *int64
}

func (s *rangeServer) handler(w http.ResponseWriter, r *http.Request) {
	atomic.AddInt64(s.requests, 1)
	body := s.body
	offset := int64(0)
	end := int64(len(body) - 1)
	if rh := r.Header.Get("Range"); rh != "" && !s.noRange {
		// Support both bytes=N- and bytes=N-M.
		if _, err := fmt.Sscanf(rh, "bytes=%d-%d", &offset, &end); err != nil {
			end = int64(len(body) - 1)
			fmt.Sscanf(rh, "bytes=%d-", &offset)
		}
		if offset >= int64(len(body)) || end >= int64(len(body)) || end < offset {
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", offset, end, len(body)))
		w.Header().Set("Content-Length", strconv.FormatInt(end-offset+1, 10))
		w.WriteHeader(http.StatusPartialContent)
		w.Write(body[offset : end+1])
		return
	}
	// Full request (or server that ignores Range).
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(http.StatusOK)
	if s.dropAfter > 0 && offset == 0 {
		// Write a prefix then drop.
		w.Write(body[:s.dropAfter])
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		// Hijack to kill the connection mid-body.
		hj, ok := w.(http.Hijacker)
		if ok {
			conn, _, _ := hj.Hijack()
			conn.Close()
			return
		}
		return
	}
	w.Write(body)
}

func TestDownloadSimple(t *testing.T) {
	m, _, dir := testManager(t)
	body := []byte(strings.Repeat("model-weights", 10000))
	sum := sha256.Sum256(body)
	srv := httptest.NewServer(http.HandlerFunc((&rangeServer{body: body, requests: new(int64)}).handler))
	defer srv.Close()

	dest := filepath.Join(dir, "out", "m.gguf")
	id, err := m.Enqueue("model", "test", filepath.Dir(dest),
		[]FileSpec{{URL: srv.URL + "/m.gguf", DestPath: dest, Size: int64(len(body)), SHA256: hex.EncodeToString(sum[:])}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	state, err := m.WaitComplete(context.Background(), id)
	if err != nil || state != "complete" {
		t.Fatalf("state=%s err=%v", state, err)
	}
	got, _ := os.ReadFile(dest)
	if !strings.EqualFold(string(got), string(body)) {
		t.Error("content mismatch")
	}
}

func TestDownloadResume(t *testing.T) {
	m, db, dir := testManager(t)
	// Hold the queue so Enqueue's pump cannot start before the partial exists.
	m.mu.Lock()
	m.limit = 0
	m.mu.Unlock()

	body := []byte(strings.Repeat("0123456789abcdef", 100000)) // 1.6 MB
	rs := &rangeServer{body: body, requests: new(int64)}
	srv := httptest.NewServer(http.HandlerFunc(rs.handler))
	defer srv.Close()

	dest := filepath.Join(dir, "big.gguf")
	id, err := m.Enqueue("model", "resume-test", dir,
		[]FileSpec{{URL: srv.URL + "/big.gguf", DestPath: dest, Size: int64(len(body))}}, nil)
	if err != nil {
		t.Fatal(err)
	}

	var partial string
	if err := db.QueryRow(`SELECT partial_path FROM download_files WHERE download_id = ?`, id).Scan(&partial); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(partial, body[:500000], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE download_files SET done_bytes = 500000 WHERE download_id = ?`, id); err != nil {
		t.Fatal(err)
	}

	m.SetConcurrency(2)
	state, err := m.WaitComplete(context.Background(), id)
	if err != nil || state != "complete" {
		t.Fatalf("state=%s err=%v", state, err)
	}
	got, _ := os.ReadFile(dest)
	if len(got) != len(body) || string(got) != string(body) {
		t.Fatalf("resumed content mismatch: %d bytes (want %d)", len(got), len(body))
	}
	if atomic.LoadInt64(rs.requests) < 1 {
		t.Fatal("expected at least one HTTP request")
	}
}

func TestDownloadInterruptedThenRetried(t *testing.T) {
	m, _, dir := testManager(t)
	body := []byte(strings.Repeat("x", 300000))
	rs := &rangeServer{body: body, dropAfter: 100000, requests: new(int64)}
	srv := httptest.NewServer(http.HandlerFunc(rs.handler))
	defer srv.Close()

	dest := filepath.Join(dir, "interrupted.gguf")
	id, _ := m.Enqueue("model", "interrupt", dir,
		[]FileSpec{{URL: srv.URL, DestPath: dest, Size: int64(len(body))}}, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	state, err := m.WaitComplete(ctx, id)
	if err == nil {
		t.Fatalf("expected failure, got state=%s", state)
	}
	if state != "failed" {
		t.Fatalf("want failed, got %s", state)
	}

	// Retry with a healthy server: partial must be resumed, not restarted.
	rs.dropAfter = 0
	if err := m.Retry(id); err != nil {
		t.Fatal(err)
	}
	state, err = m.WaitComplete(ctx, id)
	if err != nil || state != "complete" {
		t.Fatalf("retry: state=%s err=%v", state, err)
	}
	got, _ := os.ReadFile(dest)
	if string(got) != string(body) {
		t.Fatal("content mismatch after retry")
	}
}

func TestDownloadNoRangeRestart(t *testing.T) {
	m, db, dir := testManager(t)
	m.mu.Lock()
	m.limit = 0
	m.mu.Unlock()

	body := []byte(strings.Repeat("y", 200000))
	rs := &rangeServer{body: body, noRange: true, requests: new(int64)}
	srv := httptest.NewServer(http.HandlerFunc(rs.handler))
	defer srv.Close()

	dest := filepath.Join(dir, "norange.gguf")
	id, _ := m.Enqueue("model", "norange", dir,
		[]FileSpec{{URL: srv.URL, DestPath: dest, Size: int64(len(body))}}, nil)
	var partial string
	db.QueryRow(`SELECT partial_path FROM download_files WHERE download_id = ?`, id).Scan(&partial)
	os.WriteFile(partial, body[:50000], 0o644)

	m.SetConcurrency(2)
	state, err := m.WaitComplete(context.Background(), id)
	if err != nil || state != "complete" {
		t.Fatalf("state=%s err=%v", state, err)
	}
	got, _ := os.ReadFile(dest)
	if string(got) != string(body) {
		t.Fatal("no-range restart produced wrong content")
	}
	var resumable int
	db.QueryRow(`SELECT resumable FROM download_files WHERE download_id = ?`, id).Scan(&resumable)
	if resumable != 0 {
		t.Errorf("resumable flag = %d, want 0", resumable)
	}
}

func TestDownloadChecksumMismatch(t *testing.T) {
	m, _, dir := testManager(t)
	body := []byte("real content")
	srv := httptest.NewServer(http.HandlerFunc((&rangeServer{body: body, requests: new(int64)}).handler))
	defer srv.Close()

	dest := filepath.Join(dir, "bad.gguf")
	id, _ := m.Enqueue("model", "badsum", dir,
		[]FileSpec{{URL: srv.URL, DestPath: dest, Size: int64(len(body)),
			SHA256: strings.Repeat("0", 64)}}, nil)
	state, err := m.WaitComplete(context.Background(), id)
	if err == nil || state != "failed" {
		t.Fatalf("checksum mismatch must fail: state=%s", state)
	}
	if _, statErr := os.Stat(dest); !os.IsNotExist(statErr) {
		t.Error("corrupt file must not be exposed as valid output")
	}
}

func TestDownloadDiskPreflight(t *testing.T) {
	m, _, dir := testManager(t)
	m.diskFree = func(string) uint64 { return 100 } // nearly full disk
	srv := httptest.NewServer(http.HandlerFunc((&rangeServer{body: []byte("x"), requests: new(int64)}).handler))
	defer srv.Close()
	id, _ := m.Enqueue("model", "full", dir,
		[]FileSpec{{URL: srv.URL, DestPath: filepath.Join(dir, "f.gguf"), Size: 1 << 30}}, nil)
	state, err := m.WaitComplete(context.Background(), id)
	if state != "failed" || err == nil || !strings.Contains(err.Error(), "disk") {
		t.Fatalf("disk preflight not enforced: state=%s err=%v", state, err)
	}
}

func TestSplitRanges(t *testing.T) {
	parts := splitRanges(32<<20, 4, "/tmp/x.part")
	if len(parts) != 4 {
		t.Fatalf("parts=%d", len(parts))
	}
	if parts[0].start != 0 || parts[len(parts)-1].end != (32<<20)-1 {
		t.Fatalf("bad coverage: %+v", parts)
	}
	for i := 1; i < len(parts); i++ {
		if parts[i].start != parts[i-1].end+1 {
			t.Fatalf("gap/overlap at %d: %+v", i, parts)
		}
	}
}

func TestDownloadMultipart(t *testing.T) {
	m, _, dir := testManager(t)
	m.SetConnections(4)
	body := []byte(strings.Repeat("abcdefgh", (8<<20)/8)) // exactly 8 MiB
	sum := sha256.Sum256(body)
	rs := &rangeServer{body: body, requests: new(int64)}
	srv := httptest.NewServer(http.HandlerFunc(rs.handler))
	defer srv.Close()

	dest := filepath.Join(dir, "multi.gguf")
	id, err := m.Enqueue("model", "multipart", dir,
		[]FileSpec{{URL: srv.URL + "/multi.gguf", DestPath: dest, Size: int64(len(body)),
			SHA256: hex.EncodeToString(sum[:])}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	state, err := m.WaitComplete(context.Background(), id)
	if err != nil || state != "complete" {
		t.Fatalf("state=%s err=%v", state, err)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(body) || string(got) != string(body) {
		t.Fatalf("multipart content mismatch: got %d want %d", len(got), len(body))
	}
	if n := atomic.LoadInt64(rs.requests); n < 4 {
		t.Fatalf("expected >=4 range requests, got %d", n)
	}
}
