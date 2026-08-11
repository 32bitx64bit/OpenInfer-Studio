package downloads

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

// Multipart thresholds. Below these sizes a single stream is simpler and
// usually just as fast.
const (
	multipartMinSize = 8 << 20 // 8 MiB
	multipartMinPart = 2 << 20 // 2 MiB
)

var errNoRanges = errors.New("server does not support multipart ranges")

func (m *Manager) shouldMultipart(fr *fileRow, existingOffset int64) bool {
	if m.connections() <= 1 {
		return false
	}
	if fr.size < multipartMinSize || fr.resumable == 0 {
		return false
	}
	// A single-stream partial already in progress — finish that way so we
	// do not abandon resumed bytes.
	if existingOffset > 0 {
		return false
	}
	return true
}

func cleanupPartFiles(partialBase string) {
	matches, _ := filepath.Glob(partialBase + ".p*")
	for _, p := range matches {
		os.Remove(p)
	}
	os.Remove(partialBase + ".assembling")
}

type byteRange struct {
	index      int
	start, end int64 // inclusive end
	path       string
}

func splitRanges(size int64, parts int, partialBase string) []byteRange {
	if parts < 1 {
		parts = 1
	}
	maxBySize := int(size / multipartMinPart)
	if maxBySize < 1 {
		maxBySize = 1
	}
	if parts > maxBySize {
		parts = maxBySize
	}
	out := make([]byteRange, 0, parts)
	chunk := size / int64(parts)
	var start int64
	for i := 0; i < parts; i++ {
		end := start + chunk - 1
		if i == parts-1 {
			end = size - 1
		}
		out = append(out, byteRange{
			index: i,
			start: start,
			end:   end,
			path:  fmt.Sprintf("%s.p%d", partialBase, i),
		})
		start = end + 1
	}
	return out
}

func (m *Manager) downloadMultipart(ctx context.Context, id string, fr *fileRow) error {
	parts := splitRanges(fr.size, m.connections(), fr.partial)
	if len(parts) <= 1 {
		return m.downloadSingle(ctx, id, fr, 0)
	}
	if err := os.MkdirAll(filepath.Dir(fr.partial), 0o755); err != nil {
		return err
	}

	var done atomic.Int64
	for _, p := range parts {
		want := p.end - p.start + 1
		if st, err := os.Stat(p.path); err == nil {
			switch {
			case st.Size() == want:
				done.Add(want)
			case st.Size() > want:
				os.Remove(p.path)
			case st.Size() > 0:
				done.Add(st.Size())
			}
		}
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		wg       sync.WaitGroup
		errOnce  sync.Once
		firstErr error
		progress sync.Mutex
		lastEmit = time.Now()
		lastDone = done.Load()
		speed    float64
	)
	fail := func(err error) {
		errOnce.Do(func() {
			firstErr = err
			cancel()
		})
	}
	emit := func() {
		progress.Lock()
		defer progress.Unlock()
		now := time.Now()
		if now.Sub(lastEmit) < 400*time.Millisecond {
			return
		}
		cur := done.Load()
		elapsed := now.Sub(lastEmit).Seconds()
		if elapsed > 0 {
			inst := float64(cur-lastDone) / elapsed
			speed = 0.7*speed + 0.3*inst
		}
		lastEmit = now
		lastDone = cur
		m.persistProgress(id, fr.dest, cur)
		m.emitProgress(id, cur, fr.size, speed)
	}

	for _, p := range parts {
		want := p.end - p.start + 1
		if st, err := os.Stat(p.path); err == nil && st.Size() == want {
			continue
		}
		wg.Add(1)
		go func(p byteRange) {
			defer wg.Done()
			if err := m.downloadPart(ctx, fr, p, &done, emit); err != nil {
				if !errors.Is(err, context.Canceled) {
					fail(err)
				}
			}
		}(p)
	}
	wg.Wait()
	if firstErr != nil {
		m.persistProgress(id, fr.dest, done.Load())
		return firstErr
	}
	if done.Load() != fr.size {
		return fmt.Errorf("multipart incomplete: got %d of %d bytes", done.Load(), fr.size)
	}

	if err := assembleParts(fr.partial, parts); err != nil {
		return err
	}
	_, _ = m.db.Exec(`UPDATE download_files SET resumable = 1 WHERE download_id = ? AND dest_path = ?`, id, fr.dest)
	m.persistProgress(id, fr.dest, fr.size)
	return m.finalize(id, fr)
}

func (m *Manager) downloadPart(ctx context.Context, fr *fileRow, p byteRange, done *atomic.Int64, emit func()) error {
	want := p.end - p.start + 1
	var offset int64
	if st, err := os.Stat(p.path); err == nil {
		offset = st.Size()
		if offset > want {
			os.Remove(p.path)
			offset = 0
		}
		if offset == want {
			return nil
		}
	}

	req, err := m.newRequest(ctx, fr.url)
	if err != nil {
		return err
	}
	start := p.start + offset
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", start, p.end))

	resp, err := m.http.Do(req)
	if err != nil {
		return fmt.Errorf("part %d request: %w", p.index, err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusPartialContent:
		// ok
	case http.StatusOK:
		// Server ignored Range — cannot multipart safely.
		return errNoRanges
	default:
		return fmt.Errorf("part %d: HTTP %d", p.index, resp.StatusCode)
	}

	flag := os.O_CREATE | os.O_WRONLY
	if offset > 0 {
		flag |= os.O_APPEND
	} else {
		flag |= os.O_TRUNC
	}
	out, err := os.OpenFile(p.path, flag, 0o644)
	if err != nil {
		return err
	}
	defer out.Close()

	buf := make([]byte, 1<<20)
	written := offset
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := out.Write(buf[:n]); werr != nil {
				return werr
			}
			done.Add(int64(n))
			written += int64(n)
			emit()
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return fmt.Errorf("part %d interrupted: %w", p.index, rerr)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	if written != want {
		return fmt.Errorf("part %d size mismatch: got %d want %d", p.index, written, want)
	}
	return nil
}

func assembleParts(dest string, parts []byteRange) error {
	tmp := dest + ".assembling"
	out, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	for _, p := range parts {
		in, err := os.Open(p.path)
		if err != nil {
			out.Close()
			os.Remove(tmp)
			return err
		}
		_, copyErr := io.Copy(out, in)
		in.Close()
		if copyErr != nil {
			out.Close()
			os.Remove(tmp)
			return copyErr
		}
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, dest); err != nil {
		os.Remove(tmp)
		return err
	}
	for _, p := range parts {
		os.Remove(p.path)
	}
	return nil
}
