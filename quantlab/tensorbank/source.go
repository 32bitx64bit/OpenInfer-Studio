package tensorbank

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
)

// Source is an open GGUF file implementing Reader, with lazy SHA256.
type Source struct {
	f      *os.File
	path   string
	size   int64
	sha    string
	hasSHA bool
}

// OpenSource opens path for random-access reading.
func OpenSource(path string) (*Source, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("tensorbank: open %q: %w", path, err)
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("tensorbank: stat %q: %w", path, err)
	}
	if !st.Mode().IsRegular() {
		f.Close()
		return nil, fmt.Errorf("tensorbank: %q is not a regular file", path)
	}
	return &Source{f: f, path: path, size: st.Size()}, nil
}

func (s *Source) ReadAt(p []byte, off int64) (int, error) { return s.f.ReadAt(p, off) }
func (s *Source) Size() int64                             { return s.size }
func (s *Source) Path() string                            { return s.path }
func (s *Source) Close() error                            { return s.f.Close() }

// file returns the underlying *os.File for kernel copy/fadvise. Unexported
// so callers cannot close or seek it out from under the assembler.
func (s *Source) file() *os.File { return s.f }

// SHA256 streams the whole file and returns its hex digest (cached).
func (s *Source) SHA256() (string, error) {
	if s.hasSHA {
		return s.sha, nil
	}
	d, err := HashReader(s)
	if err != nil {
		return "", err
	}
	s.sha, s.hasSHA = d, true
	return d, nil
}

// HashReader streams r through SHA256 in 1 MiB chunks and returns the hex
// digest. Never loads the source wholesale.
func HashReader(r Reader) (string, error) {
	h := sha256.New()
	buf := make([]byte, 1<<20)
	var off int64
	for off < r.Size() {
		n := int64(len(buf))
		if rem := r.Size() - off; rem < n {
			n = rem
		}
		if _, err := r.ReadAt(buf[:n], off); err != nil {
			return "", fmt.Errorf("tensorbank: hash read at %d: %w", off, err)
		}
		h.Write(buf[:n])
		off += n
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// hashRange streams [off, off+length) of r through SHA256.
func hashRange(r Reader, off int64, length uint64) (string, error) {
	h := sha256.New()
	buf := make([]byte, 1<<20)
	var done uint64
	for done < length {
		n := uint64(len(buf))
		if rem := length - done; rem < n {
			n = rem
		}
		if _, err := r.ReadAt(buf[:n], off+int64(done)); err != nil {
			return "", fmt.Errorf("tensorbank: hash read at %d: %w", off+int64(done), err)
		}
		h.Write(buf[:n])
		done += n
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// rangeReader adapts a fixed range of a Reader to io.Reader.
type rangeReader struct {
	r    Reader
	off  int64
	left int64
}

func (rr *rangeReader) Read(p []byte) (int, error) {
	if rr.left == 0 {
		return 0, io.EOF
	}
	if int64(len(p)) > rr.left {
		p = p[:rr.left]
	}
	n, err := rr.r.ReadAt(p, rr.off)
	rr.off += int64(n)
	rr.left -= int64(n)
	if err == io.EOF && n > 0 && rr.left > 0 {
		return n, io.ErrUnexpectedEOF
	}
	return n, err
}
