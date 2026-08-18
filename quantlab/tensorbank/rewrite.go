package tensorbank

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
)

// ErrSkipPayload tells RewriteInPlace to leave a tensor's existing bytes
// untouched. Out-of-place Rewrite still requires a full payload write.
var ErrSkipPayload = errors.New("tensorbank: skip payload")

// PayloadFunc writes one tensor's payload (exactly ti.Length bytes) to w.
type PayloadFunc func(w io.Writer, src *Source, abs int64, ti TensorInfo) error

// CopyPayload copies one tensor payload verbatim.
func CopyPayload(w io.Writer, src *Source, abs int64, ti TensorInfo) error {
	return CopyPayloadContext(context.Background(), w, src, abs, ti)
}

// CopyPayloadContext copies one tensor payload verbatim and remains
// interruptible while processing model-sized tensors.
func CopyPayloadContext(ctx context.Context, w io.Writer, src *Source, abs int64, ti TensorInfo) error {
	return copyRangeContext(ctx, w, src, abs, abs+int64(ti.Length))
}

// Rewrite streams src to outPath, preserving header, metadata, and layout.
// fn, when non-nil, writes each tensor payload; a nil fn copies verbatim.
// Only the bytes fn writes may differ from the source; offsets and dtypes
// stay unchanged, so the result remains a loadable GGUF.
func Rewrite(src *Source, outPath string, fn PayloadFunc) error {
	return RewriteContext(context.Background(), src, outPath, fn)
}

// RewriteContext is Rewrite with cancellation checks between metadata,
// tensor, gap, and payload chunks.
func RewriteContext(ctx context.Context, src *Source, outPath string, fn PayloadFunc) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if fn == nil {
		fn = func(w io.Writer, src *Source, abs int64, ti TensorInfo) error {
			return CopyPayloadContext(ctx, w, src, abs, ti)
		}
	}
	file, err := Parse(src)
	if err != nil {
		return err
	}
	tmp := outPath + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	defer func() {
		out.Close()
		os.Remove(tmp)
	}()
	marker := file.DataOffset
	if err := copyRangeContext(ctx, out, src, 0, marker); err != nil {
		return fmt.Errorf("tensorbank: metadata copy: %w", err)
	}
	ordered := append([]TensorInfo(nil), file.Tensors...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].RelOffset < ordered[j].RelOffset })
	pos := int64(marker)
	for i, ti := range ordered {
		if err := ctx.Err(); err != nil {
			return err
		}
		abs := file.PayloadOffset(ti)
		if abs < pos {
			return fmt.Errorf("tensorbank: tensor %d overlaps previous payload", i)
		}
		if err := copyRangeContext(ctx, out, src, pos, abs); err != nil {
			return fmt.Errorf("tensorbank: gap copy: %w", err)
		}
		if err := fn(out, src, abs, ti); err != nil {
			return fmt.Errorf("tensorbank: payload %s: %w", ti.Name, err)
		}
		pos = abs + int64(ti.Length)
	}
	if err := copyRangeContext(ctx, out, src, pos, src.Size()); err != nil {
		return fmt.Errorf("tensorbank: tail copy: %w", err)
	}
	if err := out.Sync(); err != nil {
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, outPath)
}

// RewriteInPlace mutates src's payload bytes at their existing offsets.
// Header, dtype, and layout stay unchanged, so the file remains a loadable
// GGUF of the same size. Unchanged tensors should return ErrSkipPayload so
// they are not rewritten. Peak extra disk is zero; peak RAM is whatever fn
// uses for one tensor.
func RewriteInPlace(ctx context.Context, src *Source, fn PayloadFunc) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if src == nil {
		return fmt.Errorf("tensorbank: nil source")
	}
	if fn == nil {
		return nil
	}
	file, err := Parse(src)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(src.Path(), os.O_RDWR, 0)
	if err != nil {
		return fmt.Errorf("tensorbank: open %q for in-place rewrite: %w", src.Path(), err)
	}
	defer f.Close()
	ordered := append([]TensorInfo(nil), file.Tensors...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].RelOffset < ordered[j].RelOffset })
	for _, ti := range ordered {
		if err := ctx.Err(); err != nil {
			return err
		}
		abs := file.PayloadOffset(ti)
		w := &sectionWriter{f: f, base: abs, lim: int64(ti.Length)}
		err := fn(w, src, abs, ti)
		if errors.Is(err, ErrSkipPayload) {
			continue
		}
		if err != nil {
			return fmt.Errorf("tensorbank: in-place payload %s: %w", ti.Name, err)
		}
		if w.pos != w.lim {
			return fmt.Errorf("tensorbank: in-place payload %s wrote %d of %d bytes", ti.Name, w.pos, ti.Length)
		}
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("tensorbank: in-place sync: %w", err)
	}
	return nil
}

// sectionWriter writes a tensor payload into a fixed file range.
type sectionWriter struct {
	f    *os.File
	base int64
	pos  int64
	lim  int64
}

func (w *sectionWriter) Write(p []byte) (int, error) {
	if w.pos+int64(len(p)) > w.lim {
		return 0, fmt.Errorf("tensorbank: payload write exceeds tensor bounds")
	}
	n, err := w.f.WriteAt(p, w.base+w.pos)
	w.pos += int64(n)
	return n, err
}

func copyRangeContext(ctx context.Context, w io.Writer, src *Source, from, to int64) error {
	if to <= from {
		return nil
	}
	buf := make([]byte, 1<<20)
	for off := from; off < to; {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := int64(len(buf))
		if rem := to - off; rem < n {
			n = rem
		}
		if _, err := src.ReadAt(buf[:n], off); err != nil {
			return err
		}
		if _, err := w.Write(buf[:n]); err != nil {
			return err
		}
		off += n
	}
	return nil
}
