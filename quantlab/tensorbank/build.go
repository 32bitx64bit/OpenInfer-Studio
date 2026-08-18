package tensorbank

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash"
	"io"
	"math"
	"os"
	"path/filepath"

	"quantlab/core"
)

// ProgressFunc reports cumulative payload bytes copied vs total.
type ProgressFunc func(copied, total uint64)

// pick is a resolved per-tensor payload source.
type pick struct {
	info TensorInfo // chosen variant (dtype/shape from the providing anchor)
	src  *Source
	file *File
}

// ValidateAnchors checks that parsed files are compatible anchors for the
// same model: identical model identity (architecture and, when present,
// general.name) and compatible shapes for tensors present in both. Dtypes may
// differ. Trimmed variant anchors carry subsets of the primary's tensors, so
// a differing tensor count is not itself an error; every tensor a non-primary
// file does carry must exist in the primary with a matching shape. The Build
// pick logic fails closed if no anchor provides a tensor at the manifest's
// target dtype (missing coverage = programming error).
func ValidateAnchors(files []*File) error {
	if len(files) < 2 {
		return nil
	}
	base := files[0]
	for i := 1; i < len(files); i++ {
		f := files[i]
		if base.Architecture != f.Architecture {
			return fmt.Errorf("tensorbank: anchor %d architecture %q does not match %q", i, f.Architecture, base.Architecture)
		}
		if base.ModelID != "" && f.ModelID != "" && base.ModelID != f.ModelID {
			return fmt.Errorf("tensorbank: anchor %d model %q does not match %q", i, f.ModelID, base.ModelID)
		}
		for _, t := range f.Tensors {
			o, ok := base.FindTensor(t.Name)
			if !ok {
				return fmt.Errorf("tensorbank: anchor %d has tensor %q not in primary", i, t.Name)
			}
			if !sameShape(t.Shape, o.Shape) {
				return fmt.Errorf("tensorbank: anchor %d tensor %q shape %v does not match %v", i, t.Name, o.Shape, t.Shape)
			}
		}
	}
	return nil
}

func sameShape(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Build writes a derived GGUF to outPath by selecting, for every tensor in
// the primary anchor (sources[0]), the payload whose dtype exactly matches
// the manifest option target, taken from whichever anchor provides it.
// When several anchors provide the identical chosen dtype — the normal case
// for non-quantizable float tensors (norms, biases), which every
// llama-quantize anchor keeps in its original dtype — the tie is broken
// deterministically in favor of the lowest anchor index, so the primary
// source wins. Multiple identical-dtype providers are not ambiguous: the
// drift checks below (geometry, byte cost, element count) still fail-closed
// on any genuine conflict.
//
// The output preserves the primary anchor's metadata KV bytes verbatim,
// rebuilds tensor descriptors with freshly computed aligned offsets under the
// primary's general.alignment, streams payloads without loading them
// wholesale, honors context cancellation, reports progress, verifies every
// copied payload byte-for-byte against its source region, writes atomically
// (tmp file + rename), and re-validates the finished file. Any failure
// removes the partial output.
func (a *Assembler) Build(ctx context.Context, sources []*Source, manifest *core.SelectionManifest, outPath string, progress ProgressFunc) error {
	if len(sources) == 0 {
		return fmt.Errorf("tensorbank: no anchor sources")
	}
	if manifest == nil {
		return fmt.Errorf("tensorbank: nil manifest")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	abs, absErr := filepath.Abs(outPath)
	if absErr != nil {
		return fmt.Errorf("tensorbank: resolve %q: %w", outPath, absErr)
	}
	for _, s := range sources {
		sp, err := filepath.Abs(s.Path())
		if err != nil {
			return fmt.Errorf("tensorbank: resolve %q: %w", s.Path(), err)
		}
		if sp == abs {
			return fmt.Errorf("tensorbank: in-place edit rejected: output %q is also an anchor source", outPath)
		}
	}

	files := make([]*File, len(sources))
	for i, s := range sources {
		f, err := Parse(s)
		if err != nil {
			return fmt.Errorf("tensorbank: anchor %q: %w", s.Path(), err)
		}
		files[i] = f
	}
	if err := ValidateAnchors(files); err != nil {
		return err
	}
	primary := files[0]

	if manifest.SourceSHA != "" {
		sha, err := sources[0].SHA256()
		if err != nil {
			return err
		}
		if sha != manifest.SourceSHA {
			return fmt.Errorf("tensorbank: manifest source SHA %s does not match primary anchor %s", manifest.SourceSHA, sha)
		}
	}

	opts := make(map[string]core.TensorOption, len(manifest.Options))
	for _, o := range manifest.Options {
		if _, dup := opts[o.TensorName]; dup {
			return fmt.Errorf("tensorbank: manifest has duplicate tensor %q", o.TensorName)
		}
		opts[o.TensorName] = o
	}
	if len(opts) != len(primary.Tensors) {
		return fmt.Errorf("tensorbank: manifest covers %d tensors, primary anchor has %d", len(opts), len(primary.Tensors))
	}

	picks := make([]pick, len(primary.Tensors))
	var total uint64
	for i, pt := range primary.Tensors {
		o, ok := opts[pt.Name]
		if !ok {
			return fmt.Errorf("tensorbank: manifest missing tensor %q", pt.Name)
		}
		target := o.Target.BaseTensorType()
		var found []pick
		for j, f := range files {
			ti, ok := f.FindTensor(pt.Name)
			if !ok || ti.DType != target {
				continue
			}
			found = append(found, pick{info: ti, src: sources[j], file: f})
		}
		if len(found) == 0 {
			return fmt.Errorf("tensorbank: no anchor provides tensor %q as %s", pt.Name, target)
		}
		// Multiple anchors may provide the identical chosen dtype (all
		// anchors keep non-quantizable tensors in their original dtype).
		// Identical dtype plus the drift checks below make the candidates
		// interchangeable, so select deterministically: lowest anchor index.
		p := found[0]
		// Type/shape drift checks: payload length must match geometry and the
		// manifest's declared byte cost exactly.
		want, ok := target.ExactBytes(p.info.Elements)
		if !ok || want != p.info.Length {
			return fmt.Errorf("tensorbank: tensor %q payload length %d does not match geometry %d", pt.Name, p.info.Length, want)
		}
		if o.Bytes != 0 && o.Bytes != p.info.Length {
			return fmt.Errorf("tensorbank: tensor %q manifest byte cost %d does not match anchor payload %d", pt.Name, o.Bytes, p.info.Length)
		}
		if p.info.Elements != pt.Elements {
			return fmt.Errorf("tensorbank: tensor %q element drift %d vs %d", pt.Name, p.info.Elements, pt.Elements)
		}
		picks[i] = p
		total += p.info.Length
	}
	if manifest.TotalBytes != 0 && manifest.TotalBytes != total {
		return fmt.Errorf("tensorbank: manifest total %d does not match selected payload sum %d", manifest.TotalBytes, total)
	}

	// Recompute aligned relative offsets in primary tensor order.
	al := uint64(primary.Alignment)
	rel := make([]uint64, len(picks))
	var cur uint64
	for i, p := range picks {
		cur = alignUp(cur, al)
		rel[i] = cur
		cur += p.info.Length
	}

	tmp := outPath + ".tmp"
	cleanup := func(err error) error {
		os.Remove(tmp)
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	// Header size + aligned payload region: padTo + last relative end.
	var payloadEnd uint64
	if n := len(rel); n > 0 {
		payloadEnd = rel[n-1] + picks[n-1].info.Length
	}
	posHdr := uint64(24) + uint64(len(primary.KVBytes)) + tensorInfosLen(picks)
	padToHdr := alignUp(posHdr, al)
	finalSize := padToHdr + payloadEnd

	out, err := a.createDest(tmp, finalSize)
	if err != nil {
		return err
	}

	fail := func(err error) error {
		out.Close()
		return cleanup(err)
	}

	w := &errWriter{w: out}
	var hdr [24]byte
	binary.LittleEndian.PutUint32(hdr[0:4], magicGGUF)
	binary.LittleEndian.PutUint32(hdr[4:8], primary.Header.Version)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(len(picks)))
	binary.LittleEndian.PutUint64(hdr[16:24], uint64(len(primary.KVs)))
	w.Write(hdr[:])
	w.Write(primary.KVBytes) // preserves metadata verbatim
	for i, p := range picks {
		writeTensorInfo(w, p.info, rel[i])
	}
	if w.err != nil {
		return fail(fmt.Errorf("tensorbank: write header: %w", w.err))
	}

	// Pad to alignment, then stream payloads.
	pos := uint64(24) + uint64(len(primary.KVBytes)) + tensorInfosLen(picks)
	padTo := alignUp(pos, al)
	if err := writeZeros(out, padTo-pos); err != nil {
		return fail(err)
	}
	pos = padTo

	buf := make([]byte, 1<<20)
	var copied uint64
	srcHash := make([]string, len(picks))
	for i, p := range picks {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		if want := padTo + rel[i]; pos != want {
			if err := writeZeros(out, want-pos); err != nil {
				return fail(err)
			}
			pos = want
		}
		h, err := copyHashedRange(out, p.src, p.file.PayloadOffset(p.info), p.info.Length, buf)
		if err != nil {
			return fail(fmt.Errorf("tensorbank: copy tensor %q: %w", p.info.Name, err))
		}
		srcHash[i] = h
		pos += p.info.Length
		copied += p.info.Length
		if progress != nil {
			progress(copied, total)
		}
	}
	if err := a.syncFile(out); err != nil {
		return fail(fmt.Errorf("tensorbank: sync: %w", err))
	}
	if err := out.Close(); err != nil {
		return cleanup(fmt.Errorf("tensorbank: close: %w", err))
	}

	// Validate the result: re-parse and verify descriptors, metadata and
	// payload byte identity against the bytes actually written.
	if err := a.verifyResult(tmp, primary, picks, rel, srcHash); err != nil {
		return cleanup(err)
	}

	if err := os.Rename(tmp, outPath); err != nil {
		return cleanup(fmt.Errorf("tensorbank: rename to %q: %w", outPath, err))
	}
	a.syncDir(outPath)
	return nil
}

// verifyResult re-parses the written file and checks every tensor's
// descriptor, the verbatim metadata section, and every payload hash against
// the chosen source regions.
func (a *Assembler) verifyResult(path string, primary *File, picks []pick, rel []uint64, srcHash []string) error {
	s, err := OpenSource(path)
	if err != nil {
		return err
	}
	defer s.Close()
	f, err := Parse(s)
	if err != nil {
		return fmt.Errorf("tensorbank: result re-parse: %w", err)
	}
	if len(f.Tensors) != len(picks) {
		return fmt.Errorf("tensorbank: result has %d tensors, want %d", len(f.Tensors), len(picks))
	}
	if !bytes.Equal(f.KVBytes, primary.KVBytes) {
		return fmt.Errorf("tensorbank: result metadata differs from primary anchor")
	}
	for i, p := range picks {
		t, ok := f.FindTensor(p.info.Name)
		if !ok {
			return fmt.Errorf("tensorbank: result missing tensor %q", p.info.Name)
		}
		if t.DType != p.info.DType || !sameShape(t.Shape, p.info.Shape) ||
			t.RelOffset != rel[i] || t.Length != p.info.Length {
			return fmt.Errorf("tensorbank: result tensor %q descriptor mismatch", p.info.Name)
		}
		h, err := hashRange(s, f.PayloadOffset(t), t.Length)
		if err != nil {
			return err
		}
		if h != srcHash[i] {
			return fmt.Errorf("tensorbank: result tensor %q payload differs from source", p.info.Name)
		}
	}
	return nil
}

type errWriter struct {
	w   io.Writer
	err error
}

func (e *errWriter) Write(p []byte) {
	if e.err != nil {
		return
	}
	_, e.err = e.w.Write(p)
}

func writeTensorInfo(w *errWriter, t TensorInfo, relOff uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(len(t.Name)))
	w.Write(b[:])
	w.Write([]byte(t.Name))
	binary.LittleEndian.PutUint32(b[:4], uint32(len(t.Shape)))
	w.Write(b[:4])
	for _, d := range t.Shape {
		binary.LittleEndian.PutUint64(b[:], d)
		w.Write(b[:])
	}
	binary.LittleEndian.PutUint32(b[:4], t.GGMLType)
	w.Write(b[:4])
	binary.LittleEndian.PutUint64(b[:], relOff)
	w.Write(b[:])
}

func tensorInfosLen(picks []pick) uint64 {
	var n uint64
	for _, p := range picks {
		n += 8 + uint64(len(p.info.Name)) + 4 + 8*uint64(len(p.info.Shape)) + 4 + 8
	}
	return n
}

func writeZeros(w io.Writer, n uint64) error {
	var zeros [4096]byte
	for n > 0 {
		k := uint64(len(zeros))
		if n < k {
			k = n
		}
		if _, err := w.Write(zeros[:k]); err != nil {
			return fmt.Errorf("tensorbank: write padding: %w", err)
		}
		n -= k
	}
	return nil
}

func (a *Assembler) createDest(tmp string, size uint64) (*os.File, error) {
	out, err := os.OpenFile(tmp, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, fmt.Errorf("tensorbank: create %q: %w", tmp, err)
	}
	if size > 0 && size <= math.MaxInt64 {
		if err := out.Truncate(int64(size)); err != nil {
			out.Close()
			os.Remove(tmp)
			return nil, fmt.Errorf("tensorbank: preallocate %q: %w", tmp, err)
		}
	}
	return out, nil
}

func (a *Assembler) syncFile(out *os.File) error {
	if a.Scratch {
		return nil
	}
	return out.Sync()
}

func (a *Assembler) syncDir(outPath string) {
	if a.Scratch {
		return
	}
	if d, derr := os.Open(filepath.Dir(outPath)); derr == nil {
		d.Sync()
		d.Close()
	}
}

// hashingWriter hashes every byte successfully accepted by w.
type hashingWriter struct {
	w io.Writer
	h hash.Hash
}

func (hw *hashingWriter) Write(p []byte) (int, error) {
	n, err := hw.w.Write(p)
	if n > 0 {
		hw.h.Write(p[:n])
	}
	return n, err
}

// copyHashedRange copies length bytes from src@srcOff to dst's current offset
// and returns the SHA-256 of the bytes actually written. On Linux it tries
// copy_file_range (then hashes the dest range just written); otherwise it
// CopyBuffer-s through a hashingWriter so the source is not re-read.
func copyHashedRange(dst *os.File, src *Source, srcOff int64, length uint64, buf []byte) (string, error) {
	if length == 0 {
		return hex.EncodeToString(sha256.New().Sum(nil)), nil
	}
	dstOff, err := dst.Seek(0, io.SeekCurrent)
	if err != nil {
		return "", err
	}
	if length > math.MaxInt64 {
		return copyHashedBuffer(dst, src, srcOff, length, buf)
	}
	if srcF := src.file(); srcF != nil {
		adviseSequential(int(srcF.Fd()), srcOff, int64(length))
		if err := copyFileRangeAll(int(srcF.Fd()), int(dst.Fd()), srcOff, dstOff, int64(length)); err == nil {
			if _, err := dst.Seek(dstOff+int64(length), io.SeekStart); err != nil {
				return "", err
			}
			return hashFileRange(dst, dstOff, length, buf)
		}
		if _, err := dst.Seek(dstOff, io.SeekStart); err != nil {
			return "", err
		}
	}
	return copyHashedBuffer(dst, src, srcOff, length, buf)
}

func copyHashedBuffer(dst *os.File, src *Source, srcOff int64, length uint64, buf []byte) (string, error) {
	h := sha256.New()
	hw := &hashingWriter{w: dst, h: h}
	rr := &rangeReader{r: src, off: srcOff, left: int64(length)}
	n, err := io.CopyBuffer(hw, rr, buf)
	if err != nil {
		return "", err
	}
	if uint64(n) != length {
		return "", fmt.Errorf("short copy %d of %d", n, length)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func hashFileRange(f *os.File, off int64, length uint64, buf []byte) (string, error) {
	h := sha256.New()
	var done uint64
	for done < length {
		n := uint64(len(buf))
		if rem := length - done; rem < n {
			n = rem
		}
		if _, err := f.ReadAt(buf[:n], off+int64(done)); err != nil {
			return "", fmt.Errorf("tensorbank: hash dest at %d: %w", off+int64(done), err)
		}
		h.Write(buf[:n])
		done += n
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
