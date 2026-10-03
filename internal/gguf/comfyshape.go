package gguf

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// ComfyUI-GGUF's converter flattens or re-chunks some tensors so they can be
// quantized (k-quants need rows of 256) and keeps the original PyTorch shape
// in a "comfy.gguf.orig_shape.<tensor>" key. ComfyUI undoes it when loading.
// llama.cpp and stable-diffusion.cpp do not: they take the stored dims at
// face value, and stable-diffusion.cpp even derives a model's hidden size
// from them, so such a file is rejected with "wrong shape" for hundreds of
// tensors. RestoreComfyShapes writes a copy in which every tensor has the
// shape those readers expect.

const comfyOrigShapePrefix = "comfy.gguf.orig_shape."

// ComfyShapeStatus describes what restoring a GGUF's ComfyUI-GGUF tensor
// shapes involves.
type ComfyShapeStatus struct {
	Reshaped    int      // tensors stored with other dims than their orig_shape entry
	Widened     int      // of those, tensors re-encoded as F16 because their quantized rows cannot hold the original width
	Unsupported []string // tensors that cannot be restored, with the reason
	OutputBytes int64    // size of the restored copy
}

// Needed reports whether the file has any tensor to restore.
func (s ComfyShapeStatus) Needed() bool { return s.Reshaped > 0 }

type comfyKV struct {
	key        string
	start, end int64 // raw byte range in the source file
}

type comfyTensor struct {
	name      string
	dims      []uint64
	typ       uint32
	offset    uint64 // relative to the data section
	elements  uint64
	size      uint64 // payload bytes (excluding alignment padding)
	sizeKnown bool

	// plan
	newDims []uint64
	newType uint32
	widen   bool   // re-encode as F16
	newSize uint64 // payload bytes in the restored file
	newOff  uint64
}

type comfyLayout struct {
	version   uint32
	alignment uint64
	kvs       []comfyKV
	orig      map[string][]uint64 // tensor name -> PyTorch shape
	badOrig   map[string]bool     // orig_shape key present but unreadable
	tensors   []comfyTensor
	dataStart int64
	fileSize  int64
}

func readComfyLayout(f *os.File) (*comfyLayout, error) {
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	l := &comfyLayout{
		fileSize:  st.Size(),
		alignment: 32,
		orig:      map[string][]uint64{},
		badOrig:   map[string]bool{},
	}
	r := &reader{r: f}
	if magic := r.u32(); magic != magicGGUF {
		return nil, ErrBadMagic
	}
	l.version = r.u32()
	if l.version < minVersion || l.version > maxVersion {
		return nil, fmt.Errorf("%w: %d", ErrBadVersion, l.version)
	}
	tensorCount := r.u64()
	kvCount := r.u64()
	if r.err != nil {
		return nil, r.err
	}
	if tensorCount == 0 || tensorCount > maxTensorCount || kvCount > maxKVCount {
		return nil, fmt.Errorf("%w: tensors=%d kv=%d", ErrBoundsUnsafe, tensorCount, kvCount)
	}
	for i := uint64(0); i < kvCount; i++ {
		start := r.off
		key := r.str()
		value := r.value(l.fileSize)
		if r.err != nil {
			return nil, r.err
		}
		l.kvs = append(l.kvs, comfyKV{key: key, start: start, end: r.off})
		switch {
		case key == "general.alignment":
			if n, ok := toUint64(value); ok && n > 0 && n < 1<<20 {
				l.alignment = n
			}
		case strings.HasPrefix(key, comfyOrigShapePrefix):
			name := strings.TrimPrefix(key, comfyOrigShapePrefix)
			dims, ok := origShapeDims(value)
			if !ok {
				l.badOrig[name] = true
				continue
			}
			l.orig[name] = dims
		}
	}
	for i := uint64(0); i < tensorCount; i++ {
		t := comfyTensor{name: r.str()}
		nDims := r.u32()
		if r.err != nil {
			return nil, r.err
		}
		if nDims == 0 || nDims > 4 {
			return nil, fmt.Errorf("tensor %q has implausible %d dimensions", t.name, nDims)
		}
		t.elements = 1
		for d := uint32(0); d < nDims; d++ {
			dim := r.u64()
			if dim == 0 || dim > 1<<40 || t.elements > math.MaxUint64/dim {
				return nil, fmt.Errorf("tensor %q dimensions overflow", t.name)
			}
			t.dims = append(t.dims, dim)
			t.elements *= dim
		}
		t.typ = r.u32()
		t.offset = r.u64()
		if r.err != nil {
			return nil, r.err
		}
		if tt, ok := tensorTypes[t.typ]; ok && tt.blockSize > 0 {
			blocks := (t.elements + tt.blockSize - 1) / tt.blockSize
			if blocks > math.MaxUint64/tt.typeSize {
				return nil, fmt.Errorf("tensor %q size overflows", t.name)
			}
			t.size, t.sizeKnown = blocks*tt.typeSize, true
		}
		l.tensors = append(l.tensors, t)
	}
	l.dataStart = align(r.off, int64(l.alignment))
	if l.dataStart < 0 || l.dataStart > l.fileSize {
		return nil, fmt.Errorf("GGUF header size %d is unsafe", l.dataStart)
	}
	dataBytes := uint64(l.fileSize - l.dataStart)
	for i := range l.tensors {
		t := &l.tensors[i]
		end := dataBytes
		if i+1 < len(l.tensors) {
			end = l.tensors[i+1].offset
		}
		if t.offset > end || end > dataBytes {
			return nil, fmt.Errorf("tensor %q has an out-of-order or out-of-file data offset", t.name)
		}
		if !t.sizeKnown {
			t.size = end - t.offset // unknown type: carry its whole range over
		}
		if t.size > end-t.offset {
			return nil, fmt.Errorf("tensor %q overlaps the next tensor or the end of the file", t.name)
		}
	}
	return l, nil
}

// origShapeDims reads a comfy.gguf.orig_shape array (int32 elements as the
// reader captures them).
func origShapeDims(v any) ([]uint64, bool) {
	arr, ok := v.([]uint32)
	if !ok || len(arr) == 0 {
		return nil, false
	}
	out := make([]uint64, len(arr))
	for i, d := range arr {
		if int32(d) <= 0 {
			return nil, false
		}
		out[i] = uint64(d)
	}
	return out, true
}

// ggmlDimsFromTorch converts a PyTorch shape to ggml's ne order (reversed,
// with anything past the fourth dimension folded into it, which is how
// stable-diffusion.cpp reads 5-D convolution kernels).
func ggmlDimsFromTorch(shape []uint64) []uint64 {
	ne := make([]uint64, len(shape))
	for i, d := range shape {
		ne[len(shape)-1-i] = d
	}
	for len(ne) > 4 {
		ne[3] *= ne[len(ne)-1]
		ne = ne[:len(ne)-1]
	}
	return ne
}

func trimTrailingOnes(d []uint64) []uint64 {
	for len(d) > 1 && d[len(d)-1] == 1 {
		d = d[:len(d)-1]
	}
	return d
}

func sameDims(a, b []uint64) bool {
	a, b = trimTrailingOnes(a), trimTrailingOnes(b)
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

func dimsProduct(d []uint64) uint64 {
	p := uint64(1)
	for _, x := range d {
		p *= x
	}
	return p
}

// plan decides, per tensor, its restored dims and encoding, assigns offsets
// in the restored data section, and returns the status. It does not touch
// the file.
func (l *comfyLayout) plan() (ComfyShapeStatus, error) {
	var st ComfyShapeStatus
	for i := range l.tensors {
		t := &l.tensors[i]
		t.newDims, t.newType, t.newSize, t.widen = t.dims, t.typ, t.size, false
		if l.badOrig[t.name] {
			st.Reshaped++
			st.Unsupported = append(st.Unsupported, fmt.Sprintf("%s: unreadable %s%s entry", t.name, comfyOrigShapePrefix, t.name))
			continue
		}
		shape, ok := l.orig[t.name]
		if !ok {
			continue
		}
		want := ggmlDimsFromTorch(shape)
		if sameDims(want, t.dims) {
			continue
		}
		st.Reshaped++
		if dimsProduct(want) != t.elements {
			return st, fmt.Errorf("tensor %q: its orig_shape %v holds %d elements but %d are stored", t.name, shape, dimsProduct(want), t.elements)
		}
		tt, known := tensorTypes[t.typ]
		switch {
		case !known || !t.sizeKnown:
			st.Unsupported = append(st.Unsupported, fmt.Sprintf("%s: unknown tensor type %d", t.name, t.typ))
		case tt.blockSize == 1 || want[0]%tt.blockSize == 0:
			t.newDims = want // same bytes, new dims
		case dequantizers[t.typ] != nil:
			t.newDims, t.newType, t.newSize, t.widen = want, 1, t.elements*2, true
			st.Widened++
		default:
			st.Unsupported = append(st.Unsupported, fmt.Sprintf("%s: %s rows of %d cannot hold its original width %d", t.name, strings.ToUpper(ggmlTypeName(t.typ)), tt.blockSize, want[0]))
		}
	}
	var run uint64
	for i := range l.tensors {
		t := &l.tensors[i]
		t.newOff = run
		run += uint64(align(int64(t.newSize), int64(l.alignment)))
	}
	hdr, err := l.header(nil)
	if err != nil {
		return st, err
	}
	st.OutputBytes = align(int64(len(hdr)), int64(l.alignment)) + int64(run)
	return st, nil
}

func (l *comfyLayout) keptKVs() []comfyKV {
	kept := make([]comfyKV, 0, len(l.kvs))
	for _, kv := range l.kvs {
		if !strings.HasPrefix(kv.key, comfyOrigShapePrefix) {
			kept = append(kept, kv)
		}
	}
	return kept
}

// header builds the restored file's header (magic through the tensor table,
// without alignment padding). src supplies the raw bytes of the kept
// metadata; nil builds a placeholder of the right length for sizing.
func (l *comfyLayout) header(src io.ReaderAt) ([]byte, error) {
	kept := l.keptKVs()
	b := binary.LittleEndian.AppendUint32(nil, magicGGUF)
	b = binary.LittleEndian.AppendUint32(b, l.version)
	b = binary.LittleEndian.AppendUint64(b, uint64(len(l.tensors)))
	b = binary.LittleEndian.AppendUint64(b, uint64(len(kept)))
	for _, kv := range kept {
		raw := make([]byte, kv.end-kv.start)
		if src != nil {
			if _, err := src.ReadAt(raw, kv.start); err != nil {
				return nil, err
			}
		}
		b = append(b, raw...)
	}
	for _, t := range l.tensors {
		b = binary.LittleEndian.AppendUint64(b, uint64(len(t.name)))
		b = append(b, t.name...)
		b = binary.LittleEndian.AppendUint32(b, uint32(len(t.newDims)))
		for _, d := range t.newDims {
			b = binary.LittleEndian.AppendUint64(b, d)
		}
		b = binary.LittleEndian.AppendUint32(b, t.newType)
		b = binary.LittleEndian.AppendUint64(b, t.newOff)
	}
	return b, nil
}

// InspectComfyShapes reports whether path is a GGUF made with ComfyUI-GGUF's
// reshaping and what restoring it takes. A file that is not a GGUF, or has no
// such tensors, comes back with Needed() == false.
func InspectComfyShapes(path string) (ComfyShapeStatus, error) {
	f, err := os.Open(path)
	if err != nil {
		return ComfyShapeStatus{}, err
	}
	defer f.Close()
	l, err := readComfyLayout(f)
	if err != nil {
		return ComfyShapeStatus{}, err
	}
	if len(l.orig) == 0 && len(l.badOrig) == 0 {
		return ComfyShapeStatus{}, nil
	}
	return l.plan()
}

const comfyCopyChunk = 64 << 20

// RestoreComfyShapes writes a copy of sourcePath in which every tensor that
// ComfyUI-GGUF stored reshaped has its original shape back, and the
// comfy.gguf.orig_shape.* keys are gone. Tensors whose bytes already are the
// original layout (every F32/F16/BF16 one, and quantized ones whose original
// rows hold whole blocks) are copied untouched with only their dims changed.
// Quantized tensors whose original rows cannot hold whole blocks (a
// 1152-wide row against 256-element k-quant blocks) are decoded and stored
// as F16. The source is never modified.
func RestoreComfyShapes(ctx context.Context, sourcePath, destPath string, progress func(done, total int64)) (status ComfyShapeStatus, err error) {
	sourceAbs, err := filepath.Abs(sourcePath)
	if err != nil {
		return status, err
	}
	destAbs, err := filepath.Abs(destPath)
	if err != nil {
		return status, err
	}
	if sourceAbs == destAbs {
		return status, fmt.Errorf("refusing to restore a GGUF in place")
	}
	source, err := os.Open(sourcePath)
	if err != nil {
		return status, err
	}
	defer source.Close()
	l, err := readComfyLayout(source)
	if err != nil {
		return status, err
	}
	if status, err = l.plan(); err != nil {
		return status, err
	}
	if !status.Needed() {
		return status, fmt.Errorf("%s has no ComfyUI-GGUF reshaped tensors", filepath.Base(sourcePath))
	}
	if len(status.Unsupported) > 0 {
		shown := status.Unsupported
		if len(shown) > 3 {
			shown = append(append([]string{}, shown[:3]...), fmt.Sprintf("and %d more", len(status.Unsupported)-3))
		}
		return status, fmt.Errorf("cannot restore %d tensor shapes: %s", len(status.Unsupported), strings.Join(shown, "; "))
	}
	header, err := l.header(source)
	if err != nil {
		return status, err
	}

	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return status, err
	}
	dest, err := os.OpenFile(destPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return status, err
	}
	keep := false
	defer func() {
		_ = dest.Close()
		if !keep {
			_ = os.Remove(destPath)
		}
	}()

	var written int64
	lastReport := int64(-64 << 20)
	report := func(force bool) {
		if progress != nil && (force || written-lastReport >= 64<<20) {
			lastReport = written
			progress(written, status.OutputBytes)
		}
	}
	pad := func() error {
		n := align(written, int64(l.alignment)) - written
		if n == 0 {
			return nil
		}
		if _, err := dest.Write(make([]byte, n)); err != nil {
			return err
		}
		written += n
		return nil
	}
	if _, err := dest.Write(header); err != nil {
		return status, err
	}
	written = int64(len(header))
	if err := pad(); err != nil {
		return status, err
	}
	report(true)

	for i := range l.tensors {
		t := &l.tensors[i]
		if err := ctx.Err(); err != nil {
			return status, err
		}
		srcPos := l.dataStart + int64(t.offset)
		if t.widen {
			err = widenTensor(ctx, dest, source, srcPos, t, &written, report)
		} else {
			err = copyTensor(ctx, dest, source, srcPos, int64(t.size), &written, report)
		}
		if err != nil {
			return status, fmt.Errorf("tensor %q: %w", t.name, err)
		}
		if err := pad(); err != nil {
			return status, err
		}
	}
	if written != status.OutputBytes {
		return status, fmt.Errorf("restored file is %d bytes, expected %d", written, status.OutputBytes)
	}
	report(true)
	if err := dest.Close(); err != nil {
		return status, err
	}
	issues, _, err := ValidateFile(destPath)
	if err != nil {
		return status, err
	}
	if len(issues) > 0 {
		return status, fmt.Errorf("restored GGUF failed validation: %s", issues[0])
	}
	keep = true
	return status, nil
}

// copyTensor copies n payload bytes from src at pos. Chunked io.CopyN keeps
// the *os.File to *os.File path, which the kernel can copy without bouncing
// the data through userspace.
func copyTensor(ctx context.Context, dst *os.File, src *os.File, pos, n int64, written *int64, report func(bool)) error {
	if _, err := src.Seek(pos, io.SeekStart); err != nil {
		return err
	}
	for n > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		c := min(n, comfyCopyChunk)
		if _, err := io.CopyN(dst, src, c); err != nil {
			return err
		}
		n -= c
		*written += c
		report(false)
	}
	return nil
}

// widenTensor decodes a block-quantized tensor and writes it as F16.
func widenTensor(ctx context.Context, dst *os.File, src *os.File, pos int64, t *comfyTensor, written *int64, report func(bool)) error {
	tt := tensorTypes[t.typ]
	decode := dequantizers[t.typ]
	const blocksPerBatch = 4096
	totalBlocks := t.elements / tt.blockSize
	in := make([]byte, blocksPerBatch*tt.typeSize)
	var out []byte
	for done := uint64(0); done < totalBlocks; {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := min(totalBlocks-done, blocksPerBatch)
		buf := in[:n*tt.typeSize]
		if _, err := src.ReadAt(buf, pos+int64(done*tt.typeSize)); err != nil {
			return err
		}
		var err error
		if out, err = encodeF16(out[:0], decode(buf, int(n))); err != nil {
			return err
		}
		if _, err := dst.Write(out); err != nil {
			return err
		}
		done += n
		*written += int64(len(out))
		report(false)
	}
	return nil
}
