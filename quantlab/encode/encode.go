// Package encode writes stock ggml K/legacy GGUF anchors using GPTQ rounding
// and optional Viterbi block-scale search. IQ types stay on llama-quantize.
// This packer is not the default pipeline path: a 3.5 bpw hybrid is mostly
// IQ, which this package cannot pack.
package encode

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"

	"quantlab/core"
	"quantlab/profile"
	"quantlab/qtype"
	"quantlab/tensorbank"
)

// Supported reports whether d is packed by this engine (not IQ).
func Supported(d core.DType) bool {
	return qtype.PackSupported(d.BaseTensorType())
}

// Options control GPTQ packing of one pure-dtype anchor.
type Options struct {
	DType    core.DType
	Imatrix  map[string]profile.ImatrixStats
	Sketches map[int][][]float32 // keyed by input axis length
	Viterbi  bool
	GPTQ     bool
	Progress func(string)
	Context  context.Context
}

// WriteAnchor reads a float GGUF and writes a --pure-style GGUF of dtype d.
func WriteAnchor(srcPath, dstPath string, opt Options) error {
	if !Supported(opt.DType) {
		return fmt.Errorf("encode: %s is not a packed K/legacy type", opt.DType)
	}
	ctx := opt.Context
	if ctx == nil {
		ctx = context.Background()
	}
	src, err := tensorbank.OpenSource(srcPath)
	if err != nil {
		return err
	}
	defer src.Close()
	file, err := tensorbank.Parse(src)
	if err != nil {
		return err
	}
	d := opt.DType.BaseTensorType()
	payloads := make([][]byte, len(file.Tensors))
	infos := make([]tensorbank.TensorInfo, len(file.Tensors))
	copy(infos, file.Tensors)
	for i, ti := range file.Tensors {
		if err := ctx.Err(); err != nil {
			return err
		}
		if opt.Progress != nil {
			opt.Progress(ti.Name)
		}
		if !quantizable(ti) {
			buf := make([]byte, ti.Length)
			if _, err := src.ReadAt(buf, file.PayloadOffset(ti)); err != nil {
				return err
			}
			payloads[i] = buf
			continue
		}
		vals, err := readFloat(src, file, ti)
		if err != nil {
			return err
		}
		ne0 := int(ti.Shape[0])
		if ne0 <= 0 {
			ne0 = len(vals)
		}
		imp := importance(opt.Imatrix, ti.Name, len(vals), ne0)
		if opt.GPTQ {
			sk := opt.Sketches[ne0]
			if sk == nil {
				sk = profile.MakeSketches(ne0, 32, imp[:min(ne0, len(imp))], 1)
			}
			gptqCompensate(vals, ne0, sk, imp, d)
		}
		packed, _, err := qtype.PackOpts(d, vals, imp, qtype.PackOptions{
			Viterbi: opt.Viterbi,
			RowLen:  ne0,
		})
		if err != nil {
			return fmt.Errorf("encode %s: %w", ti.Name, err)
		}
		id, ok := tensorbank.GGMLTypeID(d)
		if !ok {
			return fmt.Errorf("encode: no ggml id for %s", d)
		}
		infos[i].DType = d
		infos[i].GGMLType = id
		infos[i].Length = uint64(len(packed))
		payloads[i] = packed
	}
	return writeGGUF(dstPath, file, infos, payloads)
}

func quantizable(ti tensorbank.TensorInfo) bool {
	if !ti.DType.IsFloat() {
		return false
	}
	if len(ti.Shape) < 2 || len(ti.Shape) > 3 {
		return false
	}
	return ti.Shape[0]%32 == 0 || ti.Shape[0]%256 == 0
}

func importance(im map[string]profile.ImatrixStats, name string, n, ne0 int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	if im == nil {
		return out
	}
	st, ok := im[name]
	if !ok {
		alt := name
		if len(name) > 7 && name[len(name)-7:] == ".weight" {
			alt = name[:len(name)-7]
		} else {
			alt = name + ".weight"
		}
		st, ok = im[alt]
	}
	if !ok || len(st.Values) == 0 {
		return out
	}
	for i := 0; i < n; i++ {
		out[i] = st.Values[i%len(st.Values)]
		if out[i] <= 0 {
			out[i] = 1e-6
		}
	}
	return out
}

func readFloat(src *tensorbank.Source, file *tensorbank.File, ti tensorbank.TensorInfo) ([]float32, error) {
	buf := make([]byte, ti.Length)
	if _, err := src.ReadAt(buf, file.PayloadOffset(ti)); err != nil {
		return nil, err
	}
	out := make([]float32, ti.Elements)
	es := 4
	switch ti.DType {
	case core.DTypeF16, core.DTypeBF16:
		es = 2
	}
	for i := range out {
		switch ti.DType {
		case core.DTypeF32:
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		case core.DTypeF16:
			out[i] = qtype.F16ToF32(binary.LittleEndian.Uint16(buf[i*2:]))
		case core.DTypeBF16:
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(buf[i*2:])) << 16)
		default:
			return nil, fmt.Errorf("encode: not float %s", ti.DType)
		}
		_ = es
	}
	return out, nil
}

// gptqCompensate applies block-GPTQ error feedback along each row using
// activation sketches. The packer then RTN-quantizes the compensated weights.
func gptqCompensate(w []float32, ne0 int, sketches [][]float32, imp []float32, d core.DType) {
	if ne0 <= 0 || len(w)%ne0 != 0 || !qtype.PackSupported(d) {
		return
	}
	bs := qtype.BlockSize(d)
	if bs <= 0 || ne0%bs != 0 {
		return
	}
	rows := len(w) / ne0
	damp := 1e-4
	for r := 0; r < rows; r++ {
		row := w[r*ne0 : (r+1)*ne0]
		for b := 0; b+bs <= ne0; b += bs {
			blk := row[b : b+bs]
			tmp := append([]float32(nil), blk...)
			_, rec, err := qtype.Pack(d, tmp, impSlice(imp, r*ne0+b, bs))
			if err != nil || rec == nil {
				continue
			}
			if b+bs >= ne0 {
				copy(blk, rec) // last block: keep reconstruction? No, keep compensated original for packer.
				continue
			}
			next := row[b+bs : b+2*bs]
			for i := 0; i < bs; i++ {
				e := float64(blk[i] - rec[i])
				hi := hessDiag(sketches, b+i) + damp
				for j := 0; j < bs && b+bs+j < ne0; j++ {
					hij := hessOff(sketches, b+i, b+bs+j)
					next[j] += float32(e * hij / hi)
				}
			}
		}
	}
}

func hessDiag(sk [][]float32, i int) float64 {
	if len(sk) == 0 {
		return 1
	}
	var s float64
	for _, row := range sk {
		if i < len(row) {
			s += float64(row[i]) * float64(row[i])
		}
	}
	if s == 0 {
		return 1
	}
	return s
}

func hessOff(sk [][]float32, i, j int) float64 {
	if len(sk) == 0 {
		return 0
	}
	var s float64
	for _, row := range sk {
		if i < len(row) && j < len(row) {
			s += float64(row[i]) * float64(row[j])
		}
	}
	return s
}

func impSlice(imp []float32, off, n int) []float32 {
	if imp == nil || off+n > len(imp) {
		return nil
	}
	return imp[off : off+n]
}

func writeGGUF(path string, src *tensorbank.File, infos []tensorbank.TensorInfo, payloads [][]byte) error {
	align := uint64(src.Alignment)
	if align == 0 {
		align = 32
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	defer func() {
		f.Close()
		os.Remove(tmp)
	}()
	var hdr [24]byte
	binary.LittleEndian.PutUint32(hdr[0:4], 0x46554747)
	binary.LittleEndian.PutUint32(hdr[4:8], src.Header.Version)
	if src.Header.Version == 0 {
		binary.LittleEndian.PutUint32(hdr[4:8], 3)
	}
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(len(infos)))
	binary.LittleEndian.PutUint64(hdr[16:24], src.Header.KVCount)
	if _, err := f.Write(hdr[:]); err != nil {
		return err
	}
	if _, err := f.Write(src.KVBytes); err != nil {
		return err
	}
	infoLen := uint64(0)
	for _, ti := range infos {
		infoLen += 8 + uint64(len(ti.Name)) + 4 + 8*uint64(len(ti.Shape)) + 4 + 8
	}
	dataStart := alignUp(24+uint64(len(src.KVBytes))+infoLen, align)
	rel := uint64(0)
	var b [8]byte
	for i, ti := range infos {
		rel = alignUp(rel, align)
		binary.LittleEndian.PutUint64(b[:], uint64(len(ti.Name)))
		if _, err := f.Write(b[:]); err != nil {
			return err
		}
		if _, err := io.WriteString(f, ti.Name); err != nil {
			return err
		}
		binary.LittleEndian.PutUint32(b[:4], uint32(len(ti.Shape)))
		if _, err := f.Write(b[:4]); err != nil {
			return err
		}
		for _, dim := range ti.Shape {
			binary.LittleEndian.PutUint64(b[:], dim)
			if _, err := f.Write(b[:]); err != nil {
				return err
			}
		}
		binary.LittleEndian.PutUint32(b[:4], ti.GGMLType)
		if _, err := f.Write(b[:4]); err != nil {
			return err
		}
		binary.LittleEndian.PutUint64(b[:], rel)
		if _, err := f.Write(b[:]); err != nil {
			return err
		}
		rel += uint64(len(payloads[i]))
	}
	pos, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if uint64(pos) < dataStart {
		if _, err := f.Write(make([]byte, dataStart-uint64(pos))); err != nil {
			return err
		}
	}
	off := uint64(0)
	for i, p := range payloads {
		off = alignUp(off, align)
		cur, _ := f.Seek(0, io.SeekCurrent)
		want := int64(dataStart + off)
		if cur < want {
			if _, err := f.Write(make([]byte, want-cur)); err != nil {
				return err
			}
		}
		if _, err := f.Write(p); err != nil {
			return err
		}
		off += uint64(len(p))
		_ = i
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func alignUp(n, a uint64) uint64 {
	if a == 0 {
		return n
	}
	return (n + a - 1) / a * a
}
