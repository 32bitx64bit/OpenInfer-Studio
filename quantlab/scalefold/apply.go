package scalefold

import (
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"quantlab/core"
	"quantlab/qtype"
	"quantlab/tensorbank"
)

// maxScaleBufferBytes bounds one payload transform read (16 MiB).
const maxScaleBufferBytes = 16 << 20

// Apply rewrites src to outPath with the fold transform: whitelisted
// consumers are scaled (W · diag(s)), their norm gains inversely scaled
// (g / s). Alpha-0 and skipped clusters are left alone. The output file
// reproduces the exact source layout; only float payloads change.
func Apply(src *tensorbank.Source, clusters []Cluster, outPath string) error {
	x := map[string][]float32{}
	for _, cl := range clusters {
		if cl.Skipped != "" || cl.Alpha == 0 || len(cl.Scales) == 0 {
			continue
		}
		for _, name := range cl.Consumers {
			x[name] = cl.Scales
		}
		x[cl.Norm] = inverseScales(cl.Scales)
	}
	return writeTransformed(src, outPath, x, false)
}

// ApplyImatrix rewrites an imatrix GGUF: consumers' in_sum2 rows are
// divided by s², matching the activation rescale after folding.
func ApplyImatrix(src *tensorbank.Source, clusters []Cluster, outPath string) error {
	x := map[string][]float32{}
	for _, cl := range clusters {
		if cl.Skipped != "" || cl.Alpha == 0 || len(cl.Scales) == 0 {
			continue
		}
		for _, name := range cl.Consumers {
			x[name+".in_sum2"] = inverseSquareScales(cl.Scales)
		}
	}
	if len(x) == 0 {
		return nil
	}
	return writeTransformed(src, outPath, x, true)
}

func inverseScales(s []float32) []float32 {
	out := make([]float32, len(s))
	for i, v := range s {
		if v == 0 {
			out[i] = 0
			continue
		}
		out[i] = 1 / v
	}
	return out
}

func inverseSquareScales(s []float32) []float32 {
	out := make([]float32, len(s))
	for i, v := range s {
		out[i] = 1 / (v * v)
	}
	return out
}

// writeTransformed streams src to outPath with per-name column scalers.
// imatrixTensors switches the row layout: 1-D tensors are scaled per
// their full vector rather than per 2-D row.
func writeTransformed(src *tensorbank.Source, outPath string, transforms map[string][]float32, imatrixTensors bool) error {
	file, err := tensorbank.Parse(src)
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
	// Header + metadata verbatim.
	marker := file.DataOffset
	if err := copyRange(out, src, 0, marker); err != nil {
		return fmt.Errorf("scalefold: metadata copy: %w", err)
	}
	ordered := append([]tensorbank.TensorInfo(nil), file.Tensors...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].RelOffset < ordered[j].RelOffset })
	pos := int64(marker)
	for i, ti := range ordered {
		abs := file.PayloadOffset(ti)
		if abs < pos {
			return fmt.Errorf("scalefold: tensor %d overlaps previous payload", i)
		}
		if err := copyRange(out, src, pos, abs); err != nil {
			return fmt.Errorf("scalefold: gap copy: %w", err)
		}
		rl := transforms[ti.Name]
		if rl == nil {
			if err := copyRange(out, src, abs, abs+int64(ti.Length)); err != nil {
				return fmt.Errorf("scalefold: payload copy %s: %w", ti.Name, err)
			}
		} else if err := scalePayload(out, src, abs, ti, rl, imatrixTensors); err != nil {
			return fmt.Errorf("scalefold: scale %s: %w", ti.Name, err)
		}
		pos = abs + int64(ti.Length)
	}
	if err := copyRange(out, src, pos, src.Size()); err != nil {
		return fmt.Errorf("scalefold: tail copy: %w", err)
	}
	if err := out.Sync(); err != nil {
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, outPath)
}

// scalePayload writes one tensor payload column-scaled. Row layout: for
// 2-D tensors the row length is Shape[0] elements; anything else (1-D
// norms, imatrix tensors) is scaled by vector position.
func scalePayload(w io.Writer, src *tensorbank.Source, off int64, ti tensorbank.TensorInfo,
	scales []float32, imatrixTensors bool) error {
	var ne0 uint64
	if imatrixTensors {
		ne0 = ti.Elements
	} else if len(ti.Shape) > 0 {
		ne0 = ti.Shape[0]
	}
	rows := ti.Elements / ne0
	esize := uint64(4)
	if ti.DType == core.DTypeF16 || ti.DType == core.DTypeBF16 {
		esize = 2
	}
	if ti.DType != core.DTypeF32 && ti.DType != core.DTypeF16 && ti.DType != core.DTypeBF16 {
		return fmt.Errorf("scalefold: cannot scale %s payload", ti.DType)
	}
	rowBytes := ne0 * esize
	rowsPer := maxScaleBufferBytes / rowBytes
	if rowsPer < 1 {
		rowsPer = 1
	}
	buf := make([]byte, rowBytes*rowsPer)
	for r := uint64(0); r < rows; r += rowsPer {
		n := rowsPer
		if rem := rows - r; rem < n {
			n = rem
		}
		read := rowBytes * n
		if _, err := src.ReadAt(buf[:read], off+int64(r*rowBytes)); err != nil {
			return err
		}
		for i := uint64(0); i < n*ne0; i++ {
			c := i % ne0
			if s := scales[c]; s != 1 {
				v := readScalar(buf[i*esize:], ti.DType)
				writeScalar(buf[i*esize:], ti.DType, v*float64(s))
			}
		}
		if _, err := w.Write(buf[:read]); err != nil {
			return err
		}
	}
	return nil
}

func readScalar(p []byte, d core.DType) float64 {
	switch d {
	case core.DTypeF32:
		return float64(math.Float32frombits(le32(p)))
	case core.DTypeF16:
		return float64(f16ToF32(le16(p)))
	case core.DTypeBF16:
		return float64(math.Float32frombits(uint32(le16(p)) << 16))
	}
	return 0
}

func writeScalar(p []byte, d core.DType, v float64) {
	switch d {
	case core.DTypeF32:
		put32(p, math.Float32bits(float32(v)))
	case core.DTypeF16:
		put16(p, qtype.F16Bits(float32(v)))
	case core.DTypeBF16:
		put16(p, uint16(math.Float32bits(float32(v))>>16))
	}
}

func le16(p []byte) uint16 { return uint16(p[0]) | uint16(p[1])<<8 }
func le32(p []byte) uint32 {
	return uint32(p[0]) | uint32(p[1])<<8 | uint32(p[2])<<16 | uint32(p[3])<<24
}
func put16(p []byte, v uint16) {
	p[0] = byte(v)
	p[1] = byte(v >> 8)
}
func put32(p []byte, v uint32) {
	p[0] = byte(v)
	p[1] = byte(v >> 8)
	p[2] = byte(v >> 16)
	p[3] = byte(v >> 24)
}

// copyRange streams src bytes [from, to) to w verbatim.
func copyRange(w io.Writer, src *tensorbank.Source, from, to int64) error {
	if to <= from {
		return nil
	}
	buf := make([]byte, 1<<20)
	for off := from; off < to; {
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
