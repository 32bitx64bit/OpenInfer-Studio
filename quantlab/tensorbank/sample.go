package tensorbank

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"quantlab/core"
)

// RowSample selects rows of one float tensor: Keep are row indices in
// [0, Shape[1]) applied identically to every slice along dims >= 2
// (MoE experts). Output shape is [Shape[0], len(Keep), Shape[2:]...].
type RowSample struct {
	Name string
	Keep []uint64
}

// SampleRows writes a GGUF containing the picked tensors with only the
// selected rows of each. Source tensors must be F32/F16/BF16. The output
// preserves the source dtype per tensor, the tensor names, the source
// tensor order (filtered to picks), the alignment, and the KV section
// verbatim. Atomic: tmp + rename; tmp is removed on error or cancel.
func SampleRows(ctx context.Context, srcPath, outPath string, picks []RowSample, progress ProgressFunc) error {
	src, err := OpenSource(srcPath)
	if err != nil {
		return err
	}
	defer src.Close()
	file, err := Parse(src)
	if err != nil {
		return err
	}
	byName := make(map[string]RowSample, len(picks))
	for _, p := range picks {
		byName[p.Name] = p
	}
	// Filter to picks in source order.
	var selected []TensorInfo
	var keeps [][]uint64
	for _, ti := range file.Tensors {
		p, ok := byName[ti.Name]
		if !ok {
			continue
		}
		if !ti.DType.IsFloat() || ti.DType == core.DTypeF64 {
			return fmt.Errorf("tensorbank: SampleRows: %s has non-float dtype %s", ti.Name, ti.DType)
		}
		if len(ti.Shape) < 2 {
			return fmt.Errorf("tensorbank: SampleRows: %s is not at least 2-D", ti.Name)
		}
		for _, k := range p.Keep {
			if k >= ti.Shape[1] {
				return fmt.Errorf("tensorbank: SampleRows: %s row %d out of range [0,%d)", ti.Name, k, ti.Shape[1])
			}
		}
		selected = append(selected, ti)
		keeps = append(keeps, p.Keep)
	}
	if len(selected) == 0 {
		return fmt.Errorf("tensorbank: SampleRows: no matching tensors")
	}

	align := uint64(file.Alignment)
	if align == 0 {
		align = 32
	}

	// Compute output sizes.
	var infoLen uint64
	for _, ti := range selected {
		infoLen += 8 + uint64(len(ti.Name)) + 4 + 8*uint64(len(ti.Shape)) + 4 + 8
	}
	metaEnd := 24 + uint64(len(file.KVBytes)) + infoLen
	dataStart := alignUp(metaEnd, align)

	type outSpan struct {
		relOff uint64
		length uint64
	}
	spans := make([]outSpan, len(selected))
	off := uint64(0)
	var totalPayload uint64
	for i, ti := range selected {
		off = alignUp(off, align)
		keep := uint64(len(keeps[i]))
		ne0 := ti.Shape[0]
		rowBytes := ne0 * elemSize(ti.DType)
		slices := uint64(1)
		for _, d := range ti.Shape[2:] {
			slices *= d
		}
		outLen := rowBytes * keep * slices
		spans[i] = outSpan{relOff: off, length: outLen}
		off += outLen
		totalPayload += outLen
	}
	totalSize := dataStart + off

	tmp := outPath + ".tmp"
	_ = os.Remove(tmp)
	out, err := os.OpenFile(tmp, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("tensorbank: SampleRows: create %q: %w", tmp, err)
	}
	defer func() {
		out.Close()
		os.Remove(tmp)
	}()
	if totalSize <= math.MaxInt64 {
		if err := out.Truncate(int64(totalSize)); err != nil {
			return fmt.Errorf("tensorbank: SampleRows: preallocate: %w", err)
		}
	}

	// Header + KV verbatim + tensor infos with sampled geometry.
	w := &errWriter{w: out}
	var hdr [24]byte
	binary.LittleEndian.PutUint32(hdr[0:4], 0x46554747)
	ver := file.Header.Version
	if ver == 0 {
		ver = 3
	}
	binary.LittleEndian.PutUint32(hdr[4:8], ver)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(len(selected)))
	binary.LittleEndian.PutUint64(hdr[16:24], file.Header.KVCount)
	w.Write(hdr[:])
	w.Write(file.KVBytes)
	for i, ti := range selected {
		keep := uint64(len(keeps[i]))
		ne0 := ti.Shape[0]
		slices := uint64(1)
		for _, d := range ti.Shape[2:] {
			slices *= d
		}
		outInfo := ti
		outInfo.Shape = make([]uint64, len(ti.Shape))
		copy(outInfo.Shape, ti.Shape)
		outInfo.Shape[1] = keep
		outInfo.Elements = ne0 * keep * slices
		outInfo.Length = spans[i].length
		writeTensorInfo(w, outInfo, spans[i].relOff)
	}
	if w.err != nil {
		return fmt.Errorf("tensorbank: SampleRows: write header: %w", w.err)
	}
	// Padding to dataStart.
	if pos := 24 + uint64(len(file.KVBytes)) + infoLen; pos < dataStart {
		if err := writeZeros(out, dataStart-pos); err != nil {
			return err
		}
	} else if _, err := out.Seek(int64(dataStart), io.SeekStart); err != nil {
		return err
	}

	// Stream payloads row-by-row.
	var copied uint64
	const maxBuf = 8 << 20
	for i, ti := range selected {
		if err := ctx.Err(); err != nil {
			return err
		}
		ne0 := ti.Shape[0]
		es := elemSize(ti.DType)
		rowBytes := ne0 * es
		R := ti.Shape[1]
		slices := uint64(1)
		for _, d := range ti.Shape[2:] {
			slices *= d
		}
		srcOff := file.PayloadOffset(ti)
		dstOff := int64(dataStart + spans[i].relOff)

		rowsPer := maxBuf / rowBytes
		if rowsPer < 1 {
			rowsPer = 1
		}
		buf := make([]byte, rowBytes*rowsPer)
		pos := dstOff
		for e := uint64(0); e < slices; e++ {
			sliceBase := srcOff + int64(e*R*rowBytes)
			for kr := uint64(0); kr < uint64(len(keeps[i])); kr += rowsPer {
				n := rowsPer
				if rem := uint64(len(keeps[i])) - kr; rem < n {
					n = rem
				}
				for j := uint64(0); j < n; j++ {
					row := keeps[i][kr+j]
					read := rowBytes
					if _, err := src.ReadAt(buf[j*rowBytes:(j+1)*rowBytes], sliceBase+int64(row*rowBytes)); err != nil {
						return fmt.Errorf("tensorbank: SampleRows: read %s: %w", ti.Name, err)
					}
					_ = read
				}
				if _, err := out.WriteAt(buf[:n*rowBytes], pos); err != nil {
					return fmt.Errorf("tensorbank: SampleRows: write %s: %w", ti.Name, err)
				}
				pos += int64(n * rowBytes)
				copied += n * rowBytes
				if progress != nil {
					progress(copied, totalPayload)
				}
			}
		}
	}
	if err := out.Sync(); err != nil {
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	return os.Rename(tmp, outPath)
}

// elemSize returns the byte width of one element of a float dtype.
func elemSize(d core.DType) uint64 {
	if d == core.DTypeF16 || d == core.DTypeBF16 {
		return 2
	}
	return 4
}
