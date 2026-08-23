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

const ssmConvValidationTag = "requires F32 ssm_conv1d weights"

// SSMConv1dRepairStatus describes a legacy GGUF whose recurrent convolution
// weights were stored as F16/BF16 instead of the F32 layout llama.cpp expects.
type SSMConv1dRepairStatus struct {
	Required    bool
	Repairable  bool
	TensorCount int
	AddedBytes  int64
	OutputBytes int64
}

// InspectSSMConv1d reports whether path needs the lossless F16/BF16-to-F32
// convolution repair. Other tensor types are deliberately not decoded here.
func InspectSSMConv1d(path string) (SSMConv1dRepairStatus, error) {
	st, err := os.Stat(path)
	if err != nil {
		return SSMConv1dRepairStatus{}, err
	}
	tensors, _, err := ListTensors(path)
	if err != nil {
		return SSMConv1dRepairStatus{}, err
	}
	status := SSMConv1dRepairStatus{Repairable: true, OutputBytes: st.Size()}
	for _, tensor := range tensors {
		if !strings.HasSuffix(tensor.Name, "ssm_conv1d.weight") || tensor.TypeID == 0 {
			continue
		}
		status.Required = true
		status.TensorCount++
		if tensor.TypeID != 1 && tensor.TypeID != 30 {
			status.Repairable = false
			continue
		}
		if tensor.Elements > math.MaxInt64/2 {
			return SSMConv1dRepairStatus{}, fmt.Errorf("tensor %q is too large to repair", tensor.Name)
		}
		status.AddedBytes += int64(tensor.Elements * 2)
	}
	status.Repairable = status.Required && status.Repairable
	if status.AddedBytes > math.MaxInt64-status.OutputBytes {
		return SSMConv1dRepairStatus{}, fmt.Errorf("repaired GGUF size overflows int64")
	}
	status.OutputBytes += status.AddedBytes
	return status, nil
}

// RepairableSSMConv1dIssues reports whether all validation failures are the
// legacy convolution layout that RepairSSMConv1d can safely rewrite.
func RepairableSSMConv1dIssues(path string, issues []string) bool {
	if len(issues) == 0 {
		return false
	}
	status, err := InspectSSMConv1d(path)
	if err != nil || !status.Repairable {
		return false
	}
	for _, issue := range issues {
		if !strings.Contains(issue, ssmConvValidationTag) {
			return false
		}
	}
	return true
}

type repairTensor struct {
	name                   string
	typ                    uint32
	typePos, offsetPos     int64
	offset, elements, size uint64
	sizeKnown              bool
	repair                 bool
}

type repairLayout struct {
	headerSize int64
	fileSize   int64
	alignment  uint64
	tensors    []repairTensor
}

func readRepairLayout(f *os.File) (repairLayout, error) {
	st, err := f.Stat()
	if err != nil {
		return repairLayout{}, err
	}
	fileSize := st.Size()
	r := &reader{r: f}
	if magic := r.u32(); magic != magicGGUF {
		return repairLayout{}, ErrBadMagic
	}
	version := r.u32()
	if version < minVersion || version > maxVersion {
		return repairLayout{}, fmt.Errorf("%w: %d", ErrBadVersion, version)
	}
	tensorCount := r.u64()
	kvCount := r.u64()
	if r.err != nil {
		return repairLayout{}, r.err
	}
	if tensorCount == 0 || tensorCount > maxTensorCount || kvCount > maxKVCount {
		return repairLayout{}, fmt.Errorf("%w: tensors=%d kv=%d", ErrBoundsUnsafe, tensorCount, kvCount)
	}
	alignment := uint64(32)
	for i := uint64(0); i < kvCount; i++ {
		key := r.str()
		value := r.value(fileSize)
		if r.err != nil {
			return repairLayout{}, r.err
		}
		if key == "general.alignment" {
			if n, ok := toUint64(value); ok && n > 0 && n < 1<<20 {
				alignment = n
			}
		}
	}

	tensors := make([]repairTensor, 0, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name := r.str()
		nDims := r.u32()
		if r.err != nil {
			return repairLayout{}, r.err
		}
		if nDims == 0 || nDims > 4 {
			return repairLayout{}, fmt.Errorf("tensor %q has implausible %d dimensions", name, nDims)
		}
		elements := uint64(1)
		for d := uint32(0); d < nDims; d++ {
			dim := r.u64()
			if dim > 1<<40 || (dim != 0 && elements > math.MaxUint64/dim) {
				return repairLayout{}, fmt.Errorf("tensor %q dimensions overflow", name)
			}
			elements *= dim
		}
		typePos := r.off
		typ := r.u32()
		offsetPos := r.off
		offset := r.u64()
		if r.err != nil {
			return repairLayout{}, r.err
		}
		tt, sizeKnown := tensorTypes[typ]
		var size uint64
		if sizeKnown && tt.blockSize > 0 {
			blocks := (elements + tt.blockSize - 1) / tt.blockSize
			if blocks > math.MaxUint64/tt.typeSize {
				return repairLayout{}, fmt.Errorf("tensor %q size overflows", name)
			}
			size = blocks * tt.typeSize
		}
		repair := strings.HasSuffix(name, "ssm_conv1d.weight") && typ != 0
		if repair && typ != 1 && typ != 30 {
			return repairLayout{}, fmt.Errorf("tensor %q uses unsupported repair source type %s", name, ggmlTypeName(typ))
		}
		tensors = append(tensors, repairTensor{
			name: name, typ: typ, typePos: typePos, offsetPos: offsetPos,
			offset: offset, elements: elements, size: size, sizeKnown: sizeKnown, repair: repair,
		})
	}

	headerSize := align(r.off, int64(alignment))
	if headerSize < 0 || headerSize > fileSize || headerSize > 1<<30 {
		return repairLayout{}, fmt.Errorf("GGUF header size %d is unsafe", headerSize)
	}
	dataBytes := uint64(fileSize - headerSize)
	for i, tensor := range tensors {
		if tensor.offset > dataBytes {
			return repairLayout{}, fmt.Errorf("tensor %q starts past end of file", tensor.name)
		}
		end := dataBytes
		if i+1 < len(tensors) {
			end = tensors[i+1].offset
			if end < tensor.offset {
				return repairLayout{}, fmt.Errorf("tensor %q has decreasing data offset", tensors[i+1].name)
			}
		}
		if tensor.sizeKnown && (tensor.size > math.MaxUint64-tensor.offset || tensor.offset+tensor.size > end) {
			return repairLayout{}, fmt.Errorf("tensor %q overlaps the next tensor or file boundary", tensor.name)
		}
	}
	return repairLayout{headerSize: headerSize, fileSize: fileSize, alignment: alignment, tensors: tensors}, nil
}

func patchHeaderValue(header []byte, offset int64, value uint64, width int) error {
	if offset < 0 || offset+int64(width) > int64(len(header)) {
		return fmt.Errorf("GGUF header patch offset %d is out of range", offset)
	}
	switch width {
	case 4:
		binary.LittleEndian.PutUint32(header[offset:offset+4], uint32(value))
	case 8:
		binary.LittleEndian.PutUint64(header[offset:offset+8], value)
	default:
		return fmt.Errorf("unsupported GGUF header patch width %d", width)
	}
	return nil
}

func halfToFloat32Bits(h uint16) uint32 {
	sign := uint32(h&0x8000) << 16
	exponent := uint32(h>>10) & 0x1f
	mantissa := uint32(h & 0x03ff)
	switch exponent {
	case 0:
		if mantissa == 0 {
			return sign
		}
		e := int32(-14)
		for mantissa&0x0400 == 0 {
			mantissa <<= 1
			e--
		}
		mantissa &= 0x03ff
		return sign | uint32(e+127)<<23 | mantissa<<13
	case 0x1f:
		return sign | 0x7f800000 | mantissa<<13
	default:
		return sign | (exponent+112)<<23 | mantissa<<13
	}
}

func copyRepairBytes(ctx context.Context, dst io.Writer, src io.Reader, count int64, consumed *int64, report func(bool)) error {
	buf := make([]byte, 4<<20)
	for count > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := int64(len(buf))
		if count < n {
			n = count
		}
		if _, err := io.ReadFull(src, buf[:n]); err != nil {
			return err
		}
		if _, err := dst.Write(buf[:n]); err != nil {
			return err
		}
		count -= n
		*consumed += n
		report(false)
	}
	return nil
}

func upcastConv(ctx context.Context, dst io.Writer, src io.Reader, tensor repairTensor, consumed *int64, report func(bool)) error {
	const batchElements = 32 * 1024
	in := make([]byte, batchElements*2)
	out := make([]byte, batchElements*4)
	remaining := tensor.elements
	for remaining > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := uint64(batchElements)
		if remaining < n {
			n = remaining
		}
		input := in[:n*2]
		if _, err := io.ReadFull(src, input); err != nil {
			return err
		}
		output := out[:n*4]
		for i := uint64(0); i < n; i++ {
			value := binary.LittleEndian.Uint16(input[i*2:])
			bits := uint32(value) << 16
			if tensor.typ == 1 {
				bits = halfToFloat32Bits(value)
			}
			binary.LittleEndian.PutUint32(output[i*4:], bits)
		}
		if _, err := dst.Write(output); err != nil {
			return err
		}
		remaining -= n
		*consumed += int64(n * 2)
		report(false)
	}
	return nil
}

// RepairSSMConv1d writes a job-local GGUF copy with legacy F16/BF16
// ssm_conv1d tensors losslessly expanded to F32. All other header fields and
// tensor payload bytes are preserved exactly. The source is never modified.
func RepairSSMConv1d(ctx context.Context, sourcePath, destPath string, progress func(done, total int64)) (status SSMConv1dRepairStatus, err error) {
	status, err = InspectSSMConv1d(sourcePath)
	if err != nil {
		return status, err
	}
	if !status.Repairable {
		return status, fmt.Errorf("source has no repairable F16/BF16 ssm_conv1d tensors")
	}
	sourceAbs, err := filepath.Abs(sourcePath)
	if err != nil {
		return status, err
	}
	destAbs, err := filepath.Abs(destPath)
	if err != nil {
		return status, err
	}
	if sourceAbs == destAbs {
		return status, fmt.Errorf("refusing to repair GGUF in place")
	}

	source, err := os.Open(sourcePath)
	if err != nil {
		return status, err
	}
	defer source.Close()
	layout, err := readRepairLayout(source)
	if err != nil {
		return status, err
	}
	header := make([]byte, layout.headerSize)
	if _, err := source.ReadAt(header, 0); err != nil {
		return status, err
	}

	var added uint64
	repairCount := 0
	for i := range layout.tensors {
		tensor := &layout.tensors[i]
		if tensor.offset > math.MaxUint64-added {
			return status, fmt.Errorf("repaired tensor offset overflows")
		}
		if err := patchHeaderValue(header, tensor.offsetPos, tensor.offset+added, 8); err != nil {
			return status, err
		}
		if !tensor.repair {
			continue
		}
		if tensor.size%layout.alignment != 0 {
			return status, fmt.Errorf("tensor %q cannot be expanded without changing GGUF alignment", tensor.name)
		}
		if err := patchHeaderValue(header, tensor.typePos, 0, 4); err != nil {
			return status, err
		}
		added += tensor.size
		repairCount++
	}
	if repairCount != status.TensorCount || int64(added) != status.AddedBytes {
		return status, fmt.Errorf("repair inspection changed while opening source")
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
	if _, err := dest.Write(header); err != nil {
		return status, err
	}
	if _, err := source.Seek(layout.headerSize, io.SeekStart); err != nil {
		return status, err
	}

	consumed := layout.headerSize
	lastReport := int64(-64 << 20)
	report := func(force bool) {
		if progress != nil && (force || consumed-lastReport >= 64<<20) {
			lastReport = consumed
			progress(consumed, layout.fileSize)
		}
	}
	report(true)
	if first := layout.tensors[0].offset; first > 0 {
		if err := copyRepairBytes(ctx, dest, source, int64(first), &consumed, report); err != nil {
			return status, err
		}
	}
	dataBytes := uint64(layout.fileSize - layout.headerSize)
	for i, tensor := range layout.tensors {
		end := dataBytes
		if i+1 < len(layout.tensors) {
			end = layout.tensors[i+1].offset
		}
		rangeSize := end - tensor.offset
		if !tensor.repair {
			if err := copyRepairBytes(ctx, dest, source, int64(rangeSize), &consumed, report); err != nil {
				return status, err
			}
			continue
		}
		if err := upcastConv(ctx, dest, source, tensor, &consumed, report); err != nil {
			return status, err
		}
		if padding := rangeSize - tensor.size; padding > 0 {
			if err := copyRepairBytes(ctx, dest, source, int64(padding), &consumed, report); err != nil {
				return status, err
			}
		}
	}
	report(true)
	if err := dest.Sync(); err != nil {
		return status, err
	}
	if err := dest.Close(); err != nil {
		return status, err
	}
	issues, _, err := ValidateFile(destPath)
	if err != nil {
		return status, err
	}
	if len(issues) > 0 {
		return status, fmt.Errorf("repaired GGUF failed validation: %s", issues[0])
	}
	keep = true
	return status, nil
}
