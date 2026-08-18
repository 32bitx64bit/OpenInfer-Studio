package tensorbank

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"

	"quantlab/core"
)

// GGUF metadata value type IDs (v2/v3).
type ValueType uint32

const (
	VTUint8   ValueType = 0
	VTInt8    ValueType = 1
	VTUint16  ValueType = 2
	VTInt16   ValueType = 3
	VTUint32  ValueType = 4
	VTInt32   ValueType = 5
	VTFloat32 ValueType = 6
	VTBool    ValueType = 7
	VTString  ValueType = 8
	VTArray   ValueType = 9
	VTUint64  ValueType = 10
	VTInt64   ValueType = 11
	VTFloat64 ValueType = 12
)

// Parser bounds. GGUF arrays may not nest; depth is capped at one level.
const (
	maxKVCount   = 1 << 20
	maxArrayLen  = 1 << 24
	maxStringLen = 1 << 31
	maxNameLen   = 1 << 16
	maxDims      = 4
)

// Value is one parsed metadata value. Scalar holds a Go scalar for non-array
// types (uint8, int8, uint16, int16, uint32, int32, float32, bool, string,
// uint64, int64, float64). Arr holds elements for VTArray; elements are
// guaranteed non-array.
type Value struct {
	Type   ValueType
	Scalar any
	Arr    []Value
}

// KV is one metadata key/value pair, in file order.
type KV struct {
	Key   string
	Value Value
}

// TensorInfo is one parsed tensor descriptor. RelOffset addresses bytes
// relative to the start of the tensor data region.
type TensorInfo struct {
	Name      string
	DType     core.DType
	GGMLType  uint32
	Shape     []uint64
	RelOffset uint64
	Length    uint64
	Elements  uint64
}

// File is the fully parsed, validated view over one GGUF source.
type File struct {
	Header    Header
	Alignment uint32
	KVs       []KV
	Tensors   []TensorInfo
	// KVBytes is the raw byte range covering the KV section (used to
	// preserve metadata verbatim when writing derived files).
	KVBytes []byte
	// MetaBytes is the raw byte range from the end of the fixed header
	// through the end of the tensor info section.
	MetaBytes []byte
	// DataOffset is the absolute file offset of the tensor data region.
	DataOffset int64
	// ModelID is general.name, if present.
	ModelID string
	// Architecture is general.architecture, if present.
	Architecture string
}

// Meta returns the value for key, or false.
func (f *File) Meta(key string) (Value, bool) {
	for _, kv := range f.KVs {
		if kv.Key == key {
			return kv.Value, true
		}
	}
	return Value{}, false
}

// PayloadOffset returns the absolute file offset of t's payload.
func (f *File) PayloadOffset(t TensorInfo) int64 {
	return f.DataOffset + int64(t.RelOffset)
}

// FindTensor returns the tensor info for name, or false.
func (f *File) FindTensor(name string) (TensorInfo, bool) {
	for _, t := range f.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return TensorInfo{}, false
}

// ggml type IDs (ggml_type enum) for raw per-tensor dtypes supported by the
// bundled llama.cpp. This is deliberately distinct from GGUF metadata enums
// such as general.file_type and from whole-model quantization recipe labels.
var dtypeToGGML = map[core.DType]uint32{
	core.DTypeF32:     0,
	core.DTypeF16:     1,
	core.DTypeQ4_0:    2,
	core.DTypeQ4_1:    3,
	core.DTypeQ5_0:    6,
	core.DTypeQ5_1:    7,
	core.DTypeQ8_0:    8,
	core.DTypeQ8_1:    9,
	core.DTypeQ2_K:    10,
	core.DTypeQ3_K:    11,
	core.DTypeQ4_K_T:  12,
	core.DTypeQ5_K_T:  13,
	core.DTypeQ6_K:    14,
	core.DTypeQ8_K:    15,
	core.DTypeIQ2_XXS: 16,
	core.DTypeIQ2_XS:  17,
	core.DTypeIQ3_XXS: 18,
	core.DTypeIQ1_S:   19,
	core.DTypeIQ4_NL:  20,
	core.DTypeIQ3_S:   21,
	core.DTypeIQ2_S:   22,
	core.DTypeIQ4_XS:  23,
	core.DTypeI8:      24,
	core.DTypeI16:     25,
	core.DTypeI32:     26,
	core.DTypeI64:     27,
	core.DTypeF64:     28,
	core.DTypeIQ1_M:   29,
	core.DTypeBF16:    30,
}

var ggmlToDType = map[uint32]core.DType{
	0:  core.DTypeF32,
	1:  core.DTypeF16,
	2:  core.DTypeQ4_0,
	3:  core.DTypeQ4_1,
	6:  core.DTypeQ5_0,
	7:  core.DTypeQ5_1,
	8:  core.DTypeQ8_0,
	9:  core.DTypeQ8_1,
	10: core.DTypeQ2_K,
	11: core.DTypeQ3_K,
	12: core.DTypeQ4_K_T,
	13: core.DTypeQ5_K_T,
	14: core.DTypeQ6_K,
	15: core.DTypeQ8_K,
	16: core.DTypeIQ2_XXS,
	17: core.DTypeIQ2_XS,
	18: core.DTypeIQ3_XXS,
	19: core.DTypeIQ1_S,
	20: core.DTypeIQ4_NL,
	21: core.DTypeIQ3_S,
	22: core.DTypeIQ2_S,
	23: core.DTypeIQ4_XS,
	24: core.DTypeI8,
	25: core.DTypeI16,
	26: core.DTypeI32,
	27: core.DTypeI64,
	28: core.DTypeF64,
	29: core.DTypeIQ1_M,
	30: core.DTypeBF16,
}

// GGMLTypeID maps a raw core dtype to its GGML type ID. Recipe labels are
// whole-model quantization instructions, not tensor storage types, and are
// therefore rejected.
func GGMLTypeID(d core.DType) (uint32, bool) {
	id, ok := dtypeToGGML[d]
	return id, ok
}

// DTypeFromGGML maps a GGML type ID to the explicit per-tensor core dtype.
// Returns false for unknown IDs.
func DTypeFromGGML(id uint32) (core.DType, bool) {
	d, ok := ggmlToDType[id]
	return d, ok
}

type cursor struct {
	r   Reader
	off int64
	end int64
}

func (c *cursor) remaining() int64 { return c.end - c.off }

func (c *cursor) read(p []byte) error {
	if int64(len(p)) > c.remaining() {
		return fmt.Errorf("tensorbank: truncated metadata at offset %d (need %d, have %d)", c.off, len(p), c.remaining())
	}
	if _, err := c.r.ReadAt(p, c.off); err != nil {
		return fmt.Errorf("tensorbank: read at %d: %w", c.off, err)
	}
	c.off += int64(len(p))
	return nil
}

func (c *cursor) u32() (uint32, error) {
	var b [4]byte
	if err := c.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func (c *cursor) u64() (uint64, error) {
	var b [8]byte
	if err := c.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func (c *cursor) str(max uint64) (string, error) {
	n, err := c.u64()
	if err != nil {
		return "", err
	}
	if n > max || n > uint64(c.remaining()) {
		return "", fmt.Errorf("tensorbank: string length %d out of bounds at offset %d", n, c.off)
	}
	buf := make([]byte, n)
	if err := c.read(buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func (c *cursor) value(depth int) (Value, error) {
	if depth > 1 {
		return Value{}, fmt.Errorf("tensorbank: nested array depth %d not supported", depth)
	}
	t, err := c.u32()
	if err != nil {
		return Value{}, err
	}
	vt := ValueType(t)
	if vt == VTArray {
		et, err := c.u32()
		if err != nil {
			return Value{}, err
		}
		if ValueType(et) == VTArray {
			return Value{}, fmt.Errorf("tensorbank: array of arrays not supported")
		}
		n, err := c.u64()
		if err != nil {
			return Value{}, err
		}
		if n > maxArrayLen {
			return Value{}, fmt.Errorf("tensorbank: array length %d exceeds bound", n)
		}
		v := Value{Type: VTArray, Arr: make([]Value, 0, min(n, 1024))}
		for i := uint64(0); i < n; i++ {
			e, err := c.scalar(ValueType(et))
			if err != nil {
				return Value{}, fmt.Errorf("tensorbank: array element %d: %w", i, err)
			}
			v.Arr = append(v.Arr, e)
		}
		return v, nil
	}
	return c.scalar(vt)
}

func (c *cursor) scalar(vt ValueType) (Value, error) {
	v := Value{Type: vt}
	switch vt {
	case VTUint8, VTInt8, VTBool:
		var b [1]byte
		if err := c.read(b[:]); err != nil {
			return v, err
		}
		switch vt {
		case VTUint8:
			v.Scalar = b[0]
		case VTInt8:
			v.Scalar = int8(b[0])
		default:
			v.Scalar = b[0] != 0
		}
	case VTUint16, VTInt16:
		var b [2]byte
		if err := c.read(b[:]); err != nil {
			return v, err
		}
		if vt == VTUint16 {
			v.Scalar = binary.LittleEndian.Uint16(b[:])
		} else {
			v.Scalar = int16(binary.LittleEndian.Uint16(b[:]))
		}
	case VTUint32, VTInt32, VTFloat32:
		var b [4]byte
		if err := c.read(b[:]); err != nil {
			return v, err
		}
		u := binary.LittleEndian.Uint32(b[:])
		switch vt {
		case VTUint32:
			v.Scalar = u
		case VTInt32:
			v.Scalar = int32(u)
		default:
			v.Scalar = math.Float32frombits(u)
		}
	case VTUint64, VTInt64, VTFloat64:
		var b [8]byte
		if err := c.read(b[:]); err != nil {
			return v, err
		}
		u := binary.LittleEndian.Uint64(b[:])
		switch vt {
		case VTUint64:
			v.Scalar = u
		case VTInt64:
			v.Scalar = int64(u)
		default:
			v.Scalar = math.Float64frombits(u)
		}
	case VTString:
		s, err := c.str(maxStringLen)
		if err != nil {
			return v, err
		}
		v.Scalar = s
	default:
		return v, fmt.Errorf("tensorbank: unknown metadata value type %d", uint32(vt))
	}
	return v, nil
}

func alignUp(n, a uint64) uint64 {
	if a == 0 {
		return n
	}
	return (n + a - 1) / a * a
}

// Parse fully parses and validates a GGUF v2/v3 source: header, bounded KV
// walking, tensor infos with exact payload geometry, alignment, bounds and
// overlap checks. Fails closed on any inconsistency.
func Parse(r Reader) (*File, error) {
	h, err := parseHeader(r)
	if err != nil {
		return nil, err
	}
	f := &File{Header: h, Alignment: 32}
	cur := &cursor{r: r, off: 24, end: r.Size()}

	if h.KVCount > maxKVCount {
		return nil, fmt.Errorf("tensorbank: kv count %d exceeds bound", h.KVCount)
	}
	f.KVs = make([]KV, 0, min(h.KVCount, 1024))
	for i := uint64(0); i < h.KVCount; i++ {
		key, err := cur.str(maxNameLen)
		if err != nil {
			return nil, fmt.Errorf("tensorbank: kv %d key: %w", i, err)
		}
		v, err := cur.value(0)
		if err != nil {
			return nil, fmt.Errorf("tensorbank: kv %q: %w", key, err)
		}
		f.KVs = append(f.KVs, KV{Key: key, Value: v})
	}

	for _, kv := range f.KVs {
		switch kv.Key {
		case "general.split.no", "general.split.count":
			return nil, fmt.Errorf("tensorbank: split GGUF input (%s) is not supported", kv.Key)
		case "general.alignment":
			if u, ok := kv.Value.Scalar.(uint32); ok {
				f.Alignment = u
			}
		case "general.name":
			if s, ok := kv.Value.Scalar.(string); ok {
				f.ModelID = s
			}
		case "general.architecture":
			if s, ok := kv.Value.Scalar.(string); ok {
				f.Architecture = s
			}
		}
	}
	if f.Alignment == 0 || f.Alignment > 1<<20 || f.Alignment&(f.Alignment-1) != 0 {
		return nil, fmt.Errorf("tensorbank: invalid alignment %d", f.Alignment)
	}
	kvEnd := cur.off

	if h.TensorCount > 1<<22 {
		return nil, fmt.Errorf("tensorbank: tensor count %d exceeds bound", h.TensorCount)
	}
	f.Tensors = make([]TensorInfo, 0, min(h.TensorCount, 4096))
	seen := make(map[string]struct{}, h.TensorCount)
	for i := uint64(0); i < h.TensorCount; i++ {
		name, err := cur.str(maxNameLen)
		if err != nil {
			return nil, fmt.Errorf("tensorbank: tensor %d name: %w", i, err)
		}
		// Names from untrusted headers are interpolated into tensor-type
		// override files downstream; reject unsafe names at parse time.
		if err := core.ValidateTensorName(name); err != nil {
			return nil, fmt.Errorf("tensorbank: tensor %d: %w", i, err)
		}
		if _, dup := seen[name]; dup {
			return nil, fmt.Errorf("tensorbank: duplicate tensor %q", name)
		}
		seen[name] = struct{}{}
		nd, err := cur.u32()
		if err != nil {
			return nil, err
		}
		if nd == 0 || nd > maxDims {
			return nil, fmt.Errorf("tensorbank: tensor %q rank %d out of range", name, nd)
		}
		shape := make([]uint64, nd)
		var elems uint64 = 1
		for d := range shape {
			dim, err := cur.u64()
			if err != nil {
				return nil, err
			}
			if dim == 0 {
				return nil, fmt.Errorf("tensorbank: tensor %q zero dimension %d", name, d)
			}
			if elems > math.MaxUint64/dim {
				return nil, fmt.Errorf("tensorbank: tensor %q element count overflow", name)
			}
			elems *= dim
			shape[d] = dim
		}
		gt, err := cur.u32()
		if err != nil {
			return nil, err
		}
		dt, ok := DTypeFromGGML(gt)
		if !ok {
			return nil, fmt.Errorf("tensorbank: tensor %q: unknown GGML type id %d", name, gt)
		}
		off, err := cur.u64()
		if err != nil {
			return nil, err
		}
		length, ok := dt.ExactBytes(elems)
		if !ok || length == 0 {
			return nil, fmt.Errorf("tensorbank: tensor %q: no geometry for %q", name, dt)
		}
		f.Tensors = append(f.Tensors, TensorInfo{
			Name: name, DType: dt, GGMLType: gt, Shape: shape,
			RelOffset: off, Length: length, Elements: elems,
		})
	}
	metaEnd := cur.off

	f.KVBytes = make([]byte, kvEnd-24)
	if _, err := r.ReadAt(f.KVBytes, 24); err != nil {
		return nil, fmt.Errorf("tensorbank: capture kv bytes: %w", err)
	}
	f.MetaBytes = make([]byte, metaEnd-24)
	if _, err := r.ReadAt(f.MetaBytes, 24); err != nil {
		return nil, fmt.Errorf("tensorbank: capture metadata bytes: %w", err)
	}

	dataStart := alignUp(uint64(metaEnd), uint64(f.Alignment))
	if dataStart > uint64(r.Size()) || dataStart > math.MaxInt64 {
		return nil, fmt.Errorf("tensorbank: data region start %d beyond file size %d", dataStart, r.Size())
	}
	f.DataOffset = int64(dataStart)
	dataLen := uint64(r.Size()) - dataStart

	sorted := make([]TensorInfo, len(f.Tensors))
	copy(sorted, f.Tensors)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].RelOffset < sorted[j].RelOffset })
	var prevEnd uint64
	for i, t := range sorted {
		if t.RelOffset > math.MaxInt64 || t.Length > math.MaxInt64 {
			return nil, fmt.Errorf("tensorbank: tensor %q offset/length overflow", t.Name)
		}
		end := t.RelOffset + t.Length
		if end < t.RelOffset || end > dataLen {
			return nil, fmt.Errorf("tensorbank: tensor %q payload [%d,%d) out of bounds (data region %d bytes)",
				t.Name, t.RelOffset, end, dataLen)
		}
		if i > 0 && t.RelOffset < prevEnd {
			return nil, fmt.Errorf("tensorbank: tensor %q payload overlaps previous tensor", t.Name)
		}
		prevEnd = end
	}
	return f, nil
}
