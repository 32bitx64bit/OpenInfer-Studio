package gguf

import (
	"fmt"
	"os"
	"strings"
)

// Tensor describes one GGUF tensor table entry without loading payload bytes.
type Tensor struct {
	Name     string   `json:"name"`
	TypeID   uint32   `json:"type_id"`
	TypeName string   `json:"type_name,omitempty"`
	NDims    uint32   `json:"ndims,omitempty"`
	Shape    []uint64 `json:"shape,omitempty"` // GGUF dims; Shape[0] is ne[0] / ncols
	Elements uint64   `json:"elements"`
	Bytes    uint64   `json:"bytes,omitempty"` // 0 when the ggml type layout is unknown
}

// NCols is ggml ne[0], the inner dimension llama.cpp checks against block size.
func (t Tensor) NCols() uint64 {
	if len(t.Shape) == 0 {
		return 0
	}
	return t.Shape[0]
}

// ggmlTypeNames maps ggml_type enum values to llama-quantize type tokens.
var ggmlTypeNames = map[uint32]string{
	0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1",
	6: "q5_0", 7: "q5_1", 8: "q8_0", 9: "q8_1",
	10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k",
	14: "q6_k", 15: "q8_k",
	16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s",
	20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs",
	29: "iq1_m", 30: "bf16",
	34: "tq1_0", 35: "tq2_0", 39: "mxfp4",
}

// ListTensors reads the GGUF tensor table (names, types, element counts, sizes).
func ListTensors(path string) ([]Tensor, *Metadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	fileSize := st.Size()

	r := &reader{r: f}
	if magic := r.u32(); magic != magicGGUF {
		return nil, nil, ErrBadMagic
	}
	version := r.u32()
	if r.err != nil {
		return nil, nil, r.err
	}
	if version < minVersion || version > maxVersion {
		return nil, nil, fmt.Errorf("%w: %d", ErrBadVersion, version)
	}
	tensors := r.u64()
	kvCount := r.u64()
	if r.err != nil {
		return nil, nil, r.err
	}
	if tensors > maxTensorCount || kvCount > maxKVCount {
		return nil, nil, fmt.Errorf("%w: tensors=%d kv=%d", ErrBoundsUnsafe, tensors, kvCount)
	}

	md := &Metadata{Version: version, TensorCount: tensors, Raw: map[string]any{}}
	for i := uint64(0); i < kvCount; i++ {
		if err := takeKV(r, md, fileSize); err != nil {
			return nil, nil, err
		}
	}
	md.extract()

	out := make([]Tensor, 0, tensors)
	for i := uint64(0); i < tensors; i++ {
		name := r.str()
		nDims := r.u32()
		if r.err != nil {
			return out, md, r.err
		}
		shape := make([]uint64, nDims)
		var elems uint64 = 1
		for d := uint32(0); d < nDims; d++ {
			dim := r.u64()
			shape[d] = dim
			elems *= dim
		}
		typ := r.u32()
		_ = r.u64() // offset
		if r.err != nil {
			return out, md, r.err
		}
		t := Tensor{Name: name, TypeID: typ, TypeName: ggmlTypeNames[typ], NDims: nDims, Shape: shape, Elements: elems}
		if tt, ok := tensorTypes[typ]; ok && tt.blockSize > 0 {
			blocks := (elems + tt.blockSize - 1) / tt.blockSize
			t.Bytes = blocks * tt.typeSize
		}
		out = append(out, t)
	}
	return out, md, nil
}

// ggmlTypeName returns the canonical lowercase token for a ggml type id
// ("" when unknown).
func ggmlTypeName(id uint32) string {
	return ggmlTypeNames[id]
}

func ggmlTypeID(name string) (uint32, bool) {
	want := strings.ToLower(strings.TrimSpace(name))
	if want == "" {
		return 0, false
	}
	for id, n := range ggmlTypeNames {
		if n == want {
			return id, true
		}
	}
	return 0, false
}

// BytesForType is the on-disk size of elements stored as ggml type typeName.
// Unknown types fall back to 4.5 bpw.
func BytesForType(elements uint64, typeName string) uint64 {
	id, ok := ggmlTypeID(typeName)
	if !ok {
		return uint64(float64(elements) * 4.5 / 8)
	}
	tt, ok := tensorTypes[id]
	if !ok || tt.blockSize == 0 {
		return uint64(float64(elements) * 4.5 / 8)
	}
	blocks := (elements + tt.blockSize - 1) / tt.blockSize
	return blocks * tt.typeSize
}
