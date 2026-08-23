package gguf

import (
	"fmt"
	"os"
	"strings"
)

// tensorType describes a ggml tensor type's block layout: values are stored
// in blocks of blockSize elements occupying typeSize bytes.
type tensorType struct {
	blockSize uint64
	typeSize  uint64
}

// Common ggml tensor types (ggml_type enum → layout). Unknown ids are
// tolerated: strict size checks are skipped for them and only offset
// monotonicity is verified. Sizes follow ggml's block layout exactly;
// uncertain entries are deliberately omitted rather than guessed.
var tensorTypes = map[uint32]tensorType{
	0:  {1, 4},     // F32
	1:  {1, 2},     // F16
	2:  {32, 18},   // Q4_0
	3:  {32, 20},   // Q4_1
	6:  {32, 22},   // Q5_0
	7:  {32, 24},   // Q5_1
	8:  {32, 34},   // Q8_0
	9:  {32, 40},   // Q8_1
	10: {256, 84},  // Q2_K
	11: {256, 110}, // Q3_K
	12: {256, 144}, // Q4_K
	13: {256, 176}, // Q5_K
	14: {256, 210}, // Q6_K
	15: {256, 292}, // Q8_K
	16: {256, 66},  // IQ2_XXS
	17: {256, 74},  // IQ2_XS
	18: {256, 98},  // IQ3_XXS
	19: {256, 50},  // IQ1_S
	20: {32, 18},   // IQ4_NL
	21: {256, 110}, // IQ3_S
	22: {256, 82},  // IQ2_S
	23: {32, 17},   // IQ4_XS
	24: {1, 1},     // I8
	25: {1, 2},     // I16
	26: {1, 4},     // I32
	28: {1, 8},     // F64
	29: {256, 56},  // IQ1_M
	30: {1, 2},     // BF16
	34: {256, 54},  // TQ1_0
	35: {256, 66},  // TQ2_0
	39: {32, 17},   // MXFP4
	// 42: ternary unpacked ("Q2_0 ternary", bitnet-style). Layout verified
	// against llama.cpp b10212's own offset expectation: blocks of 32
	// elements at 9 bytes (2.25 bpw).
	42: {32, 9},
}

// ValidateFile parses the full header — KV metadata plus the tensor table —
// and verifies tensor data layout consistency:
//   - offsets are non-decreasing,
//   - each tensor's computed end (type size) does not exceed the next
//     tensor's offset (when the type is known),
//   - the last tensor fits inside the file.
//
// It returns a list of issues (empty = healthy). This catches corrupt or
// truncated GGUF files — e.g. "tensor X has offset A, expected B" — before
// llama.cpp rejects them at load time.
func ValidateFile(path string) (issues []string, md *Metadata, err error) {
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

	md = &Metadata{Version: version, TensorCount: tensors, Raw: map[string]any{}}
	for i := uint64(0); i < kvCount; i++ {
		key := r.str()
		val := r.value(fileSize)
		if r.err != nil {
			return nil, nil, r.err
		}
		md.Raw[key] = val
	}
	md.extract()

	// Alignment: general.alignment, default 32.
	var alignment uint64 = 32
	if v, ok := md.Raw["general.alignment"]; ok {
		if n, ok := toUint64(v); ok && n > 0 && n < 1<<20 {
			alignment = n
		}
	}

	type tensorInfo struct {
		name   string
		typ    uint32
		offset uint64
		elems  uint64
	}
	infos := make([]tensorInfo, 0, 256)
	for i := uint64(0); i < tensors; i++ {
		name := r.str()
		nDims := r.u32()
		if r.err == nil && (nDims == 0 || nDims > 4) {
			issues = append(issues, fmt.Sprintf("tensor %q: implausible %d dimensions", name, nDims))
			break
		}
		var elems uint64 = 1
		for d := uint32(0); d < nDims; d++ {
			dim := r.u64()
			if r.err == nil && dim > 1<<40 {
				issues = append(issues, fmt.Sprintf("tensor %q: implausible dimension %d", name, dim))
				break
			}
			elems *= dim
		}
		typ := r.u32()
		off := r.u64()
		if r.err != nil {
			return nil, md, r.err
		}
		infos = append(infos, tensorInfo{name, typ, off, elems})
		if len(issues) >= 8 {
			return issues, md, nil
		}
	}
	if len(infos) == 0 {
		return issues, md, nil
	}

	// Data section starts after the header, aligned.
	dataStart := align(r.off, int64(alignment))
	if dataStart > fileSize {
		issues = append(issues, "header extends beyond end of file (truncated)")
		return issues, md, nil
	}

	ssmConvReported := false
	for i, ti := range infos {
		// llama.cpp's SSM conv kernels read ssm_conv1d weights as raw F32
		// (ggml asserts src1->nb[0] == sizeof(float) on CPU and CUDA).
		// A bf16/f16 conv aborts or garbles every linear-attention layer.
		if strings.HasSuffix(ti.name, "ssm_conv1d.weight") && ti.typ != 0 && !ssmConvReported {
			tn := ggmlTypeName(ti.typ)
			if tn == "" {
				tn = fmt.Sprintf("type %d", ti.typ)
			}
			issues = append(issues, fmt.Sprintf(
				"tensor %q: stored as %s, but llama.cpp %s",
				ti.name, tn, ssmConvValidationTag))
			ssmConvReported = true
			if len(issues) >= 8 {
				return issues, md, nil
			}
		}
		if i > 0 && ti.offset < infos[i-1].offset {
			issues = append(issues, fmt.Sprintf(
				"tensor %q: offset %d is before previous tensor end (expected >= %d) — file is corrupt",
				ti.name, dataStart+int64(ti.offset), dataStart+int64(infos[i-1].offset)))
			if len(issues) >= 8 {
				return issues, md, nil
			}
		}
		tt, known := tensorTypes[ti.typ]
		if !known || tt.blockSize == 0 {
			continue // unverifiable type: monotonicity still checked above
		}
		blocks := (ti.elems + tt.blockSize - 1) / tt.blockSize
		size := blocks * tt.typeSize
		end := ti.offset + size
		if i+1 < len(infos) {
			if end > infos[i+1].offset {
				issues = append(issues, fmt.Sprintf(
					"tensor %q: size %d bytes overlaps next tensor (offset %d) — file is corrupt",
					ti.name, size, infos[i+1].offset))
			}
		} else if dataStart+int64(end) > fileSize {
			issues = append(issues, fmt.Sprintf(
				"tensor %q: data ends %d bytes past end of file — file is truncated",
				ti.name, dataStart+int64(end)-fileSize))
		}
		if len(issues) >= 8 {
			return issues, md, nil
		}
	}
	return issues, md, nil
}

func align(v, a int64) int64 {
	if a <= 1 {
		return v
	}
	rem := v % a
	if rem == 0 {
		return v
	}
	return v + a - rem
}
