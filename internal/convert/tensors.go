package convert

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

func ggmlTypeOf(stDType string) (int, error) {
	switch strings.ToUpper(stDType) {
	case "F32":
		return GGMLF32, nil
	case "F16":
		return GGMLF16, nil
	case "BF16":
		return GGMLBF16, nil
	default:
		return 0, fmt.Errorf("cannot store safetensors dtype %s in GGUF (need F32/F16/BF16)", stDType)
	}
}

// storeType picks the GGUF tensor type for a converted model.
// F32 sources are stored as F16; F16/BF16 keep their dtype.
func storeType(srcDType string) (int, error) {
	switch strings.ToUpper(srcDType) {
	case "F32":
		return GGMLF16, nil
	case "F16":
		return GGMLF16, nil
	case "BF16":
		return GGMLBF16, nil
	default:
		return ggmlTypeOf(srcDType)
	}
}

func convertPayload(src []byte, srcDType string, dstType int) ([]byte, error) {
	srcT, err := ggmlTypeOf(srcDType)
	if err != nil {
		return nil, err
	}
	if srcT == dstType {
		out := make([]byte, len(src))
		copy(out, src)
		return out, nil
	}
	if srcT == GGMLF32 && dstType == GGMLF16 {
		return f32ToF16(src), nil
	}
	return nil, fmt.Errorf("no conversion from %s to ggml type %d", srcDType, dstType)
}

func f32ToF16(src []byte) []byte {
	n := len(src) / 4
	out := make([]byte, n*2)
	for i := 0; i < n; i++ {
		f := math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
		binary.LittleEndian.PutUint16(out[i*2:], float32ToF16bits(f))
	}
	return out
}

// IEEE 754 binary16, round-to-nearest-even.
func float32ToF16bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int((b>>23)&0xff) - 127 + 15
	frac := b & 0x7fffff
	switch {
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		frac |= 0x800000
		shift := uint(1 - exp)
		frac = (frac + (1 << (shift - 1))) >> shift
		return sign | uint16(frac>>13)
	case exp >= 31:
		if exp == 143 && frac == 0 { // Inf
			return sign | 0x7c00
		}
		return sign | 0x7c00 | 0x0200 // NaN-ish
	default:
		return sign | uint16(exp<<10) | uint16((frac+(1<<12))>>13)
	}
}

func f32Bytes(vals []float32) []byte {
	out := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func onesF32(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

func filledF32(n int, v float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func addOneF32(src []byte) []byte {
	out := make([]byte, len(src))
	n := len(src) / 4
	for i := 0; i < n; i++ {
		f := math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:])) + 1
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(f))
	}
	return out
}

func toF32(src []byte, dtype string) ([]byte, error) {
	switch strings.ToUpper(dtype) {
	case "F32":
		out := make([]byte, len(src))
		copy(out, src)
		return out, nil
	case "F16":
		n := len(src) / 2
		out := make([]byte, n*4)
		for i := 0; i < n; i++ {
			f := f16bitsToFloat32(binary.LittleEndian.Uint16(src[i*2:]))
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(f))
		}
		return out, nil
	case "BF16":
		n := len(src) / 2
		out := make([]byte, n*4)
		for i := 0; i < n; i++ {
			u := binary.LittleEndian.Uint16(src[i*2:])
			binary.LittleEndian.PutUint32(out[i*4:], uint32(u)<<16)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("cannot promote %s to F32", dtype)
	}
}

func f16bitsToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h & 0x3ff)
	var f uint32
	switch exp {
	case 0:
		if frac == 0 {
			f = sign << 31
		} else {
			// subnormal
			e := 1
			for frac&0x400 == 0 {
				frac <<= 1
				e--
			}
			frac &= 0x3ff
			f = (sign << 31) | uint32(127-15+e)<<23 | frac<<13
		}
	case 31:
		f = (sign << 31) | 0x7f800000 | frac<<13
	default:
		f = (sign << 31) | (exp+127-15)<<23 | frac<<13
	}
	return math.Float32frombits(f)
}

func fromF32(src []byte, dstType int) []byte {
	switch dstType {
	case GGMLF32:
		out := make([]byte, len(src))
		copy(out, src)
		return out
	case GGMLF16:
		return f32ToF16(src)
	case GGMLBF16:
		n := len(src) / 4
		out := make([]byte, n*2)
		for i := 0; i < n; i++ {
			u := binary.LittleEndian.Uint32(src[i*4:])
			binary.LittleEndian.PutUint16(out[i*2:], uint16(u>>16))
		}
		return out
	default:
		return src
	}
}

// transpose2D interprets src as [rows, cols] row-major and returns [cols, rows].
func transpose2D(src []byte, rows, cols int, elem int) []byte {
	out := make([]byte, len(src))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			si := (r*cols + c) * elem
			di := (c*rows + r) * elem
			copy(out[di:di+elem], src[si:si+elem])
		}
	}
	return out
}

// unpermuteHF permutes HF Q/K weights from rotate_half layout to ggml interleaved.
// t is HF [out, in] row-major. nHeads is Q or KV heads. dim1 is out features, dim2 is in features.
func unpermuteHF(t []byte, nHeads, dim1, dim2, elem int) ([]byte, error) {
	if nHeads <= 0 || dim1%(nHeads*2) != 0 {
		return nil, fmt.Errorf("unpermute: dim1=%d n_heads=%d not divisible by 2*heads", dim1, nHeads)
	}
	headHalf := dim1 / nHeads / 2
	// view(n_heads, 2, headHalf, dim2).transpose(1, 2) → (n_heads, headHalf, 2, dim2)
	out := make([]byte, len(t))
	for h := 0; h < nHeads; h++ {
		for p := 0; p < 2; p++ {
			for d := 0; d < headHalf; d++ {
				srcRow := ((h*2+p)*headHalf + d)
				dstRow := ((h*headHalf+d)*2 + p)
				srcOff := srcRow * dim2 * elem
				dstOff := dstRow * dim2 * elem
				copy(out[dstOff:dstOff+dim2*elem], t[srcOff:srcOff+dim2*elem])
			}
		}
	}
	return out, nil
}
