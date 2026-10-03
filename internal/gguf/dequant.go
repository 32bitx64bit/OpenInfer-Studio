package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Dequantizers for the block formats RestoreComfyShapes may have to widen to
// F16. They follow ggml's dequantize_row_* reference implementations: each
// takes whole blocks of src and returns the float32 values in storage order.

// dequantFunc decodes n whole blocks.
type dequantFunc func(src []byte, blocks int) []float32

var dequantizers = map[uint32]dequantFunc{
	8:  dequantQ8_0,
	12: dequantQ4K,
	13: dequantQ5K,
	14: dequantQ6K,
}

func f16At(b []byte) float32 {
	return math.Float32frombits(halfToFloat32Bits(binary.LittleEndian.Uint16(b)))
}

// float32ToHalf rounds to nearest even and reports whether the value is
// finite in F16 (overflow comes back as ±Inf with ok=false).
func float32ToHalf(f float32) (h uint16, ok bool) {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int32(bits>>23) & 0xff
	mant := bits & 0x7fffff
	switch {
	case exp == 0xff: // Inf / NaN
		if mant != 0 {
			return sign | 0x7e00, true
		}
		return sign | 0x7c00, true
	case exp > 142: // |f| >= 2^16, beyond F16 range
		return sign | 0x7c00, false
	case exp < 102: // below half the smallest subnormal
		return sign, true
	case exp < 113: // F16 subnormal (exp 102 rounds up to the smallest one)
		shift := uint(126 - exp)
		m := (mant | 0x800000)
		half := uint16(m >> shift)
		rem := m & (1<<shift - 1)
		mid := uint32(1) << (shift - 1)
		if rem > mid || (rem == mid && half&1 == 1) {
			half++
		}
		return sign | half, true
	}
	half := uint16(exp-112)<<10 | uint16(mant>>13)
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && half&1 == 1) {
		half++ // may carry into the exponent, which is the correct rounding
	}
	if half&0x7c00 == 0x7c00 {
		return sign | 0x7c00, false
	}
	return sign | half, true
}

func dequantQ8_0(src []byte, blocks int) []float32 {
	const bs = 34
	out := make([]float32, 0, blocks*32)
	for i := 0; i < blocks; i++ {
		b := src[i*bs : (i+1)*bs]
		d := f16At(b)
		for _, q := range b[2:34] {
			out = append(out, d*float32(int8(q)))
		}
	}
	return out
}

// scaleMinK4 unpacks the 6-bit scale and min of sub-block j from the 12
// packed bytes shared by Q4_K and Q5_K.
func scaleMinK4(j int, q []byte) (sc, m uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return q[j+4]&0xf | (q[j-4]>>6)<<4, q[j+4]>>4 | (q[j]>>6)<<4
}

func dequantQ4K(src []byte, blocks int) []float32 {
	const bs = 144
	out := make([]float32, 0, blocks*256)
	for i := 0; i < blocks; i++ {
		b := src[i*bs : (i+1)*bs]
		d, dmin := f16At(b[0:]), f16At(b[2:])
		scales, qs := b[4:16], b[16:144]
		is := 0
		for j := 0; j < 256; j += 64 {
			sc1, m1 := scaleMinK4(is, scales)
			sc2, m2 := scaleMinK4(is+1, scales)
			d1, mn1 := d*float32(sc1), dmin*float32(m1)
			d2, mn2 := d*float32(sc2), dmin*float32(m2)
			q := qs[j/2 : j/2+32]
			for l := 0; l < 32; l++ {
				out = append(out, d1*float32(q[l]&0xf)-mn1)
			}
			for l := 0; l < 32; l++ {
				out = append(out, d2*float32(q[l]>>4)-mn2)
			}
			is += 2
		}
	}
	return out
}

func dequantQ5K(src []byte, blocks int) []float32 {
	const bs = 176
	out := make([]float32, 0, blocks*256)
	for i := 0; i < blocks; i++ {
		b := src[i*bs : (i+1)*bs]
		d, dmin := f16At(b[0:]), f16At(b[2:])
		scales, qh, ql := b[4:16], b[16:48], b[48:176]
		is := 0
		u1, u2 := uint8(1), uint8(2)
		for j := 0; j < 256; j += 64 {
			sc1, m1 := scaleMinK4(is, scales)
			sc2, m2 := scaleMinK4(is+1, scales)
			d1, mn1 := d*float32(sc1), dmin*float32(m1)
			d2, mn2 := d*float32(sc2), dmin*float32(m2)
			q := ql[j/2 : j/2+32]
			for l := 0; l < 32; l++ {
				v := q[l] & 0xf
				if qh[l]&u1 != 0 {
					v += 16
				}
				out = append(out, d1*float32(v)-mn1)
			}
			for l := 0; l < 32; l++ {
				v := q[l] >> 4
				if qh[l]&u2 != 0 {
					v += 16
				}
				out = append(out, d2*float32(v)-mn2)
			}
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
	return out
}

func dequantQ6K(src []byte, blocks int) []float32 {
	const bs = 210
	out := make([]float32, blocks*256)
	for i := 0; i < blocks; i++ {
		b := src[i*bs : (i+1)*bs]
		ql, qh, sc := b[0:128], b[128:192], b[192:208]
		d := f16At(b[208:])
		y := out[i*256 : (i+1)*256]
		for n := 0; n < 256; n += 128 {
			half := n / 128
			ql0, qh0, sc0 := ql[half*64:], qh[half*32:], sc[half*8:]
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int(ql0[l]&0xf|(qh0[l]>>0&3)<<4) - 32
				q2 := int(ql0[l+32]&0xf|(qh0[l]>>2&3)<<4) - 32
				q3 := int(ql0[l]>>4|(qh0[l]>>4&3)<<4) - 32
				q4 := int(ql0[l+32]>>4|(qh0[l]>>6&3)<<4) - 32
				y[n+l] = d * float32(int8(sc0[is])) * float32(q1)
				y[n+l+32] = d * float32(int8(sc0[is+2])) * float32(q2)
				y[n+l+64] = d * float32(int8(sc0[is+4])) * float32(q3)
				y[n+l+96] = d * float32(int8(sc0[is+6])) * float32(q4)
			}
		}
	}
	return out
}

// encodeF16 appends the F16 little-endian encoding of vals to dst. A value
// outside F16's range is an error: silently clamping a weight would corrupt
// the model.
func encodeF16(dst []byte, vals []float32) ([]byte, error) {
	for _, v := range vals {
		h, ok := float32ToHalf(v)
		if !ok {
			return dst, fmt.Errorf("value %g does not fit F16", v)
		}
		dst = append(dst, byte(h), byte(h>>8))
	}
	return dst, nil
}
