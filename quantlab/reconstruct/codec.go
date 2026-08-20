package reconstruct

import (
	"context"
	"encoding/binary"
	"math"

	"quantlab/core"
	"quantlab/qtype"
	"quantlab/tensorbank"
)

func isFloatDType(d core.DType) bool {
	switch d {
	case core.DTypeF32, core.DTypeF16, core.DTypeBF16:
		return true
	}
	return false
}

func elemSize(d core.DType) uint64 {
	switch d {
	case core.DTypeF32:
		return 4
	case core.DTypeF16, core.DTypeBF16:
		return 2
	}
	return 0
}

func decodeScalar(p []byte, d core.DType) float32 {
	switch d {
	case core.DTypeF32:
		return math.Float32frombits(binary.LittleEndian.Uint32(p))
	case core.DTypeF16:
		return qtype.F16ToF32(binary.LittleEndian.Uint16(p))
	case core.DTypeBF16:
		return math.Float32frombits(uint32(binary.LittleEndian.Uint16(p)) << 16)
	}
	return 0
}

func encodeScalar(p []byte, d core.DType, v float32) {
	switch d {
	case core.DTypeF32:
		binary.LittleEndian.PutUint32(p, math.Float32bits(v))
	case core.DTypeF16:
		binary.LittleEndian.PutUint16(p, qtype.F16Bits(v))
	case core.DTypeBF16:
		binary.LittleEndian.PutUint16(p, uint16(math.Float32bits(v)>>16))
	}
}

func decodeBuf(dst []float32, buf []byte, d core.DType) {
	es := int(elemSize(d))
	for i := range dst {
		dst[i] = decodeScalar(buf[i*es:], d)
	}
}

func encodeBuf(buf []byte, src []float32, d core.DType) {
	es := int(elemSize(d))
	for i, v := range src {
		encodeScalar(buf[i*es:], d, v)
	}
}

func readFloatTensor(src *tensorbank.Source, file *tensorbank.File, name string) ([]float32, tensorbank.TensorInfo, bool, error) {
	return readFloatTensorContext(context.Background(), src, file, name)
}

func readFloatTensorContext(ctx context.Context, src *tensorbank.Source, file *tensorbank.File, name string) ([]float32, tensorbank.TensorInfo, bool, error) {
	ti, ok := file.FindTensor(name)
	if !ok || !isFloatDType(ti.DType) {
		return nil, ti, false, nil
	}
	buf := make([]byte, ti.Length)
	const chunk = 16 << 20
	abs := file.PayloadOffset(ti)
	for off := 0; off < len(buf); off += chunk {
		if err := ctx.Err(); err != nil {
			return nil, ti, false, err
		}
		end := off + chunk
		if end > len(buf) {
			end = len(buf)
		}
		if _, err := src.ReadAt(buf[off:end], abs+int64(off)); err != nil {
			return nil, ti, false, err
		}
	}
	out := make([]float32, ti.Elements)
	decodeBuf(out, buf, ti.DType)
	return out, ti, true, nil
}
