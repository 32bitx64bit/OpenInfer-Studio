package reconstruct

import (
	"encoding/binary"
	"io"
	"math"
	"strings"

	"quantlab/core"
	"quantlab/tensorbank"
)

// ApplyImatrix writes a GGUF imatrix whose residual-read in_sum2 256-blocks
// are flattened toward their mean, matching the Hadamard mix of input
// channels. All other tensors copy verbatim. The output is a loadable GGUF.
func ApplyImatrix(src *tensorbank.Source, outPath string) error {
	return tensorbank.Rewrite(src, outPath, func(w io.Writer, src *tensorbank.Source, abs int64, ti tensorbank.TensorInfo) error {
		if ti.DType != core.DTypeF32 || !strings.HasSuffix(ti.Name, ".in_sum2") {
			return tensorbank.CopyPayload(w, src, abs, ti)
		}
		base := strings.TrimSuffix(ti.Name, ".in_sum2")
		if !ReadsResidual(base) {
			return tensorbank.CopyPayload(w, src, abs, ti)
		}
		buf := make([]byte, ti.Length)
		if _, err := src.ReadAt(buf, abs); err != nil {
			return err
		}
		n := int(ti.Elements)
		vals := make([]float32, n)
		for i := 0; i < n; i++ {
			vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
		flatten256(vals)
		for i, v := range vals {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		_, err := w.Write(buf)
		return err
	})
}

func flatten256(v []float32) {
	for off := 0; off+hadamardBlock <= len(v); off += hadamardBlock {
		var s float32
		for i := 0; i < hadamardBlock; i++ {
			s += v[off+i]
		}
		m := s / float32(hadamardBlock)
		for i := 0; i < hadamardBlock; i++ {
			v[off+i] = m
		}
	}
}
