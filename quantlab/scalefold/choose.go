package scalefold

import (
	"encoding/binary"
	"fmt"
	"math"

	"quantlab/core"
	"quantlab/profile"
	"quantlab/qtype"
	"quantlab/tensorbank"
)

// scalesFor builds the per-channel scale vector for one alpha:
// s = (importance^alpha), geometric-mean normalized and clamped. At
// alpha 0 the result is all-ones (the no-op baseline of the grid). A nil
// importance vector behaves as the all-ones fold.
func scalesFor(imp []float64, alpha float64) []float32 {
	n := len(imp)
	out := make([]float32, n)
	if alpha == 0 || imp == nil {
		for i := range out {
			out[i] = 1
		}
		return out
	}
	var logSum float64
	for _, v := range imp {
		if v > 0 {
			logSum += math.Log(v)
		}
	}
	geo := math.Exp(logSum / float64(n))
	if geo <= 0 {
		geo = 1
	}
	for i, v := range imp {
		s := math.Pow(math.Max(v, 1e-30)/geo, alpha)
		if s < 1/scaleClamp {
			s = 1 / scaleClamp
		}
		if s > scaleClamp {
			s = scaleClamp
		}
		out[i] = float32(s)
	}
	return out
}

func decodeF32(payload []byte, d core.DType, out []float32) error {
	switch d {
	case core.DTypeF32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[i*4:]))
		}
	case core.DTypeF16:
		for i := range out {
			out[i] = f16ToF32(binary.LittleEndian.Uint16(payload[i*2:]))
		}
	case core.DTypeBF16:
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(payload[i*2:])) << 16)
		}
	default:
		return fmt.Errorf("scalefold: cannot decode %s", d)
	}
	return nil
}

func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	man := uint32(h & 0x3ff)
	if exp == 0 {
		if man == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(127 - 15 + 1)
		for man&0x400 == 0 {
			man <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (man&0x3ff)<<13)
	}
	if exp == 0x1f {
		return math.Float32frombits(sign | 0x7f800000 | man<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | man<<13)
}

// rowSSE measures the importance-weighted post-fold quantization error of
// one weight row: weights are scaled by s (transform happens in xform),
// dequantized back, and the reconstruction observed with importance
// ch/s² (the activation power after folding).
func rowSSE(d core.DType, w, xform []float32, s, ch []float32, ws *qtype.Workspace) (float64, error) {
	for i := range w {
		xform[i] = w[i] * s[i]
	}
	if _, err := qtype.QuantizeDequantWS(d, xform, nil, ws); err != nil {
		return 0, err
	}
	var sum float64
	for i := range w {
		imp := 1.0
		if ch != nil {
			imp = float64(ch[i])
		}
		imp /= float64(s[i]) * float64(s[i])
		e := float64(w[i])*float64(s[i]) - float64(xform[i])
		sum += imp * e * e
	}
	return sum, nil
}

// ChooseAlpha selects the fold exponent per cluster by measuring total
// importance-weighted post-fold quantization error over the alpha grid at
// the probe dtype. consumers with 3-D shapes are excluded from discovery;
// chunks of rows above the cap are deterministically stride-sampled.
func ChooseAlpha(src *tensorbank.Source, clusters []Cluster,
	imatrix map[string]profile.ImatrixStats, probe core.DType) ([]Cluster, error) {
	file, err := tensorbank.Parse(src)
	if err != nil {
		return nil, fmt.Errorf("scalefold: parse source: %w", err)
	}
	out := make([]Cluster, 0, len(clusters))
	for _, cl := range clusters {
		sel, before, after, skip, err := chooseAlphaForCluster(src, file, cl, imatrix, probe)
		if err != nil {
			return nil, err
		}
		if skip != "" {
			cl.Skipped = skip
			out = append(out, cl)
			continue
		}
		cl.Alpha = AlphaGrid[sel]
		cl.Scales = scalesFor(ChannelImportance(imatrix, cl, shapesForCluster(file, cl)), cl.Alpha)
		cl.ErrBefore = before
		cl.ErrAfter = after
		cl.Probe = probe
		out = append(out, cl)
	}
	return out, nil
}

// shapesForCluster resolves the shape descriptors of a cluster's consumers.
func shapesForCluster(file *tensorbank.File, cl Cluster) map[string]core.TensorDesc {
	out := map[string]core.TensorDesc{}
	for _, name := range cl.Consumers {
		if ti, ok := file.FindTensor(name); ok {
			out[name] = core.TensorDesc{Shape: append([]uint64(nil), ti.Shape...)}
		}
	}
	return out
}

func chooseAlphaForCluster(src *tensorbank.Source, file *tensorbank.File, cl Cluster,
	imatrix map[string]profile.ImatrixStats, probe core.DType) (alpha int, before, after float64, skip string, err error) {
	var ne0 uint64
	shapes := map[string]core.TensorDesc{}
	for _, name := range cl.Consumers {
		ti, ok := file.FindTensor(name)
		if !ok {
			return 0, 0, 0, "missing consumer", nil
		}
		if len(ti.Shape) != 2 || !ti.DType.IsFloat() {
			return 0, 0, 0, "consumer not float 2-D", nil
		}
		if ne0 == 0 {
			ne0 = ti.Shape[0]
		}
		if ti.Shape[0] != ne0 {
			return 0, 0, 0, "consumer dim mismatch", nil
		}
		shapes[name] = core.TensorDesc{Shape: append([]uint64(nil), ti.Shape...)}
	}
	ch64 := ChannelImportance(imatrix, cl, shapes)
	if ch64 == nil {
		return 0, 0, 0, "no imatrix data", nil
	}
	scalesGrid := make([][]float32, len(AlphaGrid))
	for i, a := range AlphaGrid {
		scalesGrid[i] = scalesFor(ch64, a)
	}
	ch32 := make([]float32, ne0)
	for j := range ch32 {
		ch32[j] = float32(ch64[j])
	}
	scores := make([]float64, len(AlphaGrid))
	ws := qtype.NewWorkspace(probe)
	defer ws.Release()
	for _, name := range cl.Consumers {
		ti, _ := file.FindTensor(name)
		rows := ti.Shape[1]
		esize := uint64(4)
		if ti.DType == core.DTypeF16 || ti.DType == core.DTypeBF16 {
			esize = 2
		}
		rowBytes := ne0 * esize
		stride := uint64(1)
		if rows > maxClusterRows {
			stride = rows / maxClusterRows
		}
		off := file.PayloadOffset(ti)
		buf := make([]byte, rowBytes)
		w := make([]float32, ne0)
		xform := make([]float32, ne0)
		for r := uint64(0); r < rows; r += stride {
			if _, err := src.ReadAt(buf, off+int64(r*rowBytes)); err != nil {
				return 0, 0, 0, "", fmt.Errorf("scalefold: read %s row: %w", name, err)
			}
			if err := decodeF32(buf, ti.DType, w); err != nil {
				return 0, 0, 0, "", err
			}
			for gi := range AlphaGrid {
				sse, err := rowSSE(probe, w, xform, scalesGrid[gi], ch32, ws)
				if err != nil {
					return 0, 0, 0, "", err
				}
				scores[gi] += sse
			}
		}
	}
	best := 0
	for i := 1; i < len(scores); i++ {
		if scores[i] < scores[best] {
			best = i
		}
	}
	return best, scores[0], scores[best], "", nil
}
