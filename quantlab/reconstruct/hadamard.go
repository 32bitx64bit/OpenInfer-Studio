package reconstruct

import (
	"context"
	"io"
	"math"
	"strings"

	"quantlab/core"
	"quantlab/scalefold"
	"quantlab/tensorbank"
)

const hadamardBlock = 256

type hadamardResult struct {
	applied bool
	dim     int
	reason  string
}

type tensorKind int

const (
	kindCopy tensorKind = iota
	kindNormOnes
	kindRightH // FWHT along ne0 (read residual / embed / output)
	kindLeftH  // FWHT along ne1 (write residual)
)

func applyHadamard(ctx context.Context, src *tensorbank.Source, outPath string) (hadamardResult, error) {
	file, err := tensorbank.Parse(src)
	if err != nil {
		return hadamardResult{}, err
	}
	d := residualWidth(file)
	if d == 0 || d%hadamardBlock != 0 {
		return hadamardResult{dim: d, reason: "residual width not a multiple of 256"}, nil
	}
	bank := &core.TensorBank{}
	for _, t := range file.Tensors {
		bank.Tensors = append(bank.Tensors, core.TensorDesc{
			Name: t.Name, DType: t.DType, Shape: t.Shape, Elements: t.Elements, Length: t.Length,
		})
	}
	clusters := scalefold.Discover(bank)
	if len(clusters) == 0 {
		return hadamardResult{dim: d, reason: "no RMSNorm clusters to absorb γ"}, nil
	}

	gamma := map[string][]float32{} // consumer -> per-input-channel scales
	onesFor := map[string]bool{}
	for _, cl := range clusters {
		if err := ctx.Err(); err != nil {
			return hadamardResult{}, err
		}
		g, _, ok, err := readFloatTensor(src, file, cl.Norm)
		if err != nil {
			return hadamardResult{}, err
		}
		if !ok || len(g) != d {
			continue
		}
		for _, name := range cl.Consumers {
			gamma[name] = g
		}
		onesFor[cl.Norm] = true
	}
	if on, g := outputNormGamma(src, file, d); on != "" && g != nil {
		onesFor[on] = true
		for _, t := range file.Tensors {
			if isOutputHead(t.Name) && int(ne0Of(t)) == d {
				gamma[t.Name] = g
			}
		}
	}
	if len(gamma) == 0 {
		return hadamardResult{dim: d, reason: "could not load RMSNorm γ"}, nil
	}

	if reason, err := incoherenceGate(ctx, src, file, d); err != nil {
		return hadamardResult{}, err
	} else if reason != "" {
		return hadamardResult{dim: d, reason: reason}, nil
	}

	kindOf := func(ti tensorbank.TensorInfo) tensorKind {
		if !isFloatDType(ti.DType) {
			return kindCopy
		}
		if onesFor[ti.Name] && len(ti.Shape) == 1 {
			return kindNormOnes
		}
		if isWriteResidual(ti.Name) && int(ne1Of(ti)) == d && uint64(d)%uint64(hadamardBlock) == 0 {
			return kindLeftH
		}
		if int(ne0Of(ti)) == d && (isReadResidual(ti.Name) || isEmbedding(ti.Name) || isOutputHead(ti.Name)) {
			return kindRightH
		}
		return kindCopy
	}

	inPlace := samePath(src.Path(), outPath)
	err = rewriteGGUF(ctx, src, outPath, func(w io.Writer, src *tensorbank.Source, abs int64, ti tensorbank.TensorInfo) error {
		switch kindOf(ti) {
		case kindNormOnes:
			return writeOnes(w, ti)
		case kindRightH:
			return streamRightH(ctx, w, src, abs, ti, gamma[ti.Name], d)
		case kindLeftH:
			return streamLeftH(ctx, w, src, abs, ti)
		default:
			if inPlace {
				return tensorbank.ErrSkipPayload
			}
			return tensorbank.CopyPayloadContext(ctx, w, src, abs, ti)
		}
	})
	if err != nil {
		return hadamardResult{}, err
	}
	return hadamardResult{applied: true, dim: d}, nil
}

func outputNormGamma(src *tensorbank.Source, file *tensorbank.File, d int) (string, []float32) {
	for _, t := range file.Tensors {
		if len(t.Shape) != 1 || int(t.Shape[0]) != d || !isFloatDType(t.DType) {
			continue
		}
		low := strings.ToLower(t.Name)
		if !strings.Contains(low, "norm") {
			continue
		}
		if strings.Contains(low, "output") || strings.Contains(low, "final") {
			g, _, ok, err := readFloatTensor(src, file, t.Name)
			if err == nil && ok {
				return t.Name, g
			}
		}
	}
	return "", nil
}

func incoherenceGate(ctx context.Context, src *tensorbank.Source, file *tensorbank.File, d int) (string, error) {
	var n, nonfinite int
	var sum, sum2, sum4, absmax float64
	sampled := 0
	for _, t := range file.Tensors {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		if sampled >= 4 {
			break
		}
		if !isReadResidual(t.Name) || !isFloatDType(t.DType) || int(ne0Of(t)) != d {
			continue
		}
		sampled++
		es := elemSize(t.DType)
		want := uint64(4096)
		if t.Elements < want {
			want = t.Elements
		}
		buf := make([]byte, want*es)
		if _, err := src.ReadAt(buf, file.PayloadOffset(t)); err != nil {
			continue
		}
		tmp := make([]float32, want)
		decodeBuf(tmp, buf, t.DType)
		for _, v := range tmp {
			x := float64(v)
			n++
			if math.IsNaN(x) || math.IsInf(x, 0) {
				nonfinite++
				continue
			}
			a := math.Abs(x)
			if a > absmax {
				absmax = a
			}
			sum += x
			sum2 += x * x
			sum4 += x * x * x * x
		}
	}
	if n == 0 {
		return "no residual-read float weights to gate on", nil
	}
	if float64(nonfinite)/float64(n) > 0.005 {
		return "non-finite residual weights; skip Hadamard", nil
	}
	finite := n - nonfinite
	if finite < 32 {
		return "too few finite residual weights", nil
	}
	mean := sum / float64(finite)
	mom2 := sum2/float64(finite) - mean*mean
	if mom2 <= 0 {
		return "degenerate residual weight variance", nil
	}
	// Excess kurtosis of a Gaussian is 0. Heavy-tailed residual weights
	// (the Hadamard win condition) sit well above that.
	kurt := (sum4/float64(finite))/(mom2*mom2) - 3
	rms := math.Sqrt(sum2 / float64(finite))
	peak := 0.0
	if rms > 0 {
		peak = absmax / rms
	}
	if kurt < 1 && peak < 6 {
		return "residual weights already near-Gaussian; skip Hadamard", nil
	}
	return "", nil
}

func writeOnes(w io.Writer, ti tensorbank.TensorInfo) error {
	es := elemSize(ti.DType)
	buf := make([]byte, ti.Length)
	one := make([]byte, es)
	encodeScalar(one, ti.DType, 1)
	for i := uint64(0); i < ti.Elements; i++ {
		copy(buf[i*es:], one)
	}
	_, err := w.Write(buf)
	return err
}

func streamRightH(ctx context.Context, w io.Writer, src *tensorbank.Source, abs int64, ti tensorbank.TensorInfo, gamma []float32, d int) error {
	ne0 := ne0Of(ti)
	rows := ti.Elements / ne0
	es := elemSize(ti.DType)
	rowBytes := ne0 * es
	buf := make([]byte, rowBytes)
	row := make([]float32, ne0)
	for r := uint64(0); r < rows; r++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if _, err := src.ReadAt(buf, abs+int64(r*rowBytes)); err != nil {
			return err
		}
		decodeBuf(row, buf, ti.DType)
		if len(gamma) == int(ne0) {
			for c := uint64(0); c < ne0; c++ {
				row[c] *= gamma[c]
			}
		}
		fwhtRows(row, d)
		encodeBuf(buf, row, ti.DType)
		if _, err := w.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

func streamLeftH(ctx context.Context, w io.Writer, src *tensorbank.Source, abs int64, ti tensorbank.TensorInfo) error {
	ne0 := ne0Of(ti)
	rows := ti.Elements / ne0
	es := elemSize(ti.DType)
	rowBytes := ne0 * es
	blockRows := uint64(hadamardBlock)
	buf := make([]byte, rowBytes*blockRows)
	rowsF := make([]float32, ne0*blockRows)
	col := make([]float64, hadamardBlock)
	for r := uint64(0); r < rows; r += blockRows {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := blockRows
		if rem := rows - r; rem < n {
			n = rem
		}
		read := rowBytes * n
		if _, err := src.ReadAt(buf[:read], abs+int64(r*rowBytes)); err != nil {
			return err
		}
		decodeBuf(rowsF[:ne0*n], buf[:read], ti.DType)
		if n == blockRows {
			for c := uint64(0); c < ne0; c++ {
				for i := uint64(0); i < blockRows; i++ {
					col[i] = float64(rowsF[i*ne0+c])
				}
				fwht(col)
				for i := uint64(0); i < blockRows; i++ {
					rowsF[i*ne0+c] = float32(col[i])
				}
			}
		}
		encodeBuf(buf[:read], rowsF[:ne0*n], ti.DType)
		if _, err := w.Write(buf[:read]); err != nil {
			return err
		}
	}
	return nil
}

func fwhtRows(row []float32, d int) {
	if int(len(row)) < d {
		d = len(row)
	}
	tmp := make([]float64, hadamardBlock)
	for off := 0; off+hadamardBlock <= d && off+hadamardBlock <= len(row); off += hadamardBlock {
		for i := 0; i < hadamardBlock; i++ {
			tmp[i] = float64(row[off+i])
		}
		fwht(tmp)
		for i := 0; i < hadamardBlock; i++ {
			row[off+i] = float32(tmp[i])
		}
	}
}

// fwht is a normalized in-place Fast Walsh–Hadamard transform (H H = I).
func fwht(x []float64) {
	n := len(x)
	for h := 1; h < n; h *= 2 {
		for i := 0; i < n; i += h * 2 {
			for j := i; j < i+h; j++ {
				a, b := x[j], x[j+h]
				x[j] = a + b
				x[j+h] = a - b
			}
		}
	}
	s := 1 / math.Sqrt(float64(n))
	for i := range x {
		x[i] *= s
	}
}
