package reconstruct

import (
	"context"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"

	"quantlab/core"
	"quantlab/graph"
	"quantlab/profile"
	"quantlab/qtype"
	"quantlab/tensorbank"
)

type prepResult struct {
	applied bool
	reason  string
	detail  string
}

func graphTensors(file *tensorbank.File) []graph.Tensor {
	out := make([]graph.Tensor, 0, len(file.Tensors))
	for _, t := range file.Tensors {
		out = append(out, graph.Tensor{Name: t.Name, Shape: t.Shape})
	}
	return out
}

func rewriteNamedFloats(ctx context.Context, src *tensorbank.Source, outPath string, names map[string]bool,
	mut func(name string, ti tensorbank.TensorInfo, v []float32) []float32) (int, error) {
	inPlace := samePath(src.Path(), outPath)
	n := 0
	err := rewriteGGUF(ctx, src, outPath, func(w io.Writer, src *tensorbank.Source, abs int64, ti tensorbank.TensorInfo) error {
		if !names[ti.Name] || !isFloatDType(ti.DType) {
			if inPlace {
				return tensorbank.ErrSkipPayload
			}
			return tensorbank.CopyPayloadContext(ctx, w, src, abs, ti)
		}
		buf := make([]byte, ti.Length)
		if _, err := src.ReadAt(buf, abs); err != nil {
			return err
		}
		vals := make([]float32, ti.Elements)
		decodeBuf(vals, buf, ti.DType)
		out := mut(ti.Name, ti, vals)
		if out == nil {
			if inPlace {
				return tensorbank.ErrSkipPayload
			}
			return tensorbank.CopyPayloadContext(ctx, w, src, abs, ti)
		}
		encodeBuf(buf, out, ti.DType)
		if _, err := w.Write(buf); err != nil {
			return err
		}
		n++
		return nil
	})
	return n, err
}

func applyPermute(ctx context.Context, src *tensorbank.Source, outPath string, imatrix map[string]profile.ImatrixStats) (prepResult, error) {
	file, err := tensorbank.Parse(src)
	if err != nil {
		return prepResult{}, err
	}
	gt := graphTensors(file)
	d := graph.Residual(gt)
	if d == 0 || d%256 != 0 {
		return prepResult{reason: "residual width not superblock aligned"}, nil
	}
	score := make([]float64, d)
	var nRead int
	for _, t := range file.Tensors {
		if !graph.ResidualRead(graph.Tensor{Name: t.Name, Shape: t.Shape}, d, gt) || !isFloatDType(t.DType) {
			continue
		}
		vals, _, ok, err := readFloatTensorContext(ctx, src, file, t.Name)
		if err != nil || !ok {
			continue
		}
		nRead++
		rows := int(ne1Of(t))
		ne0 := int(ne0Of(t))
		if ne0 != d || rows == 0 {
			continue
		}
		std := channelStd(imatrix, t.Name, uint64(d), uint64(rows))
		for c := 0; c < d; c++ {
			var s float64
			for r := 0; r < rows; r++ {
				v := float64(vals[r*ne0+c])
				s += v * v
			}
			score[c] += math.Sqrt(s/float64(rows)) * float64(std[c])
		}
	}
	if nRead == 0 {
		return prepResult{reason: "no residual-read tensors"}, nil
	}
	perm := argsortDesc(score) // new[i] = old index
	names := map[string]bool{}
	for _, t := range file.Tensors {
		g := graph.Tensor{Name: t.Name, Shape: t.Shape}
		if graph.ResidualRead(g, d, gt) || graph.ResidualWrite(g, d) {
			names[t.Name] = true
		}
	}
	n, err := rewriteNamedFloats(ctx, src, outPath, names, func(name string, ti tensorbank.TensorInfo, v []float32) []float32 {
		ne0, ne1 := int(ne0Of(ti)), int(ne1Of(ti))
		out := make([]float32, len(v))
		if ne0 == d && ne1 > 0 {
			for r := 0; r < ne1; r++ {
				for i := 0; i < d; i++ {
					out[r*ne0+i] = v[r*ne0+perm[i]]
				}
			}
			return out
		}
		if ne1 == d && ne0 > 0 {
			for i := 0; i < d; i++ {
				copy(out[i*ne0:(i+1)*ne0], v[perm[i]*ne0:(perm[i]+1)*ne0])
			}
			return out
		}
		return nil
	})
	if err != nil {
		return prepResult{}, err
	}
	if n == 0 {
		return prepResult{reason: "permute wrote no tensors"}, nil
	}
	return prepResult{applied: true, detail: fmt.Sprintf("perm %d tensors", n)}, nil
}

func argsortDesc(score []float64) []int {
	idx := make([]int, len(score))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool { return score[idx[i]] > score[idx[j]] })
	return idx
}

func applyMagR(ctx context.Context, src *tensorbank.Source, outPath string, imatrix map[string]profile.ImatrixStats) (prepResult, error) {
	file, err := tensorbank.Parse(src)
	if err != nil {
		return prepResult{}, err
	}
	names := map[string]bool{}
	for _, t := range file.Tensors {
		if isFloatDType(t.DType) && (len(t.Shape) == 2 || len(t.Shape) == 3) && !graph.ConvLike(graph.Tensor{Name: t.Name, Shape: t.Shape}) {
			names[t.Name] = true
		}
	}
	n, err := rewriteNamedFloats(ctx, src, outPath, names, func(name string, ti tensorbank.TensorInfo, v []float32) []float32 {
		return magrShrink(v, importanceVec(imatrix, name, len(v)))
	})
	if err != nil {
		return prepResult{}, err
	}
	if n == 0 {
		return prepResult{reason: "magr wrote nothing"}, nil
	}
	return prepResult{applied: true, detail: fmt.Sprintf("%d tensors", n)}, nil
}

func magrShrink(v, imp []float32) []float32 {
	const bs = 256
	out := append([]float32(nil), v...)
	pin := pinMask(v, imp, bs)
	for off := 0; off < len(v); off += bs {
		end := off + bs
		if end > len(v) {
			end = len(v)
		}
		amax := 0.0
		for i := off; i < end; i++ {
			if pin[i] {
				continue
			}
			if a := math.Abs(float64(v[i])); a > amax {
				amax = a
			}
		}
		if amax == 0 {
			continue
		}
		bestTau := amax
		bestSSE := magrSSE(v[off:end], imp[off:end], pin[off:end], amax)
		for _, f := range []float64{0.85, 0.9, 0.95, 1.0} {
			tau := amax * f
			sse := magrSSE(v[off:end], imp[off:end], pin[off:end], tau)
			if sse < bestSSE {
				bestSSE = sse
				bestTau = tau
			}
		}
		for i := off; i < end; i++ {
			if pin[i] {
				continue
			}
			a := math.Abs(float64(v[i]))
			if a > bestTau && a > 0 {
				out[i] = float32(float64(v[i]) * bestTau / a)
			}
		}
	}
	return out
}

func magrSSE(v, imp []float32, pin []bool, tau float64) float64 {
	tmp := append([]float32(nil), v...)
	for i := range tmp {
		if pin[i] {
			continue
		}
		a := math.Abs(float64(tmp[i]))
		if a > tau && a > 0 {
			tmp[i] = float32(float64(tmp[i]) * tau / a)
		}
	}
	_, err := qtype.QuantizeDequant(core.DTypeQ3_K, tmp, imp)
	if err != nil {
		// pad/trim to block
		bs := qtype.BlockSize(core.DTypeQ3_K)
		if len(tmp)%bs != 0 {
			return 1e9
		}
	}
	var s float64
	for i := range v {
		e := float64(v[i] - tmp[i])
		w := 1.0
		if i < len(imp) {
			w = float64(imp[i])
		}
		s += w * e * e
	}
	return s
}

// pinMask marks MagR super-weights that stay at their F32 values inside the
// loadable GGUF tensor. There is no FP16 sidecar patch.
func pinMask(v, imp []float32, bs int) []bool {
	pin := make([]bool, len(v))
	if len(v) == 0 {
		return pin
	}
	type pair struct {
		i int
		s float64
	}
	ps := make([]pair, len(v))
	for i, x := range v {
		w := 1.0
		if i < len(imp) {
			w = float64(imp[i])
		}
		ps[i] = pair{i, math.Abs(float64(x)) * w}
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].s > ps[j].s })
	n := len(v) / 1000
	if n < 1 {
		n = 1
	}
	if n > 8 {
		n = 8
	}
	for k := 0; k < n; k++ {
		pin[ps[k].i] = true
	}
	return pin
}

func importanceVec(im map[string]profile.ImatrixStats, name string, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	if im == nil {
		return out
	}
	st, ok := im[name]
	if !ok || len(st.Values) == 0 {
		return out
	}
	for i := 0; i < n; i++ {
		out[i] = st.Values[i%len(st.Values)]
		if out[i] <= 0 {
			out[i] = 1e-6
		}
	}
	return out
}

func applyLWC(ctx context.Context, src *tensorbank.Source, outPath string, imatrix map[string]profile.ImatrixStats) (prepResult, error) {
	file, err := tensorbank.Parse(src)
	if err != nil {
		return prepResult{}, err
	}
	names := map[string]bool{}
	for _, t := range file.Tensors {
		if isFloatDType(t.DType) && (len(t.Shape) == 2 || len(t.Shape) == 3) {
			names[t.Name] = true
		}
	}
	n, err := rewriteNamedFloats(ctx, src, outPath, names, func(name string, ti tensorbank.TensorInfo, v []float32) []float32 {
		return lwcClip(v, importanceVec(imatrix, name, len(v)))
	})
	if err != nil {
		return prepResult{}, err
	}
	if n == 0 {
		return prepResult{reason: "lwc wrote nothing"}, nil
	}
	return prepResult{applied: true, detail: fmt.Sprintf("%d tensors", n)}, nil
}

func lwcClip(v, imp []float32) []float32 {
	const bs = 256
	out := append([]float32(nil), v...)
	pin := pinMask(v, imp, bs)
	for off := 0; off < len(v); off += bs {
		end := off + bs
		if end > len(v) {
			end = len(v)
		}
		blk := v[off:end]
		im := imp[off:end]
		mn := blk[0]
		for _, x := range blk {
			if x < mn {
				mn = x
			}
		}
		shifted := make([]float32, len(blk))
		for i, x := range blk {
			shifted[i] = x - mn
		}
		amax := 0.0
		for i, x := range shifted {
			if pin[off+i] {
				continue
			}
			if a := math.Abs(float64(x)); a > amax {
				amax = a
			}
		}
		if amax == 0 {
			continue
		}
		bestA, best := 1.0, math.Inf(1)
		for _, a := range []float64{0.7, 0.8, 0.9, 1.0} {
			tmp := append([]float32(nil), shifted...)
			clip := amax * a
			for i := range tmp {
				if pin[off+i] {
					continue
				}
				if float64(tmp[i]) > clip {
					tmp[i] = float32(clip)
				}
				if float64(tmp[i]) < -clip {
					tmp[i] = float32(-clip)
				}
			}
			work := append([]float32(nil), tmp...)
			if len(work)%qtype.BlockSize(core.DTypeQ3_K) != 0 {
				bestA = a
				break
			}
			_, _ = qtype.QuantizeDequant(core.DTypeQ3_K, work, im)
			var sse float64
			for i := range tmp {
				e := float64(shifted[i] - work[i])
				w := 1.0
				if i < len(im) {
					w = float64(im[i])
				}
				sse += w * e * e
			}
			if sse < best {
				best = sse
				bestA = a
			}
		}
		clip := amax * bestA
		for i := off; i < end; i++ {
			x := float64(v[i] - mn)
			if !pin[i] {
				if x > clip {
					x = clip
				}
				if x < -clip {
					x = -clip
				}
			}
			out[i] = float32(x) + mn
		}
	}
	return out
}

func applyFreqVQ(ctx context.Context, src *tensorbank.Source, outPath string, imatrix map[string]profile.ImatrixStats) (prepResult, error) {
	file, err := tensorbank.Parse(src)
	if err != nil {
		return prepResult{}, err
	}
	gt := graphTensors(file)
	names := map[string]bool{}
	for _, t := range file.Tensors {
		if graph.UniqueLargeAxis(graph.Tensor{Name: t.Name, Shape: t.Shape}, gt) && isFloatDType(t.DType) {
			names[t.Name] = true
		}
	}
	if len(names) == 0 {
		return prepResult{reason: "no unique-large-axis tensors"}, nil
	}
	n, err := rewriteNamedFloats(ctx, src, outPath, names, func(name string, ti tensorbank.TensorInfo, v []float32) []float32 {
		return freqVQ(v, int(ne0Of(ti)), int(ne1Of(ti)))
	})
	if err != nil {
		return prepResult{}, err
	}
	if n == 0 {
		return prepResult{reason: "freqvq wrote nothing"}, nil
	}
	return prepResult{applied: true, detail: fmt.Sprintf("%d tensors", n)}, nil
}

func freqVQ(v []float32, ne0, ne1 int) []float32 {
	if ne0 <= 0 || ne1 < 32 || len(v) != ne0*ne1 {
		return nil
	}
	energy := make([]float64, ne1)
	for r := 0; r < ne1; r++ {
		var s float64
		row := v[r*ne0 : (r+1)*ne0]
		for _, x := range row {
			s += float64(x) * float64(x)
		}
		energy[r] = s
	}
	if !graph.ZipfRows(energy) {
		return nil
	}
	idx := argsortDesc(energy)
	keep := ne1 - ne1/5
	if keep < ne1/2 {
		keep = ne1 / 2
	}
	if float64(ne1-keep)*float64(ne0) > 0.08*float64(len(v)) {
		keep = ne1 - int(0.08*float64(ne1))
	}
	const k = 32
	rng := rand.New(rand.NewSource(3))
	cent := make([][]float32, k)
	for i := 0; i < k; i++ {
		src := idx[i%keep]
		cent[i] = append([]float32(nil), v[src*ne0:(src+1)*ne0]...)
	}
	rare := idx[keep:]
	for it := 0; it < 4; it++ {
		assign := make([]int, len(rare))
		for ri, r := range rare {
			best, bi := math.Inf(1), 0
			row := v[r*ne0 : (r+1)*ne0]
			for c, ct := range cent {
				var s float64
				for j := 0; j < ne0; j++ {
					e := float64(row[j] - ct[j])
					s += e * e
				}
				if s < best {
					best, bi = s, c
				}
			}
			assign[ri] = bi
		}
		if it == 3 {
			out := append([]float32(nil), v...)
			for ri, r := range rare {
				copy(out[r*ne0:(r+1)*ne0], cent[assign[ri]])
			}
			return out
		}
		_ = rng
	}
	return nil
}

func applyCentroid(ctx context.Context, src *tensorbank.Source, outPath string, imatrix map[string]profile.ImatrixStats) (prepResult, error) {
	file, err := tensorbank.Parse(src)
	if err != nil {
		return prepResult{}, err
	}
	m := graph.Analyze(graphTensors(file))
	names := map[string]bool{}
	slices := map[string]int{}
	for _, st := range m.MoE {
		if st.Slices < 4 {
			continue
		}
		names[st.Tensor.Name] = true
		slices[st.Tensor.Name] = st.Slices
	}
	if len(names) == 0 {
		return prepResult{reason: "no fused expert stacks"}, nil
	}
	n, err := rewriteNamedFloats(ctx, src, outPath, names, func(name string, ti tensorbank.TensorInfo, v []float32) []float32 {
		ns := slices[name]
		return shrinkColdExperts(v, ns)
	})
	if err != nil {
		return prepResult{}, err
	}
	if n == 0 {
		return prepResult{reason: "centroid wrote nothing"}, nil
	}
	return prepResult{applied: true, detail: fmt.Sprintf("%d stacks", n)}, nil
}

func shrinkColdExperts(v []float32, nSlice int) []float32 {
	if nSlice < 4 || len(v)%nSlice != 0 {
		return nil
	}
	span := len(v) / nSlice
	energy := make([]float64, nSlice)
	for s := 0; s < nSlice; s++ {
		var e float64
		sl := v[s*span : (s+1)*span]
		for _, x := range sl {
			e += float64(x) * float64(x)
		}
		energy[s] = e
	}
	sorted := append([]float64(nil), energy...)
	sort.Float64s(sorted)
	med := sorted[len(sorted)/2]
	cent := make([]float32, span)
	var wsum float64
	for s := 0; s < nSlice; s++ {
		if energy[s] < med {
			continue
		}
		w := energy[s]
		wsum += w
		sl := v[s*span : (s+1)*span]
		for i, x := range sl {
			cent[i] += float32(w * float64(x))
		}
	}
	if wsum <= 0 {
		return nil
	}
	for i := range cent {
		cent[i] = float32(float64(cent[i]) / wsum)
	}
	out := append([]float32(nil), v...)
	const alpha = 0.35
	for s := 0; s < nSlice; s++ {
		if energy[s] >= med {
			continue
		}
		sl := out[s*span : (s+1)*span]
		for i := range sl {
			sl[i] = sl[i]*(1-alpha) + cent[i]*alpha
		}
	}
	return out
}
