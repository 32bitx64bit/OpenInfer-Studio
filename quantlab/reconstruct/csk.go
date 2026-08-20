package reconstruct

import (
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sort"
	"sync"

	"quantlab/core"
	"quantlab/graph"
	"quantlab/profile"
	"quantlab/qtype"
	"quantlab/tensorbank"
)

const (
	cskSamples = 64
	cskLambda  = 1e-3
	// Keep the library default conservative for CLI callers that cannot
	// report host memory. Studio supplies a tighter host-aware limit.
	defaultCSKMaxWorkingSet = uint64(1 << 30)
)

type cskResult struct {
	applied bool
	layers  int
	reason  string
}

type swigluLayer struct {
	prefix         string
	gate, up, down tensorbank.TensorInfo
	linear         bool
}

func applyCSK(ctx context.Context, src *tensorbank.Source, outPath string, opts Options) (cskResult, error) {
	if len(opts.Imatrix) == 0 {
		return cskResult{reason: "no imatrix"}, nil
	}
	file, err := tensorbank.Parse(src)
	if err != nil {
		return cskResult{}, err
	}
	layers := discoverSwiGLU(file)
	if len(layers) == 0 {
		return cskResult{reason: "no SwiGLU gate/up/down triples"}, nil
	}
	type indexedLayer struct {
		index int
		layer swigluLayer
	}
	byDown := make(map[string]indexedLayer, len(layers))
	for i, layer := range layers {
		byDown[layer.down.Name] = indexedLayer{index: i, layer: layer}
	}
	maxWorking := opts.MaxWorkingSetBytes
	if maxWorking == 0 {
		maxWorking = defaultCSKMaxWorkingSet
	}
	fitsWorkingSet := false
	var smallestNeed uint64
	for _, layer := range layers {
		nEmb, nFF := int(ne0Of(layer.gate)), int(ne1Of(layer.gate))
		p := cskSamples
		if nFF < p {
			p = nFF
		}
		if nEmb <= 0 || nFF <= 0 || p < 4 {
			continue
		}
		need := estimateCSKWorkingSet(nEmb, nFF, p)
		if smallestNeed == 0 || need < smallestNeed {
			smallestNeed = need
		}
		if need <= maxWorking {
			fitsWorkingSet = true
			break
		}
	}
	if !fitsWorkingSet && smallestNeed > 0 {
		return cskResult{reason: fmt.Sprintf(
			"estimated CSK working set %.1f GiB exceeds %.1f GiB limit",
			float64(smallestNeed)/(1<<30), float64(maxWorking)/(1<<30),
		)}, nil
	}
	inPlace := samePath(src.Path(), outPath)
	applied := 0
	lastSkip := ""
	err = rewriteGGUF(ctx, src, outPath, func(w io.Writer, src *tensorbank.Source, abs int64, ti tensorbank.TensorInfo) error {
		entry, ok := byDown[ti.Name]
		if !ok {
			if inPlace {
				return tensorbank.ErrSkipPayload
			}
			return tensorbank.CopyPayloadContext(ctx, w, src, abs, ti)
		}
		if opts.Progress != nil {
			opts.Progress(Progress{
				Phase: "csk", Layer: entry.index + 1, Layers: len(layers),
				Total: int(ne1Of(entry.layer.down)), Detail: "preparing " + entry.layer.prefix,
			})
		}
		plan, reason, err := prepareCSKLayer(ctx, src, file, entry.layer, opts.Imatrix, maxWorking)
		if err != nil {
			return err
		}
		if plan == nil {
			lastSkip = reason
			if inPlace {
				return tensorbank.ErrSkipPayload
			}
			return tensorbank.CopyPayloadContext(ctx, w, src, abs, ti)
		}
		rows, err := writeCSKDown(ctx, w, src, abs, ti, plan, entry.index, len(layers), opts.Progress)
		if err != nil {
			return err
		}
		if rows > 0 {
			applied++
		} else {
			lastSkip = "no rows improved"
		}
		return nil
	})
	if err != nil {
		return cskResult{}, err
	}
	if applied == 0 {
		if !inPlace {
			_ = os.Remove(outPath)
		}
		if lastSkip == "" {
			lastSkip = "no SwiGLU layer improved by CSK"
		}
		return cskResult{reason: lastSkip}, nil
	}
	return cskResult{applied: true, layers: applied}, nil
}

func discoverSwiGLU(file *tensorbank.File) []swigluLayer {
	named := discoverNamedSwiGLU(file)
	seen := map[string]bool{}
	for _, ly := range named {
		seen[ly.down.Name] = true
	}
	byName := map[string]tensorbank.TensorInfo{}
	gts := make([]graph.Tensor, 0, len(file.Tensors))
	for _, t := range file.Tensors {
		byName[t.Name] = t
		gts = append(gts, graph.Tensor{Name: t.Name, Shape: t.Shape})
	}
	m := graph.Analyze(gts)
	out := named
	for _, mx := range m.Mixers {
		down, ok := byName[mx.Writer.Name]
		if !ok || seen[down.Name] || !isFloatDType(down.DType) {
			continue
		}
		ly := swigluLayer{prefix: mx.Prefix, down: down}
		switch mx.Kind {
		case "glu":
			if len(mx.Expanders) < 2 {
				continue
			}
			g, ok1 := byName[mx.Expanders[0].Name]
			u, ok2 := byName[mx.Expanders[1].Name]
			if !ok1 || !ok2 || !isFloatDType(g.DType) || !isFloatDType(u.DType) {
				continue
			}
			ly.gate, ly.up = g, u
		case "linear":
			if len(mx.Expanders) < 1 {
				continue
			}
			e, ok := byName[mx.Expanders[0].Name]
			if !ok || !isFloatDType(e.DType) {
				continue
			}
			ly.gate, ly.up, ly.linear = e, e, true
		default:
			continue
		}
		if ne0Of(ly.gate) != ne0Of(ly.up) || ne1Of(ly.gate) != ne0Of(ly.down) {
			continue
		}
		out = append(out, ly)
		seen[down.Name] = true
	}
	return out
}

func discoverNamedSwiGLU(file *tensorbank.File) []swigluLayer {
	by := map[string]*swigluLayer{}
	for _, t := range file.Tensors {
		if !isFloatDType(t.DType) || len(t.Shape) != 2 {
			continue
		}
		pre := layerPrefix(t.Name)
		if pre == "" {
			continue
		}
		stem := localStem(t.Name)
		ly := by[pre]
		if ly == nil {
			ly = &swigluLayer{prefix: pre}
			by[pre] = ly
		}
		switch {
		case isFFNGate(stem):
			ly.gate = t
		case isFFNUp(stem):
			ly.up = t
		case isFFNDown(stem):
			ly.down = t
		}
	}
	var keys []string
	for k, ly := range by {
		if ly.gate.Name == "" || ly.up.Name == "" || ly.down.Name == "" {
			continue
		}
		if ne0Of(ly.gate) != ne0Of(ly.up) {
			continue
		}
		if ne1Of(ly.gate) != ne0Of(ly.down) || ne1Of(ly.up) != ne0Of(ly.down) {
			continue
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]swigluLayer, 0, len(keys))
	for _, k := range keys {
		out = append(out, *by[k])
	}
	return out
}

type cskLayerPlan struct {
	// A and diff are row-major nFF×p matrices. A is the quantized hidden
	// activation sketch; diff is original-minus-quantized.
	A, diff      []float64
	cholesky     []float64
	lambda       float64
	nFF, nOut, p int
}

func prepareCSKLayer(ctx context.Context, src *tensorbank.Source, file *tensorbank.File, ly swigluLayer,
	imatrix map[string]profile.ImatrixStats, maxWorking uint64) (*cskLayerPlan, string, error) {
	nEmb := int(ne0Of(ly.gate))
	nFF := int(ne1Of(ly.gate))
	nOut := int(ne1Of(ly.down))
	if nEmb == 0 || nFF == 0 || nOut == 0 {
		return nil, "empty SwiGLU geometry", nil
	}
	if nEmb%qtype.BlockSize(core.DTypeQ3_K) != 0 {
		return nil, "embedding width is not Q3_K block aligned", nil
	}
	p := cskSamples
	if nFF < p {
		p = nFF
	}
	if p < 4 {
		return nil, "SwiGLU width is too small", nil
	}
	if need := estimateCSKWorkingSet(nEmb, nFF, p); maxWorking > 0 && need > maxWorking {
		return nil, fmt.Sprintf("estimated CSK working set %.1f GiB exceeds %.1f GiB limit",
			float64(need)/(1<<30), float64(maxWorking)/(1<<30)), nil
	}
	gate, _, ok, err := readFloatTensorContext(ctx, src, file, ly.gate.Name)
	if err != nil || !ok {
		return nil, "", err
	}
	up, _, ok, err := readFloatTensorContext(ctx, src, file, ly.up.Name)
	if err != nil || !ok {
		return nil, "", err
	}

	std := channelStd(imatrix, ly.gate.Name, uint64(nEmb), ne1Of(ly.gate))
	sk := profile.MakeSketches(nEmb, p, std, int64(fnv32(ly.prefix))+1)
	X := make([]float64, p*nEmb)
	for s, row := range sk {
		if err := ctx.Err(); err != nil {
			return nil, "", err
		}
		copy64 := X[s*nEmb : (s+1)*nEmb]
		for c, v := range row {
			copy64[c] = float64(v)
		}
	}

	hOrigA := make([]float64, p*nFF)
	hOrigB := make([]float64, p*nFF)
	for s := 0; s < p; s++ {
		x := X[s*nEmb : (s+1)*nEmb]
		if err := hiddenInto(ctx, hOrigA[s*nFF:(s+1)*nFF], gate, up, x, nEmb, nFF, ly.linear, false); err != nil {
			return nil, "", err
		}
		if !ly.linear {
			if err := hiddenInto(ctx, hOrigB[s*nFF:(s+1)*nFF], gate, up, x, nEmb, nFF, false, true); err != nil {
				return nil, "", err
			}
		}
	}

	if err := quantizeDequantContext(ctx, core.DTypeQ3_K, gate); err != nil {
		return nil, "", fmt.Errorf("reconstruct: CSK Q3_K gate: %w", err)
	}
	if !ly.linear {
		if err := quantizeDequantContext(ctx, core.DTypeQ3_K, up); err != nil {
			return nil, "", fmt.Errorf("reconstruct: CSK Q3_K up: %w", err)
		}
	} else {
		copy(up, gate)
	}

	fill := func(hOrig []float64, swap bool) (A, diff []float64, frob float64, err error) {
		A = make([]float64, nFF*p)
		diff = make([]float64, nFF*p)
		hQ := make([]float64, nFF)
		for s := 0; s < p; s++ {
			if err = hiddenInto(ctx, hQ, gate, up, X[s*nEmb:(s+1)*nEmb], nEmb, nFF, ly.linear, swap); err != nil {
				return
			}
			for i, v := range hQ {
				off := i*p + s
				A[off] = v
				e := hOrig[s*nFF+i] - v
				diff[off] = e
				frob += e * e
			}
		}
		return
	}
	A, diff, frob, err := fill(hOrigA, false)
	if err != nil {
		return nil, "", err
	}
	if !ly.linear {
		A2, diff2, frob2, err := fill(hOrigB, true)
		if err != nil {
			return nil, "", err
		}
		if frob2 > frob {
			A, diff, frob = A2, diff2, frob2
		}
	}
	lambda := cskLambda * (frob/float64(nFF*p) + 1)

	// Build AᵀA + λI and factor it once per layer. The previous implementation
	// rebuilt and refactored this invariant p×p matrix for every output row.
	gram := make([]float64, p*p)
	for r := 0; r < nFF; r++ {
		if r&63 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, "", err
			}
		}
		row := A[r*p : (r+1)*p]
		for i := 0; i < p; i++ {
			for j := 0; j <= i; j++ {
				gram[i*p+j] += row[i] * row[j]
			}
		}
	}
	for i := 0; i < p; i++ {
		for j := 0; j < i; j++ {
			gram[j*p+i] = gram[i*p+j]
		}
		gram[i*p+i] += lambda
	}
	L, ok := choleskyFactorFlat(gram, p)
	if !ok {
		return nil, "CSK normal matrix is not positive definite", nil
	}
	return &cskLayerPlan{A: A, diff: diff, cholesky: L, lambda: lambda, nFF: nFF, nOut: nOut, p: p}, "", nil
}

func estimateCSKWorkingSet(nEmb, nFF, p int) uint64 {
	if nEmb <= 0 || nFF <= 0 || p <= 0 {
		return 0
	}
	// Two float32 projections, one transient source buffer per projection,
	// original/sketch matrices, and 25% allocator/headroom margin.
	matrix := uint64(nEmb) * uint64(nFF)
	sketch := uint64(nFF) * uint64(p)
	base := matrix*12 + sketch*24 + uint64(nEmb*p)*8
	return base + base/4
}

func quantizeDequantContext(ctx context.Context, dtype core.DType, values []float32) error {
	bs := qtype.BlockSize(dtype)
	if bs <= 0 {
		return fmt.Errorf("unsupported dtype %s", dtype)
	}
	const target = 1 << 20
	chunk := target - target%bs
	if chunk < bs {
		chunk = bs
	}
	ws := qtype.NewWorkspace(dtype)
	defer ws.Release()
	for off := 0; off < len(values); off += chunk {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := off + chunk
		if end > len(values) {
			end = len(values)
		}
		if _, err := qtype.QuantizeDequantWS(dtype, values[off:end], nil, ws); err != nil {
			return err
		}
	}
	return nil
}

type cskRowResult struct {
	buf      []byte
	improved bool
	err      error
}

func writeCSKDown(ctx context.Context, w io.Writer, src *tensorbank.Source, abs int64,
	ti tensorbank.TensorInfo, plan *cskLayerPlan, layer, layers int, progress func(Progress)) (int, error) {
	if plan == nil || plan.nFF <= 0 || plan.nOut <= 0 {
		return 0, fmt.Errorf("reconstruct: invalid CSK layer plan")
	}
	es := int(elemSize(ti.DType))
	if es == 0 || uint64(plan.nFF*plan.nOut) != ti.Elements {
		return 0, fmt.Errorf("reconstruct: CSK down geometry does not match %s", ti.Name)
	}
	rowBytes := plan.nFF * es
	workers := runtime.GOMAXPROCS(0)
	if workers > 16 {
		workers = 16
	}
	if workers < 1 {
		workers = 1
	}
	if workers > plan.nOut {
		workers = plan.nOut
	}
	improved := 0
	for base := 0; base < plan.nOut; base += workers {
		if err := ctx.Err(); err != nil {
			return improved, err
		}
		n := workers
		if rem := plan.nOut - base; rem < n {
			n = rem
		}
		results := make([]cskRowResult, n)
		for i := 0; i < n; i++ {
			results[i].buf = make([]byte, rowBytes)
			if _, err := src.ReadAt(results[i].buf, abs+int64((base+i)*rowBytes)); err != nil {
				return improved, err
			}
		}
		var wg sync.WaitGroup
		for i := range results {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				results[i].improved, results[i].err = compensateCSKRow(ctx, results[i].buf, ti.DType, plan)
			}(i)
		}
		wg.Wait()
		for i := range results {
			if results[i].err != nil {
				return improved, results[i].err
			}
			if _, err := w.Write(results[i].buf); err != nil {
				return improved, err
			}
			if results[i].improved {
				improved++
			}
		}
		done := base + n
		if progress != nil {
			progress(Progress{
				Phase: "csk", Layer: layer + 1, Layers: layers,
				Current: done, Total: plan.nOut, Detail: ti.Name,
			})
		}
	}
	return improved, nil
}

func compensateCSKRow(ctx context.Context, buf []byte, dtype core.DType, plan *cskLayerPlan) (bool, error) {
	es := int(elemSize(dtype))
	w0 := make([]float64, plan.nFF)
	rhs := make([]float64, plan.p)
	for i := 0; i < plan.nFF; i++ {
		if i&255 == 0 {
			if err := ctx.Err(); err != nil {
				return false, err
			}
		}
		wi := float64(decodeScalar(buf[i*es:], dtype))
		w0[i] = wi
		row := plan.diff[i*plan.p : (i+1)*plan.p]
		for s, v := range row {
			rhs[s] += v * wi
		}
	}
	var before float64
	for _, v := range rhs {
		before += v * v
	}
	if !(before > 0) || math.IsNaN(before) || math.IsInf(before, 0) {
		return false, nil
	}
	alpha, ok := choleskySolveFlat(plan.cholesky, plan.p, rhs)
	if !ok {
		return false, nil
	}
	// Since (AᵀA + λI)α = rhs, the fitted residual is λ α. This avoids two
	// extra nFF×p matrix products per output row.
	var alpha2 float64
	for _, v := range alpha {
		alpha2 += v * v
	}
	after := plan.lambda * plan.lambda * alpha2
	if !(after < before*(1-1e-12)) || math.IsNaN(after) || math.IsInf(after, 0) {
		return false, nil
	}
	out := make([]float32, plan.nFF)
	for i := 0; i < plan.nFF; i++ {
		if i&255 == 0 {
			if err := ctx.Err(); err != nil {
				return false, err
			}
		}
		row := plan.A[i*plan.p : (i+1)*plan.p]
		correction := 0.0
		for s, v := range row {
			correction += v * alpha[s]
		}
		v := w0[i] + correction
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false, nil
		}
		out[i] = float32(v)
	}
	encodeBuf(buf, out, dtype)
	return true, nil
}

func hiddenInto(ctx context.Context, h []float64, gate, up []float32, x []float64, nEmb, nFF int, linear, swap bool) error {
	for o := 0; o < nFF; o++ {
		if o&63 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		var g, u float64
		base := o * nEmb
		for c := 0; c < nEmb; c++ {
			g += float64(gate[base+c]) * x[c]
			u += float64(up[base+c]) * x[c]
		}
		switch {
		case linear:
			h[o] = u
		case swap:
			h[o] = silu(u) * g
		default:
			h[o] = silu(g) * u
		}
	}
	return nil
}

func silu(x float64) float64 {
	if x > 20 {
		return x
	}
	if x < -20 {
		return 0
	}
	return x / (1 + math.Exp(-x))
}

func channelStd(imatrix map[string]profile.ImatrixStats, name string, ne0, rows uint64) []float32 {
	out := make([]float32, ne0)
	for i := range out {
		out[i] = 1
	}
	st, ok := imatrix[name]
	if !ok {
		alt := name
		if len(name) > 7 && name[len(name)-7:] == ".weight" {
			alt = name[:len(name)-7]
		} else {
			alt = name + ".weight"
		}
		st, ok = imatrix[alt]
	}
	if !ok || len(st.Values) == 0 || rows == 0 {
		return out
	}
	rowChunks := uint64(len(st.Values)) / rows
	if rowChunks == 0 {
		return out
	}
	sz := ne0 / rowChunks
	if sz == 0 {
		sz = 1
	}
	for c := uint64(0); c < rowChunks; c++ {
		var s float64
		n := uint64(0)
		for r := uint64(0); r < rows && r*rowChunks+c < uint64(len(st.Values)); r++ {
			s += float64(st.Values[r*rowChunks+c])
			n++
		}
		if n == 0 {
			continue
		}
		v := math.Sqrt(math.Max(s/float64(n), 1e-12))
		for j := c * sz; j < (c+1)*sz && j < ne0; j++ {
			out[j] = float32(v)
		}
	}
	return out
}

func choleskyFactorFlat(A []float64, n int) ([]float64, bool) {
	if n <= 0 || len(A) != n*n {
		return nil, false
	}
	L := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			s := A[i*n+j]
			for k := 0; k < j; k++ {
				s -= L[i*n+k] * L[j*n+k]
			}
			if i == j {
				if s <= 1e-18 {
					return nil, false
				}
				L[i*n+j] = math.Sqrt(s)
			} else {
				L[i*n+j] = s / L[j*n+j]
			}
		}
	}
	return L, true
}

func choleskySolveFlat(L []float64, n int, b []float64) ([]float64, bool) {
	if n <= 0 || len(L) != n*n || len(b) != n {
		return nil, false
	}
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		s := b[i]
		for k := 0; k < i; k++ {
			s -= L[i*n+k] * y[k]
		}
		diag := L[i*n+i]
		if diag == 0 {
			return nil, false
		}
		y[i] = s / diag
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		s := y[i]
		for k := i + 1; k < n; k++ {
			s -= L[k*n+i] * x[k]
		}
		x[i] = s / L[i*n+i]
	}
	return x, true
}

func fnv32(s string) uint32 {
	var h uint32 = 2166136261
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= 16777619
	}
	return h
}
