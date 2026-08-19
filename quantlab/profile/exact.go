package profile

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"

	"quantlab/core"
	"quantlab/qtype"
	"quantlab/tensorbank"
)

// exactUnitAligns maps a relative importance-weighted MSE (per weight) into
// the numeric range of baseSeverity so exact, heuristic, and cache-measured
// losses stay comparable inside the solver's single objective.
const exactUnitAlign = 22.0

// maxExactReadBytes bounds one row-aligned payload read (8 MiB).
const maxExactReadBytes = 8 << 20

// ExactConfig controls optional exact-table refinements. The zero value
// preserves the original importance-weighted SSE table (unit tests).
type ExactConfig struct {
	// ProbeKLD blends a cheap softmax-KLD probe of Wx vs Q(W)x into the
	// per-dtype loss so formats that win MSE but lose logit KLD (heavy
	// tails) are not preferred. Off by default.
	ProbeKLD bool
	// ProbeSamples is the synthetic-input count; 0 selects 4.
	ProbeSamples int
	// ProbeBlend is the KLD mix weight in [0,1]; 0 selects 0.35.
	ProbeBlend float64
	Context    context.Context
	// Existing seeds an identity-validated partial table. Complete tensors
	// are skipped, so a resumed solve does not reread them.
	Existing map[string]map[core.DType]float64
	// OnTensor is called after one tensor completes. Returning an error aborts
	// the build so checkpoint failures are not silently ignored.
	OnTensor func(name string, losses map[core.DType]float64) error
}

// BuildExactLossTable streams the source GGUF once and computes, for every
// quantizable tensor whose source storage is float (F32/F16/BF16) and every
// candidate dtype with a qtype reference quantizer, the
// importance-weighted squared reconstruction error
// sum_ch imp_ch * ||w_ch - Q(w_ch)||^2. imp comes from the imatrix
// per-(row, 256-chunk) Values when available, else weights are uniform.
//
// The map is keyed by tensor name, then by candidate dtype. Tensors skipped
// here fall back to the analytic estimator. progress, when non-nil,
// receives (bytesDone, bytesTotal) notifications; it may be called from
// worker goroutines.
func BuildExactLossTable(bank *core.TensorBank, candidates []core.DType, imatrix map[string]ImatrixStats,
	progress func(done, total int64)) (map[string]map[core.DType]float64, error) {
	return BuildExactLossTableCfg(bank, candidates, imatrix, progress, ExactConfig{})
}

// BuildExactLossTableCfg is BuildExactLossTable with optional probe-KLD.
func BuildExactLossTableCfg(bank *core.TensorBank, candidates []core.DType, imatrix map[string]ImatrixStats,
	progress func(done, total int64), cfg ExactConfig) (map[string]map[core.DType]float64, error) {
	if bank == nil || bank.SourcePath == "" {
		return nil, fmt.Errorf("profile: exact table needs a bank with a source")
	}
	src, err := tensorbank.OpenSource(bank.SourcePath)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	ctx := cfg.Context
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	file, err := tensorbank.Parse(src)
	if err != nil {
		return nil, fmt.Errorf("profile: parse source for exact table: %w", err)
	}

	type job struct {
		t    core.TensorDesc
		off  int64
		impV []float32 // raw per-(row, chunk) values
	}
	var jobs []job
	var totalBytes int64
	for _, t := range bank.Tensors {
		if !t.Quantizable() {
			continue
		}
		switch t.DType {
		case core.DTypeF32, core.DTypeF16, core.DTypeBF16:
		default:
			continue
		}
		any := false
		complete := true
		for _, d := range candidates {
			if qtype.Supported(d) && t.Shape[0]%uint64(qtype.BlockSize(d)) == 0 {
				any = true
				if _, ok := cfg.Existing[t.Name][d]; !ok {
					complete = false
				}
			}
		}
		if !any || complete {
			continue
		}
		ti, ok := file.FindTensor(t.Name)
		if !ok {
			continue
		}
		j := job{t: t, off: file.PayloadOffset(ti)}
		if st, ok := lookupImatrix(imatrix, t.Name); ok {
			j.impV = st.Values
		}
		jobs = append(jobs, j)
		totalBytes += int64(t.Length)
	}
	out := make(map[string]map[core.DType]float64, len(cfg.Existing)+len(jobs))
	for name, losses := range cfg.Existing {
		out[name] = make(map[core.DType]float64, len(losses))
		for d, v := range losses {
			out[name][d] = v
		}
	}
	if len(jobs) == 0 {
		return out, nil
	}

	// Filter candidates to dtypes usable per tensor shape once.
	usable := func(t core.TensorDesc) []core.DType {
		var out []core.DType
		for _, d := range candidates {
			if !qtype.Supported(d) {
				continue
			}
			if t.Shape[0]%uint64(qtype.BlockSize(d)) != 0 {
				continue
			}
			out = append(out, d)
		}
		return out
	}

	var (
		mu       sync.Mutex
		done     atomic.Int64
		workers  = 4
		firstErr error
		once     sync.Once
	)
	if n := len(jobs); n < workers {
		workers = n
	}
	fail := func(err error) {
		once.Do(func() {
			firstErr = err
			cancel()
		})
	}
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup
	for _, j := range jobs {
		wg.Add(1)
		go func(j job) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				fail(ctx.Err())
				return
			}
			defer func() { <-sem }()
			res, err := exactTensorLoss(ctx, src, j.t, j.off, usable(j.t), j.impV, cfg)
			d := done.Add(int64(j.t.Length))
			if progress != nil {
				progress(d, totalBytes)
			}
			if err != nil {
				fail(err)
				return
			}
			mu.Lock()
			out[j.t.Name] = res
			mu.Unlock()
			if cfg.OnTensor != nil {
				err = cfg.OnTensor(j.t.Name, res)
			}
			if err != nil {
				fail(err)
			}
		}(j)
	}
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

// lookupImatrix resolves alternate .weight suffixes, mirroring
// FallbackEstimator.stats.
func lookupImatrix(imatrix map[string]ImatrixStats, name string) (ImatrixStats, bool) {
	if st, ok := imatrix[name]; ok {
		return st, true
	}
	alt := name
	if hasWSuffix(name) {
		alt = trimWSuffix(name)
	} else {
		alt = name + ".weight"
	}
	if st, ok := imatrix[alt]; ok {
		return st, true
	}
	return ImatrixStats{}, false
}

func hasWSuffix(name string) bool    { return len(name) > 7 && name[len(name)-7:] == ".weight" }
func trimWSuffix(name string) string { return name[:len(name)-7] }

// exactTensorLoss streams one tensor's rows and accumulates the weighted
// SSE per candidate dtype.
func exactTensorLoss(ctx context.Context, src *tensorbank.Source, t core.TensorDesc, off int64, candidates []core.DType,
	impV []float32, cfg ExactConfig) (map[core.DType]float64, error) {
	if len(candidates) == 0 {
		return nil, nil
	}
	ne0 := t.Shape[0]
	rows := t.Elements / ne0
	esize := t.Length / t.Elements // source bytes per element
	if esize == 0 {
		esize = 4
	}
	rowBytes := ne0 * esize
	rowsPerRead := maxExactReadBytes / rowBytes
	if rowsPerRead < 1 {
		rowsPerRead = 1
	}
	rowChunks := uint64(0)
	haveImp := false
	if len(impV) > 0 && rows > 0 && uint64(len(impV))%rows == 0 {
		rowChunks = uint64(len(impV)) / rows
		if rowChunks > 0 {
			haveImp = true
		}
	}
	sums := make(map[core.DType]float64, len(candidates))
	buf := make([]byte, rowBytes*rowsPerRead)
	row := make([]float32, ne0*rowsPerRead)
	qrt := make([]float32, ne0*rowsPerRead)
	imp := make([]float32, ne0*rowsPerRead)
	ws := qtype.NewWorkspace(candidates[0])
	defer ws.Release()
	probeN := cfg.ProbeSamples
	if probeN <= 0 {
		probeN = 4
	}
	blend := cfg.ProbeBlend
	if blend <= 0 {
		blend = 0.35
	}
	var rng *rand.Rand
	if cfg.ProbeKLD {
		h := fnv.New64a()
		h.Write([]byte(t.Name))
		rng = rand.New(rand.NewSource(int64(h.Sum64())))
	}
	for r := uint64(0); r < rows; r += rowsPerRead {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n := rowsPerRead
		if rem := rows - r; rem < n {
			n = rem
		}
		read := rowBytes * n
		if _, err := src.ReadAt(buf[:read], off+int64(r*rowBytes)); err != nil {
			return nil, fmt.Errorf("profile: exact read %s: %w", t.Name, err)
		}
		decodeFloats(row[:ne0*n], buf[:read], t.DType)
		if haveImp {
			fillRowImp(imp[:ne0*n], impV, r, n, ne0, rowChunks)
		} else {
			for i := range imp[:ne0*n] {
				imp[i] = 1
			}
		}
		type scored struct {
			sse, kld float64
			bad      bool
		}
		got := make(map[core.DType]scored, len(candidates))
		var xs [][]float64
		if cfg.ProbeKLD && n >= 2 && rng != nil {
			sigma := make([]float32, ne0)
			for c := uint64(0); c < ne0; c++ {
				var is float64
				for r := uint64(0); r < n; r++ {
					is += float64(imp[r*ne0+c])
				}
				sigma[c] = float32(is / float64(n))
			}
			sk := MakeSketches(int(ne0), probeN, sigma, rng.Int63())
			xs = make([][]float64, len(sk))
			for i, row := range sk {
				xs[i] = make([]float64, len(row))
				for j, v := range row {
					xs[i][j] = float64(v)
				}
			}
		}
		for _, d := range candidates {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			copy(qrt[:ne0*n], row[:ne0*n])
			if _, err := qtype.QuantizeDequantWS(d, qrt[:ne0*n], imp[:ne0*n], ws); err != nil {
				return nil, err
			}
			var sse float64
			for i := uint64(0); i < ne0*n; i++ {
				e := float64(row[i]) - float64(qrt[i])
				sse += float64(imp[i]) * e * e
			}
			sc := scored{sse: sse}
			if math.IsNaN(sse) || math.IsInf(sse, 0) {
				sc.bad = true
			} else if len(xs) > 0 {
				sc.kld = probeSoftmaxKLD(row[:ne0*n], qrt[:ne0*n], ne0, n, xs)
			}
			got[d] = sc
		}
		var meanSSE, meanKLD float64
		okN := 0
		for _, sc := range got {
			if sc.bad {
				continue
			}
			meanSSE += sc.sse
			meanKLD += sc.kld
			okN++
		}
		if okN > 0 {
			meanSSE /= float64(okN)
			meanKLD /= float64(okN)
		}
		for d, sc := range got {
			if sc.bad {
				delete(sums, d)
				continue
			}
			loss := sc.sse
			if cfg.ProbeKLD && meanSSE > 0 {
				kldTerm := sc.kld * meanSSE / (meanKLD + 1e-12)
				loss = (1-blend)*sc.sse + blend*kldTerm
			}
			sums[d] += loss
		}
	}
	return sums, nil
}

func probeSoftmaxKLD(orig, quant []float32, ne0, nRows uint64, xs [][]float64) float64 {
	if nRows < 2 || ne0 == 0 || len(xs) == 0 {
		return 0
	}
	y := make([]float64, nRows)
	yq := make([]float64, nRows)
	var sum float64
	for _, x := range xs {
		for r := uint64(0); r < nRows; r++ {
			var a, b float64
			base := r * ne0
			for c := uint64(0); c < ne0; c++ {
				xc := x[c]
				a += float64(orig[base+c]) * xc
				b += float64(quant[base+c]) * xc
			}
			y[r] = a
			yq[r] = b
		}
		sum += softmaxKL(y, yq)
	}
	return sum / float64(len(xs))
}

func softmaxKL(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	ma, mb := a[0], b[0]
	for i := 1; i < len(a); i++ {
		if a[i] > ma {
			ma = a[i]
		}
		if b[i] > mb {
			mb = b[i]
		}
	}
	var sa, sb float64
	ea := make([]float64, len(a))
	eb := make([]float64, len(b))
	for i := range a {
		ea[i] = math.Exp(a[i] - ma)
		sa += ea[i]
		eb[i] = math.Exp(b[i] - mb)
		sb += eb[i]
	}
	if sa <= 0 || sb <= 0 {
		return 0
	}
	var kl float64
	for i := range a {
		p := ea[i] / sa
		q := eb[i] / sb
		if p > 0 && q > 1e-300 {
			kl += p * math.Log(p/q)
		}
	}
	if math.IsNaN(kl) || math.IsInf(kl, 0) || kl < 0 {
		return 0
	}
	return kl
}

func decodeFloats(dst []float32, buf []byte, d core.DType) {
	switch d {
	case core.DTypeF32:
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
	case core.DTypeF16:
		for i := range dst {
			dst[i] = f16ToF32(binary.LittleEndian.Uint16(buf[i*2:]))
		}
	case core.DTypeBF16:
		for i := range dst {
			dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(buf[i*2:])) << 16)
		}
	}
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

// fillRowImp expands per-(row, chunk) importance to per-element weights.
func fillRowImp(dst, impV []float32, rowStart, rows, ne0, rowChunks uint64) {
	for i := range dst {
		r := uint64(i)/ne0 + rowStart
		c := (uint64(i) % ne0) / 256
		if c >= rowChunks {
			c = rowChunks - 1
		}
		dst[i] = impV[r*rowChunks+c]
	}
}

// SetExactLoss installs a precomputed exact loss table (tensor -> dtype ->
// importance-weighted SSE). Estimate consults it before the analytic
// fallback. A nil map clears the table.
func (e *FallbackEstimator) SetExactLoss(t map[string]map[core.DType]float64) {
	if e == nil {
		return
	}
	e.exact = t
}

// HasExactLoss reports whether an exact entry exists for (name, target).
func (e *FallbackEstimator) HasExactLoss(name string, target core.DType) bool {
	if e == nil || e.exact == nil {
		return false
	}
	m, ok := e.exact[name]
	if !ok {
		return false
	}
	_, ok = m[target]
	return ok
}
