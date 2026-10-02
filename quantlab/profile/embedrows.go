package profile

import (
	"math"
	"runtime"
	"sort"
	"sync"

	"quantlab/core"
	"quantlab/qtype"
)

// Token-embedding rows are read by lookup, not matmul: a token's input to
// layer 0 is exactly its row, so a row's quantization error is that token's
// entire perturbation. Corpus KLD only exercises the tokens the corpus
// contains; scoring every row's error covers the rare tokens a fixed
// embedding floor exists to protect, with no corpus at all.

// RowErrorStats summarizes the per-row relative RMS reconstruction error
// ||row − Q_d(row)|| / ||row|| of one dtype over a tensor's rows.
type RowErrorStats struct {
	DType  core.DType `json:"dtype"`
	Rows   int        `json:"rows"`
	Median float64    `json:"median"`
	P999   float64    `json:"p999"`
	Max    float64    `json:"max"`
}

// EmbedRowDTypes are the rungs scored for a token-embedding floor: K/legacy
// types whose Go reference quantizer is exact (embeddings carry no imatrix,
// so IQ rungs are never legal for them).
var EmbedRowDTypes = []core.DType{
	core.DTypeQ8_0, core.DTypeQ6_K, core.DTypeQ5_K_T, core.DTypeQ4_K_T,
	core.DTypeQ3_K, core.DTypeQ2_K,
}

// EmbeddingRowErrors quantizes every row (length ne0) of a row-major float
// tensor with each dtype (uniform importance, as llama-quantize does for a
// tensor without imatrix data) and returns per-dtype row-error statistics.
// All-zero rows are skipped. Dtypes whose block size does not divide ne0
// are left out.
func EmbeddingRowErrors(vals []float32, ne0 int, dtypes []core.DType) map[core.DType]RowErrorStats {
	out := map[core.DType]RowErrorStats{}
	if ne0 <= 0 || len(vals) < ne0 || len(vals)%ne0 != 0 {
		return out
	}
	rows := len(vals) / ne0
	norms := make([]float64, rows)
	for r := 0; r < rows; r++ {
		var s float64
		for _, v := range vals[r*ne0 : (r+1)*ne0] {
			s += float64(v) * float64(v)
		}
		norms[r] = s
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	for _, d := range dtypes {
		bs := qtype.BlockSize(d)
		if bs == 0 || ne0%bs != 0 {
			continue
		}
		errs := make([]float64, rows)
		valid := make([]bool, rows)
		var wg sync.WaitGroup
		next := make(chan int, workers)
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				ws := qtype.NewWorkspace(d)
				defer ws.Release()
				buf := make([]float32, ne0)
				for r := range next {
					if !(norms[r] > 0) {
						continue
					}
					copy(buf, vals[r*ne0:(r+1)*ne0])
					sse, err := qtype.QuantizeDequantRowsWS(d, buf, nil, ne0, ws)
					if err != nil || math.IsNaN(sse) || math.IsInf(sse, 0) {
						continue
					}
					errs[r] = math.Sqrt(sse / norms[r])
					valid[r] = true
				}
			}()
		}
		for r := 0; r < rows; r++ {
			next <- r
		}
		close(next)
		wg.Wait()
		var kept []float64
		for r, ok := range valid {
			if ok {
				kept = append(kept, errs[r])
			}
		}
		if len(kept) == 0 {
			continue
		}
		sort.Float64s(kept)
		out[d] = RowErrorStats{
			DType: d, Rows: len(kept),
			Median: percentileSorted(kept, 0.5),
			P999:   percentileSorted(kept, 0.999),
			Max:    kept[len(kept)-1],
		}
	}
	return out
}

// YardstickError is the typical (median) row error at exactly the
// compression target: log-linear in bits per weight between the two scored
// rungs that bracket the target (quantization error falls roughly
// geometrically per bit), clamped to the outermost rungs. It is the noise a
// typical body weight carries at the target rate. ok=false when nothing
// usable was scored.
func YardstickError(targetBPW float64, stats map[core.DType]RowErrorStats) (float64, bool) {
	type pt struct{ bpw, med, logErr float64 }
	var pts []pt
	for d, st := range stats {
		bpw, ok := d.BitsPerWeight()
		if !ok || !(st.Median > 0) {
			continue
		}
		pts = append(pts, pt{bpw, st.Median, math.Log(st.Median)})
	}
	if len(pts) == 0 || !(targetBPW > 0) {
		return 0, false
	}
	sort.Slice(pts, func(i, j int) bool { return pts[i].bpw < pts[j].bpw })
	if targetBPW <= pts[0].bpw {
		return pts[0].med, true
	}
	last := pts[len(pts)-1]
	if targetBPW >= last.bpw {
		return last.med, true
	}
	for i := 0; i+1 < len(pts); i++ {
		a, b := pts[i], pts[i+1]
		if targetBPW >= a.bpw && targetBPW <= b.bpw {
			if b.bpw == a.bpw {
				return math.Exp(math.Min(a.logErr, b.logErr)), true
			}
			t := (targetBPW - a.bpw) / (b.bpw - a.bpw)
			return math.Exp(a.logErr + t*(b.logErr-a.logErr)), true
		}
	}
	return last.med, true
}

// ChooseEmbeddingFloor picks the cheapest scored rung whose worst rows
// (p99.9 relative error) are no noisier than yardstick, the typical row
// error at the compression target: rare tokens then lose no more than an
// ordinary token would at the target rate. The result never ranks above
// (is never more expensive than) maxFloor: this check only relaxes the
// policy floor. ok=false when no cheaper-or-equal rung qualifies.
func ChooseEmbeddingFloor(stats map[core.DType]RowErrorStats, yardstick float64, maxFloor core.DType, rank func(core.DType) int) (core.DType, bool) {
	if !(yardstick > 0) {
		return "", false
	}
	type cand struct {
		d    core.DType
		bpw  float64
		p999 float64
	}
	var cands []cand
	for d, st := range stats {
		if rank(d) < rank(maxFloor) {
			continue // more expensive than the policy floor
		}
		bpw, _ := d.BitsPerWeight()
		cands = append(cands, cand{d, bpw, st.P999})
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].bpw != cands[j].bpw {
			return cands[i].bpw < cands[j].bpw
		}
		return cands[i].d < cands[j].d
	})
	for _, c := range cands {
		if c.p999 <= yardstick {
			return c.d, true
		}
	}
	return "", false
}

// DecodeFloats decodes a little-endian F32/F16/BF16 payload into dst.
func DecodeFloats(dst []float32, buf []byte, d core.DType) { decodeFloats(dst, buf, d) }
