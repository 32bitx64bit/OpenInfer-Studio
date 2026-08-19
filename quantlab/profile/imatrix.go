package profile

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"

	"quantlab/core"
	"quantlab/tensorbank"
)

const (
	ggufMagic         = 0x46554747
	maxImatrixEntries = 1 << 20
	maxImatrixName    = core.MaxTensorNameLen
	maxImatrixFloats  = 16 << 20 // 64 MiB of F32
	maxImatrixBytes   = 64 << 20
	// maxImatrixVector bounds the retained per-tensor importance vector.
	maxImatrixVector = 1 << 20
	// maxImatrixVectorTotal bounds aggregate vector retention across one
	// imatrix file (64 MiB of f32).
	maxImatrixVectorTotal = 1 << 24
)

// vectorBudget tracks aggregate retention across an imatrix load so
// gigantic files keep aggregates for every tensor but full vectors only
// until the budget is spent.
type vectorBudget struct {
	total int
}

func (b *vectorBudget) canRetain(n int) bool {
	return n <= maxImatrixVector && b.total+n <= maxImatrixVectorTotal
}

func (b *vectorBudget) retain(n int) { b.total += n }

// LoadImatrix parses a llama-imatrix GGUF (*.in_sum2 / *.counts) or legacy
// .dat file into per-tensor aggregate importance statistics keyed by the
// corresponding model tensor name. A readable GGUF that is not an imatrix
// (no *.in_sum2 tensors) returns (nil, nil) so the solver can fall back to
// its heuristic; truncated or malformed files are errors.
func LoadImatrix(path string) (map[string]ImatrixStats, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("profile: open imatrix: %w", err)
	}
	defer f.Close()
	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return nil, fmt.Errorf("profile: read imatrix header: %w", err)
	}
	if binary.LittleEndian.Uint32(magic[:]) == ggufMagic {
		return loadImatrixGGUF(path)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("profile: rewind imatrix: %w", err)
	}
	return loadImatrixDat(f)
}

func loadImatrixGGUF(path string) (map[string]ImatrixStats, error) {
	budget := &vectorBudget{}
	src, err := tensorbank.OpenSource(path)
	if err != nil {
		return nil, fmt.Errorf("profile: open imatrix gguf: %w", err)
	}
	defer src.Close()
	file, err := tensorbank.Parse(src)
	if err != nil {
		return nil, fmt.Errorf("profile: parse imatrix gguf: %w", err)
	}
	type pair struct {
		sum2   []float32
		counts []float32
	}
	byName := map[string]*pair{}
	ensure := func(name string) *pair {
		p := byName[name]
		if p == nil {
			p = &pair{}
			byName[name] = p
		}
		return p
	}
	for _, t := range file.Tensors {
		name, kind := splitImatrixTensor(t.Name)
		if kind == "" {
			continue
		}
		if t.DType != core.DTypeF32 {
			return nil, fmt.Errorf("profile: imatrix tensor %q is %s, want F32", t.Name, t.DType)
		}
		vals, err := readF32(src, file.PayloadOffset(t), t.Length)
		if err != nil {
			return nil, fmt.Errorf("profile: imatrix tensor %q: %w", t.Name, err)
		}
		p := ensure(name)
		switch kind {
		case "in_sum2":
			p.sum2 = vals
		case "counts":
			p.counts = vals
		}
	}
	if len(byName) == 0 {
		// A model GGUF (or any non-imatrix GGUF) is not an error: Dynamic
		// still passes the path to llama-quantize, and the solver falls back
		// to the heuristic estimator.
		return nil, nil
	}
	out := make(map[string]ImatrixStats, len(byName))
	for name, p := range byName {
		if len(p.sum2) == 0 {
			continue
		}
		out[name] = aggregateImatrix(p.sum2, p.counts, budget.canRetain(len(p.sum2)))
		if out[name].Values != nil {
			budget.retain(len(out[name].Values))
		}
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func splitImatrixTensor(name string) (base, kind string) {
	switch {
	case strings.HasSuffix(name, ".in_sum2"):
		return strings.TrimSuffix(name, ".in_sum2"), "in_sum2"
	case strings.HasSuffix(name, ".counts"):
		return strings.TrimSuffix(name, ".counts"), "counts"
	}
	return "", ""
}

func readF32(r tensorbank.Reader, off int64, nbytes uint64) ([]float32, error) {
	if nbytes%4 != 0 {
		return nil, fmt.Errorf("payload length %d is not a multiple of 4", nbytes)
	}
	if nbytes > maxImatrixBytes {
		return nil, fmt.Errorf("payload %d bytes exceeds bound", nbytes)
	}
	buf := make([]byte, nbytes)
	if _, err := r.ReadAt(buf, off); err != nil {
		return nil, err
	}
	out := make([]float32, nbytes/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return out, nil
}

const imatrixEps = 1e-12

func percentileSorted(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n == 1 || p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[n-1]
	}
	pos := p * float64(n-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo < 0 {
		lo = 0
	}
	if hi >= n {
		hi = n - 1
	}
	if lo == hi {
		return sorted[lo]
	}
	w := pos - float64(lo)
	return sorted[lo]*(1-w) + sorted[hi]*w
}

func spikinessOf(mean, max, p50, p95 float64) float64 {
	s := 1.0
	if mean > imatrixEps {
		if r := max / mean; r > s {
			s = r
		}
	} else if max > imatrixEps {
		s = 64
	}
	if p50 > imatrixEps {
		if r := p95 / p50; r > s {
			s = r
		}
	}
	if s > 64 {
		s = 64
	}
	if s < 0 {
		return 0
	}
	return s
}

func aggregateImatrix(sum2, counts []float32, retainVector bool) ImatrixStats {
	nmat := len(counts)
	row := len(sum2)
	if nmat > 0 && len(sum2)%nmat == 0 {
		row = len(sum2) / nmat
	}
	var samples float64
	for _, c := range counts {
		if float64(c) > samples {
			samples = float64(c)
		}
	}
	vals := make([]float64, 0, len(sum2))
	var sum, maxv float64
	for i, v := range sum2 {
		x := float64(v)
		if nmat > 0 && row > 0 {
			ci := i / row
			if ci >= nmat {
				ci = nmat - 1
			}
			if c := float64(counts[ci]); c > 0 {
				x /= c
			}
		}
		if math.IsNaN(x) || math.IsInf(x, 0) {
			continue
		}
		if len(vals) == 0 || x > maxv {
			maxv = x
		}
		sum += x
		vals = append(vals, x)
	}
	st := ImatrixStats{Max: maxv, Samples: uint64(samples)}
	n := len(vals)
	if n == 0 {
		return st
	}
	st.Mean = sum / float64(n)
	if st.Samples == 0 {
		st.Samples = uint64(n)
	}
	sorted := append([]float64(nil), vals...)
	sort.Float64s(sorted)
	st.P50 = percentileSorted(sorted, 0.50)
	st.P95 = percentileSorted(sorted, 0.95)
	var varSum float64
	for _, x := range vals {
		d := x - st.Mean
		varSum += d * d
	}
	st.Variance = varSum / float64(n)
	st.Spikiness = spikinessOf(st.Mean, st.Max, st.P50, st.P95)
	st.Entropy, st.EffRank = shapeFeatures(vals)
	if retainVector {
		vec := make([]float32, len(vals))
		for i, v := range vals {
			vec[i] = float32(v)
		}
		st.Values = vec
	}
	return st
}

// shapeFeatures computes the normalized entropy and participation ratio of
// the per-block importance distribution.
func shapeFeatures(vals []float64) (entropy, effRank float64) {
	if len(vals) == 0 {
		return 0, 0
	}
	var sum float64
	for _, v := range vals {
		if v > 0 {
			sum += v
		}
	}
	if sum <= 0 {
		return 0, 1
	}
	var h, sum2 float64
	for _, v := range vals {
		q := v / sum
		if q > 0 {
			h -= q * math.Log2(q)
		}
		sum2 += v * v
	}
	maxH := math.Log2(float64(len(vals)))
	if maxH > 0 {
		entropy = h / maxH
	}
	if entropy < 0 {
		entropy = 0
	}
	if entropy > 1 {
		entropy = 1
	}
	if sum2 > 0 {
		effRank = sum * sum / sum2
	}
	return entropy, effRank
}

func loadImatrixDat(r io.Reader) (map[string]ImatrixStats, error) {
	var nEntries int32
	if err := binary.Read(r, binary.LittleEndian, &nEntries); err != nil {
		return nil, fmt.Errorf("profile: imatrix dat header: %w", err)
	}
	if nEntries <= 0 || nEntries > maxImatrixEntries {
		return nil, fmt.Errorf("profile: imatrix dat entry count %d out of bounds", nEntries)
	}
	out := make(map[string]ImatrixStats, nEntries)
	for i := int32(0); i < nEntries; i++ {
		var nameLen int32
		if err := binary.Read(r, binary.LittleEndian, &nameLen); err != nil {
			return nil, fmt.Errorf("profile: imatrix dat entry %d name length: %w", i, err)
		}
		if nameLen <= 0 || nameLen > maxImatrixName {
			return nil, fmt.Errorf("profile: imatrix dat entry %d name length %d out of bounds", i, nameLen)
		}
		nameBuf := make([]byte, nameLen)
		if _, err := io.ReadFull(r, nameBuf); err != nil {
			return nil, fmt.Errorf("profile: imatrix dat entry %d name: %w", i, err)
		}
		name := string(nameBuf)
		if err := core.ValidateTensorName(name); err != nil {
			return nil, fmt.Errorf("profile: imatrix dat entry %d: %w", i, err)
		}
		var ncall, nval int32
		if err := binary.Read(r, binary.LittleEndian, &ncall); err != nil {
			return nil, fmt.Errorf("profile: imatrix dat entry %d ncall: %w", i, err)
		}
		if err := binary.Read(r, binary.LittleEndian, &nval); err != nil {
			return nil, fmt.Errorf("profile: imatrix dat entry %d nval: %w", i, err)
		}
		if nval < 0 || nval > maxImatrixFloats {
			return nil, fmt.Errorf("profile: imatrix dat entry %d nval %d out of bounds", i, nval)
		}
		if nval == 0 {
			continue
		}
		vals := make([]float32, nval)
		if err := binary.Read(r, binary.LittleEndian, vals); err != nil {
			return nil, fmt.Errorf("profile: imatrix dat entry %d values: %w", i, err)
		}
		st := aggregateImatrix(vals, nil, len(vals) <= maxImatrixVector)
		if ncall > 0 {
			st.Samples = uint64(ncall)
		}
		out[name] = st
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("profile: imatrix dat has no usable per-tensor stats")
	}
	return out, nil
}
