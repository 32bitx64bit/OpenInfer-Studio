// Package graph classifies GGUF tensors by shape and residual geometry.
// Architecture names are a fallback when two layouts are equally plausible.
package graph

import (
	"math"
	"strings"
)

// Tensor is the name/shape pair the rest of QuantLab already has.
type Tensor struct {
	Name  string
	Shape []uint64
}

// Residual is the inferred residual stream width (0 if unknown).
func Residual(ts []Tensor) int {
	counts := map[uint64]int{}
	for _, t := range ts {
		if len(t.Shape) != 1 {
			continue
		}
		n := strings.ToLower(t.Name)
		if !strings.Contains(n, "norm") || !strings.HasSuffix(n, ".weight") {
			continue
		}
		counts[t.Shape[0]]++
	}
	var best uint64
	bestN := 0
	for d, n := range counts {
		if n > bestN || (n == bestN && d > best) {
			best, bestN = d, n
		}
	}
	return int(best)
}

// LayerPrefix is the GGUF blk.N / layers.N / layer.N grouping, or "".
func LayerPrefix(name string) string {
	for _, pre := range []string{"blk.", "layers.", "layer."} {
		if i := strings.Index(name, pre); i >= 0 {
			rest := name[i+len(pre):]
			if j := strings.IndexByte(rest, '.'); j >= 0 {
				return pre + rest[:j]
			}
		}
	}
	return ""
}

func ne(t Tensor, i int) uint64 {
	if i < 0 || i >= len(t.Shape) {
		return 0
	}
	return t.Shape[i]
}

func uniqueHuge(ts []Tensor) uint64 {
	counts := map[uint64]int{}
	var maxd uint64
	for _, t := range ts {
		for _, d := range t.Shape {
			counts[d]++
			if d > maxd {
				maxd = d
			}
		}
	}
	if maxd == 0 {
		return 0
	}
	var second uint64
	for d := range counts {
		if d < maxd && d > second {
			second = d
		}
	}
	if second > 0 && maxd >= second*8 && counts[maxd] <= 4 {
		return maxd
	}
	return 0
}

// UniqueLargeAxis reports whether t has one uniquely huge axis (vocab-like).
// Residual width is never vocab: a tiny fixture whose largest dim is d
// must still count as ResidualRead, not as an embedding.
func UniqueLargeAxis(t Tensor, ts []Tensor) bool {
	huge := uniqueHuge(ts)
	if huge == 0 || len(t.Shape) < 2 {
		return false
	}
	if d := Residual(ts); d > 0 && huge == uint64(d) {
		return false
	}
	n := 0
	for _, d := range t.Shape {
		if d == huge {
			n++
		}
	}
	return n == 1
}

// GQAShort reports a residual-read whose output axis is a small divisor of d.
func GQAShort(t Tensor, d int) bool {
	if d <= 0 || len(t.Shape) != 2 {
		return false
	}
	if int(ne(t, 0)) != d {
		return false
	}
	n1 := int(ne(t, 1))
	return n1 > 0 && n1 < d && n1*4 <= d && d%n1 == 0
}

// ResidualRead is rank-2, ne0==d, not vocab-wide.
func ResidualRead(t Tensor, d int, ts []Tensor) bool {
	if d <= 0 || len(t.Shape) != 2 || int(ne(t, 0)) != d {
		return false
	}
	return !UniqueLargeAxis(t, ts)
}

// ResidualWrite is rank-2, ne1==d.
func ResidualWrite(t Tensor, d int) bool {
	return d > 0 && len(t.Shape) == 2 && int(ne(t, 1)) == d
}

// Mixer is a residual writer plus the expanders that feed it.
type Mixer struct {
	Prefix    string
	Expanders []Tensor
	Writer    Tensor
	Kind      string // "glu", "linear", "write"
	Slices    int
}

// MoEStack is a rank-3 expert stack and optional 2-D router.
type MoEStack struct {
	Tensor Tensor
	Router Tensor
	Slices int
}

// Model is the geometry view of one GGUF.
type Model struct {
	Tensors []Tensor
	D       int
	Reads   []Tensor
	Writes  []Tensor
	Mixers  []Mixer
	MoE     []MoEStack
	Vocab   uint64
}

// Analyze builds a residual graph from tensor shapes.
func Analyze(ts []Tensor) Model {
	m := Model{Tensors: ts, D: Residual(ts), Vocab: uniqueHuge(ts)}
	d := m.D
	for _, t := range ts {
		if ResidualRead(t, d, ts) {
			m.Reads = append(m.Reads, t)
		}
		if ResidualWrite(t, d) {
			m.Writes = append(m.Writes, t)
		}
		if st, ok := moeStack(t, ts, d); ok {
			m.MoE = append(m.MoE, st)
		}
	}
	m.Mixers = mixers(ts, d)
	return m
}

func moeStack(t Tensor, ts []Tensor, d int) (MoEStack, bool) {
	if len(t.Shape) != 3 {
		return MoEStack{}, false
	}
	small, smallI := t.Shape[0], 0
	for i, dim := range t.Shape {
		if dim < small {
			small, smallI = dim, i
		}
	}
	if small < 2 || small > 4096 {
		return MoEStack{}, false
	}
	if d > 0 {
		hasD := false
		for i, dim := range t.Shape {
			if i != smallI && int(dim) == d {
				hasD = true
			}
		}
		if !hasD {
			return MoEStack{}, false
		}
	}
	st := MoEStack{Tensor: t, Slices: int(small)}
	for _, u := range ts {
		if len(u.Shape) != 2 || LayerPrefix(u.Name) != LayerPrefix(t.Name) {
			continue
		}
		if ne(u, 0) <= 256 && ne(u, 1) == small {
			st.Router = u
			break
		}
		if ne(u, 1) <= 256 && ne(u, 0) == small {
			st.Router = u
			break
		}
	}
	return st, true
}

func mixers(ts []Tensor, d int) []Mixer {
	if d <= 0 {
		return nil
	}
	by := map[string][]Tensor{}
	var keys []string
	seen := map[string]bool{}
	for _, t := range ts {
		if len(t.Shape) != 2 {
			continue
		}
		pre := LayerPrefix(t.Name)
		if pre == "" {
			continue
		}
		by[pre] = append(by[pre], t)
		if !seen[pre] {
			seen[pre] = true
			keys = append(keys, pre)
		}
	}
	var out []Mixer
	for _, pre := range keys {
		group := by[pre]
		var expand []Tensor
		var writers []Tensor
		for _, t := range group {
			if int(ne(t, 0)) == d && int(ne(t, 1)) != d {
				expand = append(expand, t)
			}
			if int(ne(t, 1)) == d {
				writers = append(writers, t)
			}
		}
		for _, w := range writers {
			nFF := int(ne(w, 0))
			var pair []Tensor
			for _, e := range expand {
				if int(ne(e, 1)) == nFF && int(ne(e, 0)) == d {
					pair = append(pair, e)
				}
			}
			mx := Mixer{Prefix: pre, Writer: w, Expanders: pair, Slices: 1}
			switch {
			case len(pair) >= 2:
				mx.Kind = "glu"
				mx.Expanders = pair[:2]
			case len(pair) == 1:
				mx.Kind = "linear"
			default:
				mx.Kind = "write"
			}
			out = append(out, mx)
		}
	}
	return out
}

// ZipfRows reports whether row energies look Zipf-like (vocab embeddings).
func ZipfRows(rowEnergy []float64) bool {
	if len(rowEnergy) < 32 {
		return false
	}
	cp := append([]float64(nil), rowEnergy...)
	for i := 0; i < len(cp); i++ {
		best := i
		for j := i + 1; j < len(cp); j++ {
			if cp[j] > cp[best] {
				best = j
			}
		}
		cp[i], cp[best] = cp[best], cp[i]
	}
	n := len(cp)
	head := 0.0
	for i := 0; i < n/10+1; i++ {
		head += cp[i]
	}
	var tot float64
	for _, v := range cp {
		tot += math.Abs(v)
	}
	if tot <= 0 {
		return false
	}
	return head/tot >= 0.35
}

// ConvLike is a tiny leading kernel dimension (conv1d / DW conv).
func ConvLike(t Tensor) bool {
	return len(t.Shape) >= 2 && t.Shape[0] > 0 && t.Shape[0] <= 16 && t.Shape[0] < t.Shape[len(t.Shape)-1]/4
}
