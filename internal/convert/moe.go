package convert

import (
	"fmt"
	"strconv"
	"strings"
)

func expertPlanShape(srcs []TensorRef, nExpert int) ([]int64, error) {
	if len(srcs) == 0 {
		return nil, fmt.Errorf("no expert tensors")
	}
	n := nExpert
	if n <= 0 {
		n = len(srcs)
	}
	if len(srcs) != n {
		return nil, fmt.Errorf("experts: got %d tensors, want %d", len(srcs), n)
	}
	seen := map[int]bool{}
	for _, t := range srcs {
		idx := expertIndexOf(t.Name)
		if idx < 0 || idx >= n || seen[idx] {
			return nil, fmt.Errorf("experts: bad index from %q", t.Name)
		}
		seen[idx] = true
		if !sameShape(t.Shape, srcs[0].Shape) {
			return nil, fmt.Errorf("expert shape mismatch on %q", t.Name)
		}
	}
	return append([]int64{int64(n)}, srcs[0].Shape...), nil
}

func stackExperts(srcs []TensorRef, nExpert int) ([]byte, []int64, error) {
	if len(srcs) == 0 {
		return nil, nil, fmt.Errorf("no expert tensors")
	}
	ordered := append([]TensorRef(nil), srcs...)
	type pair struct {
		idx int
		t   TensorRef
	}
	ps := make([]pair, 0, len(ordered))
	for _, t := range ordered {
		idx := expertIndexOf(t.Name)
		if idx < 0 {
			return nil, nil, fmt.Errorf("cannot parse expert index from %q", t.Name)
		}
		ps = append(ps, pair{idx, t})
	}
	for i := 0; i < len(ps); i++ {
		for j := i + 1; j < len(ps); j++ {
			if ps[j].idx < ps[i].idx {
				ps[i], ps[j] = ps[j], ps[i]
			}
		}
	}
	n := nExpert
	if n <= 0 {
		n = len(ps)
	}
	if len(ps) != n {
		return nil, nil, fmt.Errorf("experts: got %d tensors, want %d", len(ps), n)
	}
	shape0 := ps[0].t.Shape
	var payload []byte
	for e, p := range ps {
		if p.idx != e {
			return nil, nil, fmt.Errorf("experts: missing index %d (got %d from %q)", e, p.idx, p.t.Name)
		}
		if !sameShape(p.t.Shape, shape0) {
			return nil, nil, fmt.Errorf("expert %d shape %v != %v", e, p.t.Shape, shape0)
		}
		raw, err := ReadPayload(p.t)
		if err != nil {
			return nil, nil, err
		}
		payload = append(payload, raw...)
	}
	stacked := make([]int64, 0, len(shape0)+1)
	stacked = append(stacked, int64(n))
	stacked = append(stacked, shape0...)
	return payload, stacked, nil
}

func expertIndexOf(hfName string) int {
	n := strings.ReplaceAll(hfName, "language_model.", "")
	n = strings.TrimSuffix(n, ".weight")
	m := layerRe.FindStringSubmatch(n)
	rest := n
	if m != nil {
		rest = m[2]
	}
	em := expertRe.FindStringSubmatch(rest)
	if em == nil {
		return -1
	}
	idx, err := strconv.Atoi(em[1])
	if err != nil {
		return -1
	}
	return idx
}

func sameShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
