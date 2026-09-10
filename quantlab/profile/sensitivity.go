package profile

import (
	"fmt"
	"math"
	"sort"

	"quantlab/core"
)

// RoleKey groups tensors that play the same architectural role in every
// layer: the layer-local stem of the GGUF name ("attn_q", "ffn_down_exps",
// "output", "token_embd"). Sensitivity probes quantize one role at a time,
// so one measured KLD per role is what the calibrated solver consumes.
func RoleKey(name string) string { return localStem(name) }

// RoleGroup is one probe unit: the quantizable tensors that share a role.
type RoleGroup struct {
	Role     string   `json:"role"`
	Tensors  []string `json:"tensors"`
	Elements uint64   `json:"elements"`
}

// GroupByRole partitions the bank's quantizable tensors by RoleKey, in
// order of first appearance in the bank. include, when non-nil, further
// restricts membership (callers pass the anchor set's preservation rule so
// float-pinned tensors are never probed). When maxGroups > 0 and more roles
// exist, the smallest roles (by elements) are merged into a single "other"
// group so the number of probes stays bounded on exotic architectures.
func GroupByRole(bank *core.TensorBank, maxGroups int, include func(core.TensorDesc) bool) []RoleGroup {
	if bank == nil {
		return nil
	}
	var groups []RoleGroup
	idx := map[string]int{}
	for _, t := range bank.Tensors {
		if !t.Quantizable() || (include != nil && !include(t)) {
			continue
		}
		key := RoleKey(t.Name)
		i, ok := idx[key]
		if !ok {
			i = len(groups)
			idx[key] = i
			groups = append(groups, RoleGroup{Role: key})
		}
		groups[i].Tensors = append(groups[i].Tensors, t.Name)
		groups[i].Elements += t.Elements
	}
	if maxGroups <= 0 || len(groups) <= maxGroups {
		return groups
	}
	// Keep the maxGroups-1 largest roles; merge the rest into "other".
	order := make([]int, len(groups))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return groups[order[a]].Elements > groups[order[b]].Elements
	})
	keep := map[int]bool{}
	for _, i := range order[:maxGroups-1] {
		keep[i] = true
	}
	var out []RoleGroup
	other := RoleGroup{Role: "other"}
	for i, g := range groups {
		if keep[i] {
			out = append(out, g)
			continue
		}
		other.Tensors = append(other.Tensors, g.Tensors...)
		other.Elements += g.Elements
	}
	return append(out, other)
}

// RoleSensitivity is the measured KLD cost of quantizing one whole role to
// ProbeDType while every other tensor stays near-lossless.
type RoleSensitivity struct {
	Role       string     `json:"role"`
	ProbeDType core.DType `json:"probeDType"`
	// KLD is the probe's mean KLD against the float baseline with the
	// near-lossless background KLD already subtracted (floored > 0).
	KLD float64 `json:"kld"`
	// SumWSSE is the exact-table loss summed over the role's tensors at
	// ProbeDType (audit only; the per-tensor model is rung-relative).
	SumWSSE  float64  `json:"sumWSSE,omitempty"`
	Elements uint64   `json:"elements"`
	Tensors  []string `json:"tensors,omitempty"`
}

// Sensitivity is the probe-calibrated cross-tensor loss model. It replaces
// the heuristic role priors: every role's marginal KLD at a common probe
// rung was measured on this model, so losses of different roles are
// commensurable (all in KLD units) and the solver can trade bytes between
// attention, FFN, embeddings and the output head on evidence.
type Sensitivity struct {
	// Background is the mean KLD of the near-lossless reference itself.
	Background float64                    `json:"background"`
	Roles      map[string]RoleSensitivity `json:"roles"`
	// Pinned lists roles too small to be worth a probe; the calibrated
	// solver keeps them at their highest-fidelity legal option, which is
	// what any byte-aware allocation would do with them anyway.
	Pinned []string `json:"pinned,omitempty"`
}

// Validate checks the model is usable by the solver.
func (s *Sensitivity) Validate() error {
	if s == nil {
		return fmt.Errorf("profile: nil sensitivity")
	}
	if len(s.Roles) == 0 {
		return fmt.Errorf("profile: sensitivity has no roles")
	}
	for k, r := range s.Roles {
		if r.Role != "" && r.Role != k {
			return fmt.Errorf("profile: sensitivity role key %q != %q", k, r.Role)
		}
		if !(r.KLD > 0) || math.IsInf(r.KLD, 0) {
			return fmt.Errorf("profile: sensitivity role %q: non-positive KLD %v", k, r.KLD)
		}
		if r.Elements == 0 {
			return fmt.Errorf("profile: sensitivity role %q: zero elements", k)
		}
		if !r.ProbeDType.IsQuant() {
			return fmt.Errorf("profile: sensitivity role %q: probe dtype %q is not a quant type", k, r.ProbeDType)
		}
	}
	return nil
}

// Calibrated reports whether tensor name belongs to a probed role.
func (s *Sensitivity) Calibrated(name string) bool {
	if s == nil {
		return false
	}
	_, ok := s.Roles[RoleKey(name)]
	return ok
}

// IsPinned reports whether name belongs to a role the probes skipped as too
// small to matter for the byte budget.
func (s *Sensitivity) IsPinned(name string) bool {
	if s == nil {
		return false
	}
	key := RoleKey(name)
	for _, p := range s.Pinned {
		if p == key {
			return true
		}
	}
	return false
}

// Loss returns the calibrated loss (KLD units) of storing tensor t as d,
// given the tensor's exact loss row (dtype -> importance-weighted SSE).
//
// Model (rung-relative, depth-flat): the role's measured KLD at the probe
// rung is attributed to the role's tensors in proportion to their element
// counts, and each tensor's loss at any other rung scales with its own
// exact-table error relative to the probe rung:
//
//	loss(t, d) = KLD_role * elements_t / elements_role * wSSE_t(d) / wSSE_t(probe)
//
// The per-tensor wSSE ratio carries the rung shape (how fast error grows as
// bits drop for THIS tensor, including IQ codebook effects), while the
// cross-tensor and cross-role scale comes from measurement instead of the
// raw wSSE magnitude. Raw wSSE is not commensurable across roles or depths:
// its scale follows activation power, which varies orders of magnitude
// through the residual stream, and a wSSE-proportional split was measured
// to starve late layers badly.
//
// When the exact row lacks the probe rung (a shape that cannot take it),
// the row's rung closest in bytes to the probe rung is the reference.
// ok=false when the tensor's role is uncalibrated or the row is unusable.
func (s *Sensitivity) Loss(t core.TensorDesc, d core.DType, row map[core.DType]float64) (float64, bool) {
	if s == nil || len(row) == 0 {
		return 0, false
	}
	r, ok := s.Roles[RoleKey(t.Name)]
	if !ok || r.Elements == 0 {
		return 0, false
	}
	w, ok := row[d.BaseTensorType()]
	if !ok || math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
		return 0, false
	}
	ref := referenceLoss(t, r.ProbeDType, row)
	if !(ref > 0) {
		return 0, false
	}
	share := r.KLD * float64(t.Elements) / float64(r.Elements)
	return share * w / ref, true
}

// referenceLoss is row[probe] when present and positive, else the row entry
// whose byte cost is closest to the probe rung's.
func referenceLoss(t core.TensorDesc, probe core.DType, row map[core.DType]float64) float64 {
	if v, ok := row[probe.BaseTensorType()]; ok && v > 0 {
		return v
	}
	want, ok := probe.ExactBytes(t.Elements)
	if !ok {
		return 0
	}
	best, bestDiff := 0.0, uint64(math.MaxUint64)
	var bestName core.DType
	for d, v := range row {
		if !(v > 0) {
			continue
		}
		b, ok := d.ExactBytes(t.Elements)
		if !ok {
			continue
		}
		diff := b - want
		if want > b {
			diff = want - b
		}
		if diff < bestDiff || (diff == bestDiff && d < bestName) {
			best, bestDiff, bestName = v, diff, d
		}
	}
	return best
}

// ProbeDTypeFor picks the common probe rung for a role: the first of
// preferred whose block size divides every member's contiguous dimension
// and which every member's exact row covers. ok=false when none fits.
func ProbeDTypeFor(bank *core.TensorBank, members []string, preferred []core.DType,
	exact map[string]map[core.DType]float64) (core.DType, bool) {
	if bank == nil || len(members) == 0 {
		return "", false
	}
next:
	for _, d := range preferred {
		g, ok := d.BaseTensorType().Geometry()
		if !ok {
			continue
		}
		for _, name := range members {
			t, ok := bank.Find(name)
			if !ok || len(t.Shape) == 0 || t.Shape[0]%g.BlockSize != 0 {
				continue next
			}
			if exact != nil {
				if _, ok := exact[name][d.BaseTensorType()]; !ok {
					continue next
				}
			}
		}
		return d, true
	}
	return "", false
}

// DefaultProbeDTypes is the probe-rung preference order: Q3_K sits at the
// aggressive end of typical mixed-precision budgets, so its per-role KLD is
// well above measurement noise yet still in the regime where wSSE tracks
// KLD; Q4_0 covers 32-block-only shapes.
var DefaultProbeDTypes = []core.DType{core.DTypeQ3_K, core.DTypeQ4_0}
