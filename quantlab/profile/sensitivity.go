package profile

import (
	"fmt"
	"math"
	"sort"
	"strings"

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
	// ProbeDType2 / KLD2 / SumWSSE2 record an optional second probe rung
	// near the compression target (background-corrected KLD, exact-table
	// wSSE). Exponent is the fitted b of KLD ∝ wSSE^b through both probes;
	// zero means the single-rung linear model (b = 1).
	ProbeDType2 core.DType `json:"probeDType2,omitempty"`
	KLD2        float64    `json:"kld2,omitempty"`
	SumWSSE2    float64    `json:"sumWSSE2,omitempty"`
	Exponent    float64    `json:"exponent,omitempty"`
}

// Rung exponent guard. b = 1 is the small-perturbation regime (KLD
// quadratic in output error, wSSE quadratic in weight error); measured
// fits outside [0.5, 2] are treated as probe noise and clamped.
const (
	MinRungExponent = 0.5
	MaxRungExponent = 2.0
	// minRungSpread is the smallest wSSE ratio between the two probe rungs
	// that supports an exponent fit; closer rungs amplify probe noise.
	minRungSpread = 1.5
)

// FitRungExponent fits b in KLD ∝ wSSE^b through two role-level probe
// points (k = background-corrected KLD, w = summed exact-table wSSE at the
// probe rung). ok=false when either point is unusable or the rungs are too
// close in wSSE for a stable fit; callers keep the linear model.
func FitRungExponent(k1, w1, k2, w2 float64) (float64, bool) {
	for _, v := range []float64{k1, w1, k2, w2} {
		if !(v > 0) || math.IsInf(v, 0) || math.IsNaN(v) {
			return 0, false
		}
	}
	lw := math.Log(w2 / w1)
	if math.Abs(lw) < math.Log(minRungSpread) {
		return 0, false
	}
	b := math.Log(k2/k1) / lw
	if math.IsNaN(b) || math.IsInf(b, 0) {
		return 0, false
	}
	if b < MinRungExponent {
		b = MinRungExponent
	}
	if b > MaxRungExponent {
		b = MaxRungExponent
	}
	return b, true
}

// DepthBucket records the measured vs predicted KLD ratio for one
// contiguous layer range of one tensor family.
type DepthBucket struct {
	// Family is the depth family the bucket was probed for ("mix" or
	// "ffn"); empty for a combined probe covering every family.
	Family       string  `json:"family,omitempty"`
	First        int     `json:"first"`
	Last         int     `json:"last"`
	MeasuredKLD  float64 `json:"measuredKLD"`
	PredictedKLD float64 `json:"predictedKLD"`
	// Factor is the bucket's knot value ρ (measured / predicted, guarded).
	Factor float64 `json:"factor"`
	// Clamped reports that Factor hit the [DepthFactorMin, DepthFactorMax]
	// guard, i.e. the measurement asked for more than the model allows.
	Clamped bool `json:"clamped,omitempty"`
}

// DepthModel redistributes each role's probe-measured KLD across layers
// using a few depth-bucket probes. Per family, each bucket's factor ρ =
// measured / depth-flat prediction; single-layer edge buckets apply to their
// own layer, and interior layers interpolate log-linearly between the
// multi-layer bucket centers (adjacent layers never jump at a bucket edge,
// and a uniform interior stays uniform). Each role's total is then
// renormalized so it still equals its measured KLD at the probe rung.
type DepthModel struct {
	Buckets []DepthBucket `json:"buckets,omitempty"`
	// Shares maps tensor name → per-weight share_t (KLD × ρ / Z).
	Shares map[string]float64 `json:"shares,omitempty"`
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
	// Depth, when present, redistributes each role's KLD across layers
	// using measured depth-bucket probes.
	Depth *DepthModel `json:"depth,omitempty"`

	// members maps tensor name → role key for roles whose key is not the
	// tensor's own RoleKey (GroupByRole's merged "other" group). Built
	// lazily from Roles[*].Tensors.
	members map[string]string
}

// RoleOf returns the probed role a tensor belongs to: its RoleKey when
// probed, else the merged role that lists it among its tensors.
func (s *Sensitivity) RoleOf(name string) (string, bool) {
	if s == nil {
		return "", false
	}
	if key := RoleKey(name); s.Roles[key].Elements > 0 {
		return key, true
	}
	if s.members == nil {
		s.members = roleMembers(s.Roles)
	}
	key, ok := s.members[name]
	return key, ok
}

// roleMembers indexes tensors listed under a role key other than their own
// RoleKey (merged probe groups).
func roleMembers(roles map[string]RoleSensitivity) map[string]string {
	out := map[string]string{}
	for key, r := range roles {
		for _, name := range r.Tensors {
			if RoleKey(name) != key {
				out[name] = key
			}
		}
	}
	return out
}

// PinnedRate is the conservative KLD-per-wSSE rate that prices an unprobed
// tensor: the largest KLD/SumWSSE at the probe rung among probed roles of
// the tensor's depth family (every probed role for tensors outside the
// layer stack). Taking the family's most sensitive rate keeps a guess from
// harvesting a tensor no probe looked at, while still letting a tensor
// whose own error is small drop below top fidelity. ok=false when no role
// carries a usable rate.
func (s *Sensitivity) PinnedRate(name string) (float64, bool) {
	if s == nil {
		return 0, false
	}
	fam := ""
	if LayerIndex(name) >= 0 {
		fam = DepthFamily(name)
	}
	best := 0.0
	for _, r := range s.Roles {
		if !(r.SumWSSE > 0) || !(r.KLD > 0) {
			continue
		}
		// Layer tensors compare against layer roles of their family only:
		// the output head and embeddings sit outside the layer stack and
		// their logit-facing rates are not a family's.
		if fam != "" && len(r.Tensors) > 0 &&
			(LayerIndex(r.Tensors[0]) < 0 || DepthFamily(r.Tensors[0]) != fam) {
			continue
		}
		if k := r.KLD / r.SumWSSE; k > best && !math.IsInf(k, 0) {
			best = k
		}
	}
	if best == 0 && fam != "" {
		// No probed role in the family: fall back to every role.
		for _, r := range s.Roles {
			if r.SumWSSE > 0 && r.KLD > 0 {
				if k := r.KLD / r.SumWSSE; k > best && !math.IsInf(k, 0) {
					best = k
				}
			}
		}
	}
	return best, best > 0
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
		if r.Exponent != 0 && (r.Exponent < MinRungExponent || r.Exponent > MaxRungExponent || math.IsNaN(r.Exponent)) {
			return fmt.Errorf("profile: sensitivity role %q: exponent %v outside [%v,%v]", k, r.Exponent, MinRungExponent, MaxRungExponent)
		}
	}
	if s.Depth != nil {
		for name, v := range s.Depth.Shares {
			if !(v > 0) || math.IsInf(v, 0) || math.IsNaN(v) {
				return fmt.Errorf("profile: depth share %q = %v is not finite positive", name, v)
			}
		}
	}
	return nil
}

// Calibrated reports whether tensor name belongs to a probed role.
func (s *Sensitivity) Calibrated(name string) bool {
	if s == nil {
		return false
	}
	_, ok := s.RoleOf(name)
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
//	loss(t, d) = KLD_role * elements_t / elements_role * (wSSE_t(d) / wSSE_t(probe))^b
//
// b is the role's fitted rung exponent (1 without a second probe rung): it
// bends the rung curve so it passes through the second measurement near the
// compression target instead of extrapolating linearly from the probe.
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
	key, ok := s.RoleOf(t.Name)
	if !ok {
		return 0, false
	}
	r := s.Roles[key]
	if r.Elements == 0 {
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
	if s.Depth != nil {
		if v, ok := s.Depth.Shares[t.Name]; ok {
			share = v
		}
	}
	ratio := w / ref
	if r.Exponent > 0 && r.Exponent != 1 {
		ratio = math.Pow(ratio, r.Exponent)
	}
	return share * ratio, true
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

// SecondProbeDTypes is the second-rung preference order for a compression
// target (bits per weight): a rung on the far side of the operating point
// from the Q3_K probe, so the fitted exponent describes the curve where the
// solver actually allocates. Each list ends in 32-block rungs for shapes
// that cannot take 256-element super-blocks. Nil for an unset target.
func SecondProbeDTypes(targetBPW float64) []core.DType {
	switch {
	case targetBPW <= 0:
		return nil
	case targetBPW < 3.0:
		return []core.DType{core.DTypeIQ2_XS, core.DTypeQ2_K}
	case targetBPW < 4.5:
		return []core.DType{core.DTypeQ4_K_T, core.DTypeQ5_0}
	case targetBPW < 6.0:
		return []core.DType{core.DTypeQ5_K_T, core.DTypeQ5_1}
	default:
		return []core.DType{core.DTypeQ6_K, core.DTypeQ8_0}
	}
}

// DefaultProbeDTypes is the probe-rung preference order: Q3_K sits at the
// aggressive end of typical mixed-precision budgets, so its per-role KLD is
// well above measurement noise yet still in the regime where wSSE tracks
// KLD; Q4_0 covers 32-block-only shapes.
var DefaultProbeDTypes = []core.DType{core.DTypeQ3_K, core.DTypeQ4_0}

// Depth families. Token mixers (softmax/linear attention, SSM) and channel
// mixers (dense FFN, MoE experts) are probed separately: their sensitivity
// profiles through depth differ, so one shared factor per bucket misprices
// one family to fit the other.
const (
	DepthFamilyMix = "mix"
	DepthFamilyFFN = "ffn"
)

// DepthFamilies lists the families in probe order.
var DepthFamilies = []string{DepthFamilyMix, DepthFamilyFFN}

// DepthFamily classifies a layered tensor into its depth family.
func DepthFamily(name string) string {
	stem := localStem(name)
	n := strings.ToLower(name)
	if isFFNDown(stem) || isFFNUp(stem) || isFFNGate(stem) || isMoEExpert(name) ||
		strings.Contains(n, "ffn") || strings.Contains(n, "mlp") {
		return DepthFamilyFFN
	}
	return DepthFamilyMix
}

// DepthKey names one depth probe: "depth-<first>-<last>" for a combined
// probe (family ""), "depth-<family>-<first>-<last>" otherwise.
func DepthKey(family string, b [2]int) string {
	if family == "" {
		return fmt.Sprintf("depth-%d-%d", b[0], b[1])
	}
	return fmt.Sprintf("depth-%s-%d-%d", family, b[0], b[1])
}

// Depth factor guard. Single-layer edge buckets legitimately measure several
// times the depth-flat prediction; the guard only stops probe noise from
// zeroing or exploding a bucket.
const (
	DepthFactorMin = 0.125
	DepthFactorMax = 8.0
)

// DepthBuckets computes the layer-bucket edges for depth probes: {0},
// four contiguous groups of layers 1..n-2, and {n-1}. Returns nil when
// n < 8 (too few layers for meaningful buckets).
func DepthBuckets(n int) [][2]int {
	if n < 8 {
		return nil
	}
	mid := n - 2 // layers 1..n-2
	base := mid / 4
	extra := mid % 4
	buckets := [][2]int{{0, 0}}
	start := 1
	for i := 0; i < 4; i++ {
		sz := base
		if i < extra {
			sz++
		}
		buckets = append(buckets, [2]int{start, start + sz - 1})
		start += sz
	}
	buckets = append(buckets, [2]int{n - 1, n - 1})
	return buckets
}

// depthKnot is one fitted bucket of a family profile.
type depthKnot struct {
	bucket int     // index into buckets
	first  int     // bucket's first layer
	last   int     // bucket's last layer
	center float64 // (first+last)/2
	logRho float64
}

// point reports a single-layer bucket: its factor applies to that layer
// only and never bleeds into neighbours (the edge layers 0 and n-1
// routinely measure several times the flat prediction).
func (k depthKnot) point() bool { return k.first == k.last }

// interpolateKnots evaluates a family profile at layer l: a single-layer
// knot's own factor on its layer; otherwise log-linear between the centers
// of the multi-layer knots, constant beyond the outermost ones. Point knots
// never take part in the interpolation, so a large edge factor cannot leak
// into the interior, and monotone interior knots give a monotone profile.
func interpolateKnots(knots []depthKnot, l int) float64 {
	var span []depthKnot
	for _, k := range knots {
		if k.point() {
			if k.first == l {
				return math.Exp(k.logRho)
			}
			continue
		}
		span = append(span, k)
	}
	if len(span) == 0 {
		return 1
	}
	x := float64(l)
	if x <= span[0].center {
		return math.Exp(span[0].logRho)
	}
	last := span[len(span)-1]
	if x >= last.center {
		return math.Exp(last.logRho)
	}
	for i := 0; i+1 < len(span); i++ {
		a, b := span[i], span[i+1]
		if x >= a.center && x <= b.center {
			t := (x - a.center) / (b.center - a.center)
			return math.Exp(a.logRho + t*(b.logRho-a.logRho))
		}
	}
	return math.Exp(last.logRho)
}

func clampDepthFactor(f float64) (float64, bool) {
	switch {
	case !(f > 0) || math.IsNaN(f):
		return DepthFactorMin, true
	case f < DepthFactorMin:
		return DepthFactorMin, true
	case f > DepthFactorMax:
		return DepthFactorMax, true
	}
	return f, false
}

// ComputeDepthModel builds the depth model from per-bucket measured KLD
// results and the depth-flat prediction. buckets are the layer edges;
// measured maps DepthKey(family, bucket) → background-corrected KLD. A
// family without its own probe for a bucket falls back to the combined
// probe DepthKey("", bucket) when present. The depth-flat prediction per
// tensor is KLD_role × elements_t / elements_role.
func ComputeDepthModel(bank *core.TensorBank, buckets [][2]int,
	sens map[string]RoleSensitivity, measured map[string]float64,
	background float64) *DepthModel {
	_ = background // measured values arrive background-corrected
	if len(buckets) == 0 || len(measured) == 0 {
		return nil
	}
	type tinfo struct {
		name   string
		elems  uint64
		role   string
		family string
		layer  int
		pred   float64
	}
	var tensors []tinfo
	merged := roleMembers(sens)
	for _, t := range bank.Tensors {
		if !t.Quantizable() {
			continue
		}
		role := RoleKey(t.Name)
		r, ok := sens[role]
		if !ok {
			if role, ok = merged[t.Name]; ok {
				r = sens[role]
			}
		}
		if !ok || r.Elements == 0 {
			continue
		}
		tensors = append(tensors, tinfo{
			name: t.Name, elems: t.Elements, role: role,
			family: DepthFamily(t.Name), layer: LayerIndex(t.Name),
			pred: r.KLD * float64(t.Elements) / float64(r.Elements),
		})
	}
	inBucket := func(l int, b [2]int) bool { return l >= b[0] && l <= b[1] }

	dm := &DepthModel{Shares: map[string]float64{}}
	profiles := map[string][]depthKnot{}
	for _, fam := range DepthFamilies {
		// Members of this family per bucket, and the probe that measured
		// them: a family probe, else the combined probe.
		var knots []depthKnot
		measuredOf := map[int]float64{}
		for bi, b := range buckets {
			mb, ok := measured[DepthKey(fam, b)]
			combined := false
			if !ok {
				mb, ok = measured[DepthKey("", b)]
				combined = ok
			}
			if !ok || !(mb > 0) {
				continue
			}
			var pb float64
			for _, ti := range tensors {
				if inBucket(ti.layer, b) && (combined || ti.family == fam) {
					pb += ti.pred
				}
			}
			if !(pb > 0) {
				continue
			}
			// A combined probe measures every family at once: attribute
			// it to this family in proportion to the family's predicted
			// share of the bucket.
			if combined {
				var fp float64
				for _, ti := range tensors {
					if inBucket(ti.layer, b) && ti.family == fam {
						fp += ti.pred
					}
				}
				if !(fp > 0) {
					continue
				}
				mb *= fp / pb
				pb = fp
			}
			f, _ := clampDepthFactor(mb / pb)
			knots = append(knots, depthKnot{
				bucket: bi, first: b[0], last: b[1],
				center: float64(b[0]+b[1]) / 2, logRho: math.Log(f),
			})
			measuredOf[bi] = mb
		}
		if len(knots) == 0 {
			continue
		}
		profiles[fam] = knots
		for _, k := range knots {
			b := buckets[k.bucket]
			var p float64
			for _, ti := range tensors {
				if ti.family == fam && inBucket(ti.layer, b) {
					p += ti.pred
				}
			}
			// Snap knots the guard bound (exp∘log round-off) onto the bound.
			f, clamped := math.Exp(k.logRho), false
			switch {
			case f <= DepthFactorMin*(1+1e-9):
				f, clamped = DepthFactorMin, true
			case f >= DepthFactorMax*(1-1e-9):
				f, clamped = DepthFactorMax, true
			}
			dm.Buckets = append(dm.Buckets, DepthBucket{
				Family: fam, First: b[0], Last: b[1],
				MeasuredKLD: measuredOf[k.bucket], PredictedKLD: p, Factor: f, Clamped: clamped,
			})
		}
	}
	if len(dm.Buckets) == 0 {
		return nil
	}
	// share_t = KLD_r · E_t · ρ_{family(t)}(l_t) / Z_r.
	rho := func(ti tinfo) float64 {
		if ti.layer < 0 {
			return 1
		}
		knots, ok := profiles[ti.family]
		if !ok {
			return 1
		}
		return interpolateKnots(knots, ti.layer)
	}
	zr := map[string]float64{}
	for _, ti := range tensors {
		zr[ti.role] += float64(ti.elems) * rho(ti)
	}
	for _, ti := range tensors {
		r := sens[ti.role]
		z := zr[ti.role]
		if !(z > 0) {
			z = 1
		}
		dm.Shares[ti.name] = r.KLD * float64(ti.elems) * rho(ti) / z
	}
	return dm
}

// Scaled returns a copy of the model whose per-role KLD (both probe rungs)
// and depth shares are multiplied by lambda[role]; roles absent from lambda
// take def. Refinement uses it to fold in-context corrections (measured vs
// predicted marginal KLD inside the real mix) into the solver's objective.
func (s *Sensitivity) Scaled(lambda map[string]float64, def float64) *Sensitivity {
	if s == nil {
		return nil
	}
	scale := func(role string) float64 {
		if v, ok := lambda[role]; ok && v > 0 && !math.IsInf(v, 0) {
			return v
		}
		return def
	}
	out := &Sensitivity{Background: s.Background, Roles: make(map[string]RoleSensitivity, len(s.Roles))}
	out.Pinned = append(out.Pinned, s.Pinned...)
	for k, r := range s.Roles {
		f := scale(k)
		r.KLD *= f
		r.KLD2 *= f
		r.Tensors = append([]string(nil), r.Tensors...)
		out.Roles[k] = r
	}
	if s.Depth != nil {
		d := &DepthModel{Buckets: append([]DepthBucket(nil), s.Depth.Buckets...), Shares: make(map[string]float64, len(s.Depth.Shares))}
		for name, v := range s.Depth.Shares {
			role, ok := s.RoleOf(name)
			if !ok {
				d.Shares[name] = v
				continue
			}
			d.Shares[name] = v * scale(role)
		}
		out.Depth = d
	}
	return out
}
