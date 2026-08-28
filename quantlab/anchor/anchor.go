// Package anchor resolves structural preservation, hard anchor floors, and
// soft fidelity priors over a tensor bank.
//
// Policy:
//   - Roles are derived from GGUF names the way convert's workKind/layout
//     encode them. Name globs are optional extras, not the sole classifier.
//   - Norms, SSM conv1d, ssm_a (A_log), ssm_dt, and any tensor convert would
//     refuse to map are structurally preserved: they are never quantized.
//     Quantizable linear-attn projections (ssm_out, attn_qkv, attn_gate)
//     receive the same attention soft prior as softmax Q/K/V — convert
//     layout, not a model-family special case.
//   - Embeddings receive a Q6_K soft prior. The output head (output.weight /
//     lm_head) is a hard floor (Q6_K, easing to Q5_K/Q4_K at low target bpw)
//     so token logits — especially EOS vs continue — stay sharp. Attention
//     projections receive Q5_K; attention V gets an extra Q6_K ValuePrior.
//     FFN down gets a separate DownPrior (Q6 historically; weaker Q4/Q5
//     at low target bpw). Solvers fold priors into the loss landscape;
//     they never hard-pin those tensors.
//   - MoE routers (ffn_gate_inp) receive a hard floor. Explicit (user-pinned)
//     and calibration-selected anchors are also hard floors.
package anchor

import (
	"fmt"
	"sort"

	"quantlab/core"
)

// Policy configures automatic anchor derivation. Zero fields fall back to
// DefaultPolicy values.
type Policy struct {
	// EmbeddingPrior / AttentionPrior are the soft preferred minimum dtypes
	// for embeddings+output head and attention projections.
	EmbeddingPrior core.DType
	AttentionPrior core.DType
	// ValuePrior is an additional soft preference for attention V
	// (attn_v / v_proj / wv). It is never a hard floor. Fused qkv is not V.
	ValuePrior core.DType
	// DownPrior is a weaker soft preference for FFN down projections
	// (ffn_down / down_proj / w2). It is never a hard floor and must not
	// replay llama.cpp's "Q6 on every down" reflex at low target bpw.
	DownPrior core.DType
	// EmbeddingWeight / AttentionWeight / ValueWeight / DownWeight scale
	// the soft prior penalty. ExpertDownWeight, when set, replaces
	// DownWeight on MoE expert downs (weaker harvest).
	EmbeddingWeight  float64
	AttentionWeight  float64
	ValueWeight      float64
	DownWeight       float64
	ExpertDownWeight float64
	// RouterFloor is the hard minimum dtype for MoE routers (ffn_gate_inp).
	RouterFloor core.DType
	// OutputFloor is the hard minimum dtype for the lm_head / output.weight.
	// llama.cpp recipes keep this tensor high even at Q2; a soft prior is
	// harvested on huge vocabs and the model then stops mid-generation.
	OutputFloor core.DType
	// Pattern sets; '*' is the only wildcard (see core.Anchor.Matches).
	// AttentionPatterns default empty: softmax-attention priors are attached
	// per tensor from GGUF roles so linear-attn names are not glob-matched.
	EmbeddingPatterns []string
	AttentionPatterns []string
	ValuePatterns     []string
	DownPatterns      []string
	NormPatterns      []string
}

// DefaultPolicy returns the stock automatic-anchor policy.
func DefaultPolicy() Policy {
	return Policy{
		EmbeddingPrior:    core.DTypeQ6_K,
		AttentionPrior:    core.DTypeQ5_K_T,
		ValuePrior:        core.DTypeQ6_K,
		DownPrior:         core.DTypeQ6_K, // historical: same as ValuePrior when bpw is unset
		EmbeddingWeight:   1.0,
		AttentionWeight:   0.5,
		ValueWeight:       1.0,
		DownWeight:        1.0,
		ExpertDownWeight:  1.0,
		RouterFloor:       core.DTypeQ5_K_T,
		OutputFloor:       core.DTypeQ6_K,
		EmbeddingPatterns: []string{"*token_embd*", "*output*"},
		AttentionPatterns: nil, // GGUF-derived softmax-attention names only
		ValuePatterns:     nil, // GGUF-derived attn_v names only
		DownPatterns:      nil, // GGUF-derived ffn_down names only
		NormPatterns:      []string{"*_norm*"},
	}
}

// PolicyForBPW returns DefaultPolicy adjusted for a compression target.
// bpw <= 0 preserves historical DefaultPolicy (Q6_K on V and on every down).
func PolicyForBPW(bpw float64) Policy {
	return DefaultPolicy().ApplyTargetBPW(bpw)
}

// ApplyTargetBPW returns a copy of p with FFN-down priors scaled to bpw.
// Embeddings stay Q6_K; non-V attention stays Q5_K; V stays Q6_K.
// The output-head floor eases Q6_K → Q5_K → Q4_K at tighter budgets.
// Downs are never hard-pinned to Q6.
func (p Policy) ApplyTargetBPW(bpw float64) Policy {
	p = p.withDefaults()
	if bpw <= 0 {
		return p
	}
	switch {
	case bpw >= 4.5:
		// Comfortable Q4+ budgets: downs may still prefer Q5 softly.
		p.DownPrior = core.DTypeQ5_K_T
		p.DownWeight = 0.45
		p.ExpertDownWeight = 0.25
	case bpw <= 4.0:
		// Q3-ish (~3.5): do not put Q6_K on every ffn_down. Weak Q4 prior;
		// MoE expert downs weaker still so the body can be harvested.
		p.DownPrior = core.DTypeQ4_K_T
		p.DownWeight = 0.25
		p.ExpertDownWeight = 0.12
		if bpw <= 2.5 {
			p.OutputFloor = core.DTypeQ4_K_T
		} else {
			p.OutputFloor = core.DTypeQ5_K_T
		}
	default:
		p.DownPrior = core.DTypeQ5_K_T
		p.DownWeight = 0.35
		p.ExpertDownWeight = 0.18
	}
	return p
}

func (p Policy) withDefaults() Policy {
	d := DefaultPolicy()
	if p.EmbeddingPrior == "" {
		p.EmbeddingPrior = d.EmbeddingPrior
	}
	if p.AttentionPrior == "" {
		p.AttentionPrior = d.AttentionPrior
	}
	if p.ValuePrior == "" {
		p.ValuePrior = d.ValuePrior
	}
	if p.DownPrior == "" {
		p.DownPrior = d.DownPrior
	}
	if p.EmbeddingWeight == 0 {
		p.EmbeddingWeight = d.EmbeddingWeight
	}
	if p.AttentionWeight == 0 {
		p.AttentionWeight = d.AttentionWeight
	}
	if p.ValueWeight == 0 {
		p.ValueWeight = d.ValueWeight
	}
	if p.DownWeight == 0 {
		p.DownWeight = d.DownWeight
	}
	if p.ExpertDownWeight == 0 {
		p.ExpertDownWeight = p.DownWeight
	}
	if p.RouterFloor == "" {
		p.RouterFloor = d.RouterFloor
	}
	if p.OutputFloor == "" {
		p.OutputFloor = d.OutputFloor
	}
	if p.EmbeddingPatterns == nil {
		p.EmbeddingPatterns = d.EmbeddingPatterns
	}
	if p.AttentionPatterns == nil {
		p.AttentionPatterns = d.AttentionPatterns
	}
	if p.ValuePatterns == nil {
		p.ValuePatterns = d.ValuePatterns
	}
	if p.DownPatterns == nil {
		p.DownPatterns = d.DownPatterns
	}
	if p.NormPatterns == nil {
		p.NormPatterns = d.NormPatterns
	}
	return p
}

// Prior is a soft preference: tensors matching Pattern should prefer DType or
// better, with the violation penalty scaled by Weight. Priors never forbid an
// assignment; they only shape the loss landscape.
type Prior struct {
	Kind    core.AnchorKind `json:"kind"`
	Pattern string          `json:"pattern"`
	DType   core.DType      `json:"dtype"`
	Weight  float64         `json:"weight"`
}

// Set is the resolved anchor view over a specific tensor bank.
type Set struct {
	// Hard floors: explicit, calibration, output head, and router floors.
	Hard []core.Anchor `json:"hard,omitempty"`
	// Priors: soft preferences for embeddings/output and attention.
	Priors []Prior `json:"priors,omitempty"`
	// NormPatterns identify structurally preserved norm tensors.
	NormPatterns []string `json:"normPatterns,omitempty"`
	// PreservedNames are exact tensor names Derive pinned to float storage
	// (F32 norms/conv/ssm_a/ssm_dt, and unknown names fail-closed).
	PreservedNames []string `json:"preserved,omitempty"`
}

// matchPattern applies core.Anchor glob semantics to a bare pattern.
func matchPattern(pattern, tensorName string) bool {
	return core.Anchor{Pattern: pattern}.Matches(tensorName)
}

// Derive constructs the anchor set for a bank. Explicit, calibration, and
// layout-native router anchors in explicit become hard floors; any other
// user-supplied anchor kind is demoted to a soft prior (weight 1). Automatic
// soft priors for embeddings/output come from pol patterns; attention
// priors (softmax and linear-attn) are attached per tensor from GGUF roles.
// Norms, conv1d, ssm_a, ssm_dt, and unknown tensors are structurally preserved.
func Derive(bank *core.TensorBank, explicit []core.Anchor, pol Policy) (*Set, error) {
	if bank == nil {
		return nil, fmt.Errorf("anchor: nil bank")
	}
	pol = pol.withDefaults()
	if !pol.EmbeddingPrior.Valid() || !pol.AttentionPrior.Valid() || !pol.ValuePrior.Valid() || !pol.DownPrior.Valid() {
		return nil, fmt.Errorf("anchor: invalid prior dtype in policy")
	}
	if pol.RouterFloor != "" && (!pol.RouterFloor.Valid() || !pol.RouterFloor.IsQuant()) {
		return nil, fmt.Errorf("anchor: invalid router floor dtype %q", pol.RouterFloor)
	}
	if pol.OutputFloor != "" && (!pol.OutputFloor.Valid() || !pol.OutputFloor.IsQuant()) {
		return nil, fmt.Errorf("anchor: invalid output floor dtype %q", pol.OutputFloor)
	}
	s := &Set{NormPatterns: pol.NormPatterns}
	for _, a := range explicit {
		if err := a.Validate(); err != nil {
			return nil, err
		}
		switch a.Kind {
		case core.AnchorExplicit, core.AnchorCalibration, core.AnchorRouter:
			s.Hard = append(s.Hard, a)
		default:
			pat := a.Pattern
			if pat == "" {
				pat = a.Name
			}
			s.Priors = append(s.Priors, Prior{Kind: a.Kind, Pattern: pat, DType: a.MinDType, Weight: 1.0})
		}
	}
	for _, pat := range pol.EmbeddingPatterns {
		s.Priors = append(s.Priors, Prior{
			Kind: core.AnchorEmbedding, Pattern: pat,
			DType: pol.EmbeddingPrior, Weight: pol.EmbeddingWeight,
		})
	}
	for _, pat := range pol.AttentionPatterns {
		s.Priors = append(s.Priors, Prior{
			Kind: core.AnchorAttention, Pattern: pat,
			DType: pol.AttentionPrior, Weight: pol.AttentionWeight,
		})
	}
	for _, pat := range pol.ValuePatterns {
		s.Priors = append(s.Priors, Prior{
			Kind: core.AnchorAttention, Pattern: pat,
			DType: pol.ValuePrior, Weight: pol.ValueWeight,
		})
	}
	for _, pat := range pol.DownPatterns {
		s.Priors = append(s.Priors, Prior{
			Kind: core.AnchorAttention, Pattern: pat,
			DType: pol.DownPrior, Weight: pol.DownWeight,
		})
	}

	linear := linearAttnLayers(bank)
	var preserved []string
	for _, t := range bank.Tensors {
		role := classify(t.Name, linear)
		switch {
		case keepFloat(role):
			preserved = append(preserved, t.Name)
		case role == roleAttention, role == roleLinearAttn && t.Quantizable():
			s.Priors = append(s.Priors, Prior{
				Kind: core.AnchorAttention, Pattern: t.Name,
				DType: pol.AttentionPrior, Weight: pol.AttentionWeight,
			})
			if isAttnV(t.Name) {
				s.Priors = append(s.Priors, Prior{
					Kind: core.AnchorAttention, Pattern: t.Name,
					DType: pol.ValuePrior, Weight: pol.ValueWeight,
				})
			}
		case role == roleFFN && isFFNDown(t.Name):
			w := pol.DownWeight
			if isMoEExpert(t.Name) {
				w = pol.ExpertDownWeight
			}
			s.Priors = append(s.Priors, Prior{
				Kind: core.AnchorAttention, Pattern: t.Name,
				DType: pol.DownPrior, Weight: w,
			})
		case role == roleOutput && t.Quantizable() && pol.OutputFloor != "":
			s.Hard = append(s.Hard, core.Anchor{
				Kind: core.AnchorExplicit, Name: t.Name,
				MinDType: pol.OutputFloor, Reason: "output head",
			})
		case role == roleRouter && t.Quantizable():
			s.Hard = append(s.Hard, core.Anchor{
				Kind: core.AnchorRouter, Name: t.Name,
				MinDType: pol.RouterFloor, Reason: "moe router",
			})
		case role == roleRouter:
			preserved = append(preserved, t.Name)
		}
	}
	sort.Strings(preserved)
	s.PreservedNames = preserved
	sort.Slice(s.Hard, func(i, j int) bool {
		return s.Hard[i].Pattern+s.Hard[i].Name < s.Hard[j].Pattern+s.Hard[j].Name
	})
	sort.Slice(s.Priors, func(i, j int) bool {
		if s.Priors[i].Pattern != s.Priors[j].Pattern {
			return s.Priors[i].Pattern < s.Priors[j].Pattern
		}
		return s.Priors[i].Kind < s.Priors[j].Kind
	})
	return s, nil
}

// fidelityOrder is the exact-bpw fidelity ordering used by Rank: quant
// dtypes sorted by exact bits-per-weight descending (highest fidelity
// first), with equal-bpw ties broken deterministically by dtype name
// ascending. Recipe labels resolve through their base type's geometry, so a
// recipe label shares its base type's bpw and ranks adjacent to it.
var fidelityOrder = func() []core.DType {
	out := append([]core.DType(nil), core.QuantDTypes...)
	sort.Slice(out, func(i, j int) bool {
		bi, _ := out[i].BitsPerWeight()
		bj, _ := out[j].BitsPerWeight()
		if bi != bj {
			return bi > bj
		}
		return out[i] < out[j]
	})
	return out
}()

var fidelityRank = func() map[core.DType]int {
	m := make(map[core.DType]int, len(fidelityOrder))
	for i, d := range fidelityOrder {
		m[d] = i
	}
	return m
}()

// Rank returns the fidelity rank of d: floats rank above all quants (-1),
// otherwise the position of d in fidelityOrder (0 = highest fidelity).
//
// The order is fidelity-approximate by exact bits-per-weight: dtypes with
// identical bpw (e.g. Q4_K and IQ4_NL, or a recipe label and its base type)
// are ordered among themselves by name, and floors treat anything ranked
// after the floor dtype as lower fidelity — including an equal-bpq sibling
// that sorts later by name.
func Rank(d core.DType) int {
	if !d.IsQuant() {
		return -1
	}
	if r, ok := fidelityRank[d]; ok {
		return r
	}
	return len(fidelityOrder)
}

// Floor returns the highest minimum dtype any hard anchor requires for the
// tensor, and whether any hard anchor matched.
func (s *Set) Floor(tensorName string) (core.DType, bool) {
	best := -1
	var floor core.DType
	for _, a := range s.Hard {
		if !a.Matches(tensorName) {
			continue
		}
		idx := Rank(a.MinDType)
		if best == -1 || idx < best {
			best = idx
			floor = a.MinDType
		}
	}
	return floor, best != -1
}

// Preserved reports whether t is structurally preserved: a non-quantizable
// tensor (norm, bias, 1D) or one matching a norm pattern. Preserved tensors
// stay in their current float storage; no anchor or prior applies.
func (s *Set) Preserved(t core.TensorDesc) bool {
	if !t.Quantizable() {
		return true
	}
	for _, name := range s.PreservedNames {
		if name == t.Name {
			return true
		}
	}
	for _, pat := range s.NormPatterns {
		if matchPattern(pat, t.Name) {
			return true
		}
	}
	return false
}

// PriorLoss returns the soft penalty for storing t as target. It is zero when
// target is a float, preserved, or at-or-above every matching prior dtype;
// otherwise each violated prior contributes Weight times its normalized rank
// distance.
func (s *Set) PriorLoss(t core.TensorDesc, target core.DType) float64 {
	if !target.IsQuant() || s.Preserved(t) {
		return 0
	}
	rt := Rank(target)
	var pen float64
	for _, p := range s.Priors {
		if !matchPattern(p.Pattern, t.Name) {
			continue
		}
		rp := Rank(p.DType)
		if rt > rp {
			pen += p.Weight * float64(rt-rp) / float64(len(fidelityOrder))
		}
	}
	return pen
}

// Violation describes one hard-floor incompatibility in a profile.
type Violation struct {
	Tensor   string     `json:"tensor"`
	Required core.DType `json:"required"`
	Actual   core.DType `json:"actual"`
}

// Check verifies every assignment in profile respects the hard floors. A
// float target always satisfies any floor. Soft priors are never violations.
func (s *Set) Check(profile *core.Profile) []Violation {
	var out []Violation
	for _, qa := range profile.Assignments {
		floor, ok := s.Floor(qa.TensorName)
		if !ok || !qa.Target.IsQuant() {
			continue
		}
		if Rank(qa.Target) > Rank(floor) {
			out = append(out, Violation{Tensor: qa.TensorName, Required: floor, Actual: qa.Target})
		}
	}
	return out
}
