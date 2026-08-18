package core

import (
	"fmt"
	"strings"
)

// AnchorKind classifies why a tensor is pinned to a minimum fidelity.
type AnchorKind string

const (
	AnchorEmbedding   AnchorKind = "embedding"   // token embeddings / output head
	AnchorNorm        AnchorKind = "norm"        // layernorm / rmsnorm weights
	AnchorAttention   AnchorKind = "attention"   // attention projections
	AnchorRouter      AnchorKind = "router"      // MoE router (ffn_gate_inp)
	AnchorExplicit    AnchorKind = "explicit"    // user-pinned by name/pattern
	AnchorCalibration AnchorKind = "calibration" // selected by calibration data
)

// Anchor pins a tensor (by exact name or glob pattern) to at least MinDType.
type Anchor struct {
	Kind     AnchorKind `json:"kind"`
	Name     string     `json:"name"`    // exact tensor name
	Pattern  string     `json:"pattern"` // optional glob; Name wins if both set
	MinDType DType      `json:"minDType"`
	Reason   string     `json:"reason"`
}

func (a Anchor) Validate() error {
	if a.Name == "" && a.Pattern == "" {
		return fmt.Errorf("anchor: name or pattern required")
	}
	switch a.Kind {
	case AnchorEmbedding, AnchorNorm, AnchorAttention, AnchorRouter, AnchorExplicit, AnchorCalibration:
	default:
		return fmt.Errorf("anchor %q: unknown kind %q", a.Name, a.Kind)
	}
	if !a.MinDType.Valid() {
		return fmt.Errorf("anchor %q: invalid min dtype %q", a.Name, a.MinDType)
	}
	return nil
}

// Matches reports whether the anchor applies to the tensor name. Patterns use
// '*' as the only wildcard (suffix/prefix glob), avoiding filepath semantics.
func (a Anchor) Matches(tensorName string) bool {
	if a.Name != "" {
		return a.Name == tensorName
	}
	p := a.Pattern
	if p == "*" {
		return true
	}
	if strings.HasPrefix(p, "*") && strings.HasSuffix(p, "*") && len(p) > 1 {
		return strings.Contains(tensorName, strings.Trim(p, "*"))
	}
	if strings.HasPrefix(p, "*") {
		return strings.HasSuffix(tensorName, p[1:])
	}
	if strings.HasSuffix(p, "*") {
		return strings.HasPrefix(tensorName, p[:len(p)-1])
	}
	return tensorName == p
}

// TensorOption is one candidate quantization choice for one tensor, with its
// exact on-disk byte cost derived from block geometry. Solvers pick exactly
// one option per tensor.
type TensorOption struct {
	TensorName string `json:"tensorName"`
	Target     DType  `json:"target"`
	// Bytes is the exact quantized payload size (geometry-derived).
	Bytes uint64 `json:"bytes"`
	// PriorLoss is an optional predicted distortion score (higher = worse);
	// 0 means "no prior".
	PriorLoss float64 `json:"priorLoss,omitempty"`
}

func (o TensorOption) Validate() error {
	if err := ValidateTensorName(o.TensorName); err != nil {
		return fmt.Errorf("tensor option: %w", err)
	}
	if !o.Target.Valid() {
		return fmt.Errorf("tensor option %q: invalid target %q", o.TensorName, o.Target)
	}
	if o.Bytes == 0 {
		return fmt.Errorf("tensor option %q: zero byte cost", o.TensorName)
	}
	if o.PriorLoss < 0 {
		return fmt.Errorf("tensor option %q: negative prior loss", o.TensorName)
	}
	return nil
}

// OptionsFor enumerates candidate options for t across targets, computing
// exact byte costs. Targets with unknown geometry or incompatible block
// alignment are skipped. Non-quantizable tensors yield a single float option
// (their current dtype).
//
// NOTE: profile.EnumerateOptions is the solver-grade counterpart of this
// function (loss scoring, anchors, floors, Pareto pruning). The two are kept
// in sync deliberately: EnumerateOptions delegates option geometry to the
// same core.DType.ExactBytes/block-alignment rules used here. Change one,
// change the other.
func OptionsFor(t TensorDesc, targets []DType) ([]TensorOption, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	if !t.Quantizable() {
		b, ok := t.DType.ExactBytes(t.Elements)
		if !ok {
			return nil, fmt.Errorf("tensor %q: no geometry for %q", t.Name, t.DType)
		}
		return []TensorOption{{TensorName: t.Name, Target: t.DType, Bytes: b}}, nil
	}
	out := make([]TensorOption, 0, len(targets))
	for _, d := range targets {
		if !d.IsQuant() {
			continue
		}
		g, ok := d.BaseTensorType().Geometry()
		if !ok {
			continue
		}
		if t.Shape[0]%g.BlockSize != 0 {
			continue
		}
		b, _ := d.ExactBytes(t.Elements)
		out = append(out, TensorOption{TensorName: t.Name, Target: d, Bytes: b})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("tensor %q: no compatible quant target", t.Name)
	}
	return out, nil
}
