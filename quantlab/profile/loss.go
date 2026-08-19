package profile

import (
	"fmt"

	"quantlab/core"
)

// EvidenceKind classifies the source of a candidate loss.
type EvidenceKind string

const (
	// EvidenceMeasured is a real measurement with full provenance (from Cache).
	EvidenceMeasured EvidenceKind = "measured"
	// EvidenceHeuristic is the deterministic fallback estimator, optionally
	// informed by imatrix aggregate statistics. Always clearly marked.
	EvidenceHeuristic EvidenceKind = "heuristic"
)

// CandidateLoss is one scored (tensor, target dtype) candidate.
type CandidateLoss struct {
	TensorName string
	Target     core.DType
	Loss       float64
	Evidence   EvidenceKind
	// Confidence in [0,1]: 1 for measured, lower for heuristic estimates.
	Confidence float64
	// Prov is set only for measured evidence.
	Prov *core.Provenance
}

func (c CandidateLoss) Validate() error {
	if c.TensorName == "" {
		return fmt.Errorf("candidate loss: empty tensor name")
	}
	if !c.Target.Valid() {
		return fmt.Errorf("candidate loss %q: invalid target %q", c.TensorName, c.Target)
	}
	if c.Loss < 0 {
		return fmt.Errorf("candidate loss %q: negative loss", c.TensorName)
	}
	if c.Confidence < 0 || c.Confidence > 1 {
		return fmt.Errorf("candidate loss %q: confidence %v out of [0,1]", c.TensorName, c.Confidence)
	}
	switch c.Evidence {
	case EvidenceMeasured:
		if c.Prov == nil {
			return fmt.Errorf("candidate loss %q: measured evidence without provenance", c.TensorName)
		}
		if err := c.Prov.Validate(); err != nil {
			return err
		}
	case EvidenceHeuristic:
	default:
		return fmt.Errorf("candidate loss %q: unknown evidence kind %q", c.TensorName, c.Evidence)
	}
	return nil
}
