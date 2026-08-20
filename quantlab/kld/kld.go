// Package kld retains the search-history types persisted in old checkpoints.
// Overnight KLD interaction search is no longer part of the pipeline; StageSearch
// is a no-op that still exists so those checkpoints resume.
package kld

import (
	"fmt"

	"quantlab/core"
)

// Interaction records a measured pairwise effect: jointly changing tensors A
// and B costs more (synergy) or less (redundancy) than the sum of individual
// effects.
type Interaction struct {
	TensorA    string  `json:"tensorA"`
	TensorB    string  `json:"tensorB"`
	JointDelta float64 `json:"jointDelta"` // measured KLD when both degraded
	SumDelta   float64 `json:"sumDelta"`   // sum of individual KLD deltas
}

// Synergy is JointDelta - SumDelta; positive means the pair interacts badly.
func (i Interaction) Synergy() float64 { return i.JointDelta - i.SumDelta }

func (i Interaction) Validate() error {
	if i.TensorA == "" || i.TensorB == "" || i.TensorA == i.TensorB {
		return fmt.Errorf("kld: invalid interaction pair %q/%q", i.TensorA, i.TensorB)
	}
	return nil
}

// MeasuredRecord ties a cache key (candidate ID + context fingerprint, or a
// pair/probe key) to the measurement it produced. Steps carry these so a
// resumed run never remeasures a candidate ID already recorded in the same
// incumbent context.
type MeasuredRecord struct {
	Key string           `json:"key"`
	M   core.Measurement `json:"measurement"`
}

// Step is one recorded search move, persisted for resumability.
type Step struct {
	Iteration int           `json:"iteration"`
	ProfileID string        `json:"profileID"`
	KLD       float64       `json:"kld"`
	Moves     []Interaction `json:"moves,omitempty"`
	Accepted  bool          `json:"accepted"`
	// Kind classifies the step: "solo", "pair", "prune", "eval-error".
	Kind     string           `json:"kind,omitempty"`
	GroupIDs []string         `json:"groupIDs,omitempty"`
	Bytes    int64            `json:"bytes,omitempty"`
	Measured []MeasuredRecord `json:"measured,omitempty"`
	Baseline float64          `json:"baseline,omitempty"`
	Error    string           `json:"error,omitempty"`
}
