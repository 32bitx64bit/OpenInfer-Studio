// Package profile implements the rate-distortion profiling and allocation
// subsystem: measured candidate-loss caching with provenance, a deterministic
// heuristic fallback estimator, exact per-tensor option enumeration with
// Pareto pruning, and an architecture-agnostic multiple-choice
// rate-distortion solver that allocates per-tensor dtypes under an exact byte
// budget via marginal-gain (greedy Lagrangian) allocation.
package profile

import (
	"fmt"

	"quantlab/anchor"
	"quantlab/core"
)

// DefaultConfidencePenalty inflates losses by (1 + p*(1-confidence)) when
// evidence is weak. Applied when Request.ConfidencePenalty is zero.
const DefaultConfidencePenalty = 0.25

// Request is the solver input.
type Request struct {
	Bank    *core.TensorBank
	Anchors *anchor.Set // nil = no floors, priors, or preservation
	// Candidates is the allowed target dtype lattice. Nil = core.QuantDTypes.
	Candidates []core.DType
	// BudgetBytes is a hard exact-byte rate constraint; 0 = unconstrained
	// unless TargetBPW is set.
	BudgetBytes uint64
	// TargetBPW, when > 0 and BudgetBytes is 0, is converted to an exact
	// budget via BPWToBudget.
	TargetBPW float64
	// Cache, when non-nil, supplies measured candidate losses that override
	// the heuristic estimator.
	Cache *Cache
	// Imatrix, when non-nil, supplies per-tensor aggregate importance
	// statistics to the fallback estimator.
	Imatrix map[string]ImatrixStats
	// ConfidencePenalty scales the weak-evidence loss inflation;
	// 0 = DefaultConfidencePenalty.
	ConfidencePenalty float64
	// ProfileID, when empty, is derived deterministically from the solution.
	ProfileID string
	// ExactLoss, when non-nil, is a precomputed importance-weighted SSE
	// table (BuildExactLossTable) that overrides the analytic severity and
	// codebook terms of the estimator for covered (tensor, dtype) pairs.
	ExactLoss map[string]map[core.DType]float64
	// Calibration, when non-nil, corrects heuristic estimates toward
	// measured marginal KLD behavior (see calibration.go).
	Calibration *Calibration
	// DisableSwiGLUCoupling leaves the gate/up dtype equalization pass
	// out (multi-seed diversity).
	DisableSwiGLUCoupling bool
	// RolePriorScale scales the role prior in the estimator; 1 is the
	// default, 0 removes functional priors (multi-seed diversity).
	RolePriorScale float64
	// Diversity inverts the greedy gain ranking so the budget is spent on
	// the upgrades the main seeds deprioritized (anti-greedy seed).
	Diversity bool
}

// Validate checks the request is well-formed.
func (r Request) Validate() error {
	if r.Bank == nil {
		return fmt.Errorf("profile: nil bank")
	}
	if err := r.Bank.Validate(); err != nil {
		return err
	}
	for _, d := range r.Candidates {
		if !d.IsQuant() {
			return fmt.Errorf("profile: candidate %q is not a quant dtype", d)
		}
	}
	if r.TargetBPW < 0 {
		return fmt.Errorf("profile: negative target bpw")
	}
	if r.ConfidencePenalty < 0 {
		return fmt.Errorf("profile: negative confidence penalty")
	}
	return nil
}

// Diagnostics explains a solve: feasibility envelope, achieved loss, and the
// evidence mix behind the chosen options.
type Diagnostics struct {
	BudgetBytes uint64  `json:"budgetBytes,omitempty"`
	MinBytes    uint64  `json:"minBytes"` // cheapest legal assignment (floors honored)
	MaxBytes    uint64  `json:"maxBytes"` // highest-fidelity legal assignment
	TotalLoss   float64 `json:"totalLoss"`
	SlopBytes   uint64  `json:"slopBytes"` // budget left unused
	// MeasuredTensors / EstimatedTensors count assignments whose loss came
	// from measured evidence vs. heuristic estimation.
	MeasuredTensors  int `json:"measuredTensors"`
	EstimatedTensors int `json:"estimatedTensors"`
}

// Result is the solver output: a complete profile plus its exact-byte
// selection manifest and diagnostics.
type Result struct {
	Profile  *core.Profile
	Manifest *core.SelectionManifest
	Diag     Diagnostics
}

// InfeasibleError reports a budget below the cheapest legal assignment.
type InfeasibleError struct {
	BudgetBytes uint64
	MinBytes    uint64
	TargetBPW   float64
	// Constrained lists tensors pinned to a single option by structural
	// preservation or hard floors (the floor cost drivers).
	Constrained []string
}

func (e *InfeasibleError) Error() string {
	return fmt.Sprintf("profile: infeasible budget %d bytes (target bpw %.3f): minimum legal assignment needs %d bytes; constrained tensors: %v",
		e.BudgetBytes, e.TargetBPW, e.MinBytes, e.Constrained)
}

// BPWToBudget converts a target bits-per-weight into an exact byte budget:
// floor(bpw * totalElements / 8) over every tensor payload in the bank.
func BPWToBudget(bank *core.TensorBank, bpw float64) (uint64, error) {
	if bank == nil {
		return 0, fmt.Errorf("profile: nil bank")
	}
	if bpw <= 0 {
		return 0, fmt.Errorf("profile: non-positive target bpw")
	}
	var elems uint64
	for _, t := range bank.Tensors {
		elems += t.Elements
	}
	return uint64(bpw * float64(elems) / 8.0), nil
}
