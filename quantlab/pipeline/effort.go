package pipeline

import (
	"fmt"
	"math"

	"quantlab/core"
	"quantlab/state"
)

// Effort selects a preset evaluation effort profile. The type is
// declared in state (RunConfig persists and validates it there); pipeline
// re-exports it as an alias so the preset API reads exactly as specified.
type Effort = state.Effort

const (
	EffortFast     = state.EffortFast
	EffortProfiled = state.EffortProfiled
	EffortDeep     = state.EffortDeep
)

// EffortProfile is the resolved knob set for one effort preset. Explicit run
// options (CLI flags, sidecar overrides) take precedence over these values.
type EffortProfile struct {
	EvalChunks    int          // KLD validation chunks
	EvalCtx       int          // evaluation context size suggestion; 0 = caller default
	AnchorRecipes []core.DType // anchor quantization levels, aggressive-first
	Candidates    []core.DType // allowed tensor-type lattice (nil = full)
	Gates         []core.QualityGate
	// ExactEstimator enables the importance-weighted exact loss table at
	// solve time (streams the source once through the qtype reference
	// quantizers).
	ExactEstimator bool
	// Reconstruct enables extra GGUF rewrites (FTI files, permute/MagR/LWC).
	// Extra/CLI only; every preset leaves this off.
	Reconstruct bool
	// ScaleFold enables AWQ-style equivalent scaling on a job-private copy.
	// Profiled/deep on; fast off. Extra.NoScaleFold opts out.
	ScaleFold bool
	// InPlaceReconstruct enables Hadamard + CSK on a job-private source
	// without a second model-sized GGUF. Profiled/deep on; fast off.
	// Permute/MagR/LWC stay off unless Reconstruct or Extra turns them on.
	InPlaceReconstruct bool
	// ProbeKLD blends a cheap Wx softmax-KLD into the exact loss table.
	// Profiled/deep on; fast off. Extra.NoProbeKLD opts out.
	ProbeKLD bool
	// SolverFTI sharpens imatrix channel weights in memory for exact-loss
	// and Solve. llama-quantize still sees the original measured matrix.
	// No extra GGUF. Profiled/deep on; fast off. Extra.NoFTI opts out.
	SolverFTI bool
}

// qualityGateKnots are (bpw, mean-KLD, p95-KLD), descending in bits-per-weight.
// Q5 (5.5 bpw) keeps the historical 0.15 / 1.0 high-precision bar. Lower
// compression targets loosen the bar so a typical Q3 is judged as a Q3,
// not as a failed Q5. Sensitivity-probe dumps (~3+ mean KLD at 3.5 bpw)
// still fail.
var qualityGateKnots = [][3]float64{
	{8.0, 0.05, 0.40},
	{6.0, 0.08, 0.60},
	{5.5, 0.15, 1.00},
	{4.5, 0.40, 3.00},
	{3.5, 2.00, 16.0},
	{2.5, 4.00, 32.0},
	{1.5, 8.00, 64.0},
}

// QualityGateThresholds returns the default mean-KLD and p95-KLD acceptance
// caps for a target bits-per-weight. A non-positive bpw keeps the Q5
// reference (0.15 / 1.0).
func QualityGateThresholds(bpw float64) (meanKLD, p95KLD float64) {
	if bpw <= 0 {
		return 0.15, 1.0
	}
	knots := qualityGateKnots
	if bpw >= knots[0][0] {
		return knots[0][1], knots[0][2]
	}
	last := knots[len(knots)-1]
	if bpw <= last[0] {
		return last[1], last[2]
	}
	for i := 0; i < len(knots)-1; i++ {
		hi, lo := knots[i], knots[i+1]
		if bpw <= hi[0] && bpw >= lo[0] {
			t := (hi[0] - bpw) / (hi[0] - lo[0])
			return hi[1] + t*(lo[1]-hi[1]), hi[2] + t*(lo[2]-hi[2])
		}
	}
	return 0.15, 1.0
}

// GatesForTargetBPW is the default acceptance set at a compression target.
func GatesForTargetBPW(bpw float64) []core.QualityGate {
	mean, p95 := QualityGateThresholds(bpw)
	return []core.QualityGate{
		{Metric: core.MetricKLD, MaxDelta: mean},
		{Metric: core.MetricP95KLD, MaxAbsolute: p95, MaxDelta: math.MaxFloat64},
	}
}

// defaultEffortGates is the high-precision reference (Q5 / unset target):
// mean-KLD <= 0.15 and an absolute p95-KLD cap of 1.0. resolvedGates scales
// this set to Config.TargetBPW so a Q3 job is not failed for not being Q5.
func defaultEffortGates() []core.QualityGate {
	return GatesForTargetBPW(0)
}

// worstDomainGateDelta is the holdout-domain cap: 4/3 of the mean-KLD gate,
// matching the historical 0.15 → 0.20 slack on small domain corpora.
func worstDomainGateDelta(meanKLD float64) float64 {
	if meanKLD <= 0 {
		return 0.20
	}
	return meanKLD * 4.0 / 3.0
}

// EffortFor resolves an effort preset to its profile. Empty resolves to
// profiled; anything else unknown is an error.
func EffortFor(e Effort) (EffortProfile, error) {
	switch e {
	case EffortFast:
		return EffortProfile{
			EvalChunks:    2,
			EvalCtx:       512,
			AnchorRecipes: []core.DType{core.DTypeQ3_K_M, core.DTypeQ4_K_M},
			Candidates: []core.DType{
				core.DTypeQ8_0, core.DTypeQ6_K, core.DTypeQ5_K_T, core.DTypeQ5_1,
				core.DTypeIQ4_NL, core.DTypeQ4_K_T, core.DTypeQ4_1, core.DTypeIQ4_XS,
				core.DTypeQ4_0, core.DTypeIQ3_S, core.DTypeQ3_K, core.DTypeQ2_K,
				core.DTypeIQ2_XS, core.DTypeIQ2_XXS,
			},
			Gates:          defaultEffortGates(),
			ExactEstimator: false,
		}, nil
	case "", EffortProfiled:
		return EffortProfile{
			EvalChunks:         4,
			EvalCtx:            2048,
			AnchorRecipes:      []core.DType{core.DTypeQ3_K_M, core.DTypeQ3_K_L, core.DTypeQ4_K_M},
			Gates:              defaultEffortGates(),
			ExactEstimator:     true,
			ScaleFold:          true,
			InPlaceReconstruct: true,
			ProbeKLD:           true,
			SolverFTI:          true,
		}, nil
	case EffortDeep:
		return EffortProfile{
			EvalChunks: 8,
			EvalCtx:    4096,
			// Q4_K_L has no core constant; the recipe list is preset data
			// (consumed by the app adapter), so it is spelled inline rather
			// than extending core.
			AnchorRecipes:      []core.DType{core.DTypeQ3_K_M, core.DTypeQ3_K_L, core.DTypeQ4_K_M, core.DType("Q4_K_L")},
			Gates:              defaultEffortGates(),
			ExactEstimator:     true,
			ScaleFold:          true,
			InPlaceReconstruct: true,
			ProbeKLD:           true,
			SolverFTI:          true,
		}, nil
	}
	return EffortProfile{}, fmt.Errorf("pipeline: unknown effort %q (want fast, profiled, or deep)", e)
}

// effortProfile resolves the run's configured effort (empty = profiled). An
// unresolvable stored value (a checkpoint written by a newer schema) falls
// back to profiled rather than failing the run.
func (e *Engine) effortProfile() EffortProfile {
	p, err := EffortFor(Effort(e.Run.Config.Effort))
	if err != nil {
		p, _ = EffortFor(EffortProfiled)
	}
	return p
}

// resolvedGates is the effective acceptance set: an explicit opt-out yields
// none, configured gates win, and an empty Gates list falls back to
// TargetBPW-scaled defaults (fail-closed rather than trivially passing).
func (e *Engine) resolvedGates() []core.QualityGate {
	cfg := e.Run.Config
	if cfg.GatesOptOut {
		return nil
	}
	if len(cfg.Gates) > 0 {
		return cfg.Gates
	}
	gates := GatesForTargetBPW(cfg.TargetBPW)
	// Defaults are auto-raised when the calibration dir carries per-domain
	// evaluation corpora: any group whose KLD regresses beyond 4/3 of the
	// mean gate (small corpora are noisier) fails the gate.
	domains, domainErr := e.evalableDomainEvalPaths()
	if domainErr != nil || len(domains) >= 2 {
		mean := 0.15
		if len(gates) > 0 && gates[0].MaxDelta > 0 {
			mean = gates[0].MaxDelta
		}
		gates = append(append([]core.QualityGate(nil), gates...),
			core.QualityGate{Metric: core.MetricWorstDomainKLD, MaxDelta: worstDomainGateDelta(mean)})
	}
	return gates
}
