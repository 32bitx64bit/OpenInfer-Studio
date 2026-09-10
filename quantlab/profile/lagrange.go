package profile

import (
	"fmt"
	"math"
	"sort"

	"quantlab/anchor"
	"quantlab/core"
)

// solveCalibrated is the probe-calibrated allocation path of Solve.
//
// Every quantizable tensor whose role was probed gets losses in KLD units
// from Sensitivity.Loss (measured role KLD, split across the role's tensors
// and shaped per tensor by its exact-table rung ratios). Roles too small to
// probe are pinned to their highest-fidelity legal rung. Soft priors are not
// applied and gate/up coupling is off: both encode guesses about relative
// sensitivity that the probes replace with measurement. Policy hard floors
// are likewise dropped for probed roles (an output head that measured as
// insensitive should not be forced to Q6_K) and kept for unprobed ones.
//
// Allocation is an exact Lagrangian sweep over each tensor's lower convex
// frontier: hull edges are applied in descending loss-decrease-per-byte
// order until the budget binds, which is the optimum of the relaxed problem
// at that multiplier; leftover bytes are then filled greedily by the best
// remaining single-rung upgrades and polished by the bounded 2-opt pass.
func solveCalibrated(req Request, set *anchor.Set, est *FallbackEstimator, cands []core.DType, budget uint64) (*Result, error) {
	sens := req.Sensitivity
	if err := sens.Validate(); err != nil {
		return nil, err
	}
	if len(req.ExactLoss) == 0 {
		return nil, fmt.Errorf("profile: calibrated solve requires an exact loss table")
	}
	set = calibratedAnchors(set, req.Bank, sens)

	states := make([]solverState, len(req.Bank.Tensors))
	var minTotal, maxTotal uint64
	var constrained []string
	for i, t := range req.Bank.Tensors {
		opts, err := enumerateCalibrated(t, cands, set, est, sens, req.ExactLoss[t.Name])
		if err != nil {
			return nil, err
		}
		states[i] = solverState{opts: opts}
		minTotal += opts[0].Bytes
		maxTotal += opts[len(opts)-1].Bytes
		if len(opts) == 1 {
			constrained = append(constrained, t.Name)
		}
	}
	effective := budget
	if effective == 0 {
		effective = maxTotal
	}
	if effective < minTotal {
		return nil, &InfeasibleError{
			BudgetBytes: budget, MinBytes: minTotal, TargetBPW: req.TargetBPW,
			Constrained: constrained,
		}
	}
	lagrangianAllocate(states, req.Bank, effective)
	reallocate2Opt(states, req.Bank, effective)
	return assembleResult(req, set, states, budget, effective, minTotal, maxTotal)
}

// calibratedAnchors returns set without the hard floors that match any
// tensor of a probed role; the profile records the reduced set so
// Set.Check agrees with what the solver enforced.
func calibratedAnchors(set *anchor.Set, bank *core.TensorBank, sens *Sensitivity) *anchor.Set {
	out := *set
	out.Hard = nil
	for _, a := range set.Hard {
		keep := true
		for _, t := range bank.Tensors {
			if sens.Calibrated(t.Name) && a.Matches(t.Name) {
				keep = false
				break
			}
		}
		if keep {
			out.Hard = append(out.Hard, a)
		}
	}
	return &out
}

// enumerateCalibrated is EnumerateOptions for the calibrated objective:
// same legality rules (geometry, block alignment, imatrix requirement,
// structural preservation), losses from Sensitivity.Loss, floors only for
// unprobed roles, and unprobed quantizable tensors pinned to one option.
func enumerateCalibrated(t core.TensorDesc, cands []core.DType, set *anchor.Set,
	est *FallbackEstimator, sens *Sensitivity, row map[core.DType]float64) ([]ScoredOption, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	if set.Preserved(t) {
		b, ok := t.DType.ExactBytes(t.Elements)
		if !ok {
			return nil, fmt.Errorf("profile: tensor %q: no geometry for %q", t.Name, t.DType)
		}
		return []ScoredOption{{
			TensorOption: core.TensorOption{TensorName: t.Name, Target: t.DType, Bytes: b},
			Evidence:     EvidenceHeuristic, Confidence: 1,
		}}, nil
	}
	floor, hasFloor := set.Floor(t.Name)
	var legal []core.TensorOption
	for _, d := range cands {
		if !d.IsQuant() {
			continue
		}
		g, ok := d.BaseTensorType().Geometry()
		if !ok || t.Shape[0]%g.BlockSize != 0 {
			continue
		}
		if hasFloor && anchor.Rank(d) > anchor.Rank(floor) {
			continue
		}
		if d.RequiresImatrix() && !est.HasImportance(t.Name) {
			continue
		}
		b, _ := d.ExactBytes(t.Elements)
		legal = append(legal, core.TensorOption{TensorName: t.Name, Target: d, Bytes: b})
	}
	if len(legal) == 0 {
		return nil, fmt.Errorf("profile: tensor %q: no legal quant option (floor %q, candidates %v)",
			t.Name, floor, cands)
	}
	if sens.Calibrated(t.Name) {
		opts := make([]ScoredOption, 0, len(legal))
		for _, o := range legal {
			loss, ok := sens.Loss(t, o.Target, row)
			if !ok {
				continue
			}
			o.PriorLoss = loss
			opts = append(opts, ScoredOption{TensorOption: o, Loss: loss, Evidence: EvidenceMeasured, Confidence: 1})
		}
		if len(opts) > 0 {
			return ParetoPrune(opts), nil
		}
		// Degenerate row. A tensor whose reference error is zero (all-zero
		// weights) is lossless at every rung: take the cheapest. Anything
		// else (non-finite or missing entries) is pinned like an unprobed
		// tensor below.
		if ref := referenceLoss(t, sens.Roles[RoleKey(t.Name)].ProbeDType, row); ref == 0 && finiteRow(row) {
			for _, o := range legal {
				opts = append(opts, ScoredOption{TensorOption: o, Evidence: EvidenceMeasured, Confidence: 1})
			}
			return ParetoPrune(opts), nil
		}
	}
	// Unprobed (or unusable row): keep the highest-fidelity legal rung.
	// Prefer the lowest exact-table error; without a usable row, the
	// highest fidelity rank.
	var pin *ScoredOption
	pinKey := 0.0
	for _, o := range legal {
		var key float64
		if w, ok := row[o.Target.BaseTensorType()]; ok && finite(w) && w >= 0 {
			key = w
		} else {
			key = float64(anchor.Rank(o.Target)) + 1e12 // rank-only entries sort after any real error
		}
		if pin == nil || key < pinKey || (key == pinKey && o.Bytes > pin.Bytes) {
			so := ScoredOption{TensorOption: o, Evidence: EvidenceHeuristic, Confidence: 1}
			pin, pinKey = &so, key
		}
	}
	return []ScoredOption{*pin}, nil
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func finiteRow(row map[core.DType]float64) bool {
	if len(row) == 0 {
		return false
	}
	for _, v := range row {
		if !finite(v) {
			return false
		}
	}
	return true
}

// hullEdge is one step along a tensor's lower convex frontier.
type hullEdge struct {
	tensor   int
	from, to int
	slope    float64 // loss decrease per byte
	name     string
	target   core.DType
}

// lowerHull returns the indices of the frontier points on the lower convex
// hull of (bytes, loss). The frontier is bytes-ascending and loss-descending
// (ParetoPrune), so the hull is the subsequence with non-increasing
// loss-decrease-per-byte; only those rungs are ever optimal for some
// multiplier.
func lowerHull(opts []ScoredOption) []int {
	hull := make([]int, 0, len(opts))
	cross := func(o, a, b ScoredOption) float64 {
		return (float64(a.Bytes)-float64(o.Bytes))*(b.Loss-o.Loss) -
			(a.Loss-o.Loss)*(float64(b.Bytes)-float64(o.Bytes))
	}
	for i := range opts {
		for len(hull) >= 2 && cross(opts[hull[len(hull)-2]], opts[hull[len(hull)-1]], opts[i]) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, i)
	}
	return hull
}

// lagrangianAllocate sets every state's rung to the budget-constrained
// optimum of the relaxed problem, then fills leftover bytes greedily.
func lagrangianAllocate(states []solverState, bank *core.TensorBank, budget uint64) {
	var used uint64
	var edges []hullEdge
	for i := range states {
		st := &states[i]
		st.cur = 0
		used += st.opts[0].Bytes
		hull := lowerHull(st.opts)
		for k := 0; k+1 < len(hull); k++ {
			a, b := st.opts[hull[k]], st.opts[hull[k+1]]
			inc := float64(b.Bytes - a.Bytes)
			edges = append(edges, hullEdge{
				tensor: i, from: hull[k], to: hull[k+1],
				slope: (a.Loss - b.Loss) / inc,
				name:  bank.Tensors[i].Name, target: b.Target,
			})
		}
	}
	sort.Slice(edges, func(a, b int) bool {
		if edges[a].slope != edges[b].slope {
			return edges[a].slope > edges[b].slope
		}
		if edges[a].name != edges[b].name {
			return edges[a].name < edges[b].name
		}
		return edges[a].target < edges[b].target
	})
	// Hull sweep. Within one tensor hull slopes strictly decrease, so its
	// edges arrive in order; a skipped edge leaves cur != from and blocks
	// the tensor's later edges, exactly as a single multiplier would.
	for _, e := range edges {
		st := &states[e.tensor]
		if st.cur != e.from {
			continue
		}
		inc := st.opts[e.to].Bytes - st.opts[e.from].Bytes
		if used+inc > budget {
			continue
		}
		st.cur = e.to
		used += inc
	}
	// Refill: single-rung upgrades (interior frontier points included) by
	// best loss decrease per byte, then larger decrease, then name.
	for {
		best := -1
		var bestSlope, bestDec float64
		var bestInc uint64
		for i := range states {
			st := &states[i]
			if st.cur >= len(st.opts)-1 {
				continue
			}
			from, to := st.opts[st.cur], st.opts[st.cur+1]
			inc := to.Bytes - from.Bytes
			if inc == 0 || used+inc > budget {
				continue
			}
			dec := from.Loss - to.Loss
			slope := dec / float64(inc)
			better := best == -1 || slope > bestSlope ||
				(slope == bestSlope && (dec > bestDec ||
					(dec == bestDec && bank.Tensors[i].Name < bank.Tensors[best].Name)))
			if better {
				best, bestSlope, bestDec, bestInc = i, slope, dec, inc
			}
		}
		if best == -1 {
			return
		}
		states[best].cur++
		used += bestInc
	}
}
