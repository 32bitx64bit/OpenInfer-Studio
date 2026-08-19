package profile

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"

	"quantlab/anchor"
	"quantlab/core"
)

// upgradeStep is one candidate frontier step-up in the solver's greedy loop.
type upgradeStep struct {
	name     string     // tensor name (used for deterministic tie-breaking)
	target   core.DType // dtype stepped up to
	gain     float64    // loss decrease per byte
	lossDec  float64    // absolute loss decrease
	bytesInc uint64     // byte increase
}

// betterUpgrade implements the deterministic upgrade ranking: larger
// absolute loss decrease, then higher gain (loss/byte), then smaller byte
// increase, then tensor name ascending, then target dtype ascending.
//
// Discrete RD with mixed tensor sizes cannot rank on gain alone: a swarm of
// slightly-better-gain small tensors spends the budget in fragments that no
// longer fit one large fused projection. Loss is already element-weighted,
// so lossDec prefers tensors that actually move model distortion.
// firstBest marks the initial "no best yet" state (kept explicit so callers
// never index an empty best). The final target comparison always references
// the CURRENT best option's target, so banks with duplicate tensor names
// (rejected by TensorBank.Validate, but defended here anyway) cannot index
// the wrong option list.
func betterUpgrade(cand, best upgradeStep, firstBest bool) bool {
	switch {
	case firstBest:
		return true
	case cand.lossDec != best.lossDec:
		return cand.lossDec > best.lossDec
	case cand.gain != best.gain:
		return cand.gain > best.gain
	case cand.bytesInc != best.bytesInc:
		return cand.bytesInc < best.bytesInc
	case cand.name != best.name:
		return cand.name < best.name
	case cand.target != best.target:
		return cand.target < best.target
	}
	return false
}

// betterDiverse inverts betterUpgrade: prefer the smallest positive loss
// decrease (the upgrades greedy seeds deprioritize), then larger byte
// increases for more structural spread, then name for determinism.
func betterDiverse(cand, best upgradeStep) bool {
	switch {
	case cand.lossDec != best.lossDec:
		return cand.lossDec < best.lossDec
	case cand.bytesInc != best.bytesInc:
		return cand.bytesInc > best.bytesInc
	case cand.name != best.name:
		return cand.name < best.name
	case cand.target != best.target:
		return cand.target < best.target
	}
	return false
}

// solverState is one tensor's Pareto frontier plus the currently chosen rung.
type solverState struct {
	opts []ScoredOption
	cur  int
}

// Solve runs the architecture-agnostic multiple-choice rate-distortion
// allocation: exactly one dtype per tensor, chosen from each tensor's
// Pareto-pruned option frontier, minimizing total effective loss under an
// exact byte budget.
//
// Algorithm (greedy marginal-gain / Lagrangian allocation with refill):
//
//  1. Enumerate each tensor's legal options (exact bytes, hard floors
//     enforced, losses confidence-adjusted and prior-shaped) and Pareto-prune
//     them, leaving a bytes-ascending, strictly loss-descending frontier.
//  2. Start from the cheapest legal assignment (MinBytes). If even that
//     exceeds the budget, return *InfeasibleError with the feasibility
//     envelope.
//  3. Repeatedly apply the single frontier step-up with the largest
//     absolute loss decrease that still fits the remaining budget. Loss is
//     element-weighted, so this is architecture-agnostic: fused / SSM /
//     wide-FFN tensors are not packed out by a swarm of smaller names with
//     slightly better loss-per-byte. Gain (Δloss/Δbyte) breaks ties, then
//     smaller byte increase, tensor name, target dtype.
//  4. The loop terminates when no upgrade fits, so leftover budget is
//     automatically refilled to the best marginal use (no further greedy
//     repair pass is needed).
//  5. SwiGLU gate/up pairs in the same layer are coupled to a shared dtype
//     (highest-fidelity common frontier rung that still fits).
//  6. A bounded 2-opt reallocater (at most 32 attempted swaps) demotes the
//     worst loss-decrease-per-byte rung to fund the best next-rung promotion
//     when the swap cuts total effective loss without exceeding the budget.
//
// With no budget (BudgetBytes and TargetBPW both zero) the loop upgrades
// every tensor to its minimum-loss option.
func Solve(req Request) (*Result, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	set := req.Anchors
	if set == nil {
		set = &anchor.Set{}
	}
	est := NewFallbackEstimator(req.Imatrix)
	est.BindBank(req.Bank)
	est.SetExactLoss(req.ExactLoss)
	est.SetCalibration(req.Calibration)
	est.SetRoleScale(req.RolePriorScale) // negative flattens priors
	est.SetTargetBPW(req.TargetBPW)
	cands := req.Candidates
	if cands == nil {
		cands = core.QuantDTypes
	}
	confPenalty := req.ConfidencePenalty
	if confPenalty == 0 {
		confPenalty = DefaultConfidencePenalty
	}

	budget := req.BudgetBytes
	if budget == 0 && req.TargetBPW > 0 {
		b, err := BPWToBudget(req.Bank, req.TargetBPW)
		if err != nil {
			return nil, err
		}
		budget = b
	}

	states := make([]solverState, len(req.Bank.Tensors))
	var minTotal, maxTotal uint64
	var constrained []string
	for i, t := range req.Bank.Tensors {
		opts, err := EnumerateOptions(t, cands, set, req.Cache, est, confPenalty)
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

	// Marginal-gain upgrade loop with refill. Diversity inverts the
	// ranking so the budget is spent on the upgrades greedy deprioritizes.
	remaining := effective - minTotal
	for {
		bestT, bestStep := -1, 0
		var bestGain, bestLossDec float64
		var bestBytesInc uint64
		for i := range states {
			st := &states[i]
			if st.cur >= len(st.opts)-1 {
				continue
			}
			from, to := st.opts[st.cur], st.opts[st.cur+1]
			inc := to.Bytes - from.Bytes
			if inc > remaining {
				continue
			}
			dec := from.Loss - to.Loss // > 0 on a Pareto frontier
			gain := dec / float64(inc)
			cand := upgradeStep{
				name: req.Bank.Tensors[i].Name, target: to.Target,
				gain: gain, lossDec: dec, bytesInc: inc,
			}
			better := bestT == -1
			if !better {
				cur := upgradeStep{
					name: req.Bank.Tensors[bestT].Name, target: states[bestT].opts[bestStep].Target,
					gain: bestGain, lossDec: bestLossDec, bytesInc: bestBytesInc,
				}
				if req.Diversity {
					better = betterDiverse(cand, cur)
				} else {
					better = betterUpgrade(cand, cur, false)
				}
			}
			if better {
				bestT, bestStep = i, st.cur+1
				bestGain, bestLossDec, bestBytesInc = gain, dec, inc
			}
		}
		if bestT == -1 {
			break
		}
		states[bestT].cur = bestStep
		remaining -= bestBytesInc
	}

	if !req.DisableSwiGLUCoupling {
		coupleSwiGLU(states, req.Bank, effective)
	}
	// reallocate2Opt normalizes toward the greedy optimum; the diversity
	// seed deliberately avoids it.
	if !req.Diversity {
		reallocate2Opt(states, req.Bank, effective)
	}
	if !req.DisableSwiGLUCoupling {
		coupleSwiGLU(states, req.Bank, effective)
	}

	// Assemble outputs in bank order.
	assignments := make([]core.QuantAssignment, 0, len(states))
	var total uint64
	var totalLoss float64
	var measured, estimated int
	for i, st := range states {
		o := st.opts[st.cur]
		total += o.Bytes
		totalLoss += o.Loss
		if o.Evidence == EvidenceMeasured {
			measured++
		} else {
			estimated++
		}
		assignments = append(assignments, core.QuantAssignment{
			TensorName:    o.TensorName,
			Target:        o.Target,
			BitsPerWeight: float64(o.Bytes) * 8.0 / float64(req.Bank.Tensors[i].Elements),
		})
	}

	id := req.ProfileID
	if id == "" {
		h := sha256.New()
		for _, qa := range assignments {
			fmt.Fprintf(h, "%s:%s\n", qa.TensorName, qa.Target)
		}
		id = "prof-" + hex.EncodeToString(h.Sum(nil))[:16]
	}
	p := &core.Profile{
		ID:             id,
		BaseModel:      req.Bank.ModelID,
		Assignments:    assignments,
		Anchors:        append([]core.Anchor(nil), set.Hard...),
		BudgetBytes:    budget,
		EstimatedBytes: total,
	}
	if err := p.Validate(req.Bank); err != nil {
		return nil, err
	}
	if v := set.Check(p); len(v) > 0 {
		sort.Slice(v, func(i, j int) bool { return v[i].Tensor < v[j].Tensor })
		return nil, fmt.Errorf("profile: solver produced anchor violations: %+v", v)
	}
	m, err := core.ManifestFor(p, req.Bank)
	if err != nil {
		return nil, err
	}
	return &Result{
		Profile:  p,
		Manifest: m,
		Diag: Diagnostics{
			BudgetBytes:      budget,
			MinBytes:         minTotal,
			MaxBytes:         maxTotal,
			TotalLoss:        totalLoss,
			SlopBytes:        effective - total,
			MeasuredTensors:  measured,
			EstimatedTensors: estimated,
		},
	}, nil
}

// maxReallocAttempts bounds the 2-opt reallocater. Each considered
// (demote, promote) pair counts as one attempt.
const maxReallocAttempts = 32

func assignmentBytes(states []solverState) uint64 {
	var n uint64
	for i := range states {
		n += states[i].opts[states[i].cur].Bytes
	}
	return n
}

func assignmentLoss(states []solverState) float64 {
	var n float64
	for i := range states {
		n += states[i].opts[states[i].cur].Loss
	}
	return n
}

// coupleSwiGLU equalizes ffn_gate/gate_proj/w1 with ffn_up/up_proj/w3 per
// layer. Prefers the highest-fidelity dtype both frontiers share that still
// fits the budget; otherwise the lower common rung. Does not fail the solve
// if no common dtype fits.
func coupleSwiGLU(states []solverState, bank *core.TensorBank, budget uint64) {
	if bank == nil || len(states) != len(bank.Tensors) {
		return
	}
	layers := map[int][]int{}
	var ids []int
	seen := map[int]bool{}
	for i, t := range bank.Tensors {
		d := layerIndex(t.Name)
		if d < 0 {
			continue
		}
		if !seen[d] {
			ids = append(ids, d)
			seen[d] = true
		}
		layers[d] = append(layers[d], i)
	}
	sort.Ints(ids)
	for _, d := range ids {
		var gates, ups []int
		for _, i := range layers[d] { // bank order
			switch swigluHalf(bank.Tensors[i].Name) {
			case "gate":
				gates = append(gates, i)
			case "up":
				ups = append(ups, i)
			}
		}
		n := len(gates)
		if len(ups) < n {
			n = len(ups)
		}
		for k := 0; k < n; k++ {
			couplePair(states, gates[k], ups[k], budget)
		}
	}
}

func couplePair(states []solverState, i, j int, budget uint64) {
	if i < 0 || j < 0 || i >= len(states) || j >= len(states) || i == j {
		return
	}
	si, sj := &states[i], &states[j]
	if len(si.opts) == 0 || len(sj.opts) == 0 {
		return
	}
	if si.opts[si.cur].Target == sj.opts[sj.cur].Target {
		return
	}
	var others uint64
	for k := range states {
		if k == i || k == j {
			continue
		}
		others += states[k].opts[states[k].cur].Bytes
	}
	type common struct {
		rank  int
		ii    int
		jj    int
		bytes uint64
		dtype core.DType
	}
	var cands []common
	for ii, oi := range si.opts {
		for jj, oj := range sj.opts {
			if oi.Target != oj.Target {
				continue
			}
			if others+oi.Bytes+oj.Bytes > budget {
				continue
			}
			cands = append(cands, common{
				rank:  anchor.Rank(oi.Target),
				ii:    ii,
				jj:    jj,
				bytes: oi.Bytes + oj.Bytes,
				dtype: oi.Target,
			})
		}
	}
	if len(cands) == 0 {
		return
	}
	sort.Slice(cands, func(a, b int) bool {
		if cands[a].rank != cands[b].rank {
			return cands[a].rank < cands[b].rank
		}
		if cands[a].bytes != cands[b].bytes {
			return cands[a].bytes < cands[b].bytes
		}
		return cands[a].dtype < cands[b].dtype
	})
	best := cands[0]
	si.cur = best.ii
	sj.cur = best.jj
}

type reallocMove struct {
	idx    int
	name   string
	metric float64
	newCur int
	bytes  uint64
	loss   float64
}

func collectReallocMoves(states []solverState, bank *core.TensorBank) (demotes, promotes []reallocMove) {
	for i := range states {
		st := &states[i]
		if len(st.opts) < 2 {
			continue
		}
		name := bank.Tensors[i].Name
		cur := st.opts[st.cur]
		if st.cur > 0 {
			cheap := st.opts[st.cur-1]
			db := cur.Bytes - cheap.Bytes
			if db > 0 {
				dec := cheap.Loss - cur.Loss // quality paid for the current rung
				demotes = append(demotes, reallocMove{
					idx: i, name: name, metric: dec / float64(db),
					newCur: st.cur - 1, bytes: cheap.Bytes, loss: cheap.Loss,
				})
			}
		}
		if st.cur < len(st.opts)-1 {
			next := st.opts[st.cur+1]
			db := next.Bytes - cur.Bytes
			if db > 0 {
				dec := cur.Loss - next.Loss
				promotes = append(promotes, reallocMove{
					idx: i, name: name, metric: dec / float64(db),
					newCur: st.cur + 1, bytes: next.Bytes, loss: next.Loss,
				})
			}
		}
	}
	sort.Slice(demotes, func(i, j int) bool {
		if demotes[i].metric != demotes[j].metric {
			return demotes[i].metric < demotes[j].metric
		}
		return demotes[i].name < demotes[j].name
	})
	sort.Slice(promotes, func(i, j int) bool {
		if promotes[i].metric != promotes[j].metric {
			return promotes[i].metric > promotes[j].metric
		}
		return promotes[i].name < promotes[j].name
	})
	return demotes, promotes
}

// reallocate2Opt is one bounded pass of (demote worst, promote best) swaps.
func reallocate2Opt(states []solverState, bank *core.TensorBank, budget uint64) {
	if bank == nil || len(states) != len(bank.Tensors) {
		return
	}
	attempts := 0
	for attempts < maxReallocAttempts {
		demotes, promotes := collectReallocMoves(states, bank)
		if len(demotes) == 0 || len(promotes) == 0 {
			return
		}
		curB := assignmentBytes(states)
		curL := assignmentLoss(states)
		applied := false
		for _, d := range demotes {
			for _, p := range promotes {
				if d.idx == p.idx {
					continue
				}
				attempts++
				if attempts > maxReallocAttempts {
					return
				}
				curD := states[d.idx].opts[states[d.idx].cur]
				curP := states[p.idx].opts[states[p.idx].cur]
				newB := curB - curD.Bytes + d.bytes - curP.Bytes + p.bytes
				newL := curL - curD.Loss + d.loss - curP.Loss + p.loss
				if newB <= budget && newL < curL {
					states[d.idx].cur = d.newCur
					states[p.idx].cur = p.newCur
					applied = true
					break
				}
			}
			if applied {
				break
			}
		}
		if !applied {
			return
		}
	}
}
