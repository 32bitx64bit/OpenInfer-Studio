package profile

import (
	"sort"
	"time"

	"quantlab/core"
	"quantlab/kld"
)

// Search-derived cache confidence is deliberately below 1.0: group KLD is
// incumbent-context-dependent and must not lock out a later, better
// measurement. It still beats the 0.3 heuristic via DefaultConfidencePenalty.
const (
	SearchLossConfidenceSolo = 0.75
	SearchLossConfidencePair = 0.65
)

// IngestKLDHistory writes measured CandidateLoss entries from accepted KLD
// search steps into c. Solo (and pair) promotions attribute the
// incumbent-relative KLD improvement across the group's tensors at their To
// dtypes. Unknown GroupIDs, empty history, and eval-error/scan/prune steps
// are skipped. Returns the number of cache writes (including replacements).
func IngestKLDHistory(c *Cache, history []kld.Step, groups []core.MoveGroup, runID string, at time.Time) (int, error) {
	if c == nil || len(history) == 0 {
		return 0, nil
	}
	if at.IsZero() {
		at = time.Unix(0, 0).UTC()
	}
	byID := make(map[string]core.MoveGroup, len(groups))
	for _, g := range groups {
		if g.ID == "" {
			continue
		}
		byID[g.ID] = g
	}
	prevKLD := 0.0
	havePrev := false
	n := 0
	for _, step := range history {
		switch step.Kind {
		case "baseline":
			prevKLD = step.KLD
			havePrev = true
			continue
		case "solo", "pair":
			if !step.Accepted {
				continue
			}
		default:
			// scan / prune / prune-reject / eval-error: skip (context-noisy).
			if step.KLD != 0 && (step.Kind == "prune" || step.Accepted) {
				prevKLD = step.KLD
				havePrev = true
			}
			continue
		}
		delta := 0.0
		if havePrev {
			delta = prevKLD - step.KLD
		}
		prevKLD = step.KLD
		havePrev = true
		if delta < 0 {
			delta = 0
		}

		var moves []core.Move
		for _, id := range step.GroupIDs {
			g, ok := byID[id]
			if !ok {
				continue
			}
			moves = append(moves, g.Moves...)
		}
		if len(moves) == 0 {
			continue
		}
		share := delta / float64(len(moves))
		conf := SearchLossConfidenceSolo
		if step.Kind == "pair" {
			conf = SearchLossConfidencePair
		}
		prov := core.Provenance{Tool: "kld-search", RunID: runID, MeasuredAt: at}
		sort.Slice(moves, func(i, j int) bool {
			if moves[i].TensorName != moves[j].TensorName {
				return moves[i].TensorName < moves[j].TensorName
			}
			return moves[i].To < moves[j].To
		})
		for _, m := range moves {
			cl := CandidateLoss{
				TensorName: m.TensorName,
				Target:     m.To,
				Loss:       share,
				Evidence:   EvidenceMeasured,
				Confidence: conf,
				Prov:       &prov,
			}
			if err := c.PutReplace(cl); err != nil {
				return n, err
			}
			n++
		}
	}
	return n, nil
}
