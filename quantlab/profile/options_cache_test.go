package profile

import (
	"testing"
	"time"

	"quantlab/core"
)

// TestCacheConfidenceResolution: a stale measured entry (search-derived
// confidence) must not suppress a fresher exact estimate; a higher-
// confidence entry keeps winning. This is the rerun-replay guard: seeded
// caches from older code never replay the old allocation over an improved
// estimator.
func TestCacheConfidenceResolution(t *testing.T) {
	td := weightTD("blk.3.attn_q.weight", 256, 64)
	bank := &core.TensorBank{SourcePath: "/x", ModelID: "m", Tensors: []core.TensorDesc{td}}
	prov := core.Provenance{Tool: "test", RunID: "r", MeasuredAt: time.Unix(0, 0).UTC()}

	est := NewFallbackEstimator(nil)
	est.BindBank(bank)
	exactTable := map[string]map[core.DType]float64{
		td.Name: {core.DTypeQ4_K_T: 12.0},
	}
	est.SetExactLoss(exactTable)

	// Stale entry with a huge loss but only search-derived confidence 0.7.
	stale := NewCache("m", "")
	if err := stale.Put(CandidateLoss{
		TensorName: td.Name, Target: core.DTypeQ4_K_T, Loss: 1e9,
		Evidence: EvidenceMeasured, Confidence: 0.7, Prov: &prov,
	}); err != nil {
		t.Fatal(err)
	}
	opts, err := EnumerateOptions(td, []core.DType{core.DTypeQ4_K_T, core.DTypeQ4_K_S},
		nil, stale, est, DefaultConfidencePenalty)
	if err != nil {
		t.Fatal(err)
	}
	var q4 *ScoredOption
	for i, o := range opts {
		if o.Target == core.DTypeQ4_K_T {
			q4 = &opts[i]
		}
	}
	if q4 == nil {
		t.Fatal("no Q4_K option")
	}
	if q4.Loss > 1e6 {
		t.Errorf("exact estimate outranked by stale cache entry: loss %v", q4.Loss)
	}

	// Same entry at 0.95 confidence (a genuine measurement) wins again.
	strong := NewCache("m", "")
	if err := strong.Put(CandidateLoss{
		TensorName: td.Name, Target: core.DTypeQ4_K_T, Loss: 1e9,
		Evidence: EvidenceMeasured, Confidence: 0.95, Prov: &prov,
	}); err != nil {
		t.Fatal(err)
	}
	opts2, err := EnumerateOptions(td, []core.DType{core.DTypeQ4_K_T, core.DTypeQ4_K_S},
		nil, strong, est, DefaultConfidencePenalty)
	if err != nil {
		t.Fatal(err)
	}
	for _, o := range opts2 {
		if o.Target == core.DTypeQ4_K_T {
			if o.Loss < 1e6 {
				t.Errorf("high-confidence measured entry should win: loss %v", o.Loss)
			}
		}
	}
}
