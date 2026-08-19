package profile

import (
	"bytes"
	"math"
	"testing"
	"time"

	"quantlab/core"
)

// cacheFrom builds a measured cache with synthetic per-weight losses for
// every (tensor, dtype) in pairs.
func cacheFrom(t *testing.T, modelID string, pairs map[string]map[core.DType]float64) *Cache {
	t.Helper()
	c := NewCache(modelID, "sha-test")
	prov := core.Provenance{Tool: "test", RunID: "r1", MeasuredAt: time.Unix(0, 0).UTC()}
	names := make([]string, 0, len(pairs))
	for n := range pairs {
		names = append(names, n)
	}
	for _, n := range names {
		for d, v := range pairs[n] {
			if err := c.Put(CandidateLoss{
				TensorName: n, Target: d, Loss: v,
				Evidence: EvidenceMeasured, Confidence: 0.75, Prov: &prov,
			}); err != nil {
				t.Fatal(err)
			}
		}
	}
	return c
}

func TestFitCalibrationTooFewSamples(t *testing.T) {
	bank := smallBank(2)
	c := cacheFrom(t, "m", map[string]map[core.DType]float64{
		"blk.0.attn_q.weight": {core.DTypeQ4_K_T: 0.01},
		"blk.1.attn_q.weight": {core.DTypeQ4_K_T: 0.02},
	})
	est := NewFallbackEstimator(nil)
	if cal := FitCalibration(bank, c, est, "test"); cal != nil {
		t.Fatalf("expected nil calibration for 2 samples, got %+v", cal)
	}
}

func TestFitCalibrationCorrection(t *testing.T) {
	bank := smallBank(8)
	est := NewFallbackEstimator(nil)
	est.BindBank(bank)
	// Synthetic ground truth: measured per-weight loss = heuristic * 3 for
	// every tensor/dtype (a constant multiplicative offset).
	pairs := map[string]map[core.DType]float64{}
	for _, td := range bank.Tensors {
		m := map[core.DType]float64{}
		for _, d := range []core.DType{core.DTypeQ4_K_T, core.DTypeQ8_0, core.DTypeQ2_K} {
			h, _ := est.heuristic(td, d)
			m[d] = h / float64(td.Elements) * 3
		}
		pairs[td.Name] = m
	}
	c := cacheFrom(t, "m", pairs)
	cal := FitCalibration(bank, c, est, "test")
	if cal == nil {
		t.Fatal("expected fitted calibration")
	}
	if cal.Samples != 8*3 {
		t.Errorf("samples %d, want 24", cal.Samples)
	}
	if cal.R2 < 0.95 {
		// A constant multiplicative offset in log space is mostly, but not
		// entirely, explained by the feature set; the intercept catches the
		// rest. R2 stays high but need not be ~1.
		t.Errorf("R2 %.4f, want >0.95 for a constant offset", cal.R2)
	}
	est2 := NewFallbackEstimator(nil)
	est2.BindBank(bank)
	est2.SetCalibration(cal)
	td := bank.Tensors[0]
	hBefore, _ := est.heuristic(td, core.DTypeQ4_K_T)
	after, _ := est2.Estimate(td, core.DTypeQ4_K_T)
	ratio := after / hBefore
	if math.Abs(ratio-3) > 0.35 {
		t.Errorf("correction ratio %.3f, want ~3", ratio)
	}
}

func TestCalibrationClamp(t *testing.T) {
	cal := &Calibration{Features: 1, Beta: []float64{10}, Mean: []float64{0}, Std: []float64{1},
		Intercept: 0, MinCorr: calCorrMin, MaxCorr: calCorrMax}
	if got := cal.Correction([]float64{5}, 1); got != calCorrMax {
		t.Errorf("correction %v, want clamp max %v", got, calCorrMax)
	}
	if got := cal.Correction([]float64{-5}, 1); got != calCorrMin {
		t.Errorf("correction %v, want clamp min %v", got, calCorrMin)
	}
}

func TestCalibrationStoreRoundTrip(t *testing.T) {
	s := NewCalibrationStore("m", "sha", "llama")
	s.Levels[LevelModel] = &Calibration{Features: 2, Beta: []float64{0.1, -0.2},
		Mean: []float64{0, 0}, Std: []float64{1, 1}, Samples: 20, MinCorr: calCorrMin, MaxCorr: calCorrMax}
	s.Samples[LevelModel] = 20
	var buf bytes.Buffer
	if err := SaveCalibration(&buf, s); err != nil {
		t.Fatal(err)
	}
	raw := append([]byte(nil), buf.Bytes()...)
	got, err := LoadCalibration(&buf, "m", "sha")
	if err != nil {
		t.Fatal(err)
	}
	if got.Resolve("llama") == nil || got.Resolve("llama").Samples != 20 {
		t.Fatal("resolve failed")
	}
	buf2 := bytes.NewBuffer(raw)
	// identity mismatch yields nil silently
	other, err := LoadCalibration(buf2, "other", "sha2")
	if err != nil {
		t.Fatal(err)
	}
	if other != nil {
		t.Fatal("identity mismatch must yield nil")
	}
}

func smallBank(n int) *core.TensorBank {
	b := &core.TensorBank{SourcePath: "/dev/null", ModelID: "m"}
	for i := 0; i < n; i++ {
		b.Tensors = append(b.Tensors, core.TensorDesc{
			Name:  "blk." + itoa(i) + ".attn_q.weight",
			DType: core.DTypeF32, Shape: []uint64{256, 64},
			Elements: 256 * 64, Length: 256 * 64 * 4,
		})
	}
	return b
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var d []byte
	for i > 0 {
		d = append([]byte{byte('0' + i%10)}, d...)
		i /= 10
	}
	return string(d)
}
