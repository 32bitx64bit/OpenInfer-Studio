package core

import (
	"math"
	"testing"
	"time"
)

func prov() Provenance {
	return Provenance{
		Tool:        "llama-perplexity",
		ToolVersion: "b6123",
		RunID:       "run-1",
		CorpusSHA:   "deadbeef",
		MeasuredAt:  time.Date(2026, 8, 16, 8, 0, 0, 0, time.UTC),
	}
}

func TestMeasurementValidate(t *testing.T) {
	m := Measurement{ProfileID: "p1", Metric: MetricPerplexity, Value: 6.5, Baseline: 6.1, Delta: 0.4, Prov: prov()}
	if err := m.Validate(); err != nil {
		t.Fatalf("valid measurement rejected: %v", err)
	}
	bad := m
	bad.Prov.MeasuredAt = time.Time{}
	if err := bad.Validate(); err == nil {
		t.Error("zero provenance time accepted")
	}
	bad = m
	bad.Metric = "bogus"
	if err := bad.Validate(); err == nil {
		t.Error("unknown metric accepted")
	}
	bad = m
	bad.Value = -1
	if err := bad.Validate(); err == nil {
		t.Error("negative perplexity accepted")
	}
	bad = m
	bad.Prov.Tool = ""
	if err := bad.Validate(); err == nil {
		t.Error("empty tool accepted")
	}
}

func TestQualityGate(t *testing.T) {
	g := QualityGate{Metric: MetricPerplexity, MaxDelta: 0.5, MaxAbsolute: 7.0}
	if err := g.Validate(); err != nil {
		t.Fatalf("valid gate rejected: %v", err)
	}
	pass := Measurement{Metric: MetricPerplexity, Value: 6.4, Delta: 0.3}
	if !g.Passes(pass) {
		t.Error("passing measurement rejected")
	}
	if g.Passes(Measurement{Metric: MetricPerplexity, Value: 6.4, Delta: 0.6}) {
		t.Error("delta breach passed")
	}
	if g.Passes(Measurement{Metric: MetricPerplexity, Value: 7.1, Delta: 0.1}) {
		t.Error("absolute breach passed")
	}
	if g.Passes(Measurement{Metric: MetricKLD, Value: 0.01, Delta: 0.01}) {
		t.Error("wrong metric passed")
	}
	for _, bad := range []QualityGate{
		{Metric: MetricSizeBytes, MaxDelta: 1},
		{Metric: MetricKLD, MaxDelta: -1},
		{Metric: MetricKLD, MaxAbsolute: -1},
	} {
		if err := bad.Validate(); err == nil {
			t.Errorf("expected error for %+v", bad)
		}
	}
}

// TestP95KLDGate proves the p95-kld gate gates the p95 measurement, not the
// mean: a mean-low/p95-high candidate passes mean-kld and fails p95-kld.
func TestP95KLDGate(t *testing.T) {
	meanGate := QualityGate{Metric: MetricKLD, MaxDelta: 0.05}
	p95Gate := QualityGate{Metric: MetricP95KLD, MaxDelta: math.MaxFloat64, MaxAbsolute: 0.2}
	if err := meanGate.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := p95Gate.Validate(); err != nil {
		t.Fatal(err)
	}
	mean := Measurement{Metric: MetricKLD, Value: 0.01, Baseline: 0, Delta: 0.01}
	p95 := Measurement{Metric: MetricP95KLD, Value: 0.5, Baseline: 0, Delta: 0.5}
	if !meanGate.Passes(mean) {
		t.Error("mean-low measurement failed mean-kld gate")
	}
	if !p95Gate.Passes(Measurement{Metric: MetricP95KLD, Value: 0.15, Delta: 0.15}) {
		t.Error("p95-low measurement failed p95-kld gate")
	}
	if p95Gate.Passes(p95) {
		t.Error("p95-high measurement passed p95-kld gate")
	}
	if meanGate.Passes(mean) && !p95Gate.Passes(p95) {
		// documented outcome: same candidate, mean passes, p95 fails
	} else {
		t.Fatal("gate split not honored")
	}
	// Fail-closed: the p95 gate never passes on the mean measurement (and
	// vice versa) — a missing p95 measurement cannot satisfy the gate.
	if p95Gate.Passes(mean) {
		t.Error("p95 gate passed on a mean-kld measurement")
	}
	if meanGate.Passes(p95) {
		t.Error("mean gate passed on a p95-kld measurement")
	}
	// MetricP95KLD is a first-class measurement kind.
	m := Measurement{ProfileID: "p", Metric: MetricP95KLD, Value: 0.1, Delta: 0.1, Prov: prov()}
	if err := m.Validate(); err != nil {
		t.Fatalf("p95 measurement rejected: %v", err)
	}
}

func TestMoveValidateAndByteDelta(t *testing.T) {
	m := Move{TensorName: "w", From: DTypeQ4_K_T, To: DTypeQ6_K}
	if err := m.Validate(); err != nil {
		t.Fatalf("valid move rejected: %v", err)
	}
	d, err := m.ByteDelta(256)
	if err != nil {
		t.Fatal(err)
	}
	if d != 66 { // 210-144
		t.Fatalf("byte delta = %d, want 66", d)
	}
	for _, bad := range []Move{
		{TensorName: "", From: DTypeQ4_K_T, To: DTypeQ6_K},
		{TensorName: "w", From: DTypeQ4_K_T, To: DTypeQ4_K_T},
		{TensorName: "w", From: "NOPE", To: DTypeQ6_K},
	} {
		if err := bad.Validate(); err == nil {
			t.Errorf("expected error for %+v", bad)
		}
	}
}

func TestMoveGroup(t *testing.T) {
	bank := &TensorBank{SourcePath: "/m.gguf", Tensors: []TensorDesc{
		{Name: "a", DType: DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
		{Name: "b", DType: DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
	}}
	g, err := NewMoveGroup("g1", "upgrade worst pair", []Move{
		{TensorName: "a", From: DTypeQ4_K_T, To: DTypeQ6_K},
		{TensorName: "b", From: DTypeIQ2_XXS, To: DTypeQ4_K_T},
	}, bank)
	if err != nil {
		t.Fatal(err)
	}
	// a: 65536/256*(210-144)=16896; b: 65536/256*(144-66)=19968.
	if g.Bytes != 36864 {
		t.Fatalf("group bytes = %d, want 36864", g.Bytes)
	}
	if _, err := NewMoveGroup("g2", "", []Move{
		{TensorName: "a", From: DTypeQ4_K_T, To: DTypeQ6_K},
		{TensorName: "a", From: DTypeQ4_K_T, To: DTypeQ8_0},
	}, bank); err == nil {
		t.Fatal("duplicate tensor in group accepted")
	}
	if _, err := NewMoveGroup("g3", "", nil, bank); err == nil {
		t.Fatal("empty group accepted")
	}
	if _, err := NewMoveGroup("", "", []Move{{TensorName: "a", From: DTypeQ4_K_T, To: DTypeQ6_K}}, bank); err == nil {
		t.Fatal("empty id accepted")
	}
	g.Bytes = 0
	if err := g.Validate(bank); err == nil {
		t.Fatal("stale byte delta accepted")
	}
}
