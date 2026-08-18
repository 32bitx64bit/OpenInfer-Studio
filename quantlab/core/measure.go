package core

import (
	"fmt"
	"strings"
	"time"
)

// MetricKind identifies an evaluation signal.
type MetricKind string

const (
	MetricPerplexity MetricKind = "perplexity"
	MetricKLD        MetricKind = "kld"
	// MetricP95KLD is the 95th-percentile per-token KLD of a candidate
	// against the source baseline model. It is an absolute divergence:
	// the source baseline itself is 0.
	MetricP95KLD MetricKind = "p95-kld"
	// MetricCVaRKLD is the mean of the worst 5% of per-chunk/token KLD
	// samples when llama-perplexity emits that distribution. It is a
	// search ranking term, not a bit-allocation objective: default gates
	// stay mean+p95. Absent a sample list it is not synthesized from p95.
	MetricCVaRKLD   MetricKind = "cvar-kld"
	MetricMaxKLD    MetricKind = "max-kld"
	MetricRMSDeltaP MetricKind = "rms-delta-p"
	// MetricTop1Disagreement is 1 - same-top fraction, so lower is better
	// like every other search objective.
	//
	// Unsloth Divergence-300 @32 is greedy 32-token match vs BF16 on 300
	// held-out prompts. This repo has no generation harness, so Refine uses
	// llama-perplexity next-token (and prefix, when printed) same-top as
	// the trajectory-match stand-in. Do not add a second "div-horizon"
	// kind: that would duplicate this metric. Perplexity stays a recorded
	// human signal and is not this proxy.
	MetricTop1Disagreement MetricKind = "top1-disagreement"
	MetricSizeBytes        MetricKind = "size_bytes"
	// MetricWorstDomainKLD is a gate-only virtual metric: the maximum of all
	// per-domain KLD measurements (metric names below). Measurements are
	// recorded per domain; the worst-domain gate fires on the max.
	MetricWorstDomainKLD MetricKind = "worst-domain-kld"
	// MetricDomainPrefix prefixes per-domain KLD measurement metrics.
	MetricDomainPrefix MetricKind = "kld-dom."
)

// Provenance records exactly how a measurement was produced, so results are
// auditable and comparable across resumed runs.
type Provenance struct {
	Tool           string    `json:"tool"`        // e.g. "llama-perplexity"
	ToolVersion    string    `json:"toolVersion"` // binary-reported version string
	BinarySHA256   string    `json:"binarySHA256,omitempty"`
	ProducerSHA256 string    `json:"producerSHA256,omitempty"` // quantizer binary identity
	RunID          string    `json:"runID"`
	SourceSHA      string    `json:"sourceSHA,omitempty"`  // evaluated run source/payload identity
	CorpusSHA      string    `json:"corpusSHA,omitempty"`  // eval corpus digest
	ImatrixSHA     string    `json:"imatrixSHA,omitempty"` // imatrix digest, when used
	EvalContext    int       `json:"evalContext,omitempty"`
	EvalChunks     int       `json:"evalChunks,omitempty"`
	EvalThreads    int       `json:"evalThreads,omitempty"`
	EvalNGPULayers int       `json:"evalNGPULayers"`
	EvalSeed       int64     `json:"evalSeed,omitempty"`
	EvalConfigured bool      `json:"evalConfigured,omitempty"`
	MeasuredAt     time.Time `json:"measuredAt"`
}

func (p Provenance) Validate() error {
	if p.Tool == "" {
		return fmt.Errorf("provenance: empty tool")
	}
	if p.RunID == "" {
		return fmt.Errorf("provenance: empty run id")
	}
	if p.MeasuredAt.IsZero() {
		return fmt.Errorf("provenance: zero timestamp")
	}
	return nil
}

// Measurement is one scored evaluation of a candidate artifact, with full
// provenance. Lower is better for perplexity and KLD.
//
// Baseline semantics: KLD-family metrics (kld, p95-kld) are measured against
// the run's fixed reference — the source baseline model, via the baseline
// logits captured once at the evaluate stage — so Baseline is always 0 and
// Delta always equals Value. Perplexity measurements carry the baseline
// model's perplexity in Baseline, so Delta is the regression versus the
// source model. Gates must evaluate against these source-baseline semantics
// only; deltas relative to an intermediate incumbent are never recorded.
type Measurement struct {
	ProfileID string     `json:"profileID"`
	Metric    MetricKind `json:"metric"`
	Value     float64    `json:"value"`
	Baseline  float64    `json:"baseline"`
	Delta     float64    `json:"delta"` // Value - Baseline
	Prov      Provenance `json:"prov"`
}

func (m Measurement) Validate() error {
	if m.ProfileID == "" {
		return fmt.Errorf("measurement: empty profile id")
	}
	if !KnownMetric(m.Metric) {
		return fmt.Errorf("measurement: unknown metric %q", m.Metric)
	}
	if m.Metric != MetricSizeBytes && m.Value < 0 {
		return fmt.Errorf("measurement: negative %s value", m.Metric)
	}
	return m.Prov.Validate()
}

// EvalResult is the legacy lightweight evaluation record retained for the
// search history. New code should prefer Measurement.
type EvalResult struct {
	ProfileID string     `json:"profileID"`
	Metric    MetricKind `json:"metric"`
	Value     float64    `json:"value"`
	Baseline  float64    `json:"baseline,omitempty"`
	Delta     float64    `json:"delta"`
}

// QualityGate is an acceptance threshold on a measured metric. A candidate
// passes when its Delta (value minus baseline) is at most MaxDelta, and, when
// MaxAbsolute > 0, its absolute Value is at most MaxAbsolute.
type QualityGate struct {
	Metric      MetricKind `json:"metric"`
	MaxDelta    float64    `json:"maxDelta"`
	MaxAbsolute float64    `json:"maxAbsolute,omitempty"`
}

// KnownMetric reports whether k is a valid measurement metric kind.
func KnownMetric(k MetricKind) bool {
	switch k {
	case MetricPerplexity, MetricKLD, MetricP95KLD, MetricCVaRKLD, MetricMaxKLD,
		MetricRMSDeltaP, MetricTop1Disagreement, MetricSizeBytes:
		return true
	}
	return strings.HasPrefix(string(k), string(MetricDomainPrefix)) &&
		len(k) > len(MetricDomainPrefix)
}

// DomainMetric builds the per-domain KLD measurement kind for a corpus
// domain name (kld-dom.<domain>).
func DomainMetric(domain string) MetricKind { return MetricDomainPrefix + MetricKind(domain) }

func (g QualityGate) Validate() error {
	switch g.Metric {
	case MetricPerplexity, MetricKLD, MetricP95KLD, MetricCVaRKLD, MetricMaxKLD,
		MetricRMSDeltaP, MetricTop1Disagreement, MetricWorstDomainKLD:
	default:
		return fmt.Errorf("quality gate: unsupported metric %q", g.Metric)
	}
	if g.MaxDelta < 0 {
		return fmt.Errorf("quality gate: negative max delta")
	}
	if g.MaxAbsolute < 0 {
		return fmt.Errorf("quality gate: negative max absolute")
	}
	return nil
}

// Passes evaluates the gate against one measurement. A measurement of a
// different metric kind never passes (fail-closed), and a missing
// measurement is the caller's responsibility to treat as a failure.
func (g QualityGate) Passes(m Measurement) bool {
	if m.Metric != g.Metric {
		return false
	}
	if m.Delta > g.MaxDelta {
		return false
	}
	if g.MaxAbsolute > 0 && m.Value > g.MaxAbsolute {
		return false
	}
	return true
}
