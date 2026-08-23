package quantize

import (
	"encoding/json"
	"math"
	"strings"
	"sync"
	"testing"

	"quantlab/core"
)

func TestTierBPW(t *testing.T) {
	cases := []struct {
		tier string
		bpw  float64
		ok   bool
	}{
		{"q5", 5.5, true},
		{"q4", 4.5, true},
		{"q3", 3.5, true},
		{"q2", 2.5, true},
		{"Q4", 4.5, true},
		{"custom", 0, false},
		{"", 0, false},
		{"q9", 0, false},
	}
	for _, c := range cases {
		bpw, ok := TierBPW(c.tier)
		if bpw != c.bpw || ok != c.ok {
			t.Fatalf("TierBPW(%q) = %v,%v want %v,%v", c.tier, bpw, ok, c.bpw, c.ok)
		}
	}
}

func TestResolveQuantTier(t *testing.T) {
	t.Run("named tier overrides target_bpw", func(t *testing.T) {
		req := Request{QuantTier: "q3", TargetBPW: 2.0, TargetBytes: 999}
		resolveQuantTier(&req)
		if req.QuantTier != "q3" || req.TargetBPW != 3.5 || req.TargetBytes != 0 {
			t.Fatalf("resolve q3 = %#v", req)
		}
	})
	t.Run("custom keeps target_bytes", func(t *testing.T) {
		req := Request{QuantTier: "custom", TargetBytes: 1 << 20}
		resolveQuantTier(&req)
		if req.QuantTier != "custom" || req.TargetBytes != 1<<20 || req.TargetBPW != 0 {
			t.Fatalf("resolve custom = %#v", req)
		}
	})
	t.Run("empty leaves explicit overrides", func(t *testing.T) {
		req := Request{QuantTier: "", TargetBPW: 3.83}
		resolveQuantTier(&req)
		if req.QuantTier != "" || req.TargetBPW != 3.83 {
			t.Fatalf("resolve empty = %#v", req)
		}
	})
}

func TestNormalizeAdaptiveTargetDefaultsToQ4(t *testing.T) {
	req := Request{Kind: KindAdaptiveQuantize}
	warnings, err := normalizeAdaptiveTarget(&req)
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 0 {
		t.Fatalf("unexpected warnings: %v", warnings)
	}
	if req.TargetBPW != 4.5 {
		t.Fatalf("default target_bpw = %v, want 4.5 (q4)", req.TargetBPW)
	}
}

func TestQuantlabTierLabel(t *testing.T) {
	cases := []struct {
		tier  string
		label string
	}{
		{"q5", "OID-Q5_K_XL"},
		{"q4", "OID-Q4_K_XL"},
		{"q3", "OID-Q3_K_XL"},
		{"q2", "OID-Q2_K_XL"},
	}
	for _, c := range cases {
		if got := quantlabTierLabel(c.tier, nil); got != c.label {
			t.Fatalf("quantlabTierLabel(%q, nil) = %q want %q", c.tier, got, c.label)
		}
	}
	// An empty tier (legacy request with target_bytes) maps the achieved BPW.
	if got := quantlabTierLabel("", nil); got != "OID-Q4_K_XL" {
		t.Fatalf("quantlabTierLabel(\"\", nil) = %q, want default OID-Q4_K_XL", got)
	}
}

func TestQuantlabLabelMapsAchievedBPW(t *testing.T) {
	manifest := func(target core.DType, bytes uint64) *core.SelectionManifest {
		return &core.SelectionManifest{
			ProfileID:  "test",
			Options:    []core.TensorOption{{TensorName: "blk.0.w", Target: target, Bytes: bytes}},
			TotalBytes: bytes,
		}
	}
	cases := []struct {
		target core.DType
		label  string
	}{
		{core.DTypeQ5_K_T, "OID-Q5_K_XL"},
		{core.DTypeQ4_K_T, "OID-Q4_K_XL"},
		{core.DTypeQ3_K, "OID-Q3_K_XL"},
		{core.DTypeQ2_K, "OID-Q2_K_XL"},
	}
	for _, c := range cases {
		m := manifest(c.target, 1000)
		if got := quantlabLabel(m); got != c.label {
			t.Fatalf("quantlabLabel(%s) = %q want %q (achieved bpw %v)",
				c.target, got, c.label, manifestAchievedBPW(m))
		}
	}
	// Custom tier falls back to the achieved-BPW mapping.
	if got := quantlabTierLabel("custom", manifest(core.DTypeQ4_K_T, 1000)); got != "OID-Q4_K_XL" {
		t.Fatalf("custom q4 manifest label = %q", got)
	}
	if got := quantlabLabel(nil); got != "OID-Q4_K_XL" {
		t.Fatalf("nil manifest label = %q, want OID-Q4_K_XL", got)
	}
}

func TestManifestAchievedBPWMixed(t *testing.T) {
	// 75% Q4_K_T (4.5 bpw) + 25% Q2_K (2.625 bpw) by bytes. The achieved
	// bpw is the byte-weighted harmonic mean (total_bits/total_elements),
	// which lands in the Q3 band.
	q4 := core.TensorOption{TensorName: "a", Target: core.DTypeQ4_K_T, Bytes: 7500}
	q2 := core.TensorOption{TensorName: "b", Target: core.DTypeQ2_K, Bytes: 2500}
	m := &core.SelectionManifest{ProfileID: "mix", Options: []core.TensorOption{q4, q2}, TotalBytes: 10000}
	bpw := manifestAchievedBPW(m)
	// 10000 / (7500/4.5 + 2500/2.625) ~= 3.818
	if math.Abs(bpw-3.818) > 0.01 {
		t.Fatalf("mixed achieved bpw = %v, want ~3.818", bpw)
	}
	if tierPrefixFromBPW(bpw) != "Q3" {
		t.Fatalf("mixed tier prefix = %q, want Q3", tierPrefixFromBPW(bpw))
	}
}

// observerSink captures quant.progress events for observer mapping tests.
type observerSink struct {
	mu     sync.Mutex
	events []Progress
}

func (s *observerSink) Publish(event string, payload any) {
	if p, ok := payload.(Progress); ok && event == "quant.progress" {
		s.mu.Lock()
		s.events = append(s.events, p)
		s.mu.Unlock()
	}
}

func (s *observerSink) snapshot() []Progress {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Progress, len(s.events))
	copy(out, s.events)
	return out
}

func insertObserverJob(t *testing.T, env *quantEnv) string {
	t.Helper()
	const id = "obs-job"
	raw, _ := json.Marshal(Request{Kind: KindAdaptiveQuantize, Effort: "fast"})
	if _, err := env.qm.db.Exec(`INSERT INTO quant_jobs(id,kind,state,stage,progress,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		id, KindAdaptiveQuantize, "running", "", 0, "", "", "", 0, "", string(raw), "{}", "", now(), now()); err != nil {
		t.Fatal(err)
	}
	return id
}

func TestQuantlabObserverMapsStagesAndMonotonicProgress(t *testing.T) {
	env := newQuantEnv(t, false)
	id := insertObserverJob(t, env)
	sink := &observerSink{}
	env.qm.events = sink
	obs := &quantlabObserver{m: env.qm, jobID: id}

	wantNames := []string{"analyze", "anchor", "solve", "quantize", "validate", "search", "finalize"}
	stages := []core.Stage{core.StageAssemble, core.StageAnchor, core.StageSolve,
		core.StageQuantize, core.StageEvaluate, core.StageSearch, core.StageEmit}
	var prevOverall float64
	for i, s := range stages {
		obs.StageStarted(s)
		obs.StageProgress(s, 0.5, "mid "+wantNames[i])
		j, err := env.qm.Get(id)
		if err != nil {
			t.Fatal(err)
		}
		if j.Stage != wantNames[i] {
			t.Fatalf("stage %s: job stage = %q, want %q", s, j.Stage, wantNames[i])
		}
		if j.Progress < prevOverall-1e-9 {
			t.Fatalf("stage %s: overall %v decreased from %v", s, j.Progress, prevOverall)
		}
		prevOverall = j.Progress
	}
	// Every mapped stage name was published at least once.
	published := map[string]bool{}
	for _, p := range sink.snapshot() {
		published[p.Stage] = true
	}
	for _, want := range wantNames {
		if !published[want] {
			t.Fatalf("stage name %q was not published; got %v", want, published)
		}
	}
}

func TestQuantlabObserverQuantizeAnchorStepCounter(t *testing.T) {
	env := newQuantEnv(t, false)
	id := insertObserverJob(t, env)
	sink := &observerSink{}
	env.qm.events = sink
	obs := &quantlabObserver{m: env.qm, jobID: id}

	obs.StageStarted(core.StageQuantize)
	for _, f := range []float64{0.25, 0.5, 0.75, 1.0} {
		obs.StageProgress(core.StageQuantize, f, "anchor")
	}
	snaps := sink.snapshot()
	last := snaps[len(snaps)-1]
	if last.Current != 4 || last.Total != 4 {
		t.Fatalf("anchor counter = %d/%d, want 4/4", last.Current, last.Total)
	}
	j, _ := env.qm.Get(id)
	if j.ProgressCurrent != 4 || j.ProgressTotal != 4 {
		t.Fatalf("persisted counter = %d/%d, want 4/4", j.ProgressCurrent, j.ProgressTotal)
	}
	// The assembly sub-phase reports no step counter (current/total clear).
	obs.StageProgress(core.StageQuantize, 0.5, "assembling candidate")
	j, _ = env.qm.Get(id)
	if j.ProgressCurrent != 0 || j.ProgressTotal != 0 {
		t.Fatalf("assembly counter = %d/%d, want 0/0", j.ProgressCurrent, j.ProgressTotal)
	}
}

func TestQuantlabObserverMeasurementFeedback(t *testing.T) {
	env := newQuantEnv(t, false)
	id := insertObserverJob(t, env)
	sink := &observerSink{}
	env.qm.events = sink
	obs := &quantlabObserver{m: env.qm, jobID: id}

	obs.StageStarted(core.StageSearch)
	obs.StageProgress(core.StageSearch, 0.5, "assembling candidate p1")
	for i := 0; i < 3; i++ {
		obs.Measurement(core.Measurement{ProfileID: "p", Metric: core.MetricKLD, Value: 0.01 * float64(i+1)})
	}
	j, _ := env.qm.Get(id)
	if !strings.Contains(j.ProgressMessage, "measurement 3") {
		t.Fatalf("persisted message = %q, want it to contain measurement 3", j.ProgressMessage)
	}
	// The search bar advanced past the assembly sliver toward 0.9.
	if j.StageProgress < 0.3 {
		t.Fatalf("search stage_progress = %v, expected to advance per measurement", j.StageProgress)
	}
}

func TestQuantlabObserverProgressNeverDecreases(t *testing.T) {
	env := newQuantEnv(t, false)
	id := insertObserverJob(t, env)
	sink := &observerSink{}
	env.qm.events = sink
	obs := &quantlabObserver{m: env.qm, jobID: id}

	obs.StageStarted(core.StageQuantize)
	obs.StageProgress(core.StageQuantize, 0.8, "assembling candidate")
	obs.StageProgress(core.StageQuantize, 0.2, "assembling candidate")
	j, _ := env.qm.Get(id)
	// Assembly remap of 0.8 is 0.9; a later 0.2 (remap 0.6) must clamp to it.
	if j.StageProgress < 0.89 {
		t.Fatalf("stage_progress decreased to %v after clamp", j.StageProgress)
	}
}

func TestEffortCalibrationPreset(t *testing.T) {
	if got := effortCalibrationPreset("fast"); got != "quick" {
		t.Fatalf("fast = %q", got)
	}
	if got := effortCalibrationPreset("deep"); got != "research" {
		t.Fatalf("deep = %q", got)
	}
	if got := effortCalibrationPreset("profiled"); got != "thorough" {
		t.Fatalf("profiled = %q", got)
	}
	if got := effortCalibrationPreset(""); got != "thorough" {
		t.Fatalf("empty = %q", got)
	}
	req := Request{Effort: "fast"}
	applyEffortCalibration(&req, "fast")
	if req.CalibrationPreset != "quick" {
		t.Fatalf("applied preset = %q", req.CalibrationPreset)
	}
	req.CalibrationPreset = "research"
	applyEffortCalibration(&req, "fast")
	if req.CalibrationPreset != "research" {
		t.Fatalf("explicit preset overwritten: %q", req.CalibrationPreset)
	}
}
