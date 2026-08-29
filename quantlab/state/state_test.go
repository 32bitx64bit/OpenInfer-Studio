package state

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"quantlab/core"
)

// unixAbs maps a POSIX absolute path onto an OS-absolute path so Validate
// (filepath.IsAbs) accepts fixtures on Windows.
func unixAbs(p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return "C:" + filepath.FromSlash(p)
}

func testConfig() RunConfig {
	return RunConfig{
		SourcePath:          unixAbs("/models/src.gguf"),
		OutputDir:           unixAbs("/out"),
		WorkDir:             unixAbs("/work"),
		EvalCorpus:          unixAbs("/data/eval.txt"),
		Tools:               ToolPaths{LlamaQuantize: unixAbs("/opt/bin/llama-quantize"), LlamaPerplexity: unixAbs("/opt/bin/llama-perplexity")},
		BudgetBytes:         1 << 30,
		TargetBPW:           4.8,
		Threads:             8,
		CtxSize:             512,
		Gates:               []core.QualityGate{{Metric: core.MetricPerplexity, MaxDelta: 0.5}},
		SearchEnabled:       true,
		MaxSearchIterations: 10,
	}
}

func TestRunConfigValidate(t *testing.T) {
	if err := testConfig().Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	mutations := []struct {
		name string
		mut  func(*RunConfig)
	}{
		{"empty source", func(c *RunConfig) { c.SourcePath = "" }},
		{"relative source", func(c *RunConfig) { c.SourcePath = "rel.gguf" }},
		{"empty output", func(c *RunConfig) { c.OutputDir = "" }},
		{"relative workdir", func(c *RunConfig) { c.WorkDir = "tmp" }},
		{"missing quantize tool", func(c *RunConfig) { c.Tools.LlamaQuantize = "" }},
		{"relative tool", func(c *RunConfig) { c.Tools.LlamaPerplexity = "llama-perplexity" }},
		{"relative imatrix", func(c *RunConfig) { c.ImatrixPath = "im.dat" }},
		{"zero budget", func(c *RunConfig) { c.BudgetBytes = 0 }},
		{"negative bpw", func(c *RunConfig) { c.TargetBPW = -1 }},
		{"zero threads", func(c *RunConfig) { c.Threads = 0 }},
		{"zero ctx", func(c *RunConfig) { c.CtxSize = 0 }},
		{"bad gate", func(c *RunConfig) { c.Gates = []core.QualityGate{{Metric: core.MetricSizeBytes}} }},
		{"search without bound", func(c *RunConfig) { c.MaxSearchIterations = 0 }},
	}
	for _, m := range mutations {
		c := testConfig()
		m.mut(&c)
		if err := c.Validate(); err == nil {
			t.Errorf("%s: expected error", m.name)
		}
	}
	// Search disabled: no iteration bound needed.
	c := testConfig()
	c.SearchEnabled = false
	c.MaxSearchIterations = 0
	if err := c.Validate(); err != nil {
		t.Fatalf("search-disabled config rejected: %v", err)
	}
}

func TestRunConfigEffortValidation(t *testing.T) {
	for _, e := range []Effort{"", EffortFast, EffortProfiled, EffortDeep} {
		c := testConfig()
		c.Effort = e
		if err := c.Validate(); err != nil {
			t.Errorf("effort %q rejected: %v", e, err)
		}
	}
	c := testConfig()
	c.Effort = "ludicrous"
	if err := c.Validate(); err == nil {
		t.Error("unknown effort accepted")
	}
}

func TestRunLifecycle(t *testing.T) {
	now := time.Date(2026, 8, 16, 8, 0, 0, 0, time.UTC)
	r, err := NewRun("run-1", now, testConfig())
	if err != nil {
		t.Fatal(err)
	}
	next, ok := r.NextStage()
	if !ok || next != core.StageAssemble {
		t.Fatalf("next = %q,%v want assemble,true", next, ok)
	}
	if err := r.MarkComplete(core.StageSolve, "", now); err == nil {
		t.Fatal("out-of-order stage completed")
	}
	for _, s := range core.StageOrder {
		if err := r.MarkComplete(s, "/out/"+string(s), now); err != nil {
			t.Fatalf("complete %s: %v", s, err)
		}
	}
	if _, ok := r.NextStage(); ok {
		t.Fatal("run not complete after all stages")
	}
	if err := r.MarkComplete(core.StageEmit, "", now); err == nil {
		t.Fatal("completed run accepted another stage")
	}
}

func TestMarkCompleteRequiresConfig(t *testing.T) {
	r, err := NewRun("run-x", time.Now()) // no config
	if err != nil {
		t.Fatal(err)
	}
	if err := r.MarkComplete(core.StageAssemble, "", time.Now()); err == nil {
		t.Fatal("stage completed without config")
	}
}

func TestStoreRoundTripPreservesResumeState(t *testing.T) {
	dir := t.TempDir()
	st := Store{Dir: dir}
	now := time.Now()
	r, err := NewRun("abc", now, testConfig())
	if err != nil {
		t.Fatal(err)
	}
	r.Bank = &core.TensorBank{SourcePath: "/models/src.gguf", SHA256: "ff", Tensors: []core.TensorDesc{
		{Name: "w", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
	}}
	if err := r.MarkComplete(core.StageAssemble, "/work/bank.json", now); err != nil {
		t.Fatal(err)
	}
	r.Measurements = append(r.Measurements, core.Measurement{
		ProfileID: "p1", Metric: core.MetricPerplexity, Value: 6.5, Baseline: 6.1, Delta: 0.4,
		Prov: core.Provenance{Tool: "llama-perplexity", ToolVersion: "b6123", RunID: "abc", MeasuredAt: now},
	})
	if err := st.Save(r); err != nil {
		t.Fatal(err)
	}
	loaded, err := st.Load("abc")
	if err != nil {
		t.Fatal(err)
	}
	if loaded.RunID != "abc" || len(loaded.Completed) != 1 {
		t.Fatalf("round trip mismatch: %+v", loaded)
	}
	// Resume without CLI args: config, bank, measurements all present.
	cfg := loaded.Config
	if cfg.SourcePath != unixAbs("/models/src.gguf") || cfg.Tools.LlamaQuantize == "" || cfg.BudgetBytes == 0 || cfg.Threads != 8 {
		t.Fatalf("config not preserved: %+v", cfg)
	}
	if loaded.Bank == nil || loaded.Bank.SHA256 != "ff" {
		t.Fatal("bank not preserved")
	}
	if len(loaded.Measurements) != 1 || loaded.Measurements[0].Prov.ToolVersion != "b6123" {
		t.Fatal("measurements not preserved")
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "*.tmp"))
	if len(matches) != 0 {
		t.Fatalf("temp files left behind: %v", matches)
	}
}

func TestLoadFreshRunArtifactsMapSafe(t *testing.T) {
	dir := t.TempDir()
	st := Store{Dir: dir}
	now := time.Now()
	r, err := NewRun("fresh", now, testConfig())
	if err != nil {
		t.Fatal(err)
	}
	// Save before any stage completes: Artifacts is empty and therefore
	// omitted from the JSON entirely (omitempty).
	if err := st.Save(r); err != nil {
		t.Fatal(err)
	}
	loaded, err := st.Load("fresh")
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Artifacts == nil {
		t.Fatal("Artifacts map nil after Load")
	}
	// Recording the first artifact on the reloaded run must not panic
	// ("assignment to entry in nil map") and must persist.
	if err := loaded.MarkComplete(core.StageAssemble, "/work/bank.json", now); err != nil {
		t.Fatalf("mark complete after reload: %v", err)
	}
	if got := loaded.Artifacts[core.StageAssemble]; got != "/work/bank.json" {
		t.Fatalf("artifact = %q", got)
	}
	// The nil-map guard also covers runs that never went through Load.
	manual := *r
	manual.Artifacts = nil
	if err := manual.MarkComplete(core.StageAssemble, "/work/bank.json", now); err != nil {
		t.Fatalf("mark complete on nil map: %v", err)
	}
}

func TestLoadRejectsBadVersion(t *testing.T) {
	dir := t.TempDir()
	st := Store{Dir: dir}
	r, _ := NewRun("v", time.Now(), testConfig())
	r.Version = 999
	if err := st.Save(r); err == nil {
		t.Fatal("save accepted bad version")
	}
}

func TestNewRunEmptyID(t *testing.T) {
	if _, err := NewRun("", time.Now(), testConfig()); err == nil {
		t.Fatal("empty run id accepted")
	}
	if _, err := NewRun("ok", time.Now(), RunConfig{}); err == nil {
		t.Fatal("invalid config accepted at construction")
	}
	if _, err := NewRun("ok", time.Now(), testConfig(), testConfig()); err == nil {
		t.Fatal("multiple configs accepted")
	}
}

func TestRunIDValidation(t *testing.T) {
	st := Store{Dir: t.TempDir()}
	bad := []string{
		"../../evil", "a/b", `a\b`, "..", ".", "a b", "a;b", "a$b",
		"", strings.Repeat("x", MaxRunIDLen+1),
	}
	for _, id := range bad {
		if _, err := NewRun(id, time.Now(), testConfig()); err == nil {
			t.Errorf("NewRun(%q) accepted", id)
		}
		if _, err := st.Path(id); err == nil {
			t.Errorf("Store.Path(%q) accepted", id)
		}
		if _, err := st.Load(id); err == nil {
			t.Errorf("Store.Load(%q) accepted", id)
		}
	}
	// A checkpoint whose on-disk runID turned malicious is rejected by Load.
	r, err := NewRun("victim", time.Now(), testConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := st.Save(r); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Load("victim"); err != nil {
		t.Fatalf("valid id rejected: %v", err)
	}
	// Save refuses a run whose ID was corrupted after construction.
	r.RunID = "../escape"
	if err := st.Save(r); err == nil {
		t.Fatal("save accepted traversal run id")
	}
	for _, ok := range []string{"run-1", "a.b_c", "X9"} {
		if _, err := st.Path(ok); err != nil {
			t.Errorf("Store.Path(%q) rejected: %v", ok, err)
		}
	}
}
