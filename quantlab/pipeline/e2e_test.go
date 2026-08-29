package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"quantlab/core"
	"quantlab/orchestrate"
	"quantlab/state"
	"quantlab/tensorbank"
)

func testExe(dir, name string) string {
	if runtime.GOOS == "windows" && filepath.Ext(name) == "" {
		name += ".exe"
	}
	return filepath.Join(dir, name)
}

// buildFakeTool compiles the synthetic tool binary from testdata (argv-only
// go build; skipped when no toolchain is available).
func buildFakeTool(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go toolchain not available to build the fake tool")
	}
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate test source dir")
	}
	pkgDir := filepath.Dir(thisFile)
	bin := testExe(t.TempDir(), "faketool")
	cmd := exec.Command("go", "build", "-o", bin, "./testdata/faketool")
	cmd.Dir = pkgDir
	cmd.Env = os.Environ()
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build faketool: %v\n%s", err, out)
	}
	return bin
}

// TestSyntheticEndToEnd runs the full plan->resume pipeline against real
// subprocesses: the compiled faketool stands in for llama-quantize and
// llama-perplexity, driven by the production orchestrate.OSRunner.
func TestSyntheticEndToEnd(t *testing.T) {
	tool := buildFakeTool(t)
	dir := t.TempDir()
	src := filepath.Join(dir, "tiny.gguf")
	if err := writeGGUF(src, testTensors()); err != nil {
		t.Fatal(err)
	}
	calib := filepath.Join(dir, "corpus")
	if err := os.MkdirAll(calib, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(calib, "docs.txt"), []byte("alpha beta gamma delta epsilon zeta\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gates, err := ParseGates("mean-kld=0.5,p95-kld=0.03")
	if err != nil {
		t.Fatal(err)
	}
	store := state.Store{Dir: filepath.Join(dir, "state")}
	r, err := Plan(PlanOptions{
		SourcePath:      src,
		StateDir:        store.Dir,
		OutputDir:       filepath.Join(dir, "out"),
		CalibrationDir:  calib,
		LlamaQuantize:   tool,
		LlamaPerplexity: tool,
		BudgetBytes:     150000,
		Threads:         2,
		CtxSize:         512,
		Gates:           gates,
		RunID:           "e2e",
		Now:             time.Now(),
		Stdout:          os.Stdout,
	})
	if err != nil {
		t.Fatal(err)
	}
	eng, err := NewEngine(store, r, orchestrate.OSRunner{Env: []string{}}, os.Stdout)
	if err != nil {
		t.Fatal(err)
	}
	if err := eng.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	done, err := store.Load("e2e")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := done.NextStage(); ok {
		t.Fatalf("run incomplete: %v", done.Completed)
	}
	// Final artifact parses, carries quantized tensors, and matches the
	// manifest's dtype selection exactly.
	var final string
	ents, err := os.ReadDir(done.Config.OutputDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		if filepath.Ext(e.Name()) == ".gguf" {
			final = filepath.Join(done.Config.OutputDir, e.Name())
		}
	}
	if final == "" {
		t.Fatal("no final gguf emitted")
	}
	s, err := tensorbank.OpenSource(final)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	pf, err := tensorbank.Parse(s)
	if err != nil {
		t.Fatal(err)
	}
	if len(pf.Tensors) != len(done.Manifest.Options) {
		t.Fatalf("emitted model has %d tensors, manifest has %d", len(pf.Tensors), len(done.Manifest.Options))
	}
	quantTensors := 0
	for _, o := range done.Manifest.Options {
		ti, ok := pf.FindTensor(o.TensorName)
		if !ok {
			t.Fatalf("emitted model missing tensor %q", o.TensorName)
		}
		if ti.DType != o.Target.BaseTensorType() {
			t.Fatalf("tensor %q emitted as %s, manifest wants %s", o.TensorName, ti.DType, o.Target)
		}
		if ti.DType.IsQuant() {
			quantTensors++
		}
	}
	if quantTensors == 0 {
		t.Fatal("no quantized tensors in emitted model")
	}
	// Report with passing gates and recorded measurements.
	var found FoundReport
	rp := filepath.Join(done.Config.OutputDir, "tiny-e2e.report.json")
	data, err := os.ReadFile(rp)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &found); err != nil {
		t.Fatal(err)
	}
	if !found.GatesPass || found.Output.Bytes == 0 {
		t.Fatalf("report: %+v", found)
	}
	if !hasMeasurementPub(done, "baseline", core.MetricPerplexity) {
		t.Fatal("baseline measurement missing")
	}
	var p95 float64
	for _, m := range done.Measurements {
		if m.Metric == core.MetricP95KLD {
			p95 = m.Value
		}
	}
	if p95 != 0.025 {
		t.Fatalf("p95 KLD = %v, want parser value 0.025", p95)
	}
}

func buildFakePerplexity(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go toolchain not available to build fakeperplexity")
	}
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate test source dir")
	}
	repoRoot := filepath.Join(filepath.Dir(thisFile), "..", "..")
	bin := testExe(t.TempDir(), "llama-perplexity")
	cmd := exec.Command("go", "build", "-o", bin, "./tests/fakeperplexity")
	cmd.Dir = repoRoot
	cmd.Env = os.Environ()
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build fakeperplexity: %v\n%s", err, out)
	}
	return bin
}

type toolMux struct {
	quant orchestrate.Runner
	ppl   orchestrate.Runner
}

func (m toolMux) Run(ctx context.Context, iv orchestrate.Invocation) (orchestrate.Result, error) {
	switch iv.Tool {
	case orchestrate.ToolLlamaQuantize:
		return m.quant.Run(ctx, iv)
	case orchestrate.ToolPerplexity:
		return m.ppl.Run(ctx, iv)
	default:
		return orchestrate.Result{}, fmt.Errorf("unknown tool %s", iv.Tool)
	}
}

type stripKLDRunner struct {
	inner orchestrate.Runner
}

func (s stripKLDRunner) Run(ctx context.Context, iv orchestrate.Invocation) (orchestrate.Result, error) {
	if iv.Tool == orchestrate.ToolPerplexity {
		filtered := make([]string, 0, len(iv.Argv))
		for _, a := range iv.Argv {
			if a == "--kl-divergence" {
				continue
			}
			filtered = append(filtered, a)
		}
		iv.Argv = filtered
	}
	return s.inner.Run(ctx, iv)
}

func planEvalOSRunner(t *testing.T, f *fixture, runID, pplPath string) *state.Run {
	t.Helper()
	g, err := ParseGates("mean-kld=0.5")
	if err != nil {
		t.Fatal(err)
	}
	r, err := Plan(PlanOptions{
		SourcePath:      f.src,
		StateDir:        f.stateDir,
		OutputDir:       f.outDir,
		CalibrationDir:  f.calibDir,
		LlamaQuantize:   f.src,
		LlamaPerplexity: pplPath,
		BudgetBytes:     f.budget,
		Threads:         2,
		CtxSize:         512,
		Gates:           g,
		RunID:           runID,
		Now:             time.Now(),
		Stdout:          io.Discard,
	})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// TestEvaluateOSRunnerFakePerplexity covers evaluate through the production
// OSRunner against tests/fakeperplexity (which prints KLD only when
// --kl-divergence is present). Missing that flag must fail at evaluate.
func TestEvaluateOSRunnerFakePerplexity(t *testing.T) {
	ppl := buildFakePerplexity(t)

	t.Run("records KLD", func(t *testing.T) {
		f := newFixture(t, 150000)
		r := planEvalOSRunner(t, f, "os-kld", ppl)
		mux := toolMux{quant: f.runner, ppl: orchestrate.OSRunner{Env: []string{}}}
		e, err := NewEngine(state.Store{Dir: f.stateDir}, r, mux, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		e.StageLimit = 5
		if err := e.Resume(context.Background()); err != nil {
			t.Fatal(err)
		}
		done, err := (state.Store{Dir: f.stateDir}).Load("os-kld")
		if err != nil {
			t.Fatal(err)
		}
		if !hasMeasurementPub(done, done.BestProfileID, core.MetricKLD) {
			t.Fatal("candidate KLD missing after evaluate")
		}
		var sawEval, sawSearch bool
		for _, s := range done.Completed {
			if s == core.StageEvaluate {
				sawEval = true
			}
			if s == core.StageSearch {
				sawSearch = true
			}
		}
		if !sawEval {
			t.Fatal("StageEvaluate not completed")
		}
		if sawSearch {
			t.Fatal("search ran despite StageLimit")
		}
	})

	t.Run("missing --kl-divergence fails at evaluate", func(t *testing.T) {
		f := newFixture(t, 150000)
		r := planEvalOSRunner(t, f, "os-nokld", ppl)
		mux := toolMux{
			quant: f.runner,
			ppl:   stripKLDRunner{inner: orchestrate.OSRunner{Env: []string{}}},
		}
		e, err := NewEngine(state.Store{Dir: f.stateDir}, r, mux, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		err = e.Resume(context.Background())
		if err == nil || !strings.Contains(err.Error(), "produced no KLD") {
			t.Fatalf("error = %v", err)
		}
		if !strings.Contains(err.Error(), "stage evaluate") {
			t.Fatalf("want evaluate-stage failure, got %v", err)
		}
		got, err := (state.Store{Dir: f.stateDir}).Load("os-nokld")
		if err != nil {
			t.Fatal(err)
		}
		for _, s := range got.Completed {
			if s == core.StageEvaluate {
				t.Fatal("StageEvaluate completed without KLD")
			}
			if s == core.StageSearch {
				t.Fatal("search stage completed after evaluate failed")
			}
		}
	})
}

type FoundReport struct {
	GatesPass bool       `json:"gatesPass"`
	Output    reportFile `json:"output"`
}
