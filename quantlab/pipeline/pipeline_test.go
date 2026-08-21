package pipeline

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"quantlab/core"
	"quantlab/orchestrate"
	"quantlab/state"
	"quantlab/tensorbank"
)

// --- tiny synthetic GGUF construction -------------------------------------

type gtensor struct {
	name  string
	dt    core.DType
	shape []uint64
}

func testTensors() []gtensor {
	return []gtensor{
		{"blk.0.attn_q.weight", core.DTypeF16, []uint64{256, 256}},
		{"blk.0.ffn_down.weight", core.DTypeF16, []uint64{512, 256}},
		{"blk.1.attn_q.weight", core.DTypeF16, []uint64{256, 256}},
		{"blk.0.attn_norm.weight", core.DTypeF32, []uint64{256}},
	}
}

func ggmlID(dt core.DType) uint32 {
	id, ok := tensorbank.GGMLTypeID(dt)
	if !ok {
		panic("no ggml id for " + string(dt))
	}
	return id
}

func alignUpU(n, a uint64) uint64 { return (n + a - 1) / a * a }

func writeKVString(w io.Writer, k, v string) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(len(k)))
	w.Write(b[:])
	w.Write([]byte(k))
	var t [4]byte
	binary.LittleEndian.PutUint32(t[:], uint32(8)) // VTString
	w.Write(t[:])
	binary.LittleEndian.PutUint64(b[:], uint64(len(v)))
	w.Write(b[:])
	w.Write([]byte(v))
}

func writeKVUint32(w io.Writer, k string, v uint32) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(len(k)))
	w.Write(b[:])
	w.Write([]byte(k))
	var t [4]byte
	binary.LittleEndian.PutUint32(t[:], uint32(4)) // VTUint32
	w.Write(t[:])
	var v4 [4]byte
	binary.LittleEndian.PutUint32(v4[:], v)
	w.Write(v4[:])
}

// writeGGUF writes a minimal valid GGUF v3 file with the given tensors.
func writeGGUF(path string, tensors []gtensor) error {
	type rec struct {
		t   gtensor
		rel uint64
	}
	var recs []rec
	var cur uint64
	for _, t := range tensors {
		elems := uint64(1)
		for _, d := range t.shape {
			elems *= d
		}
		cur = alignUpU(cur, 32)
		recs = append(recs, rec{t, cur})
		l, _ := t.dt.ExactBytes(elems)
		cur += l
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var hdr [24]byte
	binary.LittleEndian.PutUint32(hdr[0:4], 0x46554747)
	binary.LittleEndian.PutUint32(hdr[4:8], 3)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(len(recs)))
	binary.LittleEndian.PutUint64(hdr[16:24], 3)
	f.Write(hdr[:])
	writeKVString(f, "general.architecture", "llama")
	writeKVString(f, "general.name", "tiny")
	writeKVUint32(f, "general.alignment", 32)
	for _, r := range recs {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], uint64(len(r.t.name)))
		f.Write(b[:])
		f.Write([]byte(r.t.name))
		var n [4]byte
		binary.LittleEndian.PutUint32(n[:], uint32(len(r.t.shape)))
		f.Write(n[:])
		for _, d := range r.t.shape {
			binary.LittleEndian.PutUint64(b[:], d)
			f.Write(b[:])
		}
		binary.LittleEndian.PutUint32(n[:], ggmlID(r.t.dt))
		f.Write(n[:])
		binary.LittleEndian.PutUint64(b[:], r.rel)
		f.Write(b[:])
	}
	// Pad to alignment relative to the metadata start is handled by offsets:
	// the data region begins at alignUp(metaEnd, 32); write zeros until each
	// tensor's aligned absolute offset, then its payload.
	metaEnd, _ := f.Seek(0, io.SeekCurrent)
	dataStart := alignUpU(uint64(metaEnd), 32)
	if pad := dataStart - uint64(metaEnd); pad > 0 {
		f.Write(make([]byte, pad))
	}
	rng := rand.New(rand.NewSource(7))
	for i, r := range recs {
		abs := dataStart + r.rel
		if pos, _ := f.Seek(0, io.SeekCurrent); pos < int64(abs) {
			f.Write(make([]byte, abs-uint64(pos)))
		}
		elems := uint64(1)
		for _, d := range r.t.shape {
			elems *= d
		}
		l, _ := r.t.dt.ExactBytes(elems)
		payload := make([]byte, l)
		rng.Read(payload)
		payload[0] = byte(i) // deterministic marker
		f.Write(payload)
	}
	return f.Close()
}

// fakeQuantizeGGUF rewrites a GGUF storing every quantizable tensor in the
// target dtype (zero payload for converted tensors), mimicking llama-quantize
// output closely enough for streaming assembly.
func fakeQuantizeGGUF(t *testing.T, in, out string, target core.DType, pure bool) {
	t.Helper()
	s, err := tensorbank.OpenSource(in)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	f, err := tensorbank.Parse(s)
	if err != nil {
		t.Fatal(err)
	}
	target = target.BaseTensorType()
	quantizable := func(shape []uint64) bool {
		return len(shape) == 2 && (shape[0]%256 == 0 || shape[0]%32 == 0)
	}
	type rec struct {
		ti  tensorbank.TensorInfo
		rel uint64
	}
	al := uint64(f.Alignment)
	var recs []rec
	var cur uint64
	for _, ti := range f.Tensors {
		nt := ti
		if quantizable(ti.Shape) {
			nt.DType = target
			if !pure && target == core.DTypeQ3_K && strings.Contains(ti.Name, "ffn_down") {
				nt.DType = core.DTypeQ4_K_T
			}
			nt.GGMLType = ggmlID(nt.DType)
			l, _ := nt.DType.ExactBytes(nt.Elements)
			nt.Length = l
		}
		cur = alignUpU(cur, al)
		recs = append(recs, rec{nt, cur})
		cur += nt.Length
	}
	of, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	var hdr [24]byte
	binary.LittleEndian.PutUint32(hdr[0:4], 0x46554747)
	binary.LittleEndian.PutUint32(hdr[4:8], f.Header.Version)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(len(recs)))
	binary.LittleEndian.PutUint64(hdr[16:24], uint64(len(f.KVs)))
	of.Write(hdr[:])
	of.Write(f.KVBytes)
	for _, r := range recs {
		var b [8]byte
		var n [4]byte
		binary.LittleEndian.PutUint64(b[:], uint64(len(r.ti.Name)))
		of.Write(b[:])
		of.Write([]byte(r.ti.Name))
		binary.LittleEndian.PutUint32(n[:], uint32(len(r.ti.Shape)))
		of.Write(n[:])
		for _, d := range r.ti.Shape {
			binary.LittleEndian.PutUint64(b[:], d)
			of.Write(b[:])
		}
		binary.LittleEndian.PutUint32(n[:], r.ti.GGMLType)
		of.Write(n[:])
		binary.LittleEndian.PutUint64(b[:], r.rel)
		of.Write(b[:])
	}
	metaEnd := 24 + len(f.KVBytes)
	for _, r := range recs {
		metaEnd += 8 + len(r.ti.Name) + 4 + 8*len(r.ti.Shape) + 4 + 8
	}
	dataStart := alignUpU(uint64(metaEnd), al)
	if pad := int64(dataStart) - (24 + int64(len(f.KVBytes))); pad > 0 {
		// pad from current position (end of tensor infos)
	}
	if pos, _ := of.Seek(0, io.SeekCurrent); pos < int64(dataStart) {
		of.Write(make([]byte, int64(dataStart)-pos))
	}
	buf := make([]byte, 1<<16)
	for i, r := range recs {
		abs := dataStart + r.rel
		if pos, _ := of.Seek(0, io.SeekCurrent); pos < int64(abs) {
			of.Write(make([]byte, int64(abs)-pos))
		}
		if r.ti.DType == f.Tensors[i].DType && r.ti.Length == f.Tensors[i].Length {
			off := f.PayloadOffset(f.Tensors[i])
			left := r.ti.Length
			for left > 0 {
				n := uint64(len(buf))
				if left < n {
					n = left
				}
				if _, err := s.ReadAt(buf[:n], off+int64(r.ti.Length-left)); err != nil {
					t.Fatal(err)
				}
				of.Write(buf[:n])
				left -= n
			}
		} else {
			of.Write(make([]byte, r.ti.Length))
		}
	}
	if err := of.Close(); err != nil {
		t.Fatal(err)
	}
}

// --- fake runner ------------------------------------------------------------

const fakeQuantizeHelp = `usage: fake-quantize [options] <in> <out> <type> [threads]
options:
 --imatrix FILE
 --tensor-type FILE
 --output-tensor-type TYPE
 --token-embedding-type TYPE
 --pure
 --keep-split
 --dry-run
 --version
version: fake-q1
types: Q8_0 Q6_K Q5_K Q5_1 Q5_0 Q4_K Q4_1 Q4_0 Q3_K Q2_K IQ4_NL IQ4_XS IQ3_S IQ3_XXS IQ2_S IQ2_M IQ2_XXS IQ1_M
`

const fakePerplexityHelp = `usage: fake-perplexity [options]
 -m MODEL
 -f FILE
 -c N
 -chunks N
 -t N
 -ngl N
 -s SEED
 --kl-divergence
 --kl-divergence-base FILE
 --version
version: fake-p1
`

type fakeRunner struct {
	mu          sync.Mutex
	t           *testing.T
	quantRuns   int
	pplRuns     int
	baselinePPL float64
	candPPL     float64
	candKLD     float64
	// noP95 omits the p95 KLD line from candidate evaluations, simulating a
	// tool that reports mean KLD only.
	noP95 bool
	// omitKLD prints candidate PPL without Mean KLD even when
	// --kl-divergence is present, simulating a PPL-only harness.
	omitKLD     bool
	noPure      bool
	lastPPLArgv []string
	// lastQuantInTensors is the tensor count of the GGUF passed to the most
	// recent (non-help) llama-quantize run, used to assert IQ jobs quantize
	// a trimmed subset rather than the full source.
	lastQuantInTensors int
	lastQuantType      core.DType
}

func newFakeRunner(t *testing.T) *fakeRunner {
	return &fakeRunner{t: t, baselinePPL: 11.5, candPPL: 11.7245, candKLD: 0.0125}
}

func (f *fakeRunner) Run(ctx context.Context, iv orchestrate.Invocation) (orchestrate.Result, error) {
	f.t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := iv.Validate(); err != nil {
		return orchestrate.Result{}, err
	}
	if len(iv.Argv) > 0 && (iv.Argv[0] == "--help" || iv.Argv[0] == "--version") {
		if iv.Tool == orchestrate.ToolLlamaQuantize {
			help := fakeQuantizeHelp
			if f.noPure {
				help = strings.ReplaceAll(help, " --pure\n", "")
			}
			return orchestrate.Result{Stdout: help}, nil
		}
		return orchestrate.Result{Stdout: fakePerplexityHelp}, nil
	}
	switch iv.Tool {
	case orchestrate.ToolLlamaQuantize:
		in, out, dtype, dry, pure := parseQuantizeArgv(iv.Argv)
		f.quantRuns++
		f.lastQuantType = dtype
		f.lastQuantInTensors = countGGUFTensors(f.t, in)
		if !dry {
			fakeQuantizeGGUF(f.t, in, out, dtype, pure)
		}
		return orchestrate.Result{Stdout: "fake quantize ok\n"}, nil
	case orchestrate.ToolPerplexity:
		model, logits, compare := parsePerplexityArgv(iv.Argv)
		_ = model
		f.pplRuns++
		f.lastPPLArgv = append([]string(nil), iv.Argv...)
		if compare {
			if f.omitKLD {
				return orchestrate.Result{Stdout: fmt.Sprintf("Final estimate: PPL = %.4f +/- 0.1000\n", f.candPPL)}, nil
			}
			out := fmt.Sprintf("Final estimate: PPL = %.4f +/- 0.1000\nmean KLD: %.6f\n", f.candPPL, f.candKLD)
			if !f.noP95 {
				out += fmt.Sprintf("p95 KLD: %.6f\n", 2*f.candKLD)
			}
			return orchestrate.Result{Stdout: out}, nil
		}
		if err := os.WriteFile(logits, []byte("fake logits"), 0o644); err != nil {
			return orchestrate.Result{}, err
		}
		return orchestrate.Result{Stdout: fmt.Sprintf("Final estimate: PPL = %.4f +/- 0.1000\n", f.baselinePPL)}, nil
	}
	return orchestrate.Result{}, fmt.Errorf("fake runner: unknown tool %q", iv.Tool)
}

func parseQuantizeArgv(argv []string) (in, out string, dtype core.DType, dry, pure bool) {
	var pos []string
	valueFlags := map[string]bool{
		"--imatrix": true, "--tensor-type": true,
		"--output-tensor-type": true, "--token-embedding-type": true,
	}
	for i := 0; i < len(argv); i++ {
		a := argv[i]
		if strings.HasPrefix(a, "-") {
			if valueFlags[a] {
				i++
				continue
			}
			if a == "--dry-run" {
				dry = true
			}
			if a == "--pure" {
				pure = true
			}
			continue
		}
		pos = append(pos, a)
	}
	if len(pos) < 3 {
		panic("fake quantize: bad argv " + fmt.Sprint(argv))
	}
	return pos[0], pos[1], core.DType(pos[2]), dry, pure
}

func countGGUFTensors(t *testing.T, path string) int {
	t.Helper()
	s, err := tensorbank.OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	f, err := tensorbank.Parse(s)
	if err != nil {
		t.Fatal(err)
	}
	return len(f.Tensors)
}

func parsePerplexityArgv(argv []string) (model, logits string, compare bool) {
	for i := 0; i < len(argv); i++ {
		switch argv[i] {
		case "-m", "--model":
			model = argv[i+1]
			i++
		case "--kl-divergence-base":
			logits = argv[i+1]
			i++
		case "--kl-divergence":
			compare = true
		}
	}
	return model, logits, compare
}

// --- shared test fixtures -----------------------------------------------------

type fixture struct {
	t        *testing.T
	dir      string
	src      string
	calibDir string
	outDir   string
	stateDir string
	budget   uint64
	runner   *fakeRunner
}

func newFixture(t *testing.T, budget uint64) *fixture {
	t.Helper()
	dir := t.TempDir()
	f := &fixture{t: t, dir: dir}
	f.src = filepath.Join(dir, "tiny.gguf")
	if err := writeGGUF(f.src, testTensors()); err != nil {
		t.Fatal(err)
	}
	f.calibDir = filepath.Join(dir, "corpus")
	if err := os.MkdirAll(f.calibDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(f.calibDir, "docs.txt"), []byte("the quick brown fox jumps over the lazy dog\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	f.outDir = filepath.Join(dir, "out")
	f.stateDir = filepath.Join(dir, "state")
	f.runner = newFakeRunner(t)
	f.budget = budget
	return f
}

func (f *fixture) plan(runID string, gates string) *state.Run {
	f.t.Helper()
	g, err := ParseGates(gates)
	if err != nil {
		f.t.Fatal(err)
	}
	r, err := Plan(PlanOptions{
		SourcePath:      f.src,
		StateDir:        f.stateDir,
		OutputDir:       f.outDir,
		CalibrationDir:  f.calibDir,
		LlamaQuantize:   f.src,
		LlamaPerplexity: f.src,
		BudgetBytes:     f.budget,
		Threads:         2,
		CtxSize:         512,
		Gates:           g,
		RunID:           runID,
		Now:             time.Now(),
		Stdout:          io.Discard,
	})
	if err != nil {
		f.t.Fatal(err)
	}
	return r
}

func (f *fixture) engine(r *state.Run) *Engine {
	f.t.Helper()
	return f.rawEngine(r)
}

func (f *fixture) rawEngine(r *state.Run) *Engine {
	f.t.Helper()
	e, err := NewEngine(state.Store{Dir: f.stateDir}, r, f.runner, io.Discard)
	if err != nil {
		f.t.Fatal(err)
	}
	return e
}

// --- tests ---------------------------------------------------------------------

func TestParseGates(t *testing.T) {
	gates, err := ParseGates("mean-kld=0.05, p95-kld=0.2")
	if err != nil {
		t.Fatal(err)
	}
	if len(gates) != 2 {
		t.Fatalf("got %d gates", len(gates))
	}
	if gates[0].Metric != core.MetricKLD || gates[0].MaxDelta != 0.05 {
		t.Fatalf("gate0 = %+v", gates[0])
	}
	if gates[1].MaxAbsolute != 0.2 {
		t.Fatalf("gate1 = %+v", gates[1])
	}
	for _, bad := range []string{"kld=1", "mean-kld=x", "mean-kld=-1", "foo=1"} {
		if _, err := ParseGates(bad); err == nil {
			t.Errorf("expected error for %q", bad)
		}
	}
	if g, err := ParseGates(""); err != nil || len(g) != 0 {
		t.Errorf("empty gates: %v %v", g, err)
	}
	cvar, err := ParseGates("cvar-kld=0.4")
	if err != nil {
		t.Fatal(err)
	}
	if len(cvar) != 1 || cvar[0].Metric != core.MetricCVaRKLD || cvar[0].MaxAbsolute != 0.4 {
		t.Fatalf("cvar gate = %+v", cvar)
	}
}

func validGGUF(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "valid.gguf")
	if err := writeGGUF(p, testTensors()); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestPlanValidationErrors(t *testing.T) {
	dir := t.TempDir()
	junk := filepath.Join(dir, "junk.gguf")
	os.WriteFile(junk, []byte("not a gguf"), 0o644)
	calib := filepath.Join(dir, "corpus")
	os.MkdirAll(calib, 0o755)
	os.WriteFile(filepath.Join(calib, "a.txt"), []byte("hello world hello world"), 0o644)

	cases := []struct {
		name string
		opts PlanOptions
		want string
	}{
		{"missing src", PlanOptions{StateDir: dir, CalibrationDir: calib, LlamaQuantize: junk, LlamaPerplexity: junk, BudgetBytes: 1000}, "-src"},
		{"missing tools", PlanOptions{SourcePath: junk, StateDir: dir, CalibrationDir: calib, BudgetBytes: 1000}, "-quantize"},
		{"no budget", PlanOptions{SourcePath: validGGUF(t), StateDir: dir, CalibrationDir: calib, LlamaQuantize: validGGUF(t), LlamaPerplexity: validGGUF(t)}, "budget"},
		{"bad gguf", PlanOptions{SourcePath: junk, StateDir: dir, CalibrationDir: calib, LlamaQuantize: junk, LlamaPerplexity: junk, BudgetBytes: 1000}, "GGUF"},
		{"empty calib", PlanOptions{SourcePath: validGGUF(t), StateDir: dir, CalibrationDir: filepath.Join(dir, "empty"), LlamaQuantize: validGGUF(t), LlamaPerplexity: validGGUF(t), BudgetBytes: 1000}, "calibration"},
	}
	for _, tc := range cases {
		_, err := Plan(tc.opts)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want error containing %q", tc.name, err, tc.want)
		}
	}
}

func TestPlanTargetBPW(t *testing.T) {
	f := newFixture(t, 0)
	f.budget = 0
	g, err := ParseGates("")
	if err != nil {
		t.Fatal(err)
	}
	r, err := Plan(PlanOptions{
		SourcePath:      f.src,
		StateDir:        f.stateDir,
		CalibrationDir:  f.calibDir,
		LlamaQuantize:   f.src,
		LlamaPerplexity: f.src,
		TargetBPW:       6.0,
		Threads:         2,
		CtxSize:         512,
		Gates:           g,
		Stdout:          io.Discard,
		RunID:           "bpw",
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Config.BudgetBytes == 0 {
		t.Fatal("budget not derived from target bpw")
	}
	if got := r.Config.CalibrationCorpus; !strings.HasSuffix(got, "calibration.txt") {
		t.Fatalf("calibration corpus = %q", got)
	}
	if _, err := os.Stat(filepath.Join(f.calibDir, "manifest.json")); err != nil {
		t.Fatal("corpus manifest not written")
	}
}

func TestResumeAllStages(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.plan("full", "mean-kld=0.5")
	e := f.engine(r)
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	store := state.Store{Dir: f.stateDir}
	done, err := store.Load("full")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := done.NextStage(); ok {
		t.Fatalf("run incomplete: %+v", done.Completed)
	}
	// Final artifact + sidecar + report.
	outs, err := os.ReadDir(f.outDir)
	if err != nil {
		t.Fatal(err)
	}
	var ggufs, sidecars, reports int
	for _, o := range outs {
		switch {
		case strings.HasSuffix(o.Name(), ".gguf"):
			ggufs++
		case strings.HasSuffix(o.Name(), ".oid-plan.json"):
			sidecars++
		case strings.HasSuffix(o.Name(), ".report.json"):
			reports++
		}
	}
	if ggufs != 1 || sidecars != 1 || reports != 1 {
		t.Fatalf("outputs: %d gguf, %d sidecar, %d report", ggufs, sidecars, reports)
	}
	// The emitted model parses and holds quantized tensors.
	var finalPath string
	for _, o := range outs {
		if strings.HasSuffix(o.Name(), ".gguf") {
			finalPath = filepath.Join(f.outDir, o.Name())
		}
	}
	s, err := tensorbank.OpenSource(finalPath)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	pf, err := tensorbank.Parse(s)
	if err != nil {
		t.Fatal(err)
	}
	quant := false
	for _, ti := range pf.Tensors {
		if ti.DType.IsQuant() {
			quant = true
		}
	}
	if !quant {
		t.Fatal("emitted model has no quantized tensors")
	}
	// Measurements recorded with provenance.
	if !hasMeasurementPub(done, "baseline", core.MetricPerplexity) {
		t.Fatal("baseline perplexity missing")
	}
	if !hasMeasurementPub(done, done.BestProfileID, core.MetricKLD) {
		t.Fatal("candidate KLD missing")
	}
	// Scratch cleanup.
	if _, err := os.Stat(filepath.Join(done.Config.WorkDir, "anchors")); !os.IsNotExist(err) {
		t.Fatalf("anchor scratch not cleaned: %v", err)
	}
	// Baseline measured exactly once.
	if f.runner.pplRuns < 2 {
		t.Fatalf("expected baseline+candidate evals, got %d", f.runner.pplRuns)
	}
	baseRuns := f.runner.pplRuns
	// Resume of a complete run is a no-op and remeasures nothing.
	e2 := f.engine(done)
	if err := e2.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if f.runner.pplRuns != baseRuns {
		t.Fatalf("complete-run resume remeasured: %d -> %d", baseRuns, f.runner.pplRuns)
	}
}

func TestEvaluatePPLOnlyFailsClosed(t *testing.T) {
	f := newFixture(t, 150000)
	f.runner.omitKLD = true
	r := f.plan("nokld", "mean-kld=0.5")
	err := f.engine(r).Resume(context.Background())
	if err == nil || !strings.Contains(err.Error(), "produced no KLD") {
		t.Fatalf("error = %v", err)
	}
	if !strings.Contains(err.Error(), "stage evaluate") {
		t.Fatalf("want evaluate-stage failure, got %v", err)
	}
	store := state.Store{Dir: f.stateDir}
	got, err := store.Load("nokld")
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range got.Completed {
		if s == core.StageEvaluate {
			t.Fatal("StageEvaluate completed on PPL-only candidate")
		}
		if s == core.StageSearch {
			t.Fatal("search ran after PPL-only evaluate")
		}
	}
}

func hasMeasurementPub(r *state.Run, profileID string, metric core.MetricKind) bool {
	for _, m := range r.Measurements {
		if m.ProfileID == profileID && m.Metric == metric {
			return true
		}
	}
	return false
}

func TestStageLimitAndResumeWithoutRemeasure(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.plan("lim", "mean-kld=0.5")
	store := state.Store{Dir: f.stateDir}

	// First resume executes exactly three stages.
	e := f.engine(r)
	e.StageLimit = 3
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	r1, err := store.Load("lim")
	if err != nil {
		t.Fatal(err)
	}
	if len(r1.Completed) != 3 || r1.Completed[2] != core.StageSolve {
		t.Fatalf("completed = %v", r1.Completed)
	}

	// Continue to completion.
	e = f.engine(r1)
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	fullRuns := f.runner.pplRuns
	fullQuant := f.runner.quantRuns

	// Simulate a crash after evaluate: stages beyond quantize are forgotten
	// but measurements survive; resume must not remeasure baseline/candidate.
	r2, err := store.Load("lim")
	if err != nil {
		t.Fatal(err)
	}
	r2.Completed = r2.Completed[:4] // keep assemble..quantize
	r2.BestProfileID = r2.Manifest.ProfileID
	if err := store.Save(r2); err != nil {
		t.Fatal(err)
	}
	e = f.engine(r2)
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if f.runner.pplRuns != fullRuns {
		t.Fatalf("resume remeasured evals: %d -> %d; last argv=%v", fullRuns, f.runner.pplRuns, f.runner.lastPPLArgv)
	}
	if f.runner.quantRuns != fullQuant {
		t.Fatalf("resume re-quantized: %d -> %d", fullQuant, f.runner.quantRuns)
	}
}

func TestEvaluateRegeneratesTruncatedBaselineLogits(t *testing.T) {
	f := newFixture(t, 150000)
	run := f.plan("truncated-logits", "mean-kld=0.5")
	engine := f.engine(run)
	engine.StageLimit = 5 // through evaluate
	if err := engine.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	before := f.runner.pplRuns
	var baselineOnly []core.Measurement
	for _, measurement := range run.Measurements {
		if measurement.ProfileID == "baseline" {
			baselineOnly = append(baselineOnly, measurement)
		}
	}
	run.Measurements = baselineOnly
	run.Completed = run.Completed[:4]
	if err := os.WriteFile(engine.logitsPath(), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	store := state.Store{Dir: f.stateDir}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}
	engine = f.engine(run)
	engine.StageLimit = 1
	if err := engine.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := f.runner.pplRuns - before; got != 2 {
		t.Fatalf("truncated logits were reused: added runs=%d, want baseline recapture + candidate", got)
	}
}

func TestDryRun(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.plan("dry", "mean-kld=0.5")
	e := f.engine(r)
	e.DryRun = true
	var buf bytes.Buffer
	e.Out = &buf
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if f.runner.quantRuns != 0 || f.runner.pplRuns != 0 {
		t.Fatalf("dry-run executed tools: quant=%d ppl=%d", f.runner.quantRuns, f.runner.pplRuns)
	}
	if !strings.Contains(buf.String(), "plan:") {
		t.Fatalf("dry-run printed no plans:\n%s", buf.String())
	}
	// Checkpoint untouched: no stages completed, no artifacts.
	store := state.Store{Dir: f.stateDir}
	r2, err := store.Load("dry")
	if err != nil {
		t.Fatal(err)
	}
	if len(r2.Completed) != 0 {
		t.Fatalf("dry-run persisted completions: %v", r2.Completed)
	}
	if _, err := os.Stat(filepath.Join(r2.Config.WorkDir, "candidate.gguf")); !os.IsNotExist(err) {
		t.Fatal("dry-run wrote a candidate artifact")
	}
	if _, err := os.ReadDir(f.outDir); err == nil {
		t.Fatal("dry-run wrote final artifacts")
	}
}

func TestQuantizeRequiresAdvertisedPureBeforeAnchorExecution(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.plan("nopure", "mean-kld=0.5")
	e := f.engine(r)
	e.StageLimit = 3
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	r, err := (state.Store{Dir: f.stateDir}).Load("nopure")
	if err != nil {
		t.Fatal(err)
	}
	f.runner.noPure = true
	err = f.engine(r).Resume(context.Background())
	if err == nil || !strings.Contains(err.Error(), "--pure") {
		t.Fatalf("missing --pure error = %v", err)
	}
	if f.runner.quantRuns != 0 {
		t.Fatalf("anchor execution started despite missing --pure: %d", f.runner.quantRuns)
	}
}

func TestResumeRejectsChangedSourceBeforeTools(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.plan("source-sha", "mean-kld=0.5")
	e := f.engine(r)
	e.StageLimit = 1
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	// Same shape and valid GGUF, but distinct payload bytes.
	file, err := os.OpenFile(f.src, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteAt([]byte{0x7f}, fileStatSize(t, f.src)-1); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	r, err = (state.Store{Dir: f.stateDir}).Load("source-sha")
	if err != nil {
		t.Fatal(err)
	}
	err = f.engine(r).Resume(context.Background())
	if err == nil || !strings.Contains(err.Error(), "SHA256 changed") {
		t.Fatalf("source replacement error = %v", err)
	}
	if f.runner.quantRuns != 0 || f.runner.pplRuns != 0 {
		t.Fatalf("tools ran after source replacement: quant=%d ppl=%d", f.runner.quantRuns, f.runner.pplRuns)
	}
	if _, err := (state.Store{Dir: f.stateDir}).Load("source-sha"); err != nil {
		t.Fatalf("checkpoint was deleted: %v", err)
	}
}

func fileStatSize(t *testing.T, path string) int64 {
	t.Helper()
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return st.Size()
}

func TestEmitReportContents(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.plan("reporty", "mean-kld=0.5,p95-kld=0.2")
	e := f.engine(r)
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(f.outDir, "tiny-reporty.report.json"))
	if err != nil {
		t.Fatal(err)
	}
	var rep RunReport
	if err := json.Unmarshal(data, &rep); err != nil {
		t.Fatal(err)
	}
	if rep.RunID != "reporty" || rep.Output.Bytes == 0 || rep.Source.Bytes == 0 {
		t.Fatalf("report header incomplete: %+v", rep)
	}
	if len(rep.Gates) != 2 || !rep.GatesPass {
		t.Fatalf("gates = %+v pass=%v", rep.Gates, rep.GatesPass)
	}
	if len(rep.Measurements) == 0 {
		t.Fatal("report has no measurements")
	}
	var sidecar RecipeSidecar
	sd, err := os.ReadFile(filepath.Join(f.outDir, "tiny-reporty.oid-plan.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(sd, &sidecar); err != nil {
		t.Fatal(err)
	}
	if sidecar.ProfileID == "" || len(sidecar.Assignments) == 0 || sidecar.TotalBytes == 0 {
		t.Fatalf("sidecar incomplete: %+v", sidecar)
	}
}

// readReport loads the emitted run report for runID from the fixture out dir.
func (f *fixture) readReport(runID string) RunReport {
	f.t.Helper()
	data, err := os.ReadFile(filepath.Join(f.outDir, "tiny-"+runID+".report.json"))
	if err != nil {
		f.t.Fatal(err)
	}
	var rep RunReport
	if err := json.Unmarshal(data, &rep); err != nil {
		f.t.Fatal(err)
	}
	return rep
}

// TestEmitCrashWindowResume simulates a crash after the final artifact was
// renamed into place but before the checkpoint save (fail-injected by
// blocking the store's tmp path): the destination GGUF, sidecar and report
// already exist, the checkpoint still shows emit incomplete, and a plain
// resume must verify the moved artifact and finish without manual surgery.
func TestEmitCrashWindowResume(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.plan("crash", "mean-kld=0.5")
	store := state.Store{Dir: f.stateDir}

	e := f.engine(r)
	e.StageLimit = 6 // everything except emit
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	r1, err := store.Load("crash")
	if err != nil {
		t.Fatal(err)
	}
	if next, _ := r1.NextStage(); next != core.StageEmit {
		t.Fatalf("next stage = %s, want emit", next)
	}

	// Fail-inject the post-emit checkpoint save: a directory at the store's
	// tmp path makes the atomic write fail AFTER the artifact move.
	cpPath, err := store.Path("crash")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(cpPath+".tmp", 0o755); err != nil {
		t.Fatal(err)
	}
	e = f.engine(r1)
	err = e.Resume(context.Background())
	if err == nil || !strings.Contains(err.Error(), "checkpoint save") {
		t.Fatalf("expected checkpoint save failure, got %v", err)
	}

	// Crash state: final GGUF moved to the destination, sidecar and report
	// already written (they precede the move), checkpoint still at emit.
	dest := filepath.Join(f.outDir, "tiny-crash.gguf")
	if st, err := os.Stat(dest); err != nil || st.Size() == 0 {
		t.Fatalf("destination missing after move: %v", err)
	}
	if _, err := os.Stat(filepath.Join(f.outDir, "tiny-crash.oid-plan.json")); err != nil {
		t.Fatal("sidecar not written before the move")
	}
	if _, err := os.Stat(filepath.Join(f.outDir, "tiny-crash.report.json")); err != nil {
		t.Fatal("report not written before the move")
	}
	if _, err := os.Stat(r1.Artifacts[core.StageQuantize]); !os.IsNotExist(err) {
		t.Fatal("staged candidate not moved")
	}
	r2, err := store.Load("crash")
	if err != nil {
		t.Fatal(err)
	}
	if next, _ := r2.NextStage(); next != core.StageEmit {
		t.Fatalf("checkpoint advanced past emit after failed save: %v", r2.Completed)
	}

	// Resume heals: the pre-existing destination is verified against the
	// exact planned artifact size, the move is not repeated, and the run
	// completes.
	if err := os.Remove(cpPath + ".tmp"); err != nil {
		t.Fatal(err)
	}
	e = f.engine(r2)
	if err := e.Resume(context.Background()); err != nil {
		t.Fatalf("resume after crash failed: %v", err)
	}
	done, err := store.Load("crash")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := done.NextStage(); ok {
		t.Fatalf("run incomplete after healing resume: %v", done.Completed)
	}
	var ggufs int
	outs, _ := os.ReadDir(f.outDir)
	for _, o := range outs {
		if strings.HasSuffix(o.Name(), ".gguf") {
			ggufs++
		}
	}
	if ggufs != 1 {
		t.Fatalf("expected exactly 1 emitted gguf, got %d", ggufs)
	}
}

// TestEmitHardFailsOverBudget proves BudgetBytes caps the complete final
// artifact: when the checkpoint's budget is shrunk below the true artifact
// size, emit fails BEFORE publishing anything (no destination, no sidecar,
// no report).
func TestEmitHardFailsOverBudget(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.plan("tight", "mean-kld=0.5")
	store := state.Store{Dir: f.stateDir}
	e := f.engine(r)
	e.StageLimit = 6
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	r1, err := store.Load("tight")
	if err != nil {
		t.Fatal(err)
	}
	staged := r1.Artifacts[core.StageQuantize]
	if staged == "" {
		t.Fatal("quantize stage left no artifact")
	}
	st, err := os.Stat(staged)
	if err != nil {
		t.Fatal(err)
	}
	r1.Config.BudgetBytes = uint64(st.Size()) - 1
	if err := store.Save(r1); err != nil {
		t.Fatal(err)
	}
	e = f.engine(r1)
	err = e.Resume(context.Background())
	if err == nil || !strings.Contains(err.Error(), "exceeding budget") {
		t.Fatalf("expected budget failure, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(f.outDir, "tiny-tight.gguf")); !os.IsNotExist(err) {
		t.Fatal("over-budget artifact was published")
	}
	if _, err := os.Stat(filepath.Join(f.outDir, "tiny-tight.oid-plan.json")); !os.IsNotExist(err) {
		t.Fatal("sidecar published before budget check")
	}
	if _, err := os.Stat(filepath.Join(f.outDir, "tiny-tight.report.json")); !os.IsNotExist(err) {
		t.Fatal("report published before budget check")
	}
	// The staged artifact survives, so widening the budget resumes cleanly.
	r2, err := store.Load("tight")
	if err != nil {
		t.Fatal(err)
	}
	r2.Config.BudgetBytes = uint64(st.Size())
	if err := store.Save(r2); err != nil {
		t.Fatal(err)
	}
	if err := f.engine(r2).Resume(context.Background()); err != nil {
		t.Fatalf("resume with exact budget failed: %v", err)
	}
}

// TestEmitGateSplitReported drives a mean-low/p95-high candidate end to end:
// the mean-kld gate passes and the p95-kld gate fails in the emitted report.
func TestEmitGateSplitReported(t *testing.T) {
	f := newFixture(t, 150000)
	f.runner.candKLD = 0.15 // mean 0.15, p95 0.30
	r := f.plan("split", "mean-kld=0.5,p95-kld=0.2")
	if err := f.engine(r).Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	rep := f.readReport("split")
	if rep.GatesPass {
		t.Fatal("gates passed despite p95 breach")
	}
	if len(rep.Gates) != 2 {
		t.Fatalf("gates = %+v", rep.Gates)
	}
	mean, p95 := rep.Gates[0], rep.Gates[1]
	if !mean.Measured || !mean.Pass {
		t.Errorf("mean-kld gate = %+v, want measured pass", mean)
	}
	if !p95.Measured || p95.Pass {
		t.Errorf("p95-kld gate = %+v, want measured FAIL", p95)
	}
	if p95.Value < 0.29 || p95.Value > 0.31 {
		t.Errorf("p95 value = %v, want ~0.30", p95.Value)
	}
}

// TestEmitMissingP95FailsClosed proves a p95-kld gate never passes when the
// tool reported no p95 measurement: the gate is evaluated fail-closed.
func TestEmitMissingP95FailsClosed(t *testing.T) {
	f := newFixture(t, 150000)
	f.runner.noP95 = true
	r := f.plan("nop95", "mean-kld=0.5,p95-kld=0.2")
	if err := f.engine(r).Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	rep := f.readReport("nop95")
	if rep.GatesPass {
		t.Fatal("gates passed with p95 measurement missing")
	}
	var p95 *GateReport
	for i := range rep.Gates {
		if rep.Gates[i].Metric == core.MetricP95KLD {
			p95 = &rep.Gates[i]
		}
	}
	if p95 == nil {
		t.Fatalf("p95 gate missing from report: %+v", rep.Gates)
	}
	if p95.Measured || p95.Pass {
		t.Errorf("p95 gate = %+v, want unmeasured fail", p95)
	}
}

// TestEmitSkipsUnmeasuredWorstDomain: a worst-domain gate with no holdout
// measurement is omitted, not failed. Mean and p95 still fail-closed.
func TestEmitSkipsUnmeasuredWorstDomain(t *testing.T) {
	f := newFixture(t, 150000)
	g, err := ParseGates("mean-kld=0.5,p95-kld=1.0,worst-domain-kld=0.5")
	if err != nil {
		t.Fatal(err)
	}
	r, err := Plan(PlanOptions{
		SourcePath:      f.src,
		StateDir:        f.stateDir,
		OutputDir:       f.outDir,
		CalibrationDir:  f.calibDir,
		LlamaQuantize:   f.src,
		LlamaPerplexity: f.src,
		BudgetBytes:     f.budget,
		Threads:         2,
		CtxSize:         512,
		Gates:           g,
		RunID:           "nodomain",
		Now:             time.Now(),
		Stdout:          io.Discard,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.engine(r).Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	rep := f.readReport("nodomain")
	if !rep.GatesPass {
		t.Fatalf("gates failed: %+v", rep.Gates)
	}
	if rep.GatesConfigured != 2 {
		t.Fatalf("gatesConfigured = %d, want 2 (worst-domain omitted)", rep.GatesConfigured)
	}
	for _, gr := range rep.Gates {
		if gr.Metric == core.MetricWorstDomainKLD {
			t.Fatalf("unmeasured worst-domain gate was reported: %+v", gr)
		}
		if !gr.Measured || !gr.Pass {
			t.Errorf("required gate = %+v, want measured pass", gr)
		}
	}
}

// TestEmitNoGatesConfigured covers the explicit opt-out ("-gates none"):
// GatesPass is vacuously true and GatesConfigured says why, explicitly. An
// empty gate list without opt-out now resolves to the effort profile
// defaults; see TestGateDefaultsFromEffort.
func TestEmitNoGatesConfigured(t *testing.T) {
	f := newFixture(t, 150000)
	r, err := Plan(PlanOptions{
		SourcePath:      f.src,
		StateDir:        f.stateDir,
		OutputDir:       f.outDir,
		CalibrationDir:  f.calibDir,
		LlamaQuantize:   f.src,
		LlamaPerplexity: f.src,
		BudgetBytes:     f.budget,
		Threads:         2,
		CtxSize:         512,
		GatesOptOut:     true,
		RunID:           "nogates",
		Now:             time.Now(),
		Stdout:          io.Discard,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.engine(r).Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	rep := f.readReport("nogates")
	if rep.GatesConfigured != 0 {
		t.Fatalf("gatesConfigured = %d, want 0", rep.GatesConfigured)
	}
	if !rep.GatesPass || len(rep.Gates) != 0 {
		t.Fatalf("gates = %+v pass=%v", rep.Gates, rep.GatesPass)
	}
}

// TestIQVariantQuantizesTrimmedSubset: IQ2_S --pure on the full source
// aborts in llama.cpp when any 2D tensor lacks an imatrix row. The quantize
// stage must trim to the keep set first so llama-quantize only sees covered
// tensors.
func TestIQVariantQuantizesTrimmedSubset(t *testing.T) {
	f := newFixture(t, 150000)
	r := f.plan("iqsubset", "mean-kld=1.0")
	im := filepath.Join(f.dir, "im.bin")
	if err := os.WriteFile(im, []byte("imatrix"), 0o644); err != nil {
		t.Fatal(err)
	}
	r.Config.ImatrixPath = im
	store := state.Store{Dir: f.stateDir}
	if err := store.Save(r); err != nil {
		t.Fatal(err)
	}
	e := f.rawEngine(r)
	e.StageLimit = 1
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	r1, err := store.Load("iqsubset")
	if err != nil {
		t.Fatal(err)
	}
	b, ok := core.DTypeIQ2_S.ExactBytes(256 * 256)
	if !ok {
		t.Fatal("IQ2_S geometry")
	}
	r1.Config.ImatrixPath = im
	r1.Manifest = &core.SelectionManifest{
		ProfileID: "iq-subset",
		Options: []core.TensorOption{
			{TensorName: "blk.0.attn_q.weight", Target: core.DTypeIQ2_S, Bytes: b},
		},
	}
	if err := store.Save(r1); err != nil {
		t.Fatal(err)
	}
	srcTensors := countGGUFTensors(t, f.src)
	if srcTensors < 2 {
		t.Fatalf("source tensor count %d, want a full model", srcTensors)
	}
	e = f.rawEngine(r1)
	keep := map[string]struct{}{"blk.0.attn_q.weight": {}}
	keepFor := func(core.DType) (map[string]struct{}, error) { return keep, nil }
	if err := e.runTrimmedAnchorJobs(context.Background(), []core.DType{core.DTypeIQ2_S}, keepFor, e.variantsDir(), "meta.json"); err != nil {
		t.Fatal(err)
	}
	if f.runner.lastQuantType != core.DTypeIQ2_M {
		t.Fatalf("quantized %s, want IQ2_M", f.runner.lastQuantType)
	}
	if f.runner.lastQuantInTensors != 1 {
		t.Fatalf("IQ2_S quantize saw %d tensors, want 1 (trimmed subset, not the %d-tensor source)",
			f.runner.lastQuantInTensors, srcTensors)
	}
}

func TestSolveBudgetEqualsPayloadBudget(t *testing.T) {
	bank := &core.TensorBank{Tensors: []core.TensorDesc{
		{Name: "a", DType: core.DTypeF16, Shape: []uint64{256, 256}, Length: 131072, Elements: 65536},
	}}
	e := &Engine{Run: &state.Run{
		Bank:   bank,
		Config: state.RunConfig{BudgetBytes: 1 << 20},
	}}
	if got, want := e.solveBudget(), e.payloadBudget(); got != want {
		t.Fatalf("solveBudget = %d, want payloadBudget %d", got, want)
	}
}
