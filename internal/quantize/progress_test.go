package quantize

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

func TestParseProgressLine(t *testing.T) {
	c, tot, ok := ParseProgressLine("[ 12/291] blk.0.attn_q.weight")
	if !ok || c != 12 || tot != 291 {
		t.Fatalf("quantize progress: %d/%d ok=%v", c, tot, ok)
	}
	c, tot, ok = ParseProgressLine("chunk 3 / 100 ppl = 1.2")
	if !ok || c != 3 || tot != 100 {
		t.Fatalf("imatrix chunk: %d/%d ok=%v", c, tot, ok)
	}
	c, tot, ok = ParseProgressLine("saved imatrix with 20 chunks")
	if !ok || c != 20 {
		t.Fatalf("saved chunks: %d/%d ok=%v", c, tot, ok)
	}
	if _, _, ok := ParseProgressLine("hello"); ok {
		t.Fatal("noise should not parse")
	}
	if fraction(5, 10, 0) != 0.5 {
		t.Fatal("fraction")
	}
}

func TestSplitTerminalLinesHandlesCarriageReturnProgress(t *testing.T) {
	scanner := bufio.NewScanner(strings.NewReader("[ 1/3] one\r[ 2/3] two\r\n[ 3/3] three\n"))
	scanner.Split(splitTerminalLines)
	var lines []string
	for scanner.Scan() {
		if line := strings.TrimSpace(scanner.Text()); line != "" {
			lines = append(lines, line)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if len(lines) != 3 || lines[1] != "[ 2/3] two" {
		t.Fatalf("terminal lines = %#v", lines)
	}
}

func TestCommandProgressTrackerEstimatesSilentIMatrixPasses(t *testing.T) {
	start := time.Unix(1000, 0)
	tracker := newCommandProgressTracker(start)
	if sample, ok := tracker.Observe("compute_imatrix: computing over 169 chunks, n_ctx=4096", start.Add(90*time.Second)); !ok || sample.Total != 169 {
		t.Fatalf("total sample = %+v ok=%v", sample, ok)
	}
	first, ok := tracker.Observe("87.07 seconds per pass - ETA 4 hours 5.23 minutes", start.Add(180*time.Second))
	if !ok || first.Current != 1 || first.ETASeconds < 14000 {
		t.Fatalf("first timed sample = %+v ok=%v", first, ok)
	}
	later, ok := tracker.Estimate(start.Add(180*time.Second + 870*time.Second))
	if !ok || later.Current != 10 || later.Total != 169 || later.Progress < 0.064 || later.Progress > 0.066 {
		t.Fatalf("estimated sample = %+v ok=%v", later, ok)
	}
	if !later.Estimated || later.ETASeconds <= 0 {
		t.Fatalf("expected estimated ETA: %+v", later)
	}
}

func TestStageRangeAndOverallETA(t *testing.T) {
	j := &Job{Kind: KindAdaptiveQuantize, Request: Request{Effort: "deep"}}
	repairStart, repairEnd := stageRange(j, "repair_source")
	if repairStart != 0 || repairEnd != 0.02 {
		t.Fatalf("repair range = %v..%v", repairStart, repairEnd)
	}
	start, end := stageRange(j, "imatrix")
	if start != 0.02 || end != 0.05 {
		t.Fatalf("imatrix range = %v..%v", start, end)
	}
	overall := start + (end-start)*0.5
	eta := overallETA(3600, overall, end)
	if eta <= 3600 {
		t.Fatalf("overall ETA %d should include later stages", eta)
	}
	if math.Abs(overall-0.035) > 1e-9 {
		t.Fatalf("imatrix mid-stage progress = %v, want 0.035", overall)
	}
	qStart, qEnd := stageRange(j, "quantize")
	if qStart != 0.20 || qEnd != 0.65 {
		t.Fatalf("quantize range = %v..%v", qStart, qEnd)
	}
	anchorStart, anchorEnd := stageRange(j, "anchor")
	solveStart, solveEnd := stageRange(j, "solve")
	searchStart, searchEnd := stageRange(j, "search")
	finalStart, finalEnd := stageRange(j, "finalize")
	if anchorStart != 0.10 || anchorEnd != 0.15 || solveStart != 0.15 || solveEnd != 0.20 ||
		searchStart != 0.92 || searchEnd != 0.93 || finalStart != 0.93 || finalEnd != 1 {
		t.Fatalf("quantlab stage ranges = anchor %v..%v solve %v..%v search %v..%v finalize %v..%v",
			anchorStart, anchorEnd, solveStart, solveEnd, searchStart, searchEnd, finalStart, finalEnd)
	}
}

func TestStageTextQuantlabLabels(t *testing.T) {
	cases := []struct {
		kind, stage, want string
	}{
		{KindAdaptiveQuantize, "imatrix", "Building importance matrix"},
		{KindAdaptiveQuantize, "analyze", "Analyzing source tensors"},
		{KindAdaptiveQuantize, "anchor", "Planning anchors"},
		{KindAdaptiveQuantize, "solve", "Solving bit allocation"},
		{KindAdaptiveQuantize, "quantize", "Quantizing anchors"},
		{KindAdaptiveQuantize, "validate", "Validating against source (KLD)"},
		{KindAdaptiveQuantize, "search", "Preparing output"},
		{KindAdaptiveQuantize, "finalize", "Publishing model"},
		{KindQuantize, "quantize", "Quantizing weights"},
		{"", "", "Running"},
	}
	for _, c := range cases {
		if got := StageText(c.kind, c.stage); got != c.want {
			t.Fatalf("StageText(%q,%q) = %q, want %q", c.kind, c.stage, got, c.want)
		}
	}
}

func TestStepCounter(t *testing.T) {
	cases := []struct {
		count int
		frac  float64
		cur   int
		tot   int
	}{
		{1, 0.25, 1, 4},
		{2, 0.5, 2, 4},
		{4, 1.0, 4, 4},
		{1, 0.01, 1, 100},
		{0, 0, 0, 0},
		{3, 0, 3, 0},
	}
	for _, c := range cases {
		cur, tot := stepCounter(c.count, c.frac)
		if cur != c.cur || tot != c.tot {
			t.Fatalf("stepCounter(%d, %v) = %d/%d, want %d/%d", c.count, c.frac, cur, tot, c.cur, c.tot)
		}
	}
}

func TestRemapQuantizeProgress(t *testing.T) {
	if f := remapQuantizeProgress("anchor Q4_K", 0.5); math.Abs(f-0.25) > 1e-9 {
		t.Fatalf("anchor remap = %v, want 0.25", f)
	}
	if f := remapQuantizeProgress("assembling candidate", 0.5); math.Abs(f-0.75) > 1e-9 {
		t.Fatalf("assembly remap = %v, want 0.75", f)
	}
	// Anchors fill the first half, assembly the second: the boundary is
	// monotonic (last anchor complete does not exceed assembly start).
	if f1, f2 := remapQuantizeProgress("anchor", 1.0), remapQuantizeProgress("assembling", 0.0); f1 > f2 {
		t.Fatalf("anchor complete %v should not exceed assembly start %v", f1, f2)
	}
}

func TestMeasurementStageFractionMonotonic(t *testing.T) {
	var prev float64
	for m := 0; m <= 10; m++ {
		f := measurementStageFraction(m, 1.0)
		if f < prev-1e-9 {
			t.Fatalf("measurement %d: fraction %v decreased from %v", m, f, prev)
		}
		if f > 0.9+1e-9 {
			t.Fatalf("measurement %d: fraction %v exceeds 0.9 cap", m, f)
		}
		prev = f
	}
	if f := measurementStageFraction(0, 0.0); math.Abs(f-0.05) > 1e-9 {
		t.Fatalf("zero measurement fraction = %v, want 0.05", f)
	}
}

func TestStageRangeQuantlabMonotonic(t *testing.T) {
	j := &Job{Kind: KindAdaptiveQuantize, Request: Request{Effort: "fast"}}
	stages := []string{"repair_source", "imatrix", "analyze", "anchor", "solve", "quantize", "validate", "search", "finalize"}
	var prevEnd float64
	for _, s := range stages {
		start, end := stageRange(j, s)
		if start < 0 || end <= start {
			t.Fatalf("stage %s: invalid range %v..%v", s, start, end)
		}
		if start < prevEnd-1e-9 {
			t.Fatalf("stage %s: start %v < previous end %v (progress could decrease)", s, start, prevEnd)
		}
		prevEnd = end
	}
	if math.Abs(prevEnd-1.0) > 1e-9 {
		t.Fatalf("final stage end = %v, want 1.0", prevEnd)
	}
}

func TestSummarizeToolFailure(t *testing.T) {
	tail := `
0.06.019.429 I system_info: n_threads = 15 (n_threads_batch = 15) / 16 | CPU : SSE3 = 1 | AVX512 = 1
0.06.019.433 I compute_imatrix: tokenizing the input ..
0.06.022.065 E compute_imatrix: you need at least 1024 tokens for a context of 512 tokens
0.06.022.065 E compute_imatrix: the data file you provided tokenizes to only 915 tokens
`
	err := summarizeToolFailure(fmt.Errorf("exit status 1"), tail)
	if err == nil {
		t.Fatal("expected error")
	}
	s := err.Error()
	if !strings.Contains(s, "1024") || !strings.Contains(s, "915") {
		t.Fatalf("ugly or incomplete error: %q", s)
	}
	if strings.Contains(s, "SSE3") || strings.Contains(s, "AVX512") {
		t.Fatalf("CPU flags leaked into error: %q", s)
	}

	qerr := summarizeToolFailure(fmt.Errorf("exit status 1"), `
llama_model_quantize_impl: did not find weights for output.weight
llama_model_quantize: failed to quantize: requantizing from type q8_0 is disabled
llama_quantize: failed to quantize model from '/tmp/model.gguf'
load_imatrix: loaded 416 importance matrix entries
`)
	if qerr == nil || !strings.Contains(qerr.Error(), "already quantized") {
		t.Fatalf("requantize error: %v", qerr)
	}

	shape := summarizeToolFailure(fmt.Errorf("exit status 1"), `
0.00.364.404 E llama_model_load: error loading model: check_tensor_dims: tensor 'token_embd.weight' has wrong shape; expected   5120, 248077, got   5120, 248320,      1,      1
0.00.364.409 E llama_model_load_from_file_impl: failed to load model
`)
	if shape == nil {
		t.Fatal("expected shape error")
	}
	s2 := shape.Error()
	if !strings.Contains(s2, "token_embd.weight") || !strings.Contains(s2, "248077") || !strings.Contains(s2, "248320") {
		t.Fatalf("shape error missing dims: %q", s2)
	}
	if strings.Contains(s2, "Qwen") || strings.Contains(s2, "llama_model_load_from_file_impl") {
		t.Fatalf("shape error should be architecture-agnostic and not dump load internals: %q", s2)
	}
}

func TestEnsureCalibrationRefreshes(t *testing.T) {
	env := newQuantEnv(t, true)
	dest, err := env.qm.EnsureCalibration()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dest, []byte("too short\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	again, err := env.qm.EnsureCalibration()
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(again)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) < 8000 {
		t.Fatalf("calibration not refreshed, size=%d", len(b))
	}
}
