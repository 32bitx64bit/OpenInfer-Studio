package quantize

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var allFlags = []string{
	"--allow-requantize", "--leave-output-tensor", "--pure", "--imatrix",
	"--output-tensor-type", "--token-embedding-type", "--tensor-type",
	"--tensor-type-file", "--keep-split", "--override-kv",
	"--model", "-m", "--file", "-f", "--output-file", "-o",
	"--n-gpu-layers", "-ngl", "--chunks", "--chunk", "--parse-special",
	"--process-output", "--no-ppl", "--threads", "-t", "--in-file",
	"--show-statistics", "--ctx-size", "-c", "--prior-weight", "--dry-run",
}

func TestPlanQuantizeArgv(t *testing.T) {
	args, err := PlanQuantize(Request{
		FType: "Q4_K", Threads: 7, AllowRequantize: true, Pure: true,
		KeepSplit: true, OutputTensorType: "q8_0",
	}, allFlags, "/in.gguf", "/out.gguf", "/im.gguf", "")
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{"--allow-requantize", "--pure", "--keep-split", "--imatrix", "/im.gguf", "--output-tensor-type", "q8_0", "/in.gguf", "/out.gguf", "Q4_K_M", "7"} {
		if !containsArg(args, want) && !strings.Contains(joined, want) {
			t.Errorf("missing %q in %v", want, args)
		}
	}
	if args[len(args)-1] != "7" || args[len(args)-2] != "Q4_K_M" {
		t.Errorf("positional tail = %v", args[len(args)-3:])
	}
}

func TestPlanQuantizeGatesUnknownFlags(t *testing.T) {
	args, err := PlanQuantize(Request{FType: "Q8_0", Pure: true, AllowRequantize: true},
		[]string{}, "/in", "/out", "", "")
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range args {
		if strings.HasPrefix(a, "--") {
			t.Errorf("ungated flag %s in %v", a, args)
		}
	}
}

func TestPlanQuantizeIMatrixRequiredByBinary(t *testing.T) {
	_, err := PlanQuantize(Request{FType: "Q4_K_M"}, []string{}, "/in", "/out", "/im.gguf", "")
	if err == nil {
		t.Fatal("expected error when --imatrix is not advertised")
	}
}

func TestPlanQuantizeOptionalPriorWeight(t *testing.T) {
	zero := 0.0
	args, err := PlanQuantize(Request{FType: "Q4_K_M", PriorWeight: &zero}, allFlags, "/in", "/out", "/im", "")
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(args, " "); !strings.Contains(got, "--prior-weight 0") {
		t.Fatalf("advertised prior weight missing: %v", args)
	}
	args, err = PlanQuantize(Request{FType: "Q4_K_M", PriorWeight: &zero}, []string{"--imatrix"}, "/in", "/out", "/im", "")
	if err != nil {
		t.Fatal(err)
	}
	if containsArg(args, "--prior-weight") {
		t.Fatalf("unadvertised prior weight must be omitted: %v", args)
	}
	negative := -1.0
	if _, err := PlanQuantize(Request{FType: "Q4_K_M", PriorWeight: &negative}, allFlags, "/in", "/out", "/im", ""); err == nil {
		t.Fatal("negative prior weight should fail")
	}
}

func TestPlanQuantizeDryRunSyntax(t *testing.T) {
	args, err := PlanQuantizeDryRun(Request{FType: "Q4_K_M", Threads: 3}, allFlags, "/in.gguf", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(args, " "); got != "--dry-run /in.gguf Q4_K_M 3" {
		t.Fatalf("dry-run argv = %q", got)
	}
	if _, err := PlanQuantizeDryRun(Request{FType: "Q4_K_M"}, nil, "/in", "", ""); err == nil {
		t.Fatal("dry run should require an advertised flag")
	}
}

func TestPlanQuantizeTensorFileFallsBackToRepeatedOverrides(t *testing.T) {
	p := filepath.Join(t.TempDir(), "tensor-types.txt")
	if err := os.WriteFile(p, []byte("blk.0.attn_q.weight=iq3_s\nblk.0.ffn_down.weight=q4_k\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	flags := []string{"--tensor-type"}
	args, err := PlanQuantize(Request{FType: "Q4_K_M"}, flags, "/in", "/out", "", p)
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{"--tensor-type blk.0.attn_q.weight=iq3_s", "--tensor-type blk.0.ffn_down.weight=q4_k"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("missing fallback override %q in %v", want, args)
		}
	}
	if containsArg(args, "--tensor-type-file") {
		t.Fatalf("unadvertised tensor-type-file was passed: %v", args)
	}
}

func TestPlanIMatrixAndCombine(t *testing.T) {
	args, err := PlanIMatrix(Request{Chunks: 20, GPULayers: 99, ParseSpecial: true, ProcessOutput: true, Threads: 4},
		allFlags, "/m.gguf", "/cal.txt", "/o.gguf")
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	for _, w := range []string{"-m", "/m.gguf", "-f", "/cal.txt", "-o", "/o.gguf", "--chunks", "20", "--parse-special", "--process-output"} {
		if !strings.Contains(joined, w) {
			t.Errorf("missing %s in %v", w, args)
		}
	}
	if containsArg(args, "--ctx-size") {
		t.Fatalf("quick/explicit chunks should not pass ctx-size: %v", args)
	}
	cont, err := PlanIMatrix(Request{Chunks: 612, ChunkSkip: 10, IMatrixInFile: "/partial.gguf"},
		allFlags, "/m.gguf", "/cal.txt", "/o.gguf")
	if err != nil {
		t.Fatal(err)
	}
	joinedCont := strings.Join(cont, " ")
	for _, w := range []string{"--chunk", "10", "--in-file", "/partial.gguf"} {
		if !strings.Contains(joinedCont, w) {
			t.Errorf("continue argv missing %s in %v", w, cont)
		}
	}
	if _, err := PlanIMatrix(Request{IMatrixInFile: "/partial.gguf"}, []string{"--model", "-m", "--file", "-f", "--output-file", "-o"},
		"/m.gguf", "/cal.txt", "/o.gguf"); err == nil {
		t.Fatal("continue without advertised --in-file should fail")
	}
	comb, err := PlanCombine(allFlags, []string{"/a.gguf", "/b.gguf"}, "/c.gguf", "/m.gguf")
	if err != nil {
		t.Fatal(err)
	}
	nIn := 0
	joinedIn := ""
	gotModel := ""
	for i, a := range comb {
		if a == "--in-file" {
			nIn++
			if i+1 < len(comb) {
				joinedIn = comb[i+1]
			}
		}
		if a == "--model" || a == "-m" {
			if i+1 < len(comb) {
				gotModel = comb[i+1]
			}
		}
	}
	if nIn != 1 || joinedIn != "/a.gguf,/b.gguf" {
		t.Fatalf("combine should pass comma-separated --in-file, got %v", comb)
	}
	if gotModel != "/m.gguf" {
		t.Fatalf("combine should pass --model for argparse, got %v", comb)
	}
	if _, err := PlanCombine(allFlags, []string{"/a.gguf", "/b.gguf"}, "/c.gguf", ""); err == nil {
		t.Fatal("combine should require a model path")
	}
	if _, err := PlanCombine([]string{"-o"}, []string{"/a", "/b"}, "/c", "/m.gguf"); err == nil {
		t.Fatal("combine should require --in-file")
	}
}

func TestPlanIMatrixDefaultChunks(t *testing.T) {
	args, err := PlanIMatrix(Request{}, allFlags, "/m.gguf", "/cal.txt", "/o.gguf")
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--chunks") || !strings.Contains(joined, "200") {
		t.Fatalf("default preset missing --chunks 200 in %v", args)
	}
}

func TestPlanIMatrixCtxSize(t *testing.T) {
	for _, preset := range []string{"thorough", "research"} {
		args, err := PlanIMatrix(Request{CalibrationPreset: preset}, allFlags, "/m.gguf", "/cal.txt", "/o.gguf")
		if err != nil {
			t.Fatal(err)
		}
		joined := strings.Join(args, " ")
		if !strings.Contains(joined, "--ctx-size") || !strings.Contains(joined, "4096") {
			t.Errorf("%s missing --ctx-size 4096 in %v", preset, args)
		}
		wantChunks := "500"
		if preset == "research" {
			wantChunks = "750"
		}
		if !strings.Contains(joined, wantChunks) {
			t.Errorf("%s missing --chunks %s in %v", preset, wantChunks, args)
		}
	}
	args, err := PlanIMatrix(Request{Kind: KindAdaptiveQuantize, CalibrationPreset: "quick"}, allFlags, "/m.gguf", "/cal.txt", "/o.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if !containsArg(args, "--ctx-size") {
		t.Fatalf("adaptive imatrix should pass ctx-size: %v", args)
	}
	gated, err := PlanIMatrix(Request{CalibrationPreset: "thorough"}, []string{"--model", "-m", "--file", "-f", "--output-file", "-o", "--chunks"},
		"/m.gguf", "/cal.txt", "/o.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if containsArg(gated, "--ctx-size") || containsArg(gated, "-c") {
		t.Fatalf("ctx-size must be gated on advertised flags: %v", gated)
	}
}

func TestPlanIMatrixCtxSizeClampsShortFile(t *testing.T) {
	short := filepath.Join(t.TempDir(), "short.txt")
	if err := os.WriteFile(short, []byte(strings.Repeat("token ", 2000)), 0o644); err != nil {
		t.Fatal(err)
	}
	args, err := PlanIMatrix(Request{Kind: KindAdaptiveQuantize}, allFlags, "/m.gguf", short, "/o.gguf")
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "4096") {
		t.Fatalf("short calibration must not request ctx 4096: %v", args)
	}
	if !containsArg(args, "--ctx-size") {
		t.Fatalf("still want a clamped ctx-size: %v", args)
	}
}

func TestIMatrixCtxForFile(t *testing.T) {
	dir := t.TempDir()
	long := filepath.Join(dir, "long.txt")
	// 5 chars/token * 2 * 4096, plus slack
	if err := os.WriteFile(long, []byte(strings.Repeat("abcdefghij", 5000)), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := imatrixCtxForFile(long, 4096); got != 2048 {
		t.Fatalf("10k-token file ctx=%d want 2048 (4 unique windows)", got)
	}
	if got := imatrixCtxForFile("/missing.txt", 4096); got != 4096 {
		t.Fatalf("missing file should keep want, got %d", got)
	}
	huge := filepath.Join(dir, "huge.txt")
	if err := os.WriteFile(huge, []byte(strings.Repeat("abcdefghij", 9000)), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := imatrixCtxForFile(huge, 4096); got != 4096 {
		t.Fatalf("90k-char file ctx=%d want 4096", got)
	}
}

func TestIMatrixTokenEstimateIsConservativeForCJK(t *testing.T) {
	p := filepath.Join(t.TempDir(), "cjk.txt")
	if err := os.WriteFile(p, []byte(strings.Repeat("測", 5000)), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := fileTokenEstimate(p); got != 5000 {
		t.Fatalf("CJK token estimate=%d want 5000", got)
	}
	if got := imatrixCtxForFile(p, 4096); got != 1024 {
		t.Fatalf("CJK calibration ctx=%d want 1024", got)
	}
}

func TestClampIMatrixChunks(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "cal.txt")
	if err := os.WriteFile(p, []byte(strings.Repeat("abcdefghij", 1000)), 0o644); err != nil {
		t.Fatal(err)
	}
	// 2000 tokens, ctx 512 → 3 unique windows → max 12 chunks
	got := clampIMatrixChunks(p, 512, 750)
	if got > 16 || got < 8 {
		t.Fatalf("clamped chunks=%d", got)
	}
	if clampIMatrixChunks("/missing", 512, 250) != 250 {
		t.Fatal("missing file should keep want")
	}
}

func containsArg(args []string, want string) bool {
	for _, a := range args {
		if a == want {
			return true
		}
	}
	return false
}
