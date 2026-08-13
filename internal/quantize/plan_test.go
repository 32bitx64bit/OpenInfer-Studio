package quantize

import (
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
	"--show-statistics",
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

func TestPlanIMatrixAndCombine(t *testing.T) {
	args, err := PlanIMatrix(Request{Chunks: 20, GPULayers: 99, ParseSpecial: true, Threads: 4},
		allFlags, "/m.gguf", "/cal.txt", "/o.gguf")
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	for _, w := range []string{"-m", "/m.gguf", "-f", "/cal.txt", "-o", "/o.gguf", "--chunks", "20", "--parse-special"} {
		if !strings.Contains(joined, w) {
			t.Errorf("missing %s in %v", w, args)
		}
	}
	comb, err := PlanCombine(allFlags, []string{"/a.gguf", "/b.gguf"}, "/c.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if !containsArg(comb, "--in-file") {
		t.Fatalf("combine argv: %v", comb)
	}
	if _, err := PlanCombine([]string{"-o"}, []string{"/a", "/b"}, "/c"); err == nil {
		t.Fatal("combine should require --in-file")
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
