package quantize

import (
	"fmt"
	"os"
	"strings"
	"testing"
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

func TestParseIMatrixStats(t *testing.T) {
	m := ParseIMatrixStats("token_embd.weight  ZD score: 8.2\nblk.0.attn_v.weight  zd: 6.1\n")
	if m["token_embd.weight"] != 8.2 {
		t.Fatalf("got %#v", m)
	}
	if m["blk.0.attn_v.weight"] != 6.1 {
		t.Fatalf("got %#v", m)
	}
}
