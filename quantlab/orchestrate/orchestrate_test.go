package orchestrate

import (
	"path/filepath"
	"strings"
	"testing"

	"quantlab/core"
)

func TestInvocationValidate(t *testing.T) {
	ok := Invocation{Tool: ToolLlamaQuantize, Path: filepath.Join(t.TempDir(), "llama-quantize"), Argv: []string{"a", "b"}}
	if err := ok.Validate(); err != nil {
		t.Fatalf("valid invocation rejected: %v", err)
	}
	for _, iv := range []Invocation{
		{Tool: "rm", Path: "/bin/rm"},                                                // unknown tool
		{Tool: ToolLlamaQuantize, Path: "relative/path"},                             // not absolute
		{Tool: ToolLlamaQuantize, Path: unixAbs("/bin/x; rm -rf /")},                 // metachars
		{Tool: ToolLlamaQuantize, Path: unixAbs("/bin/x"), Argv: []string{"a\x00b"}}, // NUL in argv
	} {
		if err := iv.Validate(); err == nil {
			t.Errorf("expected error for %+v", iv)
		}
	}
}

func TestQuantizeJobInvocation(t *testing.T) {
	j := QuantizeJob{ProfileID: "p1", InPath: "/in.gguf", OutPath: "/out.gguf", Type: core.DTypeQ4_K, Threads: 8}
	iv, err := j.Invocation(unixAbs("/opt/bin/llama-quantize"))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--threads", "8", "/in.gguf", "/out.gguf", "Q4_K_M"}
	if strings.Join(iv.Argv, " ") != strings.Join(want, " ") {
		t.Fatalf("argv = %v, want %v", iv.Argv, want)
	}
	j.OutPath = j.InPath
	if _, err := j.Invocation(unixAbs("/opt/bin/llama-quantize")); err == nil {
		t.Fatal("same in/out accepted")
	}
	j.OutPath = "/out.gguf"
	j.Type = core.DTypeF16
	if _, err := j.Invocation(unixAbs("/opt/bin/llama-quantize")); err == nil {
		t.Fatal("float quantize type accepted")
	}
}

func TestPerplexityJobInvocation(t *testing.T) {
	j := PerplexityJob{ModelPath: "/m.gguf", TextPath: "/wiki.txt", CtxSize: 512}
	iv, err := j.Invocation(unixAbs("/opt/bin/llama-perplexity"))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"-m", "/m.gguf", "-f", "/wiki.txt", "-c", "512"}
	if strings.Join(iv.Argv, " ") != strings.Join(want, " ") {
		t.Fatalf("argv = %v, want %v", iv.Argv, want)
	}
	j.CtxSize = 0
	if _, err := j.Invocation(unixAbs("/opt/bin/llama-perplexity")); err == nil {
		t.Fatal("zero ctx accepted")
	}
}

func TestParseEvalMetricsPerplexity(t *testing.T) {
	out := "perplexity: calculating...\n\nFinal estimate: PPL = 6.1234 +/- 0.05\n"
	m, err := ParseEvalMetrics(out)
	if err != nil {
		t.Fatal(err)
	}
	if !m.HasPPL || m.Perplexity < 6.1233 || m.Perplexity > 6.1235 {
		t.Fatalf("ppl = %v (has=%v), want ~6.1234", m.Perplexity, m.HasPPL)
	}
	if _, err := ParseEvalMetrics("no estimate here"); err == nil {
		t.Fatal("garbage output parsed")
	}
}
