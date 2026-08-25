package runtimes

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func TestParseToolFlags(t *testing.T) {
	help := `usage: llama-quantize [--help] [--imatrix]
  --allow-requantize
  --keep-split
  -h, --help
`
	got := parseToolFlags(help)
	for _, want := range []string{"--allow-requantize", "--imatrix", "--keep-split", "--help"} {
		if !ToolHasFlag(got, want) {
			t.Errorf("missing %s in %v", want, got)
		}
	}
	if ToolHasFlag(got, "-quantize") {
		t.Errorf("should not parse -quantize from llama-quantize: %v", got)
	}
	if ToolHasFlag(got, "--dry-run") {
		t.Fatal("unexpected --dry-run")
	}
}

func TestParseQuantizeSize(t *testing.T) {
	tests := []struct {
		output string
		want   int64
	}{
		{"llama_model_quantize_internal: quant size = 1234567 bytes", 1234567},
		{"llama_model_quantize_internal: quant size  =  4685.30 MiB (4.89 BPW)", 4912893133},
		{"quant size = 1.5 GB", 1500000000},
		{"quant size = 1 MiB\nquant size = 2 MiB", 2 << 20},
	}
	for _, tt := range tests {
		got, ok := ParseQuantizeSize(tt.output)
		if !ok || got != tt.want {
			t.Errorf("ParseQuantizeSize(%q) = %d, %v; want %d, true", tt.output, got, ok, tt.want)
		}
	}
	if _, ok := ParseQuantizeSize("model size = 12 MiB"); ok {
		t.Fatal("model size must not be mistaken for quant size")
	}
}

func TestProbeToolsFindsPerplexitySibling(t *testing.T) {
	dir := t.TempDir()
	name := func(base string) string {
		if runtime.GOOS == "windows" {
			return base + ".exe"
		}
		return base
	}
	build := func(pkg, binary string) string {
		t.Helper()
		path := filepath.Join(dir, name(binary))
		cmd := exec.Command("go", "build", "-o", path, "github.com/openinfer/openinfer-studio/tests/"+pkg)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("build %s: %v\n%s", pkg, err, out)
		}
		return path
	}
	server := build("fakeserver", "llama-server")
	perplexity := build("fakeperplexity", "llama-perplexity")
	snapshot := probeTools(server)
	if !snapshot.Perplexity.Present || snapshot.Perplexity.Path != perplexity {
		t.Fatalf("perplexity tool = %+v", snapshot.Perplexity)
	}
	for _, flag := range []string{"--model", "--file", "--kl-divergence-base", "--kl-divergence", "--chunks"} {
		if !ToolHasFlag(snapshot.Perplexity.Flags, flag) {
			t.Errorf("perplexity flags missing %s: %v", flag, snapshot.Perplexity.Flags)
		}
	}
}
