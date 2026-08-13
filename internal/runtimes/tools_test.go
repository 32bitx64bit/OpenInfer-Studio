package runtimes

import "testing"

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
