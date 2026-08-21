package pipeline

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"quantlab/orchestrate"
)

func TestParseQwenKLDOutputFixture(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate test fixture")
	}
	data, err := os.ReadFile(filepath.Join(filepath.Dir(thisFile), "testdata", "qwen-kld.txt"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := orchestrate.ParseEvalMetrics(string(data))
	if err != nil {
		t.Fatal(err)
	}
	if !m.HasKLD || !m.HasMeanKLD || !m.HasP95 || !m.HasMax ||
		m.MeanKLD != 0.0125 || m.MaxKLD != 0.03 || m.P95KLD != 0.025 {
		t.Fatalf("KLD = %+v", m)
	}
	if !m.HasRMS || m.RMSDeltaP != 0.001 || !m.HasSameTop || m.SameTop != 0.991 {
		t.Fatalf("auxiliary metrics = %+v", m)
	}
}
