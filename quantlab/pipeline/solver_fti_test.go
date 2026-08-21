package pipeline

import (
	"os"
	"path/filepath"
	"testing"

	"quantlab/profile"
	"quantlab/state"
)

func TestApplySolverFTISharpensWithoutDisk(t *testing.T) {
	dir := t.TempDir()
	orig := []float32{1, 1, 1, 8}
	name := "blk.0.attn_q.weight"
	stats := map[string]profile.ImatrixStats{
		name: {Mean: 2.75, Values: append([]float32(nil), orig...)},
	}
	e := &Engine{Run: &state.Run{Config: state.RunConfig{Effort: EffortProfiled, WorkDir: dir}}}
	e.applySolverFTI(stats)

	got := stats[name].Values
	want := profile.SharpenValues(orig, profile.DefaultFTIPower)
	if len(got) != len(want) {
		t.Fatalf("values len %d, want %d", len(got), len(want))
	}
	if got[3] <= orig[3] {
		t.Fatalf("peak %v should grow from %v", got[3], orig[3])
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("values[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	est := profile.NewFallbackEstimator(stats)
	if est == nil || !est.HasImportance(name) {
		t.Fatal("estimator should consume the sharpened map")
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("solver FTI wrote disk: %v", namesOf(entries))
	}

	fastStats := map[string]profile.ImatrixStats{
		name: {Mean: 2.75, Values: append([]float32(nil), orig...)},
	}
	fast := &Engine{Run: &state.Run{Config: state.RunConfig{Effort: EffortFast, WorkDir: dir}}}
	fast.applySolverFTI(fastStats)
	if fastStats[name].Values[3] != orig[3] {
		t.Fatalf("fast solver FTI changed peak %v", fastStats[name].Values[3])
	}

	optOut := map[string]profile.ImatrixStats{
		name: {Mean: 2.75, Values: append([]float32(nil), orig...)},
	}
	no := &Engine{
		Run:   &state.Run{Config: state.RunConfig{Effort: EffortProfiled, WorkDir: dir}},
		Extra: ExtraConfig{NoFTI: true},
	}
	no.applySolverFTI(optOut)
	if optOut[name].Values[3] != orig[3] {
		t.Fatalf("NoFTI still sharpened peak %v", optOut[name].Values[3])
	}

	ftiPath := filepath.Join(dir, "fti-imatrix.gguf")
	if err := os.WriteFile(ftiPath, []byte("not-a-real-gguf"), 0o600); err != nil {
		t.Fatal(err)
	}
	skip := map[string]profile.ImatrixStats{
		name: {Mean: 2.75, Values: append([]float32(nil), orig...)},
	}
	fileFTI := &Engine{
		Run:   &state.Run{Config: state.RunConfig{Effort: EffortProfiled, WorkDir: dir}},
		Extra: ExtraConfig{FTIImatrixPath: ftiPath},
	}
	fileFTI.applySolverFTI(skip)
	if skip[name].Values[3] != orig[3] {
		t.Fatalf("active FTI GGUF still in-memory-sharpened peak %v", skip[name].Values[3])
	}
}

func namesOf(entries []os.DirEntry) []string {
	out := make([]string, len(entries))
	for i, e := range entries {
		out[i] = e.Name()
	}
	return out
}
