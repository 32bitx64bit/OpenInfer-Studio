package pipeline

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"quantlab/anchor"
	"quantlab/core"
	"quantlab/profile"
	"quantlab/qtype"
	"quantlab/state"
	"quantlab/tensorbank"
)

// finiteF16Payloads rewrites every F16 tensor payload of a fixture GGUF with
// small finite values so the exact loss table (which streams the source
// through the reference quantizers) produces usable rows; writeGGUF fills
// payloads with random bytes, most of which decode to NaN/Inf.
func finiteF16Payloads(t *testing.T, path string) {
	t.Helper()
	s, err := tensorbank.OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	f, err := tensorbank.Parse(s)
	if err != nil {
		s.Close()
		t.Fatal(err)
	}
	type span struct {
		off int64
		n   uint64
	}
	var spans []span
	for _, ti := range f.Tensors {
		if ti.DType == core.DTypeF16 {
			spans = append(spans, span{f.PayloadOffset(ti), ti.Elements})
		}
	}
	s.Close()
	fh, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer fh.Close()
	rng := rand.New(rand.NewSource(11))
	for _, sp := range spans {
		buf := make([]byte, 2*sp.n)
		for i := uint64(0); i < sp.n; i++ {
			v := float32(rng.NormFloat64() * 0.05)
			binary.LittleEndian.PutUint16(buf[2*i:], qtype.F16Bits(v))
		}
		if _, err := fh.WriteAt(buf, sp.off); err != nil {
			t.Fatal(err)
		}
	}
}

// probeKLD is the fake harness's model of this fixture: attention is very
// sensitive to Q3_K, ffn_down barely, Q8_0 costs a small background.
func probeKLD(t *testing.T) func(path string) (float64, bool) {
	return func(path string) (float64, bool) {
		s, err := tensorbank.OpenSource(path)
		if err != nil {
			return 0, false
		}
		defer s.Close()
		f, err := tensorbank.Parse(s)
		if err != nil {
			return 0, false
		}
		kld := 0.001
		for _, ti := range f.Tensors {
			switch {
			case ti.DType == core.DTypeF16 || ti.DType == core.DTypeF32:
			case ti.DType == core.DTypeQ8_0:
				kld += 0.0005
			case strings.Contains(ti.Name, "attn_q"):
				kld += 0.4
			case strings.Contains(ti.Name, "ffn_down"):
				kld += 0.01
			}
		}
		return kld, true
	}
}

func TestSolveCalibratesSensitivityWithProbes(t *testing.T) {
	f := newFixture(t, 230000)
	finiteF16Payloads(t, f.src)
	f.runner.kldForModel = probeKLD(t)
	r := f.planEffort("sens", "profiled", nil)
	e := f.engine(r)
	e.StageLimit = 3 // assemble, anchor, solve
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	store := state.Store{Dir: f.stateDir}
	done, err := store.Load("sens")
	if err != nil {
		t.Fatal(err)
	}
	if next, ok := done.NextStage(); !ok || next != core.StageQuantize {
		t.Fatalf("next stage = %v %v, want quantize", next, ok)
	}

	// The probe record: background plus one probe per role (norms excluded).
	var st sensitivityState
	data, err := os.ReadFile(filepath.Join(done.Config.WorkDir, "sensitivity.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &st); err != nil {
		t.Fatal(err)
	}
	if st.Background == nil || *st.Background <= 0 {
		t.Fatalf("no background KLD: %+v", st)
	}
	if len(st.Probes) != 2 || st.Probes["attn_q"].ProbeDType != core.DTypeQ3_K || st.Probes["ffn_down"].ProbeDType != core.DTypeQ3_K {
		t.Fatalf("probes = %+v", st.Probes)
	}
	if st.Probes["attn_q"].KLD <= st.Probes["ffn_down"].KLD {
		t.Fatalf("attn_q probe %v should exceed ffn_down probe %v", st.Probes["attn_q"].KLD, st.Probes["ffn_down"].KLD)
	}
	// Background + 2 probes = 3 KLD evaluations, plus one baseline capture.
	if got := len(f.runner.evaluated); got != 3 {
		t.Fatalf("KLD evaluations = %d (%v), want 3", got, f.runner.evaluated)
	}
	for _, p := range f.runner.evaluated {
		if _, err := os.Stat(p); err == nil {
			t.Errorf("probe model %s left on disk", p)
		}
	}
	if _, err := os.Stat(filepath.Join(done.Config.WorkDir, "anchors", "probes")); !os.IsNotExist(err) {
		t.Errorf("probe scratch not removed: %v", err)
	}

	// The solve consumed the calibration and spent bytes on attention.
	var prof struct {
		Profile     *core.Profile        `json:"profile"`
		Sensitivity *profile.Sensitivity `json:"sensitivity"`
	}
	data, err = os.ReadFile(done.Artifacts[core.StageSolve])
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &prof); err != nil {
		t.Fatal(err)
	}
	if prof.Sensitivity == nil || len(prof.Sensitivity.Roles) != 2 {
		t.Fatalf("profile.json sensitivity = %+v", prof.Sensitivity)
	}
	target := map[string]core.DType{}
	for _, qa := range prof.Profile.Assignments {
		target[qa.TensorName] = qa.Target
	}
	if anchor.Rank(target["blk.0.attn_q.weight"]) >= anchor.Rank(target["blk.0.ffn_down.weight"]) {
		t.Errorf("attn_q %s should outrank ffn_down %s under a binding budget", target["blk.0.attn_q.weight"], target["blk.0.ffn_down.weight"])
	}
	if target["blk.0.attn_norm.weight"] != core.DTypeF32 {
		t.Errorf("norm = %s, want preserved", target["blk.0.attn_norm.weight"])
	}

	// Resume: re-executing solve must reuse the checkpointed probes and the
	// exact table without touching llama-perplexity again.
	done.Completed = done.Completed[:2]
	delete(done.Artifacts, core.StageSolve)
	if err := store.Save(done); err != nil {
		t.Fatal(err)
	}
	before := len(f.runner.evaluated)
	e2 := f.engine(done)
	e2.StageLimit = 1
	if err := e2.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := len(f.runner.evaluated); got != before {
		t.Fatalf("resumed solve re-ran %d probe evaluations", got-before)
	}
}

func TestFastEffortSkipsSensitivityProbes(t *testing.T) {
	f := newFixture(t, 230000)
	finiteF16Payloads(t, f.src)
	f.runner.kldForModel = probeKLD(t)
	r := f.planEffort("nosens", "fast", nil)
	e := f.engine(r)
	e.StageLimit = 3
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(f.runner.evaluated) != 0 {
		t.Fatalf("fast solve ran %d KLD evaluations", len(f.runner.evaluated))
	}
	if _, err := os.Stat(filepath.Join(r.Config.WorkDir, "sensitivity.json")); !os.IsNotExist(err) {
		t.Fatal("fast effort wrote sensitivity.json")
	}
	// Explicit opt-out on profiled behaves the same.
	f2 := newFixture(t, 230000)
	finiteF16Payloads(t, f2.src)
	f2.runner.kldForModel = probeKLD(t)
	r2 := f2.planEffort("optout", "profiled", func(o *PlanOptions) { o.NoSensitivity = true })
	e2 := f2.engine(r2)
	e2.StageLimit = 3
	if err := e2.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(f2.runner.evaluated) != 0 {
		t.Fatalf("-no-sensitivity solve ran %d KLD evaluations", len(f2.runner.evaluated))
	}
}
