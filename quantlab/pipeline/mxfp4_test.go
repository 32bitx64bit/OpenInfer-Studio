package pipeline

import (
	"context"
	"path/filepath"
	"testing"

	"quantlab/core"
	"quantlab/orchestrate"
	"quantlab/tensorbank"
)

// MXFP4 expert stacks (gpt-oss) are the model's native storage: the run
// must keep them bit-for-bit, never requantize them, and still optimize
// the float tensors around them.
func TestMXFP4ExpertsKeptNative(t *testing.T) {
	f := newFixture(t, 400000)
	tensors := []gtensor{
		{"blk.0.attn_q.weight", core.DTypeF16, []uint64{256, 256}},
		{"blk.0.ffn_down_exps.weight", core.DTypeMXFP4, []uint64{256, 64, 4}},
		{"blk.0.attn_norm.weight", core.DTypeF32, []uint64{256}},
		{"blk.1.attn_q.weight", core.DTypeF16, []uint64{256, 256}},
	}
	if err := writeGGUF(f.src, tensors); err != nil {
		t.Fatal(err)
	}
	finiteF16Payloads(t, f.src)
	r := f.planEffort("mx", "profiled", func(o *PlanOptions) { o.GatesOptOut = true })
	e := f.engine(r)
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, d := range f.runner.quantTypes {
		if d.BaseTensorType() == core.DTypeMXFP4 {
			t.Fatalf("llama-quantize was asked to write MXFP4 (%v)", f.runner.quantTypes)
		}
	}
	var kept bool
	for _, o := range e.Run.Manifest.Options {
		if o.TensorName == "blk.0.ffn_down_exps.weight" {
			kept = o.Target == core.DTypeMXFP4
		}
	}
	if !kept {
		t.Fatal("MXFP4 experts not kept native in the manifest")
	}
	out := e.Run.Artifacts[core.StageEmit]
	if out == "" {
		t.Fatal("no emitted artifact")
	}
	s, err := tensorbank.OpenSource(out)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	file, err := tensorbank.Parse(s)
	if err != nil {
		t.Fatal(err)
	}
	ti, ok := file.FindTensor("blk.0.ffn_down_exps.weight")
	if !ok || ti.DType != core.DTypeMXFP4 {
		t.Fatalf("emitted experts = %+v %v, want MXFP4", ti, ok)
	}
}

// The requantize guard must look at each job's own input: a row sample or
// trimmed subset of float tensors is not a requantization even when the
// model also holds MXFP4 experts.
func TestRequantizeFlagFollowsJobInput(t *testing.T) {
	f := newFixture(t, 400000)
	dir := t.TempDir()
	write := func(name string, ts []gtensor) string {
		p := filepath.Join(dir, name)
		if err := writeGGUF(p, ts); err != nil {
			t.Fatal(err)
		}
		return p
	}
	full := write("full.gguf", []gtensor{
		{"blk.0.attn_q.weight", core.DTypeF16, []uint64{256, 256}},
		{"blk.0.ffn_down_exps.weight", core.DTypeMXFP4, []uint64{256, 64, 4}},
	})
	floats := write("floats.gguf", []gtensor{{"blk.0.attn_q.weight", core.DTypeF16, []uint64{256, 256}}})
	q8 := write("q8.gguf", []gtensor{{"blk.0.attn_q.weight", core.DTypeQ8_0, []uint64{256, 256}}})
	if err := writeGGUF(f.src, []gtensor{
		{"blk.0.attn_q.weight", core.DTypeF16, []uint64{256, 256}},
		{"blk.0.ffn_down_exps.weight", core.DTypeMXFP4, []uint64{256, 64, 4}},
	}); err != nil {
		t.Fatal(err)
	}
	r := f.planEffort("rq", "fast", nil)
	e := f.engine(r)
	for path, want := range map[string]bool{full: false, floats: false, q8: true} {
		req := orchestrate.QuantizeRequest{SourcePath: path}
		e.fillQuantizeRequest(&req)
		if req.SourceQuantized != want {
			t.Errorf("%s: SourceQuantized = %v, want %v", filepath.Base(path), req.SourceQuantized, want)
		}
	}
}
