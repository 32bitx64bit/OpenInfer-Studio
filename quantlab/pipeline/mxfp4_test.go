package pipeline

import (
	"context"
	"testing"

	"quantlab/core"
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
