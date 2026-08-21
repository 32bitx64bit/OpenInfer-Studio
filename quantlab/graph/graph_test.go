package graph

import "testing"

func TestAnalyzeGLUMixer(t *testing.T) {
	const d, nFF uint64 = 256, 512
	ts := []Tensor{
		{Name: "blk.0.attn_norm.weight", Shape: []uint64{d}},
		{Name: "blk.0.ffn_norm.weight", Shape: []uint64{d}},
		{Name: "blk.0.a.weight", Shape: []uint64{d, nFF}},
		{Name: "blk.0.b.weight", Shape: []uint64{d, nFF}},
		{Name: "blk.0.c.weight", Shape: []uint64{nFF, d}},
	}
	m := Analyze(ts)
	if m.D != int(d) {
		t.Fatalf("residual = %d, want %d", m.D, d)
	}
	found := false
	for _, mx := range m.Mixers {
		if mx.Kind == "glu" && mx.Writer.Name == "blk.0.c.weight" && len(mx.Expanders) == 2 {
			found = true
		}
	}
	if !found {
		t.Fatalf("glu mixer not found: %+v", m.Mixers)
	}
}

func TestUniqueLargeAxisAndGQA(t *testing.T) {
	const d uint64 = 256
	ts := []Tensor{
		{Name: "blk.0.attn_norm.weight", Shape: []uint64{d}},
		{Name: "token.weight", Shape: []uint64{d, 32000}},
		{Name: "output.weight", Shape: []uint64{d, 32000}},
		{Name: "blk.0.k.weight", Shape: []uint64{d, 64}},
	}
	if !UniqueLargeAxis(ts[1], ts) {
		t.Fatal("expected unique large axis on token.weight")
	}
	if !GQAShort(ts[3], int(d)) {
		t.Fatal("expected GQA-short K")
	}
}

func TestResidualWidthIsNotUniqueLargeAxis(t *testing.T) {
	const d uint64 = 256
	ts := []Tensor{
		{Name: "blk.0.attn_norm.weight", Shape: []uint64{d}},
		{Name: "blk.0.attn_q.weight", Shape: []uint64{d, 8}},
	}
	if UniqueLargeAxis(ts[1], ts) {
		t.Fatal("residual-width attn_q must not count as vocab-like")
	}
	if !ResidualRead(ts[1], int(d), ts) {
		t.Fatal("expected residual-read attn_q")
	}
}

func TestMoEStack(t *testing.T) {
	const d, nFF, nExp uint64 = 256, 512, 8
	ts := []Tensor{
		{Name: "blk.0.ffn_norm.weight", Shape: []uint64{d}},
		{Name: "blk.0.ffn_up_exps.weight", Shape: []uint64{d, nFF, nExp}},
		{Name: "blk.0.ffn_gate_inp.weight", Shape: []uint64{d, nExp}},
	}
	m := Analyze(ts)
	if len(m.MoE) != 1 || m.MoE[0].Slices != 8 || m.MoE[0].Router.Name == "" {
		t.Fatalf("moe = %+v", m.MoE)
	}
}

func TestZipfRows(t *testing.T) {
	e := make([]float64, 100)
	for i := range e {
		e[i] = 1.0 / float64(i+1)
	}
	if !ZipfRows(e) {
		t.Fatal("zipf row energy not detected")
	}
}
