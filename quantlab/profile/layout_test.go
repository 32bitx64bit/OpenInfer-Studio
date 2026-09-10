package profile

import (
	"math"
	"path/filepath"
	"testing"

	"quantlab/core"
	"quantlab/qtype"
)

func TestLayoutFor(t *testing.T) {
	// 2-D weight: one value per input channel.
	l, ok := LayoutFor(make([]float32, 256), 256, 4)
	if !ok || l.Experts != 1 {
		t.Fatalf("2-D layout: ok=%v experts=%d", ok, l.Experts)
	}
	// Fused expert stack: experts vectors, rows split evenly.
	l, ok = LayoutFor(make([]float32, 3*256), 256, 12)
	if !ok || l.Experts != 3 {
		t.Fatalf("expert layout: ok=%v experts=%d", ok, l.Experts)
	}
	// Per-row vectors (the old misreading) do not fit.
	if _, ok := LayoutFor(make([]float32, 4), 256, 4); ok {
		t.Error("per-row vector must not be accepted as a channel layout")
	}
	if _, ok := LayoutFor(make([]float32, 2*256), 256, 5); ok {
		t.Error("experts must partition rows evenly")
	}
	if _, ok := LayoutFor(nil, 256, 4); ok {
		t.Error("empty vector")
	}
}

func TestLayoutRowAndFill(t *testing.T) {
	const ne0 = 4
	vals := []float32{
		1, 2, 3, 4, // expert 0
		10, 20, 30, 40, // expert 1
	}
	l, ok := LayoutFor(vals, ne0, 6)
	if !ok || l.Experts != 2 {
		t.Fatal("layout")
	}
	if got := l.Row(vals, 2); got[0] != 1 {
		t.Errorf("row 2 -> expert 0, got %v", got)
	}
	if got := l.Row(vals, 3); got[0] != 10 {
		t.Errorf("row 3 -> expert 1, got %v", got)
	}
	dst := make([]float32, 2*ne0)
	l.Fill(dst, vals, 2, 2)
	want := []float32{1, 2, 3, 4, 10, 20, 30, 40}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("Fill = %v, want %v", dst, want)
		}
	}
	mean := ChannelMean(vals, ne0, 6)
	if mean[0] != 5.5 || mean[3] != 22 {
		t.Errorf("ChannelMean = %v", mean)
	}
	if ChannelMean(vals, 3, 6) != nil {
		t.Error("ChannelMean must reject a non-fitting vector")
	}
}

// TestExactLossWeightsColumns checks the exact table applies the imatrix
// vector per INPUT channel: the table entry must equal the reference
// quantizer's weighted SSE when every row is weighted by the same
// ne0-length vector, and a vector of the wrong length must fall back to
// uniform weights rather than being misapplied.
func TestExactLossWeightsColumns(t *testing.T) {
	const ne0, rows = 256, 8
	w := make([]float32, ne0*rows)
	for i := range w {
		w[i] = float32(math.Sin(float64(i)*0.731)) * 0.37
	}
	tensor := core.TensorDesc{Name: "blk.0.attn_q.weight", DType: core.DTypeF32,
		Shape: []uint64{ne0, rows}, Elements: ne0 * rows, Length: ne0 * rows * 4}
	// Strongly non-uniform per-channel importance.
	imp := make([]float32, ne0)
	for c := range imp {
		imp[c] = float32(1 + 30*(c%7))
	}
	expected := func(perChannel []float32) float64 {
		full := make([]float32, ne0*rows)
		for r := 0; r < rows; r++ {
			copy(full[r*ne0:], perChannel)
		}
		q := append([]float32(nil), w...)
		if _, err := qtype.QuantizeDequant(core.DTypeQ4_0, q, full); err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range w {
			e := float64(w[i]) - float64(q[i])
			s += float64(full[i]) * e * e
		}
		return s
	}
	got := exactLossFromValues(t, tensor, w, imp)[core.DTypeQ4_0]
	want := expected(imp)
	if math.Abs(got-want) > 1e-6*want {
		t.Errorf("weighted table = %v, want %v", got, want)
	}
	uniform := make([]float32, ne0)
	for c := range uniform {
		uniform[c] = 1
	}
	base := expected(uniform)
	if math.Abs(got-base) < 1e-3*base {
		t.Fatalf("test is not discriminating: weighted %v ~ uniform %v", got, base)
	}
	// A per-row vector (the old misreading) does not fit the layout and
	// must fall back to uniform weights.
	if got := exactLossFromValues(t, tensor, w, make([]float32, rows))[core.DTypeQ4_0]; math.Abs(got-base) > 1e-9*base {
		t.Errorf("per-row vector should fall back to uniform: %v vs %v", got, base)
	}
}

// exactLossFromValues writes tensor to a scratch GGUF and builds its exact
// loss table for Q4_0 with the given per-channel importance.
func exactLossFromValues(t *testing.T, tensor core.TensorDesc, w, imp []float32) map[core.DType]float64 {
	t.Helper()
	model := filepath.Join(t.TempDir(), "model.gguf")
	writeF32GGUF2(t, model, int(tensor.Shape[0]), int(tensor.Shape[1]), map[string][]float32{tensor.Name: w})
	bank := &core.TensorBank{SourcePath: model, ModelID: "layout-test", Tensors: []core.TensorDesc{tensor}}
	imatrix := map[string]ImatrixStats{tensor.Name: {Mean: 1, Samples: 1, Values: imp}}
	table, err := BuildExactLossTable(bank, []core.DType{core.DTypeQ4_0}, imatrix, nil)
	if err != nil {
		t.Fatal(err)
	}
	return table[tensor.Name]
}
