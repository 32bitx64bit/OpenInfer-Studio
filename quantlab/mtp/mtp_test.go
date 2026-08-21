package mtp

import (
	"slices"
	"testing"

	"quantlab/core"
	"quantlab/tensorbank"
)

func fakeBank(names ...string) *core.TensorBank {
	b := &core.TensorBank{SourcePath: "model.gguf", Arch: "qwen3"}
	for _, n := range names {
		b.Tensors = append(b.Tensors, core.TensorDesc{
			Name: n, DType: core.DTypeF32, Shape: []uint64{8},
			Length: 32, Elements: 8,
		})
	}
	return b
}

func trunkKVs(arch string, block, nextn uint32) []tensorbank.KV {
	return []tensorbank.KV{
		{Key: "general.architecture", Value: tensorbank.Value{Type: tensorbank.VTString, Scalar: arch}},
		{Key: arch + ".block_count", Value: tensorbank.Value{Type: tensorbank.VTUint32, Scalar: block}},
		{Key: arch + ".nextn_predict_layers", Value: tensorbank.Value{Type: tensorbank.VTUint32, Scalar: nextn}},
	}
}

func TestShouldDrop(t *testing.T) {
	cases := []struct {
		bpw  float64
		want bool
	}{
		{0, false},
		{-1, false},
		{2.0, true},
		{2.74, true},
		{2.75, false},
		{2.9, false},
		{4.5, false},
	}
	for _, tc := range cases {
		if got := ShouldDrop(tc.bpw); got != tc.want {
			t.Errorf("ShouldDrop(%v)=%v, want %v", tc.bpw, got, tc.want)
		}
	}
}

func TestSelectDropsExtraBlocksAndNextnNames(t *testing.T) {
	names := []string{
		"token_embd.weight",
		"blk.0.attn_q.weight",
		"blk.32.attn_q.weight",
		"blk.63.attn_q.weight",
		"blk.63.ffn_down.weight",
		"blk.64.attn_q.weight",
		"blk.64.ffn_down.weight",
		"nextn.eh_proj.weight",
		"output.weight",
		"foo.mtp.head.weight",
	}
	drop, ok := Select(fakeBank(names...), trunkKVs("qwen3", 64, 1), "qwen3")
	if !ok {
		t.Fatal("Select returned ok=false")
	}
	want := []string{
		"blk.64.attn_q.weight",
		"blk.64.ffn_down.weight",
		"nextn.eh_proj.weight",
		"foo.mtp.head.weight",
	}
	if !slices.Equal(drop, want) {
		t.Fatalf("drop=%v, want %v", drop, want)
	}
}

func TestSelectLayersPrefix(t *testing.T) {
	names := []string{
		"layers.0.attn_q.weight",
		"layers.63.attn_q.weight",
		"layers.64.attn_q.weight",
		"token_embd.weight",
	}
	drop, ok := Select(fakeBank(names...), trunkKVs("llama", 64, 1), "llama")
	if !ok {
		t.Fatal("Select returned ok=false")
	}
	if !slices.Equal(drop, []string{"layers.64.attn_q.weight"}) {
		t.Fatalf("drop=%v", drop)
	}
}

func TestSelectNoopWhenNextnZeroAndNoNames(t *testing.T) {
	names := []string{"blk.0.attn_q.weight", "blk.63.attn_q.weight", "output.weight"}
	kvs := trunkKVs("qwen3", 64, 0)
	if drop, ok := Select(fakeBank(names...), kvs, "qwen3"); ok || len(drop) != 0 {
		t.Fatalf("drop=%v ok=%v", drop, ok)
	}
}

func TestSelectDropsNextnNamesWhenKVIsZero(t *testing.T) {
	names := []string{"blk.0.attn_q.weight", "blk.63.attn_q.weight", "nextn.foo.weight"}
	drop, ok := Select(fakeBank(names...), trunkKVs("qwen3", 64, 0), "qwen3")
	if !ok || !slices.Equal(drop, []string{"nextn.foo.weight"}) {
		t.Fatalf("drop=%v ok=%v", drop, ok)
	}
}

func TestSelectNoopExtraBlocksWhenNextnZero(t *testing.T) {
	names := []string{"blk.0.attn_q.weight", "blk.64.attn_q.weight"}
	drop, ok := Select(fakeBank(names...), trunkKVs("qwen3", 64, 0), "qwen3")
	if ok || len(drop) != 0 {
		t.Fatalf("extra blk.64 dropped without nextn KV: drop=%v ok=%v", drop, ok)
	}
}

func TestSelectNoopDedicatedSidecar(t *testing.T) {
	names := []string{
		"blk.64.attn_q.weight",
		"nextn.eh_proj.weight",
	}
	if drop, ok := Select(fakeBank(names...), trunkKVs("qwen3", 64, 1), "qwen3"); ok || len(drop) != 0 {
		t.Fatalf("sidecar stripped: drop=%v ok=%v", drop, ok)
	}
}

func TestSelectNoopWouldEmptyFile(t *testing.T) {
	names := []string{"nextn.eh_proj.weight", "foo.mtp.head.weight"}
	kvs := []tensorbank.KV{
		{Key: "general.architecture", Value: tensorbank.Value{Type: tensorbank.VTString, Scalar: "qwen3"}},
		{Key: "qwen3.block_count", Value: tensorbank.Value{Type: tensorbank.VTUint32, Scalar: uint32(0)}},
		{Key: "qwen3.nextn_predict_layers", Value: tensorbank.Value{Type: tensorbank.VTUint32, Scalar: uint32(1)}},
	}
	if drop, ok := Select(fakeBank(names...), kvs, "qwen3"); ok || len(drop) != 0 {
		t.Fatalf("empty-keep strip: drop=%v ok=%v", drop, ok)
	}
}

func TestZeroNextnLayersUpdatesExistingOnly(t *testing.T) {
	kvs := []tensorbank.KV{
		{Key: "general.architecture", Value: tensorbank.Value{Type: tensorbank.VTString, Scalar: "qwen3"}},
		{Key: "qwen3.nextn_predict_layers", Value: tensorbank.Value{Type: tensorbank.VTUint32, Scalar: uint32(1)}},
		{Key: "qwen3.block_count", Value: tensorbank.Value{Type: tensorbank.VTUint32, Scalar: uint32(64)}},
	}
	out := ZeroNextnLayers(kvs, "qwen3")
	if len(out) != 3 {
		t.Fatalf("added keys: %d", len(out))
	}
	got, ok := out[1].Value.Scalar.(uint32)
	if !ok || got != 0 {
		t.Fatalf("nextn=%v %T", out[1].Value.Scalar, out[1].Value.Scalar)
	}
	if _, ok := kvUint32(out, "nextn_predict_layers"); ok {
		t.Fatal("inserted alias that never existed")
	}
}

func TestImpliedBPW(t *testing.T) {
	b := fakeBank("a", "b") // 16 elements
	got := ImpliedBPW(b, 4) // 4 bytes * 8 / 16 = 2.0
	if got != 2.0 {
		t.Fatalf("ImpliedBPW=%v, want 2", got)
	}
	if ImpliedBPW(b, 0) != 0 || ImpliedBPW(nil, 10) != 0 {
		t.Fatal("zero cases")
	}
}

func TestNextnPredictLayersAliases(t *testing.T) {
	kvs := []tensorbank.KV{
		{Key: "nextn_predict_layers", Value: tensorbank.Value{Type: tensorbank.VTUint32, Scalar: uint32(2)}},
	}
	if n := NextnPredictLayers(kvs, "qwen3"); n != 2 {
		t.Fatalf("alias=%d", n)
	}
	kvs = []tensorbank.KV{
		{Key: "fork.nextn_predict_layers", Value: tensorbank.Value{Type: tensorbank.VTUint64, Scalar: uint64(3)}},
	}
	if n := NextnPredictLayers(kvs, "qwen3"); n != 3 {
		t.Fatalf("suffix scan=%d", n)
	}
}
