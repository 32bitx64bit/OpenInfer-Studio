package tensorbank

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"quantlab/core"
)

func sampleKVs() []KV {
	return []KV{
		{Key: "general.architecture", Value: Value{Type: VTString, Scalar: "llama"}},
		{Key: "general.name", Value: Value{Type: VTString, Scalar: "sample-test"}},
		{Key: "general.alignment", Value: Value{Type: VTUint32, Scalar: uint32(32)}},
	}
}

func sampleSpecs() []spec {
	return []spec{
		{"blk.0.attn.weight", core.DTypeF16, []uint64{64, 16}},       // 2-D
		{"blk.0.ffn_exps.weight", core.DTypeF16, []uint64{64, 8, 2}}, // 3-D [ne0, R, experts]
		{"blk.0.norm.weight", core.DTypeF32, []uint64{64}},
	}
}

func TestSampleRowsRoundTrip(t *testing.T) {
	data, _ := buildGGUF(t, 3, 32, sampleKVs(), sampleSpecs())
	srcPath := filepath.Join(t.TempDir(), "src.gguf")
	if err := os.WriteFile(srcPath, data, 0o644); err != nil {
		t.Fatal(err)
	}

	picks := []RowSample{
		{Name: "blk.0.attn.weight", Keep: []uint64{0, 4, 8, 12}}, // 4 of 16 rows
		{Name: "blk.0.ffn_exps.weight", Keep: []uint64{1, 3, 5}}, // 3 of 8 rows per expert
	}
	outPath := filepath.Join(t.TempDir(), "sample.gguf")
	if err := SampleRows(context.Background(), srcPath, outPath, picks, nil); err != nil {
		t.Fatalf("SampleRows: %v", err)
	}

	out, err := OpenSource(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	f, err := Parse(out)
	if err != nil {
		t.Fatalf("re-parse: %v", err)
	}
	if len(f.Tensors) != 2 {
		t.Fatalf("sampled %d tensors, want 2 (picks only)", len(f.Tensors))
	}
	// 2-D: shape becomes [64, 4].
	t2d, ok := f.FindTensor("blk.0.attn.weight")
	if !ok {
		t.Fatal("missing 2-D tensor")
	}
	if len(t2d.Shape) != 2 || t2d.Shape[0] != 64 || t2d.Shape[1] != 4 {
		t.Errorf("2-D shape = %v, want [64 4]", t2d.Shape)
	}
	// 3-D: shape becomes [64, 3, 2].
	t3d, ok := f.FindTensor("blk.0.ffn_exps.weight")
	if !ok {
		t.Fatal("missing 3-D tensor")
	}
	if len(t3d.Shape) != 3 || t3d.Shape[0] != 64 || t3d.Shape[1] != 3 || t3d.Shape[2] != 2 {
		t.Errorf("3-D shape = %v, want [64 3 2]", t3d.Shape)
	}
	// KV bytes verbatim.
	src, err := OpenSource(srcPath)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	sf, err := Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	if string(f.KVBytes) != string(sf.KVBytes) {
		t.Error("KV bytes differ from source")
	}
	// Source order preserved: attn before ffn_exps.
	if f.Tensors[0].Name != "blk.0.attn.weight" || f.Tensors[1].Name != "blk.0.ffn_exps.weight" {
		t.Errorf("tensor order = %s, %s", f.Tensors[0].Name, f.Tensors[1].Name)
	}
}

func TestSampleRowsRejectsQuantized(t *testing.T) {
	data, _ := buildGGUF(t, 3, 32, sampleKVs(), []spec{
		{"blk.0.q.weight", core.DTypeQ8_0, []uint64{64, 16}},
	})
	srcPath := filepath.Join(t.TempDir(), "q.gguf")
	if err := os.WriteFile(srcPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	err := SampleRows(context.Background(), srcPath, filepath.Join(t.TempDir(), "o.gguf"),
		[]RowSample{{Name: "blk.0.q.weight", Keep: []uint64{0}}}, nil)
	if err == nil {
		t.Fatal("quantized source accepted")
	}
}

func TestSampleRowsRejectsOutOfRange(t *testing.T) {
	data, _ := buildGGUF(t, 3, 32, sampleKVs(), sampleSpecs())
	srcPath := filepath.Join(t.TempDir(), "src.gguf")
	if err := os.WriteFile(srcPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	err := SampleRows(context.Background(), srcPath, filepath.Join(t.TempDir(), "o.gguf"),
		[]RowSample{{Name: "blk.0.attn.weight", Keep: []uint64{99}}}, nil)
	if err == nil {
		t.Fatal("out-of-range row accepted")
	}
}
