package tensorbank

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"quantlab/core"
)

func kvHeavy(n int) []KV {
	kvs := anchorKVs("model")
	for i := 0; i < n; i++ {
		kvs = append(kvs, KV{
			Key:   fmt.Sprintf("llama.custom.metadata.key.%03d", i),
			Value: Value{Type: VTString, Scalar: "some moderately long metadata value to inflate the KV section"},
		})
	}
	return kvs
}

// TestPlannedArtifactSizeMatchesBuild proves PlannedArtifactSize is exact:
// it equals the byte size of the file Assembler.Build actually writes, for
// default alignment, alignment 64, and a metadata-heavy source.
func TestPlannedArtifactSizeMatchesBuild(t *testing.T) {
	specs := []spec{
		{"tok_embd", core.DTypeF16, []uint64{128, 4}},
		{"attn_v", core.DTypeF16, []uint64{128, 4}},
		{"ffn_down", core.DTypeF16, []uint64{64, 8}},
		{"norm", core.DTypeF32, []uint64{7}},
	}
	cases := []struct {
		name  string
		align uint32
		kvs   []KV
	}{
		{"default alignment", 32, anchorKVs("model")},
		{"alignment 64", 64, []KV{
			{Key: "general.architecture", Value: Value{Type: VTString, Scalar: "llama"}},
			{Key: "general.name", Value: Value{Type: VTString, Scalar: "model"}},
			{Key: "general.alignment", Value: Value{Type: VTUint32, Scalar: uint32(64)}},
		}},
		{"metadata heavy", 32, kvHeavy(50)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data, _ := buildGGUF(t, 3, tc.align, tc.kvs, specs)
			srcs := openSources(t, data)
			bank, err := NewAssembler().Assemble(srcs[0], srcs[0].Path(), "")
			if err != nil {
				t.Fatal(err)
			}
			m := manifestFor(t, specs, func(s spec) core.DType { return s.dt }, "")
			planned, ok := PlannedArtifactSize(bank, m)
			if !ok {
				t.Fatal("planned size not computable for assembled bank")
			}
			outPath := filepath.Join(t.TempDir(), "out.gguf")
			if err := mustBuild(t, context.Background(), srcs, m, outPath, nil); err != nil {
				t.Fatal(err)
			}
			st, err := os.Stat(outPath)
			if err != nil {
				t.Fatal(err)
			}
			if planned != uint64(st.Size()) {
				t.Fatalf("planned %d != built %d", planned, st.Size())
			}
			// The reserve always covers the true non-payload overhead.
			overhead := uint64(st.Size()) - m.TotalBytes
			if r := OverheadReserve(bank); r < overhead {
				t.Fatalf("reserve %d < actual overhead %d", r, overhead)
			}
		})
	}
}

// TestPlannedArtifactSizeMixedManifest checks exactness when tensor payloads
// come from a second anchor at a different dtype (payload sizes change).
func TestPlannedArtifactSizeMixedManifest(t *testing.T) {
	specs := []spec{
		{"tok_embd", core.DTypeF16, []uint64{128, 4}},
		{"attn_v", core.DTypeF16, []uint64{128, 4}},
	}
	bSpecs := []spec{
		{"tok_embd", core.DTypeQ8_0, []uint64{128, 4}},
		{"attn_v", core.DTypeQ8_0, []uint64{128, 4}},
	}
	aData, _ := buildGGUF(t, 3, 32, anchorKVs("model"), specs)
	bData, _ := buildGGUF(t, 3, 32, anchorKVs("model"), bSpecs)
	srcs := openSources(t, aData, bData)
	bank, err := NewAssembler().Assemble(srcs[0], srcs[0].Path(), "")
	if err != nil {
		t.Fatal(err)
	}
	m := manifestFor(t, specs, func(s spec) core.DType { return core.DTypeQ8_0 }, "")
	planned, ok := PlannedArtifactSize(bank, m)
	if !ok {
		t.Fatal("planned size not computable")
	}
	outPath := filepath.Join(t.TempDir(), "out.gguf")
	if err := mustBuild(t, context.Background(), srcs, m, outPath, nil); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(outPath)
	if err != nil {
		t.Fatal(err)
	}
	if planned != uint64(st.Size()) {
		t.Fatalf("planned %d != built %d", planned, st.Size())
	}
}

// TestPlannedArtifactSizeFallback documents the fail-closed fallback: banks
// without KV metadata length (older checkpoints) yield ok=false so callers
// use payload accounting plus the conservative reserve instead.
func TestPlannedArtifactSizeFallback(t *testing.T) {
	if _, ok := PlannedArtifactSize(nil, &core.SelectionManifest{}); ok {
		t.Fatal("nil bank accepted")
	}
	bank := &core.TensorBank{SourcePath: "/m.gguf", Tensors: []core.TensorDesc{
		{Name: "a", DType: core.DTypeF16, Shape: []uint64{4, 4}, Elements: 16, Length: 32},
	}}
	m := &core.SelectionManifest{Options: []core.TensorOption{
		{TensorName: "a", Target: core.DTypeF16, Bytes: 32},
	}}
	if _, ok := PlannedArtifactSize(bank, nil); ok {
		t.Fatal("nil manifest accepted")
	}
	if _, ok := PlannedArtifactSize(bank, m); ok {
		t.Fatal("bank without KV metadata length yielded an exact size")
	}
	// Manifest must cover every bank tensor exactly once.
	bank.KVMetadataLen = 8
	if _, ok := PlannedArtifactSize(bank, &core.SelectionManifest{}); ok {
		t.Fatal("incomplete manifest accepted")
	}
	m2 := &core.SelectionManifest{Options: []core.TensorOption{
		{TensorName: "a", Target: core.DTypeF16, Bytes: 32},
		{TensorName: "a", Target: core.DTypeF16, Bytes: 32},
	}}
	if _, ok := PlannedArtifactSize(bank, m2); ok {
		t.Fatal("duplicate manifest option accepted")
	}
}

// TestOverheadReserveTightness bounds the reserve so the solver never wastes
// more than the documented worst case: the true metadata section plus one
// alignment unit of slack per tensor.
func TestOverheadReserveTightness(t *testing.T) {
	data, _ := buildGGUF(t, 3, 32, anchorKVs("model"), []spec{
		{"a", core.DTypeF16, []uint64{128, 4}},
		{"b", core.DTypeF16, []uint64{64, 8}},
	})
	srcs := openSources(t, data)
	bank, err := NewAssembler().Assemble(srcs[0], srcs[0].Path(), "")
	if err != nil {
		t.Fatal(err)
	}
	m := manifestFor(t, []spec{
		{"a", core.DTypeF16, []uint64{128, 4}},
		{"b", core.DTypeF16, []uint64{64, 8}},
	}, func(s spec) core.DType { return s.dt }, "")
	outPath := filepath.Join(t.TempDir(), "out.gguf")
	if err := mustBuild(t, context.Background(), srcs, m, outPath, nil); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(outPath)
	if err != nil {
		t.Fatal(err)
	}
	overhead := uint64(st.Size()) - m.TotalBytes
	r := OverheadReserve(bank)
	if r < overhead {
		t.Fatalf("reserve %d < actual overhead %d", r, overhead)
	}
	// Slack is only the unused per-tensor padding bound.
	if max := overhead + 2*(bankAlignment(bank)-1); r > max {
		t.Fatalf("reserve %d exceeds worst-case bound %d", r, max)
	}
	if OverheadReserve(nil) != 0 || OverheadReserve(&core.TensorBank{}) != 0 {
		t.Fatal("empty bank reserve nonzero")
	}
}
