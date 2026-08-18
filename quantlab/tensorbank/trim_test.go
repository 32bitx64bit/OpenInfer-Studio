package tensorbank

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"quantlab/core"
)

func trimAnchorKVs() []KV {
	return []KV{
		{Key: "general.architecture", Value: Value{Type: VTString, Scalar: "llama"}},
		{Key: "general.name", Value: Value{Type: VTString, Scalar: "trim-model"}},
		{Key: "general.alignment", Value: Value{Type: VTUint32, Scalar: uint32(32)}},
	}
}

func trimSpecs() []spec {
	return []spec{
		{"blk.0.attn.weight", core.DTypeQ8_0, []uint64{128, 32}},
		{"blk.0.ffn.weight", core.DTypeQ8_0, []uint64{128, 32}},
		{"blk.0.norm.weight", core.DTypeF32, []uint64{128}},
		{"blk.1.attn.weight", core.DTypeQ8_0, []uint64{128, 32}},
	}
}

func writeAnchorFile(t *testing.T, specs []spec, align uint32) (*Source, map[string][]byte) {
	t.Helper()
	data, pay := buildGGUF(t, 3, align, trimAnchorKVs(), specs)
	dir := t.TempDir()
	path := filepath.Join(dir, "anchor.gguf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s, pay
}

func TestTrimProducesValidGGUFWithMatchingPayloads(t *testing.T) {
	specs := trimSpecs()
	src, pay := writeAnchorFile(t, specs, 32)
	anchorFile, err := Parse(src)
	if err != nil {
		t.Fatal(err)
	}

	keep := map[string]struct{}{
		"blk.0.attn.weight": {},
		"blk.0.norm.weight": {},
	}
	outPath := filepath.Join(t.TempDir(), "trimmed.gguf")
	if err := Trim(context.Background(), src.Path(), keep, outPath, nil); err != nil {
		t.Fatalf("trim: %v", err)
	}

	out, err := OpenSource(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	f, err := Parse(out)
	if err != nil {
		t.Fatalf("trimmed file fails Parse: %v", err)
	}
	if len(f.Tensors) != len(keep) {
		t.Fatalf("trimmed has %d tensors, want %d", len(f.Tensors), len(keep))
	}
	// Metadata preserved verbatim.
	if !strings.HasPrefix(string(f.KVBytes), string(anchorFile.KVBytes)) {
		// KVBytes should be identical; direct compare below.
	}
	if string(f.KVBytes) != string(anchorFile.KVBytes) {
		t.Error("metadata not preserved verbatim")
	}
	// Per-tensor: dtype, shape, alignment, payload hash matches parent.
	for _, ti := range f.Tensors {
		if ti.RelOffset%uint64(f.Alignment) != 0 {
			t.Errorf("%s offset %d not aligned", ti.Name, ti.RelOffset)
		}
		got := make([]byte, ti.Length)
		out.ReadAt(got, f.PayloadOffset(ti))
		if string(got) != string(pay[ti.Name]) {
			t.Errorf("%s payload differs from anchor", ti.Name)
		}
	}
}

func TestTrimPreservesTensorTableOrdering(t *testing.T) {
	specs := trimSpecs()
	src, _ := writeAnchorFile(t, specs, 32)
	anchorFile, _ := Parse(src)

	keep := map[string]struct{}{
		"blk.0.norm.weight": {},
		"blk.0.attn.weight": {},
		"blk.1.attn.weight": {},
		"blk.0.ffn.weight":  {},
	}
	outPath := filepath.Join(t.TempDir(), "trimmed.gguf")
	if err := Trim(context.Background(), src.Path(), keep, outPath, nil); err != nil {
		t.Fatal(err)
	}
	out, _ := OpenSource(outPath)
	defer out.Close()
	f, err := Parse(out)
	if err != nil {
		t.Fatal(err)
	}
	// Output order must match the anchor's order (not the keep map order).
	var wantOrder []string
	for _, t := range anchorFile.Tensors {
		if _, ok := keep[t.Name]; ok {
			wantOrder = append(wantOrder, t.Name)
		}
	}
	if len(f.Tensors) != len(wantOrder) {
		t.Fatalf("tensor count %d, want %d", len(f.Tensors), len(wantOrder))
	}
	for i, ti := range f.Tensors {
		if ti.Name != wantOrder[i] {
			t.Errorf("position %d: got %q, want %q", i, ti.Name, wantOrder[i])
		}
	}
}

func TestTrimOffsetsAtBothAlignments(t *testing.T) {
	for _, align := range []uint32{32, 64} {
		t.Run(fmt.Sprintf("align%d", align), func(t *testing.T) {
			kvs := []KV{
				{Key: "general.architecture", Value: Value{Type: VTString, Scalar: "llama"}},
				{Key: "general.alignment", Value: Value{Type: VTUint32, Scalar: align}},
			}
			specs := []spec{
				{"a", core.DTypeF32, []uint64{100}},
				{"b", core.DTypeF16, []uint64{70, 3}},
				{"c", core.DTypeF32, []uint64{50}},
			}
			data, _ := buildGGUF(t, 3, align, kvs, specs)
			dir := t.TempDir()
			anchorPath := filepath.Join(dir, "anchor.gguf")
			os.WriteFile(anchorPath, data, 0o644)

			keep := map[string]struct{}{"a": {}, "c": {}}
			outPath := filepath.Join(dir, "trimmed.gguf")
			if err := Trim(context.Background(), anchorPath, keep, outPath, nil); err != nil {
				t.Fatal(err)
			}
			out, _ := OpenSource(outPath)
			defer out.Close()
			f, err := Parse(out)
			if err != nil {
				t.Fatal(err)
			}
			if f.Alignment != align {
				t.Fatalf("alignment %d, want %d", f.Alignment, align)
			}
			if f.DataOffset%int64(align) != 0 {
				t.Errorf("data offset %d not aligned to %d", f.DataOffset, align)
			}
			for _, ti := range f.Tensors {
				if ti.RelOffset%uint64(align) != 0 {
					t.Errorf("%s offset %d not aligned to %d", ti.Name, ti.RelOffset, align)
				}
			}
		})
	}
}

func TestTrimRejectsEmptyKeep(t *testing.T) {
	src, _ := writeAnchorFile(t, trimSpecs(), 32)
	outPath := filepath.Join(t.TempDir(), "trimmed.gguf")
	err := Trim(context.Background(), src.Path(), map[string]struct{}{}, outPath, nil)
	if err == nil || !strings.Contains(err.Error(), "empty") {
		t.Fatalf("want empty-keep error, got %v", err)
	}
}

func TestTrimRejectsMissingTensorInKeep(t *testing.T) {
	src, _ := writeAnchorFile(t, trimSpecs(), 32)
	keep := map[string]struct{}{
		"blk.0.attn.weight": {},
		"ghost.tensor":      {},
	}
	outPath := filepath.Join(t.TempDir(), "trimmed.gguf")
	err := Trim(context.Background(), src.Path(), keep, outPath, nil)
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("want missing-tensor error, got %v", err)
	}
}

func TestTrimCancellationLeavesNoPartialFile(t *testing.T) {
	specs := []spec{
		{"w1", core.DTypeF16, []uint64{512, 4}},
		{"w2", core.DTypeF16, []uint64{512, 4}},
		{"w3", core.DTypeF16, []uint64{512, 4}},
	}
	src, _ := writeAnchorFile(t, specs, 32)
	keep := map[string]struct{}{"w1": {}, "w2": {}, "w3": {}}
	dir := t.TempDir()
	outPath := filepath.Join(dir, "trimmed.gguf")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := Trim(ctx, src.Path(), keep, outPath, nil); err == nil {
		t.Fatal("pre-cancelled context accepted")
	}
	assertNoPartial(t, outPath)

	// cancel mid-copy via progress callback
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	err := Trim(ctx2, src.Path(), keep, outPath, func(c, tot uint64) {
		if c >= tot/3 {
			cancel2()
		}
	})
	if err == nil {
		t.Fatal("mid-copy cancellation accepted")
	}
	assertNoPartial(t, outPath)
}

func TestTrimAtomicCleanupOnFailure(t *testing.T) {
	specs := []spec{
		{"w1", core.DTypeF16, []uint64{512, 4}},
		{"w2", core.DTypeF16, []uint64{512, 4}},
	}
	src, _ := writeAnchorFile(t, specs, 32)
	keep := map[string]struct{}{"w1": {}, "w2": {}}
	dir := t.TempDir()
	outPath := filepath.Join(dir, "trimmed.gguf")

	// Truncate the source after the first tensor is copied → read failure
	// mid-stream, after the tmp file exists.
	data, _ := os.ReadFile(src.Path())
	var copiedOnce bool
	err := Trim(context.Background(), src.Path(), keep, outPath, func(c, tot uint64) {
		if !copiedOnce {
			copiedOnce = true
			if err := os.Truncate(src.Path(), int64(len(data)-1024)); err != nil {
				t.Errorf("truncate: %v", err)
			}
		}
	})
	if err == nil {
		t.Fatal("truncated source accepted")
	}
	assertNoPartial(t, outPath)
}

func TestTrimRejectsInPlaceEdit(t *testing.T) {
	src, _ := writeAnchorFile(t, trimSpecs(), 32)
	keep := map[string]struct{}{"blk.0.attn.weight": {}}
	err := Trim(context.Background(), src.Path(), keep, src.Path(), nil)
	if err == nil || !strings.Contains(err.Error(), "in-place") {
		t.Fatalf("want in-place rejection, got %v", err)
	}
}

func TestTrimProgressReports(t *testing.T) {
	specs := trimSpecs()
	src, _ := writeAnchorFile(t, specs, 32)
	keep := map[string]struct{}{
		"blk.0.attn.weight": {},
		"blk.0.norm.weight": {},
	}
	outPath := filepath.Join(t.TempDir(), "trimmed.gguf")
	var lastCopied, lastTotal uint64
	var calls int
	if err := Trim(context.Background(), src.Path(), keep, outPath, func(c, tot uint64) {
		lastCopied, lastTotal = c, tot
		calls++
	}); err != nil {
		t.Fatal(err)
	}
	if calls != len(keep) || lastCopied != lastTotal || lastTotal == 0 {
		t.Errorf("progress: calls=%d copied=%d total=%d", calls, lastCopied, lastTotal)
	}
}
