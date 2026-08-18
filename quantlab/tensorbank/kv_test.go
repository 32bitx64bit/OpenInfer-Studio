package tensorbank

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"quantlab/core"
)

func TestEncodeKVsRoundTrip(t *testing.T) {
	kvs := allKVTypes()
	data, _ := buildGGUF(t, 3, 32, kvs, []spec{
		{"w1", core.DTypeF16, []uint64{64, 8}},
	})
	f, _ := openParse(t, data)
	encoded := EncodeKVs(f.KVs)
	if !bytes.Equal(encoded, f.KVBytes) {
		t.Fatalf("EncodeKVs length %d, KVBytes %d", len(encoded), len(f.KVBytes))
	}
	if !bytes.Equal(EncodeKVs(kvs), f.KVBytes) {
		t.Fatal("EncodeKVs(original) does not match parsed KVBytes")
	}
}

func TestSetScalarUpdatesExistingOnly(t *testing.T) {
	kvs := []KV{
		{Key: "general.architecture", Value: Value{Type: VTString, Scalar: "qwen3"}},
		{Key: "qwen3.nextn_predict_layers", Value: Value{Type: VTUint32, Scalar: uint32(1)}},
		{Key: "qwen3.block_count", Value: Value{Type: VTUint32, Scalar: uint32(64)}},
	}
	out := SetScalar(kvs, "qwen3.nextn_predict_layers", 0)
	if got, ok := out[1].Value.Scalar.(uint32); !ok || got != 0 {
		t.Fatalf("nextn=%#v", out[1].Value.Scalar)
	}
	if out[1].Value.Type != VTUint32 {
		t.Fatalf("type changed to %d", out[1].Value.Type)
	}
	if len(SetScalar(kvs, "nextn_predict_layers", 0)) != len(kvs) {
		t.Fatal("inserted a key that never existed")
	}
	if got, ok := kvs[1].Value.Scalar.(uint32); !ok || got != 1 {
		t.Fatal("SetScalar mutated input")
	}
}

func TestTrimWithMetadataPatchesNextn(t *testing.T) {
	kvs := []KV{
		{Key: "general.architecture", Value: Value{Type: VTString, Scalar: "qwen3"}},
		{Key: "general.name", Value: Value{Type: VTString, Scalar: "mtp-model"}},
		{Key: "general.alignment", Value: Value{Type: VTUint32, Scalar: uint32(32)}},
		{Key: "qwen3.block_count", Value: Value{Type: VTUint32, Scalar: uint32(64)}},
		{Key: "qwen3.nextn_predict_layers", Value: Value{Type: VTUint32, Scalar: uint32(1)}},
		{Key: "nextn_predict_layers", Value: Value{Type: VTUint32, Scalar: uint32(1)}},
		{Key: "general.nextn_predict_layers", Value: Value{Type: VTUint32, Scalar: uint32(1)}},
	}
	specs := []spec{
		{"blk.0.attn_q.weight", core.DTypeF32, []uint64{32}},
		{"blk.64.attn_q.weight", core.DTypeF32, []uint64{32}},
		{"nextn.eh_proj.weight", core.DTypeF32, []uint64{32}},
	}
	data, pay := buildGGUF(t, 3, 32, kvs, specs)
	dir := t.TempDir()
	anchorPath := filepath.Join(dir, "anchor.gguf")
	if err := os.WriteFile(anchorPath, data, 0o644); err != nil {
		t.Fatal(err)
	}

	keep := map[string]struct{}{"blk.0.attn_q.weight": {}}
	patched := SetScalar(kvs, "qwen3.nextn_predict_layers", 0)
	patched = SetScalar(patched, "nextn_predict_layers", 0)
	patched = SetScalar(patched, "general.nextn_predict_layers", 0)
	outPath := filepath.Join(dir, "stripped.gguf")
	if err := TrimWithMetadata(context.Background(), anchorPath, keep, outPath, patched, nil); err != nil {
		t.Fatalf("TrimWithMetadata: %v", err)
	}

	s, err := OpenSource(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	f, err := Parse(s)
	if err != nil {
		t.Fatal(err)
	}
	if len(f.Tensors) != 1 || f.Tensors[0].Name != "blk.0.attn_q.weight" {
		t.Fatalf("tensors=%v", f.Tensors)
	}
	v, ok := f.Meta("qwen3.nextn_predict_layers")
	if !ok {
		t.Fatal("missing nextn key")
	}
	n, ok := v.Scalar.(uint32)
	if !ok || n != 0 {
		t.Fatalf("nextn=%#v, want 0", v.Scalar)
	}
	for _, key := range []string{"nextn_predict_layers", "general.nextn_predict_layers"} {
		v, ok := f.Meta(key)
		if !ok {
			t.Fatalf("missing %s", key)
		}
		if n, ok := v.Scalar.(uint32); !ok || n != 0 {
			t.Fatalf("%s=%#v", key, v.Scalar)
		}
	}
	gotPay := make([]byte, f.Tensors[0].Length)
	s.ReadAt(gotPay, f.PayloadOffset(f.Tensors[0]))
	if string(gotPay) != string(pay["blk.0.attn_q.weight"]) {
		t.Fatal("payload changed")
	}

	// Default Trim still copies KV verbatim (nextn stays 1).
	verbatim := filepath.Join(dir, "verbatim.gguf")
	if err := Trim(context.Background(), anchorPath, keep, verbatim, nil); err != nil {
		t.Fatal(err)
	}
	vs, err := OpenSource(verbatim)
	if err != nil {
		t.Fatal(err)
	}
	defer vs.Close()
	vf, err := Parse(vs)
	if err != nil {
		t.Fatal(err)
	}
	v, _ = vf.Meta("qwen3.nextn_predict_layers")
	if n, _ := v.Scalar.(uint32); n != 1 {
		t.Fatalf("Trim mutated nextn to %v", v.Scalar)
	}
}
