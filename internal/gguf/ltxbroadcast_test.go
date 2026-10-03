package gguf

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func ltxFixture(t *testing.T) string {
	t.Helper()
	return writeComfyGGUF(t, []comfyFixTensor{
		{name: "keyframes_abs_pos_embedding", typ: 1, dims: []uint64{5}, data: []byte{0, 0x3c, 0, 0xc1, 1, 0, 0, 0x80, 0, 0x7c}},
		{name: "audio_scale_shift_table", typ: 30, dims: []uint64{4}, data: []byte{0x80, 0x3f, 0x20, 0xc0, 1, 0, 0, 0x80}},
		{name: "transformer_blocks.0.scale_shift_table", typ: 1, dims: []uint64{16}, data: make([]byte, 32)},
		{name: "transformer_blocks.0.attn_qkv.weight", typ: 2, dims: []uint64{32}, data: pattern(18, 17)},
	})
}

func TestRepairLTXBroadcastLosslessAndCanonical(t *testing.T) {
	src := ltxFixture(t)
	original, _ := os.ReadFile(src)
	st, err := InspectLTXBroadcast(src)
	if err != nil || st.TensorCount != 3 || st.AddedBytes != 50 {
		t.Fatalf("inspection = %+v, %v", st, err)
	}
	dst := filepath.Join(t.TempDir(), "fixed.gguf")
	var progress []int64
	got, err := RepairLTXBroadcast(context.Background(), src, dst, func(done, total int64) {
		if done < 0 || done > total || (len(progress) > 0 && done < progress[len(progress)-1]) {
			t.Errorf("invalid progress %d/%d", done, total)
		}
		progress = append(progress, done)
	})
	if err != nil || got != st {
		t.Fatalf("repair = %+v, %v", got, err)
	}
	if len(progress) < 2 || progress[len(progress)-1] != st.OutputBytes {
		t.Fatalf("incomplete progress: %v", progress)
	}
	if issues, _, err := ValidateFile(dst); err != nil || len(issues) > 0 {
		t.Fatalf("invalid copy: %v %v", issues, err)
	}
	if st, err := InspectLTXBroadcast(dst); err != nil || st.Needed() {
		t.Fatalf("copy still needs repair: %+v %v", st, err)
	}
	_, beforeMD, _ := ListTensors(src)
	_, afterMD, _ := ListTensors(dst)
	if !reflect.DeepEqual(beforeMD.Raw, afterMD.Raw) {
		t.Fatal("metadata changed")
	}
	for name, want := range map[string][]uint32{
		"keyframes_abs_pos_embedding": {0x3f800000, 0xc0200000, 0x33800000, 0x80000000, 0x7f800000},
		"audio_scale_shift_table":     {0x3f800000, 0xc0200000, 0x00010000, 0x80000000},
	} {
		b := payload(t, dst, name)
		for i, bits := range want {
			if got := binary.LittleEndian.Uint32(b[i*4:]); got != bits {
				t.Fatalf("%s value %d = %x, want %x", name, i, got, bits)
			}
		}
	}
	if !bytes.Equal(payload(t, dst, "transformer_blocks.0.attn_qkv.weight"), pattern(18, 17)) {
		t.Fatal("quantized weights changed")
	}
	f, err := os.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	l, err := readRepairLayout(f)
	if err != nil {
		t.Fatal(err)
	}
	var off uint64
	for _, ten := range l.tensors {
		if ten.offset != off {
			t.Fatalf("noncanonical offset for %s: %d, want %d", ten.name, ten.offset, off)
		}
		off += uint64(align(int64(ten.size), int64(l.alignment)))
	}
	if after, _ := os.ReadFile(src); !bytes.Equal(original, after) {
		t.Fatal("source was modified")
	}
}

func TestRepairLTXBroadcastCancellationAndExistingDest(t *testing.T) {
	src := ltxFixture(t)
	dst := filepath.Join(t.TempDir(), "fixed.gguf")
	ctx, cancel := context.WithCancel(context.Background())
	_, err := RepairLTXBroadcast(ctx, src, dst, func(_, _ int64) { cancel() })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled repair: %v", err)
	}
	if _, err := os.Stat(dst); !os.IsNotExist(err) {
		t.Fatalf("partial copy left behind: %v", err)
	}
	before, _ := os.ReadFile(src)
	if _, err := RepairLTXBroadcast(context.Background(), src, src, nil); err == nil {
		t.Fatal("in-place repair was accepted")
	}
	if after, _ := os.ReadFile(src); !bytes.Equal(before, after) {
		t.Fatal("source changed")
	}
}

func TestLTXBroadcastRequiresModelSignature(t *testing.T) {
	src := writeComfyGGUF(t, []comfyFixTensor{{name: "keyframes_abs_pos_embedding", typ: 1, dims: []uint64{16}, data: make([]byte, 32)}})
	st, err := InspectLTXBroadcast(src)
	if err != nil || st.Needed() {
		t.Fatalf("a single name triggered repair: %+v %v", st, err)
	}
}

func TestLTXBroadcastReportsKnownModelWithUnsupportedParameterSize(t *testing.T) {
	src := writeComfyGGUF(t, []comfyFixTensor{
		{name: "keyframes_abs_pos_embedding", typ: 1, dims: []uint64{1<<20 + 1}, data: make([]byte, (1<<20+1)*2)},
		{name: "audio_scale_shift_table", typ: 1, dims: []uint64{16}, data: make([]byte, 32)},
		{name: "transformer_blocks.0.scale_shift_table", typ: 1, dims: []uint64{16}, data: make([]byte, 32)},
	})
	st, err := InspectLTXBroadcast(src)
	if err == nil || !st.Detected {
		t.Fatalf("known unsupported LTX was treated as an unrelated model: %+v %v", st, err)
	}
}
