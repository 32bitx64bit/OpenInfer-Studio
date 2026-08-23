package gguf

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestRepairSSMConv1dBF16(t *testing.T) {
	source := buildGGUFWithTensors(t, []struct {
		name string
		typ  uint32
		dims []uint64
		off  uint64
	}{
		{"blk.0.ssm_conv1d.weight", 30, []uint64{16}, 0},
		{"blk.0.attn_qkv.weight", 30, []uint64{16}, 32},
	}, 512)

	f, err := os.OpenFile(source, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	layout, err := readRepairLayout(f)
	if err != nil {
		f.Close()
		t.Fatal(err)
	}
	values := []uint16{0x3f80, 0xc020, 0x0000, 0x7f80} // 1, -2.5, 0, +Inf in BF16.
	for i, value := range values {
		var raw [2]byte
		binary.LittleEndian.PutUint16(raw[:], value)
		if _, err := f.WriteAt(raw[:], layout.headerSize+int64(i*2)); err != nil {
			f.Close()
			t.Fatal(err)
		}
	}
	other := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	if _, err := f.WriteAt(other, layout.headerSize+32); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	issues, _, err := ValidateFile(source)
	if err != nil {
		t.Fatal(err)
	}
	if !RepairableSSMConv1dIssues(source, issues) {
		t.Fatalf("legacy validation issues are not repairable: %v", issues)
	}

	dest := filepath.Join(t.TempDir(), "repaired.gguf")
	var progressCalls int
	status, err := RepairSSMConv1d(context.Background(), source, dest, func(done, total int64) {
		progressCalls++
		if done < 0 || done > total {
			t.Fatalf("repair progress %d/%d", done, total)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if !status.Repairable || status.TensorCount != 1 || status.AddedBytes != 32 {
		t.Fatalf("repair status = %+v", status)
	}
	if progressCalls < 2 {
		t.Fatalf("repair progress called %d times", progressCalls)
	}
	if issues, _, err := ValidateFile(dest); err != nil || len(issues) != 0 {
		t.Fatalf("repaired validation issues=%v err=%v", issues, err)
	}

	repaired, err := os.Open(dest)
	if err != nil {
		t.Fatal(err)
	}
	defer repaired.Close()
	repairedLayout, err := readRepairLayout(repaired)
	if err != nil {
		t.Fatal(err)
	}
	if got := repairedLayout.tensors[0].typ; got != 0 {
		t.Fatalf("conv type = %d, want F32", got)
	}
	if got := repairedLayout.tensors[1].typ; got != 30 {
		t.Fatalf("unrelated tensor type = %d, want BF16", got)
	}
	if got := repairedLayout.tensors[1].offset; got != 64 {
		t.Fatalf("shifted tensor offset = %d, want 64", got)
	}
	for i, want := range []float32{1, -2.5, 0, float32(math.Inf(1))} {
		var raw [4]byte
		if _, err := repaired.ReadAt(raw[:], repairedLayout.headerSize+int64(i*4)); err != nil {
			t.Fatal(err)
		}
		got := math.Float32frombits(binary.LittleEndian.Uint32(raw[:]))
		if got != want {
			t.Fatalf("converted value %d = %v, want %v", i, got, want)
		}
	}
	gotOther := make([]byte, len(other))
	if _, err := repaired.ReadAt(gotOther, repairedLayout.headerSize+64); err != nil {
		t.Fatal(err)
	}
	for i := range other {
		if gotOther[i] != other[i] {
			t.Fatalf("unrelated payload changed: got %v want %v", gotOther, other)
		}
	}
}

func TestRepairSSMConv1dRefusesInPlace(t *testing.T) {
	source := buildGGUFWithTensors(t, []struct {
		name string
		typ  uint32
		dims []uint64
		off  uint64
	}{
		{"blk.0.ssm_conv1d.weight", 30, []uint64{16}, 0},
	}, 256)
	if _, err := RepairSSMConv1d(context.Background(), source, source, nil); err == nil {
		t.Fatal("expected in-place repair refusal")
	}
}
