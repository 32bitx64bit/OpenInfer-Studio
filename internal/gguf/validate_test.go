package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// buildGGUFWithTensors writes a small GGUF with a tensor table and padded
// data section. tensors: name, type, dims, offset.
func buildGGUFWithTensors(t *testing.T, tensors []struct {
	name string
	typ  uint32
	dims []uint64
	off  uint64
}, dataPad int) string {
	t.Helper()
	var b bytes.Buffer
	w32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	w64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	wstr := func(s string) { w64(uint64(len(s))); b.WriteString(s) }

	binary.Write(&b, binary.LittleEndian, uint32(magicGGUF))
	w32(3)
	w64(uint64(len(tensors)))
	w64(1) // one kv: general.architecture
	wstr("general.architecture")
	w32(tString)
	wstr("llama")
	for _, ti := range tensors {
		wstr(ti.name)
		w32(uint32(len(ti.dims)))
		for _, d := range ti.dims {
			w64(d)
		}
		w32(ti.typ)
		w64(ti.off)
	}
	// Pad the data section.
	for b.Len() < dataPad {
		b.WriteByte(0)
	}
	path := filepath.Join(t.TempDir(), "t.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestValidateHealthy(t *testing.T) {
	// Two F32 tensors of 8 elements (32 bytes each), sequential offsets.
	path := buildGGUFWithTensors(t, []struct {
		name string
		typ  uint32
		dims []uint64
		off  uint64
	}{
		{"a.weight", 0, []uint64{8}, 0},
		{"b.weight", 0, []uint64{8}, 32},
	}, 4096)
	issues, _, err := ValidateFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(issues) != 0 {
		t.Errorf("healthy file flagged: %v", issues)
	}
}

func TestValidateOverlap(t *testing.T) {
	// a.weight claims 64 bytes (16 F32) but b starts at offset 32.
	path := buildGGUFWithTensors(t, []struct {
		name string
		typ  uint32
		dims []uint64
		off  uint64
	}{
		{"a.weight", 0, []uint64{16}, 0},
		{"b.weight", 0, []uint64{8}, 32},
	}, 4096)
	issues, _, err := ValidateFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(issues) == 0 {
		t.Fatal("overlap not detected")
	}
}

func TestValidateDecreasingOffset(t *testing.T) {
	path := buildGGUFWithTensors(t, []struct {
		name string
		typ  uint32
		dims []uint64
		off  uint64
	}{
		{"a.weight", 0, []uint64{8}, 64},
		{"b.weight", 0, []uint64{8}, 32}, // goes backwards
	}, 4096)
	issues, _, err := ValidateFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(issues) == 0 {
		t.Fatal("decreasing offset not detected")
	}
}

func TestValidateTruncated(t *testing.T) {
	// Last tensor wants 1024 F32 (4096 bytes) but file is far too small.
	path := buildGGUFWithTensors(t, []struct {
		name string
		typ  uint32
		dims []uint64
		off  uint64
	}{
		{"a.weight", 0, []uint64{1024}, 0},
	}, 2048)
	issues, _, err := ValidateFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(issues) == 0 {
		t.Fatal("truncation not detected")
	}
}

func TestValidateUnknownTypeTolerated(t *testing.T) {
	// Unknown tensor type id: monotonic offsets pass, no false positive.
	path := buildGGUFWithTensors(t, []struct {
		name string
		typ  uint32
		dims []uint64
		off  uint64
	}{
		{"a.weight", 200, []uint64{16}, 0},
		{"b.weight", 0, []uint64{8}, 128},
	}, 4096)
	issues, _, err := ValidateFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(issues) != 0 {
		t.Errorf("unknown type must not produce false positives: %v", issues)
	}
}

func TestValidateSSMConv1dMustBeF32(t *testing.T) {
	path := buildGGUFWithTensors(t, []struct {
		name string
		typ  uint32
		dims []uint64
		off  uint64
	}{
		{"blk.0.ssm_conv1d.weight", 30, []uint64{4, 64}, 0}, // bf16
		{"blk.0.attn_qkv.weight", 30, []uint64{256, 64}, 512},
	}, 4096)
	issues, _, err := ValidateFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(issues) == 0 || !strings.Contains(issues[0], "ssm_conv1d") {
		t.Fatalf("bf16 ssm_conv1d must be flagged, got %v", issues)
	}

	ok := buildGGUFWithTensors(t, []struct {
		name string
		typ  uint32
		dims []uint64
		off  uint64
	}{
		{"blk.0.ssm_conv1d.weight", 0, []uint64{4, 64}, 0}, // f32
		{"blk.0.attn_qkv.weight", 30, []uint64{256, 64}, 1024},
	}, 65536)
	issues, _, err = ValidateFile(ok)
	if err != nil {
		t.Fatal(err)
	}
	if len(issues) != 0 {
		t.Fatalf("f32 ssm_conv1d must not be flagged, got %v", issues)
	}
}
