package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The fixtures in testdata/dequant were produced by ggml itself
// (dequantize_row_* and ggml_fp32_to_fp16_row from a stable-diffusion.cpp
// build) from random valid blocks, so these tests pin the decoders to the
// reference rather than to this file's reading of it.

func readF32s(t *testing.T, name string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "dequant", name))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

func TestDequantMatchesGGML(t *testing.T) {
	for _, c := range []struct {
		name string
		typ  uint32
	}{{"q8_0", 8}, {"q4_k", 12}, {"q5_k", 13}, {"q6_k", 14}} {
		t.Run(c.name, func(t *testing.T) {
			blocks, err := os.ReadFile(filepath.Join("testdata", "dequant", c.name+".blocks"))
			if err != nil {
				t.Fatal(err)
			}
			want := readF32s(t, c.name+".f32")
			tt := tensorTypes[c.typ]
			n := len(blocks) / int(tt.typeSize)
			got := dequantizers[c.typ](blocks, n)
			if len(got) != len(want) || uint64(len(got)) != uint64(n)*tt.blockSize {
				t.Fatalf("decoded %d values from %d blocks, ggml gave %d", len(got), n, len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("value %d: got %v, ggml %v", i, got[i], want[i])
				}
			}
		})
	}
}

func TestFloat32ToHalfMatchesGGML(t *testing.T) {
	in := readF32s(t, "f16.in")
	raw, err := os.ReadFile(filepath.Join("testdata", "dequant", "f16.out"))
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range in {
		want := binary.LittleEndian.Uint16(raw[i*2:])
		got, ok := float32ToHalf(v)
		if !ok || got != want {
			t.Errorf("float32ToHalf(%g) = %#04x ok=%v, ggml %#04x", v, got, ok, want)
		}
	}
}

func TestFloat32ToHalfRoundTripsEveryHalf(t *testing.T) {
	for h := 0; h < 1<<16; h++ {
		if h&0x7c00 == 0x7c00 { // Inf / NaN
			continue
		}
		f := math.Float32frombits(halfToFloat32Bits(uint16(h)))
		got, ok := float32ToHalf(f)
		if !ok || got != uint16(h) {
			t.Fatalf("half %#04x -> %v -> %#04x ok=%v", h, f, got, ok)
		}
	}
}

func TestFloat32ToHalfOverflow(t *testing.T) {
	for _, v := range []float32{65520, 1e6, -1e9} {
		if _, ok := float32ToHalf(v); ok {
			t.Errorf("float32ToHalf(%g) reported a finite F16", v)
		}
	}
	if _, err := encodeF16(nil, []float32{1, 70000}); err == nil {
		t.Error("encodeF16 accepted a value beyond F16 range")
	}
}
