package profile

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"quantlab/core"
	"quantlab/tensorbank"
)

func writeF32GGUF(t *testing.T, path string, tensors map[string][]float32) {
	t.Helper()
	type rec struct {
		name  string
		vals  []float32
		rel   uint64
		elems uint64
	}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	var recs []rec
	var cur uint64
	for _, name := range names {
		vals := tensors[name]
		cur = (cur + 31) / 32 * 32
		recs = append(recs, rec{name: name, vals: vals, rel: cur, elems: uint64(len(vals))})
		cur += uint64(len(vals)) * 4
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var hdr [24]byte
	binary.LittleEndian.PutUint32(hdr[0:4], 0x46554747)
	binary.LittleEndian.PutUint32(hdr[4:8], 3)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(len(recs)))
	binary.LittleEndian.PutUint64(hdr[16:24], 2)
	f.Write(hdr[:])
	writeKVString(f, "general.type", "imatrix")
	writeKVUint32(f, "general.alignment", 32)
	for _, r := range recs {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], uint64(len(r.name)))
		f.Write(b[:])
		f.Write([]byte(r.name))
		var n [4]byte
		binary.LittleEndian.PutUint32(n[:], 1)
		f.Write(n[:])
		binary.LittleEndian.PutUint64(b[:], r.elems)
		f.Write(b[:])
		binary.LittleEndian.PutUint32(n[:], 0)
		f.Write(n[:])
		binary.LittleEndian.PutUint64(b[:], r.rel)
		f.Write(b[:])
	}
	metaEnd, _ := f.Seek(0, io.SeekCurrent)
	dataStart := (uint64(metaEnd) + 31) / 32 * 32
	if pad := dataStart - uint64(metaEnd); pad > 0 {
		f.Write(make([]byte, pad))
	}
	for _, r := range recs {
		abs := dataStart + r.rel
		if pos, _ := f.Seek(0, io.SeekCurrent); pos < int64(abs) {
			f.Write(make([]byte, abs-uint64(pos)))
		}
		buf := make([]byte, len(r.vals)*4)
		for i, v := range r.vals {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		f.Write(buf)
	}
}

func writeImatrixGGUF(t *testing.T, path string, sum2, counts map[string][]float32) {
	t.Helper()
	tensors := map[string][]float32{}
	for name, v := range sum2 {
		tensors[name+".in_sum2"] = v
		if c, ok := counts[name]; ok {
			tensors[name+".counts"] = c
		}
	}
	writeF32GGUF(t, path, tensors)
}

func writeKVString(w io.Writer, k, v string) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(len(k)))
	w.Write(b[:])
	w.Write([]byte(k))
	var typ [4]byte
	binary.LittleEndian.PutUint32(typ[:], uint32(tensorbank.VTString))
	w.Write(typ[:])
	binary.LittleEndian.PutUint64(b[:], uint64(len(v)))
	w.Write(b[:])
	w.Write([]byte(v))
}

func writeKVUint32(w io.Writer, k string, v uint32) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(len(k)))
	w.Write(b[:])
	w.Write([]byte(k))
	var typ [4]byte
	binary.LittleEndian.PutUint32(typ[:], uint32(tensorbank.VTUint32))
	w.Write(typ[:])
	var v4 [4]byte
	binary.LittleEndian.PutUint32(v4[:], v)
	w.Write(v4[:])
}

func writeImatrixDat(t *testing.T, path string, entries map[string][]float32, ncall int32) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	names := make([]string, 0, len(entries))
	for n := range entries {
		names = append(names, n)
	}
	// Stable order is not required by the parser.
	if err := binary.Write(f, binary.LittleEndian, int32(len(names))); err != nil {
		t.Fatal(err)
	}
	for _, name := range names {
		vals := entries[name]
		if err := binary.Write(f, binary.LittleEndian, int32(len(name))); err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write([]byte(name)); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(f, binary.LittleEndian, ncall); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(f, binary.LittleEndian, int32(len(vals))); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(f, binary.LittleEndian, vals); err != nil {
			t.Fatal(err)
		}
	}
}

func TestLoadImatrixGGUF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "im.gguf")
	writeImatrixGGUF(t, path,
		map[string][]float32{
			"blk.0.attn_q.weight":   {10, 20, 30, 40},
			"blk.0.ffn_up.weight":   {2, 2, 2, 2},
			"blk.0.ffn_down.weight": {1, 1},
		},
		map[string][]float32{
			"blk.0.attn_q.weight": {4},
			"blk.0.ffn_up.weight": {2},
		},
	)
	got, err := LoadImatrix(path)
	if err != nil {
		t.Fatal(err)
	}
	q, ok := got["blk.0.attn_q.weight"]
	if !ok {
		t.Fatalf("missing attn_q: %v", got)
	}
	// mean((10,20,30,40)/4) = 25/4 = 6.25
	if math.Abs(q.Mean-6.25) > 1e-6 {
		t.Errorf("attn_q mean = %v, want 6.25", q.Mean)
	}
	if q.Max != 10 {
		t.Errorf("attn_q max = %v, want 10", q.Max)
	}
	if q.Samples != 4 {
		t.Errorf("attn_q samples = %d, want 4", q.Samples)
	}
	if q.P50 <= 0 || q.P95 < q.P50 {
		t.Errorf("attn_q percentiles P50=%v P95=%v", q.P50, q.P95)
	}
	if q.Spikiness < 1 {
		t.Errorf("attn_q spikiness = %v, want >= 1", q.Spikiness)
	}
	up := got["blk.0.ffn_up.weight"]
	if math.Abs(up.Mean-1) > 1e-6 {
		t.Errorf("ffn_up mean = %v, want 1", up.Mean)
	}
	down := got["blk.0.ffn_down.weight"]
	if down.Mean != 1 {
		t.Errorf("ffn_down mean = %v, want 1 (no counts)", down.Mean)
	}
}

func TestLoadImatrixDat(t *testing.T) {
	path := filepath.Join(t.TempDir(), "im.dat")
	writeImatrixDat(t, path, map[string][]float32{
		"blk.0.attn_q.weight": {2, 4, 6},
		"token_embd.weight":   {1, 1, 1},
	}, 8)
	got, err := LoadImatrix(path)
	if err != nil {
		t.Fatal(err)
	}
	q := got["blk.0.attn_q.weight"]
	if math.Abs(q.Mean-4) > 1e-6 {
		t.Errorf("mean = %v, want 4", q.Mean)
	}
	if q.Max != 6 {
		t.Errorf("max = %v, want 6", q.Max)
	}
	if q.Samples != 8 {
		t.Errorf("samples = %d, want ncall 8", q.Samples)
	}
}

func TestLoadImatrixNonImatrixGGUF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	writeF32GGUF(t, path, map[string][]float32{
		"blk.0.attn_q.weight": {1, 2, 3, 4},
	})
	got, err := LoadImatrix(path)
	if err != nil {
		t.Fatalf("non-imatrix GGUF should not be a hard error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("got stats from non-imatrix GGUF: %v", got)
	}
}

func TestAggregateImatrixSpikinessFlatVsOneHot(t *testing.T) {
	flat := aggregateImatrix([]float32{1, 1, 1, 1, 1, 1, 1, 1}, nil, true)
	spiky := aggregateImatrix([]float32{0, 0, 0, 0, 0, 0, 0, 10}, nil, true)
	if flat.Spikiness > 1.05 {
		t.Errorf("flat Spikiness = %v, want ~1", flat.Spikiness)
	}
	if flat.Variance != 0 {
		t.Errorf("flat Variance = %v, want 0", flat.Variance)
	}
	if math.Abs(flat.P50-1) > 1e-9 || math.Abs(flat.P95-1) > 1e-9 {
		t.Errorf("flat percentiles P50=%v P95=%v, want 1", flat.P50, flat.P95)
	}
	if !(spiky.Spikiness > flat.Spikiness*2) {
		t.Fatalf("one-hot Spikiness %v not well above flat %v", spiky.Spikiness, flat.Spikiness)
	}
	if spiky.Variance <= flat.Variance {
		t.Errorf("one-hot variance %v not above flat %v", spiky.Variance, flat.Variance)
	}

	td := core.TensorDesc{
		Name: "blk.1.ffn_up.weight", DType: core.DTypeF16,
		Shape: []uint64{256, 256}, Length: 131072, Elements: 65536,
	}
	lossFlat, _ := NewFallbackEstimator(map[string]ImatrixStats{td.Name: flat}).Estimate(td, core.DTypeQ2_K)
	lossSpiky, _ := NewFallbackEstimator(map[string]ImatrixStats{td.Name: spiky}).Estimate(td, core.DTypeQ2_K)
	if !(lossSpiky > lossFlat) {
		t.Fatalf("spiky Q2_K loss %v not above flat %v", lossSpiky, lossFlat)
	}
}
