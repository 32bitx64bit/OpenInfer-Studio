package profile

import (
	"path/filepath"
	"testing"
)

func TestSharpenValuesPeaks(t *testing.T) {
	v := []float32{1, 1, 1, 8}
	got := SharpenValues(v, 2)
	if got[3] <= v[3] {
		t.Fatalf("peak %v should grow, got %v", v[3], got[3])
	}
	if got[0] >= v[0] {
		t.Fatalf("floor %v should shrink, got %v", v[0], got[0])
	}
}

func TestWriteSharpenedImatrixGGUF(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "im.gguf")
	writeImatrixGGUF(t, src,
		map[string][]float32{"blk.0.attn_q.weight": {1, 1, 1, 16}},
		map[string][]float32{"blk.0.attn_q.weight": {4}})
	dst := filepath.Join(dir, "fti.gguf")
	if err := WriteSharpenedImatrix(src, dst, 2); err != nil {
		t.Fatal(err)
	}
	orig, err := LoadImatrix(src)
	if err != nil {
		t.Fatal(err)
	}
	got, err := LoadImatrix(dst)
	if err != nil {
		t.Fatal(err)
	}
	o := orig["blk.0.attn_q.weight"]
	g := got["blk.0.attn_q.weight"]
	if len(g.Values) != len(o.Values) {
		t.Fatalf("values %d vs %d", len(g.Values), len(o.Values))
	}
	if g.Values[len(g.Values)-1] <= o.Values[len(o.Values)-1] {
		t.Fatalf("sharpened peak %v <= original %v", g.Values[len(g.Values)-1], o.Values[len(o.Values)-1])
	}
}

func TestWriteImatrixGGUFFromStats(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "im.gguf")
	st := map[string]ImatrixStats{
		"blk.0.ffn_up.weight": {Mean: 2, Values: []float32{1, 2, 3}, Samples: 8},
	}
	if err := WriteImatrixGGUF(p, st); err != nil {
		t.Fatal(err)
	}
	got, err := LoadImatrix(p)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := got["blk.0.ffn_up.weight"]; !ok {
		t.Fatalf("missing tensor: %v", got)
	}
}
