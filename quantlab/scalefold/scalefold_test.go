package scalefold

import (
	"encoding/binary"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"quantlab/core"
	"quantlab/profile"
	"quantlab/tensorbank"
)

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
	var val [4]byte
	binary.LittleEndian.PutUint32(val[:], v)
	w.Write(val[:])
}

// writeModel writes a rank-mixed GGUF: name -> (shape, payload floats).
func writeModel(t *testing.T, path string, tensors map[string]struct {
	Shape []uint64
	Vals  []float32
}) {
	t.Helper()
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	for i := 1; i < len(names); i++ {
		for j := i; j > 0 && names[j-1] > names[j]; j-- {
			names[j-1], names[j] = names[j], names[j-1]
		}
	}
	type rec struct {
		name string
		rel  uint64
	}
	var recs []rec
	var cur uint64
	for _, n := range names {
		cur = (cur + 31) / 32 * 32
		recs = append(recs, rec{name: n, rel: cur})
		cur += uint64(len(tensors[n].Vals)) * 4
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
	binary.LittleEndian.PutUint64(hdr[16:24], 4)
	f.Write(hdr[:])
	writeKVString(f, "general.type", "model")
	writeKVString(f, "general.name", "fold-test")
	writeKVString(f, "general.architecture", "llama")
	writeKVUint32(f, "general.alignment", 32)
	for _, r := range recs {
		rec := tensors[r.name]
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], uint64(len(r.name)))
		f.Write(b[:])
		f.Write([]byte(r.name))
		var n [4]byte
		binary.LittleEndian.PutUint32(n[:], uint32(len(rec.Shape)))
		f.Write(n[:])
		for _, d := range rec.Shape {
			binary.LittleEndian.PutUint64(b[:], d)
			f.Write(b[:])
		}
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
		buf := make([]byte, len(tensors[r.name].Vals)*4)
		for i, v := range tensors[r.name].Vals {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		f.Write(buf)
	}
}

// foldToyBank builds a model with an attention cluster and an FFN cluster.
func foldToyBank() (map[string]struct {
	Shape []uint64
	Vals  []float32
}, *core.TensorBank) {
	rng := rand.New(rand.NewSource(11))
	const D = 256 // hidden (norm length = ne0)
	gauss := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(rng.NormFloat64())
		}
		return out
	}
	spiky := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(rng.NormFloat64()) * (0.2 + 4*float32(i%D)/float32(D))
		}
		return out
	}
	ts := map[string]struct {
		Shape []uint64
		Vals  []float32
	}{
		"blk.0.attn_norm.weight":   {[]uint64{D}, gauss(D)},
		"blk.0.ffn_norm.weight":    {[]uint64{D}, gauss(D)},
		"blk.0.attn_q.weight":      {[]uint64{D, 8}, spiky(D * 8)},
		"blk.0.attn_k.weight":      {[]uint64{D, 8}, spiky(D * 8)},
		"blk.0.attn_v.weight":      {[]uint64{D, 8}, spiky(D * 8)},
		"blk.0.attn_output.weight": {[]uint64{D, 8}, spiky(D * 8)},
		"blk.0.ffn_gate.weight":    {[]uint64{D, 16}, spiky(D * 16)},
		"blk.0.ffn_up.weight":      {[]uint64{D, 16}, spiky(D * 16)},
		"blk.0.ffn_down.weight":    {[]uint64{256, 8}, spiky(256 * 8)},
	}
	bank := &core.TensorBank{SourcePath: "toy", ModelID: "fold-test"}
	for name, tv := range ts {
		bank.Tensors = append(bank.Tensors, core.TensorDesc{
			Name: name, DType: core.DTypeF32, Shape: append([]uint64(nil), tv.Shape...),
			Elements: prod(tv.Shape), Length: prod(tv.Shape) * 4,
		})
	}
	return ts, bank
}

func prod(s []uint64) uint64 {
	n := uint64(1)
	for _, d := range s {
		n *= d
	}
	return n
}

func TestDiscover(t *testing.T) {
	_, bank := foldToyBank()
	clusters := Discover(bank)
	if len(clusters) != 2 {
		t.Fatalf("want 2 clusters (attn + ffn), got %d: %+v", len(clusters), clusters)
	}
	var attn, ffn *Cluster
	for i := range clusters {
		switch clusters[i].Norm {
		case "blk.0.attn_norm.weight":
			attn = &clusters[i]
		case "blk.0.ffn_norm.weight":
			ffn = &clusters[i]
		}
	}
	if attn == nil || ffn == nil {
		t.Fatalf("clusters: %+v", clusters)
	}
	for _, c := range attn.Consumers {
		if c == "blk.0.attn_output.weight" || c == "blk.0.ffn_down.weight" {
			t.Errorf("forbidden consumer in attn cluster: %s", c)
		}
	}
	if len(attn.Consumers) != 3 || len(ffn.Consumers) != 2 {
		t.Errorf("consumer sets: attn=%v ffn=%v", attn.Consumers, ffn.Consumers)
	}
}

func TestChooseAlphaFloor(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const D = 512
	// Weight rows with a large second-half range paired with low second-half
	// importance: folding difficulty into those channels must help.
	spiky := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			mag := 1.0
			if i%512 >= 256 {
				mag = 20
			}
			out[i] = float32(rng.NormFloat64()) * float32(mag) * 0.05
		}
		return out
	}
	ts := map[string]struct {
		Shape []uint64
		Vals  []float32
	}{
		"blk.0.attn_norm.weight": {[]uint64{D}, make([]float32, D)},
		"blk.0.attn_q.weight":    {[]uint64{D, 8}, spiky(D * 8)},
		"blk.0.attn_k.weight":    {[]uint64{D, 8}, spiky(D * 8)},
		"blk.0.attn_v.weight":    {[]uint64{D, 8}, spiky(D * 8)},
	}
	for i := range ts["blk.0.attn_norm.weight"].Vals {
		ts["blk.0.attn_norm.weight"].Vals[i] = 1
	}
	dir := t.TempDir()
	model := filepath.Join(dir, "model.gguf")
	writeModel(t, model, ts)
	src, err := tensorbank.OpenSource(model)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	// Per-(row, chunk) importance: first half strong, second half weak.
	// 8 rows x 2 chunks (D=512 -> 2 chunks of 256).
	vals := make([]float32, 8*2)
	for r := 0; r < 8; r++ {
		vals[r*2] = 4.0    // strong first half
		vals[r*2+1] = 0.05 // weak second half
	}
	st := ImatrixVec(vals, 8)
	imatrix := map[string]profile.ImatrixStats{
		"blk.0.attn_q.weight": st,
		"blk.0.attn_k.weight": st,
		"blk.0.attn_v.weight": st,
	}
	clusters := []Cluster{{Norm: "blk.0.attn_norm.weight", Consumers: []string{
		"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"}}}
	got, err := ChooseAlpha(src, clusters, imatrix, core.DTypeQ4_K_T)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d clusters", len(got))
	}
	if got[0].Skipped != "" {
		t.Fatalf("cluster skipped unexpectedly: %s", got[0].Skipped)
	}
	if got[0].ErrAfter > got[0].ErrBefore {
		t.Errorf("chosen fold worse than no-op: after %v before %v", got[0].ErrAfter, got[0].ErrBefore)
	}
	if got[0].Alpha != 0 && got[0].ErrAfter >= got[0].ErrBefore {
		t.Errorf("nonzero alpha should improve the error: after %v before %v", got[0].ErrAfter, got[0].ErrBefore)
	}
	if got[0].Alpha == 0 && got[0].ErrAfter < got[0].ErrBefore {
		t.Errorf("alpha=0 cannot improve over itself")
	}
}

// ImatrixVec builds ImatrixStats with retained per-(row, chunk) values.
func ImatrixVec(vals []float32, rows int) profile.ImatrixStats {
	st := profile.ImatrixStats{Samples: 1, Values: vals}
	return st
}

func TestApply(t *testing.T) {
	ts, _ := foldToyBank()
	dir := t.TempDir()
	model := filepath.Join(dir, "model.gguf")
	writeModel(t, model, ts)
	src, err := tensorbank.OpenSource(model)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	scales := make([]float32, 256)
	for i := range scales {
		scales[i] = 2.0
	}
	clusters := []Cluster{{
		Norm:      "blk.0.attn_norm.weight",
		Consumers: []string{"blk.0.attn_q.weight"},
		Alpha:     0.5,
		Scales:    scales,
	}}
	outPath := filepath.Join(dir, "folded.gguf")
	if err := Apply(src, clusters, outPath); err != nil {
		t.Fatal(err)
	}
	folded, err := tensorbank.OpenSource(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer folded.Close()
	file, err := tensorbank.Parse(folded)
	if err != nil {
		t.Fatal(err)
	}
	read := func(name string) []float32 {
		ti, ok := file.FindTensor(name)
		if !ok {
			t.Fatalf("missing tensor %s", name)
		}
		buf := make([]byte, ti.Length)
		if _, err := folded.ReadAt(buf, file.PayloadOffset(ti)); err != nil {
			t.Fatal(err)
		}
		out := make([]float32, len(buf)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
		return out
	}
	origQ := ts["blk.0.attn_q.weight"].Vals
	foldQ := read("blk.0.attn_q.weight")
	for i := range origQ {
		if got, want := foldQ[i], origQ[i]*2; math.Abs(float64(got-want)) > 1e-5 {
			t.Fatalf("attn_q[%d] = %v, want %v", i, got, want)
		}
	}
	origG := ts["blk.0.attn_norm.weight"].Vals
	foldG := read("blk.0.attn_norm.weight")
	for i := range origG {
		if got, want := foldG[i], origG[i]/2; math.Abs(float64(got-want)) > 1e-5 {
			t.Fatalf("norm[%d] = %v, want %v", i, got, want)
		}
	}
	origU := ts["blk.0.ffn_down.weight"].Vals
	foldU := read("blk.0.ffn_down.weight")
	for i := range origU {
		if foldU[i] != origU[i] {
			t.Fatalf("untouched tensor changed: ffn_down[%d]", i)
		}
	}
}

func TestApplyImatrix(t *testing.T) {
	dir := t.TempDir()
	// imatrix GGUF: in_sum2 [rows], counts placeholder
	ts := map[string]struct {
		Shape []uint64
		Vals  []float32
	}{
		"blk.0.attn_q.weight.in_sum2": {[]uint64{256}, make([]float32, 256)},
	}
	for i := range ts["blk.0.attn_q.weight.in_sum2"].Vals {
		ts["blk.0.attn_q.weight.in_sum2"].Vals[i] = float32(i + 1)
	}
	path := filepath.Join(dir, "im.gguf")
	writeModel(t, path, ts)
	src, err := tensorbank.OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	scales := make([]float32, 256)
	for i := range scales {
		scales[i] = 4.0
	}
	clusters := []Cluster{{
		Norm:      "blk.0.attn_norm.weight",
		Consumers: []string{"blk.0.attn_q.weight"},
		Alpha:     0.5,
		Scales:    scales,
	}}
	outPath := filepath.Join(dir, "im-folded.gguf")
	if err := ApplyImatrix(src, clusters, outPath); err != nil {
		t.Fatal(err)
	}
	fi, err := tensorbank.OpenSource(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer fi.Close()
	file, err := tensorbank.Parse(fi)
	if err != nil {
		t.Fatal(err)
	}
	ti, _ := file.FindTensor("blk.0.attn_q.weight.in_sum2")
	buf := make([]byte, ti.Length)
	if _, err := fi.ReadAt(buf, file.PayloadOffset(ti)); err != nil {
		t.Fatal(err)
	}
	for i := uint64(0); i < 4; i++ {
		got := math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		want := float32(i+1) / 16
		if math.Abs(float64(got-want)) > 1e-5 {
			t.Fatalf("in_sum2[%d] = %v, want %v", i, got, want)
		}
	}
}
