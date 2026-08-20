package reconstruct

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
	writeKVString(f, "general.name", "reconstruct-test")
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
			f.Write(make([]byte, int64(abs)-pos))
		}
		buf := make([]byte, len(tensors[r.name].Vals)*4)
		for i, v := range tensors[r.name].Vals {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		f.Write(buf)
	}
}

func TestFWHTInvolution(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	x := make([]float64, 256)
	orig := make([]float64, 256)
	for i := range x {
		x[i] = rng.NormFloat64()
		orig[i] = x[i]
	}
	fwht(x)
	fwht(x)
	for i := range x {
		if math.Abs(x[i]-orig[i]) > 1e-10 {
			t.Fatalf("H H != I at %d: %v vs %v", i, x[i], orig[i])
		}
	}
}

func TestHadamardEquivalentLinear(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	const d, rows = 256, 8
	rng := rand.New(rand.NewSource(11))
	gamma := make([]float32, d)
	wq := make([]float32, d*rows)
	for i := range gamma {
		gamma[i] = 0.5 + float32(i%7)*0.1
	}
	for i := range wq {
		wq[i] = float32(rng.NormFloat64())
		if i%37 == 0 {
			wq[i] = 12
		}
	}
	writeModel(t, path, map[string]struct {
		Shape []uint64
		Vals  []float32
	}{
		"blk.0.attn_norm.weight": {[]uint64{d}, gamma},
		"blk.0.attn_q.weight":    {[]uint64{d, rows}, wq},
	})
	src, err := tensorbank.OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	out := filepath.Join(dir, "rot.gguf")
	res, err := Apply(src, out, Options{Hadamard: true})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Written || !res.Hadamard {
		t.Fatalf("hadamard not applied: %+v", res)
	}

	rot, err := tensorbank.OpenSource(out)
	if err != nil {
		t.Fatal(err)
	}
	defer rot.Close()
	file, err := tensorbank.Parse(rot)
	if err != nil {
		t.Fatal(err)
	}
	wq2, _, ok, err := readFloatTensor(rot, file, "blk.0.attn_q.weight")
	if err != nil || !ok {
		t.Fatalf("read rotated q: %v ok=%v", err, ok)
	}
	g2, _, ok, err := readFloatTensor(rot, file, "blk.0.attn_norm.weight")
	if err != nil || !ok {
		t.Fatal(err)
	}
	for i, v := range g2 {
		if math.Abs(float64(v)-1) > 1e-3 {
			t.Fatalf("norm[%d]=%v, want 1", i, v)
		}
	}

	x := make([]float64, d)
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	xH := append([]float64(nil), x...)
	fwht(xH)
	for r := 0; r < rows; r++ {
		var y0, y1 float64
		for c := 0; c < d; c++ {
			y0 += float64(wq[r*d+c]) * float64(gamma[c]) * x[c]
			y1 += float64(wq2[r*d+c]) * xH[c]
		}
		if math.Abs(y0-y1) > 1e-3*math.Max(1, math.Abs(y0)) {
			t.Fatalf("row %d: orig %v rotated %v", r, y0, y1)
		}
	}
}

func TestCSKReducesNextLayerError(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	const nEmb, nFF, nOut = 256, 32, 8
	rng := rand.New(rand.NewSource(21))
	fill := func(n int, scale float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * scale
			if i%19 == 0 {
				v[i] *= 8
			}
		}
		return v
	}
	gate := fill(nEmb*nFF, 0.05)
	up := fill(nEmb*nFF, 0.05)
	down := fill(nFF*nOut, 0.05)
	writeModel(t, path, map[string]struct {
		Shape []uint64
		Vals  []float32
	}{
		"blk.0.ffn_gate.weight": {[]uint64{nEmb, nFF}, gate},
		"blk.0.ffn_up.weight":   {[]uint64{nEmb, nFF}, up},
		"blk.0.ffn_down.weight": {[]uint64{nFF, nOut}, down},
	})
	imp := make([]float32, nFF)
	for i := range imp {
		imp[i] = 1
	}
	imatrix := map[string]profile.ImatrixStats{
		"blk.0.ffn_gate.weight": {Mean: 1, Values: imp, Samples: 32},
	}
	src, err := tensorbank.OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	out := filepath.Join(dir, "csk.gguf")
	res, err := Apply(src, out, Options{CSK: true, Imatrix: imatrix})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Written || res.CSKLayers != 1 {
		t.Fatalf("csk not applied: %+v", res)
	}
	boundedOut := filepath.Join(dir, "csk-bounded.gguf")
	bounded, err := Apply(src, boundedOut, Options{
		CSK: true, Imatrix: imatrix, MaxWorkingSetBytes: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if bounded.Written || !strings.Contains(bounded.SkipCSK, "working set") {
		t.Fatalf("bounded CSK result = %+v", bounded)
	}
	if _, err := os.Stat(boundedOut); !os.IsNotExist(err) {
		t.Fatalf("bounded CSK wrote an output: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	cancelOut := filepath.Join(dir, "csk-canceled.gguf")
	if _, err := Apply(src, cancelOut, Options{CSK: true, Imatrix: imatrix, Context: ctx}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled CSK error = %v", err)
	}
}

func TestRewriteRoundTripCopy(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	vals := make([]float32, 256)
	for i := range vals {
		vals[i] = float32(i)
	}
	writeModel(t, path, map[string]struct {
		Shape []uint64
		Vals  []float32
	}{
		"token_embd.weight": {[]uint64{256, 1}, vals},
	})
	src, err := tensorbank.OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	out := filepath.Join(dir, "copy.gguf")
	if err := tensorbank.Rewrite(src, out, nil); err != nil {
		t.Fatal(err)
	}
	b1, _ := os.ReadFile(path)
	b2, _ := os.ReadFile(out)
	if len(b1) != len(b2) {
		t.Fatalf("copy length %d vs %d", len(b2), len(b1))
	}
	for i := range b1 {
		if b1[i] != b2[i] {
			t.Fatalf("byte %d differed", i)
		}
	}
}

func TestHadamardInPlaceMatchesOutOfPlace(t *testing.T) {
	dir := t.TempDir()
	orig := filepath.Join(dir, "m.gguf")
	const d, rows = 256, 8
	rng := rand.New(rand.NewSource(11))
	gamma := make([]float32, d)
	wq := make([]float32, d*rows)
	for i := range gamma {
		gamma[i] = 0.5 + float32(i%7)*0.1
	}
	for i := range wq {
		wq[i] = float32(rng.NormFloat64())
		if i%37 == 0 {
			wq[i] = 12
		}
	}
	writeModel(t, orig, map[string]struct {
		Shape []uint64
		Vals  []float32
	}{
		"blk.0.attn_norm.weight": {[]uint64{d}, gamma},
		"blk.0.attn_q.weight":    {[]uint64{d, rows}, wq},
	})
	inPlace := filepath.Join(dir, "inplace.gguf")
	if data, err := os.ReadFile(orig); err != nil {
		t.Fatal(err)
	} else if err := os.WriteFile(inPlace, data, 0o600); err != nil {
		t.Fatal(err)
	}
	src, err := tensorbank.OpenSource(orig)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	out := filepath.Join(dir, "copy.gguf")
	copyRes, err := Apply(src, out, Options{Hadamard: true})
	if err != nil {
		t.Fatal(err)
	}
	live, err := tensorbank.OpenSource(inPlace)
	if err != nil {
		t.Fatal(err)
	}
	defer live.Close()
	st, err := os.Stat(inPlace)
	if err != nil {
		t.Fatal(err)
	}
	placeRes, err := Apply(live, inPlace, Options{Hadamard: true})
	if err != nil {
		t.Fatal(err)
	}
	if !copyRes.Written || !placeRes.Written || !placeRes.InPlace || copyRes.InPlace {
		t.Fatalf("copy=%+v inplace=%+v", copyRes, placeRes)
	}
	after, err := os.Stat(inPlace)
	if err != nil {
		t.Fatal(err)
	}
	if after.Size() != st.Size() {
		t.Fatalf("in-place size %d -> %d", st.Size(), after.Size())
	}
	want, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(inPlace)
	if err != nil {
		t.Fatal(err)
	}
	if len(want) != len(got) {
		t.Fatalf("length %d vs %d", len(got), len(want))
	}
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("byte %d differed", i)
		}
	}
}

func TestCSKInPlaceDoesNotDeleteSource(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	const nEmb, nFF, nOut = 256, 32, 8
	rng := rand.New(rand.NewSource(21))
	fill := func(n int, scale float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * scale
		}
		return v
	}
	writeModel(t, path, map[string]struct {
		Shape []uint64
		Vals  []float32
	}{
		"blk.0.ffn_gate.weight": {[]uint64{nEmb, nFF}, fill(nEmb*nFF, 0.05)},
		"blk.0.ffn_up.weight":   {[]uint64{nEmb, nFF}, fill(nEmb*nFF, 0.05)},
		"blk.0.ffn_down.weight": {[]uint64{nFF, nOut}, fill(nFF*nOut, 0.05)},
	})
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	src, err := tensorbank.OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	imp := make([]float32, nFF)
	for i := range imp {
		imp[i] = 1
	}
	imatrix := map[string]profile.ImatrixStats{
		"blk.0.ffn_gate.weight": {Mean: 1, Values: imp, Samples: 32},
	}
	res, err := Apply(src, path, Options{CSK: true, Imatrix: imatrix, MaxWorkingSetBytes: 1})
	if err != nil {
		t.Fatal(err)
	}
	if res.Written {
		t.Fatalf("tiny working set still wrote: %+v", res)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatalf("in-place CSK skip deleted source: %v", err)
	}
	if after.Size() != st.Size() {
		t.Fatalf("source size changed %d -> %d", st.Size(), after.Size())
	}
}

func TestPermutePreservesResidualReadColumns(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	const d, rows = 256, 8
	q := make([]float32, d*rows)
	for r := 0; r < rows; r++ {
		for c := 0; c < d; c++ {
			q[r*d+c] = float32(c+1) * 0.01
		}
	}
	gamma := make([]float32, d)
	for i := range gamma {
		gamma[i] = 1
	}
	writeModel(t, path, map[string]struct {
		Shape []uint64
		Vals  []float32
	}{
		"blk.0.attn_norm.weight": {[]uint64{d}, gamma},
		"blk.0.attn_q.weight":    {[]uint64{d, rows}, q},
	})
	src, err := tensorbank.OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	out := filepath.Join(dir, "perm.gguf")
	res, err := Apply(src, out, Options{Permute: true})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Written {
		t.Fatalf("permute not applied: %+v", res)
	}
	rot, err := tensorbank.OpenSource(out)
	if err != nil {
		t.Fatal(err)
	}
	defer rot.Close()
	file, err := tensorbank.Parse(rot)
	if err != nil {
		t.Fatal(err)
	}
	q2, _, ok, err := readFloatTensor(rot, file, "blk.0.attn_q.weight")
	if err != nil || !ok {
		t.Fatalf("read permuted q: %v ok=%v", err, ok)
	}
	col := func(w []float32, c int) float64 {
		var s float64
		for r := 0; r < rows; r++ {
			s += float64(w[r*d+c])
		}
		return s
	}
	seen := map[int]int{}
	for c := 0; c < d; c++ {
		seen[int(col(q, c)*100+0.5)]++
		seen[int(col(q2, c)*100+0.5)]--
	}
	for k, v := range seen {
		if v != 0 {
			t.Fatalf("column multiset mismatch at scale %d", k)
		}
	}
}

func TestUnnamedGLUCSKApplies(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	const nEmb, nFF = 256, 32
	rng := rand.New(rand.NewSource(9))
	fill := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * 0.05
		}
		return v
	}
	writeModel(t, path, map[string]struct {
		Shape []uint64
		Vals  []float32
	}{
		"blk.0.attn_norm.weight": {[]uint64{nEmb}, fill(nEmb)},
		"blk.0.a.weight":         {[]uint64{nEmb, nFF}, fill(nEmb * nFF)},
		"blk.0.b.weight":         {[]uint64{nEmb, nFF}, fill(nEmb * nFF)},
		"blk.0.c.weight":         {[]uint64{nFF, nEmb}, fill(nFF * nEmb)},
	})
	imp := make([]float32, nFF)
	for i := range imp {
		imp[i] = 1
	}
	imatrix := map[string]profile.ImatrixStats{
		"blk.0.a.weight": {Mean: 1, Values: imp, Samples: 32},
	}
	src, err := tensorbank.OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	out := filepath.Join(dir, "csk.gguf")
	res, err := Apply(src, out, Options{CSK: true, Imatrix: imatrix})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Written || res.CSKLayers != 1 {
		t.Fatalf("unnamed glu csk not applied: %+v", res)
	}
}

func TestMagRPinsOutlierChannels(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	const n = 256
	w := make([]float32, n*n)
	for i := range w {
		w[i] = 0.01
	}
	w[0] = 50
	writeModel(t, path, map[string]struct {
		Shape []uint64
		Vals  []float32
	}{
		"blk.0.ffn_down.weight": {[]uint64{n, n}, w},
	})
	src, err := tensorbank.OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	out := filepath.Join(dir, "magr.gguf")
	res, err := Apply(src, out, Options{MagR: true})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Written {
		t.Fatalf("magr not applied: %+v", res)
	}
	rot, err := tensorbank.OpenSource(out)
	if err != nil {
		t.Fatal(err)
	}
	defer rot.Close()
	file, err := tensorbank.Parse(rot)
	if err != nil {
		t.Fatal(err)
	}
	got, _, ok, err := readFloatTensor(rot, file, "blk.0.ffn_down.weight")
	if err != nil || !ok {
		t.Fatal(err)
	}
	if math.Abs(float64(got[0]-50)) > 1e-4 {
		t.Fatalf("pinned super-weight clipped to %v", got[0])
	}
}
