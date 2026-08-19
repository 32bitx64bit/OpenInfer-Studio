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
)

// writeF32GGUF2 writes a GGUF with rank-2 tensors shaped [ne0, rows].
func writeF32GGUF2(t *testing.T, path string, ne0, rows int, tensors map[string][]float32) {
	t.Helper()
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	type rec struct {
		name string
		vals []float32
		rel  uint64
	}
	var recs []rec
	var cur uint64
	for _, name := range names {
		vals := tensors[name]
		cur = (cur + 31) / 32 * 32
		recs = append(recs, rec{name: name, vals: vals, rel: cur})
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
	binary.LittleEndian.PutUint64(hdr[16:24], 4)
	f.Write(hdr[:])
	writeKVString(f, "general.type", "model")
	writeKVString(f, "general.name", "exact-test")
	writeKVString(f, "general.architecture", "llama")
	writeKVUint32(f, "general.alignment", 32)
	for _, r := range recs {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], uint64(len(r.name)))
		f.Write(b[:])
		f.Write([]byte(r.name))
		var n [4]byte
		binary.LittleEndian.PutUint32(n[:], 2)
		f.Write(n[:])
		binary.LittleEndian.PutUint64(b[:], uint64(ne0))
		f.Write(b[:])
		binary.LittleEndian.PutUint64(b[:], uint64(rows))
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

func TestBuildExactLossTable(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model.gguf")
	const ne0, rows = 256, 4
	vals := make([]float32, ne0*rows)
	for i := range vals {
		vals[i] = float32(math.Sin(float64(i) * 0.371))
	}
	writeF32GGUF2(t, model, ne0, rows, map[string][]float32{"blk.0.attn_q.weight": vals})

	impPath := filepath.Join(dir, "imp.gguf")
	sum2 := make([]float32, rows)
	counts := make([]float32, rows)
	for r := range sum2 {
		sum2[r] = float32(r+1) * 8
		counts[r] = 8
	}
	writeImatrixGGUF(t, impPath,
		map[string][]float32{"blk.0.attn_q.weight": sum2},
		map[string][]float32{"blk.0.attn_q.weight": counts})

	imatrix, err := LoadImatrix(impPath)
	if err != nil {
		t.Fatal(err)
	}
	st, ok := imatrix["blk.0.attn_q.weight"]
	if !ok {
		t.Fatalf("no imatrix stats: %v", imatrix)
	}
	if len(st.Values) != rows {
		t.Fatalf("Values len %d, want %d", len(st.Values), rows)
	}
	if st.Entropy <= 0 || st.Entropy > 1 {
		t.Errorf("entropy %v out of (0,1]", st.Entropy)
	}
	if st.EffRank < 1 || st.EffRank > float64(rows) {
		t.Errorf("effRank %v out of [1,%d]", st.EffRank, rows)
	}

	bank := &core.TensorBank{
		SourcePath: model,
		ModelID:    "exact-test",
		Tensors: []core.TensorDesc{{
			Name: "blk.0.attn_q.weight", DType: core.DTypeF32,
			Shape: []uint64{ne0, rows}, Elements: ne0 * rows, Length: ne0 * rows * 4,
		}},
	}
	var progCalls int
	table, err := BuildExactLossTable(bank, []core.DType{core.DTypeQ8_0, core.DTypeQ4_K_T}, imatrix,
		func(done, total int64) { progCalls++ })
	if err != nil {
		t.Fatal(err)
	}
	m := table["blk.0.attn_q.weight"]
	if m == nil || len(m) != 2 {
		t.Fatalf("table entry missing: %v", m)
	}
	if m[core.DTypeQ8_0] >= m[core.DTypeQ4_K_T] {
		t.Errorf("Q8_0 sse %v should be below Q4_K sse %v", m[core.DTypeQ8_0], m[core.DTypeQ4_K_T])
	}
	if progCalls == 0 {
		t.Error("progress never called")
	}

	est := NewFallbackEstimator(imatrix)
	est.BindBank(bank)
	l8, _ := est.Estimate(bank.Tensors[0], core.DTypeQ8_0)
	est2 := NewFallbackEstimator(imatrix)
	est2.BindBank(bank)
	est2.SetExactLoss(table)
	x8, xc8 := est2.Estimate(bank.Tensors[0], core.DTypeQ8_0)
	x4, xc4 := est2.Estimate(bank.Tensors[0], core.DTypeQ4_K_T)
	if !(x8 < x4) {
		t.Errorf("exact estimate Q8_0 %v should be below Q4_K %v", x8, x4)
	}
	if xc8 != 0.9 || xc4 != 0.9 {
		t.Errorf("exact confidence (%v,%v), want 0.9", xc8, xc4)
	}
	if !(x8 < l8*1e3) {
		t.Errorf("exact Q8_0 loss %v wildly above heuristic %v", x8, l8)
	}
	if !est2.HasExactLoss("blk.0.attn_q.weight", core.DTypeQ8_0) {
		t.Error("HasExactLoss false")
	}
	if est.HasExactLoss("blk.0.attn_q.weight", core.DTypeQ8_0) {
		t.Error("HasExactLoss true without table")
	}
}

func TestBuildExactLossTableUniform(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model.gguf")
	const ne0, rows = 256, 2
	vals := make([]float32, ne0*rows)
	for i := range vals {
		vals[i] = float32(float64(i%37)-18) * 0.125
	}
	writeF32GGUF2(t, model, ne0, rows, map[string][]float32{"blk.0.ffn_up.weight": vals})
	bank := &core.TensorBank{
		SourcePath: model,
		ModelID:    "exact-uniform",
		Tensors: []core.TensorDesc{{
			Name: "blk.0.ffn_up.weight", DType: core.DTypeF32,
			Shape: []uint64{ne0, rows}, Elements: ne0 * rows, Length: ne0 * rows * 4,
		}},
	}
	table, err := BuildExactLossTable(bank, []core.DType{core.DTypeQ6_K}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	m := table["blk.0.ffn_up.weight"]
	if m == nil {
		t.Fatal("missing entry")
	}
	if m[core.DTypeQ6_K] <= 0 {
		t.Errorf("uniform sse %v should be > 0", m[core.DTypeQ6_K])
	}
	est := NewFallbackEstimator(nil)
	est.SetExactLoss(table)
	_, c := est.Estimate(bank.Tensors[0], core.DTypeQ6_K)
	if c != 0.8 {
		t.Errorf("uniform exact confidence %v, want 0.8", c)
	}
}

func TestBuildExactLossTableProbeKLD(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model.gguf")
	const ne0, rows = 256, 8
	vals := make([]float32, ne0*rows)
	for i := range vals {
		vals[i] = float32(math.Sin(float64(i)*0.371) * 0.5)
		if i%64 == 0 {
			vals[i] = 6
		}
	}
	writeF32GGUF2(t, model, ne0, rows, map[string][]float32{"blk.0.attn_v.weight": vals})
	bank := &core.TensorBank{
		SourcePath: model,
		ModelID:    "probe-kld",
		Tensors: []core.TensorDesc{{
			Name: "blk.0.attn_v.weight", DType: core.DTypeF32,
			Shape: []uint64{ne0, rows}, Elements: ne0 * rows, Length: ne0 * rows * 4,
		}},
	}
	plain, err := BuildExactLossTable(bank, []core.DType{core.DTypeQ8_0, core.DTypeQ4_K_T}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	probed, err := BuildExactLossTableCfg(bank, []core.DType{core.DTypeQ8_0, core.DTypeQ4_K_T}, nil, nil, ExactConfig{ProbeKLD: true})
	if err != nil {
		t.Fatal(err)
	}
	p, b := probed["blk.0.attn_v.weight"], plain["blk.0.attn_v.weight"]
	if p == nil || b == nil {
		t.Fatal("missing table")
	}
	if !(p[core.DTypeQ8_0] < p[core.DTypeQ4_K_T]) {
		t.Errorf("probe-KLD Q8_0 %v should stay below Q4_K %v", p[core.DTypeQ8_0], p[core.DTypeQ4_K_T])
	}
	if p[core.DTypeQ4_K_T] == b[core.DTypeQ4_K_T] {
		t.Error("probe-KLD should change Q4_K loss vs pure IWSE")
	}
}
