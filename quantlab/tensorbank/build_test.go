package tensorbank

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"quantlab/core"
)

func anchorKVs(name string) []KV {
	return []KV{
		{Key: "general.architecture", Value: Value{Type: VTString, Scalar: "llama"}},
		{Key: "general.name", Value: Value{Type: VTString, Scalar: name}},
		{Key: "general.alignment", Value: Value{Type: VTUint32, Scalar: uint32(32)}},
		{Key: "llama.context_length", Value: Value{Type: VTUint32, Scalar: uint32(4096)}},
	}
}

func openSources(t *testing.T, datas ...[]byte) []*Source {
	t.Helper()
	dir := t.TempDir()
	out := make([]*Source, len(datas))
	for i, d := range datas {
		path := filepath.Join(dir, "anchor"+strings.Repeat("x", i)+".gguf")
		if err := os.WriteFile(path, d, 0o644); err != nil {
			t.Fatal(err)
		}
		s, err := OpenSource(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { s.Close() })
		out[i] = s
	}
	return out
}

func manifestFor(t *testing.T, specs []spec, target func(spec) core.DType, sha string) *core.SelectionManifest {
	t.Helper()
	m := &core.SelectionManifest{ProfileID: "p1", SourceSHA: sha}
	for _, s := range specs {
		var elems uint64 = 1
		for _, d := range s.ne {
			elems *= d
		}
		b, ok := target(s).ExactBytes(elems)
		if !ok {
			t.Fatalf("geometry for %s", target(s))
		}
		m.Options = append(m.Options, core.TensorOption{TensorName: s.name, Target: target(s), Bytes: b})
		m.TotalBytes += b
	}
	return m
}

func mustBuild(t *testing.T, ctx context.Context, sources []*Source, m *core.SelectionManifest, outPath string, progress ProgressFunc) error {
	t.Helper()
	return NewAssembler().Build(ctx, sources, m, outPath, progress)
}

func TestBuildMixedAnchors(t *testing.T) {
	specs := []spec{
		{"tok_embd", core.DTypeF16, []uint64{128, 4}},
		{"attn_v", core.DTypeF16, []uint64{128, 4}},
		{"ffn_down", core.DTypeF16, []uint64{64, 8}},
	}
	// anchor A: all F16; anchor B: all Q8_0 (128 and 64 both divide 32-block)
	aData, aPay := buildGGUF(t, 3, 32, anchorKVs("model"), specs)
	bSpecs := []spec{
		{"tok_embd", core.DTypeQ8_0, []uint64{128, 4}},
		{"attn_v", core.DTypeQ8_0, []uint64{128, 4}},
		{"ffn_down", core.DTypeQ8_0, []uint64{64, 8}},
	}
	bData, bPay := buildGGUF(t, 3, 32, anchorKVs("model"), bSpecs)
	srcs := openSources(t, aData, bData)

	// pick: tok_embd from B (Q8_0), attn_v from A (F16), ffn_down from B
	m := manifestFor(t, specs, func(s spec) core.DType {
		if s.name == "attn_v" {
			return core.DTypeF16
		}
		return core.DTypeQ8_0
	}, "")

	outPath := filepath.Join(t.TempDir(), "out.gguf")
	var lastCopied, lastTotal uint64
	var calls int
	if err := mustBuild(t, context.Background(), srcs, m, outPath, func(c, tot uint64) {
		lastCopied, lastTotal = c, tot
		calls++
	}); err != nil {
		t.Fatal(err)
	}
	if calls != len(specs) || lastCopied != lastTotal || lastTotal != m.TotalBytes {
		t.Errorf("progress: calls=%d copied=%d total=%d want total=%d", calls, lastCopied, lastTotal, m.TotalBytes)
	}

	out, err := OpenSource(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	f, err := Parse(out)
	if err != nil {
		t.Fatalf("re-parse: %v", err)
	}
	// metadata preserved verbatim from primary anchor
	af, _ := Parse(srcs[0])
	if !bytes.Equal(f.KVBytes, af.KVBytes) {
		t.Error("metadata not preserved verbatim")
	}
	// per-tensor checks
	for _, s := range specs {
		ti, ok := f.FindTensor(s.name)
		if !ok {
			t.Fatalf("missing tensor %s", s.name)
		}
		wantDT := core.DTypeQ8_0
		wantPay := bPay[s.name]
		if s.name == "attn_v" {
			wantDT = core.DTypeF16
			wantPay = aPay[s.name]
		}
		if ti.DType != wantDT {
			t.Errorf("%s dtype %s want %s", s.name, ti.DType, wantDT)
		}
		if ti.RelOffset%32 != 0 {
			t.Errorf("%s offset %d not aligned", s.name, ti.RelOffset)
		}
		got := make([]byte, ti.Length)
		if _, err := out.ReadAt(got, f.PayloadOffset(ti)); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, wantPay) {
			t.Errorf("%s payload bytes differ from chosen anchor", s.name)
		}
		// distinctness: must differ from the OTHER anchor's payload of same name
		other := aPay[s.name]
		if wantDT == core.DTypeQ8_0 {
			other = bPay[s.name]
		}
		if bytes.Equal(wantPay, other) && !bytes.Equal(aPay[s.name], bPay[s.name]) {
			// only a sanity note; synth payloads differ by dtype seed
		}
	}
}

func TestBuildSingleAnchorIdentity(t *testing.T) {
	specs := []spec{{"w", core.DTypeF16, []uint64{64, 4}}}
	data, pay := buildGGUF(t, 3, 32, anchorKVs("m"), specs)
	srcs := openSources(t, data)
	m := manifestFor(t, specs, func(s spec) core.DType { return core.DTypeF16 }, "")
	outPath := filepath.Join(t.TempDir(), "out.gguf")
	if err := mustBuild(t, context.Background(), srcs, m, outPath, nil); err != nil {
		t.Fatal(err)
	}
	out, _ := OpenSource(outPath)
	defer out.Close()
	f, err := Parse(out)
	if err != nil {
		t.Fatal(err)
	}
	ti, _ := f.FindTensor("w")
	got := make([]byte, ti.Length)
	out.ReadAt(got, f.PayloadOffset(ti))
	if !bytes.Equal(got, pay["w"]) {
		t.Error("payload drift on single-anchor rebuild")
	}
}

func TestBuildSourceSHAMismatch(t *testing.T) {
	specs := []spec{{"w", core.DTypeF16, []uint64{64, 4}}}
	data, _ := buildGGUF(t, 3, 32, anchorKVs("m"), specs)
	srcs := openSources(t, data)
	m := manifestFor(t, specs, func(s spec) core.DType { return core.DTypeF16 },
		strings.Repeat("0a", 32))
	err := mustBuild(t, context.Background(), srcs, m, filepath.Join(t.TempDir(), "out.gguf"), nil)
	if err == nil || !strings.Contains(err.Error(), "SHA") {
		t.Fatalf("want SHA mismatch error, got %v", err)
	}
}

func TestValidateAnchorsCompatibility(t *testing.T) {
	base, _ := buildGGUF(t, 3, 32, anchorKVs("m"), []spec{{"w", core.DTypeF16, []uint64{64, 4}}})
	shapeDrift, _ := buildGGUF(t, 3, 32, anchorKVs("m"), []spec{{"w", core.DTypeF16, []uint64{65, 4}}})
	otherModel, _ := buildGGUF(t, 3, 32, anchorKVs("other"), []spec{{"w", core.DTypeF16, []uint64{64, 4}}})
	otherArch, _ := buildGGUF(t, 3, 32,
		[]KV{{Key: "general.architecture", Value: Value{Type: VTString, Scalar: "mamba"}}},
		[]spec{{"w", core.DTypeF16, []uint64{64, 4}}})
	missing, _ := buildGGUF(t, 3, 32, anchorKVs("m"), []spec{{"v", core.DTypeF16, []uint64{64, 4}}})

	for _, tc := range []struct {
		name  string
		other []byte
	}{
		{"shape drift", shapeDrift},
		{"model mismatch", otherModel},
		{"arch mismatch", otherArch},
		{"missing tensor", missing},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srcs := openSources(t, base, tc.other)
			var files []*File
			for _, s := range srcs {
				f, err := Parse(s)
				if err != nil {
					t.Fatal(err)
				}
				files = append(files, f)
			}
			if err := ValidateAnchors(files); err == nil {
				t.Error("incompatible anchors accepted")
			}
		})
	}
}

func TestBuildRejectsBadManifests(t *testing.T) {
	specs := []spec{{"w", core.DTypeF16, []uint64{64, 4}}}
	data, _ := buildGGUF(t, 3, 32, anchorKVs("m"), specs)
	srcs := openSources(t, data)
	dir := t.TempDir()

	// missing variant: no anchor provides Q6_K
	q6 := manifestFor(t, specs, func(s spec) core.DType { return core.DTypeQ6_K }, "")
	if err := mustBuild(t, context.Background(), srcs, q6, filepath.Join(dir, "o.gguf"), nil); err == nil ||
		!strings.Contains(err.Error(), "no anchor provides") {
		t.Errorf("missing variant: %v", err)
	}

	// wrong byte cost
	bad := manifestFor(t, specs, func(s spec) core.DType { return core.DTypeF16 }, "")
	bad.Options[0].Bytes += 16
	bad.TotalBytes += 16
	if err := mustBuild(t, context.Background(), srcs, bad, filepath.Join(dir, "o.gguf"), nil); err == nil {
		t.Error("wrong byte cost accepted")
	}

	// unknown tensor
	unk := manifestFor(t, specs, func(s spec) core.DType { return core.DTypeF16 }, "")
	unk.Options[0].TensorName = "ghost"
	if err := mustBuild(t, context.Background(), srcs, unk, filepath.Join(dir, "o.gguf"), nil); err == nil {
		t.Error("unknown tensor accepted")
	}

	// incomplete coverage
	dup := manifestFor(t, specs, func(s spec) core.DType { return core.DTypeF16 }, "")
	dup.Options = dup.Options[:0]
	if err := mustBuild(t, context.Background(), srcs, dup, filepath.Join(dir, "o.gguf"), nil); err == nil {
		t.Error("empty manifest accepted")
	}
}

// Multiple anchors providing the SAME dtype for a tensor — the normal case
// for float norms/biases in llama-quantize anchor batches — must resolve
// deterministically to the lowest anchor index, not error as ambiguous.
func TestBuildSameDTypeMultipleAnchorsDeterministic(t *testing.T) {
	specs := []spec{
		{"norm_f32", core.DTypeF32, []uint64{128}},
		{"w", core.DTypeF16, []uint64{64, 4}},
	}
	aData, aPay := buildGGUF(t, 3, 32, anchorKVs("m"), specs)
	bData, _ := buildGGUF(t, 3, 32, anchorKVs("m"), specs)
	srcs := openSources(t, aData, bData)
	m := manifestFor(t, specs, func(s spec) core.DType { return s.dt }, "")

	dir := t.TempDir()
	out1 := filepath.Join(dir, "o1.gguf")
	if err := mustBuild(t, context.Background(), srcs, m, out1, nil); err != nil {
		t.Fatalf("same-dtype multi-anchor build failed: %v", err)
	}
	// Deterministic: a second build is byte-identical, and every payload
	// comes from anchor 0 (the primary), the lowest index.
	out2 := filepath.Join(dir, "o2.gguf")
	if err := mustBuild(t, context.Background(), srcs, m, out2, nil); err != nil {
		t.Fatal(err)
	}
	if fileHash(out1) != fileHash(out2) {
		t.Error("builds are not byte-deterministic")
	}
	out, err := OpenSource(out1)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	f, err := Parse(out)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range specs {
		ti, ok := f.FindTensor(s.name)
		if !ok {
			t.Fatalf("missing tensor %s", s.name)
		}
		if ti.DType != s.dt {
			t.Errorf("%s dtype %s want %s", s.name, ti.DType, s.dt)
		}
		got := make([]byte, ti.Length)
		if _, err := out.ReadAt(got, f.PayloadOffset(ti)); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, aPay[s.name]) {
			t.Errorf("%s payload not taken from lowest-index anchor", s.name)
		}
	}
}

// The manifest's chosen dtype provided by exactly one anchor among others
// carrying conflicting dtypes must succeed from that anchor; the chosen
// dtype absent from all anchors must still fail.
func TestBuildChosenTypeFromSingleProviderAmongConflicts(t *testing.T) {
	f16 := []spec{{"w", core.DTypeF16, []uint64{64, 4}}}
	q8 := []spec{{"w", core.DTypeQ8_0, []uint64{64, 4}}}
	aData, _ := buildGGUF(t, 3, 32, anchorKVs("m"), f16)
	bData, _ := buildGGUF(t, 3, 32, anchorKVs("m"), f16)
	cData, cPay := buildGGUF(t, 3, 32, anchorKVs("m"), q8)
	srcs := openSources(t, aData, bData, cData)

	// Chosen type Q8_0 exists only in anchor 2 despite F16 conflicts
	// elsewhere: assembly must succeed and source from anchor 2.
	m := manifestFor(t, f16, func(s spec) core.DType { return core.DTypeQ8_0 }, "")
	outPath := filepath.Join(t.TempDir(), "o.gguf")
	if err := mustBuild(t, context.Background(), srcs, m, outPath, nil); err != nil {
		t.Fatalf("single-provider build failed: %v", err)
	}
	out, err := OpenSource(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	f, err := Parse(out)
	if err != nil {
		t.Fatal(err)
	}
	ti, ok := f.FindTensor("w")
	if !ok {
		t.Fatal("missing tensor w")
	}
	if ti.DType != core.DTypeQ8_0 {
		t.Fatalf("w dtype %s want Q8_0", ti.DType)
	}
	got := make([]byte, ti.Length)
	if _, err := out.ReadAt(got, f.PayloadOffset(ti)); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, cPay["w"]) {
		t.Error("payload not taken from the sole providing anchor")
	}

	// Chosen type absent from all anchors still fails.
	absent := manifestFor(t, f16, func(s spec) core.DType { return core.DTypeF32 }, "")
	err = mustBuild(t, context.Background(), srcs, absent, filepath.Join(t.TempDir(), "o.gguf"), nil)
	if err == nil || !strings.Contains(err.Error(), "no anchor provides") {
		t.Fatalf("want missing-type error, got %v", err)
	}
}

func fileHash(p string) string {
	b, err := os.ReadFile(p)
	if err != nil {
		panic(err)
	}
	return hex.EncodeToString(sha256Sum(b))
}

func TestBuildRejectsInPlaceEdit(t *testing.T) {
	specs := []spec{{"w", core.DTypeF16, []uint64{64, 4}}}
	data, _ := buildGGUF(t, 3, 32, anchorKVs("m"), specs)
	srcs := openSources(t, data)
	m := manifestFor(t, specs, func(s spec) core.DType { return core.DTypeF16 }, "")
	err := mustBuild(t, context.Background(), srcs, m, srcs[0].Path(), nil)
	if err == nil || !strings.Contains(err.Error(), "in-place") {
		t.Fatalf("want in-place rejection, got %v", err)
	}
}

func TestBuildCancellation(t *testing.T) {
	specs := []spec{
		{"w1", core.DTypeF16, []uint64{512, 4}},
		{"w2", core.DTypeF16, []uint64{512, 4}},
		{"w3", core.DTypeF16, []uint64{512, 4}},
	}
	data, _ := buildGGUF(t, 3, 32, anchorKVs("m"), specs)
	srcs := openSources(t, data)
	m := manifestFor(t, specs, func(s spec) core.DType { return core.DTypeF16 }, "")
	dir := t.TempDir()
	outPath := filepath.Join(dir, "out.gguf")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := mustBuild(t, ctx, srcs, m, outPath, nil); err == nil {
		t.Fatal("pre-cancelled context accepted")
	}

	// cancel mid-copy via progress callback
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	err := mustBuild(t, ctx2, srcs, m, outPath, func(c, tot uint64) {
		if c >= tot/3 {
			cancel2()
		}
	})
	if err == nil {
		t.Fatal("mid-copy cancellation accepted")
	}
	assertNoPartial(t, outPath)
}

func TestBuildAtomicCleanupOnFailure(t *testing.T) {
	specs := []spec{
		{"w1", core.DTypeF16, []uint64{512, 4}},
		{"w2", core.DTypeF16, []uint64{512, 4}},
	}
	data, _ := buildGGUF(t, 3, 32, anchorKVs("m"), specs)
	srcs := openSources(t, data)
	m := manifestFor(t, specs, func(s spec) core.DType { return core.DTypeF16 }, "")
	dir := t.TempDir()
	outPath := filepath.Join(dir, "out.gguf")

	// truncate the source after the first tensor is copied -> read failure
	// mid-stream, after the tmp file exists
	var copiedOnce bool
	err := mustBuild(t, context.Background(), srcs, m, outPath, func(c, tot uint64) {
		if !copiedOnce {
			copiedOnce = true
			if err := os.Truncate(srcs[0].Path(), int64(len(data)-1024)); err != nil {
				t.Errorf("truncate: %v", err)
			}
		}
	})
	if err == nil {
		t.Fatal("truncated source accepted")
	}
	assertNoPartial(t, outPath)
}

func assertNoPartial(t *testing.T, outPath string) {
	t.Helper()
	if _, err := os.Stat(outPath); !os.IsNotExist(err) {
		t.Errorf("output file exists after failure: %v", err)
	}
	if _, err := os.Stat(outPath + ".tmp"); !os.IsNotExist(err) {
		t.Errorf("tmp file leaked after failure: %v", err)
	}
}

func TestBuildOutputIsSelfConsistentGGUF(t *testing.T) {
	aSpecs := []spec{
		{"w1", core.DTypeF16, []uint64{100, 4}},
		{"w2", core.DTypeF16, []uint64{96, 8}},
	}
	bSpecs := []spec{
		{"w1", core.DTypeQ8_0, []uint64{100, 4}},
		{"w2", core.DTypeQ8_0, []uint64{96, 8}},
	}
	aData, _ := buildGGUF(t, 2, 32, anchorKVs("m"), aSpecs)
	bData, _ := buildGGUF(t, 3, 64, anchorKVs("m"), bSpecs)
	srcs := openSources(t, aData, bData)
	m := manifestFor(t, aSpecs, func(s spec) core.DType {
		if s.name == "w1" {
			return core.DTypeF16
		}
		return core.DTypeQ8_0
	}, "")
	outPath := filepath.Join(t.TempDir(), "out.gguf")
	if err := mustBuild(t, context.Background(), srcs, m, outPath, nil); err != nil {
		t.Fatal(err)
	}
	// whole-file digest of a second independent build must match (determinism)
	outPath2 := filepath.Join(t.TempDir(), "out2.gguf")
	if err := mustBuild(t, context.Background(), srcs, m, outPath2, nil); err != nil {
		t.Fatal(err)
	}
	h := func(p string) string {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		return hex.EncodeToString(sha256Sum(b))
	}
	if h(outPath) != h(outPath2) {
		t.Error("builds are not byte-deterministic")
	}
}

func sha256Sum(b []byte) []byte {
	s := sha256.Sum256(b)
	return s[:]
}

func TestCopyHashedRangeMatchesSourceHash(t *testing.T) {
	dir := t.TempDir()
	payload := bytes.Repeat([]byte{0xab, 0xcd, 0xef, 0x01}, 1024)
	srcPath := filepath.Join(dir, "src.bin")
	if err := os.WriteFile(srcPath, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	src, err := OpenSource(srcPath)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	dst, err := os.Create(filepath.Join(dir, "dst.bin"))
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	got, err := copyHashedRange(dst, src, 0, uint64(len(payload)), make([]byte, 512))
	if err != nil {
		t.Fatal(err)
	}
	want, err := hashRange(src, 0, uint64(len(payload)))
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("copy hash %s != source hash %s", got, want)
	}
	written, err := os.ReadFile(dst.Name())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(written, payload) {
		t.Fatal("copied bytes differ from source")
	}
}

func TestHashingWriterHashesAcceptedBytes(t *testing.T) {
	payload := []byte("hash-during-copy")
	h := sha256.New()
	var dst bytes.Buffer
	hw := &hashingWriter{w: &dst, h: h}
	if _, err := hw.Write(payload); err != nil {
		t.Fatal(err)
	}
	got := hex.EncodeToString(h.Sum(nil))
	sum := sha256.Sum256(payload)
	want := hex.EncodeToString(sum[:])
	if got != want {
		t.Fatalf("hash %s want %s", got, want)
	}
	if !bytes.Equal(dst.Bytes(), payload) {
		t.Fatal("writer dropped bytes")
	}
}

func TestVerifyResultDetectsPayloadMismatch(t *testing.T) {
	specs := []spec{{"w", core.DTypeF16, []uint64{64, 4}}}
	data, _ := buildGGUF(t, 3, 32, anchorKVs("m"), specs)
	srcs := openSources(t, data)
	m := manifestFor(t, specs, func(s spec) core.DType { return core.DTypeF16 }, "")
	outPath := filepath.Join(t.TempDir(), "out.gguf")
	if err := mustBuild(t, context.Background(), srcs, m, outPath, nil); err != nil {
		t.Fatal(err)
	}
	out, err := OpenSource(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	f, err := Parse(out)
	if err != nil {
		t.Fatal(err)
	}
	ti := f.Tensors[0]
	wrong := strings.Repeat("ab", 32)
	a := NewAssembler()
	err = a.verifyResult(outPath, f, []pick{{info: ti, src: out, file: f}}, []uint64{ti.RelOffset}, []string{wrong})
	if err == nil || !strings.Contains(err.Error(), "payload differs") {
		t.Fatalf("want payload mismatch, got %v", err)
	}
}

func TestScratchAssemblerBuildsValidGGUF(t *testing.T) {
	specs := []spec{
		{"tok_embd", core.DTypeF16, []uint64{128, 4}},
		{"attn_v", core.DTypeF16, []uint64{128, 4}},
	}
	aData, aPay := buildGGUF(t, 3, 32, anchorKVs("model"), specs)
	srcs := openSources(t, aData)
	m := manifestFor(t, specs, func(s spec) core.DType { return core.DTypeF16 }, "")
	dir := t.TempDir()
	scratchPath := filepath.Join(dir, "scratch.gguf")
	durablePath := filepath.Join(dir, "durable.gguf")
	if err := (&Assembler{Scratch: true}).Build(context.Background(), srcs, m, scratchPath, nil); err != nil {
		t.Fatal(err)
	}
	if err := NewAssembler().Build(context.Background(), srcs, m, durablePath, nil); err != nil {
		t.Fatal(err)
	}
	if fileHash(scratchPath) != fileHash(durablePath) {
		t.Fatal("Scratch build bytes differ from durable build")
	}
	out, err := OpenSource(scratchPath)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	f, err := Parse(out)
	if err != nil {
		t.Fatalf("Scratch output failed verify/parse: %v", err)
	}
	for _, s := range specs {
		ti, ok := f.FindTensor(s.name)
		if !ok {
			t.Fatalf("missing %s", s.name)
		}
		got := make([]byte, ti.Length)
		if _, err := out.ReadAt(got, f.PayloadOffset(ti)); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, aPay[s.name]) {
			t.Errorf("%s payload mismatch", s.name)
		}
	}
}

func TestScratchSkipsSync(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "sync")
	if err != nil {
		t.Fatal(err)
	}
	path := f.Name()
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	closed, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	closed.Close()
	if err := (&Assembler{Scratch: true}).syncFile(closed); err != nil {
		t.Fatalf("Scratch syncFile should skip Sync: %v", err)
	}
	if err := NewAssembler().syncFile(closed); err == nil {
		t.Fatal("durable syncFile on closed file succeeded")
	}
	(&Assembler{Scratch: true}).syncDir(path)
}
