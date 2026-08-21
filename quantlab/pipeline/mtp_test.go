package pipeline

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"testing"

	"quantlab/core"
	"quantlab/mtp"
	"quantlab/state"
	"quantlab/tensorbank"
)

func mtpTensors() []gtensor {
	return []gtensor{
		{"token_embd.weight", core.DTypeF32, []uint64{8}},
		{"blk.0.attn_q.weight", core.DTypeF32, []uint64{8}},
		{"blk.63.attn_q.weight", core.DTypeF32, []uint64{8}},
		{"blk.64.attn_q.weight", core.DTypeF32, []uint64{8}},
		{"nextn.eh_proj.weight", core.DTypeF32, []uint64{8}},
		{"output.weight", core.DTypeF32, []uint64{8}},
	}
}

func writeMTPGGUF(path string, tensors []gtensor, arch string, block, nextn uint32) error {
	type rec struct {
		t   gtensor
		rel uint64
	}
	var recs []rec
	var cur uint64
	for _, t := range tensors {
		elems := uint64(1)
		for _, d := range t.shape {
			elems *= d
		}
		cur = alignUpU(cur, 32)
		recs = append(recs, rec{t, cur})
		l, _ := t.dt.ExactBytes(elems)
		cur += l
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var hdr [24]byte
	binary.LittleEndian.PutUint32(hdr[0:4], 0x46554747)
	binary.LittleEndian.PutUint32(hdr[4:8], 3)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(len(recs)))
	binary.LittleEndian.PutUint64(hdr[16:24], 5)
	if _, err := f.Write(hdr[:]); err != nil {
		return err
	}
	writeKVString(f, "general.architecture", arch)
	writeKVString(f, "general.name", "mtp-tiny")
	writeKVUint32(f, "general.alignment", 32)
	writeKVUint32(f, arch+".block_count", block)
	writeKVUint32(f, arch+".nextn_predict_layers", nextn)
	for _, r := range recs {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], uint64(len(r.t.name)))
		f.Write(b[:])
		f.Write([]byte(r.t.name))
		var n [4]byte
		binary.LittleEndian.PutUint32(n[:], uint32(len(r.t.shape)))
		f.Write(n[:])
		for _, d := range r.t.shape {
			binary.LittleEndian.PutUint64(b[:], d)
			f.Write(b[:])
		}
		binary.LittleEndian.PutUint32(n[:], ggmlID(r.t.dt))
		f.Write(n[:])
		binary.LittleEndian.PutUint64(b[:], r.rel)
		f.Write(b[:])
	}
	metaEnd, _ := f.Seek(0, io.SeekCurrent)
	dataStart := alignUpU(uint64(metaEnd), 32)
	if pad := dataStart - uint64(metaEnd); pad > 0 {
		f.Write(make([]byte, pad))
	}
	for i, r := range recs {
		abs := dataStart + r.rel
		if pos, _ := f.Seek(0, io.SeekCurrent); pos < int64(abs) {
			f.Write(make([]byte, abs-uint64(pos)))
		}
		elems := uint64(1)
		for _, d := range r.t.shape {
			elems *= d
		}
		l, _ := r.t.dt.ExactBytes(elems)
		payload := bytes.Repeat([]byte{byte(i + 1)}, int(l))
		f.Write(payload)
	}
	return nil
}

func assembleMTPEngine(t *testing.T, src string, bpw float64, budget uint64) *Engine {
	t.Helper()
	dir := t.TempDir()
	work := filepath.Join(dir, "work")
	if err := os.MkdirAll(work, 0o755); err != nil {
		t.Fatal(err)
	}
	s, err := tensorbank.OpenSource(src)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	bank, err := tensorbank.NewAssembler().Assemble(s, src, "")
	if err != nil {
		t.Fatal(err)
	}
	return &Engine{
		Store: state.Store{Dir: dir},
		Run: &state.Run{
			RunID: "mtp",
			Config: state.RunConfig{
				SourcePath:  src,
				WorkDir:     work,
				TargetBPW:   bpw,
				BudgetBytes: budget,
			},
			Bank: bank,
		},
		Out: io.Discard,
	}
}

func TestStageDropMTPStripsFusedHeads(t *testing.T) {
	src := filepath.Join(t.TempDir(), "src.gguf")
	if err := writeMTPGGUF(src, mtpTensors(), "qwen3", 64, 1); err != nil {
		t.Fatal(err)
	}
	origSize, _ := os.Stat(src)
	e := assembleMTPEngine(t, src, 2.0, 1<<20)
	budget := e.Run.Config.BudgetBytes
	nBefore := len(e.Run.Bank.Tensors)
	if err := e.stageDropMTP(context.Background()); err != nil {
		t.Fatal(err)
	}
	if e.Run.Config.BudgetBytes != budget {
		t.Fatalf("BudgetBytes changed: %d -> %d", budget, e.Run.Config.BudgetBytes)
	}
	if e.Extra.MTPStrippedPath == "" {
		t.Fatal("MTPStrippedPath not set")
	}
	if got := e.payloadSource(); got != e.Extra.MTPStrippedPath {
		t.Fatalf("payloadSource=%s", got)
	}
	if samePath(e.payloadSource(), src) {
		t.Fatal("payload still original source")
	}
	names := map[string]struct{}{}
	for _, tn := range e.Run.Bank.Tensors {
		names[tn.Name] = struct{}{}
	}
	for _, wantGone := range []string{"blk.64.attn_q.weight", "nextn.eh_proj.weight"} {
		if _, ok := names[wantGone]; ok {
			t.Fatalf("bank still has %s", wantGone)
		}
	}
	for _, want := range []string{"blk.0.attn_q.weight", "blk.63.attn_q.weight", "token_embd.weight", "output.weight"} {
		if _, ok := names[want]; !ok {
			t.Fatalf("bank missing %s", want)
		}
	}
	if len(e.Run.Bank.Tensors) >= nBefore {
		t.Fatalf("tensor count %d, was %d", len(e.Run.Bank.Tensors), nBefore)
	}

	s, err := tensorbank.OpenSource(e.Extra.MTPStrippedPath)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	f, err := tensorbank.Parse(s)
	if err != nil {
		t.Fatal(err)
	}
	v, ok := f.Meta("qwen3.nextn_predict_layers")
	if !ok {
		t.Fatal("nextn key missing")
	}
	if n, _ := v.Scalar.(uint32); n != 0 {
		t.Fatalf("nextn=%v, want 0", v.Scalar)
	}
	if _, ok := f.FindTensor("blk.64.attn_q.weight"); ok {
		t.Fatal("stripped file still has blk.64")
	}

	// Original GGUF is unchanged (baseline eval still uses it).
	st, _ := os.Stat(src)
	if st.Size() != origSize.Size() {
		t.Fatal("source GGUF mutated")
	}
	orig, _ := tensorbank.OpenSource(src)
	defer orig.Close()
	of, _ := tensorbank.Parse(orig)
	ov, _ := of.Meta("qwen3.nextn_predict_layers")
	if n, _ := ov.Scalar.(uint32); n != 1 {
		t.Fatalf("source nextn mutated to %v", ov.Scalar)
	}
}

func TestStageDropMTPNoopQ4(t *testing.T) {
	src := filepath.Join(t.TempDir(), "src.gguf")
	if err := writeMTPGGUF(src, mtpTensors(), "qwen3", 64, 1); err != nil {
		t.Fatal(err)
	}
	e := assembleMTPEngine(t, src, 4.5, 0)
	if err := e.stageDropMTP(context.Background()); err != nil {
		t.Fatal(err)
	}
	if e.Extra.MTPStrippedPath != "" {
		t.Fatal("Q4 job stripped MTP")
	}
	if len(e.Run.Bank.Tensors) != len(mtpTensors()) {
		t.Fatal("bank mutated")
	}
}

func TestStageDropMTPImpliedBPWFromBudget(t *testing.T) {
	src := filepath.Join(t.TempDir(), "src.gguf")
	if err := writeMTPGGUF(src, mtpTensors(), "qwen3", 64, 1); err != nil {
		t.Fatal(err)
	}
	s, _ := tensorbank.OpenSource(src)
	defer s.Close()
	bank, _ := tensorbank.NewAssembler().Assemble(s, src, "")
	// Implied bpw 2.0 < 2.75.
	budget, err := func() (uint64, error) {
		var elems uint64
		for _, t := range bank.Tensors {
			elems += t.Elements
		}
		return uint64(2.0 * float64(elems) / 8.0), nil
	}()
	if err != nil {
		t.Fatal(err)
	}
	if !mtp.ShouldDrop(mtp.ImpliedBPW(bank, budget)) {
		t.Fatal("implied bpw should drop")
	}
	e := assembleMTPEngine(t, src, 0, budget)
	if err := e.stageDropMTP(context.Background()); err != nil {
		t.Fatal(err)
	}
	if e.Extra.MTPStrippedPath == "" {
		t.Fatal("budget-only Q2 job did not strip MTP")
	}

	e4 := assembleMTPEngine(t, src, 0, budget*4)
	if err := e4.stageDropMTP(context.Background()); err != nil {
		t.Fatal(err)
	}
	if e4.Extra.MTPStrippedPath != "" {
		t.Fatal("large budget stripped MTP")
	}
}

func TestStageDropMTPSidecarNoop(t *testing.T) {
	src := filepath.Join(t.TempDir(), "sidecar.gguf")
	if err := writeMTPGGUF(src, []gtensor{
		{"blk.64.attn_q.weight", core.DTypeF32, []uint64{8}},
		{"nextn.eh_proj.weight", core.DTypeF32, []uint64{8}},
	}, "qwen3", 64, 1); err != nil {
		t.Fatal(err)
	}
	e := assembleMTPEngine(t, src, 2.0, 0)
	if err := e.stageDropMTP(context.Background()); err != nil {
		t.Fatal(err)
	}
	if e.Extra.MTPStrippedPath != "" {
		t.Fatal("sidecar stripped")
	}
}

func TestStageDropMTPResumeSkipsRewrite(t *testing.T) {
	src := filepath.Join(t.TempDir(), "src.gguf")
	if err := writeMTPGGUF(src, mtpTensors(), "qwen3", 64, 1); err != nil {
		t.Fatal(err)
	}
	e := assembleMTPEngine(t, src, 2.0, 0)
	if err := e.stageDropMTP(context.Background()); err != nil {
		t.Fatal(err)
	}
	stripped := e.Extra.MTPStrippedPath
	st, err := os.Stat(stripped)
	if err != nil {
		t.Fatal(err)
	}
	mod := st.ModTime()

	// Simulate assemble restart: bank is the original again.
	s, _ := tensorbank.OpenSource(src)
	defer s.Close()
	bank, _ := tensorbank.NewAssembler().Assemble(s, src, "")
	e.Run.Bank = bank
	if err := e.stageDropMTP(context.Background()); err != nil {
		t.Fatal(err)
	}
	st2, _ := os.Stat(stripped)
	if !st2.ModTime().Equal(mod) {
		t.Fatal("resume rewrote stripped GGUF")
	}
	for _, tn := range e.Run.Bank.Tensors {
		if tn.Name == "blk.64.attn_q.weight" {
			t.Fatal("resume left MTP tensors in bank")
		}
	}
}

func TestPayloadSourcePrefersMTP(t *testing.T) {
	dir := t.TempDir()
	mtpPath := filepath.Join(dir, "source-no-mtp.gguf")
	recon := filepath.Join(dir, "reconstructed.gguf")
	if err := os.WriteFile(mtpPath, []byte("m"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(recon, []byte("r"), 0o644); err != nil {
		t.Fatal(err)
	}
	e := &Engine{
		Run: &state.Run{Config: state.RunConfig{SourcePath: filepath.Join(dir, "src.gguf")}},
		Extra: ExtraConfig{
			MTPStrippedPath:         mtpPath,
			ReconstructedSourcePath: recon,
			FoldedSourcePath:        filepath.Join(dir, "folded.gguf"),
		},
	}
	if got := e.payloadSource(); got != mtpPath {
		t.Fatalf("payloadSource=%s, want MTP", got)
	}
	os.Remove(mtpPath)
	if got := e.payloadSource(); got != recon {
		t.Fatalf("missing MTP file should fall through, got %s", got)
	}
}
