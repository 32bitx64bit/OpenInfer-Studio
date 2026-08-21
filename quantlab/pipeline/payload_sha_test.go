package pipeline

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"quantlab/core"
	"quantlab/state"
	"quantlab/tensorbank"
)

func TestBuildUsesRewrittenPayloadSHA(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src.gguf")
	recon := filepath.Join(dir, "reconstructed.gguf")
	if err := writeGGUF(src, testTensors()); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)-1] ^= 0xff
	if err := os.WriteFile(recon, data, 0o644); err != nil {
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

	as := make([]core.QuantAssignment, 0, len(bank.Tensors))
	for _, tn := range bank.Tensors {
		as = append(as, core.QuantAssignment{TensorName: tn.Name, Target: tn.DType, BitsPerWeight: 16})
	}
	m, err := core.ManifestFor(&core.Profile{ID: "p", Assignments: as}, bank)
	if err != nil {
		t.Fatal(err)
	}
	if m.SourceSHA != bank.SHA256 {
		t.Fatalf("manifest SHA = %s, want original bank %s", m.SourceSHA, bank.SHA256)
	}

	prim, err := tensorbank.OpenSource(recon)
	if err != nil {
		t.Fatal(err)
	}
	defer prim.Close()

	err = tensorbank.NewAssembler().Build(context.Background(), []*tensorbank.Source{prim}, m, filepath.Join(dir, "bad.gguf"), nil)
	if err == nil || !strings.Contains(err.Error(), "SHA") {
		t.Fatalf("want SHA mismatch without bind, got %v", err)
	}

	e := &Engine{
		Store: state.Store{Dir: dir},
		Run: &state.Run{
			RunID:  "payload-sha",
			Config: state.RunConfig{SourcePath: src},
			Bank:   bank,
		},
		Extra: ExtraConfig{ReconstructedSourcePath: recon},
	}
	if err := e.bindManifestSource(m); err != nil {
		t.Fatal(err)
	}
	if m.SourceSHA == bank.SHA256 {
		t.Fatal("bind left original source SHA on rewritten payload")
	}
	if err := tensorbank.NewAssembler().Build(context.Background(), []*tensorbank.Source{prim}, m, filepath.Join(dir, "ok.gguf"), nil); err != nil {
		t.Fatal(err)
	}
}
