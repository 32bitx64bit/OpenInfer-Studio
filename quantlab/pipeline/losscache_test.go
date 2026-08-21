package pipeline

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"quantlab/core"
	"quantlab/kld"
	"quantlab/profile"
	"quantlab/state"
)

func TestIngestSearchLossCacheNoopEmpty(t *testing.T) {
	e := &Engine{Run: &state.Run{RunID: "r", Bank: &core.TensorBank{ModelID: "m", SHA256: "s"}}}
	if err := e.ingestSearchLossCache(); err != nil {
		t.Fatal(err)
	}
	if e.lossCache != nil {
		t.Fatal("empty history populated cache")
	}
}

func TestIngestAndLoadLossCacheSidecar(t *testing.T) {
	work := t.TempDir()
	storeDir := t.TempDir()
	bank := &core.TensorBank{
		SourcePath: "/m.gguf", ModelID: "m", SHA256: "abc",
		Tensors: []core.TensorDesc{weightLike("blk.0.attn_v.weight")},
	}
	e := &Engine{
		Store: state.Store{Dir: storeDir},
		Run: &state.Run{
			RunID:  "run1",
			Bank:   bank,
			Config: state.RunConfig{WorkDir: work},
			SearchHistory: []kld.Step{
				{Kind: "baseline", KLD: 1.0},
				{Kind: "solo", Accepted: true, KLD: 0.5, GroupIDs: []string{"g1"}},
			},
			MoveGroups: []core.MoveGroup{{
				ID: "g1",
				Moves: []core.Move{
					{TensorName: "blk.0.attn_v.weight", From: core.DTypeQ2_K, To: core.DTypeQ4_K_T},
				},
			}},
		},
		now: func() time.Time { return time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC) },
	}
	if err := e.ingestSearchLossCache(); err != nil {
		t.Fatal(err)
	}
	workPath := filepath.Join(work, "loss-cache.json")
	sidePath := filepath.Join(storeDir, "run1.loss-cache.json")
	if _, err := os.Stat(workPath); err != nil {
		t.Fatalf("workDir cache missing: %v", err)
	}
	if _, err := os.Stat(sidePath); err != nil {
		t.Fatalf("store sidecar missing: %v", err)
	}
	// Simulate scratch cleanup: only sidecar remains.
	os.Remove(workPath)
	got := e.loadLossCache()
	if got == nil {
		t.Fatal("sidecar not loaded after workDir cleanup")
	}
	cl, ok := got.Get("blk.0.attn_v.weight", core.DTypeQ4_K_T)
	if !ok || cl.Prov == nil || cl.Prov.Tool != "kld-search" {
		t.Fatalf("sidecar entry: %+v ok=%v", cl, ok)
	}
	if cl.Confidence < 0.6 || cl.Confidence > 0.8 {
		t.Fatalf("confidence %v", cl.Confidence)
	}
}

func weightLike(name string) core.TensorDesc {
	return core.TensorDesc{
		Name: name, DType: core.DTypeF16, Shape: []uint64{256, 4},
		Length: 2048, Elements: 1024,
	}
}

func TestLoadLossCacheIdentityMismatch(t *testing.T) {
	work := t.TempDir()
	c := profile.NewCache("other", "sha")
	prov := core.Provenance{Tool: "kld-search", RunID: "r", MeasuredAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
	_ = c.Put(profile.CandidateLoss{
		TensorName: "blk.0.attn_v.weight", Target: core.DTypeQ4_K_T, Loss: 0.1,
		Evidence: profile.EvidenceMeasured, Confidence: 0.7, Prov: &prov,
	})
	f, err := os.Create(filepath.Join(work, "loss-cache.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Save(f); err != nil {
		t.Fatal(err)
	}
	f.Close()
	e := &Engine{
		Run: &state.Run{
			RunID:  "run1",
			Bank:   &core.TensorBank{ModelID: "m", SHA256: "abc", SourcePath: "/m.gguf"},
			Config: state.RunConfig{WorkDir: work},
		},
	}
	if got := e.loadLossCache(); got != nil {
		t.Fatal("identity mismatch must not leak into solve")
	}
}
