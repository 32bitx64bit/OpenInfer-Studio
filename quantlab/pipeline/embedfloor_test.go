package pipeline

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"quantlab/anchor"
	"quantlab/core"
)

func TestEmbedRowFloorRelaxesUntiedEmbedding(t *testing.T) {
	f := newFixture(t, 0)
	tensors := []gtensor{
		{"token_embd.weight", core.DTypeF16, []uint64{256, 2048}},
		{"blk.0.attn_q.weight", core.DTypeF16, []uint64{256, 256}},
		{"blk.0.attn_norm.weight", core.DTypeF32, []uint64{256}},
		{"output.weight", core.DTypeF16, []uint64{256, 2048}},
	}
	if err := writeGGUF(f.src, tensors); err != nil {
		t.Fatal(err)
	}
	finiteF16Payloads(t, f.src)
	r := f.planEffort("emb", "profiled", func(o *PlanOptions) {
		o.BudgetBytes = 0
		o.TargetBPW = 4.5
	})
	e := f.engine(r)
	e.StageLimit = 2 // assemble, anchor
	if err := e.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(r.Config.WorkDir, "anchors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var set anchor.Set
	if err := json.Unmarshal(data, &set); err != nil {
		t.Fatal(err)
	}
	emb, _ := set.Floor("token_embd.weight")
	out, _ := set.Floor("output.weight")
	if anchor.Rank(emb) <= anchor.Rank(core.DTypeQ6_K) {
		t.Errorf("embedding floor = %s, want relaxed below the Q6_K policy for well-behaved rows", emb)
	}
	if out != core.DTypeQ6_K {
		t.Errorf("output floor = %s, want the untouched Q6_K policy", out)
	}
	if _, err := os.Stat(filepath.Join(r.Config.WorkDir, "embed-floor.json")); err != nil {
		t.Errorf("embed-floor record not written: %v", err)
	}

	// Opt-out keeps the policy floor.
	e2 := f.engine(r)
	e2.Extra.NoEmbedRowFloor = true
	set2, err := e2.deriveAnchors(e2.Run.Bank)
	if err != nil {
		t.Fatal(err)
	}
	if fl, _ := set2.Floor("token_embd.weight"); fl != core.DTypeQ6_K {
		t.Errorf("NoEmbedRowFloor: embedding floor = %s, want Q6_K", fl)
	}
}
