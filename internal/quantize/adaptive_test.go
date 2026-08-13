package quantize

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/models"
)

func writeTinyGGUF(t *testing.T, path string, tensors []struct {
	name string
	typ  uint32
	dims []uint64
	off  uint64
}, pad int) {
	t.Helper()
	var b bytes.Buffer
	w32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	w64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	wstr := func(s string) { w64(uint64(len(s))); b.WriteString(s) }
	binary.Write(&b, binary.LittleEndian, uint32(0x46554747))
	w32(3)
	w64(uint64(len(tensors)))
	w64(1)
	wstr("general.architecture")
	w32(8)
	wstr("llama")
	for _, ti := range tensors {
		wstr(ti.name)
		w32(uint32(len(ti.dims)))
		for _, d := range ti.dims {
			w64(d)
		}
		w32(ti.typ)
		w64(ti.off)
	}
	for b.Len() < pad {
		b.WriteByte(0)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestPlanAdaptiveAssignments(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.gguf")
	writeTinyGGUF(t, path, []struct {
		name string
		typ  uint32
		dims []uint64
		off  uint64
	}{
		{"token_embd.weight", 1, []uint64{256}, 0},
		{"blk.0.ffn_gate.weight", 1, []uint64{256}, 512},
		{"output.weight", 1, []uint64{256}, 1024},
		{"blk.0.attn_norm.weight", 0, []uint64{8}, 1536},
	}, 4096)
	plan, err := PlanAdaptive(path, "balanced", 0, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Label != "OpenInfer Adaptive Balanced" {
		t.Errorf("label=%q", plan.Label)
	}
	if plan.Assignments["token_embd.weight"] == "" {
		t.Fatal("missing token_embd assignment")
	}
	if plan.Assignments["token_embd.weight"] == "q2_k" {
		t.Fatal("embeddings should stay higher precision")
	}
	tf := filepath.Join(t.TempDir(), "t.txt")
	if err := writeTensorTypeFile(tf, plan.Assignments); err != nil {
		t.Fatal(err)
	}
}

func TestListCompanionsProjector(t *testing.T) {
	src := models.Model{ID: "a", Alias: "main", PrimaryPath: "/m/model.gguf", ProjectorPath: "/m/mmproj.gguf"}
	got := ListCompanions(src, nil, "Q4_K_M")
	if len(got) != 1 || got[0].Kind != "projector" || got[0].DefaultFType != "Q8_0" {
		t.Fatalf("%+v", got)
	}
}

func TestIsSplitPath(t *testing.T) {
	if !isSplitPath("/x/model-00001-of-00002.gguf") {
		t.Fatal("split")
	}
	if isSplitPath("/x/model.gguf") {
		t.Fatal("not split")
	}
}
