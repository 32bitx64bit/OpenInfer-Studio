package models

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/sdmodel"
)

// writeSafetensors writes a safetensors file with the given tensor names
// (header only; the scanner never reads tensor data).
func writeSafetensors(t *testing.T, path string, names []string) {
	t.Helper()
	header := map[string]any{}
	for _, n := range names {
		header[n] = map[string]any{"dtype": "BF16", "shape": []int{1}, "data_offsets": []int{0, 0}}
	}
	body, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	var n [8]byte
	binary.LittleEndian.PutUint64(n[:], uint64(len(body)))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(n[:], body...), 0o644); err != nil {
		t.Fatal(err)
	}
}

// mystery is a diffusion transformer no tensor signature knows.
func mystery() []string {
	var names []string
	for _, p := range []string{"alpha", "beta", "gamma", "delta"} {
		for i := 0; i < 5; i++ {
			names = append(names, "stack."+p+"."+string(rune('0'+i))+".weight")
		}
	}
	return names
}

func registered(t *testing.T, lib *Library, path string) *Model {
	t.Helper()
	if _, err := lib.Scan(); err != nil {
		t.Fatal(err)
	}
	id := lib.IDForPath(path)
	if id == "" {
		return nil
	}
	m, err := lib.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// The reported case: a downloaded video model whose architecture matches no
// tensor signature and whose names say nothing never reached the library.
func TestScanUnknownArchitectureNeedsTheDownloadsRecord(t *testing.T) {
	lib := testLibrary(t)
	dir := filepath.Join(lib.managed, "acme--mystery-net", "model-mystery_bf16")
	file := filepath.Join(dir, "mystery_bf16.safetensors")
	writeSafetensors(t, file, mystery())
	if registered(t, lib, file) != nil {
		t.Fatal("with no evidence at all the file must stay out (it could be anything)")
	}
	if err := sdmodel.WriteManifest(dir, sdmodel.Manifest{Version: 1, Repo: "acme/mystery-net", DiffusionKind: "video",
		Files: map[string]string{"mystery_bf16.safetensors": sdmodel.RoleModel}}); err != nil {
		t.Fatal(err)
	}
	m := registered(t, lib, file)
	if m == nil {
		t.Fatal("the download recorded this file as the diffusion model: it must be in the library")
	}
	if !IsDiffusionModel(*m) || DiffusionKind(*m) != "video" {
		t.Fatalf("row = modality %q kind %q", m.Modality, DiffusionKind(*m))
	}
	// A rescan keeps one row.
	if _, err := lib.Scan(); err != nil {
		t.Fatal(err)
	}
	all, _ := lib.List()
	if len(all) != 1 {
		t.Fatalf("rows after rescan = %d", len(all))
	}
}

func TestScanRecordedComponentsAreNotModels(t *testing.T) {
	lib := testLibrary(t)
	dir := filepath.Join(lib.managed, "acme--mystery-net", "g")
	model := filepath.Join(dir, "a_model.safetensors")
	enc := filepath.Join(dir, "weights_b.safetensors")
	vae := filepath.Join(dir, "weights_c.safetensors")
	for _, f := range []string{model, enc, vae} {
		writeSafetensors(t, f, mystery())
	}
	_ = sdmodel.WriteManifest(dir, sdmodel.Manifest{Version: 1, Files: map[string]string{
		"a_model.safetensors": sdmodel.RoleModel, "weights_b.safetensors": sdmodel.RoleT5XXL, "weights_c.safetensors": sdmodel.RoleVAE,
	}})
	if registered(t, lib, model) == nil {
		t.Fatal("model missing")
	}
	if lib.IDForPath(enc) != "" || lib.IDForPath(vae) != "" {
		t.Error("files recorded as a text encoder / VAE must not become library models")
	}
}

func TestScanWiderEvidenceForFilesDownloadedBeforeTheRecord(t *testing.T) {
	cases := []struct {
		name  string
		repo  string
		file  string
		names []string
		want  bool
		kind  string
	}{
		{"Wan-style tensors", "acme--unnamed", "w.safetensors",
			[]string{"patch_embedding.weight", "text_embedding.0.weight", "time_embedding.0.weight", "blocks.0.cross_attn.k.weight", "blocks.0.self_attn.q.weight"}, true, "video"},
		{"comfy-org repo, unknown tensors", "Comfy-Org--MiniMax-H3", "model_bf16.safetensors", mystery(), true, "video"},
		{"name hint, unknown tensors", "acme--stuff", "minimax_h3_fp8.safetensors", mystery(), true, "video"},
		{"language-model tensors in a comfy-org repo", "Comfy-Org--MiniMax-H3", "model.safetensors",
			[]string{"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "lm_head.weight"}, false, ""},
		{"text encoder by name in a comfy-org repo", "Comfy-Org--MiniMax-H3", "umt5_xxl_fp16.safetensors", mystery(), false, ""},
		{"LLM shard whose name mentions minimax", "MiniMaxAI--MiniMax-M2", "model-00001-of-00130.safetensors",
			[]string{"model.embed_tokens.weight", "model.layers.0.mlp.gate.weight"}, false, ""},
	}
	for _, c := range cases {
		lib := testLibrary(t)
		file := filepath.Join(lib.managed, c.repo, "g", c.file)
		writeSafetensors(t, file, c.names)
		m := registered(t, lib, file)
		if (m != nil) != c.want {
			t.Errorf("%s: in library = %v, want %v", c.name, m != nil, c.want)
			continue
		}
		if m != nil && c.kind != "" && DiffusionKind(*m) != c.kind {
			t.Errorf("%s: kind = %q, want %q", c.name, DiffusionKind(*m), c.kind)
		}
	}
}

func TestScanRecordedGGUFModelOfUnknownArchitectureIsADiffusionModel(t *testing.T) {
	lib := testLibrary(t)
	dir := filepath.Join(lib.managed, "acme--mystery-gguf", "g")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	file := writeTestGGUF(t, dir, "mystery-Q8_0.gguf", "Mystery")
	enc := writeTestGGUF(t, dir, "textenc-Q8_0.gguf", "Encoder")
	_ = sdmodel.WriteManifest(dir, sdmodel.Manifest{Version: 1, DiffusionKind: "video", Files: map[string]string{
		"mystery-Q8_0.gguf": sdmodel.RoleModel, "textenc-Q8_0.gguf": sdmodel.RoleLLM,
	}})
	m := registered(t, lib, file)
	if m == nil {
		t.Fatal("recorded GGUF model missing")
	}
	if !IsDiffusionModel(*m) || DiffusionKind(*m) != "video" {
		t.Errorf("recorded GGUF model: modality %q kind %q", m.Modality, DiffusionKind(*m))
	}
	if lib.IDForPath(enc) != "" {
		t.Error("a GGUF recorded as a text encoder must not be listed as a chat model")
	}
}

func TestSchemaVersionForcesRescanOfExistingDownloads(t *testing.T) {
	lib := testLibrary(t)
	file := filepath.Join(lib.managed, "Comfy-Org--MiniMax-H3", "g", "model_bf16.safetensors")
	writeSafetensors(t, file, mystery())
	// A launch with the previous schema stored rescans and finds the file.
	scanned, _, _, err := lib.EnsureFresh("13")
	if err != nil || !scanned {
		t.Fatalf("scanned=%v err=%v", scanned, err)
	}
	if lib.IDForPath(file) == "" {
		t.Fatal("an existing download should appear after the upgrade rescan")
	}
}
