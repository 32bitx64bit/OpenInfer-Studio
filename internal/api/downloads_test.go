package api

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/internal/downloads"
	"github.com/openinfer/openinfer-studio/internal/huggingface"
	"github.com/openinfer/openinfer-studio/internal/sdmodel"
	"github.com/openinfer/openinfer-studio/migrations"
)

func files(paths ...string) []enqueueFile {
	out := make([]enqueueFile, len(paths))
	for i, p := range paths {
		out[i] = enqueueFile{Path: p, Size: 1}
	}
	return out
}

func TestDownloadDestsFlatByDefault(t *testing.T) {
	got, err := downloadDests(files("Q4_K_M/model-00001-of-00002.gguf", "Q4_K_M/model-00002-of-00002.gguf", "mmproj-F16.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	want := "model-00001-of-00002.gguf,model-00002-of-00002.gguf,mmproj-F16.gguf"
	if strings.Join(got, ",") != want {
		t.Fatalf("dests = %v, want %s", got, want)
	}
}

func TestDownloadDestsCollidingNamesKeepFolders(t *testing.T) {
	got, err := downloadDests(files(
		"unet/diffusion_pytorch_model.safetensors",
		"vae/diffusion_pytorch_model.safetensors",
		"flux1-dev-fp8.safetensors",
	))
	if err != nil {
		t.Fatal(err)
	}
	want := "unet/diffusion_pytorch_model.safetensors,vae/diffusion_pytorch_model.safetensors,flux1-dev-fp8.safetensors"
	if strings.Join(got, ",") != want {
		t.Fatalf("dests = %v, want %s", got, want)
	}
}

func TestDownloadDestsCollisionIsCaseInsensitive(t *testing.T) {
	got, err := downloadDests(files("a/Model.safetensors", "b/model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if got[0] == got[1] || !strings.Contains(got[0], "/") {
		t.Fatalf("case-different names must still be kept apart: %v", got)
	}
}

func TestDownloadDestsExplicitDestWins(t *testing.T) {
	in := files("vae/diffusion_pytorch_model.safetensors", "ae.safetensors")
	in[0].Dest = "vae/diffusion_pytorch_model.safetensors"
	got, err := downloadDests(in)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != "vae/diffusion_pytorch_model.safetensors" || got[1] != "ae.safetensors" {
		t.Fatalf("dests = %v", got)
	}
}

func TestDownloadDestsRejectsUnsafeAndDuplicate(t *testing.T) {
	bad := []string{"../evil.bin", "/etc/passwd", "a/../../b", "C:/x.bin", ".", "a\\..\\..\\b"}
	for _, d := range bad {
		in := files("m.bin")
		in[0].Dest = d
		if _, err := downloadDests(in); err == nil {
			t.Errorf("dest %q accepted", d)
		}
	}
	// The same file twice cannot be saved twice.
	if _, err := downloadDests(files("a/x.bin", "a/x.bin")); err == nil {
		t.Error("duplicate path accepted")
	}
	// Explicit dests that collide are rejected, not silently merged.
	in := files("a/x.bin", "b/x.bin")
	in[0].Dest, in[1].Dest = "x.bin", "x.bin"
	if _, err := downloadDests(in); err == nil {
		t.Error("colliding explicit dests accepted")
	}
}

// The repository endpoint carries the download plan the picker renders, and
// the legacy group fields stay for API callers.
func TestHFRepoReturnsPlan(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/Model-GGUF", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "org/Model-GGUF", "tags": []string{"gguf"}, "pipeline_tag": "text-generation",
			"siblings": []any{},
		})
	})
	mux.HandleFunc("/api/models/org/Model-GGUF/tree/main", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"path": "Model-Q4_K_M.gguf", "size": 400, "type": "file"},
			{"path": "Model-Q8_0.gguf", "size": 800, "type": "file"},
			{"path": "mmproj-F16.gguf", "size": 50, "type": "file"},
		})
	})
	mux.HandleFunc("/org/Model-GGUF/raw/main/README.md", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(404)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	hf := huggingface.NewClient()
	hf.SetBaseURL(srv.URL)

	h := &handlers{d: &Deps{HF: hf, Layout: &config.Layout{Models: t.TempDir()}}}
	req := httptest.NewRequest(http.MethodGet, "/api/v1/hf/repo/org/Model-GGUF", nil)
	req.SetPathValue("author", "org")
	req.SetPathValue("name", "Model-GGUF")
	rec := httptest.NewRecorder()
	h.hfRepo(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var body struct {
		Plan   huggingface.Plan        `json:"plan"`
		Groups []huggingface.FileGroup `json:"groups"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Plan.Kind != huggingface.PlanChat || len(body.Plan.Components) != 2 {
		t.Fatalf("plan = %+v", body.Plan)
	}
	if got := body.Plan.Components[0].Default; !strings.Contains(got, "q4_k_m") {
		t.Errorf("default model option = %q", got)
	}
	if len(body.Groups) != 2 {
		t.Errorf("legacy groups field changed: %d", len(body.Groups))
	}
}

type nullEvents struct{}

func (nullEvents) Publish(string, any) {}

// A generator download records, next to the files, what each one is: the
// library scanner cannot tell from tensor names for a new architecture.
func TestEnqueueRecordsGeneratorRolesForTheLibrary(t *testing.T) {
	dir := t.TempDir()
	db, err := database.Open(dir, migrations.FS)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	partial := filepath.Join(dir, "partial")
	_ = os.MkdirAll(partial, 0o755)
	dl := downloads.NewManager(db.DB, partial, nullEvents{}, slog.New(slog.NewTextHandler(io.Discard, nil)),
		func(string) uint64 { return 1 << 40 })

	hubSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("weights"))
	}))
	defer hubSrv.Close()
	hf := huggingface.NewClient()
	hf.SetBaseURL(hubSrv.URL)

	models := filepath.Join(dir, "models")
	h := &handlers{d: &Deps{HF: hf, DL: dl, Layout: &config.Layout{Models: models}}}

	body := `{"kind":"model","label":"x","repo":"Comfy-Org/MiniMax-H3","group":"model-minimax_h3_bf16","plan":"generator","diffusion":"video",
	 "files":[
	  {"path":"split_files/diffusion_models/minimax_h3_bf16.safetensors","size":7,"role":"model"},
	  {"path":"split_files/text_encoders/umt5_xxl_fp16.safetensors","size":7,"role":"t5xxl","dest":"text_encoders/umt5_xxl_fp16.safetensors"},
	  {"path":"split_files/vae/vae.safetensors","size":7,"role":"vae"},
	  {"path":"split_files/clip_vision/cv.safetensors","size":7,"role":"made-up-role"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/downloads", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.enqueueDownload(rec, req)
	if rec.Code != 201 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var out struct{ ID string }
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if state, err := dl.WaitComplete(context.Background(), out.ID); err != nil || state != "complete" {
		t.Fatalf("state=%s err=%v", state, err)
	}

	group := filepath.Join(models, "Comfy-Org--MiniMax-H3", "model-minimax_h3_bf16")
	m := sdmodel.ReadManifest(group)
	if m == nil {
		t.Fatal("no manifest written next to the downloaded files")
	}
	want := map[string]string{
		"minimax_h3_bf16.safetensors":             sdmodel.RoleModel,
		"text_encoders/umt5_xxl_fp16.safetensors": sdmodel.RoleT5XXL,
		"vae.safetensors":                         sdmodel.RoleVAE,
	}
	if len(m.Files) != len(want) || m.Repo != "Comfy-Org/MiniMax-H3" || m.DiffusionKind != "video" {
		t.Fatalf("manifest = %+v", m)
	}
	for rel, role := range want {
		if m.Files[rel] != role {
			t.Errorf("manifest[%s] = %q, want %q", rel, m.Files[rel], role)
		}
		if _, err := os.Stat(filepath.Join(group, filepath.FromSlash(rel))); err != nil {
			t.Errorf("file %s is not where the manifest says: %v", rel, err)
		}
	}
	if role, ok := sdmodel.DeclaredRole(filepath.Join(group, "minimax_h3_bf16.safetensors")); !ok || role != sdmodel.RoleModel {
		t.Errorf("DeclaredRole = %q,%v", role, ok)
	}

	// A chat download records nothing.
	chat := `{"kind":"model","label":"c","repo":"org/Model-GGUF","group":"g","plan":"chat","files":[{"path":"M-Q4_K_M.gguf","size":7,"role":"model"}]}`
	rec = httptest.NewRecorder()
	h.enqueueDownload(rec, httptest.NewRequest(http.MethodPost, "/api/v1/downloads", strings.NewReader(chat)))
	if rec.Code != 201 {
		t.Fatalf("chat status %d: %s", rec.Code, rec.Body.String())
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	_, _ = dl.WaitComplete(context.Background(), out.ID)
	if sdmodel.ReadManifest(filepath.Join(models, "org--Model-GGUF", "g")) != nil {
		t.Error("a chat download must not write a generator manifest")
	}
}
