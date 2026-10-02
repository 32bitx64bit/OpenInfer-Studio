package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/huggingface"
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
