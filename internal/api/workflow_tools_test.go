package api

import (
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/workflow"
)

func TestWorkflowMaskEndpointCreatesUsableGrayscaleArtifact(t *testing.T) {
	h, _, root, dir := newSaveTest(t)
	source := filepath.Join(dir, "source #%.png")
	f, err := os.Create(source)
	if err != nil {
		t.Fatal(err)
	}
	err = png.Encode(f, image.NewGray(image.Rect(0, 0, 32, 16)))
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(map[string]any{"image_path": source, "strokes": []workflow.MaskStroke{{Radius: .1, Points: [][2]float64{{.5, .5}}}}})
	w := httptest.NewRecorder()
	h.paintWorkflowMask(w, httptest.NewRequest(http.MethodPost, "/api/v1/workflow/masks", strings.NewReader(string(body))))
	var result struct {
		Path string `json:"path"`
		URL  string `json:"file_url"`
	}
	if w.Code != http.StatusCreated || json.Unmarshal(w.Body.Bytes(), &result) != nil {
		t.Fatalf("mask response: %d %s", w.Code, w.Body.String())
	}
	u, err := url.Parse(result.URL)
	rel, relErr := filepath.Rel(root, result.Path)
	urlPath := u.Path
	if runtime.GOOS == "windows" {
		urlPath = strings.TrimPrefix(urlPath, "/")
	}
	if err != nil || u.Scheme != "file" || filepath.FromSlash(urlPath) != result.Path || relErr != nil || !filepath.IsLocal(rel) {
		t.Fatalf("invalid artifact locations: %+v", result)
	}
	f, err = os.Open(result.Path)
	if err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil || img.Bounds().Dx() != 32 || img.Bounds().Dy() != 16 {
		t.Fatalf("unusable mask artifact: %v", err)
	}
	if color.GrayModel.Convert(img.At(16, 8)).(color.Gray).Y != 255 || color.GrayModel.Convert(img.At(0, 0)).(color.Gray).Y != 0 {
		t.Fatal("artifact must paint white and preserve black")
	}
	bad, _ := json.Marshal(map[string]any{"image_path": source, "strokes": []workflow.MaskStroke{{Radius: .51}}})
	w = httptest.NewRecorder()
	h.paintWorkflowMask(w, httptest.NewRequest(http.MethodPost, "/api/v1/workflow/masks", strings.NewReader(string(bad))))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("invalid brush accepted: %d", w.Code)
	}
}

func TestWorkflowFileEndpointsRoundTripNativeDocument(t *testing.T) {
	svc := newWorkflowService(t, fakeMedia{dir: t.TempDir()})
	h := &handlers{d: &Deps{Workflow: svc}}
	record, err := svc.Store().Create("Shared workflow", json.RawMessage(txt2imgJSON))
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/v1/workflows/{id}/export", h.exportWorkflow)
	mux.HandleFunc("POST /api/v1/workflows/import", h.importWorkflow)
	path := filepath.Join(t.TempDir(), "workflow.json")
	body, _ := json.Marshal(map[string]any{"path": path})
	status, out := call(t, mux, http.MethodPost, "/api/v1/workflows/"+record.ID+"/export", string(body))
	if status != http.StatusOK || out["ok"] != true {
		t.Fatalf("export: %d %v", status, out)
	}
	status, out = call(t, mux, http.MethodPost, "/api/v1/workflows/import", string(body))
	if status != http.StatusCreated || out["name"] != "Shared workflow" || out["id"] == record.ID {
		t.Fatalf("import: %d %v", status, out)
	}
	graph := out["graph"].(map[string]any)
	if len(graph["nodes"].([]any)) != 5 {
		t.Fatal("import lost workflow nodes")
	}
}
