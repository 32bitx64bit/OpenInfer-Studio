package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

func TestMediaGenerationCapabilitiesModelSelectionAndUnknownState(t *testing.T) {
	h, _, _, _ := newSaveTest(t)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/v1/media/servers/{id}/capabilities", h.mediaGenerationCapabilities)
	mux.HandleFunc("GET /api/v1/media/capabilities", h.mediaGenerationCapabilities)
	for _, url := range []string{"/api/v1/media/servers/model-1/capabilities", "/api/v1/media/capabilities?model_id=model-1"} {
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest(http.MethodGet, url, nil))
		var caps mediagen.APICapabilities
		if w.Code != http.StatusOK || json.Unmarshal(w.Body.Bytes(), &caps) != nil || caps.ModelID != "model-1" || caps.Known || caps.Reason == "" {
			t.Fatalf("response=%d %s", w.Code, w.Body.String())
		}
		if caps.SupportedModes == nil || caps.Samplers == nil || caps.Schedulers == nil || caps.FeaturesByMode == nil {
			t.Fatal("unknown capabilities must expose empty collections")
		}
	}
	for _, url := range []string{"/api/v1/media/capabilities", "/api/v1/media/servers/model-1/capabilities?model_id=other"} {
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest(http.MethodGet, url, nil))
		if w.Code != http.StatusBadRequest {
			t.Fatalf("missing/conflicting id accepted: %d", w.Code)
		}
	}
}

func TestMediaGenerationCapabilitiesUnavailableManager(t *testing.T) {
	h := &handlers{d: &Deps{}}
	w := httptest.NewRecorder()
	h.mediaGenerationCapabilities(w, httptest.NewRequest(http.MethodGet, "/api/v1/media/capabilities?model_id=m", nil))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d", w.Code)
	}
}
