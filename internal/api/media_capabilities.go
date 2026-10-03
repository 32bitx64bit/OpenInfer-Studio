package api

import (
	"net/http"
	"strings"
)

// mediaGenerationCapabilities serves loaded-model capability discovery.
// The server path's id is a library model ID; model_id selects the query form.
// Discovery is
// read-only: it never loads a model or probes CLI flags as HTTP evidence.
func (h *handlers) mediaGenerationCapabilities(w http.ResponseWriter, r *http.Request) {
	if h.d.Media == nil {
		writeErr(w, http.StatusServiceUnavailable, "media generation unavailable", nil)
		return
	}
	modelID := strings.TrimSpace(r.PathValue("id"))
	queryID := strings.TrimSpace(r.URL.Query().Get("model_id"))
	if modelID != "" && queryID != "" && modelID != queryID {
		writeErr(w, http.StatusBadRequest, "model_id conflicts with server id", nil)
		return
	}
	if modelID == "" {
		modelID = queryID
	}
	if modelID == "" {
		writeErr(w, http.StatusBadRequest, "model_id is required", nil)
		return
	}
	caps, err := h.d.Media.GenerationCapabilities(modelID)
	if err != nil {
		writeErr(w, http.StatusBadGateway, "reading loaded-model generation capabilities failed", err)
		return
	}
	writeJSON(w, http.StatusOK, caps)
}
