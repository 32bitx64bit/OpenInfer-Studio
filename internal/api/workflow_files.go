package api

import (
	"net/http"
)

func (h *handlers) importWorkflow(w http.ResponseWriter, r *http.Request) {
	svc := h.workflowSvc(w)
	if svc == nil {
		return
	}
	var body struct {
		Path string `json:"path"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	rec, err := svc.Store().Import(body.Path)
	if err != nil {
		writeWorkflowErr(w, "importing workflow", err)
		return
	}
	writeJSON(w, http.StatusCreated, rec)
}

func (h *handlers) exportWorkflow(w http.ResponseWriter, r *http.Request) {
	svc := h.workflowSvc(w)
	if svc == nil {
		return
	}
	var body struct {
		Path string `json:"path"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	if err := svc.Store().Export(r.PathValue("id"), body.Path); err != nil {
		writeWorkflowErr(w, "exporting workflow", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "path": body.Path})
}
