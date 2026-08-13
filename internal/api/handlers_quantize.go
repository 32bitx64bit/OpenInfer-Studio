package api

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/instances"
	"github.com/openinfer/openinfer-studio/internal/quantize"
)

func (h *handlers) requireQuant(w http.ResponseWriter) *quantize.Manager {
	if h.d.Quant == nil {
		writeErr(w, http.StatusServiceUnavailable, "quantization is not available", nil)
		return nil
	}
	return h.d.Quant
}

func (h *handlers) runtimeTools(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if h.d.RT == nil {
		writeErr(w, 500, "runtimes unavailable", nil)
		return
	}
	if _, err := h.d.RT.Get(id); err != nil {
		writeErr(w, 404, "runtime not found", err)
		return
	}
	tools, err := h.d.RT.Tools(id)
	if err != nil {
		writeErr(w, 500, "probing runtime tools", err)
		return
	}
	var types []quantize.FType
	if q := h.d.Quant; q != nil {
		types, _, _ = q.Types(id)
	}
	writeJSON(w, 200, map[string]any{"tools": tools, "types": types, "runtime_id": id})
}

func (h *handlers) quantizeTypes(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	runtimeID := r.URL.Query().Get("runtime_id")
	types, tools, err := q.Types(runtimeID)
	if err != nil {
		writeErr(w, 400, "listing quantization types", err)
		return
	}
	writeJSON(w, 200, map[string]any{"types": types, "tools": tools, "runtime_id": runtimeID})
}

func (h *handlers) quantizePreview(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	var req quantize.Request
	if !decodeJSON(w, r, &req) {
		return
	}
	out, err := q.Preview(req)
	if err != nil {
		writeErr(w, 400, "preview failed", err)
		return
	}
	writeJSON(w, 200, out)
}

func (h *handlers) startQuantizeJob(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	var req quantize.Request
	if !decodeJSON(w, r, &req) {
		return
	}
	job, err := q.Start(req)
	if err != nil {
		writeErr(w, 400, "could not start quantization job", err)
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"job": job, "id": job.ID})
}

func (h *handlers) listQuantizeJobs(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	jobs, err := q.List()
	if err != nil {
		writeErr(w, 500, "listing quantization jobs", err)
		return
	}
	if jobs == nil {
		jobs = []quantize.Job{}
	}
	writeJSON(w, 200, map[string]any{"jobs": jobs})
}

func (h *handlers) getQuantizeJob(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	job, err := q.Get(r.PathValue("id"))
	if err != nil {
		writeErr(w, 404, "job not found", err)
		return
	}
	tail, _ := q.LogTail(job.ID, 8192)
	writeJSON(w, 200, map[string]any{"job": job, "log": tail})
}

func (h *handlers) cancelQuantizeJob(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	if err := q.Cancel(r.PathValue("id")); err != nil {
		writeErr(w, 400, "could not cancel job", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) deleteQuantizeJob(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	if err := q.Delete(r.PathValue("id")); err != nil {
		writeErr(w, 400, "could not remove job", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) clearQuantizeHistory(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	n, err := q.ClearHistory()
	if err != nil {
		writeErr(w, 500, "could not clear quantization history", err)
		return
	}
	writeJSON(w, 200, map[string]any{"removed": n})
}

func (h *handlers) listIMatrices(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	rows, err := q.ListIMatrices(r.URL.Query().Get("model_id"))
	if err != nil {
		writeErr(w, 500, "listing imatrices", err)
		return
	}
	writeJSON(w, 200, map[string]any{"imatrices": rows})
}

func (h *handlers) importIMatrix(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	var body struct {
		Path          string `json:"path"`
		SourceModelID string `json:"source_model_id"`
		DatasetLabel  string `json:"dataset_label"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	im, err := q.ImportIMatrix(body.SourceModelID, body.Path, body.DatasetLabel)
	if err != nil {
		writeErr(w, 400, "importing imatrix", err)
		return
	}
	writeJSON(w, 200, map[string]any{"imatrix": im})
}

func (h *handlers) deleteIMatrix(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	delFile := true
	if v := strings.TrimSpace(r.URL.Query().Get("delete_file")); v != "" {
		delFile, _ = strconv.ParseBool(v)
	}
	if err := q.DeleteIMatrix(r.PathValue("id"), delFile); err != nil {
		writeErr(w, 400, "deleting imatrix", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

// LoadedModelIDs returns model IDs currently occupying VRAM/RAM.
func LoadedModelIDs(im *instances.Manager) []string {
	if im == nil {
		return []string{}
	}
	var ids []string
	for _, inst := range im.List() {
		switch inst.State {
		case instances.StateReady, instances.StateBusy, instances.StateLoading,
			instances.StateStarting, instances.StateSleeping:
			ids = append(ids, inst.ModelID)
		}
	}
	return ids
}
