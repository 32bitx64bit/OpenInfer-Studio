package api

import (
	"fmt"
	"math"
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

// quantizeRequestBody is the API-facing quantize job body. A pointer retains
// whether clients supplied `effort`, so the deprecated adaptive_mode field is
// only forwarded when effort is absent. adaptive_preset passes through for the
// backend's deprecated preset → target_bpw translation.
type quantizeRequestBody struct {
	quantize.Request
	Effort *string `json:"effort"`
}

// decodeQuantizeRequest reads and validates a quantize job body. An explicit
// `effort` and the deprecated `adaptive_mode` alias are kept on their
// respective request fields. The backend resolves their precedence.
func decodeQuantizeRequest(w http.ResponseWriter, r *http.Request) (quantize.Request, bool) {
	var body quantizeRequestBody
	if !decodeJSON(w, r, &body) {
		return quantize.Request{}, false
	}
	req := body.Request
	effortField := "adaptive_mode"
	effort := strings.ToLower(strings.TrimSpace(req.AdaptiveMode))
	if body.Effort != nil {
		effortField = "effort"
		effort = strings.ToLower(strings.TrimSpace(*body.Effort))
		req.Effort = effort
		req.AdaptiveMode = ""
	} else {
		req.AdaptiveMode = effort
	}
	if !validateQuantizeEffort(w, effortField, effort) {
		return quantize.Request{}, false
	}
	req.QuantTier = strings.ToLower(strings.TrimSpace(req.QuantTier))
	if !validateQuantTier(w, req.QuantTier) {
		return quantize.Request{}, false
	}
	if req.QuantTier == quantize.QuantTierCustom && req.TargetBytes <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid target_bytes", fmt.Errorf("quant_tier %q requires a positive target_bytes", req.QuantTier))
		return quantize.Request{}, false
	}
	return req, true
}

func validateQuantizeEffort(w http.ResponseWriter, field, effort string) bool {
	switch effort {
	case "", "fast", "profiled", "deep":
		return true
	default:
		writeErr(w, http.StatusBadRequest, "invalid "+field, fmt.Errorf(
			"%s must be \"fast\", \"profiled\", or \"deep\" (got %q); omit it for the default \"profiled\"", field, effort))
		return false
	}
}

// validateQuantTier accepts the OpenInfer Dynamic compression tier values. An
// empty value is valid and defers to the backend default (q4).
func validateQuantTier(w http.ResponseWriter, tier string) bool {
	switch tier {
	case "", "q5", "q4", "q3", "q2", "custom":
		return true
	default:
		writeErr(w, http.StatusBadRequest, "invalid quant_tier", fmt.Errorf(
			"quant_tier must be \"q5\", \"q4\", \"q3\", \"q2\", or \"custom\" (got %q); omit it for the default \"q4\"", tier))
		return false
	}
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
	req, ok := decodeQuantizeRequest(w, r)
	if !ok {
		return
	}
	out, err := q.Preview(req)
	if err != nil {
		writeErr(w, 400, "preview failed", err)
		return
	}
	writeJSON(w, 200, out)
}

func (h *handlers) quantizeFromHFPreview(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	repo, req, dynamic, ok := decodeFromHFPreviewRequest(w, r)
	if !ok {
		return
	}
	var (
		out *quantize.FromHFPreview
		err error
	)
	if dynamic {
		out, err = q.ProbeFromHFForRequest(r.Context(), repo, req)
	} else {
		out, err = q.ProbeFromHF(r.Context(), repo)
	}
	if err != nil {
		status := 502
		if ae, ok := err.(interface{ HTTPStatus() int }); ok {
			status = ae.HTTPStatus()
		}
		writeErr(w, status, "Hugging Face convert preview failed", err)
		return
	}
	writeJSON(w, 200, out)
}

// decodeFromHFPreviewRequest reads the optional Dynamic sizing query. A
// repo-only request intentionally remains the original conversion-only probe.
func decodeFromHFPreviewRequest(w http.ResponseWriter, r *http.Request) (string, quantize.Request, bool, bool) {
	query := r.URL.Query()
	repo := strings.TrimSpace(query.Get("repo"))
	if repo == "" {
		writeErr(w, http.StatusBadRequest, "repo query parameter is required", nil)
		return "", quantize.Request{}, false, false
	}

	dynamic := false
	if raw, present := query["dynamic"]; present {
		var err error
		dynamic, err = strconv.ParseBool(strings.TrimSpace(raw[0]))
		if err != nil {
			writeErr(w, http.StatusBadRequest, "invalid dynamic", fmt.Errorf("dynamic must be true or false"))
			return "", quantize.Request{}, false, false
		}
	}

	effort := strings.ToLower(strings.TrimSpace(query.Get("effort")))
	if !validateQuantizeEffort(w, "effort", effort) {
		return "", quantize.Request{}, false, false
	}
	req := quantize.Request{Kind: quantize.KindFromHF, Effort: effort}
	if dynamic && req.Effort == "" {
		// Match the backend's default effort while marking this as Dynamic.
		req.Effort = "profiled"
	}

	if raw, present := query["target_bpw"]; present {
		value, err := strconv.ParseFloat(strings.TrimSpace(raw[0]), 64)
		if err != nil || math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
			writeErr(w, http.StatusBadRequest, "invalid target_bpw", fmt.Errorf("target_bpw must be a non-negative number"))
			return "", quantize.Request{}, false, false
		}
		req.TargetBPW = value
	}
	if raw, present := query["target_bytes"]; present {
		value, err := strconv.ParseInt(strings.TrimSpace(raw[0]), 10, 64)
		if err != nil || value < 0 {
			writeErr(w, http.StatusBadRequest, "invalid target_bytes", fmt.Errorf("target_bytes must be a non-negative integer"))
			return "", quantize.Request{}, false, false
		}
		req.TargetBytes = value
	}

	if raw, present := query["generate_imatrix"]; present {
		value, err := strconv.ParseBool(strings.TrimSpace(raw[0]))
		if err != nil {
			writeErr(w, http.StatusBadRequest, "invalid generate_imatrix", fmt.Errorf("generate_imatrix must be true or false"))
			return "", quantize.Request{}, false, false
		}
		req.GenerateIMatrix = value
	} else {
		req.GenerateIMatrix = dynamic
	}

	tier := strings.ToLower(strings.TrimSpace(query.Get("quant_tier")))
	if !validateQuantTier(w, tier) {
		return "", quantize.Request{}, false, false
	}
	req.QuantTier = tier
	if tier == quantize.QuantTierCustom && req.TargetBytes <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid target_bytes", fmt.Errorf("quant_tier %q requires a positive target_bytes", tier))
		return "", quantize.Request{}, false, false
	}
	return repo, req, dynamic, true
}

func (h *handlers) startQuantizeJob(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	req, ok := decodeQuantizeRequest(w, r)
	if !ok {
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

func (h *handlers) pauseQuantizeJob(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	if err := q.Pause(r.PathValue("id")); err != nil {
		writeErr(w, 400, "could not pause job", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) resumeQuantizeJob(w http.ResponseWriter, r *http.Request) {
	q := h.requireQuant(w)
	if q == nil {
		return
	}
	if err := q.Resume(r.PathValue("id")); err != nil {
		writeErr(w, 400, "could not resume job", err)
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
