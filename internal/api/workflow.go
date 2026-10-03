package api

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/openinfer/openinfer-studio/internal/workflow"
)

// workflowSvc returns the workflow service, or writes 503 when the backend
// was started without one.
func (h *handlers) workflowSvc(w http.ResponseWriter) *workflow.Service {
	if h.d.Workflow == nil {
		writeErr(w, http.StatusServiceUnavailable, "workflows unavailable", nil)
		return nil
	}
	return h.d.Workflow
}

func writeWorkflowErr(w http.ResponseWriter, msg string, err error) {
	switch {
	case errors.Is(err, workflow.ErrNotFound):
		writeErr(w, http.StatusNotFound, "workflow not found", err)
	case errors.Is(err, workflow.ErrInvalid):
		writeErr(w, http.StatusBadRequest, msg, err)
	default:
		writeErr(w, http.StatusInternalServerError, msg, err)
	}
}

// workflowNodeTypes serves the node registry. QML draws the palette, sockets,
// parameter widgets and inspector from this document, so a new node type
// needs no QML. Availability reflects the selected sd.cpp runtime.
func (h *handlers) workflowNodeTypes(w http.ResponseWriter, r *http.Request) {
	svc := h.workflowSvc(w)
	if svc == nil {
		return
	}
	info, caps := svc.Capabilities(r.URL.Query().Get("runtime_id"))
	apiCaps := svc.ModelCapabilities(r.URL.Query().Get("model_id"))
	caps.API = apiCaps
	writeJSON(w, http.StatusOK, map[string]any{
		"node_types":       svc.Registry().View(caps),
		"port_types":       workflow.AllPortTypes,
		"runtime":          info,
		"api_capabilities": apiCaps,
	})
}

func (h *handlers) listWorkflows(w http.ResponseWriter, r *http.Request) {
	svc := h.workflowSvc(w)
	if svc == nil {
		return
	}
	list, err := svc.Store().List()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "listing workflows", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"workflows": list})
}

type workflowBody struct {
	Name  *string         `json:"name"`
	Graph json.RawMessage `json:"graph"`
}

func (h *handlers) createWorkflow(w http.ResponseWriter, r *http.Request) {
	svc := h.workflowSvc(w)
	if svc == nil {
		return
	}
	var body workflowBody
	if !decodeJSON(w, r, &body) {
		return
	}
	name := "Untitled workflow"
	if body.Name != nil {
		name = *body.Name
	}
	rec, err := svc.Store().Create(name, body.Graph)
	if err != nil {
		writeWorkflowErr(w, "saving workflow", err)
		return
	}
	writeJSON(w, http.StatusCreated, rec)
}

func (h *handlers) getWorkflow(w http.ResponseWriter, r *http.Request) {
	svc := h.workflowSvc(w)
	if svc == nil {
		return
	}
	rec, err := svc.Store().Get(r.PathValue("id"))
	if err != nil {
		writeWorkflowErr(w, "loading workflow", err)
		return
	}
	writeJSON(w, http.StatusOK, rec)
}

func (h *handlers) putWorkflow(w http.ResponseWriter, r *http.Request) {
	svc := h.workflowSvc(w)
	if svc == nil {
		return
	}
	var body workflowBody
	if !decodeJSON(w, r, &body) {
		return
	}
	rec, err := svc.Store().Update(r.PathValue("id"), body.Name, body.Graph)
	if err != nil {
		writeWorkflowErr(w, "saving workflow", err)
		return
	}
	writeJSON(w, http.StatusOK, rec)
}

func (h *handlers) deleteWorkflow(w http.ResponseWriter, r *http.Request) {
	svc := h.workflowSvc(w)
	if svc == nil {
		return
	}
	if err := svc.Store().Delete(r.PathValue("id")); err != nil {
		writeWorkflowErr(w, "deleting workflow", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// validateWorkflow checks a graph and, when it is sound, returns the plan a
// run would execute: stages, the servers they need, and whether each is
// reused, started or restarted. Nothing is started or written.
func (h *handlers) validateWorkflow(w http.ResponseWriter, r *http.Request) {
	svc := h.workflowSvc(w)
	if svc == nil {
		return
	}
	var body struct {
		Graph     json.RawMessage `json:"graph"`
		Only      string          `json:"only"`
		RuntimeID string          `json:"runtime_id"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	g, err := workflow.ParseGraph(body.Graph)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid graph", err)
		return
	}
	plan, issues, info := svc.Plan(g, body.RuntimeID, workflow.Options{Only: body.Only})
	errs, warns := splitIssues(issues)
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":       len(errs) == 0,
		"errors":   errs,
		"warnings": warns,
		"plan":     plan,
		"runtime":  info,
	})
}

func (h *handlers) workflowExec(w http.ResponseWriter) (*workflow.Service, *workflow.Executor) {
	svc := h.workflowSvc(w)
	if svc == nil {
		return nil, nil
	}
	if svc.Executor() == nil {
		writeErr(w, http.StatusServiceUnavailable, "running workflows is unavailable", nil)
		return nil, nil
	}
	return svc, svc.Executor()
}

// startWorkflowRun plans a graph and queues it. A graph with problems is
// rejected with the same issue list /validate returns, so the canvas can mark
// them; nothing is queued. Progress arrives as workflow.* events.
func (h *handlers) startWorkflowRun(w http.ResponseWriter, r *http.Request) {
	svc, exec := h.workflowExec(w)
	if svc == nil {
		return
	}
	var body struct {
		Graph      json.RawMessage `json:"graph"`
		Only       string          `json:"only"`
		RuntimeID  string          `json:"runtime_id"`
		WorkflowID string          `json:"workflow_id"`
		Force      bool            `json:"force"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	g, err := workflow.ParseGraph(body.Graph)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid graph", err)
		return
	}
	plan, issues, _ := svc.Plan(g, body.RuntimeID, workflow.Options{Only: body.Only})
	errs, warns := splitIssues(issues)
	if plan == nil || len(errs) > 0 || len(plan.Stages) == 0 {
		msg := "graph has problems"
		if len(errs) == 0 {
			msg = "nothing to run"
		}
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": msg, "errors": errs, "warnings": warns})
		return
	}
	view, err := exec.SubmitWithOptions(plan, workflow.RunOptions{Graph: &g, WorkflowID: body.WorkflowID, Only: body.Only, Force: body.Force})
	if errors.Is(err, workflow.ErrQueueFull) {
		writeErr(w, http.StatusTooManyRequests, "too many runs queued", err)
		return
	}
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "starting run", err)
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"run": view, "warnings": warns})
}

func splitIssues(issues []workflow.Issue) (errs, warns []workflow.Issue) {
	errs, warns = []workflow.Issue{}, []workflow.Issue{}
	for _, i := range issues {
		if i.Severity == workflow.SeverityError {
			errs = append(errs, i)
		} else {
			warns = append(warns, i)
		}
	}
	return errs, warns
}

func (h *handlers) listWorkflowRuns(w http.ResponseWriter, r *http.Request) {
	_, exec := h.workflowExec(w)
	if exec == nil {
		return
	}
	runs, err := exec.History()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "reading run history", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"runs": runs})
}

func (h *handlers) getWorkflowRun(w http.ResponseWriter, r *http.Request) {
	_, exec := h.workflowExec(w)
	if exec == nil {
		return
	}
	view, ok := exec.Get(r.PathValue("id"))
	if !ok {
		writeErr(w, http.StatusNotFound, "run not found", nil)
		return
	}
	writeJSON(w, http.StatusOK, view)
}

func (h *handlers) cancelWorkflowRun(w http.ResponseWriter, r *http.Request) {
	_, exec := h.workflowExec(w)
	if exec == nil {
		return
	}
	if err := exec.Cancel(r.PathValue("id")); err != nil {
		if errors.Is(err, workflow.ErrRunNotFound) {
			writeErr(w, http.StatusNotFound, "run not found", err)
			return
		}
		writeErr(w, http.StatusInternalServerError, "cancel failed", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}
