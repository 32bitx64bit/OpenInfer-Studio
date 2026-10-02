package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/internal/mediagen"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/workflow"
	"github.com/openinfer/openinfer-studio/migrations"
)

type fakeLibrary struct{}

func (fakeLibrary) Get(id string) (*models.Model, error) {
	if id != "m1" {
		return nil, fmt.Errorf("model %s not found", id)
	}
	return &models.Model{ID: "m1", Alias: "flux-2-klein-9b", PrimaryPath: "/models/flux.gguf", SizeBytes: 5 << 30, Modality: "diffusion"}, nil
}

type fakeMedia struct {
	flags   []string
	running map[string]mediagen.LoadSettings
	dir     string
	jobs    *[]mediagen.GenerateParams // records generations when set
}

func (f fakeMedia) MediaDir() string { return f.dir }

func (f fakeMedia) RunStage(_ context.Context, _ string, p mediagen.GenerateParams, _ *mediagen.LoadOverrides) (*mediagen.Job, error) {
	if f.jobs == nil {
		return nil, fmt.Errorf("fake media cannot generate")
	}
	*f.jobs = append(*f.jobs, p)
	rel := "images/fake-0.png"
	abs := filepath.Join(f.dir, filepath.FromSlash(rel))
	_ = os.MkdirAll(filepath.Dir(abs), 0o755)
	if err := os.WriteFile(abs, []byte("PNG"), 0o644); err != nil {
		return nil, err
	}
	return &mediagen.Job{ID: "job-1", State: mediagen.StateComplete, OutputPaths: []string{rel}}, nil
}

func (f fakeMedia) RunningSettings(id string) (mediagen.LoadSettings, bool) {
	s, ok := f.running[id]
	return s, ok
}

func (f fakeMedia) RuntimeCapabilities(string) (string, []string, error) {
	return "sd-1", f.flags, nil
}

func newWorkflowMux(t *testing.T, svc *workflow.Service) http.Handler {
	t.Helper()
	h := &handlers{d: &Deps{Workflow: svc}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/v1/workflow/node-types", h.workflowNodeTypes)
	mux.HandleFunc("GET /api/v1/workflows", h.listWorkflows)
	mux.HandleFunc("POST /api/v1/workflows", h.createWorkflow)
	mux.HandleFunc("POST /api/v1/workflows/validate", h.validateWorkflow)
	mux.HandleFunc("POST /api/v1/workflow/runs", h.startWorkflowRun)
	mux.HandleFunc("GET /api/v1/workflow/runs", h.listWorkflowRuns)
	mux.HandleFunc("GET /api/v1/workflow/runs/{id}", h.getWorkflowRun)
	mux.HandleFunc("POST /api/v1/workflow/runs/{id}/cancel", h.cancelWorkflowRun)
	mux.HandleFunc("GET /api/v1/workflows/{id}", h.getWorkflow)
	mux.HandleFunc("PUT /api/v1/workflows/{id}", h.putWorkflow)
	mux.HandleFunc("DELETE /api/v1/workflows/{id}", h.deleteWorkflow)
	return mux
}

func newWorkflowService(t *testing.T, media fakeMedia) *workflow.Service {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "database")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	db, err := database.Open(dir, migrations.FS)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	svc := workflow.NewService(db.DB, fakeLibrary{}, media, nil)
	t.Cleanup(svc.Close)
	return svc
}

func call(t *testing.T, mux http.Handler, method, path, body string) (int, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	var out map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &out)
	return w.Code, out
}

const txt2imgJSON = `{
  "version": 1,
  "nodes": [
    {"id": "n1", "type": "checkpoint.load", "pos": [0, 0], "params": {"model": {"library_id": "m1", "name": "flux"}}},
    {"id": "n2", "type": "prompt", "pos": [0, 0], "params": {"text": "a fisherman"}},
    {"id": "n3", "type": "latent.empty", "pos": [0, 0], "params": {"width": 896, "height": 1152}},
    {"id": "n4", "type": "sample", "pos": [0, 0], "params": {"seed": 7}},
    {"id": "n5", "type": "image.save", "pos": [0, 0]}
  ],
  "edges": [
    {"from": ["n1", "model"], "to": ["n4", "model"]},
    {"from": ["n1", "clip"], "to": ["n4", "clip"]},
    {"from": ["n2", "cond"], "to": ["n4", "positive"]},
    {"from": ["n3", "size"], "to": ["n4", "start"]},
    {"from": ["n4", "image"], "to": ["n5", "image"]}
  ]
}`

func TestWorkflowCRUD(t *testing.T) {
	mux := newWorkflowMux(t, newWorkflowService(t, fakeMedia{}))

	code, rec := call(t, mux, "POST", "/api/v1/workflows", `{"name": "Portrait", "graph": `+txt2imgJSON+`}`)
	if code != 201 || rec["id"] == "" || rec["name"] != "Portrait" {
		t.Fatalf("create = %d %v", code, rec)
	}
	id := rec["id"].(string)

	code, list := call(t, mux, "GET", "/api/v1/workflows", "")
	items, _ := list["workflows"].([]any)
	if code != 200 || len(items) != 1 || items[0].(map[string]any)["node_count"] != float64(5) {
		t.Fatalf("list = %d %v", code, list)
	}

	code, got := call(t, mux, "GET", "/api/v1/workflows/"+id, "")
	graph, _ := got["graph"].(map[string]any)
	if code != 200 || graph["version"] != float64(1) {
		t.Fatalf("get = %d %v", code, got)
	}

	code, up := call(t, mux, "PUT", "/api/v1/workflows/"+id, `{"name": "Portrait v2"}`)
	if code != 200 || up["name"] != "Portrait v2" {
		t.Fatalf("rename = %d %v", code, up)
	}

	if code, _ := call(t, mux, "DELETE", "/api/v1/workflows/"+id, ""); code != 200 {
		t.Fatalf("delete = %d", code)
	}
	if code, _ := call(t, mux, "GET", "/api/v1/workflows/"+id, ""); code != 404 {
		t.Fatalf("get after delete = %d, want 404", code)
	}
	if code, _ := call(t, mux, "DELETE", "/api/v1/workflows/"+id, ""); code != 404 {
		t.Fatalf("second delete = %d, want 404", code)
	}
}

func TestWorkflowCreateRejectsBadInput(t *testing.T) {
	mux := newWorkflowMux(t, newWorkflowService(t, fakeMedia{}))
	for name, body := range map[string]string{
		"not json":       `{`,
		"unknown field":  `{"name": "x", "graph": {"version": 1}, "extra": 1}`,
		"missing graph":  `{"name": "x"}`,
		"future version": `{"name": "x", "graph": {"version": 99}}`,
		"empty name":     `{"name": " ", "graph": {"version": 1}}`,
	} {
		if code, out := call(t, mux, "POST", "/api/v1/workflows", body); code != 400 {
			t.Errorf("%s: status = %d (%v), want 400", name, code, out)
		}
	}
}

func TestWorkflowDraftsSaveEvenWhenInvalid(t *testing.T) {
	mux := newWorkflowMux(t, newWorkflowService(t, fakeMedia{}))
	draft := `{"name": "wip", "graph": {"version": 1, "nodes": [{"id": "n1", "type": "sample", "pos": [0,0]}], "edges": []}}`
	if code, out := call(t, mux, "POST", "/api/v1/workflows", draft); code != 201 {
		t.Fatalf("draft = %d %v", code, out)
	}
}

func TestWorkflowValidateReturnsAPlan(t *testing.T) {
	mux := newWorkflowMux(t, newWorkflowService(t, fakeMedia{flags: []string{"vae"}}))

	code, out := call(t, mux, "POST", "/api/v1/workflows/validate", `{"graph": `+txt2imgJSON+`}`)
	if code != 200 || out["ok"] != true {
		t.Fatalf("validate = %d %v", code, out)
	}
	plan := out["plan"].(map[string]any)
	stages := plan["stages"].([]any)
	if len(stages) != 2 {
		t.Fatalf("stages = %v", stages)
	}
	gen := stages[0].(map[string]any)["generate"].(map[string]any)
	params := gen["params"].(map[string]any)
	if params["prompt"] != "a fisherman" || params["width"] != float64(896) || params["seed"] != float64(7) {
		t.Fatalf("generate params = %v", params)
	}
	servers := plan["servers"].([]any)
	if len(servers) != 1 || servers[0].(map[string]any)["action"] != "start" {
		t.Fatalf("servers = %v", servers)
	}

	// the same graph with a server already running plans a reuse
	running := fakeMedia{flags: []string{"vae"}, running: map[string]mediagen.LoadSettings{"m1": {FlashAttention: true}}}
	mux = newWorkflowMux(t, newWorkflowService(t, running))
	_, out = call(t, mux, "POST", "/api/v1/workflows/validate", `{"graph": `+txt2imgJSON+`}`)
	servers = out["plan"].(map[string]any)["servers"].([]any)
	if servers[0].(map[string]any)["action"] != "reuse" {
		t.Fatalf("running server: servers = %v, want reuse", servers)
	}
}

func TestWorkflowValidateReportsProblemsPerNode(t *testing.T) {
	mux := newWorkflowMux(t, newWorkflowService(t, fakeMedia{}))
	broken := strings.Replace(txt2imgJSON, `"library_id": "m1"`, `"library_id": "ghost"`, 1)
	broken = strings.Replace(broken, `"text": "a fisherman"`, `"text": ""`, 1)

	code, out := call(t, mux, "POST", "/api/v1/workflows/validate", `{"graph": `+broken+`}`)
	if code != 200 || out["ok"] != false || out["plan"] != nil {
		t.Fatalf("validate = %d %v", code, out)
	}
	codes := map[string]string{}
	for _, e := range out["errors"].([]any) {
		m := e.(map[string]any)
		codes[m["code"].(string)] = m["node"].(string)
	}
	if codes["plan.model_missing"] != "n1" || codes["plan.prompt_empty"] != "n4" {
		t.Fatalf("errors = %v", codes)
	}
}

func TestWorkflowValidateRejectsMalformedGraph(t *testing.T) {
	mux := newWorkflowMux(t, newWorkflowService(t, fakeMedia{}))
	if code, _ := call(t, mux, "POST", "/api/v1/workflows/validate", `{"graph": {"version": 3}}`); code != 400 {
		t.Fatalf("future version: status = %d, want 400", code)
	}
}

func TestWorkflowNodeTypesFollowRuntimeCapabilities(t *testing.T) {
	mux := newWorkflowMux(t, newWorkflowService(t, fakeMedia{flags: []string{"vae"}}))
	code, out := call(t, mux, "GET", "/api/v1/workflow/node-types", "")
	if code != 200 {
		t.Fatalf("node-types = %d", code)
	}
	avail := map[string]bool{}
	for _, v := range out["node_types"].([]any) {
		m := v.(map[string]any)
		avail[m["type"].(string)] = m["available"].(bool)
	}
	if !avail["sample"] || !avail["vae.load"] || avail["lora.load"] {
		t.Fatalf("availability = %v, want sample and vae.load on, lora.load off", avail)
	}
	if rt := out["runtime"].(map[string]any); rt["known"] != true {
		t.Fatalf("runtime = %v", rt)
	}
	if types := out["port_types"].([]any); len(types) != 9 {
		t.Fatalf("port_types = %v", types)
	}
}

func TestWorkflowEndpointsReturn503WithoutService(t *testing.T) {
	mux := newWorkflowMux(t, nil)
	for _, p := range []string{"/api/v1/workflows", "/api/v1/workflow/node-types"} {
		if code, _ := call(t, mux, "GET", p, ""); code != 503 {
			t.Errorf("GET %s = %d, want 503", p, code)
		}
	}
}

func TestWorkflowRunEndToEnd(t *testing.T) {
	var gens []mediagen.GenerateParams
	svc := newWorkflowService(t, fakeMedia{flags: []string{"vae"}, dir: t.TempDir(), jobs: &gens})
	mux := newWorkflowMux(t, svc)

	code, out := call(t, mux, "POST", "/api/v1/workflow/runs", `{"graph": `+txt2imgJSON+`}`)
	if code != 202 {
		t.Fatalf("start run = %d %v", code, out)
	}
	run := out["run"].(map[string]any)
	id := run["id"].(string)
	if run["state"] != "queued" {
		t.Fatalf("run = %v", run)
	}

	var final map[string]any
	for i := 0; i < 400; i++ {
		_, final = call(t, mux, "GET", "/api/v1/workflow/runs/"+id, "")
		if final["state"] == "complete" || final["state"] == "failed" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if final["state"] != "complete" {
		t.Fatalf("final run = %v", final)
	}
	nodes := final["nodes"].(map[string]any)
	sample := nodes["n4"].(map[string]any)
	if sample["state"] != "done" || sample["message"] != "seed 7" {
		t.Fatalf("sample node = %v", sample)
	}
	if len(gens) != 1 || gens[0].Prompt != "a fisherman" || gens[0].Width != 896 {
		t.Fatalf("generations = %+v", gens)
	}

	code, list := call(t, mux, "GET", "/api/v1/workflow/runs", "")
	if code != 200 || len(list["runs"].([]any)) != 1 {
		t.Fatalf("list runs = %d %v", code, list)
	}
	if code, _ := call(t, mux, "POST", "/api/v1/workflow/runs/"+id+"/cancel", ""); code != 200 {
		t.Fatalf("cancel finished run = %d", code)
	}
	if code, _ := call(t, mux, "GET", "/api/v1/workflow/runs/nope", ""); code != 404 {
		t.Fatalf("unknown run = %d, want 404", code)
	}
	if code, _ := call(t, mux, "POST", "/api/v1/workflow/runs/nope/cancel", ""); code != 404 {
		t.Fatalf("cancel unknown = %d, want 404", code)
	}
}

func TestWorkflowRunRejectsProblemGraphsWithIssues(t *testing.T) {
	mux := newWorkflowMux(t, newWorkflowService(t, fakeMedia{dir: t.TempDir()}))
	broken := strings.Replace(txt2imgJSON, `"library_id": "m1"`, `"library_id": "ghost"`, 1)
	code, out := call(t, mux, "POST", "/api/v1/workflow/runs", `{"graph": `+broken+`}`)
	if code != 400 || out["error"] != "graph has problems" {
		t.Fatalf("run = %d %v", code, out)
	}
	if errs := out["errors"].([]any); len(errs) == 0 || errs[0].(map[string]any)["code"] != "plan.model_missing" {
		t.Fatalf("errors = %v", out["errors"])
	}

	noSave := strings.Replace(txt2imgJSON, `{"id": "n5", "type": "image.save", "pos": [0, 0]}`, `{"id": "n5", "type": "prompt", "pos": [0, 0]}`, 1)
	noSave = strings.Replace(noSave, `{"from": ["n4", "image"], "to": ["n5", "image"]}`, `{"from": ["n2", "cond"], "to": ["n4", "negative"]}`, 1)
	code, out = call(t, mux, "POST", "/api/v1/workflow/runs", `{"graph": `+noSave+`}`)
	if code != 400 || out["error"] != "nothing to run" {
		t.Fatalf("no output node = %d %v", code, out)
	}
}
