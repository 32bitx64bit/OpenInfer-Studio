package mediagen

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/migrations"
)

// fakeSD is a minimal native sdcpp API: img_gen accepts a job, the job poll
// answers with the configured terminal document.
type fakeSD struct {
	srv      *httptest.Server
	requests []map[string]any
	poll     map[string]any
}

func newFakeSD(t *testing.T, poll map[string]any) *fakeSD {
	t.Helper()
	f := &fakeSD{poll: poll}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /sdcpp/v1/img_gen", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.requests = append(f.requests, body)
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(map[string]string{"id": "sd-1"})
	})
	mux.HandleFunc("GET /sdcpp/v1/jobs/sd-1", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(f.poll)
	})
	f.srv = httptest.NewServer(mux)
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeSD) port(t *testing.T) int {
	t.Helper()
	u, err := url.Parse(f.srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	p, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// newJobManager returns a manager over a temp database with a ready
// supervised server for "model-1" pointing at the fake.
func newJobManager(t *testing.T, f *fakeSD) *Manager {
	t.Helper()
	dir := t.TempDir()
	dbDir := filepath.Join(dir, "database")
	if err := os.MkdirAll(dbDir, 0o755); err != nil {
		t.Fatal(err)
	}
	db, err := database.Open(dbDir, migrations.FS)
	if err != nil {
		t.Fatalf("open database: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	m := NewManager(db.DB, &config.Layout{DataDir: dir}, nil, nil, nil, nil)
	m.servers["model-1"] = &server{modelID: "model-1", port: f.port(t), ready: true, state: ServerReady}

	old := sdPollInterval
	sdPollInterval = 5 * time.Millisecond
	t.Cleanup(func() { sdPollInterval = old })
	return m
}

func insertJob(t *testing.T, m *Manager, id string, p GenerateParams) {
	t.Helper()
	body, _ := json.Marshal(p)
	kind := p.Kind
	if kind == "" {
		kind = KindImage
	}
	ts := now()
	if _, err := m.db.Exec(`INSERT INTO media_jobs(id,model_id,kind,state,prompt,params_json,seed,created_at,updated_at)
		VALUES (?,?,?,?,?,?,?,?,?)`, id, "model-1", kind, StateQueued, p.Prompt, string(body), p.Seed, ts, ts); err != nil {
		t.Fatalf("insert job: %v", err)
	}
}

func TestRunJobCompletesAndStoresOutput(t *testing.T) {
	f := newFakeSD(t, map[string]any{
		"status": "completed",
		"result": map[string]any{
			"output_format": "png",
			"images":        []any{map[string]any{"b64_json": base64.StdEncoding.EncodeToString([]byte("PNGDATA"))}},
		},
	})
	m := newJobManager(t, f)
	p := GenerateParams{Kind: KindImage, Prompt: "a fisherman", Width: 896, Height: 1152, Steps: 28, CFGScale: 1, Seed: 7, Sampler: "euler"}
	ValidateGenerateParams(&p)
	insertJob(t, m, "job-ok", p)

	m.runJob(context.Background(), "job-ok", "model-1", KindImage, p, nil)

	j, err := m.Get("job-ok")
	if err != nil {
		t.Fatal(err)
	}
	if j.State != StateComplete {
		t.Fatalf("state = %q (error %q), want complete", j.State, j.Error)
	}
	if len(j.OutputPaths) != 1 {
		t.Fatalf("outputs = %v, want one", j.OutputPaths)
	}
	raw, err := os.ReadFile(filepath.Join(m.MediaDir(), filepath.FromSlash(j.OutputPaths[0])))
	if err != nil || string(raw) != "PNGDATA" {
		t.Fatalf("stored output = %q, %v; want PNGDATA", raw, err)
	}
	if len(f.requests) != 1 {
		t.Fatalf("img_gen calls = %d, want 1", len(f.requests))
	}
	got := f.requests[0]
	if got["prompt"] != "a fisherman" || got["width"] != float64(896) || got["height"] != float64(1152) || got["seed"] != float64(7) {
		t.Fatalf("request body = %v", got)
	}
}

func TestRunJobFailureIsRecorded(t *testing.T) {
	f := newFakeSD(t, map[string]any{
		"status": "failed",
		"error":  map[string]any{"message": "out of memory"},
	})
	m := newJobManager(t, f)
	p := GenerateParams{Kind: KindImage, Prompt: "x", Seed: -1}
	insertJob(t, m, "job-fail", p)

	m.runJob(context.Background(), "job-fail", "model-1", KindImage, p, nil)

	j, err := m.Get("job-fail")
	if err != nil {
		t.Fatal(err)
	}
	if j.State != StateFailed || j.Error != "out of memory" {
		t.Fatalf("state=%q error=%q, want failed / out of memory", j.State, j.Error)
	}
}

func okPoll() map[string]any {
	return map[string]any{
		"status": "completed",
		"result": map[string]any{
			"output_format": "png",
			"images":        []any{map[string]any{"b64_json": base64.StdEncoding.EncodeToString([]byte("PNGDATA"))}},
		},
	}
}

func TestRunStageReturnsFinishedJob(t *testing.T) {
	f := newFakeSD(t, okPoll())
	m := newJobManager(t, f)

	j, err := m.RunStage(context.Background(), "model-1", GenerateParams{Prompt: "a boat", Seed: 3}, nil)
	if err != nil {
		t.Fatalf("RunStage: %v", err)
	}
	if j.State != StateComplete || len(j.OutputPaths) != 1 {
		t.Fatalf("job = state %q outputs %v, want complete with one output", j.State, j.OutputPaths)
	}
}

func TestRunStageSurfacesFailureMessage(t *testing.T) {
	f := newFakeSD(t, map[string]any{"status": "failed", "error": map[string]any{"message": "out of memory"}})
	m := newJobManager(t, f)

	j, err := m.RunStage(context.Background(), "model-1", GenerateParams{Prompt: "a boat"}, nil)
	if err == nil || err.Error() != "out of memory" {
		t.Fatalf("err = %v, want out of memory", err)
	}
	if j == nil || j.State != StateFailed {
		t.Fatalf("job = %+v, want the failed job returned alongside the error", j)
	}
}

func TestRunStageCancelsJobWhenContextEnds(t *testing.T) {
	f := newFakeSD(t, map[string]any{"status": "generating"})
	m := newJobManager(t, f)
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(40*time.Millisecond, cancel)

	j, err := m.RunStage(ctx, "model-1", GenerateParams{Prompt: "a boat"}, nil)
	if err != context.Canceled {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if j == nil || j.State != StateCanceled {
		t.Fatalf("job = %+v, want state canceled", j)
	}
}

func TestRunStageReusesServerThatSatisfiesOverrides(t *testing.T) {
	f := newFakeSD(t, okPoll())
	m := newJobManager(t, f)
	m.servers["model-1"].resolved = LoadSettings{VAE: "/models/flux2-vae.safetensors", FlashAttention: true}

	ov := &LoadOverrides{VAE: "/models/flux2-vae.safetensors"}
	if _, err := m.RunStage(context.Background(), "model-1", GenerateParams{Prompt: "a boat"}, ov); err != nil {
		t.Fatalf("RunStage: %v", err)
	}
	if _, ok := m.ServerPort("model-1"); !ok {
		t.Fatal("a satisfied override must keep the running server")
	}
	if len(f.requests) != 1 {
		t.Fatalf("img_gen calls = %d, want 1", len(f.requests))
	}
}
