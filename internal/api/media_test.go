package api

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/internal/mediagen"
	"github.com/openinfer/openinfer-studio/migrations"
)

func TestSaveTargets(t *testing.T) {
	t.Run("single output lands exactly on dest", func(t *testing.T) {
		got := saveTargets("/tmp/photo.png", 1)
		want := []string{"/tmp/photo.png"}
		if len(got) != 1 || got[0] != want[0] {
			t.Fatalf("saveTargets = %v, want %v", got, want)
		}
	})

	t.Run("batch extras are numbered before the extension", func(t *testing.T) {
		got := saveTargets(filepath.FromSlash("/tmp/photo.png"), 3)
		want := []string{
			filepath.FromSlash("/tmp/photo.png"),
			filepath.FromSlash("/tmp/photo-1.png"),
			filepath.FromSlash("/tmp/photo-2.png"),
		}
		if len(got) != len(want) {
			t.Fatalf("saveTargets len = %d, want %d", len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("saveTargets[%d] = %q, want %q", i, got[i], want[i])
			}
		}
	})

	t.Run("name without extension is numbered bare", func(t *testing.T) {
		got := saveTargets("/tmp/photo", 2)
		if got[0] != "/tmp/photo" || got[1] != "/tmp/photo-1" {
			t.Fatalf("saveTargets = %v, want [/tmp/photo /tmp/photo-1]", got)
		}
	})
}

// newSaveTest wires a real media manager over a temp database and returns the
// handler, its open database, media dir, and temp root.
func newSaveTest(t *testing.T) (*handlers, *sql.DB, string, string) {
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
	mgr := mediagen.NewManager(db.DB, &config.Layout{DataDir: dir}, nil, nil, nil, nil)
	return &handlers{d: &Deps{Media: mgr}}, db.DB, mgr.MediaDir(), dir
}

func insertSaveJob(t *testing.T, db *sql.DB, id string, outputs []string) {
	t.Helper()
	paths, err := json.Marshal(map[string]any{"outputs": outputs})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO media_jobs
		(id,model_id,kind,state,prompt,params_json,output_path,output_format,result_json,created_at,updated_at)
		VALUES (?,?,?,?,?,?,?,?,?,?,?)`,
		id, "model-1", "image", "complete", "a cat", "{}",
		outputs[0], "png", string(paths),
		"2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"); err != nil {
		t.Fatalf("insert job: %v", err)
	}
}

func postSave(t *testing.T, h *handlers, id, body string) *httptest.ResponseRecorder {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/v1/media/jobs/{id}/save", h.saveMediaJob)
	req := httptest.NewRequest("POST", "/api/v1/media/jobs/"+id+"/save", strings.NewReader(body))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	return w
}

func TestSaveMediaJobSavesBatchWithSuffixes(t *testing.T) {
	h, db, mediaDir, dir := newSaveTest(t)
	outputs := []string{"images/a.png", "images/b.png"}
	for _, rel := range outputs {
		p := filepath.Join(mediaDir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("content-"+rel), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	insertSaveJob(t, db, "job-batch", outputs)

	dest := filepath.Join(dir, "photo.png")
	body, _ := json.Marshal(map[string]string{"dest_path": dest})
	w := postSave(t, h, "job-batch", string(body))
	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	want := map[string]string{
		dest:                              "content-" + outputs[0],
		filepath.Join(dir, "photo-1.png"): "content-" + outputs[1],
	}
	for path, content := range want {
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if string(got) != content {
			t.Fatalf("%s = %q, want %q", path, got, content)
		}
	}
}

func TestSaveMediaJobSingleOutputLandsOnDest(t *testing.T) {
	h, db, mediaDir, dir := newSaveTest(t)
	p := filepath.Join(mediaDir, "images", "only.png")
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("solo"), 0o644); err != nil {
		t.Fatal(err)
	}
	insertSaveJob(t, db, "job-solo", []string{"images/only.png"})

	dest := filepath.Join(dir, "out.webp")
	body, _ := json.Marshal(map[string]string{"dest_path": dest})
	w := postSave(t, h, "job-solo", string(body))
	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if _, err := os.Stat(dest); err != nil {
		t.Fatalf("dest missing: %v", err)
	}
	if _, err := os.Stat(dest + "-0"); err == nil {
		t.Fatal("single-output save must not create numbered siblings")
	}
}

func TestSaveMediaJobRejectsRelativeDest(t *testing.T) {
	h, db, _, _ := newSaveTest(t)
	insertSaveJob(t, db, "job-rel", []string{"images/a.png"})

	w := postSave(t, h, "job-rel", `{"dest_path":"relative/photo.png"}`)
	if w.Code != 400 {
		t.Fatalf("status = %d, want 400; body = %s", w.Code, w.Body.String())
	}
}

func TestSaveMediaJobUnknownJob(t *testing.T) {
	h, _, _, dir := newSaveTest(t)
	body, _ := json.Marshal(map[string]string{"dest_path": filepath.Join(dir, "x.png")})
	w := postSave(t, h, "job-missing", string(body))
	if w.Code != 404 {
		t.Fatalf("status = %d, want 404; body = %s", w.Code, w.Body.String())
	}
}

func TestSaveMediaJobMissingOutputFile(t *testing.T) {
	h, db, _, dir := newSaveTest(t)
	insertSaveJob(t, db, "job-gone", []string{"images/never-written.png"})

	body, _ := json.Marshal(map[string]string{"dest_path": filepath.Join(dir, "x.png")})
	w := postSave(t, h, "job-gone", string(body))
	if w.Code != 404 {
		t.Fatalf("status = %d, want 404; body = %s", w.Code, w.Body.String())
	}
}
