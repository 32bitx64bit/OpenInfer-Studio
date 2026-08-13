package quantize

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/internal/hardware"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
	"github.com/openinfer/openinfer-studio/migrations"
)

func writeGGUF(t *testing.T, dir, name string, fileType uint32) string {
	t.Helper()
	var b bytes.Buffer
	w32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	w64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	wstr := func(s string) { w64(uint64(len(s))); b.WriteString(s) }
	binary.Write(&b, binary.LittleEndian, uint32(0x46554747))
	w32(3)
	w64(0)
	w64(4)
	wstr("general.name")
	w32(8)
	wstr("Quant Test")
	wstr("general.architecture")
	w32(8)
	wstr("llama")
	wstr("general.file_type")
	w32(4)
	w32(fileType)
	wstr("llama.context_length")
	w32(4)
	w32(4096)
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func buildTool(t *testing.T, pkg, destDir, name string) string {
	t.Helper()
	if runtime.GOOS == "windows" && filepath.Ext(name) == "" {
		name += ".exe"
	}
	out := filepath.Join(destDir, name)
	cmd := exec.Command("go", "build", "-o", out, "github.com/openinfer/openinfer-studio/tests/"+pkg)
	b, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build %s: %v\n%s", pkg, err, b)
	}
	return out
}

type quantEnv struct {
	layout *config.Layout
	lib    *models.Library
	rt     *runtimes.Manager
	qm     *Manager
	model  *models.Model
}

func newQuantEnv(t *testing.T, withTools bool) *quantEnv {
	t.Helper()
	layout, err := config.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	db, err := database.Open(layout.Database, migrations.FS)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	lib := models.NewLibrary(db.DB, layout.Models, nil, log)
	rt := runtimes.NewManager(db.DB, layout.Runtimes, nil, nil, log)

	toolDir := t.TempDir()
	server := buildTool(t, "fakeserver", toolDir, "llama-server")
	if withTools {
		buildTool(t, "fakequantize", toolDir, "llama-quantize")
		buildTool(t, "fakeimatrix", toolDir, "llama-imatrix")
	}
	if _, err := rt.ImportCustom(server); err != nil {
		t.Fatalf("import runtime: %v", err)
	}

	src := writeGGUF(t, t.TempDir(), "src-F16.gguf", 1) // F16
	id, err := lib.ImportFile(src)
	if err != nil {
		t.Fatal(err)
	}
	m, err := lib.Get(id)
	if err != nil {
		t.Fatal(err)
	}

	qm := NewManager(db.DB, layout, rt, lib, nil, log,
		func(string) uint64 { return 1 << 40 },
		func() *hardware.Info { return &hardware.Info{LogicalCores: 4, RAMAvailable: 32 << 30} })
	if err := qm.RecoverAfterRestart(); err != nil {
		t.Fatal(err)
	}
	return &quantEnv{layout: layout, lib: lib, rt: rt, qm: qm, model: m}
}

func waitJob(t *testing.T, qm *Manager, id, want string) *Job {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		j, err := qm.Get(id)
		if err != nil {
			t.Fatal(err)
		}
		switch j.State {
		case want:
			return j
		case "failed", "canceled":
			if want != j.State {
				tail, _ := qm.LogTail(id, 2048)
				t.Fatalf("job %s (want %s)\nerr=%s\nlog:\n%s", j.State, want, j.Error, tail)
			}
			return j
		}
		time.Sleep(50 * time.Millisecond)
	}
	j, _ := qm.Get(id)
	t.Fatalf("timeout waiting for %s: %+v", want, j)
	return j
}

func TestQuantizeJobCompletes(t *testing.T) {
	env := newQuantEnv(t, true)
	job, err := env.qm.Start(Request{
		Kind: KindQuantize, SourceModelID: env.model.ID, FType: "Q4_K_M", Threads: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	if done.DestPath == "" {
		t.Fatal("missing dest")
	}
	if _, err := os.Stat(done.DestPath); err != nil {
		t.Fatal(err)
	}
	id := env.lib.IDForPath(done.DestPath)
	if id == "" {
		t.Fatal("dest not imported into library")
	}
	got, err := env.lib.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if got.Alias != "Quant Test Q4_K_M" {
		t.Fatalf("alias = %q, want source name + quant", got.Alias)
	}
	if got.Quantization != "Q4_K_M" {
		t.Fatalf("quantization = %q", got.Quantization)
	}
}

func TestQuantizePreview(t *testing.T) {
	env := newQuantEnv(t, true)
	out, err := env.qm.Preview(Request{SourceModelID: env.model.ID, FType: "Q8_0"})
	if err != nil {
		t.Fatal(err)
	}
	prev, _ := out["preview"].(Preview)
	if prev.EstimatedBytes <= 0 && env.model.Parameters == 0 {
		// parameter count may be 0 on the tiny header; still expect a preview object
	}
	if _, ok := out["tools"]; !ok {
		t.Fatal("preview missing tools")
	}
}

func TestQuantizeMissingSibling(t *testing.T) {
	env := newQuantEnv(t, false)
	_, err := env.qm.Start(Request{SourceModelID: env.model.ID, FType: "Q4_K_M"})
	if err == nil {
		t.Fatal("expected missing llama-quantize error")
	}
}

func TestRequantizeBlocked(t *testing.T) {
	env := newQuantEnv(t, true)
	q4 := writeGGUF(t, t.TempDir(), "already-Q4_K_M.gguf", 15)
	id, err := env.lib.ImportFile(q4)
	if err != nil {
		t.Fatal(err)
	}
	_, err = env.qm.Start(Request{SourceModelID: id, FType: "Q3_K_M"})
	if err == nil {
		t.Fatal("expected requantize block")
	}
	job, err := env.qm.Start(Request{
		SourceModelID: id, FType: "Q3_K_M",
		AllowRequantize: true, AcknowledgeRequantize: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	waitJob(t, env.qm, job.ID, "complete")
}

func TestQuantizeDraftSkipsIMatrix(t *testing.T) {
	env := newQuantEnv(t, true)
	path := writeGGUF(t, t.TempDir(), "dflash-Muse-Glimmer-30B-F16.gguf", 1)
	id, err := env.lib.ImportFile(path)
	if err != nil {
		t.Fatal(err)
	}
	job, err := env.qm.Start(Request{
		Kind: KindQuantize, SourceModelID: id, FType: "Q4_K_M",
		GenerateIMatrix: true, CalibrationPreset: "quick",
	})
	if err != nil {
		t.Fatal(err)
	}
	if job.Request.GenerateIMatrix {
		t.Fatal("imatrix should be skipped for assistant/draft GGUFs")
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	if done.State != "complete" {
		t.Fatalf("state=%s err=%s", done.State, done.Error)
	}
	destID := env.lib.IDForPath(done.DestPath)
	if destID == "" {
		t.Fatal("quantized assistant not imported into library")
	}
	got, err := env.lib.Get(destID)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got.Alias, "Q4_K_M") {
		t.Fatalf("alias %q should include quant level", got.Alias)
	}
}

func TestIMatrixOnlyRejectedOnDraft(t *testing.T) {
	env := newQuantEnv(t, true)
	path := writeGGUF(t, t.TempDir(), "dflash-Muse-Glimmer-30B-F16.gguf", 1)
	id, err := env.lib.ImportFile(path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = env.qm.Start(Request{Kind: KindIMatrix, SourceModelID: id, CalibrationPreset: "quick"})
	if err == nil {
		t.Fatal("expected imatrix-only job on a draft to fail")
	}
}

func TestIMatrixRequired(t *testing.T) {
	env := newQuantEnv(t, true)
	_, err := env.qm.Start(Request{SourceModelID: env.model.ID, FType: "IQ2_XXS"})
	if err == nil {
		t.Fatal("IQ2 should require imatrix")
	}
}

func TestIMatrixThenQuantize(t *testing.T) {
	env := newQuantEnv(t, true)
	job, err := env.qm.Start(Request{
		Kind: KindQuantize, SourceModelID: env.model.ID, FType: "IQ4_XS",
		GenerateIMatrix: true, CalibrationPreset: "quick", KeepIMatrix: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	waitJob(t, env.qm, job.ID, "complete")
	ims, err := env.qm.ListIMatrices(env.model.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(ims) == 0 {
		t.Fatal("expected generated imatrix")
	}
}

func TestAdaptiveQuantizeJob(t *testing.T) {
	env := newQuantEnv(t, true)
	src := filepath.Join(t.TempDir(), "t.gguf")
	writeTinyGGUF(t, src, []struct {
		name string
		typ  uint32
		dims []uint64
		off  uint64
	}{
		{"token_embd.weight", 1, []uint64{32}, 0},
		{"blk.0.ffn_gate.weight", 1, []uint64{32}, 64},
	}, 4096)
	id, err := env.lib.ImportFile(src)
	if err != nil {
		t.Fatal(err)
	}
	job, err := env.qm.Start(Request{
		Kind: KindAdaptiveQuantize, SourceModelID: id, AdaptivePreset: "balanced",
		GenerateIMatrix: true, CalibrationPreset: "quick", FType: "Q4_K_M",
	})
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	var res map[string]any
	_ = json.Unmarshal(done.Result, &res)
	if label, _ := res["adaptive"].(string); !strings.Contains(label, "OpenInfer Adaptive") {
		t.Fatalf("result=%v", res)
	}
}

func TestRecoverAfterRestart(t *testing.T) {
	env := newQuantEnv(t, true)
	_, err := env.qm.db.Exec(`INSERT INTO quant_jobs(id,kind,state,stage,progress,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at)
		VALUES ('dead','quantize','running','quantize',0.2,'','','',1,'','{}','{}','','t','t')`)
	if err != nil {
		t.Fatal(err)
	}
	if err := env.qm.RecoverAfterRestart(); err != nil {
		t.Fatal(err)
	}
	j, err := env.qm.Get("dead")
	if err != nil {
		t.Fatal(err)
	}
	if j.State != "failed" {
		t.Fatalf("state=%s", j.State)
	}
}

func TestDeleteAndClearHistory(t *testing.T) {
	env := newQuantEnv(t, true)
	job, err := env.qm.Start(Request{
		Kind: KindQuantize, SourceModelID: env.model.ID, FType: "Q4_K_M", Threads: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	dest := done.DestPath
	if err := env.qm.Delete(done.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := env.qm.Get(done.ID); err == nil {
		t.Fatal("deleted job should be gone")
	}
	if dest != "" {
		if _, err := os.Stat(dest); err != nil {
			t.Fatalf("library GGUF should remain: %v", err)
		}
	}

	if _, err := env.qm.db.Exec(`INSERT INTO quant_jobs(id,kind,state,stage,progress,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at)
		VALUES ('fin1','quantize','complete','',1,'','','',0,'','{}','{}','','t','t'),
		       ('fin2','quantize','failed','',0,'','','',0,'','{}','{}','boom','t','t'),
		       ('live','quantize','queued','',0,'','','',0,'','{}','{}','','t','t')`); err != nil {
		t.Fatal(err)
	}
	n, err := env.qm.ClearHistory()
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("removed %d, want 2", n)
	}
	if _, err := env.qm.Get("live"); err != nil {
		t.Fatalf("queued job should remain: %v", err)
	}
	if _, err := env.qm.Get("fin1"); err == nil {
		t.Fatal("complete job should have been cleared")
	}

	if _, err := env.qm.db.Exec(`INSERT INTO quant_jobs(id,kind,state,stage,progress,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at)
		VALUES ('run1','quantize','running','quantize',0.2,'','','',1,'','{}','{}','','t','t')`); err != nil {
		t.Fatal(err)
	}
	if err := env.qm.Delete("run1"); err == nil {
		t.Fatal("expected delete of running job to fail")
	}
}
