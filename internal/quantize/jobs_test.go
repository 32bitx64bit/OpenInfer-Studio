package quantize

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/convert"
	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/internal/downloads"
	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/hardware"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
	"github.com/openinfer/openinfer-studio/migrations"
	"quantlab/core"
	"quantlab/orchestrate"
	"quantlab/pipeline"
	"quantlab/state"
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

func buildQuantlabTool(t *testing.T, destDir, name string) string {
	t.Helper()
	if runtime.GOOS == "windows" && filepath.Ext(name) == "" {
		name += ".exe"
	}
	out := filepath.Join(destDir, name)
	cmd := exec.Command("go", "build", "-o", out, "github.com/openinfer/openinfer-studio/internal/quantize/testtool")
	b, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build quantlab test tool: %v\n%s", err, b)
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
		buildTool(t, "fakeperplexity", toolDir, "llama-perplexity")
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

// newQuantlabEnv keeps the normal test fakes for non-adaptive coverage while
// installing an anchor-capable quantize/perplexity tool for quantlab runs.
func newQuantlabEnv(t *testing.T) *quantEnv {
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
	buildQuantlabTool(t, toolDir, "llama-quantize")
	buildQuantlabTool(t, toolDir, "llama-perplexity")
	buildTool(t, "fakeimatrix", toolDir, "llama-imatrix")
	if _, err := rt.ImportCustom(server); err != nil {
		t.Fatalf("import runtime: %v", err)
	}

	src := writeGGUF(t, t.TempDir(), "src-F16.gguf", 1)
	id, err := lib.ImportFile(src)
	if err != nil {
		t.Fatal(err)
	}
	model, err := lib.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	qm := NewManager(db.DB, layout, rt, lib, nil, log,
		func(string) uint64 { return 1 << 40 },
		func() *hardware.Info { return &hardware.Info{LogicalCores: 4, RAMAvailable: 32 << 30} })
	if err := qm.RecoverAfterRestart(); err != nil {
		t.Fatal(err)
	}
	return &quantEnv{layout: layout, lib: lib, rt: rt, qm: qm, model: model}
}

type quantlabTestTensor struct {
	name string
	typ  uint32
	dims []uint64
}

// writeQuantlabTestGGUF writes a complete, aligned GGUF with enough real
// payload for quantlab's tensorbank parser and anchor assembler.
func writeQuantlabTestGGUF(t *testing.T, path, modelName string, tensors []quantlabTestTensor) {
	t.Helper()
	const alignment = uint64(32)
	type record struct {
		quantlabTestTensor
		offset uint64
		bytes  uint64
	}
	bytesFor := func(typ uint32, elements uint64) uint64 {
		switch typ {
		case 0:
			return elements * 4
		case 1:
			return elements * 2
		default:
			t.Fatalf("unsupported test tensor type %d", typ)
			return 0
		}
	}
	alignUp := func(n uint64) uint64 { return (n + alignment - 1) / alignment * alignment }
	var records []record
	var cursor uint64
	for _, tensor := range tensors {
		var elements uint64 = 1
		for _, dim := range tensor.dims {
			elements *= dim
		}
		cursor = alignUp(cursor)
		records = append(records, record{quantlabTestTensor: tensor, offset: cursor, bytes: bytesFor(tensor.typ, elements)})
		cursor += records[len(records)-1].bytes
	}

	var b bytes.Buffer
	w32 := func(v uint32) { _ = binary.Write(&b, binary.LittleEndian, v) }
	w64 := func(v uint64) { _ = binary.Write(&b, binary.LittleEndian, v) }
	wstr := func(v string) { w64(uint64(len(v))); b.WriteString(v) }
	wKVString := func(key, value string) {
		wstr(key)
		w32(8)
		wstr(value)
	}
	wKVUint32 := func(key string, value uint32) {
		wstr(key)
		w32(4)
		w32(value)
	}
	w32(0x46554747)
	w32(3)
	w64(uint64(len(records)))
	w64(5)
	wKVString("general.name", modelName)
	wKVString("general.architecture", "llama")
	wKVUint32("general.file_type", 1)
	wKVUint32("general.alignment", uint32(alignment))
	wKVUint32("llama.context_length", 4096)
	for _, record := range records {
		wstr(record.name)
		w32(uint32(len(record.dims)))
		for _, dim := range record.dims {
			w64(dim)
		}
		w32(record.typ)
		w64(record.offset)
	}
	dataStart := alignUp(uint64(b.Len()))
	for uint64(b.Len()) < dataStart {
		b.WriteByte(0)
	}
	for _, record := range records {
		want := dataStart + record.offset
		for uint64(b.Len()) < want {
			b.WriteByte(0)
		}
		b.Write(make([]byte, record.bytes))
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func importQuantlabTestModel(t *testing.T, env *quantEnv, name string) *models.Model {
	t.Helper()
	path := filepath.Join(t.TempDir(), "adaptive-source.gguf")
	writeQuantlabTestGGUF(t, path, name, []quantlabTestTensor{
		{"token_embd.weight", 1, []uint64{256, 128}},
		{"blk.0.attn_q.weight", 1, []uint64{256, 128}},
		{"blk.0.attn_v.weight", 1, []uint64{256, 128}},
		{"blk.0.ffn_gate.weight", 1, []uint64{256, 128}},
		{"blk.0.ffn_down.weight", 1, []uint64{256, 128}},
		{"output.weight", 1, []uint64{256, 128}},
	})
	id, err := env.lib.ImportFile(path)
	if err != nil {
		t.Fatal(err)
	}
	model, err := env.lib.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

// dummyCheckpointConfig is a valid quantlab RunConfig with OS-absolute paths.
func dummyCheckpointConfig(t *testing.T) state.RunConfig {
	t.Helper()
	dir := t.TempDir()
	tool := filepath.Join(dir, "tool")
	return state.RunConfig{
		SourcePath:  filepath.Join(dir, "source.gguf"),
		OutputDir:   filepath.Join(dir, "out"),
		WorkDir:     filepath.Join(dir, "work"),
		EvalCorpus:  filepath.Join(dir, "eval.txt"),
		BudgetBytes: 1,
		CtxSize:     512,
		Threads:     1,
		Tools:       state.ToolPaths{LlamaQuantize: tool, LlamaPerplexity: tool},
	}
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
		case "failed", "canceled", "paused", "complete":
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
	if prev.EstimatedBytes <= 0 {
		t.Fatalf("preview estimate = %d", prev.EstimatedBytes)
	}
	if _, ok := out["tools"]; !ok {
		t.Fatal("preview missing tools")
	}
}

func TestDetailedProgressPersistsAcrossReload(t *testing.T) {
	env := newQuantEnv(t, true)
	job, err := env.qm.Start(Request{Kind: KindQuantize, SourceModelID: env.model.ID, FType: "Q8_0"})
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	if err := env.qm.setState(done.ID, "running", "starting", 0, ""); err != nil {
		t.Fatal(err)
	}
	if err := env.qm.setStage(done.ID, "quantize", 0); err != nil {
		t.Fatal(err)
	}
	env.qm.emitProgressSample(done.ID, commandProgressSample{
		Current: 25, Total: 100, Progress: 0.25, ETASeconds: 300,
		Message: "[ 25/100] tensor", Estimated: true,
	})
	got, err := env.qm.Get(done.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.StageProgress != 0.25 || got.ProgressCurrent != 25 || got.ProgressTotal != 100 {
		t.Fatalf("persisted counters = %+v", got)
	}
	if got.StageETASeconds != 300 || got.ETASeconds <= got.StageETASeconds {
		t.Fatalf("persisted ETA = stage %d overall %d", got.StageETASeconds, got.ETASeconds)
	}
	if got.ProgressMessage != "[ 25/100] tensor" {
		t.Fatalf("persisted message = %q", got.ProgressMessage)
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

func TestIMatrixMixedCalibrationPass(t *testing.T) {
	env := newQuantEnv(t, true)
	job, err := env.qm.Start(Request{
		Kind: KindIMatrix, SourceModelID: env.model.ID, CalibrationPreset: "quick", KeepIMatrix: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	tail, _ := env.qm.LogTail(done.ID, 8192)
	if strings.Contains(tail, "-prose.gguf") || strings.Contains(tail, "combining") {
		t.Fatalf("domain-split combine should not run:\n%s", tail)
	}
	if !strings.Contains(tail, "all.txt") && !strings.Contains(tail, "mixed") {
		t.Fatalf("expected one mixed calibration file in log:\n%s", tail)
	}
	ims, err := env.qm.ListIMatrices(env.model.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(ims) == 0 {
		t.Fatal("expected generated imatrix")
	}
	if !strings.Contains(ims[0].DatasetLabel, "mixed") && !strings.Contains(ims[0].DatasetLabel, "chat") {
		t.Fatalf("label=%q", ims[0].DatasetLabel)
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

func TestResolveIMatrixRejectsDifferentSource(t *testing.T) {
	env := newQuantEnv(t, true)
	imatrixPath := filepath.Join(t.TempDir(), "matrix.gguf")
	if err := os.WriteFile(imatrixPath, []byte("matrix"), 0o644); err != nil {
		t.Fatal(err)
	}
	imatrix, err := env.qm.ImportIMatrix(env.model.ID, imatrixPath, "bound")
	if err != nil {
		t.Fatal(err)
	}
	otherPath := filepath.Join(t.TempDir(), "other.gguf")
	if err := copyFile(env.model.PrimaryPath, otherPath); err != nil {
		t.Fatal(err)
	}
	otherID, err := env.lib.ImportFile(otherPath)
	if err != nil {
		t.Fatal(err)
	}
	other, err := env.lib.Get(otherID)
	if err != nil {
		t.Fatal(err)
	}
	_, err = env.qm.resolveIMatrixPath(context.Background(), &Job{}, Request{IMatrixID: imatrix.ID}, other, runtimes.ToolsSnapshot{})
	if err == nil || !strings.Contains(err.Error(), "belongs to model") {
		t.Fatalf("cross-model imatrix reuse was not rejected: %v", err)
	}
}

func adaptiveRequest(modelID string) Request {
	return Request{
		Kind: KindAdaptiveQuantize, SourceModelID: modelID, GenerateIMatrix: true,
		CalibrationPreset: "quick", TargetBytes: 1 << 20,
	}
}

func resultMap(t *testing.T, j *Job) map[string]any {
	t.Helper()
	var result map[string]any
	if err := json.Unmarshal(j.Result, &result); err != nil {
		t.Fatal(err)
	}
	return result
}

func TestQuantlabAdaptiveFastJobCompletesAndAdopts(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Quantlab Fast")
	req := adaptiveRequest(model.ID)
	req.Effort = "fast"
	job, err := env.qm.Start(req)
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	result := resultMap(t, done)
	if ftype, _ := result["ftype"].(string); !strings.HasPrefix(ftype, "OID-") {
		t.Fatalf("ftype/result = %q / %#v", ftype, result)
	}
	if modelID, _ := result["model_id"].(string); modelID == "" || env.lib.IDForPath(done.DestPath) != modelID {
		t.Fatalf("model was not adopted: %#v", result)
	}
	if recipe, _ := result["recipe_path"].(string); recipe == "" {
		t.Fatalf("recipe sidecar missing from result: %#v", result)
	} else if _, err := os.Stat(recipe); err != nil {
		t.Fatal(err)
	}
	if report, _ := result["report_path"].(string); report == "" {
		t.Fatalf("final report sidecar missing from result: %#v", result)
	} else if _, err := os.Stat(report); err != nil {
		t.Fatal(err)
	} else {
		var reportData map[string]any
		data, err := os.ReadFile(report)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, &reportData); err != nil {
			t.Fatal(err)
		}
		output, _ := reportData["output"].(map[string]any)
		if got, _ := output["path"].(string); got != done.DestPath {
			t.Fatalf("adopted report output.path = %q, want %q", got, done.DestPath)
		}
	}
	quality, _ := result["quality_gate"].(map[string]any)
	if passed, _ := quality["passed"].(bool); !passed {
		t.Fatalf("quality gate failed: %#v", result)
	}
	tail, _ := env.qm.LogTail(done.ID, 16384)
	if !strings.Contains(tail, "ctx=512 effort=fast") {
		t.Fatalf("fast eval ctx not applied:\n%s", tail)
	}
}

func TestPublishQuantlabArtifactIsIdempotentAndKeepsCheckpoint(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "ql", "out", "emitted.gguf")
	writeQuantlabTestGGUF(t, source, "publication", []quantlabTestTensor{{"blk.0.ffn_down.weight", 1, []uint64{256, 64}}})
	dest := filepath.Join(dir, "models", "published.gguf")
	if err := publishQuantlabArtifact(source, dest); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(source); err != nil {
		t.Fatalf("checkpoint artifact was moved: %v", err)
	}
	if err := publishQuantlabArtifact(source, dest); err != nil {
		t.Fatalf("verified destination was not reused: %v", err)
	}
	if err := os.WriteFile(dest, []byte("untrusted"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := publishQuantlabArtifact(source, dest); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("mismatched destination accepted: %v", err)
	}
	if _, err := os.Stat(dest); !os.IsNotExist(err) {
		t.Fatalf("mismatched destination was retained: %v", err)
	}
}

func TestRecoverAfterPublicationCopyResumesWithoutDuplicate(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Publication Resume")
	const id = "publication-resume"
	root, outDir, workDir, stateDir, calibDir := env.qm.quantlabDirs(id)
	if err := os.MkdirAll(calibDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string]string{
		"manifest.json": "{\"version\":1}", "calibration.txt": "calibration text\n", "evaluation.txt": "evaluation text\n",
	} {
		if err := os.WriteFile(filepath.Join(calibDir, name), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	req := Request{Kind: KindAdaptiveQuantize, SourceModelID: model.ID, TargetBytes: 1 << 20, Effort: "fast"}
	rt, err := env.qm.resolveRuntime(req, model)
	if err != nil {
		t.Fatal(err)
	}
	tools, err := env.qm.toolsFor(rt)
	if err != nil {
		t.Fatal(err)
	}
	run, err := pipeline.Plan(pipeline.PlanOptions{
		SourcePath: model.PrimaryPath, OutputDir: outDir, WorkDir: workDir, StateDir: stateDir,
		CalibrationDir: calibDir, BudgetBytes: 1 << 20, LlamaQuantize: tools.Quantize.Path,
		LlamaPerplexity: tools.Perplexity.Path, LlamaImatrix: tools.IMatrix.Path,
		CtxSize: 512, Threads: 1, Effort: "fast", RunID: id, Stdout: io.Discard,
	})
	if err != nil {
		t.Fatal(err)
	}
	eng, err := pipeline.NewEngine(state.Store{Dir: stateDir}, run, orchestrate.OSRunner{Env: append(os.Environ(), runtimes.LibPathEnv(tools.Quantize.Path)...)}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if err := eng.Resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	emitted := run.Artifacts[core.StageEmit]
	if emitted == "" {
		t.Fatal("pipeline did not emit checkpoint artifact")
	}
	dest, err := env.qm.destFor(model, quantlabLabel(run.Manifest), "")
	if err != nil {
		t.Fatal(err)
	}
	if err := publishQuantlabArtifact(emitted, dest); err != nil {
		t.Fatal(err)
	}
	// This is the crash window: the job is still running with no result, while
	// ql/out and the verified destination both exist.
	raw, _ := json.Marshal(req)
	if _, err := env.qm.db.Exec(`INSERT INTO quant_jobs(id,kind,state,stage,progress,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`, id, KindAdaptiveQuantize, "running", "finalize", .95, rt.ID, model.ID, dest, 1, filepath.Join(env.layout.QuantLogs, id+".log"), string(raw), "{}", "", now(), now()); err != nil {
		t.Fatal(err)
	}
	if err := env.qm.RecoverAfterRestart(); err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, id, "complete")
	if got := env.lib.IDForPath(dest); got == "" {
		t.Fatal("published model was not adopted")
	}
	if resultMap(t, done)["dest_path"] != dest {
		t.Fatalf("result did not reuse published destination: %#v", resultMap(t, done))
	}
	if _, err := os.Stat(filepath.Join(root, "out", filepath.Base(emitted))); !os.IsNotExist(err) {
		t.Fatalf("checkpoint scratch was not cleaned after durable result: %v", err)
	}
}

func TestFromHFAdaptivePreviewPeakIsEffortAware(t *testing.T) {
	env := newQuantEnv(t, false)
	env.qm.SetProbeFn(func(context.Context, string) (*FromHFPreview, error) {
		return &FromHFPreview{ProbeResult: convert.ProbeResult{Compatible: true, SnapshotBytes: 100, EstimatedGGUFBytes: 1000}}, nil
	})
	fast, err := env.qm.ProbeFromHFForRequest(context.Background(), "org/model", Request{Kind: KindFromHF, Effort: "fast", TargetBPW: 4})
	if err != nil {
		t.Fatal(err)
	}
	profiled, err := env.qm.ProbeFromHFForRequest(context.Background(), "org/model", Request{Kind: KindFromHF, Effort: "profiled", TargetBPW: 4})
	if err != nil {
		t.Fatal(err)
	}
	deep, err := env.qm.ProbeFromHFForRequest(context.Background(), "org/model", Request{Kind: KindFromHF, Effort: "deep", TargetBPW: 4})
	if err != nil {
		t.Fatal(err)
	}
	if !(fast.DiskPeakBytes < profiled.DiskPeakBytes && profiled.DiskPeakBytes < deep.DiskPeakBytes) {
		t.Fatalf("adaptive preview peaks fast/profiled/deep = %d/%d/%d", fast.DiskPeakBytes, profiled.DiskPeakBytes, deep.DiskPeakBytes)
	}
	if fast.DiskPeakBytes < 256<<20+100+3*1000 {
		t.Fatalf("adaptive preview undercounts generated imatrix/source/candidate/final: %d", fast.DiskPeakBytes)
	}
}

func TestFromHFAdaptivePreview57GiBCase(t *testing.T) {
	env := newQuantEnv(t, false)
	const gib = int64(1) << 30
	env.qm.SetProbeFn(func(context.Context, string) (*FromHFPreview, error) {
		return &FromHFPreview{ProbeResult: convert.ProbeResult{
			Compatible: true, SnapshotBytes: 57 * gib, EstimatedGGUFBytes: 57 * gib,
		}}, nil
	})
	probe := func(effort string) int64 {
		out, err := env.qm.ProbeFromHFForRequest(context.Background(), "org/model",
			Request{Kind: KindFromHF, Effort: effort, QuantTier: QuantTierQ4, GenerateIMatrix: true})
		if err != nil {
			t.Fatal(err)
		}
		return out.DiskPeakBytes
	}
	fastPeak, profiledPeak, deepPeak := probe("fast"), probe("profiled"), probe("deep")
	if !(fastPeak < profiledPeak && profiledPeak < deepPeak) {
		t.Fatalf("monotonicity: %d/%d/%d", fastPeak, profiledPeak, deepPeak)
	}
	// The old coarse multiplier showed ~285 GiB for deep; the
	// streaming-variant-bank estimate must stay near the true scratch gate.
	if deepPeak >= 200*gib {
		t.Fatalf("deep preview regressed to coarse multiplier: %.1f GiB", float64(deepPeak)/float64(gib))
	}
	if fastPeak < 125*gib || fastPeak > 160*gib {
		t.Fatalf("fast preview out of expected band: %.1f GiB", float64(fastPeak)/float64(gib))
	}
	if deepPeak < 160*gib || deepPeak > 195*gib {
		t.Fatalf("deep preview out of expected band: %.1f GiB", float64(deepPeak)/float64(gib))
	}
	// Reusing an existing GGUF must drop the snapshot from the peak.
	env.qm.SetProbeFn(func(context.Context, string) (*FromHFPreview, error) {
		return &FromHFPreview{ProbeResult: convert.ProbeResult{
			Compatible: true, SnapshotBytes: 57 * gib, EstimatedGGUFBytes: 57 * gib,
		}, ReusedModelID: "exists"}, nil
	})
	reused, err := env.qm.ProbeFromHFForRequest(context.Background(), "org/model",
		Request{Kind: KindFromHF, Effort: "deep", QuantTier: QuantTierQ4})
	if err != nil {
		t.Fatal(err)
	}
	if reused.DiskPeakBytes+57*gib != deepPeak {
		t.Fatalf("reused-source preview must exclude the snapshot: %.1f vs %.1f GiB",
			float64(reused.DiskPeakBytes)/float64(gib), float64(deepPeak)/float64(gib))
	}
}

func TestQuantlabAdaptiveDefaultsToProfiled(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Quantlab Default")
	job, err := env.qm.Start(adaptiveRequest(model.ID))
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	if done.Request.Effort != "profiled" {
		t.Fatalf("persisted effort = %q", done.Request.Effort)
	}
	if effort, _ := resultMap(t, done)["effort"].(string); effort != "profiled" {
		t.Fatalf("result effort = %q", effort)
	}
}

func TestQuantlabAdaptiveDefaultsTargetBPW(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Quantlab Default Target")
	job, err := env.qm.Start(Request{Kind: KindAdaptiveQuantize, SourceModelID: model.ID, GenerateIMatrix: true, CalibrationPreset: "quick"})
	if err != nil {
		t.Fatal(err)
	}
	if job.Request.TargetBPW != defaultAdaptiveTargetBPW {
		t.Fatalf("persisted target_bpw = %v", job.Request.TargetBPW)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	if got, _ := resultMap(t, done)["target_bpw"].(float64); got != defaultAdaptiveTargetBPW {
		t.Fatalf("result target_bpw = %v", got)
	}
	tail, _ := env.qm.LogTail(done.ID, 16384)
	if !strings.Contains(tail, "ctx=2048 effort=profiled") {
		t.Fatalf("profiled eval ctx not applied:\n%s", tail)
	}
}

func TestFindReusableIMatrixMatchesPresetNotCustomPath(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Reuse Matrix")
	src := filepath.Join(t.TempDir(), "matrix.gguf")
	if err := os.WriteFile(src, []byte("matrix"), 0o644); err != nil {
		t.Fatal(err)
	}
	im, err := env.qm.ImportIMatrix(model.ID, src, "quick")
	if err != nil {
		t.Fatal(err)
	}
	got := env.qm.findReusableIMatrix(model, Request{CalibrationPreset: "quick"})
	if got == nil || got.ID != im.ID {
		t.Fatalf("quick reuse = %+v, want %s", got, im.ID)
	}
	if env.qm.findReusableIMatrix(model, Request{CalibrationPreset: "thorough"}) != nil {
		t.Fatal("thorough should not reuse a quick matrix")
	}
	if env.qm.findReusableIMatrix(model, Request{CalibrationPreset: "quick", CalibrationPath: "/tmp/custom.txt"}) != nil {
		t.Fatal("custom calibration path should not reuse")
	}
}

func TestQuantlabReusesIMatrixForSameSource(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Quantlab IMatrix Reuse")
	req := adaptiveRequest(model.ID)
	req.Effort = "fast"
	first, err := env.qm.Start(req)
	if err != nil {
		t.Fatal(err)
	}
	waitJob(t, env.qm, first.ID, "complete")
	ims, err := env.qm.ListIMatrices(model.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(ims) != 1 {
		t.Fatalf("after first job: %d imatrices", len(ims))
	}
	second, err := env.qm.Start(req)
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, second.ID, "complete")
	ims, err = env.qm.ListIMatrices(model.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(ims) != 1 {
		t.Fatalf("second job regenerated imatrix: %d rows", len(ims))
	}
	tail, _ := env.qm.LogTail(done.ID, 16384)
	if !strings.Contains(tail, "reusing importance matrix") {
		t.Fatalf("second job did not reuse imatrix:\n%s", tail)
	}
}

func TestQuantlabEffortScalesCalibrationPreset(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Quantlab Effort Calib")
	job, err := env.qm.Start(Request{
		Kind: KindAdaptiveQuantize, SourceModelID: model.ID, GenerateIMatrix: true,
		TargetBytes: 1 << 20, Effort: "fast",
	})
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	if done.Request.CalibrationPreset != "quick" {
		t.Fatalf("fast calibration preset = %q, want quick", done.Request.CalibrationPreset)
	}
	if !done.Request.ParseSpecial || !done.Request.ProcessOutput {
		t.Fatalf("adaptive imatrix flags parse_special=%v process_output=%v, want both true", done.Request.ParseSpecial, done.Request.ProcessOutput)
	}
}

func TestSeedAndAdoptQuantlabLossCache(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Loss Cache")
	work := t.TempDir()
	stateDir := t.TempDir()
	stable := env.qm.modelLossCachePath(model.ID)
	if err := os.MkdirAll(filepath.Dir(stable), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stable, []byte(`{"version":1,"modelID":"x"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	env.qm.seedQuantlabLossCache(model.ID, work, stateDir, "run1")
	if fileSize(filepath.Join(work, "loss-cache.json")) <= 0 {
		t.Fatal("workDir cache not seeded")
	}
	if fileSize(filepath.Join(stateDir, "run1.loss-cache.json")) <= 0 {
		t.Fatal("sidecar not seeded")
	}
	os.Remove(stable)
	if err := os.WriteFile(filepath.Join(stateDir, "run1.loss-cache.json"), []byte(`{"version":1,"modelID":"y"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	env.qm.adoptQuantlabLossCache(model, stateDir, "run1")
	got, err := os.ReadFile(stable)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(got), `"modelID":"y"`) {
		t.Fatalf("adopted cache = %s", got)
	}
}

func TestQuantlabDiskPreflightRejectsInsufficientScratch(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Quantlab Disk")
	env.qm.diskFree = func(path string) uint64 {
		if path == env.layout.QuantJobs {
			return 1
		}
		return 1 << 40
	}
	// Start must queue the job immediately without the synchronous
	// GGUF-parsing preflight; the preflight now runs in the execution
	// goroutine and transitions the job to failed on insufficient disk.
	job, err := env.qm.Start(adaptiveRequest(model.ID))
	if err != nil {
		t.Fatalf("Start returned error instead of queueing: %v", err)
	}
	if _, err := env.qm.Get(job.ID); err != nil {
		t.Fatalf("queued job not visible: %v", err)
	}
	done := waitJob(t, env.qm, job.ID, "failed")
	if !strings.Contains(done.Error, "Dynamic scratch") {
		t.Fatalf("disk preflight error = %q", done.Error)
	}
}

func TestQuantlabScratchEstimateUsesOutputBudgetForPublication(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Quantlab Estimate")
	const target = uint64(1 << 20)
	scratch, output, err := env.qm.quantlabScratchEstimate(model, Request{
		Kind: KindAdaptiveQuantize, SourceModelID: model.ID,
		TargetBytes: int64(target), Effort: "fast",
	})
	if err != nil {
		t.Fatal(err)
	}
	if scratch == 0 {
		t.Fatal("scratch estimate is zero")
	}
	if output != target {
		t.Fatalf("publication output reserve = %d, want target budget %d", output, target)
	}
}

func TestQuantlabCSKWorkingSetTracksAvailableMemory(t *testing.T) {
	if got := quantlabCSKWorkingSet(0); got != 1<<30 {
		t.Fatalf("unknown-memory limit = %d, want 1 GiB", got)
	}
	if got := quantlabCSKWorkingSet(4 << 30); got != 1<<30 {
		t.Fatalf("4 GiB host limit = %d, want 1 GiB", got)
	}
	if got := quantlabCSKWorkingSet(16 << 30); got != 4<<30 {
		t.Fatalf("16 GiB host limit = %d, want 4 GiB", got)
	}
	if got := quantlabCSKWorkingSet(128 << 30); got != 8<<30 {
		t.Fatalf("large host limit = %d, want 8 GiB cap", got)
	}
}

// TestQuantlabStartReturnsBeforeDiskPreflight verifies Start() returns
// quickly for a quantlab job without parsing the source GGUF: the
// scratch-disk preflight (which reaches diskFree(QuantJobs) only after
// parsing the source) is deferred into the execution goroutine, and the
// queued job is immediately visible via the jobs list GET.
func TestQuantlabStartReturnsBeforeDiskPreflight(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Quantlab Fast Start")
	release := make(chan struct{})
	called := make(chan struct{}, 1)
	env.qm.diskFree = func(path string) uint64 {
		if path == env.layout.QuantJobs {
			select {
			case called <- struct{}{}:
			default:
			}
			<-release
			return 1 << 40
		}
		return 1 << 40
	}
	type startRes struct {
		job *Job
		err error
	}
	resCh := make(chan startRes, 1)
	go func() {
		job, err := env.qm.Start(adaptiveRequest(model.ID))
		resCh <- startRes{job, err}
	}()
	var job *Job
	select {
	case r := <-resCh:
		if r.err != nil {
			close(release)
			t.Fatalf("Start: %v", r.err)
		}
		job = r.job
	case <-time.After(3 * time.Second):
		close(release)
		t.Fatal("Start() blocked on synchronous GGUF-parsing disk preflight")
	}
	jobs, err := env.qm.List()
	if err != nil {
		close(release)
		t.Fatalf("List: %v", err)
	}
	found := false
	for _, jj := range jobs {
		if jj.ID == job.ID {
			found = true
			break
		}
	}
	if !found {
		close(release)
		t.Fatal("queued job not visible in jobs list")
	}
	select {
	case <-called:
	case <-time.After(3 * time.Second):
		close(release)
		t.Fatal("disk preflight did not run in execution goroutine")
	}
	close(release)
	waitJob(t, env.qm, job.ID, "complete")
}

func TestQuantlabOutputValidationRejectsCorruptArtifact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corrupt.gguf")
	if err := os.WriteFile(path, []byte("not a gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := validateQuantlabOutput(path); err == nil {
		t.Fatal("corrupt emitted artifact was accepted")
	}
}

func TestScanAndResultRejectsMissingAdoption(t *testing.T) {
	env := newQuantEnv(t, true)
	if err := env.qm.scanAndResult("missing", filepath.Join(t.TempDir(), "outside.gguf"), Request{}, env.model, nil, "Q4_K_M"); err == nil || !strings.Contains(err.Error(), "not adopted") {
		t.Fatalf("missing adoption error = %v", err)
	}
}

func TestQuantlabAdaptiveModeAlias(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Quantlab Alias")
	req := adaptiveRequest(model.ID)
	req.AdaptiveMode = "fast"
	job, err := env.qm.Start(req)
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	if done.Request.Effort != "fast" {
		t.Fatalf("adaptive_mode alias was not resolved: %#v", done.Request)
	}
	if effort, _ := resultMap(t, done)["effort"].(string); effort != "fast" {
		t.Fatalf("result effort = %q", effort)
	}
}

func TestQuantlabAdaptivePresetTranslatesAndWarns(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "Quantlab Preset")
	req := adaptiveRequest(model.ID)
	req.TargetBytes = 0
	req.AdaptivePreset = "compact"
	req.Effort = "fast"
	job, err := env.qm.Start(req)
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	result := resultMap(t, done)
	if bpw, _ := result["target_bpw"].(float64); bpw != 3.8 {
		t.Fatalf("translated target_bpw = %v; result=%#v", bpw, result)
	}
	warnings, _ := result["warnings"].([]any)
	if len(warnings) == 0 || !strings.Contains(fmt.Sprint(warnings[0]), "deprecated") {
		t.Fatalf("preset warning missing: %#v", result)
	}
}

func TestQuantlabAdaptiveGateFailureStillAdopts(t *testing.T) {
	env := newQuantlabEnv(t)
	model := importQuantlabTestModel(t, env, "reject-quality")
	req := adaptiveRequest(model.ID)
	req.Effort = "deep"
	job, err := env.qm.Start(req)
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	result := resultMap(t, done)
	if pass, _ := result["gates_pass"].(bool); pass {
		t.Fatalf("gates_pass true despite high KLD: %#v", result)
	}
	quality, _ := result["quality_gate"].(map[string]any)
	if passed, _ := quality["passed"].(bool); passed || len(quality) == 0 {
		t.Fatalf("gate evidence missing: %#v", result)
	}
	if _, blocked := result["publication_blocked"]; blocked {
		t.Fatalf("publication was blocked: %#v", result)
	}
	warnings, _ := result["warnings"].([]any)
	if len(warnings) == 0 || !strings.Contains(fmt.Sprint(warnings), "Quality check exceeded") {
		t.Fatalf("quality warning missing: %#v", result)
	}
	if done.DestPath == "" {
		t.Fatal("gate miss did not publish artifact")
	}
	if _, err := os.Stat(done.DestPath); err != nil {
		t.Fatalf("published artifact missing: %v", err)
	}
	if modelID, _ := result["model_id"].(string); modelID == "" || env.lib.IDForPath(done.DestPath) != modelID {
		t.Fatalf("gate-miss artifact was not adopted: %#v", result)
	}
}

func TestQuantlabAdaptiveRepairsSSMConvSource(t *testing.T) {
	env := newQuantlabEnv(t)
	source := filepath.Join(t.TempDir(), "legacy-hybrid-F16.gguf")
	writeQuantlabTestGGUF(t, source, "Legacy Hybrid", []quantlabTestTensor{
		{"token_embd.weight", 1, []uint64{256, 64}},
		{"output.weight", 1, []uint64{256, 64}},
		{"blk.0.attn_gate.weight", 1, []uint64{256, 64}},
		{"blk.0.attn_qkv.weight", 1, []uint64{256, 64}},
		{"blk.0.ssm_alpha.weight", 1, []uint64{256, 64}},
		{"blk.0.ssm_beta.weight", 1, []uint64{256, 64}},
		{"blk.0.ssm_conv1d.weight", 1, []uint64{16}},
		{"blk.0.ssm_out.weight", 1, []uint64{256, 64}},
		{"blk.0.ffn_gate.weight", 1, []uint64{256, 64}},
		{"blk.0.ffn_up.weight", 1, []uint64{256, 64}},
		{"blk.0.ffn_down.weight", 1, []uint64{256, 64}},
	})
	id, err := env.lib.ImportFile(source)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := env.qm.Start(Request{Kind: KindQuantize, SourceModelID: id, FType: "Q4_K_M"}); err == nil || !strings.Contains(err.Error(), "ssm_conv1d") {
		t.Fatalf("ordinary quantization should reject legacy SSM source, got %v", err)
	}
	req := adaptiveRequest(id)
	req.Effort = "fast"
	job, err := env.qm.Start(req)
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	if repaired, _ := resultMap(t, done)["source_repaired"].(bool); !repaired {
		t.Fatal("result did not report source repair")
	}
	if issues, _, err := gguf.ValidateFile(done.DestPath); err != nil || len(issues) != 0 {
		t.Fatalf("quantized repaired output issues=%v err=%v", issues, err)
	}
	tensors, _, err := gguf.ListTensors(done.DestPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, tensor := range tensors {
		if tensor.Name == "blk.0.ssm_conv1d.weight" {
			if tensor.TypeName != "f32" {
				t.Fatalf("output SSM convolution type = %s, want f32", tensor.TypeName)
			}
			if _, err := os.Stat(filepath.Join(env.layout.QuantJobs, done.ID, "source-ssm-f32.gguf")); !os.IsNotExist(err) {
				t.Fatalf("temporary repaired source was not removed: %v", err)
			}
			if issues, _, err := gguf.ValidateFile(source); err != nil || !gguf.RepairableSSMConv1dIssues(source, issues) {
				t.Fatalf("original source was unexpectedly modified: issues=%v err=%v", issues, err)
			}
			return
		}
	}
	t.Fatal("output is missing SSM convolution tensor")
}

type quantlabStageSink struct {
	qm              *Manager
	cancelOnAnalyze bool
	pauseOnStage    string

	mu     sync.Mutex
	stages []string
	once   sync.Once
}

func (s *quantlabStageSink) Publish(event string, payload any) {
	p, ok := payload.(Progress)
	if !ok || event != "quant.progress" {
		return
	}
	s.mu.Lock()
	s.stages = append(s.stages, p.Stage)
	s.mu.Unlock()
	if s.cancelOnAnalyze && p.Stage == "analyze" {
		s.once.Do(func() { go func() { _ = s.qm.Cancel(p.ID) }() })
	}
	if s.pauseOnStage != "" && p.Stage == s.pauseOnStage {
		s.once.Do(func() { go func() { _ = s.qm.Pause(p.ID) }() })
	}
}

func (s *quantlabStageSink) orderedStages() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	wanted := map[string]bool{"analyze": true, "anchor": true, "solve": true, "quantize": true, "validate": true, "search": true, "finalize": true}
	seen := map[string]bool{}
	var out []string
	for _, stage := range s.stages {
		if wanted[stage] && !seen[stage] {
			seen[stage] = true
			out = append(out, stage)
		}
	}
	return out
}

func TestQuantlabAdaptiveCancellationLeavesNoPartialAdoption(t *testing.T) {
	env := newQuantlabEnv(t)
	sink := &quantlabStageSink{qm: env.qm, cancelOnAnalyze: true}
	env.qm.events = sink
	model := importQuantlabTestModel(t, env, "Quantlab Cancel")
	req := adaptiveRequest(model.ID)
	req.Effort = "profiled"
	job, err := env.qm.Start(req)
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "canceled")
	if done.DestPath != "" {
		if _, err := os.Stat(done.DestPath); !os.IsNotExist(err) {
			t.Fatalf("canceled artifact retained: %v", err)
		}
	}
	if _, err := os.Stat(filepath.Join(env.layout.QuantJobs, done.ID, quantlabDirName)); !os.IsNotExist(err) {
		t.Fatalf("quantlab scratch retained after cancellation: %v", err)
	}
}

func TestQuantlabAdaptiveProgressStagesAreOrdered(t *testing.T) {
	env := newQuantlabEnv(t)
	sink := &quantlabStageSink{qm: env.qm}
	env.qm.events = sink
	model := importQuantlabTestModel(t, env, "Quantlab Progress")
	req := adaptiveRequest(model.ID)
	req.Effort = "fast"
	job, err := env.qm.Start(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = waitJob(t, env.qm, job.ID, "complete")
	got := sink.orderedStages()
	want := []string{"analyze", "anchor", "solve", "quantize", "validate", "search", "finalize"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("stage order = %v, want %v", got, want)
	}
}

func TestRecoverAfterRestart(t *testing.T) {
	env := newQuantEnv(t, true)
	_, err := env.qm.db.Exec(`INSERT INTO quant_jobs(id,kind,state,stage,progress,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at)
		VALUES ('dead','quantize','running','quantize',0.2,'','','',1,'','{}','{}','','t','t')`)
	if err != nil {
		t.Fatal(err)
	}
	repairPath := filepath.Join(env.layout.QuantJobs, "dead", "source-ssm-f32.gguf")
	if err := os.MkdirAll(filepath.Dir(repairPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(repairPath, []byte("partial"), 0o644); err != nil {
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
	if _, err := os.Stat(repairPath); !os.IsNotExist(err) {
		t.Fatalf("stale repaired source was not removed: %v", err)
	}
}

func TestRecoverAfterRestartRequeuesQuantlabCheckpoint(t *testing.T) {
	env := newQuantEnv(t, true)
	const id = "resumable"
	stateDir := filepath.Join(env.layout.QuantJobs, id, quantlabDirName, "state")
	run, err := state.NewRun(id, time.Now(), dummyCheckpointConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	if err := (state.Store{Dir: stateDir}).Save(run); err != nil {
		t.Fatal(err)
	}
	req := Request{Kind: KindAdaptiveQuantize, SourceModelID: env.model.ID, TargetBPW: 3.83, Effort: "fast"}
	raw, _ := json.Marshal(req)
	if _, err := env.qm.db.Exec(`INSERT INTO quant_jobs(id,kind,state,stage,progress,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`, id, KindAdaptiveQuantize, "running", "validate", .75, "", env.model.ID, "", 1, "", string(raw), "{}", "", "t", "t"); err != nil {
		t.Fatal(err)
	}
	// Keep recovery observable without allowing the queued test fixture to run.
	env.qm.mu.Lock()
	env.qm.busy = true
	env.qm.mu.Unlock()
	defer func() {
		env.qm.mu.Lock()
		env.qm.busy = false
		env.qm.mu.Unlock()
	}()
	if err := env.qm.RecoverAfterRestart(); err != nil {
		t.Fatal(err)
	}
	got, err := env.qm.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if got.State != "queued" {
		t.Fatalf("resumable state = %s", got.State)
	}
	if _, err := os.Stat(stateDir); err != nil {
		t.Fatalf("resumable checkpoint was removed: %v", err)
	}
}

func TestPauseQueuedAndResumeCompletes(t *testing.T) {
	env := newQuantEnv(t, true)
	env.qm.mu.Lock()
	env.qm.busy = true
	env.qm.mu.Unlock()
	job, err := env.qm.Start(Request{
		Kind: KindQuantize, SourceModelID: env.model.ID, FType: "Q4_K_M", Threads: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := env.qm.Pause(job.ID); err != nil {
		t.Fatal(err)
	}
	paused, err := env.qm.Get(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if paused.State != "paused" {
		t.Fatalf("state=%s", paused.State)
	}
	if err := env.qm.Pause(job.ID); err != nil {
		t.Fatalf("pause of paused job: %v", err)
	}
	if err := env.qm.Resume(job.ID); err != nil {
		t.Fatal(err)
	}
	queued, err := env.qm.Get(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if queued.State != "queued" {
		t.Fatalf("resumed state=%s", queued.State)
	}
	env.qm.mu.Lock()
	env.qm.busy = false
	env.qm.mu.Unlock()
	env.qm.kick()
	done := waitJob(t, env.qm, job.ID, "complete")
	if done.DestPath == "" {
		t.Fatal("missing dest after resume")
	}
}

func TestCancelPausedJob(t *testing.T) {
	env := newQuantEnv(t, true)
	env.qm.mu.Lock()
	env.qm.busy = true
	env.qm.mu.Unlock()
	job, err := env.qm.Start(Request{
		Kind: KindQuantize, SourceModelID: env.model.ID, FType: "Q4_K_M", Threads: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := env.qm.Pause(job.ID); err != nil {
		t.Fatal(err)
	}
	paused := waitJob(t, env.qm, job.ID, "paused")
	if err := env.qm.Cancel(paused.ID); err != nil {
		t.Fatal(err)
	}
	got, err := env.qm.Get(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.State != "canceled" {
		t.Fatalf("state=%s", got.State)
	}
	if err := env.qm.Resume(job.ID); err == nil {
		t.Fatal("resume of canceled job should fail")
	}
}

func TestPauseRunningQuantlabKeepsCheckpoint(t *testing.T) {
	env := newQuantlabEnv(t)
	sink := &quantlabStageSink{qm: env.qm, pauseOnStage: "quantize"}
	env.qm.events = sink
	model := importQuantlabTestModel(t, env, "Quantlab Pause")
	req := adaptiveRequest(model.ID)
	req.Effort = "fast"
	job, err := env.qm.Start(req)
	if err != nil {
		t.Fatal(err)
	}
	paused := waitJob(t, env.qm, job.ID, "paused")
	qlDir := filepath.Join(env.layout.QuantJobs, paused.ID, quantlabDirName)
	if _, err := os.Stat(qlDir); err != nil {
		t.Fatalf("quantlab scratch removed on pause: %v", err)
	}
	if paused.PID != 0 {
		t.Fatalf("paused pid=%d, want 0", paused.PID)
	}
	if err := env.qm.Resume(paused.ID); err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, paused.ID, "complete")
	if done.DestPath == "" {
		t.Fatal("missing dest after resume")
	}
}

func TestPauseCompleteFails(t *testing.T) {
	env := newQuantEnv(t, true)
	job, err := env.qm.Start(Request{
		Kind: KindQuantize, SourceModelID: env.model.ID, FType: "Q4_K_M", Threads: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	if err := env.qm.Pause(done.ID); err == nil {
		t.Fatal("pause of complete job should fail")
	}
	if err := env.qm.Resume(done.ID); err == nil {
		t.Fatal("resume of complete job should fail")
	}
}

func TestResumeWhilePausingRejected(t *testing.T) {
	env := newQuantEnv(t, true)
	if _, err := env.qm.db.Exec(`INSERT INTO quant_jobs(id,kind,state,stage,progress,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at)
		VALUES ('p1','quantize','pausing','quantize',0.2,'','','',0,'','{}','{}','','t','t')`); err != nil {
		t.Fatal(err)
	}
	if err := env.qm.Resume("p1"); err == nil {
		t.Fatal("resume while pausing should fail")
	}
	if err := env.qm.Delete("p1"); err == nil {
		t.Fatal("delete of pausing job should fail")
	}
}

func TestDeletePausedJob(t *testing.T) {
	env := newQuantEnv(t, true)
	env.qm.mu.Lock()
	env.qm.busy = true
	env.qm.mu.Unlock()
	job, err := env.qm.Start(Request{
		Kind: KindQuantize, SourceModelID: env.model.ID, FType: "Q4_K_M", Threads: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := env.qm.Pause(job.ID); err != nil {
		t.Fatal(err)
	}
	paused := waitJob(t, env.qm, job.ID, "paused")
	if err := env.qm.Delete(paused.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := env.qm.Get(job.ID); err == nil {
		t.Fatal("deleted paused job should be gone")
	}
}

func TestRecoverAfterRestartLeavesPaused(t *testing.T) {
	env := newQuantEnv(t, true)
	const id = "held"
	if _, err := env.qm.db.Exec(`INSERT INTO quant_jobs(id,kind,state,stage,progress,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`, id, KindAdaptiveQuantize, "paused", "search", .5, "", env.model.ID, "", 0, "", `{"kind":"adaptive_quantize"}`, "{}", "", "t", "t"); err != nil {
		t.Fatal(err)
	}
	env.qm.mu.Lock()
	env.qm.busy = true
	env.qm.mu.Unlock()
	if err := env.qm.RecoverAfterRestart(); err != nil {
		t.Fatal(err)
	}
	got, err := env.qm.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if got.State != "paused" {
		t.Fatalf("paused job became %s", got.State)
	}
}

func TestRecoverAfterRestartPausingBecomesPaused(t *testing.T) {
	env := newQuantEnv(t, true)
	const id = "flushing"
	stateDir := filepath.Join(env.layout.QuantJobs, id, quantlabDirName, "state")
	run, err := state.NewRun(id, time.Now(), dummyCheckpointConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	if err := (state.Store{Dir: stateDir}).Save(run); err != nil {
		t.Fatal(err)
	}
	req := Request{Kind: KindAdaptiveQuantize, SourceModelID: env.model.ID, TargetBPW: 3.83, Effort: "fast"}
	raw, _ := json.Marshal(req)
	if _, err := env.qm.db.Exec(`INSERT INTO quant_jobs(id,kind,state,stage,progress,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`, id, KindAdaptiveQuantize, "pausing", "validate", .75, "", env.model.ID, "", 1, "", string(raw), "{}", "", "t", "t"); err != nil {
		t.Fatal(err)
	}
	repairPath := filepath.Join(env.layout.QuantJobs, id, "source-ssm-f32.gguf")
	if err := os.WriteFile(repairPath, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	env.qm.mu.Lock()
	env.qm.busy = true
	env.qm.mu.Unlock()
	if err := env.qm.RecoverAfterRestart(); err != nil {
		t.Fatal(err)
	}
	got, err := env.qm.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if got.State != "paused" {
		t.Fatalf("pausing job became %s, want paused", got.State)
	}
	if got.PID != 0 {
		t.Fatalf("pid=%d, want 0", got.PID)
	}
	if _, err := os.Stat(stateDir); err != nil {
		t.Fatalf("checkpoint was removed: %v", err)
	}
	if _, err := os.Stat(repairPath); err != nil {
		t.Fatalf("repaired source was removed: %v", err)
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
		       ('live','quantize','queued','',0,'','','',0,'','{}','{}','','t','t'),
		       ('hold','quantize','paused','quantize',0.4,'','','',0,'','{}','{}','','t','t')`); err != nil {
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
	if _, err := env.qm.Get("hold"); err != nil {
		t.Fatalf("paused job should remain: %v", err)
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

func TestFromHFJobConvertsThenQuantizes(t *testing.T) {
	env := newQuantEnv(t, true)
	repo := "test/tiny-glimmer"
	env.qm.SetProbeFn(func(ctx context.Context, r string) (*FromHFPreview, error) {
		return &FromHFPreview{
			ProbeResult: convert.ProbeResult{
				Compatible:         true,
				Architecture:       "muse-glimmer",
				Adapter:            "muse-glimmer",
				WeightDType:        "BF16",
				SnapshotBytes:      1000,
				EstimatedGGUFBytes: 1000,
				Files: []convert.NeededFile{
					{Path: "config.json", Size: 100},
					{Path: "tokenizer.json", Size: 100},
					{Path: "tokenizer_config.json", Size: 50},
					{Path: "model.safetensors", Size: 800},
				},
			},
			Repo: repo,
			SHA:  "abc123",
		}, nil
	})
	env.qm.SetSnapshotFn(func(ctx context.Context, destDir string, files []downloads.FileSpec) error {
		return convert.WriteTinyGlimmerSnapshot(destDir)
	})
	job, err := env.qm.Start(Request{
		Kind: KindFromHF, HFRepo: repo, FType: "Q4_K_M", Threads: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	done := waitJob(t, env.qm, job.ID, "complete")
	if done.DestPath == "" {
		t.Fatal("missing dest")
	}
	found := env.lib.HighPrecisionFromRepo(repo)
	if found == nil {
		t.Fatal("library missing converted high-precision GGUF")
	}
	if !HighPrecision(found.Quantization) {
		t.Fatalf("converted quant %s", found.Quantization)
	}
	if found.SourceRepo != repo {
		t.Fatalf("source_repo %q", found.SourceRepo)
	}
	if !strings.Contains(found.Alias, "tiny-glimmer") {
		t.Fatalf("converted alias %q should be the repo name, not author/name", found.Alias)
	}
}

func TestCanSkipFromHFProbe(t *testing.T) {
	dir := t.TempDir()
	layout := &config.Layout{QuantJobs: dir}
	id := "skip-probe"
	src := filepath.Join(dir, "src.gguf")
	eval := filepath.Join(dir, "eval.txt")
	if err := os.WriteFile(src, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(eval, []byte("text\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	req := Request{Kind: KindFromHF, SourceModelID: "model-1", Effort: "deep"}
	if canSkipFromHFProbe(layout, id, req) {
		t.Fatal("skipped probe without a checkpoint")
	}
	if canSkipFromHFProbe(layout, id, Request{Kind: KindFromHF, Effort: "deep"}) {
		t.Fatal("skipped probe without a source model")
	}
	if canSkipFromHFProbe(layout, id, Request{Kind: KindFromHF, SourceModelID: "model-1", FType: "Q4_K_M"}) {
		t.Fatal("skipped probe for a non-Dynamic from-HF job")
	}
	stateDir := filepath.Join(dir, id, quantlabDirName, "state")
	if err := os.MkdirAll(stateDir, 0o755); err != nil {
		t.Fatal(err)
	}
	run, err := state.NewRun(id, time.Now(), state.RunConfig{
		SourcePath:  src,
		OutputDir:   dir,
		WorkDir:     dir,
		Tools:       state.ToolPaths{LlamaQuantize: src, LlamaPerplexity: src},
		EvalCorpus:  eval,
		BudgetBytes: 1024,
		Threads:     1,
		CtxSize:     512,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := (state.Store{Dir: stateDir}).Save(run); err != nil {
		t.Fatal(err)
	}
	if !canSkipFromHFProbe(layout, id, req) {
		t.Fatal("did not skip probe with a valid Dynamic checkpoint")
	}
}

func writeVocabLayoutGGUF(t *testing.T, dir, name string, tokens []string, embd, vocabRows uint64) string {
	t.Helper()
	var b bytes.Buffer
	w32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	w64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	wstr := func(s string) { w64(uint64(len(s))); b.WriteString(s) }
	const (
		tUint32 = 4
		tString = 8
		tArray  = 9
	)
	binary.Write(&b, binary.LittleEndian, uint32(0x46554747))
	w32(3)
	w64(1)
	w64(3)
	wstr("general.architecture")
	w32(tString)
	wstr("llama")
	wstr("llama.embedding_length")
	w32(tUint32)
	w32(uint32(embd))
	wstr("tokenizer.ggml.tokens")
	w32(tArray)
	w32(tString)
	w64(uint64(len(tokens)))
	for _, s := range tokens {
		wstr(s)
	}
	wstr("token_embd.weight")
	w32(2)
	w64(embd)
	w64(vocabRows)
	w32(0)
	w64(0)
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestDestForConvertOverwritesInconsistentVocab(t *testing.T) {
	env := newQuantEnv(t, false)
	repo := "org/padded-embed"
	dest, err := env.qm.destForConvert(repo, "BF16")
	if err != nil {
		t.Fatal(err)
	}
	mismatch := writeVocabLayoutGGUF(t, t.TempDir(), "bad.gguf", []string{"a", "b", "c", "d"}, 8, 6)
	if err := copyFile(mismatch, dest); err != nil {
		t.Fatal(err)
	}
	again, err := env.qm.destForConvert(repo, "BF16")
	if err != nil {
		t.Fatal(err)
	}
	if again != dest {
		t.Fatalf("inconsistent GGUF should be overwritten in place, got %s want %s", again, dest)
	}

	aligned := writeVocabLayoutGGUF(t, t.TempDir(), "ok.gguf", []string{"a", "b", "c", "d"}, 8, 4)
	if err := copyFile(aligned, dest); err != nil {
		t.Fatal(err)
	}
	fresh, err := env.qm.destForConvert(repo, "BF16")
	if err != nil {
		t.Fatal(err)
	}
	if fresh == dest {
		t.Fatal("consistent GGUF must not be clobbered")
	}
}

func TestSupportsQuantlabEvaluationRequiresKLDivergence(t *testing.T) {
	tool := runtimes.ToolInfo{
		Present: true,
		Path:    "/bin/llama-perplexity",
		Flags:   []string{"--model", "--file", "--kl-divergence-base", "--chunks"},
	}
	if supportsQuantlabEvaluation(tool) {
		t.Fatal("accepted without --kl-divergence")
	}
	tool.Flags = append(tool.Flags, "--kl-divergence")
	if !supportsQuantlabEvaluation(tool) {
		t.Fatal("rejected with --kl-divergence")
	}
}
