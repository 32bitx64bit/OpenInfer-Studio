package quantize

import (
	"bufio"
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/diagnostics"
	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/hardware"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/processes"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
	"github.com/openinfer/openinfer-studio/internal/storage"
)

// EventSink fans progress to the control-API hub.
type EventSink interface {
	Publish(event string, payload any)
}

// Stopper unloads a running instance.
type Stopper interface {
	Stop(modelID string, force bool) error
}

// Job is the API view of one quantization / imatrix job.
type Job struct {
	ID            string          `json:"id"`
	Kind          string          `json:"kind"`
	State         string          `json:"state"`
	Stage         string          `json:"stage"`
	Progress      float64         `json:"progress"`
	RuntimeID     string          `json:"runtime_id"`
	SourceModelID string          `json:"source_model_id"`
	DestPath      string          `json:"dest_path"`
	PID           int             `json:"pid"`
	LogPath       string          `json:"log_path"`
	Request       Request         `json:"request"`
	Result        json.RawMessage `json:"result"`
	Error         string          `json:"error"`
	CreatedAt     string          `json:"created_at"`
	UpdatedAt     string          `json:"updated_at"`
	FinishedAt    string          `json:"finished_at"`
}

// Manager persists and runs quantization jobs.
type Manager struct {
	db          *sql.DB
	layout      *config.Layout
	rt          *runtimes.Manager
	lib         *models.Library
	loadedIDsFn func() []string
	stopper     Stopper
	events      EventSink
	log         *slog.Logger
	diskFree    func(string) uint64
	hardware    func() *hardware.Info

	mu        sync.Mutex
	busy      bool
	current   *processes.Handle
	currentID string
	cancel    context.CancelFunc
}

func NewManager(db *sql.DB, layout *config.Layout, rt *runtimes.Manager, lib *models.Library,
	events EventSink, log *slog.Logger, diskFree func(string) uint64, hw func() *hardware.Info) *Manager {
	if log == nil {
		log = slog.Default()
	}
	return &Manager{db: db, layout: layout, rt: rt, lib: lib, events: events, log: log, diskFree: diskFree, hardware: hw}
}

func (m *Manager) SetInstanceHooks(loadedIDs func() []string, stop Stopper) {
	m.loadedIDsFn = loadedIDs
	m.stopper = stop
}

func now() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// RecoverAfterRestart fails jobs that were running when the process died.
func (m *Manager) RecoverAfterRestart() error {
	_, err := m.db.Exec(`UPDATE quant_jobs SET state='failed', error=?, updated_at=?, finished_at=?
		WHERE state IN ('running','canceling')`,
		"interrupted by application restart", now(), now())
	if err != nil {
		return err
	}
	m.kick()
	return nil
}

func (m *Manager) EnsureCalibration() (string, error) {
	dest := filepath.Join(m.layout.QuantCalibration, "default.txt")
	want := defaultCalibrationBytes()
	if b, err := os.ReadFile(dest); err == nil && bytes.Equal(b, want) {
		return dest, nil
	}
	if err := os.MkdirAll(m.layout.QuantCalibration, 0o755); err != nil {
		return "", err
	}
	return dest, os.WriteFile(dest, want, 0o644)
}

func (m *Manager) resolveRuntime(req Request, src *models.Model) (*runtimes.Runtime, error) {
	id := req.RuntimeID
	if id == "" && src != nil {
		id = src.PinnedRuntime
	}
	if id != "" {
		return m.rt.Get(id)
	}
	pref, err := m.rt.Preferred()
	if err != nil {
		return nil, err
	}
	if pref == nil {
		return nil, fmt.Errorf("no llama.cpp runtime installed")
	}
	return pref, nil
}

func (m *Manager) toolsFor(rt *runtimes.Runtime) (runtimes.ToolsSnapshot, error) {
	return m.rt.Tools(rt.ID)
}

// Preview assembles a cheap size/fit estimate (no llama-quantize --dry-run).
func (m *Manager) Preview(req Request) (map[string]any, error) {
	src, err := m.lib.Get(req.SourceModelID)
	if err != nil {
		return nil, err
	}
	rt, err := m.resolveRuntime(req, src)
	if err != nil {
		return nil, err
	}
	types, tools, err := m.Types(rt.ID)
	if err != nil {
		return nil, err
	}
	ftypeName := CanonicalFType(req.FType)
	if ftypeName == "" {
		ftypeName = "Q4_K_M"
	}
	ft, ok := LookupFType(ftypeName)
	if !ok {
		ft = FType{Name: ftypeName, BPW: 4.89, Band: BandBalanced, IMatrix: IMatrixRecommended, Advertised: true}
	}
	for _, t := range types {
		if strings.EqualFold(t.Name, ftypeName) {
			ft = t
			break
		}
	}
	var hw *hardware.Info
	if m.hardware != nil {
		hw = m.hardware()
	}
	lib, _ := m.lib.List()
	if lib == nil {
		lib = []models.Model{}
	}
	var draft *models.Model
	if req.DraftModelID != "" {
		if d, err := m.lib.Get(req.DraftModelID); err == nil {
			draft = d
		}
	}
	rec := Recommend(*src, hw, types, draft)
	if req.FType == "" {
		ft, _ = LookupFType(rec.FType)
		ftypeName = rec.FType
	}
	var draftBytes int64
	if draft != nil {
		draftBytes = draft.SizeBytes
	}
	disk := uint64(0)
	if m.diskFree != nil {
		disk = m.diskFree(m.layout.Models)
	}
	prev := BuildPreview(EstimateInput{
		Source: *src, FType: ft, Hardware: hw, DiskFree: disk,
		DraftBytes: draftBytes, Context: src.ContextLength,
	}, rec)
	if models.IsSpeculativeDraft(*src) {
		prev.Warnings = append(prev.Warnings, "This is a speculative assistant/draft GGUF. llama-imatrix cannot initialize DFlash/EAGLE/MTP without the main model, so quantization will skip the importance matrix.")
		if RequiresIMatrix(ft.Name) && req.IMatrixID == "" {
			prev.Blockers = append(prev.Blockers, ft.Name+" needs an importance matrix, which cannot be built for this draft. Use Q4_K_M, Q5_K_M, or Q8_0.")
		}
	}
	comps := ListCompanions(*src, lib, ftypeName)
	loaded := m.loadedIDs()
	out := map[string]any{
		"preview":       prev,
		"companions":    comps,
		"types":         types,
		"tools":         tools,
		"runtime_id":    rt.ID,
		"loaded_models": loaded,
		"split_source":  isSplitPath(src.PrimaryPath),
	}
	return out, nil
}

func advertisedAll() []FType {
	var out []FType
	for _, t := range knownFTypes {
		if t.AliasOf != "" {
			continue
		}
		t.Advertised = true
		out = append(out, t)
	}
	return out
}

func (m *Manager) loadedIDs() []string {
	if m.loadedIDsFn == nil {
		return []string{}
	}
	ids := m.loadedIDsFn()
	if ids == nil {
		return []string{}
	}
	return ids
}

func (m *Manager) Types(runtimeID string) ([]FType, runtimes.ToolsSnapshot, error) {
	var rt *runtimes.Runtime
	var err error
	if runtimeID != "" {
		rt, err = m.rt.Get(runtimeID)
	} else {
		rt, err = m.rt.Preferred()
	}
	if err != nil {
		return nil, runtimes.ToolsSnapshot{}, err
	}
	if rt == nil {
		return nil, runtimes.ToolsSnapshot{}, fmt.Errorf("no llama.cpp runtime installed")
	}
	tools, err := m.toolsFor(rt)
	if err != nil {
		return nil, tools, err
	}
	var types []FType
	if tools.Quantize.Present && tools.Quantize.Path != "" {
		if help, herr := runtimes.ProbeHelp(tools.Quantize.Path); herr == nil {
			types = ParseQuantizeTypes(help)
		}
	}
	if len(types) == 0 && tools.Quantize.Present {
		types = advertisedAll()
	}
	return types, tools, nil
}

// Start validates and queues a job. Returns the job id.
func (m *Manager) Start(req Request) (*Job, error) {
	req.Kind = normalizeKind(req.Kind)
	src, err := m.lib.Get(req.SourceModelID)
	if err != nil && req.Kind != KindCombineIMatrix {
		return nil, err
	}
	rt, err := m.resolveRuntime(req, src)
	if err != nil {
		return nil, err
	}
	tools, err := m.toolsFor(rt)
	if err != nil {
		return nil, err
	}
	if req.Kind != KindCombineIMatrix && !tools.Quantize.Present && (req.Kind == KindQuantize || req.Kind == KindAdaptiveQuantize) {
		return nil, fmt.Errorf("runtime %s has no llama-quantize next to llama-server; install an official llama.cpp build", rt.ID)
	}
	if src != nil && models.IsSpeculativeDraft(*src) {
		if req.Kind == KindIMatrix {
			return nil, fmt.Errorf("llama-imatrix cannot load a speculative assistant GGUF (%s) without the main model context", filepath.Base(src.PrimaryPath))
		}
		// Quantize the assistant; skip imatrix (DFlash/EAGLE need ctx_other).
		req.GenerateIMatrix = false
	}
	if (req.Kind == KindIMatrix || req.GenerateIMatrix || req.Kind == KindCombineIMatrix) && !tools.IMatrix.Present {
		return nil, fmt.Errorf("runtime %s has no llama-imatrix next to llama-server", rt.ID)
	}
	if src != nil && !HighPrecision(src.Quantization) && (req.Kind == KindQuantize || req.Kind == KindAdaptiveQuantize) {
		if !req.AllowRequantize || !req.AcknowledgeRequantize {
			return nil, fmt.Errorf("source is already quantized (%s); set allow_requantize and acknowledge_requantize to continue", src.Quantization)
		}
	}
	ftName := CanonicalFType(req.FType)
	if ft, ok := LookupFType(ftName); ok && ft.Experimental && !req.AcknowledgeExperimental {
		return nil, fmt.Errorf("%s is experimental; set acknowledge_experimental to continue", ftName)
	}
	if RequiresIMatrix(ftName) && req.IMatrixID == "" && !req.GenerateIMatrix && req.Kind != KindIMatrix && req.Kind != KindCombineIMatrix {
		return nil, fmt.Errorf("%s requires an importance matrix (generate_imatrix or imatrix_id)", ftName)
	}
	if src != nil {
		if issues, _, verr := gguf.ValidateFile(src.PrimaryPath); verr != nil {
			return nil, fmt.Errorf("source GGUF: %w", verr)
		} else if len(issues) > 0 {
			return nil, fmt.Errorf("source GGUF failed validation: %s", issues[0])
		}
		if m.diskFree != nil && (req.Kind == KindQuantize || req.Kind == KindAdaptiveQuantize) {
			ft, _ := LookupFType(CanonicalFType(req.FType))
			if ft.Name == "" {
				ft, _ = LookupFType("Q4_K_M")
			}
			est := EstimateSize(*src, ft)
			need := uint64(est) + uint64(est)/10
			if free := m.diskFree(m.layout.Models); free > 0 && need > free {
				return nil, fmt.Errorf("not enough free disk space for the estimated output")
			}
		}
	}

	id := uuid.NewString()
	jobDir := filepath.Join(m.layout.QuantJobs, id)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		return nil, err
	}
	logPath := filepath.Join(m.layout.QuantLogs, id+".log")
	_ = os.MkdirAll(m.layout.QuantLogs, 0o755)
	body, _ := json.Marshal(req)
	ts := now()
	_, err = m.db.Exec(`INSERT INTO quant_jobs(id,kind,state,stage,progress,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		id, req.Kind, "queued", "", 0, rt.ID, req.SourceModelID, "", 0, logPath, string(body), "{}", "", ts, ts)
	if err != nil {
		return nil, err
	}
	m.kick()
	return m.Get(id)
}

func (m *Manager) List() ([]Job, error) {
	rows, err := m.db.Query(`SELECT id FROM quant_jobs ORDER BY created_at DESC LIMIT 100`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if rows.Scan(&id) == nil {
			ids = append(ids, id)
		}
	}
	out := make([]Job, 0, len(ids))
	for _, id := range ids {
		j, err := m.Get(id)
		if err == nil {
			out = append(out, *j)
		}
	}
	return out, nil
}

func (m *Manager) Get(id string) (*Job, error) {
	var j Job
	var reqJSON, resJSON string
	err := m.db.QueryRow(`SELECT id,kind,state,stage,progress,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at,finished_at
		FROM quant_jobs WHERE id = ?`, id).Scan(
		&j.ID, &j.Kind, &j.State, &j.Stage, &j.Progress, &j.RuntimeID, &j.SourceModelID, &j.DestPath,
		&j.PID, &j.LogPath, &reqJSON, &resJSON, &j.Error, &j.CreatedAt, &j.UpdatedAt, &j.FinishedAt)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("job %s not found", id)
	}
	if err != nil {
		return nil, err
	}
	_ = json.Unmarshal([]byte(reqJSON), &j.Request)
	j.Result = json.RawMessage(resJSON)
	return &j, nil
}

func (m *Manager) Cancel(id string) error {
	j, err := m.Get(id)
	if err != nil {
		return err
	}
	switch j.State {
	case "queued":
		return m.setState(id, "canceled", j.Stage, j.Progress, "canceled")
	case "running", "canceling":
		m.mu.Lock()
		if m.currentID == id && m.cancel != nil {
			m.cancel()
		}
		if m.currentID == id && m.current != nil {
			_ = m.current.KillTree()
		}
		m.mu.Unlock()
		return m.setState(id, "canceling", j.Stage, j.Progress, "")
	default:
		return fmt.Errorf("job is %s", j.State)
	}
}

// Delete removes a finished (or still-queued) job from history. Running
// jobs must be canceled first. Output GGUFs in the library are left in place.
func (m *Manager) Delete(id string) error {
	j, err := m.Get(id)
	if err != nil {
		return err
	}
	switch j.State {
	case "running", "canceling":
		return fmt.Errorf("job is %s; cancel it first", j.State)
	}
	return m.removeJobs([]Job{*j})
}

// ClearHistory deletes complete/failed/canceled jobs. Queued and running
// jobs are left untouched. Quantized models already in the library stay.
func (m *Manager) ClearHistory() (int, error) {
	all, err := m.List()
	if err != nil {
		return 0, err
	}
	var done []Job
	for _, j := range all {
		switch j.State {
		case "complete", "failed", "canceled":
			done = append(done, j)
		}
	}
	if err := m.removeJobs(done); err != nil {
		return 0, err
	}
	return len(done), nil
}

func (m *Manager) removeJobs(jobs []Job) error {
	if len(jobs) == 0 {
		return nil
	}
	tx, err := m.db.Begin()
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	for _, j := range jobs {
		if _, err := tx.Exec(`DELETE FROM quant_jobs WHERE id=?`, j.ID); err != nil {
			return err
		}
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	for _, j := range jobs {
		if j.LogPath != "" {
			_ = os.Remove(j.LogPath)
		}
		if m.layout != nil && j.ID != "" {
			_ = os.RemoveAll(filepath.Join(m.layout.QuantJobs, j.ID))
		}
	}
	return nil
}

func (m *Manager) LogTail(id string, max int) (string, error) {
	j, err := m.Get(id)
	if err != nil {
		return "", err
	}
	b, err := os.ReadFile(j.LogPath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	s := diagnostics.Redact(string(b))
	if max > 0 && len(s) > max {
		s = lastNRunes(s, max)
	}
	return s, nil
}

func (m *Manager) kick() {
	m.mu.Lock()
	if m.busy {
		m.mu.Unlock()
		return
	}
	m.busy = true
	m.mu.Unlock()
	go func() {
		defer func() {
			m.mu.Lock()
			m.busy = false
			m.mu.Unlock()
			// Try the next queued job without holding busy across the call.
			var n int
			_ = m.db.QueryRow(`SELECT COUNT(1) FROM quant_jobs WHERE state='queued'`).Scan(&n)
			if n > 0 {
				m.kick()
			}
		}()
		id := m.nextQueued()
		if id == "" {
			return
		}
		m.execute(id)
	}()
}

func (m *Manager) nextQueued() string {
	var id string
	_ = m.db.QueryRow(`SELECT id FROM quant_jobs WHERE state='queued' ORDER BY created_at LIMIT 1`).Scan(&id)
	return id
}

func (m *Manager) execute(id string) {
	j, err := m.Get(id)
	if err != nil {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	m.mu.Lock()
	m.cancel = cancel
	m.mu.Unlock()
	defer cancel()

	_ = m.setState(id, "running", "starting", 0, "")
	m.publishState(id, "running", "")
	err = m.run(ctx, j)
	if errors.Is(err, context.Canceled) {
		m.cleanupPartial(j)
		_ = m.setState(id, "canceled", j.Stage, j.Progress, "canceled")
		m.publishState(id, "canceled", "canceled")
		return
	}
	if err != nil {
		m.cleanupPartial(j)
		msg := err.Error()
		_ = m.setState(id, "failed", j.Stage, j.Progress, msg)
		m.publishState(id, "failed", msg)
		return
	}
	_ = m.setState(id, "complete", "done", 1, "")
	m.publishState(id, "complete", "")
}

func (m *Manager) cleanupPartial(j *Job) {
	if j == nil {
		return
	}
	cur, err := m.Get(j.ID)
	if err != nil {
		return
	}
	_ = os.RemoveAll(filepath.Join(m.layout.QuantJobs, j.ID, "partial.gguf"))
	if cur.State != "complete" && cur.DestPath != "" {
		if _, _, err := gguf.ValidateFile(cur.DestPath); err != nil {
			_ = os.Remove(cur.DestPath)
		}
	}
}

func (m *Manager) run(ctx context.Context, j *Job) error {
	req := j.Request
	src, err := m.lib.Get(req.SourceModelID)
	if err != nil && req.Kind != KindCombineIMatrix {
		return err
	}
	rt, err := m.resolveRuntime(req, src)
	if err != nil {
		return err
	}
	tools, err := m.toolsFor(rt)
	if err != nil {
		return err
	}
	if req.UnloadFirst && m.stopper != nil {
		for _, id := range m.loadedIDs() {
			_ = m.stopper.Stop(id, false)
		}
	}

	switch req.Kind {
	case KindIMatrix:
		return m.runIMatrixOnly(ctx, j, req, src, tools)
	case KindCombineIMatrix:
		return m.runCombine(ctx, j, req, tools)
	case KindAdaptiveQuantize:
		return m.runAdaptive(ctx, j, req, src, rt, tools)
	default:
		return m.runQuantize(ctx, j, req, src, rt, tools)
	}
}

func (m *Manager) runIMatrixOnly(ctx context.Context, j *Job, req Request, src *models.Model, tools runtimes.ToolsSnapshot) error {
	path, err := m.generateIMatrix(ctx, j, req, src, tools)
	if err != nil {
		return err
	}
	return m.setResult(j.ID, map[string]any{"imatrix_path": path})
}

func (m *Manager) runCombine(ctx context.Context, j *Job, req Request, tools runtimes.ToolsSnapshot) error {
	_ = m.setStage(j.ID, "combine_imatrix", 0)
	var inputs []string
	for _, id := range req.CombineIMatrixIDs {
		im, err := m.getIMatrix(id)
		if err != nil {
			return err
		}
		inputs = append(inputs, im.Path)
	}
	out := filepath.Join(m.layout.QuantIMatrices, uuid.NewString()+".gguf")
	args, err := PlanCombine(tools.IMatrix.Flags, inputs, out)
	if err != nil {
		return err
	}
	if err := m.runCmd(ctx, j, tools.IMatrix.Path, args); err != nil {
		return fmt.Errorf("llama-imatrix combine: %w", err)
	}
	label := "combined"
	modelID := req.SourceModelID
	im, err := m.recordGeneratedIMatrix(modelID, out, label, 0, "combined")
	if err != nil {
		return err
	}
	return m.setResult(j.ID, map[string]any{"imatrix_id": im.ID, "imatrix_path": out})
}

func (m *Manager) generateIMatrix(ctx context.Context, j *Job, req Request, src *models.Model, tools runtimes.ToolsSnapshot) (string, error) {
	if src == nil {
		return "", fmt.Errorf("imatrix requires a source model")
	}
	if models.IsSpeculativeDraft(*src) {
		return "", fmt.Errorf("llama-imatrix cannot load a speculative assistant GGUF (%s) without the main model context", filepath.Base(src.PrimaryPath))
	}
	_ = m.setStage(j.ID, "imatrix", 0)
	cal := req.CalibrationPath
	if cal == "" {
		var err error
		cal, err = m.EnsureCalibration()
		if err != nil {
			return "", err
		}
	}
	out := filepath.Join(m.layout.QuantIMatrices, j.ID+".gguf")
	args, err := PlanIMatrix(req, tools.IMatrix.Flags, src.PrimaryPath, cal, out)
	if err != nil {
		return "", err
	}
	if err := m.runCmd(ctx, j, tools.IMatrix.Path, args); err != nil {
		return "", fmt.Errorf("llama-imatrix: %w", err)
	}
	label := req.CalibrationPreset
	if label == "" {
		label = filepath.Base(cal)
	}
	chunks := req.Chunks
	if chunks == 0 {
		chunks = presetChunks(req.CalibrationPreset)
	}
	if _, err := m.recordGeneratedIMatrix(src.ID, out, label, chunks, "generated"); err != nil {
		return "", err
	}
	return out, nil
}

func (m *Manager) imatrixStats(exe, path string, flags []string) map[string]float64 {
	if exe == "" || path == "" || !flagOK(flags, "--show-statistics") {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	args := []string{}
	if flagOK(flags, "--in-file") {
		args = append(args, "--in-file", path)
	}
	args = append(args, "--show-statistics")
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Dir = filepath.Dir(exe)
	cmd.Env = append(os.Environ(), runtimes.LibPathEnv(exe)...)
	out, _ := cmd.CombinedOutput()
	return ParseIMatrixStats(string(out))
}

func (m *Manager) resolveIMatrixPath(ctx context.Context, j *Job, req Request, src *models.Model, tools runtimes.ToolsSnapshot) (string, error) {
	if req.IMatrixID != "" {
		im, err := m.getIMatrix(req.IMatrixID)
		if err != nil {
			return "", err
		}
		return im.Path, nil
	}
	if req.GenerateIMatrix {
		if src != nil && models.IsSpeculativeDraft(*src) {
			if j.LogPath != "" {
				if f, err := os.OpenFile(j.LogPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644); err == nil {
					fmt.Fprintf(f, "skipping llama-imatrix: %s is a speculative assistant/draft GGUF (needs main-model context)\n", filepath.Base(src.PrimaryPath))
					f.Close()
				}
			}
			return "", nil
		}
		return m.generateIMatrix(ctx, j, req, src, tools)
	}
	return "", nil
}

func (m *Manager) destFor(src *models.Model, ftype, outputName string) (string, error) {
	base := outputName
	if base == "" {
		base = models.QuantizedAlias(src.Alias, ftype)
		if p := sidecarFilePrefix(src.PrimaryPath); p != "" &&
			!strings.HasPrefix(strings.ToLower(base), strings.ToLower(p)) {
			base = p + base
		}
		base = safeName(base)
	} else {
		base = safeName(base)
	}
	dir := filepath.Join(m.layout.Models, "local--"+base, "files")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	dest := filepath.Join(dir, base+".gguf")
	if _, err := os.Stat(dest); err == nil {
		dir = filepath.Join(m.layout.Models, "local--"+base+"-"+uuid.NewString()[:8], "files")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return "", err
		}
		dest = filepath.Join(dir, base+".gguf")
	}
	if _, err := storage.ValidateInside(m.layout.Models, dest); err != nil {
		return "", err
	}
	return dest, nil
}

func (m *Manager) runQuantize(ctx context.Context, j *Job, req Request, src *models.Model, rt *runtimes.Runtime, tools runtimes.ToolsSnapshot) error {
	imatrixPath, err := m.resolveIMatrixPath(ctx, j, req, src, tools)
	if err != nil {
		return err
	}
	ftype := CanonicalFType(req.FType)
	if ftype == "" {
		ftype = "Q4_K_M"
	}
	if isSplitPath(src.PrimaryPath) && !req.KeepSplit {
		req.KeepSplit = flagOK(tools.Quantize.Flags, "--keep-split")
	}
	if needsRequantizeFlag(src.Quantization) {
		req.AllowRequantize = true
	}
	dest, err := m.destFor(src, ftype, req.OutputName)
	if err != nil {
		return err
	}
	_, _ = m.db.Exec(`UPDATE quant_jobs SET dest_path=?, updated_at=? WHERE id=?`, dest, now(), j.ID)
	_ = m.setStage(j.ID, "quantize", 0.05)
	args, err := PlanQuantize(req, tools.Quantize.Flags, src.PrimaryPath, dest, imatrixPath, req.TensorTypeFile)
	if err != nil {
		return err
	}
	if err := m.runCmd(ctx, j, tools.Quantize.Path, args); err != nil {
		_ = os.Remove(dest)
		return fmt.Errorf("llama-quantize: %w", err)
	}
	extras, err := m.afterQuantize(ctx, j, req, src, dest, tools, rt)
	if err != nil {
		return err
	}
	return m.scanAndResult(j.ID, dest, req, src, extras)
}

func (m *Manager) runAdaptive(ctx context.Context, j *Job, req Request, src *models.Model, rt *runtimes.Runtime, tools runtimes.ToolsSnapshot) error {
	imatrixPath, err := m.resolveIMatrixPath(ctx, j, req, src, tools)
	if err != nil {
		return err
	}
	_ = m.setStage(j.ID, "adaptive", 0.05)
	if imatrixPath != "" {
		req.GenerateIMatrix = false
		var imid string
		_ = m.db.QueryRow(`SELECT id FROM imatrices WHERE path = ? ORDER BY created_at DESC LIMIT 1`, imatrixPath).Scan(&imid)
		req.IMatrixID = imid
	}
	stats := m.imatrixStats(tools.IMatrix.Path, imatrixPath, tools.IMatrix.Flags)
	plan, err := PlanAdaptive(src.PrimaryPath, req.AdaptivePreset, req.TargetBPW, req.TargetBytes, stats)
	if err != nil {
		return err
	}
	tf := filepath.Join(m.layout.QuantJobs, j.ID, "tensor-types.txt")
	if err := os.MkdirAll(filepath.Dir(tf), 0o755); err != nil {
		return err
	}
	if err := writeTensorTypeFile(tf, plan.Assignments); err != nil {
		return err
	}
	req.TensorTypeFile = tf
	if req.FType == "" {
		req.FType = "Q4_K_M"
	}
	j.Request = req
	if err := m.runQuantize(ctx, j, req, src, rt, tools); err != nil {
		return err
	}
	cur, _ := m.Get(j.ID)
	res := map[string]any{"adaptive": plan.Label, "target_bpw": plan.TargetBPW}
	if cur != nil && len(cur.Result) > 0 {
		var prev map[string]any
		_ = json.Unmarshal(cur.Result, &prev)
		for k, v := range prev {
			res[k] = v
		}
	}
	return m.setResult(j.ID, res)
}

type quantOut struct {
	dest  string
	src   *models.Model
	ftype string
}

func (m *Manager) afterQuantize(ctx context.Context, j *Job, req Request, src *models.Model, dest string, tools runtimes.ToolsSnapshot, rt *runtimes.Runtime) ([]quantOut, error) {
	_ = rt
	var extra []quantOut
	destDir := filepath.Dir(dest)
	if req.CopyProjector && src.ProjectorPath != "" && !req.QuantizeProjector {
		_ = copyFile(src.ProjectorPath, filepath.Join(destDir, filepath.Base(src.ProjectorPath)))
	}
	if req.QuantizeProjector && src.ProjectorPath != "" && tools.Quantize.Present {
		pf := req.ProjectorFType
		if pf == "" {
			pf = "Q8_0"
		}
		pdest := filepath.Join(destDir, strings.TrimSuffix(filepath.Base(src.ProjectorPath), ".gguf")+"-"+pf+".gguf")
		_ = m.setStage(j.ID, "quantize_projector", 0.85)
		args, err := PlanQuantize(Request{FType: pf, Threads: req.Threads, AllowRequantize: true}, tools.Quantize.Flags, src.ProjectorPath, pdest, "", "")
		if err == nil {
			if err := m.runCmd(ctx, j, tools.Quantize.Path, args); err != nil {
				m.log.Warn("projector quantize failed", "err", err)
			}
		}
	}
	if req.QuantizeDraft && req.DraftModelID != "" {
		d, err := m.lib.Get(req.DraftModelID)
		if err == nil {
			df := req.DraftFType
			if df == "" {
				df = moreAggressive(req.FType)
			}
			ddest, err := m.destFor(d, df, "")
			if err == nil {
				_ = m.setStage(j.ID, "quantize_draft", 0.9)
				args, err := PlanQuantize(Request{
					FType: df, Threads: req.Threads,
					AllowRequantize: needsRequantizeFlag(d.Quantization),
				}, tools.Quantize.Flags, d.PrimaryPath, ddest, "", "")
				if err == nil {
					if err := m.runCmd(ctx, j, tools.Quantize.Path, args); err != nil {
						m.log.Warn("draft quantize failed", "err", err)
					} else {
						extra = append(extra, quantOut{dest: ddest, src: d, ftype: df})
					}
				}
			}
		}
	}
	if req.DeleteIntermediates && !req.KeepIMatrix {
		// generated imatrix named after job id
		_ = os.Remove(filepath.Join(m.layout.QuantIMatrices, j.ID+".gguf"))
	}
	return extra, nil
}

func (m *Manager) adoptIfPresent(dest string, src *models.Model, ftype string) string {
	if m.lib == nil || dest == "" {
		return ""
	}
	id := m.lib.IDForPath(dest)
	if id == "" {
		id = idForNearbyGGUF(m.lib, dest)
	}
	if id == "" {
		return ""
	}
	alias := ""
	if src != nil {
		alias = src.Alias
	}
	if _, err := m.lib.AdoptQuantized(id, alias, ftype, src); err != nil {
		m.log.Warn("adopt quantized model", "path", dest, "err", err)
	}
	return id
}

func idForNearbyGGUF(lib *models.Library, dest string) string {
	dir := filepath.Dir(dest)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if !strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
			continue
		}
		if id := lib.IDForPath(filepath.Join(dir, e.Name())); id != "" {
			return id
		}
	}
	return ""
}

func (m *Manager) scanAndResult(jobID, dest string, req Request, src *models.Model, extras []quantOut) error {
	if m.lib != nil {
		_, _ = m.lib.Scan()
	}
	id := ""
	alias := ""
	if m.lib != nil {
		id = m.adoptIfPresent(dest, src, CanonicalFType(req.FType))
		if id != "" {
			if m2, err := m.lib.Get(id); err == nil {
				alias = m2.Alias
			}
		}
		for _, extra := range extras {
			_ = m.adoptIfPresent(extra.dest, extra.src, extra.ftype)
		}
	}
	return m.setResult(jobID, map[string]any{
		"dest_path": dest,
		"model_id":  id,
		"alias":     alias,
		"ftype":     CanonicalFType(req.FType),
	})
}

func (m *Manager) runCmd(ctx context.Context, j *Job, exe string, args []string) error {
	if exe == "" {
		return fmt.Errorf("empty executable")
	}
	logFile, err := os.OpenFile(j.LogPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer logFile.Close()
	fmt.Fprintf(logFile, "\n# %s %s\n", exe, strings.Join(args, " "))

	pr, pw, err := os.Pipe()
	if err != nil {
		return err
	}
	env := map[string]string{}
	for _, kv := range runtimes.LibPathEnv(exe) {
		if k, v, ok := strings.Cut(kv, "="); ok {
			env[k] = v
		}
	}
	spec := processes.Spec{Exe: exe, Args: args, Dir: filepath.Dir(exe), Env: env}
	m.mu.Lock()
	h, err := processes.Start(spec, pw, pw)
	if err != nil {
		m.mu.Unlock()
		pw.Close()
		pr.Close()
		return err
	}
	m.current = h
	m.currentID = j.ID
	pid := 0
	if h.Cmd.Process != nil {
		pid = h.Cmd.Process.Pid
	}
	m.mu.Unlock()
	_, _ = m.db.Exec(`UPDATE quant_jobs SET pid=?, updated_at=? WHERE id=?`, pid, now(), j.ID)

	scanDone := make(chan struct{})
	go func() {
		defer close(scanDone)
		defer pr.Close()
		sc := bufio.NewScanner(pr)
		sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
		for sc.Scan() {
			line := diagnostics.Redact(sc.Text())
			_, _ = io.WriteString(logFile, line+"\n")
			if c, t, ok := ParseProgressLine(line); ok {
				m.emitProgress(j.ID, c, t, line)
			}
		}
	}()

	waitCh := make(chan error, 1)
	go func() {
		_, werr := h.Wait()
		waitCh <- werr
	}()
	var waitErr error
	select {
	case <-ctx.Done():
		_ = h.KillTree()
		waitErr = <-waitCh
		if waitErr == nil {
			waitErr = ctx.Err()
		}
	case waitErr = <-waitCh:
	}
	_ = pw.Close()
	<-scanDone
	m.mu.Lock()
	if m.currentID == j.ID {
		m.current = nil
		m.currentID = ""
	}
	m.mu.Unlock()
	_, _ = m.db.Exec(`UPDATE quant_jobs SET pid=0, updated_at=? WHERE id=?`, now(), j.ID)
	if waitErr != nil {
		tail, _ := m.LogTail(j.ID, 8192)
		return summarizeToolFailure(waitErr, tail)
	}
	return nil
}

func (m *Manager) emitProgress(id string, current, total int, msg string) {
	j, err := m.Get(id)
	if err != nil {
		return
	}
	p := fraction(current, total, j.Progress)
	// Keep stage-relative progress from collapsing backwards to 0.
	if p < j.Progress && total > 0 {
		p = j.Progress
	}
	_, _ = m.db.Exec(`UPDATE quant_jobs SET progress=?, updated_at=? WHERE id=?`, p, now(), id)
	if m.events != nil {
		m.events.Publish("quant.progress", Progress{
			ID: id, State: j.State, Stage: j.Stage, Current: current, Total: total, Progress: p, Message: msg,
		})
	}
}

func (m *Manager) setStage(id, stage string, progress float64) error {
	_, err := m.db.Exec(`UPDATE quant_jobs SET stage=?, progress=?, updated_at=? WHERE id=?`, stage, progress, now(), id)
	if m.events != nil {
		m.events.Publish("quant.progress", Progress{ID: id, Stage: stage, Progress: progress})
	}
	return err
}

func (m *Manager) setState(id, state, stage string, progress float64, errStr string) error {
	finished := ""
	if state == "complete" || state == "failed" || state == "canceled" {
		finished = now()
	}
	_, err := m.db.Exec(`UPDATE quant_jobs SET state=?, stage=?, progress=?, error=?, updated_at=?, finished_at=? WHERE id=?`,
		state, stage, progress, errStr, now(), finished, id)
	return err
}

func (m *Manager) setResult(id string, result map[string]any) error {
	b, _ := json.Marshal(result)
	_, err := m.db.Exec(`UPDATE quant_jobs SET result_json=?, updated_at=? WHERE id=?`, string(b), now(), id)
	return err
}

func (m *Manager) publishState(id, state, errStr string) {
	if m.events == nil {
		return
	}
	j, _ := m.Get(id)
	payload := map[string]any{"id": id, "state": state, "error": errStr}
	if j != nil {
		payload["kind"] = j.Kind
		payload["source_model_id"] = j.SourceModelID
		payload["dest_path"] = j.DestPath
		var res map[string]any
		_ = json.Unmarshal(j.Result, &res)
		if mid, _ := res["model_id"].(string); mid != "" {
			payload["model_id"] = mid
		}
		if ft, _ := res["ftype"].(string); ft != "" {
			payload["ftype"] = ft
		}
	}
	m.events.Publish("quant.state_changed", payload)
}
