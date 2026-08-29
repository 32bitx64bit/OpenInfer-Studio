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
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/diagnostics"
	"github.com/openinfer/openinfer-studio/internal/downloads"
	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/hardware"
	"github.com/openinfer/openinfer-studio/internal/huggingface"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/processes"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
	"github.com/openinfer/openinfer-studio/internal/storage"
	"quantlab/state"
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
	ID              string          `json:"id"`
	Kind            string          `json:"kind"`
	State           string          `json:"state"`
	Stage           string          `json:"stage"`
	Progress        float64         `json:"progress"`
	StageProgress   float64         `json:"stage_progress"`
	ProgressCurrent int             `json:"progress_current"`
	ProgressTotal   int             `json:"progress_total"`
	StageETASeconds int64           `json:"stage_eta_seconds"`
	ETASeconds      int64           `json:"eta_seconds"`
	ProgressMessage string          `json:"progress_message,omitempty"`
	StageStartedAt  string          `json:"stage_started_at,omitempty"`
	RuntimeID       string          `json:"runtime_id"`
	SourceModelID   string          `json:"source_model_id"`
	DestPath        string          `json:"dest_path"`
	PID             int             `json:"pid"`
	LogPath         string          `json:"log_path"`
	Request         Request         `json:"request"`
	Result          json.RawMessage `json:"result"`
	Error           string          `json:"error"`
	CreatedAt       string          `json:"created_at"`
	UpdatedAt       string          `json:"updated_at"`
	FinishedAt      string          `json:"finished_at"`
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
	hf          *huggingface.Client
	dl          *downloads.Manager
	snapshotFn  SnapshotFn
	probeFn     func(ctx context.Context, repo string) (*FromHFPreview, error)
	dlIDs       sync.Map // job id → download id

	mu        sync.Mutex
	busy      bool
	current   *processes.Handle
	currentID string
	runningID string
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

// RecoverAfterRestart requeues resumable Dynamic jobs with a valid checkpoint.
// Jobs the user paused stay paused (including a pause that was still flushing).
// Other interrupted work has no process-independent checkpoint and fails.
func (m *Manager) RecoverAfterRestart() error {
	rows, err := m.db.Query(`SELECT id,kind,state,request_json FROM quant_jobs WHERE state IN ('running','canceling','pausing')`)
	if err != nil {
		return err
	}
	var interrupted []string
	var resumable []string
	var pausing []string
	for rows.Next() {
		var id, kind, st, raw string
		if err := rows.Scan(&id, &kind, &st, &raw); err != nil {
			rows.Close()
			return err
		}
		if st == "pausing" {
			pausing = append(pausing, id)
			continue
		}
		var req Request
		_ = json.Unmarshal([]byte(raw), &req)
		if usesQuantlab(req) && quantlabCheckpointValid(filepath.Join(m.layout.QuantJobs, id, quantlabDirName, "state"), id) {
			resumable = append(resumable, id)
		} else if kind != "" && !containsString(resumable, id) {
			interrupted = append(interrupted, id)
		}
	}
	if err := rows.Close(); err != nil {
		return err
	}
	for _, id := range pausing {
		if _, err := m.db.Exec(`UPDATE quant_jobs SET state='paused',pid=0,stage_eta_seconds=0,eta_seconds=0,error='',updated_at=?,finished_at='' WHERE id=?`, now(), id); err != nil {
			return err
		}
	}
	for _, id := range resumable {
		if _, err := m.db.Exec(`UPDATE quant_jobs SET state='queued',pid=0,stage_eta_seconds=0,eta_seconds=0,error='',updated_at=?,finished_at='' WHERE id=?`, now(), id); err != nil {
			return err
		}
	}
	for _, id := range interrupted {
		if _, err := m.db.Exec(`UPDATE quant_jobs SET state='failed',stage_eta_seconds=0,eta_seconds=0,error=?, updated_at=?, finished_at=? WHERE id=?`, "interrupted by application restart", now(), now(), id); err != nil {
			return err
		}
	}
	for _, id := range interrupted {
		_ = os.Remove(filepath.Join(m.layout.QuantJobs, id, "source-ssm-f32.gguf"))
	}
	m.kick()
	return nil
}

func quantlabCheckpointValid(stateDir, runID string) bool {
	run, err := (state.Store{Dir: stateDir}).Load(runID)
	return err == nil && run != nil && run.RunID == runID && run.Config.SourcePath != ""
}

func containsString(values []string, value string) bool {
	for _, v := range values {
		if v == value {
			return true
		}
	}
	return false
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
	if req.Kind == KindFromHF {
		if strings.TrimSpace(req.HFRepo) == "" {
			return nil, fmt.Errorf("hf_repo is required for kind=from_hf")
		}
		prev, perr := m.ProbeFromHFForRequest(context.Background(), req.HFRepo, req)
		if perr != nil {
			return nil, perr
		}
		if !prev.Compatible && prev.ReusedModelID == "" {
			reason := prev.Reason
			if reason == "" {
				reason = "repository is not convertible"
			}
			return nil, fmt.Errorf("%s", reason)
		}
		if prev.ReusedModelID != "" && req.SourceModelID == "" {
			req.SourceModelID = prev.ReusedModelID
		}
	}
	src, err := m.lib.Get(req.SourceModelID)
	if err != nil && req.Kind != KindCombineIMatrix && req.Kind != KindFromHF {
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
	if req.TargetBPW < 0 {
		return nil, fmt.Errorf("target_bpw must be non-negative")
	}
	if req.TargetBytes < 0 {
		return nil, fmt.Errorf("target_bytes must be non-negative")
	}
	quantlabRequested := usesQuantlab(req)
	adaptiveMode := strings.ToLower(strings.TrimSpace(req.AdaptiveMode))
	effort := strings.ToLower(strings.TrimSpace(req.Effort))
	if effort == "" {
		effort = adaptiveMode
	}
	switch effort {
	case "", "fast", "profiled", "deep":
	default:
		return nil, fmt.Errorf("effort must be fast, profiled, or deep")
	}
	switch adaptiveMode {
	case "", "fast", "profiled", "deep":
	default:
		return nil, fmt.Errorf("adaptive_mode must be fast, profiled, or deep")
	}
	if quantlabRequested {
		if effort == "" {
			effort = "profiled"
		}
		req.Effort = effort
		if _, err := normalizeAdaptiveTarget(&req); err != nil {
			return nil, err
		}
	}
	if quantlabRequested && !supportsQuantlabEvaluation(tools.Perplexity) {
		return nil, fmt.Errorf("OpenInfer Dynamic requires a compatible llama-perplexity beside the selected runtime for quantlab quality gates")
	}
	if req.Kind != KindCombineIMatrix && !tools.Quantize.Present &&
		(req.Kind == KindQuantize || req.Kind == KindAdaptiveQuantize || req.Kind == KindFromHF) {
		return nil, fmt.Errorf("runtime %s has no llama-quantize next to llama-server; install an official llama.cpp build", rt.ID)
	}
	if quantlabRequested && !flagOK(tools.Quantize.Flags, "--tensor-type-file") && !flagOK(tools.Quantize.Flags, "--tensor-type") {
		return nil, fmt.Errorf("the selected runtime does not advertise per-tensor quantization overrides")
	}
	if quantlabRequested && !flagOK(tools.Quantize.Flags, "--pure") {
		return nil, fmt.Errorf("OpenInfer Dynamic requires llama-quantize --pure so tensor-bank anchors retain every requested per-tensor dtype")
	}
	if quantlabRequested && !tools.IMatrix.Present {
		return nil, fmt.Errorf("OpenInfer Dynamic requires llama-imatrix beside the selected runtime")
	}
	if src != nil && models.IsSpeculativeDraft(*src) {
		if req.Kind == KindIMatrix {
			return nil, fmt.Errorf("llama-imatrix cannot load a speculative assistant GGUF (%s) without the main model context", filepath.Base(src.PrimaryPath))
		}
		req.GenerateIMatrix = false
	}
	if quantlabRequested &&
		src != nil && !models.IsSpeculativeDraft(*src) && req.IMatrixID == "" && tools.IMatrix.Present {
		req.GenerateIMatrix = true
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
	if RequiresIMatrix(ftName) && req.IMatrixID == "" && !req.GenerateIMatrix &&
		req.Kind != KindIMatrix && req.Kind != KindCombineIMatrix && req.Kind != KindFromHF {
		return nil, fmt.Errorf("%s requires an importance matrix (generate_imatrix or imatrix_id)", ftName)
	}
	// Dynamic (quantlab) jobs defer source GGUF validation and the
	// scratch-disk preflight into the execution goroutine: both parse the
	// source GGUF and exceed the QML HTTP timeout for large BF16 sources.
	// Start only does cheap checks here so the queued job is visible at once.
	if src != nil && !quantlabRequested {
		if issues, _, verr := gguf.ValidateFile(src.PrimaryPath); verr != nil {
			return nil, fmt.Errorf("source GGUF: %w", verr)
		} else if len(issues) > 0 {
			return nil, fmt.Errorf("source GGUF failed validation: %s", issues[0])
		}
		if m.diskFree != nil && req.Kind == KindQuantize {
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
	err := m.db.QueryRow(`SELECT id,kind,state,stage,progress,stage_progress,progress_current,progress_total,stage_eta_seconds,eta_seconds,progress_message,stage_started_at,runtime_id,source_model_id,dest_path,pid,log_path,request_json,result_json,error,created_at,updated_at,finished_at
		FROM quant_jobs WHERE id = ?`, id).Scan(
		&j.ID, &j.Kind, &j.State, &j.Stage, &j.Progress, &j.StageProgress, &j.ProgressCurrent, &j.ProgressTotal,
		&j.StageETASeconds, &j.ETASeconds, &j.ProgressMessage, &j.StageStartedAt, &j.RuntimeID, &j.SourceModelID, &j.DestPath,
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
	m.cancelAttachedDownload(id)
	switch j.State {
	case "queued", "paused":
		if j.State == "paused" {
			m.cleanupPartial(j)
		}
		if err := m.setState(id, "canceled", j.Stage, j.Progress, "canceled"); err != nil {
			return err
		}
		m.publishState(id, "canceled", "canceled")
		return nil
	case "running", "canceling", "pausing":
		m.interrupt(id)
		return m.setState(id, "canceling", j.Stage, j.Progress, "")
	default:
		return fmt.Errorf("job is %s", j.State)
	}
}

// Pause stops a queued or running job without discarding Dynamic checkpoints,
// the repaired source copy, or a partial llama-imatrix GGUF. The GPU is freed
// once tools exit; Resume re-queues the same job.
func (m *Manager) Pause(id string) error {
	for range 3 {
		j, err := m.Get(id)
		if err != nil {
			return err
		}
		switch j.State {
		case "paused", "pausing":
			return nil
		case "queued":
			ok, err := m.casState(id, "queued", "paused", j.Stage, j.Progress, "")
			if err != nil {
				return err
			}
			if ok {
				m.publishState(id, "paused", "")
				return nil
			}
		case "running":
			ok, err := m.casState(id, "running", "pausing", j.Stage, j.Progress, "")
			if err != nil {
				return err
			}
			if !ok {
				break
			}
			m.pauseAttachedDownload(id)
			m.interrupt(id)
			m.publishState(id, "pausing", "")
			return nil
		default:
			return fmt.Errorf("job is %s", j.State)
		}
	}
	j, err := m.Get(id)
	if err != nil {
		return err
	}
	switch j.State {
	case "paused", "pausing":
		return nil
	default:
		return fmt.Errorf("job is %s", j.State)
	}
}

// Resume re-queues a paused job. Dynamic work continues from the last
// checkpoint. llama-imatrix continues from imatrix.chunk_count on the
// job's output GGUF when --in-file/--from-chunk are advertised.
func (m *Manager) Resume(id string) error {
	j, err := m.Get(id)
	if err != nil {
		return err
	}
	switch j.State {
	case "paused":
		ok, err := m.casState(id, "paused", "queued", j.Stage, j.Progress, "")
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("job is %s", j.State)
		}
		m.publishState(id, "queued", "")
		m.kick()
		return nil
	case "pausing":
		return fmt.Errorf("job is still pausing")
	default:
		return fmt.Errorf("job is %s", j.State)
	}
}

func (m *Manager) interrupt(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.runningID == id && m.cancel != nil {
		m.cancel()
	}
	if m.currentID == id && m.current != nil {
		_ = m.current.KillTree()
	}
}

func (m *Manager) pauseAttachedDownload(id string) {
	if v, ok := m.dlIDs.Load(id); ok {
		if did, ok := v.(string); ok && did != "" && m.dl != nil {
			_ = m.dl.Pause(did)
		}
	}
}

func (m *Manager) cancelAttachedDownload(id string) {
	if v, ok := m.dlIDs.Load(id); ok {
		if did, ok := v.(string); ok && did != "" && m.dl != nil {
			_ = m.dl.Cancel(did)
		}
	}
}

func (m *Manager) pauseIntent(id string) bool {
	j, err := m.Get(id)
	return err == nil && j != nil && (j.State == "pausing" || j.State == "paused")
}

// Delete removes a finished, paused, or still-queued job from history.
// Running jobs must be canceled or paused first. Output GGUFs in the library
// are left in place.
func (m *Manager) Delete(id string) error {
	j, err := m.Get(id)
	if err != nil {
		return err
	}
	switch j.State {
	case "running", "canceling", "pausing":
		return fmt.Errorf("job is %s; cancel or wait for pause first", j.State)
	}
	return m.removeJobs([]Job{*j})
}

// ClearHistory deletes complete/failed/canceled jobs. Queued, paused, and
// running jobs are left untouched. Quantized models already in the library stay.
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
	id := m.nextQueued()
	if id == "" {
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
	if j.State != "queued" {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	m.mu.Lock()
	m.cancel = cancel
	m.runningID = id
	m.mu.Unlock()
	defer func() {
		cancel()
		m.mu.Lock()
		if m.runningID == id {
			m.runningID = ""
			m.cancel = nil
		}
		m.mu.Unlock()
	}()

	ok, err := m.casState(id, "queued", "running", "starting", 0, "")
	if err != nil || !ok {
		return
	}
	m.publishState(id, "running", "")
	err = m.run(ctx, j)
	if err == nil {
		_ = m.setState(id, "complete", "done", 1, "")
		m.publishState(id, "complete", "")
		return
	}
	cur, _ := m.Get(id)
	if cur != nil && (cur.State == "pausing" || cur.State == "paused") {
		_ = m.setState(id, "paused", cur.Stage, cur.Progress, "")
		m.publishState(id, "paused", "")
		return
	}
	// A subprocess killed mid-cancel (notably llama-imatrix during the
	// deferred "analyze" disk preflight, which does not accept a context)
	// surfaces as a signal error rather than context.Canceled, so honor
	// the interrupt whenever the context is done.
	if errors.Is(err, context.Canceled) || ctx.Err() != nil {
		m.cleanupPartial(j)
		_ = m.setState(id, "canceled", j.Stage, j.Progress, "canceled")
		m.publishState(id, "canceled", "canceled")
		return
	}
	m.cleanupPartial(j)
	msg := err.Error()
	_ = m.setState(id, "failed", j.Stage, j.Progress, msg)
	m.publishState(id, "failed", msg)
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
	_ = os.Remove(filepath.Join(m.layout.QuantJobs, j.ID, "candidate.gguf"))
	// A valid Dynamic checkpoint can contain hours of measured candidates and
	// a validated gate-failing artifact. Keep it on failure/cancellation so a
	// retry or application restart can resume; invalid scratch is disposable.
	stateDir := filepath.Join(m.layout.QuantJobs, j.ID, quantlabDirName, "state")
	if !usesQuantlab(cur.Request) || !quantlabCheckpointValid(stateDir, j.ID) {
		_ = os.RemoveAll(filepath.Join(m.layout.QuantJobs, j.ID, quantlabDirName))
	}
	if cur.DestPath != "" {
		_ = os.Remove(filepath.Join(m.layout.QuantJobs, j.ID, "candidate-"+filepath.Base(cur.DestPath)))
	}
	_ = os.Remove(filepath.Join(m.layout.QuantJobs, j.ID, "source-ssm-f32.gguf"))
	if cur.State != "complete" && cur.DestPath != "" {
		if _, _, err := gguf.ValidateFile(cur.DestPath); err != nil {
			_ = os.Remove(cur.DestPath)
		}
	}
}

func (m *Manager) run(ctx context.Context, j *Job) error {
	req := j.Request
	src, err := m.lib.Get(req.SourceModelID)
	if err != nil && req.Kind != KindCombineIMatrix && req.Kind != KindFromHF {
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

	// Dynamic jobs run the GGUF-parsing scratch-disk preflight here — as a
	// brief first "analyze" stage before the quantlab pipeline — instead of
	// in Start(), so Start() stays within the QML HTTP timeout. A failure
	// transitions the job to failed with the same disk error. from_hf jobs
	// already preflight inside runFromHF, and a resuming run already passed
	// this gate before the crash.
	if req.Kind == KindAdaptiveQuantize && src != nil {
		stateDir := filepath.Join(m.layout.QuantJobs, j.ID, quantlabDirName, "state")
		if !quantlabCheckpointValid(stateDir, j.ID) {
			_ = m.setStage(j.ID, "analyze", 0)
			if err := m.checkQuantlabDisk(src, req); err != nil {
				return err
			}
		}
	}

	switch req.Kind {
	case KindIMatrix:
		return m.runIMatrixOnly(ctx, j, req, src, tools)
	case KindCombineIMatrix:
		return m.runCombine(ctx, j, req, src, tools)
	case KindAdaptiveQuantize:
		return m.runQuantlabAdaptive(ctx, j, req, src, rt, tools)
	case KindFromHF:
		return m.runFromHF(ctx, j, req, rt, tools)
	default:
		return m.runQuantize(ctx, j, req, src, rt, tools, "")
	}
}

func (m *Manager) runIMatrixOnly(ctx context.Context, j *Job, req Request, src *models.Model, tools runtimes.ToolsSnapshot) error {
	path, err := m.generateIMatrix(ctx, j, req, src, tools)
	if err != nil {
		return err
	}
	return m.setResult(j.ID, map[string]any{"imatrix_path": path})
}

func (m *Manager) runCombine(ctx context.Context, j *Job, req Request, src *models.Model, tools runtimes.ToolsSnapshot) error {
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
	modelPath := ""
	if src != nil {
		modelPath = src.PrimaryPath
	}
	args, err := PlanCombine(tools.IMatrix.Flags, inputs, out, modelPath)
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
	files, err := m.prepareJobCalibration(j, src, cal)
	if err != nil {
		return "", err
	}
	out := filepath.Join(m.layout.QuantIMatrices, j.ID+".gguf")
	// One mixed llama-imatrix pass. Splitting prose/facts/code and averaging
	// three matrices is not the same as mixed activations, and each short
	// domain file overfits when --chunks is large. Unsloth Dynamic 2.0 uses
	// a single mixed calibration set.
	if len(files) > 1 {
		one, err := concatPrepared(filepath.Dir(files[0].Path), files)
		if err != nil {
			return "", err
		}
		one.Domain = "mixed"
		files = []preparedCal{one}
	}
	wanted := effectiveIMatrixChunks(req, files[0].Path)
	done, skip, inFile, cleanup, err := prepareIMatrixContinue(out, wanted)
	if err != nil {
		return "", err
	}
	defer cleanup()
	if done {
		return m.finishGeneratedIMatrix(src.ID, out, req, files, wanted)
	}
	if skip > 0 {
		req.ChunkSkip = skip
		req.IMatrixInFile = inFile
		if j.LogPath != "" {
			if f, err := os.OpenFile(j.LogPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644); err == nil {
				fmt.Fprintf(f, "continuing importance matrix from chunk %d (%s)\n", skip, inFile)
				f.Close()
			}
		}
	}
	args, err := PlanIMatrix(req, tools.IMatrix.Flags, src.PrimaryPath, files[0].Path, out)
	if err != nil {
		return "", err
	}
	if err := m.runCmd(ctx, j, tools.IMatrix.Path, args); err != nil {
		return "", fmt.Errorf("llama-imatrix: %w", err)
	}
	return m.finishGeneratedIMatrix(src.ID, out, req, files, wanted)
}

// prepareIMatrixContinue copies a partial llama-imatrix GGUF so --in-file and
// --from-chunk can continue without llama.cpp truncating the same path it reads.
func prepareIMatrixContinue(out string, wanted int) (done bool, skip int, inFile string, cleanup func(), err error) {
	cleanup = func() {}
	n := imatrixChunkCount(out)
	if n <= 0 {
		return false, 0, "", cleanup, nil
	}
	if wanted > 0 && n >= wanted {
		return true, n, "", cleanup, nil
	}
	inFile = out + ".continue"
	if err := copyFile(out, inFile); err != nil {
		return false, 0, "", cleanup, fmt.Errorf("preserve partial imatrix: %w", err)
	}
	return false, n, inFile, func() { _ = os.Remove(inFile) }, nil
}

func effectiveIMatrixChunks(req Request, calibrationPath string) int {
	chunks := req.Chunks
	if chunks == 0 {
		chunks = presetChunks(req.CalibrationPreset)
	}
	ctx := 512
	if wantIMatrixCtx(req) {
		ctx = imatrixCtxForFile(calibrationPath, 4096)
		if ctx < 512 {
			ctx = 512
		}
	}
	return clampIMatrixChunks(calibrationPath, ctx, chunks)
}

func (m *Manager) finishGeneratedIMatrix(modelID, out string, req Request, files []preparedCal, chunks int) (string, error) {
	label := preparedCalLabel(req.CalibrationPreset, files)
	if req.CalibrationPreset == "" && req.CalibrationPath != "" {
		label = preparedCalLabel(filepath.Base(req.CalibrationPath), files)
	}
	if _, err := m.recordGeneratedIMatrix(modelID, out, label, chunks, "generated"); err != nil {
		return "", err
	}
	return out, nil
}

func (m *Manager) resolveIMatrixPath(ctx context.Context, j *Job, req Request, src *models.Model, tools runtimes.ToolsSnapshot) (string, error) {
	if req.IMatrixID != "" {
		im, err := m.getIMatrix(req.IMatrixID)
		if err != nil {
			return "", err
		}
		if src != nil && im.SourceModelID != "" && im.SourceModelID != src.ID {
			return "", fmt.Errorf("importance matrix belongs to model %s, not selected source %s", im.SourceModelID, src.ID)
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
		if im := m.findReusableIMatrix(src, req); im != nil {
			if j.LogPath != "" {
				if f, err := os.OpenFile(j.LogPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644); err == nil {
					fmt.Fprintf(f, "reusing importance matrix %s (%s, %d chunks)\n", im.Path, im.DatasetLabel, im.NChunks)
					f.Close()
				}
			}
			return im.Path, nil
		}
		return m.generateIMatrix(ctx, j, req, src, tools)
	}
	return "", nil
}

// findReusableIMatrix returns a previously generated matrix for the same
// source and calibration preset. Custom CalibrationPath never reuses. Chunk
// counts are not compared: llama-imatrix clamps --chunks to the file, so a
// strict match would regenerate the matrix we just wrote.
func (m *Manager) findReusableIMatrix(src *models.Model, req Request) *IMatrix {
	if m == nil || m.db == nil || src == nil || strings.TrimSpace(src.ID) == "" {
		return nil
	}
	if strings.TrimSpace(req.CalibrationPath) != "" {
		return nil
	}
	list, err := m.ListIMatrices(src.ID)
	if err != nil {
		return nil
	}
	preset := strings.ToLower(strings.TrimSpace(req.CalibrationPreset))
	if preset == "" {
		preset = "calibration"
	}
	for i := range list {
		im := &list[i]
		if fileSize(im.Path) <= 0 {
			continue
		}
		label := strings.ToLower(strings.TrimSpace(im.DatasetLabel))
		if label != "" && !strings.HasPrefix(label, preset) {
			continue
		}
		return im
	}
	return nil
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

func (m *Manager) runQuantize(ctx context.Context, j *Job, req Request, src *models.Model, rt *runtimes.Runtime, tools runtimes.ToolsSnapshot, quantLabel string) error {
	dest, _, effectiveReq, err := m.runQuantizeCandidate(ctx, j, req, src, tools, quantLabel, false)
	if err != nil {
		return err
	}
	extras, err := m.afterQuantize(ctx, j, effectiveReq, src, dest, tools, rt)
	if err != nil {
		return err
	}
	return m.scanAndResult(j.ID, dest, effectiveReq, src, extras, quantLabel)
}

// runQuantizeCandidate writes a candidate but deliberately does not scan or
// adopt it. Dynamic quantization uses this boundary to run its quality gate
// before a model can become visible in the library.
func (m *Manager) runQuantizeCandidate(ctx context.Context, j *Job, req Request, src *models.Model, tools runtimes.ToolsSnapshot, quantLabel string, staged bool) (string, string, Request, error) {
	imatrixPath, err := m.resolveIMatrixPath(ctx, j, req, src, tools)
	if err != nil {
		return "", "", req, err
	}
	ftype := CanonicalFType(req.FType)
	if ftype == "" {
		ftype = "Q4_K_M"
	}
	if quantLabel == "" {
		quantLabel = ftype
	}
	if isSplitPath(src.PrimaryPath) && !req.KeepSplit {
		req.KeepSplit = flagOK(tools.Quantize.Flags, "--keep-split")
	}
	if needsRequantizeFlag(src.Quantization) {
		req.AllowRequantize = true
	}
	dest, err := m.destFor(src, quantLabel, req.OutputName)
	if err != nil {
		return "", "", req, err
	}
	outputPath := dest
	if staged {
		outputPath = filepath.Join(m.layout.QuantJobs, j.ID, "candidate-"+filepath.Base(dest))
		_ = os.Remove(outputPath)
	}
	_, _ = m.db.Exec(`UPDATE quant_jobs SET dest_path=?, updated_at=? WHERE id=?`, dest, now(), j.ID)
	_ = m.setStage(j.ID, "quantize", 0.05)
	args, err := PlanQuantize(req, tools.Quantize.Flags, src.PrimaryPath, outputPath, imatrixPath, req.TensorTypeFile)
	if err != nil {
		return "", "", req, err
	}
	if size, ok := m.quantizeDryRunSize(ctx, req, tools.Quantize, src.PrimaryPath, imatrixPath, req.TensorTypeFile); ok && m.diskFree != nil {
		need := uint64(size) + uint64(size)/10
		if free := m.diskFree(m.layout.Models); free > 0 && need > free {
			return "", "", req, fmt.Errorf("not enough free disk space for the verified output size")
		}
	}
	if err := m.runCmd(ctx, j, tools.Quantize.Path, args); err != nil {
		_ = os.Remove(outputPath)
		return "", "", req, fmt.Errorf("llama-quantize: %w", err)
	}
	return dest, outputPath, req, nil
}

func (m *Manager) quantizeDryRunSize(ctx context.Context, req Request, tool runtimes.ToolInfo, inPath, imatrixPath, tensorFile string) (int64, bool) {
	if !flagOK(tool.Flags, "--dry-run") {
		return 0, false
	}
	args, err := PlanQuantizeDryRun(req, tool.Flags, inPath, imatrixPath, tensorFile)
	if err != nil {
		return 0, false
	}
	probeCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	out, err := runtimes.RunTool(probeCtx, tool, args)
	if err != nil {
		return 0, false
	}
	return runtimes.ParseQuantizeSize(out)
}

func (m *Manager) prepareQuantlabSource(ctx context.Context, j *Job, src *models.Model) (*models.Model, bool, func(), error) {
	cleanup := func() {}
	if src == nil {
		return nil, false, cleanup, fmt.Errorf("quantlab source is missing")
	}
	status, err := gguf.InspectSSMConv1d(src.PrimaryPath)
	if err != nil {
		return nil, false, cleanup, fmt.Errorf("inspect source SSM convolution layout: %w", err)
	}
	if !status.Required {
		return src, false, cleanup, nil
	}
	if !status.Repairable {
		return nil, false, cleanup, fmt.Errorf("source has non-F32 ssm_conv1d tensors that cannot be repaired; reconvert from the Hugging Face weights")
	}
	if isSplitPath(src.PrimaryPath) {
		return nil, false, cleanup, fmt.Errorf("legacy split GGUF has non-F32 ssm_conv1d tensors; reconvert from the Hugging Face weights")
	}

	// Repair writes a full temporary source before quantlab creates its output.
	required := uint64(status.OutputBytes) + uint64(src.SizeBytes) + uint64(src.SizeBytes)/10
	if m.diskFree != nil {
		if free := m.diskFree(m.layout.QuantJobs); free > 0 && required > free {
			return nil, false, cleanup, fmt.Errorf(
				"legacy SSM layout needs %.1f GiB of temporary disk space for repair and quantization; %.1f GiB is available",
				float64(required)/(1<<30), float64(free)/(1<<30))
		}
	}

	jobDir := filepath.Join(m.layout.QuantJobs, j.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		return nil, false, cleanup, err
	}
	repairedPath := filepath.Join(jobDir, "source-ssm-f32.gguf")
	_ = m.setStage(j.ID, "repair_source", 0)
	m.log.Info("repairing legacy SSM convolution layout", "job", j.ID, "source", src.PrimaryPath, "tensors", status.TensorCount)
	_, err = gguf.RepairSSMConv1d(ctx, src.PrimaryPath, repairedPath, func(done, total int64) {
		progress := 0.0
		if total > 0 {
			progress = float64(done) / float64(total)
		}
		m.emitProgressSample(j.ID, commandProgressSample{
			Current:  int(done / (1 << 20)),
			Total:    int((total + (1 << 20) - 1) / (1 << 20)),
			Progress: progress,
			Message:  "repairing legacy SSM convolution layout",
		})
	})
	if err != nil {
		return nil, false, cleanup, fmt.Errorf("repair legacy SSM convolution layout: %w", err)
	}
	working := *src
	working.PrimaryPath = repairedPath
	working.SizeBytes = status.OutputBytes
	cleanup = func() { _ = os.Remove(repairedPath) }
	return &working, true, cleanup, nil
}

func discardValidatedCandidate(dest, candidatePath string) {
	if dest == "" && candidatePath == "" {
		return
	}
	_ = os.Remove(candidatePath)
	base := strings.TrimSuffix(dest, filepath.Ext(dest))
	_ = os.Remove(dest)
	_ = os.Remove(base + ".oid-plan.json")
	_ = os.Remove(base + ".tensor-types.txt")
	_ = os.Remove(base + ".quantlab-report.json")
	// destFor creates two private parent directories for this candidate.
	_ = os.Remove(filepath.Dir(dest))
	_ = os.Remove(filepath.Dir(filepath.Dir(dest)))
}

func supportsQuantlabEvaluation(tool runtimes.ToolInfo) bool {
	return tool.Present && tool.Path != "" &&
		flagOK(tool.Flags, "--model", "-m") &&
		flagOK(tool.Flags, "--file", "-f") &&
		flagOK(tool.Flags, "--kl-divergence-base") &&
		flagOK(tool.Flags, "--kl-divergence") &&
		flagOK(tool.Flags, "--chunks", "-chunks")
}

func (m *Manager) ensureQuantlabCorpus(j *Job, req Request, src *models.Model) error {
	heldOut := filepath.Join(m.layout.QuantJobs, j.ID, "calibration", "validation.txt")
	if st, err := os.Stat(heldOut); err == nil && !st.IsDir() && st.Size() > 0 {
		return nil
	}
	calibrationPath := req.CalibrationPath
	if calibrationPath == "" {
		var err error
		calibrationPath, err = m.EnsureCalibration()
		if err != nil {
			return fmt.Errorf("prepare quantlab evaluation corpus: %w", err)
		}
	}
	if _, err := m.prepareJobCalibration(j, src, calibrationPath); err != nil {
		return fmt.Errorf("prepare quantlab evaluation corpus: %w", err)
	}
	if st, err := os.Stat(heldOut); err != nil || st.IsDir() || st.Size() == 0 {
		return fmt.Errorf("quantlab evaluation corpus is unavailable")
	}
	return nil
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
		if err := copyFile(src.ProjectorPath, filepath.Join(destDir, filepath.Base(src.ProjectorPath))); err != nil {
			m.log.Warn("copy projector next to quantized GGUF", "err", err)
		}
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
	if req.DeleteIntermediates && !req.KeepIMatrix && !usesQuantlab(req) {
		// generated imatrix named after job id. Dynamic jobs keep the file so
		// the next run on the same source can reuse it.
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

func (m *Manager) scanAndResult(jobID, dest string, req Request, src *models.Model, extras []quantOut, quantLabel string) error {
	if m.lib != nil {
		if _, err := m.lib.Scan(); err != nil {
			return fmt.Errorf("scan quantized model: %w", err)
		}
	}
	if quantLabel == "" {
		quantLabel = CanonicalFType(req.FType)
	}
	id := ""
	alias := ""
	if m.lib != nil {
		id = m.adoptIfPresent(dest, src, quantLabel)
		if id == "" {
			return fmt.Errorf("quantized model was not adopted by the library")
		}
		if id != "" {
			if m2, err := m.lib.Get(id); err == nil {
				alias = m2.Alias
			}
		}
		for _, extra := range extras {
			_ = m.adoptIfPresent(extra.dest, extra.src, extra.ftype)
		}
	} else {
		return fmt.Errorf("quantized model library is unavailable")
	}
	return m.setResult(jobID, map[string]any{
		"dest_path": dest,
		"model_id":  id,
		"alias":     alias,
		"ftype":     quantLabel,
	})
}

func (m *Manager) runCmd(ctx context.Context, j *Job, exe string, args []string) error {
	return m.runCmdInternal(ctx, j, exe, args)
}

func (m *Manager) runCmdInternal(ctx context.Context, j *Job, exe string, args []string) error {
	if exe == "" {
		return fmt.Errorf("empty executable")
	}
	logFile, err := os.OpenFile(j.LogPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer logFile.Close()
	fmt.Fprintf(logFile, "\n# %s %s\n", exe, strings.Join(args, " "))

	commandStarted := time.Now()
	tracker := newCommandProgressTracker(commandStarted)
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

	trackerDone := make(chan struct{})
	var trackerWG sync.WaitGroup
	trackerWG.Add(1)
	go func() {
		defer trackerWG.Done()
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-trackerDone:
				return
			case at := <-ticker.C:
				if sample, ok := tracker.Estimate(at); ok {
					m.emitProgressSample(j.ID, sample)
				}
			}
		}
	}()
	scanDone := make(chan struct{})
	go func() {
		defer close(scanDone)
		defer pr.Close()
		sc := bufio.NewScanner(pr)
		sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
		sc.Split(splitTerminalLines)
		for sc.Scan() {
			line := diagnostics.Redact(sc.Text())
			_, _ = io.WriteString(logFile, line+"\n")
			if sample, ok := tracker.Observe(line, time.Now()); ok {
				m.emitProgressSample(j.ID, sample)
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
	close(trackerDone)
	trackerWG.Wait()
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

func (m *Manager) emitProgressSample(id string, sample commandProgressSample) {
	j, err := m.Get(id)
	if err != nil {
		return
	}
	stageProgress := clampProgress(sample.Progress)
	if stageProgress < j.StageProgress {
		stageProgress = j.StageProgress
	}
	start, end := stageRange(j, j.Stage)
	if start < 0 || end <= start {
		start, end = j.Progress, 1
	}
	overall := start + (end-start)*stageProgress
	if overall < j.Progress {
		overall = j.Progress
	}
	eta := overallETA(sample.ETASeconds, overall, end)
	message := sample.Message
	if message == "" {
		message = j.ProgressMessage
	}
	_, _ = m.db.Exec(`UPDATE quant_jobs SET progress=?,stage_progress=?,progress_current=?,progress_total=?,stage_eta_seconds=?,eta_seconds=?,progress_message=?,updated_at=? WHERE id=?`,
		overall, stageProgress, sample.Current, sample.Total, sample.ETASeconds, eta, message, now(), id)
	if m.events != nil {
		m.events.Publish("quant.progress", Progress{
			ID: id, State: j.State, Stage: j.Stage, Current: sample.Current, Total: sample.Total,
			Progress: overall, StageProgress: stageProgress, StageETASeconds: sample.ETASeconds,
			ETASeconds: eta, Message: message, Estimated: sample.Estimated,
		})
	}
}

func (m *Manager) setStage(id, stage string, progress float64) error {
	j, getErr := m.Get(id)
	if getErr != nil {
		return getErr
	}
	start, _ := stageRange(j, stage)
	if start < 0 {
		start = progress
	}
	if start < j.Progress {
		start = j.Progress
	}
	started := now()
	_, err := m.db.Exec(`UPDATE quant_jobs SET stage=?,progress=?,stage_progress=0,progress_current=0,progress_total=0,stage_eta_seconds=0,eta_seconds=0,progress_message='',stage_started_at=?,updated_at=? WHERE id=?`,
		stage, start, started, started, id)
	if m.events != nil {
		m.events.Publish("quant.progress", Progress{ID: id, State: j.State, Stage: stage, Progress: start})
	}
	return err
}

func (m *Manager) setState(id, state, stage string, progress float64, errStr string) error {
	_, err := m.transitionState(id, "", state, stage, progress, errStr)
	return err
}

func (m *Manager) casState(id, from, to, stage string, progress float64, errStr string) (bool, error) {
	return m.transitionState(id, from, to, stage, progress, errStr)
}

func (m *Manager) transitionState(id, from, to, stage string, progress float64, errStr string) (bool, error) {
	finished := ""
	if to == "complete" || to == "failed" || to == "canceled" {
		finished = now()
	}
	stageProgress := -1.0
	if to == "running" {
		stageProgress = 0
	} else if to == "complete" {
		stageProgress = 1
	}
	q := `UPDATE quant_jobs SET state=?,stage=?,progress=?,stage_progress=CASE WHEN ?>=0 THEN ? ELSE stage_progress END,stage_eta_seconds=0,eta_seconds=0,error=?,updated_at=?,finished_at=?,pid=CASE WHEN ?='running' THEN pid ELSE 0 END WHERE id=?`
	args := []any{to, stage, progress, stageProgress, stageProgress, errStr, now(), finished, to, id}
	if from != "" {
		q += ` AND state=?`
		args = append(args, from)
	}
	res, err := m.db.Exec(q, args...)
	if err != nil {
		return false, err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, err
	}
	return n > 0, nil
}

func (m *Manager) setResult(id string, result map[string]any) error {
	b, _ := json.Marshal(result)
	_, err := m.db.Exec(`UPDATE quant_jobs SET result_json=?, updated_at=? WHERE id=?`, string(b), now(), id)
	return err
}

func (m *Manager) persistRequest(id string, req Request) error {
	b, err := json.Marshal(req)
	if err != nil {
		return err
	}
	_, err = m.db.Exec(`UPDATE quant_jobs SET source_model_id=?, request_json=?, updated_at=? WHERE id=?`, req.SourceModelID, string(b), now(), id)
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
