// Package instances supervises one llama-server process per loaded model.
package instances

import (
	"bufio"
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/openinfer/openinfer-studio/internal/diagnostics"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/processes"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

// State machine states.
const (
	StateUnloaded = "unloaded"
	StateStarting = "starting"
	StateLoading  = "loading"
	StateReady    = "ready"
	StateBusy     = "busy"
	StateSleeping = "sleeping"
	StateStopping = "stopping"
	StateFailed   = "failed"
	StateCrashed  = "crashed"
)

type EventSink interface {
	Publish(event string, payload any)
}

// Instance is the live (or last-known) state of a loaded model.
type Instance struct {
	ID           string    `json:"id"`
	ModelID      string    `json:"model_id"`
	ModelAlias   string    `json:"model_alias"`
	PresetID     string    `json:"preset_id"`
	RuntimeID    string    `json:"runtime_id"`
	PID          int       `json:"pid"`
	Port         int       `json:"port"`
	State        string    `json:"state"`
	Requests     int64     `json:"requests"`
	Args         []string  `json:"args"`
	Command      string    `json:"command"`
	Warnings     []string  `json:"warnings"`
	StartedAt    time.Time `json:"started_at"`
	LastActivity time.Time `json:"last_activity"`
	ExitCode     int       `json:"exit_code"`
	Failure      string    `json:"failure"`
	FailureClass string    `json:"failure_class"`
	LogFile      string    `json:"log_file"`
	Backend      string    `json:"backend"`
}

type liveInstance struct {
	Instance
	apiKey       string
	handle       *processes.Handle
	diffEngine   *DiffusionEngine
	diffShim     *DiffusionShim
	cancel       context.CancelFunc
	done         chan struct{}
	lastActivity *Activity
}

// Manager owns all model instances.
type Manager struct {
	db             *sql.DB
	events         EventSink
	log            *slog.Logger
	logDir         string
	tempDir        string
	cacheDir       string
	runtimes       *runtimes.Manager
	library        *models.Library
	maxLoaded      int
	startupTimeout time.Duration

	mu        sync.Mutex
	instances map[string]*liveInstance // keyed by model ID (one per model)
}

func NewManager(db *sql.DB, rt *runtimes.Manager, lib *models.Library,
	events EventSink, log *slog.Logger, logDir, tempDir, cacheDir string) *Manager {
	return &Manager{
		db: db, runtimes: rt, library: lib, events: events, log: log,
		logDir: logDir, tempDir: tempDir, cacheDir: cacheDir,
		maxLoaded: 8, startupTimeout: 10 * time.Minute,
		instances: map[string]*liveInstance{},
	}
}

// SetMaxLoaded limits simultaneously loaded models.
func (m *Manager) SetMaxLoaded(n int) {
	if n >= 1 && n <= 32 {
		m.maxLoaded = n
	}
}

// SetStartupTimeout adjusts the readiness ceiling.
func (m *Manager) SetStartupTimeout(d time.Duration) {
	if d >= 10*time.Second && d <= time.Hour {
		m.startupTimeout = d
	}
}

// List returns current instances.
func (m *Manager) List() []Instance {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Instance, 0, len(m.instances))
	for _, li := range m.instances {
		out = append(out, li.Instance)
	}
	return out
}

// Get returns the instance for a model, if any.
func (m *Manager) Get(modelID string) (*Instance, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if li, ok := m.instances[modelID]; ok {
		cp := li.Instance
		return &cp, true
	}
	return nil, false
}

// Endpoint describes how to reach a ready instance (used by chat & proxy).
type Endpoint struct {
	URL    string
	APIKey string
	Alias  string
}

// EndpointFor returns the loopback endpoint of a ready model instance.
func (m *Manager) EndpointFor(modelID string) (Endpoint, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	li, ok := m.instances[modelID]
	if !ok {
		return Endpoint{}, fmt.Errorf("model %s is not loaded", modelID)
	}
	if li.State != StateReady && li.State != StateBusy {
		return Endpoint{}, fmt.Errorf("model %s is %s, not ready", modelID, li.State)
	}
	alias := li.ModelAlias
	return Endpoint{
		URL:    fmt.Sprintf("http://127.0.0.1:%d", li.Port),
		APIKey: li.apiKey,
		Alias:  alias,
	}, nil
}

// Touch bumps activity and request counters (proxy/chat bookkeeping).
func (m *Manager) Touch(modelID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if li, ok := m.instances[modelID]; ok {
		li.LastActivity = time.Now()
		li.Requests++
	}
}

// ResolveRuntime picks the runtime for a model:
// explicit override > model pin > global preferred > any healthy install.
func (m *Manager) ResolveRuntime(modelID, override string) (*runtimes.Runtime, error) {
	if override != "" {
		rt, err := m.runtimes.Get(override)
		if err != nil {
			return nil, fmt.Errorf("selected runtime %q: %w", override, err)
		}
		return rt, nil
	}
	mdl, err := m.library.Get(modelID)
	if err != nil {
		return nil, err
	}
	if mdl.PinnedRuntime != "" {
		return m.runtimes.Get(mdl.PinnedRuntime)
	}
	rt, err := m.runtimes.Preferred()
	if err != nil || rt != nil {
		return rt, err
	}
	// Fall back to any healthy runtime.
	all, err := m.runtimes.List()
	if err != nil {
		return nil, err
	}
	for i := range all {
		if all[i].Healthy {
			return &all[i], nil
		}
	}
	if len(all) > 0 {
		return &all[0], nil
	}
	return nil, errors.New("no llama.cpp runtime installed; install one from the Runtimes page")
}

// Preview builds the command without starting anything (command preview UI).
func (m *Manager) Preview(modelID string, s LoadSettings) (BuildResult, error) {
	mdl, err := m.library.Get(modelID)
	if err != nil {
		return BuildResult{}, err
	}
	rt, err := m.ResolveRuntime(modelID, s.RuntimeID)
	if err != nil {
		return BuildResult{}, err
	}
	help, _ := m.runtimes.HelpOutput(rt.ID)
	caps := rt.Capabilities
	if help != "" {
		// Prefer live --help so newly recognized flags (e.g. --spec-type) work
		// even when runtime_capabilities was snapshotted on an older build.
		caps = runtimes.ParseCapabilities(help)
	}
	br := BuildArgs(s, mdl.PrimaryPath, mdl.ProjectorPath, caps, help,
		"127.0.0.1", 0, "<generated-per-process>")
	br.Resolutions = append([]Resolution{{
		Setting: "Runtime", Auto: s.RuntimeID, Resolved: rt.ID + " (" + rt.Backend + ")",
	}}, br.Resolutions...)
	return br, nil
}

// ResolveModelID maps a public model name (model ID, instance alias, or
// model alias) to a loaded model ID for proxy routing.
func (m *Manager) ResolveModelID(name string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for modelID, li := range m.instances {
		if li.State != StateReady && li.State != StateBusy {
			continue
		}
		if modelID == name || li.ModelAlias == name || li.ID == name {
			return modelID, nil
		}
	}
	return "", fmt.Errorf("no loaded model matches %q", name)
}

// LoadedModels lists names of ready instances for /v1/models.
func (m *Manager) LoadedModels() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := []string{}
	for _, li := range m.instances {
		if li.State == StateReady || li.State == StateBusy {
			out = append(out, li.ModelAlias)
		}
	}
	return out
}

// allocatePort binds :0 on loopback and releases it, returning the port.
// The small race before the child binds is handled by llama-server retries
// and classified as port-conflict if it ever loses.
func allocatePort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port, nil
}

func genAPIKey() string {
	var b [24]byte
	_, _ = rand.Read(b[:])
	return "oi-" + hex.EncodeToString(b[:])
}

// Start loads a model with the given settings. If already loading/ready it
// returns the current instance.
func (m *Manager) Start(modelID string, s LoadSettings) (*Instance, error) {
	m.mu.Lock()
	if li, ok := m.instances[modelID]; ok {
		st := li.State
		m.mu.Unlock()
		if st == StateStarting || st == StateLoading || st == StateReady || st == StateBusy {
			cp := li.Instance
			return &cp, nil
		}
		// failed/crashed/stopping: fall through and restart fresh
		m.mu.Lock()
		delete(m.instances, modelID)
	}
	loaded := 0
	for _, li := range m.instances {
		if li.State == StateReady || li.State == StateBusy || li.State == StateLoading || li.State == StateStarting {
			loaded++
		}
	}
	if loaded >= m.maxLoaded {
		m.mu.Unlock()
		return nil, fmt.Errorf("maximum loaded models (%d) reached; unload one first", m.maxLoaded)
	}
	m.mu.Unlock()

	mdl, err := m.library.Get(modelID)
	if err != nil {
		return nil, err
	}
	rt, err := m.ResolveRuntime(modelID, s.RuntimeID)
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(rt.ExecutablePath); err != nil {
		return nil, fmt.Errorf("runtime %s executable missing (%s); reinstall or pick another runtime", rt.ID, rt.ExecutablePath)
	}

	port, err := allocatePort()
	if err != nil {
		return nil, err
	}
	apiKey := genAPIKey()

	var meta struct {
		IsDiffusion  bool   `json:"is_diffusion"`
		CanvasLength uint32 `json:"canvas_length"`
	}
	_ = json.Unmarshal(mdl.Metadata, &meta)
	// Path/arch fallback when library metadata is stale (pre-schema-7).
	if !meta.IsDiffusion {
		if isDiff, canvas := detectDiffusionFallback(mdl); isDiff {
			meta.IsDiffusion = true
			meta.CanvasLength = canvas
		}
	}

	var br BuildResult
	if meta.IsDiffusion {
		diffExe, err := ResolveDiffusionBinary(rt.ExecutablePath)
		if err != nil {
			return nil, err
		}
		ngl := resolveNGL(s)
		br = BuildResult{
			Args:    []string{mdl.PrimaryPath},
			Command: fmt.Sprintf("%s %s  (NGL=%d MAXTOK=auto)", filepath.Base(diffExe), filepath.Base(mdl.PrimaryPath), ngl),
			Resolutions: []Resolution{
				{Setting: "Runner", Auto: "llama-server", Resolved: filepath.Base(diffExe)},
				{Setting: "GPU offload", Auto: s.GPUOffload, Resolved: fmt.Sprintf("NGL=%d", ngl)},
			},
			Warnings: []string{"block-diffusion model: using llama-diffusion visual server (not autoregressive llama-server)"},
		}
	} else {
		help, _ := m.runtimes.HelpOutput(rt.ID)
		caps := rt.Capabilities
		if help != "" {
			caps = runtimes.ParseCapabilities(help)
		}
		br = BuildArgs(s, mdl.PrimaryPath, mdl.ProjectorPath, caps, help, "127.0.0.1", port, apiKey)
	}

	instID := uuid.NewString()
	alias := s.Alias
	if alias == "" {
		alias = mdl.Alias
	}
	li := &liveInstance{
		Instance: Instance{
			ID: instID, ModelID: modelID, ModelAlias: alias, RuntimeID: rt.ID,
			Port: port, State: StateStarting, Args: br.Args, Command: br.Command,
			Warnings: br.Warnings, StartedAt: time.Now().UTC(), LastActivity: time.Now().UTC(),
			Backend: rt.Backend, ExitCode: -1,
			LogFile: filepath.Join(m.logDir, modelID+".log"),
		},
		apiKey: apiKey,
		done:   make(chan struct{}),
	}
	m.mu.Lock()
	m.instances[modelID] = li
	m.mu.Unlock()
	m.setState(li, StateStarting)
	m.emit(li)

	_, _ = m.db.Exec(`INSERT INTO instances(id,model_id,runtime_id,pid,port,state,args_json,started_at,created_at)
		VALUES (?,?,?,?,?,?,?,?,?)`,
		instID, modelID, rt.ID, 0, port, StateStarting, mustJSON(br.Args), nowUTC(), nowUTC())

	if meta.IsDiffusion {
		go m.superviseDiffusion(li, rt, s, mdl, meta.CanvasLength)
	} else {
		go m.supervise(li, rt, br, s)
	}
	cp := li.Instance
	return &cp, nil
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func nowUTC() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// supervise launches the process and watches readiness and lifetime.
func (m *Manager) supervise(li *liveInstance, rt *runtimes.Runtime, br BuildResult, s LoadSettings) {
	logFile, err := os.OpenFile(li.LogFile, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		m.failInstance(li, -1, fmt.Errorf("opening instance log: %w", err))
		return
	}
	defer logFile.Close()

	// Live streams (log tail for consoles, /slots activity monitor) run
	// for the lifetime of this supervise call.
	actx, acancel := context.WithCancel(context.Background())
	defer acancel()
	go m.streamLogs(actx, li)

	fmt.Fprintf(logFile, "%s starting %s ===\n", launchMarker(li.ID), li.StartedAt.Format(time.RFC3339))
	fmt.Fprintf(logFile, "runtime: %s (%s %s)\ncommand: %s\n", rt.ID, rt.Backend, rt.VersionOutput,
		diagnostics.Redact(strings.Join(br.Args, " ")))

	// App-owned environment (temp/cache dirs) is trusted and applied directly;
	// only user-supplied overrides go through the allowlist filter.
	env := defaultEnv(m.tempDir, m.cacheDir)
	userEnv, rejected := FilterEnv(s.EnvOverrides)
	for k, v := range userEnv {
		env[k] = v
	}
	for _, k := range rejected {
		fmt.Fprintf(logFile, "note: env override %s rejected by allowlist\n", k)
	}
	// Put the runtime's own directory on the library search path: official
	// llama.cpp archives ship libllama-*.so / DLLs next to the binaries.
	for _, kv := range runtimes.LibPathEnv(rt.ExecutablePath) {
		k, v, _ := strings.Cut(kv, "=")
		env[k] = v
	}

	handle, err := processes.Start(processes.Spec{
		Exe:  rt.ExecutablePath,
		Args: br.Args,
		Dir:  filepath.Dir(rt.ExecutablePath),
		Env:  env,
	}, logFile, logFile)
	if err != nil {
		m.failInstance(li, -1, fmt.Errorf("launching llama-server: %w", err))
		return
	}
	li.handle = handle
	li.PID = handle.Cmd.Process.Pid
	_, _ = m.db.Exec(`UPDATE instances SET pid = ? WHERE id = ?`, li.PID, li.ID)
	m.setState(li, StateLoading)
	m.emit(li)

	// Readiness probe in background; process exit in foreground.
	ctx, cancel := context.WithTimeout(context.Background(), m.startupTimeout)
	li.cancel = cancel
	readyCh := make(chan error, 1)
	go m.probeReady(ctx, li, readyCh)

	exitCh := make(chan int, 1)
	go func() {
		code, _ := handle.Wait()
		exitCh <- code
	}()

	select {
	case err := <-readyCh:
		cancel()
		if err != nil {
			m.failInstance(li, -2, err) // -2 = never became ready
			handle.KillTree()
			<-exitCh
			return
		}
		m.setState(li, StateReady)
		m.library.RecordLoad(li.ModelID, li.RuntimeID, "ok")
		go m.monitorActivity(actx, li)
		// Persist last-known-good configuration when requested. Only on
		// success — failures keep the previous good preset.
		if s.SaveOnSuccess {
			if cfg, err := json.Marshal(s); err == nil {
				if err := m.library.SaveLastGood(li.ModelID, cfg); err != nil {
					m.log.Warn("saving last-known-good preset failed", "err", err)
				}
			}
		}
		m.emit(li)
		// Wait for exit after ready.
		code := <-exitCh
		m.handleExit(li, code)
	case code := <-exitCh:
		cancel()
		m.handleExit(li, code)
	}
	close(li.done)
}

// probeReady polls /health until the server reports readiness.
func (m *Manager) probeReady(ctx context.Context, li *liveInstance, out chan<- error) {
	url := fmt.Sprintf("http://127.0.0.1:%d/health", li.Port)
	client := &http.Client{Timeout: 3 * time.Second}
	deadline := time.Now().Add(m.startupTimeout)
	for {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		req.Header.Set("Authorization", "Bearer "+li.apiKey)
		resp, err := client.Do(req)
		if err == nil {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				out <- nil
				return
			}
			// 503 = still loading; keep waiting. Log the body for diagnostics.
			_ = body
		}
		if time.Now().After(deadline) {
			out <- fmt.Errorf("model did not become ready within %s", m.startupTimeout)
			return
		}
		select {
		case <-ctx.Done():
			out <- ctx.Err()
			return
		case <-time.After(500 * time.Millisecond):
		}
	}
}

// detectDiffusionFallback uses path/arch when metadata_json lacks is_diffusion.
func detectDiffusionFallback(mdl *models.Model) (bool, uint32) {
	arch := strings.ToLower(mdl.Architecture)
	var raw map[string]any
	_ = json.Unmarshal(mdl.Metadata, &raw)
	return importDetectDiffusion(arch, mdl.Alias, mdl.PrimaryPath, raw)
}

// superviseDiffusion launches llama-diffusion-gemma-visual-server behind an
// OpenAI-compatible HTTP shim so chat/readiness keep working.
func (m *Manager) superviseDiffusion(li *liveInstance, rt *runtimes.Runtime, s LoadSettings, mdl *models.Model, canvas uint32) {
	logFile, err := os.OpenFile(li.LogFile, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		m.failInstance(li, -1, fmt.Errorf("opening instance log: %w", err))
		return
	}
	defer logFile.Close()

	actx, acancel := context.WithCancel(context.Background())
	defer acancel()
	go m.streamLogs(actx, li)

	diffExe, err := ResolveDiffusionBinary(rt.ExecutablePath)
	if err != nil {
		m.failInstance(li, -1, err)
		return
	}

	fmt.Fprintf(logFile, "%s starting %s ===\n", launchMarker(li.ID), li.StartedAt.Format(time.RFC3339))
	fmt.Fprintf(logFile, "runtime: %s (%s %s)\ndiffusion-server: %s\nmodel: %s\n",
		rt.ID, rt.Backend, rt.VersionOutput, diffExe, mdl.PrimaryPath)

	env := defaultEnv(m.tempDir, m.cacheDir)
	userEnv, rejected := FilterEnv(s.EnvOverrides)
	for k, v := range userEnv {
		env[k] = v
	}
	for _, k := range rejected {
		fmt.Fprintf(logFile, "note: env override %s rejected by allowlist\n", k)
	}
	for _, kv := range runtimes.LibPathEnv(rt.ExecutablePath) {
		k, v, _ := strings.Cut(kv, "=")
		env[k] = v
	}

	ngl := resolveNGL(s)
	maxTok := s.ContextLength // may be 0 → auto
	if canvas == 0 {
		canvas = 256
	}

	m.setState(li, StateLoading)
	m.emit(li)

	engine, err := StartDiffusionEngine(DiffusionLaunch{
		Exe:       diffExe,
		ModelPath: mdl.PrimaryPath,
		WorkDir:   filepath.Dir(diffExe),
		Env:       env,
		NGL:       ngl,
		MaxTok:    maxTok,
		Canvas:    canvas,
		Log:       logFile,
		TempDir:   m.tempDir,
	})
	if err != nil {
		m.failInstance(li, -1, fmt.Errorf("starting diffusion visual server: %w", err))
		return
	}
	li.diffEngine = engine
	li.handle = engine.handle
	li.PID = engine.PID()
	_, _ = m.db.Exec(`UPDATE instances SET pid = ? WHERE id = ?`, li.PID, li.ID)

	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", li.Port))
	if err != nil {
		_ = engine.Close()
		m.failInstance(li, -1, fmt.Errorf("binding diffusion shim: %w", err))
		return
	}
	shim := StartDiffusionShim(ln, engine, li.apiKey, li.ModelAlias)
	li.diffShim = shim

	m.setState(li, StateReady)
	m.library.RecordLoad(li.ModelID, li.RuntimeID, "ok")
	if s.SaveOnSuccess {
		if cfg, err := json.Marshal(s); err == nil {
			if err := m.library.SaveLastGood(li.ModelID, cfg); err != nil {
				m.log.Warn("saving last-known-good preset failed", "err", err)
			}
		}
	}
	m.emit(li)

	exitCh := make(chan int, 1)
	go func() {
		code, _ := engine.Wait()
		exitCh <- code
	}()

	code := <-exitCh
	_ = shim.Close()
	m.handleExit(li, code)
	close(li.done)
}

// launchMarker starts each launch's section in the append-only log file.
func launchMarker(id string) string { return "=== openinfer instance " + id }

// LogSectionForLaunch trims a log tail to the most recent launch of the
// given instance. The file is append-only across launches, and stale
// errors from earlier runs must not influence crash classification.
func LogSectionForLaunch(tail, instID string) string {
	if i := strings.LastIndex(tail, launchMarker(instID)); i >= 0 {
		return tail[i:]
	}
	return tail
}

// handleExit classifies and records a process exit.
func (m *Manager) handleExit(li *liveInstance, code int) {
	m.mu.Lock()
	st := li.State
	m.mu.Unlock()
	if st == StateStopping {
		m.setState(li, StateUnloaded)
		li.ExitCode = code
		_, _ = m.db.Exec(`UPDATE instances SET state=?, exit_code=?, stopped_at=? WHERE id=?`,
			StateUnloaded, code, nowUTC(), li.ID)
		m.emit(li)
		// Unloaded instances are removed from the active map.
		m.mu.Lock()
		delete(m.instances, li.ModelID)
		m.mu.Unlock()
		return
	}
	// Unexpected exit → crashed (or failed during load).
	tail := LogSectionForLaunch(readLogTail(li.LogFile, 64<<10), li.ID)
	class := diagnostics.Classify(tail, code, false)
	newState := StateCrashed
	if st == StateStarting || st == StateLoading {
		newState = StateFailed
	}
	m.mu.Lock()
	li.ExitCode = code
	li.Failure = class.Summary
	li.FailureClass = string(class.Class)
	m.mu.Unlock()
	m.setState(li, newState)
	m.library.RecordLoad(li.ModelID, li.RuntimeID, newState+": "+string(class.Class))
	_, _ = m.db.Exec(`UPDATE instances SET state=?, exit_code=?, failure=?, stopped_at=? WHERE id=?`,
		newState, code, class.Summary, nowUTC(), li.ID)
	_, _ = m.db.Exec(`INSERT INTO diagnostic_events(id,kind,severity,summary,detail,created_at)
		VALUES (?,?,?,?,?,?)`, uuid.NewString(), "instance."+newState, "error",
		class.Summary, diagnostics.Redact(tail[len(tail)-min(len(tail), 8192):]), nowUTC())
	m.emit(li)
}

// Stop requests graceful unload; force kills the tree.
func (m *Manager) Stop(modelID string, force bool) error {
	m.mu.Lock()
	li, ok := m.instances[modelID]
	m.mu.Unlock()
	if !ok {
		return nil // already unloaded
	}
	if li.State == StateFailed || li.State == StateCrashed {
		// Dead already; just drop the record.
		m.mu.Lock()
		delete(m.instances, modelID)
		m.mu.Unlock()
		m.emit(li)
		return nil
	}
	m.setState(li, StateStopping)
	m.emit(li)
	if li.cancel != nil {
		li.cancel()
	}
	if li.diffShim != nil {
		_ = li.diffShim.Close()
	}
	if li.diffEngine != nil {
		if force {
			li.diffEngine.forceKill()
			return nil
		}
		go func() {
			_ = li.diffEngine.Close()
		}()
		go func() {
			select {
			case <-li.done:
			case <-time.After(20 * time.Second):
				li.diffEngine.forceKill()
			}
		}()
		return nil
	}
	if li.handle != nil {
		if force {
			return li.handle.KillTree()
		}
		if err := li.handle.Signal(); err != nil {
			return li.handle.KillTree()
		}
		// Escalate to kill if graceful stop takes too long.
		go func() {
			select {
			case <-li.done:
			case <-time.After(15 * time.Second):
				_ = li.handle.KillTree()
			}
		}()
	}
	return nil
}

// Restart unloads then reloads with the same or new settings.
func (m *Manager) Restart(modelID string, s LoadSettings) (*Instance, error) {
	_ = m.Stop(modelID, false)
	// Wait briefly for teardown.
	for i := 0; i < 100; i++ {
		m.mu.Lock()
		_, ok := m.instances[modelID]
		m.mu.Unlock()
		if !ok {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	return m.Start(modelID, s)
}

// StopAll terminates every instance (backend shutdown).
func (m *Manager) StopAll() {
	m.mu.Lock()
	ids := make([]string, 0, len(m.instances))
	for id := range m.instances {
		ids = append(ids, id)
	}
	m.mu.Unlock()
	for _, id := range ids {
		_ = m.Stop(id, true)
	}
}

// Logs returns the tail of an instance log (redacted).
func (m *Manager) Logs(modelID string, maxBytes int64) (string, error) {
	m.mu.Lock()
	li, ok := m.instances[modelID]
	m.mu.Unlock()
	path := ""
	if ok {
		path = li.LogFile
	} else {
		path = filepath.Join(m.logDir, modelID+".log")
	}
	if maxBytes <= 0 || maxBytes > 1<<20 {
		maxBytes = 256 << 10
	}
	return readLogTail(path, maxBytes), nil
}

func (m *Manager) setState(li *liveInstance, st string) {
	m.mu.Lock()
	li.State = st
	m.mu.Unlock()
	m.events.Publish("instance.state_changed", map[string]any{
		"model_id": li.ModelID, "instance_id": li.ID, "state": st,
		"failure": li.Failure, "failure_class": li.FailureClass,
	})
}

func (m *Manager) emit(li *liveInstance) {
	m.events.Publish("instance.updated", li.Instance)
}

func (m *Manager) failInstance(li *liveInstance, code int, err error) {
	m.mu.Lock()
	li.ExitCode = code
	li.Failure = err.Error()
	li.FailureClass = string(diagnostics.Classify(err.Error(), code, strings.Contains(err.Error(), "ready")).Class)
	m.mu.Unlock()
	m.setState(li, StateFailed)
	m.library.RecordLoad(li.ModelID, li.RuntimeID, "failed: "+err.Error())
	_, _ = m.db.Exec(`UPDATE instances SET state=?, exit_code=?, failure=?, stopped_at=? WHERE id=?`,
		StateFailed, code, err.Error(), nowUTC(), li.ID)
	close(li.done)
}

// defaultEnv builds the app-owned environment for child processes.
func defaultEnv(tempDir, cacheDir string) map[string]string {
	return map[string]string{
		"TMPDIR":          tempDir,
		"TEMP":            tempDir,
		"TMP":             tempDir,
		"XDG_CACHE_HOME":  cacheDir,
		"LLAMA_CACHE":     filepath.Join(cacheDir, "llama"),
		"GGML_CACHE_PATH": filepath.Join(cacheDir, "llama"),
	}
}

// readLogTail returns the last maxBytes of a log file.
func readLogTail(path string, maxBytes int64) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return ""
	}
	start := int64(0)
	if st.Size() > maxBytes {
		start = st.Size() - maxBytes
	}
	buf := make([]byte, st.Size()-start)
	if _, err := f.ReadAt(buf, start); err != nil && err != io.EOF {
		return ""
	}
	return string(buf)
}

// ScanLogLines streams appended lines of the instance log to out until ctx
// is canceled. Existing content is skipped (use Logs for history).
func (m *Manager) ScanLogLines(ctx context.Context, modelID string, out chan<- string) error {
	m.mu.Lock()
	li, ok := m.instances[modelID]
	m.mu.Unlock()
	if !ok {
		return fmt.Errorf("model %s is not loaded", modelID)
	}
	return m.scanLogLines(ctx, li, out)
}

func (m *Manager) scanLogLines(ctx context.Context, li *liveInstance, out chan<- string) error {
	f, err := os.Open(li.LogFile)
	if err != nil {
		return err
	}
	defer f.Close()
	offset, _ := f.Seek(0, io.SeekEnd)
	reader := bufio.NewReader(f)
	var pending string
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}
		line, err := reader.ReadString('\n')
		if err == nil {
			select {
			case out <- pending + strings.TrimRight(line, "\n"):
			case <-ctx.Done():
				return nil
			}
			pending = ""
			offset += int64(len(line))
			continue
		}
		pending += line
		// Handle truncation/rotation.
		if st, statErr := f.Stat(); statErr == nil && st.Size() < offset {
			offset = 0
			f.Seek(0, io.SeekStart)
			reader = bufio.NewReader(f)
		}
		time.Sleep(300 * time.Millisecond)
	}
}
