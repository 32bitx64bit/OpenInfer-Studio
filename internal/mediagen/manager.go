// Package mediagen runs one-shot stable-diffusion.cpp generations.
//
// Design: sd.cpp's sd-server is a single-model daemon with an OpenAI-style
// image API plus a native async sdcpp API (POST /sdcpp/v1/img_gen,
// POST /sdcpp/v1/vid_gen, GET /sdcpp/v1/jobs/{id}). The manager below
// supervises one sd-server per loaded diffusion model (same loopback +
// random port + readiness model as instances.Manager) and submits async
// sdcpp jobs, polling until completion. Small-file jobs (single PNG) are
// stored in the managed media dir; video containers land there too.
//
// Only flags the sd-server --help advertises are ever passed (AGENTS.md
// rule). Model weights are always argv values, never shell strings.
package mediagen

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/processes"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

// Modality of one media job.
const (
	KindImage = "image"
	KindVideo = "video"
)

// Job states.
const (
	StateQueued   = "queued"
	StateRunning  = "running"
	StateComplete = "complete"
	StateFailed   = "failed"
	StateCanceled = "canceled"
)

// LoadSettings is the sd-server launch configuration for one diffusion
// model. Zero values select automatic defaults. Only fields the runtime's
// --help advertises are emitted.
type LoadSettings struct {
	Threads             int    `json:"threads"`               // 0 = auto
	VAE                 string `json:"vae"`                   // optional standalone VAE path
	TAESD               string `json:"taesd"`                 // optional tiny decoder
	ESRGAN              string `json:"esrgan"`                // optional upscaler
	HiresUpscalersDir   string `json:"hires_upscalers_dir"`   // native model-upscale catalog directory
	ControlNet          string `json:"control_net"`           // optional
	LLM                 string `json:"llm"`                   // LLM text encoder (qwen-image / flux2)
	LLMVision           string `json:"llm_vision"`            // --llm_vision (LLM ViT)
	T5XXL               string `json:"t5xxl"`                 // T5-XXL / UMT5 text encoder (flux / sd3 / wan)
	ClipL               string `json:"clip_l"`                // CLIP-L text encoder
	ClipG               string `json:"clip_g"`                // CLIP-G text encoder
	ClipVision          string `json:"clip_vision"`           // CLIP vision encoder
	Tokenizer           string `json:"tokenizer"`             // tokenizer.json override
	LoraModelDir        string `json:"lora_model_dir"`        // --lora-model-dir
	FlashAttention      bool   `json:"flash_attention"`       // --diffusion-fa / --fa
	VAETiling           bool   `json:"vae_tiling"`            // --vae-tiling
	TemporalTiling      bool   `json:"temporal_tiling"`       // --temporal-tiling (video)
	WeightType          string `json:"weight_type"`           // --type f16|q8_0|q4_K|…
	VAEFormat           string `json:"vae_format"`            // --vae-format auto|flux|sd3|flux2|wan
	Backend             string `json:"backend"`               // --backend assignment
	ParamsBackend       string `json:"params_backend"`        // --params-backend assignment
	MaxVRAM             string `json:"max_vram"`              // --max-vram GiB budget
	AutoFit             string `json:"auto_fit"`              // --auto-fit on|off ("" = runtime default)
	OffloadCPU          bool   `json:"offload_to_cpu"`        // --offload-to-cpu
	Mmap                bool   `json:"mmap"`                  // --mmap
	EagerLoad           bool   `json:"eager_load"`            // --eager-load
	DiffusionConvDirect bool   `json:"diffusion_conv_direct"` // --diffusion-conv-direct
	VAEConvDirect       bool   `json:"vae_conv_direct"`       // --vae-conv-direct
	RawArgs             string `json:"raw_args"`              // extra sd-server flags (validated against --help)
	RuntimeID           string `json:"runtime_id"`            // explicit runtime override
	SaveOnSuccess       bool   `json:"save_on_success"`       // store as "Last known good" once ready
}

// DefaultLoadSettings returns safe automatic defaults. They match the load
// dialog defaults so a one-click auto-start behaves like a dialog start.
func DefaultLoadSettings() LoadSettings {
	return LoadSettings{FlashAttention: true, VAETiling: true, SaveOnSuccess: true}
}

// GenerateParams is one image/video generation request. The zero value
// selects txt2img defaults. Local inputs are encoded only at submission.
type GenerateParams struct {
	Kind             string   `json:"kind"` // image|video ("" = image)
	Prompt           string   `json:"prompt"`
	NegativePrompt   string   `json:"negative_prompt"`
	Width            int      `json:"width"`  // 0 = server default
	Height           int      `json:"height"` // 0 = server default
	Steps            int      `json:"steps"`  // 0 = server default
	CFGScale         float64  `json:"cfg_scale"`
	Seed             int64    `json:"seed"` // -1 = random (sd.cpp default)
	Sampler          string   `json:"sampler"`
	Scheduler        string   `json:"scheduler"`
	BatchCount       int      `json:"batch_count"`
	VideoFrames      int      `json:"video_frames"`
	FPS              int      `json:"fps"`
	Guidance         float64  `json:"guidance"`        // distilled guidance (FLUX-class); 0 = server default
	OutputFormat     string   `json:"output_format"`   // png|jpeg|webp|webm|avi
	InitImage        string   `json:"init_image"`      // base64 or data URL (img2img)
	InitImagePath    string   `json:"init_image_path"` // local image file (img2img); read + encoded by the backend
	Strength         float64  `json:"strength"`        // img2img denoising strength
	MaskImage        string   `json:"mask_image,omitempty"`
	MaskImagePath    string   `json:"mask_image_path,omitempty"`
	ControlImage     string   `json:"control_image,omitempty"`
	ControlImagePath string   `json:"control_image_path,omitempty"`
	ControlStrength  float64  `json:"control_strength,omitempty"`
	RefImages        []string `json:"ref_images,omitempty"`
	RefImagePaths    []string `json:"ref_image_paths,omitempty"`
	Lora             []Lora   `json:"lora,omitempty"`
	CFGScaleExplicit bool     `json:"-"` // preserve a caller's explicit zero
	GuidanceExplicit bool     `json:"-"`
}

// Job is the API view of one media generation.
type Job struct {
	Progress    *Progress       `json:"progress,omitempty"`
	ID          string          `json:"id"`
	ModelID     string          `json:"model_id"`
	Kind        string          `json:"kind"`
	State       string          `json:"state"`
	Prompt      string          `json:"prompt"`
	Params      json.RawMessage `json:"params"`
	OutputPath  string          `json:"output_path"`
	Format      string          `json:"output_format"`
	OutputPaths []string        `json:"output_paths,omitempty"`
	Width       int             `json:"width"`
	Height      int             `json:"height"`
	Frames      int             `json:"frames"`
	Seed        int64           `json:"seed"`
	RuntimeID   string          `json:"runtime_id"`
	PID         int             `json:"pid"`
	LogPath     string          `json:"log_path"`
	Result      json.RawMessage `json:"result"`
	Error       string          `json:"error"`
	CreatedAt   string          `json:"created_at"`
	UpdatedAt   string          `json:"updated_at"`
	FinishedAt  string          `json:"finished_at"`
}

type EventSink interface {
	Publish(event string, payload any)
}

func now() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// Server states surfaced by ListServers / media.server_state. "stopped" is
// only ever published as an event (a stopped server is removed from the
// map, not listed).
const (
	ServerStarting = "starting"
	ServerReady    = "ready"
	ServerFailed   = "failed"
	ServerStopped  = "stopped"
)

// Server is one supervised sd-server bound to a single diffusion model.
// All fields are read by ListServers/ServerPort/StopServer from other
// goroutines, so every access (read or write) after the claim is inserted
// into Manager.servers must hold Manager.mu.
type server struct {
	modelID       string
	runtimeID     string
	exe           string
	modelPath     string
	port          int
	nativePort    int // runtime-internal loopback port; never exposed to callers
	handle        *processes.Handle
	logFile       string
	help          string
	caps          []string
	client        *http.Client
	apiKey        string // process-local secret; never persisted or logged
	gateway       *sdGateway
	cacheSnapshot *launchCacheSnapshot
	ready         bool
	starting      bool          // launch in flight (claim in m.servers)
	started       chan struct{} // closed when the launch attempt finishes

	state     string // starting|ready|failed
	errMsg    string
	detail    string             // what a "starting" server is doing, when it is not just loading weights
	cancel    context.CancelFunc // aborts the launch's own work (restoring model files) before the process exists
	launchCtx context.Context
	startedAt string
	updatedAt string
	settings  LoadSettings
	resolved  LoadSettings // settings after local companion pairing (what ran)
}

// ServerView is the API view of one supervised sd-server
// (GET /api/v1/media/servers).
type ServerView struct {
	ModelID   string          `json:"model_id"`
	State     string          `json:"state"` // starting|ready|failed
	Port      int             `json:"port"`
	PID       int             `json:"pid"`
	RuntimeID string          `json:"runtime_id"`
	LogPath   string          `json:"log_path"`
	Error     string          `json:"error"`
	Detail    string          `json:"detail,omitempty"` // progress text while starting, e.g. restoring tensor shapes
	LogTail   string          `json:"log_tail,omitempty"`
	StartedAt string          `json:"started_at"`
	UpdatedAt string          `json:"updated_at"`
	Settings  json.RawMessage `json:"settings,omitempty"`
}

// jobHandle tracks the live state of one in-flight media job: its
// cancellation function, which model/port it is running against, and the
// sd.cpp job id (once submitted) so Cancel can reach the server too.
type jobHandle struct {
	cancel     context.CancelCauseFunc
	modelID    string
	port       int
	sdJobID    string
	progress   *Progress
	onProgress func(Progress)
	terminal   bool
}

// errServerCrashed is the cancellation cause used when a job is aborted
// because its sd-server exited unexpectedly (as opposed to a user Cancel).
// runJob checks this to avoid overwriting the "failed" state that
// failJobsForModel already wrote with a plain "canceled".
var errServerCrashed = errors.New("diffusion server crashed")

// Manager owns supervised sd-servers and media jobs.
type Manager struct {
	db     *sql.DB
	layout *config.Layout
	rt     *runtimes.Manager
	lib    *models.Library
	events EventSink
	log    *slog.Logger
	http   *http.Client

	restoreMu sync.Mutex // one restored-GGUF copy is written at a time

	mu       sync.Mutex
	servers  map[string]*server // keyed by model ID
	jobs     map[string]*jobHandle
	capCache map[string][]string // runtime ID -> parsed sd-server capabilities
}

func NewManager(db *sql.DB, layout *config.Layout, rt *runtimes.Manager, lib *models.Library, events EventSink, log *slog.Logger) *Manager {
	if log == nil {
		log = slog.Default()
	}
	return &Manager{
		db: db, layout: layout, rt: rt, lib: lib, events: events, log: log,
		http:     sdHTTPClient(),
		servers:  map[string]*server{},
		jobs:     map[string]*jobHandle{},
		capCache: map[string][]string{},
	}
}

// MediaDir returns the managed media output directory, creating it.
func (m *Manager) MediaDir() string {
	dir := ""
	if m.layout != nil {
		dir = filepath.Join(m.layout.DataDir, "media")
	} else {
		dir = filepath.Join(os.TempDir(), "openinfer-media")
	}
	_ = os.MkdirAll(dir, 0o755)
	_ = os.MkdirAll(filepath.Join(dir, "images"), 0o755)
	_ = os.MkdirAll(filepath.Join(dir, "videos"), 0o755)
	return dir
}

func allocatePort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port, nil
}

// ResolveRuntime picks the sd.cpp runtime for a model: explicit override >
// model pin (when it points at an sd.cpp runtime) > preferred sd.cpp >
// any sd.cpp runtime.
func (m *Manager) ResolveRuntime(modelID, override string) (*runtimes.Runtime, error) {
	if override != "" {
		rt, err := m.rt.Get(override)
		if err != nil {
			return nil, err
		}
		if !IsSDRuntime(rt) {
			return nil, fmt.Errorf("runtime %s is not a stable-diffusion.cpp runtime", rt.ID)
		}
		return rt, nil
	}
	if modelID != "" && m.lib != nil {
		if mdl, err := m.lib.Get(modelID); err == nil && mdl.PinnedRuntime != "" {
			if rt, err := m.rt.Get(mdl.PinnedRuntime); err == nil && IsSDRuntime(rt) {
				return rt, nil
			}
		}
	}
	if pref, err := m.rt.Preferred(); err == nil && pref != nil && IsSDRuntime(pref) {
		return pref, nil
	}
	all, err := m.rt.List()
	if err != nil {
		return nil, err
	}
	for i := range all {
		if IsSDRuntime(&all[i]) {
			return &all[i], nil
		}
	}
	return nil, fmt.Errorf("no stable-diffusion.cpp runtime installed; install one from the Runtimes page")
}

// EnsureServer starts (or returns) the supervised sd-server for a model.
// Concurrent callers share one launch: an in-flight start is awaited rather
// than double-launching the process.
func (m *Manager) EnsureServer(modelID string, s LoadSettings) (int, error) {
	m.mu.Lock()
	for {
		sv, ok := m.servers[modelID]
		if ok && sv.ready && sv.handle != nil {
			port := sv.port
			m.mu.Unlock()
			return port, nil
		}
		if ok && sv.starting {
			ch := sv.started
			m.mu.Unlock()
			<-ch // launch attempt finished (ready or failed); re-check
			m.mu.Lock()
			if m.servers[modelID] != sv {
				m.mu.Unlock()
				return 0, fmt.Errorf("diffusion server for model %s was stopped during startup", modelID)
			}
			if !sv.ready {
				err := fmt.Errorf("diffusion server startup failed: %s", sv.errMsg)
				m.mu.Unlock()
				return 0, err
			}
			continue
		}
		break
	}
	// Claim the launch slot so parallel callers wait instead of racing.
	previous := m.servers[modelID]
	ctx, cancel := context.WithCancel(context.Background())
	claim := &server{
		modelID: modelID, starting: true, started: make(chan struct{}),
		state: ServerStarting, startedAt: now(), updatedAt: now(),
		launchCtx: ctx, cancel: cancel, settings: s,
	}
	m.servers[modelID] = claim
	m.mu.Unlock()
	if previous != nil {
		m.closeSDServer(previous)
	}

	port, err := m.startServer(modelID, s, claim)
	m.mu.Lock()
	stillOurs := m.servers[modelID] == claim
	if err == nil && (!stillOurs || ctx.Err() != nil || claim.state == ServerFailed) {
		err = fmt.Errorf("diffusion server for model %s was stopped during startup", modelID)
	}
	if err != nil {
		// Keep the failed entry (with its error + log path) so the Library
		// can show diagnostics, same as a failed llama-server instance. It
		// is cleared on the next start or stop of this model.
		claim.starting = false
		claim.ready = false
		claim.state = ServerFailed
		claim.errMsg = err.Error()
		claim.detail = ""
		claim.port = 0
		claim.updatedAt = now()
		m.mu.Unlock()
		m.closeSDServer(claim)
		close(claim.started)
		if stillOurs {
			m.publish("media.server_state", map[string]any{"model_id": modelID, "state": ServerFailed, "error": err.Error()})
		}
		return 0, err
	}
	claim.starting = false
	claim.ready = true
	claim.state = ServerReady
	claim.updatedAt = now()
	m.mu.Unlock()
	close(claim.started)
	if s.SaveOnSuccess && m.lib != nil {
		if raw, jerr := json.Marshal(s); jerr == nil {
			if serr := m.lib.SaveLastGood(modelID, raw); serr != nil {
				m.log.Warn("saving last-known-good diffusion settings failed", "model_id", modelID, "err", serr)
			}
		}
	}
	return port, nil
}

// startServer resolves the runtime, launches sd-server, and waits for
// readiness. The claim is filled in place (handle set before the wait so
// StopServer can cancel mid-launch). Local pipeline companions (VAE,
// text encoders, tokenizer) are paired automatically when the settings
// leave them empty.
func (m *Manager) startServer(modelID string, s LoadSettings, sv *server) (port int, retErr error) {
	m.mu.Lock()
	ctx := sv.launchCtx
	m.mu.Unlock()
	var endpoint *sdEndpoint
	var handle *processes.Handle
	defer func() {
		if retErr != nil {
			endpoint.close()
			if handle != nil {
				_ = handle.KillTree()
			}
		}
	}()
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	mdl, err := m.lib.Get(modelID)
	if err != nil {
		return 0, err
	}
	rt, err := m.ResolveRuntime(modelID, s.RuntimeID)
	if err != nil {
		return 0, err
	}
	// PrepareLaunch resolves the sd-server binary (never sd-cli), pairs local
	// pipeline companions, parses capabilities from --help (probing directly
	// when the recorded snapshot is empty/stale rather than launching blind),
	// validates companion paths, and builds the argv — the same helper
	// previewDiffusionLoad uses, so the dialog shows exactly what runs here.
	rawHelp, _ := m.rt.HelpOutput(rt.ID)
	exe, args, resolved, help, caps, warnings, err := PrepareLaunch(rt, rawHelp, mdl.PrimaryPath, s)
	if err != nil {
		return 0, err
	}
	sourceFiles, cacheErr := cacheLaunchFiles(exe, args)
	m.mu.Lock()
	sv.resolved = resolved
	m.mu.Unlock()
	for _, w := range warnings {
		m.log.Warn("sd-server launch warning", "model_id", modelID, "warning", w)
	}

	logDir := filepath.Join(m.MediaDir(), "logs")
	_ = os.MkdirAll(logDir, 0o755)
	logPath := filepath.Join(logDir, sanitizeID(modelID)+".log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return 0, err
	}
	defer logFile.Close()
	// GGUFs converted with ComfyUI-GGUF carry reshaped tensors that
	// stable-diffusion.cpp cannot read: swap in restored copies first.
	setDetail := func(detail string) {
		m.mu.Lock()
		if m.servers[modelID] != sv || ctx.Err() != nil {
			m.mu.Unlock()
			return
		}
		changed := sv.detail != detail
		sv.detail = detail
		sv.updatedAt = now()
		m.mu.Unlock()
		if !changed {
			return
		}
		m.publish("media.server_state", map[string]any{"model_id": modelID, "state": ServerStarting, "detail": detail})
	}
	if err := m.restoreReshapedGGUFs(ctx, args, logFile, setDetail); err != nil {
		if ctx.Err() != nil {
			return 0, fmt.Errorf("diffusion server for model %s was stopped during startup", modelID)
		}
		return 0, err
	}
	if err := m.prepareLTXBroadcast(ctx, args, rt.Backend, logFile, setDetail); err != nil {
		if ctx.Err() != nil {
			return 0, fmt.Errorf("diffusion server for model %s was stopped during startup", modelID)
		}
		return 0, err
	}
	setDetail("")
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	args, endpoint, err = configureSDTransport(help, args)
	if err != nil {
		return 0, err
	}
	if err := m.attachSDTransport(sv, endpoint); err != nil {
		return 0, err
	}
	port = endpoint.port
	loadedFiles, loadedCacheErr := cacheLaunchFiles(exe, args)
	if cacheErr == nil && loadedCacheErr == nil {
		m.mu.Lock()
		sv.cacheSnapshot = &launchCacheSnapshot{Settings: resolved, Args: cacheArgs(args), SourceFiles: sourceFiles, LoadedFiles: loadedFiles}
		m.mu.Unlock()
	}
	fmt.Fprintf(logFile, "=== sd-server starting %s model=%s ===\n", now(), mdl.PrimaryPath)
	fmt.Fprintf(logFile, "=== argv: %s ===\n", strings.Join(redactSDArgs(args), " "))

	env := SDLaunchEnvironment(exe, args, rt.Backend)
	if env["GGML_CUDA_DISABLE_GRAPHS"] != "" {
		fmt.Fprintf(logFile, "=== %s (GGML_CUDA_DISABLE_GRAPHS=1) ===\n", ltxHIPGraphWarning)
	}
	h, err := processes.Start(processes.Spec{
		Exe: exe, Args: args, Dir: filepath.Dir(exe), Env: env,
	}, logFile, logFile)
	if err != nil {
		return 0, fmt.Errorf("launching sd-server: %w", err)
	}
	handle = h
	// Reap this exact launch and close only its own gateway. The public
	// listener closes even when the process dies before readiness.
	exited := make(chan struct{})
	go func() {
		_, _ = h.Wait()
		endpoint.close()
		close(exited)
	}()
	m.mu.Lock()
	if m.servers[modelID] != sv || ctx.Err() != nil || sv.state != ServerStarting {
		m.mu.Unlock()
		return 0, context.Canceled
	}
	sv.runtimeID = rt.ID
	sv.exe = exe
	sv.modelPath = mdl.PrimaryPath
	sv.handle = h
	sv.logFile = logPath
	sv.help = help
	sv.caps = caps
	sv.client = sdHTTPClient()
	sv.updatedAt = now()
	m.mu.Unlock()

	if err := m.waitReady(sv, exited, 10*time.Minute); err != nil {
		_ = h.KillTree()
		return 0, err
	}
	// Watch for a ready server exiting later (crash/OOM): a dead "ready"
	// entry would otherwise sit in the map forever and every subsequent job
	// would fail or hang against a closed port. A deliberate StopServer
	// removes the entry from m.servers before killing the process, so this
	// goroutine's map re-check (see watchServerExit) naturally distinguishes
	// a crash from an intentional stop.
	go m.watchServerExit(modelID, sv, exited)
	return port, nil
}

// attachSDTransport refuses to publish a listener for a stopped or replaced
// launch claim. Its caller owns cleanup until startup completes.
func (m *Manager) attachSDTransport(sv *server, endpoint *sdEndpoint) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.servers[sv.modelID] != sv || sv.launchCtx.Err() != nil || sv.state != ServerStarting {
		return context.Canceled
	}
	sv.port, sv.nativePort = endpoint.port, endpoint.nativePort
	sv.apiKey, sv.gateway = endpoint.key, endpoint.gateway
	return nil
}

// closeSDServer captures resources under the manager lock, then tears down
// this exact launch outside it. It cannot close a replacement's gateway.
func (m *Manager) closeSDServer(sv *server) {
	m.mu.Lock()
	gate, cancel, handle := sv.gateway, sv.cancel, sv.handle
	m.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	gate.Close()
	if handle != nil {
		_ = handle.KillTree()
	}
}

// watchServerExit observes a ready sd-server's process exit. If the entry
// is still the live one in Manager.servers (i.e. this was not a deliberate
// StopServer, which removes it first), the exit is a crash: the server is
// marked failed with a log tail, its port is cleared, events fire, and any
// in-flight jobs for the model are failed promptly instead of hanging or
// retrying against a dead port.
func (m *Manager) watchServerExit(modelID string, sv *server, exited <-chan struct{}) {
	// A Linux process can spend minutes writing a core dump before Wait
	// returns. It has already crashed, so fail jobs and release its process
	// group as soon as the kernel reports that state.
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-exited:
			m.handleServerExit(modelID, sv)
			return
		case <-ticker.C:
			m.mu.Lock()
			if m.servers[modelID] != sv || sv.state == ServerFailed {
				m.mu.Unlock()
				return
			}
			h, port, path := sv.handle, sv.port, sv.logFile
			m.mu.Unlock()
			if h.IsCoreDumping() {
				m.abortServer(modelID, port, coreDumpFailure(path))
				return
			}
		}
	}
}

func (m *Manager) handleServerExit(modelID string, sv *server) {
	m.mu.Lock()
	cur, ok := m.servers[modelID]
	if !ok || cur != sv || sv.state == ServerFailed {
		m.mu.Unlock()
		return // stopped deliberately, or already replaced by a newer launch
	}
	logFile := sv.logFile
	m.mu.Unlock()

	errMsg := runtimeFailure(readLogFrom(logFile, 0), logFile)
	if sv.handle != nil && sv.handle.Cmd != nil && sv.handle.Cmd.ProcessState != nil {
		errMsg = sv.handle.Cmd.ProcessState.String() + "\n" + errMsg
	}

	m.mu.Lock()
	cur, ok = m.servers[modelID]
	if !ok || cur != sv || sv.state == ServerFailed {
		m.mu.Unlock()
		return
	}
	sv.ready = false
	sv.state = ServerFailed
	sv.errMsg = errMsg
	sv.port = 0
	sv.updatedAt = now()
	m.mu.Unlock()
	m.closeSDServer(sv)

	m.publish("media.server_error", map[string]any{"model_id": modelID, "error": errMsg})
	m.publish("media.server_state", map[string]any{"model_id": modelID, "state": ServerFailed, "error": errMsg})
	m.failJobsForModel(modelID, "diffusion server crashed: "+errMsg)
}

// failJobsForModel fails every in-flight job for a model promptly (used
// when its sd-server crashes). It cancels each job's context with
// errServerCrashed so runJob's own error handling does not overwrite the
// "failed" state written here with a plain "canceled".
func (m *Manager) failJobsForModel(modelID, reason string) {
	m.mu.Lock()
	var failed *server
	if sv := m.servers[modelID]; sv != nil && sv.state == ServerFailed {
		failed = sv
	}
	var ids []string
	for id, jh := range m.jobs {
		if jh.modelID == modelID && !jh.terminal {
			ids = append(ids, id)
			jh.cancel(errServerCrashed)
		}
	}
	m.mu.Unlock()
	if failed != nil {
		m.closeSDServer(failed)
	}
	for _, id := range ids {
		_ = m.setState(id, StateFailed, "", reason)
		m.publish("media.progress", map[string]any{"id": id, "model_id": modelID, "state": StateFailed, "error": reason, "message": "Diffusion server crashed"})
	}
}

// StartServer launches the sd-server for a model in the background and
// returns immediately. Readiness events: media.server_starting,
// media.server_ready {model_id, port}, media.server_error {model_id, error}.
// Model weights can take minutes to load; callers must not block a request.
func (m *Manager) StartServer(modelID string, s LoadSettings) error {
	if _, err := m.lib.Get(modelID); err != nil {
		return err
	}
	if _, err := m.ResolveRuntime(modelID, s.RuntimeID); err != nil {
		return err
	}
	m.publish("media.server_starting", map[string]any{"model_id": modelID})
	m.publish("media.server_state", map[string]any{"model_id": modelID, "state": ServerStarting})
	go func() {
		port, err := m.EnsureServer(modelID, s)
		if err != nil {
			m.publish("media.server_error", map[string]any{"model_id": modelID, "error": err.Error()})
			// media.server_state was already published by EnsureServer's
			// failure path (it knows the final error precisely).
			return
		}
		m.publish("media.server_ready", map[string]any{"model_id": modelID, "port": port})
		m.publish("media.server_state", map[string]any{"model_id": modelID, "state": ServerReady, "port": port})
	}()
	return nil
}

// waitReady polls /sdcpp/v1/capabilities until the server answers, failing
// immediately when the process exits (with the log tail as the error) —
// model/flag problems surface in seconds instead of the full deadline.
func (m *Manager) waitReady(sv *server, exited <-chan struct{}, timeout time.Duration) error {
	m.mu.Lock()
	port := sv.port
	logFile := sv.logFile
	modelID, modelPath, settings, handle := sv.modelID, sv.modelPath, sv.resolved, sv.handle
	launchCtx := sv.launchCtx
	m.mu.Unlock()
	deadline := time.Now().Add(timeout)
	url := fmt.Sprintf("http://127.0.0.1:%d/sdcpp/v1/capabilities", port)
	for {
		if err := launchCtx.Err(); err != nil {
			return err
		}
		if handle.IsCoreDumping() {
			return errors.New(coreDumpFailure(logFile))
		}
		ctx, cancel := context.WithTimeout(launchCtx, 5*time.Second)
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if resp, err := m.doSD(req, port); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				cancel()
				return nil
			}
		}
		cancel()
		select {
		case <-exited:
			tail, _ := os.ReadFile(logFile)
			msg := startupFailure(string(tail), logFile, modelPath, settings)
			// The per-model sd-server log is not the application log: put the
			// failure where the Logs page shows it too.
			m.log.Error("sd-server exited during startup", "model_id", modelID, "log", logFile, "error", msg)
			return errors.New(msg)
		default:
		}
		if time.Now().After(deadline) {
			tail, _ := os.ReadFile(logFile)
			return fmt.Errorf("sd-server did not become ready within %s:\n%s", timeout, lastLines(string(tail), 8))
		}
		timer := time.NewTimer(300 * time.Millisecond)
		select {
		case <-timer.C:
		case <-launchCtx.Done():
			timer.Stop()
			return launchCtx.Err()
		case <-exited:
			timer.Stop()
		}
	}
}

// StopServer stops the supervised sd-server for a model. Removing the map
// entry before killing the process is what lets watchServerExit tell a
// deliberate stop apart from a crash.
func (m *Manager) StopServer(modelID string) {
	m.mu.Lock()
	sv, ok := m.servers[modelID]
	delete(m.servers, modelID)
	m.mu.Unlock()
	if ok {
		m.closeSDServer(sv)
	}
	m.publish("media.server_stopped", map[string]any{"model_id": modelID})
	m.publish("media.server_state", map[string]any{"model_id": modelID, "state": ServerStopped})
}

// StopAll stops every supervised sd-server. Called on backend shutdown:
// Linux relies on PR_SET_PDEATHSIG today, but macOS/Windows would otherwise
// orphan multi-GB VRAM processes when the backend exits.
func (m *Manager) StopAll() {
	m.mu.Lock()
	ids := make([]string, 0, len(m.servers))
	for id := range m.servers {
		ids = append(ids, id)
	}
	m.mu.Unlock()
	for _, id := range ids {
		m.StopServer(id)
	}
}

// ListServers returns the current view of every supervised sd-server
// (GET /api/v1/media/servers). Failed entries keep their error and log path
// until the next start or stop of that model; the log tail is only read
// for failed entries.
func (m *Manager) ListServers() []ServerView {
	type snap struct {
		modelID, runtimeID, logFile, errMsg, detail, state, startedAt, updatedAt string
		port                                                                     int
		pid                                                                      int
		settings                                                                 LoadSettings
	}
	m.mu.Lock()
	snaps := make([]snap, 0, len(m.servers))
	for id, sv := range m.servers {
		pid := 0
		if sv.handle != nil && sv.handle.Cmd != nil && sv.handle.Cmd.Process != nil {
			pid = sv.handle.Cmd.Process.Pid
		}
		snaps = append(snaps, snap{
			modelID: id, runtimeID: sv.runtimeID, logFile: sv.logFile, errMsg: sv.errMsg, detail: sv.detail,
			state: sv.state, startedAt: sv.startedAt, updatedAt: sv.updatedAt,
			port: sv.port, pid: pid, settings: sv.settings,
		})
	}
	m.mu.Unlock()

	out := make([]ServerView, 0, len(snaps))
	for _, s := range snaps {
		v := ServerView{
			ModelID: s.modelID, State: s.state, Port: s.port, PID: s.pid,
			RuntimeID: s.runtimeID, LogPath: s.logFile, Error: s.errMsg, Detail: s.detail,
			StartedAt: s.startedAt, UpdatedAt: s.updatedAt,
		}
		if b, err := json.Marshal(s.settings); err == nil {
			v.Settings = b
		}
		if s.state == ServerFailed && s.logFile != "" {
			if tail, err := os.ReadFile(s.logFile); err == nil {
				v.LogTail = lastLines(string(tail), 40)
			}
		}
		out = append(out, v)
	}
	return out
}

// ServerPort returns the sd-server port for a model, if running.
func (m *Manager) ServerPort(modelID string) (int, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if sv, ok := m.servers[modelID]; ok && sv.ready {
		return sv.port, true
	}
	return 0, false
}

// Capabilities proxies the sd-server capability document for a model.
func (m *Manager) Capabilities(ctx context.Context, modelID string) (map[string]any, error) {
	return m.fetchCapabilities(ctx, modelID)
}

// StartGenerate validates a request, records a queued media job, and runs
// the generation in the background. It returns the queued job immediately
// so callers can respond 202 without blocking on model load + sampling.
// Progress and completion flow over media.progress events.
func (m *Manager) StartGenerate(modelID string, p GenerateParams) (*Job, error) {
	return m.StartGenerateWith(modelID, p, nil)
}

// StartGenerateWith is StartGenerate for callers that also pin sd-server
// launch inputs (the node-graph executor). When ov is non-nil and the model's
// running server does not already satisfy it, the server is restarted with
// the overrides applied before the job is submitted.
func (m *Manager) StartGenerateWith(modelID string, p GenerateParams, ov *LoadOverrides) (*Job, error) {
	j, _, err := m.startGenerate(modelID, p, ov, nil)
	return j, err
}

// RunStage runs one generation to completion and returns the finished job.
// It is the seam the workflow executor drives: same job row, events and
// cancel path as StartGenerate, but blocking. A canceled ctx cancels the job
// (and best-effort requests runtime cancellation) and returns ctx's error; a failed job returns the
// job alongside an error carrying its message.
func (m *Manager) RunStage(ctx context.Context, modelID string, p GenerateParams, ov *LoadOverrides) (*Job, error) {
	return m.RunStageWithProgress(ctx, modelID, p, ov, nil)
}

// RunStageWithProgress forwards this job's observed progress to the workflow
// executing it. The callback is scoped to the job, so simultaneous generations
// cannot update another workflow node.
func (m *Manager) RunStageWithProgress(ctx context.Context, modelID string, p GenerateParams, ov *LoadOverrides, onProgress func(Progress)) (*Job, error) {
	j, done, err := m.startGenerate(modelID, p, ov, onProgress)
	if err != nil {
		return nil, err
	}
	select {
	case <-done:
	case <-ctx.Done():
		_ = m.Cancel(j.ID)
		<-done
		if final, gerr := m.Get(j.ID); gerr == nil {
			return final, ctx.Err()
		}
		return j, ctx.Err()
	}
	final, err := m.Get(j.ID)
	if err != nil {
		return j, err
	}
	switch final.State {
	case StateComplete:
		return final, nil
	case StateCanceled:
		return final, fmt.Errorf("media job %s canceled", final.ID)
	default:
		msg := final.Error
		if msg == "" {
			msg = "generation " + final.State
		}
		return final, errors.New(msg)
	}
}

// startGenerate validates, records the queued job and runs it in the
// background; the returned channel closes once the job goroutine is done.
func (m *Manager) startGenerate(modelID string, p GenerateParams, ov *LoadOverrides, onProgress func(Progress)) (*Job, <-chan struct{}, error) {
	kind := p.Kind
	if kind == "" {
		kind = KindImage
	}
	if kind != KindImage && kind != KindVideo {
		return nil, nil, fmt.Errorf("invalid kind %q", p.Kind)
	}
	if strings.TrimSpace(p.Prompt) == "" {
		return nil, nil, fmt.Errorf("prompt is required")
	}
	p.RefImages = append([]string(nil), p.RefImages...)
	p.RefImagePaths = append([]string(nil), p.RefImagePaths...)
	p.Lora = append([]Lora(nil), p.Lora...)
	if err := validateGenerationInputs(p); err != nil {
		return nil, nil, err
	}
	ValidateGenerateParams(&p)

	// lib and rt are only ever nil in tests that inject a ready server.
	if m.lib != nil {
		if _, err := m.lib.Get(modelID); err != nil {
			return nil, nil, err
		}
	}
	// Fail fast when no sd.cpp runtime exists (fast DB check) instead of
	// surfacing it minutes later as a job failure.
	if m.rt != nil {
		if _, err := m.ResolveRuntime(modelID, ""); err != nil {
			return nil, nil, err
		}
	}

	id := uuid.NewString()
	body, _ := json.Marshal(p)
	ts := now()
	if _, err := m.db.Exec(`INSERT INTO media_jobs(id,model_id,kind,state,prompt,params_json,seed,created_at,updated_at)
		VALUES (?,?,?,?,?,?,?,?,?)`, id, modelID, kind, StateQueued, p.Prompt, string(body), p.Seed, ts, ts); err != nil {
		return nil, nil, err
	}
	jctx, cancel := context.WithCancelCause(context.Background())
	m.mu.Lock()
	m.jobs[id] = &jobHandle{cancel: cancel, modelID: modelID, onProgress: onProgress}
	m.mu.Unlock()
	done := make(chan struct{})
	go func() {
		defer func() {
			m.mu.Lock()
			delete(m.jobs, id)
			m.mu.Unlock()
			cancel(nil)
			close(done)
		}()
		m.runJob(jctx, id, modelID, kind, p, ov)
	}()
	j, err := m.Get(id)
	if err != nil {
		return nil, nil, err
	}
	return j, done, nil
}

// jobCanceledByCrash reports whether ctx was canceled by failJobsForModel
// rather than a user Cancel: the caller already wrote the DB row and
// published media.progress, so runJob's own handling becomes a no-op.
func jobCanceledByCrash(ctx context.Context) bool {
	return errors.Is(context.Cause(ctx), errServerCrashed)
}

// runJob executes one media generation: auto-start the sd-server when
// needed, submit the sdcpp job, poll to completion, and store the outputs.
// Progress events flow over media.progress.
func (m *Manager) runJob(jctx context.Context, id, modelID, kind string, p GenerateParams, ov *LoadOverrides) {
	port, ok := m.ServerPort(modelID)
	if ok && ov != nil && !m.serverSatisfies(modelID, *ov) {
		// The graph needs different launch inputs (VAE, text encoder, LoRA
		// dir) than the running server has; sd-server only reads them at
		// launch, so replace it.
		m.publish("media.progress", map[string]any{"id": id, "model_id": modelID, "state": StateRunning, "sd_status": "reloading model", "message": "Reloading model with different components…"})
		m.StopServer(modelID)
		ok = false
	}
	if !ok {
		// Auto-start with the model's last-known-good settings when present
		// (same fallback the load dialog uses), else safe defaults, so
		// Image Studio works with one click.
		m.publish("media.progress", map[string]any{"id": id, "model_id": modelID, "state": StateRunning, "sd_status": "loading model", "message": "Loading model…"})
		settings := DefaultLoadSettings()
		if m.lib != nil {
			if raw, ok := m.lib.LastGood(modelID); ok {
				s := DefaultLoadSettings()
				if err := json.Unmarshal(raw, &s); err == nil {
					settings = s
				}
			}
		}
		if ov != nil && !ov.IsZero() {
			ov.Apply(&settings)
			// A graph-shaped launch is not the user's tuned configuration;
			// do not record it as the model's last known good.
			settings.SaveOnSuccess = false
		}
		var err error
		type launchResult struct {
			port int
			err  error
		}
		launched := make(chan launchResult, 1)
		go func() { port, err := m.EnsureServer(modelID, settings); launched <- launchResult{port, err} }()
		select {
		case result := <-launched:
			port, err = result.port, result.err
		case <-jctx.Done():
			if !jobCanceledByCrash(jctx) {
				_ = m.setState(id, StateCanceled, "", "canceled")
			}
			return
		}
		if err != nil {
			msg := err.Error()
			_ = m.setState(id, StateFailed, "", msg)
			m.publish("media.progress", map[string]any{"id": id, "model_id": modelID, "state": StateFailed, "error": msg})
			return
		}
	}
	if jctx.Err() != nil {
		if jobCanceledByCrash(jctx) {
			return
		}
		_ = m.setState(id, StateCanceled, "", "canceled")
		m.publish("media.progress", map[string]any{"id": id, "state": StateCanceled, "message": "Canceled"})
		return
	}

	m.setState(id, StateRunning, "", "")
	runtimeID, logPath, pid := m.serverMeta(modelID)
	_, _ = m.db.Exec(`UPDATE media_jobs SET runtime_id=?, log_path=?, pid=?, updated_at=? WHERE id=?`, runtimeID, logPath, pid, now(), id)
	m.publish("media.progress", map[string]any{"id": id, "model_id": modelID, "state": StateRunning, "message": "Submitting job…"})

	failRequest := func(err error) {
		if jctx.Err() != nil && !jobCanceledByCrash(jctx) {
			_ = m.setState(id, StateCanceled, "", "canceled")
			return
		}
		if jobCanceledByCrash(jctx) {
			return
		}
		_ = m.setState(id, StateFailed, "", err.Error())
		m.publish("media.progress", map[string]any{"id": id, "state": StateFailed, "error": err.Error()})
	}
	apiCaps, err := m.generationCapabilities(jctx, modelID)
	if err != nil {
		failRequest(err)
		return
	}
	if err = apiCaps.ValidateGeneration(p); err != nil {
		failRequest(err)
		return
	}
	settings, _ := m.RunningSettings(modelID)
	reqParams, err := convertPromptLora(p, apiCaps, settings)
	if err != nil {
		failRequest(err)
		return
	}
	reqParams, err = resolveGenerationImages(reqParams)
	if err != nil {
		failRequest(err)
		return
	}
	if !m.serverMatchesPort(modelID, port) {
		failRequest(ErrServerNotRunning)
		return
	}

	endpoint := "/sdcpp/v1/img_gen"
	if kind == KindVideo {
		endpoint = "/sdcpp/v1/vid_gen"
	}
	sdBody := SDRequestBody(reqParams)
	logOffset := logSize(logPath) // what the server logs from here on belongs to this job
	jobID, status, err := m.submitSDJob(jctx, port, endpoint, sdBody)
	if err != nil {
		// Fall back to the synchronous OpenAI-compat endpoint only when the
		// native async API itself is missing (older sd.cpp builds); a 400
		// (bad params) or 429 (queue full) must surface as-is, not retry
		// through a second, different submission.
		if shouldFallbackToOpenAICompat(kind, status) && canFallbackGeneration(reqParams) {
			if _, ferr := m.generateOpenAICompat(jctx, id, modelID, reqParams); ferr != nil {
				msg := firstErr(err, ferr)
				_ = m.setState(id, StateFailed, "", msg)
				m.publish("media.progress", map[string]any{"id": id, "state": StateFailed, "error": msg})
			}
			return
		}
		if shouldFallbackToOpenAICompat(kind, status) && !canFallbackGeneration(reqParams) {
			err = fmt.Errorf("native generation API unavailable; compatibility fallback cannot preserve the requested inputs or sampling controls: %w", err)
		}
		msg := err.Error()
		_ = m.setState(id, StateFailed, "", msg)
		m.publish("media.progress", map[string]any{"id": id, "state": StateFailed, "error": msg})
		return
	}

	m.mu.Lock()
	if jh, ok := m.jobs[id]; ok {
		jh.sdJobID = jobID
		jh.port = port
	}
	m.mu.Unlock()
	if jctx.Err() != nil {
		m.cancelSDJob(port, jobID)
	}

	tracker := newLogProgress(logPath, logOffset)
	result, err := m.pollSDJob(jctx, modelID, port, jobID, func(st string) error {
		progress := tracker.snapshot(id, st)
		if tracker.fatal != "" {
			msg := runtimeFailure(tracker.fatal, logPath)
			m.abortServer(modelID, port, msg)
			return errors.New(msg)
		}
		m.publish("media.progress", map[string]any{"id": id, "model_id": modelID, "state": StateRunning, "sd_status": st, "message": progress.Message, "progress": progress})
		return nil
	})
	if err != nil {
		if jctx.Err() != nil {
			if jobCanceledByCrash(jctx) {
				return
			}
			_ = m.setState(id, StateCanceled, "", "canceled")
			m.publish("media.progress", map[string]any{"id": id, "state": StateCanceled, "message": "Canceled"})
			return
		}
		msg := explainJobFailure(readLogFrom(logPath, logOffset), err.Error())
		_ = m.setState(id, StateFailed, "", msg)
		m.publish("media.progress", map[string]any{"id": id, "state": StateFailed, "error": msg})
		return
	}

	paths, resJSON, err := m.storeSDResult(id, modelID, kind, p, result)
	if err != nil {
		msg := err.Error()
		_ = m.setState(id, StateFailed, "", msg)
		m.publish("media.progress", map[string]any{"id": id, "state": StateFailed, "error": msg})
		return
	}
	_, _ = m.db.Exec(`UPDATE media_jobs SET state=?, output_path=?, result_json=?, width=?, height=?, frames=?, finished_at=?, updated_at=? WHERE id=?`,
		StateComplete, firstPath(paths), string(resJSON), p.Width, p.Height, p.VideoFrames, now(), now(), id)
	m.publish("media.progress", map[string]any{"id": id, "model_id": modelID, "state": StateComplete, "outputs": paths, "message": "Complete"})
}

// serverMeta returns the supervised server's runtime id, log path and OS
// pid for a model (used to fill media_jobs.runtime_id/log_path/pid).
func (m *Manager) serverMeta(modelID string) (runtimeID, logPath string, pid int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	sv, ok := m.servers[modelID]
	if !ok {
		return "", "", 0
	}
	runtimeID = sv.runtimeID
	logPath = sv.logFile
	if sv.handle != nil && sv.handle.Cmd != nil && sv.handle.Cmd.Process != nil {
		pid = sv.handle.Cmd.Process.Pid
	}
	return runtimeID, logPath, pid
}

// RunningSettings returns the settings the model's ready sd-server was
// launched with (after companion pairing), if one is running.
func (m *Manager) RunningSettings(modelID string) (LoadSettings, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	sv, ok := m.servers[modelID]
	if !ok || !sv.ready {
		return LoadSettings{}, false
	}
	if sv.resolved == (LoadSettings{}) {
		return sv.settings, true // server injected without a launch record
	}
	return sv.resolved, true
}

// serverSatisfies reports whether the model's running server already has the
// given launch overrides in effect.
func (m *Manager) serverSatisfies(modelID string, ov LoadOverrides) bool {
	have, ok := m.RunningSettings(modelID)
	return ok && ov.SatisfiedBy(have)
}

// serverMatchesPort reports whether the model's supervised server is still
// ready and bound to port (used by pollSDJob to bail out promptly when the
// server has been stopped or replaced mid-poll).
func (m *Manager) serverMatchesPort(modelID string, port int) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	sv, ok := m.servers[modelID]
	return ok && sv.ready && sv.port == port
}

// shouldFallbackToOpenAICompat reports whether a failed native sdcpp submit
// should retry through the synchronous OpenAI-compat endpoint: only when
// the native endpoint itself is missing (older sd.cpp build), never for a
// request-level rejection like 400 (bad params) or 429 (queue full).
func shouldFallbackToOpenAICompat(kind string, status int) bool {
	return kind == KindImage && (status == http.StatusNotFound || status == http.StatusMethodNotAllowed)
}

// generateOpenAICompat uses POST /v1/images/generations (synchronous,
// base64 b64_json) for older sd.cpp builds.
func (m *Manager) generateOpenAICompat(ctx context.Context, id, modelID string, p GenerateParams) (*Job, error) {
	port, _ := m.ServerPort(modelID)
	size := ""
	if p.Width > 0 && p.Height > 0 {
		size = fmt.Sprintf("%dx%d", p.Width, p.Height)
	}
	payload := map[string]any{"prompt": p.Prompt, "n": maxInt(p.BatchCount, 1)}
	if size != "" {
		payload["size"] = size
	}
	if p.OutputFormat != "" {
		payload["output_format"] = strings.ToLower(p.OutputFormat)
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("http://127.0.0.1:%d/v1/images/generations", port), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := m.doSD(req, port)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out struct {
		Data []struct {
			B64 string `json:"b64_json"`
		} `json:"data"`
		OutputFormat string `json:"output_format"`
		Error        string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("sd-server HTTP %d: %s", resp.StatusCode, out.Error)
	}
	if len(out.Data) == 0 {
		return nil, fmt.Errorf("sd-server returned no images")
	}
	ext := "png"
	if out.OutputFormat != "" {
		ext = strings.ToLower(out.OutputFormat)
	} else if p.OutputFormat != "" {
		ext = strings.ToLower(p.OutputFormat)
	}
	if ext == "jpeg" {
		ext = "jpg"
	}
	if !containsChoice([]string{"png", "jpg", "webp"}, ext) {
		return nil, fmt.Errorf("sd-server returned unsupported image format %q", ext)
	}
	var paths []string
	for i, d := range out.Data {
		raw, err := base64.StdEncoding.DecodeString(d.B64)
		if err != nil {
			return nil, fmt.Errorf("decoding image %d: %w", i, err)
		}
		if len(raw) > 512<<20 {
			return nil, fmt.Errorf("output %d exceeds 512 MiB store limit", i)
		}
		name := fmt.Sprintf("%s-%d.%s", sanitizeID(id), i, ext)
		dest := filepath.Join(m.MediaDir(), "images", name)
		if err := os.WriteFile(dest, raw, 0o644); err != nil {
			return nil, err
		}
		paths = append(paths, filepath.Join("images", name))
	}
	resJSON, _ := json.Marshal(map[string]any{"outputs": paths, "via": "openai-compat"})
	_, _ = m.db.Exec(`UPDATE media_jobs SET state=?, output_path=?, result_json=?, width=?, height=?, finished_at=?, updated_at=? WHERE id=?`,
		StateComplete, firstPath(paths), string(resJSON), p.Width, p.Height, now(), now(), id)
	m.publish("media.progress", map[string]any{"id": id, "model_id": modelID, "state": StateComplete, "outputs": paths})
	return m.Get(id)
}

// submitSDJob posts an async job; returns the sd.cpp job id and the HTTP
// status code (0 on transport error) so callers can tell "endpoint missing"
// (404/405) apart from a request-level rejection (400/429/…).
func (m *Manager) submitSDJob(ctx context.Context, port int, endpoint string, body map[string]any) (string, int, error) {
	ctx, cancel := context.WithTimeout(ctx, sdSubmitRequestTimeout)
	defer cancel()
	payload, err := json.Marshal(body)
	if err != nil {
		return "", 0, fmt.Errorf("encoding sd-server request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("http://127.0.0.1:%d%s", port, endpoint), bytes.NewReader(payload))
	if err != nil {
		return "", 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := m.doSD(req, port)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	var out struct {
		ID      string `json:"id"`
		Error   string `json:"error"`
		Message string `json:"message"`
	}
	decodeErr := json.NewDecoder(resp.Body).Decode(&out)
	if resp.StatusCode != 202 {
		msg := out.Error
		if msg == "" {
			msg = out.Message
		}
		if msg == "" {
			msg = fmt.Sprintf("HTTP %d", resp.StatusCode)
		}
		return "", resp.StatusCode, fmt.Errorf("sd-server submit: %s", msg)
	}
	if decodeErr != nil {
		return "", resp.StatusCode, fmt.Errorf("reading sd-server submission acknowledgement: %w", decodeErr)
	}
	if out.ID == "" {
		return "", resp.StatusCode, fmt.Errorf("sd-server returned no job id")
	}
	return out.ID, resp.StatusCode, nil
}

// sdPollInterval/sdPollErrorBound are package vars (not consts) so tests can
// shrink them instead of waiting out the real ~20s transport-error bound.
var (
	sdPollInterval         = 1500 * time.Millisecond
	sdPollErrorBound       = 20 * time.Second
	sdPollRequestTimeout   = 5 * time.Second
	sdSubmitRequestTimeout = 30 * time.Second
)

// pollSDJob polls GET /sdcpp/v1/jobs/{id} until terminal. It never spins
// forever: it fails fast once the model's supervised server is no longer
// the one it started against, on HTTP 404/410 (job gone), or after
// sdPollErrorBound of consecutive transport errors / missing status.
func (m *Manager) pollSDJob(ctx context.Context, modelID string, port int, jobID string, onTick func(string) error) (map[string]any, error) {
	url := fmt.Sprintf("http://127.0.0.1:%d/sdcpp/v1/jobs/%s", port, jobID)
	var errSince time.Time
	bump := func(reason string) error {
		if errSince.IsZero() {
			errSince = time.Now()
			return nil
		}
		if time.Since(errSince) > sdPollErrorBound {
			return fmt.Errorf("polling sd-server job %s: %s", jobID, reason)
		}
		return nil
	}
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(sdPollInterval):
		}
		if !m.serverMatchesPort(modelID, port) {
			return nil, fmt.Errorf("diffusion server for model %s is no longer running", modelID)
		}
		pollCtx, cancel := context.WithTimeout(ctx, sdPollRequestTimeout)
		req, err := http.NewRequestWithContext(pollCtx, http.MethodGet, url, nil)
		if err != nil {
			cancel()
			return nil, err
		}
		resp, err := m.doSD(req, port)
		if err != nil {
			cancel()
			if onTick != nil {
				if err := onTick("unresponsive"); err != nil {
					return nil, err
				}
			}
			if ferr := bump(err.Error()); ferr != nil {
				return nil, ferr
			}
			continue // transient; keep polling within the error bound
		}
		if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusGone {
			resp.Body.Close()
			cancel()
			return nil, fmt.Errorf("sd-server job %s no longer exists (HTTP %d)", jobID, resp.StatusCode)
		}
		var doc map[string]any
		decodeErr := json.NewDecoder(resp.Body).Decode(&doc)
		resp.Body.Close()
		cancel()
		st, _ := doc["status"].(string)
		if decodeErr != nil || resp.StatusCode != http.StatusOK || st == "" {
			if onTick != nil {
				if err := onTick("unresponsive"); err != nil {
					return nil, err
				}
			}
			reason := "sd-server returned no status"
			if decodeErr != nil {
				reason = "reading sd-server job response: " + decodeErr.Error()
			} else if resp.StatusCode != http.StatusOK {
				reason = fmt.Sprintf("sd-server job poll returned HTTP %d", resp.StatusCode)
			}
			if ferr := bump(reason); ferr != nil {
				return nil, ferr
			}
			continue
		}
		errSince = time.Time{}
		if onTick != nil {
			if err := onTick(st); err != nil {
				return nil, err
			}
		}
		switch st {
		case "completed":
			return doc, nil
		case "failed", "cancelled":
			msg := "generation failed"
			if text, ok := doc["error"].(string); ok && text != "" {
				msg = text
			}
			if e, ok := doc["error"].(map[string]any); ok {
				if s, ok := e["message"].(string); ok && s != "" {
					msg = s
				}
			}
			return nil, fmt.Errorf("%s", msg)
		case "queued", "generating":
			// progress; keep polling
		}
	}
}

// storeSDResult decodes the completed sdcpp payload into the media dir.
func (m *Manager) storeSDResult(id, modelID, kind string, p GenerateParams, doc map[string]any) ([]string, json.RawMessage, error) {
	res, _ := doc["result"].(map[string]any)
	if res == nil {
		return nil, nil, fmt.Errorf("sd-server returned no result")
	}
	// Image jobs: result.images[] = {b64_json}. Video jobs: result.b64_json
	// container + mime_type/output_format/fps/frame_count.
	var blobs []string
	var mime, outFmt string
	if arr, ok := res["images"].([]any); ok {
		for _, item := range arr {
			if obj, ok := item.(map[string]any); ok {
				if b, ok := obj["b64_json"].(string); ok && b != "" {
					blobs = append(blobs, b)
				}
			}
		}
		outFmt, _ = res["output_format"].(string)
	}
	if b, ok := res["b64_json"].(string); ok && b != "" {
		blobs = append(blobs, b)
		mime, _ = res["mime_type"].(string)
		outFmt, _ = res["output_format"].(string)
	}
	if len(blobs) == 0 {
		return nil, nil, fmt.Errorf("sd-server returned no media payload")
	}
	sub := "images"
	ext := "png"
	if kind == KindVideo {
		sub = "videos"
		ext = "webm"
	}
	if outFmt != "" {
		ext = strings.ToLower(outFmt)
	} else if p.OutputFormat != "" {
		ext = strings.ToLower(p.OutputFormat)
	}
	if ext == "jpeg" {
		ext = "jpg"
	}
	formats := []string{"png", "jpg", "webp"}
	if kind == KindVideo {
		formats = []string{"webm", "avi", "mp4"}
	}
	if !containsChoice(formats, ext) {
		return nil, nil, fmt.Errorf("sd-server returned unsupported output format %q", ext)
	}
	var paths []string
	for i, b := range blobs {
		raw, err := base64.StdEncoding.DecodeString(b)
		if err != nil {
			// Some builds emit data URLs.
			if j := strings.Index(b, ","); j >= 0 {
				raw, err = base64.StdEncoding.DecodeString(b[j+1:])
			}
			if err != nil {
				return nil, nil, fmt.Errorf("decoding output %d: %w", i, err)
			}
		}
		// Cap stored outputs: base64 payloads expand ~4/3 on disk.
		if len(raw) > 512<<20 {
			return nil, nil, fmt.Errorf("output %d exceeds 512 MiB store limit", i)
		}
		dest := filepath.Join(m.MediaDir(), sub, fmt.Sprintf("%s-%d.%s", sanitizeID(id), i, ext))
		if err := os.WriteFile(dest, raw, 0o644); err != nil {
			return nil, nil, err
		}
		paths = append(paths, filepath.Join(sub, filepath.Base(dest)))
	}
	meta := map[string]any{"outputs": paths, "via": "sdcpp", "mime": mime, "format": ext}
	if fps, ok := res["fps"]; ok {
		meta["fps"] = fps
	}
	if fc, ok := res["frame_count"]; ok {
		meta["frame_count"] = fc
	}
	resJSON, _ := json.Marshal(meta)
	return paths, resJSON, nil
}

// Cancel cancels a queued/running media job. For a job still tracked in
// memory it cancels the Go context (stopping polling) and best-effort posts
// the sd.cpp cancel endpoint. GPU abort depends on the runtime's advertised
// cancel_generating capability; some builds can only cancel queued jobs.
// A job id from before a backend restart (no longer in memory) still gets
// its DB row marked canceled if it is non-terminal.
func (m *Manager) Cancel(id string) error {
	m.mu.Lock()
	jh, ok := m.jobs[id]
	var cancel context.CancelCauseFunc
	var port int
	var sdJobID string
	if ok && jh != nil {
		cancel, port, sdJobID = jh.cancel, jh.port, jh.sdJobID
	}
	m.mu.Unlock()
	if ok && jh != nil {
		cancel(nil) // plain user cancel, not errServerCrashed
		if sdJobID != "" && port != 0 {
			m.cancelSDJob(port, sdJobID)
		}
		return nil
	}
	var state string
	if err := m.db.QueryRow(`SELECT state FROM media_jobs WHERE id=?`, id).Scan(&state); err != nil {
		if err == sql.ErrNoRows {
			return fmt.Errorf("media job %s not found", id)
		}
		return err
	}
	if state == StateQueued || state == StateRunning {
		_ = m.setState(id, StateCanceled, "", "canceled")
		m.publish("media.progress", map[string]any{"id": id, "state": StateCanceled, "message": "Canceled"})
	}
	return nil
}

// cancelSDJob best-effort posts the sd.cpp cancel endpoint; the Go-side
// context cancellation is the primary mechanism and this is fire-and-forget
// (the server may already be gone, or the job already terminal).
func (m *Manager) cancelSDJob(port int, sdJobID string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("http://127.0.0.1:%d/sdcpp/v1/jobs/%s/cancel", port, sdJobID), nil)
	if err != nil {
		return
	}
	resp, err := m.doSD(req, port)
	if err != nil {
		return
	}
	resp.Body.Close()
}

// RecoverAfterRestart marks media jobs left queued/running by an unclean
// backend exit as failed, mirroring instances/quantize restart recovery.
// Without this, a killed or crashed backend leaves rows stuck forever since
// nothing is left in memory to ever finish them.
func (m *Manager) RecoverAfterRestart() error {
	_, err := m.db.Exec(`UPDATE media_jobs SET state=?, error=?, updated_at=? WHERE state IN (?,?)`,
		StateFailed, "interrupted: backend restarted", now(), StateQueued, StateRunning)
	return err
}

// Get returns one media job.
func (m *Manager) Get(id string) (*Job, error) {
	var j Job
	var params, result string
	err := m.db.QueryRow(`SELECT id,model_id,kind,state,prompt,params_json,output_path,output_format,
		width,height,frames,seed,runtime_id,pid,log_path,result_json,error,created_at,updated_at,finished_at
		FROM media_jobs WHERE id=?`, id).Scan(
		&j.ID, &j.ModelID, &j.Kind, &j.State, &j.Prompt, &params, &j.OutputPath, &j.Format,
		&j.Width, &j.Height, &j.Frames, &j.Seed, &j.RuntimeID, &j.PID, &j.LogPath,
		&result, &j.Error, &j.CreatedAt, &j.UpdatedAt, &j.FinishedAt)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("media job %s not found", id)
	}
	if err != nil {
		return nil, err
	}
	j.Params = json.RawMessage(params)
	m.mu.Lock()
	if h := m.jobs[id]; h != nil && h.progress != nil {
		p := *h.progress
		j.Progress = &p
	}
	m.mu.Unlock()
	j.Result = json.RawMessage(result)
	var meta struct {
		Outputs []string `json:"outputs"`
	}
	if json.Unmarshal(j.Result, &meta) == nil && len(meta.Outputs) > 0 {
		j.OutputPaths = meta.Outputs
	} else if j.OutputPath != "" {
		j.OutputPaths = []string{j.OutputPath}
	}
	return &j, nil
}

// List returns recent media jobs (newest first).
func (m *Manager) List(modelID string, limit int) ([]Job, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	var rows *sql.Rows
	var err error
	if modelID != "" {
		rows, err = m.db.Query(`SELECT id FROM media_jobs WHERE model_id=? ORDER BY created_at DESC LIMIT ?`, modelID, limit)
	} else {
		rows, err = m.db.Query(`SELECT id FROM media_jobs ORDER BY created_at DESC LIMIT ?`, limit)
	}
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
		if j, err := m.Get(id); err == nil {
			out = append(out, *j)
		}
	}
	return out, nil
}

func (m *Manager) setState(id, state, outPath, errMsg string) error {
	_, err := m.db.Exec(`UPDATE media_jobs SET state=?, output_path=?, error=?, updated_at=? WHERE id=?`,
		state, outPath, errMsg, now(), id)
	return err
}

func (m *Manager) publish(event string, payload any) {
	if event == "media.progress" {
		if data, ok := payload.(map[string]any); ok {
			id, _ := data["id"].(string)
			state, _ := data["state"].(string)
			var callback func(Progress)
			p, hasProgress := data["progress"].(Progress)
			if !hasProgress {
				p.JobID = id
				p.Message, _ = data["message"].(string)
				p.Phase, _ = data["sd_status"].(string)
			}
			m.mu.Lock()
			if h := m.jobs[id]; h != nil {
				if h.terminal && state == StateRunning {
					m.mu.Unlock()
					return
				}
				h.terminal = state == StateFailed || state == StateCanceled || state == StateComplete
				h.progress = &p
				if state == StateRunning {
					callback = h.onProgress
				}
			}
			m.mu.Unlock()
			if callback != nil {
				callback(p)
			}
		}
	}
	if m.events != nil {
		m.events.Publish(event, payload)
	}
}

func firstPath(paths []string) string {
	if len(paths) == 0 {
		return ""
	}
	return paths[0]
}

func firstErr(a, b error) string {
	if a != nil {
		if b != nil {
			return a.Error() + "; fallback: " + b.Error()
		}
		return a.Error()
	}
	if b != nil {
		return b.Error()
	}
	return "generation failed"
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func sanitizeID(id string) string {
	out := make([]rune, 0, len(id))
	for _, r := range id {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			out = append(out, r)
		} else {
			out = append(out, '-')
		}
	}
	return string(out)
}

func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) <= n {
		return s
	}
	return strings.Join(lines[len(lines)-n:], "\n")
}

// maxInitImageInlineBytes bounds an inline base64 init_image: it has to fit
// the control-API's request body limit alongside everything else.
const maxInitImageInlineBytes = 3 << 20

// maxInitImagePathBytes bounds a local init_image_path file read from disk.
const maxInitImagePathBytes = 20 << 20

// checkInitImageSize rejects an inline init_image over the transport cap
// instead of letting ValidateGenerateParams silently drop it, which would
// otherwise turn img2img into txt2img without telling the caller.
func checkInitImageSize(p GenerateParams) error {
	if len(p.InitImage) > maxInitImageInlineBytes {
		return fmt.Errorf("init_image exceeds %d MiB limit; use init_image_path for larger images", maxInitImageInlineBytes>>20)
	}
	return nil
}

// initImageExts are the local image file extensions accepted for
// init_image_path (img2img).
var initImageExts = map[string]string{
	".png":  "image/png",
	".jpg":  "image/jpeg",
	".jpeg": "image/jpeg",
	".webp": "image/webp",
	".bmp":  "image/bmp",
}

// validateInitImagePath checks an init_image_path candidate without reading
// its contents: must be absolute, a regular file, an accepted image
// extension, and no larger than maxInitImagePathBytes.
func validateInitImagePath(path string) error {
	if !filepath.IsAbs(path) {
		return fmt.Errorf("init_image_path must be an absolute path")
	}
	if _, ok := initImageExts[strings.ToLower(filepath.Ext(path))]; !ok {
		return fmt.Errorf("init_image_path must be a png/jpg/jpeg/webp/bmp file")
	}
	st, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("init_image_path: %w", err)
	}
	if !st.Mode().IsRegular() {
		return fmt.Errorf("init_image_path must be a regular file")
	}
	if st.Size() > maxInitImagePathBytes {
		return fmt.Errorf("init_image_path exceeds %d MiB limit", maxInitImagePathBytes>>20)
	}
	return nil
}

// loadInitImageDataURL validates, reads, and base64-encodes a local
// init_image_path into a data: URL for the sdcpp init_image field.
func loadInitImageDataURL(path string) (string, error) {
	if err := validateInitImagePath(path); err != nil {
		return "", err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("reading init image: %w", err)
	}
	if len(raw) > maxInitImagePathBytes {
		return "", fmt.Errorf("init_image_path exceeds %d MiB limit", maxInitImagePathBytes>>20)
	}
	mime := initImageExts[strings.ToLower(filepath.Ext(path))]
	if mime == "" {
		mime = "application/octet-stream"
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(raw), nil
}
