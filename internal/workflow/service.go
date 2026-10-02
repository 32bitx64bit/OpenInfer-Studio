package workflow

import (
	"database/sql"
	"fmt"
	"image"
	_ "image/gif" // register decoders for DecodeConfig
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
	"github.com/openinfer/openinfer-studio/internal/models"
)

// Models is the slice of the model library the planner needs.
type Models interface {
	Get(id string) (*models.Model, error)
}

// Media is the slice of the media manager the service needs: capabilities
// and running-server state for planning, plus the stage runner for execution.
type Media interface {
	StageRunner
	RunningSettings(modelID string) (mediagen.LoadSettings, bool)
	RuntimeCapabilities(runtimeID string) (id string, flags []string, err error)
}

// RuntimeInfo says which sd.cpp runtime capabilities were read from.
type RuntimeInfo struct {
	ID    string `json:"id,omitempty"`
	Known bool   `json:"known"`
	Error string `json:"error,omitempty"`
}

// Service ties the store, registry and planner to the live app: the model
// library, the media manager and the files on disk.
type Service struct {
	store    *Store
	registry *Registry
	models   Models
	media    Media
	exec     *Executor
}

// NewService returns the workflow service. db must have migration 0005.
// events may be nil; media may be nil, which disables planning against a
// runtime and running graphs.
func NewService(db *sql.DB, lib Models, media Media, events EventSink) *Service {
	s := &Service{store: NewStore(db), registry: NewRegistry(), models: lib, media: media}
	if media != nil {
		s.exec = NewExecutor(media, events, nil)
	}
	return s
}

// Executor returns the run queue, or nil when running is unavailable.
func (s *Service) Executor() *Executor { return s.exec }

// Close stops the executor.
func (s *Service) Close() {
	if s.exec != nil {
		s.exec.Close()
	}
}

// Store returns the workflow store.
func (s *Service) Store() *Store { return s.store }

// Registry returns the node registry.
func (s *Service) Registry() *Registry { return s.registry }

// Capabilities reads what the selected sd.cpp runtime advertises. When there
// is no usable runtime, Known is false and nodes that need flags are marked
// unavailable rather than guessed at.
func (s *Service) Capabilities(runtimeID string) (RuntimeInfo, Caps) {
	if s.media == nil {
		return RuntimeInfo{Error: "media generation unavailable"}, Caps{}
	}
	id, flags, err := s.media.RuntimeCapabilities(runtimeID)
	if err != nil {
		return RuntimeInfo{ID: id, Error: err.Error()}, Caps{}
	}
	return RuntimeInfo{ID: id, Known: true}, Caps{Known: true, Flags: flags}
}

// Plan validates and compiles a graph against the live app.
func (s *Service) Plan(g Graph, runtimeID string, opts Options) (*Plan, []Issue, RuntimeInfo) {
	info, caps := s.Capabilities(runtimeID)
	plan, issues := Build(g, s.registry, serviceEnv{s}, caps, opts)
	return plan, issues, info
}

// serviceEnv implements Env over the library, media manager and filesystem.
type serviceEnv struct{ s *Service }

func (e serviceEnv) ResolveModel(libraryID string) (ModelInfo, error) {
	if e.s.models == nil {
		return ModelInfo{}, fmt.Errorf("model library unavailable")
	}
	m, err := e.s.models.Get(libraryID)
	if err != nil {
		return ModelInfo{}, err
	}
	if m.Modality != "diffusion" {
		return ModelInfo{}, fmt.Errorf("%s is not a diffusion model", filepath.Base(m.PrimaryPath))
	}
	info := ModelInfo{ID: m.ID, Name: m.Alias, Path: m.PrimaryPath, Size: m.SizeBytes}
	if info.Name == "" {
		info.Name = filepath.Base(m.PrimaryPath)
	}
	if st, err := os.Stat(m.PrimaryPath); err == nil {
		info.ModTime = st.ModTime().UnixNano()
	}
	return info, nil
}

func (e serviceEnv) FileMeta(path string) (FileMeta, error) {
	st, err := os.Stat(path)
	if err != nil {
		return FileMeta{}, err
	}
	if !st.Mode().IsRegular() {
		return FileMeta{}, fmt.Errorf("not a regular file")
	}
	meta := FileMeta{Size: st.Size(), ModTime: st.ModTime().UnixNano()}
	switch strings.ToLower(filepath.Ext(path)) {
	case ".png", ".jpg", ".jpeg", ".gif":
		if f, err := os.Open(path); err == nil {
			if cfg, _, err := image.DecodeConfig(f); err == nil {
				meta.Width, meta.Height = cfg.Width, cfg.Height
			}
			f.Close()
		}
	}
	return meta, nil
}

func (e serviceEnv) RunningServer(modelID string) (mediagen.LoadSettings, bool) {
	if e.s.media == nil {
		return mediagen.LoadSettings{}, false
	}
	return e.s.media.RunningSettings(modelID)
}
