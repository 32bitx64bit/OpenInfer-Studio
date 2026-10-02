package workflow

import "github.com/openinfer/openinfer-studio/internal/mediagen"

// ModelInfo identifies a library model for planning and cache keys.
type ModelInfo struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Path    string `json:"path"`
	Size    int64  `json:"size"`
	ModTime int64  `json:"mod_time"` // unix nanoseconds
}

// FileMeta identifies a local file (LoRA, VAE, text encoder, input image).
// Width and Height are filled for images the backend can read the size of.
type FileMeta struct {
	Size    int64
	ModTime int64
	Width   int
	Height  int
}

// Env is everything the planner needs from the outside world. The API layer
// implements it over the model library and the media manager; tests use a
// fake, so planning is pure and needs no database or process.
type Env interface {
	// ResolveModel looks up a library model by id.
	ResolveModel(libraryID string) (ModelInfo, error)
	// FileMeta stats a local file and, for images, reads its dimensions.
	FileMeta(path string) (FileMeta, error)
	// RunningServer returns the settings a ready sd-server for the model was
	// launched with (after companion pairing), if one is running.
	RunningServer(modelID string) (mediagen.LoadSettings, bool)
}
