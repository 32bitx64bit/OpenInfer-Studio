package sdmodel

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// ManifestName is the small file a generator download leaves next to its
// weights. The download picker knows what each file is (the diffusion model,
// a VAE, a text encoder, …) because the user ticked it as that; the library
// scanner cannot always work it out from tensor names, which for a new
// architecture match nothing it knows. The manifest carries the picker's
// answer to the scanner.
const ManifestName = ".openinfer-download.json"

// Roles recorded in a manifest besides the Role* component roles.
const (
	// RoleModel marks the diffusion model / checkpoint itself.
	RoleModel = "model"
	// RoleComponent marks a pipeline part whose exact kind the picker could
	// not say (a diffusers text_encoder/ folder): not a model, not wired by
	// the manifest — companion discovery classifies it as before.
	RoleComponent = "component"
)

// Manifest records what the files of one download are.
type Manifest struct {
	Version int    `json:"version"`
	Repo    string `json:"repo,omitempty"`
	// DiffusionKind is "image", "video" or "both" as the repository was
	// detected; empty when unknown.
	DiffusionKind string `json:"diffusion_kind,omitempty"`
	// Files maps a slash path relative to the manifest's folder to its role.
	Files map[string]string `json:"files"`
}

const maxManifestBytes = 1 << 20

// ReadManifest reads dir's manifest; nil when there is none or it is unusable.
func ReadManifest(dir string) *Manifest {
	f, err := os.Open(filepath.Join(dir, ManifestName))
	if err != nil {
		return nil
	}
	defer f.Close()
	var m Manifest
	dec := json.NewDecoder(io.LimitReader(f, maxManifestBytes))
	if err := dec.Decode(&m); err != nil || m.Version != 1 || m.Files == nil {
		return nil
	}
	return &m
}

// WriteManifest merges add into dir's manifest (creating it) so a later
// download into the same folder keeps what an earlier one recorded. The
// write is atomic.
func WriteManifest(dir string, add Manifest) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	cur := ReadManifest(dir)
	if cur == nil {
		cur = &Manifest{Version: 1, Files: map[string]string{}}
	}
	if add.Repo != "" {
		cur.Repo = add.Repo
	}
	if add.DiffusionKind != "" {
		cur.DiffusionKind = add.DiffusionKind
	}
	for rel, role := range add.Files {
		cur.Files[filepath.ToSlash(rel)] = role
	}
	b, err := json.MarshalIndent(cur, "", " ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ManifestName+".*")
	if err != nil {
		return err
	}
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	if err := os.Rename(tmp.Name(), filepath.Join(dir, ManifestName)); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	return nil
}

// manifestDepth is how many folders above a file are searched for its
// manifest: the download folder itself plus a few levels of kept repo
// folders (vae/, text_encoders/, vae_1_0/x/).
const manifestDepth = 4

// declared finds the manifest entry for path.
func declared(path string) (*Manifest, string, bool) {
	path = filepath.Clean(path)
	dir := filepath.Dir(path)
	for i := 0; i < manifestDepth; i++ {
		if m := ReadManifest(dir); m != nil {
			rel, err := filepath.Rel(dir, path)
			if err == nil {
				if role, ok := m.Files[filepath.ToSlash(rel)]; ok {
					return m, role, true
				}
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return nil, "", false
}

// DeclaredRole returns the role the download recorded for path: RoleModel,
// RoleComponent or one of the Role* component roles.
func DeclaredRole(path string) (string, bool) {
	_, role, ok := declared(path)
	return role, ok
}

// DeclaredKind returns "image" or "video" when the download recorded the
// repository's modality for path (a repository that does both reports "").
func DeclaredKind(path string) string {
	m, _, ok := declared(path)
	if !ok {
		return ""
	}
	switch strings.ToLower(m.DiffusionKind) {
	case "image":
		return "image"
	case "video":
		return "video"
	}
	return ""
}
