package workflow

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

type cacheEntry struct {
	Outputs  []string `json:"outputs"`
	Sizes    []int64  `json:"sizes"`
	Modified []int64  `json:"modified"`
	JobID    string   `json:"job_id"`
	Message  string   `json:"message"`
	Seed     *int64   `json:"seed,omitempty"`
}

// A generation cache identity includes the effective running configuration and
// runtime binary, as supplied by the media manager. Unknown identities do not cache.
type generationCacheIdentity interface {
	WorkflowCacheIdentity(modelID string, overrides *mediagen.LoadOverrides) (string, error)
}

func (e *Executor) cacheKey(r *run, st Stage, servers map[string]ServerNeed) string {
	if st.Volatile || st.Kind == StageSave {
		return ""
	}
	identity := "cpu-image-v1"
	if st.Kind == StageGenerate || st.Kind == StageUpscale {
		provider, ok := e.runner.(generationCacheIdentity)
		if !ok {
			return ""
		}
		server := ""
		if st.Generate != nil {
			server = st.Generate.Server
		} else {
			server = st.Upscale.Server
		}
		need := servers[server]
		var err error
		identity, err = provider.WorkflowCacheIdentity(need.ModelID, &need.Overrides)
		if err != nil || identity == "" {
			return ""
		}
	}
	inputs := e.cacheInputIdentity(r, st)
	if inputs == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(identity + ":" + inputs + ":" + st.Key))
	return hex.EncodeToString(sum[:])
}

func (e *Executor) cacheInputIdentity(r *run, st Stage) string {
	identity := "inputs-v1"
	paths := []string{}
	add := func(ref *ImageRef) {
		if ref != nil && ref.Path != "" {
			paths = append(paths, ref.Path)
		}
	}
	if st.Generate != nil {
		add(st.Generate.Init)
		add(st.Generate.Mask)
		add(st.Generate.Control)
		add(st.Generate.Reference)
		for _, l := range st.Generate.Params.Lora {
			paths = append(paths, l.Path)
		}
	}
	if st.Resize != nil {
		add(&st.Resize.Src)
	}
	if st.Image != nil {
		add(&st.Image.Src)
		add(st.Image.Other)
	}
	if st.Upscale != nil {
		add(&st.Upscale.Src)
	}
	for _, path := range paths {
		real, err := filepath.EvalSymlinks(path)
		if err != nil {
			return ""
		}
		info, err := os.Stat(real)
		if err != nil {
			return ""
		}
		raw, _ := json.Marshal(fileIdentity{Path: real, Size: info.Size(), ModTime: info.ModTime().UnixNano()})
		identity += ":" + string(raw)
	}
	for _, dep := range st.Deps {
		for _, rel := range r.outputs[dep] {
			info, err := os.Stat(filepath.Join(e.runner.MediaDir(), filepath.FromSlash(rel)))
			if err != nil {
				return ""
			}
			raw, _ := json.Marshal(fileIdentity{Path: rel, Size: info.Size(), ModTime: info.ModTime().UnixNano()})
			identity += ":" + string(raw)
		}
	}
	return identity
}

func (e *Executor) cached(key string) (cacheEntry, bool) {
	var entry cacheEntry
	if e.db == nil || key == "" {
		return entry, false
	}
	var raw string
	if e.db.QueryRow(`SELECT entry_json FROM workflow_cache WHERE key=?`, key).Scan(&raw) != nil || json.Unmarshal([]byte(raw), &entry) != nil {
		return entry, false
	}
	if len(entry.Outputs) == 0 || len(entry.Outputs) != len(entry.Sizes) || len(entry.Outputs) != len(entry.Modified) {
		return entry, false
	}
	for i, rel := range entry.Outputs {
		if filepath.IsAbs(rel) || !filepath.IsLocal(filepath.FromSlash(rel)) {
			return cacheEntry{}, false
		}
		path := filepath.Join(e.runner.MediaDir(), filepath.FromSlash(rel))
		real, err := filepath.EvalSymlinks(path)
		if err != nil {
			return cacheEntry{}, false
		}
		root, err := filepath.EvalSymlinks(e.runner.MediaDir())
		if err != nil {
			return cacheEntry{}, false
		}
		inside, err := filepath.Rel(root, real)
		if err != nil || !filepath.IsLocal(inside) {
			return cacheEntry{}, false
		}
		st, err := os.Stat(real)
		if err != nil || !st.Mode().IsRegular() || st.Size() != entry.Sizes[i] || st.ModTime().UnixNano() != entry.Modified[i] {
			return cacheEntry{}, false
		}
	}
	return entry, true
}

func (e *Executor) putCache(key string, entry cacheEntry) {
	if e.db == nil || key == "" {
		return
	}
	for _, rel := range entry.Outputs {
		st, err := os.Stat(filepath.Join(e.runner.MediaDir(), filepath.FromSlash(rel)))
		if err != nil {
			return
		}
		entry.Sizes = append(entry.Sizes, st.Size())
		entry.Modified = append(entry.Modified, st.ModTime().UnixNano())
	}
	raw, err := json.Marshal(entry)
	if err != nil {
		return
	}
	if _, err := e.db.Exec(`INSERT INTO workflow_cache(key,entry_json,updated_at) VALUES(?,?,?)
        ON CONFLICT(key) DO UPDATE SET entry_json=excluded.entry_json,updated_at=excluded.updated_at`, key, string(raw), ts()); err != nil {
		e.log.Warn("saving workflow cache", "err", err)
	}
}
