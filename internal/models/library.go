// Package models manages the local model library: scanning, GGUF metadata,
// split/projector grouping, aliases, favorites, notes, presets, and files.
package models

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/openinfer/openinfer-studio/internal/gguf"
)

var now = func() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// ErrEmptyDisplayName is returned when a PATCH tries to clear the library name.
var ErrEmptyDisplayName = errors.New("display name cannot be empty")

// Filename quant / split patterns used when deriving a display alias.
var (
	aliasQuantRe = regexp.MustCompile(`(?i)[.\-_]((?:(?:UD|OID)-)?(?:IQ[1-4](?:_[A-Z0-9]+)+|Q[1-8]_[A-Z0-9_]+(?:_[SMXL])?|F16|F32|BF16|TQ[12]_0|MXFP4|Q4_0))`)
	aliasSplitRe = regexp.MustCompile(`(?i)-(\d{5})-of-(\d{5})$`)
	// Trailing quant on a display name, including a space separator
	// ("Muse Glimmer 30B Assistant Q4_K_M").
	trailingQuantRe = regexp.MustCompile(`(?i)[.\-_\s]+(?:(?:UD|OID)-)?(?:IQ[1-4](?:_[A-Z0-9]+)+|Q[1-8]_[A-Z0-9_]+(?:_[SMXL])?|F16|F32|BF16|TQ[12]_0|MXFP4|Q4_0)$`)
)

// goodAlias reports whether a GGUF general.name (or existing DB alias) is
// usable as a display name. Upstream converters sometimes leave stubs like
// "Hf" in general.name.
func goodAlias(s string) bool {
	s = strings.TrimSpace(s)
	if len(s) < 4 {
		return false
	}
	if strings.Contains(s, "/") {
		// Hugging Face repo ids (author/name) are not display names.
		return false
	}
	switch strings.ToLower(s) {
	case "model", "gguf", "untitled", "unknown", "none":
		return false
	}
	return true
}

// DisplayNameFromRepo turns a Hugging Face id into a library title
// ("Blackfrost-AI/Muse-Glimmer-30B-Abliterated-BF16" → "Muse-Glimmer-30B-Abliterated").
func DisplayNameFromRepo(repo string) string {
	repo = strings.TrimSpace(repo)
	if i := strings.LastIndex(repo, "/"); i >= 0 {
		repo = repo[i+1:]
	}
	return StripTrailingQuant(repo)
}

// StripTrailingQuant removes a filename/display-name quant suffix so a new
// level can be appended (F16 → Q4_K_M, Q8_0 → Q4_K_M, …).
func StripTrailingQuant(s string) string {
	s = strings.TrimSpace(s)
	for {
		next := strings.TrimSpace(trailingQuantRe.ReplaceAllString(s, ""))
		if next == s {
			return next
		}
		s = next
	}
}

// QuantizedAlias is the library display name for a newly quantized GGUF:
// the source model's name with the new quant level at the end.
func QuantizedAlias(name, ftype string) string {
	name = strings.TrimSpace(name)
	ftype = strings.TrimSpace(ftype)
	base := StripTrailingQuant(name)
	if base == "" {
		base = name
	}
	if ftype == "" {
		return base
	}
	return strings.TrimSpace(base + " " + ftype)
}

func isLocalLayout(path string) bool {
	for p := filepath.Clean(path); p != "." && p != string(os.PathSeparator); {
		if strings.HasPrefix(filepath.Base(p), "local--") {
			return true
		}
		next := filepath.Dir(p)
		if next == p {
			break
		}
		p = next
	}
	return false
}

// deriveAlias chooses a human-readable library alias. Prefer a solid
// general.name; otherwise build one from the GGUF filename or the managed
// Hugging Face repo folder (author--Repo-Name-GGUF/…). Local quantize/import
// outputs append the quant level so they are distinct from the source.
func deriveAlias(primaryPath, ggufName, quantization string) string {
	alias := deriveAliasBase(primaryPath, ggufName)
	if quantization != "" && isLocalLayout(primaryPath) {
		return QuantizedAlias(alias, quantization)
	}
	return alias
}

func deriveAliasBase(primaryPath, ggufName string) string {
	ggufName = strings.TrimSpace(ggufName)
	if i := strings.LastIndex(ggufName, "/"); i >= 0 && i+1 < len(ggufName) {
		ggufName = ggufName[i+1:]
	}
	if goodAlias(ggufName) {
		return strings.TrimSpace(ggufName)
	}
	base := strings.TrimSuffix(filepath.Base(primaryPath), ".gguf")
	base = aliasSplitRe.ReplaceAllString(base, "")
	cleaned := aliasQuantRe.ReplaceAllString(base, "")
	cleaned = regexp.MustCompile(`[-_.]{2,}`).ReplaceAllString(cleaned, "-")
	cleaned = strings.Trim(cleaned, "._- ")
	if goodAlias(cleaned) {
		return cleaned
	}
	if goodAlias(base) {
		return base
	}
	// Managed download layout: …/author--Repo-Name-GGUF/<group>/<file>.gguf
	repoDir := filepath.Base(filepath.Dir(filepath.Dir(primaryPath)))
	if _, name, ok := strings.Cut(repoDir, "--"); ok {
		repoDir = name
	}
	repoDir = regexp.MustCompile(`(?i)-gguf$`).ReplaceAllString(repoDir, "")
	repoDir = strings.ReplaceAll(repoDir, "-", " ")
	repoDir = strings.Join(strings.Fields(repoDir), " ")
	if goodAlias(repoDir) {
		return repoDir
	}
	if cleaned != "" {
		return cleaned
	}
	if base != "" {
		return base
	}
	if strings.TrimSpace(ggufName) != "" {
		return strings.TrimSpace(ggufName)
	}
	return filepath.Base(primaryPath)
}

// Model is the library view of one logical model (possibly split).
type Model struct {
	ID            string          `json:"id"`
	Alias         string          `json:"alias"`
	Favorite      bool            `json:"favorite"`
	Notes         string          `json:"notes"`
	PrimaryPath   string          `json:"primary_path"`
	ProjectorPath string          `json:"projector_path"`
	SizeBytes     int64           `json:"size_bytes"`
	Quantization  string          `json:"quantization"`
	Architecture  string          `json:"architecture"`
	Parameters    int64           `json:"parameters"`
	ContextLength int             `json:"context_length"`
	Metadata      json.RawMessage `json:"metadata"`
	SourceRepo    string          `json:"source_repo"`
	PinnedRuntime string          `json:"pinned_runtime"`
	PinnedBackend string          `json:"pinned_backend"`
	LastLoadedAt  string          `json:"last_loaded_at"`
	LastRuntime   string          `json:"last_runtime"`
	LastResult    string          `json:"last_result"`
	Files         []string        `json:"files"` // all shards + projector
	CreatedAt     string          `json:"created_at"`
}

type EventSink interface {
	Publish(event string, payload any)
}

type Library struct {
	db       *sql.DB
	managed  string // managed models dir
	log      *slog.Logger
	events   EventSink
	maxDepth int
}

func NewLibrary(db *sql.DB, managedDir string, events EventSink, log *slog.Logger) *Library {
	return &Library{db: db, managed: managedDir, events: events, log: log, maxDepth: 6}
}

// Directories returns registered model directories (managed first).
func (l *Library) Directories() ([]map[string]any, error) {
	out := []map[string]any{{"id": "managed", "path": l.managed, "managed": true}}
	rows, err := l.db.Query(`SELECT id, path FROM model_directories ORDER BY created_at`)
	if err != nil {
		return out, nil
	}
	defer rows.Close()
	for rows.Next() {
		var id, p string
		if rows.Scan(&id, &p) == nil {
			out = append(out, map[string]any{"id": id, "path": p, "managed": false})
		}
	}
	return out, nil
}

// AddDirectory registers an extra model directory.
func (l *Library) AddDirectory(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	st, err := os.Stat(abs)
	if err != nil || !st.IsDir() {
		return "", fmt.Errorf("not a directory: %s", abs)
	}
	id := uuid.NewString()
	_, err = l.db.Exec(`INSERT INTO model_directories(id,path,managed,created_at) VALUES (?,?,0,?)`, id, abs, now())
	return id, err
}

// RemoveDirectory unregisters a directory (files are never deleted).
func (l *Library) RemoveDirectory(id string) error {
	_, err := l.db.Exec(`DELETE FROM model_directories WHERE id = ?`, id)
	return err
}

var splitSuffix = regexp.MustCompile(`(?i)-\d{5}-of-\d{5}\.gguf$`)

// Scan walks all registered directories, parses GGUF headers, groups split
// sets and pairs projectors, and upserts the library.
func (l *Library) Scan() (int, error) {
	dirs, err := l.Directories()
	if err != nil {
		return 0, err
	}
	type found struct {
		path string
		size int64
	}
	var ggufs []found
	for _, d := range dirs {
		root := d["path"].(string)
		baseDepth := strings.Count(filepath.Clean(root), string(os.PathSeparator))
		_ = filepath.WalkDir(root, func(p string, e os.DirEntry, err error) error {
			if err != nil {
				return nil // unreadable entries are skipped, never fatal
			}
			depth := strings.Count(filepath.Clean(p), string(os.PathSeparator)) - baseDepth
			if e.IsDir() {
				if depth > l.maxDepth {
					return filepath.SkipDir
				}
				// Skip hidden dirs and our own partial trees.
				if strings.HasPrefix(e.Name(), ".") && depth > 0 {
					return filepath.SkipDir
				}
				return nil
			}
			if strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
				if st, err := e.Info(); err == nil {
					ggufs = append(ggufs, found{p, st.Size()})
				}
			}
			return nil
		})
	}

	// Group by model stem (split sets share a stem).
	groups := map[string][]found{}
	projectors := map[string]found{} // by directory
	for _, g := range ggufs {
		lower := strings.ToLower(filepath.Base(g.path))
		if strings.Contains(lower, "mmproj") || strings.Contains(lower, "mm-proj") {
			projectors[filepath.Dir(g.path)] = g
			continue
		}
		stem := g.path
		if splitSuffix.MatchString(stem) {
			stem = splitSuffix.ReplaceAllString(stem, "")
		} else {
			stem = strings.TrimSuffix(stem, ".gguf")
		}
		groups[stem] = append(groups[stem], g)
	}

	count := 0
	for _, files := range groups {
		sort.Slice(files, func(i, j int) bool { return files[i].path < files[j].path })
		primary := files[0].path
		// Full validation: header metadata + tensor-table integrity. Files
		// with tensor errors are kept in the library but flagged, so the user
		// sees the problem before attempting a load.
		tensorIssues, md, err := gguf.ValidateFile(primary)
		if err != nil {
			l.log.Warn("skipping unreadable gguf", "path", primary, "err", err)
			continue
		}
		if len(tensorIssues) > 0 {
			l.log.Warn("gguf tensor validation failed", "path", primary, "issues", tensorIssues)
		}
		// Path/name heuristics catch draft sidecars (mtp-, eagle3-, …) and
		// community draft names that still use a normal architecture string.
		md.ApplySpeculativeFlags(primary)
		md.ApplyEmbeddingFlags(primary)
		md.ApplyDiffusionFlags(primary)
		if q := gguf.OverlayDynamicQuant(primary, md.Name, ""); q != "" {
			md.Quantization = q
		}
		var total int64
		for _, f := range files {
			total += f.size
		}
		var proj string
		var projHasVision, projHasAudio bool
		// Never pair an mmproj with a speculative draft or embedding model —
		// neither is a multimodal chat target.
		if !md.SpeculativeDraft && !md.IsEmbedding {
			if p, ok := projectors[filepath.Dir(primary)]; ok {
				proj = p.path
				total += p.size
				if _, pmd, perr := gguf.ValidateFile(proj); perr == nil {
					// Trust parsed encoder flags. Do not let filename heuristics
					// override a vision-only mmproj that sets has_audio=false.
					projHasVision = pmd.HasVision
					projHasAudio = pmd.HasAudio
					if pmd.Projector && !projHasVision && !projHasAudio {
						projHasVision = true
					}
				} else {
					// Parse failed — fall back to conservative filename hints.
					plower := strings.ToLower(filepath.Base(proj) + " " + filepath.Base(primary))
					for _, h := range []string{
						"ultravox", "voxtral", "whisper", "qwen2-audio", "qwen3-asr",
						"-asr-", "_asr_", "seallm-audio", "-audio-", "_audio_",
						"-omni-", "_omni_", "omni-",
					} {
						if strings.Contains(plower, h) {
							projHasAudio = true
						}
					}
					for _, h := range []string{"llava", "vision", "vl-", "pixtral", "internvl", "smolvlm"} {
						if strings.Contains(plower, h) {
							projHasVision = true
						}
					}
					if !projHasVision && !projHasAudio {
						projHasVision = true // paired mmproj ⇒ assume vision
					}
				}
			}
		}
		hasVision := md.HasVision || projHasVision
		hasAudio := md.HasAudio || projHasAudio
		multimodal := md.Multimodal || hasVision || hasAudio || proj != ""
		if md.SpeculativeDraft || md.IsEmbedding {
			hasVision, hasAudio, multimodal = false, false, false
			proj = ""
		}
		if md.IsDiffusion {
			hasVision, hasAudio, multimodal = false, false, false
			proj = ""
		}
		meta := map[string]any{
			"name": md.Name, "tokenizer": md.Tokenizer,
			"multimodal": multimodal, "has_vision": hasVision, "has_audio": hasAudio,
			"speculative_draft":    md.SpeculativeDraft,
			"has_mtp":              md.HasMTP,
			"nextn_predict_layers": md.NextnPredictLayers,
			"spec_type":            md.SpecType,
			"is_embedding":         md.IsEmbedding,
			"is_reranker":          md.IsReranker,
			"pooling_type":         md.PoolingType,
			"embedding_length_out": md.EmbeddingLengthOut,
			"is_diffusion":         md.IsDiffusion,
			"canvas_length":        md.CanvasLength,
			"version":              md.Version,
			"block_count":          md.BlockCount, "head_count": md.HeadCount,
			"head_count_kv": md.HeadCountKV, "head_count_kv_layers": md.HeadCountKVLayers,
			"head_dim": md.HeadDim, "value_dim": md.ValueDim,
			"head_dim_swa": md.HeadDimSWA, "value_dim_swa": md.ValueDimSWA,
			"sliding_window": md.SlidingWindow, "sliding_window_pattern": md.SlidingWindowPattern,
			"shared_kv_layers": md.SharedKVLayers, "full_attention_interval": md.FullAttentionInterval,
			"ssm_state_size": md.SSMStateSize, "ssm_inner_size": md.SSMInnerSize,
			"embedding_length": md.Embedding,
			"tensor_errors":    tensorIssues,
		}
		if md.Reasoning.Controllable() || md.Reasoning.CanPreserve {
			meta["reasoning"] = md.Reasoning
		}
		metaJSON, _ := json.Marshal(meta)
		id := stableID(primary)
		alias := deriveAlias(primary, md.Name, md.Quantization)
		_, err = l.db.Exec(`INSERT INTO models
			(id,alias,primary_path,projector_path,size_bytes,quantization,architecture,parameters,
			 context_length,metadata_json,created_at,updated_at)
			VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
			ON CONFLICT(id) DO UPDATE SET
			 primary_path=excluded.primary_path, projector_path=excluded.projector_path,
			 size_bytes=excluded.size_bytes, quantization=excluded.quantization,
			 architecture=excluded.architecture, parameters=excluded.parameters,
			 context_length=excluded.context_length, metadata_json=excluded.metadata_json,
			 updated_at=excluded.updated_at,
			 alias=CASE
			   WHEN length(trim(models.alias)) < 4
			     OR lower(trim(models.alias)) IN ('model','gguf','untitled','unknown','none')
			   THEN excluded.alias
			   WHEN instr(models.alias, '/') > 0
			   THEN excluded.alias
			   WHEN trim(models.alias) = trim(COALESCE(json_extract(models.metadata_json, '$.name'), ''))
			   THEN excluded.alias
			   ELSE models.alias
			 END`,
			id, alias, primary, proj, total, md.Quantization, md.Architecture,
			int64(md.Parameters), int(md.ContextLength), string(metaJSON), now(), now())
		if err != nil {
			l.log.Warn("upsert model failed", "path", primary, "err", err)
			continue
		}
		_, _ = l.db.Exec(`DELETE FROM model_files WHERE model_id = ?`, id)
		for i, f := range files {
			role := "shard"
			if i == 0 {
				role = "primary"
			}
			_, _ = l.db.Exec(`INSERT INTO model_files(id,model_id,path,role,size_bytes) VALUES (?,?,?,?,?)`,
				uuid.NewString(), id, f.path, role, f.size)
		}
		if proj != "" {
			_, _ = l.db.Exec(`INSERT INTO model_files(id,model_id,path,role,size_bytes) VALUES (?,?,?,'projector',?)`,
				uuid.NewString(), id, proj, projectors[filepath.Dir(primary)].size)
		}
		count++
	}

	// Remove DB rows whose primary file vanished.
	rows, err := l.db.Query(`SELECT id, primary_path FROM models`)
	if err == nil {
		defer rows.Close()
		type row struct{ id, p string }
		var stale []row
		for rows.Next() {
			var r row
			if rows.Scan(&r.id, &r.p) == nil {
				if _, err := os.Stat(r.p); os.IsNotExist(err) {
					stale = append(stale, r)
				}
			}
		}
		for _, s := range stale {
			_, _ = l.db.Exec(`DELETE FROM models WHERE id = ?`, s.id)
		}
	}

	if l.events != nil {
		l.events.Publish("library.scanned", map[string]any{"models": count})
	}
	return count, nil
}

// stableID derives a collision-resistant ID from the path (stable across
// rescans, so settings survive).
func stableID(path string) string {
	return uuid.NewSHA1(uuid.NameSpaceURL, []byte("openinfer-model:"+filepath.Clean(path))).String()
}

// List returns all models, favorites first.
func (l *Library) List() ([]Model, error) {
	rows, err := l.db.Query(`SELECT id,alias,favorite,notes,primary_path,projector_path,size_bytes,
		quantization,architecture,parameters,context_length,metadata_json,source_repo,
		pinned_runtime,pinned_backend,last_loaded_at,last_runtime,last_result,created_at
		FROM models ORDER BY favorite DESC, alias COLLATE NOCASE`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Model
	for rows.Next() {
		var m Model
		var fav int
		var meta string
		if err := rows.Scan(&m.ID, &m.Alias, &fav, &m.Notes, &m.PrimaryPath, &m.ProjectorPath,
			&m.SizeBytes, &m.Quantization, &m.Architecture, &m.Parameters, &m.ContextLength,
			&meta, &m.SourceRepo, &m.PinnedRuntime, &m.PinnedBackend, &m.LastLoadedAt,
			&m.LastRuntime, &m.LastResult, &m.CreatedAt); err != nil {
			return nil, err
		}
		m.Favorite = fav == 1
		m.Metadata = json.RawMessage(meta)
		out = append(out, m)
	}
	for i := range out {
		fs, _ := l.filesOf(out[i].ID)
		out[i].Files = fs
	}
	return out, nil
}

// Get returns one model.
func (l *Library) Get(id string) (*Model, error) {
	all, err := l.List()
	if err != nil {
		return nil, err
	}
	for i := range all {
		if all[i].ID == id {
			return &all[i], nil
		}
	}
	return nil, fmt.Errorf("model %s not found", id)
}

func (l *Library) filesOf(id string) ([]string, error) {
	rows, err := l.db.Query(`SELECT path FROM model_files WHERE model_id = ? ORDER BY rowid`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var p string
		if rows.Scan(&p) == nil {
			out = append(out, p)
		}
	}
	return out, nil
}

// Update edits mutable model fields.
func (l *Library) Update(id string, alias *string, favorite *bool, notes *string,
	pinnedRuntime *string, pinnedBackend *string) error {
	m, err := l.Get(id)
	if err != nil {
		return err
	}
	if alias != nil {
		a := strings.TrimSpace(*alias)
		if a == "" {
			return ErrEmptyDisplayName
		}
		alias = &a
	} else {
		alias = &m.Alias
	}
	fav := m.Favorite
	if favorite != nil {
		fav = *favorite
	}
	if notes == nil {
		notes = &m.Notes
	}
	if pinnedRuntime == nil {
		pinnedRuntime = &m.PinnedRuntime
	}
	if pinnedBackend == nil {
		pinnedBackend = &m.PinnedBackend
	}
	fi := 0
	if fav {
		fi = 1
	}
	_, err = l.db.Exec(`UPDATE models SET alias=?, favorite=?, notes=?, pinned_runtime=?,
		pinned_backend=?, updated_at=? WHERE id=?`,
		*alias, fi, *notes, *pinnedRuntime, *pinnedBackend, now(), id)
	if err != nil {
		return err
	}
	if l.events != nil {
		l.events.Publish("library.model_updated", map[string]any{
			"id": id, "alias": *alias,
		})
	}
	return nil
}

// AdoptQuantized stamps a freshly quantized GGUF in the library: display
// name is the source alias plus the new quant level, and source_repo /
// runtime pins are copied from the model that was quantized.
func (l *Library) AdoptQuantized(id, srcAlias, ftype string, inherit *Model) (string, error) {
	if id == "" {
		return "", fmt.Errorf("missing model id")
	}
	m, err := l.Get(id)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(srcAlias) == "" {
		srcAlias = m.Alias
	}
	alias := QuantizedAlias(srcAlias, ftype)
	pinRT, pinBE := m.PinnedRuntime, m.PinnedBackend
	if inherit != nil {
		if pinRT == "" {
			pinRT = inherit.PinnedRuntime
		}
		if pinBE == "" {
			pinBE = inherit.PinnedBackend
		}
	}
	if err := l.Update(id, &alias, nil, nil, &pinRT, &pinBE); err != nil {
		return "", err
	}
	repo := m.SourceRepo
	if inherit != nil && repo == "" && inherit.SourceRepo != "" {
		repo = inherit.SourceRepo
	}
	if repo == "" {
		repo = "local/" + safeImportName(alias+".gguf")
	}
	q := strings.TrimSpace(ftype)
	if q == "" {
		q = m.Quantization
	}
	_, err = l.db.Exec(`UPDATE models SET source_repo=?, quantization=?, updated_at=? WHERE id=?`,
		repo, q, now(), id)
	if l.events != nil {
		l.events.Publish("library.model_imported", map[string]any{
			"id": id, "alias": alias, "quantization": q, "source": "quantize",
		})
	}
	return alias, err
}

// RecordLoad stores the outcome of a load attempt.
func (l *Library) RecordLoad(id, runtimeID, result string) {
	_, _ = l.db.Exec(`UPDATE models SET last_loaded_at=?, last_runtime=?, last_result=? WHERE id=?`,
		now(), runtimeID, result, id)
	_, _ = l.db.Exec(`INSERT INTO model_runtime_history(id,model_id,runtime_id,result,created_at)
		VALUES (?,?,?,?,?)`, uuid.NewString(), id, runtimeID, result, now())
}

// Delete removes a model from the library. When deleteFiles is true, files
// are removed only if they live inside the managed models directory — the
// API layer confirms with the user and passes the exact paths first.
func (l *Library) Delete(id string, deleteFiles bool) ([]string, error) {
	m, err := l.Get(id)
	if err != nil {
		return nil, err
	}
	removed := []string{}
	if deleteFiles {
		managedClean := filepath.Clean(l.managed)
		for _, f := range m.Files {
			cf := filepath.Clean(f)
			if !strings.HasPrefix(cf, managedClean+string(os.PathSeparator)) {
				return removed, fmt.Errorf("refusing to delete file outside managed directory: %s", cf)
			}
		}
		for _, f := range m.Files {
			if err := os.Remove(f); err != nil && !os.IsNotExist(err) {
				return removed, fmt.Errorf("deleting %s: %w", f, err)
			}
			removed = append(removed, f)
		}
		// Best-effort: prune empty group/repo dirs left by an import/download.
		pruneEmptyParents(m.PrimaryPath, managedClean)
	}
	_, err = l.db.Exec(`DELETE FROM models WHERE id = ?`, id)
	return removed, err
}

func pruneEmptyParents(filePath, stopAt string) {
	dir := filepath.Dir(filePath)
	stopAt = filepath.Clean(stopAt)
	for {
		clean := filepath.Clean(dir)
		if clean == stopAt || !strings.HasPrefix(clean, stopAt+string(os.PathSeparator)) {
			return
		}
		entries, err := os.ReadDir(clean)
		if err != nil || len(entries) > 0 {
			return
		}
		if err := os.Remove(clean); err != nil {
			return
		}
		dir = filepath.Dir(clean)
	}
}

// ImportFile copies a GGUF from disk into the managed models directory (same
// ownership model as Hugging Face downloads), then rescans the library.
// Sibling split shards and an mmproj in the same source directory are copied
// alongside the selected file. The original path is left untouched.
//
// Layout: <managed>/local--<SafeName>/files/<basename.gguf>
//
// Files already inside the managed tree are registered in place (no copy).
func (l *Library) ImportFile(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	st, err := os.Stat(abs)
	if err != nil || st.IsDir() {
		return "", fmt.Errorf("invalid model path %q", abs)
	}
	if !strings.HasSuffix(strings.ToLower(abs), ".gguf") {
		return "", fmt.Errorf("not a GGUF file: %s", abs)
	}
	if _, err := gguf.ParseFile(abs); err != nil {
		return "", fmt.Errorf("not a readable GGUF model: %w", err)
	}

	managedClean := filepath.Clean(l.managed)
	if strings.HasPrefix(filepath.Clean(abs), managedClean+string(os.PathSeparator)) {
		if _, err := l.Scan(); err != nil {
			return "", err
		}
		return stableID(abs), nil
	}

	sources, err := importBundle(abs)
	if err != nil {
		return "", err
	}

	repoName := safeImportName(filepath.Base(abs))
	destDir := filepath.Join(l.managed, "local--"+repoName, "files")
	if conflict, err := destConflict(destDir, sources); err != nil {
		return "", err
	} else if conflict {
		repoName = repoName + "-" + uuid.NewString()[:8]
		destDir = filepath.Join(l.managed, "local--"+repoName, "files")
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return "", err
	}

	var primaryDest string
	for _, src := range sources {
		dest := filepath.Join(destDir, filepath.Base(src))
		if err := copyFile(src, dest); err != nil {
			return "", fmt.Errorf("copying %s: %w", filepath.Base(src), err)
		}
		if filepath.Clean(src) == filepath.Clean(abs) {
			primaryDest = dest
		}
	}
	if primaryDest == "" {
		primaryDest = filepath.Join(destDir, filepath.Base(abs))
	}

	id := stableID(primaryDest)
	if _, err := l.Scan(); err != nil {
		return "", err
	}
	// Scan may pick a different primary among split shards; resolve by path.
	if _, err := l.Get(id); err != nil {
		if resolved := l.idForPath(primaryDest); resolved != "" {
			id = resolved
		} else {
			return "", fmt.Errorf("imported file was not registered by library scan")
		}
	}
	// Stamp source after scan so ON CONFLICT does not wipe it.
	_, _ = l.db.Exec(`UPDATE models SET source_repo = ? WHERE id = ?`, "local/"+repoName, id)
	if l.events != nil {
		l.events.Publish("library.model_imported", map[string]any{
			"id": id, "path": primaryDest, "source": abs,
		})
	}
	return id, nil
}

// IDForPath returns the library id whose primary, projector, or shard path
// matches path, or "" if none.
func (l *Library) IDForPath(path string) string {
	return l.idForPath(path)
}

func (l *Library) idForPath(path string) string {
	want := filepath.Clean(path)
	all, err := l.List()
	if err != nil {
		return ""
	}
	for _, m := range all {
		if filepath.Clean(m.PrimaryPath) == want || filepath.Clean(m.ProjectorPath) == want {
			return m.ID
		}
		for _, f := range m.Files {
			if filepath.Clean(f) == want {
				return m.ID
			}
		}
	}
	return ""
}

// SetSourceRepo stamps the Hugging Face id (or local/…) after Scan, which
// otherwise leaves source_repo empty on upsert.
func (l *Library) SetSourceRepo(id, repo string) error {
	if id == "" {
		return fmt.Errorf("missing model id")
	}
	_, err := l.db.Exec(`UPDATE models SET source_repo=?, updated_at=? WHERE id=?`, repo, now(), id)
	return err
}

// SetAlias overwrites the display name after convert/import.
func (l *Library) SetAlias(id, alias string) error {
	id = strings.TrimSpace(id)
	alias = strings.TrimSpace(alias)
	if id == "" || alias == "" {
		return fmt.Errorf("missing model id or alias")
	}
	_, err := l.db.Exec(`UPDATE models SET alias=?, updated_at=? WHERE id=?`, alias, now(), id)
	return err
}

// HighPrecisionFromRepo returns a library F32/F16/BF16 (else Q8) whose
// source_repo matches the Hugging Face id. Files whose tokenizer length,
// vocab_size, and embedding/output rows disagree are skipped so from-HF
// reconverts instead of feeding llama.cpp a GGUF it will reject.
func (l *Library) HighPrecisionFromRepo(repo string) *Model {
	repo = strings.TrimSpace(repo)
	if repo == "" {
		return nil
	}
	all, err := l.List()
	if err != nil {
		return nil
	}
	var q8 *Model
	for i := range all {
		m := &all[i]
		if m.SourceRepo != repo {
			continue
		}
		q := strings.ToUpper(strings.TrimSpace(m.Quantization))
		q = strings.TrimPrefix(q, "UD-")
		q = strings.TrimPrefix(q, "OID-")
		switch q {
		case "F32", "F16", "BF16":
			if gguf.CheckVocabLayout(m.PrimaryPath) != nil {
				continue
			}
			return m
		case "Q8_0", "":
			if q8 == nil && gguf.CheckVocabLayout(m.PrimaryPath) == nil {
				q8 = m
			}
		}
	}
	return q8
}

// importBundle returns the selected GGUF plus same-directory split shards and
// an mmproj sibling when present.
func importBundle(primary string) ([]string, error) {
	dir := filepath.Dir(primary)
	base := filepath.Base(primary)
	lower := strings.ToLower(base)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return []string{primary}, nil
	}

	stem := primary
	if splitSuffix.MatchString(stem) {
		stem = splitSuffix.ReplaceAllString(stem, "")
	} else {
		stem = strings.TrimSuffix(stem, filepath.Ext(stem))
	}
	stemBase := filepath.Base(stem)

	out := []string{primary}
	seen := map[string]bool{filepath.Clean(primary): true}
	var mmproj string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		nlower := strings.ToLower(name)
		if !strings.HasSuffix(nlower, ".gguf") {
			continue
		}
		full := filepath.Join(dir, name)
		if seen[filepath.Clean(full)] {
			continue
		}
		if strings.Contains(nlower, "mmproj") || strings.Contains(nlower, "mm-proj") {
			if !strings.Contains(lower, "mmproj") && !strings.Contains(lower, "mm-proj") {
				mmproj = full
			}
			continue
		}
		candStem := full
		if splitSuffix.MatchString(candStem) {
			candStem = splitSuffix.ReplaceAllString(candStem, "")
		} else {
			candStem = strings.TrimSuffix(candStem, filepath.Ext(candStem))
		}
		if filepath.Base(candStem) == stemBase {
			out = append(out, full)
			seen[filepath.Clean(full)] = true
		}
	}
	if mmproj != "" {
		out = append(out, mmproj)
	}
	sort.Strings(out)
	return out, nil
}

func safeImportName(filename string) string {
	base := strings.TrimSuffix(filename, filepath.Ext(filename))
	base = aliasSplitRe.ReplaceAllString(base, "")
	base = strings.TrimSpace(base)
	if base == "" {
		base = "model"
	}
	repl := strings.NewReplacer("/", "-", "\\", "-", "..", "", " ", "-", "\t", "-")
	base = repl.Replace(base)
	base = regexp.MustCompile(`[^A-Za-z0-9._-]+`).ReplaceAllString(base, "-")
	base = regexp.MustCompile(`-{2,}`).ReplaceAllString(base, "-")
	base = strings.Trim(base, ".-_")
	if base == "" {
		base = "model"
	}
	if len(base) > 80 {
		base = base[:80]
	}
	return base
}

func destConflict(destDir string, sources []string) (bool, error) {
	for _, src := range sources {
		dest := filepath.Join(destDir, filepath.Base(src))
		st, err := os.Stat(dest)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return false, err
		}
		sst, err := os.Stat(src)
		if err != nil {
			return false, err
		}
		if st.Size() != sst.Size() {
			return true, nil
		}
	}
	return false, nil
}

func copyFile(src, dst string) error {
	if filepath.Clean(src) == filepath.Clean(dst) {
		return nil
	}
	if st, err := os.Stat(dst); err == nil {
		sst, serr := os.Stat(src)
		if serr == nil && st.Size() == sst.Size() {
			return nil // identical-sized file already present
		}
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp := dst + ".copying"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dst)
}
