package api

import (
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

// fileURL builds a proper file:// URL for a local path, including the
// Windows drive-letter form (file:///C:/...).
func fileURL(abs string) string {
	abs = filepath.ToSlash(abs)
	if runtime.GOOS == "windows" && !strings.HasPrefix(abs, "/") {
		abs = "/" + abs
	}
	return "file://" + (&url.URL{Path: abs}).EscapedPath()
}

// mediaFileServe serves a generated media file from the managed media dir.
// Paths are validated to stay inside the media root; nothing outside is
// ever served.
func (h *handlers) mediaFile(w http.ResponseWriter, r *http.Request) {
	rel := strings.TrimPrefix(r.URL.Path, "/api/v1/media/file/")
	if rel == "" || strings.Contains(rel, "..") {
		writeErr(w, 400, "invalid media path", nil)
		return
	}
	root := ""
	if h.d.Media != nil {
		root = h.d.Media.MediaDir()
	}
	if root == "" {
		writeErr(w, 404, "media store unavailable", nil)
		return
	}
	abs := filepath.Join(root, filepath.FromSlash(rel))
	cleanRoot, _ := filepath.Abs(root)
	cleanAbs, _ := filepath.Abs(abs)
	if cleanAbs != cleanRoot && !strings.HasPrefix(cleanAbs, cleanRoot+string(os.PathSeparator)) {
		writeErr(w, 403, "media path escapes managed root", nil)
		return
	}
	st, err := os.Stat(cleanAbs)
	if err != nil || st.IsDir() {
		writeErr(w, 404, "media file not found", nil)
		return
	}
	switch strings.ToLower(filepath.Ext(cleanAbs)) {
	case ".png":
		w.Header().Set("Content-Type", "image/png")
	case ".jpg", ".jpeg":
		w.Header().Set("Content-Type", "image/jpeg")
	case ".webp":
		// Still images and animated webp share the type; browsers sniff.
		w.Header().Set("Content-Type", "image/webp")
	case ".webm":
		w.Header().Set("Content-Type", "video/webm")
	case ".avi":
		w.Header().Set("Content-Type", "video/x-msvideo")
	default:
		w.Header().Set("Content-Type", "application/octet-stream")
	}
	http.ServeFile(w, r, cleanAbs)
}

func mediaJobView(j *mediagen.Job, mediaDir string) map[string]any {
	if j == nil {
		return nil
	}
	outputs := j.OutputPaths
	var urls []string
	var fileURLs []string
	var localPaths []string
	for _, p := range outputs {
		urls = append(urls, "/api/v1/media/file/"+mediaRel(p))
		if mediaDir != "" {
			abs := p
			if !filepath.IsAbs(abs) {
				abs = filepath.Join(mediaDir, filepath.FromSlash(p))
			}
			localPaths = append(localPaths, abs)
			fileURLs = append(fileURLs, fileURL(abs))
		}
	}
	return map[string]any{
		"id": j.ID, "model_id": j.ModelID, "kind": j.Kind, "state": j.State,
		"prompt": j.Prompt, "params": j.Params,
		"output_path": j.OutputPath, "output_paths": outputs, "urls": urls,
		"file_urls": fileURLs, "local_paths": localPaths,
		"output_format": j.Format, "width": j.Width, "height": j.Height,
		"frames": j.Frames, "seed": j.Seed, "runtime_id": j.RuntimeID,
		"log_path": j.LogPath, "result": j.Result, "error": j.Error,
		"progress":   j.Progress,
		"created_at": j.CreatedAt, "updated_at": j.UpdatedAt, "finished_at": j.FinishedAt,
	}
}

func mediaRel(p string) string {
	// Result paths are stored media-root-relative ("images/<id>.png").
	// Older absolute result paths fall back to subdir+basename so the
	// file-serve route still resolves inside the media root.
	if p == "" {
		return ""
	}
	if filepath.IsAbs(p) {
		base := filepath.Base(p)
		dir := filepath.Base(filepath.Dir(p))
		if dir == "images" || dir == "videos" || dir == "logs" {
			return dir + "/" + base
		}
		return base
	}
	return filepath.ToSlash(filepath.Clean(p))
}

func (h *handlers) listMediaJobs(w http.ResponseWriter, r *http.Request) {
	if h.d.Media == nil {
		writeErr(w, 503, "media generation unavailable", nil)
		return
	}
	jobs, err := h.d.Media.List(r.URL.Query().Get("model_id"), 50)
	if err != nil {
		writeErr(w, 500, "listing media jobs", err)
		return
	}
	mediaDir := h.d.Media.MediaDir()
	out := make([]map[string]any, 0, len(jobs))
	for i := range jobs {
		out = append(out, mediaJobView(&jobs[i], mediaDir))
	}
	writeJSON(w, 200, map[string]any{"jobs": out})
}

func (h *handlers) getMediaJob(w http.ResponseWriter, r *http.Request) {
	if h.d.Media == nil {
		writeErr(w, 503, "media generation unavailable", nil)
		return
	}
	j, err := h.d.Media.Get(r.PathValue("id"))
	if err != nil {
		writeErr(w, 404, "media job not found", err)
		return
	}
	writeJSON(w, 200, mediaJobView(j, h.d.Media.MediaDir()))
}

// saveTargets maps n job outputs onto a user-chosen destination path: the
// first output lands exactly on dest, later ones get "-<i>" before the
// extension so a batch lands beside the picked name.
func saveTargets(dest string, n int) []string {
	out := make([]string, 0, n)
	ext := filepath.Ext(dest)
	stem := strings.TrimSuffix(dest, ext)
	for i := 0; i < n; i++ {
		if i == 0 {
			out = append(out, dest)
		} else {
			out = append(out, stem+"-"+strconv.Itoa(i)+ext)
		}
	}
	return out
}

// copyFile writes src to dst atomically (temp file + rename).
func copyFile(src, dst string) error {
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

// saveMediaJob copies a job's generated outputs out of the managed media dir
// to a user-chosen absolute path ("Save to disk"). A single-output job lands
// exactly on dest_path; a batch writes the rest beside it as name-1.ext, …
func (h *handlers) saveMediaJob(w http.ResponseWriter, r *http.Request) {
	if h.d.Media == nil {
		writeErr(w, 503, "media generation unavailable", nil)
		return
	}
	var body struct {
		DestPath string `json:"dest_path"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	dest := filepath.Clean(body.DestPath)
	if dest == "" || dest == "." {
		writeErr(w, 400, "dest_path required", nil)
		return
	}
	if !filepath.IsAbs(dest) {
		writeErr(w, 400, "dest_path must be an absolute path", nil)
		return
	}
	j, err := h.d.Media.Get(r.PathValue("id"))
	if err != nil {
		writeErr(w, 404, "media job not found", err)
		return
	}
	outs := j.OutputPaths
	if len(outs) == 0 {
		writeErr(w, 400, "job has no output files", nil)
		return
	}
	mediaDir := h.d.Media.MediaDir()
	targets := saveTargets(dest, len(outs))
	saved := make([]string, 0, len(outs))
	for i, p := range outs {
		src := p
		if !filepath.IsAbs(src) {
			src = filepath.Join(mediaDir, filepath.FromSlash(p))
		}
		if _, err := os.Stat(src); err != nil {
			writeErr(w, 404, "output file not found", err)
			return
		}
		if err := copyFile(src, targets[i]); err != nil {
			writeErr(w, 500, "save failed", err)
			return
		}
		saved = append(saved, targets[i])
	}
	writeJSON(w, 200, map[string]any{"ok": true, "saved": saved})
}

func (h *handlers) cancelMediaJob(w http.ResponseWriter, r *http.Request) {
	if h.d.Media == nil {
		writeErr(w, 503, "media generation unavailable", nil)
		return
	}
	if err := h.d.Media.Cancel(r.PathValue("id")); err != nil {
		writeErr(w, 500, "cancel failed", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) mediaCapabilities(w http.ResponseWriter, r *http.Request) {
	if h.d.Media == nil {
		writeErr(w, 503, "media generation unavailable", nil)
		return
	}
	caps, err := h.d.Media.Capabilities(r.Context(), r.PathValue("id"))
	if err != nil {
		writeErr(w, 400, "diffusion server not running", err)
		return
	}
	writeJSON(w, 200, caps)
}

func (h *handlers) mediaServer(w http.ResponseWriter, r *http.Request) {
	if h.d.Media == nil {
		writeErr(w, 503, "media generation unavailable", nil)
		return
	}
	modelID := r.PathValue("id")
	switch {
	case strings.HasSuffix(r.URL.Path, "/start"):
		// Already up: report the port so the UI can settle immediately.
		if port, ok := h.d.Media.ServerPort(modelID); ok {
			writeJSON(w, 200, map[string]any{"ok": true, "state": "running", "port": port})
			return
		}
		var s mediagen.LoadSettings
		if r.Body != nil && r.ContentLength != 0 {
			raw := map[string]json.RawMessage{}
			if !decodeJSON(w, r, &raw) {
				return
			}
			merged, _ := json.Marshal(mediagen.DefaultLoadSettings())
			base := map[string]json.RawMessage{}
			_ = json.Unmarshal(merged, &base)
			for k, v := range raw {
				base[k] = v
			}
			final, _ := json.Marshal(base)
			if err := json.Unmarshal(final, &s); err != nil {
				writeErr(w, 400, "invalid diffusion load settings", err)
				return
			}
		}
		// Model weights load for minutes — 202 immediately and report
		// readiness over media.server_ready / media.server_error events.
		if err := h.d.Media.StartServer(modelID, s); err != nil {
			writeErr(w, 400, "starting diffusion server failed", err)
			return
		}
		writeJSON(w, 202, map[string]any{"ok": true, "state": "starting"})
	case strings.HasSuffix(r.URL.Path, "/stop"):
		h.d.Media.StopServer(modelID)
		writeJSON(w, 200, map[string]bool{"ok": true})
	default:
		writeErr(w, 404, "unknown media server action", nil)
	}
}

func (h *handlers) generateMedia(w http.ResponseWriter, r *http.Request) {
	if h.d.Media == nil {
		writeErr(w, 503, "media generation unavailable", nil)
		return
	}
	var p mediagen.GenerateParams
	// Merge over validated defaults so omitted knobs stay unset.
	raw := map[string]json.RawMessage{}
	if r.Body != nil && r.ContentLength != 0 {
		if !decodeJSON(w, r, &raw) {
			return
		}
		merged, _ := json.Marshal(mediagen.GenerateParams{Seed: -1})
		base := map[string]json.RawMessage{}
		_ = json.Unmarshal(merged, &base)
		for k, v := range raw {
			base[k] = v
		}
		final, _ := json.Marshal(base)
		if err := json.Unmarshal(final, &p); err != nil {
			writeErr(w, 400, "invalid generation params", err)
			return
		}
	} else {
		p.Seed = -1
	}
	if len(p.Prompt) > 8000 {
		writeErr(w, 400, "prompt too large", nil)
		return
	}
	if len(p.NegativePrompt) > 8000 {
		writeErr(w, 400, "negative prompt too large", nil)
		return
	}
	// Generations run for minutes (model load + sampling). StartGenerate
	// records the job and works in the background; this handler answers 202
	// immediately so the request never rides the 120s control-API ceiling
	// (server.go withTimeouts).
	j, err := h.d.Media.StartGenerate(r.PathValue("id"), p)
	if err != nil {
		writeErr(w, 400, "generation failed", err)
		return
	}
	writeJSON(w, 202, mediaJobView(j, h.d.Media.MediaDir()))
}

func (h *handlers) listMediaServers(w http.ResponseWriter, r *http.Request) {
	if h.d.Media == nil {
		writeJSON(w, 200, map[string]any{"servers": []any{}})
		return
	}
	servers := h.d.Media.ListServers()
	if servers == nil {
		servers = []mediagen.ServerView{}
	}
	writeJSON(w, 200, map[string]any{"servers": servers})
}
