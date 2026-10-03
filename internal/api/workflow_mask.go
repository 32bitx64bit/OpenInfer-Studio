package api

import (
	"net/http"
	"net/url"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/workflow"
)

func (h *handlers) paintWorkflowMask(w http.ResponseWriter, r *http.Request) {
	if h.d.Media == nil {
		writeErr(w, 503, "media store unavailable", nil)
		return
	}
	var body struct {
		ImagePath string                `json:"image_path"`
		Strokes   []workflow.MaskStroke `json:"strokes"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	path, err := workflow.PaintMaskContext(r.Context(), h.d.Media.MediaDir(), body.ImagePath, body.Strokes)
	if err != nil {
		writeWorkflowErr(w, "painting mask", err)
		return
	}
	p := filepath.ToSlash(path)
	if runtime.GOOS == "windows" && !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	writeJSON(w, 201, map[string]any{"path": path, "file_url": "file://" + (&url.URL{Path: p}).EscapedPath()})
}
