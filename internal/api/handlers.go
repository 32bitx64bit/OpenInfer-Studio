package api

import (
	"context"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/openinfer/openinfer-studio/internal/auth"
	"github.com/openinfer/openinfer-studio/internal/downloads"
	"github.com/openinfer/openinfer-studio/internal/hardware"
	"github.com/openinfer/openinfer-studio/internal/huggingface"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
	"github.com/openinfer/openinfer-studio/internal/version"
)

type handlers struct {
	d *Deps

	hwMu   sync.Mutex
	hwInfo *hardware.Info
	hwAt   time.Time
}

func (h *handlers) status(w http.ResponseWriter, r *http.Request) {
	info := version.Info()
	writeJSON(w, 200, map[string]any{
		"app": "OpenInfer Studio", "api": 1,
		"version": info["version"], "commit": info["commit"], "date": info["date"],
		"goos": info["goos"], "goarch": info["goarch"],
		"data_dir": h.d.Layout.DataDir,
	})
}

func (h *handlers) hardwareInfo() *hardware.Info {
	h.hwMu.Lock()
	defer h.hwMu.Unlock()
	if h.hwInfo != nil && time.Since(h.hwAt) < 10*time.Second {
		return h.hwInfo
	}
	h.hwInfo = hardware.Detect(h.d.Layout.Models, h.d.Layout.Runtimes)
	h.hwAt = time.Now()
	return h.hwInfo
}

func (h *handlers) hardware(w http.ResponseWriter, r *http.Request) {
	if r.URL.Query().Get("refresh") == "1" {
		h.hwMu.Lock()
		h.hwInfo = nil
		h.hwMu.Unlock()
	}
	info := h.hardwareInfo()
	rec := hardware.RecommendBackend(info)
	writeJSON(w, 200, map[string]any{"hardware": info, "recommendation": rec})
}

func (h *handlers) getSettings(w http.ResponseWriter, r *http.Request) {
	all, err := h.d.Settings.All()
	if err != nil {
		writeErr(w, 500, "reading settings", err)
		return
	}
	// Never leak secrets: hf token lives in the OS keychain only.
	delete(all, "hf_token")
	writeJSON(w, 200, all)
}

func (h *handlers) putSetting(w http.ResponseWriter, r *http.Request) {
	key := r.PathValue("key")
	if strings.Contains(strings.ToLower(key), "token") || strings.Contains(strings.ToLower(key), "secret") {
		writeErr(w, 400, "secrets must use dedicated endpoints", nil)
		return
	}
	var body struct {
		Value string `json:"value"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	if len(body.Value) > 1<<16 {
		writeErr(w, 400, "setting value too large", nil)
		return
	}
	if err := h.d.Settings.Set(key, body.Value); err != nil {
		writeErr(w, 500, "saving setting", err)
		return
	}
	h.applySetting(key, body.Value)
	writeJSON(w, 200, map[string]string{"key": key, "value": body.Value})
}

// applySetting propagates settings that affect running services.
func (h *handlers) applySetting(key, value string) {
	switch key {
	case "downloads.concurrency":
		if n, err := strconv.Atoi(value); err == nil {
			h.d.DL.SetConcurrency(n)
		}
	case "downloads.connections":
		if n, err := strconv.Atoi(value); err == nil {
			h.d.DL.SetConnections(n)
		}
	case "instances.max_loaded":
		if n, err := strconv.Atoi(value); err == nil {
			h.d.IM.SetMaxLoaded(n)
		}
	case "instances.startup_timeout_sec":
		if n, err := strconv.Atoi(value); err == nil {
			h.d.IM.SetStartupTimeout(time.Duration(n) * time.Second)
		}
	case "chat.streaming":
		h.d.Chat.SetStreaming(value != "0")
	}
}

func (h *handlers) hfSearch(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query().Get("q")
	sort := r.URL.Query().Get("sort")
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	res, err := h.d.HF.Search(r.Context(), q, sort, limit)
	if err != nil {
		status := 502
		if ae, ok := err.(interface{ HTTPStatus() int }); ok {
			status = ae.HTTPStatus()
		}
		writeErr(w, status, "Hugging Face search failed", err)
		return
	}
	writeJSON(w, 200, map[string]any{"results": res})
}

func (h *handlers) hfRepo(w http.ResponseWriter, r *http.Request) {
	repo := r.PathValue("author") + "/" + r.PathValue("name")
	info, err := h.d.HF.Repo(r.Context(), repo)
	if err != nil {
		writeErr(w, 502, "loading repository failed", err)
		return
	}
	groups, projectors := huggingface.GroupFiles(info.Files)
	filePaths := make([]string, 0, len(info.Files))
	for _, f := range info.Files {
		filePaths = append(filePaths, f.Path)
	}
	modalities := huggingface.DetectModalities(info.ID, info.PipelineTag, info.Tags, filePaths)
	writeJSON(w, 200, map[string]any{
		"repo": info, "groups": groups, "projectors": projectors,
		"modalities":    modalities,
		"mtp":           huggingface.DetectMTP(info.ID, info.Tags, filePaths),
		"embedding":     huggingface.DetectEmbedding(info.ID, info.PipelineTag, info.Tags, filePaths),
		"download_base": h.d.Layout.Models,
	})
}

func (h *handlers) hfTokenStatus(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{"configured": h.d.HF.HasToken()})
}

func (h *handlers) hfSetToken(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Token string `json:"token"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	if body.Token == "" || len(body.Token) > 512 {
		writeErr(w, 400, "invalid token", nil)
		return
	}
	if err := auth.StoreHuggingFaceToken(body.Token); err != nil {
		writeErr(w, 500, "storing token in OS credential vault failed", err)
		return
	}
	h.d.HF.SetToken(body.Token)
	writeJSON(w, 200, map[string]any{"configured": true})
}

func (h *handlers) hfDeleteToken(w http.ResponseWriter, r *http.Request) {
	if err := auth.DeleteHuggingFaceToken(); err != nil {
		writeErr(w, 500, "removing token", err)
		return
	}
	h.d.HF.SetToken("")
	writeJSON(w, 200, map[string]any{"configured": false})
}

func (h *handlers) listDownloads(w http.ResponseWriter, r *http.Request) {
	items, err := h.d.DL.List()
	if err != nil {
		writeErr(w, 500, "listing downloads", err)
		return
	}
	if items == nil {
		items = []downloads.Item{}
	}
	writeJSON(w, 200, map[string]any{"downloads": items})
}

type enqueueRequest struct {
	Kind  string `json:"kind"` // model|runtime
	Label string `json:"label"`
	Repo  string `json:"repo"`
	Group string `json:"group"` // group ID for folder naming
	Files []struct {
		Path string `json:"path"`
		Size int64  `json:"size"`
		URL  string `json:"url,omitempty"`
	} `json:"files"`
}

func (h *handlers) enqueueDownload(w http.ResponseWriter, r *http.Request) {
	var req enqueueRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.Kind == "" {
		req.Kind = "model"
	}
	if len(req.Files) == 0 || len(req.Files) > 64 {
		writeErr(w, 400, "a download needs between 1 and 64 files", nil)
		return
	}
	// Destination: managed models dir / repo / group / filename
	safeRepo := strings.NewReplacer("/", "--", "..", "").Replace(req.Repo)
	safeGroup := strings.NewReplacer("/", "-", "..", "").Replace(req.Group)
	if safeGroup == "" {
		safeGroup = "files"
	}
	destDir := filepath.Join(h.d.Layout.Models, safeRepo, safeGroup)
	specs := make([]downloads.FileSpec, 0, len(req.Files))
	for _, f := range req.Files {
		if f.Path == "" || strings.Contains(f.Path, "..") {
			writeErr(w, 400, "invalid file path "+f.Path, nil)
			return
		}
		url := f.URL
		if url == "" {
			url = h.d.HF.DownloadURL(req.Repo, f.Path)
		}
		specs = append(specs, downloads.FileSpec{
			URL:      url,
			DestPath: filepath.Join(destDir, filepath.Base(f.Path)),
			Size:     f.Size,
		})
	}
	id, err := h.d.DL.Enqueue(req.Kind, req.Label, destDir, specs,
		map[string]string{"repo": req.Repo, "group": req.Group})
	if err != nil {
		writeErr(w, 500, "enqueue failed", err)
		return
	}
	writeJSON(w, 201, map[string]string{"id": id})
}

func (h *handlers) dlAction(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var err error
	switch {
	case strings.HasSuffix(r.URL.Path, "/pause"):
		err = h.d.DL.Pause(id)
	case strings.HasSuffix(r.URL.Path, "/resume"):
		err = h.d.DL.Resume(id)
	case strings.HasSuffix(r.URL.Path, "/cancel"):
		err = h.d.DL.Cancel(id)
	case strings.HasSuffix(r.URL.Path, "/retry"):
		err = h.d.DL.Retry(id)
	}
	if err != nil {
		writeErr(w, 500, "download action failed", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) dlReorder(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Position int `json:"position"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	if err := h.d.DL.Reorder(r.PathValue("id"), body.Position); err != nil {
		writeErr(w, 500, "reorder failed", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) dlDelete(w http.ResponseWriter, r *http.Request) {
	if err := h.d.DL.Delete(r.PathValue("id")); err != nil {
		writeErr(w, 500, "delete failed", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) listRuntimes(w http.ResponseWriter, r *http.Request) {
	all, err := h.d.RT.List()
	if err != nil {
		writeErr(w, 500, "listing runtimes", err)
		return
	}
	if all == nil {
		all = []runtimes.Runtime{}
	}
	writeJSON(w, 200, map[string]any{"runtimes": all})
}

func (h *handlers) listReleases(w http.ResponseWriter, r *http.Request) {
	rels, err := h.d.RT.Feed().Latest(r.Context())
	if err != nil {
		writeErr(w, 502, "checking llama.cpp releases failed", err)
		return
	}
	hw := h.hardwareInfo()
	prof := runtimes.MachineProfile{
		OS: hw.OS, Arch: hw.Arch, Vulkan: hw.Vulkan, CUDA: hw.CUDA,
		HIP: hw.HIP, Metal: hw.Metal, SYCL: hw.SYCL,
	}
	for _, g := range hw.GPUs {
		if prof.GPUVendor == "" {
			prof.GPUVendor = g.Vendor
		}
		if g.Vendor == "nvidia" {
			prof.GPUVendor = "nvidia"
		}
	}
	prefer := r.URL.Query().Get("backend")
	type relView struct {
		runtimes.Release
		Matches []runtimes.AssetMatch `json:"matches"`
	}
	out := []relView{}
	for _, rel := range rels {
		out = append(out, relView{rel, runtimes.ResolveAssets(rel, prof, prefer)})
	}
	writeJSON(w, 200, map[string]any{"releases": out})
}

func (h *handlers) installRuntime(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Tag     string `json:"tag"`
		Asset   string `json:"asset"` // asset name
		Backend string `json:"backend"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	rels, err := h.d.RT.Feed().Latest(r.Context())
	if err != nil {
		writeErr(w, 502, "checking releases failed", err)
		return
	}
	var rel *runtimes.Release
	for i := range rels {
		if rels[i].Tag == req.Tag {
			rel = &rels[i]
			break
		}
	}
	if rel == nil {
		writeErr(w, 404, "release not found", nil)
		return
	}
	var match *runtimes.AssetMatch
	hw := h.hardwareInfo()
	prof := runtimes.MachineProfile{OS: hw.OS, Arch: hw.Arch, Vulkan: hw.Vulkan, CUDA: hw.CUDA,
		HIP: hw.HIP, Metal: hw.Metal, SYCL: hw.SYCL}
	matches := runtimes.ResolveAssets(*rel, prof, req.Backend)
	for i := range matches {
		if matches[i].Asset.Name == req.Asset {
			match = &matches[i]
			break
		}
	}
	if match == nil {
		writeErr(w, 404, "asset not found in release", nil)
		return
	}
	// Install runs async; download progress flows over download.* events,
	// lifecycle over runtime.installing / installed / install_failed.
	if h.d.Hub != nil {
		h.d.Hub.Publish("runtime.installing", map[string]any{
			"tag": req.Tag, "backend": req.Backend, "asset": req.Asset,
		})
	}
	go func() {
		id, err := h.d.RT.Install(context.Background(), *rel, *match)
		if err != nil {
			h.d.Logs.Logger("runtimes", 0).Error("runtime install failed", "err", err)
			if h.d.Hub != nil {
				h.d.Hub.Publish("runtime.install_failed", map[string]any{
					"tag": req.Tag, "backend": req.Backend, "error": err.Error(),
				})
			}
			return
		}
		h.d.Logs.Logger("runtimes", 0).Info("runtime installed", "id", id)
	}()
	writeJSON(w, 202, map[string]string{"status": "installing"})
}

func (h *handlers) importRuntime(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Path string `json:"path"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	id, err := h.d.RT.ImportCustom(req.Path)
	if err != nil {
		writeErr(w, 400, "import failed", err)
		return
	}
	writeJSON(w, 201, map[string]string{"id": id})
}

func (h *handlers) removeRuntime(w http.ResponseWriter, r *http.Request) {
	if err := h.d.RT.Remove(r.PathValue("id")); err != nil {
		writeErr(w, 409, "removing runtime failed", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) preferRuntime(w http.ResponseWriter, r *http.Request) {
	if err := h.d.RT.SetPreferred(r.PathValue("id")); err != nil {
		writeErr(w, 500, "setting preferred runtime", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) healthRuntime(w http.ResponseWriter, r *http.Request) {
	ok, out, err := h.d.RT.HealthCheck(r.PathValue("id"))
	if err != nil {
		writeJSON(w, 200, map[string]any{"healthy": false, "version_output": "", "error": err.Error()})
		return
	}
	writeJSON(w, 200, map[string]any{"healthy": ok, "version_output": out})
}

func (h *handlers) runtimeCaps(w http.ResponseWriter, r *http.Request) {
	rt, err := h.d.RT.Get(r.PathValue("id"))
	if err != nil {
		writeErr(w, 404, "runtime not found", err)
		return
	}
	help, _ := h.d.RT.HelpOutput(rt.ID)
	writeJSON(w, 200, map[string]any{
		"capabilities": rt.Capabilities, "help": help, "version_output": rt.VersionOutput,
	})
}
