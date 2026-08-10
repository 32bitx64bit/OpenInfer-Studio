package api

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/chat"
	"github.com/openinfer/openinfer-studio/internal/diagnostics"
	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/instances"
	"github.com/openinfer/openinfer-studio/internal/models"
)

func (h *handlers) listModels(w http.ResponseWriter, r *http.Request) {
	all, err := h.d.Lib.List()
	if err != nil {
		writeErr(w, 500, "listing models", err)
		return
	}
	if all == nil {
		all = []models.Model{}
	}
	writeJSON(w, 200, map[string]any{"models": all})
}

func (h *handlers) scanModels(w http.ResponseWriter, r *http.Request) {
	n, err := h.d.Lib.Scan()
	if err != nil {
		writeErr(w, 500, "scan failed", err)
		return
	}
	// Manual rescan also marks metadata schema current so startup EnsureFresh
	// does not immediately redo the same work.
	if h.d.Settings != nil {
		_ = h.d.Settings.Set("library.metadata_schema", models.MetadataSchemaVersion)
	}
	writeJSON(w, 200, map[string]any{"models": n})
}

func (h *handlers) importModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Path string `json:"path"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	id, err := h.d.Lib.ImportFile(req.Path)
	if err != nil {
		writeErr(w, 400, "import failed", err)
		return
	}
	writeJSON(w, 201, map[string]string{"id": id})
}

func (h *handlers) getModel(w http.ResponseWriter, r *http.Request) {
	m, err := h.d.Lib.Get(r.PathValue("id"))
	if err != nil {
		writeErr(w, 404, "model not found", err)
		return
	}
	presets, _ := h.d.Lib.ListPresets(m.ID)
	if presets == nil {
		presets = []models.Preset{}
	}
	inst, loaded := h.d.IM.Get(m.ID)
	resp := map[string]any{"model": m, "presets": presets, "loaded": loaded}
	if loaded {
		resp["instance"] = inst
	}
	writeJSON(w, 200, resp)
}

func (h *handlers) patchModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Alias         *string `json:"alias"`
		Favorite      *bool   `json:"favorite"`
		Notes         *string `json:"notes"`
		PinnedRuntime *string `json:"pinned_runtime"`
		PinnedBackend *string `json:"pinned_backend"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	if err := h.d.Lib.Update(r.PathValue("id"), req.Alias, req.Favorite, req.Notes,
		req.PinnedRuntime, req.PinnedBackend); err != nil {
		writeErr(w, 500, "updating model", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) deleteModel(w http.ResponseWriter, r *http.Request) {
	deleteFiles := r.URL.Query().Get("delete_files") == "1"
	// Return affected paths first so the UI can confirm; the UI then re-calls
	// with confirmed=1.
	if r.URL.Query().Get("confirmed") != "1" {
		m, err := h.d.Lib.Get(r.PathValue("id"))
		if err != nil {
			writeErr(w, 404, "model not found", err)
			return
		}
		writeJSON(w, 200, map[string]any{"requires_confirmation": true, "paths": m.Files})
		return
	}
	removed, err := h.d.Lib.Delete(r.PathValue("id"), deleteFiles)
	if err != nil {
		writeErr(w, 409, "delete failed", err)
		return
	}
	writeJSON(w, 200, map[string]any{"ok": true, "deleted_files": removed})
}

func (h *handlers) listPresets(w http.ResponseWriter, r *http.Request) {
	p, err := h.d.Lib.ListPresets(r.PathValue("id"))
	if err != nil {
		writeErr(w, 500, "listing presets", err)
		return
	}
	if p == nil {
		p = []models.Preset{}
	}
	writeJSON(w, 200, map[string]any{"presets": p})
}

func (h *handlers) savePreset(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name      string          `json:"name"`
		Settings  json.RawMessage `json:"settings"`
		IsDefault bool            `json:"is_default"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	if len(req.Settings) == 0 {
		req.Settings = json.RawMessage(`{}`)
	}
	id, err := h.d.Lib.SavePreset(r.PathValue("id"), r.PathValue("pid"), req.Name, req.Settings, req.IsDefault)
	if err != nil {
		writeErr(w, 400, "saving preset", err)
		return
	}
	writeJSON(w, 200, map[string]string{"id": id})
}

func (h *handlers) deletePreset(w http.ResponseWriter, r *http.Request) {
	if err := h.d.Lib.DeletePreset(r.PathValue("id"), r.PathValue("pid")); err != nil {
		writeErr(w, 500, "deleting preset", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func decodeSettings(w http.ResponseWriter, r *http.Request) (instances.LoadSettings, bool) {
	s := instances.DefaultSettings()
	if r.Body == nil || r.ContentLength == 0 {
		return s, true
	}
	// Merge over defaults: decode into map then overlay JSON.
	raw := map[string]json.RawMessage{}
	if !decodeJSON(w, r, &raw) {
		return s, false
	}
	merged, _ := json.Marshal(s)
	base := map[string]json.RawMessage{}
	_ = json.Unmarshal(merged, &base)
	for k, v := range raw {
		base[k] = v
	}
	final, _ := json.Marshal(base)
	if err := json.Unmarshal(final, &s); err != nil {
		writeErr(w, 400, "invalid load settings", err)
		return s, false
	}
	// Numeric validation.
	if s.ContextLength < 0 || s.ContextLength > 1<<22 {
		writeErr(w, 400, "context_length out of range", nil)
		return s, false
	}
	if s.GPULayers < 0 || s.GPULayers > 4096 {
		writeErr(w, 400, "gpu_layers out of range", nil)
		return s, false
	}
	if s.Threads < 0 || s.Threads > 1024 {
		writeErr(w, 400, "threads out of range", nil)
		return s, false
	}
	if s.ThreadsBatch < 0 || s.ThreadsBatch > 1024 {
		writeErr(w, 400, "threads_batch out of range", nil)
		return s, false
	}
	if s.Prio < -2 || s.Prio > 3 {
		writeErr(w, 400, "prio out of range", nil)
		return s, false
	}
	if s.Poll < -1 || s.Poll > 100 {
		writeErr(w, 400, "poll out of range", nil)
		return s, false
	}
	if s.NCPUMoe < 0 || s.NCPUMoe > 4096 {
		writeErr(w, 400, "n_cpu_moe out of range", nil)
		return s, false
	}
	if s.CacheReuse < 0 || s.CacheReuse > 1<<20 {
		writeErr(w, 400, "cache_reuse out of range", nil)
		return s, false
	}
	if s.DraftMax < 0 || s.DraftMax > 256 {
		writeErr(w, 400, "draft_max out of range", nil)
		return s, false
	}
	if s.DraftMin < 0 || s.DraftMin > 256 {
		writeErr(w, 400, "draft_min out of range", nil)
		return s, false
	}
	if s.DraftModel != "" {
		if !filepath.IsAbs(s.DraftModel) {
			writeErr(w, 400, "draft_model must be an absolute path", nil)
			return s, false
		}
		if st, err := os.Stat(s.DraftModel); err != nil || st.IsDir() {
			writeErr(w, 400, "draft_model path does not exist", err)
			return s, false
		}
	}
	switch s.GPUOffload {
	case "", "auto", "all", "none", "custom":
	default:
		writeErr(w, 400, "invalid gpu_offload value", nil)
		return s, false
	}
	switch s.FlashAttention {
	case "", "auto", "on", "off":
	default:
		writeErr(w, 400, "invalid flash_attention value", nil)
		return s, false
	}
	for _, pair := range []struct {
		name, v string
	}{
		{"kv_offload", s.KVOffload},
		{"op_offload", s.OpOffload},
		{"kv_unified", s.KVUnified},
		{"fit", s.Fit},
	} {
		switch strings.ToLower(strings.TrimSpace(pair.v)) {
		case "", "on", "off":
		default:
			writeErr(w, 400, "invalid "+pair.name+" value", nil)
			return s, false
		}
	}
	switch s.NUMA {
	case "", "distribute", "isolate", "numactl":
	default:
		writeErr(w, 400, "invalid numa value", nil)
		return s, false
	}
	switch s.SplitMode {
	case "", "none", "layer", "row", "tensor":
	default:
		writeErr(w, 400, "invalid split_mode value", nil)
		return s, false
	}
	switch strings.ToLower(strings.TrimSpace(s.Pooling)) {
	case "", "none", "mean", "cls", "last", "rank":
	default:
		writeErr(w, 400, "invalid pooling value", nil)
		return s, false
	}
	return s, true
}

func (h *handlers) previewLoad(w http.ResponseWriter, r *http.Request) {
	s, ok := decodeSettings(w, r)
	if !ok {
		return
	}
	h.applyMultimodalDefaults(r.PathValue("id"), &s)
	h.applyEmbedderDefaults(r.PathValue("id"), &s)
	br, err := h.d.IM.Preview(r.PathValue("id"), s)
	if err != nil {
		writeErr(w, 400, "preview failed", err)
		return
	}
	writeJSON(w, 200, br)
}

// draftCandidates lists library models that can be used as a speculative
// decoding draft for the target. With filtering on (default), only detected
// speculative sidecars (mtp-/eagle3-/dflash-/dspark-/…) are returned. Pass
// ?filter=0 or disable load.filter_incompatible_drafts to list every model.
func (h *handlers) draftCandidates(w http.ResponseWriter, r *http.Request) {
	target, err := h.d.Lib.Get(r.PathValue("id"))
	if err != nil {
		writeErr(w, 404, "model not found", err)
		return
	}
	all, err := h.d.Lib.List()
	if err != nil {
		writeErr(w, 500, "listing models", err)
		return
	}
	filter := true
	switch r.URL.Query().Get("filter") {
	case "0", "false", "off":
		filter = false
	case "1", "true", "on":
		filter = true
	default:
		if h.d.Settings != nil && h.d.Settings.Get("load.filter_incompatible_drafts", "1") == "0" {
			filter = false
		}
	}
	cands := models.FilterDraftCandidates(*target, all, filter)
	if cands == nil {
		cands = []models.DraftCandidate{}
	}
	writeJSON(w, 200, map[string]any{
		"candidates": cands,
		"filtered":   filter,
	})
}

// estimateLoad projects memory use for a settings draft against detected
// VRAM/RAM. Numbers are heuristics and presented as estimates in the UI.
func (h *handlers) estimateLoad(w http.ResponseWriter, r *http.Request) {
	s, ok := decodeSettings(w, r)
	if !ok {
		return
	}
	h.applyEmbedderDefaults(r.PathValue("id"), &s)
	m, err := h.d.Lib.Get(r.PathValue("id"))
	if err != nil {
		writeErr(w, 404, "model not found", err)
		return
	}
	var meta struct {
		BlockCount            uint32   `json:"block_count"`
		HeadCountKV           uint32   `json:"head_count_kv"`
		HeadCountKVLayers     []uint32 `json:"head_count_kv_layers"`
		HeadDim               uint32   `json:"head_dim"`
		ValueDim              uint32   `json:"value_dim"`
		HeadDimSWA            uint32   `json:"head_dim_swa"`
		ValueDimSWA           uint32   `json:"value_dim_swa"`
		SlidingWindow         uint32   `json:"sliding_window"`
		SlidingWindowPattern  []bool   `json:"sliding_window_pattern"`
		SharedKVLayers        uint32   `json:"shared_kv_layers"`
		FullAttentionInterval uint32   `json:"full_attention_interval"`
		SSMStateSize          uint32   `json:"ssm_state_size"`
		SSMInnerSize          uint32   `json:"ssm_inner_size"`
		EmbeddingLength       uint32   `json:"embedding_length"`
		HasVision             bool     `json:"has_vision"`
		HasAudio              bool     `json:"has_audio"`
	}
	_ = json.Unmarshal(m.Metadata, &meta)
	// Prefer a live GGUF header parse for architecture/KV fields. Library
	// metadata_json can lag behind parser improvements (SWA, hybrid, etc.).
	if md, err := gguf.ParseFile(m.PrimaryPath); err == nil {
		if md.BlockCount > 0 {
			meta.BlockCount = md.BlockCount
		}
		if md.HeadCountKV > 0 {
			meta.HeadCountKV = md.HeadCountKV
		}
		if len(md.HeadCountKVLayers) > 0 {
			meta.HeadCountKVLayers = md.HeadCountKVLayers
		}
		if md.HeadDim > 0 {
			meta.HeadDim = md.HeadDim
		}
		if md.ValueDim > 0 {
			meta.ValueDim = md.ValueDim
		}
		meta.HeadDimSWA = md.HeadDimSWA
		meta.ValueDimSWA = md.ValueDimSWA
		meta.SlidingWindow = md.SlidingWindow
		meta.SlidingWindowPattern = md.SlidingWindowPattern
		meta.SharedKVLayers = md.SharedKVLayers
		meta.FullAttentionInterval = md.FullAttentionInterval
		meta.SSMStateSize = md.SSMStateSize
		meta.SSMInnerSize = md.SSMInnerSize
		if md.Embedding > 0 {
			meta.EmbeddingLength = md.Embedding
		}
	}

	hw := h.hardwareInfo()
	var vram uint64
	for _, g := range hw.GPUs {
		vram += g.VRAM
	}
	offload := s.GPUOffload != "none"
	var proj int64
	if m.ProjectorPath != "" {
		if st, err := os.Stat(m.ProjectorPath); err == nil {
			proj = st.Size()
		}
	}
	var draftWeights int64
	if s.DraftModel != "" {
		if st, err := os.Stat(s.DraftModel); err == nil {
			draftWeights = st.Size()
		}
	}
	// Resolve offloaded layer count for GPU/CPU placement.
	offloaded := meta.BlockCount
	switch s.GPUOffload {
	case "none":
		offloaded = 0
	case "custom":
		if s.GPULayers > 0 {
			offloaded = uint32(s.GPULayers)
		} else {
			offloaded = 0
		}
		if meta.BlockCount > 0 && offloaded > meta.BlockCount {
			offloaded = meta.BlockCount
		}
	default: // auto|all
		if meta.BlockCount == 0 {
			offloaded = 999 // full-offload sentinel when layer count unknown
		}
	}
	est := instances.EstimateMemory(instances.EstimateInput{
		Weights: m.SizeBytes - proj, DraftWeights: draftWeights, Projector: proj,
		Layers: meta.BlockCount, KVHeads: meta.HeadCountKV,
		HeadCountKVLayers: meta.HeadCountKVLayers,
		HeadDim:           meta.HeadDim, ValueDim: meta.ValueDim,
		HeadDimSWA: meta.HeadDimSWA, ValueDimSWA: meta.ValueDimSWA,
		SlidingWindow: meta.SlidingWindow, SlidingWindowPattern: meta.SlidingWindowPattern,
		SharedKVLayers: meta.SharedKVLayers, FullAttentionInterval: meta.FullAttentionInterval,
		SSMStateSize: meta.SSMStateSize, SSMInnerSize: meta.SSMInnerSize,
		EmbeddingLength: meta.EmbeddingLength,
		Ctx:             s.ContextLength, ModelContext: m.ContextLength,
		CacheK: s.CacheTypeK, CacheV: s.CacheTypeV,
		Slots: s.Parallel, BatchSize: s.BatchSize, UBatchSize: s.UBatchSize,
		GPUOffload: offload, OffloadedLayers: offloaded,
		VRAM: vram, RAM: hw.RAMAvailable,
		HasVision: meta.HasVision, HasAudio: meta.HasAudio,
		NoMmproj: s.NoMmproj, NoMmprojOffload: s.NoMmprojOffload,
		FlashAttention: s.FlashAttention,
	})
	writeJSON(w, 200, est)
}

func (h *handlers) loadModel(w http.ResponseWriter, r *http.Request) {
	s, ok := decodeSettings(w, r)
	if !ok {
		return
	}
	h.applyMultimodalDefaults(r.PathValue("id"), &s)
	h.applyEmbedderDefaults(r.PathValue("id"), &s)
	inst, err := h.d.IM.Start(r.PathValue("id"), s)
	if err != nil {
		writeErr(w, 400, "load failed", err)
		return
	}
	writeJSON(w, 202, inst)
}

// applyMultimodalDefaults enables Jinja when omitted and the model is
// multimodal — most audio/vision chat templates require it.
func (h *handlers) applyMultimodalDefaults(modelID string, s *instances.LoadSettings) {
	if s.Jinja != nil {
		return
	}
	m, err := h.d.Lib.Get(modelID)
	if err != nil {
		return
	}
	var meta struct {
		HasAudio         bool `json:"has_audio"`
		HasVision        bool `json:"has_vision"`
		Multimodal       bool `json:"multimodal"`
		SpeculativeDraft bool `json:"speculative_draft"`
		IsEmbedding      bool `json:"is_embedding"`
	}
	_ = json.Unmarshal(m.Metadata, &meta)
	if meta.SpeculativeDraft || meta.IsEmbedding {
		return
	}
	if m.ProjectorPath != "" || meta.HasAudio || meta.HasVision || meta.Multimodal {
		t := true
		s.Jinja = &t
	}
}

// applyEmbedderDefaults turns on --embedding for detected embedders/rerankers
// and prefills --pooling from GGUF metadata when the client left it unset.
func (h *handlers) applyEmbedderDefaults(modelID string, s *instances.LoadSettings) {
	m, err := h.d.Lib.Get(modelID)
	if err != nil {
		return
	}
	var meta struct {
		IsEmbedding bool   `json:"is_embedding"`
		IsReranker  bool   `json:"is_reranker"`
		PoolingType string `json:"pooling_type"`
	}
	_ = json.Unmarshal(m.Metadata, &meta)
	if !meta.IsEmbedding && !meta.IsReranker {
		return
	}
	if s.Embedding == nil {
		t := true
		s.Embedding = &t
	}
	if strings.TrimSpace(s.Pooling) == "" {
		if meta.IsReranker {
			s.Pooling = "rank"
		} else if p := strings.TrimSpace(meta.PoolingType); p != "" {
			s.Pooling = p
		}
	}
}

func (h *handlers) unloadModel(w http.ResponseWriter, r *http.Request) {
	force := r.URL.Query().Get("force") == "1"
	if err := h.d.IM.Stop(r.PathValue("id"), force); err != nil {
		writeErr(w, 500, "unload failed", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) restartModel(w http.ResponseWriter, r *http.Request) {
	s, ok := decodeSettings(w, r)
	if !ok {
		return
	}
	h.applyMultimodalDefaults(r.PathValue("id"), &s)
	h.applyEmbedderDefaults(r.PathValue("id"), &s)
	inst, err := h.d.IM.Restart(r.PathValue("id"), s)
	if err != nil {
		writeErr(w, 400, "restart failed", err)
		return
	}
	writeJSON(w, 202, inst)
}

func (h *handlers) listInstances(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{"instances": h.d.IM.List()})
}

// modelActivity returns the last /slots sample for a loaded model: busy
// flag, active request count and per-request token counts.
func (h *handlers) modelActivity(w http.ResponseWriter, r *http.Request) {
	act, _ := h.d.IM.Activity(r.PathValue("id"))
	writeJSON(w, 200, map[string]any{"activity": act})
}

func (h *handlers) modelLogs(w http.ResponseWriter, r *http.Request) {
	text, err := h.d.IM.Logs(r.PathValue("id"), 256<<10)
	if err != nil {
		writeErr(w, 500, "reading logs", err)
		return
	}
	writeJSON(w, 200, map[string]string{"log": diagnostics.Redact(text)})
}

// modelDiagnostics assembles the full failure report for a model.
func (h *handlers) modelDiagnostics(w http.ResponseWriter, r *http.Request) {
	m, err := h.d.Lib.Get(r.PathValue("id"))
	if err != nil {
		writeErr(w, 404, "model not found", err)
		return
	}
	inst, _ := h.d.IM.Get(m.ID)
	logTail, _ := h.d.IM.Logs(m.ID, 128<<10)
	redactHome := r.URL.Query().Get("redact_home") == "1"
	if redactHome {
		if home, err := os.UserHomeDir(); err == nil {
			logTail = diagnostics.RedactPaths(logTail, home)
		}
	}
	logTail = diagnostics.Redact(logTail)

	// Sanitize the instance: the args vector contains the per-process API key.
	var instView any
	if inst != nil {
		cp := *inst
		cp.Args = redactArgs(cp.Args)
		cp.Command = diagnostics.Redact(cp.Command)
		instView = cp
	}
	hw := h.hardwareInfo()
	report := map[string]any{
		"model":         m,
		"instance":      instView,
		"log_tail":      logTail,
		"ram_total":     hw.RAMTotal,
		"ram_available": hw.RAMAvailable,
		"gpus":          hw.GPUs,
	}
	if inst != nil {
		// Classify only the current launch's log section: the file is
		// append-only and earlier launches' errors would mislead the match.
		class := diagnostics.Classify(instances.LogSectionForLaunch(logTail, inst.ID), inst.ExitCode, false)
		report["classification"] = class
		if rt, err := h.d.RT.Get(inst.RuntimeID); err == nil {
			report["runtime"] = map[string]any{
				"id": rt.ID, "backend": rt.Backend, "version_output": rt.VersionOutput,
			}
		}
	}
	writeJSON(w, 200, report)
}

func (h *handlers) listDirs(w http.ResponseWriter, r *http.Request) {
	dirs, err := h.d.Lib.Directories()
	if err != nil {
		writeErr(w, 500, "listing directories", err)
		return
	}
	writeJSON(w, 200, map[string]any{"directories": dirs})
}

func (h *handlers) addDir(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Path string `json:"path"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	id, err := h.d.Lib.AddDirectory(req.Path)
	if err != nil {
		writeErr(w, 400, "adding directory", err)
		return
	}
	writeJSON(w, 201, map[string]string{"id": id})
}

func (h *handlers) removeDir(w http.ResponseWriter, r *http.Request) {
	if err := h.d.Lib.RemoveDirectory(r.PathValue("id")); err != nil {
		writeErr(w, 500, "removing directory", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

// redactArgs masks values of sensitive flags in an argument vector.
func redactArgs(args []string) []string {
	out := make([]string, len(args))
	copy(out, args)
	for i, a := range out {
		if (a == "--api-key" || a == "--hf-token" || a == "--api-key-file") && i+1 < len(out) {
			out[i+1] = "[REDACTED]"
		}
	}
	return out
}

func (h *handlers) listConversations(w http.ResponseWriter, r *http.Request) {
	all, err := h.d.Chat.ListConversations(r.URL.Query().Get("archived") == "1")
	if err != nil {
		writeErr(w, 500, "listing conversations", err)
		return
	}
	if all == nil {
		all = []chat.Conversation{}
	}
	writeJSON(w, 200, map[string]any{"conversations": all})
}

func (h *handlers) createConversation(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ModelID string `json:"model_id"`
		Title   string `json:"title"`
		System  string `json:"system"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	c, err := h.d.Chat.CreateConversation(req.ModelID, req.Title, req.System)
	if err != nil {
		writeErr(w, 500, "creating conversation", err)
		return
	}
	writeJSON(w, 201, c)
}

func (h *handlers) patchConversation(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Title    *string         `json:"title"`
		ModelID  *string         `json:"model_id"`
		System   *string         `json:"system"`
		Archived *bool           `json:"archived"`
		Params   json.RawMessage `json:"params"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	id := r.PathValue("id")
	var err error
	switch {
	case req.Title != nil:
		err = h.d.Chat.RenameConversation(id, *req.Title)
	case req.ModelID != nil:
		err = h.d.Chat.SetConversationModel(id, *req.ModelID)
	case req.System != nil:
		err = h.d.Chat.SetSystemPrompt(id, *req.System)
	case req.Archived != nil:
		err = h.d.Chat.ArchiveConversation(id, *req.Archived)
	case len(req.Params) > 0:
		err = h.d.Chat.SetParams(id, req.Params)
	}
	if err != nil {
		writeErr(w, 500, "updating conversation", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) deleteConversation(w http.ResponseWriter, r *http.Request) {
	if err := h.d.Chat.DeleteConversation(r.PathValue("id")); err != nil {
		writeErr(w, 500, "deleting conversation", err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) listMessages(w http.ResponseWriter, r *http.Request) {
	msgs, err := h.d.Chat.Messages(r.PathValue("id"))
	if err != nil {
		writeErr(w, 500, "listing messages", err)
		return
	}
	if msgs == nil {
		msgs = []chat.Message{}
	}
	writeJSON(w, 200, map[string]any{"messages": msgs})
}

func (h *handlers) generate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ParentID string           `json:"parent_id"`
		Content  string           `json:"content"`
		Params   chat.GenParams   `json:"params"`
		Audio    *chat.AudioInput `json:"audio"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	if len(req.Content) > 1<<20 {
		writeErr(w, 400, "message too large", nil)
		return
	}
	if req.Audio != nil {
		if h.d.Settings.Get("experimental.audio_models", "0") != "1" {
			writeErr(w, 400, "experimental audio models are disabled; enable them in Settings", nil)
			return
		}
		conv, err := h.d.Chat.GetConversation(r.PathValue("id"))
		if err != nil {
			writeErr(w, 404, "conversation not found", err)
			return
		}
		mdl, err := h.d.Lib.Get(conv.ModelID)
		if err != nil {
			writeErr(w, 400, "model not found for conversation", err)
			return
		}
		var meta struct {
			HasAudio bool `json:"has_audio"`
		}
		_ = json.Unmarshal(mdl.Metadata, &meta)
		if !meta.HasAudio {
			writeErr(w, 400, "selected model does not support audio input; enable Audio models in Settings and rescan the library if this model has an audio mmproj", nil)
			return
		}
	}
	msgID, err := h.d.Chat.Generate(r.Context(), r.PathValue("id"), req.ParentID, req.Content, req.Params, req.Audio)
	if err != nil {
		writeErr(w, 400, "generation failed to start", err)
		return
	}
	writeJSON(w, 202, map[string]string{"message_id": msgID})
}

func (h *handlers) stopGeneration(w http.ResponseWriter, r *http.Request) {
	h.d.Chat.Stop(r.PathValue("id"))
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) getServer(w http.ResponseWriter, r *http.Request) {
	cfg := h.d.Proxy.Config()
	writeJSON(w, 200, map[string]any{
		"running": h.d.Proxy.Running(), "config": cfg,
		"clients": h.d.Proxy.Clients(),
	})
}

func (h *handlers) putServer(w http.ResponseWriter, r *http.Request) {
	var cfg struct {
		Port      int    `json:"port"`
		Bind      string `json:"bind"`
		AllowLAN  bool   `json:"allow_lan"`
		CORS      string `json:"cors"`
		Autostart bool   `json:"autostart"`
	}
	if !decodeJSON(w, r, &cfg) {
		return
	}
	cur := h.d.Proxy.Config()
	cur.Port = cfg.Port
	cur.Bind = cfg.Bind
	cur.AllowLAN = cfg.AllowLAN
	cur.CORS = cfg.CORS
	cur.Autostart = cfg.Autostart
	if err := h.d.Proxy.UpdateConfig(cur); err != nil {
		writeErr(w, 400, "invalid server config", err)
		return
	}
	// If the public API is already running and HostIt is enabled, refresh
	// the tunnel against the (possibly new) local port.
	if h.d.HostIt != nil && h.d.Proxy.Running() {
		_ = h.d.HostIt.SyncTimeout(h.d.Proxy.Config().Port, true)
	}
	writeJSON(w, 200, map[string]any{
		"ok": true, "config": h.d.Proxy.Config(),
		"restart_required": h.d.Proxy.Running(),
	})
}

func (h *handlers) serverStart(w http.ResponseWriter, r *http.Request) {
	if err := h.d.Proxy.Start(); err != nil {
		writeErr(w, 400, "starting server failed", err)
		return
	}
	if h.d.HostIt != nil {
		if err := h.d.HostIt.SyncTimeout(h.d.Proxy.Config().Port, true); err != nil {
			// Public API is up; HostIt failure is reported but not fatal.
			writeJSON(w, 200, map[string]any{"ok": true, "hostit_error": err.Error()})
			return
		}
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) serverStop(w http.ResponseWriter, r *http.Request) {
	if h.d.HostIt != nil {
		_ = h.d.HostIt.SyncTimeout(0, false)
	}
	h.d.Proxy.Stop()
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (h *handlers) getHostIt(w http.ResponseWriter, r *http.Request) {
	if h.d.HostIt == nil {
		writeJSON(w, 200, map[string]any{"enabled": false, "available": false})
		return
	}
	writeJSON(w, 200, h.d.HostIt.Status(r.Context()))
}

func (h *handlers) putHostIt(w http.ResponseWriter, r *http.Request) {
	if h.d.HostIt == nil {
		writeErr(w, 500, "hostit bridge unavailable", nil)
		return
	}
	var req struct {
		Enabled   bool   `json:"enabled"`
		AgentURL  string `json:"agent_url"`
		Domain    string `json:"domain"`
		RouteName string `json:"route_name"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	h.d.HostIt.SetConfig(req.Enabled, req.AgentURL, req.Domain, req.RouteName)
	running := h.d.Proxy.Running()
	err := h.d.HostIt.SyncTimeout(h.d.Proxy.Config().Port, running)
	out := h.d.HostIt.Status(r.Context())
	out["ok"] = err == nil
	if err != nil {
		out["sync_error"] = err.Error()
	}
	writeJSON(w, 200, out)
}

func (h *handlers) syncHostIt(w http.ResponseWriter, r *http.Request) {
	if h.d.HostIt == nil {
		writeErr(w, 500, "hostit bridge unavailable", nil)
		return
	}
	running := h.d.Proxy.Running()
	err := h.d.HostIt.SyncTimeout(h.d.Proxy.Config().Port, running)
	out := h.d.HostIt.Status(r.Context())
	out["ok"] = err == nil
	if err != nil {
		out["sync_error"] = err.Error()
		writeJSON(w, 200, out)
		return
	}
	writeJSON(w, 200, out)
}

func (h *handlers) serverRegenKey(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]string{"api_key": h.d.Proxy.RegenerateKey()})
}

func (h *handlers) serverRequests(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{"requests": h.d.Proxy.RecentRequests()})
}

func (h *handlers) logFiles(w http.ResponseWriter, r *http.Request) {
	type lf struct {
		Name string `json:"name"`
		Size int64  `json:"size"`
		Dir  string `json:"dir"`
	}
	var out []lf
	for _, dir := range []string{h.d.Layout.AppLogs, h.d.Layout.InstLogs} {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".log") {
				continue
			}
			st, err := e.Info()
			if err != nil {
				continue
			}
			out = append(out, lf{Name: e.Name(), Size: st.Size(), Dir: dir})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	if out == nil {
		out = []lf{}
	}
	writeJSON(w, 200, map[string]any{"files": out, "app_log_dir": h.d.Layout.AppLogs, "instance_log_dir": h.d.Layout.InstLogs})
}

func (h *handlers) logTail(w http.ResponseWriter, r *http.Request) {
	name := filepath.Base(r.URL.Query().Get("name"))
	if name == "" || name == "." {
		writeErr(w, 400, "name required", nil)
		return
	}
	var path string
	for _, dir := range []string{h.d.Layout.AppLogs, h.d.Layout.InstLogs} {
		p := filepath.Join(dir, name)
		if _, err := os.Stat(p); err == nil {
			path = p
			break
		}
	}
	if path == "" {
		writeErr(w, 404, "log file not found", nil)
		return
	}
	f, err := os.Open(path)
	if err != nil {
		writeErr(w, 500, "reading log", err)
		return
	}
	defer f.Close()
	st, _ := f.Stat()
	const maxTail = 512 << 10
	start := int64(0)
	if st.Size() > maxTail {
		start = st.Size() - maxTail
	}
	buf := make([]byte, st.Size()-start)
	n, _ := f.ReadAt(buf, start)
	writeJSON(w, 200, map[string]string{"content": diagnostics.Redact(string(buf[:n]))})
}
