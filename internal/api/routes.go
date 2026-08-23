package api

import (
	"net/http"

	"github.com/openinfer/openinfer-studio/internal/chat"
	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/internal/diagnostics"
	"github.com/openinfer/openinfer-studio/internal/downloads"
	"github.com/openinfer/openinfer-studio/internal/hostit"
	"github.com/openinfer/openinfer-studio/internal/huggingface"
	"github.com/openinfer/openinfer-studio/internal/instances"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/proxy"
	"github.com/openinfer/openinfer-studio/internal/quantize"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

// Deps are the backend services exposed over the control API.
type Deps struct {
	Hub      *Hub
	Layout   *config.Layout
	DB       *database.DB
	Settings *config.Settings
	HF       *huggingface.Client
	DL       *downloads.Manager
	RT       *runtimes.Manager
	Lib      *models.Library
	IM       *instances.Manager
	Chat     *chat.Service
	Proxy    *proxy.Server
	HostIt   *hostit.Bridge
	Logs     *diagnostics.Manager
	Quant    *quantize.Manager
}

// RegisterRoutes wires every REST endpoint and the event WebSocket.
func (s *Server) RegisterRoutes(d *Deps) {
	h := &handlers{d: d}

	s.Handle("GET /api/v1/status", h.status)
	s.Handle("GET /api/v1/hardware", h.hardware)

	s.Handle("GET /api/v1/settings", h.getSettings)
	s.Handle("PUT /api/v1/settings/{key}", h.putSetting)

	s.Handle("GET /api/v1/hf/search", h.hfSearch)
	s.Handle("GET /api/v1/hf/repo/{author}/{name}", h.hfRepo)
	s.Handle("GET /api/v1/hf/token", h.hfTokenStatus)
	s.Handle("PUT /api/v1/hf/token", h.hfSetToken)
	s.Handle("DELETE /api/v1/hf/token", h.hfDeleteToken)

	s.Handle("GET /api/v1/downloads", h.listDownloads)
	s.Handle("POST /api/v1/downloads", h.enqueueDownload)
	s.Handle("POST /api/v1/downloads/{id}/pause", h.dlAction)
	s.Handle("POST /api/v1/downloads/{id}/resume", h.dlAction)
	s.Handle("POST /api/v1/downloads/{id}/cancel", h.dlAction)
	s.Handle("POST /api/v1/downloads/{id}/retry", h.dlAction)
	s.Handle("POST /api/v1/downloads/{id}/reorder", h.dlReorder)
	s.Handle("DELETE /api/v1/downloads/{id}", h.dlDelete)

	s.Handle("GET /api/v1/models", h.listModels)
	s.Handle("POST /api/v1/models/scan", h.scanModels)
	s.Handle("POST /api/v1/models/import", h.importModel)
	s.Handle("GET /api/v1/models/{id}", h.getModel)
	s.Handle("PATCH /api/v1/models/{id}", h.patchModel)
	s.Handle("DELETE /api/v1/models/{id}", h.deleteModel)
	s.Handle("GET /api/v1/models/{id}/presets", h.listPresets)
	s.Handle("POST /api/v1/models/{id}/presets", h.savePreset)
	s.Handle("PUT /api/v1/models/{id}/presets/{pid}", h.savePreset)
	s.Handle("DELETE /api/v1/models/{id}/presets/{pid}", h.deletePreset)
	s.Handle("GET /api/v1/models/{id}/logs", h.modelLogs)
	s.Handle("POST /api/v1/models/{id}/preview", h.previewLoad)
	s.Handle("POST /api/v1/models/{id}/estimate", h.estimateLoad)
	s.Handle("GET /api/v1/models/{id}/draft-candidates", h.draftCandidates)
	s.Handle("POST /api/v1/models/{id}/load", h.loadModel)
	s.Handle("POST /api/v1/models/{id}/unload", h.unloadModel)
	s.Handle("POST /api/v1/models/{id}/restart", h.restartModel)
	s.Handle("GET /api/v1/models/{id}/diagnostics", h.modelDiagnostics)
	s.Handle("GET /api/v1/models/{id}/activity", h.modelActivity)

	s.Handle("GET /api/v1/directories", h.listDirs)
	s.Handle("POST /api/v1/directories", h.addDir)
	s.Handle("DELETE /api/v1/directories/{id}", h.removeDir)

	s.Handle("GET /api/v1/runtimes", h.listRuntimes)
	s.Handle("GET /api/v1/runtimes/releases", h.listReleases)
	s.Handle("POST /api/v1/runtimes/install", h.installRuntime)
	s.Handle("POST /api/v1/runtimes/import", h.importRuntime)
	s.Handle("DELETE /api/v1/runtimes/{id}", h.removeRuntime)
	s.Handle("POST /api/v1/runtimes/{id}/preferred", h.preferRuntime)
	s.Handle("POST /api/v1/runtimes/{id}/health", h.healthRuntime)
	s.Handle("GET /api/v1/runtimes/{id}/capabilities", h.runtimeCaps)
	s.Handle("GET /api/v1/runtimes/{id}/tools", h.runtimeTools)

	s.Handle("GET /api/v1/quantize/types", h.quantizeTypes)
	s.Handle("GET /api/v1/quantize/from-hf/preview", h.quantizeFromHFPreview)
	s.Handle("POST /api/v1/quantize/preview", h.quantizePreview)
	s.Handle("POST /api/v1/quantize/jobs", h.startQuantizeJob)
	s.Handle("GET /api/v1/quantize/jobs", h.listQuantizeJobs)
	s.Handle("POST /api/v1/quantize/jobs/clear-history", h.clearQuantizeHistory)
	s.Handle("GET /api/v1/quantize/jobs/{id}", h.getQuantizeJob)
	s.Handle("POST /api/v1/quantize/jobs/{id}/cancel", h.cancelQuantizeJob)
	s.Handle("POST /api/v1/quantize/jobs/{id}/pause", h.pauseQuantizeJob)
	s.Handle("POST /api/v1/quantize/jobs/{id}/resume", h.resumeQuantizeJob)
	s.Handle("DELETE /api/v1/quantize/jobs/{id}", h.deleteQuantizeJob)
	s.Handle("GET /api/v1/quantize/imatrices", h.listIMatrices)
	s.Handle("POST /api/v1/quantize/imatrices/import", h.importIMatrix)
	s.Handle("DELETE /api/v1/quantize/imatrices/{id}", h.deleteIMatrix)

	s.Handle("GET /api/v1/instances", h.listInstances)

	s.Handle("GET /api/v1/chat", h.listConversations)
	s.Handle("POST /api/v1/chat", h.createConversation)
	s.Handle("PATCH /api/v1/chat/{id}", h.patchConversation)
	s.Handle("DELETE /api/v1/chat/{id}", h.deleteConversation)
	s.Handle("GET /api/v1/chat/{id}/messages", h.listMessages)
	s.Handle("POST /api/v1/chat/{id}/generate", h.generate)
	s.Handle("POST /api/v1/chat/{id}/stop", h.stopGeneration)

	s.Handle("GET /api/v1/server", h.getServer)
	s.Handle("PUT /api/v1/server", h.putServer)
	s.Handle("POST /api/v1/server/start", h.serverStart)
	s.Handle("POST /api/v1/server/stop", h.serverStop)
	s.Handle("POST /api/v1/server/regenerate-key", h.serverRegenKey)
	s.Handle("GET /api/v1/server/requests", h.serverRequests)

	s.Handle("GET /api/v1/hostit", h.getHostIt)
	s.Handle("PUT /api/v1/hostit", h.putHostIt)
	s.Handle("POST /api/v1/hostit/sync", h.syncHostIt)

	s.Handle("GET /api/v1/logs/files", h.logFiles)
	s.Handle("GET /api/v1/logs/tail", h.logTail)

	s.HandleEvents("/api/v1/events")
}

var _ = http.MethodGet
