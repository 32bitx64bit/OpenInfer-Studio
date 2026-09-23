import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import ".."
import "../components"

// Image / video generation studio backed by stable-diffusion.cpp (sd-server).
//
// Model picker lists diffusion checkpoints from the library (modality ===
// "diffusion"). Mode selector switches txt2img / img2img / video; the
// backend auto-starts the sd-server on first generate. Preview shows the
// latest output; history lists recent media jobs with local file URLs.
Item {
    id: page
    property var api
    property var events

    property var library: []
    property string selectedModelId: ""
    onSelectedModelIdChanged: {
        page.capabilities = null
        page.serverState = ""
        page.serverError = ""
        page.currentServer = null
        page.reloadServers()
    }
    property string mode: "txt2img"       // txt2img|img2img|video
    property string prompt: ""
    property string negativePrompt: ""
    property int genWidth: 512
    property int genHeight: 512
    property int steps: 20
    property real cfgScale: 7.0
    property int seed: -1
    property string sampler: ""
    property string scheduler: ""
    property int batchCount: 1
    property int videoFrames: 33
    property int fps: 16
    property string outputFormat: "png"
    property real strength: 0.75
    property real guidance: 0            // distilled guidance; 0 = default
    property string initImagePath: ""
    property string initImageName: ""
    property string initImageUrl: ""     // file:// URL, for local preview only

    property bool generating: false
    property bool canceling: false
    property string currentJobId: ""
    property string previewUrl: ""
    property string previewKind: "image"
    property string errorText: ""
    property string progressMessage: ""
    property string progressSdStatus: ""
    property var jobs: []

    // sd-server status for the selected model, from /api/v1/media/servers.
    // "" = not loaded (auto-starts on Generate with last-known-good settings).
    property string serverState: ""
    property string serverError: ""
    property var currentServer: null

    // /api/v1/models/{id}/media/capabilities for the selected model, once
    // its server is ready. Seeded once per model id so it does not clobber
    // user edits on later refreshes.
    property var capabilities: null
    property var capabilitiesSeededFor: ({})

    signal configureLibrary()
    signal loadModel(string modelId)

    function isImageGen(m) {
        if (!m) return false
        var meta = m.metadata || {}
        // Components (pipeline-only checkpoints: VAE/text-encoder splits)
        // are never selectable models, even if a metadata flag for it lands.
        if (meta.is_component || meta.component) return false
        if (m.modality === "diffusion" || meta.modality === "diffusion") return true
        if (meta.diffusion_kind || meta.sd_family) return true
        if (meta.is_diffusion) return false
        var path = String(m.primary_path || m.alias || "").toLowerCase()
        var fams = ["qwen-image", "qwen_image", "flux", "sdxl", "sd-xl",
                    "stable-diffusion", "stable_diffusion", "chroma", "z-image",
                    "wan2", "wan-2", "wanx", "ltx", "hunyuanvideo", "hunyuan-video"]
        for (var i = 0; i < fams.length; i++) {
            if (path.indexOf(fams[i]) >= 0) return true
        }
        return false
    }

    // file:///C:/Users/... -> C:/Users/... ; file:///home/x -> /home/x
    function fileUrlToLocalPath(fileUrl) {
        var s = String(fileUrl)
        if (s.indexOf("file://") !== 0) return s
        s = s.substring(7)
        try { s = decodeURIComponent(s) } catch (e) {}
        if (/^\/[A-Za-z]:\//.test(s)) s = s.substring(1)
        return s
    }

    // Loadable media URL for a job: only file_urls (local file:// paths) are
    // usable directly — the relative /api/v1/media/file/... urls require a
    // bearer token the Image element / Qt.openUrlExternally cannot send.
    function mediaFileUrl(job) {
        if (!job) return ""
        var furls = job.file_urls || []
        return furls.length > 0 ? furls[0] : ""
    }

    // Display-only label when no file_urls are available yet (e.g. a job
    // still queued). Never used as an Image source or openUrlExternally
    // target since it requires auth the UI does not attach here.
    function mediaDisplayLabel(job) {
        if (!job) return ""
        var lp = job.local_paths || []
        if (lp.length > 0) return lp[0]
        var urls = job.urls || []
        if (urls.length > 0) return apiBase + urls[0]
        return ""
    }

    function diffusionModels() {
        return (page.library || []).filter(function(m) { return page.isImageGen(m) })
    }

    function selectedModel() {
        for (var i = 0; i < page.library.length; i++) {
            if (page.library[i].id === page.selectedModelId) return page.library[i]
        }
        return null
    }

    function reload() {
        api.get("/api/v1/models", function(st, data) {
            if (st !== 200) return
            page.library = (data && data.models) || []
            var mods = page.diffusionModels()
            if (!page.selectedModelId && mods.length > 0) page.selectedModelId = mods[0].id
            else if (page.selectedModelId && !page.selectedModel()) {
                page.selectedModelId = mods.length > 0 ? mods[0].id : ""
            }
        })
        page.reloadJobs()
        page.reloadServers()
    }

    function reloadJobs() {
        api.get("/api/v1/media/jobs", function(st, data) {
            if (st !== 200) return
            page.jobs = (data && data.jobs) || []
        })
    }

    // Live sd-server status for the selected model (GET /media/servers).
    function reloadServers() {
        api.get("/api/v1/media/servers", function(st, data) {
            if (st !== 200) return
            var servers = (data && data.servers) || []
            page.currentServer = null
            for (var i = 0; i < servers.length; i++) {
                if (servers[i].model_id === page.selectedModelId) {
                    page.currentServer = servers[i]
                    break
                }
            }
            page.serverState = page.currentServer ? (page.currentServer.state || "") : ""
            page.serverError = page.currentServer ? (page.currentServer.error || "") : ""
            if (page.serverState === "ready") page.loadCapabilities()
        })
    }

    // Sampler/scheduler/format lists and sane defaults come from the running
    // server's capability document — flow models (Qwen-Image/FLUX) reject
    // euler_a + discrete + CFG 7.0, so the UI must not hardcode them.
    function loadCapabilities() {
        if (!page.selectedModelId) return
        api.get("/api/v1/models/" + page.selectedModelId + "/media/capabilities", function(st, data) {
            if (st !== 200 || !data) return
            page.capabilities = data
            // Seed defaults once per model so later refreshes never clobber
            // user edits.
            if (!page.capabilitiesSeededFor[page.selectedModelId]) {
                page.capabilitiesSeededFor[page.selectedModelId] = true
                var modes = data.supported_modes || ["img_gen"]
                var modeKey = page.mode === "video" ? "vid_gen" : "img_gen"
                if (modes.indexOf(modeKey) < 0) modeKey = modes.length > 0 ? modes[0] : "img_gen"
                var d = (data.defaults_by_mode || {})[modeKey] || {}
                if (d.width) page.genWidth = d.width
                if (d.height) page.genHeight = d.height
                if (typeof d.seed === "number") page.seed = d.seed
                var sp = d.sample_params || {}
                if (sp.sample_steps) page.steps = sp.sample_steps
                var g = sp.guidance || {}
                if (typeof g.txt_cfg === "number") page.cfgScale = g.txt_cfg
                if (typeof g.distilled_guidance === "number") page.guidance = g.distilled_guidance
            }
        })
    }

    function samplerModel() {
        var list = (page.capabilities && page.capabilities.samplers) || []
        return [""].concat(list)
    }

    function schedulerModel() {
        var list = (page.capabilities && page.capabilities.schedulers) || []
        return [""].concat(list)
    }

    function formatModel() {
        if (page.mode === "video") {
            var vf = (page.capabilities && page.capabilities.output_formats_by_mode
                      && page.capabilities.output_formats_by_mode.vid_gen) || ["webm", "avi"]
            return vf
        }
        var imf = (page.capabilities && page.capabilities.output_formats_by_mode
                   && page.capabilities.output_formats_by_mode.img_gen) || ["png", "jpeg", "webp"]
        return imf
    }

    function modeSupported(m) {
        if (!page.capabilities || !page.capabilities.supported_modes) return true
        var modes = page.capabilities.supported_modes
        return m === "video" ? modes.indexOf("vid_gen") >= 0 : modes.indexOf("img_gen") >= 0
    }

    function unloadCurrent() {
        if (!page.selectedModelId) return
        api.post("/api/v1/models/" + page.selectedModelId + "/media/server/stop", {}, function() {
            page.reloadServers()
        })
    }

    function mediaUrl(job) {
        if (!job) return ""
        var urls = job.urls || []
        if (urls.length > 0) return urls[0]
        return ""
    }

    function pickInitImage() {
        initImageDialog.open()
    }

    function clearInitImage() {
        page.initImagePath = ""
        page.initImageName = ""
        page.initImageUrl = ""
    }

    function generate() {
        page.errorText = ""
        if (!page.selectedModelId) { page.errorText = "Pick a diffusion checkpoint first."; return }
        if (page.prompt.trim() === "") { page.errorText = "Prompt is required."; return }
        if (page.mode === "img2img" && page.initImagePath === "") {
            page.errorText = "Choose an input image first."
            return
        }
        var kind = page.mode === "video" ? "video" : "image"
        var body = {
            "kind": kind,
            "prompt": page.prompt,
            "negative_prompt": page.negativePrompt,
            "width": page.genWidth,
            "height": page.genHeight,
            "steps": page.steps,
            "cfg_scale": page.cfgScale,
            "seed": page.seed,
            "sampler": page.sampler,
            "scheduler": page.scheduler,
            "guidance": page.guidance,
            "output_format": kind === "video"
                ? (page.outputFormat === "png" ? "webm" : page.outputFormat)
                : page.outputFormat
        }
        if (kind === "image") {
            body.batch_count = page.batchCount
            if (page.mode === "img2img") {
                body.strength = page.strength
                // The backend reads and encodes the file itself; QML has no
                // FileReader, and a raw path keeps large images off the JSON
                // control path.
                if (page.initImagePath !== "") body.init_image_path = page.initImagePath
            }
        } else {
            body.video_frames = page.videoFrames
            body.fps = page.fps
        }
        page.generating = true
        // 202 returns immediately with a queued job; media.progress events
        // drive completion. Stay in the generating state until one lands.
        api.post("/api/v1/models/" + page.selectedModelId + "/media/generate", body, function(st, data) {
            if (st === 202 && data) {
                page.currentJobId = data.id || ""
                page.previewKind = kind
                if (page.mediaUrl(data)) page.previewUrl = page.mediaUrl(data)
                page.reloadJobs()
                // Terminal on arrival (e.g. instant failure): settle now.
                if (data.state === "complete") {
                    page.generating = false
                    if (page.mediaUrl(data)) page.previewUrl = page.mediaUrl(data)
                } else if (data.state === "failed" || data.state === "canceled") {
                    page.generating = false
                    page.errorText = data.error || ("Job " + data.state)
                }
            } else {
                page.generating = false
                page.errorText = (data && (data.detail || data.error)) || ("HTTP " + st)
            }
        })
    }

    function cancelCurrent() {
        if (!page.currentJobId || page.canceling) return
        page.canceling = true
        api.post("/api/v1/media/jobs/" + page.currentJobId + "/cancel", {}, function() {
            // The canceled media.progress event (or this response) confirms
            // the cancel; settle then rather than optimistically clearing.
            page.canceling = false
            page.generating = false
            page.reloadJobs()
        })
    }

    Connections {
        target: page.events
        function onEventReceived(name, payload) {
            if (name === "media.progress" && payload) {
                if (payload.id === page.currentJobId) {
                    page.progressMessage = payload.message || ""
                    page.progressSdStatus = payload.sd_status || ""
                    if (payload.state === "complete") {
                        page.generating = false
                        page.progressMessage = ""
                        page.progressSdStatus = ""
                        page.reloadJobs()
                        // Refresh the preview from the finished job row.
                        api.get("/api/v1/media/jobs/" + payload.id, function(st, data) {
                            if (st === 200 && page.mediaFileUrl(data)) page.previewUrl = page.mediaFileUrl(data)
                        })
                    } else if (payload.state === "failed" || payload.state === "canceled") {
                        page.generating = false
                        page.progressMessage = ""
                        page.progressSdStatus = ""
                        if (payload.state === "failed") {
                            page.errorText = payload.error || "Job failed"
                        }
                        page.reloadJobs()
                    }
                } else {
                    // Only reload the history on terminal states — a foreign
                    // job's per-tick progress must not thrash the list.
                    var s = payload.state || ""
                    if (s === "complete" || s === "failed" || s === "canceled") page.reloadJobs()
                }
            } else if (name === "media.server_state") {
                page.reloadServers()
                if (payload && payload.model_id === page.selectedModelId
                        && (payload.state === "ready" || payload.state === "stopped")) {
                    page.capabilities = null
                    if (payload.state === "ready") page.loadCapabilities()
                }
            } else if (name === "media.server_starting" || name === "media.server_ready"
                       || name === "media.server_error" || name === "media.server_stopped") {
                page.reloadServers()
            } else if (name === "library.scanned" || name === "library.model_imported"
                || name === "library.model_updated") {
                page.reload()
            }
        }
    }

    RowLayout {
        anchors.fill: parent
        anchors.margins: AppTheme.pad
        spacing: AppTheme.gap * 1.5

        // Left: controls
        ScrollView {
            Layout.preferredWidth: 380
            Layout.fillHeight: true
            clip: true
            ColumnLayout {
                width: 380 - 8
                spacing: AppTheme.gap

                PageHeader {
                    title: "Image Studio"
                    subtitle: "Generate images and video with stable-diffusion.cpp."
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Model"
                    hint: "Diffusion checkpoints from your library."
                    AppComboBox {
                        id: modelCombo
                        width: parent.width
                        textRole: "alias"
                        valueRole: "id"
                        subtitleRole: "quantization"
                        model: page.diffusionModels()
                        currentIndex: {
                            var mods = page.diffusionModels()
                            for (var i = 0; i < mods.length; i++) {
                                if (mods[i].id === page.selectedModelId) return i
                            }
                            return -1
                        }
                        onActivated: function(i) {
                            var m = page.diffusionModels()[i]
                            if (m) page.selectedModelId = m.id
                        }
                        onModelChanged: {
                            // Reselect after library reloads; diffusionModels()
                            // is a fresh array each evaluation.
                            var mods = page.diffusionModels()
                            for (var i = 0; i < mods.length; i++) {
                                if (mods[i].id === page.selectedModelId) {
                                    modelCombo.currentIndex = i
                                    return
                                }
                            }
                            modelCombo.currentIndex = -1
                        }
                    }
                }

                // Live sd-server status for the selected model. Generate
                // still works when unloaded — the backend auto-starts with
                // last-known-good settings.
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Rectangle {
                        width: 10; height: 10; radius: 5
                        color: page.serverState === "ready" ? AppTheme.success
                             : page.serverState === "starting" ? AppTheme.warning
                             : page.serverState === "failed" ? AppTheme.danger
                             : AppTheme.textFaint
                    }
                    Label {
                        Layout.fillWidth: true
                        text: {
                            if (page.serverState === "ready") return "Server ready"
                            if (page.serverState === "starting") return "Loading model…"
                            if (page.serverState === "failed") return "Failed: " + (page.serverError || "unknown error")
                            return "Not loaded — Generate will start it automatically"
                        }
                        color: page.serverState === "failed" ? AppTheme.danger : AppTheme.text
                        elide: Text.ElideRight
                    }
                    AppButton {
                        visible: page.serverState !== "ready" && page.serverState !== "starting"
                        text: "Load…"
                        flat: true
                        onClicked: page.loadModel(page.selectedModelId)
                    }
                    AppButton {
                        visible: page.serverState === "ready" || page.serverState === "starting"
                        text: "Unload"
                        flat: true
                        onClicked: page.unloadCurrent()
                    }
                }
                Label {
                    visible: page.serverState === "failed" && page.currentServer
                             && (page.currentServer.log_tail || "") !== ""
                    Layout.fillWidth: true
                    text: page.currentServer ? (page.currentServer.log_tail || "") : ""
                    color: AppTheme.textFaint
                    font.family: "monospace"
                    font.pixelSize: AppTheme.fontSmall
                    wrapMode: Text.WrapAnywhere
                    maximumLineCount: 8
                    elide: Text.ElideRight
                }

                Label {
                    visible: page.diffusionModels().length === 0
                    Layout.fillWidth: true
                    text: page.selectedModelId === ""
                        ? "No diffusion checkpoints yet. Download one from Discover (Image / video corpus) or import a .safetensors file — then press Rescan in My library."
                        : "Selected checkpoint is no longer in the library. Pick another model."
                    color: AppTheme.warning
                    wrapMode: Text.WordWrap
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Mode"
                    hint: "txt2img, img2img (strength), or video."
                    AppComboBox {
                        id: modeCombo
                        width: parent.width
                        textRole: "text"
                        valueRole: "value"
                        model: [
                            { "text": "Text → image", "value": "txt2img" },
                            { "text": "Image → image", "value": "img2img" },
                            { "text": "Text → video", "value": "video" }
                        ]
                        currentIndex: page.mode === "img2img" ? 1 : (page.mode === "video" ? 2 : 0)
                        onActivated: function(i) {
                            page.mode = ["txt2img", "img2img", "video"][i] || "txt2img"
                            if (page.mode === "video" && page.outputFormat === "png") page.outputFormat = "webm"
                            if (page.mode !== "video" && page.outputFormat === "webm") page.outputFormat = "png"
                        }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Prompt"
                    hint: page.mode === "img2img" ? "Describe what the input image should become." : ""
                    AppTextArea {
                        width: parent.width
                        implicitHeight: 90
                        placeholderText: page.mode === "img2img"
                            ? "Turn this photo into a watercolor painting…"
                            : "A misty harbor at dawn, oil painting…"
                        text: page.prompt
                        onTextChanged: page.prompt = text
                    }
                }

                // img2img input: picked image is read as a data URL and sent
                // as init_image; prompt describes the transformation.
                FormField {
                    visible: page.mode === "img2img"
                    Layout.fillWidth: true
                    label: "Input image"
                    hint: "The starting image to transform."
                    ColumnLayout {
                        width: parent.width
                        spacing: 6
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            Rectangle {
                                Layout.preferredWidth: 72
                                Layout.preferredHeight: 72
                                radius: AppTheme.radiusSmall
                                color: AppTheme.surfaceHi
                                border.color: AppTheme.border
                                border.width: 1
                                clip: true
                                Image {
                                    anchors.fill: parent
                                    anchors.margins: 2
                                    visible: page.initImageUrl !== ""
                                    source: page.initImageUrl
                                    fillMode: Image.PreserveAspectCrop
                                    asynchronous: true
                                    cache: false
                                }
                                Label {
                                    anchors.centerIn: parent
                                    visible: page.initImageUrl === ""
                                    text: "🖼"
                                    font.pixelSize: 26
                                    color: AppTheme.textFaint
                                }
                            }
                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 2
                                Label {
                                    Layout.fillWidth: true
                                    text: page.initImageName !== "" ? page.initImageName : "No image selected"
                                    color: page.initImageName !== "" ? AppTheme.text : AppTheme.textFaint
                                    elide: Text.ElideMiddle
                                }
                                Label {
                                    visible: page.initImageName !== ""
                                    Layout.fillWidth: true
                                    text: page.initImagePath
                                    color: AppTheme.textFaint
                                    font.pixelSize: AppTheme.fontSmall
                                    elide: Text.ElideMiddle
                                }
                            }
                        }
                        RowLayout {
                            spacing: 8
                            AppButton { text: "Choose image…"; onClicked: page.pickInitImage() }
                            AppButton {
                                visible: page.initImageUrl !== ""
                                text: "Clear"
                                flat: true
                                onClicked: page.clearInitImage()
                            }
                        }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Negative prompt"
                    AppTextArea {
                        width: parent.width
                        implicitHeight: 56
                        placeholderText: "blurry, watermark…"
                        text: page.negativePrompt
                        onTextChanged: page.negativePrompt = text
                    }
                }

                // Size + sampling in one combined surface per pair: outer
                // arrows on their respective sides, divider in the middle.
                FormField {
                    Layout.fillWidth: true
                    label: "Size"
                    hint: "Output resolution in pixels."
                    DualStepper {
                        width: parent.width
                        leftTitle: "Width"
                        rightTitle: "Height"
                        leftFrom: 64; leftTo: 4096; leftStep: 64
                        rightFrom: 64; rightTo: 4096; rightStep: 64
                        leftValue: page.genWidth
                        rightValue: page.genHeight
                        onLeftValueChanged: page.genWidth = leftValue
                        onRightValueChanged: page.genHeight = rightValue
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Sampling"
                    hint: "Denoising steps and RNG seed (-1 = random)."
                    DualStepper {
                        width: parent.width
                        leftTitle: "Steps"
                        rightTitle: "Seed"
                        leftFrom: 1; leftTo: 300; leftStep: 1
                        rightFrom: -1; rightTo: 2147483647; rightStep: 1
                        leftValue: page.steps
                        rightValue: page.seed
                        onLeftValueChanged: page.steps = leftValue
                        onRightValueChanged: page.seed = rightValue
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Guidance (CFG) — " + page.cfgScale.toFixed(1)
                    AppSlider { width: parent.width; from: 0; to: 30; stepSize: 0.5; value: page.cfgScale; onValueChanged: page.cfgScale = value }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    FormField {
                        Layout.fillWidth: true
                        label: "Sampler"
                        hint: "Default = server per-model default."
                        AppComboBox {
                            width: parent.width
                            model: page.samplerModel()
                            currentIndex: Math.max(0, model.indexOf(page.sampler))
                            onActivated: function(i) { page.sampler = model[i] }
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Scheduler"
                        hint: "Default = server per-model default."
                        AppComboBox {
                            width: parent.width
                            model: page.schedulerModel()
                            currentIndex: Math.max(0, model.indexOf(page.scheduler))
                            onActivated: function(i) { page.scheduler = model[i] }
                        }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Distilled guidance — " + page.guidance.toFixed(1)
                    hint: "Flow-model guidance (0 = server default)."
                    AppSlider { width: parent.width; from: 0; to: 20; stepSize: 0.25; value: page.guidance; onValueChanged: page.guidance = value }
                }

                FormField {
                    visible: page.mode !== "video"
                    Layout.fillWidth: true
                    label: "Batch count"
                    AppSpinBox { width: parent.width; from: 1; to: 8; value: page.batchCount; onValueChanged: page.batchCount = value }
                }

                FormField {
                    visible: page.mode === "img2img"
                    Layout.fillWidth: true
                    label: "Denoise strength — " + page.strength.toFixed(2)
                    hint: "Higher = further from the input image."
                    AppSlider { width: parent.width; from: 0; to: 1; stepSize: 0.05; value: page.strength; onValueChanged: page.strength = value }
                }

                RowLayout {
                    visible: page.mode === "video"
                    Layout.fillWidth: true
                    spacing: 8
                    FormField {
                        Layout.fillWidth: true
                        label: "Frames"
                        AppSpinBox { width: parent.width; from: 1; to: 512; value: page.videoFrames; onValueChanged: page.videoFrames = value }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "FPS"
                        AppSpinBox { width: parent.width; from: 1; to: 60; value: page.fps; onValueChanged: page.fps = value }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Format"
                    AppComboBox {
                        width: parent.width
                        model: page.formatModel()
                        currentIndex: Math.max(0, model.indexOf(page.outputFormat))
                        onActivated: function(i) { page.outputFormat = model[i] }
                    }
                }

                Label {
                    visible: page.errorText !== ""
                    Layout.fillWidth: true
                    text: page.errorText
                    color: AppTheme.danger
                    wrapMode: Text.WordWrap
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    AppButton {
                        text: page.generating ? "Generating…" : "Generate"
                        primary: true
                        // Keep clickable even with no selection so generate()
                        // can explain what is missing instead of doing nothing.
                        enabled: !page.generating
                        Layout.fillWidth: true
                        onClicked: page.generate()
                    }
                    AppButton {
                        visible: page.generating
                        text: page.canceling ? "Canceling…" : "Cancel"
                        enabled: !page.canceling
                        onClicked: page.cancelCurrent()
                    }
                }
                Label {
                    visible: page.generating && (page.progressMessage !== "" || page.progressSdStatus !== "")
                    Layout.fillWidth: true
                    text: page.progressMessage !== "" ? page.progressMessage
                        : (page.progressSdStatus !== "" ? "Generating… (" + page.progressSdStatus + ")" : "")
                    color: AppTheme.textFaint
                    horizontalAlignment: Text.AlignHCenter
                }
                BusyIndicator { visible: page.generating; Layout.alignment: Qt.AlignHCenter }
            }
        }

        // Right: preview + history
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: AppTheme.gap

            Card {
                Layout.fillWidth: true
                Layout.fillHeight: true
                implicitHeight: 320
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 6
                    Label {
                        text: page.previewUrl === "" ? "Preview" : (page.previewKind === "video" ? "Preview (video)" : "Preview")
                        color: AppTheme.text
                        font.weight: Font.DemiBold
                    }
                    Image {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        visible: page.previewUrl !== "" && page.previewKind !== "video"
                        source: page.previewUrl
                        fillMode: Image.PreserveAspectFit
                        asynchronous: true
                        cache: false
                    }
                    Label {
                        visible: page.previewUrl !== "" && page.previewKind === "video"
                        Layout.fillWidth: true
                        text: "Video saved — open from history below."
                        color: AppTheme.textDim
                        wrapMode: Text.WordWrap
                    }
                    Label {
                        visible: page.previewUrl === ""
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        text: "Nothing generated yet."
                        color: AppTheme.textFaint
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                }
            }

            AppGroupBox {
                Layout.fillWidth: true
                Layout.preferredHeight: 220
                title: "History (" + page.jobs.length + ")"
                ListView {
                    anchors.fill: parent
                    clip: true
                    spacing: 6
                    model: page.jobs
                    delegate: Card {
                        width: ListView.view.width - 4
                        implicitHeight: hrow.implicitHeight + 16
                        RowLayout {
                            id: hrow
                            anchors.fill: parent
                            anchors.margins: 8
                            spacing: 10
                            Image {
                                visible: (modelData.urls || []).length > 0 && modelData.kind !== "video"
                                source: (modelData.urls || [])[0] || ""
                                Layout.preferredWidth: 64
                                Layout.preferredHeight: 64
                                fillMode: Image.PreserveAspectCrop
                                asynchronous: true
                                cache: false
                            }
                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 2
                                Label {
                                    text: modelData.prompt || "(no prompt)"
                                    color: AppTheme.text
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }
                                Label {
                                    text: modelData.kind + " · " + modelData.state
                                        + ((modelData.width || 0) > 0 ? " · " + modelData.width + "×" + modelData.height : "")
                                        + " · seed " + modelData.seed
                                    color: AppTheme.textDim
                                    font.pixelSize: AppTheme.fontSmall
                                }
                                Label {
                                    visible: (modelData.error || "") !== ""
                                    text: modelData.error
                                    color: AppTheme.danger
                                    font.pixelSize: AppTheme.fontSmall
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }
                            }
                            AppButton {
                                visible: (modelData.urls || []).length > 0
                                text: modelData.kind === "video" ? "Open" : "View"
                                flat: true
                                onClicked: {
                                    page.previewUrl = (modelData.urls || [])[0]
                                    page.previewKind = modelData.kind
                                    if (modelData.kind === "video")
                                        Qt.openUrlExternally(page.previewUrl)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Component.onCompleted: { reload(); reloadServers() }

    FileDialog {
        id: initImageDialog
        title: "Choose input image"
        fileMode: FileDialog.OpenFile
        nameFilters: ["Images (*.png *.jpg *.jpeg *.webp *.bmp)", "All files (*)"]
        onAccepted: {
            var url = selectedFile.toString()
            var path = page.fileUrlToLocalPath(url)
            page.initImagePath = path
            page.initImageName = path.split(/[/\\]/).pop()
            // Preview directly from the file URL; the backend reads the
            // local path itself (init_image_path) so no FileReader/XHR is
            // needed in QML and large files stay off the JSON control path.
            page.initImageUrl = url
        }
    }
}
