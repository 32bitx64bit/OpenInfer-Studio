import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtCore
import ".."
import "../components"

// Model load configuration: one scrollable page. Advanced/Expert sections
// collapse behind persisted toggles. Memory estimate updates live; the
// generated command preview sits at the very end.
Dialog {
    id: root
    property var api
    property string modelId: ""
    property var model: null
    property var runtimes: []
    property var presets: []

    signal loaded()

    title: model ? ("Load — " + (model.alias || model.id)) : "Load model"
    modal: true
    width: Math.min(680, parent ? parent.width - 64 : 680)
    height: Math.min(720, parent ? parent.height - 48 : 720)
    anchors.centerIn: parent
    padding: AppTheme.padSmall
    transformOrigin: Item.Center
    enter: DialogEnter {}
    exit: DialogExit {}
    Overlay.modal: Rectangle { color: AppTheme.overlay }

    background: Rectangle {
        color: AppTheme.bg
        border.color: AppTheme.border
        radius: AppTheme.radius
    }
    Component.onCompleted: AppTheme.applyPalette(root)

    // Persisted expand/collapse state.
    Settings {
        category: "OpenInferStudio/LoadDialog"
        property alias advancedOpen: advancedToggle.checked
        property alias expertOpen: expertToggle.checked
    }

    property var settings: ({
        "context_length": 4096, "gpu_offload": "all", "gpu_layers": 0,
        "threads": 0, "flash_attention": "auto", "parallel": 0,
        "batch_size": 0, "ubatch_size": 0, "cache_type_k": "", "cache_type_v": "",
        "no_mmap": false, "mlock": false, "main_gpu": -1, "split_mode": "",
        "tensor_split": "", "device": "", "numa": "",
        "cont_batching": null, "cache_reuse": 0,
        "threads_batch": 0, "prio": -2, "poll": -1,
        "cpu_moe": false, "n_cpu_moe": 0,
        "kv_offload": "", "op_offload": "", "kv_unified": "",
        "swa_full": false, "fit": "", "no_warmup": false,
        "rope_scaling": "", "alias": "", "raw_args": "",
        "jinja": false, "no_mmproj": false, "no_mmproj_offload": false,
        "reasoning_preserve": null,
        "draft_model": "", "draft_max": 0, "draft_min": 0, "spec_type": "",
        "embedding": false, "pooling": "",
        "runtime_id": "",
        "save_on_success": true
    })
    property string selectedRuntime: ""   // mirrors settings.runtime_id; "" = auto
    property var preview: null
    property var estimate: null
    property string loadError: ""
    property var draftCandidates: []
    property bool draftFiltered: true
    readonly property bool hasProjector: root.model && root.model.projector_path !== ""
    readonly property bool isEmbedding: !!(root.model && root.model.metadata
        && (root.model.metadata.is_embedding || root.model.metadata.is_reranker))
    readonly property bool isReranker: !!(root.model && root.model.metadata
        && root.model.metadata.is_reranker)
    readonly property bool speculativeEnabled: !!(root.settings.draft_model && root.settings.draft_model !== "")
            || !!(root.settings.spec_type && root.settings.spec_type !== "")
    readonly property bool hasMTP: {
        if (root.isEmbedding) return false
        if (!root.model || !root.model.metadata) return false
        return !!root.model.metadata.has_mtp
    }

    function isMultimodal(m) {
        if (!m) return false
        var meta = m.metadata || {}
        if (meta.speculative_draft || meta.is_embedding || meta.is_reranker) return false
        if (m.projector_path && m.projector_path !== "") return true
        return !!(meta.multimodal || meta.has_audio || meta.has_vision)
    }

    // Muse Glimmer needs --jinja even as a text-only OID quant (no mmproj).
    // Thinking models need it so per-request chat_template_kwargs (reasoning
    // effort / enable_thinking) actually reach the template.
    function needsJinja(m) {
        if (!m) return false
        var meta = m.metadata || {}
        if (meta.speculative_draft || meta.is_embedding || meta.is_reranker) return false
        if (root.isMultimodal(m)) return true
        if (meta.reasoning && meta.reasoning.style) return true
        if (meta.reasoning && meta.reasoning.can_preserve) return true
        var arch = String(m.architecture || "").toLowerCase()
        if (arch.indexOf("assistant") >= 0) return false
        return arch.indexOf("muse-glimmer") >= 0
            || arch.indexOf("muse_glimmer") >= 0
            || arch.indexOf("museglimmer") >= 0
    }

    function canPreserveReasoning(m) {
        if (!m) return false
        var meta = m.metadata || {}
        if (meta.speculative_draft || meta.is_embedding || meta.is_reranker) return false
        return !!(meta.reasoning && meta.reasoning.can_preserve)
    }

    function isMuseGlimmerChat(m) {
        if (!m) return false
        var meta = m.metadata || {}
        if (meta.speculative_draft) return false
        var arch = String(m.architecture || "").toLowerCase()
        if (arch.indexOf("assistant") >= 0) return false
        return arch.indexOf("muse-glimmer") >= 0
            || arch.indexOf("muse_glimmer") >= 0
            || arch.indexOf("museglimmer") >= 0
    }

    function defaultPooling(meta) {
        if (!meta) return ""
        if (meta.is_reranker) return "rank"
        return meta.pooling_type || ""
    }

    function openFor(m) {
        model = m
        modelId = m.id
        // Reassign the whole settings object so QML bindings refresh. Mutating
        // keys in place does not notify FormFields still bound to the previous model.
        var meta = m.metadata || {}
        var embedder = !!(meta.is_embedding || meta.is_reranker)
        // Fused-trunk MTP: default on; presets (Last known good) can still override.
        var defaultMTP = !!(meta.has_mtp && !meta.speculative_draft && !embedder)
        var next = {
            "context_length": 4096, "gpu_offload": "all", "gpu_layers": 0,
            "threads": 0, "flash_attention": "auto", "parallel": 0,
            "batch_size": 0, "ubatch_size": 0, "cache_type_k": "", "cache_type_v": "",
            "no_mmap": false, "mlock": false, "main_gpu": -1, "split_mode": "",
            "tensor_split": "", "device": "", "numa": "",
            "cont_batching": null, "cache_reuse": 0,
            "threads_batch": 0, "prio": -2, "poll": -1,
            "cpu_moe": false, "n_cpu_moe": 0,
            "kv_offload": "", "op_offload": "", "kv_unified": "",
            "swa_full": false, "fit": "", "no_warmup": false,
            "rope_scaling": "", "alias": m.alias || "", "raw_args": "",
            "jinja": embedder ? false : root.needsJinja(m),
            "chat_template_kwargs": root.isMuseGlimmerChat(m) ? "{\"reasoning_strength\":\"low\"}" : "",
            "reasoning_preserve": (!embedder && root.canPreserveReasoning(m)) ? true : null,
            "no_mmproj": false, "no_mmproj_offload": false,
            "draft_model": "",
            "draft_max": defaultMTP ? 2 : 0,
            "draft_min": 0,
            "spec_type": defaultMTP ? "draft-mtp" : "",
            "embedding": embedder,
            "pooling": embedder ? root.defaultPooling(meta) : "",
            "runtime_id": m.pinned_runtime || "",
            "save_on_success": true
        }

        settings = next
        if (aliasField) aliasField.text = next.alias
        selectedRuntime = next.runtime_id
        loadError = ""
        preview = null
        estimate = null
        draftCandidates = []
        api.get("/api/v1/runtimes", function(st, data) {
            if (st === 200) root.runtimes = (data && data.runtimes) || []
        })
        root.reloadDraftCandidates()
        api.get("/api/v1/models/" + m.id + "/presets", function(st, data) {
            if (st !== 200) return
            // Ignore stale responses if the user opened another model.
            if (root.modelId !== m.id) return
            root.presets = (data && data.presets) || []
            // Prefill from last-known-good, else the default preset.
            var applied = false
            for (var i = 0; i < root.presets.length; i++) {
                if (root.presets[i].name === "Last known good") {
                    root.applyPreset(root.presets[i])
                    applied = true
                    break
                }
            }
            if (!applied) {
                for (var j = 0; j < root.presets.length; j++) {
                    if (root.presets[j].is_default) { root.applyPreset(root.presets[j]); break }
                }
            }
            // Presets may omit jinja; Glimmer and multimodal still need it.
            if (root.settings.jinja === undefined || root.settings.jinja === null || (root.needsJinja(m) && !root.settings.jinja)) {
                root.setSetting("jinja", root.needsJinja(m))
            }
            if (root.isMuseGlimmerChat(m) && !root.settings.chat_template_kwargs) {
                root.setSetting("chat_template_kwargs", "{\"reasoning_strength\":\"low\"}")
            }
            if (root.canPreserveReasoning(m) && (root.settings.reasoning_preserve === undefined
                    || root.settings.reasoning_preserve === null)) {
                root.setSetting("reasoning_preserve", true)
            }
            // Embedder defaults when presets omit embedding/pooling.
            if (embedder) {
                if (root.settings.embedding === undefined || root.settings.embedding === null)
                    root.setSetting("embedding", true)
                if (!root.settings.pooling)
                    root.setSetting("pooling", root.defaultPooling(meta))
            }
            // Presets must not leave a stale alias from another model; always
            // prefer this model's library alias when the field is empty.
            if (!root.settings.alias) {
                root.setSetting("alias", m.alias || "")
                if (aliasField) aliasField.text = m.alias || ""
            }
            // Keep runtime combo in sync with preset/pin.
            root.selectedRuntime = root.settings.runtime_id || m.pinned_runtime || ""
            if (root.settings.runtime_id === undefined || root.settings.runtime_id === null) {
                root.setSetting("runtime_id", root.selectedRuntime)
            }
        })
        open()
        scheduleRefresh()
    }

    // Debounced refresh of preview + estimate on any settings change.
    Timer {
        id: refreshTimer
        interval: 350
        onTriggered: root.refreshNow()
    }
    function scheduleRefresh() { refreshTimer.restart() }

    function refreshNow() {
        if (!modelId) return
        api.post("/api/v1/models/" + modelId + "/preview", settings, function(st, data) {
            if (st === 200) root.preview = data
        })
        api.post("/api/v1/models/" + modelId + "/estimate", settings, function(st, data) {
            if (st === 200) root.estimate = data
        })
    }

    function setSetting(key, value) {
        settings[key] = value
        settingsChanged()
        scheduleRefresh()
    }

    function reloadDraftCandidates() {
        if (!root.modelId) return
        var id = root.modelId
        api.get("/api/v1/models/" + id + "/draft-candidates", function(st, data) {
            if (st !== 200 || root.modelId !== id) return
            root.draftCandidates = (data && data.candidates) || []
            root.draftFiltered = !data || data.filtered !== false
        })
    }

    function draftChoices() {
        var none = { "id": "", "alias": "None (disabled)", "primary_path": "", "quantization": "", "compatible": true }
        var list = [none].concat(root.draftCandidates || [])
        var path = root.settings.draft_model || ""
        if (path === "") return list
        for (var i = 0; i < list.length; i++) {
            if ((list[i].primary_path || "") === path) return list
        }
        // Preset / manual path not in the filtered set — keep it selectable.
        list.push({
            "id": "_selected",
            "alias": path.split("/").pop() || path,
            "primary_path": path,
            "quantization": "",
            "compatible": false,
            "reason": "not in filtered candidates"
        })
        return list
    }

    function draftIndex() {
        var path = root.settings.draft_model || ""
        var m = root.draftChoices()
        for (var i = 0; i < m.length; i++) {
            if ((m[i].primary_path || "") === path) return i
        }
        return 0
    }

    function applyPreset(p) {
        try {
            var incoming = JSON.parse(JSON.stringify(p.settings))
            var s = Object.assign({}, root.settings)
            for (var k in incoming) s[k] = incoming[k]
            // Keep the library alias unless the preset explicitly set one.
            if (!s.alias && root.model) s.alias = root.model.alias || ""
            if (s.runtime_id === undefined || s.runtime_id === null)
                s.runtime_id = root.model ? (root.model.pinned_runtime || "") : ""
            root.settings = s
            root.selectedRuntime = s.runtime_id || ""
            if (aliasField) aliasField.text = s.alias || ""
            scheduleRefresh()
        } catch (e) {}
    }

    function runtimeChoices() {
        return [{ "id": "", "build": "Auto (preferred)", "backend": "", "architecture": "" }]
            .concat(root.runtimes || [])
    }

    function runtimeIndex() {
        var id = root.settings.runtime_id || root.selectedRuntime || ""
        var m = root.runtimeChoices()
        for (var i = 0; i < m.length; i++) {
            if ((m[i].id || "") === id) return i
        }
        return 0
    }

    function maxCtx() {
        var mc = root.model ? root.model.context_length : 0
        if (mc <= 0) mc = 131072
        return Math.min(mc, 1048576)
    }

    function maxLayers() {
        var md = root.model ? root.model.metadata : null
        var n = md && md.block_count ? md.block_count : 0
        return n > 0 ? n : 99
    }

    contentItem: ColumnLayout {
        spacing: 0

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: AppTheme.pad
            clip: true
            ScrollBar.vertical.policy: ScrollBar.AsNeeded

            ColumnLayout {
                width: root.width - AppTheme.pad * 2 - 60
                spacing: AppTheme.gap

                // Corrupt-file warning
                Label {
                    Layout.fillWidth: true
                    visible: root.model && root.model.metadata && root.model.metadata.tensor_errors
                        && root.model.metadata.tensor_errors.length > 0
                    text: "⚠ This model file failed integrity validation: "
                        + (root.model ? (root.model.metadata.tensor_errors || []).join("; ") : "")
                        + "\nLoading will very likely fail. Re-download or pick another quantization."
                    color: AppTheme.danger
                    wrapMode: Text.WordWrap
                }

                Label {
                    Layout.fillWidth: true
                    visible: root.isEmbedding
                    text: root.isReranker
                          ? "This is a reranker model. It serves ranking via embeddings mode (pooling=rank), not Chat."
                          : "This is an embedding model. It serves /v1/embeddings, not Chat."
                    color: AppTheme.info
                    wrapMode: Text.WordWrap
                    padding: 8
                    background: Rectangle {
                        color: AppTheme.surface
                        border.color: AppTheme.border
                        radius: AppTheme.radius
                    }
                }

                // Memory estimate
                Card {
                    Layout.fillWidth: true
                    implicitHeight: estCol.implicitHeight + 20
                    ColumnLayout {
                        id: estCol
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 4
                        RowLayout {
                            Label {
                                text: "Estimated memory"
                                color: AppTheme.textDim
                                font.pixelSize: AppTheme.fontSmall
                            }
                            Item { Layout.fillWidth: true }
                            Label {
                                text: root.estimate
                                    ? AppTheme.bytes(root.estimate.total_bytes) + " total"
                                    : "…"
                                color: root.estimate
                                    ? (root.estimate.fits ? AppTheme.success : AppTheme.danger)
                                    : AppTheme.textDim
                                font.weight: Font.DemiBold
                            }
                        }

                        // GPU VRAM row
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2
                            visible: root.estimate && (root.estimate.gpu_budget_bytes > 0
                                     || root.estimate.gpu_bytes > 0
                                     || root.settings.gpu_offload !== "none")
                            RowLayout {
                                Label {
                                    text: "GPU VRAM"
                                    color: AppTheme.textDim
                                    font.pixelSize: AppTheme.fontSmall
                                }
                                Item { Layout.fillWidth: true }
                                Label {
                                    text: {
                                        if (!root.estimate) return ""
                                        var used = root.estimate.gpu_bytes || 0
                                        var bud = root.estimate.gpu_budget_bytes || 0
                                        if (bud > 0)
                                            return AppTheme.bytes(used) + " / " + AppTheme.bytes(bud)
                                        return AppTheme.bytes(used)
                                    }
                                    color: root.estimate
                                        ? ((root.estimate.fits_gpu !== false) ? AppTheme.text : AppTheme.danger)
                                        : AppTheme.textDim
                                    font.pixelSize: AppTheme.fontSmall
                                    font.weight: Font.DemiBold
                                }
                            }
                            AppProgressBar {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 8
                                from: 0; to: 1
                                value: {
                                    if (!root.estimate) return 0
                                    var bud = root.estimate.gpu_budget_bytes || 0
                                    if (bud <= 0) return root.estimate.gpu_bytes > 0 ? 1 : 0
                                    return Math.min(1, (root.estimate.gpu_bytes || 0) / bud)
                                }
                            }
                        }

                        // System RAM row
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2
                            visible: !!root.estimate
                            RowLayout {
                                Label {
                                    text: root.estimate && root.estimate.budget_kind === "unified RAM"
                                          ? "System RAM (unified)"
                                          : "System RAM"
                                    color: AppTheme.textDim
                                    font.pixelSize: AppTheme.fontSmall
                                }
                                Item { Layout.fillWidth: true }
                                Label {
                                    text: {
                                        if (!root.estimate) return ""
                                        var used = root.estimate.cpu_bytes || 0
                                        // Unified: GPU+CPU compete for RAM.
                                        if (root.estimate.budget_kind === "unified RAM")
                                            used = (root.estimate.gpu_bytes || 0) + (root.estimate.cpu_bytes || 0)
                                        var bud = root.estimate.cpu_budget_bytes || 0
                                        if (bud > 0)
                                            return AppTheme.bytes(used) + " / " + AppTheme.bytes(bud)
                                        return AppTheme.bytes(used)
                                    }
                                    color: root.estimate
                                        ? ((root.estimate.fits_cpu !== false) ? AppTheme.text : AppTheme.danger)
                                        : AppTheme.textDim
                                    font.pixelSize: AppTheme.fontSmall
                                    font.weight: Font.DemiBold
                                }
                            }
                            AppProgressBar {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 8
                                from: 0; to: 1
                                value: {
                                    if (!root.estimate) return 0
                                    var used = root.estimate.cpu_bytes || 0
                                    if (root.estimate.budget_kind === "unified RAM")
                                        used = (root.estimate.gpu_bytes || 0) + (root.estimate.cpu_bytes || 0)
                                    var bud = root.estimate.cpu_budget_bytes || 0
                                    if (bud <= 0) return used > 0 ? 1 : 0
                                    return Math.min(1, used / bud)
                                }
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            text: {
                                if (!root.estimate) return ""
                                var e = root.estimate
                                var parts = []
                                parts.push("weights " + AppTheme.bytes(e.weights_bytes || 0))
                                if ((e.draft_weights_bytes || 0) > 0)
                                    parts.push("draft " + AppTheme.bytes(e.draft_weights_bytes))
                                if ((e.projector_bytes || 0) > 0)
                                    parts.push("mmproj " + AppTheme.bytes(e.projector_bytes))
                                parts.push("KV " + AppTheme.bytes(e.kv_cache_bytes || 0))
                                if ((e.recurrent_bytes || 0) > 0)
                                    parts.push("recurrent " + AppTheme.bytes(e.recurrent_bytes))
                                parts.push("compute " + AppTheme.bytes(e.compute_bytes || 0))
                                if ((e.media_bytes || 0) > 0)
                                    parts.push("media " + AppTheme.bytes(e.media_bytes))
                                parts.push("overhead " + AppTheme.bytes(e.overhead_bytes || 0))
                                var line = parts.join(" · ")
                                if (e.offload_fraction > 0 && e.offload_fraction < 1)
                                    line += "  |  offload " + Math.round(e.offload_fraction * 100) + "%"
                                if (e.note && e.note !== "")
                                    line += "  —  " + e.note
                                return line
                            }
                            color: AppTheme.textFaint
                            font.pixelSize: AppTheme.fontSmall
                            wrapMode: Text.WordWrap
                        }
                        Label {
                            visible: root.estimate && !root.estimate.fits
                            Layout.fillWidth: true
                            text: {
                                if (!root.estimate) return ""
                                var bits = []
                                if (root.estimate.fits_gpu === false) bits.push("GPU VRAM")
                                if (root.estimate.fits_cpu === false) bits.push("system RAM")
                                if (bits.length === 0) bits.push("available memory")
                                return "Likely exceeds " + bits.join(" and ")
                                    + " — reduce context, KV type, or GPU layers, or Load anyway if you know this machine can handle it."
                            }
                            color: AppTheme.warning
                            font.pixelSize: AppTheme.fontSmall
                            wrapMode: Text.WordWrap
                        }
                    }
                }

                // Presets
                RowLayout {
                    Layout.fillWidth: true
                    visible: root.presets.length > 0
                    Label { text: "Preset:"; color: AppTheme.textDim }
                    AppComboBox {
                        Layout.fillWidth: true
                        model: root.presets
                        textRole: "name"
                        onActivated: function(i) { root.applyPreset(root.presets[i]) }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Runtime"
                    hint: "llama.cpp build used for this model. Auto uses the pinned or preferred runtime."
                    AppComboBox {
                        id: runtimeCombo
                        width: parent.width
                        model: root.runtimeChoices()
                        textRole: "build"
                        currentIndex: root.runtimeIndex()
                        onActivated: function(i) {
                            var id = model[i].id || ""
                            root.selectedRuntime = id
                            root.setSetting("runtime_id", id)
                        }
                        delegate: ItemDelegate {
                            text: modelData.id === "" ? modelData.build
                                : modelData.build + " · " + modelData.backend + " · " + modelData.architecture
                            width: parent ? parent.width : implicitWidth
                        }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Context length"
                    hint: "Tokens of context. KV-cache memory grows linearly with this."
                    argName: "--ctx-size"
                    ColumnLayout {
                        width: parent.width
                        spacing: 4
                        RowLayout {
                            width: parent.width
                            spacing: 12
                            AppSlider {
                                id: ctxSlider
                                Layout.fillWidth: true
                                from: 512
                                to: root.maxCtx()
                                stepSize: 512
                                value: root.settings.context_length
                                onMoved: root.setSetting("context_length", Math.round(value))
                            }
                            AppSpinBox {
                                from: 512
                                to: 1048576
                                stepSize: 512
                                editable: true
                                value: root.settings.context_length
                                onValueModified: root.setSetting("context_length", value)
                            }
                        }
                        Label {
                            text: "model maximum: " + (root.model && root.model.context_length > 0
                                  ? root.model.context_length : "unknown")
                            color: AppTheme.textFaint
                            font.pixelSize: AppTheme.fontSmall
                        }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "GPU offload"
                    hint: "Layers offloaded to the GPU. All is fastest when it fits; fewer layers spill the rest to system RAM."
                    argName: "--n-gpu-layers"
                    ColumnLayout {
                        width: parent.width
                        spacing: 4
                        Row {
                            spacing: 8
                            AppComboBox {
                                id: offloadCombo
                                model: [
                                    { "text": "All layers", "value": "all" },
                                    { "text": "Custom", "value": "custom" },
                                    { "text": "None (CPU only)", "value": "none" }
                                ]
                                textRole: "text"
                                currentIndex: root.settings.gpu_offload === "custom" ? 1
                                    : root.settings.gpu_offload === "none" ? 2 : 0
                                onActivated: function(i) {
                                    root.setSetting("gpu_offload", model[i].value)
                                    if (model[i].value === "custom" && root.settings.gpu_layers <= 0)
                                        root.setSetting("gpu_layers", root.maxLayers())
                                }
                            }
                            Label {
                                visible: root.settings.gpu_offload === "custom"
                                text: root.settings.gpu_layers + " / " + root.maxLayers()
                                    + (root.settings.gpu_layers >= root.maxLayers() ? " (all)" : "")
                                color: AppTheme.textDim
                                anchors.verticalCenter: parent.verticalCenter
                            }
                        }
                        RowLayout {
                            width: parent.width
                            spacing: 12
                            visible: root.settings.gpu_offload === "custom"
                            AppSlider {
                                Layout.fillWidth: true
                                from: 0
                                to: root.maxLayers()
                                stepSize: 1
                                value: root.settings.gpu_layers
                                onMoved: root.setSetting("gpu_layers", Math.round(value))
                            }
                            AppSpinBox {
                                from: 0
                                to: root.maxLayers()
                                editable: true
                                value: root.settings.gpu_layers
                                onValueModified: root.setSetting("gpu_layers", value)
                            }
                        }
                        Label {
                            visible: root.settings.gpu_offload === "custom"
                            text: "model has " + root.maxLayers() + " layers"
                            color: AppTheme.textFaint
                            font.pixelSize: AppTheme.fontSmall
                        }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "CPU threads"
                    hint: "0 = automatic."
                    argName: "--threads"
                    AppSpinBox {
                        from: 0; to: 1024; editable: true
                        value: root.settings.threads
                        onValueModified: root.setSetting("threads", value)
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Flash Attention"
                    hint: "Reduces KV-cache memory. Auto = enabled."
                    argName: "--flash-attn"
                    AppComboBox {
                        model: ["auto", "on", "off"]
                        currentIndex: Math.max(0, model.indexOf(root.settings.flash_attention))
                        onActivated: function(i) { root.setSetting("flash_attention", model[i]) }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Parallel slots"
                    hint: "Simultaneous requests. Each slot gets the full context length; KV memory = context × slots."
                    argName: "--parallel"
                    AppSpinBox {
                        from: 0; to: 64; editable: true
                        value: root.settings.parallel
                        onValueModified: root.setSetting("parallel", value)
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    visible: root.isEmbedding
                    label: "Embedding mode"
                    hint: "Restricts llama-server to embeddings (Developer API /v1/embeddings). Leave on for dedicated embedders."
                    argName: "--embedding"
                    AppSwitch {
                        checked: !!root.settings.embedding
                        onToggled: root.setSetting("embedding", checked)
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    visible: root.isEmbedding
                    label: "Pooling"
                    hint: "Token aggregation for embeddings. Empty uses the model default. Rerankers typically use rank."
                    argName: "--pooling"
                    AppComboBox {
                        model: [
                            { "value": "", "label": "Model default" },
                            { "value": "none", "label": "none" },
                            { "value": "mean", "label": "mean" },
                            { "value": "cls", "label": "cls" },
                            { "value": "last", "label": "last" },
                            { "value": "rank", "label": "rank" }
                        ]
                        textRole: "label"
                        currentIndex: {
                            var v = root.settings.pooling || ""
                            var m = model
                            for (var i = 0; i < m.length; i++)
                                if (m[i].value === v) return i
                            return 0
                        }
                        onActivated: function(i) {
                            root.setSetting("pooling", model[i].value)
                        }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    visible: root.hasMTP && !root.isEmbedding
                    label: "Built-in MTP"
                    hint: "Enabled by default for this GGUF (NextN / Multi-Token Prediction heads). Uses --spec-type draft-mtp without a separate draft file. Turn off here if you prefer single-token decode."
                    argName: "--spec-type"
                    AppSwitch {
                        checked: root.settings.spec_type === "draft-mtp" && !root.settings.draft_model
                        onToggled: {
                            if (checked) {
                                root.setSetting("draft_model", "")
                                root.setSetting("spec_type", "draft-mtp")
                                if (!root.settings.draft_max)
                                    root.setSetting("draft_max", 2)
                            } else if (root.settings.spec_type === "draft-mtp") {
                                root.setSetting("spec_type", "")
                            }
                        }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    visible: !root.isEmbedding
                    label: "Speculative draft"
                    hint: root.draftFiltered
                          ? "Only detected speculative drafts (mtp- / gemma4-assistant / eagle3- / dflash- / dspark-). Turn off Settings → Filter draft model picker to choose any GGUF."
                          : "All library models (filter off). Prefer official sidecars / gemma4-assistant when available."
                    argName: "--model-draft"
                    AppComboBox {
                        id: draftCombo
                        width: parent.width
                        model: root.draftChoices()
                        textRole: "alias"
                        currentIndex: root.draftIndex()
                        onActivated: function(i) {
                            var item = model[i]
                            var path = item.primary_path || ""
                            root.setSetting("draft_model", path)
                            if (path === "") {
                                if (root.settings.spec_type !== "draft-mtp" || !root.hasMTP)
                                    root.setSetting("spec_type", "")
                                return
                            }
                            // Prefer library metadata / sidecar-inferred type (mtp-, eagle3-, …).
                            var st = item.spec_type || (item.metadata && item.metadata.spec_type) || ""
                            if (!st && item.metadata && item.metadata.has_mtp) st = "draft-mtp"
                            if (!st && (item.architecture === "gemma4-assistant" || item.architecture === "gemma4_assistant")) st = "draft-mtp"
                            if (!st && item.architecture === "eagle3") st = "draft-eagle3"
                            if (!st && (item.architecture === "dflash" || item.architecture === "dflash-draft"
                                    || item.architecture === "muse-glimmer-assistant"
                                    || item.architecture === "muse_glimmer_assistant"
                                    || item.architecture === "museglimmer-assistant")) st = "draft-dflash"
                            if (!st && item.architecture === "dspark") st = "draft-dspark"
                            if (!st) {
                                var base = (path.split("/").pop() || "").toLowerCase()
                                if (base.indexOf("mtp-") === 0) st = "draft-mtp"
                                else if (base.indexOf("eagle3-") === 0) st = "draft-eagle3"
                                else if (base.indexOf("dflash-") === 0) st = "draft-dflash"
                                else if (base.indexOf("dspark-") === 0) st = "draft-dspark"
                                else if (base.indexOf("assistant") >= 0 && base.indexOf("glimmer") >= 0) st = "draft-dflash"
                                else if (base.indexOf("assistant") >= 0 && (base.indexOf("gemma-4") >= 0 || base.indexOf("gemma4") >= 0)) st = "draft-mtp"
                                else st = "draft-simple"
                            }
                            root.setSetting("spec_type", st)
                        }
                        delegate: ItemDelegate {
                            width: parent ? parent.width : implicitWidth
                            text: {
                                if (modelData.id === "") return modelData.alias
                                var t = modelData.alias
                                if (modelData.quantization) t += " · " + modelData.quantization
                                var st = modelData.spec_type || (modelData.metadata && modelData.metadata.spec_type) || ""
                                if (st) t += " · " + st
                                if (modelData.compatible === false && modelData.reason)
                                    t += "  (" + modelData.reason + ")"
                                return t
                            }
                        }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    visible: root.speculativeEnabled && !root.isEmbedding
                    label: "Draft tokens (max)"
                    hint: "Max tokens drafted per step. 0 = runtime default (typically 3–16)."
                    argName: "--spec-draft-n-max"
                    AppSpinBox {
                        from: 0; to: 64; editable: true
                        value: root.settings.draft_max || 0
                        onValueModified: root.setSetting("draft_max", value)
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    visible: root.speculativeEnabled && !root.isEmbedding
                    label: "Spec type"
                    hint: "llama.cpp --spec-type. Auto-selected from mtp-/eagle3-/dflash-/dspark- sidecars and draft architecture."
                    argName: "--spec-type"
                    AppComboBox {
                        model: [
                            { "text": "draft-simple", "value": "draft-simple" },
                            { "text": "draft-eagle3", "value": "draft-eagle3" },
                            { "text": "draft-dflash", "value": "draft-dflash" },
                            { "text": "draft-dspark", "value": "draft-dspark" },
                            { "text": "draft-mtp", "value": "draft-mtp" },
                            { "text": "ngram-mod", "value": "ngram-mod" },
                            { "text": "ngram-simple", "value": "ngram-simple" },
                            { "text": "ngram-cache", "value": "ngram-cache" }
                        ]
                        textRole: "text"
                        currentIndex: {
                            var v = root.settings.spec_type || "draft-simple"
                            for (var i = 0; i < model.length; i++)
                                if (model[i].value === v) return i
                            return 0
                        }
                        onActivated: function(i) { root.setSetting("spec_type", model[i].value) }
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    label: "Remember on success"
                    hint: "Save as this model's default after a successful load. Failed loads keep the previous default."
                    AppSwitch {
                        checked: root.settings.save_on_success
                        onToggled: root.setSetting("save_on_success", checked)
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    visible: !root.isEmbedding
                    label: "Multimodal projector"
                    hint: !root.hasProjector
                          ? "No projector paired with this model."
                          : (root.settings.no_mmproj
                             ? "Paired projector will not be loaded."
                             : root.model.projector_path)
                    argName: "--mmproj"
                    supported: root.hasProjector && !root.settings.no_mmproj
                    Label {
                        text: !root.hasProjector ? "None"
                              : (root.settings.no_mmproj ? "Skipped" : "Paired automatically")
                        color: AppTheme.textDim
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    visible: !root.isEmbedding
                    label: "Jinja chat template"
                    hint: "Required by Muse Glimmer and many multimodal models. Enabled by default for those architectures, including text-only Glimmer quants."
                    argName: "--jinja"
                    AppSwitch {
                        checked: !!root.settings.jinja
                        onToggled: root.setSetting("jinja", checked)
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    visible: !root.isEmbedding && root.canPreserveReasoning(root.model)
                    label: "Preserve reasoning"
                    hint: "Keep earlier turns' thinking in context so the model can continue from it. Uses extra tokens; disable to save context. Enabled by default when the chat template supports it."
                    argName: "--reasoning-preserve"
                    AppSwitch {
                        checked: !!root.settings.reasoning_preserve
                        onToggled: root.setSetting("reasoning_preserve", checked)
                    }
                }

                FormField {
                    Layout.fillWidth: true
                    visible: root.hasProjector && !root.isEmbedding
                    label: "Keep projector on CPU"
                    hint: "Disable GPU offload for the multimodal projector (saves VRAM)."
                    argName: "--no-mmproj-offload"
                    supported: !root.settings.no_mmproj
                    AppSwitch {
                        enabled: !root.settings.no_mmproj
                        checked: !!root.settings.no_mmproj_offload
                        onToggled: root.setSetting("no_mmproj_offload", checked)
                    }
                }

                ToolButton {
                    id: advancedToggle
                    checkable: true
                    text: (checked ? "▾" : "▸") + " Advanced"
                    font.weight: Font.DemiBold
                }
                ColumnLayout {
                    id: advancedSection
                    Layout.fillWidth: true
                    visible: advancedToggle.checked
                    spacing: AppTheme.gap
                    opacity: 1
                    onVisibleChanged: {
                        if (visible) {
                            opacity = 0
                            Qt.callLater(function() { advancedSection.opacity = 1 })
                        }
                    }
                    Behavior on opacity { NumberAnimation { duration: AppTheme.motion; easing.type: Easing.OutCubic } }

                    FormField {
                        Layout.fillWidth: true
                        visible: !root.isEmbedding
                        label: "Skip multimodal projector"
                        hint: root.hasProjector
                              ? "Load text-only: omit the paired vision/audio projector (saves memory)."
                              : "Only available when a multimodal projector is paired with this model."
                        argName: "--no-mmproj"
                        supported: root.hasProjector
                        AppSwitch {
                            enabled: root.hasProjector
                            checked: !!root.settings.no_mmproj
                            onToggled: root.setSetting("no_mmproj", checked)
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Batch size"; argName: "--batch-size"
                        hint: "Prompt processing batch. 0 = runtime default."
                        AppSpinBox { from: 0; to: 65536; editable: true; value: root.settings.batch_size
                            onValueModified: root.setSetting("batch_size", value) }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Micro-batch size"; argName: "--ubatch-size"
                        hint: "Physical batch. 0 = runtime default."
                        AppSpinBox { from: 0; to: 65536; editable: true; value: root.settings.ubatch_size
                            onValueModified: root.setSetting("ubatch_size", value) }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "KV cache type K"; argName: "--cache-type-k"
                        hint: "Quantizing the K cache saves memory."
                        AppComboBox {
                            width: parent.width
                            model: ["", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
                            currentIndex: Math.max(0, model.indexOf(root.settings.cache_type_k))
                            onActivated: function(i) { root.setSetting("cache_type_k", model[i]) }
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "KV cache type V"; argName: "--cache-type-v"
                        hint: "Quantizing the V cache saves memory."
                        AppComboBox {
                            width: parent.width
                            model: ["", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
                            currentIndex: Math.max(0, model.indexOf(root.settings.cache_type_v))
                            onActivated: function(i) { root.setSetting("cache_type_v", model[i]) }
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Memory mapping"; argName: "--no-mmap"
                        hint: "Disable to load the whole model into RAM up front."
                        AppSwitch { checked: !root.settings.no_mmap
                            onToggled: root.setSetting("no_mmap", !checked) }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Memory locking"; argName: "--mlock"
                        hint: "Keep the model resident (prevents swapping)."
                        AppSwitch { checked: root.settings.mlock
                            onToggled: root.setSetting("mlock", checked) }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Model alias"; argName: "--alias"
                        hint: "API name for this session only. Does not change the name on My library."
                        AppTextField {
                            id: aliasField
                            width: 320
                            text: root.settings.alias
                            onEditingFinished: root.setSetting("alias", text)
                        }
                    }
                }

                ToolButton {
                    id: expertToggle
                    checkable: true
                    text: (checked ? "▾" : "▸") + " Expert"
                    font.weight: Font.DemiBold
                }
                ColumnLayout {
                    id: expertSection
                    Layout.fillWidth: true
                    visible: expertToggle.checked
                    spacing: AppTheme.gap
                    opacity: 1
                    onVisibleChanged: {
                        if (visible) {
                            opacity = 0
                            Qt.callLater(function() { expertSection.opacity = 1 })
                        }
                    }
                    Behavior on opacity { NumberAnimation { duration: AppTheme.motion; easing.type: Easing.OutCubic } }

                    Label {
                        Layout.fillWidth: true
                        text: "Performance and multi-GPU knobs. Empty / default leaves llama.cpp defaults. Unsupported flags are skipped in the command preview."
                        color: AppTheme.textFaint
                        font.pixelSize: AppTheme.fontSmall
                        wrapMode: Text.WordWrap
                    }

                    FormField {
                        Layout.fillWidth: true
                        label: "Batch threads"
                        argName: "--threads-batch"
                        hint: "CPU threads for prompt / batch processing. 0 = same as --threads. Higher can speed prefills."
                        AppSpinBox {
                            from: 0; to: 256; editable: true
                            value: root.settings.threads_batch || 0
                            onValueModified: root.setSetting("threads_batch", value)
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Continuous batching"
                        argName: "--cont-batching"
                        hint: "Dynamic batching across slots. Improves multi-request throughput; leave default for single-chat."
                        AppComboBox {
                            model: [
                                { "text": "default", "value": "" },
                                { "text": "on", "value": "on" },
                                { "text": "off", "value": "off" }
                            ]
                            textRole: "text"
                            currentIndex: {
                                var v = root.settings.cont_batching
                                if (v === true) return 1
                                if (v === false) return 2
                                return 0
                            }
                            onActivated: function(i) {
                                var v = model[i].value
                                if (v === "") root.setSetting("cont_batching", null)
                                else root.setSetting("cont_batching", v === "on")
                            }
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Cache reuse"
                        argName: "--cache-reuse"
                        hint: "Min chunk size (tokens) to reuse from the prompt cache via KV shifting. 0 = off / runtime default."
                        AppSpinBox {
                            from: 0; to: 65536; editable: true
                            value: root.settings.cache_reuse || 0
                            onValueModified: root.setSetting("cache_reuse", value)
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Process priority"
                        argName: "--prio"
                        hint: "OS scheduling priority for the llama-server process."
                        AppComboBox {
                            model: [
                                { "text": "default", "value": -2 },
                                { "text": "low", "value": -1 },
                                { "text": "normal", "value": 0 },
                                { "text": "medium", "value": 1 },
                                { "text": "high", "value": 2 },
                                { "text": "realtime", "value": 3 }
                            ]
                            textRole: "text"
                            currentIndex: {
                                var v = root.settings.prio
                                if (v === undefined || v === null) return 0
                                for (var i = 0; i < model.length; i++)
                                    if (model[i].value === v) return i
                                return 0
                            }
                            onActivated: function(i) { root.setSetting("prio", model[i].value) }
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Work poll"
                        argName: "--poll"
                        hint: "CPU polling while waiting for work (0 = sleep, 50 = default, 100 = aggressive). Can cut latency at the cost of idle CPU."
                        AppSpinBox {
                            from: -1; to: 100; editable: true
                            value: (root.settings.poll === undefined || root.settings.poll === null)
                                   ? -1 : root.settings.poll
                            textFromValue: function(v) { return v < 0 ? "default" : String(v) }
                            valueFromText: function(t) {
                                t = String(t).trim().toLowerCase()
                                if (t === "" || t === "default") return -1
                                var n = parseInt(t, 10)
                                return isNaN(n) ? -1 : n
                            }
                            onValueModified: root.setSetting("poll", value)
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "NUMA"
                        argName: "--numa"
                        hint: "NUMA placement on multi-socket CPUs. Most desktops leave this empty."
                        AppComboBox {
                            model: ["", "distribute", "isolate", "numactl"]
                            currentIndex: Math.max(0, model.indexOf(root.settings.numa || ""))
                            onActivated: function(i) { root.setSetting("numa", model[i]) }
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Fit to device memory"
                        argName: "--fit"
                        hint: "Auto-adjust unset args (context / layers) so the load fits VRAM."
                        AppComboBox {
                            model: [
                                { "text": "default", "value": "" },
                                { "text": "on", "value": "on" },
                                { "text": "off", "value": "off" }
                            ]
                            textRole: "text"
                            currentIndex: {
                                var v = root.settings.fit || ""
                                for (var i = 0; i < model.length; i++)
                                    if (model[i].value === v) return i
                                return 0
                            }
                            onActivated: function(i) { root.setSetting("fit", model[i].value) }
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "KV cache offload"
                        argName: "--kv-offload"
                        hint: "Keep KV cache on GPU (on) or force it to system RAM (off). Off frees VRAM but is slower."
                        AppComboBox {
                            model: [
                                { "text": "default", "value": "" },
                                { "text": "on", "value": "on" },
                                { "text": "off", "value": "off" }
                            ]
                            textRole: "text"
                            currentIndex: {
                                var v = root.settings.kv_offload || ""
                                for (var i = 0; i < model.length; i++)
                                    if (model[i].value === v) return i
                                return 0
                            }
                            onActivated: function(i) { root.setSetting("kv_offload", model[i].value) }
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Op offload"
                        argName: "--op-offload"
                        hint: "Offload host tensor ops to the GPU device (default on). Rarely disable unless debugging."
                        AppComboBox {
                            model: [
                                { "text": "default", "value": "" },
                                { "text": "on", "value": "on" },
                                { "text": "off", "value": "off" }
                            ]
                            textRole: "text"
                            currentIndex: {
                                var v = root.settings.op_offload || ""
                                for (var i = 0; i < model.length; i++)
                                    if (model[i].value === v) return i
                                return 0
                            }
                            onActivated: function(i) { root.setSetting("op_offload", model[i].value) }
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Unified KV buffer"
                        argName: "--kv-unified"
                        hint: "Single KV buffer shared across sequences. Helps multi-slot / continuous batching."
                        AppComboBox {
                            model: [
                                { "text": "default", "value": "" },
                                { "text": "on", "value": "on" },
                                { "text": "off", "value": "off" }
                            ]
                            textRole: "text"
                            currentIndex: {
                                var v = root.settings.kv_unified || ""
                                for (var i = 0; i < model.length; i++)
                                    if (model[i].value === v) return i
                                return 0
                            }
                            onActivated: function(i) { root.setSetting("kv_unified", model[i].value) }
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Full SWA cache"
                        argName: "--swa-full"
                        hint: "Use a full-size sliding-window attention cache. More memory; can help some SWA models."
                        AppSwitch {
                            checked: !!root.settings.swa_full
                            onToggled: root.setSetting("swa_full", checked)
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Keep MoE on CPU"
                        argName: "--cpu-moe"
                        hint: "Leave all Mixture-of-Experts weights in system RAM. Frees VRAM on MoE models."
                        AppSwitch {
                            checked: !!root.settings.cpu_moe
                            onToggled: root.setSetting("cpu_moe", checked)
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "MoE layers on CPU"
                        argName: "--n-cpu-moe"
                        hint: "Keep MoE weights of the first N layers on CPU. 0 = unset (use --cpu-moe for all)."
                        AppSpinBox {
                            from: 0; to: 512; editable: true
                            value: root.settings.n_cpu_moe || 0
                            onValueModified: root.setSetting("n_cpu_moe", value)
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Main GPU"
                        argName: "--main-gpu"
                        hint: "Primary GPU index for single-GPU or row-split intermediate/KV. -1 = unset."
                        AppSpinBox {
                            from: -1; to: 15; editable: true
                            value: (root.settings.main_gpu === undefined || root.settings.main_gpu === null)
                                   ? -1 : root.settings.main_gpu
                            onValueModified: root.setSetting("main_gpu", value)
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Devices"
                        argName: "--device"
                        hint: "Comma-separated device list for offload (e.g. Vulkan0,Vulkan1). Empty = all; none = no offload."
                        AppTextField {
                            width: parent.width
                            text: root.settings.device || ""
                            placeholderText: "Vulkan0,Vulkan1"
                            onEditingFinished: root.setSetting("device", text.trim())
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Split mode"
                        argName: "--split-mode"
                        hint: "How to split across GPUs: layer (pipelined), row, or tensor."
                        AppComboBox {
                            model: ["", "none", "layer", "row", "tensor"]
                            currentIndex: Math.max(0, model.indexOf(root.settings.split_mode || ""))
                            onActivated: function(i) { root.setSetting("split_mode", model[i]) }
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Tensor split"
                        argName: "--tensor-split"
                        hint: "Per-GPU fractions, e.g. 3,1 for a 75/25 split."
                        AppTextField {
                            width: parent.width
                            text: root.settings.tensor_split || ""
                            placeholderText: "3,1"
                            onEditingFinished: root.setSetting("tensor_split", text.trim())
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        visible: !root.isEmbedding
                        label: "Draft tokens (min)"
                        argName: "--spec-draft-n-min"
                        hint: "Minimum draft tokens per speculative step. 0 = runtime default."
                        AppSpinBox {
                            from: 0; to: 64; editable: true
                            value: root.settings.draft_min || 0
                            onValueModified: root.setSetting("draft_min", value)
                        }
                    }
                    FormField {
                        Layout.fillWidth: true
                        label: "Skip warmup"
                        argName: "--no-warmup"
                        hint: "Skip the empty warmup run at start. Faster load; first real request may be slower."
                        AppSwitch {
                            checked: !!root.settings.no_warmup
                            onToggled: root.setSetting("no_warmup", checked)
                        }
                    }

                    FormField {
                        Layout.fillWidth: true
                        label: "Raw llama.cpp arguments"
                        hint: "Space-separated flags supported by the selected runtime. Unsafe input is rejected."
                        AppTextArea {
                            width: parent.width
                            height: 72
                            text: root.settings.raw_args
                            placeholderText: "--override-kv llama.attention.head_count=int:8"
                            onEditingFinished: root.setSetting("raw_args", text)
                        }
                    }
                    Label {
                        text: "Environment overrides are restricted to an allowlist (GGML_*, CUDA_VISIBLE_DEVICES, …)."
                        color: AppTheme.textFaint
                        font.pixelSize: AppTheme.fontSmall
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                }

                AppGroupBox {
                    Layout.fillWidth: true
                    title: "Generated command"
                    ColumnLayout {
                        width: parent.width
                        Repeater {
                            model: root.preview ? root.preview.resolutions || [] : []
                            Label {
                                text: modelData.setting + ": " + modelData.auto + " → " + modelData.resolved
                                color: AppTheme.textDim
                                font.pixelSize: AppTheme.fontSmall
                            }
                        }
                        Repeater {
                            model: root.preview ? root.preview.warnings || [] : []
                            Label {
                                text: "⚠ " + modelData
                                color: AppTheme.warning
                                font.pixelSize: AppTheme.fontSmall
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                        }
                        ScrollView {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 56
                            clip: true
                            TextEdit {
                                width: parent.width
                                readOnly: true
                                text: root.preview ? "llama-server " + root.preview.command : "…"
                                color: AppTheme.text
                                font.family: "monospace"
                                font.pixelSize: AppTheme.fontSmall
                                wrapMode: Text.WrapAnywhere
                            }
                        }
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: footerRow.implicitHeight + 16
            color: AppTheme.bgAlt
            ColumnLayout {
                id: footerRow
                anchors.fill: parent
                anchors.margins: 8
                spacing: 6
                Label {
                    visible: root.loadError !== ""
                    Layout.fillWidth: true
                    text: root.loadError
                    color: AppTheme.danger
                    wrapMode: Text.WordWrap
                }
                RowLayout {
                    Layout.fillWidth: true
                    AppButton {
                        text: "Save preset…"
                        flat: true
                        onClicked: savePresetDialog.open()
                    }
                    Item { Layout.fillWidth: true }
                    AppButton { text: "Cancel"; onClicked: root.close() }
                    AppButton {
                        readonly property bool overBudget: !!(root.estimate && !root.estimate.fits)
                        text: overBudget ? "Load anyway" : "Load model"
                        primary: !overBudget
                        danger: overBudget
                        ToolTip.visible: hovered && overBudget
                        ToolTip.text: "Estimate exceeds detected memory. Load anyway if you know this machine can handle it."
                        onClicked: {
                            root.loadError = ""
                            root.api.post("/api/v1/models/" + root.modelId + "/load", root.settings,
                                function(st, data) {
                                    if (st === 202) {
                                        root.loaded()
                                        root.close()
                                    } else {
                                        root.loadError = (data && (data.detail || data.error)) || ("HTTP " + st)
                                    }
                                })
                        }
                    }
                }
            }
        }
    }

    Dialog {
        id: savePresetDialog
        title: "Save preset"
        modal: true
        anchors.centerIn: root.parent
        standardButtons: Dialog.Save | Dialog.Cancel
        transformOrigin: Item.Center
        enter: DialogEnter {}
        exit: DialogExit {}
        Overlay.modal: Rectangle { color: AppTheme.overlay }
        background: Rectangle {
            color: AppTheme.bg
            border.color: AppTheme.border
            radius: AppTheme.radius
        }
        Column {
            spacing: 8
            AppTextField { id: presetNameField; placeholderText: "Preset name"; width: 280 }
            AppCheckBox { id: presetDefault; text: "Make default" }
        }
        onAccepted: {
            root.api.post("/api/v1/models/" + root.modelId + "/presets",
                { "name": presetNameField.text, "settings": root.settings, "is_default": presetDefault.checked },
                function(st, data) {
                    if (st === 200) {
                        root.api.get("/api/v1/models/" + root.modelId + "/presets", function(s2, d2) {
                            if (s2 === 200) root.presets = (d2 && d2.presets) || []
                        })
                    }
                })
        }
    }
}
