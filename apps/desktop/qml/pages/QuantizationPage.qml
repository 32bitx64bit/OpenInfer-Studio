import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtQuick.Window
import ".."
import "../components"

Item {
    id: page
    property var api
    property var events

    property var models: []
    property var runtimes: []
    property var jobs: []
    readonly property bool hasFinishedJobs: {
        for (var i = 0; i < jobs.length; i++) {
            var s = jobs[i] && jobs[i].state
            if (s === "complete" || s === "failed" || s === "canceled")
                return true
        }
        return false
    }
    property var imatrices: []
    property var companions: []
    property var types: []
    property var tools: ({})
    property var preview: null
    property var fit: null
    property string sourceId: ""
    property string pendingSourceId: ""
    property string ftype: "Q4_K_M"
    property string quality: "balanced" // best|balanced|smaller|custom|adaptive
    property string runtimeId: ""
    property string outputName: ""
    property int threads: 0
    property bool allowRequantize: false
    property bool ackRequantize: false
    property bool ackExperimental: false
    property bool ackRAM: false
    property bool leaveOutput: false
    property bool pure: false
    property bool keepSplit: false
    property bool generateIMatrix: false
    property string imatrixId: ""
    property string calibrationPreset: "standard"
    property string calibrationPath: ""
    property int gpuLayers: 0
    property bool parseSpecial: false
    property bool processOutput: false
    property bool keepIMatrix: true
    property bool copyProjector: true
    property bool quantizeProjector: false
    property string projectorFType: "Q8_0"
    property bool quantizeDraft: false
    property string draftModelId: ""
    property string draftFType: "Q4_K_S"
    property string outputTensorType: ""
    property string tokenEmbeddingType: ""
    property string tensorTypesText: ""
    property string adaptivePreset: ""
    property bool unloadFirst: false
    property var loadedModels: []
    property string errorText: ""
    property string statusText: ""
    property bool hideQ8AndBelow: true
    property bool jobSubmitting: false

    function prefillModel(id) {
        page.pendingSourceId = id || ""
        page.sourceId = id || ""
        schedulePreview()
        reloadIMatrices()
    }

    function reload() {
        api.get("/api/v1/models", function(st, data) {
            if (st === 200) page.models = (data && data.models) || []
            if (page.pendingSourceId !== "") {
                page.sourceId = page.pendingSourceId
                page.pendingSourceId = ""
            } else {
                page.pickDefaultSource()
            }
            schedulePreview()
            reloadIMatrices()
        })
        api.get("/api/v1/runtimes", function(st, data) {
            if (st !== 200) return
            page.runtimes = (data && data.runtimes) || []
            if (page.runtimeId === "") {
                for (var i = 0; i < page.runtimes.length; i++) {
                    if (page.runtimes[i].preferred) {
                        page.runtimeId = page.runtimes[i].id
                        break
                    }
                }
                if (page.runtimeId === "" && page.runtimes.length > 0)
                    page.runtimeId = page.runtimes[0].id
            }
        })
        reloadJobs()
    }

    function reloadJobs() {
        api.get("/api/v1/quantize/jobs", function(st, data) {
            if (st === 200) page.jobs = (data && data.jobs) || []
        })
    }

    function clearJobHistory() {
        api.post("/api/v1/quantize/jobs/clear-history", {}, function() { page.reloadJobs() })
    }

    function reloadIMatrices() {
        if (page.sourceId === "") {
            page.imatrices = []
            return
        }
        api.get("/api/v1/quantize/imatrices?model_id=" + encodeURIComponent(page.sourceId), function(st, data) {
            if (st === 200) page.imatrices = (data && data.imatrices) || []
        })
    }

    function sourceModel() {
        for (var i = 0; i < page.models.length; i++)
            if (page.models[i].id === page.sourceId) return page.models[i]
        return null
    }

    function sourceChoiceLabel(m) {
        if (!m) return ""
        var label = (m.alias || m.id) + " · " + (m.quantization || "unknown")
        var meta = m.metadata || {}
        if (meta.speculative_draft) {
            var st = String(meta.spec_type || "")
            if (st.indexOf("draft-") === 0) st = st.substring(6)
            label += " · " + (st || "assistant")
        }
        return label
    }

    function sourceChoices() {
        var out = []
        var seen = {}
        for (var i = 0; i < page.models.length; i++) {
            var m = page.models[i]
            if (page.hideQ8AndBelow && !AppTheme.isFullPrecisionQuant(m.quantization))
                continue
            seen[m.id] = true
            out.push({ "id": m.id, "label": page.sourceChoiceLabel(m) })
        }
        if (page.sourceId !== "" && !seen[page.sourceId]) {
            var cur = page.sourceModel()
            if (cur)
                out.unshift({ "id": cur.id, "label": page.sourceChoiceLabel(cur) })
        }
        return out
    }

    function pickDefaultSource() {
        var choices = page.sourceChoices()
        if (page.sourceId !== "") {
            for (var i = 0; i < choices.length; i++)
                if (choices[i].id === page.sourceId)
                    return
            if (page.models.length === 0)
                return
        }
        page.sourceId = choices.length > 0 ? choices[0].id : ""
    }

    function hasFlag(name) {
        var q = page.tools && page.tools.quantize ? page.tools.quantize : null
        var flags = (q && q.flags) || []
        return flags.indexOf(name) >= 0
    }

    function hasIMatrixFlag(name) {
        var q = page.tools && page.tools.imatrix ? page.tools.imatrix : null
        var flags = (q && q.flags) || []
        return flags.indexOf(name) >= 0
    }

    function quantizePresent() {
        return !!(page.tools && page.tools.quantize && page.tools.quantize.present)
    }

    function imatrixPresent() {
        return !!(page.tools && page.tools.imatrix && page.tools.imatrix.present)
    }

    function pickType(names) {
        for (var i = 0; i < names.length; i++) {
            for (var j = 0; j < page.types.length; j++) {
                if (page.types[j].name === names[i] && !page.types[j].alias_of)
                    return names[i]
            }
        }
        return names[0]
    }

    function isIQ(name) {
        var u = String(name || "").toUpperCase()
        return u.indexOf("IQ") === 0 || u === "Q2_K" || u === "Q2_K_S"
    }

    function ftypeMeta(name) {
        for (var i = 0; i < page.types.length; i++)
            if (page.types[i].name === name) return page.types[i]
        return null
    }

    function customTypes() {
        var out = []
        for (var i = 0; i < page.types.length; i++) {
            var t = page.types[i]
            if (t.alias_of) continue
            if (!advancedToggle.checked && (t.band === "repack" || t.experimental)) continue
            out.push(t.name)
        }
        return out
    }

    function applyQuality(q) {
        page.quality = q
        page.adaptivePreset = ""
        if (q === "best") page.ftype = "Q8_0"
        else if (q === "balanced") page.ftype = "Q4_K_M"
        else if (q === "smaller") {
            page.ftype = page.pickType(["IQ4_XS", "Q3_K_M", "Q4_K_S"])
            page.generateIMatrix = page.isIQ(page.ftype)
            page.calibrationPreset = "standard"
        } else if (q === "adaptive") {
            page.adaptivePreset = "balanced"
            page.ftype = "Q4_K_M"
            page.generateIMatrix = true
        }
        if (page.quality !== "smaller" && page.quality !== "adaptive" && !isIQ(page.ftype))
            page.generateIMatrix = page.generateIMatrix && page.quality === "custom"
        schedulePreview()
    }

    function requestBody() {
        var tensors = []
        String(page.tensorTypesText || "").split(/[\n,]/).forEach(function(s) {
            s = s.trim()
            if (s !== "") tensors.push(s)
        })
        var kind = "quantize"
        if (page.adaptivePreset !== "") kind = "adaptive_quantize"
        return {
            "kind": kind,
            "runtime_id": page.runtimeId,
            "source_model_id": page.sourceId,
            "ftype": page.ftype,
            "output_name": page.outputName,
            "threads": page.threads,
            "allow_requantize": page.allowRequantize,
            "leave_output_tensor": page.leaveOutput,
            "pure": page.pure,
            "keep_split": page.keepSplit,
            "output_tensor_type": page.outputTensorType,
            "token_embedding_type": page.tokenEmbeddingType,
            "tensor_types": tensors,
            "tensor_type_file": "",
            "imatrix_id": page.generateIMatrix ? "" : page.imatrixId,
            "generate_imatrix": page.generateIMatrix,
            "calibration_path": page.calibrationPath,
            "calibration_preset": page.calibrationPreset,
            "chunks": 0,
            "chunk_skip": 0,
            "gpu_layers": page.gpuLayers,
            "parse_special": page.parseSpecial,
            "process_output": page.processOutput,
            "combine_imatrix_ids": [],
            "delete_intermediates": !page.keepIMatrix,
            "keep_imatrix": page.keepIMatrix,
            "quantize_projector": page.quantizeProjector,
            "projector_ftype": page.projectorFType,
            "copy_projector": page.copyProjector && !page.quantizeProjector,
            "draft_model_id": page.draftModelId,
            "quantize_draft": page.quantizeDraft,
            "draft_ftype": page.draftFType,
            "adaptive_preset": page.adaptivePreset,
            "target_bpw": 0,
            "target_bytes": 0,
            "acknowledge_requantize": page.ackRequantize,
            "acknowledge_experimental": page.ackExperimental,
            "unload_first": page.unloadFirst
        }
    }

    function schedulePreview() {
        previewTimer.restart()
    }

    function runPreview() {
        if (page.sourceId === "") {
            page.preview = null
            return
        }
        api.post("/api/v1/quantize/preview", requestBody(), function(st, data) {
            if (st !== 200 || !data) {
                page.errorText = (data && (data.detail || data.error)) || ""
                return
            }
            page.errorText = ""
            page.preview = data.preview || null
            page.companions = data.companions || []
            page.types = data.types || []
            page.tools = data.tools || {}
            page.fit = (data.preview && data.preview.fit) || null
            page.loadedModels = data.loaded_models || []
            if (page.threads <= 0 && data.preview && data.preview.threads_default)
                page.threads = data.preview.threads_default
            if (data.preview && data.preview.recommended_ftype && page.quality === "balanced" && page.ftype === "Q4_K_M") {
                // Keep the user's balanced chip; show recommended separately.
            }
            if (data.split_source && page.hasFlag("--keep-split"))
                page.keepSplit = true
            var drafts = page.companions.filter(function(c) { return c.kind === "draft" })
            if (drafts.length > 0 && page.draftModelId === "")
                page.draftModelId = drafts[0].model_id || ""
        })
    }

    function startJob() {
        if (page.jobSubmitting) return
        page.jobSubmitting = true
        page.errorText = ""
        page.statusText = ""
        api.post("/api/v1/quantize/jobs", requestBody(), function(st, data) {
            page.jobSubmitting = false
            if (st !== 202 && st !== 200) {
                page.errorText = (data && (data.detail || data.error))
                    || (st === 0 ? "Backend unreachable or timed out" : ("HTTP " + st))
                return
            }
            page.statusText = "Job queued"
            page.reloadJobs()
            var win = page.Window.window
            if (win && win.toast) win.toast("Quantization job queued", "success")
            if (win && win.refreshActivityBadge) win.refreshActivityBadge()
        })
    }

    function startDisabledReason() {
        if (page.sourceId === "") return "Pick a source model"
        if (!page.quantizePresent()) return "This runtime has no llama-quantize next to llama-server"
        var p = page.preview
        if (p && p.blockers && p.blockers.length) return p.blockers[0]
        if (p && !p.high_precision_source && (!page.allowRequantize || !page.ackRequantize))
            return "Requantize is blocked until you enable it in Advanced"
        var meta = page.ftypeMeta(page.ftype)
        if (meta && meta.experimental && !page.ackExperimental)
            return "Experimental type — confirm in Advanced"
        var needIM = (p && p.imatrix_required) || page.isIQ(page.ftype)
        if (needIM && !page.generateIMatrix && page.imatrixId === "")
            return "This type needs an importance matrix"
        if (needIM && page.generateIMatrix && !page.imatrixPresent())
            return "This runtime has no llama-imatrix"
        if (p && !p.ram_ok && !page.ackRAM)
            return "Not enough free RAM — confirm in Advanced to continue anyway"
        return ""
    }

    Timer { id: previewTimer; interval: 250; onTriggered: page.runPreview() }

    Connections {
        target: page.events
        function onEventReceived(name, payload) {
            if (name === "quant.progress" || name === "quant.state_changed")
                page.reloadJobs()
            if (name === "library.scanned" || name === "library.model_imported")
                page.reload()
        }
    }

    FileDialog {
        id: calDialog
        title: "Calibration text"
        fileMode: FileDialog.OpenFile
        nameFilters: ["Text files (*.txt)", "All files (*)"]
        onAccepted: {
            var u = String(selectedFile)
            page.calibrationPath = u.replace(/^file:\/\//, "")
        }
    }
    FileDialog {
        id: imDialog
        title: "Import importance matrix"
        fileMode: FileDialog.OpenFile
        nameFilters: ["IMatrix (*.gguf *.dat)", "All files (*)"]
        onAccepted: {
            var u = String(selectedFile).replace(/^file:\/\//, "")
            api.post("/api/v1/quantize/imatrices/import", {
                "path": u, "source_model_id": page.sourceId, "dataset_label": ""
            }, function(st, data) {
                if (st !== 200) page.errorText = (data && (data.detail || data.error)) || "Import failed"
                else {
                    page.imatrixId = data.imatrix ? data.imatrix.id : ""
                    page.generateIMatrix = false
                    page.reloadIMatrices()
                }
            })
        }
    }

    ScrollView {
        anchors.fill: parent
        clip: true
        contentWidth: availableWidth

        ColumnLayout {
            width: page.width - AppTheme.pad * 2
            x: AppTheme.pad
            y: AppTheme.pad
            spacing: AppTheme.gap

            PageHeader {
                title: "Quantization"
                subtitle: "Shrink a GGUF with the selected llama.cpp runtime. Jobs run in the background and land in your library."
            }

            Label {
                visible: page.errorText !== ""
                text: page.errorText
                color: AppTheme.danger
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
            Label {
                visible: page.statusText !== "" && page.errorText === ""
                text: page.statusText
                color: AppTheme.success
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }

            SectionCard {
                title: "Source"
                subtitle: "Prefer F16, BF16, or F32. Requantizing a smaller type loses quality."
                Layout.fillWidth: true
                FormField {
                    Layout.fillWidth: true
                    label: "Model"
                    AppComboBox {
                        Layout.fillWidth: true
                        width: parent.width
                        model: page.sourceChoices()
                        textRole: "label"
                        currentIndex: {
                            var choices = page.sourceChoices()
                            for (var i = 0; i < choices.length; i++)
                                if (choices[i].id === page.sourceId) return i
                            return 0
                        }
                        onActivated: function(i) {
                            var choices = page.sourceChoices()
                            if (i < 0 || i >= choices.length) return
                            page.sourceId = choices[i].id
                            page.reloadIMatrices()
                            page.schedulePreview()
                        }
                    }
                }
                AppCheckBox {
                    text: "Hide Q8 and below"
                    checked: page.hideQ8AndBelow
                    ToolTip.visible: hovered
                    ToolTip.text: "Show only F32, F16, and BF16 sources. Q8, K-quants, IQ, and Unsloth UD files are hidden."
                    onToggled: {
                        page.hideQ8AndBelow = checked
                        if (checked) {
                            var m = page.sourceModel()
                            if (m && !AppTheme.isFullPrecisionQuant(m.quantization))
                                page.sourceId = ""
                        }
                        page.pickDefaultSource()
                        page.reloadIMatrices()
                        page.schedulePreview()
                    }
                }
                Label {
                    visible: page.hideQ8AndBelow && page.models.length > 0 && page.sourceChoices().length === 0
                    text: "No F16, BF16, or F32 models in the library. Download a full-precision GGUF, or turn off Hide Q8 and below."
                    color: AppTheme.warning
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                    font.pixelSize: AppTheme.fontSmall
                }
                Label {
                    visible: !!page.sourceModel()
                    text: {
                        var m = page.sourceModel()
                        if (!m) return ""
                        return (m.quantization || "unknown") + " · " + AppTheme.bytes(m.size_bytes)
                            + (m.architecture ? " · " + m.architecture : "")
                    }
                    color: AppTheme.textDim
                    font.pixelSize: AppTheme.fontSmall
                }
            }

            SectionCard {
                visible: page.companions.length > 0
                title: "Related files"
                Layout.fillWidth: true
                Repeater {
                    model: page.companions
                    delegate: RowLayout {
                        Layout.fillWidth: true
                        Tag { text: modelData.kind; tone: AppTheme.accent }
                        Label {
                            text: modelData.alias || modelData.path
                            color: AppTheme.text
                            elide: Text.ElideMiddle
                            Layout.fillWidth: true
                        }
                        Label {
                            text: AppTheme.bytes(modelData.size_bytes)
                            color: AppTheme.textFaint
                            font.pixelSize: AppTheme.fontSmall
                        }
                        AppCheckBox {
                            visible: modelData.kind === "projector"
                            text: page.quantizeProjector ? "Quantize mmproj" : "Copy mmproj"
                            checked: page.quantizeProjector ? true : page.copyProjector
                            onToggled: {
                                if (checked && advancedToggle.checked && page.quantizeProjector)
                                    return
                                page.copyProjector = checked
                                if (!checked) page.quantizeProjector = false
                            }
                        }
                        AppCheckBox {
                            visible: modelData.kind === "draft"
                            text: "Quantize draft"
                            checked: page.quantizeDraft && page.draftModelId === modelData.model_id
                            onToggled: {
                                page.quantizeDraft = checked
                                if (checked) page.draftModelId = modelData.model_id
                            }
                        }
                    }
                }
            }

            SectionCard {
                title: "Quality"
                Layout.fillWidth: true
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Repeater {
                        model: [
                            { "id": "best", "label": "Best quality (Q8_0)" },
                            { "id": "balanced", "label": "Balanced (Q4_K_M)" },
                            { "id": "smaller", "label": "Smaller (IQ4_XS)" },
                            { "id": "custom", "label": "Custom…" }
                        ]
                        delegate: AppButton {
                            text: modelData.label
                            primary: page.quality === modelData.id
                            onClicked: page.applyQuality(modelData.id)
                        }
                    }
                }
                RowLayout {
                    visible: !!(page.preview && page.preview.recommended_ftype)
                    Layout.fillWidth: true
                    Tag { text: "Recommended"; tone: AppTheme.success }
                    Label {
                        text: {
                            var p = page.preview
                            if (!p || !p.recommended_ftype) return ""
                            return p.recommended_ftype + " — " + (p.recommend_reason || "")
                        }
                        color: AppTheme.textDim
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                    AppButton {
                        text: "Use"
                        flat: true
                        onClicked: {
                            var p = page.preview
                            if (!p || !p.recommended_ftype) return
                            page.quality = "custom"
                            page.ftype = p.recommended_ftype
                            if (page.isIQ(page.ftype)) page.generateIMatrix = true
                            page.schedulePreview()
                        }
                    }
                }
                FormField {
                    visible: page.quality === "custom" || advancedToggle.checked
                    Layout.fillWidth: true
                    label: "Exact type"
                    AppComboBox {
                        width: parent.width
                        model: page.customTypes()
                        currentIndex: Math.max(0, model.indexOf(page.ftype))
                        onActivated: function(i) {
                            page.ftype = model[i]
                            page.quality = "custom"
                            if (page.isIQ(page.ftype)) page.generateIMatrix = true
                            page.schedulePreview()
                        }
                    }
                }
            }

            SectionCard {
                title: "Estimate"
                Layout.fillWidth: true
                GridLayout {
                    columns: 2
                    columnSpacing: 24
                    rowSpacing: 6
                    Layout.fillWidth: true
                    Label { text: "Output size"; color: AppTheme.textFaint }
                    Label { text: page.preview ? AppTheme.bytes(page.preview.estimated_bytes) : "—"; color: AppTheme.text }
                    Label { text: "Change vs source"; color: AppTheme.textFaint }
                    Label {
                        text: page.preview ? ((page.preview.delta_bytes > 0 ? "+" : "") + AppTheme.bytes(page.preview.delta_bytes)) : "—"
                        color: AppTheme.text
                    }
                    Label { text: "Fits at default context"; color: AppTheme.textFaint }
                    Label {
                        text: page.fit ? (page.fit.fits ? "Likely fits (" + page.fit.budget_kind + ")" : "May not fit") : "—"
                        color: page.fit && page.fit.fits ? AppTheme.success : AppTheme.warning
                    }
                    Label { text: "Free disk"; color: AppTheme.textFaint }
                    Label {
                        text: page.preview ? AppTheme.bytes(page.preview.disk_free_bytes) : "—"
                        color: page.preview && page.preview.disk_ok ? AppTheme.text : AppTheme.danger
                    }
                    Label { text: "Importance matrix"; color: AppTheme.textFaint }
                    Label {
                        text: page.preview && page.preview.imatrix_required ? "Required"
                            : (page.preview && page.preview.imatrix_recommended ? "Recommended" : "Optional")
                        color: AppTheme.text
                    }
                }
                Repeater {
                    model: (page.preview && page.preview.warnings) || []
                    delegate: Label {
                        text: "• " + modelData
                        color: AppTheme.warning
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                }
                Repeater {
                    model: (page.preview && page.preview.blockers) || []
                    delegate: Label {
                        text: "• " + modelData
                        color: AppTheme.danger
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                }
            }

            SectionCard {
                title: "Importance matrix"
                visible: advancedToggle.checked || page.generateIMatrix || page.isIQ(page.ftype)
                    || (page.preview && (page.preview.imatrix_required || page.preview.imatrix_recommended))
                Layout.fillWidth: true
                AppCheckBox {
                    text: "Generate a new imatrix first (then quantize)"
                    checked: page.generateIMatrix
                    enabled: page.imatrixPresent()
                    onToggled: { page.generateIMatrix = checked; if (checked) page.imatrixId = "" }
                }
                FormField {
                    visible: !page.generateIMatrix
                    Layout.fillWidth: true
                    label: "Reuse existing"
                    AppComboBox {
                        width: parent.width
                        model: ["(none)"].concat(page.imatrices.map(function(im) {
                            return im.dataset_label + " · " + im.origin
                        }))
                        onActivated: function(i) {
                            page.imatrixId = i <= 0 ? "" : page.imatrices[i - 1].id
                        }
                    }
                }
                RowLayout {
                    AppButton { text: "Import…"; flat: true; onClicked: imDialog.open() }
                    AppButton {
                        visible: page.imatrices.length >= 2 && page.hasIMatrixFlag("--in-file")
                        text: "Combine two latest"
                        flat: true
                        onClicked: {
                            var body = page.requestBody()
                            body.kind = "combine_imatrix"
                            body.combine_imatrix_ids = [page.imatrices[0].id, page.imatrices[1].id]
                            page.api.post("/api/v1/quantize/jobs", body, function(st, data) {
                                if (st !== 202 && st !== 200)
                                    page.errorText = (data && (data.detail || data.error)) || "Combine failed"
                                else page.reloadJobs()
                            })
                        }
                    }
                    FormField {
                        visible: page.generateIMatrix
                        label: "Calibration"
                        AppComboBox {
                            model: ["quick", "standard", "thorough"]
                            currentIndex: Math.max(0, model.indexOf(page.calibrationPreset))
                            onActivated: function(i) { page.calibrationPreset = model[i] }
                        }
                    }
                    AppButton {
                        visible: page.generateIMatrix
                        text: page.calibrationPath !== "" ? "Custom file set" : "Custom file…"
                        flat: true
                        onClicked: calDialog.open()
                    }
                }
                Label {
                    visible: !page.imatrixPresent()
                    text: "This runtime has no llama-imatrix. Install an official llama.cpp build to generate matrices."
                    color: AppTheme.warning
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }

            ToolButton {
                id: advancedToggle
                checkable: true
                text: (checked ? "▾" : "▸") + " Advanced"
                font.weight: Font.DemiBold
            }

            ColumnLayout {
                visible: advancedToggle.checked
                Layout.fillWidth: true
                spacing: AppTheme.gap

                SectionCard {
                    title: "Options"
                    Layout.fillWidth: true
                    FormField {
                        Layout.fillWidth: true
                        label: "Output name"
                        hint: "Stored under the managed models directory."
                        AppTextField {
                            width: parent.width
                            text: page.outputName
                            placeholderText: "alias-Q4_K_M"
                            onTextChanged: page.outputName = text
                        }
                    }
                    FormField {
                        label: "Threads"
                        hint: "Last positional argument on llama-quantize."
                        AppSpinBox {
                            from: 0; to: 256; value: page.threads
                            onValueModified: page.threads = value
                        }
                    }
                    FormField {
                        label: "Pure"
                        argName: "--pure"
                        supported: page.hasFlag("--pure")
                        hint: "Disable mixed higher-bit tensors inside k-quants."
                        AppSwitch { checked: page.pure; onToggled: page.pure = checked }
                    }
                    FormField {
                        label: "Leave output tensor"
                        argName: "--leave-output-tensor"
                        supported: page.hasFlag("--leave-output-tensor")
                        AppSwitch { checked: page.leaveOutput; onToggled: page.leaveOutput = checked }
                    }
                    FormField {
                        label: "Keep split shards"
                        argName: "--keep-split"
                        supported: page.hasFlag("--keep-split")
                        AppSwitch { checked: page.keepSplit; onToggled: page.keepSplit = checked }
                    }
                    FormField {
                        label: "Allow requantize"
                        argName: "--allow-requantize"
                        supported: page.hasFlag("--allow-requantize")
                        hint: "Needed for Q6 and below. Q8 sources pass --allow-requantize automatically."
                        AppSwitch { checked: page.allowRequantize; onToggled: page.allowRequantize = checked }
                    }
                    AppCheckBox {
                        visible: page.allowRequantize
                        text: "I understand requantizing will lose quality"
                        checked: page.ackRequantize
                        onToggled: page.ackRequantize = checked
                    }
                    AppCheckBox {
                        visible: !!(page.ftypeMeta(page.ftype) && page.ftypeMeta(page.ftype).experimental)
                        text: "I accept this experimental quantization type"
                        checked: page.ackExperimental
                        onToggled: page.ackExperimental = checked
                    }
                    AppCheckBox {
                        visible: page.preview && !page.preview.ram_ok
                        text: "Start anyway (source may not fit in free RAM)"
                        checked: page.ackRAM
                        onToggled: page.ackRAM = checked
                    }
                    FormField {
                        label: "Output tensor type"
                        argName: "--output-tensor-type"
                        supported: page.hasFlag("--output-tensor-type")
                        AppTextField {
                            width: parent.width
                            text: page.outputTensorType
                            placeholderText: "q8_0"
                            onTextChanged: page.outputTensorType = text
                        }
                    }
                    FormField {
                        label: "Token embedding type"
                        argName: "--token-embedding-type"
                        supported: page.hasFlag("--token-embedding-type")
                        AppTextField {
                            width: parent.width
                            text: page.tokenEmbeddingType
                            placeholderText: "q8_0"
                            onTextChanged: page.tokenEmbeddingType = text
                        }
                    }
                    FormField {
                        label: "Tensor overrides"
                        argName: "--tensor-type"
                        supported: page.hasFlag("--tensor-type")
                        hint: "One TENSOR=TYPE per line."
                        AppTextArea {
                            width: parent.width
                            implicitHeight: 72
                            text: page.tensorTypesText
                            onTextChanged: page.tensorTypesText = text
                        }
                    }
                    FormField {
                        visible: page.quantizeProjector || (page.sourceModel() && page.sourceModel().projector_path)
                        label: "Projector type"
                        AppComboBox {
                            model: ["Q8_0", "Q6_K", "Q4_K_M", "F16"]
                            currentIndex: Math.max(0, model.indexOf(page.projectorFType))
                            onActivated: function(i) {
                                page.projectorFType = model[i]
                                page.quantizeProjector = true
                                page.copyProjector = false
                            }
                        }
                    }
                    FormField {
                        label: "IMatrix GPU layers"
                        argName: "-ngl"
                        supported: page.hasIMatrixFlag("-ngl") || page.hasIMatrixFlag("--n-gpu-layers")
                        AppSpinBox {
                            from: 0; to: 999; value: page.gpuLayers
                            onValueModified: page.gpuLayers = value
                        }
                    }
                    AppCheckBox {
                        text: "Keep generated imatrix for reuse"
                        checked: page.keepIMatrix
                        onToggled: page.keepIMatrix = checked
                    }
                    AppCheckBox {
                        visible: page.loadedModels.length > 0 || page.unloadFirst
                        text: "Unload loaded models first (recommended before imatrix GPU offload)"
                        checked: page.unloadFirst
                        onToggled: page.unloadFirst = checked
                    }
                    AppCheckBox {
                        text: "Parse special tokens in calibration"
                        checked: page.parseSpecial
                        onToggled: page.parseSpecial = checked
                    }
                }

                SectionCard {
                    title: "OpenInfer Adaptive"
                    subtitle: "Mixed precision from tensor-name heuristics and optional imatrix statistics."
                    Layout.fillWidth: true
                    RowLayout {
                        Repeater {
                            model: [
                                { "id": "quality", "label": "Adaptive Quality" },
                                { "id": "balanced", "label": "Adaptive Balanced" },
                                { "id": "compact", "label": "Adaptive Compact" }
                            ]
                            delegate: AppButton {
                                text: modelData.label
                                primary: page.adaptivePreset === modelData.id
                                onClicked: {
                                    page.adaptivePreset = modelData.id
                                    page.quality = "adaptive"
                                    page.generateIMatrix = true
                                    page.schedulePreview()
                                }
                            }
                        }
                        AppButton {
                            text: "Off"
                            flat: true
                            visible: page.adaptivePreset !== ""
                            onClicked: { page.adaptivePreset = ""; page.quality = "custom"; page.schedulePreview() }
                        }
                    }
                    Label {
                        visible: !page.hasFlag("--tensor-type-file")
                        text: "This runtime’s llama-quantize does not advertise --tensor-type-file, so Adaptive cannot run."
                        color: AppTheme.warning
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Label {
                    text: page.startDisabledReason()
                    color: AppTheme.warning
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                    visible: page.startDisabledReason() !== ""
                }
                AppButton {
                    text: page.jobSubmitting ? "Starting…" : "Start"
                    primary: true
                    enabled: !page.jobSubmitting && page.startDisabledReason() === ""
                    onClicked: page.startJob()
                }
            }

            SectionCard {
                title: "Jobs"
                Layout.fillWidth: true
                EmptyState {
                    visible: page.jobs.length === 0
                    icon: "▣"
                    title: "No quantization jobs yet"
                    hint: "Start a job above."
                }
                RowLayout {
                    Layout.fillWidth: true
                    visible: page.jobs.length > 0
                    Item { Layout.fillWidth: true }
                    AppButton {
                        text: "Clear history"
                        flat: true
                        enabled: page.hasFinishedJobs
                        onClicked: page.clearJobHistory()
                    }
                }
                Repeater {
                    model: page.jobs
                    delegate: Card {
                        Layout.fillWidth: true
                        implicitHeight: jcol.implicitHeight + 20
                        ColumnLayout {
                            id: jcol
                            anchors.fill: parent
                            anchors.margins: 10
                            spacing: 6
                            RowLayout {
                                Layout.fillWidth: true
                                StatusDot { state: modelData.state }
                                Label {
                                    text: (modelData.kind || "quantize") + " · " + (modelData.request && modelData.request.ftype ? modelData.request.ftype : "")
                                    color: AppTheme.text
                                    font.weight: Font.DemiBold
                                    Layout.fillWidth: true
                                }
                                Label {
                                    text: modelData.stage || modelData.state
                                    color: AppTheme.textDim
                                    font.pixelSize: AppTheme.fontSmall
                                }
                            }
                            AppProgressBar {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 8
                                from: 0; to: 1
                                value: modelData.progress || 0
                            }
                            Label {
                                visible: !!(modelData.state === "complete" && modelData.result && modelData.result.alias)
                                text: "Added to library as " + ((modelData.result && modelData.result.alias) || "")
                                color: AppTheme.textDim
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                            Label {
                                visible: !!(modelData.error && modelData.error !== "")
                                text: modelData.error || ""
                                color: AppTheme.danger
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 6
                                Item { Layout.fillWidth: true }
                                AppButton {
                                    visible: ["queued", "running", "canceling"].indexOf(modelData.state) >= 0
                                    text: "Cancel"
                                    flat: true
                                    onClicked: page.api.post("/api/v1/quantize/jobs/" + modelData.id + "/cancel", {}, function() { page.reloadJobs() })
                                }
                                AppButton {
                                    visible: ["complete", "failed", "canceled"].indexOf(modelData.state) >= 0
                                    text: "Remove"
                                    flat: true
                                    onClicked: page.api.del("/api/v1/quantize/jobs/" + modelData.id, function() { page.reloadJobs() })
                                }
                            }
                        }
                    }
                }
            }

            Item { Layout.preferredHeight: AppTheme.pad }
        }
    }

    Component.onCompleted: reload()
}
