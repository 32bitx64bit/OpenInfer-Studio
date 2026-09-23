import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import ".."
import "../components"

Item {
    id: page
    property var api
    property var events
    property var recommendation: null

    property var installed: []
    property var releases: []
    property var sdReleases: []
    property bool checking: false
    property bool checkingSD: false
    property string backendFilter: ""   // user override for asset resolution
    property string errorText: ""
    property string sdErrorText: ""
    property var runtimeDownloads: []   // active runtime downloads
    property var liveProgress: ({})
    property var pendingInstalls: []    // downloaded, extracting/verifying

    function reload() {
        api.get("/api/v1/runtimes", function(st, data) {
            if (st === 200) page.installed = (data && data.runtimes) || []
        })
        api.get("/api/v1/downloads", function(st, data) {
            if (st !== 200) return
            page.runtimeDownloads = AppTheme.keepRows(page.runtimeDownloads, ((data && data.downloads) || []).filter(function(d) {
                return d.kind === "runtime" && ["queued", "active", "paused", "failed"].indexOf(d.state) >= 0
            }))
        })
    }

    function checkReleases() {
        page.checking = true
        page.errorText = ""
        var q = page.backendFilter !== "" ? "?backend=" + page.backendFilter : ""
        api.get("/api/v1/runtimes/releases" + q, function(st, data) {
            page.checking = false
            if (st === 200) page.releases = (data && data.releases) || []
            else page.errorText = (data && (data.detail || data.error)) || ("HTTP " + st)
        })
    }

    function checkSDReleases() {
        page.checkingSD = true
        page.sdErrorText = ""
        var q = "?kind=diffusion" + (page.backendFilter !== "" ? "&backend=" + page.backendFilter : "")
        api.get("/api/v1/runtimes/releases" + q, function(st, data) {
            page.checkingSD = false
            if (st === 200) page.sdReleases = (data && data.releases) || []
            else page.sdErrorText = (data && (data.detail || data.error)) || ("HTTP " + st)
        })
    }

    function installRelease(tag, asset, backend, kind) {
        var body = { "tag": tag, "asset": asset, "backend": backend }
        if (kind === "diffusion") body.kind = "diffusion"
        page.api.post("/api/v1/runtimes/install", body, function(st, data) {
            if (st !== 202) {
                var msg = (data && (data.detail || data.error)) || "install failed"
                if (kind === "diffusion") page.sdErrorText = msg
                else page.errorText = msg
            }
        })
    }

    Connections {
        target: page.events
        function onEventReceived(name, payload) {
            switch (name) {
            case "download.progress": {
                var p = {}
                var old = page.liveProgress
                for (var k in old) p[k] = old[k]
                p[payload.id] = payload
                page.liveProgress = p
                break
            }
            case "download.state_changed":
                page.reload()
                break
            case "runtime.installing": {
                var li = page.pendingInstalls.slice()
                li.push(payload)
                page.pendingInstalls = li
                break
            }
            case "runtime.installed":
                page.pendingInstalls = []
                page.errorText = ""
                page.reload()
                break
            case "runtime.install_failed":
                page.pendingInstalls = []
                page.errorText = "Runtime install failed: " + (payload.error || "unknown error")
                page.reload()
                break
            }
        }
    }

    // Poll while a runtime install is active so progress moves even if an
    // event is missed.
    Timer {
        interval: 1500
        repeat: true
        running: page.runtimeDownloads.length > 0 || page.pendingInstalls.length > 0
        onTriggered: page.reload()
    }

    ScrollView {
        anchors.fill: parent
        anchors.margins: AppTheme.pad
        clip: true
        ColumnLayout {
            width: page.width - AppTheme.pad * 2
            spacing: AppTheme.gap * 1.5

            PageHeader {
                title: "Runtimes"
                subtitle: "Install and manage llama.cpp builds. Most people only need the recommended backend for this machine."
            }

            // Recommendation banner
            Card {
                Layout.fillWidth: true
                visible: page.recommendation !== null
                implicitHeight: recCol.implicitHeight + 24
                ColumnLayout {
                    id: recCol
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 4
                    Label {
                        text: "Recommended backend: " + (page.recommendation ? page.recommendation.backend.toUpperCase() : "—")
                        color: AppTheme.accent
                        font.weight: Font.DemiBold
                    }
                    Label {
                        Layout.fillWidth: true
                        text: page.recommendation ? page.recommendation.reason
                            + "  (alternatives: " + (page.recommendation.alternatives || []).join(", ") + ")" : ""
                        color: AppTheme.textDim
                        font.pixelSize: AppTheme.fontSmall
                        wrapMode: Text.WordWrap
                    }
                }
            }

            // In-progress installations
            AppGroupBox {
                Layout.fillWidth: true
                visible: page.runtimeDownloads.length > 0 || page.pendingInstalls.length > 0
                title: "Installations in progress"
                ColumnLayout {
                    width: parent.width
                    spacing: 8
                    Repeater {
                        model: page.runtimeDownloads
                        delegate: ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 4
                            RowLayout {
                                Label { text: modelData.label; color: AppTheme.text; Layout.fillWidth: true; elide: Text.ElideMiddle }
                                Label {
                                    text: {
                                        var live = page.liveProgress[modelData.id]
                                        var done = live ? live.done_bytes : modelData.done_bytes
                                        var total = live && live.total_bytes ? live.total_bytes : modelData.total_bytes
                                        return AppTheme.bytes(done) + " / " + AppTheme.bytes(total)
                                    }
                                    color: AppTheme.textDim
                                    font.pixelSize: AppTheme.fontSmall
                                }
                            }
                            AppProgressBar {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 8
                                from: 0; to: 1
                                value: {
                                    var live = page.liveProgress[modelData.id]
                                    var done = live ? live.done_bytes : modelData.done_bytes
                                    var total = live && live.total_bytes ? live.total_bytes : modelData.total_bytes
                                    return total > 0 ? done / total : 0
                                }
                            }
                            Label {
                                visible: modelData.error !== ""
                                text: modelData.error
                                color: AppTheme.danger
                                font.pixelSize: AppTheme.fontSmall
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                        }
                    }
                    Repeater {
                        model: page.pendingInstalls
                        delegate: RowLayout {
                            Layout.fillWidth: true
                            BusyIndicator { running: true; implicitWidth: 24; implicitHeight: 24 }
                            Label {
                                text: "Verifying and installing " + modelData.tag + " (" + modelData.backend + ")…"
                                color: AppTheme.textDim
                            }
                        }
                    }
                }
            }

            // Installed runtimes
            AppGroupBox {
                Layout.fillWidth: true
                title: "Installed runtimes (" + page.installed.length + ")"
                ColumnLayout {
                    width: parent.width
                    spacing: 8

                    EmptyState {
                        visible: page.installed.length === 0
                        width: parent.width
                        icon: "⚙"
                        title: "No runtimes installed"
                        hint: "Install an official llama.cpp build below, or import a custom llama-server binary or archive (.zip / .tar.gz)."
                    }

                    Repeater {
                        model: page.installed
                        delegate: Card {
                            Layout.fillWidth: true
                            implicitHeight: rtCol.implicitHeight + 20
                            ColumnLayout {
                                id: rtCol
                                anchors.fill: parent
                                anchors.margins: 10
                                spacing: 4
                                RowLayout {
                                    Layout.fillWidth: true
                                    StatusDot { state: modelData.healthy ? "ready" : "failed" }
                                    Label { text: modelData.id; color: AppTheme.text; font.weight: Font.DemiBold; Layout.fillWidth: true; elide: Text.ElideMiddle }
                                    Tag { text: modelData.backend; tone: AppTheme.info }
                                    Tag { visible: modelData.preferred; text: "preferred"; tone: AppTheme.success }
                                    Tag { visible: modelData.source === "custom-import"; text: "custom"; tone: AppTheme.warning }
                                }
                                Label {
                                    text: modelData.platform + "/" + modelData.architecture
                                        + " · installed " + (modelData.installed_at || "").substring(0, 10)
                                        + ((modelData.used_by_models && modelData.used_by_models.length)
                                            ? " · pinned by: " + modelData.used_by_models.join(", ") : "")
                                    color: AppTheme.textDim
                                    font.pixelSize: AppTheme.fontSmall
                                }
                                Label {
                                    Layout.fillWidth: true
                                    text: (modelData.version_output || "").split("\n")[0]
                                    color: AppTheme.textFaint
                                    font.pixelSize: AppTheme.fontSmall
                                    elide: Text.ElideRight
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 6
                                    AppButton {
                                        text: "Health test"
                                        flat: true
                                        onClicked: page.api.post("/api/v1/runtimes/" + modelData.id + "/health", {},
                                            function(st, data) {
                                                page.errorText = data && data.healthy
                                                    ? "" : ("Health check failed: " + ((data && data.error) || ""))
                                                page.reload()
                                            })
                                    }
                                    AppButton {
                                        visible: !modelData.preferred
                                        text: "Set preferred"
                                        flat: true
                                        onClicked: page.api.post("/api/v1/runtimes/" + modelData.id + "/preferred", {},
                                            function() { page.reload() })
                                    }
                                    AppButton {
                                        text: "Open directory"
                                        flat: true
                                        onClicked: Qt.openUrlExternally("file://" + modelData.install_dir)
                                    }
                                    AppButton {
                                        text: "Capabilities"
                                        flat: true
                                        onClicked: {
                                            page.api.get("/api/v1/runtimes/" + modelData.id + "/capabilities", function(st, data) {
                                                if (st === 200) {
                                                    capsDialog.caps = data.capabilities || []
                                                    capsDialog.versionOut = data.version_output || ""
                                                    capsDialog.helpText = data.help || ""
                                                    capsDialog.open()
                                                }
                                            })
                                        }
                                    }
                                    Item { Layout.fillWidth: true }
                                    AppButton {
                                        text: "Remove"
                                        flat: true
                                        onClicked: {
                                            removeDialog.runtimeId = modelData.id
                                            removeDialog.message = "Remove runtime " + modelData.id + "? Models pinning it must be repinned first."
                                            removeDialog.open()
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Release discovery — one shared card component per engine, same
            // layout: backend filter + check/import row, error line, then
            // release cards with asset rows.
            ReleaseList {
                engineTitle: "llama.cpp"
                engineKind: ""
                checking: page.checking
                releases: page.releases
                errorText: page.errorText
                checkLabel: "Check for releases"
                pageBackendFilter: page.backendFilter
                onPageBackendFilterChanged: page.backendFilter = pageBackendFilter
                onCheck: page.checkReleases()
                onInstall: function(tag, asset, backend) { page.installRelease(tag, asset, backend, "") }
                onImportCustom: importRuntimeDialog.open()
            }

            // stable-diffusion.cpp releases (image/video engine)
            ReleaseList {
                engineTitle: "stable-diffusion.cpp"
                engineKind: "diffusion"
                engineNote: "Image and video generator engine (SD · SDXL · FLUX · Wan). Same install pipeline as llama.cpp, including the Windows CUDA cudart bundle."
                checking: page.checkingSD
                releases: page.sdReleases
                errorText: page.sdErrorText
                checkLabel: "Check for sd.cpp releases"
                pageBackendFilter: page.backendFilter
                onPageBackendFilterChanged: page.backendFilter = pageBackendFilter
                onCheck: page.checkSDReleases()
                onInstall: function(tag, asset, backend) { page.installRelease(tag, asset, backend, "diffusion") }
                onImportCustom: importRuntimeDialog.open()
            }
        }
    }

    AppDialog {
        id: capsDialog
        property var caps: []
        property string versionOut: ""
        property string helpText: ""
        title: "Runtime capabilities"
        width: 520
        height: 420
        standardButtons: Dialog.NoButton
        ColumnLayout {
            anchors.fill: parent
            Label { text: capsDialog.versionOut.split("\n")[0]; color: AppTheme.textDim }
            Label { text: "Supported settings (" + capsDialog.caps.length + ")"; color: AppTheme.text; font.weight: Font.DemiBold }
            ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                TextEdit {
                    width: parent.width
                    readOnly: true
                    text: capsDialog.caps.join("\n")
                    color: AppTheme.text
                    font.family: "monospace"
                    font.pixelSize: AppTheme.fontSmall
                }
            }
            AppButton { text: "Close"; onClicked: capsDialog.close(); Layout.alignment: Qt.AlignRight }
        }
    }

    ConfirmDialog {
        id: removeDialog
        property string runtimeId: ""
        confirmText: "Remove"
        onConfirmed: page.api.del("/api/v1/runtimes/" + runtimeId, function(st, data) {
            if (st !== 200)
                page.errorText = (data && (data.detail || data.error)) || "remove failed"
            page.reload()
        })
    }

    FileDialog {
        id: importRuntimeDialog
        title: "Select llama-server, sd-server, or archive"
        nameFilters: [
            "Server or archive (llama-server* sd-server* sd-cli* sd_cli* *.zip *.tar.gz *.tgz)",
            "All files (*)"
        ]
        onAccepted: page.api.post("/api/v1/runtimes/import",
            { "path": String(selectedFile).replace("file://", "") },
            function(st, data) {
                if (st !== 201) page.errorText = (data && (data.detail || data.error)) || "import failed"
                page.reload()
            })
    }

    Component.onCompleted: reload()
}
