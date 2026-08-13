import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "services"
import "components"
import "pages"
import "dialogs"
import "."

ApplicationWindow {
    id: window
    title: "OpenInfer Studio"
    width: 1280
    height: 800
    minimumWidth: 900
    minimumHeight: 600
    visible: true
    color: AppTheme.bg

    Component.onCompleted: {
        AppTheme.applyPalette(window)
        window.reloadAll()
    }
    Connections {
        target: AppTheme
        function onModeChanged() { AppTheme.applyPalette(window) }
        function onDarkChanged() { AppTheme.applyPalette(window) }
    }

    Api { id: api }
    Events {
        id: events
        onReconnected: window.reloadAll()
        onEventReceived: function(name, payload) {
            switch (name) {
            case "instance.state_changed":
            case "instance.updated":
                window.refreshInstances()
                break
            case "download.state_changed":
                window.refreshActivityBadge()
                if (payload.state === "complete")
                    window.toast((payload.label || "Download") + " is ready in your library", "success")
                break
            case "download.progress":
                // DownloadsPage handles live progress; badge only needs state changes.
                break
            case "quant.state_changed":
                if (payload.state === "complete")
                    window.toast((payload.ftype ? payload.ftype : "Quantized model") + " is ready in your library", "success")
                else if (payload.state === "failed")
                    window.toast("Quantization failed" + (payload.error ? ": " + payload.error : ""), "error")
                break
            case "runtime.installed":
                window.toast("Runtime installed: " + (payload.id || ""), "success")
                break
            case "library.scanned":
                break
            case "log.entry":
                break
            }
        }
    }

    property var instances: []
    property var hardware: null
    property var recommendation: null
    property int downloadCount: 0
    property bool experimentalAudioModels: false
    property bool onboardingCompleted: true
    property bool onboardingPrompted: false
    property string currentRoute: "chat"
    property string previousRoute: "library"

    function routeIndex(route) {
        var routes = ["chat", "models", "library", "developer", "runtimes", "quantize",
                      "downloads", "logs", "settings", "model-detail"]
        return routes.indexOf(route)
    }
    function goTo(route) {
        if (route === "model-detail") previousRoute = currentRoute
        if (route !== "model-detail" && currentRoute === "model-detail")
            previousRoute = route
        currentRoute = route
    }
    function goBack() {
        goTo(previousRoute || "library")
    }

    function refreshSettings() {
        api.get("/api/v1/settings", function(st, data) {
            if (st !== 200 || !data) return
            window.experimentalAudioModels = (data["experimental.audio_models"] || "0") === "1"
            window.onboardingCompleted = (data["onboarding.completed"] || "0") === "1"
            window.maybeOpenOnboarding()
        })
    }
    function maybeOpenOnboarding() {
        if (window.onboardingCompleted || window.onboardingPrompted)
            return
        // Wait for hardware recommendation when possible so step 2 can pick a backend.
        if (window.hardware === null && window.recommendation === null)
            return
        window.onboardingPrompted = true
        // Existing installs: skip the wizard silently once they already have
        // a runtime. Only prompt true first-run empties.
        api.get("/api/v1/runtimes", function(rst, rdata) {
            var runtimes = (rst === 200 && rdata) ? (rdata.runtimes || []) : []
            if (runtimes.length > 0) {
                api.put("/api/v1/settings/onboarding.completed", { "value": "1" }, function() {})
                window.onboardingCompleted = true
                return
            }
            onboardingDialog.openWizard()
        })
    }
    function refreshInstances() {
        api.get("/api/v1/instances", function(st, data) {
            if (st === 200 && data) window.instances = data.instances || []
        })
    }
    function refreshActivityBadge() {
        api.get("/api/v1/downloads", function(st, data) {
            if (st === 200 && data)
                window.downloadCount = (data.downloads || []).filter(
                    function(d) { return d.state === "active" || d.state === "queued" }).length
        })
    }
    function reloadAll() {
        refreshInstances()
        refreshActivityBadge()
        refreshSettings()
        api.get("/api/v1/hardware", function(st, data) {
            if (st === 200 && data) {
                window.hardware = data.hardware
                window.recommendation = data.recommendation
            } else {
                window.hardware = window.hardware || ({})
            }
            window.maybeOpenOnboarding()
        })
        for (var i = 0; i < stack.count; i++) {
            var p = stack.itemAt(i)
            if (p && p.reload) p.reload()
        }
    }

    ListModel { id: toastModel }
    function toast(text, kind) {
        toastModel.append({ "text": text, "kind": kind || "info" })
        toastTimer.restart()
    }
    Timer {
        id: toastTimer
        interval: 5000
        onTriggered: if (toastModel.count > 0) toastModel.remove(0)
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Calm global bar: app identity on the left, activity on the right.
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 56
            color: AppTheme.bgAlt
            border.color: AppTheme.border

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: AppTheme.pad
                anchors.rightMargin: AppTheme.pad
                spacing: AppTheme.gap

                Column {
                    spacing: 1
                    Text {
                        text: "OpenInfer Studio"
                        color: AppTheme.text
                        font.pixelSize: AppTheme.fontTitle
                        font.weight: Font.DemiBold
                    }
                    Text {
                        text: "Local inference, made practical"
                        color: AppTheme.textFaint
                        font.pixelSize: AppTheme.fontSmall
                    }
                }
                Item { Layout.fillWidth: true }

                // Keep operational context available without turning the header
                // into a permanent status dashboard.
                RowLayout {
                    visible: window.instances.length > 0
                    spacing: 6
                    Text {
                        text: window.instances.filter(function(i) {
                            return ["ready", "busy", "loading", "starting"].indexOf(i.state) >= 0
                        }).length + " active model" + (window.instances.length === 1 ? "" : "s")
                        color: AppTheme.textDim
                        font.pixelSize: AppTheme.fontSmall
                    }
                    Rectangle {
                        width: 7; height: 7; radius: 4
                        color: window.instances.some(function(i) { return i.state === "busy" })
                            ? AppTheme.success : AppTheme.info
                    }
                }
                AppButton {
                    text: window.downloadCount > 0
                          ? "Activity · " + window.downloadCount : "Activity"
                    primary: window.downloadCount > 0
                    onClicked: window.goTo("downloads")
                }
                Row {
                    visible: api.lastError !== "" && !events.connected
                    spacing: 6
                    Rectangle { width: 8; height: 8; radius: 4; color: AppTheme.danger; anchors.verticalCenter: parent.verticalCenter }
                    Text { text: "Backend offline"; color: AppTheme.danger; font.pixelSize: AppTheme.fontSmall }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            // Navigation is task-oriented. Operational pages stay available in
            // the secondary group without competing with the everyday flow.
            Rectangle {
                Layout.fillHeight: true
                Layout.preferredWidth: window.width < 1060 ? 64 : 196
                color: AppTheme.bgAlt
                border.color: AppTheme.border

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 8
                    spacing: 4
                    Repeater {
                        model: [
                            { "label": "Chat", "route": "chat", "glyph": "◌" },
                            { "label": "Browse models", "route": "models", "glyph": "⌕" },
                            { "label": "My library", "route": "library", "glyph": "▦" }
                        ]
                        delegate: NavButton {
                            text: modelData.label
                            glyph: modelData.glyph
                            compact: window.width < 1060
                            current: window.currentRoute === modelData.route
                                || (window.currentRoute === "model-detail" && modelData.route === "library")
                            onClicked: window.goTo(modelData.route)
                        }
                    }
                    Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: AppTheme.border; Layout.topMargin: 8 }
                    Label {
                        visible: window.width >= 1060
                        text: "TOOLS"
                        color: AppTheme.textFaint
                        font.pixelSize: AppTheme.fontSmall
                        font.weight: Font.DemiBold
                        font.letterSpacing: 1
                        Layout.fillWidth: true
                        horizontalAlignment: Text.AlignHCenter
                        Layout.topMargin: 6
                    }
                    Repeater {
                        model: [
                            { "label": "Downloads", "route": "downloads", "glyph": "↓" },
                            { "label": "Runtimes", "route": "runtimes", "glyph": "⚙" },
                            { "label": "Quantization", "route": "quantize", "glyph": "▣" },
                            { "label": "Developer API", "route": "developer", "glyph": "⌘" },
                            { "label": "Logs", "route": "logs", "glyph": "≡" }
                        ]
                        delegate: NavButton {
                            text: modelData.label
                            glyph: modelData.glyph
                            compact: window.width < 1060
                            current: window.currentRoute === modelData.route
                            onClicked: window.goTo(modelData.route)
                        }
                    }
                    Item { Layout.fillHeight: true }
                    NavButton {
                        text: "Settings"
                        glyph: "⚙"
                        compact: window.width < 1060
                        current: window.currentRoute === "settings"
                        onClicked: window.goTo("settings")
                    }
                }
            }

            // Pages
            StackLayout {
                id: stack
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: window.routeIndex(window.currentRoute)

                ChatPage {
                    api: api
                    events: events
                    experimentalAudio: window.experimentalAudioModels
                    onOpenLibrary: window.goTo("library")
                    onConfigureModel: function(modelId) {
                        window.goTo("library")
                        libraryPage.openLoad(modelId)
                    }
                }
                DiscoverPage {
                    api: api
                    events: events
                    experimentalAudio: window.experimentalAudioModels
                    onDownloadQueued: function(label) {
                        window.refreshActivityBadge()
                        window.toast(label + " added to downloads", "success")
                    }
                }
                LibraryPage {
                    id: libraryPage
                    api: api
                    events: events
                    experimentalAudio: window.experimentalAudioModels
                    onOpenDetail: function(modelId) { window.openInstanceDetail(modelId) }
                    onBrowseModels: window.goTo("models")
                    onQuantizeModel: function(modelId) {
                        quantizePage.prefillModel(modelId)
                        window.goTo("quantize")
                    }
                }
                DeveloperPage { api: api; events: events }
                RuntimesPage  { api: api; events: events; recommendation: window.recommendation }
                QuantizationPage { id: quantizePage; api: api; events: events }
                DownloadsPage { api: api; events: events }
                LogsPage      { api: api; events: events }
                SettingsPage  {
                    api: api
                    events: events
                    onSettingChanged: function(key, value) {
                        if (key === "experimental.audio_models")
                            window.experimentalAudioModels = value === "1"
                    }
                    onReplayOnboarding: {
                        window.onboardingCompleted = false
                        window.onboardingPrompted = false
                        onboardingDialog.openWizard()
                    }
                }
                InstanceDetailPage {
                    id: instanceDetailPage
                    api: api
                    events: events
                    onBack: window.goBack()
                    onQuantizeRequested: {
                        quantizePage.prefillModel(instanceDetailPage.modelId)
                        window.goTo("quantize")
                    }
                }
            }
        }
    }

    // Detail pages are routed by name and retain the originating page for Back.
    function openInstanceDetail(modelId) {
        instanceDetailPage.modelId = modelId
        instanceDetailPage.enter()
        goTo("model-detail")
    }

    // Toast overlay
    Column {
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 16
        spacing: 8
        z: 100
        Repeater {
            model: toastModel
            delegate: Rectangle {
                width: toastText.implicitWidth + 32
                height: 40
                radius: AppTheme.radius
                color: model.kind === "success" ? AppTheme.success
                     : model.kind === "error" ? AppTheme.danger : AppTheme.surfaceHi
                border.color: AppTheme.border
                Text {
                    id: toastText
                    anchors.centerIn: parent
                    text: model.text
                    color: model.kind === "info" ? AppTheme.text : "white"
                    font.pixelSize: AppTheme.fontBody
                }
                MouseArea { anchors.fill: parent; onClicked: toastModel.remove(index) }
            }
        }
    }

    OnboardingDialog {
        id: onboardingDialog
        parent: Overlay.overlay
        api: api
        events: events
        recommendation: window.recommendation
        stack: stack
        onFinished: window.onboardingCompleted = true
        onGoChat: window.goTo("chat")
        onGoRuntimes: window.goTo("runtimes")
        onGoModels: window.goTo("models")
    }
}
