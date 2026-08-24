import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".."
import "../components"

// First-run setup: welcome → runtime → a clear next step toward first chat.
Dialog {
    id: root
    property var api
    property var events
    property var recommendation: null
    property var stack: null

    property int step: 0
    property var installed: []
    property bool installing: false
    property bool checking: false
    property string statusText: ""
    property string errorText: ""
    property real installProgress: -1

    readonly property int stepCount: 3
    readonly property bool hasRuntime: installed.length > 0

    signal finished()
    signal goChat()
    signal goRuntimes()
    signal goModels()

    title: ""
    modal: true
    anchors.centerIn: parent
    width: Math.min(560, (parent ? parent.width : 560) - 48)
    height: Math.min(480, (parent ? parent.height : 480) - 48)
    padding: 0
    closePolicy: Popup.NoAutoClose
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

    function openWizard() {
        root.step = 0
        root.errorText = ""
        root.statusText = ""
        root.installing = false
        root.installProgress = -1
        root.refresh()
        root.open()
    }

    function refresh() {
        if (!root.api) return
        root.api.get("/api/v1/runtimes", function(st, data) {
            if (st === 200) root.installed = (data && data.runtimes) || []
        })
    }

    function completeAndClose() {
        if (root.api) {
            root.api.put("/api/v1/settings/onboarding.completed", { "value": "1" }, function() {})
        }
        root.close()
        root.finished()
    }

    function skip() {
        root.completeAndClose()
    }

    function installRecommended() {
        root.errorText = ""
        root.statusText = "Looking up llama.cpp releases…"
        root.checking = true
        var backend = ""
        if (root.recommendation && root.recommendation.backend)
            backend = root.recommendation.backend
        var q = backend !== "" ? ("?backend=" + backend) : ""
        root.api.get("/api/v1/runtimes/releases" + q, function(st, data) {
            root.checking = false
            if (st !== 200) {
                root.errorText = (data && (data.detail || data.error)) || ("HTTP " + st)
                root.statusText = ""
                return
            }
            var releases = (data && data.releases) || []
            if (releases.length === 0) {
                root.errorText = "No matching releases found for this machine."
                root.statusText = ""
                return
            }
            var rel = releases[0]
            var matches = rel.matches || []
            if (matches.length === 0) {
                root.errorText = "No installable asset for the recommended backend."
                root.statusText = ""
                return
            }
            var m = matches[0]
            root.installing = true
            root.statusText = "Downloading " + m.asset.name + "…"
            root.api.post("/api/v1/runtimes/install", {
                "tag": rel.tag,
                "asset": m.asset.name,
                "backend": m.backend
            }, function(ist, idata) {
                if (ist !== 202) {
                    root.installing = false
                    root.errorText = (idata && (idata.detail || idata.error)) || "Install failed"
                    root.statusText = ""
                }
            })
        })
    }

    Connections {
        target: root.events
        function onEventReceived(name, payload) {
            if (!root.visible) return
            switch (name) {
            case "download.progress":
                if (payload && payload.kind === "runtime" && payload.total > 0)
                    root.installProgress = payload.received / payload.total
                break
            case "runtime.installing":
                root.installing = true
                root.statusText = "Extracting and verifying…"
                break
            case "runtime.installed":
                root.installing = false
                root.installProgress = 1
                root.statusText = "Runtime ready."
                root.refresh()
                break
            case "runtime.install_failed":
                root.installing = false
                root.installProgress = -1
                root.errorText = "Install failed: " + (payload.error || "unknown error")
                root.statusText = ""
                break
            case "download.state_changed":
                root.refresh()
                break
            }
        }
    }

    Timer {
        interval: 2000
        repeat: true
        running: root.visible
        onTriggered: root.refresh()
    }

    NumberAnimation {
        id: stepFade
        property: "opacity"
        to: 1
        duration: AppTheme.motion
        easing.type: Easing.OutCubic
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: headerCol.implicitHeight + 24
            color: AppTheme.bgAlt
            border.color: AppTheme.border
            ColumnLayout {
                id: headerCol
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 16
                spacing: 6
                Label {
                    text: "Welcome to OpenInfer Studio"
                    color: AppTheme.text
                    font.pixelSize: AppTheme.fontTitle
                    font.weight: Font.DemiBold
                }
                Label {
                    text: "Step " + (root.step + 1) + " of " + root.stepCount
                    color: AppTheme.textFaint
                    font.pixelSize: AppTheme.fontSmall
                }
                Row {
                    spacing: 6
                    Repeater {
                        model: root.stepCount
                        Rectangle {
                            width: 28; height: 4; radius: 2
                            color: index <= root.step ? AppTheme.accent : AppTheme.border
                            Behavior on color { ColorAnimation { duration: AppTheme.motion } }
                        }
                    }
                }
            }
        }

        StackLayout {
            id: onboardingStack
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 16
            currentIndex: root.step
            property bool ready: false
            Component.onCompleted: ready = true
            onCurrentIndexChanged: {
                if (!ready)
                    return
                for (var i = 0; i < count; i++) {
                    var p = itemAt(i)
                    if (p)
                        p.opacity = 1
                }
                var item = itemAt(currentIndex)
                if (!item)
                    return
                item.opacity = 0
                stepFade.target = item
                stepFade.start()
            }

            // Step 0 — Welcome
            ColumnLayout {
                spacing: 12
                Label {
                    Layout.fillWidth: true
                    text: "Run GGUF models locally with llama.cpp. Models, chats, and hardware info stay on this machine — no accounts or cloud inference."
                    wrapMode: Text.WordWrap
                    color: AppTheme.textDim
                }
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: hwCol.implicitHeight + 20
                    radius: AppTheme.radius
                    color: AppTheme.surface
                    border.color: AppTheme.border
                    ColumnLayout {
                        id: hwCol
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 10
                        spacing: 4
                        Label {
                            text: "Detected hardware"
                            color: AppTheme.text
                            font.weight: Font.DemiBold
                        }
                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            color: AppTheme.textDim
                            font.pixelSize: AppTheme.fontSmall
                            text: {
                                if (!root.recommendation)
                                    return "Scanning…"
                                var b = (root.recommendation.backend || "cpu").toUpperCase()
                                return "Recommended runtime backend: " + b
                                    + "\n" + (root.recommendation.reason || "")
                            }
                        }
                    }
                }
                Label {
                    Layout.fillWidth: true
                    text: "Next you’ll install a llama.cpp runtime. Models can be downloaded or imported whenever you’re ready."
                    wrapMode: Text.WordWrap
                    color: AppTheme.textFaint
                    font.pixelSize: AppTheme.fontSmall
                }
                Item { Layout.fillHeight: true }
            }

            // Step 1 — Runtime
            ColumnLayout {
                spacing: 12
                Label {
                    Layout.fillWidth: true
                    text: root.hasRuntime
                        ? "A llama.cpp runtime is installed. You’re ready to download a model and chat."
                        : "Install an official llama.cpp build matched to this machine. Models cannot load without one."
                    wrapMode: Text.WordWrap
                    color: AppTheme.textDim
                }
                AppButton {
                    text: root.hasRuntime ? "Runtime installed"
                        : (root.installing || root.checking ? "Working…" : "Install recommended build")
                    enabled: !root.hasRuntime && !root.installing && !root.checking
                    primary: !root.hasRuntime
                    onClicked: root.installRecommended()
                }
                AppProgressBar {
                    Layout.fillWidth: true
                    visible: root.installing && root.installProgress >= 0
                    value: Math.max(0, root.installProgress)
                    from: 0; to: 1
                }
                Label {
                    Layout.fillWidth: true
                    visible: root.statusText !== ""
                    text: root.statusText
                    color: AppTheme.info
                    font.pixelSize: AppTheme.fontSmall
                    wrapMode: Text.WordWrap
                }
                Label {
                    Layout.fillWidth: true
                    visible: root.errorText !== ""
                    text: root.errorText
                    color: AppTheme.danger
                    font.pixelSize: AppTheme.fontSmall
                    wrapMode: Text.WordWrap
                }
                AppButton {
                    flat: true
                    text: "Advanced options on Runtimes page…"
                    onClicked: {
                        root.completeAndClose()
                        root.goRuntimes()
                    }
                }
                Item { Layout.fillHeight: true }
            }

            // Step 2 — Continue the first-model workflow without forcing it.
            ColumnLayout {
                spacing: AppTheme.gap
                Label {
                    Layout.fillWidth: true
                    text: "Your runtime is ready. The next step is choosing a model to run locally."
                    color: AppTheme.textDim
                    wrapMode: Text.WordWrap
                }
                SectionCard {
                    Layout.fillWidth: true
                    title: "Find a model"
                    subtitle: "Browse compatible GGUF models and start with a quantization that fits your hardware."
                    AppButton {
                        text: "Browse models"
                        primary: true
                        onClicked: {
                            root.completeAndClose()
                            root.goModels()
                        }
                    }
                }
                SectionCard {
                    Layout.fillWidth: true
                    title: "I already have models"
                    subtitle: "Go straight to chat. You can import a GGUF or select a local model whenever you are ready."
                    AppButton {
                        text: "Open chat"
                        onClicked: {
                            root.completeAndClose()
                            root.goChat()
                        }
                    }
                }
                Item { Layout.fillHeight: true }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 56
            color: AppTheme.bgAlt
            border.color: AppTheme.border
            RowLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8
                AppButton {
                    flat: true
                    text: "Skip setup"
                    onClicked: root.skip()
                }
                Item { Layout.fillWidth: true }
                AppButton {
                    flat: true
                    text: "Back"
                    enabled: root.step > 0
                    onClicked: root.step = Math.max(0, root.step - 1)
                }
                AppButton {
                    text: root.step >= root.stepCount - 1 ? "Finish" : "Next"
                    primary: true
                    enabled: !(root.step === 1 && !root.hasRuntime && root.installing)
                    onClicked: {
                        if (root.step < root.stepCount - 1) {
                            root.errorText = ""
                            root.step++
                            return
                        }
                        if (!root.hasRuntime) {
                            root.errorText = "Install a runtime first, or use Skip setup."
                            return
                        }
                        root.completeAndClose()
                        root.goChat()
                    }
                }
            }
        }
    }
}
