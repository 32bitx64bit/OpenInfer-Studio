import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".."
import "../components"

Item {
    id: page
    property var api
    property var events

    property var downloads: []
    property var liveProgress: ({})   // id → {done, total, speed, eta}

    function reload() {
        api.get("/api/v1/downloads", function(st, data) {
            if (st === 200) page.downloads = AppTheme.keepRows(page.downloads, (data && data.downloads) || [])
        })
    }

    Connections {
        target: page.events
        function onEventReceived(name, payload) {
            if (name === "download.progress") {
                // QML compares var by reference; copy so bindings update.
                var p = {}
                var old = page.liveProgress
                for (var k in old) p[k] = old[k]
                p[payload.id] = payload
                page.liveProgress = p
                // Start (not restart) so a burst of events cannot starve the timer.
                if (!throttle.running) throttle.start()
            } else if (name === "download.state_changed") {
                page.reload()
            }
        }
    }
    Timer { id: throttle; interval: 800; onTriggered: page.reload() }

    // Event-independent fallback: while anything is active, poll every 2 s
    // so progress still advances if the event socket is degraded.
    Timer {
        interval: 2000
        repeat: true
        running: page.downloads.some(function(d) { return d.state === "active" || d.state === "queued" })
        onTriggered: page.reload()
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: AppTheme.pad
        spacing: AppTheme.gap

        PageHeader {
            title: "Activity"
            subtitle: "Track model downloads. Completed models appear in your library automatically."
        }

        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 8
            model: page.downloads
            add: Transition {
                ParallelAnimation {
                    NumberAnimation { property: "opacity"; from: 0; to: 1; duration: AppTheme.motion; easing.type: Easing.OutCubic }
                    NumberAnimation { property: "y"; duration: AppTheme.motion; easing.type: Easing.OutCubic }
                }
            }
            displaced: Transition {
                NumberAnimation { property: "y"; duration: AppTheme.motion; easing.type: Easing.OutCubic }
            }

            EmptyState {
                visible: page.downloads.length === 0
                anchors.centerIn: parent
                icon: "↓"
                title: "No activity"
                hint: "Model downloads appear here."
            }

            delegate: Card {
                width: ListView.view.width
                implicitHeight: dcol.implicitHeight + 20
                ColumnLayout {
                    id: dcol
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 6

                    RowLayout {
                        Layout.fillWidth: true
                        StatusDot { state: modelData.state }
                        Label {
                            text: modelData.label
                            color: AppTheme.text
                            font.weight: Font.DemiBold
                            elide: Text.ElideMiddle
                            Layout.fillWidth: true
                        }
                        Tag { text: modelData.kind; tone: AppTheme.accent }
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

                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: {
                                var live = page.liveProgress[modelData.id]
                                if (live && modelData.state === "active") {
                                    var eta = live.eta_seconds
                                    return AppTheme.bytes(live.speed_bps) + "/s"
                                        + (eta > 0 ? "  ·  ~" + Math.round(eta / 60) + " min left" : "")
                                }
                                return modelData.state
                            }
                            color: AppTheme.textDim
                            font.pixelSize: AppTheme.fontSmall
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            visible: modelData.error !== ""
                            text: modelData.error
                            color: AppTheme.danger
                            font.pixelSize: AppTheme.fontSmall
                            elide: Text.ElideRight
                            Layout.maximumWidth: 420
                        }
                    }

                    // Per-file details
                    Repeater {
                        model: modelData.files
                        delegate: RowLayout {
                            Layout.fillWidth: true
                            Label {
                                text: "  " + modelData.dest_path.split("/").pop()
                                color: AppTheme.textFaint
                                font.pixelSize: AppTheme.fontSmall
                                font.family: "monospace"
                                Layout.fillWidth: true
                                elide: Text.ElideMiddle
                            }
                            Label {
                                text: modelData.state + "  " + AppTheme.bytes(modelData.done_bytes) + "/" + AppTheme.bytes(modelData.total_bytes)
                                color: AppTheme.textFaint
                                font.pixelSize: AppTheme.fontSmall
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        Item { Layout.fillWidth: true }
                        AppButton {
                            visible: modelData.state === "active"
                            text: "Pause"; flat: true
                            onClicked: page.api.post("/api/v1/downloads/" + modelData.id + "/pause", {}, function() { page.reload() })
                        }
                        AppButton {
                            visible: ["paused", "failed", "canceled"].indexOf(modelData.state) >= 0
                            text: modelData.state === "failed" ? "Retry" : "Resume"; flat: true
                            onClicked: page.api.post("/api/v1/downloads/" + modelData.id + "/resume", {}, function() { page.reload() })
                        }
                        AppButton {
                            visible: ["active", "queued", "paused"].indexOf(modelData.state) >= 0
                            text: "Cancel"; flat: true
                            onClicked: page.api.post("/api/v1/downloads/" + modelData.id + "/cancel", {}, function() { page.reload() })
                        }
                        AppButton {
                            text: "Remove"; flat: true
                            onClicked: page.api.del("/api/v1/downloads/" + modelData.id, function() { page.reload() })
                        }
                    }
                }
            }
        }
    }

    Component.onCompleted: reload()
}
