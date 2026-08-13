import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".."
import "../components"

Item {
    id: page
    property var api
    property var events
    property string modelId: ""

    signal back()
    signal quantizeRequested()

    property var instance: null
    property var activity: null
    property var consoleLines: []
    property bool follow: true

    function enter() {
        page.consoleLines = []
        page.activity = null
        page.instance = null
        page.reloadStatus()
        page.reloadLogs()
    }

    function reloadStatus() {
        if (page.modelId === "") return
        api.get("/api/v1/instances", function(st, data) {
            if (st !== 200) return
            var list = (data && data.instances) || []
            page.instance = null
            for (var i = 0; i < list.length; i++)
                if (list[i].model_id === page.modelId) page.instance = list[i]
            if (!page.instance) page.activity = null
        })
        api.get("/api/v1/models/" + page.modelId + "/activity", function(st, data) {
            if (st === 200) page.activity = (data && data.activity) || null
        })
    }

    function reloadLogs() {
        if (page.modelId === "") return
        api.get("/api/v1/models/" + page.modelId + "/logs", function(st, data) {
            if (st !== 200 || !data) return
            var lines = String(data.log || "").split("\n")
            if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop()
            page.consoleLines = lines.slice(-2000)
        })
    }

    function reload() {
        page.reloadStatus()
        page.reloadLogs()
    }

    Connections {
        target: page.events
        function onEventReceived(name, payload) {
            if (!payload || payload.model_id !== page.modelId) return
            if (name === "instance.activity") {
                page.activity = payload
            } else if (name === "instance.log") {
                var arr = page.consoleLines.concat(payload.lines || [])
                if (arr.length > 2000) arr = arr.slice(arr.length - 2000)
                page.consoleLines = arr
            } else if (name === "instance.state_changed" || name === "instance.updated") {
                page.reloadStatus()
            }
        }
    }

    Timer {
        interval: 2000
        repeat: true
        running: page.visible && page.modelId !== ""
        onTriggered: page.reloadStatus()
    }

    function uptime() {
        if (!page.instance || !page.instance.started_at) return "—"
        var s = Math.max(0, (Date.now() - Date.parse(page.instance.started_at)) / 1000)
        if (s < 60) return Math.floor(s) + "s"
        if (s < 3600) return Math.floor(s / 60) + "m " + Math.floor(s % 60) + "s"
        return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m"
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: AppTheme.pad
        spacing: AppTheme.gap

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            AppButton { text: "← Back"; flat: true; onClicked: page.back() }
            StatusDot { state: page.instance ? page.instance.state : "" }
            Label {
                text: page.instance ? (page.instance.model_alias || page.instance.model_id) : page.modelId
                color: AppTheme.text
                font.pixelSize: AppTheme.fontTitle
                font.weight: Font.DemiBold
                elide: Text.ElideRight
                Layout.fillWidth: true
            }
            AppButton { text: "Quantize…"; flat: true; onClicked: page.quantizeRequested() }
            Label {
                visible: page.activity && page.activity.busy
                text: page.activity
                    ? "Processing " + page.activity.active_requests + " request"
                      + (page.activity.active_requests === 1 ? "" : "s")
                      + " · " + page.activity.decoded_total + " tok · "
                      + page.activity.tokens_per_second.toFixed(1) + " tok/s"
                    : ""
                color: AppTheme.success
                font.pixelSize: AppTheme.fontBody
            }
        }

        Card {
            Layout.fillWidth: true
            implicitHeight: detailsGrid.implicitHeight + 20
            GridLayout {
                id: detailsGrid
                anchors.fill: parent
                anchors.margins: 10
                columns: 4
                columnSpacing: 24
                rowSpacing: 6

                Repeater {
                    model: [
                        { "k": "State",    "v": page.instance ? page.instance.state : "unloaded" },
                        { "k": "PID",      "v": page.instance && page.instance.pid > 0 ? page.instance.pid : "—" },
                        { "k": "Port",     "v": page.instance && page.instance.port > 0 ? page.instance.port : "—" },
                        { "k": "Runtime",  "v": page.instance ? page.instance.runtime_id : "—" },
                        { "k": "Backend",  "v": page.instance ? page.instance.backend : "—" },
                        { "k": "Uptime",   "v": page.uptime() },
                        { "k": "Requests", "v": page.instance ? page.instance.requests : 0 },
                        { "k": "Speed",    "v": page.activity && page.activity.busy
                                                ? page.activity.tokens_per_second.toFixed(1) + " tok/s" : "idle" }
                    ]
                    delegate: ColumnLayout {
                        spacing: 0
                        Label { text: modelData.k; color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall }
                        Label {
                            text: String(modelData.v)
                            color: modelData.k === "State" ? AppTheme.stateColor(String(modelData.v)) : AppTheme.text
                            font.pixelSize: AppTheme.fontBody
                            elide: Text.ElideRight
                            Layout.maximumWidth: 220
                        }
                    }
                }
            }
        }

        Card {
            Layout.fillWidth: true
            implicitHeight: reqCol.implicitHeight + 20
            visible: page.instance !== null
            ColumnLayout {
                id: reqCol
                anchors.fill: parent
                anchors.margins: 10
                spacing: 6
                Label {
                    text: "Active requests"
                    color: AppTheme.text
                    font.weight: Font.DemiBold
                }
                Label {
                    visible: !page.activity || page.activity.active_requests === 0
                    text: "No requests in flight."
                    color: AppTheme.textFaint
                    font.pixelSize: AppTheme.fontSmall
                }
                Repeater {
                    model: page.activity
                        ? page.activity.slots.filter(function(s) { return s.processing }) : []
                    delegate: RowLayout {
                        Layout.fillWidth: true
                        spacing: 12
                        Rectangle {
                            width: 8; height: 8; radius: 4
                            color: AppTheme.success
                            Layout.alignment: Qt.AlignVCenter
                        }
                        Label {
                            text: "slot " + modelData.id + (modelData.task_id ? " · task " + modelData.task_id : "")
                            color: AppTheme.textDim
                            font.pixelSize: AppTheme.fontSmall
                            font.family: "monospace"
                        }
                        Label {
                            text: "prompt: " + modelData.prompt_tokens + " tok"
                            color: AppTheme.textDim
                            font.pixelSize: AppTheme.fontSmall
                        }
                        Label {
                            text: "generated: " + modelData.decoded + " tok"
                            color: AppTheme.text
                            font.pixelSize: AppTheme.fontSmall
                        }
                        Label {
                            text: modelData.tokens_per_second > 0
                                ? modelData.tokens_per_second.toFixed(1) + " tok/s" : ""
                            color: AppTheme.success
                            font.pixelSize: AppTheme.fontSmall
                        }
                        Item { Layout.fillWidth: true }
                    }
                }
            }
        }

        Card {
            Layout.fillWidth: true
            Layout.fillHeight: true
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 6
                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: "Console"
                        color: AppTheme.text
                        font.weight: Font.DemiBold
                    }
                    Item { Layout.fillWidth: true }
                    AppButton {
                        text: page.follow ? "Following ▾" : "Follow"
                        flat: true
                        checkable: true
                        checked: page.follow
                        onClicked: page.follow = !page.follow
                    }
                    AppButton { text: "Clear"; flat: true; onClicked: page.consoleLines = [] }
                    AppButton { text: "Refresh"; flat: true; onClicked: page.reload() }
                }
                ListView {
                    id: consoleView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: page.consoleLines
                    onCountChanged: if (page.follow) positionViewAtEnd()
                    delegate: Label {
                        width: consoleView.width
                        text: modelData
                        color: AppTheme.textDim
                        font.family: "monospace"
                        font.pixelSize: AppTheme.fontSmall
                        wrapMode: Text.NoWrap
                        elide: Text.ElideRight
                    }
                    ScrollBar.vertical: ScrollBar {}
                }
            }
        }
    }
}
