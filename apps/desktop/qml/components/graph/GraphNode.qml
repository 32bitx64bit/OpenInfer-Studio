import QtQuick
import QtQuick.Controls
import "../.."
import "../"
import "../../js/graphModel.js" as GM

// One node on the canvas, drawn entirely from its node-type spec: header,
// socket rows, parameter widgets, run status and previews. It knows nothing
// about specific node types; GraphPage (host) owns the graph and all edits.
Rectangle {
    id: root

    property var node          // {id, type, pos, params, title?}
    property var spec          // node-type descriptor from /workflow/node-types
    property var host          // GraphPage
    property bool selected: false

    readonly property var runState: host ? host.runStateFor(node.id) : null
    readonly property string st: runState ? (runState.state || "") : ""
    readonly property var workProgress: runState ? runState.progress || null : null
    readonly property var shownIssues: host ? host.issuesFor(node.id) : []
    readonly property int rows: Math.max(GM.list(spec.inputs).length, GM.list(spec.outputs).length)
    readonly property color catColor: GM.categoryColor(spec.category)
    property var visibleParams: {
        if (!host) return []
        host.visRev
        return GM.list(spec.params).filter(function(ps) { return host.paramVisible(node, ps) })
    }
    property real nowMs: Date.now()

    x: node.pos[0]
    y: node.pos[1]
    width: GM.nodeWidth(node.type)
    height: col.implicitHeight + 6
    radius: AppTheme.radius
    color: AppTheme.surface
    border.width: selected || st === "running" || st === "failed" ? 2 : 1
    border.color: selected ? AppTheme.accentHi
                : st === "failed" ? AppTheme.danger
                : st === "running" ? AppTheme.accent
                : shownIssues.length > 0 ? AppTheme.danger
                : AppTheme.border
    opacity: st === "pending" ? 0.82 : 1
    z: selected ? 5 : 1

    onXChanged: if (body.drag.active) host.nodeMoved(node.id, x, y)
    onYChanged: if (body.drag.active) host.nodeMoved(node.id, x, y)

    // Soft outline behind a selected node.
    Rectangle {
        anchors.fill: parent
        anchors.margins: -4
        z: -1
        radius: root.radius + 4
        color: "transparent"
        border.width: 4
        border.color: Qt.alpha(AppTheme.accentHi, 0.18)
        visible: root.selected
    }

    // Underlay: select, drag and context menu from any area not covered by a control.
    MouseArea {
        id: body
        anchors.fill: parent
        z: -1
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        drag.target: root
        drag.threshold: 4
        onPressed: function(m) {
            host.select(node.id)
            if (m.button === Qt.RightButton) host.openMenu(node.id)
        }
    }

    Column {
        id: col
        width: parent.width

        // ---- header ----
        Rectangle {
            id: header
            width: parent.width
            height: GM.HEADER
            radius: root.radius - 1
            color: AppTheme.surfaceHi
            Rectangle {   // square off the bottom corners
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                height: parent.radius
                color: parent.color
            }
            Rectangle {
                id: catDot
                x: 10
                anchors.verticalCenter: parent.verticalCenter
                width: 8; height: 8; radius: 2
                color: root.catColor
            }
            Text {
                anchors.left: catDot.right
                anchors.leftMargin: 8
                anchors.right: tag.left
                anchors.rightMargin: 6
                anchors.verticalCenter: parent.verticalCenter
                text: node.title || spec.title
                color: AppTheme.text
                font.pixelSize: AppTheme.fontSmall + 1
                font.weight: Font.DemiBold
                elide: Text.ElideRight
            }
            Text {
                id: tag
                anchors.right: parent.right
                anchors.rightMargin: 10
                anchors.verticalCenter: parent.verticalCenter
                visible: text !== ""
                color: root.st === "failed" ? AppTheme.danger
                     : root.st === "running" ? AppTheme.accentHi
                     : root.st === "done" ? AppTheme.success
                     : AppTheme.textFaint
                font.pixelSize: AppTheme.fontSmall
                text: {
                    switch (root.st) {
                    case "running":
                        var s = Math.max(0, Math.round((root.nowMs - (root.runState.startedMs || root.nowMs)) / 1000))
                        return "running " + Math.floor(s / 60) + ":" + (s % 60 < 10 ? "0" : "") + (s % 60)
                    case "done":
                        return root.runState.ms ? "done " + (root.runState.ms / 1000).toFixed(1) + " s" : "done"
                    case "failed": return "failed"
                    case "canceled": return "canceled"
                    case "pending": return "waiting"
                    default: return root.shownIssues.length > 0 ? "needs attention" : ""
                    }
                }
            }
            // Use observed counters. Unknown phases remain indeterminate.
            Rectangle {
                visible: root.st === "running"
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                height: 2
                color: AppTheme.border
                clip: true
                Rectangle {
                    id: runner
                    visible: !root.workProgress || !(root.workProgress.total > 0)
                    width: parent.width * 0.4
                    height: 2
                    color: AppTheme.accentHi
                    SequentialAnimation on x {
                        running: root.st === "running" && runner.visible
                        loops: Animation.Infinite
                        NumberAnimation { from: -runner.width; to: header.width; duration: 1600 }
                    }
                }
                Rectangle {
                    visible: !!root.workProgress && root.workProgress.total > 0
                    width: parent.width * (visible ? Math.max(0, Math.min(1, (root.workProgress.current || 0) / root.workProgress.total)) : 0)
                    height: 2
                    color: AppTheme.accentHi
                }
            }
        }
        Timer {
            running: root.st === "running"
            interval: 500
            repeat: true
            onTriggered: root.nowMs = Date.now()
        }

        Item { width: 1; height: 8 }

        // ---- socket rows (labels; sockets are drawn separately) ----
        Repeater {
            model: root.rows
            delegate: Item {
                width: col.width
                height: GM.ROW
                readonly property var inPort: index < GM.list(spec.inputs).length ? spec.inputs[index] : null
                readonly property var outPort: index < GM.list(spec.outputs).length ? spec.outputs[index] : null
                Text {
                    visible: inPort !== null
                    x: 16
                    anchors.verticalCenter: parent.verticalCenter
                    text: {
                        if (!inPort) return ""
                        var t = GM.types(inPort)
                        return inPort.name + (t.length > 1 ? "  (" + t.join(" or ").toLowerCase() + ")" : "")
                    }
                    color: inPort && inPort.required ? AppTheme.textDim : AppTheme.textFaint
                    font.pixelSize: AppTheme.fontSmall
                }
                Text {
                    visible: outPort !== null
                    anchors.right: parent.right
                    anchors.rightMargin: 16
                    anchors.verticalCenter: parent.verticalCenter
                    text: outPort ? outPort.name : ""
                    color: AppTheme.textDim
                    font.pixelSize: AppTheme.fontSmall
                }
            }
        }

        // ---- parameters ----
        Item {
            width: parent.width
            height: paramCol.implicitHeight > 0 ? paramCol.implicitHeight + 14 : 4
            Column {
                id: paramCol
                x: 10
                y: 8
                width: parent.width - 20
                spacing: 6
                Repeater {
                    id: paramRepeater
                    model: root.visibleParams
                    delegate: Loader {
                        id: loader
                        width: paramCol.width
                        property var ps: modelData
                        source: root.widgetFile(ps)
                        onLoaded: root.bindWidget(item, ps)
                    }
                }
            }
        }

        // ---- footer: results, messages, issues ----
        Item {
            id: footer
            width: parent.width
            // Visibility must not depend on the height of its own children:
            // a hidden item reports its children as hidden, so they would
            // never contribute a height and the footer would stay hidden.
            visible: root.hasFooter
            height: root.hasFooter ? footerCol.implicitHeight + 12 : 0
            Rectangle {
                visible: footerCol.implicitHeight > 0
                anchors.top: parent.top
                width: parent.width
                height: 1
                color: AppTheme.border
            }
            Column {
                id: footerCol
                x: 10
                y: 8
                width: parent.width - 20
                spacing: 6

                Text {
                    width: parent.width
                    visible: text !== ""
                    wrapMode: Text.Wrap
                    font.pixelSize: AppTheme.fontSmall
                    color: root.st === "failed" ? AppTheme.danger : AppTheme.textDim
                    text: root.runState && root.runState.message ? root.runState.message : ""
                }
                Text {
                    width: parent.width
                    visible: root.st === "running" && !!root.workProgress && (root.workProgress.quiet_ms >= 15000 || root.workProgress.server_responding === false)
                    wrapMode: Text.Wrap
                    font.pixelSize: AppTheme.fontSmall
                    color: AppTheme.warning
                    text: {
                        if (!root.workProgress) return ""
                        if (root.workProgress.server_responding === false) return "Server is not responding; checking…"
                        var s = Math.floor(root.workProgress.quiet_ms / 1000)
                        return "Server responding; no new progress for " + Math.floor(s / 60) + ":" + (s % 60 < 10 ? "0" : "") + (s % 60)
                    }
                }
                Image {
                    id: preview
                    visible: root.previewUrl !== "" && !root.isVideo && status === Image.Ready
                    width: parent.width
                    height: visible ? Math.min(root.previewMax, width * implicitHeight / Math.max(1, implicitWidth)) : 0
                    source: root.isVideo ? "" : root.previewUrl
                    sourceSize.width: 640
                    fillMode: Image.PreserveAspectFit
                    asynchronous: true
                    cache: false
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: Qt.openUrlExternally(root.previewUrl)
                    }
                }
                AppButton {
                    visible: root.isVideo && root.previewUrl !== ""
                    text: "Open video"
                    flat: true
                    implicitHeight: 28
                    onClicked: Qt.openUrlExternally(root.previewUrl)
                }
                Repeater {
                    model: root.shownIssues
                    delegate: Text {
                        width: footerCol.width
                        wrapMode: Text.Wrap
                        text: modelData.message
                        color: AppTheme.danger
                        font.pixelSize: AppTheme.fontSmall
                    }
                }
            }
        }
    }

    // ---- sockets ----
    Repeater {
        model: GM.list(spec.inputs)
        delegate: Socket {
            host: root.host; nodeId: root.node.id; cardColor: root.color; nodeWidth: root.width
            port: modelData; index: model.index; isOutput: false
        }
    }
    Repeater {
        model: GM.list(spec.outputs)
        delegate: Socket {
            host: root.host; nodeId: root.node.id; cardColor: root.color; nodeWidth: root.width
            port: modelData; index: model.index; isOutput: true
        }
    }

    readonly property bool hasFooter: shownIssues.length > 0
        || (runState !== null && ((runState.message || "") !== "" || previewUrl !== ""))

    // ---- previews ----
    readonly property string previewUrl: {
        var urls = runState && runState.file_urls ? runState.file_urls : []
        if (urls.length === 0) return ""
        if (node.type === "image.load") return ""
        return urls[0]
    }
    readonly property bool isVideo: /\.(webm|avi|mp4)$/i.test(previewUrl)
    readonly property real previewMax: (node.type === "image.save" || node.type === "video.save") ? 280 : 150

    function widgetFile(ps) {
        switch (ps.kind) {
        case "int": case "float": case "seed": return "NodeNumber.qml"
        case "enum": return "NodeCombo.qml"
        case "text": return "NodeText.qml"
        case "model": return "NodeModelPick.qml"
        case "bool": return "NodeBool.qml"
        default: return "NodeLine.qml"
        }
    }
    function bindWidget(item, ps) {
        item.spec = ps
        item.value = host.getParam(node, ps)
        if (item.options !== undefined)
            item.options = ps.kind === "model" ? host.modelOptions : host.paramOptions(ps)
        item.edited.connect(function(v) { host.setParam(node.id, ps.name, v) })
        if (ps.kind === "path" && item.browseClicked)
            item.browseClicked.connect(function() { host.browse(node.id, ps.name) })
    }
    function syncParams() {
        for (var i = 0; i < paramRepeater.count; i++) {
            var l = paramRepeater.itemAt(i)
            if (l && l.item && l.item.syncFrom) l.item.syncFrom(host.getParam(node, l.ps))
        }
    }
    Connections {
        target: root.host
        function onParamsRevChanged() { root.syncParams() }
        function onModelOptionsChanged() {
            for (var i = 0; i < paramRepeater.count; i++) {
                var l = paramRepeater.itemAt(i)
                if (l && l.item && l.ps.kind === "model") l.item.options = host.modelOptions
            }
        }
    }
}
