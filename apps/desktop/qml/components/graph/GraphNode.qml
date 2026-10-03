pragma ComponentBehavior: Bound
import QtQuick
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
    readonly property bool collapsed: host ? root.host.isCollapsed(root.node.id) : !!root.node.collapsed
    property int previewIndex: 0
    property var widgetsByName: ({})

    readonly property var runState: host ? root.host.runStateFor(root.node.id) : null
    readonly property string st: runState ? (runState.state || "") : ""
    readonly property var workProgress: runState ? runState.progress || null : null
    readonly property var shownIssues: host ? root.host.issuesFor(root.node.id) : []
    readonly property int rows: Math.max(GM.list(root.spec.inputs).length, GM.list(root.spec.outputs).length)
    readonly property color catColor: GM.categoryColor(root.spec.category)
    property var visibleParams: {
        if (!host) return []
        root.host.visRev
        return GM.list(root.spec.params).filter(function(ps) { return root.host.paramVisible(node, ps) })
    }
    property real nowMs: Date.now()

    x: { if (host) root.host.layoutRev; return root.node.pos[0] }
    y: { if (host) root.host.layoutRev; return root.node.pos[1] }
    width: GM.nodeWidth(root.node.type)
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

    onHeightChanged: if (host && node) root.host.measureNode(root.node.id, height)
    Component.onCompleted: if (host && node) root.host.measureNode(root.node.id, height)

    // Soft outline behind a selected root.node.
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
        preventStealing: true
        onPressed: function(m) {
            if (m.button === Qt.RightButton) { root.host.openMenu(root.node.id); return }
            root.host.beginMove(root.node.id, m.modifiers, mapToItem(root.host.world, m.x, m.y))
        }
        onPositionChanged: function(m) { if (pressed) root.host.moveNodes(mapToItem(root.host.world, m.x, m.y)) }
        onReleased: root.host.endMove(false)
        onCanceled: root.host.endMove(true)
        onDoubleClicked: root.host.toggleCollapse([root.node.id])
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
                text: root.node.title || root.spec.title
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
                        if (root.runState.cached) return "cached"
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
                id: socketRow
                required property int index
                width: col.width
                height: GM.ROW
                readonly property var inPort: index < GM.list(root.spec.inputs).length ? root.spec.inputs[index] : null
                readonly property var outPort: index < GM.list(root.spec.outputs).length ? root.spec.outputs[index] : null
                Text {
                    visible: socketRow.inPort !== null
                    x: 16
                    anchors.verticalCenter: parent.verticalCenter
                    text: {
                        if (!socketRow.inPort) return ""
                        var t = GM.types(socketRow.inPort)
                        return socketRow.inPort.name + (t.length > 1 ? "  (" + t.join(" or ").toLowerCase() + ")" : "")
                    }
                    color: socketRow.inPort && socketRow.inPort.required ? AppTheme.textDim : AppTheme.textFaint
                    font.pixelSize: AppTheme.fontSmall
                }
                Text {
                    visible: socketRow.outPort !== null
                    anchors.right: parent.right
                    anchors.rightMargin: 16
                    anchors.verticalCenter: parent.verticalCenter
                    text: socketRow.outPort ? socketRow.outPort.name : ""
                    color: AppTheme.textDim
                    font.pixelSize: AppTheme.fontSmall
                }
            }
        }

        // ---- parameters ----
        Item {
            width: parent.width
            visible: !root.collapsed
            height: root.collapsed ? 0 : paramCol.implicitHeight > 0 ? paramCol.implicitHeight + 14 : 4
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
                        required property var modelData
                        width: paramCol.width
                        property var ps: modelData
                        enabled: !root.host.inspectingRun
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
            visible: root.hasFooter && !root.collapsed
            height: visible ? footerCol.implicitHeight + 12 : 0
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
                    visible: !!root.runState && root.runState.seed !== undefined && root.runState.seed !== null
                    text: visible ? "Seed " + root.runState.seed : ""
                    color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall
                }
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
                    objectName: "nodePreview"
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
                        onClicked: root.host.viewImages(root.previewSources, root.previewIndex)
                    }
                }
                AppButton {
                    visible: root.isVideo && root.previewUrl !== ""
                    text: "Open video"
                    flat: true
                    implicitHeight: 28
                    onClicked: Qt.openUrlExternally(root.previewUrl)
                }
                Row {
                    visible: root.previewSources.length > 1
                    spacing: 8
                    AppButton {
                        objectName: "previousPreview"
                        text: "‹"; flat: true; implicitHeight: 26
                        enabled: root.previewIndex > 0
                        onClicked: root.previewIndex--
                    }
                    Text {
                        anchors.verticalCenter: parent.verticalCenter
                        text: (root.previewIndex + 1) + " / " + root.previewSources.length
                        color: AppTheme.textDim; font.pixelSize: AppTheme.fontSmall
                    }
                    AppButton {
                        objectName: "nextPreview"
                        text: "›"; flat: true; implicitHeight: 26
                        enabled: root.previewIndex + 1 < root.previewSources.length
                        onClicked: root.previewIndex++
                    }
                }
                Text {
                    width: parent.width
                    visible: root.node.type === "image.load" && root.previewUrl !== "" && preview.status === Image.Error
                    text: "Input image is unavailable"
                    color: AppTheme.warning; font.pixelSize: AppTheme.fontSmall
                }
                Repeater {
                    model: root.shownIssues
                    delegate: Text {
                        required property var modelData
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
        model: GM.list(root.spec.inputs)
        delegate: Socket {
            required property var modelData
            required index
            host: root.host; nodeId: root.node.id; cardColor: root.color; nodeWidth: root.width
            port: modelData; isOutput: false
        }
    }
    Repeater {
        model: GM.list(root.spec.outputs)
        delegate: Socket {
            required property var modelData
            required index
            host: root.host; nodeId: root.node.id; cardColor: root.color; nodeWidth: root.width
            port: modelData; isOutput: true
        }
    }

    readonly property bool hasFooter: shownIssues.length > 0
        || previewUrl !== "" || (runState !== null && ((runState.message || "") !== "" || runState.seed !== undefined))

    // ---- previews ----
    readonly property var previewSources: {
        if (host) root.host.graphRev
        if (root.node.type === "image.load" || root.node.type === "mask.load") {
            var ps = GM.list(root.spec.params).filter(function(p) { return p.kind === "path" })[0]
            var path = ps && host ? root.host.getParam(node, ps) : ""
            return path ? [GM.localPathToFileUrl(path)] : []
        }
        return runState && runState.file_urls ? runState.file_urls : []
    }
    onPreviewSourcesChanged: previewIndex = Math.max(0, Math.min(previewIndex, previewSources.length - 1))
    readonly property string previewUrl: previewSources[previewIndex] || ""
    readonly property bool isVideo: /\.(webm|avi|mp4)$/i.test(previewUrl)
    readonly property real previewMax: (root.node.type === "image.save" || root.node.type === "video.save") ? 280 : 150

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
        widgetsByName[ps.name] = item
        item.spec = ps
        item.value = root.host.getParam(node, ps)
        if (item.options !== undefined)
            item.options = ps.kind === "model" ? root.host.modelOptions : root.host.paramOptions(ps)
        item.edited.connect(function(v) { root.host.setParam(root.node.id, ps.name, v, ps.kind === "text" || ps.kind === "string") })
        if (ps.kind === "path" && item.browseClicked)
            item.browseClicked.connect(function() { root.host.browse(root.node.id, ps.name) })
    }
    function syncParams() {
        for (var i = 0; i < visibleParams.length; i++) {
            var ps = visibleParams[i], widget = widgetsByName[ps.name]
            if (widget && widget.syncFrom) widget.syncFrom(root.host.getParam(node, ps))
        }
    }
    Connections {
        target: root.host
        function onParamsRevChanged() { root.syncParams() }
        function onModelOptionsChanged() {
            for (var i = 0; i < root.visibleParams.length; i++) {
                var ps = root.visibleParams[i], widget = root.widgetsByName[ps.name]
                if (widget && ps.kind === "model") widget.options = root.host.modelOptions
            }
        }
    }
}
