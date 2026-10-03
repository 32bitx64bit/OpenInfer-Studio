import QtQuick
import "../.."
import "../../js/graphModel.js" as GM

Rectangle {
    id: root
    property var host
    property real viewportWidth: 1
    property real viewportHeight: 1
    readonly property var bounds: {
        if (!host) return null
        host.layoutRev; host.graphRev
        return GM.sectionBounds(host.canvasGraph(), host.specs, host.nodeIds, host.nodeSizes)
    }
    readonly property real mapScale: bounds ? Math.min((width - 20) / Math.max(1, bounds.w), (height - 20) / Math.max(1, bounds.h)) : 1
    readonly property real offsetX: bounds ? (width - bounds.w * mapScale) / 2 - bounds.x * mapScale : 0
    readonly property real offsetY: bounds ? (height - bounds.h * mapScale) / 2 - bounds.y * mapScale : 0
    width: 170; height: 110
    z: 250
    color: Qt.alpha(AppTheme.bgAlt, 0.94)
    border.color: AppTheme.border
    radius: 6
    clip: true
    Canvas {
        id: map
        anchors.fill: parent
        Connections {
            target: root.host
            function onLayoutRevChanged() { map.requestPaint() }
            function onGraphRevChanged() { map.requestPaint() }
            function onSelectedIdsChanged() { map.requestPaint() }
            function onPanXChanged() { map.requestPaint() }
            function onPanYChanged() { map.requestPaint() }
            function onZoomChanged() { map.requestPaint() }
        }
        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            if (!root.host || !root.bounds) return
            var host = root.host, s = root.mapScale, ox = root.offsetX, oy = root.offsetY
            host.canvasGraph().nodes.forEach(function(n) {
                var b = GM.nodeRect(n, host.specs, host.nodeSizes)
                ctx.fillStyle = host.isSelected(n.id) ? AppTheme.accentHi : GM.categoryColor((host.specs[n.type] || {}).category)
                ctx.globalAlpha = 0.7
                ctx.fillRect(ox + b.x * s, oy + b.y * s, Math.max(2, b.w * s), Math.max(2, b.h * s))
            })
            ctx.globalAlpha = 1
            ctx.strokeStyle = AppTheme.text
            ctx.lineWidth = 1
            ctx.strokeRect(ox - host.panX / host.zoom * s, oy - host.panY / host.zoom * s,
                           root.viewportWidth / host.zoom * s, root.viewportHeight / host.zoom * s)
        }
    }
    MouseArea {
        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        function navigate(x, y) {
            if (!root.host || !root.bounds) return
            root.host.panX = root.viewportWidth / 2 - (x - root.offsetX) / root.mapScale * root.host.zoom
            root.host.panY = root.viewportHeight / 2 - (y - root.offsetY) / root.mapScale * root.host.zoom
        }
        onPressed: function(m) { navigate(m.x, m.y) }
        onPositionChanged: function(m) { if (pressed) navigate(m.x, m.y) }
        onReleased: root.host.saveViewport()
    }
}
