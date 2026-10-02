import QtQuick
import "../.."
import "../../js/graphModel.js" as GM

// A node socket: type-coloured, filled when wired, glows while a compatible
// wire is being dragged. Pressing starts (or picks up) a wire; the press
// keeps the pointer, so the wire follows it across the whole canvas.
Item {
    id: sk

    property var host
    property string nodeId: ""
    property color cardColor: AppTheme.surface
    property var port
    property int index: 0
    property bool isOutput: false
    property real nodeWidth: 232

    readonly property bool wired: host.isConnected(nodeId, port.name, isOutput)
    readonly property bool hot: host.socketHot(nodeId, port.name, isOutput)
    readonly property color tint: GM.typeColor(GM.types(port)[0])

    x: isOutput ? nodeWidth - 7 : -7
    y: GM.FIRST_ROW + GM.ROW * index - 7
    width: 14
    height: 14

    Rectangle {
        anchors.centerIn: parent
        width: sk.hot ? 22 : 0
        height: width
        radius: width / 2
        color: Qt.alpha(sk.tint, 0.3)
        Behavior on width { NumberAnimation { duration: AppTheme.motionFast } }
    }
    Rectangle {
        anchors.fill: parent
        radius: 7
        color: sk.wired || sk.hot ? sk.tint : sk.cardColor
        border.width: 2
        border.color: sk.wired || sk.hot ? sk.cardColor : sk.tint
    }
    MouseArea {
        anchors.fill: parent
        anchors.margins: -7
        cursorShape: Qt.CrossCursor
        preventStealing: true
        onPressed: function(m) { sk.host.beginWire(sk.nodeId, sk.port.name, sk.isOutput, mapToItem(sk.host.world, m.x, m.y)) }
        onPositionChanged: function(m) { if (pressed) sk.host.moveWire(mapToItem(sk.host.world, m.x, m.y)) }
        onReleased: function(m) { sk.host.endWire(mapToItem(sk.host.world, m.x, m.y)) }
    }
}
