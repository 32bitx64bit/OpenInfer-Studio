import QtQuick
import QtQuick.Controls
import "../.."

// Single-line text or file-path parameter. Paths get a browse button.
Rectangle {
    id: root

    property var spec
    property var value
    readonly property bool isPath: !!spec && spec.kind === "path"
    readonly property string label: spec ? (spec.label || spec.name.replace(/^output_/, "").replace(/_/g, " ")) : ""
    signal edited(var v)
    signal browseClicked()

    function syncFrom(v) {
        root.value = v
        if (!input.activeFocus) input.text = v === undefined || v === null ? "" : String(v)
    }

    implicitHeight: 26
    radius: AppTheme.radiusSmall
    color: AppTheme.bg
    border.width: 1
    border.color: input.activeFocus ? AppTheme.borderFocus : AppTheme.border

    Text {
        id: caption
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.verticalCenter: parent.verticalCenter
        text: root.label
        color: AppTheme.textDim
        font.pixelSize: AppTheme.fontSmall
    }
    Rectangle {
        id: browse
        visible: root.isPath
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: visible ? 26 : 0
        radius: AppTheme.radiusSmall
        color: browseArea.containsMouse ? AppTheme.surfaceHover : "transparent"
        Text { anchors.centerIn: parent; text: "…"; color: AppTheme.textDim; font.pixelSize: AppTheme.fontBody }
        MouseArea {
            id: browseArea
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: Qt.PointingHandCursor
            onClicked: root.browseClicked()
        }
    }
    TextInput {
        id: input
        anchors.left: caption.right
        anchors.leftMargin: 8
        anchors.right: browse.visible ? browse.left : parent.right
        anchors.rightMargin: 8
        anchors.verticalCenter: parent.verticalCenter
        color: AppTheme.text
        font.pixelSize: AppTheme.fontSmall + 1
        selectByMouse: true
        clip: true
        onEditingFinished: { if (text !== root.value) { root.value = text; root.edited(text) } }
    }
    Text {
        visible: input.text === "" && !input.activeFocus
        anchors.left: input.left
        anchors.right: input.right
        anchors.verticalCenter: parent.verticalCenter
        text: root.isPath ? "choose a file…" : ""
        color: AppTheme.textFaint
        font.pixelSize: AppTheme.fontSmall + 1
    }
    onValueChanged: if (!input.activeFocus) input.text = value === undefined || value === null ? "" : String(value)
}
