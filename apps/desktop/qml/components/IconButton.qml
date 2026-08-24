import QtQuick
import QtQuick.Controls
import ".."

ToolButton {
    id: root

    property string iconText: ""
    property string description: ""
    property bool selected: false

    text: iconText
    Accessible.name: description !== "" ? description : iconText
    hoverEnabled: true
    implicitWidth: 36
    implicitHeight: 36
    transformOrigin: Item.Center
    scale: root.down && root.enabled ? AppTheme.motionPressScale : 1

    Behavior on scale { NumberAnimation { duration: AppTheme.motionFast; easing.type: Easing.OutCubic } }

    contentItem: Text {
        text: root.iconText
        color: root.selected ? AppTheme.accent
             : root.hovered || root.down ? AppTheme.text : AppTheme.textDim
        font.pixelSize: 17
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
    }
    background: Rectangle {
        radius: AppTheme.radiusSmall
        color: root.selected ? AppTheme.surfaceSelected
             : root.down ? AppTheme.surfaceHi
             : root.hovered ? AppTheme.surfaceHover : AppTheme.ghost(AppTheme.surfaceHover)
        border.color: root.activeFocus ? AppTheme.borderFocus : AppTheme.ghost(AppTheme.borderFocus)
        border.width: root.activeFocus ? 1 : 0
        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
        Behavior on border.color { ColorAnimation { duration: AppTheme.motionFast } }
    }
    ToolTip.visible: hovered && description !== ""
    ToolTip.text: description
}
