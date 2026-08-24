import QtQuick
import QtQuick.Controls
import ".."

CheckBox {
    id: root

    hoverEnabled: true
    spacing: 8

    indicator: Rectangle {
        implicitWidth: 18
        implicitHeight: 18
        x: root.leftPadding
        y: parent.height / 2 - height / 2
        radius: 4
        color: root.checked ? AppTheme.accent : AppTheme.surface
        border.color: root.checked ? AppTheme.accent
                     : root.hovered ? AppTheme.textFaint : AppTheme.border
        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
        Behavior on border.color { ColorAnimation { duration: AppTheme.motionFast } }

        Text {
            anchors.centerIn: parent
            opacity: root.checked ? 1 : 0
            text: "✓"
            color: AppTheme.onAccent
            font.pixelSize: 12
            font.weight: Font.Bold
            Behavior on opacity { NumberAnimation { duration: AppTheme.motionFast; easing.type: Easing.OutCubic } }
        }
    }

    contentItem: Text {
        text: root.text
        font: root.font
        color: root.enabled ? AppTheme.text : AppTheme.textFaint
        verticalAlignment: Text.AlignVCenter
        leftPadding: root.indicator.width + root.spacing
        wrapMode: Text.WordWrap
    }
}
