import QtQuick
import QtQuick.Controls
import ".."

Switch {
    id: root

    hoverEnabled: true
    implicitHeight: 28
    implicitWidth: text === "" ? indicator.width : implicitContentWidth + leftPadding + rightPadding

    indicator: Rectangle {
        implicitWidth: 44
        implicitHeight: 24
        x: root.text === "" ? (parent.width - width) / 2 : root.leftPadding
        y: parent.height / 2 - height / 2
        radius: 12
        color: root.checked ? AppTheme.accent : AppTheme.surfaceHi
        border.color: root.checked ? AppTheme.accent : AppTheme.border
        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
        Behavior on border.color { ColorAnimation { duration: AppTheme.motionFast } }

        Rectangle {
            x: root.checked ? parent.width - width - 3 : 3
            anchors.verticalCenter: parent.verticalCenter
            width: 18
            height: 18
            radius: 9
            color: root.checked ? AppTheme.onAccent : AppTheme.surface
            border.color: root.checked ? AppTheme.accentHi : AppTheme.border
            Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
            Behavior on border.color { ColorAnimation { duration: AppTheme.motionFast } }

            Behavior on x { NumberAnimation { duration: AppTheme.motionFast; easing.type: Easing.OutCubic } }
        }
    }

    contentItem: Text {
        visible: root.text !== ""
        text: root.text
        font: root.font
        color: root.enabled ? AppTheme.text : AppTheme.textFaint
        verticalAlignment: Text.AlignVCenter
        leftPadding: root.indicator.width + root.spacing
    }
}
