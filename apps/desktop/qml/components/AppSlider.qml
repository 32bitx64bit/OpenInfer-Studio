import QtQuick
import QtQuick.Controls
import ".."

Slider {
    id: root

    hoverEnabled: true
    implicitHeight: 28

    background: Rectangle {
        x: root.leftPadding
        y: root.topPadding + root.availableHeight / 2 - height / 2
        implicitWidth: 200
        implicitHeight: 6
        width: root.availableWidth
        height: implicitHeight
        radius: 3
        color: AppTheme.surfaceHi
        border.color: AppTheme.border

        Rectangle {
            width: root.visualPosition * parent.width
            height: parent.height
            radius: 3
            color: AppTheme.accent
        }
    }

    handle: Rectangle {
        x: root.leftPadding + root.visualPosition * (root.availableWidth - width)
        y: root.topPadding + root.availableHeight / 2 - height / 2
        implicitWidth: 16
        implicitHeight: 16
        radius: 8
        color: root.pressed ? AppTheme.accentHi : (root.hovered ? AppTheme.accentHi : AppTheme.accent)
        border.color: AppTheme.onAccent
        border.width: 1
        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
    }
}
