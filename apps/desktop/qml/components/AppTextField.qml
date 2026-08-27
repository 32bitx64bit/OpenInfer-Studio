import QtQuick
import QtQuick.Controls
import ".."

TextField {
    id: root

    selectByMouse: true
    color: AppTheme.text
    placeholderTextColor: AppTheme.textFaint
    leftPadding: 12
    rightPadding: 12
    topPadding: 8
    bottomPadding: 8
    implicitWidth: 240
    implicitHeight: 36

    background: Rectangle {
        implicitWidth: 240
        implicitHeight: 36
        radius: AppTheme.radiusSmall
        color: AppTheme.surface
        border.width: 1
        border.color: root.activeFocus ? AppTheme.borderFocus
                     : root.hovered ? AppTheme.textFaint : AppTheme.border
        Behavior on border.color { ColorAnimation { duration: AppTheme.motionFast } }
    }
}
