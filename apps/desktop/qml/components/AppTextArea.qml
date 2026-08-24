import QtQuick
import QtQuick.Controls
import ".."

TextArea {
    id: root

    selectByMouse: true
    color: AppTheme.text
    placeholderTextColor: AppTheme.textFaint
    wrapMode: TextArea.Wrap
    leftPadding: 12
    rightPadding: 12
    topPadding: 10
    bottomPadding: 10

    background: Rectangle {
        radius: AppTheme.radiusSmall
        color: AppTheme.surface
        border.width: 1
        border.color: root.activeFocus ? AppTheme.borderFocus
                     : root.hovered ? AppTheme.textFaint : AppTheme.border
        Behavior on border.color { ColorAnimation { duration: AppTheme.motionFast } }
    }
}
