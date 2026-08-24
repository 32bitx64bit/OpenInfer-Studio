import QtQuick
import QtQuick.Controls
import ".."

TextField {
    id: root

    property string searchLabel: "Search"

    Accessible.name: searchLabel
    leftPadding: 32
    selectByMouse: true
    color: AppTheme.text
    placeholderTextColor: AppTheme.textFaint

    background: Rectangle {
        radius: AppTheme.radiusSmall
        color: AppTheme.surface
        border.width: 1
        border.color: root.activeFocus ? AppTheme.borderFocus : root.hovered ? AppTheme.textFaint : AppTheme.border
        Behavior on border.color { ColorAnimation { duration: AppTheme.motionFast } }
        Text {
            anchors.left: parent.left
            anchors.leftMargin: 10
            anchors.verticalCenter: parent.verticalCenter
            text: "⌕"
            color: AppTheme.textFaint
            font.pixelSize: 16
        }
    }
}
