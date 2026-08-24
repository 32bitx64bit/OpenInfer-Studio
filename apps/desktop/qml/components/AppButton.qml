import QtQuick
import QtQuick.Controls
import ".."

Button {
    id: root

    property bool primary: false
    property bool danger: false
    property string accessibleDescription: ""

    Accessible.name: accessibleDescription !== "" ? accessibleDescription : text
    hoverEnabled: true
    padding: flat ? 8 : 10
    leftPadding: flat ? 10 : 14
    rightPadding: flat ? 10 : 14
    topPadding: flat ? 6 : 8
    bottomPadding: flat ? 6 : 8
    implicitHeight: flat ? 32 : 36
    transformOrigin: Item.Center
    scale: root.down && root.enabled ? AppTheme.motionPressScale : 1

    Behavior on scale { NumberAnimation { duration: AppTheme.motionFast; easing.type: Easing.OutCubic } }

    contentItem: Text {
        text: root.text
        color: {
            if (!root.enabled) return AppTheme.textFaint
            if (root.primary || root.danger) return AppTheme.onAccent
            return root.hovered || root.down ? AppTheme.text : AppTheme.textDim
        }
        font.pixelSize: AppTheme.fontBody
        font.weight: root.primary || root.danger ? Font.DemiBold : Font.Medium
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
    }
    background: Rectangle {
        radius: AppTheme.radiusSmall
        border.width: root.flat || root.primary || root.danger ? 0 : 1
        border.color: root.activeFocus ? AppTheme.borderFocus : AppTheme.border
        color: {
            if (!root.enabled) return root.flat ? AppTheme.ghost(AppTheme.surfaceHover) : AppTheme.surfaceHi
            if (root.danger) return root.down ? Qt.darker(AppTheme.danger, 1.12) : AppTheme.danger
            if (root.primary) return root.down ? AppTheme.accentHi : AppTheme.accent
            if (root.flat)
                return root.down ? AppTheme.surfaceHi
                     : root.hovered ? AppTheme.surfaceHover : AppTheme.ghost(AppTheme.surfaceHover)
            return root.down ? AppTheme.surfaceHi
                 : root.hovered ? AppTheme.surfaceHover : AppTheme.surface
        }
        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
        Behavior on border.color { ColorAnimation { duration: AppTheme.motionFast } }
    }
}
