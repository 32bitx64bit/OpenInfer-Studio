import QtQuick
import QtQuick.Controls
import ".."

// Light theme wrapper — keep Qt SpinBox behavior, only restyle the chrome.
SpinBox {
    id: root

    editable: true
    hoverEnabled: true
    implicitHeight: 36
    palette.text: AppTheme.text
    palette.buttonText: AppTheme.textDim
    palette.highlight: AppTheme.accent
    palette.highlightedText: AppTheme.onAccent
    palette.base: AppTheme.surface
    palette.button: AppTheme.surfaceHi
    palette.mid: AppTheme.border

    background: Rectangle {
        radius: AppTheme.radiusSmall
        color: AppTheme.surface
        border.width: 1
        border.color: root.activeFocus ? AppTheme.borderFocus
                     : root.hovered ? AppTheme.textFaint : AppTheme.border
        Behavior on border.color { ColorAnimation { duration: AppTheme.motionFast } }
    }
}
