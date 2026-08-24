import QtQuick
import ".."

// Restrained rounded surface used across pages.
Rectangle {
    id: root
    property bool hoverHighlight: true

    radius: AppTheme.radius
    color: hoverHighlight && cardHover.hovered ? AppTheme.surfaceHover : AppTheme.surface
    border.color: hoverHighlight && cardHover.hovered ? AppTheme.textFaint : AppTheme.border
    border.width: 1

    HoverHandler { id: cardHover }

    Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
    Behavior on border.color { ColorAnimation { duration: AppTheme.motionFast } }
}
