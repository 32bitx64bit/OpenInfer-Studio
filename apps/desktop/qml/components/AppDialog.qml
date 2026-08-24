import QtQuick
import QtQuick.Controls
import ".."

// Shared dialog chrome so overlays match the app surfaces.
Dialog {
    id: root

    modal: true
    anchors.centerIn: parent
    padding: AppTheme.pad
    topPadding: AppTheme.pad
    bottomPadding: AppTheme.pad
    transformOrigin: Item.Center
    enter: DialogEnter {}
    exit: DialogExit {}
    Overlay.modal: Rectangle { color: AppTheme.overlay }

    header: Label {
        visible: root.title !== ""
        text: root.title
        color: AppTheme.text
        font.pixelSize: AppTheme.fontTitle
        font.weight: Font.DemiBold
        leftPadding: AppTheme.pad
        rightPadding: AppTheme.pad
        topPadding: AppTheme.pad
        bottomPadding: AppTheme.gapTight
    }

    background: Rectangle {
        color: AppTheme.bg
        border.color: AppTheme.border
        radius: AppTheme.radius
    }

    Component.onCompleted: AppTheme.applyPalette(root)
}
