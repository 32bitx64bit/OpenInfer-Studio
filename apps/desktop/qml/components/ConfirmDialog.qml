import QtQuick
import QtQuick.Controls
import ".."

// Destructive-action confirmation dialog.
Dialog {
    id: root
    property string confirmText: "Delete"
    property string message: ""
    property var paths: []      // exact affected paths, when relevant
    property bool destructive: true
    signal confirmed()

    title: "Confirm"
    modal: true
    anchors.centerIn: parent
    width: Math.min(480, parent ? parent.width - 64 : 480)
    standardButtons: Dialog.NoButton
    padding: AppTheme.pad
    transformOrigin: Item.Center
    enter: DialogEnter {}
    exit: DialogExit {}
    Overlay.modal: Rectangle { color: AppTheme.overlay }

    background: Rectangle {
        color: AppTheme.bg
        border.color: AppTheme.border
        radius: AppTheme.radius
    }
    Component.onCompleted: AppTheme.applyPalette(root)

    header: Label {
        text: root.title
        color: AppTheme.text
        font.pixelSize: AppTheme.fontTitle
        font.weight: Font.DemiBold
        leftPadding: AppTheme.pad
        rightPadding: AppTheme.pad
        topPadding: AppTheme.pad
    }

    Column {
        width: parent.width
        spacing: AppTheme.gap
        Label { text: root.message; wrapMode: Text.WordWrap; width: parent.width; color: AppTheme.text }
        Column {
            visible: root.paths.length > 0
            width: parent.width
            spacing: 2
            Repeater {
                model: root.paths
                Label {
                    text: modelData
                    font.pixelSize: AppTheme.fontSmall
                    color: AppTheme.textDim
                    elide: Text.ElideMiddle
                    width: parent.width
                }
            }
        }
        Row {
            spacing: 8
            anchors.right: parent.right
            AppButton { text: "Cancel"; onClicked: root.close() }
            AppButton {
                text: root.confirmText
                primary: !root.destructive
                danger: root.destructive
                onClicked: { root.confirmed(); root.close() }
            }
        }
    }
}
