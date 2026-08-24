import QtQuick
import QtQuick.Layouts
import ".."

Card {
    id: root
    hoverHighlight: false

    property string title: ""
    property string subtitle: ""
    default property alias content: contentSlot.data

    implicitHeight: cardColumn.implicitHeight + AppTheme.pad * 2

    ColumnLayout {
        id: cardColumn
        anchors.fill: parent
        anchors.margins: AppTheme.pad
        spacing: AppTheme.gap

        ColumnLayout {
            visible: root.title !== "" || root.subtitle !== ""
            Layout.fillWidth: true
            spacing: 2
            Text {
                text: root.title
                color: AppTheme.text
                font.pixelSize: AppTheme.fontTitle
                font.weight: Font.DemiBold
            }
            Text {
                visible: root.subtitle !== ""
                Layout.fillWidth: true
                text: root.subtitle
                color: AppTheme.textDim
                font.pixelSize: AppTheme.fontBody
                wrapMode: Text.WordWrap
            }
        }

        ColumnLayout {
            id: contentSlot
            Layout.fillWidth: true
            spacing: AppTheme.gapTight
        }
    }
}
