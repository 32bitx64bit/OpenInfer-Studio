import QtQuick
import QtQuick.Controls
import ".."

ComboBox {
    id: root

    implicitHeight: 36
    leftPadding: 12
    rightPadding: 28
    hoverEnabled: true
    font.pixelSize: AppTheme.fontBody
    property string subtitleRole: ""
    property int popupMinWidth: 0

    contentItem: Text {
        leftPadding: 0
        rightPadding: root.indicator.width + 8
        text: root.displayText
        font: root.font
        color: root.enabled ? AppTheme.text : AppTheme.textFaint
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
    }

    background: Rectangle {
        radius: AppTheme.radiusSmall
        color: root.down || root.popup.visible ? AppTheme.surfaceHi
             : root.hovered ? AppTheme.surfaceHover : AppTheme.surface
        border.width: 1
        border.color: root.activeFocus || root.popup.visible ? AppTheme.borderFocus : AppTheme.border
        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
        Behavior on border.color { ColorAnimation { duration: AppTheme.motionFast } }
    }

    indicator: Text {
        x: root.width - width - 10
        y: root.topPadding + (root.availableHeight - height) / 2
        text: root.popup.visible ? "▴" : "▾"
        color: AppTheme.textDim
        font.pixelSize: AppTheme.fontSmall
    }

    popup: Popup {
        y: root.height + 2
        width: Math.max(root.width, root.popupMinWidth)
        padding: 4
        implicitHeight: Math.min(contentItem.implicitHeight + padding * 2, 360)
        margins: 0
        enter: Transition {
            NumberAnimation { property: "opacity"; from: 0; to: 1; duration: AppTheme.motionFast; easing.type: Easing.OutCubic }
        }
        exit: Transition {
            NumberAnimation { property: "opacity"; from: 1; to: 0; duration: AppTheme.motionFast; easing.type: Easing.OutCubic }
        }

        contentItem: ListView {
            clip: true
            implicitHeight: contentHeight
            boundsBehavior: Flickable.StopAtBounds
            spacing: 1
            model: root.popup.visible ? root.delegateModel : null
            currentIndex: root.highlightedIndex
            ScrollBar.vertical: ScrollBar {
                policy: parent.contentHeight > parent.height ? ScrollBar.AsNeeded : ScrollBar.AlwaysOff
            }
        }

        background: Rectangle {
            color: AppTheme.bg
            border.color: AppTheme.border
            border.width: 1
            radius: AppTheme.radiusSmall
        }
    }

    delegate: ItemDelegate {
        width: ListView.view ? ListView.view.width : root.width
        implicitHeight: root.subtitleRole !== "" ? 44 : 32
        hoverEnabled: true
        highlighted: root.highlightedIndex === index
        padding: 0
        leftPadding: 14
        rightPadding: 10

        contentItem: Column {
            spacing: 1
            Text {
                width: parent.width
                text: {
                    if (root.textRole && typeof modelData === "object" && modelData
                            && modelData[root.textRole] !== undefined)
                        return modelData[root.textRole]
                    if (root.textRole && model && model[root.textRole] !== undefined)
                        return model[root.textRole]
                    return modelData !== undefined ? String(modelData) : ""
                }
                color: AppTheme.text
                elide: Text.ElideRight
                font.pixelSize: AppTheme.fontBody
                font.weight: root.currentIndex === index ? Font.DemiBold : Font.Normal
            }
            Text {
                width: parent.width
                visible: root.subtitleRole !== "" && text !== ""
                text: {
                    if (!root.subtitleRole || typeof modelData !== "object" || !modelData)
                        return ""
                    return modelData[root.subtitleRole] || ""
                }
                color: AppTheme.textFaint
                elide: Text.ElideRight
                font.pixelSize: AppTheme.fontSmall
            }
        }

        background: Rectangle {
            radius: 4
            color: parent.highlighted ? AppTheme.surfaceHover
                 : root.currentIndex === index ? AppTheme.surfaceSelected
                 : AppTheme.ghost(AppTheme.surfaceHover)
            Rectangle {
                width: 2
                height: parent.height - 10
                radius: 1
                x: 4
                anchors.verticalCenter: parent.verticalCenter
                color: AppTheme.accent
                visible: root.currentIndex === index
            }
        }
    }
}
