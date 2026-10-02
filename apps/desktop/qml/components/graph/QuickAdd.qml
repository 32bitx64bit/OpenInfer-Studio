import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../.."
import "../../js/graphModel.js" as GM

// Search-to-add popup for graph nodes. Opened by double-click or Tab on the
// canvas, from the palette, or by dropping a wire on empty space (then it is
// filtered to nodes that accept what the wire carries).
Popup {
    id: root

    property var specs: []                 // every node type
    property string heading: "Add a node"
    property var accepts: null             // function(spec) -> bool, or null for all
    signal picked(var spec)

    readonly property var results: {
        var q = field.text
        var out = []
        for (var i = 0; i < specs.length; i++) {
            var s = specs[i]
            if (accepts && !accepts(s)) continue
            if (!GM.matchesQuery(s, q)) continue
            out.push(s)
        }
        out.sort(function(a, b) {
            if (a.available !== b.available) return a.available ? -1 : 1
            var ca = GM.CATEGORY_ORDER.indexOf(a.category), cb = GM.CATEGORY_ORDER.indexOf(b.category)
            return ca !== cb ? ca - cb : a.title.localeCompare(b.title)
        })
        return out
    }

    function openAtItem(item, px, py) {
        var p = item.mapToItem(Overlay.overlay, px, py)
        x = Math.max(8, Math.min(p.x, Overlay.overlay.width - width - 8))
        y = Math.max(8, Math.min(p.y, Overlay.overlay.height - 380))
        field.text = ""
        list.currentIndex = 0
        open()
    }
    function choose(i) {
        var s = results[i]
        if (!s || !s.available) return
        close()
        picked(s)
    }

    parent: Overlay.overlay
    width: 380
    padding: 0
    modal: false
    focus: true
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
    onOpened: field.forceActiveFocus()

    background: Rectangle {
        radius: AppTheme.radius
        color: AppTheme.surface
        border.width: 1
        border.color: AppTheme.borderFocus
    }

    contentItem: ColumnLayout {
        spacing: 0
        Text {
            Layout.margins: 12
            Layout.bottomMargin: 4
            text: root.heading.toUpperCase()
            color: AppTheme.textFaint
            font.pixelSize: AppTheme.fontSmall
            font.weight: Font.DemiBold
            font.letterSpacing: 1
        }
        TextField {
            id: field
            Layout.fillWidth: true
            Layout.leftMargin: 10
            Layout.rightMargin: 10
            Layout.preferredHeight: 34
            placeholderText: "Search nodes…"
            color: AppTheme.text
            placeholderTextColor: AppTheme.textFaint
            font.pixelSize: AppTheme.fontBody
            leftPadding: 10
            background: Rectangle {
                radius: AppTheme.radiusSmall
                color: AppTheme.bg
                border.width: 1
                border.color: AppTheme.borderFocus
            }
            onTextChanged: list.currentIndex = 0
            Keys.onDownPressed: list.currentIndex = Math.min(list.count - 1, list.currentIndex + 1)
            Keys.onUpPressed: list.currentIndex = Math.max(0, list.currentIndex - 1)
            Keys.onReturnPressed: root.choose(list.currentIndex)
            Keys.onEnterPressed: root.choose(list.currentIndex)
        }
        ListView {
            id: list
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(300, Math.max(48, contentHeight))
            Layout.topMargin: 8
            Layout.bottomMargin: 6
            clip: true
            model: root.results
            currentIndex: 0
            boundsBehavior: Flickable.StopAtBounds
            delegate: Rectangle {
                width: ListView.view.width
                height: 44
                color: ListView.isCurrentItem ? AppTheme.surfaceSelected : (hover.hovered ? AppTheme.surfaceHover : "transparent")
                opacity: modelData.available ? 1 : 0.5
                HoverHandler { id: hover }
                Rectangle {
                    x: 14
                    anchors.verticalCenter: parent.verticalCenter
                    width: 8; height: 8; radius: 2
                    color: GM.categoryColor(modelData.category)
                }
                Column {
                    anchors.left: parent.left
                    anchors.leftMargin: 32
                    anchors.right: parent.right
                    anchors.rightMargin: 12
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 1
                    Text {
                        width: parent.width
                        text: modelData.title
                        color: AppTheme.text
                        font.pixelSize: AppTheme.fontBody
                        font.weight: Font.DemiBold
                        elide: Text.ElideRight
                    }
                    Text {
                        width: parent.width
                        text: modelData.available ? (modelData.description || "") : modelData.reason
                        color: modelData.available ? AppTheme.textFaint : AppTheme.warning
                        font.pixelSize: AppTheme.fontSmall
                        elide: Text.ElideRight
                    }
                }
                MouseArea {
                    anchors.fill: parent
                    onClicked: root.choose(index)
                }
            }
            Text {
                visible: list.count === 0
                anchors.centerIn: parent
                text: "No node matches"
                color: AppTheme.textFaint
                font.pixelSize: AppTheme.fontSmall
            }
        }
    }
}
