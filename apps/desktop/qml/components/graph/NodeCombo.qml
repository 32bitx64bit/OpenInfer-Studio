import QtQuick
import QtQuick.Controls
import "../.."
import "../"

// Compact dropdown parameter for a graph node. options is [{text, value}];
// a current value that is not in the list is kept visible rather than lost.
Rectangle {
    id: root

    property var spec
    property var value
    property var options: []
    readonly property string label: spec ? (spec.label || spec.name.replace(/^output_/, "").replace(/_/g, " ")) : ""
    signal edited(var v)

    readonly property var shown: {
        var list = options || []
        var found = false
        for (var i = 0; i < list.length; i++) if (list[i].value === value) found = true
        if (!found && value !== undefined && value !== null && value !== "")
            list = list.concat([{ text: String(value), value: value }])
        return list
    }
    function indexOfValue(v) {
        for (var i = 0; i < shown.length; i++) if (shown[i].value === v) return i
        return 0
    }
    function syncFrom(v) { root.value = v }

    implicitHeight: 26
    color: "transparent"

    Text {
        id: caption
        anchors.left: parent.left
        anchors.leftMargin: 2
        anchors.verticalCenter: parent.verticalCenter
        width: 66
        text: root.label
        color: AppTheme.textDim
        font.pixelSize: AppTheme.fontSmall
        elide: Text.ElideRight
    }
    AppComboBox {
        id: combo
        anchors.left: caption.right
        anchors.right: parent.right
        height: 26
        implicitHeight: 26
        implicitWidth: 100
        leftPadding: 8
        font.pixelSize: AppTheme.fontSmall + 1
        textRole: "text"
        model: root.shown
        currentIndex: root.indexOfValue(root.value)
        onActivated: function(i) {
            var v = root.shown[i].value
            root.value = v
            root.edited(v)
        }
    }
}
