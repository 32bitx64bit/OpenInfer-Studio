import QtQuick
import QtQuick.Controls
import "../.."
import "../"

// Library model picker. The stored value is {library_id, name}; a model that
// is no longer in the library stays selected and reads as missing.
Rectangle {
    id: root

    property var spec
    property var value
    property var options: []     // [{id, name}]
    signal edited(var v)

    readonly property string currentId: value && value.library_id ? value.library_id : ""
    readonly property var shown: {
        var list = options || []
        for (var i = 0; i < list.length; i++) if (list[i].id === currentId) return list
        if (currentId !== "")
            return list.concat([{ id: currentId, name: ((value && value.name) || currentId) + " (missing)" }])
        return list
    }
    function indexOfId(id) {
        for (var i = 0; i < shown.length; i++) if (shown[i].id === id) return i
        return -1
    }
    function syncFrom(v) { root.value = v }

    implicitHeight: 26
    color: "transparent"

    AppComboBox {
        id: combo
        anchors.fill: parent
        implicitHeight: 26
        implicitWidth: 100
        leftPadding: 8
        font.pixelSize: AppTheme.fontSmall + 1
        textRole: "name"
        displayText: currentIndex >= 0 ? root.shown[currentIndex].name : "Choose a model…"
        model: root.shown
        currentIndex: root.indexOfId(root.currentId)
        onActivated: function(i) {
            var m = root.shown[i]
            var v = { library_id: m.id, name: m.name.replace(/ \(missing\)$/, "") }
            root.value = v
            root.edited(v)
        }
    }
}
