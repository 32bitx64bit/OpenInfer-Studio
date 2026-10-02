import QtQuick
import QtQuick.Controls
import "../.."

// Multi-line text parameter (prompts). Every edit is reported so the graph
// and the backend validation stay in step with what is on screen.
Rectangle {
    id: root

    property var spec
    property var value
    property int boxHeight: 96
    property bool syncing: false
    signal edited(var v)

    function syncFrom(v) {
        root.value = v
        if (area.activeFocus) return
        syncing = true
        area.text = v === undefined || v === null ? "" : String(v)
        syncing = false
    }

    implicitHeight: boxHeight
    radius: AppTheme.radiusSmall
    color: AppTheme.bg
    border.width: 1
    border.color: area.activeFocus ? AppTheme.borderFocus : AppTheme.border

    Flickable {
        id: flick
        anchors.fill: parent
        anchors.margins: 1
        clip: true
        contentWidth: width
        contentHeight: area.implicitHeight
        boundsBehavior: Flickable.StopAtBounds
        ScrollBar.vertical: ScrollBar { policy: flick.contentHeight > flick.height ? ScrollBar.AsNeeded : ScrollBar.AlwaysOff }

        TextArea.flickable: TextArea {
            id: area
            wrapMode: TextEdit.Wrap
            color: AppTheme.text
            placeholderText: root.spec ? (root.spec.label || root.spec.name) : ""
            placeholderTextColor: AppTheme.textFaint
            font.pixelSize: AppTheme.fontSmall + 1
            selectByMouse: true
            padding: 8
            background: null
            onTextChanged: if (!root.syncing) { root.value = text; root.edited(text) }
        }
    }
    // The host assigns value just after creation (and on external changes).
    onValueChanged: {
        var v = value === undefined || value === null ? "" : String(value)
        if (area.activeFocus || area.text === v) return
        syncing = true
        area.text = v
        syncing = false
    }
}
