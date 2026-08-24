import QtQuick
import QtQuick.Controls
import ".."

// Label + explanation + control slot, used by settings forms. Every control
// gets a label, a short explanation and a tooltip with the llama.cpp flag.
Column {
    id: root
    property string label: ""
    property string hint: ""        // one-line explanation
    property string argName: ""     // e.g. --ctx-size, shown in tooltip
    property bool supported: true   // greyed when the runtime lacks support
    default property alias content: holder.data

    spacing: 4
    opacity: supported ? 1.0 : 0.55
    Behavior on opacity { NumberAnimation { duration: AppTheme.motion; easing.type: Easing.OutCubic } }

    Row {
        spacing: 6
        Label {
            text: root.label
            color: AppTheme.text
            font.pixelSize: AppTheme.fontBody
            font.weight: Font.Medium
        }
        Label {
            visible: root.argName !== ""
            text: root.argName + (root.supported ? "" : "  (unsupported by this runtime)")
            color: root.supported ? AppTheme.textFaint : AppTheme.warning
            font.pixelSize: AppTheme.fontSmall
            anchors.verticalCenter: parent.verticalCenter
        }
    }
    Label {
        visible: root.hint !== ""
        text: root.hint
        color: AppTheme.textDim
        font.pixelSize: AppTheme.fontSmall
        wrapMode: Text.WordWrap
        width: root.width
    }
    Item {
        id: holder
        width: root.width
        implicitHeight: childrenRect.height
    }
}
