import QtQuick
import QtQuick.Controls
import "../.."

// Compact numeric parameter for a graph node (int, float, seed). Commits on
// Enter or focus loss, clamped to the spec's min/max; never fires per key.
Rectangle {
    id: root

    property var spec
    property var value
    readonly property bool integer: !!spec && (spec.kind === "int" || spec.kind === "seed")
    readonly property string label: spec ? (spec.label || spec.name.replace(/^output_/, "").replace(/_/g, " ")) : ""
    signal edited(var v)

    function fmt(v) {
        if (v === undefined || v === null || v === "") return ""
        var n = Number(v)
        if (isNaN(n)) return String(v)
        return integer ? String(Math.round(n)) : String(parseFloat(n.toFixed(4)))
    }
    function syncFrom(v) {
        root.value = v
        if (!input.activeFocus) input.text = fmt(v)
    }
    function commit() {
        var n = Number(input.text.replace(",", "."))
        if (input.text.trim() === "" || isNaN(n)) { input.text = fmt(root.value); return }
        if (integer) n = Math.round(n)
        if (spec.kind === "seed") n = Math.max(-1, n)
        if (spec.min !== undefined && spec.min !== null) n = Math.max(spec.min, n)
        if (spec.max !== undefined && spec.max !== null) n = Math.min(spec.max, n)
        input.text = fmt(n)
        if (n !== root.value) { root.value = n; root.edited(n) }
    }

    implicitHeight: 26
    radius: AppTheme.radiusSmall
    color: AppTheme.bg
    border.width: 1
    border.color: input.activeFocus ? AppTheme.borderFocus : AppTheme.border

    Text {
        id: caption
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.verticalCenter: parent.verticalCenter
        text: root.label
        color: AppTheme.textDim
        font.pixelSize: AppTheme.fontSmall
    }
    TextInput {
        id: input
        anchors.left: caption.right
        anchors.leftMargin: 6
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.verticalCenter: parent.verticalCenter
        horizontalAlignment: Text.AlignRight
        color: AppTheme.text
        font.pixelSize: AppTheme.fontSmall + 1
        selectByMouse: true
        clip: true
        inputMethodHints: Qt.ImhFormattedNumbersOnly
        onEditingFinished: root.commit()
    }
    // A hint for the one value that is not a plain number.
    Text {
        visible: root.spec && root.spec.kind === "seed" && Number(root.value) < 0 && !input.activeFocus
        anchors.left: caption.right
        anchors.leftMargin: 6
        anchors.verticalCenter: parent.verticalCenter
        text: "(random)"
        color: AppTheme.textFaint
        font.pixelSize: AppTheme.fontSmall
    }
    Component.onCompleted: input.text = fmt(value)
    onValueChanged: if (!input.activeFocus) input.text = fmt(value)
}
