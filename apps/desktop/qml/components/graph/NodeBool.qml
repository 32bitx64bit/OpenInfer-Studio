import QtQuick
import QtQuick.Controls
import "../.."

Item {
    id: root
    property var spec
    property var value
    signal edited(var v)
    function syncFrom(v) { root.value = v }
    implicitHeight: 26
    Text {
        anchors.left: parent.left
        anchors.leftMargin: 2
        anchors.verticalCenter: parent.verticalCenter
        text: root.spec ? (root.spec.label || root.spec.name.replace(/_/g, " ")) : ""
        color: AppTheme.textDim
        font.pixelSize: AppTheme.fontSmall
    }
    Switch {
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        checked: !!root.value
        onToggled: { root.value = checked; root.edited(checked) }
    }
}
