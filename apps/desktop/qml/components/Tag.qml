import QtQuick
import ".."

// Small pill label for quantizations, backends, states.
Rectangle {
    property alias text: label.text
    property color tone: AppTheme.accent
    implicitWidth: label.implicitWidth + 14
    implicitHeight: 20
    radius: 10
    color: Qt.alpha(tone, 0.16)
    border.color: Qt.alpha(tone, 0.4)

    Behavior on color { ColorAnimation { duration: AppTheme.motion } }
    Behavior on border.color { ColorAnimation { duration: AppTheme.motion } }

    Text {
        id: label
        anchors.centerIn: parent
        font.pixelSize: AppTheme.fontSmall
        color: tone
        Behavior on color { ColorAnimation { duration: AppTheme.motion } }
    }
}
