import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".."

ColumnLayout {
    id: root
    property string icon: "◇"
    property string title: "Nothing here yet"
    property string hint: ""
    property string actionText: ""
    signal actionTriggered()

    spacing: 8
    opacity: 0
    transform: Translate {
        id: appearShift
        y: 8
        Behavior on y { NumberAnimation { duration: AppTheme.motion; easing.type: Easing.OutCubic } }
    }

    function playAppear() {
        opacity = 0
        appearShift.y = 8
        Qt.callLater(function() {
            if (!root.visible)
                return
            root.opacity = 1
            appearShift.y = 0
        })
    }

    Component.onCompleted: if (visible) playAppear()
    onVisibleChanged: if (visible) playAppear()

    Behavior on opacity { NumberAnimation { duration: AppTheme.motion; easing.type: Easing.OutCubic } }

    Label {
        text: root.icon
        font.pixelSize: 40
        color: AppTheme.textFaint
        Layout.alignment: Qt.AlignHCenter
    }
    Label {
        text: root.title
        font.pixelSize: AppTheme.fontTitle
        color: AppTheme.textDim
        Layout.alignment: Qt.AlignHCenter
    }
    Label {
        text: root.hint
        font.pixelSize: AppTheme.fontSmall
        color: AppTheme.textFaint
        wrapMode: Text.WordWrap
        horizontalAlignment: Text.AlignHCenter
        Layout.alignment: Qt.AlignHCenter
        Layout.maximumWidth: 360
        Layout.fillWidth: true
    }
    AppButton {
        visible: root.actionText !== ""
        text: root.actionText
        primary: true
        Layout.alignment: Qt.AlignHCenter
        onClicked: root.actionTriggered()
    }
}
