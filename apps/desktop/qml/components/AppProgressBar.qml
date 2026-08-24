import QtQuick
import QtQuick.Controls
import ".."

ProgressBar {
    id: root

    implicitHeight: 8
    from: 0
    to: 1

    // Pixel width is layout * ratio. Never animate that product: a one-tick
    // parent.width of 0 (relayout) or a remounted delegate would tween the
    // fill back to empty, then snap forward again.
    property real fillRatio: 0

    onVisualPositionChanged: root.syncFill()
    Component.onCompleted: root.syncFill()

    function syncFill() {
        var p = root.visualPosition
        if (!(p > 0))
            p = 0
        else if (p > 1)
            p = 1
        if (p <= 0.001 && root.fillRatio > 0.001) {
            zeroGuard.restart()
            return
        }
        zeroGuard.stop()
        root.fillRatio = p
    }

    Timer {
        id: zeroGuard
        interval: 50
        repeat: false
        onTriggered: {
            if (root.visualPosition <= 0.001)
                root.fillRatio = 0
        }
    }

    background: Rectangle {
        implicitHeight: 8
        radius: 4
        color: AppTheme.surfaceHi
        border.color: AppTheme.border
        border.width: 1
    }

    contentItem: Item {
        id: track
        implicitHeight: 8
        clip: true
        property real lastWidth: 0
        onWidthChanged: if (width > 1) lastWidth = width

        Rectangle {
            height: parent.height
            radius: 4
            color: AppTheme.accent
            width: (track.width > 1 ? track.width : track.lastWidth) * root.fillRatio
        }
    }
}
