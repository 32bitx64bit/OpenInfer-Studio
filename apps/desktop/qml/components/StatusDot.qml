import QtQuick
import ".."

// Colored status indicator with subtle pulse while in-progress.
Rectangle {
    property string state: ""
    width: 10; height: 10; radius: 5
    color: AppTheme.stateColor(state)

    SequentialAnimation on opacity {
        running: state === "loading" || state === "starting" || state === "active" || state === "running" || state === "queued"
        loops: Animation.Infinite
        NumberAnimation { to: 0.35; duration: 700 }
        NumberAnimation { to: 1.0; duration: 700 }
    }
}
