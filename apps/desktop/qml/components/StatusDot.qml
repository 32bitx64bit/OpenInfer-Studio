import QtQuick
import ".."

// Colored status indicator with subtle pulse while in-progress.
Rectangle {
    property string state: ""
    width: 10; height: 10; radius: 5
    color: AppTheme.stateColor(state)

    Behavior on color { ColorAnimation { duration: AppTheme.motion } }

    SequentialAnimation on opacity {
        running: state === "loading" || state === "starting" || state === "active" || state === "running" || state === "queued"
        loops: Animation.Infinite
        NumberAnimation { to: 0.35; duration: AppTheme.motionPulse }
        NumberAnimation { to: 1.0; duration: AppTheme.motionPulse }
    }
}
