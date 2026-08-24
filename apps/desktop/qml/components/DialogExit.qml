import QtQuick
import ".."

Transition {
    ParallelAnimation {
        NumberAnimation { property: "opacity"; from: 1; to: 0; duration: AppTheme.motion; easing.type: Easing.OutCubic }
        NumberAnimation { property: "scale"; from: 1; to: 0.96; duration: AppTheme.motion; easing.type: Easing.OutCubic }
    }
}
