import QtQuick
import ".."

Transition {
    ParallelAnimation {
        NumberAnimation { property: "opacity"; from: 0; to: 1; duration: AppTheme.motionSlow; easing.type: Easing.OutCubic }
        NumberAnimation { property: "scale"; from: 0.96; to: 1; duration: AppTheme.motionSlow; easing.type: Easing.OutCubic }
    }
}
