import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".."

// DualStepper: two numeric steppers sharing one combined surface with a
// center divider. Left arrows sit on the outer-left edge, right arrows on
// the outer-right edge. Values are editable text with wheel support.
Rectangle {
    id: root

    property string leftTitle: ""
    property string rightTitle: ""
    property int leftValue: 0
    property int rightValue: 0
    property int leftFrom: 0
    property int leftTo: 100
    property int rightFrom: 0
    property int rightTo: 100
    property int leftStep: 1
    property int rightStep: 1

    implicitHeight: 58
    radius: AppTheme.radiusSmall
    color: AppTheme.surface
    border.width: 1
    border.color: (leftField.activeFocus || rightField.activeFocus) ? AppTheme.borderFocus : AppTheme.border
    Behavior on border.color { ColorAnimation { duration: AppTheme.motionFast } }

    function clampLeft(v) { return Math.max(root.leftFrom, Math.min(root.leftTo, v)) }
    function clampRight(v) { return Math.max(root.rightFrom, Math.min(root.rightTo, v)) }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // Left adjuster: arrows outer-left, readout fills.
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0
            ColumnLayout {
                Layout.fillHeight: true
                Layout.leftMargin: 3
                Layout.topMargin: 5
                Layout.bottomMargin: 5
                spacing: 0
                ToolButton {
                    Layout.preferredWidth: 30
                    Layout.fillHeight: true
                    padding: 2
                    text: "▲"
                    font.pixelSize: 10
                    autoRepeat: true
                    focusPolicy: Qt.NoFocus
                    Accessible.name: "Increase " + root.leftTitle
                    onClicked: root.leftValue = root.clampLeft(root.leftValue + root.leftStep)
                }
                ToolButton {
                    Layout.preferredWidth: 30
                    Layout.fillHeight: true
                    padding: 2
                    text: "▼"
                    font.pixelSize: 10
                    autoRepeat: true
                    focusPolicy: Qt.NoFocus
                    Accessible.name: "Decrease " + root.leftTitle
                    onClicked: root.leftValue = root.clampLeft(root.leftValue - root.leftStep)
                }
            }
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0
                Label {
                    Layout.fillWidth: true
                    Layout.topMargin: 4
                    text: root.leftTitle
                    color: AppTheme.textDim
                    font.pixelSize: AppTheme.fontSmall
                    horizontalAlignment: Text.AlignHCenter
                }
                TextInput {
                    id: leftField
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    text: root.leftValue
                    color: AppTheme.text
                    font.pixelSize: AppTheme.fontBody
                    font.weight: Font.DemiBold
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    selectByMouse: true
                    validator: IntValidator { bottom: root.leftFrom; top: root.leftTo }
                    onEditingFinished: {
                        var v = parseInt(text, 10)
                        root.leftValue = root.clampLeft(isNaN(v) ? root.leftValue : v)
                        text = root.leftValue
                        focus = false
                    }
                    WheelHandler {
                        onWheel: (event) => {
                            if (event.angleDelta.y > 0) root.leftValue = root.clampLeft(root.leftValue + root.leftStep)
                            else if (event.angleDelta.y < 0) root.leftValue = root.clampLeft(root.leftValue - root.leftStep)
                        }
                    }
                }
            }
            Item { Layout.preferredWidth: 6 }
        }

        // Divider: two adjusters, one surface.
        Rectangle {
            Layout.preferredWidth: 1
            Layout.fillHeight: true
            Layout.topMargin: 10
            Layout.bottomMargin: 10
            color: AppTheme.border
        }

        // Right adjuster: readout fills, arrows outer-right.
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0
            Item { Layout.preferredWidth: 6 }
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0
                Label {
                    Layout.fillWidth: true
                    Layout.topMargin: 4
                    text: root.rightTitle
                    color: AppTheme.textDim
                    font.pixelSize: AppTheme.fontSmall
                    horizontalAlignment: Text.AlignHCenter
                }
                TextInput {
                    id: rightField
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    text: root.rightValue
                    color: AppTheme.text
                    font.pixelSize: AppTheme.fontBody
                    font.weight: Font.DemiBold
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    selectByMouse: true
                    validator: IntValidator { bottom: root.rightFrom; top: root.rightTo }
                    onEditingFinished: {
                        var v = parseInt(text, 10)
                        root.rightValue = root.clampRight(isNaN(v) ? root.rightValue : v)
                        text = root.rightValue
                        focus = false
                    }
                    WheelHandler {
                        onWheel: (event) => {
                            if (event.angleDelta.y > 0) root.rightValue = root.clampRight(root.rightValue + root.rightStep)
                            else if (event.angleDelta.y < 0) root.rightValue = root.clampRight(root.rightValue - root.rightStep)
                        }
                    }
                }
            }
            ColumnLayout {
                Layout.fillHeight: true
                Layout.rightMargin: 3
                Layout.topMargin: 5
                Layout.bottomMargin: 5
                spacing: 0
                ToolButton {
                    Layout.preferredWidth: 30
                    Layout.fillHeight: true
                    padding: 2
                    text: "▲"
                    font.pixelSize: 10
                    autoRepeat: true
                    focusPolicy: Qt.NoFocus
                    Accessible.name: "Increase " + root.rightTitle
                    onClicked: root.rightValue = root.clampRight(root.rightValue + root.rightStep)
                }
                ToolButton {
                    Layout.preferredWidth: 30
                    Layout.fillHeight: true
                    padding: 2
                    text: "▼"
                    font.pixelSize: 10
                    autoRepeat: true
                    focusPolicy: Qt.NoFocus
                    Accessible.name: "Decrease " + root.rightTitle
                    onClicked: root.rightValue = root.clampRight(root.rightValue - root.rightStep)
                }
            }
        }
    }
}
