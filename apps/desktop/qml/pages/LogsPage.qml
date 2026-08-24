import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".."
import "../components"

Item {
    id: page
    property var api
    property var events

    property var files: []
    property string appLogDir: ""
    property string instLogDir: ""
    property string selectedFile: ""
    property string content: ""
    property var live: []             // live log entries
    property bool paused: false
    property string levelFilter: ""
    property string searchText: ""
    property string sourceFilter: ""

    function reload() {
        api.get("/api/v1/logs/files", function(st, data) {
            if (st === 200) {
                page.files = data.files || []
                page.appLogDir = data.app_log_dir || ""
                page.instLogDir = data.instance_log_dir || ""
            }
        })
    }

    function tail(name) {
        page.selectedFile = name
        api.get("/api/v1/logs/tail?name=" + encodeURIComponent(name), function(st, data) {
            if (st === 200) page.content = data.content || ""
        })
    }

    Connections {
        target: page.events
        function onEventReceived(name, payload) {
            if (name !== "log.entry" || page.paused) return
            var arr = page.live
            arr.push({
                "time": (payload.time || "").substring(11, 19),
                "source": payload.source,
                "level": payload.level,
                "message": payload.message
            })
            if (arr.length > 500) arr.shift()
            page.live = arr
        }
    }

    function levelColor(l) {
        switch (l) {
        case "ERROR": return AppTheme.danger
        case "WARN": return AppTheme.warning
        case "DEBUG": case "TRACE": return AppTheme.textFaint
        default: return AppTheme.textDim
        }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // File list
        ColumnLayout {
            Layout.fillHeight: true
            Layout.preferredWidth: 260
            Layout.margins: AppTheme.pad
            spacing: 8
            Label { text: "Log files"; font.pixelSize: AppTheme.fontTitle; font.weight: Font.DemiBold; color: AppTheme.text }
            AppButton {
                text: "Open log directory"
                flat: true
                onClicked: Qt.openUrlExternally("file://" + page.appLogDir)
            }
            ListView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                model: page.files
                add: Transition {
                    NumberAnimation { property: "opacity"; from: 0; to: 1; duration: AppTheme.motion; easing.type: Easing.OutCubic }
                }
                displaced: Transition {
                    NumberAnimation { property: "y"; duration: AppTheme.motion; easing.type: Easing.OutCubic }
                }
                delegate: ItemDelegate {
                    width: ListView.view.width
                    highlighted: page.selectedFile === modelData.name
                    text: modelData.name + "  (" + AppTheme.bytes(modelData.size) + ")"
                    onClicked: page.tail(modelData.name)
                }
            }
        }

        Rectangle { width: 1; Layout.fillHeight: true; color: AppTheme.border }

        // Viewer: file tail or live stream
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: AppTheme.pad
            spacing: 8

            TabBar {
                id: viewTabs
                Layout.fillWidth: true
                background: Rectangle {
                    color: AppTheme.bgAlt
                    radius: AppTheme.radiusSmall
                    border.color: AppTheme.border
                }
                TabButton {
                    text: "File view"
                    width: implicitWidth + 24
                    contentItem: Text {
                        text: parent.text
                        color: parent.checked ? AppTheme.accent : AppTheme.textDim
                        font.weight: parent.checked ? Font.DemiBold : Font.Normal
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
                    }
                    background: Rectangle {
                        color: parent.checked ? AppTheme.surfaceSelected : "transparent"
                        radius: AppTheme.radiusSmall
                        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
                    }
                }
                TabButton {
                    text: "Live stream"
                    width: implicitWidth + 24
                    contentItem: Text {
                        text: parent.text
                        color: parent.checked ? AppTheme.accent : AppTheme.textDim
                        font.weight: parent.checked ? Font.DemiBold : Font.Normal
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
                    }
                    background: Rectangle {
                        color: parent.checked ? AppTheme.surfaceSelected : "transparent"
                        radius: AppTheme.radiusSmall
                        Behavior on color { ColorAnimation { duration: AppTheme.motionFast } }
                    }
                }
            }

            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: viewTabs.currentIndex

                // File tail
                ColumnLayout {
                    spacing: 8
                    RowLayout {
                        AppButton {
                            text: "Refresh"
                            enabled: page.selectedFile !== ""
                            onClicked: page.tail(page.selectedFile)
                        }
                        Label {
                            text: page.selectedFile
                            color: AppTheme.textDim
                        }
                    }
                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        TextEdit {
                            readOnly: true
                            width: parent.width
                            text: page.content || "Select a log file."
                            color: AppTheme.text
                            font.family: "monospace"
                            font.pixelSize: AppTheme.fontSmall
                        }
                    }
                }

                // Live stream
                ColumnLayout {
                    spacing: 8
                    RowLayout {
                        spacing: 8
                        AppTextField {
                            Layout.fillWidth: true
                            placeholderText: "Filter text…"
                            onTextChanged: page.searchText = text
                        }
                        AppComboBox {
                            model: ["", "ERROR", "WARN", "INFO", "DEBUG"]
                            onActivated: function(i) { page.levelFilter = model[i] }
                        }
                        AppButton { text: page.paused ? "Resume" : "Pause"; onClicked: page.paused = !page.paused }
                        AppButton { text: "Clear"; onClicked: page.live = [] }
                    }
                    ListView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        model: page.live.filter(function(e) {
                            if (page.levelFilter !== "" && e.level !== page.levelFilter) return false
                            if (page.searchText !== "" &&
                                e.message.toLowerCase().indexOf(page.searchText.toLowerCase()) < 0 &&
                                e.source.toLowerCase().indexOf(page.searchText.toLowerCase()) < 0) return false
                            return true
                        })
                        delegate: RowLayout {
                            width: ListView.view.width
                            spacing: 10
                            Label { text: modelData.time; color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall }
                            Label { text: modelData.source; color: AppTheme.accent; font.pixelSize: AppTheme.fontSmall; width: 90; elide: Text.ElideRight }
                            Label { text: modelData.level; color: page.levelColor(modelData.level); font.pixelSize: AppTheme.fontSmall; width: 50 }
                            Label { text: modelData.message; color: AppTheme.textDim; font.pixelSize: AppTheme.fontSmall; Layout.fillWidth: true; elide: Text.ElideRight }
                        }
                    }
                }
            }
        }
    }

    Component.onCompleted: reload()
}
