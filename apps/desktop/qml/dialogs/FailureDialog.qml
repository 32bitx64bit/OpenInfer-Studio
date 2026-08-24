import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".."
import "../components"

// Failure diagnostics: plain-language summary + full technical detail,
// with retry / safe-settings / CPU-fallback / copy actions.
Dialog {
    id: root
    property var api
    property string modelId: ""
    property var report: null

    signal retry()
    signal retrySafe()
    signal retryCpu()

    title: "Model failed to load"
    modal: true
    width: Math.min(720, parent ? parent.width - 64 : 720)
    height: Math.min(620, parent ? parent.height - 64 : 620)
    anchors.centerIn: parent
    padding: AppTheme.pad
    transformOrigin: Item.Center
    enter: DialogEnter {}
    exit: DialogExit {}
    Overlay.modal: Rectangle { color: AppTheme.overlay }

    background: Rectangle {
        color: AppTheme.bg
        border.color: AppTheme.border
        radius: AppTheme.radius
    }
    Component.onCompleted: AppTheme.applyPalette(root)

    function openFor(mid) {
        modelId = mid
        api.get("/api/v1/models/" + mid + "/diagnostics", function(st, data) {
            if (st === 200) root.report = data
        })
        open()
    }

    contentItem: ColumnLayout {
        spacing: AppTheme.gap
        visible: root.report !== null

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            Tag {
                text: root.report && root.report.classification ? root.report.classification.class : "unknown"
                tone: AppTheme.danger
            }
            Label {
                Layout.fillWidth: true
                text: root.report && root.report.classification ? root.report.classification.summary : "Loading…"
                color: AppTheme.text
                font.weight: Font.DemiBold
                wrapMode: Text.WordWrap
            }
        }

        Label {
            Layout.fillWidth: true
            visible: text !== ""
            text: root.report && root.report.classification ? "Suggested: " + root.report.classification.suggestion : ""
            color: AppTheme.textDim
            wrapMode: Text.WordWrap
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 2
            columnSpacing: 16
            rowSpacing: 4
            Label { text: "Model path"; color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall }
            Label { text: root.report && root.report.model ? root.report.model.primary_path : ""; color: AppTheme.textDim; font.pixelSize: AppTheme.fontSmall; elide: Text.ElideMiddle; Layout.fillWidth: true }
            Label { text: "Runtime"; color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall }
            Label { text: root.report && root.report.runtime ? root.report.runtime.id + " (" + root.report.runtime.backend + ")" : "—"; color: AppTheme.textDim; font.pixelSize: AppTheme.fontSmall }
            Label { text: "Exit code"; color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall }
            Label { text: root.report && root.report.instance ? String(root.report.instance.exit_code) : "—"; color: AppTheme.textDim; font.pixelSize: AppTheme.fontSmall }
            Label { text: "System RAM"; color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall }
            Label { text: root.report ? AppTheme.bytes(root.report.ram_total) + " total, " + AppTheme.bytes(root.report.ram_available) + " free" : ""; color: AppTheme.textDim; font.pixelSize: AppTheme.fontSmall }
        }

        AppGroupBox {
            Layout.fillWidth: true
            title: "Command"
            ScrollView {
                anchors.fill: parent
                implicitHeight: 48
                clip: true
                TextEdit {
                    readOnly: true
                    width: parent.width
                    text: root.report && root.report.instance ? root.report.instance.command : ""
                    color: AppTheme.text
                    font.family: "monospace"
                    font.pixelSize: AppTheme.fontSmall
                    wrapMode: Text.WrapAnywhere
                }
            }
        }

        AppGroupBox {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Log (tail)"
            ScrollView {
                anchors.fill: parent
                clip: true
                TextEdit {
                    readOnly: true
                    width: parent.width
                    text: root.report ? root.report.log_tail : ""
                    color: AppTheme.text
                    font.family: "monospace"
                    font.pixelSize: AppTheme.fontSmall
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            AppButton {
                text: "Copy report"
                onClicked: {
                    var r = root.report
                    if (!r) return
                    var text = JSON.stringify(r, null, 2)
                    reportCopy.text = text
                    reportCopy.selectAll()
                    reportCopy.copy()
                }
                TextEdit { id: reportCopy; visible: false }
            }
            Item { Layout.fillWidth: true }
            AppButton { text: "Close"; onClicked: root.close() }
            AppButton { text: "Try CPU fallback"; onClicked: { root.retryCpu(); root.close() } }
            AppButton { text: "Retry safe settings"; onClicked: { root.retrySafe(); root.close() } }
            AppButton { text: "Retry"; primary: true; onClicked: { root.retry(); root.close() } }
        }
    }
}
