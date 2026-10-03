pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../.."
import ".."
import "RunHistory.js" as History

Rectangle {
    id: root
    property var runData: ({})
    property bool expanded: false
    property bool cancelPending: false
    property string cancelError: ""
    property var images: History.resultImages(runData)
    property var nodes: History.nodeRows(runData)
    property var promptRows: History.prompts(runData)
    signal inspectRequested()
    signal restoreRequested()
    signal cancelRequested()
    signal toggleDetails()
    signal viewRequested(int imageIndex)
    signal useRequested(string fileUrl)
    signal compareRequested(string fileUrl)

    implicitHeight: body.implicitHeight + 2 * AppTheme.padSmall
    color: AppTheme.surface
    border.color: AppTheme.border
    radius: AppTheme.radius

    ColumnLayout {
        id: body
        x: AppTheme.padSmall
        y: AppTheme.padSmall
        width: Math.max(0, root.width - 2 * AppTheme.padSmall)
        spacing: AppTheme.gapTight
        RowLayout {
            Layout.fillWidth: true
            Label {
                Layout.fillWidth: true
                text: History.title(root.runData)
                textFormat: Text.PlainText
                color: AppTheme.text
                font.weight: Font.DemiBold
                elide: Text.ElideRight
            }
            Tag { text: root.runData.state || ""; tone: AppTheme.stateColor(root.runData.state || "") }
            AppButton {
                objectName: "cancelRunButton"
                visible: History.active(root.runData)
                text: root.cancelPending ? "Canceling…" : "Cancel"
                enabled: !root.cancelPending
                flat: true
                accessibleDescription: "Cancel run " + root.runData.id
                onClicked: root.cancelRequested()
            }
        }
        Label {
            Layout.fillWidth: true
            text: History.timestamp(root.runData.created_at) + (root.runData.only ? " · Node: " + root.runData.only : "")
            color: AppTheme.textFaint
            font.pixelSize: AppTheme.fontSmall
            textFormat: Text.PlainText
            wrapMode: Text.Wrap
        }
        Label {
            Layout.fillWidth: true
            visible: root.promptRows.length > 0 && !root.expanded
            text: root.promptRows.map(function(prompt) { return prompt.text }).join(" · ")
            textFormat: Text.PlainText
            maximumLineCount: 2
            wrapMode: Text.Wrap
            elide: Text.ElideRight
            color: AppTheme.textDim
            font.pixelSize: AppTheme.fontSmall
        }
        Label {
            Layout.fillWidth: true
            visible: History.seedSummary(root.runData) !== ""
            text: "Resolved seed · " + History.seedSummary(root.runData)
            textFormat: Text.PlainText
            color: AppTheme.textDim
            font.pixelSize: AppTheme.fontSmall
            wrapMode: Text.Wrap
        }

        Repeater {
            model: History.active(root.runData) ? root.nodes.filter(function(row) { return row.status.state === "running" }) : []
            delegate: ColumnLayout {
                    id: progressRow
                required property var modelData
                Layout.fillWidth: true
                spacing: 4
                Label {
                    Layout.fillWidth: true
                    text: progressRow.modelData.title + " · " + History.progressText(progressRow.modelData.status)
                    textFormat: Text.PlainText
                    color: AppTheme.textDim
                    wrapMode: Text.Wrap
                    font.pixelSize: AppTheme.fontSmall
                }
                AppProgressBar {
                    Layout.fillWidth: true
                    visible: progressRow.modelData.status.progress && progressRow.modelData.status.progress.total > 0
                    from: 0
                    to: progressRow.modelData.status.progress && progressRow.modelData.status.progress.total > 0 ? progressRow.modelData.status.progress.total : 1
                    value: progressRow.modelData.status.progress ? (progressRow.modelData.status.progress.current || 0) : 0
                }
            }
        }
        Label {
            Layout.fillWidth: true
            visible: !!root.runData.error || root.cancelError !== ""
            text: root.cancelError || root.runData.error || ""
            textFormat: Text.PlainText
            wrapMode: Text.Wrap
            color: AppTheme.danger
            font.pixelSize: AppTheme.fontSmall
        }
        Flow {
            Layout.fillWidth: true
            spacing: AppTheme.gapTight
            Repeater {
                model: root.images
                delegate: Column {
                    id: resultTile
                    required property var modelData
                    required property int index
                    width: Math.max(100, Math.min(156, body.width))
                    spacing: 4
                    AppButton {
                        objectName: "resultThumbnail_" + resultTile.index
                        width: parent.width
                        height: 112
                        padding: 4
                        accessibleDescription: "Open result " + (resultTile.index + 1) + " from " + resultTile.modelData.nodeTitle
                        contentItem: Item {
                            Image {
                                id: thumbnail
                                anchors.fill: parent
                                source: resultTile.modelData.fileUrl
                                sourceSize: Qt.size(312, 224)
                                fillMode: Image.PreserveAspectFit
                                asynchronous: true
                            }
                            Label {
                                anchors.centerIn: parent
                                width: parent.width
                                horizontalAlignment: Text.AlignHCenter
                                wrapMode: Text.Wrap
                                visible: thumbnail.status === Image.Error || thumbnail.status === Image.Loading
                                text: thumbnail.status === Image.Loading ? "Loading…" : "File unavailable"
                                font.pixelSize: AppTheme.fontSmall
                                color: AppTheme.textFaint
                            }
                        }
                        onClicked: root.viewRequested(resultTile.index)
                    }
                    Label {
                        width: parent.width
                        text: resultTile.modelData.nodeTitle + (resultTile.modelData.saved ? " · Saved" : "")
                        textFormat: Text.PlainText
                        color: AppTheme.textFaint
                        font.pixelSize: AppTheme.fontSmall
                        elide: Text.ElideRight
                    }
                    Row {
                        spacing: 2
                        AppButton { objectName: "useResult_" + resultTile.index; text: "Use"; flat: true; accessibleDescription: "Use result " + (resultTile.index + 1); onClicked: root.useRequested(resultTile.modelData.fileUrl) }
                        AppButton { objectName: "compareResult_" + resultTile.index; text: "Compare"; flat: true; accessibleDescription: "Compare result " + (resultTile.index + 1); onClicked: root.compareRequested(resultTile.modelData.fileUrl) }
                    }
                }
            }
        }
        Flow {
            Layout.fillWidth: true
            spacing: 4
            AppButton { objectName: "inspectRunButton"; text: "Inspect"; flat: true; onClicked: root.inspectRequested() }
            AppButton { objectName: "restoreRunButton"; text: "Restore graph"; flat: true; enabled: !!root.runData.graph; onClicked: root.restoreRequested() }
            AppButton { objectName: "runDetailsButton"; text: root.expanded ? "Hide details" : "Details"; flat: true; onClicked: root.toggleDetails() }
        }
        ColumnLayout {
            visible: root.expanded
            Layout.fillWidth: true
            spacing: AppTheme.gapTight
            Label {
                Layout.fillWidth: true
                text: "Run " + (root.runData.id || "")
                    + (root.runData.workflow_id ? "\nWorkflow " + root.runData.workflow_id : "")
                    + (root.runData.started_at ? "\nStarted " + History.timestamp(root.runData.started_at) : "")
                    + (root.runData.finished_at ? "\nFinished " + History.timestamp(root.runData.finished_at) : "")
                textFormat: Text.PlainText
                wrapMode: Text.Wrap
                color: AppTheme.textDim
                font.pixelSize: AppTheme.fontSmall
            }
            Repeater {
                model: root.promptRows
                delegate: ColumnLayout {
                    id: promptRow
                    required property var modelData
                    Layout.fillWidth: true
                    Label { text: promptRow.modelData.title; textFormat: Text.PlainText; color: AppTheme.text; font.pixelSize: AppTheme.fontSmall }
                    AppTextArea {
                        Layout.fillWidth: true
                        text: promptRow.modelData.text
                        textFormat: TextEdit.PlainText
                        readOnly: true
                        font.pixelSize: AppTheme.fontSmall
                        Accessible.name: promptRow.modelData.title
                    }
                }
            }
            Repeater {
                model: root.nodes
                delegate: ColumnLayout {
                    id: nodeDetail
                    required property var modelData
                    Layout.fillWidth: true
                    spacing: 4
                    Label {
                        Layout.fillWidth: true
                        text: nodeDetail.modelData.title + " · " + (nodeDetail.modelData.status.state || "")
                            + (nodeDetail.modelData.status.cached ? " · Cached" : "")
                            + (nodeDetail.modelData.status.ms !== undefined ? " · " + History.duration(nodeDetail.modelData.status.ms) : "")
                        textFormat: Text.PlainText
                        wrapMode: Text.Wrap
                        font.pixelSize: AppTheme.fontSmall
                        color: AppTheme.stateColor(nodeDetail.modelData.status.state || "")
                    }
                    Label {
                        Layout.fillWidth: true
                        text: History.progressText(nodeDetail.modelData.status)
                            + (History.seed(nodeDetail.modelData.status) !== "" ? "\nResolved seed: " + History.seed(nodeDetail.modelData.status) : "")
                            + (nodeDetail.modelData.status.job_id ? "\nJob " + nodeDetail.modelData.status.job_id : "")
                        textFormat: Text.PlainText
                        wrapMode: Text.Wrap
                        color: AppTheme.textDim
                        font.pixelSize: AppTheme.fontSmall
                    }
                    AppTextArea {
                        Layout.fillWidth: true
                        visible: Object.keys(nodeDetail.modelData.params).length > 0 || (nodeDetail.modelData.status.outputs || []).length > 0
                        text: JSON.stringify(nodeDetail.modelData.params, null, 2)
                            + ((nodeDetail.modelData.status.outputs || []).length ? "\nOutputs:\n" + nodeDetail.modelData.status.outputs.join("\n") : "")
                        textFormat: TextEdit.PlainText
                        readOnly: true
                        font.pixelSize: AppTheme.fontSmall
                        Accessible.name: nodeDetail.modelData.title + " parameters and output paths"
                    }
                }
            }
        }
    }
}
