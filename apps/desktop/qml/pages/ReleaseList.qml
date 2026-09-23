import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".."
import "../components"

// ReleaseList: official release discovery for one engine (llama.cpp or
// stable-diffusion.cpp). Shared so both engines keep the exact same layout:
// title + backend filter + check/import row, error line, release cards.
Item {
    id: root

    property string engineTitle: ""
    property string engineKind: ""      // "" (llama) or "diffusion" (sd.cpp)
    property string engineNote: ""
    property bool checking: false
    property var releases: []
    property string errorText: ""
    property string checkLabel: "Check for releases"
    property var pageBackendFilter: ""

    signal check()
    signal install(string tag, string asset, string backend)
    signal importCustom()

    implicitHeight: box.implicitHeight

    AppGroupBox {
        id: box
        anchors.left: parent.left
        anchors.right: parent.right
        title: "Official " + root.engineTitle + " releases"

        ColumnLayout {
            width: parent.width
            spacing: 8
        Label {
            visible: root.engineNote !== ""
            Layout.fillWidth: true
            text: root.engineNote
            color: AppTheme.textDim
            font.pixelSize: AppTheme.fontSmall
            wrapMode: Text.WordWrap
        }
        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            Label { text: "Backend:"; color: AppTheme.textDim }
            AppComboBox {
                model: [
                    { "text": "Auto (recommended)", "value": "" },
                    { "text": "CPU", "value": "cpu" },
                    { "text": "Vulkan", "value": "vulkan" },
                    { "text": "CUDA", "value": "cuda" },
                    { "text": "HIP/ROCm", "value": "hip" },
                    { "text": "Metal", "value": "metal" },
                    { "text": "SYCL", "value": "sycl" }
                ]
                textRole: "text"; valueRole: "value"
                onActivated: function(i) { root.pageBackendFilter = model[i].value }
            }
            AppButton {
                text: root.checking ? "Checking…" : root.checkLabel
                enabled: !root.checking
                primary: true
                onClicked: root.check()
            }
            Item { Layout.fillWidth: true }
            AppButton {
                text: "Import custom build…"
                onClicked: root.importCustom()
            }
        }
        Label {
            visible: root.errorText !== ""
            Layout.fillWidth: true
            text: root.errorText
            color: AppTheme.danger
            wrapMode: Text.WordWrap
        }
        Repeater {
            model: root.releases
            delegate: Card {
                id: relCard
                property string relTag: modelData.tag
                Layout.fillWidth: true
                implicitHeight: relCol.implicitHeight + 16
                ColumnLayout {
                    id: relCol
                    anchors.fill: parent
                    anchors.margins: 8
                    spacing: 6
                    RowLayout {
                        Label { text: modelData.tag; color: AppTheme.text; font.weight: Font.DemiBold }
                        Label { text: modelData.published_at.substring(0, 10); color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall }
                        Item { Layout.fillWidth: true }
                    }
                    Repeater {
                        model: (modelData.matches || []).slice(0, 5)
                        delegate: RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            Label {
                                text: modelData.asset.name
                                color: AppTheme.textDim
                                font.pixelSize: AppTheme.fontSmall
                                elide: Text.ElideMiddle
                                Layout.fillWidth: true
                            }
                            Tag { text: modelData.backend; tone: AppTheme.accent }
                            Label { text: AppTheme.bytes(modelData.asset.size); color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall }
                            AppButton {
                                text: "Install"
                                onClicked: root.install(relCard.relTag, modelData.asset.name, modelData.backend)
                            }
                        }
                    }
                } // relCol
            } // release Card
        } // releases Repeater
        } // content ColumnLayout
    } // AppGroupBox
}
