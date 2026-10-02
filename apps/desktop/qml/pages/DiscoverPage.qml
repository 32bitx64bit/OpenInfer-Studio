import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".."
import "../components"

Item {
    id: page
    property var api
    property var events
    property bool experimentalAudio: false

    property var results: []
    property bool searching: false
    property string searchError: ""
    property var detail: null
    property var detailPlan: null
    property string detailMTP: ""
    property string detailDraft: ""
    property string detailEmbedding: ""
    property string detailDiffusion: ""
    property bool detailLoading: false
    property bool cardOpen: false
    property bool hasToken: false
    signal downloadQueued(string label)

    function modalityLabel(mods) {
        if (!mods || mods.length === 0) return ""
        var a = mods.indexOf("audio") >= 0
        var v = mods.indexOf("vision") >= 0
        if (a && v) return "audio+vision"
        if (a) return "audio"
        if (v) return "vision"
        return ""
    }

    function mtpLabel(kind) {
        if (kind === "mtp-draft") return "MTP draft"
        if (kind === "mtp") return "MTP"
        return ""
    }

    function embeddingLabel(kind) {
        if (kind === "reranker") return "reranker"
        if (kind === "embedding") return "embedding"
        return ""
    }

    function diffusionLabel(kind) {
        if (kind === "video") return "video"
        if (kind === "both") return "image+video"
        if (kind === "image") return "image"
        return ""
    }

    function draftLabel(kind) {
        var st = String(kind || "")
        if (st.indexOf("draft-") === 0)
            st = st.substring(6)
        if (st === "dflash") return "DFlash"
        if (st === "eagle3") return "EAGLE3"
        if (st === "dspark") return "DSpark"
        if (st === "mtp" || st === "mtp-draft") return "MTP draft"
        if (st === "simple" || st === "draft") return "draft"
        return st
    }

    function reload() {
        api.get("/api/v1/hf/token", function(st, data) {
            if (st === 200) page.hasToken = data && data.configured
        })
    }

    function search() {
        page.searching = true
        page.searchError = ""
        var q = encodeURIComponent(searchField.text)
        var sort = sortCombo.currentValue
        // One corpus: GGUF chat models and image/video generators together.
        api.get("/api/v1/hf/search?q=" + q + "&sort=" + sort + "&limit=40&kind=all", function(st, data) {
            page.searching = false
            if (st === 200) {
                page.results = (data && data.results) || []
            } else {
                page.searchError = (data && (data.detail || data.error)) || ("HTTP " + st)
            }
        })
    }

    function openRepo(repoId) {
        page.detailLoading = true
        page.detail = null
        page.detailPlan = null
        page.cardOpen = false
        page.detailMTP = ""
        page.detailDraft = ""
        page.detailEmbedding = ""
        page.detailDiffusion = ""
        detailDialog.open()
        api.get("/api/v1/hf/repo/" + repoId, function(st, data) {
            page.detailLoading = false
            if (st === 200) {
                page.detail = data.repo
                page.detailMTP = data.mtp || ""
                page.detailDraft = data.draft || ""
                page.detailEmbedding = data.embedding || ""
                page.detailDiffusion = data.diffusion || ""
                page.detailPlan = data.plan || { "kind": "", "components": [], "notes": [] }
            } else {
                page.searchError = (data && (data.detail || data.error)) || ("HTTP " + st)
                detailDialog.close()
            }
        })
    }

    // Download what the picker confirmed: every ticked component, each at its
    // chosen precision, as one queued download.
    function downloadPlan(req) {
        if (!page.detail || !req.files || req.files.length === 0)
            return
        var name = page.detail.id
        var label = req.summary !== "" ? name + " · " + req.summary : name
        api.post("/api/v1/downloads", {
            "kind": "model",
            "label": label,
            "repo": name,
            "group": req.group,
            "files": req.files
        }, function(st, data) {
            if (st !== 201)
                page.searchError = (data && (data.detail || data.error)) || "download failed"
            else
                page.downloadQueued(label)
        })
        detailDialog.close()
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: AppTheme.pad
        spacing: AppTheme.gap

        PageHeader {
            title: "Browse models"
            subtitle: "Find chat models and image/video generators on Hugging Face."
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            SearchField {
                id: searchField
                Layout.fillWidth: true
                placeholderText: "Search Hugging Face for models…"
                searchLabel: "Search Hugging Face models"
                onAccepted: page.search()
            }
            AppComboBox {
                id: sortCombo
                model: [
                    { "text": "Relevance", "value": "" },
                    { "text": "Downloads", "value": "downloads" },
                    { "text": "Likes", "value": "likes" },
                    { "text": "Trending", "value": "trending" },
                    { "text": "Recently updated", "value": "lastModified" }
                ]
                textRole: "text"
                valueRole: "value"
            }
            AppButton { text: "Search"; primary: true; onClicked: page.search() }
        }

        Label {
            visible: page.searchError !== ""
            Layout.fillWidth: true
            text: page.searchError
            color: AppTheme.danger
            wrapMode: Text.WordWrap
        }

        BusyIndicator { visible: page.searching; Layout.alignment: Qt.AlignHCenter }

        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 8
            model: page.results
            add: Transition {
                ParallelAnimation {
                    NumberAnimation { property: "opacity"; from: 0; to: 1; duration: AppTheme.motion; easing.type: Easing.OutCubic }
                    NumberAnimation { property: "y"; duration: AppTheme.motion; easing.type: Easing.OutCubic }
                }
            }
            populate: Transition {
                ParallelAnimation {
                    NumberAnimation { property: "opacity"; from: 0; to: 1; duration: AppTheme.motion; easing.type: Easing.OutCubic }
                    NumberAnimation { property: "y"; duration: AppTheme.motion; easing.type: Easing.OutCubic }
                }
            }
            displaced: Transition {
                NumberAnimation { property: "y"; duration: AppTheme.motion; easing.type: Easing.OutCubic }
            }

            EmptyState {
                visible: page.results.length === 0 && !page.searching
                anchors.centerIn: parent
                icon: "⌕"
                title: "Search for models"
                hint: "Search Hugging Face for GGUF chat models and image/video generators. Open one to pick the file set you want."
            }

            delegate: Card {
                width: ListView.view.width
                implicitHeight: row.implicitHeight + 20
                RowLayout {
                    id: row
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 12
                    Rectangle {
                        width: 40; height: 40; radius: 20
                        color: AppTheme.accent
                        Text {
                            anchors.centerIn: parent
                            text: (modelData.author || "?").substring(0, 2).toUpperCase()
                            color: AppTheme.onAccent
                            font.weight: Font.Bold
                        }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 2
                        RowLayout {
                            Text { text: modelData.id; color: AppTheme.text; font.weight: Font.DemiBold; elide: Text.ElideRight; Layout.fillWidth: true }
                            Tag {
                                visible: page.mtpLabel(modelData.mtp) !== ""
                                text: page.mtpLabel(modelData.mtp)
                                tone: AppTheme.warning
                                Layout.minimumWidth: implicitWidth
                            }
                            Tag {
                                visible: page.draftLabel(modelData.draft) !== ""
                                text: page.draftLabel(modelData.draft)
                                tone: AppTheme.warning
                                Layout.minimumWidth: implicitWidth
                            }
                            Tag {
                                visible: page.embeddingLabel(modelData.embedding) !== ""
                                text: page.embeddingLabel(modelData.embedding)
                                tone: AppTheme.info
                                Layout.minimumWidth: implicitWidth
                            }
                            Tag {
                                visible: page.experimentalAudio && page.modalityLabel(modelData.modalities) !== ""
                                text: page.modalityLabel(modelData.modalities)
                                tone: AppTheme.success
                                Layout.minimumWidth: implicitWidth
                            }
                            Tag {
                                visible: page.diffusionLabel(modelData.diffusion) !== ""
                                text: page.diffusionLabel(modelData.diffusion)
                                tone: AppTheme.accent
                                Layout.minimumWidth: implicitWidth
                            }
                            Tag { visible: modelData.gated !== false && modelData.gated !== null; text: "gated"; tone: AppTheme.warning; Layout.minimumWidth: implicitWidth }
                            Tag { visible: modelData.private; text: "private"; tone: AppTheme.danger; Layout.minimumWidth: implicitWidth }
                        }
                        RowLayout {
                            spacing: 12
                            Text { text: "↓ " + modelData.downloads; color: AppTheme.textDim; font.pixelSize: AppTheme.fontSmall }
                            Text { text: "likes " + modelData.likes; color: AppTheme.textDim; font.pixelSize: AppTheme.fontSmall }
                            Text { text: (modelData.tags || []).slice(0, 5).join("  "); color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall; elide: Text.ElideRight; Layout.fillWidth: true }
                        }
                    }
                    AppButton { text: "Details"; onClicked: page.openRepo(modelData.id) }
                }
            }
        }
    }

    // Repository detail popup — nearly full-window, modal, always closable.
    Dialog {
        id: detailDialog
        anchors.centerIn: parent
        width: page.width * 0.92
        height: page.height * 0.92
        modal: true
        standardButtons: Dialog.NoButton
        padding: 0
        transformOrigin: Item.Center
        enter: DialogEnter {}
        exit: DialogExit {}
        Overlay.modal: Rectangle { color: AppTheme.overlay }

        background: Rectangle {
            color: AppTheme.bg
            radius: AppTheme.radius
            border.color: AppTheme.border
        }

        contentItem: ColumnLayout {
            spacing: 0

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 48
                color: AppTheme.bgAlt
                radius: AppTheme.radius
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: AppTheme.pad
                    anchors.rightMargin: 8
                    spacing: 8
                    Label {
                        Layout.fillWidth: true
                        text: page.detail ? page.detail.id : "Loading repository…"
                        font.pixelSize: AppTheme.fontTitle
                        font.weight: Font.DemiBold
                        color: AppTheme.text
                        elide: Text.ElideMiddle
                    }
                    Tag {
                        visible: page.mtpLabel(page.detailMTP) !== ""
                        text: page.mtpLabel(page.detailMTP)
                        tone: AppTheme.warning
                        Layout.minimumWidth: implicitWidth
                    }
                    Tag {
                        visible: page.draftLabel(page.detailDraft) !== ""
                        text: page.draftLabel(page.detailDraft)
                        tone: AppTheme.warning
                        Layout.minimumWidth: implicitWidth
                    }
                    Tag {
                        visible: page.embeddingLabel(page.detailEmbedding) !== ""
                        text: page.embeddingLabel(page.detailEmbedding)
                        tone: AppTheme.info
                        Layout.minimumWidth: implicitWidth
                    }
                    Tag {
                        visible: page.diffusionLabel(page.detailDiffusion) !== ""
                        text: page.diffusionLabel(page.detailDiffusion)
                        tone: AppTheme.accent
                        Layout.minimumWidth: implicitWidth
                    }
                    AppButton {
                        text: "Open in browser"
                        flat: true
                        visible: page.detail !== null
                        onClicked: Qt.openUrlExternally("https://huggingface.co/" + page.detail.id)
                    }
                    IconButton {
                        iconText: "✕"
                        description: "Close"
                        onClicked: detailDialog.close()
                    }
                }
            }

            BusyIndicator {
                visible: page.detailLoading
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 40
            }

            ColumnLayout {
                visible: page.detail !== null
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.margins: AppTheme.pad
                spacing: AppTheme.gap

                Label {
                    Layout.fillWidth: true
                    visible: page.detail && page.detail.gated !== false && page.detail.gated !== null
                    text: "This repository is gated: accept its terms on Hugging Face, then add your access token in Settings."
                    color: AppTheme.warning
                    wrapMode: Text.WordWrap
                }

                Label {
                    text: "Downloads: " + (page.detail ? page.detail.downloads : 0)
                        + "   Likes: " + (page.detail ? page.detail.likes : 0)
                    color: AppTheme.textDim
                    font.pixelSize: AppTheme.fontSmall
                }

                DownloadPlanPicker {
                    id: planPicker
                    objectName: "planPicker"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    plan: page.detailPlan
                    experimentalAudio: page.experimentalAudio
                    onConfirmed: function(request) { page.downloadPlan(request) }
                }

                // The picker is the point of this dialog: the card stays
                // folded until asked for.
                AppButton {
                    text: page.cardOpen ? "Hide model card ▴" : "Show model card ▾"
                    flat: true
                    onClicked: page.cardOpen = !page.cardOpen
                }

                AppGroupBox {
                    visible: page.cardOpen
                    Layout.fillWidth: true
                    Layout.preferredHeight: 220
                    title: "Model card"
                    ScrollView {
                        anchors.fill: parent
                        clip: true
                        TextEdit {
                            width: parent.width
                            readOnly: true
                            text: page.detail ? (page.detail.card || "No model card.") : ""
                            textFormat: TextEdit.MarkdownText
                            wrapMode: TextEdit.Wrap
                            color: AppTheme.textDim
                            font.pixelSize: AppTheme.fontSmall
                        }
                    }
                }
            }

            // Footer with an explicit close action (Escape also closes).
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 48
                color: AppTheme.bgAlt
                radius: AppTheme.radius
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: AppTheme.pad
                    anchors.rightMargin: AppTheme.pad
                    Item { Layout.fillWidth: true }
                    AppButton {
                        text: "Close"
                        onClicked: detailDialog.close()
                    }
                }
            }
        }
    }

    Component.onCompleted: reload()
}
