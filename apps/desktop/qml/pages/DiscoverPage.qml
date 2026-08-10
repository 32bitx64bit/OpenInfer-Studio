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
    property var detailGroups: []
    property var detailProjectors: []
    property var detailModalities: []
    property string detailMTP: ""
    property string detailEmbedding: ""
    property bool withVision: true
    property bool showFilePaths: false
    property bool detailLoading: false
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

    function projectorToggleLabel() {
        var bytes = AppTheme.bytes(page.projectorBytes())
        if (page.experimentalAudio) {
            var mod = page.modalityLabel(page.detailModalities)
            var hint = mod !== "" ? (" · " + mod) : ""
            return "Download with multimodal projector (mmproj · " + bytes + ")" + hint
        }
        return "Download with vision (mmproj · " + bytes + ")"
    }

    function groupModalityTag(group) {
        if (page.experimentalAudio) {
            var label = page.modalityLabel(page.detailModalities)
            if (label !== "") return label
            if (group.vision) return "multimodal"
            return ""
        }
        return group.vision ? "vision" : ""
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
        api.get("/api/v1/hf/search?q=" + q + "&sort=" + sort + "&limit=40", function(st, data) {
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
        page.detailModalities = []
        page.detailMTP = ""
        page.detailEmbedding = ""
        page.withVision = true
        page.showFilePaths = false
        detailDialog.open()
        api.get("/api/v1/hf/repo/" + repoId, function(st, data) {
            page.detailLoading = false
            if (st === 200) {
                page.detail = data.repo
                page.detailGroups = data.groups || []
                page.detailProjectors = data.projectors || []
                page.detailModalities = data.modalities || []
                page.detailMTP = data.mtp || ""
                page.detailEmbedding = data.embedding || ""
            } else {
                page.searchError = (data && (data.detail || data.error)) || ("HTTP " + st)
                detailDialog.close()
            }
        })
    }

    function projectorBytes() {
        var t = 0
        for (var i = 0; i < page.detailProjectors.length; i++) t += page.detailProjectors[i].size
        return t
    }

    function downloadGroup(group) {
        var files = group.files.map(function(f) { return { "path": f.path, "size": f.size } })
        var hasProjector = group.files.some(function(f) { return f.kind === "projector" })
        if (page.withVision && !hasProjector) {
            for (var i = 0; i < page.detailProjectors.length; i++)
                files.push({ "path": page.detailProjectors[i].path, "size": page.detailProjectors[i].size })
        }
        api.post("/api/v1/downloads", {
            "kind": "model",
            "label": (page.detail ? page.detail.id : "") + " " + group.label,
            "repo": page.detail.id,
            "group": group.id,
            "files": files
        }, function(st, data) {
            if (st !== 201)
                page.searchError = (data && (data.detail || data.error)) || "download failed"
            else
                page.downloadQueued((page.detail ? page.detail.id : "Model") + " · " + group.label)
        })
        detailDialog.close()
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: AppTheme.pad
        spacing: AppTheme.gap

        PageHeader {
            title: "Browse models"
            subtitle: "Find GGUF models on Hugging Face. Start with a quantization that fits your hardware, then reveal advanced files only when needed."
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            SearchField {
                id: searchField
                Layout.fillWidth: true
                placeholderText: "Search Hugging Face for GGUF models…"
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

            EmptyState {
                visible: page.results.length === 0 && !page.searching
                anchors.centerIn: parent
                icon: "⌕"
                title: "Search for models"
                hint: "Search Hugging Face for GGUF repositories. Results are grouped by quantization, split set, and projector files."
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
                        visible: page.embeddingLabel(page.detailEmbedding) !== ""
                        text: page.embeddingLabel(page.detailEmbedding)
                        tone: AppTheme.info
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

                RowLayout {
                    Layout.fillWidth: true
                    visible: page.detailProjectors.length > 0
                    spacing: 8
                    AppSwitch {
                        id: visionToggle
                        checked: page.withVision
                        onToggled: page.withVision = checked
                    }
                    Label {
                        text: page.projectorToggleLabel()
                        color: AppTheme.text
                        ToolTip.visible: visionHover.hovered
                        ToolTip.text: {
                            var paths = page.detailProjectors.map(function(p) { return p.path }).join("\n")
                            if (page.experimentalAudio && page.modalityLabel(page.detailModalities) !== "")
                                return "Includes projector for " + page.modalityLabel(page.detailModalities) + "\n" + paths
                            return paths
                        }
                        HoverHandler { id: visionHover }
                        MouseArea {
                            anchors.fill: parent
                            onClicked: visionToggle.toggle()
                        }
                    }
                    Item { Layout.fillWidth: true }
                }

                AppCheckBox {
                    visible: page.detailGroups.length > 0
                    text: "Show individual file paths"
                    checked: page.showFilePaths
                    onToggled: page.showFilePaths = checked
                }

                AppGroupBox {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    title: "Available files (" + page.detailGroups.length + " groups)"
                    ListView {
                        anchors.fill: parent
                        clip: true
                        spacing: 8
                        model: page.detailGroups
                        delegate: Card {
                            width: ListView.view.width - 4
                            implicitHeight: gcol.implicitHeight + 20
                            ColumnLayout {
                                id: gcol
                                anchors.fill: parent
                                anchors.margins: 10
                                spacing: 6
                                RowLayout {
                                    Layout.fillWidth: true
                                    Label { text: modelData.label; color: AppTheme.text; font.weight: Font.DemiBold }
                                    Tag { visible: modelData.split; text: modelData.parts + " parts"; tone: AppTheme.info; Layout.minimumWidth: implicitWidth }
                                    Tag {
                                        visible: page.mtpLabel(modelData.mtp) !== ""
                                        text: page.mtpLabel(modelData.mtp)
                                        tone: AppTheme.warning
                                        Layout.minimumWidth: implicitWidth
                                    }
                                    Tag {
                                        visible: page.groupModalityTag(modelData) !== ""
                                        text: page.groupModalityTag(modelData)
                                        tone: AppTheme.success
                                        Layout.minimumWidth: implicitWidth
                                    }
                                    Item { Layout.fillWidth: true }
                                    Label {
                                        text: {
                                            var t = modelData.total_bytes
                                            var hasProj = modelData.files.some(function(f) { return f.kind === "projector" })
                                            if (page.withVision && page.detailProjectors.length > 0 && !hasProj) {
                                                var suffix = page.experimentalAudio ? " (incl. mmproj)" : " (incl. vision)"
                                                return AppTheme.bytes(t + page.projectorBytes()) + suffix
                                            }
                                            return AppTheme.bytes(t)
                                        }
                                        color: AppTheme.textDim
                                        font.pixelSize: AppTheme.fontSmall
                                    }
                                }
                                Label {
                                    text: "Estimated memory: ~" + AppTheme.bytes(modelData.est_memory_bytes) + " (estimate)"
                                    color: AppTheme.textFaint
                                    font.pixelSize: AppTheme.fontSmall
                                }
                                Repeater {
                                    model: page.showFilePaths ? modelData.files : []
                                    Label {
                                        text: "  " + modelData.path + "  ·  " + AppTheme.bytes(modelData.size)
                                        color: AppTheme.textDim
                                        font.pixelSize: AppTheme.fontSmall
                                        font.family: "monospace"
                                        elide: Text.ElideMiddle
                                        Layout.fillWidth: true
                                    }
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    Item { Layout.fillWidth: true }
                                    AppButton {
                                        text: "Download"
                                        primary: true
                                        onClicked: page.downloadGroup(modelData)
                                    }
                                }
                            }
                        }
                    }
                }

                AppGroupBox {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 160
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
