import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import ".."
import "../components"

Item {
    id: page
    property var api
    property var events
    property bool experimentalAudio: false

    property var conversations: []
    property var currentConv: null
    property var messages: []          // tree nodes
    property var chain: []             // active branch (root→leaf)
    property var library: []
    property var instances: []
    property bool generating: false
    property bool loadingModel: false
    property string waitingForModelId: ""
    property var pendingGeneration: null
    property string streamingId: ""
    // Live text of the in-flight message; the bubble binds these directly.
    property string streamContent: ""
    property string streamReasoning: ""
    property var lastStats: null
    property string errorText: ""
    property var expandedReasoning: ({})   // messageId → bool
    property string pendingAudioPath: ""
    property string pendingAudioName: ""
    property bool conversationsLoading: false
    property bool messagesLoading: false
    property bool showArchived: false
    signal openLibrary()
    signal configureModel(string modelId)

    function modelState() {
        if (!page.currentConv) return "Choose a model"
        var inst = page.instances.find(function(i) { return i.model_id === page.currentConv.model_id })
        if (!inst) return "Not loaded"
        if (inst.state === "ready" || inst.state === "busy") return "Ready"
        return inst.state
    }

    Shortcut { sequence: "Ctrl+N"; onActivated: page.newConversation() }
    Shortcut { sequence: "Ctrl+,"; onActivated: paramsDrawer.open() }
    Shortcut {
        sequence: "Escape"
        enabled: page.generating
        onActivated: {
            if (page.currentConv) page.api.post("/api/v1/chat/" + page.currentConv.id + "/stop", {}, function() {})
        }
    }

    function isSpeculativeDraft(m) {
        if (!m) return false
        var meta = m.metadata || {}
        if (meta.speculative_draft) return true
        var arch = String(m.architecture || "").toLowerCase()
        if (arch === "gemma4-assistant" || arch === "gemma4_assistant"
                || arch === "eagle3" || arch === "dflash" || arch === "dflash-draft"
                || arch === "dspark"
                || arch === "muse-glimmer-assistant" || arch === "muse_glimmer_assistant"
                || arch === "museglimmer-assistant")
            return true
        // Path fallback if library metadata hasn't been rescanned yet.
        var path = String(m.primary_path || m.alias || "").toLowerCase()
        var base = path.split("/").pop() || path
        if (base.indexOf("mtp-") === 0
            || base.indexOf("eagle3-") === 0
            || base.indexOf("dflash-") === 0
            || base.indexOf("dspark-") === 0)
            return true
        if (base.indexOf("assistant") >= 0 && (base.indexOf("glimmer") >= 0 || base.indexOf("gemma-4") >= 0 || base.indexOf("gemma4") >= 0))
            return true
        return false
    }

    function isEmbeddingModel(m) {
        if (!m) return false
        var meta = m.metadata || {}
        return !!(meta.is_embedding || meta.is_reranker)
    }

    // Chat targets only — MTP/EAGLE/DFlash/DSpark sidecars and embedders
    // belong elsewhere (draft picker / Developer API), not as conversation models.
    function chatModels() {
        return (page.library || []).filter(function(m) {
            return !page.isSpeculativeDraft(m) && !page.isEmbeddingModel(m)
        })
    }

    function currentModel() {
        if (!page.currentConv) return null
        for (var i = 0; i < page.library.length; i++) {
            if (page.library[i].id === page.currentConv.model_id) return page.library[i]
        }
        return null
    }
    function modelSupportsAudio() {
        var m = page.currentModel()
        if (!m || !m.metadata) return false
        if (m.metadata.speculative_draft) return false
        return !!m.metadata.has_audio
    }
    function canAttachAudio() {
        return page.experimentalAudio && page.modelSupportsAudio()
    }

    function reasoningExpanded(id) { return !!page.expandedReasoning[id] }
    function toggleReasoning(id) {
        var m = {}
        for (var k in page.expandedReasoning) m[k] = page.expandedReasoning[k]
        m[id] = !m[id]
        page.expandedReasoning = m
    }
    function reasoningPreview(text) {
        if (!text) return ""
        var one = text.replace(/\s+/g, " ").trim()
        return one.length > 90 ? one.substring(0, 90) + "…" : one
    }

    function reload() {
        page.conversationsLoading = true
        api.get("/api/v1/chat" + (page.showArchived ? "?archived=1" : ""), function(st, data) {
            page.conversationsLoading = false
            if (st === 200) page.conversations = (data && data.conversations) || []
        })
        api.get("/api/v1/models", function(st, data) {
            if (st === 200) page.library = (data && data.models) || []
        })
        api.get("/api/v1/instances", function(st, data) {
            if (st === 200) page.instances = (data && data.instances) || []
        })
    }

    function openConversation(c) {
        page.currentConv = c
        page.errorText = ""
        page.messagesLoading = true
        paramsDrawer.loadForConversation(c)
        page.syncModelSelector()
        api.get("/api/v1/chat/" + c.id + "/messages", function(st, data) {
            page.messagesLoading = false
            if (st !== 200) return
            page.messages = (data && data.messages) || []
            page.chain = page.buildChain(page.latestLeaf())
            chatList.positionViewAtEnd()
        })
    }

    // Keep the selector aligned with the conversation's actual model,
    // including after the library reloads (which resets the combo).
    function syncModelSelector() {
        var models = page.chatModels()
        if (!page.currentConv || models.length === 0) return
        for (var i = 0; i < models.length; i++) {
            if (models[i].id === page.currentConv.model_id) {
                if (modelSelector.currentIndex !== i) modelSelector.currentIndex = i
                return
            }
        }
    }
    onLibraryChanged: syncModelSelector()

    function latestLeaf() {
        var hasChild = {}
        for (var i = 0; i < page.messages.length; i++)
            if (page.messages[i].parent_id) hasChild[page.messages[i].parent_id] = true
        var leaf = ""
        for (var j = page.messages.length - 1; j >= 0; j--)
            if (!hasChild[page.messages[j].id]) { leaf = page.messages[j].id; break }
        return leaf
    }

    function buildChain(leafId) {
        var byId = {}
        for (var i = 0; i < page.messages.length; i++) byId[page.messages[i].id] = page.messages[i]
        var chain = []
        var cur = leafId
        var guard = 0
        while (cur && byId[cur] && guard++ < 10000) {
            chain.unshift(byId[cur])
            cur = byId[cur].parent_id
        }
        return chain
    }

    function siblingsOf(msg) {
        var out = []
        for (var i = 0; i < page.messages.length; i++)
            if (page.messages[i].parent_id === msg.parent_id && page.messages[i].id !== msg.id)
                out.push(page.messages[i])
        return out
    }

    function newConversation() {
        var mid = modelSelector.currentModelId()
        if (!mid) {
            page.errorText = "Choose a local model before creating a chat. Browse models to download or import one."
            return
        }
        api.post("/api/v1/chat", { "model_id": mid, "title": "New chat" },
            function(st, data) {
                if (st === 201) {
                    page.reload()
                    page.openConversation(data)
                }
            })
    }

    function startGeneration(body) {
        if (!page.currentConv) return
        api.post("/api/v1/chat/" + page.currentConv.id + "/generate", body,
            function(st, data) {
                if (st !== 202) {
                    page.generating = false
                    page.errorText = (data && (data.detail || data.error)) || ("HTTP " + st)
                } else {
                    page.beginStreamingMessage(data.message_id)
                }
            })
    }

    function tryStartPendingGeneration() {
        if (!page.pendingGeneration || !page.waitingForModelId) return
        var ready = page.instances.some(function(i) {
            return i.model_id === page.waitingForModelId && (i.state === "ready" || i.state === "busy")
        })
        var failed = page.instances.some(function(i) {
            return i.model_id === page.waitingForModelId && ["failed", "crashed"].indexOf(i.state) >= 0
        })
        if (ready) {
            var request = page.pendingGeneration
            page.pendingGeneration = null
            page.waitingForModelId = ""
            page.loadingModel = false
            modelWaitTimer.stop()
            page.startGeneration(request)
        } else if (failed) {
            page.pendingGeneration = null
            page.waitingForModelId = ""
            page.loadingModel = false
            page.generating = false
            modelWaitTimer.stop()
            page.errorText = "This model could not load. Open the Library to review diagnostics or configure a safer load."
        }
    }

    function startWhenModelReady(body) {
        if (!page.currentConv) return
        // A generation starts only after the selected model is ready. The old
        // parallel load/generate requests raced on slower hardware.
        var mid = page.currentConv.model_id
        if (!mid) {
            page.generating = false
            page.errorText = "Choose a model for this chat before generating."
            return
        }
        var loaded = page.instances.some(function(i) {
            return i.model_id === mid && (i.state === "ready" || i.state === "busy")
        })
        if (!loaded && mid) {
            page.loadingModel = true
            page.waitingForModelId = mid
            page.pendingGeneration = body
            modelWaitTimer.restart()
            api.post("/api/v1/models/" + mid + "/load", {}, function(st, data) {
                if (st !== 202) {
                    page.generating = false
                    page.loadingModel = false
                    page.pendingGeneration = null
                    page.waitingForModelId = ""
                    modelWaitTimer.stop()
                    page.errorText = "Auto-load failed: " + ((data && (data.detail || data.error)) || st)
                }
            })
            return
        }
        page.startGeneration(body)
    }

    function send() {
        var text = input.text.trim()
        var hasAudio = page.pendingAudioPath !== ""
        if ((!text && !hasAudio) || !page.currentConv || page.generating) return
        if (hasAudio && !page.canAttachAudio()) {
            page.errorText = "Audio attachments require experimental audio models and an audio-capable model."
            return
        }
        input.text = ""
        page.errorText = ""
        page.generating = true
        var body = { "content": text, "params": paramsDrawer.params }
        if (hasAudio) {
            body.audio = {
                "path": page.pendingAudioPath,
                "name": page.pendingAudioName
            }
            page.pendingAudioPath = ""
            page.pendingAudioName = ""
        }
        page.startWhenModelReady(body)
    }

    // Append the assistant message to the visible chain immediately so
    // tokens stream into it in place — no separate floating bubble.
    function beginStreamingMessage(messageId) {
        page.streamingId = messageId
        page.streamContent = ""
        page.streamReasoning = ""
        var msg = {
            "id": messageId, "conv_id": page.currentConv.id,
            "parent_id": page.chain.length ? page.chain[page.chain.length - 1].id : "",
            "role": "assistant", "content": "", "reasoning": "",
            "created_at": new Date().toISOString()
        }
        var msgs = page.messages.slice()
        msgs.push(msg)
        page.messages = msgs
        var ch = page.chain.slice()
        ch.push(msg)
        page.chain = ch
        chatList.positionViewAtEnd()
    }

    function regenerate(assistantMsg) {
        if (!page.currentConv || page.generating) return
        page.generating = true
        page.startWhenModelReady(
            { "parent_id": assistantMsg.parent_id, "content": "", "params": paramsDrawer.params })
    }

    Connections {
        target: page.events
        function onEventReceived(name, payload) {
            if (name === "chat.token" && page.currentConv && payload.conv_id === page.currentConv.id) {
                if (payload.error) {
                    page.errorText = payload.error
                    page.generating = false
                    page.streamingId = ""
                    page.reloadMessagesSoon()
                    return
                }
                if (payload.replace || payload.snapshot !== undefined || payload.reasoning_snapshot !== undefined) {
                    if (payload.snapshot !== undefined)
                        page.streamContent = payload.snapshot
                    if (payload.reasoning_snapshot !== undefined)
                        page.streamReasoning = payload.reasoning_snapshot
                    if (followTimer.running === false) followTimer.start()
                } else if (payload.delta || payload.reasoning_delta) {
                    if (payload.delta) page.streamContent += payload.delta
                    if (payload.reasoning_delta) page.streamReasoning += payload.reasoning_delta
                    if (followTimer.running === false) followTimer.start()
                }
                if (payload.done) {
                    page.generating = false
                    page.streamingId = ""
                    page.streamContent = ""
                    page.streamReasoning = ""
                    page.lastStats = payload.stats || null
                    page.reloadMessagesSoon()
                }
            }
            if (name === "chat.message" && page.currentConv && payload.conv_id === page.currentConv.id) {
                page.reloadMessagesSoon()
            }
            if (name === "instance.state_changed") {
                page.reload()
                if (page.loadingModel) {
                    modelWaitTimer.restart()
                }
            }
        }
    }

    Timer {
        id: reloadTimer
        interval: 150
        onTriggered: {
            if (!page.currentConv) return
            page.api.get("/api/v1/chat/" + page.currentConv.id + "/messages", function(st, data) {
                if (st !== 200) return
                page.messages = (data && data.messages) || []
                page.chain = page.buildChain(page.latestLeaf())
                chatList.positionViewAtEnd()
            })
        }
    }
    function reloadMessagesSoon() { reloadTimer.restart() }

    Timer {
        id: modelWaitTimer
        interval: 650
        repeat: true
        onTriggered: {
            page.api.get("/api/v1/instances", function(st, data) {
                if (st === 200) {
                    page.instances = (data && data.instances) || []
                    page.tryStartPendingGeneration()
                }
            })
        }
    }

    // Follow-scroll while streaming (throttled).
    Timer {
        id: followTimer
        interval: 120
        onTriggered: chatList.positionViewAtEnd()
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // Conversation list
        Rectangle {
            Layout.fillHeight: true
            width: 240
            color: AppTheme.bgAlt
            border.color: AppTheme.border
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 8
                spacing: 8
                RowLayout {
                    Layout.fillWidth: true
                    AppButton { text: "+ New chat"; primary: true; Layout.fillWidth: true; onClicked: page.newConversation() }
                }
                SearchField {
                    id: convSearch
                    Layout.fillWidth: true
                    placeholderText: "Search chats…"
                    searchLabel: "Search chats"
                }
                AppButton {
                    Layout.fillWidth: true
                    text: page.showArchived ? "Show active chats" : "View archived chats"
                    onClicked: {
                        page.showArchived = !page.showArchived
                        page.currentConv = null
                        page.chain = []
                        page.reload()
                    }
                }
                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: page.conversations.filter(function(c) {
                        return convSearch.text === "" ||
                            c.title.toLowerCase().indexOf(convSearch.text.toLowerCase()) >= 0
                    })
                    delegate: ItemDelegate {
                        width: ListView.view.width
                        height: 40
                        highlighted: page.currentConv && page.currentConv.id === modelData.id
                        text: modelData.title
                        onClicked: page.openConversation(modelData)
                        contentItem: Row {
                            spacing: 6
                            Text {
                                text: modelData.title
                                color: AppTheme.text
                                elide: Text.ElideRight
                                width: parent.width - 40
                                anchors.verticalCenter: parent.verticalCenter
                            }
                        }
                        Menu {
                            id: convMenu
                            MenuItem {
                                text: "Rename"
                                onTriggered: {
                                    renameDialog.targetConv = modelData
                                    renameDialog.open()
                                }
                            }
                            MenuItem {
                                text: modelData.archived ? "Unarchive" : "Archive"
                                onTriggered: page.api.patch("/api/v1/chat/" + modelData.id,
                                    { "archived": !modelData.archived }, function() { page.reload() })
                            }
                            MenuItem {
                                text: "Delete"
                                onTriggered: {
                                    deleteConvDialog.targetConv = modelData
                                    deleteConvDialog.open()
                                }
                            }
                        }
                        TapHandler { acceptedButtons: Qt.RightButton; onTapped: convMenu.popup() }
                    }
                }
            }
        }

        // Main chat area
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            // Toolbar: model selector, params, stats
            Rectangle {
                Layout.fillWidth: true
                height: 44
                color: AppTheme.bgAlt
                border.color: AppTheme.border
                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 8
                    spacing: 10
                    Label { text: "Model"; color: AppTheme.textDim }
                    AppComboBox {
                        id: modelSelector
                        Layout.preferredWidth: 280
                        model: page.chatModels()
                        textRole: "alias"
                        function currentModelId() {
                            var models = page.chatModels()
                            return currentIndex >= 0 && models.length ? models[currentIndex].id : ""
                        }
                        onActivated: function(i) {
                            var models = page.chatModels()
                            if (!page.currentConv || !models.length) return
                            var mid = models[i].id
                            page.api.patch("/api/v1/chat/" + page.currentConv.id,
                                { "model_id": mid }, function(st) {
                                    if (st === 200) {
                                        // Update the local copy immediately:
                                        // generation reads currentConv.model_id.
                                        page.currentConv.model_id = mid
                                        page.currentConvChanged()
                                        page.reload()
                                    }
                                })
                        }
                        Component.onCompleted: {
                            if (page.currentConv) {
                                var models = page.chatModels()
                                for (var i = 0; i < models.length; i++)
                                    if (models[i].id === page.currentConv.model_id) { currentIndex = i; break }
                            }
                        }
                        delegate: ItemDelegate {
                            width: modelSelector.width
                            contentItem: Row {
                                spacing: 6
                                Text { text: modelData.alias; color: AppTheme.text }
                                Text { text: modelData.quantization; color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall }
                            }
                        }
                    }
                    Tag {
                        text: page.loadingModel ? "Loading…" : page.modelState()
                        tone: page.loadingModel ? AppTheme.info
                            : page.modelState() === "Ready" ? AppTheme.success : AppTheme.warning
                    }
                    AppButton {
                        text: "Configure"
                        onClicked: paramsDrawer.open()
                    }
                    AppButton {
                        text: "System prompt"
                        enabled: page.currentConv !== null
                        onClicked: systemPromptDialog.openFor(page.currentConv)
                    }
                    AppButton {
                        visible: page.currentConv !== null && page.modelState() !== "Ready"
                        text: "Model settings"
                        onClicked: {
                            if (page.currentConv) page.configureModel(page.currentConv.model_id)
                        }
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        visible: page.lastStats !== null
                        text: page.lastStats
                            ? (page.lastStats.tokens_per_second || 0).toFixed(1) + " tok/s"
                              + " · TTFT " + (page.lastStats.ttft_seconds || 0).toFixed(2) + "s"
                              + " · " + (page.lastStats.prompt_tokens || 0) + "+" + (page.lastStats.completion_tokens || 0) + " tok"
                            : ""
                        color: AppTheme.textDim
                        font.pixelSize: AppTheme.fontSmall
                    }
                }
            }

            // Messages
            ListView {
                id: chatList
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                spacing: 10
                topMargin: 12
                bottomMargin: 12
                model: page.chain

                EmptyState {
                    visible: page.chain.length === 0
                    anchors.centerIn: parent
                    icon: "◎"
                    title: page.messagesLoading ? "Opening conversation…" : page.currentConv ? "Start the conversation" : "Select or create a chat"
                    hint: page.currentConv ? "Your selected model will load when you send the first message."
                        : page.chatModels().length ? "Create a chat with a local model, or browse models to get started."
                        : "No chat models yet. Embedding models load from Library and serve /v1/embeddings via Developer API — or browse for a chat GGUF."
                    actionText: page.currentConv ? "" : (page.chatModels().length ? "New chat" : "Browse models")
                    onActionTriggered: {
                        if (page.chatModels().length) page.newConversation()
                        else page.openLibrary()
                    }
                }

                delegate: ColumnLayout {
                    width: chatList.width - 48
                    x: 24
                    spacing: 2

                    // Sender row: name + branch indicator + actions
                    RowLayout {
                        Layout.fillWidth: true
                        layoutDirection: modelData.role === "user" ? Qt.RightToLeft : Qt.LeftToRight
                        Label {
                            text: modelData.role === "user" ? "You" : (page.currentConv ? modelName() : "Assistant")
                            function modelName() {
                                for (var i = 0; i < page.library.length; i++)
                                    if (page.library[i].id === page.currentConv.model_id) return page.library[i].alias
                                return "Assistant"
                            }
                            color: modelData.role === "user" ? AppTheme.accent : AppTheme.text
                            font.weight: Font.DemiBold
                            font.pixelSize: AppTheme.fontSmall
                        }
                        Label {
                            visible: page.siblingsOf(modelData).length > 0
                            text: "⑂ " + (1 + page.siblingsOf(modelData).length) + " branches"
                            color: AppTheme.textFaint
                            font.pixelSize: AppTheme.fontSmall
                            MouseArea { anchors.fill: parent; onClicked: branchSheet.openFor(modelData) }
                        }
                        Item { Layout.fillWidth: true }
                        AppButton {
                            flat: true; text: "Copy"
                            onClicked: { copyArea.text = modelData.content; copyArea.selectAll(); copyArea.copy() }
                        }
                        AppButton {
                            flat: true; visible: modelData.role === "assistant"
                            text: "Regenerate"
                            onClicked: page.regenerate(modelData)
                        }
                        AppButton {
                            flat: true; visible: modelData.role === "user"
                            text: "Edit"
                            onClicked: { editDialog.targetMsg = modelData; editDialog.open() }
                        }
                    }

                    // Bubble row: user right-aligned with avatar, assistant left.
                    RowLayout {
                        Layout.fillWidth: true
                        layoutDirection: modelData.role === "user" ? Qt.RightToLeft : Qt.LeftToRight
                        spacing: 8

                        // Avatar
                        Rectangle {
                            Layout.alignment: Qt.AlignTop
                            width: 28; height: 28; radius: 14
                            color: modelData.role === "user" ? AppTheme.accent : AppTheme.surfaceHi
                            border.color: modelData.id === page.streamingId ? AppTheme.accent
                                : modelData.role === "user" ? AppTheme.accent : AppTheme.border
                            SequentialAnimation on opacity {
                                running: modelData.id === page.streamingId
                                loops: Animation.Infinite
                                NumberAnimation { to: 0.45; duration: 700 }
                                NumberAnimation { to: 1.0; duration: 700 }
                            }
                            Text {
                                anchors.centerIn: parent
                                text: modelData.role === "user" ? "Y"
                                    : ((page.currentConv ? modelName() : "A").substring(0, 1).toUpperCase())
                                function modelName() {
                                    for (var i = 0; i < page.library.length; i++)
                                        if (page.library[i].id === page.currentConv.model_id) return page.library[i].alias
                                    return "Assistant"
                                }
                                color: modelData.role === "user" ? AppTheme.onAccent : AppTheme.textDim
                                font.weight: Font.Bold
                                font.pixelSize: 12
                            }
                        }

                        // Bubble
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.maximumWidth: chatList.width - 110
                            Layout.minimumHeight: bubbleCol.implicitHeight + 20
                            radius: AppTheme.radius
                            color: modelData.role === "user" ? Qt.alpha(AppTheme.accent, 0.14) : AppTheme.surface
                            border.color: modelData.role === "user" ? Qt.alpha(AppTheme.accent, 0.45) : AppTheme.border

                            ColumnLayout {
                                id: bubbleCol
                                anchors.fill: parent
                                anchors.margins: 10
                                spacing: 6

                                // In-bubble reasoning: collapsed preview by
                                // default, expandable with a chevron.
                                ColumnLayout {
                                    Layout.fillWidth: true
                                    visible: msgReasoning() !== ""
                                    function msgReasoning() {
                                        return modelData.id === page.streamingId
                                            ? page.streamReasoning : (modelData.reasoning || "")
                                    }
                                    spacing: 2
                                    Rectangle {
                                        Layout.fillWidth: true
                                        height: 1
                                        visible: page.reasoningExpanded(modelData.id)
                                        color: AppTheme.border
                                    }
                                    RowLayout {
                                        id: reasoningHeader
                                        Layout.fillWidth: true
                                        spacing: 6
                                        Label {
                                            text: (page.reasoningExpanded(modelData.id) ? "▾" : "▸") + " Reasoning"
                                            color: AppTheme.textFaint
                                            font.pixelSize: AppTheme.fontSmall
                                            font.weight: Font.Medium
                                        }
                                        Label {
                                            visible: !page.reasoningExpanded(modelData.id)
                                            Layout.fillWidth: true
                                            text: page.reasoningPreview(modelData.id === page.streamingId
                                                ? page.streamReasoning : (modelData.reasoning || ""))
                                            color: AppTheme.textFaint
                                            font.italic: true
                                            font.pixelSize: AppTheme.fontSmall
                                            elide: Text.ElideRight
                                        }
                                        Item { Layout.fillWidth: true }
                                        HoverHandler { cursorShape: Qt.PointingHandCursor }
                                        TapHandler {
                                            onTapped: page.toggleReasoning(modelData.id)
                                        }
                                    }
                                    Label {
                                        Layout.fillWidth: true
                                        visible: page.reasoningExpanded(modelData.id)
                                        text: (modelData.id === page.streamingId
                                            ? page.streamReasoning : modelData.reasoning) || ""
                                        wrapMode: Text.WordWrap
                                        color: AppTheme.textDim
                                        font.pixelSize: AppTheme.fontSmall
                                    }
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    visible: modelData.id === page.streamingId && page.streamContent === ""
                                    spacing: 6
                                    BusyIndicator { running: true; implicitWidth: 16; implicitHeight: 16 }
                                    Label {
                                        text: (modelData.id === page.streamingId ? page.streamReasoning : (modelData.reasoning || "")) !== "" ? "Thinking…" : "Waiting for first token…"
                                        color: AppTheme.textFaint
                                        font.italic: true
                                        font.pixelSize: AppTheme.fontSmall
                                    }
                                }

                                ChatMessageContent {
                                    Layout.fillWidth: true
                                    visible: (modelData.id === page.streamingId ? page.streamContent : modelData.content) !== ""
                                    content: modelData.id === page.streamingId ? page.streamContent : modelData.content
                                    streaming: modelData.id === page.streamingId
                                }

                                Label {
                                    Layout.fillWidth: true
                                    visible: (modelData.error || "") !== ""
                                    text: "⚠ " + modelData.error
                                    color: AppTheme.danger
                                    font.pixelSize: AppTheme.fontSmall
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }
                    }
                }
            }

            // Error bar
            Rectangle {
                Layout.fillWidth: true
                visible: page.errorText !== ""
                height: errLabel.implicitHeight + 16
                color: Qt.alpha(AppTheme.danger, 0.12)
                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 8
                    Label {
                        id: errLabel
                        Layout.fillWidth: true
                        text: page.errorText
                        color: AppTheme.danger
                        wrapMode: Text.WordWrap
                    }
                    AppButton { text: "Clear"; flat: true; onClicked: page.errorText = "" }
                }
            }

            // Composer — auto-grows with wrapped text, then scrolls internally.
            Rectangle {
                id: composerBar
                Layout.fillWidth: true
                Layout.preferredHeight: composerCol.implicitHeight + 20
                Layout.minimumHeight: 72
                color: AppTheme.bgAlt
                border.color: AppTheme.border

                ColumnLayout {
                    id: composerCol
                    // Size from content upward — do not anchors.fill, or the
                    // bar's preferredHeight and this column fight each other.
                    width: parent.width - 20
                    x: 10
                    y: 10
                    spacing: 8

                    RowLayout {
                        visible: page.pendingAudioPath !== ""
                        Layout.fillWidth: true
                        Label {
                            Layout.fillWidth: true
                            text: "Audio: " + page.pendingAudioName
                            color: AppTheme.textDim
                            elide: Text.ElideMiddle
                            font.pixelSize: AppTheme.fontSmall
                        }
                        AppButton {
                            text: "Remove"
                            flat: true
                            onClicked: { page.pendingAudioPath = ""; page.pendingAudioName = "" }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        AppButton {
                            visible: page.canAttachAudio()
                            Layout.alignment: Qt.AlignBottom
                            text: "Audio"
                            enabled: page.currentConv !== null && !page.generating
                            ToolTip.visible: audioTip.hovered
                            ToolTip.text: "Attach a WAV (16 kHz mono preferred). Experimental llama.cpp audio input."
                            HoverHandler { id: audioTip }
                            onClicked: audioDialog.open()
                        }

                        // TextArea is not scrollable alone — attach it to a
                        // Flickable that grows with content up to max height.
                        Flickable {
                            id: inputFlick
                            Layout.fillWidth: true
                            Layout.alignment: Qt.AlignBottom
                            Layout.preferredHeight: {
                                var minH = 44
                                var maxH = 220
                                var h = contentHeight
                                if (h < minH) h = minH
                                if (h > maxH) h = maxH
                                return h
                            }
                            Layout.maximumHeight: 220
                            contentWidth: width
                            contentHeight: Math.max(input.implicitHeight, 44)
                            clip: true
                            boundsBehavior: Flickable.StopAtBounds
                            flickableDirection: Flickable.VerticalFlick
                            interactive: contentHeight > height + 1

                            TextArea.flickable: AppTextArea {
                                id: input
                                enabled: page.currentConv !== null
                                wrapMode: TextArea.Wrap
                                selectByMouse: true
                                persistentSelection: true
                                placeholderText: page.currentConv
                                    ? "Message (Enter to send, Shift+Enter for newline)"
                                    : "Create a chat first"
                                background: Rectangle {
                                    radius: AppTheme.radiusSmall
                                    color: AppTheme.surface
                                    border.width: 1
                                    border.color: input.activeFocus ? AppTheme.accent : AppTheme.border
                                }
                                Keys.onReturnPressed: function(e) {
                                    if (e.modifiers & Qt.ShiftModifier) {
                                        e.accepted = false
                                        return
                                    }
                                    e.accepted = true
                                    page.send()
                                }
                                // Keep caret visible while typing past the fold.
                                onCursorRectangleChanged: {
                                    if (!inputFlick.interactive) return
                                    var y = cursorRectangle.y
                                    var bottom = y + cursorRectangle.height
                                    if (y < inputFlick.contentY)
                                        inputFlick.contentY = Math.max(0, y)
                                    else if (bottom > inputFlick.contentY + inputFlick.height)
                                        inputFlick.contentY = bottom - inputFlick.height
                                }
                            }

                            ScrollBar.vertical: ScrollBar {
                                policy: inputFlick.contentHeight > inputFlick.height
                                        ? ScrollBar.AsNeeded : ScrollBar.AlwaysOff
                            }
                        }

                        AppButton {
                            visible: !page.generating
                            Layout.alignment: Qt.AlignBottom
                            text: "Send"
                            primary: true
                            enabled: page.currentConv !== null
                                     && (input.text.trim() !== "" || page.pendingAudioPath !== "")
                            onClicked: page.send()
                        }
                        AppButton {
                            visible: page.generating
                            Layout.alignment: Qt.AlignBottom
                            text: "Stop"
                            danger: true
                            onClicked: {
                                if (page.currentConv)
                                    page.api.post("/api/v1/chat/" + page.currentConv.id + "/stop", {}, function() {})
                                page.generating = false
                            }
                        }
                    }
                }
            }
        }
    }

    FileDialog {
        id: audioDialog
        title: "Attach audio"
        fileMode: FileDialog.OpenFile
        nameFilters: ["Audio files (*.wav *.mp3 *.flac *.ogg *.m4a *.webm)", "All files (*)"]
        onAccepted: {
            var url = selectedFile.toString()
            var path = url.indexOf("file://") === 0 ? decodeURIComponent(url.replace("file://", "")) : url
            // Windows file:///C:/... — strip leading slash before drive letter.
            if (path.length >= 3 && path.charAt(0) === "/" && path.charAt(2) === ":")
                path = path.substring(1)
            page.pendingAudioPath = path
            page.pendingAudioName = path.split(/[/\\]/).pop()
        }
    }

    // Generation parameters drawer
    Drawer {
        id: paramsDrawer
        edge: Qt.RightEdge
        width: 340
        height: page.height
        background: Rectangle { color: AppTheme.bg; border.color: AppTheme.border }

        property var params: ({
            "temperature": 0.7, "top_p": 0.95, "top_k": 40, "min_p": 0.05,
            "repeat_penalty": 1.1, "max_tokens": 2048
        })

        function loadForConversation(conversation) {
            var defaults = {
                "temperature": 0.7, "top_p": 0.95, "top_k": 40, "min_p": 0.05,
                "repeat_penalty": 1.1, "max_tokens": 2048
            }
            var saved = conversation && conversation.params ? conversation.params : {}
            params = Object.assign({}, defaults, saved)
        }
        function updateParam(key, value) {
            var next = Object.assign({}, params)
            next[key] = value
            params = next
            saveParamsTimer.restart()
        }

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: AppTheme.pad
            spacing: AppTheme.gap
            Label { text: "Generation parameters"; font.pixelSize: AppTheme.fontTitle; color: AppTheme.text }

            Repeater {
                model: [
                    { "key": "temperature", "label": "Temperature", "hint": "Sampling randomness. 0 = deterministic.", "min": 0.0, "max": 2.0, "step": 0.05 },
                    { "key": "top_p", "label": "Top-p", "hint": "Nucleus sampling threshold.", "min": 0.0, "max": 1.0, "step": 0.01 },
                    { "key": "top_k", "label": "Top-k", "hint": "Candidate pool size. 0 disables.", "min": 0, "max": 200, "step": 1 },
                    { "key": "min_p", "label": "Min-p", "hint": "Minimum token probability relative to the best token.", "min": 0.0, "max": 1.0, "step": 0.01 },
                    { "key": "repeat_penalty", "label": "Repeat penalty", "hint": "Penalize repeated tokens. 1.0 = off.", "min": 0.5, "max": 2.0, "step": 0.05 },
                    { "key": "max_tokens", "label": "Max output tokens", "hint": "Generation cap for one response.", "min": 16, "max": 65536, "step": 16 }
                ]
                delegate: FormField {
                    Layout.fillWidth: true
                    label: modelData.label
                    hint: modelData.hint
                    Row {
                        spacing: 8
                        AppSlider {
                            id: slider
                            width: 180
                            from: modelData.min; to: modelData.max; stepSize: modelData.step
                            value: paramsDrawer.params[modelData.key]
                            onMoved: paramsDrawer.updateParam(modelData.key, value)
                        }
                        Label {
                            text: Number(slider.value).toFixed(modelData.step < 1 ? 2 : 0)
                            color: AppTheme.textDim
                            width: 48
                        }
                    }
                }
            }
            Item { Layout.fillHeight: true }
            Label {
                text: page.currentConv ? "Saved automatically for this chat." : "Create a chat to save parameters."
                color: AppTheme.textFaint
                font.pixelSize: AppTheme.fontSmall
            }
        }
    }

    Timer {
        id: saveParamsTimer
        interval: 400
        onTriggered: {
            if (!page.currentConv) return
            page.api.patch("/api/v1/chat/" + page.currentConv.id,
                { "params": paramsDrawer.params }, function(st) {
                    if (st === 200) page.currentConv.params = paramsDrawer.params
                })
        }
    }

    AppDialog {
        id: systemPromptDialog
        property var targetConv: null
        title: "System prompt"
        modal: true
        anchors.centerIn: page
        width: Math.min(560, page.width - 48)
        standardButtons: Dialog.Save | Dialog.Cancel
        function openFor(conversation) {
            targetConv = conversation
            systemPromptArea.text = conversation ? (conversation.system || "") : ""
            open()
        }
        ColumnLayout {
            width: parent.width
            spacing: AppTheme.gapTight
            Label {
                Layout.fillWidth: true
                text: "Set the behavior and context this chat should keep for every response."
                color: AppTheme.textDim
                wrapMode: Text.WordWrap
            }
            AppTextArea {
                id: systemPromptArea
                Layout.fillWidth: true
                Layout.preferredHeight: 180
                wrapMode: TextArea.Wrap
                selectByMouse: true
                placeholderText: "You are a helpful local assistant…"
            }
        }
        onAccepted: if (targetConv) page.api.patch("/api/v1/chat/" + targetConv.id,
            { "system": systemPromptArea.text }, function(st) {
                if (st === 200) {
                    targetConv.system = systemPromptArea.text
                    page.currentConvChanged()
                    page.reload()
                }
            })
    }

    // Rename dialog
    AppDialog {
        id: renameDialog
        property var targetConv: null
        title: "Rename chat"
        modal: true
        anchors.centerIn: page
        standardButtons: Dialog.Save | Dialog.Cancel
        AppTextField { id: renameField; width: 300; text: renameDialog.targetConv ? renameDialog.targetConv.title : "" }
        onAccepted: if (targetConv) page.api.patch("/api/v1/chat/" + targetConv.id,
            { "title": renameField.text }, function() { page.reload() })
    }

    ConfirmDialog {
        id: deleteConvDialog
        property var targetConv: null
        message: "Delete this conversation permanently?"
        confirmText: "Delete"
        onConfirmed: page.api.del("/api/v1/chat/" + targetConv.id, function() {
            if (page.currentConv && page.currentConv.id === targetConv.id) {
                page.currentConv = null
                page.chain = []
            }
            page.reload()
        })
    }

    // Edit user message → branches the conversation.
    AppDialog {
        id: editDialog
        property var targetMsg: null
        title: "Edit message (creates a branch)"
        modal: true
        anchors.centerIn: page
        width: 480
        standardButtons: Dialog.Save | Dialog.Cancel
        AppTextArea {
            id: editArea
            width: 440
            height: 160
            text: editDialog.targetMsg ? editDialog.targetMsg.content : ""
            wrapMode: TextArea.Wrap
        }
        onAccepted: if (targetMsg) {
            if (page.generating) return
            page.generating = true
            page.startWhenModelReady(
                { "parent_id": targetMsg.parent_id, "content": editArea.text, "params": paramsDrawer.params })
        }
    }

    // Branch picker
    AppDialog {
        id: branchSheet
        property var branches: []
        function openFor(msg) {
            branches = page.siblingsOf(msg).concat([msg])
            open()
        }
        title: "Conversation branches"
        modal: true
        anchors.centerIn: page
        width: 420
        ListView {
            width: parent.width
            implicitHeight: Math.min(300, contentHeight)
            model: branchSheet.branches
            delegate: ItemDelegate {
                width: parent.width
                text: modelData.content.substring(0, 80)
                onClicked: {
                    page.chain = page.buildChain(modelData.id)
                    branchSheet.close()
                }
            }
        }
    }

    TextEdit { id: copyArea; visible: false }

    Component.onCompleted: reload()
}
