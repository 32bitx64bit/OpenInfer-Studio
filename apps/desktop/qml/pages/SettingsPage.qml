import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".."
import "../components"

Item {
    id: page
    property var api
    property var events

    property var settings: ({})
    property var hardware: null
    property var recommendation: null
    property var directories: []
    property bool hfConfigured: false
    property string statusText: ""
    property var appStatus: null

    signal settingChanged(string key, string value)
    signal replayOnboarding()

    function reload() {
        api.get("/api/v1/status", function(st, data) {
            if (st === 200) page.appStatus = data
        })
        api.get("/api/v1/settings", function(st, data) {
            if (st === 200) {
                page.settings = data || {}
                var theme = page.settings["ui.theme"] || "system"
                AppTheme.mode = theme
            }
        })
        api.get("/api/v1/hardware", function(st, data) {
            if (st === 200) {
                page.hardware = data.hardware
                page.recommendation = data.recommendation
            }
        })
        api.get("/api/v1/directories", function(st, data) {
            if (st === 200) page.directories = (data && data.directories) || []
        })
        api.get("/api/v1/hf/token", function(st, data) {
            if (st === 200) page.hfConfigured = data && data.configured
        })
    }

    function setSetting(key, value) {
        var v = String(value)
        api.put("/api/v1/settings/" + key, { "value": v }, function(st, data) {
            if (st === 200) {
                var next = Object.assign({}, page.settings)
                next[key] = v
                page.settings = next
                page.settingChanged(key, v)
                page.statusText = "Saved."
                statusClear.restart()
            }
        })
    }
    Timer { id: statusClear; interval: 2500; onTriggered: page.statusText = "" }

    ScrollView {
        anchors.fill: parent
        anchors.margins: AppTheme.pad
        clip: true
        ColumnLayout {
            width: page.width - AppTheme.pad * 2
            spacing: AppTheme.gap * 1.5

            PageHeader {
                title: "Settings"
                subtitle: "Control appearance, storage, connections, and advanced behavior."
            }
            Label {
                visible: page.statusText !== ""
                text: page.statusText
                color: AppTheme.success
                font.pixelSize: AppTheme.fontSmall
            }

            // Appearance
            AppGroupBox {
                Layout.fillWidth: true
                title: "Appearance"
                FormField {
                    width: parent.width
                    label: "Theme"
                    hint: "System follows the OS appearance."
                    AppComboBox {
                        model: ["system", "dark", "light"]
                        currentIndex: Math.max(0, model.indexOf(page.settings["ui.theme"] || "system"))
                        onActivated: function(i) {
                            AppTheme.mode = model[i]
                            page.setSetting("ui.theme", model[i])
                        }
                    }
                }
            }

            // Hardware
            AppGroupBox {
                Layout.fillWidth: true
                title: "Hardware"
                ColumnLayout {
                    width: parent.width
                    spacing: 6
                    visible: page.hardware !== null
                    GridLayout {
                        columns: 2
                        columnSpacing: 20
                        rowSpacing: 4
                        width: parent.width
                        Repeater {
                            model: page.hardware ? [
                                ["OS", page.hardware.os + " " + (page.hardware.os_version || "")],
                                ["CPU", page.hardware.cpu_model + " (" + page.hardware.logical_cores + " threads)"],
                                ["Features", (page.hardware.cpu_features || []).join(", ")],
                                ["RAM", AppTheme.bytes(page.hardware.ram_total) + " total, " + AppTheme.bytes(page.hardware.ram_available) + " available"],
                                ["GPUs", (page.hardware.gpus || []).map(function(g) { return g.name }).join("; ") || "none detected"],
                                ["Vulkan / CUDA / HIP / Metal", [page.hardware.vulkan ? "Vulkan" : "", page.hardware.cuda ? "CUDA" : "", page.hardware.hip ? "HIP" : "", page.hardware.metal ? "Metal" : ""].filter(function(s){return s!==""}).join(" · ") || "none"],
                                ["Free disk (models)", AppTheme.bytes(page.hardware.disk_free_models)],
                                ["Free disk (runtimes)", AppTheme.bytes(page.hardware.disk_free_runtimes)]
                            ] : []
                            delegate: Row {
                                spacing: 8
                                Label { text: modelData[0]; color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall; width: 160 }
                                Label { text: modelData[1]; color: AppTheme.textDim; font.pixelSize: AppTheme.fontSmall }
                            }
                        }
                    }
                    Label {
                        visible: page.recommendation !== null
                        Layout.fillWidth: true
                        text: page.recommendation ? "Recommended backend: " + page.recommendation.backend.toUpperCase()
                            + " — " + page.recommendation.reason : ""
                        color: AppTheme.accent
                        wrapMode: Text.WordWrap
                    }
                    RowLayout {
                        AppButton {
                            text: "Refresh"
                            onClicked: page.api.get("/api/v1/hardware?refresh=1", function(st, data) {
                                if (st === 200) { page.hardware = data.hardware; page.recommendation = data.recommendation }
                            })
                        }
                        AppButton {
                            text: "Copy report"
                            flat: true
                            onClicked: {
                                clip.text = JSON.stringify({ "hardware": page.hardware, "recommendation": page.recommendation }, null, 2)
                                clip.selectAll(); clip.copy()
                            }
                        }
                    }
                }
            }

            // Hugging Face token
            AppGroupBox {
                Layout.fillWidth: true
                title: "Connections · Hugging Face"
                ColumnLayout {
                    width: parent.width
                    spacing: 6
                    Label {
                        Layout.fillWidth: true
                        text: "A token is only needed for gated or private repositories. It is stored in the operating system credential vault, never in plain text."
                        color: AppTheme.textDim
                        wrapMode: Text.WordWrap
                    }
                    RowLayout {
                        spacing: 8
                        AppTextField {
                            id: tokenField
                            Layout.fillWidth: true
                            echoMode: TextInput.Password
                            placeholderText: page.hfConfigured ? "Token configured (enter to replace)" : "hf_…"
                        }
                        AppButton {
                            text: "Save token"
                            onClicked: page.api.put("/api/v1/hf/token", { "token": tokenField.text }, function(st) {
                                if (st === 200) { page.hfConfigured = true; tokenField.text = "" }
                            })
                        }
                        AppButton {
                            text: "Remove"
                            visible: page.hfConfigured
                            onClicked: page.api.del("/api/v1/hf/token", function() { page.hfConfigured = false })
                        }
                    }
                }
            }

            // Limits
            AppGroupBox {
                Layout.fillWidth: true
                title: "Advanced behavior"
                GridLayout {
                    columns: 2
                    width: parent.width
                    columnSpacing: 20
                    FormField {
                        label: "Concurrent downloads"
                        hint: "1–8 simultaneous downloads."
                        AppSpinBox {
                            from: 1; to: 8
                            value: parseInt(page.settings["downloads.concurrency"] || "2")
                            onValueModified: page.setSetting("downloads.concurrency", value)
                        }
                    }
                    FormField {
                        label: "Connections per file"
                        hint: "Parallel Range streams for large files (speeds up Hugging Face). 1 = single stream."
                        AppSpinBox {
                            from: 1; to: 16
                            value: parseInt(page.settings["downloads.connections"] || "8")
                            onValueModified: page.setSetting("downloads.connections", value)
                        }
                    }
                    FormField {
                        label: "Max loaded models"
                        hint: "Simultaneously loaded models (1–32)."
                        AppSpinBox {
                            from: 1; to: 32
                            value: parseInt(page.settings["instances.max_loaded"] || "8")
                            onValueModified: page.setSetting("instances.max_loaded", value)
                        }
                    }
                    FormField {
                        label: "Model startup timeout (s)"
                        hint: "How long to wait for a model to become ready."
                        AppSpinBox {
                            from: 30; to: 3600; stepSize: 30
                            value: parseInt(page.settings["instances.startup_timeout_sec"] || "600")
                            onValueModified: page.setSetting("instances.startup_timeout_sec", value)
                        }
                    }
                    FormField {
                        label: "Stream responses"
                        hint: "Show tokens as they generate. Disable to wait for the full reply."
                        AppSwitch {
                            checked: (page.settings["chat.streaming"] || "1") !== "0"
                            onToggled: page.setSetting("chat.streaming", checked ? "1" : "0")
                        }
                    }
                    FormField {
                        label: "Runtime update checks"
                        hint: "Check llama.cpp releases when opening the Runtimes page."
                        AppSwitch {
                            checked: (page.settings["runtimes.update_checks"] || "1") === "1"
                            onToggled: page.setSetting("runtimes.update_checks", checked ? "1" : "0")
                        }
                    }
                    FormField {
                        label: "Filter draft model picker"
                        hint: "When on (default), the load dialog only lists detected speculative drafts (mtp-, gemma4-assistant, eagle3-, dflash-, dspark-, …). Turn off to pick any library GGUF as a draft-simple companion."
                        AppSwitch {
                            checked: (page.settings["load.filter_incompatible_drafts"] || "1") !== "0"
                            onToggled: page.setSetting("load.filter_incompatible_drafts", checked ? "1" : "0")
                        }
                    }
                }
            }

            // Experimental features (upstream llama.cpp audio is experimental)
            AppGroupBox {
                Layout.fillWidth: true
                title: "Advanced · Experimental"
                FormField {
                    width: parent.width
                    label: "Audio models"
                    hint: "Enable audio / multimodal-audio discovery, labeling, and chat attachments. llama.cpp audio input via libmtmd is experimental and may have reduced quality; use a recent multimodal-capable runtime."
                    AppSwitch {
                        checked: (page.settings["experimental.audio_models"] || "0") === "1"
                        onToggled: page.setSetting("experimental.audio_models", checked ? "1" : "0")
                    }
                }
            }

            // Model directories
            AppGroupBox {
                Layout.fillWidth: true
                title: "Storage · Model directories"
                ColumnLayout {
                    width: parent.width
                    spacing: 4
                    Repeater {
                        model: page.directories
                        delegate: RowLayout {
                            Layout.fillWidth: true
                            Label {
                                text: modelData.path + (modelData.managed ? "  (managed)" : "")
                                color: AppTheme.textDim
                                font.family: "monospace"
                                font.pixelSize: AppTheme.fontSmall
                                elide: Text.ElideMiddle
                                Layout.fillWidth: true
                            }
                            AppButton {
                                visible: !modelData.managed
                                text: "Remove"
                                flat: true
                                onClicked: page.api.del("/api/v1/directories/" + modelData.id, function() { page.reload() })
                            }
                        }
                    }
                    RowLayout {
                        AppTextField { id: dirField; Layout.fillWidth: true; placeholderText: "/path/to/models" }
                        AppButton {
                            text: "Add directory"
                            onClicked: page.api.post("/api/v1/directories", { "path": dirField.text }, function(st, data) {
                                if (st !== 201) page.statusText = (data && (data.detail || data.error)) || "failed"
                                dirField.text = ""
                                page.reload()
                            })
                        }
                    }
                }
            }

            // About / version (single source: internal/version/VERSION via API)
            AppGroupBox {
                Layout.fillWidth: true
                title: "About"
                ColumnLayout {
                    width: parent.width
                    spacing: 6
                    Label {
                        text: "OpenInfer Studio"
                        color: AppTheme.text
                        font.weight: Font.DemiBold
                    }
                    AppButton {
                        text: "Show setup guide"
                        flat: true
                        onClicked: page.replayOnboarding()
                    }
                    GridLayout {
                        columns: 2
                        columnSpacing: 20
                        rowSpacing: 4
                        width: parent.width
                        Repeater {
                            model: {
                                var uiVer = (typeof appVersion !== "undefined" && appVersion) ? appVersion : ""
                                var coreVer = page.appStatus ? (page.appStatus.version || "") : ""
                                var commit = page.appStatus ? (page.appStatus.commit || "") : ""
                                var built = page.appStatus ? (page.appStatus.date || "") : ""
                                var plat = page.appStatus
                                    ? ((page.appStatus.goos || "") + "/" + (page.appStatus.goarch || ""))
                                    : ""
                                return [
                                    ["App version", uiVer || coreVer || "—"],
                                    ["Backend version", coreVer || "—"],
                                    ["Commit", commit && commit !== "dev" ? commit : (commit || "—")],
                                    ["Built", built && built !== "unknown" ? built : "—"],
                                    ["Platform", plat || "—"],
                                    ["Data directory", page.appStatus ? (page.appStatus.data_dir || "—") : "—"]
                                ]
                            }
                            delegate: Row {
                                spacing: 8
                                Label { text: modelData[0]; color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall; width: 140 }
                                Label {
                                    text: String(modelData[1])
                                    color: AppTheme.textDim
                                    font.pixelSize: AppTheme.fontSmall
                                    font.family: (modelData[0] === "Data directory" || modelData[0] === "Commit")
                                        ? "monospace" : Qt.application.font.family
                                    elide: Text.ElideMiddle
                                    width: Math.min(420, page.width - AppTheme.pad * 4 - 160)
                                }
                            }
                        }
                    }
                    Label {
                        Layout.fillWidth: true
                        text: "Version information is used for diagnostics and future update checks."
                        color: AppTheme.textFaint
                        font.pixelSize: AppTheme.fontSmall
                        wrapMode: Text.WordWrap
                    }
                }
            }
        }
    }

    TextEdit { id: clip; visible: false }
    Component.onCompleted: reload()
}
