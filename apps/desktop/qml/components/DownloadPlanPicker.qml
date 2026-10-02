import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".."

// Component picker for a Hugging Face repository.
//
// The repository is a list of components (the model, a vision projector, a
// drafter, a VAE, text encoders, …). Each one can be left out with its
// checkbox and, when the repository stores it in several precisions, has a
// dropdown to pick one. A single confirm downloads exactly what is ticked.
// Everything it shows comes from the backend's download plan
// (GET /api/v1/hf/repo/{repo} → plan).
Item {
    id: root

    // Plan object from the API; null until a repository is loaded.
    property var plan: null
    property bool experimentalAudio: false
    property bool showFilePaths: false

    // Component id → { on: bool, opt: option id, touched: bool }. Always
    // replaced, never edited in place, so bindings re-evaluate.
    property var selection: ({})

    // Emitted by the Download button: { group, files, summary }.
    signal confirmed(var request)

    implicitHeight: layout.implicitHeight

    // ---- plan access ------------------------------------------------------

    function comps() {
        return root.plan && root.plan.components ? root.plan.components : []
    }

    function optsOf(c) {
        return c && c.options ? c.options : []
    }

    function optById(c, id) {
        var o = root.optsOf(c)
        for (var i = 0; i < o.length; i++)
            if (o[i].id === id) return o[i]
        return null
    }

    function optIndex(c) {
        var s = root.selection[c.id]
        var o = root.optsOf(c)
        if (!s) return -1
        for (var i = 0; i < o.length; i++)
            if (o[i].id === s.opt) return i
        return -1
    }

    function current(c) {
        var s = root.selection[c.id]
        return s ? root.optById(c, s.opt) : null
    }

    function isOn(c) {
        var s = root.selection[c.id]
        return !!(s && s.on)
    }

    function compById(id) {
        var cs = root.comps()
        for (var i = 0; i < cs.length; i++)
            if (cs[i].id === id) return cs[i]
        return null
    }

    // ---- selection --------------------------------------------------------

    function copySelection() {
        var out = {}
        for (var k in root.selection)
            out[k] = { on: root.selection[k].on, opt: root.selection[k].opt, touched: root.selection[k].touched }
        return out
    }

    // Balanced defaults from the backend. Experimental parts stay off until
    // the matching feature is enabled in Settings.
    function reset() {
        var sel = {}
        var cs = root.comps()
        for (var i = 0; i < cs.length; i++) {
            var c = cs[i]
            var on = !!c.selected
            if (c.experimental && !root.experimentalAudio) on = false
            sel[c.id] = { on: on, opt: c.default, touched: false }
        }
        root.selection = sel
    }

    onPlanChanged: reset()
    onExperimentalAudioChanged: reset()

    function setOn(c, on) {
        var sel = root.copySelection()
        sel[c.id].on = on
        root.selection = sel
    }

    function setOption(c, optId) {
        var sel = root.copySelection()
        sel[c.id].opt = optId
        sel[c.id].touched = true
        root.selection = sel
        root.followAll()
    }

    // A component that follows another (the drafter follows the model) tracks
    // that component's precision until the user picks one for it by hand.
    function followAll() {
        var cs = root.comps()
        var sel = null
        for (var i = 0; i < cs.length; i++) {
            var d = cs[i]
            if (!d.follow_precision_of || !root.selection[d.id] || root.selection[d.id].touched) continue
            var src = root.compById(d.follow_precision_of)
            var srcOpt = src ? root.current(src) : null
            if (!srcOpt) continue
            var pick = root.followPick(d, srcOpt)
            if (pick && pick.id !== root.selection[d.id].opt) {
                if (!sel) sel = root.copySelection()
                sel[d.id].opt = pick.id
            }
        }
        if (sel) root.selection = sel
    }

    function followPick(d, srcOpt) {
        var o = root.optsOf(d)
        var cur = root.current(d)
        var same = []
        for (var i = 0; i < o.length; i++)
            if (srcOpt.precision !== "" && o[i].precision === srcOpt.precision) same.push(o[i])
        if (same.length === 0) {
            var best = null
            for (var j = 0; j < o.length; j++)
                if (!best || Math.abs(o[j].bits - srcOpt.bits) < Math.abs(best.bits - srcOpt.bits)) best = o[j]
            return best
        }
        // Keep the same kind of drafter (DFlash, EAGLE3, …) when it exists.
        if (cur && cur.kind)
            for (var k = 0; k < same.length; k++)
                if (same[k].kind === cur.kind) return same[k]
        return same[0]
    }

    // Presets move every component's precision, never to a different build:
    // when a component holds several models (Wan 14B and 1.3B), the preset
    // stays with the one picked. What is ticked stays too.
    function applyPreset(kind) {
        var cs = root.comps()
        var sel = root.copySelection()
        for (var i = 0; i < cs.length; i++) {
            var c = cs[i]
            if (!sel[c.id]) continue
            var pool = root.sameVariant(c, root.current(c))
            var pick = null
            if (kind === "balanced") {
                var rec = root.optById(c, c.default)
                pick = root.nearest(pool, rec)
            } else {
                pick = root.extreme(c, kind, pool)
            }
            if (pick) sel[c.id].opt = pick.id
            sel[c.id].touched = false
        }
        root.selection = sel
        root.followAll()
    }

    // Options of the same model as `cur` (all of them when there is only one).
    function sameVariant(c, cur) {
        var o = root.optsOf(c)
        if (!cur || !cur.variant) return o
        var same = o.filter(function(x) { return x.variant === cur.variant })
        return same.length > 0 ? same : o
    }

    // The option in pool nearest `target` in bits (the target itself when it
    // is in the pool); ties go to the smaller file.
    function nearest(pool, target) {
        if (!target) return pool.length > 0 ? pool[0] : null
        var best = null
        for (var i = 0; i < pool.length; i++) {
            var x = pool[i]
            if (!best) { best = x; continue }
            var dx = Math.abs(x.bits - target.bits), db = Math.abs(best.bits - target.bits)
            if (x.id === target.id || dx < db || (dx === db && best.id !== target.id && x.total_bytes < best.total_bytes))
                best = x
        }
        return best
    }

    // "smallest" / "best" option of a pool, skipping options that are
    // unlikely to load while a sound one exists and never choosing 32-bit
    // floats when a 16-bit-or-smaller option exists.
    function extreme(c, kind, opts) {
        var o = opts || root.optsOf(c)
        var pool = o.filter(function(x) { return !x.warn })
        if (pool.length === 0) pool = o
        if (kind === "best") {
            var small = pool.filter(function(x) { return x.bits <= 16 })
            if (small.length > 0) pool = small
        }
        var best = null
        for (var i = 0; i < pool.length; i++) {
            var x = pool[i]
            if (!best) { best = x; continue }
            var better = kind === "smallest" ? x.bits < best.bits : x.bits > best.bits
            if (x.bits === best.bits) {
                // Same precision width: the smaller file, then FP16 over BF16
                // (the reference format every backend accelerates).
                better = x.total_bytes < best.total_bytes
                    || (x.total_bytes === best.total_bytes && x.precision === "fp16" && best.precision !== "fp16")
            }
            if (better) best = x
        }
        return best
    }

    // ---- results ----------------------------------------------------------

    function totals() {
        var cs = root.comps()
        var t = { bytes: 0, files: 0, n: 0, of: cs.length }
        for (var i = 0; i < cs.length; i++) {
            if (!root.isOn(cs[i])) continue
            var o = root.current(cs[i])
            if (!o) continue
            t.n++
            t.bytes += o.total_bytes
            t.files += o.files.length
        }
        return t
    }

    // Folder name for the download: the first ticked component's option.
    function groupId() {
        var cs = root.comps()
        for (var i = 0; i < cs.length; i++) {
            var o = root.isOn(cs[i]) ? root.current(cs[i]) : null
            if (o) return o.id
        }
        return ""
    }

    function summary() {
        var cs = root.comps()
        var parts = []
        for (var i = 0; i < cs.length; i++) {
            var o = root.isOn(cs[i]) ? root.current(cs[i]) : null
            if (!o) continue
            var text = o.label + (o.variant ? " " + o.variant : "")
            parts.push(cs[i].short ? cs[i].short + " " + text : text)
        }
        return parts.join(" + ")
    }

    function buildRequest() {
        var cs = root.comps()
        var files = []
        var seen = {}
        for (var i = 0; i < cs.length; i++) {
            var o = root.isOn(cs[i]) ? root.current(cs[i]) : null
            if (!o) continue
            for (var j = 0; j < o.files.length; j++) {
                var f = o.files[j]
                if (seen[f.path]) continue
                seen[f.path] = true
                var entry = { "path": f.path, "size": f.size }
                if (f.dest) entry["dest"] = f.dest
                files.push(entry)
            }
        }
        return { "group": root.groupId(), "files": files, "summary": root.summary() }
    }

    function optionText(o) {
        var t = o.label
        if (o.variant) t += "  ·  " + o.variant
        t += "  —  " + AppTheme.bytes(o.total_bytes)
        if (o.recommended) t += "  ★"
        if (o.warn) t += "  ⚠"
        return t
    }

    function optionSubtitle(o) {
        var parts = []
        for (var i = 0; o.tags && i < o.tags.length; i++) parts.push(o.tags[i])
        if (o.recommended) parts.push("recommended")
        if (o.warn) parts.push("may not load")
        return parts.join(" · ")
    }

    function comboModel(c) {
        return root.optsOf(c).map(function(o) {
            return { "text": root.optionText(o), "subtitle": root.optionSubtitle(o), "value": o.id }
        })
    }

    function tagTone(tag) {
        if (tag === "UD") return AppTheme.warning
        if (tag === "OID") return AppTheme.accent
        return AppTheme.info
    }

    ColumnLayout {
        id: layout
        anchors.fill: parent
        spacing: AppTheme.gap

        RowLayout {
            Layout.fillWidth: true
            spacing: 6
            visible: root.comps().length > 0
            Label {
                text: "Precision"
                color: AppTheme.textDim
                font.pixelSize: AppTheme.fontSmall
            }
            AppButton {
                text: "Smallest"
                flat: true
                onClicked: root.applyPreset("smallest")
                ToolTip.visible: hovered
                ToolTip.text: "Lowest precision that should load, for every component"
            }
            AppButton {
                text: "Balanced"
                flat: true
                onClicked: root.applyPreset("balanced")
                ToolTip.visible: hovered
                ToolTip.text: "The recommended pick for each component"
            }
            AppButton {
                text: "Best"
                flat: true
                onClicked: root.applyPreset("best")
                ToolTip.visible: hovered
                ToolTip.text: "Highest quality that fits in 16 bits, for every component"
            }
            Item { Layout.fillWidth: true }
            AppCheckBox {
                text: "Show file paths"
                checked: root.showFilePaths
                onToggled: root.showFilePaths = checked
            }
        }

        Repeater {
            model: root.plan && root.plan.notes ? root.plan.notes : []
            Label {
                Layout.fillWidth: true
                text: modelData
                color: AppTheme.warning
                font.pixelSize: AppTheme.fontSmall
                wrapMode: Text.WordWrap
            }
        }

        EmptyState {
            visible: root.plan !== null && root.comps().length === 0
            Layout.fillWidth: true
            Layout.preferredHeight: 120
            icon: "∅"
            title: "Nothing to download"
            hint: "No model files this app can use were found in this repository."
        }

        ListView {
            id: list
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 120
            clip: true
            spacing: 8
            model: root.comps()
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

            delegate: Card {
                id: card
                readonly property var comp: modelData
                readonly property var opt: root.current(comp)
                readonly property bool on: root.isOn(comp)

                width: ListView.view.width - 8
                implicitHeight: col.implicitHeight + 20
                hoverHighlight: false
                opacity: on ? 1 : 0.62
                Behavior on opacity { NumberAnimation { duration: AppTheme.motionFast } }

                ColumnLayout {
                    id: col
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 6

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 10
                        AppCheckBox {
                            id: includeBox
                            text: card.comp.label
                            font.weight: Font.DemiBold
                            checked: card.on
                            onToggled: root.setOn(card.comp, checked)
                        }
                        Tag {
                            visible: !!card.comp.experimental
                            text: "experimental"
                            tone: AppTheme.warning
                            Layout.minimumWidth: implicitWidth
                        }
                        Item { Layout.fillWidth: true }
                        AppComboBox {
                            id: optionCombo
                            enabled: card.on
                            implicitWidth: 360
                            popupMinWidth: 420
                            subtitleRole: "subtitle"
                            textRole: "text"
                            valueRole: "value"
                            model: root.comboModel(card.comp)
                            currentIndex: root.optIndex(card.comp)
                            onActivated: function(index) {
                                root.setOption(card.comp, card.comp.options[index].id)
                                // The user's pick replaced the binding; restore it
                                // so presets and "follow" updates still show.
                                currentIndex = Qt.binding(function() { return root.optIndex(card.comp) })
                            }
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        visible: text !== ""
                        text: card.comp.hint || ""
                        color: AppTheme.textDim
                        font.pixelSize: AppTheme.fontSmall
                        wrapMode: Text.WordWrap
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        visible: card.opt !== null
                        spacing: 6
                        Repeater {
                            model: card.opt && card.opt.tags ? card.opt.tags : []
                            Tag {
                                text: modelData
                                tone: root.tagTone(modelData)
                                Layout.minimumWidth: implicitWidth
                            }
                        }
                        Label {
                            visible: card.opt !== null && card.opt.est_memory_bytes > 0
                            text: card.opt ? "Estimated memory: ~" + AppTheme.bytes(card.opt.est_memory_bytes) + " (estimate)" : ""
                            color: AppTheme.textFaint
                            font.pixelSize: AppTheme.fontSmall
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: card.opt ? AppTheme.bytes(card.opt.total_bytes) : ""
                            color: AppTheme.textDim
                            font.pixelSize: AppTheme.fontSmall
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        visible: card.on && card.opt !== null && !!card.opt.warn
                        text: card.opt ? "⚠ " + card.opt.warn : ""
                        color: AppTheme.warning
                        font.pixelSize: AppTheme.fontSmall
                        wrapMode: Text.WordWrap
                    }

                    Repeater {
                        model: root.showFilePaths && card.opt ? card.opt.files : []
                        Label {
                            text: "  " + modelData.path + "  ·  " + AppTheme.bytes(modelData.size)
                            color: AppTheme.textDim
                            font.pixelSize: AppTheme.fontSmall
                            font.family: "monospace"
                            elide: Text.ElideMiddle
                            Layout.fillWidth: true
                        }
                    }
                }
            }
        }

        Label {
            Layout.fillWidth: true
            visible: {
                var m = root.compById("model")
                if (!m || root.isOn(m)) return false
                return root.totals().n > 0
            }
            text: "The model itself is not ticked: the other files are saved on their own, not next to a model."
            color: AppTheme.textFaint
            font.pixelSize: AppTheme.fontSmall
            wrapMode: Text.WordWrap
        }

        RowLayout {
            Layout.fillWidth: true
            visible: root.comps().length > 0
            spacing: AppTheme.gap
            Label {
                id: totalLabel
                readonly property var t: root.totals()
                text: t.n + " of " + t.of + " parts  ·  " + t.files + (t.files === 1 ? " file" : " files")
                    + "  ·  " + AppTheme.bytes(t.bytes)
                color: AppTheme.text
                font.weight: Font.DemiBold
            }
            Item { Layout.fillWidth: true }
            AppButton {
                text: "Download"
                primary: true
                enabled: totalLabel.t.n > 0
                onClicked: root.confirmed(root.buildRequest())
            }
        }
    }
}
