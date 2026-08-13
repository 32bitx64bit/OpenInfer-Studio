pragma Singleton
import QtQuick

// OpenInfer Studio theme. Semantic tokens keep a calm, focused interface
// consistent across light, dark and system appearances.
QtObject {
    // mode: "system" | "dark" | "light"
    property string mode: "system"
    readonly property bool dark: mode === "dark"
        || (mode === "system" && Qt.styleHints.colorScheme === Qt.ColorScheme.Dark)

    // Surfaces
    readonly property color bg:        dark ? "#0f1419" : "#f6f8fa"
    readonly property color bgAlt:     dark ? "#151c23" : "#eef2f5"
    readonly property color surface:   dark ? "#1a232c" : "#ffffff"
    readonly property color surfaceHi: dark ? "#25313c" : "#f1f5f7"
    readonly property color surfaceHover: dark ? "#202b35" : "#e8f0f1"
    readonly property color surfaceSelected: dark ? "#173a38" : "#dff3ef"
    readonly property color border:    dark ? "#2d3a46" : "#d5dfe4"
    readonly property color borderFocus: dark ? "#4dd8c8" : "#0d8a7d"
    readonly property color overlay:   dark ? "#cc000000" : "#66000000"

    // Text
    readonly property color text:      dark ? "#e6edf3" : "#1a2330"
    readonly property color textDim:   dark ? "#9aa8b5" : "#55677a"
    readonly property color textFaint: dark ? "#67727f" : "#8fa0b0"

    // Accents (original OpenInfer identity: teal/cyan on dark)
    readonly property color accent:    dark ? "#35c4b5" : "#0d8a7d"
    readonly property color accentHi:  dark ? "#4dd8c8" : "#0a6e64"
    readonly property color onAccent:  dark ? "#06251f" : "#ffffff"

    // Status
    readonly property color success:   dark ? "#4cc38a" : "#18794e"
    readonly property color warning:   dark ? "#e5b45a" : "#a06a00"
    readonly property color danger:    dark ? "#e06c75" : "#c0392b"
    readonly property color info:      dark ? "#6aa8e0" : "#2266aa"

    // Metrics
    readonly property int radius: 10
    readonly property int radiusSmall: 6
    readonly property int gapTight: 8
    readonly property int gap: 12
    readonly property int gapLoose: 20
    readonly property int padSmall: 12
    readonly property int pad: 18
    readonly property int padLarge: 28

    readonly property int fontBody: 13
    readonly property int fontSmall: 11
    readonly property int fontTitle: 16
    readonly property int fontHero: 22

    function stateColor(state) {
        switch (state) {
        case "ready": case "complete": case "busy": return success
        case "loading": case "starting": case "active": case "queued": case "running": return info
        case "sleeping": case "paused": case "canceling": return warning
        case "failed": case "crashed": case "canceled": return danger
        default: return textFaint
        }
    }

    function bytes(n) {
        if (n === undefined || n === null) return "—"
        const units = ["B", "KiB", "MiB", "GiB", "TiB"]
        let v = n, i = 0
        while (v >= 1024 && i < units.length - 1) { v /= 1024; i++ }
        return (i === 0 ? v : v.toFixed(1)) + " " + units[i]
    }

    function tokensPerSec(v) {
        return v ? v.toFixed(1) + " tok/s" : "—"
    }

    // F32 / F16 / BF16, or unknown (empty). Q8, K-quants, IQ, Unsloth UD, etc. are not.
    function isFullPrecisionQuant(q) {
        var u = String(q || "").toUpperCase()
        if (u.indexOf("UD-") === 0)
            u = u.substring(3)
        return u === "" || u === "F32" || u === "F16" || u === "BF16"
    }

    function isUnslothDynamicQuant(q) {
        return String(q || "").toUpperCase().indexOf("UD-") === 0
    }

    function quantTagTone(q) {
        return isUnslothDynamicQuant(q) ? warning : info
    }

    // Apply Fusion palette so stock controls inherit the app theme.
    function applyPalette(target) {
        if (!target || !target.palette)
            return
        target.palette.window = bg
        target.palette.windowText = text
        target.palette.base = surface
        target.palette.alternateBase = bgAlt
        target.palette.text = text
        target.palette.button = surfaceHi
        target.palette.buttonText = text
        target.palette.brightText = text
        target.palette.highlight = accent
        target.palette.highlightedText = onAccent
        target.palette.light = surfaceHi
        target.palette.mid = border
        target.palette.midlight = surface
        target.palette.dark = border
        target.palette.shadow = overlay
        target.palette.link = accent
        target.palette.linkVisited = accentHi
        target.palette.placeholderText = textFaint
        target.palette.toolTipBase = surfaceHi
        target.palette.toolTipText = text
    }
}
