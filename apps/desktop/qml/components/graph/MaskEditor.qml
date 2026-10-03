import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import "../.."
import ".."

Window {
    id: root
    property var api
    property string source: ""
    property var strokes: []
    // Brush diameter in displayed image pixels. Each stroke stores its radius
    // relative to the image's larger side, independent of later window sizing.
    property real brushSize: 32
    property bool erase: false
    property bool saving: false
    property string errorMessage: ""
    property bool _painting: false
    property int _strokeIndex: -1
    property int _pointCount: 0
    property int _requestSerial: 0
    property int maxStrokes: 1000
    property int maxPoints: 20000
    property bool imageWithinLimits: picture.status === Image.Ready
        && picture.sourceSize.width > 0 && picture.sourceSize.height > 0
        && picture.sourceSize.width <= 4096 && picture.sourceSize.height <= 4096
    signal maskCreated(string path)

    title: "Image Studio · Paint mask"
    width: 1060
    height: 780
    minimumWidth: 640
    minimumHeight: 440
    color: AppTheme.bg
    visible: false

    function open() {
        if (saving) { show(); raise(); requestActivate(); return }
        clear()
        if (!localPath(source)) errorMessage = "Choose a local image before painting a mask."
        show()
        raise()
        requestActivate()
    }
    function clear() {
        if (saving) return
        strokes = []; _painting = false; _strokeIndex = -1; _pointCount = 0
        errorMessage = ""
        overlay.requestPaint()
    }
    function localPath(fileUrl) {
        var match = String(fileUrl || "").match(/^file:\/\/([^/]*)(\/.*)$/i)
        if (!match || (match[1] !== "" && match[1].toLowerCase() !== "localhost")) return ""
        var path
        try { path = decodeURIComponent(match[2]) } catch (e) { return "" }
        if (/^\/[a-z]:\//i.test(path)) path = path.substring(1)
        return path.indexOf("\u0000") === -1 ? path : ""
    }
    function pointAt(stageX, stageY) {
        if (picture.status !== Image.Ready || overlay.width <= 0 || overlay.height <= 0) return null
        var x = (stageX - overlay.x) / overlay.width
        var y = (stageY - overlay.y) / overlay.height
        if (x < 0 || x > 1 || y < 0 || y > 1) return null
        return [Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))]
    }
    function beginStroke(stageX, stageY) {
        if (saving) return false
        if (!imageWithinLimits) { errorMessage = "Mask images must be at most 4096 × 4096 pixels."; return false }
        var point = pointAt(stageX, stageY)
        if (!point) return false
        if (strokes.length >= maxStrokes || _pointCount >= maxPoints) {
            errorMessage = "This mask has reached the painting limit. Clear it to start again."
            return false
        }
        var radius = Math.max(0.00001, Math.min(0.5, brushSize / 2 / Math.max(overlay.width, overlay.height)))
        strokes = strokes.concat([{points: [point], radius: radius, erase: erase}])
        _strokeIndex = strokes.length - 1; _painting = true; ++_pointCount
        errorMessage = ""
        overlay.requestPaint()
        return true
    }
    function appendPoint(stageX, stageY) {
        if (!_painting || saving) return
        var point = pointAt(stageX, stageY)
        if (!point) { endStroke(); return }
        if (_pointCount >= maxPoints) {
            errorMessage = "This mask has reached the painting limit. Clear it to start again."
            endStroke(); return
        }
        var previous = strokes[_strokeIndex].points
        var last = previous[previous.length - 1]
        // Avoid duplicate stationary samples while retaining short brush taps.
        var dx = (point[0] - last[0]) * overlay.width
        var dy = (point[1] - last[1]) * overlay.height
        if (dx * dx + dy * dy < 1) return
        var next = strokes.slice(), current = next[_strokeIndex]
        next[_strokeIndex] = {points: current.points.concat([point]), radius: current.radius, erase: current.erase}
        strokes = next; ++_pointCount
        overlay.requestPaint()
    }
    function endStroke() { _painting = false; _strokeIndex = -1 }
    function undoStroke() {
        if (saving || !strokes.length) return
        endStroke()
        _pointCount -= strokes[strokes.length - 1].points.length
        strokes = strokes.slice(0, -1)
        overlay.requestPaint()
    }
    function submit() {
        endStroke()
        if (!api || saving) return
        var path = localPath(source)
        if (!path || picture.status !== Image.Ready) { errorMessage = "The source image is unavailable."; return }
        if (!imageWithinLimits) { errorMessage = "Mask images must be at most 4096 × 4096 pixels."; return }
        if (!strokes.some(function(stroke) { return !stroke.erase })) { errorMessage = "Paint an area to regenerate first."; return }
        var serial = ++_requestSerial, submittedSource = source, client = api
        saving = true; errorMessage = ""
        client.post("/api/v1/workflow/masks", {image_path: path, strokes: strokes}, function(status, data) {
            if (serial !== root._requestSerial || submittedSource !== root.source || client !== root.api || !root.saving) return
            root.saving = false
            if (status >= 200 && status < 300 && data && typeof data.path === "string" && data.path !== "") {
                root.maskCreated(data.path)
                root.close()
            } else {
                root.errorMessage = (data && (data.detail || data.error)) || "Could not save this mask. Try again."
            }
        })
    }
    onSourceChanged: { ++_requestSerial; saving = false; clear() }
    onApiChanged: { ++_requestSerial; saving = false }
    onClosing: function(close) { close.accepted = !root.saving }
    Component.onCompleted: AppTheme.applyPalette(root)

    Shortcut { sequence: "Escape"; enabled: root.visible && !root.saving; onActivated: root.close() }
    Shortcut { sequence: "Ctrl+Z"; enabled: root.visible && !root.saving; onActivated: root.undoStroke() }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: AppTheme.padSmall
        spacing: AppTheme.gap
        Flow {
            Layout.fillWidth: true
            spacing: AppTheme.gapTight
            Label { text: "Brush diameter"; color: AppTheme.textDim; height: 36; verticalAlignment: Text.AlignVCenter }
            AppSlider { objectName: "maskBrushSlider"; width: 180; height: 36; from: 2; to: 160; value: root.brushSize; enabled: !root.saving; onMoved: root.brushSize = value }
            Label { text: Math.round(root.brushSize) + " px"; color: AppTheme.textDim; height: 36; verticalAlignment: Text.AlignVCenter }
            AppCheckBox { objectName: "maskEraseToggle"; text: "Erase"; checked: root.erase; enabled: !root.saving; onToggled: root.erase = checked }
            AppButton { objectName: "maskUndoButton"; text: "Undo stroke"; enabled: !root.saving && root.strokes.length > 0; onClicked: root.undoStroke() }
            AppButton { objectName: "maskClearButton"; text: "Clear"; enabled: !root.saving && root.strokes.length > 0; onClicked: root.clear() }
        }
        Rectangle {
            id: stage
            objectName: "maskPaintStage"
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: AppTheme.bgAlt
            border.color: AppTheme.border
            clip: true
            Image {
                id: picture
                objectName: "maskSourceImage"
                anchors.fill: parent
                source: root.source
                fillMode: Image.PreserveAspectFit
                asynchronous: true
                cache: false
                onStatusChanged: if (status === Image.Ready && !root.imageWithinLimits) root.errorMessage = "Mask images must be at most 4096 × 4096 pixels."
            }
            Canvas {
                id: overlay
                objectName: "maskOverlay"
                width: picture.status === Image.Ready ? picture.paintedWidth : 0
                height: picture.status === Image.Ready ? picture.paintedHeight : 0
                x: (stage.width - width) / 2
                y: (stage.height - height) / 2
                opacity: 0.55
                onWidthChanged: requestPaint()
                onHeightChanged: requestPaint()
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.clearRect(0, 0, width, height)
                    ctx.lineCap = "round"; ctx.lineJoin = "round"
                    root.strokes.forEach(function(stroke) {
                        ctx.globalCompositeOperation = stroke.erase ? "destination-out" : "source-over"
                        ctx.strokeStyle = "white"; ctx.fillStyle = "white"
                        var radius = stroke.radius * Math.max(overlay.width, overlay.height)
                        ctx.lineWidth = radius * 2
                        ctx.beginPath()
                        stroke.points.forEach(function(point, index) {
                            if (index === 0) ctx.moveTo(point[0] * overlay.width, point[1] * overlay.height)
                            else ctx.lineTo(point[0] * overlay.width, point[1] * overlay.height)
                        })
                        ctx.stroke()
                        if (stroke.points.length === 1) {
                            var point = stroke.points[0]
                            ctx.beginPath(); ctx.arc(point[0] * overlay.width, point[1] * overlay.height, radius, 0, Math.PI * 2); ctx.fill()
                        }
                    })
                    ctx.globalCompositeOperation = "source-over"
                }
            }
            BusyIndicator { anchors.centerIn: parent; running: picture.status === Image.Loading }
            Label {
                anchors.centerIn: parent
                text: "Image unavailable"
                visible: picture.status === Image.Error || root.source === ""
                color: AppTheme.textDim
            }
            MouseArea {
                id: pointer
                objectName: "maskPaintPointer"
                anchors.fill: parent
                enabled: !root.saving && picture.status === Image.Ready
                hoverEnabled: true
                acceptedButtons: Qt.LeftButton
                cursorShape: Qt.CrossCursor
                property bool paintGesture: false
                onPressed: function(mouse) { paintGesture = root.beginStroke(mouse.x, mouse.y) }
                onPositionChanged: function(mouse) {
                    if (!pressed || !paintGesture) return
                    if (root._painting) root.appendPoint(mouse.x, mouse.y)
                    else root.beginStroke(mouse.x, mouse.y)
                }
                onReleased: { root.endStroke(); paintGesture = false }
                onCanceled: { root.endStroke(); paintGesture = false }
                Rectangle {
                    x: pointer.mouseX - width / 2
                    y: pointer.mouseY - height / 2
                    width: root.brushSize
                    height: width
                    radius: width / 2
                    visible: pointer.containsMouse && root.pointAt(pointer.mouseX, pointer.mouseY) !== null
                    color: "transparent"
                    border.color: root.erase ? AppTheme.danger : "white"
                    border.width: 1
                }
            }
        }
        Label {
            Layout.fillWidth: true
            text: "Painted areas (white) will be regenerated. Erased and untouched areas are preserved."
            color: AppTheme.textDim
            font.pixelSize: AppTheme.fontSmall
            wrapMode: Text.Wrap
        }
        Label {
            Layout.fillWidth: true
            visible: root.errorMessage !== ""
            text: root.errorMessage
            textFormat: Text.PlainText
            wrapMode: Text.Wrap
            color: AppTheme.danger
        }
        RowLayout {
            Layout.fillWidth: true
            Label { text: root.strokes.length + " strokes"; color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall }
            Item { Layout.fillWidth: true }
            AppButton { objectName: "maskCloseButton"; text: "Cancel"; enabled: !root.saving; onClicked: root.close() }
            AppButton { objectName: "maskSaveButton"; text: root.saving ? "Saving…" : "Use mask"; primary: true; enabled: !root.saving && !!root.api && root.imageWithinLimits && root.strokes.length > 0; onClicked: root.submit() }
        }
    }
}
