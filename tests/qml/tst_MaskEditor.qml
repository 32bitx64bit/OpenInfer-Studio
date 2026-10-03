import QtQuick
import QtTest
import "../../apps/desktop/qml/components/graph"

TestCase {
    id: testCase
    name: "MaskEditor"
    width: 1100
    height: 900
    visible: true
    when: windowShown
    property var editor
    property var stage
    property var overlay

    QtObject {
        id: fakeApi
        property var posts: []
        property int status: 201
        property bool deferReply: false
        property var pending
        function post(path, body, callback) {
            posts = posts.concat([{path: path, body: JSON.parse(JSON.stringify(body))}])
            if (deferReply) pending = callback
            else callback(status, status === 201 ? {path: "/media/masks/painted.png", file_url: "file:///media/masks/painted.png"} : {error: "Mask save failed"})
        }
    }
    Component { id: maskComponent; MaskEditor {} }
    function init() {
        fakeApi.posts = []; fakeApi.status = 201; fakeApi.deferReply = false; fakeApi.pending = null
        editor = createTemporaryObject(maskComponent, testCase, {api: fakeApi, source: String(Qt.resolvedUrl("fixtures/result-wide.svg"))})
        verify(editor !== null)
        editor.open()
        tryCompare(editor, "visible", true)
        tryCompare(findChild(editor.contentItem, "maskSourceImage"), "status", Image.Ready)
        stage = findChild(editor.contentItem, "maskPaintStage")
        overlay = findChild(editor.contentItem, "maskOverlay")
        tryVerify(function() { return overlay.width > 0 && overlay.height > 0 })
    }
    function cleanup() { if (editor) editor.close(); editor = null; stage = null; overlay = null }
    function center() { return [overlay.x + overlay.width / 2, overlay.y + overlay.height / 2] }
    function draw() {
        var point = center()
        verify(editor.beginStroke(point[0], point[1]))
        editor.appendPoint(point[0] + 30, point[1] + 10)
        editor.endStroke()
    }
    function test_letterboxRejected_normalization_andBoundedPoints() {
        verify(overlay.y > 0)
        compare(editor.pointAt(stage.width / 2, overlay.y / 2), null)
        compare(editor.beginStroke(stage.width / 2, overlay.y / 2), false)
        draw()
        compare(editor.strokes.length, 1)
        compare(editor.strokes[0].points[0], [0.5, 0.5])
        fuzzyCompare(editor.strokes[0].radius, editor.brushSize / 2 / Math.max(overlay.width, overlay.height), 0.000001)
        editor.strokes[0].points.forEach(function(point) { verify(point[0] >= 0 && point[0] <= 1 && point[1] >= 0 && point[1] <= 1) })
        verify(!editor.beginStroke(-50, stage.height + 50))
    }
    function test_pointerPainting_andEraseFlag_andClear() {
        var pointer = findChild(editor.contentItem, "maskPaintPointer"), point = center()
        mousePress(pointer, point[0], point[1])
        mouseMove(pointer, point[0] + 50, point[1] + 20)
        mouseRelease(pointer, point[0] + 50, point[1] + 20)
        compare(editor.strokes.length, 1)
        verify(editor.strokes[0].points.length >= 2)
        compare(editor.strokes[0].erase, false)
        editor.erase = true
        draw()
        compare(editor.strokes[1].erase, true)
        editor.undoStroke()
        compare(editor.strokes.length, 1)
        mouseClick(findChild(editor.contentItem, "maskClearButton"))
        compare(editor.strokes.length, 0)
    }
    function test_resize_keepsNormalizedStrokes() {
        draw()
        var before = JSON.stringify(editor.strokes)
        editor.width = 750; editor.height = 600
        wait(40)
        compare(JSON.stringify(editor.strokes), before)
    }
    function test_localPath_decoding_noRemoteAuthority() {
        compare(editor.localPath("file:///tmp/image%20one%23two.png"), "/tmp/image one#two.png")
        compare(editor.localPath("file://localhost/tmp/image.png"), "/tmp/image.png")
        compare(editor.localPath("file:///C:/Images/image%20one.png"), "C:/Images/image one.png")
        compare(editor.localPath("file://remote/share/image.png"), "")
        compare(editor.localPath("https://example.org/image.png"), "")
        compare(editor.localPath("file:///tmp/bad%QQ.png"), "")
    }
    function test_submitPayload_andMaskCreatedSignal() {
        draw()
        var createdPath = ""
        editor.maskCreated.connect(function(path) { createdPath = path })
        mouseClick(findChild(editor.contentItem, "maskSaveButton"))
        compare(fakeApi.posts.length, 1)
        compare(fakeApi.posts[0].path, "/api/v1/workflow/masks")
        compare(fakeApi.posts[0].body.image_path, editor.localPath(editor.source))
        compare(fakeApi.posts[0].body.strokes.length, 1)
        compare(createdPath, "/media/masks/painted.png")
        compare(editor.visible, false)
    }
    function test_failedSave_retainsPainting_andShowsError() {
        draw()
        fakeApi.status = 400
        editor.submit()
        compare(editor.errorMessage, "Mask save failed")
        compare(editor.strokes.length, 1)
        compare(editor.visible, true)
        compare(editor.saving, false)
    }
    function test_staleResponseAfterSourceChange_ignored() {
        draw()
        fakeApi.deferReply = true
        editor.submit()
        compare(editor.saving, true)
        var createdPath = ""
        editor.maskCreated.connect(function(path) { createdPath = path })
        editor.source = String(Qt.resolvedUrl("fixtures/result-tall.svg"))
        fakeApi.pending(201, {path: "/media/masks/stale.png"})
        compare(createdPath, "")
        compare(editor.saving, false)
        compare(editor.strokes.length, 0)
    }
    function test_strokeAndPointLimits() {
        editor.maxStrokes = 1
        draw()
        var point = center()
        verify(!editor.beginStroke(point[0], point[1]))
        verify(editor.errorMessage !== "")
        editor.clear(); editor.maxStrokes = 1000; editor.maxPoints = 1
        verify(editor.beginStroke(point[0], point[1]))
        editor.appendPoint(point[0] + 30, point[1])
        compare(editor.strokes[0].points.length, 1)
        verify(editor.errorMessage !== "")
    }
    function test_oversizedSource_reportsLimit_andDisablesSave() {
        editor.source = String(Qt.resolvedUrl("fixtures/result-oversized.svg"))
        tryCompare(findChild(editor.contentItem, "maskSourceImage"), "status", Image.Ready)
        compare(editor.imageWithinLimits, false)
        verify(editor.errorMessage.indexOf("4096") !== -1)
        compare(findChild(editor.contentItem, "maskSaveButton").enabled, false)
        var point = center()
        verify(!editor.beginStroke(point[0], point[1]))
        compare(editor.strokes.length, 0)
    }
}
