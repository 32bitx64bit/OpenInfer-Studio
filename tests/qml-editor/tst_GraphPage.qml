import QtQuick
import QtTest
import "../../apps/desktop/qml/pages"
import "../../apps/desktop/qml/js/graphModel.js" as GM

TestCase {
    id: test
    name: "GraphEditorPage"
    width: 1600; height: 900
    visible: true
    when: windowShown
    property var editor
    readonly property string fixtureUrl: Qt.resolvedUrl("fixtures/input.svg").toString()
    readonly property var registry: [
        { type: "prompt", title: "Prompt", category: "prompt", available: true,
            inputs: [], outputs: [{ name: "cond", type: "COND" }], params: [{ name: "text", kind: "text", default: "" }] },
        { type: "image.load", title: "Image", category: "input", available: true,
            inputs: [], outputs: [{ name: "image", type: "IMAGE" }], params: [{ name: "path", kind: "path" }] },
        { type: "mask.load", title: "Mask", category: "input", available: true,
            inputs: [], outputs: [{ name: "mask", type: "MASK" }], params: [{ name: "path", kind: "path" }] },
        { type: "sample", title: "Sample", category: "sampling", available: true,
            inputs: [{ name: "start", type: ["IMAGE", "SIZE"] }, { name: "mask", type: "MASK" }],
            outputs: [{ name: "image", type: "IMAGE" }], params: [
                { name: "steps", kind: "int", default: 20 }, { name: "cfg", kind: "float", default: 7 },
                { name: "guidance", kind: "float", default: 3 }, { name: "strength", kind: "float", default: 1 },
                { name: "seed", kind: "seed", default: -1 }, { name: "control_strength", kind: "float", default: 0.5 },
                { name: "sampler", kind: "enum", from: "capabilities.samplers" },
                { name: "scheduler", kind: "enum", from: "capabilities.schedulers" } ] },
        { type: "sample.video", title: "Video", category: "sampling", available: true,
            inputs: [{ name: "start", type: "SIZE" }], outputs: [], params: [
                { name: "steps", kind: "int" }, { name: "frames", kind: "int" }, { name: "fps", kind: "int" } ] },
        { type: "latent.empty", title: "Size", category: "input", available: true,
            inputs: [], outputs: [{ name: "size", type: "SIZE" }], params: [
                { name: "width", kind: "int", default: 1024 }, { name: "height", kind: "int", default: 1024 },
                { name: "batch", kind: "int", default: 1 } ] },
        { type: "checkpoint.load", title: "Checkpoint", category: "loaders", available: true,
            inputs: [], outputs: [{ name: "model", type: "MODEL" }], params: [{ name: "model", kind: "model" }] }
    ]
    QtObject {
        id: eventBus
        signal eventReceived(string name, var payload)
        signal reconnected()
    }
    QtObject {
        id: client
        property var calls: []
        property var pendingPuts: []
        property var pendingValidations: []
        property bool deferPuts: false
        property bool deferValidation: false
        property int nextRun: 0
        property var runMap: ({})
        property var capabilities: ({})
        property var typeRequests: []
        property var workflowMap: ({})
        function get(path, callback) {
            if (path.indexOf("/api/v1/workflow/node-types") === 0) {
                typeRequests.push(path)
                callback(200, { node_types: test.registry, api_capabilities: capabilities })
            }
            else if (path === "/api/v1/models") callback(200, { models: [] })
            else if (path === "/api/v1/workflows") callback(200, { workflows: [] })
            else if (path === "/api/v1/workflow/runs") callback(200, { runs: Object.keys(runMap).map(function(id) { return runMap[id] }) })
            else if (path.indexOf("/api/v1/workflow/runs/") === 0) callback(200, runMap[path.split("/").pop()])
            else if (path.indexOf("/api/v1/workflows/") === 0 && workflowMap[path.split("/").pop()]) callback(200, GM.clone(workflowMap[path.split("/").pop()]))
            else callback(404, {})
        }
        function put(path, body, callback) {
            calls.push({ method: "PUT", path: path, body: GM.clone(body) })
            if (deferPuts) pendingPuts.push(callback); else callback(200, {})
        }
        function post(path, body, callback) {
            calls.push({ method: "POST", path: path, body: GM.clone(body) })
            if (path === "/api/v1/workflows/validate") {
                if (deferValidation) pendingValidations.push(callback)
                else callback(200, { errors: [], warnings: [] })
            } else if (path === "/api/v1/workflow/runs") {
                var nodes = {}
                body.graph.nodes.forEach(function(n) { nodes[n.id] = { state: "pending" } })
                var run = { id: "run" + (++nextRun), state: "queued", graph: GM.clone(body.graph),
                    workflow_id: body.workflow_id, only: body.only, nodes: nodes }
                runMap[run.id] = run
                callback(202, { run: run })
            } else if (path === "/api/v1/workflows/import") callback(201, { id: "imported" })
            else callback(200, {})
        }
        function del(path, callback) { callback(200, {}) }
    }
    Component { id: pageComponent; GraphPage { api: client; events: eventBus } }
    function fixture() {
        var g = GM.newGraph("Editor")
        g.nodes = [ { id: "n1", type: "prompt", pos: [20, 20], params: { text: "one" } },
                    { id: "n2", type: "prompt", pos: [400, 40], params: { text: "two" } } ]
        return g
    }
    function init() {
        client.calls = []; client.pendingPuts = []; client.pendingValidations = []
        client.deferPuts = false; client.deferValidation = false
        client.nextRun = 0; client.runMap = ({})
        client.capabilities = ({}); client.typeRequests = []
        client.workflowMap = ({})
        editor = createTemporaryObject(pageComponent, test, { width: 1600, height: 900 })
        verify(editor !== null)
        editor.workflowId = "w1"; editor.graph = fixture(); editor.paletteOpen = false
        editor.panX = 40; editor.panY = 40; editor.zoom = 1
        editor.rebuild()
        wait(30)
        client.calls = []
    }
    function cleanup() {
        editor.destroy(); editor = null
        wait(0)
    }
    function test_editUndoRedoAndDuplicateInternalWires() {
        editor.setParam("n1", "text", "one a", true)
        editor.setParam("n1", "text", "one ab", true)
        compare(editor.history.past.length, 1)
        editor.undoEdit(); compare(editor.nodeById.n1.params.text, "one")
        editor.redoEdit(); compare(editor.nodeById.n1.params.text, "one ab")
        editor.graph.edges = [{ from: ["n1", "cond"], to: ["n2", "input"] }]
        editor.rebuild(); editor.setSelection(["n1", "n2"]); editor.duplicateSelected()
        compare(editor.selectedIds, ["n3", "n4"])
        compare(editor.graph.edges[1].from[0], "n3")
        compare(editor.graph.edges[1].to[0], "n4")
        editor.undoEdit(); compare(editor.graph.nodes.length, 2)
        editor.redoEdit(); compare(editor.graph.nodes.length, 4)
    }
    function test_ctrlShiftMarqueeAndGroupDragWithPointer() {
        var view = findChild(editor, "graphViewport"), a = findChild(editor, "graphNode_n1"), b = findChild(editor, "graphNode_n2")
        verify(view && a && b)
        mouseClick(a, 50, 15)
        compare(editor.selectedIds, ["n1"])
        mouseClick(b, 50, 15, Qt.LeftButton, Qt.ControlModifier)
        compare(editor.selectedIds, ["n1", "n2"])
        mouseClick(b, 50, 15, Qt.LeftButton, Qt.ShiftModifier)
        compare(editor.selectedIds, ["n1"])
        mousePress(view, 35, 35)
        mouseMove(view, 750, 320, 50)
        mouseRelease(view, 750, 320)
        compare(editor.selectedIds, ["n1", "n2"])
        var p1 = editor.nodeById.n1.pos.slice(), p2 = editor.nodeById.n2.pos.slice()
        mousePress(a, 50, 15)
        mouseMove(view, p1[0] + editor.panX + 100, p1[1] + editor.panY + 65, 50)
        mouseRelease(view, p1[0] + editor.panX + 100, p1[1] + editor.panY + 65)
        compare(editor.nodeById.n1.pos, [p1[0] + 50, p1[1] + 50])
        compare(editor.nodeById.n2.pos, [p2[0] + 50, p2[1] + 50])
        compare(a.x, editor.nodeById.n1.pos[0])
        compare(b.x, editor.nodeById.n2.pos[0])
        compare(editor.history.past.length, 1)
        editor.undoEdit(); compare(editor.nodeById.n1.pos, p1); compare(editor.nodeById.n2.pos, p2)
    }
    function test_systemClipboardAndKeyboardUndo() {
        editor.setSelection(["n1", "n2"])
        var view = findChild(editor, "graphViewport")
        view.forceActiveFocus()
        keyClick(Qt.Key_C, Qt.ControlModifier)
        keyClick(Qt.Key_V, Qt.ControlModifier)
        compare(editor.graph.nodes.length, 4)
        keyClick(Qt.Key_Z, Qt.ControlModifier)
        compare(editor.graph.nodes.length, 2)
        keyClick(Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
        compare(editor.graph.nodes.length, 4)
    }
    function test_frameNoteCollapseAlignmentAndDeleteUndo() {
        editor.setSelection(["n1", "n2"])
        editor.saveFrame(-1, "Prompts", "Remember this")
        compare(editor.graph.groups[0].note, "Remember this")
        editor.toggleCollapse(editor.selectedIds)
        verify(editor.nodeById.n1.collapsed && editor.nodeById.n2.collapsed)
        editor.alignSelected("left"); compare(editor.nodeById.n2.pos[0], 20)
        editor.deleteSelected()
        compare(editor.graph.groups[0].nodes.length, 0)
        editor.undoEdit(); compare(editor.graph.groups[0].nodes, ["n1", "n2"])
        editor.undoEdit(); compare(editor.nodeById.n2.pos[0], 400)
        editor.undoEdit(); verify(!editor.nodeById.n1.collapsed)
    }
    function test_queueForceAndSnapshotIsolation() {
        editor.forceRun = true
        editor.run(""); verify(editor.running)
        editor.run("n1"); compare(client.nextRun, 2)
        compare(editor.activeRunCount, 2)
        var posts = client.calls.filter(function(c) { return c.path === "/api/v1/workflow/runs" })
        verify(posts[0].body.force); compare(posts[0].body.workflow_id, "w1")
        verify(editor.runStateFor("n1") !== null)
        editor.setParam("n1", "text", "new editor input", true)
        compare(editor.runStateFor("n1"), null)
        var current = GM.clone(editor.graph), old = GM.clone(client.runMap.run1)
        editor.inspectRun(old)
        verify(editor.inspectingRun)
        compare(editor.graph, current)
        compare(editor.nodeById.n1.params.text, "one")
        editor.setParam("n1", "text", "blocked")
        compare(editor.graph, current)
        editor.returnToEditor(); compare(editor.nodeById.n1.params.text, "new editor input")
        editor.restoreRun(old); compare(editor.nodeById.n1.params.text, "one")
        editor.undoEdit(); compare(editor.nodeById.n1.params.text, "new editor input")
        old.workflow_id = "other"; editor.applyRunView(old, true)
        compare(editor.runStateFor("n1"), null)
    }
    function test_saveSerializationAndStaleValidation() {
        client.deferPuts = true
        editor.save()
        editor.setParam("n1", "text", "second")
        editor.save()
        compare(client.pendingPuts.length, 1)
        client.pendingPuts.shift()(200, {})
        compare(client.pendingPuts.length, 1)
        var puts = client.calls.filter(function(c) { return c.method === "PUT" })
        compare(puts.length, 2)
        compare(puts[1].body.graph.nodes[0].params.text, "second")
        client.pendingPuts.shift()(200, {})
        compare(editor.saveState, "saved")
        client.deferValidation = true
        editor.validate()
        editor.setParam("n1", "text", "third")
        client.pendingValidations.shift()(200, { errors: [{ node: "n1", severity: "error", message: "Old graph" }] })
        compare(editor.issues.length, 0)
    }
    function test_importExportThroughBackendAndExportAfterSave() {
        client.deferPuts = true
        editor.exportWorkflow("/tmp/workflow.json")
        compare(client.calls.filter(function(c) { return /\/export$/.test(c.path) }).length, 0)
        client.pendingPuts.shift()(200, {})
        var exports = client.calls.filter(function(c) { return /\/export$/.test(c.path) })
        compare(exports.length, 1); compare(exports[0].body.path, "/tmp/workflow.json")
        editor.importWorkflow("/tmp/import.json")
        var imports = client.calls.filter(function(c) { return /\/import$/.test(c.path) })
        compare(imports[0].body, { path: "/tmp/import.json" })
    }
    function test_batchInputPreviewAndMaskAttachment() {
        editor.graph.nodes = [ { id: "n1", type: "image.load", pos: [20, 20], params: { path: GM.fileUrlToLocalPath(fixtureUrl) } },
            { id: "n2", type: "sample", pos: [400, 40], params: {} } ]
        editor.graph.edges = [{ from: ["n1", "image"], to: ["n2", "start"] }]
        editor.rebuild(); wait(0)
        var input = findChild(editor, "graphNode_n1")
        compare(input.previewSources, [fixtureUrl])
        tryCompare(findChild(input, "nodePreview"), "status", Image.Ready)
        editor.addPaintedMask("n1", GM.fileUrlToLocalPath(fixtureUrl))
        compare(editor.nodeById.n3.type, "mask.load")
        compare(GM.edgeInto(editor.graph, "n2", "mask").from, ["n3", "mask"])
        var r = { id: "batch", state: "complete", workflow_id: "w1", graph: GM.clone(editor.graph),
            nodes: { n2: { state: "done", file_urls: [fixtureUrl, fixtureUrl + "?batch=2", fixtureUrl + "?batch=3"] } } }
        editor.applyRunView(r, true)
        wait(30)
        var sample = findChild(editor, "graphNode_n2")
        compare(sample.previewSources.length, 3)
        var next = findChild(sample, "nextPreview"), previous = findChild(sample, "previousPreview")
        mouseClick(next); compare(sample.previewIndex, 1)
        mouseClick(next); compare(sample.previewUrl, fixtureUrl + "?batch=3")
        verify(!next.enabled)
        mouseClick(previous); compare(sample.previewIndex, 1)
        editor.setParam("n1", "path", "")
        compare(sample.previewSources.length, 0)
    }
    function test_advertisedChoicesAndCheckpointCapabilities() {
        compare(editor.paramOptions({ from: "capabilities.samplers" }), [{ text: "default", value: "" }])
        client.capabilities = { known: true, samplers: ["euler", "custom"], schedulers: ["karras"],
            upscalers: [{ name: "esrgan", model: true, image_upscale: true }, { name: "unsupported", model: true, image_upscale: false }, { name: "builtin", model: false, image_upscale: true }] }
        editor.graph.nodes.push({ id: "n3", type: "checkpoint.load", pos: [800, 20], params: { model: { library_id: "m 1" } } })
        editor.touch(true)
        compare(client.typeRequests[client.typeRequests.length - 1], "/api/v1/workflow/node-types?model_id=m%201")
        compare(editor.paramOptions({ from: "capabilities.samplers" }).map(function(p) { return p.value }), ["", "euler", "custom"])
        compare(editor.paramOptions({ from: "capabilities.upscalers" }).map(function(p) { return p.value }), ["", "esrgan"])
        var count = client.typeRequests.length
        eventBus.eventReceived("media.server_state", {})
        compare(client.typeRequests.length, count + 1)
    }
    function test_editorScreenshotAndFrameDrag() {
        editor.panY = 140
        editor.setSelection(["n1", "n2"])
        editor.saveFrame(-1, "Prompt section", "Select with Ctrl/Shift; move this frame together.")
        wait(0)
        var b = editor.frameBounds(editor.graph.groups[0]), view = findChild(editor, "graphViewport")
        var x = b.x + editor.panX + 30, y = b.y + editor.panY + 15
        var before = editor.nodeById.n1.pos.slice()
        mousePress(view, x, y)
        mouseMove(view, x + 30, y + 30, 50)
        mouseRelease(view, x + 30, y + 30)
        compare(editor.nodeById.n1.pos, [before[0] + 30, before[1] + 30])
        compare(findChild(editor, "graphNode_n1").x, editor.nodeById.n1.pos[0])
        editor.historyOpen = true
        wait(50)
        grabImage(editor).save("/tmp/openinfer-editor.png")
    }
    function findTextInput(item) {
        if (item.cursorPosition !== undefined && item.selectByMouse !== undefined) return item
        var children = item.children || []
        for (var i = 0; i < children.length; i++) {
            var found = findTextInput(children[i])
            if (found) return found
        }
        return null
    }
    function test_promptTypingUsesCoalescedDocumentUndo() {
        var area = findTextInput(findChild(editor, "graphNode_n1"))
        verify(area !== null)
        area.forceActiveFocus()
        keyClick(Qt.Key_End)
        keyClick(Qt.Key_A); keyClick(Qt.Key_B)
        compare(editor.nodeById.n1.params.text, "oneab")
        compare(editor.history.past.length, 1)
        keyClick(Qt.Key_Z, Qt.ControlModifier)
        compare(editor.nodeById.n1.params.text, "one")
        keyClick(Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
        compare(editor.nodeById.n1.params.text, "oneab")
    }
    function test_resolvedSeedsUseSourceSnapshotAndPreserveBadges() {
        editor.graph.nodes = [{ id: "n1", type: "sample", pos: [20, 20], params: { seed: -1 } }]
        editor.rebuild()
        var source = GM.clone(editor.graph), resolved = GM.clone(source)
        resolved.nodes[0].params.seed = 482910337
        var run = { id: "random", state: "complete", workflow_id: "w1", source_graph: source, graph: resolved,
            nodes: { n1: { state: "done", seed: 482910337, cached: true } } }
        editor.applyRunView(run, true)
        verify(editor.runMatchesEditor)
        compare(editor.runStateFor("n1").seed, 482910337)
        verify(editor.runStateFor("n1").cached)
        eventBus.eventReceived("workflow.node_state", { run_id: "random", node_id: "n1", state: "done", seed: 7, cached: true })
        compare(editor.runStateFor("n1").seed, 7)
        verify(editor.runStateFor("n1").cached)
        var revision = editor.documentRev
        editor.inspectRun(run)
        compare(editor.nodeById.n1.params.seed, 482910337)
        compare(editor.graph.nodes[0].params.seed, -1)
        compare(editor.documentRev, revision)
    }
    function test_runtimeDefaultsAtomicAndOnlyAdvertisedFields() {
        editor.graph.nodes.push({ id: "n3", type: "sample", pos: [700, 20], params: {
            seed: 123, steps: 30, cfg: 9, guidance: 7, strength: 0.6, control_strength: 0.75 } })
        editor.graph.nodes.push({ id: "n4", type: "latent.empty", pos: [20, 400], params: { width: 1024, height: 1024, batch: 1 } })
        editor.graph.edges.push({ from: ["n4", "size"], to: ["n3", "start"] })
        editor.rebuild(); editor.setSelection(["n3"])
        compare(editor.selectedRuntimeDefaults, null)
        editor.apiCapabilities = { known: true, defaults_by_mode: {
            img_gen: { width: 640, height: 384, batch: 3, steps: 8, cfg: 1, sampler: "euler" },
            vid_gen: { width: 832, height: 480, frames: 33, fps: 16, steps: 20 } } }
        var before = GM.clone(editor.graph)
        editor.applyRuntimeDefaults()
        compare(editor.history.past.length, 1)
        compare(editor.nodeById.n3.params.steps, 8)
        compare(editor.nodeById.n3.params.seed, 123)
        compare(editor.nodeById.n3.params.guidance, 7)
        compare(editor.nodeById.n3.params.strength, 0.6)
        compare(editor.nodeById.n3.params.control_strength, 0.75)
        compare(editor.nodeById.n1.params.text, "one")
        compare(editor.nodeById.n4.params, { width: 640, height: 384, batch: 3 })
        editor.undoEdit(); compare(editor.graph, before)
        editor.redoEdit(); compare(editor.nodeById.n3.params.steps, 8)
        editor.graph.nodes.push({ id: "n5", type: "sample.video", pos: [1000, 20], params: {} })
        editor.graph.edges.push({ from: ["n4", "size"], to: ["n5", "start"] })
        editor.rebuild(); editor.setSelection(["n5"])
        editor.applyRuntimeDefaults()
        compare(editor.nodeById.n5.params, { frames: 33, fps: 16, steps: 20 })
        compare(editor.nodeById.n4.params, { width: 832, height: 480, batch: 3 })
    }
    function test_failedSaveKeepsWorkflowOpenBeforeSwitch() {
        client.workflowMap = { w2: { id: "w2", name: "Other", graph: GM.newGraph("Other") } }
        client.deferPuts = true
        editor.setParam("n1", "text", "unsaved")
        editor.openWorkflow("w2")
        compare(editor.workflowId, "w1")
        client.pendingPuts.shift()(500, {})
        compare(editor.workflowId, "w1")
        compare(editor.nodeById.n1.params.text, "unsaved")
        editor.openWorkflow("w2")
        client.pendingPuts.shift()(200, {})
        compare(editor.workflowId, "w2")
        compare(editor.history.past.length, 0)
    }
}
