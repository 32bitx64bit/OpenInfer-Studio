import QtQuick
import QtTest
import "../../apps/desktop/qml/components/graph"
import "../../apps/desktop/qml/components/graph/RunHistory.js" as History

TestCase {
    id: testCase
    name: "RunHistory"
    width: 640
    height: 900
    visible: true
    when: windowShown
    property var panel
    property string wide: String(Qt.resolvedUrl("fixtures/result-wide.svg"))
    property string tall: String(Qt.resolvedUrl("fixtures/result-tall.svg"))

    QtObject {
        id: fakeApi
        property var responseRuns: []
        property var gets: []
        property var posts: []
        property var pending: []
        property bool deferReply: false
        property bool deferCancel: false
        property var pendingCancels: []
        property int getStatus: 200
        property int postStatus: 200
        function get(path, callback) {
            gets = gets.concat([path])
            if (deferReply) pending = pending.concat([callback])
            else callback(getStatus, getStatus === 200 ? {runs: responseRuns} : {error: "Refresh failed"})
        }
        function post(path, body, callback) {
            posts = posts.concat([{path: path, body: body}])
            if (deferCancel) { pendingCancels = pendingCancels.concat([callback]); return }
            callback(postStatus, postStatus === 200 ? {ok: true} : {error: "Cancel failed"})
        }
    }
    QtObject {
        id: fakeEvents
        signal eventReceived(string name, var payload)
        signal reconnected()
    }
    Component { id: panelComponent; RunHistoryPanel { width: 600; height: 860 } }
    Component { id: viewerComponent; ImageViewer {} }

    function run(id, runState, workflow) {
        return {id: id, state: runState, workflow_id: workflow || "w1", only: "save",
            created_at: "2026-10-03T09:00:00Z", started_at: "2026-10-03T09:00:01Z",
            graph: {version: 1, name: "Forest study", edges: [], nodes: [
                {id: "prompt", type: "prompt", title: "Positive prompt", params: {text: "misty forest, red fox"}},
                {id: "sample", type: "sample", params: {seed: -1, steps: 20}},
                {id: "save", type: "image.save", params: {prefix: "forest"}}
            ]}, nodes: {
                sample: {state: "done", seed: 482910337, message: "seed 482910337", file_urls: [wide, tall], outputs: ["wide.svg", "tall.svg"], ms: 1900},
                save: {state: "done", cached: true, file_urls: [wide], outputs: ["wide.svg"], ms: 0}
            }}
    }
    function init() {
        fakeApi.responseRuns = []; fakeApi.gets = []; fakeApi.posts = []; fakeApi.pending = []
        fakeApi.deferReply = false; fakeApi.getStatus = 200; fakeApi.postStatus = 200
        fakeApi.deferCancel = false; fakeApi.pendingCancels = []
    }
    function cleanup() { panel = null }
    function makePanel(items, workflow) {
        fakeApi.responseRuns = items
        panel = createTemporaryObject(panelComponent, testCase, {api: fakeApi, events: fakeEvents, workflowId: workflow || ""})
        verify(panel !== null)
        tryCompare(panel, "refreshing", false)
        wait(30)
        return panel
    }
    function test_results_allBranches_savedFirst_deduplicated() {
        var data = run("r1", "complete")
        var images = History.resultImages(data)
        compare(images.length, 2)
        compare(images[0].nodeId, "save")
        compare(images[0].saved, true)
        compare(images[1].fileUrl, tall)
        data.nodes.sample.file_urls.push(wide.replace("file:///", "file://localhost/"))
        data.nodes.sample.file_urls.push("file:///tmp/movie.mp4", "https://example.org/image.png")
        compare(History.resultImages(data).length, 2)
    }
    function test_search_andResolvedSeedFromBackend() {
        var data = run("r1", "complete")
        verify(History.matches(data, "FOREST fox"))
        verify(!History.matches(data, "forest city"))
        compare(History.seed(data.nodes.sample), "482910337")
        compare(History.seed({seed: 0, message: "seed 8"}), "0")
        compare(History.seed({message: "seed 77"}), "77")
        compare(History.seed({message: "sampling", seed: -1}), "")
    }
    function test_queueGlobal_historyWorkflowFilter_search() {
        makePanel([run("other-active", "running", "w2"), run("ours", "complete"), run("other", "failed", "w2")], "w1")
        compare(panel.queueRuns.length, 1)
        compare(panel.historyRuns.length, 1)
        compare(panel.historyRuns[0].id, "ours")
        var search = findChild(panel, "runHistorySearch")
        search.text = "city"
        compare(panel.historyRuns.length, 0)
        compare(panel.queueRuns.length, 1)
        search.text = "fox"
        findChild(panel, "runWorkflowFilter").checked = false
        compare(panel.historyRuns.length, 2)
    }
    function test_events_reconnect_andProgress_reconcileDurableSnapshots() {
        var data = run("r1", "running")
        data.nodes.sample = {state: "running", progress: {phase: "sampling", current: 3, total: 20, unit: "steps", message: "Sampling: step 3/20"}}
        makePanel([data])
        var retainedHistory = run("finished", "complete")
        fakeApi.responseRuns = [data, retainedHistory]
        panel.reload()
        var historyCard = findChild(panel, "historyRun_finished")
        var changed = JSON.parse(JSON.stringify(data))
        changed.nodes.sample.progress.current = 8
        fakeApi.responseRuns = [changed, retainedHistory]
        fakeEvents.eventReceived("workflow.node_state", {run_id: "r1", node_id: "sample"})
        tryVerify(function() { return panel.runs[0].nodes.sample.progress.current === 8 })
        compare(findChild(panel, "historyRun_finished"), historyCard)
        changed = JSON.parse(JSON.stringify(changed)); changed.state = "complete"
        fakeApi.responseRuns = [changed]
        fakeEvents.reconnected()
        tryCompare(panel, "queueRuns", [])
        compare(panel.historyRuns.length, 1)
    }
    function test_inflightRefresh_coalescesEvents_withoutInventingState() {
        makePanel([run("r1", "queued")])
        fakeApi.deferReply = true
        panel.reload()
        compare(fakeApi.pending.length, 1)
        fakeEvents.eventReceived("workflow.run_finished", {run_id: "r1", state: "complete"})
        compare(panel.runs[0].state, "queued")
        fakeApi.deferReply = false
        fakeApi.responseRuns = [run("r1", "complete")]
        fakeApi.pending[0](200, {runs: [run("r1", "running")]})
        tryVerify(function() { return panel.runs[0].state === "complete" })
    }
    function test_cancelIndividual_failureRetainsRunState() {
        makePanel([run("r/1", "queued"), run("r2", "running")])
        fakeApi.postStatus = 500
        var row = findChild(panel, "queueRun_r/1")
        var button = findChild(row, "cancelRunButton")
        mouseClick(button)
        compare(fakeApi.posts.length, 1)
        compare(fakeApi.posts[0].path, "/api/v1/workflow/runs/r%2F1/cancel")
        compare(panel.runs[0].state, "queued")
        compare(panel._cancelErrors["r/1"], "Cancel failed")
    }
    function test_refreshFailure_preservesResults() {
        makePanel([run("r1", "complete")])
        fakeApi.getStatus = 503
        panel.reload()
        compare(panel.historyRuns.length, 1)
        compare(panel.refreshError, "Refresh failed")
    }
    function test_cancelSuccess_reconcilesBackend_andIgnoresLateRetryCallback() {
        makePanel([run("r1", "queued")])
        fakeApi.deferCancel = true
        panel.cancelRun(panel.runs[0])
        var first = fakeApi.pendingCancels[0]
        first(0, null)
        compare(panel.runs[0].state, "queued")
        panel.cancelRun(panel.runs[0])
        var second = fakeApi.pendingCancels[1], serial = panel._cancelPending.r1
        first(500, {error: "Late error"})
        compare(panel._cancelPending.r1, serial)
        compare(panel._cancelErrors.r1, undefined)
        fakeApi.responseRuns = [run("r1", "canceled")]
        second(200, {ok: true})
        compare(panel.runs[0].state, "canceled")
        compare(panel.queueRuns.length, 0)
    }
    function test_apiChange_ignoresOldRefreshCallback() {
        makePanel([run("r1", "running")])
        fakeApi.deferReply = true
        panel.reload()
        var previousCallback = fakeApi.pending[0]
        panel.api = null
        previousCallback(200, {runs: [run("obsolete", "complete")]})
        compare(panel.runs.length, 0)
        compare(panel.refreshing, false)
    }
    function test_cardActions_emitDurableRunAndLocalUrl() {
        var snapshot = run("r1", "complete")
        snapshot.source_graph = JSON.parse(JSON.stringify(snapshot.graph))
        snapshot.graph.nodes[1].params.seed = 482910337
        makePanel([snapshot])
        var row = findChild(panel, "historyRun_r1"), inspected, restored, used, compared
        panel.inspectRun.connect(function(data) { inspected = data })
        panel.restoreRun.connect(function(data) { restored = data })
        panel.useImage.connect(function(url) { used = url })
        panel.compareImage.connect(function(url) { compared = url })
        mouseClick(findChild(row, "inspectRunButton"))
        mouseClick(findChild(row, "restoreRunButton"))
        mouseClick(findChild(row, "useResult_0"))
        mouseClick(findChild(row, "compareResult_0"))
        compare(inspected.id, "r1"); compare(restored.graph.name, "Forest study")
        compare(inspected.source_graph.nodes[1].params.seed, -1)
        compare(restored.source_graph.nodes[1].params.seed, -1)
        compare(inspected.graph.nodes[1].params.seed, 482910337)
        compare(restored.graph.nodes[1].params.seed, 482910337)
        compare(inspected.nodes.sample.seed, 482910337)
        compare(restored.nodes.save.cached, true)
        compare(row.nodes[0].params.seed, 482910337)
        compare(used, wide); compare(compared, wide)
        mouseClick(findChild(row, "runDetailsButton"))
        verify(row.expanded)
    }
    function test_viewer_batch_compare_zoomPan_reset() {
        var viewer = createTemporaryObject(viewerComponent, testCase, {sources: [wide, tall], currentIndex: 0, comparisonSource: tall})
        verify(viewer !== null)
        viewer.open()
        tryCompare(viewer, "visible", true)
        var viewport = findChild(viewer.contentItem, "mainImageViewport")
        tryCompare(viewport, "imageStatus", Image.Ready)
        verify(viewer.comparing)
        viewport.zoomAt(4, viewport.width / 2, viewport.height / 2)
        compare(viewport.zoomFactor, 4)
        verify(viewport.maxPanX > 0)
        mousePress(findChild(viewport, "imagePanArea"), viewport.width / 2, viewport.height / 2)
        mouseMove(findChild(viewport, "imagePanArea"), viewport.width / 2 + 50, viewport.height / 2)
        mouseRelease(findChild(viewport, "imagePanArea"), viewport.width / 2 + 50, viewport.height / 2)
        verify(viewport.panX > 0)
        viewer.next()
        compare(viewer.currentIndex, 1)
        compare(viewport.zoomFactor, 1)
        viewer.previous()
        compare(viewer.currentIndex, 0)
        viewer.sources = [wide]
        compare(viewer.currentIndex, 0)
        viewer.close()
    }
}
