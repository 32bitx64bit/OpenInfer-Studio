import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtQuick.Shapes
import ".."
import "../components"
import "../components/graph"
import "../js/graphModel.js" as GM

// Node-graph view of Image Studio. The canvas edits a plain JSON graph; the Go
// backend (internal/workflow) owns every decision about it: which nodes
// exist, what may connect, what a run would do, and running it. This page
// draws, edits, autosaves, asks the backend to validate on each edit, and
// shows run progress from workflow.* events.
Item {
    id: page

    property var api
    property var events
    property string preferredModelId: ""

    // ---- data from the backend ----
    property var nodeTypes: []
    property var specs: ({})
    property var runtimeInfo: ({})
    property var modelOptions: []
    property var workflows: []

    // ---- the open graph ----
    property string workflowId: ""
    property var graph: GM.newGraph()
    property var nodeById: ({})
    property var nodeIds: []
    property var edgeList: []
    property int graphRev: 0       // any content change
    property int visRev: 0         // changes which params/sockets are shown
    property int edgeRev: 0        // wires changed
    property int paramsRev: 0      // params changed from outside a widget
    property int layoutRev: 0      // a node moved
    property string selectedId: ""
    property var wireDrag: null
    property string saveState: ""

    // ---- viewport ----
    property real panX: 40
    property real panY: 40
    property real zoom: 1
    readonly property alias world: world

    // ---- validation and runs ----
    property var issues: []
    property var issueMap: ({})
    property bool showIssues: false
    property var planInfo: null
    property var runStates: ({})
    property string runId: ""
    property bool running: false
    property string runError: ""
    property string hintText: ""

    readonly property var samplerNames: ["", "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2", "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd"]
    readonly property var schedulerNames: ["", "discrete", "karras", "exponential", "ays", "gits", "smoothstep", "sgm_uniform", "simple", "lcm"]
    property bool paletteOpen: true

    // ------------------------------------------------------------------
    // Loading
    // ------------------------------------------------------------------
    function loadNodeTypes() {
        api.get("/api/v1/workflow/node-types", function(st, data) {
            if (st !== 200 || !data) return
            var map = {}
            var list = data.node_types || []
            for (var i = 0; i < list.length; i++) map[list[i].type] = list[i]
            page.specs = map
            page.nodeTypes = list
            page.runtimeInfo = data.runtime || {}
            if (page.workflowId) page.rebuild()
        })
    }
    function loadLibrary() {
        api.get("/api/v1/models", function(st, data) {
            if (st !== 200) return
            var all = (data && data.models) || []
            var out = []
            for (var i = 0; i < all.length; i++)
                if (GM.isImageGen(all[i])) out.push({ id: all[i].id, name: all[i].alias || String(all[i].primary_path || "").split(/[/\\]/).pop() })
            page.modelOptions = out
        })
    }
    function loadWorkflows(thenOpen) {
        api.get("/api/v1/workflows", function(st, data) {
            if (st !== 200) return
            page.workflows = (data && data.workflows) || []
            if (thenOpen === "first" && page.workflows.length > 0 && !page.workflowId)
                page.openWorkflow(page.workflows[0].id)
        })
    }
    function openWorkflow(id) {
        api.get("/api/v1/workflows/" + id, function(st, data) {
            if (st !== 200 || !data) return
            var g = data.graph || GM.newGraph()
            g.nodes = g.nodes || []
            g.edges = g.edges || []
            for (var i = 0; i < g.nodes.length; i++) {
                g.nodes[i].params = g.nodes[i].params || {}
                g.nodes[i].pos = g.nodes[i].pos || [0, 0]
            }
            g.name = data.name
            page.workflowId = data.id
            page.graph = g
            page.selectedId = ""
            page.runStates = ({})
            page.runError = ""
            page.showIssues = false
            page.rebuild()
            if (g.view && g.view.zoom) {
                page.panX = g.view.x; page.panY = g.view.y; page.zoom = g.view.zoom
            } else {
                Qt.callLater(page.fitView)
            }
            page.validateSoon.restart()
        })
    }

    // ------------------------------------------------------------------
    // Graph state
    // ------------------------------------------------------------------
    function rebuild() {
        var map = {}
        var ids = []
        for (var i = 0; i < graph.nodes.length; i++) {
            var n = graph.nodes[i]
            map[n.id] = n
            if (specs[n.type]) ids.push(n.id)
        }
        nodeById = map
        nodeIds = ids
        edgeList = graph.edges.slice()
        edgeRev++; visRev++; layoutRev++; graphRev++
    }
    function touch(structural) {
        graphRev++
        if (structural) rebuild()
        saveState = "dirty"
        autosave.restart()
        validateSoon.restart()
    }
    function nodeSpec(id) {
        var n = nodeById[id]
        return n ? specs[n.type] : null
    }

    // Host API used by GraphNode ----------------------------------------
    function getParam(node, ps) {
        var v = GM.effectiveParam(specs[node.type], node, ps.name)
        return v === undefined ? "" : v
    }
    function setParam(nodeId, name, value) {
        var node = nodeById[nodeId]
        if (!node) return
        node.params = node.params || {}
        node.params[name] = value
        var spec = specs[node.type]
        if (GM.affectsLayout(spec, name)) visRev++
        touch(false)
    }
    function setParamExternal(nodeId, name, value) {
        setParam(nodeId, name, value)
        paramsRev++
    }
    function paramVisible(node, ps) {
        visRev
        return GM.paramVisible(graph, specs, node, ps)
    }
    function paramOptions(ps) {
        var list = ps.from === "capabilities.samplers" ? samplerNames
                 : ps.from === "capabilities.schedulers" ? schedulerNames
                 : (ps.options || [])
        return list.map(function(o) { return { text: o === "" ? "default" : o, value: o } })
    }
    function isConnected(nodeId, port, isOutput) {
        edgeRev
        for (var i = 0; i < graph.edges.length; i++) {
            var e = graph.edges[i]
            if (isOutput ? (e.from[0] === nodeId && e.from[1] === port) : (e.to[0] === nodeId && e.to[1] === port)) return true
        }
        return false
    }
    function issuesFor(nodeId) {
        if (!showIssues && selectedId !== nodeId) return []
        return issueMap[nodeId] || []
    }
    function runStateFor(nodeId) { return runStates[nodeId] || null }
    function select(id) {
        selectedId = id
        viewport.forceActiveFocus()
    }
    function nodeMoved(id, x, y) {
        var n = nodeById[id]
        if (!n) return
        n.pos = [Math.round(x), Math.round(y)]
        layoutRev++
        saveState = "dirty"
        autosave.restart()
    }
    function openMenu(id) {
        selectedId = id
        nodeMenu.popup()
    }

    // Wires ---------------------------------------------------------------
    function edgePoints(edge) {
        layoutRev
        var fn = nodeById[edge.from[0]], tn = nodeById[edge.to[0]]
        if (!fn || !tn) return null
        var fs = specs[fn.type], ts = specs[tn.type]
        if (!fs || !ts) return null
        return {
            a: GM.socketPos(fn, fs, edge.from[1], true),
            b: GM.socketPos(tn, ts, edge.to[1], false),
            type: GM.types(fs.outputs[Math.max(0, GM.portIndex(fs.outputs, edge.from[1]))])[0]
        }
    }
    function edgeDim(edge) {
        if (selectedId === "") return false
        return edge.from[0] !== selectedId && edge.to[0] !== selectedId
    }
    function dragPoints() {
        layoutRev
        var d = wireDrag
        if (!d) return null
        var n = nodeById[d.fromNode]
        var sp = n ? specs[n.type] : null
        if (!sp) return null
        var anchor = GM.socketPos(n, sp, d.fromPort, d.fromOutput)
        var ports = d.fromOutput ? sp.outputs : sp.inputs
        var port = ports[Math.max(0, GM.portIndex(ports, d.fromPort))]
        return {
            a: d.fromOutput ? anchor : { x: d.x, y: d.y },
            b: d.fromOutput ? { x: d.x, y: d.y } : anchor,
            type: GM.types(port)[0]
        }
    }
    function beginWire(nodeId, port, isOutput, pt) {
        if (!isOutput) {
            var e = GM.edgeInto(graph, nodeId, port)
            if (e) {   // pick the wire up by its input end
                GM.removeEdge(graph, e)
                edgeList = graph.edges.slice(); edgeRev++
                wireDrag = { fromNode: e.from[0], fromPort: e.from[1], fromOutput: true, x: pt.x, y: pt.y, rerouted: true }
                return
            }
        }
        wireDrag = { fromNode: nodeId, fromPort: port, fromOutput: isOutput, x: pt.x, y: pt.y, rerouted: false }
    }
    function moveWire(pt) {
        var d = wireDrag
        if (!d) return
        wireDrag = { fromNode: d.fromNode, fromPort: d.fromPort, fromOutput: d.fromOutput, x: pt.x, y: pt.y, rerouted: d.rerouted }
    }
    // Is (nodeId, port) on the far side of the wire being dragged, and does it fit?
    function socketAccepts(nodeId, port, isOutput) {
        var d = wireDrag
        if (!d || isOutput === d.fromOutput) return false
        var from = d.fromOutput ? { node: d.fromNode, port: d.fromPort } : { node: nodeId, port: port }
        var to = d.fromOutput ? { node: nodeId, port: port } : { node: d.fromNode, port: d.fromPort }
        return GM.connectError(graph, specs, from, to) === ""
    }
    function socketHot(nodeId, port, isOutput) { return socketAccepts(nodeId, port, isOutput) }
    function socketAt(pt, wantOutput) {
        var best = null, bestDist = 22 * 22
        for (var i = 0; i < nodeIds.length; i++) {
            var n = nodeById[nodeIds[i]]
            var sp = specs[n.type]
            var ports = wantOutput ? GM.list(sp.outputs) : GM.list(sp.inputs)
            for (var j = 0; j < ports.length; j++) {
                var p = GM.socketPos(n, sp, ports[j].name, wantOutput)
                var dd = (p.x - pt.x) * (p.x - pt.x) + (p.y - pt.y) * (p.y - pt.y)
                if (dd < bestDist) { bestDist = dd; best = { node: n.id, port: ports[j].name } }
            }
        }
        return best
    }
    function endWire(pt) {
        var d = wireDrag
        wireDrag = null
        if (!d) return
        var target = socketAt(pt, !d.fromOutput)
        if (target) {
            var from = d.fromOutput ? { node: d.fromNode, port: d.fromPort } : target
            var to = d.fromOutput ? target : { node: d.fromNode, port: d.fromPort }
            var why = GM.connectError(graph, specs, from, to)
            if (why === "") {
                GM.connect(graph, from, to)
                touch(true)
                return
            }
            hintText = why
            hintTimer.restart()
        }
        if (d.rerouted) { touch(true); return }          // a dropped, picked-up wire is deleted
        if (!target) openQuickAddForWire(d, pt)
    }

    // Adding and removing nodes -------------------------------------------
    function addNodeAt(spec, x, y, params) {
        var n = GM.addNode(graph, spec, x, y, params)
        touch(true)
        selectedId = n.id
        return n
    }
    function viewCenter() {
        return { x: (viewport.width / 2 - panX) / zoom, y: (viewport.height / 2 - panY) / zoom }
    }
    function addFromPalette(spec) {
        var c = viewCenter()
        var jitter = (graph.nodes.length % 6) * 18
        addNodeAt(spec, c.x - GM.nodeWidth(spec.type) / 2 + jitter, c.y - 80 + jitter, null)
    }
    function deleteSelected() {
        if (!selectedId) return
        GM.removeNode(graph, selectedId)
        selectedId = ""
        touch(true)
    }
    function duplicateSelected() {
        var n = nodeById[selectedId]
        if (!n) return
        var c = GM.duplicateNode(graph, n, specs[n.type])
        touch(true)
        selectedId = c.id
    }

    property var quickCtx: null
    function openQuickAdd(wx, wy) {
        quickCtx = { x: wx, y: wy, wire: null }
        quickAdd.heading = "Add a node"
        quickAdd.accepts = null
        quickAdd.specs = nodeTypes
        quickAdd.openAtItem(world, wx, wy)
    }
    function openQuickAddForWire(d, pt) {
        var n = nodeById[d.fromNode]
        var sp = specs[n.type]
        var ports = d.fromOutput ? sp.outputs : sp.inputs
        var carried = ports[Math.max(0, GM.portIndex(ports, d.fromPort))]
        quickCtx = { x: pt.x, y: pt.y, wire: d }
        quickAdd.heading = d.fromOutput ? "Accepts " + GM.types(carried).join(" or ") : "Makes " + GM.types(carried).join(" or ")
        quickAdd.specs = nodeTypes
        quickAdd.accepts = function(s) {
            var side = d.fromOutput ? GM.list(s.inputs) : GM.list(s.outputs)
            for (var i = 0; i < side.length; i++)
                if (d.fromOutput ? GM.compatible(carried, side[i]) : GM.compatible(side[i], carried)) return true
            return false
        }
        quickAdd.openAtItem(world, pt.x, pt.y)
    }
    function quickPicked(spec) {
        var c = quickCtx || viewCenter()
        var d = c.wire
        var x = c.x, y = c.y
        if (d) {
            var n = nodeById[d.fromNode]
            var carried = (d.fromOutput ? specs[n.type].outputs : specs[n.type].inputs)[GM.portIndex(d.fromOutput ? specs[n.type].outputs : specs[n.type].inputs, d.fromPort)]
            var side = d.fromOutput ? GM.list(spec.inputs) : GM.list(spec.outputs)
            var idx = -1
            for (var i = 0; i < side.length && idx < 0; i++)
                if (d.fromOutput ? GM.compatible(carried, side[i]) : GM.compatible(side[i], carried)) idx = i
            x = d.fromOutput ? c.x + 24 : c.x - GM.nodeWidth(spec.type) - 24
            y = c.y - GM.FIRST_ROW - GM.ROW * Math.max(0, idx)
            var node = GM.addNode(graph, spec, x, y, null)
            if (idx >= 0) {
                if (d.fromOutput) GM.connect(graph, { node: d.fromNode, port: d.fromPort }, { node: node.id, port: side[idx].name })
                else GM.connect(graph, { node: node.id, port: side[idx].name }, { node: d.fromNode, port: d.fromPort })
            }
            touch(true)
            selectedId = node.id
            return
        }
        addNodeAt(spec, x, y, null)
    }

    // View -------------------------------------------------------------------
    function zoomAt(px, py, factor) {
        var nz = Math.max(0.25, Math.min(2.5, zoom * factor))
        var k = nz / zoom
        panX = px - (px - panX) * k
        panY = py - (py - panY) * k
        zoom = nz
        saveState = saveState === "" ? "" : "dirty"
    }
    function fitView() {
        var b = GM.bounds(graph, specs)
        if (!b || viewport.width < 50) return
        var z = Math.min((viewport.width - 80) / b.w, (viewport.height - 80) / b.h, 1)
        zoom = Math.max(0.3, z)
        panX = (viewport.width - b.w * zoom) / 2 - b.x * zoom
        panY = Math.max(24, (viewport.height - b.h * zoom) / 2 - b.y * zoom)
    }

    // ------------------------------------------------------------------
    // Saving and validating
    // ------------------------------------------------------------------
    Timer {
        id: autosave
        interval: 1200
        onTriggered: page.save()
    }
    function save() {
        if (!workflowId) return
        saveState = "saving"
        var doc = GM.forApi(graph, { x: panX, y: panY, zoom: zoom })
        api.put("/api/v1/workflows/" + workflowId, { name: graph.name || "Untitled workflow", graph: doc }, function(st) {
            saveState = st === 200 ? "saved" : "error"
        })
    }
    function rename(name) {
        name = (name || "").trim()
        if (!name || name === graph.name) return
        graph.name = name
        api.put("/api/v1/workflows/" + workflowId, { name: name }, function(st) {
            if (st === 200) page.loadWorkflows("")
        })
    }
    property alias validateSoon: validateTimer
    Timer {
        id: validateTimer
        interval: 450
        onTriggered: page.validate()
    }
    function validate() {
        if (!workflowId) return
        api.post("/api/v1/workflows/validate", { graph: GM.forApi(graph) }, function(st, data) {
            if (st !== 200 || !data) return
            page.issues = (data.errors || []).concat(data.warnings || [])
            page.issueMap = GM.issuesByNode(data.errors || [])
            page.planInfo = data.plan || null
            if ((data.errors || []).length === 0) page.showIssues = false
        })
    }
    function errorCount() {
        var n = 0
        for (var i = 0; i < issues.length; i++) if (issues[i].severity === "error") n++
        return n
    }
    function planSummary() {
        if (!planInfo || !planInfo.servers || planInfo.servers.length === 0) return ""
        var s = planInfo.servers
        var parts = []
        for (var i = 0; i < s.length; i++) {
            var verb = s[i].action === "reuse" ? "already loaded" : s[i].action === "restart" ? "will reload" : "will load"
            parts.push(s[i].model_name + " " + verb)
        }
        return parts.join(" · ")
    }

    // ------------------------------------------------------------------
    // Templates and workflow management
    // ------------------------------------------------------------------
    // Run to here only makes sense for nodes that do work (the loaders and
    // prompts feeding a Sample run together with it).
    function canRunTo(id) {
        var n = nodeById[id]
        if (!n) return false
        return n.type === "sample" || n.type === "sample.video" || n.type === "image.resize"
            || n.type === "image.save" || n.type === "video.save"
    }
    function pickModel() {
        if (preferredModelId) for (var i = 0; i < modelOptions.length; i++) if (modelOptions[i].id === preferredModelId) return modelOptions[i]
        return modelOptions.length > 0 ? modelOptions[0] : null
    }
    function createFromTemplate(kind) {
        if (nodeTypes.length === 0) { hintText = "Still loading node types, try again in a moment."; hintTimer.restart(); return }
        var m = pickModel()
        if (!m) { hintText = "Add a diffusion model to your library first (Browse models)."; hintTimer.restart(); return }
        var g = GM.template(kind, specs, { library_id: m.id, name: m.name })
        api.post("/api/v1/workflows", { name: g.name, graph: GM.forApi(g) }, function(st, data) {
            if (st === 201 && data) {
                page.workflowId = ""
                page.loadWorkflows("")
                page.openWorkflow(data.id)
            } else {
                page.hintText = "Could not create the workflow."
                hintTimer.restart()
            }
        })
    }
    function deleteWorkflow() {
        if (!workflowId) return
        var id = workflowId
        api.del("/api/v1/workflows/" + id, function(st) {
            if (st !== 200) return
            page.workflowId = ""
            page.graph = GM.newGraph()
            page.nodeIds = []; page.edgeList = []; page.nodeById = ({})
            page.runStates = ({})
            page.loadWorkflows("first")
        })
    }

    // ------------------------------------------------------------------
    // Running
    // ------------------------------------------------------------------
    function run(only) {
        if (!workflowId || running) return
        runError = ""
        autosave.stop(); save()
        api.post("/api/v1/workflow/runs", { graph: GM.forApi(graph), only: only || "" }, function(st, data) {
            if (st === 202 && data && data.run) {
                page.runId = data.run.id
                var states = {}
                for (var id in data.run.nodes) states[id] = { state: data.run.nodes[id].state }
                page.runStates = states
                page.running = true
                page.showIssues = false
            } else if (st === 400 && data) {
                page.issueMap = GM.issuesByNode(data.errors || [])
                page.issues = (data.errors || []).concat(data.warnings || [])
                page.showIssues = true
                var first = (data.errors || [])[0]
                page.runError = first ? first.message : (data.error || "Nothing to run")
            } else {
                page.runError = (data && (data.detail || data.error)) || "Could not start the run"
            }
        })
    }
    function cancelRun() {
        if (!runId) return
        api.post("/api/v1/workflow/runs/" + runId + "/cancel", {}, function() {})
    }
    function onRunEvent(name, p) {
        if (!p || p.run_id !== runId) return
        if (name === "workflow.node_state") {
            var next = {}
            for (var k in runStates) next[k] = runStates[k]
            var prev = runStates[p.node_id] || {}
            next[p.node_id] = {
                state: p.state, message: p.message || "", file_urls: p.file_urls || [], outputs: p.outputs || [],
                ms: p.ms || 0, progress: p.progress || null, job_id: p.job_id || "",
                startedMs: prev.startedMs || (p.state === "running" ? Date.now() : 0)
            }
            runStates = next
        } else if (name === "workflow.run_finished") {
            running = false
            if (p.state === "complete") advanceSeeds()
            if (p.state === "failed") runError = p.error || "The run failed"
            if (p.state === "canceled") runError = ""
        }
    }
    // "increment" and "random" seed modes act after a run, like ComfyUI's
    // control-after-generate: the next run starts from the new seed.
    function advanceSeeds() {
        var changed = false
        for (var i = 0; i < graph.nodes.length; i++) {
            var n = graph.nodes[i]
            if (n.type !== "sample" && n.type !== "sample.video") continue
            var rs = runStates[n.id]
            if (!rs || rs.state !== "done") continue
            var mode = GM.effectiveParam(specs[n.type], n, "seed_mode")
            var seed = Number(GM.effectiveParam(specs[n.type], n, "seed"))
            if (mode === "increment" && seed >= 0) { n.params.seed = seed + 1; changed = true }
            else if (mode === "random" && seed !== -1) { n.params.seed = -1; changed = true }
        }
        if (changed) { paramsRev++; touch(false) }
    }

    Connections {
        target: page.events
        function onEventReceived(name, payload) {
            if (name.indexOf("workflow.") === 0) page.onRunEvent(name, payload)
            else if (name === "library.scanned" || name === "library.model_imported" || name === "library.model_updated") page.loadLibrary()
        }
        function onReconnected() {
            if (page.runId && page.running) {
                api.get("/api/v1/workflow/runs/" + page.runId, function(st, data) {
                    if (st !== 200 || !data) return
                    page.applyRunView(data)
                })
            }
        }
    }

    function applyRunView(data) {
        var states = {}
        for (var id in data.nodes) {
            var n = data.nodes[id]
            var prev = page.runStates[id] || {}
            states[id] = { state: n.state, message: n.message || "", file_urls: n.file_urls || [],
                outputs: n.outputs || [], ms: n.ms || 0, progress: n.progress || null, job_id: n.job_id || "",
                startedMs: prev.startedMs || (n.state === "running" ? Date.now() : 0) }
        }
        page.runStates = states
        page.running = data.state === "running" || data.state === "queued"
        if (data.state === "failed") page.runError = data.error || "The run failed"
    }

    // Reconcile authoritative state even when a terminal event was missed.
    Timer {
        interval: 3000
        running: page.running && page.runId !== ""
        repeat: true
        onTriggered: {
            var id = page.runId
            page.api.get("/api/v1/workflow/runs/" + id, function(st, data) {
                if (st === 200 && data && page.runId === id) page.applyRunView(data)
            })
        }
    }

    Component.onCompleted: { loadNodeTypes(); loadLibrary(); loadWorkflows("first") }
    onVisibleChanged: if (visible && specs) { loadNodeTypes(); loadLibrary() }

    Timer { id: hintTimer; interval: 3500; onTriggered: page.hintText = "" }

    // ------------------------------------------------------------------
    // UI
    // ------------------------------------------------------------------
    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // ---- toolbar ----
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 52
            color: AppTheme.bgAlt
            border.color: AppTheme.border
            border.width: 0
            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: AppTheme.border }
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                spacing: 8

                AppButton {
                    text: page.paletteOpen ? "◂ Nodes" : "▸ Nodes"
                    flat: true
                    onClicked: page.paletteOpen = !page.paletteOpen
                }
                AppComboBox {
                    id: wfCombo
                    Layout.preferredWidth: 210
                    model: page.workflows
                    textRole: "name"
                    displayText: page.workflowId ? (page.graph.name || "Untitled workflow") : "No workflow"
                    currentIndex: {
                        for (var i = 0; i < page.workflows.length; i++) if (page.workflows[i].id === page.workflowId) return i
                        return -1
                    }
                    onActivated: function(i) { page.openWorkflow(page.workflows[i].id) }
                }
                AppButton {
                    text: "New ▾"
                    onClicked: templateMenu.popup()
                    Menu {
                        id: templateMenu
                        y: parent.height
                        Repeater {
                            model: GM.templateList()
                            delegate: MenuItem {
                                text: modelData.title + "  —  " + modelData.hint
                                onTriggered: page.createFromTemplate(modelData.id)
                            }
                        }
                    }
                }
                AppTextField {
                    id: nameField
                    visible: page.workflowId !== ""
                    Layout.preferredWidth: 170
                    implicitHeight: 32
                    placeholderText: "Workflow name"
                    text: page.graph.name || ""
                    onEditingFinished: page.rename(text)
                }
                AppButton {
                    visible: page.workflowId !== ""
                    text: "Delete"
                    flat: true
                    onClicked: deleteConfirm.open()
                }
                Label {
                    visible: page.saveState !== ""
                    text: page.saveState === "saving" ? "Saving…" : page.saveState === "saved" ? "Saved" : page.saveState === "error" ? "Save failed" : "Unsaved"
                    color: page.saveState === "error" ? AppTheme.danger : AppTheme.textFaint
                    font.pixelSize: AppTheme.fontSmall
                }

                Item { Layout.fillWidth: true }

                Label {
                    visible: page.workflowId !== "" && page.errorCount() > 0
                    text: page.errorCount() + (page.errorCount() === 1 ? " problem" : " problems")
                    color: AppTheme.warning
                    font.pixelSize: AppTheme.fontSmall
                    font.weight: Font.DemiBold
                }
                Label {
                    visible: page.workflowId !== "" && page.errorCount() === 0 && page.planSummary() !== ""
                    text: page.planSummary()
                    color: AppTheme.textDim
                    font.pixelSize: AppTheme.fontSmall
                    elide: Text.ElideRight
                    Layout.maximumWidth: 320
                }
                AppButton {
                    visible: page.running
                    text: "Cancel"
                    danger: true
                    onClicked: page.cancelRun()
                }
                AppButton {
                    text: "Run to here"
                    enabled: page.workflowId !== "" && page.canRunTo(page.selectedId) && !page.running
                    onClicked: page.run(page.selectedId)
                }
                AppButton {
                    text: page.running ? "Running…" : "▶  Run"
                    primary: true
                    enabled: page.workflowId !== "" && !page.running
                    onClicked: page.run("")
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            // ---- palette ----
            Rectangle {
                visible: page.paletteOpen
                Layout.preferredWidth: 216
                Layout.fillHeight: true
                color: AppTheme.bgAlt
                Rectangle { anchors.right: parent.right; width: 1; height: parent.height; color: AppTheme.border }
                Flickable {
                    anchors.fill: parent
                    anchors.rightMargin: 1
                    contentHeight: paletteCol.implicitHeight + 16
                    clip: true
                    boundsBehavior: Flickable.StopAtBounds
                    ScrollBar.vertical: ScrollBar {}
                    Column {
                        id: paletteCol
                        x: 10
                        y: 10
                        width: parent.width - 20
                        spacing: 2
                        Repeater {
                            model: GM.CATEGORY_ORDER
                            delegate: Column {
                                id: catBlock
                                readonly property string cat: modelData
                                readonly property var members: page.nodeTypes.filter(function(s) { return s.category === catBlock.cat })
                                width: paletteCol.width
                                visible: members.length > 0
                                spacing: 2
                                Item { width: 1; height: 8 }
                                Text {
                                    text: (GM.CATEGORY_TITLES[cat] || cat).toUpperCase()
                                    color: AppTheme.textFaint
                                    font.pixelSize: AppTheme.fontSmall
                                    font.weight: Font.DemiBold
                                    font.letterSpacing: 1
                                    leftPadding: 4
                                }
                                Repeater {
                                    model: catBlock.members
                                    delegate: Rectangle {
                                        width: catBlock.width
                                        height: 30
                                        radius: AppTheme.radiusSmall
                                        color: palHover.hovered && modelData.available ? AppTheme.surfaceHover : "transparent"
                                        opacity: modelData.available ? 1 : 0.5
                                        HoverHandler { id: palHover }
                                        Rectangle {
                                            x: 8
                                            anchors.verticalCenter: parent.verticalCenter
                                            width: 8; height: 8; radius: 2
                                            color: GM.categoryColor(modelData.category)
                                        }
                                        Text {
                                            x: 26
                                            anchors.verticalCenter: parent.verticalCenter
                                            text: modelData.title
                                            color: AppTheme.text
                                            font.pixelSize: AppTheme.fontBody
                                        }
                                        ToolTip.visible: palHover.hovered && (modelData.description || !modelData.available)
                                        ToolTip.delay: 500
                                        ToolTip.text: modelData.available ? (modelData.description || "") : modelData.reason
                                        MouseArea {
                                            anchors.fill: parent
                                            enabled: modelData.available && page.workflowId !== ""
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: page.addFromPalette(modelData)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ---- canvas ----
            Item {
                id: viewport
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                focus: true
                activeFocusOnTab: true

                Keys.onDeletePressed: page.deleteSelected()
                Keys.onPressed: function(e) {
                    if (e.key === Qt.Key_Backspace) { page.deleteSelected(); e.accepted = true }
                    else if (e.key === Qt.Key_Tab && page.workflowId !== "") {
                        var c = page.viewCenter()
                        page.openQuickAdd(c.x - 100, c.y - 60)
                        e.accepted = true
                    } else if (e.key === Qt.Key_D && (e.modifiers & Qt.ControlModifier)) { page.duplicateSelected(); e.accepted = true }
                    else if ((e.key === Qt.Key_Return || e.key === Qt.Key_Enter) && (e.modifiers & Qt.ControlModifier)) { page.run(""); e.accepted = true }
                    else if (e.key === Qt.Key_Escape) { page.selectedId = ""; e.accepted = true }
                }

                // dot grid
                Canvas {
                    id: grid
                    anchors.fill: parent
                    property color dot: AppTheme.dark ? "#27323d" : "#cfd9e0"
                    onDotChanged: requestPaint()
                    onWidthChanged: requestPaint()
                    onHeightChanged: requestPaint()
                    Connections {
                        target: page
                        function onPanXChanged() { grid.requestPaint() }
                        function onPanYChanged() { grid.requestPaint() }
                        function onZoomChanged() { grid.requestPaint() }
                    }
                    onPaint: {
                        var ctx = getContext("2d")
                        ctx.reset()
                        ctx.fillStyle = AppTheme.dark ? "#0c1116" : "#eef2f5"
                        ctx.fillRect(0, 0, width, height)
                        var step = 22 * page.zoom
                        while (step < 12) step *= 2
                        ctx.fillStyle = dot
                        var ox = ((page.panX % step) + step) % step
                        var oy = ((page.panY % step) + step) % step
                        for (var x = ox; x < width; x += step)
                            for (var y = oy; y < height; y += step)
                                ctx.fillRect(Math.round(x), Math.round(y), 1.6, 1.6)
                    }
                }

                // pan, deselect, add
                MouseArea {
                    id: bg
                    anchors.fill: parent
                    acceptedButtons: Qt.LeftButton | Qt.MiddleButton
                    property real sx: 0
                    property real sy: 0
                    property real px: 0
                    property real py: 0
                    property bool moved: false
                    onPressed: function(m) {
                        sx = m.x; sy = m.y; px = page.panX; py = page.panY; moved = false
                        viewport.forceActiveFocus()
                    }
                    onPositionChanged: function(m) {
                        if (!pressed) return
                        if (Math.abs(m.x - sx) + Math.abs(m.y - sy) > 3) moved = true
                        page.panX = px + m.x - sx
                        page.panY = py + m.y - sy
                        cursorShape = Qt.ClosedHandCursor
                    }
                    onReleased: { cursorShape = Qt.ArrowCursor; if (moved) page.saveState = page.saveState === "" ? "" : "dirty" }
                    onClicked: if (!moved) page.selectedId = ""
                    onDoubleClicked: function(m) {
                        if (page.workflowId === "") return
                        var p = bg.mapToItem(world, m.x, m.y)
                        page.openQuickAdd(p.x, p.y)
                    }
                }

                WheelHandler {
                    acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
                    onWheel: function(e) {
                        var dy = e.angleDelta.y !== 0 ? e.angleDelta.y : e.pixelDelta.y * 2
                        if (dy === 0) return
                        page.zoomAt(point.position.x, point.position.y, Math.pow(1.0016, dy))
                    }
                }

                Item {
                    id: world
                    x: page.panX
                    y: page.panY
                    scale: page.zoom
                    transformOrigin: Item.TopLeft
                    width: 1
                    height: 1

                    Repeater {
                        model: page.edgeList
                        delegate: Shape {
                            id: wire
                            readonly property var pts: page.edgePoints(modelData)
                            visible: pts !== null
                            opacity: page.edgeDim(modelData) ? 0.35 : 0.95
                            preferredRendererType: Shape.CurveRenderer
                            ShapePath {
                                strokeColor: wire.pts ? GM.typeColor(wire.pts.type) : "transparent"
                                strokeWidth: 2.5
                                fillColor: "transparent"
                                capStyle: ShapePath.RoundCap
                                PathSvg { path: wire.pts ? GM.wirePath(wire.pts.a, wire.pts.b) : "" }
                            }
                        }
                    }
                    Shape {
                        id: dragWire
                        z: 100
                        readonly property var pts: page.dragPoints()
                        visible: pts !== null
                        opacity: 0.7
                        preferredRendererType: Shape.CurveRenderer
                        ShapePath {
                            strokeColor: dragWire.pts ? GM.typeColor(dragWire.pts.type) : "transparent"
                            strokeWidth: 2.5
                            fillColor: "transparent"
                            capStyle: ShapePath.RoundCap
                            PathSvg { path: dragWire.pts ? GM.wirePath(dragWire.pts.a, dragWire.pts.b) : "" }
                        }
                    }

                    Repeater {
                        model: page.nodeIds
                        delegate: GraphNode {
                            node: page.nodeById[modelData]
                            spec: page.specs[page.nodeById[modelData].type]
                            host: page
                            selected: page.selectedId === modelData
                        }
                    }
                }

                // ---- empty state ----
                Column {
                    visible: page.workflowId === ""
                    anchors.centerIn: parent
                    spacing: 14
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: "Build a workflow"
                        color: AppTheme.text
                        font.pixelSize: AppTheme.fontHero
                        font.weight: Font.DemiBold
                    }
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        width: 380
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.Wrap
                        text: page.modelOptions.length === 0
                              ? "Wire models, prompts and samplers into a graph. First, add a diffusion model to your library from Browse models."
                              : "Wire models, prompts and samplers into a graph, then run it. Start from a template:"
                        color: AppTheme.textDim
                        font.pixelSize: AppTheme.fontBody
                    }
                    Flow {
                        anchors.horizontalCenter: parent.horizontalCenter
                        width: 420
                        spacing: 8
                        visible: page.modelOptions.length > 0
                        Repeater {
                            model: GM.templateList()
                            delegate: AppButton {
                                text: modelData.title
                                primary: index === 0
                                onClicked: page.createFromTemplate(modelData.id)
                            }
                        }
                    }
                }

                // ---- overlays ----
                Rectangle {
                    visible: page.workflowId !== ""
                    anchors.left: parent.left
                    anchors.bottom: parent.bottom
                    anchors.margins: 12
                    width: zoomRow.implicitWidth + 8
                    height: 34
                    radius: AppTheme.radiusSmall + 2
                    color: AppTheme.bgAlt
                    border.width: 1
                    border.color: AppTheme.border
                    Row {
                        id: zoomRow
                        anchors.centerIn: parent
                        spacing: 2
                        AppButton { text: "−"; flat: true; implicitHeight: 28; leftPadding: 10; rightPadding: 10; onClicked: page.zoomAt(viewport.width / 2, viewport.height / 2, 1 / 1.2) }
                        Label { width: 44; horizontalAlignment: Text.AlignHCenter; anchors.verticalCenter: parent.verticalCenter; text: Math.round(page.zoom * 100) + "%"; color: AppTheme.text; font.pixelSize: AppTheme.fontSmall }
                        AppButton { text: "+"; flat: true; implicitHeight: 28; leftPadding: 10; rightPadding: 10; onClicked: page.zoomAt(viewport.width / 2, viewport.height / 2, 1.2) }
                        AppButton { text: "Fit"; flat: true; implicitHeight: 28; onClicked: page.fitView() }
                    }
                }
                Text {
                    visible: page.workflowId !== "" && page.hintText === ""
                    anchors.left: parent.left
                    anchors.bottom: parent.bottom
                    anchors.leftMargin: 14
                    anchors.bottomMargin: 54
                    text: "Scroll to zoom · drag to pan · double-click or Tab to add a node · drag from a socket to connect"
                    color: AppTheme.textFaint
                    font.pixelSize: AppTheme.fontSmall
                }
                Rectangle {
                    visible: page.hintText !== "" || page.runError !== ""
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 16
                    width: Math.min(parent.width - 40, hintLabel.implicitWidth + 28)
                    height: hintLabel.implicitHeight + 18
                    radius: AppTheme.radiusSmall + 2
                    color: page.runError !== "" ? AppTheme.danger : AppTheme.surfaceHi
                    border.width: 1
                    border.color: page.runError !== "" ? AppTheme.danger : AppTheme.border
                    Label {
                        id: hintLabel
                        anchors.centerIn: parent
                        width: parent.width - 28
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.Wrap
                        text: page.runError !== "" ? page.runError : page.hintText
                        color: page.runError !== "" ? AppTheme.onAccent : AppTheme.text
                        font.pixelSize: AppTheme.fontSmall + 1
                    }
                    MouseArea { anchors.fill: parent; enabled: page.runError !== ""; onClicked: page.runError = "" }
                }
            }
        }
    }

    QuickAdd {
        id: quickAdd
        onPicked: function(spec) { page.quickPicked(spec) }
    }

    Menu {
        id: nodeMenu
        MenuItem { text: "Run to here"; enabled: !page.running && page.canRunTo(page.selectedId); onTriggered: page.run(page.selectedId) }
        MenuItem { text: "Duplicate"; onTriggered: page.duplicateSelected() }
        MenuSeparator {}
        MenuItem { text: "Delete"; onTriggered: page.deleteSelected() }
    }

    ConfirmDialog {
        id: deleteConfirm
        title: "Delete workflow?"
        message: "“" + (page.graph.name || "Untitled workflow") + "” will be removed. Generated images are kept."
        confirmText: "Delete"
        onConfirmed: page.deleteWorkflow()
    }

    FileDialog {
        id: fileDialog
        property string nodeId: ""
        property string paramName: ""
        fileMode: FileDialog.OpenFile
        onAccepted: {
            var path = GM.fileUrlToLocalPath(selectedFile.toString())
            page.setParamExternal(nodeId, paramName, path)
        }
    }
    function browse(nodeId, paramName) {
        var n = nodeById[nodeId]
        fileDialog.nodeId = nodeId
        fileDialog.paramName = paramName
        if (n && n.type === "image.load") {
            fileDialog.title = "Choose an input image"
            fileDialog.nameFilters = ["Images (*.png *.jpg *.jpeg *.gif)", "All files (*)"]
        } else {
            fileDialog.title = "Choose a file"
            fileDialog.nameFilters = ["Model files (*.safetensors *.gguf *.ckpt *.pt *.bin)", "All files (*)"]
        }
        fileDialog.open()
    }
}
