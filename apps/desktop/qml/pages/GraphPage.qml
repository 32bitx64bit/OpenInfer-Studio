pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtQuick.Shapes
import QtQuick.Window
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
    property var apiCapabilities: ({})
    property string capabilityModelId: ""
    property int capabilityRequest: 0
    property var modelOptions: []
    property var workflows: []

    // ---- the open graph ----
    property string workflowId: ""
    property var graph: GM.newGraph()
    property var nodeById: ({})
    property var nodeIds: []
    property var nodeCards: []
    property var edgeList: []
    property int graphRev: 0       // any content change
    property int documentRev: 0    // editor changes; inspection does not dirty saves
    property int visRev: 0         // changes which params/sockets are shown
    property int edgeRev: 0        // wires changed
    property int paramsRev: 0      // params changed from outside a widget
    property int layoutRev: 0      // a node moved
    property string selectedId: ""
    property var selectedIds: []
    property var history: GM.newHistory()
    property int historyRev: 0
    readonly property bool canUndo: { historyRev; return history.past.length > 0 && !inspectingRun }
    readonly property bool canRedo: { historyRev; return history.future.length > 0 && !inspectingRun }
    property var moveGesture: null
    property var wireBefore: null
    property var nodeSizes: ({})
    property var marquee: null
    property bool historyOpen: false
    property bool minimapOpen: true
    property string comparisonSource: ""
    property var editorView: null
    property var saveQueue: []
    property bool saveInFlight: false
    property int openRequest: 0
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
    property var selectedRun: null
    property bool inspectingRun: false
    property var activeRuns: ({})
    property var submittedRuns: ({})
    property var seedAdvanced: ({})
    property var runRequestSerial: ({})
    property int runListSerial: 0
    property int runEventRevision: 0
    property bool forceRun: false
    readonly property int activeRunCount: Object.keys(activeRuns).length
    readonly property var selectedRuntimeDefaults: {
        graphRev
        if (inspectingRun || selectedIds.length !== 1 || apiCapabilities.known !== true) return null
        var node = nodeById[selectedId]
        if (!node || (node.type !== "sample" && node.type !== "sample.video")) return null
        var modes = apiCapabilities.defaults_by_mode || {}
        var defaults = modes[node.type === "sample.video" ? "vid_gen" : "img_gen"]
        return defaults && Object.keys(defaults).length > 0 ? defaults : null
    }
    readonly property bool runMatchesEditor: {
        graphRev
        return !!selectedRun && !!selectedRun.graph && selectedRun.workflow_id === workflowId
            && GM.executionKey(selectedRun.source_graph || selectedRun.graph) === GM.executionKey(graph)
    }
    property string runError: ""
    property string hintText: ""

    property bool paletteOpen: true

    // ------------------------------------------------------------------
    // Loading
    // ------------------------------------------------------------------
    function graphModelId() {
        for (var i = 0; i < graph.nodes.length; i++) {
            var n = graph.nodes[i]
            if (n.type === "checkpoint.load" && n.params && n.params.model && n.params.model.library_id)
                return n.params.model.library_id
        }
        return preferredModelId || ""
    }
    function loadNodeTypes() {
        var model = graphModelId(), request = ++capabilityRequest
        capabilityModelId = model
        api.get("/api/v1/workflow/node-types" + (model ? "?model_id=" + encodeURIComponent(model) : ""), function(st, data) {
            if (st !== 200 || !data || request !== page.capabilityRequest || model !== page.graphModelId()) return
            var map = {}
            var list = data.node_types || []
            for (var i = 0; i < list.length; i++) map[list[i].type] = list[i]
            page.specs = map
            page.nodeTypes = list
            page.runtimeInfo = data.runtime || {}
            page.apiCapabilities = data.api_capabilities || {}
            if (page.workflowId) { page.rebuild(); page.validateSoon.restart() }
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
        autosave.stop()
        validateSoon.stop()
        var request = ++openRequest
        var fetch = function() { page.api.get("/api/v1/workflows/" + id, function(st, data) {
            if (st !== 200 || !data || request !== page.openRequest) return
            var g = data.graph || GM.newGraph()
            g.nodes = g.nodes || []
            g.edges = g.edges || []
            g.groups = g.groups || []
            for (var i = 0; i < g.nodes.length; i++) {
                g.nodes[i].params = g.nodes[i].params || {}
                g.nodes[i].pos = g.nodes[i].pos || [0, 0]
            }
            g.name = data.name
            page.workflowId = data.id
            page.graph = g
            page.documentRev++
            page.inspectingRun = false
            page.editorView = null
            page.selectedRun = null
            page.runId = ""
            page.running = false
            page.setSelection([])
            page.history = GM.newHistory(); page.historyRev++
            page.nodeSizes = ({})
            page.saveState = "saved"
            page.runStates = ({})
            page.runError = ""
            page.showIssues = false
            page.issues = []; page.issueMap = ({}); page.planInfo = null
            page.rebuild()
            page.loadNodeTypes()
            if (g.view && g.view.zoom) {
                page.panX = g.view.x; page.panY = g.view.y; page.zoom = g.view.zoom
            } else {
                Qt.callLater(page.fitView)
            }
            page.validateSoon.restart()
        }) }
        if (workflowId && (saveState === "dirty" || saveState === "error" || saveState === "saving")) {
            save(function(ok) {
                if (request !== page.openRequest) return
                if (ok) fetch()
                else { page.hintText = "Save failed; current workflow stays open"; hintTimer.restart() }
            })
        } else fetch()
    }

    // ------------------------------------------------------------------
    // Graph state
    // ------------------------------------------------------------------
    function canvasGraph() { return inspectingRun && selectedRun && selectedRun.graph ? selectedRun.graph : graph }
    function rebuild() {
        var map = {}
        var ids = []
        var cards = []
        var shown = canvasGraph()
        for (var i = 0; i < shown.nodes.length; i++) {
            var n = shown.nodes[i]
            map[n.id] = n
            if (specs[n.type]) { ids.push(n.id); cards.push({ node: n, spec: specs[n.type] }) }
        }
        nodeById = map
        nodeIds = ids
        nodeCards = cards
        edgeList = shown.edges.slice()
        edgeRev++; visRev++; layoutRev++; graphRev++
    }
    function touch(structural) {
        graphRev++
        documentRev++
        if (structural) rebuild()
        saveState = "dirty"
        autosave.restart()
        validateSoon.restart()
        if (graphModelId() !== capabilityModelId) loadNodeTypes()
    }
    function snapshot() { return { graph: GM.clone(graph), selectedIds: selectedIds.slice() } }
    function commitEdit(before, label, key) {
        if (GM.recordEdit(history, before, snapshot(), label, key || "", Date.now())) historyRev++
    }
    function applySnapshot(state) {
        if (!state) return
        viewport.forceActiveFocus()
        graph = GM.clone(state.graph)
        setSelection(state.selectedIds || [])
        paramsRev++
        touch(true)
    }
    function undoEdit() { if (canUndo) { applySnapshot(GM.undo(history)); historyRev++ } }
    function redoEdit() { if (canRedo) { applySnapshot(GM.redo(history)); historyRev++ } }
    function setSelection(ids) {
        selectedIds = ids.slice()
        selectedId = ids.length ? ids[ids.length - 1] : ""
    }
    function isSelected(id) { return selectedIds.indexOf(id) >= 0 }
    function measureNode(id, height) {
        if (!nodeSizes[id] || Math.abs(nodeSizes[id].h - height) > 0.5) {
            nodeSizes[id] = { h: height }
            layoutRev++
        }
    }
    function isCollapsed(id) { visRev; return !!nodeById[id] && !!nodeById[id].collapsed }
    function toggleCollapse(ids) {
        if (inspectingRun || !ids.length) return
        var before = snapshot(), collapse = !ids.every(function(id) { return nodeById[id] && nodeById[id].collapsed })
        ids.forEach(function(id) { if (nodeById[id]) nodeById[id].collapsed = collapse })
        visRev++; layoutRev++; touch(false); commitEdit(before, collapse ? "Collapse selection" : "Expand selection")
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
    function setParam(nodeId, name, value, coalesce) {
        if (inspectingRun) return
        var node = nodeById[nodeId]
        if (!node) return
        if (JSON.stringify(GM.effectiveParam(specs[node.type], node, name)) === JSON.stringify(value)) return
        var before = snapshot()
        node.params = node.params || {}
        node.params[name] = value
        var spec = specs[node.type]
        if (GM.affectsLayout(spec, name)) visRev++
        touch(false)
        commitEdit(before, "Edit " + name, coalesce ? nodeId + ":" + name : "")
    }
    function setParamExternal(nodeId, name, value) {
        setParam(nodeId, name, value)
        paramsRev++
    }
    function paramVisible(node, ps) {
        visRev
        return GM.paramVisible(canvasGraph(), specs, node, ps)
    }
    function paramOptions(ps) {
        var advertised = ps.from && ps.from.indexOf("capabilities.") === 0
        var field = advertised ? ps.from.substring("capabilities.".length) : ""
        var known = apiCapabilities.known === true
        var choices = known && Array.isArray(apiCapabilities[field]) ? apiCapabilities[field] : []
        if (field === "upscalers") choices = choices.filter(function(v) { return v && v.model === true && v.image_upscale === true }).map(function(v) { return v.name })
        var list = advertised ? [""].concat(choices.filter(function(v) { return typeof v === "string" && v !== "" })) : (ps.options || [])
        return list.map(function(o) { return { text: o === "" ? "default" : o, value: o } })
    }
    function isConnected(nodeId, port, isOutput) {
        edgeRev
        var shown = canvasGraph()
        for (var i = 0; i < shown.edges.length; i++) {
            var e = shown.edges[i]
            if (isOutput ? (e.from[0] === nodeId && e.from[1] === port) : (e.to[0] === nodeId && e.to[1] === port)) return true
        }
        return false
    }
    function issuesFor(nodeId) {
        if (inspectingRun) return []
        if (!showIssues && selectedId !== nodeId) return []
        return issueMap[nodeId] || []
    }
    function runStateFor(nodeId) { return inspectingRun || runMatchesEditor ? runStates[nodeId] || null : null }
    function select(id, modifiers, preserve) {
        var next = selectedIds.slice()
        if (modifiers & (Qt.ControlModifier | Qt.ShiftModifier)) {
            var at = next.indexOf(id)
            if (at >= 0) next.splice(at, 1); else next.push(id)
            setSelection(next)
        } else if (!preserve || !isSelected(id)) setSelection(id ? [id] : [])
        viewport.forceActiveFocus()
    }
    function beginMove(id, modifiers, pt) {
        select(id, modifiers, true)
        if (inspectingRun || !isSelected(id)) return
        startMove(pt)
    }
    function startMove(pt) {
        var starts = {}
        selectedIds.forEach(function(id) { if (nodeById[id]) starts[id] = nodeById[id].pos.slice() })
        autosave.stop()
        moveGesture = { point: pt, starts: starts, before: snapshot(), moved: false }
    }
    function moveNodes(pt) {
        if (!moveGesture) return
        var dx = pt.x - moveGesture.point.x, dy = pt.y - moveGesture.point.y
        if (!moveGesture.moved && Math.abs(dx) + Math.abs(dy) < 4 / zoom) return
        moveGesture.moved = true
        for (var id in moveGesture.starts) {
            var p = moveGesture.starts[id]
            nodeById[id].pos = [Math.round(p[0] + dx), Math.round(p[1] + dy)]
        }
        layoutRev++
    }
    function endMove(cancel) {
        var gesture = moveGesture
        moveGesture = null
        if (!gesture) return
        if (cancel && gesture.moved) { applySnapshot(gesture.before); return }
        if (gesture.moved) { touch(false); commitEdit(gesture.before, "Move selection") }
        else if (saveState === "dirty") autosave.restart()
    }
    function openMenu(id) {
        if (!isSelected(id)) setSelection([id])
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
        if (!selectedIds.length) return false
        return !isSelected(edge.from[0]) && !isSelected(edge.to[0])
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
        if (inspectingRun) return
        wireBefore = snapshot()
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
                commitEdit(wireBefore, "Connect nodes")
                wireBefore = null
                return
            }
            hintText = why
            hintTimer.restart()
        }
        if (d.rerouted) { touch(true); commitEdit(wireBefore, "Disconnect nodes"); wireBefore = null; return }
        wireBefore = null
        if (!target) openQuickAddForWire(d, pt)
    }

    // Adding and removing nodes -------------------------------------------
    function addNodeAt(spec, x, y, params) {
        if (inspectingRun || !workflowId) return null
        var before = snapshot()
        var n = GM.addNode(graph, spec, x, y, params)
        touch(true)
        setSelection([n.id])
        commitEdit(before, "Add node")
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
        if (!selectedIds.length || inspectingRun) return
        var before = snapshot()
        selectedIds.forEach(function(id) { GM.removeNode(graph, id) })
        setSelection([])
        touch(true)
        commitEdit(before, "Delete selection")
    }
    function duplicateSelected() {
        if (!selectedIds.length || inspectingRun) return
        var before = snapshot(), section = GM.copySection(graph, selectedIds)
        var b = GM.sectionBounds(graph, specs, selectedIds, nodeSizes)
        setSelection(GM.pasteSection(graph, section, b.x + 28, b.y + 28))
        touch(true)
        commitEdit(before, "Duplicate section")
    }
    function selectConnected() { setSelection(GM.connectedSection(canvasGraph(), selectedIds)) }
    function copySelected() {
        if (!selectedIds.length) return
        clipboard.text = JSON.stringify(GM.copySection(canvasGraph(), selectedIds))
        clipboard.selectAll(); clipboard.copy(); clipboard.deselect()
    }
    function pasteSelected() {
        if (inspectingRun || !workflowId) return
        clipboard.clear(); clipboard.paste()
        try {
            var section = GM.parseSection(clipboard.text, specs), before = snapshot(), c = viewCenter()
            setSelection(GM.pasteSection(graph, section, c.x - 100, c.y - 60))
            touch(true); commitEdit(before, "Paste section")
        } catch (error) { hintText = String(error.message || error); hintTimer.restart() }
    }
    function alignSelected(mode) {
        if (selectedIds.length < 2 || inspectingRun) return
        var before = snapshot()
        GM.alignNodes(graph, specs, selectedIds, mode, nodeSizes)
        layoutRev++; touch(false); commitEdit(before, "Align " + mode)
    }
    function applyRuntimeDefaults() {
        var defaults = selectedRuntimeDefaults, node = nodeById[selectedId]
        if (!defaults || !node) return
        var before = snapshot()
        var sampleFields = ["steps", "sampler", "scheduler", "cfg", "guidance", "strength", "frames", "fps"]
        var parameters = GM.list(specs[node.type].params).map(function(p) { return p.name })
        node.params = node.params || {}
        sampleFields.forEach(function(key) {
            if (parameters.indexOf(key) >= 0 && Object.prototype.hasOwnProperty.call(defaults, key)) node.params[key] = defaults[key]
        })
        var start = GM.edgeInto(graph, node.id, "start")
        var size = start ? GM.findNode(graph, start.from[0]) : null
        if (size && size.type === "latent.empty") {
            size.params = size.params || {}
            var sizeParams = GM.list(specs[size.type].params).map(function(p) { return p.name })
            var sizeFields = ["width", "height", "batch"]
            sizeFields.forEach(function(key) {
                if (sizeParams.indexOf(key) >= 0 && Object.prototype.hasOwnProperty.call(defaults, key)) size.params[key] = defaults[key]
            })
        }
        if (JSON.stringify(before.graph) !== JSON.stringify(graph)) {
            paramsRev++; visRev++; touch(false); commitEdit(before, "Apply runtime defaults")
        }
    }
    function frameBounds(group) {
        layoutRev
        var b = GM.sectionBounds(canvasGraph(), specs, GM.list(group.nodes), nodeSizes)
        if (!b) return null
        var titleHeight = 32 + (group.note ? 40 : 0)
        return { x: b.x - 20, y: b.y - titleHeight - 12, w: b.w + 40, h: b.h + titleHeight + 32, titleHeight: titleHeight }
    }
    function editFrame(index) {
        if (inspectingRun || (!selectedIds.length && index < 0)) return
        frameDialog.groupIndex = index
        frameText.text = index < 0 ? "Section" : graph.groups[index].title
        frameNote.text = index < 0 ? "" : graph.groups[index].note || ""
        frameDialog.open()
    }
    function saveFrame(index, title, note) {
        if (inspectingRun || !title.trim()) return
        var before = snapshot()
        graph.groups = GM.list(graph.groups)
        if (index < 0) graph.groups.push({ title: title.trim(), note: note, color: "#6aa8e0", nodes: selectedIds.slice() })
        else { graph.groups[index].title = title.trim(); graph.groups[index].note = note }
        touch(true); commitEdit(before, "Edit frame and note")
    }
    function removeFrame(index) {
        if (inspectingRun) return
        var before = snapshot()
        graph.groups.splice(index, 1)
        touch(true); commitEdit(before, "Remove frame")
    }

    property var quickCtx: null
    function openQuickAdd(wx, wy) {
        if (inspectingRun) return
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
        if (inspectingRun) return
        var before = snapshot()
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
            setSelection([node.id])
            commitEdit(before, "Add connected node")
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
        saveViewport()
    }
    function fitView() {
        var b = GM.sectionBounds(canvasGraph(), specs, nodeIds, nodeSizes)
        if (!b || viewport.width < 50) return
        var z = Math.min((viewport.width - 80) / b.w, (viewport.height - 80) / b.h, 1)
        zoom = Math.max(0.3, z)
        panX = (viewport.width - b.w * zoom) / 2 - b.x * zoom
        panY = Math.max(24, (viewport.height - b.h * zoom) / 2 - b.y * zoom)
        saveViewport()
    }
    function saveViewport() {
        if (inspectingRun || !workflowId) return
        documentRev++; saveState = "dirty"; autosave.restart()
    }

    // ------------------------------------------------------------------
    // Saving and validating
    // ------------------------------------------------------------------
    Timer {
        id: autosave
        interval: 1200
        onTriggered: page.save()
    }
    function save(then) {
        if (!workflowId) return
        saveState = "saving"
        var view = inspectingRun && editorView ? editorView : { x: panX, y: panY, zoom: zoom }
        var job = { id: workflowId, rev: documentRev, doc: GM.forApi(graph, view), then: typeof then === "function" ? then : null }
        // Serialize writes so a slower previous save cannot overwrite a later edit.
        var last = saveQueue[saveQueue.length - 1]
        if (last && last.id === job.id && !last.then && !job.then) saveQueue[saveQueue.length - 1] = job
        else saveQueue.push(job)
        drainSaves()
    }
    function drainSaves() {
        if (saveInFlight || !saveQueue.length) return
        var job = saveQueue.shift()
        saveInFlight = true
        api.put("/api/v1/workflows/" + job.id, { name: job.doc.name || "Untitled workflow", graph: job.doc }, function(st, data) {
            page.saveInFlight = false
            if (page.workflowId === job.id) page.saveState = st !== 200 ? "error" : page.documentRev === job.rev ? "saved" : "dirty"
            if (job.then) job.then(st === 200, data)
            page.drainSaves()
        })
    }
    function rename(name) {
        if (inspectingRun) return
        name = (name || "").trim()
        if (!name || name === graph.name) return
        var before = snapshot()
        graph.name = name
        touch(false); commitEdit(before, "Rename workflow")
        save(function(ok) { if (ok) page.loadWorkflows("") })
    }
    property alias validateSoon: validateTimer
    Timer {
        id: validateTimer
        interval: 450
        onTriggered: page.validate()
    }
    function validate() {
        if (!workflowId) return
        var id = workflowId, revision = documentRev
        api.post("/api/v1/workflows/validate", { graph: GM.forApi(graph) }, function(st, data) {
            if (st !== 200 || !data || id !== page.workflowId || revision !== page.documentRev) return
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
        return ["sample", "sample.video", "image.resize", "image.save", "video.save", "image.crop",
                "image.pad", "image.rotate", "image.blend", "image.pick", "image.upscale"].indexOf(n.type) >= 0
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
            page.inspectingRun = false; page.selectedRun = null; page.runId = ""; page.running = false
            page.history = GM.newHistory(); page.historyRev++; page.setSelection([])
            page.nodeIds = []; page.edgeList = []; page.nodeById = ({})
            page.nodeCards = []
            page.runStates = ({})
            page.loadWorkflows("first")
        })
    }

    // ------------------------------------------------------------------
    // Running
    // ------------------------------------------------------------------
    function run(only) {
        if (!workflowId || inspectingRun) return
        runError = ""
        autosave.stop(); save()
        var doc = GM.forApi(graph), id = workflowId, key = GM.executionKey(doc)
        api.post("/api/v1/workflow/runs", { graph: doc, only: only || "", workflow_id: id, force: forceRun }, function(st, data) {
            if (st === 202 && data && data.run) {
                // Use the returned snapshot; never invent run nodes or outputs.
                page.submittedRuns[data.run.id] = true
                page.trackRun(data.run)
                if (id === page.workflowId && !page.inspectingRun) {
                    page.applyRunView(data.run, true)
                    page.showIssues = false
                }
            } else if (st === 400 && data) {
                if (id !== page.workflowId || key !== GM.executionKey(page.graph)) return
                page.issueMap = GM.issuesByNode(data.errors || [])
                page.issues = (data.errors || []).concat(data.warnings || [])
                page.showIssues = true
                var first = (data.errors || [])[0]
                page.runError = first ? first.message : (data.error || "Nothing to run")
            } else {
                if (id !== page.workflowId) return
                page.runError = (data && (data.detail || data.error)) || "Could not start the run"
            }
        })
    }
    function cancelRun() {
        if (!runId) return
        api.post("/api/v1/workflow/runs/" + runId + "/cancel", {}, function(st, data) {
            if (st !== 200) page.runError = (data && (data.detail || data.error)) || "Could not cancel the run"
        })
    }
    function onRunEvent(name, p) {
        if (!p || !p.run_id) return
        runEventRevision++
        runRequestSerial[p.run_id] = (runRequestSerial[p.run_id] || 0) + 1
        if (name === "workflow.run_queued" || name === "workflow.run_started" || name === "workflow.run_finished") {
            refreshRun(p.run_id)
            return
        }
        if (p.run_id !== runId) return
        if (name === "workflow.node_state") {
            var next = {}
            for (var k in runStates) next[k] = runStates[k]
            var prev = runStates[p.node_id] || {}
            next[p.node_id] = {
                state: p.state, message: p.message || "", file_urls: p.file_urls || [], outputs: p.outputs || [],
                ms: p.ms || 0, progress: p.progress || null, job_id: p.job_id || "",
                seed: p.seed === undefined ? prev.seed : p.seed,
                cached: p.cached === undefined ? prev.cached : p.cached,
                startedMs: prev.startedMs || (p.state === "running" ? Date.now() : 0)
            }
            runStates = next
        }
    }
    // "increment" and "random" seed modes act after a run, like ComfyUI's
    // control-after-generate: the next run starts from the new seed.
    function advanceSeeds(runView) {
        if (inspectingRun || !runMatchesEditor || !submittedRuns[runView.id] || seedAdvanced[runView.id]) return
        seedAdvanced[runView.id] = true
        var before = snapshot()
        var changed = false
        for (var i = 0; i < graph.nodes.length; i++) {
            var n = graph.nodes[i]
            if (n.type !== "sample" && n.type !== "sample.video") continue
            var rs = runView.nodes[n.id]
            if (!rs || rs.state !== "done") continue
            var mode = GM.effectiveParam(specs[n.type], n, "seed_mode")
            var seed = Number(GM.effectiveParam(specs[n.type], n, "seed"))
            if (mode === "increment" && seed >= 0) { n.params.seed = seed + 1; changed = true }
            else if (mode === "random" && seed !== -1) { n.params.seed = -1; changed = true }
        }
        if (changed) { paramsRev++; touch(false); commitEdit(before, "Advance run seeds") }
    }

    Connections {
        target: page.events
        function onEventReceived(name, payload) {
            if (name.indexOf("workflow.") === 0) page.onRunEvent(name, payload)
            else if (name === "media.server_state") page.loadNodeTypes()
            else if (name === "library.scanned" || name === "library.model_imported" || name === "library.model_updated") page.loadLibrary()
        }
        function onReconnected() {
            page.refreshRuns()
            if (page.runId) page.refreshRun(page.runId)
        }
    }

    function trackRun(data) {
        var next = Object.assign({}, activeRuns)
        if (data.state === "running" || data.state === "queued") next[data.id] = data
        else delete next[data.id]
        activeRuns = next
    }
    function applyRunView(data, selectRun) {
        if (!data || !data.id) return
        trackRun(data)
        if (!selectRun && data.id !== runId) return
        page.runId = data.id
        page.selectedRun = data
        var states = {}
        for (var id in (data.nodes || {})) {
            var n = data.nodes[id]
            var prev = page.runStates[id] || {}
            states[id] = { state: n.state, message: n.message || "", file_urls: n.file_urls || [],
                outputs: n.outputs || [], ms: n.ms || 0, progress: n.progress || null, job_id: n.job_id || "",
                seed: n.seed, cached: n.cached === true,
                startedMs: prev.startedMs || (n.state === "running" ? Date.now() : 0) }
        }
        page.runStates = states
        page.running = data.state === "running" || data.state === "queued"
        page.runError = data.state === "failed" ? data.error || "The run failed" : ""
        if (data.state === "complete") advanceSeeds(data)
    }
    function refreshRun(id) {
        var serial = (runRequestSerial[id] || 0) + 1
        runRequestSerial[id] = serial
        api.get("/api/v1/workflow/runs/" + id, function(st, data) {
            if (st === 200 && data && page.runRequestSerial[id] === serial) page.applyRunView(data, false)
        })
    }
    function refreshRuns() {
        var serial = ++runListSerial, revision = runEventRevision
        api.get("/api/v1/workflow/runs", function(st, data) {
            if (st !== 200 || !data || serial !== page.runListSerial || revision !== page.runEventRevision) return
            var next = {}
            GM.list(data.runs).forEach(function(r) {
                if (r.state === "queued" || r.state === "running") next[r.id] = r
                if (r.id === page.runId) page.applyRunView(r, false)
            })
            page.activeRuns = next
        })
    }
    function inspectRun(run) {
        if (!run || !run.graph) { hintText = "This run has no stored graph snapshot"; hintTimer.restart(); return }
        if (!inspectingRun) editorView = { x: panX, y: panY, zoom: zoom }
        inspectingRun = true
        applyRunView(run, true)
        setSelection([]); nodeSizes = ({}); rebuild(); Qt.callLater(fitView)
    }
    function returnToEditor() {
        if (!inspectingRun) return
        inspectingRun = false
        if (editorView) { panX = editorView.x; panY = editorView.y; zoom = editorView.zoom }
        editorView = null
        setSelection([]); nodeSizes = ({}); rebuild(); validateSoon.restart()
    }
    function restoreRun(run) {
        if (!workflowId || !run || !run.graph) return
        returnToEditor()
        var before = snapshot(), name = graph.name
        graph = GM.clone(run.graph); graph.name = name
        setSelection([]); nodeSizes = ({}); paramsRev++
        touch(true); commitEdit(before, "Restore run snapshot"); Qt.callLater(fitView)
    }
    function useImage(fileUrl) {
        returnToEditor()
        if (!workflowId || !specs["image.load"]) return
        var n = nodeById[selectedId]
        var params = GM.list(specs["image.load"].params)
        var pathParam = params.filter(function(p) { return p.kind === "path" })[0]
        if (!pathParam) return
        if (n && n.type === "image.load") setParamExternal(n.id, pathParam.name, GM.fileUrlToLocalPath(fileUrl))
        else {
            var c = viewCenter(), p = {}
            p[pathParam.name] = GM.fileUrlToLocalPath(fileUrl)
            addNodeAt(specs["image.load"], c.x - 120, c.y - 80, p)
        }
    }
    function viewImages(sources, index) {
        imageViewer.sources = sources
        imageViewer.currentIndex = index || 0
        imageViewer.open()
    }
    function compareImage(fileUrl) {
        comparisonSource = fileUrl
        hintText = "Comparison selected; open another image to compare it."
        hintTimer.restart()
    }

    // Reconcile authoritative state even when a terminal event was missed.
    Timer {
        interval: 3000
        running: page.activeRunCount > 0
        repeat: true
        onTriggered: {
            page.refreshRuns()
            if (page.runId) page.refreshRun(page.runId)
        }
    }

    Component.onCompleted: { loadNodeTypes(); loadLibrary(); loadWorkflows("first"); refreshRuns() }
    onVisibleChanged: if (visible && specs) { loadNodeTypes(); loadLibrary() }

    Timer { id: hintTimer; interval: 3500; onTriggered: page.hintText = "" }

    Shortcut { sequences: [StandardKey.Undo]; enabled: page.visible && page.canUndo; onActivated: page.undoEdit() }
    Shortcut { sequences: [StandardKey.Redo]; enabled: page.visible && page.canRedo; onActivated: page.redoEdit() }

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
                                required property var modelData
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
                    enabled: !page.inspectingRun
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
                    visible: page.workflowId !== "" && !page.inspectingRun && page.errorCount() > 0
                    text: page.errorCount() + (page.errorCount() === 1 ? " problem" : " problems")
                    color: AppTheme.warning
                    font.pixelSize: AppTheme.fontSmall
                    font.weight: Font.DemiBold
                }
                Label {
                    visible: page.workflowId !== "" && !page.inspectingRun && page.errorCount() === 0 && page.planSummary() !== ""
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
                    enabled: page.workflowId !== "" && page.canRunTo(page.selectedId) && !page.inspectingRun
                    onClicked: page.run(page.selectedId)
                }
                AppButton {
                    text: page.activeRunCount > 0 ? "Queue run (" + page.activeRunCount + ")" : "▶  Run"
                    primary: true
                    enabled: page.workflowId !== "" && !page.inspectingRun
                    onClicked: page.run("")
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            color: AppTheme.bgAlt
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 12; anchors.rightMargin: 12
                spacing: 6
                AppButton { text: "Undo"; flat: true; enabled: page.canUndo; onClicked: page.undoEdit() }
                AppButton { text: "Redo"; flat: true; enabled: page.canRedo; onClicked: page.redoEdit() }
                AppButton {
                    text: "Selection ▾"; flat: true; enabled: page.selectedIds.length > 0
                    onClicked: selectionMenu.popup()
                }
                AppButton { text: "Import JSON"; flat: true; onClicked: importDialog.open() }
                AppButton {
                    text: "Export JSON"; flat: true; enabled: page.workflowId !== "" && !page.inspectingRun
                    onClicked: { exportDialog.workflowId = page.workflowId; exportDialog.open() }
                }
                CheckBox { text: "Force rerun"; checked: page.forceRun; enabled: !page.inspectingRun; onToggled: page.forceRun = checked }
                AppButton {
                    text: "Runtime defaults"; flat: true
                    enabled: page.selectedRuntimeDefaults !== null
                    onClicked: page.applyRuntimeDefaults()
                    ToolTip.visible: hovered
                    ToolTip.text: "Apply advertised defaults to the selected sampler and its connected size; one undo step"
                }
                Label {
                    Layout.fillWidth: true
                    elide: Text.ElideRight
                    text: page.inspectingRun ? "Run " + page.runId.substring(0, 8) + " · " + (page.selectedRun.state || "") + " · snapshot"
                        : page.selectedIds.length > 1 ? page.selectedIds.length + " selected"
                        : "Drag to select · Ctrl/Shift adds · Middle drag pans"
                    color: AppTheme.textFaint; font.pixelSize: AppTheme.fontSmall
                }
                AppButton { text: "Back to editor"; visible: page.inspectingRun; onClicked: page.returnToEditor() }
                AppButton { text: "Map"; flat: true; onClicked: page.minimapOpen = !page.minimapOpen }
                AppButton { text: "History"; flat: !page.historyOpen; onClicked: page.historyOpen = !page.historyOpen }
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
                                required property var modelData
                                readonly property string cat: modelData
                                readonly property var members: page.nodeTypes.filter(function(s) { return s.category === catBlock.cat })
                                width: paletteCol.width
                                visible: members.length > 0
                                spacing: 2
                                Item { width: 1; height: 8 }
                                Text {
                                    text: (GM.CATEGORY_TITLES[catBlock.cat] || catBlock.cat).toUpperCase()
                                    color: AppTheme.textFaint
                                    font.pixelSize: AppTheme.fontSmall
                                    font.weight: Font.DemiBold
                                    font.letterSpacing: 1
                                    leftPadding: 4
                                }
                                Repeater {
                                    model: catBlock.members
                                    delegate: Rectangle {
                                        id: paletteNode
                                        required property var modelData
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
                                            color: GM.categoryColor(paletteNode.modelData.category)
                                        }
                                        Text {
                                            x: 26
                                            anchors.verticalCenter: parent.verticalCenter
                                            text: paletteNode.modelData.title
                                            color: AppTheme.text
                                            font.pixelSize: AppTheme.fontBody
                                        }
                                        ToolTip.visible: palHover.hovered && (modelData.description || !modelData.available)
                                        ToolTip.delay: 500
                                        ToolTip.text: modelData.available ? (modelData.description || "") : modelData.reason
                                        MouseArea {
                                            anchors.fill: parent
                                            enabled: paletteNode.modelData.available && page.workflowId !== "" && !page.inspectingRun
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: page.addFromPalette(paletteNode.modelData)
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
                objectName: "graphViewport"
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                focus: true
                activeFocusOnTab: true

                Keys.onDeletePressed: page.deleteSelected()
                Keys.onPressed: function(e) {
                    if (e.key === Qt.Key_Backspace) { page.deleteSelected(); e.accepted = true }
                    else if (e.key === Qt.Key_Tab && page.workflowId !== "" && !page.inspectingRun) {
                        var c = page.viewCenter()
                        page.openQuickAdd(c.x - 100, c.y - 60)
                        e.accepted = true
                    } else if (e.key === Qt.Key_D && (e.modifiers & Qt.ControlModifier)) { page.duplicateSelected(); e.accepted = true }
                    else if (e.key === Qt.Key_A && (e.modifiers & Qt.ControlModifier)) { page.setSelection(page.nodeIds); e.accepted = true }
                    else if (e.key === Qt.Key_C && (e.modifiers & Qt.ControlModifier)) { page.copySelected(); e.accepted = true }
                    else if (e.key === Qt.Key_V && (e.modifiers & Qt.ControlModifier)) { page.pasteSelected(); e.accepted = true }
                    else if (e.key === Qt.Key_Z && (e.modifiers & Qt.ControlModifier)) {
                        if (e.modifiers & Qt.ShiftModifier) page.redoEdit(); else page.undoEdit()
                        e.accepted = true
                    }
                    else if (e.key === Qt.Key_Y && (e.modifiers & Qt.ControlModifier)) { page.redoEdit(); e.accepted = true }
                    else if ((e.key === Qt.Key_Return || e.key === Qt.Key_Enter) && (e.modifiers & Qt.ControlModifier)) { page.run(""); e.accepted = true }
                    else if (e.key === Qt.Key_Escape) {
                        if (page.wireDrag && page.wireBefore) { page.graph = page.wireBefore.graph; page.wireDrag = null; page.wireBefore = null; page.rebuild() }
                        page.setSelection([]); e.accepted = true
                    }
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

                // Left drag selects a rectangle; middle drag pans the viewport.
                MouseArea {
                    id: bg
                    anchors.fill: parent
                    acceptedButtons: Qt.LeftButton | Qt.MiddleButton
                    property real sx: 0
                    property real sy: 0
                    property real px: 0
                    property real py: 0
                    property bool moved: false
                    property bool panning: false
                    property var baseSelection: []
                    onPressed: function(m) {
                        sx = m.x; sy = m.y; px = page.panX; py = page.panY; moved = false
                        panning = m.button === Qt.MiddleButton
                        baseSelection = m.modifiers & (Qt.ControlModifier | Qt.ShiftModifier) ? page.selectedIds.slice() : []
                        if (!panning) page.marquee = { x: sx, y: sy, w: 0, h: 0 }
                        viewport.forceActiveFocus()
                    }
                    onPositionChanged: function(m) {
                        if (!pressed) return
                        if (Math.abs(m.x - sx) + Math.abs(m.y - sy) > 3) moved = true
                        if (panning) {
                            page.panX = px + m.x - sx; page.panY = py + m.y - sy
                            cursorShape = Qt.ClosedHandCursor
                        } else {
                            page.marquee = { x: Math.min(sx, m.x), y: Math.min(sy, m.y), w: Math.abs(m.x - sx), h: Math.abs(m.y - sy) }
                            var r = page.marquee
                            var hits = GM.selectionInRect(page.canvasGraph(), page.specs,
                                { x: (r.x - page.panX) / page.zoom, y: (r.y - page.panY) / page.zoom, w: r.w / page.zoom, h: r.h / page.zoom }, page.nodeSizes)
                            page.setSelection(baseSelection.concat(hits.filter(function(id) { return baseSelection.indexOf(id) < 0 })))
                        }
                    }
                    onReleased: {
                        cursorShape = Qt.ArrowCursor
                        if (panning && moved) page.saveViewport()
                        else if (!panning && !moved) page.setSelection(baseSelection)
                        page.marquee = null
                    }
                    onCanceled: page.marquee = null
                    onDoubleClicked: function(m) {
                        if (page.workflowId === "" || page.inspectingRun) return
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
                        model: { page.graphRev; return GM.list(page.canvasGraph().groups) }
                        delegate: Rectangle {
                            id: frame
                            required property var modelData
                            required property int index
                            readonly property int groupIndex: index
                            readonly property var bounds: page.frameBounds(modelData)
                            x: bounds ? bounds.x : 0; y: bounds ? bounds.y : 0
                            width: bounds ? bounds.w : 0; height: bounds ? bounds.h : 0
                            visible: bounds !== null
                            z: -1
                            radius: 8
                            color: Qt.alpha(modelData.color || "#6aa8e0", 0.06)
                            border.color: Qt.alpha(modelData.color || "#6aa8e0", 0.5)
                            border.width: 1
                            Rectangle {
                                width: parent.width; height: frame.bounds ? frame.bounds.titleHeight : 32
                                color: Qt.alpha(frame.modelData.color || "#6aa8e0", 0.14)
                                Text {
                                    x: 12; y: 7; width: parent.width - 70
                                    text: frame.modelData.title; color: AppTheme.text; font.pixelSize: AppTheme.fontBody; elide: Text.ElideRight
                                }
                                Text {
                                    x: 12; y: 30; width: parent.width - 24; height: 38
                                    visible: !!frame.modelData.note; text: frame.modelData.note || ""
                                    color: AppTheme.textDim; font.pixelSize: AppTheme.fontSmall; wrapMode: Text.Wrap; elide: Text.ElideRight
                                }
                                MouseArea {
                                    anchors.fill: parent
                                    acceptedButtons: Qt.LeftButton | Qt.RightButton
                                    onPressed: function(m) {
                                        if (m.button === Qt.RightButton) { frameMenu.groupIndex = frame.groupIndex; frameMenu.popup(); return }
                                        page.setSelection(frame.modelData.nodes)
                                        viewport.forceActiveFocus()
                                        if (!page.inspectingRun) page.startMove(mapToItem(world, m.x, m.y))
                                    }
                                    onPositionChanged: function(m) { if (pressed) page.moveNodes(mapToItem(world, m.x, m.y)) }
                                    onReleased: page.endMove(false)
                                    onCanceled: page.endMove(true)
                                    onDoubleClicked: page.editFrame(frame.groupIndex)
                                }
                            }
                        }
                    }

                    Repeater {
                        model: page.edgeList
                        delegate: Shape {
                            id: wire
                            required property var modelData
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
                        model: page.nodeCards
                        delegate: GraphNode {
                            required property var modelData
                            objectName: "graphNode_" + modelData.node.id
                            // Repeater roles are QVariant copies. Read live
                            // nodes from the map; the role is a teardown fallback.
                            node: page.nodeById[modelData.node.id] || modelData.node
                            spec: page.specs[modelData.node.type] || modelData.spec
                            host: page
                            selected: page.isSelected(modelData.node.id)
                        }
                    }
                }

                Rectangle {
                    visible: page.marquee !== null
                    x: page.marquee ? page.marquee.x : 0; y: page.marquee ? page.marquee.y : 0
                    width: page.marquee ? page.marquee.w : 0; height: page.marquee ? page.marquee.h : 0
                    color: Qt.alpha(AppTheme.accent, 0.12); border.color: AppTheme.accentHi; z: 200
                }
                GraphMinimap {
                    visible: page.minimapOpen && page.nodeIds.length > 0
                    anchors.right: parent.right; anchors.top: parent.top
                    anchors.margins: 12
                    host: page
                    viewportWidth: viewport.width; viewportHeight: viewport.height
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
                                required property var modelData
                                required property int index
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
                    text: "Scroll to zoom · middle drag to pan · double-click or Tab to add · drag a socket to connect"
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

            RunHistoryPanel {
                visible: page.historyOpen
                Layout.preferredWidth: 300
                Layout.fillHeight: true
                api: page.api
                events: page.events
                workflowId: page.workflowId
                onInspectRun: function(run) { page.inspectRun(run) }
                onRestoreRun: function(run) { page.restoreRun(run) }
                onUseImage: function(fileUrl) { page.useImage(fileUrl) }
                onCompareImage: function(fileUrl) { page.compareImage(fileUrl) }
            }
        }
    }

    QuickAdd {
        id: quickAdd
        onPicked: function(spec) { page.quickPicked(spec) }
    }

    Menu {
        id: nodeMenu
        MenuItem { text: "Run to here"; enabled: !page.inspectingRun && page.canRunTo(page.selectedId); onTriggered: page.run(page.selectedId) }
        MenuItem { text: "Paint mask…"; enabled: !page.inspectingRun && !!page.nodeById[page.selectedId] && page.nodeById[page.selectedId].type === "image.load"; onTriggered: page.paintMask() }
        MenuItem { text: "Select connected section"; onTriggered: page.selectConnected() }
        MenuItem { text: "Copy"; onTriggered: page.copySelected() }
        MenuItem { text: "Paste"; enabled: !page.inspectingRun; onTriggered: page.pasteSelected() }
        MenuItem { text: "Duplicate"; enabled: !page.inspectingRun; onTriggered: page.duplicateSelected() }
        MenuItem { text: "Collapse / expand"; enabled: !page.inspectingRun; onTriggered: page.toggleCollapse(page.selectedIds) }
        MenuItem { text: "Frame selection / add note…"; enabled: !page.inspectingRun; onTriggered: page.editFrame(-1) }
        MenuSeparator {}
        MenuItem { text: "Delete"; enabled: !page.inspectingRun; onTriggered: page.deleteSelected() }
    }
    Menu {
        id: selectionMenu
        MenuItem { text: "Select connected section"; onTriggered: page.selectConnected() }
        MenuItem { text: "Copy (Ctrl+C)"; onTriggered: page.copySelected() }
        MenuItem { text: "Duplicate (Ctrl+D)"; enabled: !page.inspectingRun; onTriggered: page.duplicateSelected() }
        MenuItem { text: "Frame selection / add note…"; enabled: !page.inspectingRun; onTriggered: page.editFrame(-1) }
        MenuItem { text: "Collapse / expand"; enabled: !page.inspectingRun; onTriggered: page.toggleCollapse(page.selectedIds) }
        MenuSeparator {}
        Repeater {
            model: ["left", "top", "right", "bottom"]
            delegate: MenuItem {
                required property var modelData
                text: "Align " + modelData
                enabled: page.selectedIds.length > 1 && !page.inspectingRun
                onTriggered: page.alignSelected(modelData)
            }
        }
        MenuItem { text: "Delete selection"; enabled: !page.inspectingRun; onTriggered: page.deleteSelected() }
    }
    Menu {
        id: frameMenu
        property int groupIndex: -1
        MenuItem { text: "Edit frame / note…"; enabled: !page.inspectingRun; onTriggered: page.editFrame(frameMenu.groupIndex) }
        MenuItem { text: "Collapse / expand members"; enabled: !page.inspectingRun; onTriggered: page.toggleCollapse(page.graph.groups[frameMenu.groupIndex].nodes) }
        MenuItem { text: "Remove frame"; enabled: !page.inspectingRun; onTriggered: page.removeFrame(frameMenu.groupIndex) }
    }
    Dialog {
        id: frameDialog
        property int groupIndex: -1
        title: groupIndex < 0 ? "Frame selection" : "Edit frame"
        modal: true
        anchors.centerIn: parent
        width: 380
        standardButtons: Dialog.Ok | Dialog.Cancel
        onAccepted: page.saveFrame(groupIndex, frameText.text, frameNote.text)
        ColumnLayout {
            width: parent.width
            AppTextField { id: frameText; Layout.fillWidth: true; placeholderText: "Frame title" }
            TextArea {
                id: frameNote
                Layout.fillWidth: true; Layout.preferredHeight: 110
                placeholderText: "Notes about this section"
                wrapMode: TextEdit.Wrap; selectByMouse: true
                color: AppTheme.text
            }
        }
    }

    TextEdit { id: clipboard; visible: false }
    ImageViewer {
        id: imageViewer
        objectName: "graphImageViewer"
        comparisonSource: page.comparisonSource
    }
    Loader {
        id: maskLoader
        objectName: "graphMaskLoader"
        active: false
        source: "../components/graph/MaskEditor.qml"
        property string imageNodeId: ""
        property string imageSource: ""
        property string targetWorkflowId: ""
        onLoaded: {
            var editor = item as MaskEditor
            editor.api = page.api
            editor.source = imageSource
            editor.maskCreated.connect(function(path) {
                if (maskLoader.targetWorkflowId === page.workflowId) page.addPaintedMask(maskLoader.imageNodeId, path)
            })
            editor.open()
        }
    }
    function paintMask() {
        var n = nodeById[selectedId], sp = n && specs[n.type]
        if (!n || n.type !== "image.load" || inspectingRun) return
        var pathParam = GM.list(sp.params).filter(function(p) { return p.kind === "path" })[0]
        var path = pathParam ? getParam(n, pathParam) : ""
        if (!path) { hintText = "Choose an input image before painting a mask"; hintTimer.restart(); return }
        maskLoader.active = false
        maskLoader.imageNodeId = n.id; maskLoader.targetWorkflowId = workflowId
        maskLoader.imageSource = GM.localPathToFileUrl(path)
        maskLoader.active = true
    }
    function addPaintedMask(imageId, path) {
        if (inspectingRun || !specs["mask.load"] || !nodeById[imageId]) return
        var before = snapshot(), image = nodeById[imageId], params = {}
        var ps = GM.list(specs["mask.load"].params).filter(function(p) { return p.kind === "path" })[0]
        if (!ps) return
        params[ps.name] = path
        var mask = GM.addNode(graph, specs["mask.load"], image.pos[0], image.pos[1] + 260, params)
        var targets = graph.edges.filter(function(e) { return e.from[0] === imageId && e.to[1] === "start" })
        targets.forEach(function(e) {
            var target = GM.findNode(graph, e.to[0])
            var from = { node: mask.id, port: "mask" }, to = { node: e.to[0], port: "mask" }
            if (target && target.type === "sample" && GM.connectError(graph, specs, from, to) === "") GM.connect(graph, from, to)
        })
        setSelection([mask.id]); touch(true); commitEdit(before, "Add painted mask")
    }

    FileDialog {
        id: importDialog
        title: "Import workflow JSON"
        fileMode: FileDialog.OpenFile
        nameFilters: ["Workflow JSON (*.json)"]
        onAccepted: page.importWorkflow(GM.fileUrlToLocalPath(selectedFile.toString()))
    }
    FileDialog {
        id: exportDialog
        property string workflowId: ""
        title: "Export workflow JSON"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "json"
        nameFilters: ["Workflow JSON (*.json)"]
        onAccepted: {
            if (workflowId !== page.workflowId || page.inspectingRun) return
            page.exportWorkflow(GM.fileUrlToLocalPath(selectedFile.toString()))
        }
    }
    function importWorkflow(path) {
        api.post("/api/v1/workflows/import", { path: path }, function(st, data) {
            if ((st === 200 || st === 201) && data && data.id) {
                page.loadWorkflows(""); page.openWorkflow(data.id)
            } else { page.hintText = (data && (data.detail || data.error)) || "Could not import workflow JSON"; hintTimer.restart() }
        })
    }
    function exportWorkflow(path) {
        var id = workflowId
        autosave.stop()
        save(function(ok, data) {
            if (!ok) { page.hintText = "Save failed; could not export workflow"; hintTimer.restart(); return }
            page.api.post("/api/v1/workflows/" + id + "/export", { path: path }, function(st, result) {
                page.hintText = st === 200 ? "Workflow exported" : (result && (result.detail || result.error)) || "Could not export workflow JSON"
                hintTimer.restart()
            })
        })
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
        if (n && (n.type === "image.load" || n.type === "mask.load")) {
            fileDialog.title = "Choose an input image"
            fileDialog.nameFilters = ["Images (*.png *.jpg *.jpeg *.gif)", "All files (*)"]
        } else {
            fileDialog.title = "Choose a file"
            fileDialog.nameFilters = ["Model files (*.safetensors *.gguf *.ckpt *.pt *.bin)", "All files (*)"]
        }
        fileDialog.open()
    }
}
