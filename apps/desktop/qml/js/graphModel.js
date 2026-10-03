.pragma library

// Pure helpers for the node-graph canvas. No QML, no network: everything the
// canvas decides about a graph that is not drawing. The Go backend remains
// the authority (it re-validates every edit and every run); this file exists
// so the canvas can refuse an impossible wire instantly while you drag.

var HEADER = 30
var ROW = 24
var FIRST_ROW = 50      // y of the first socket row's centre, from the node top

var TYPE_COLORS = {
    MODEL: "#b48ef0", CLIP: "#e5a14a", VAE: "#ef6f8a", COND: "#c9d65a",
    SIZE: "#6aa8e0", IMAGE: "#35c4b5", VIDEO: "#d98ad9", MASK: "#aab4bf", SCALAR: "#8593a1"
}
var CATEGORY_COLORS = {
    loaders: "#b48ef0", prompt: "#c9d65a", input: "#8593a1",
    sampling: "#35c4b5", image: "#6aa8e0", output: "#e5a14a"
}
var CATEGORY_TITLES = {
    loaders: "Loaders", prompt: "Prompts", input: "Inputs",
    sampling: "Sampling", image: "Image", output: "Output"
}
var CATEGORY_ORDER = ["loaders", "prompt", "input", "sampling", "image", "output"]

// Extra search words, so "upscale" finds Resize and "ksampler" finds Sample.
var ALIASES = {
    "image.resize": "upscale scale hires enlarge size",
    "sample": "ksampler generate diffusion txt2img img2img render",
    "sample.video": "wan animate clip generate",
    "checkpoint.load": "model load unet",
    "textencoder.load": "clip t5 llm qwen",
    "vae.load": "decoder",
    "lora.load": "adapter style",
    "prompt": "text conditioning positive negative",
    "latent.empty": "size resolution width height batch",
    "image.load": "input picture photo",
    "image.save": "output export write",
    "video.save": "output export write"
}

// Does a search query match a node type? Every word must hit somewhere.
function matchesQuery(spec, query) {
    var q = String(query || "").trim().toLowerCase()
    if (q === "") return true
    var hay = (spec.title + " " + spec.type + " " + (spec.description || "") + " " +
               (CATEGORY_TITLES[spec.category] || "") + " " + (ALIASES[spec.type] || "")).toLowerCase()
    var words = q.split(/\s+/)
    for (var i = 0; i < words.length; i++) if (hay.indexOf(words[i]) < 0) return false
    return true
}

function typeColor(t) { return TYPE_COLORS[t] || "#8593a1" }
function categoryColor(c) { return CATEGORY_COLORS[c] || "#8593a1" }

function list(v) { return v || [] }

// A port's type is a string or an array of strings.
function types(port) {
    var t = port.type
    return Array.isArray(t) ? t : [t]
}

function compatible(outPort, inPort) {
    var o = types(outPort), i = types(inPort)
    for (var a = 0; a < o.length; a++)
        for (var b = 0; b < i.length; b++)
            if (o[a] === i[b]) return true
    return false
}

function nodeWidth(type) {
    switch (type) {
    case "prompt": case "sample": case "sample.video":
    case "image.save": case "video.save": case "image.load":
        return 260
    default:
        return 232
    }
}

function portIndex(ports, name) {
    for (var i = 0; i < ports.length; i++) if (ports[i].name === name) return i
    return -1
}

// Centre of a socket in canvas (world) coordinates.
function socketPos(node, spec, portName, isOutput) {
    var ports = isOutput ? list(spec.outputs) : list(spec.inputs)
    var i = Math.max(0, portIndex(ports, portName))
    return {
        x: node.pos[0] + (isOutput ? nodeWidth(node.type) : 0),
        y: node.pos[1] + FIRST_ROW + ROW * i
    }
}

// SVG path for a wire between two points: horizontal tangents, S-curve.
function wirePath(a, b) {
    var dx = Math.abs(b.x - a.x)
    var d = Math.max(36, Math.min(180, dx * 0.5))
    return "M " + a.x + " " + a.y + " C " + (a.x + d) + " " + a.y + ", " + (b.x - d) + " " + b.y + ", " + b.x + " " + b.y
}

// ---- graph document -------------------------------------------------------

function newGraph(name) {
    return { version: 1, name: name || "", nodes: [], edges: [], groups: [], view: { x: 40, y: 40, zoom: 1 } }
}

function clone(o) { return JSON.parse(JSON.stringify(o)) }

function findNode(graph, id) {
    for (var i = 0; i < graph.nodes.length; i++) if (graph.nodes[i].id === id) return graph.nodes[i]
    return null
}

function nextId(graph) {
    var max = 0
    for (var i = 0; i < graph.nodes.length; i++) {
        var m = /^n(\d+)$/.exec(graph.nodes[i].id)
        if (m) max = Math.max(max, parseInt(m[1], 10))
    }
    return "n" + (max + 1)
}

function defaultParams(spec) {
    var p = {}
    var ps = list(spec.params)
    for (var i = 0; i < ps.length; i++)
        if (ps[i].default !== undefined && ps[i].default !== null) p[ps[i].name] = ps[i].default
    return p
}

function addNode(graph, spec, x, y, params) {
    var node = { id: nextId(graph), type: spec.type, pos: [Math.round(x), Math.round(y)], params: defaultParams(spec) }
    if (params) for (var k in params) node.params[k] = params[k]
    graph.nodes.push(node)
    return node
}

function removeNode(graph, id) {
    graph.nodes = graph.nodes.filter(function(n) { return n.id !== id })
    graph.edges = graph.edges.filter(function(e) { return e.from[0] !== id && e.to[0] !== id })
    graph.groups = list(graph.groups).map(function(g) {
        var copy = clone(g)
        copy.nodes = list(g.nodes).filter(function(n) { return n !== id })
        return copy
    })
}

function duplicateNode(graph, node, spec) {
    var copy = addNode(graph, spec, node.pos[0] + 28, node.pos[1] + 28, clone(node.params || {}))
    if (node.title) copy.title = node.title
    return copy
}

function edgeInto(graph, nodeId, port) {
    for (var i = 0; i < graph.edges.length; i++) {
        var e = graph.edges[i]
        if (e.to[0] === nodeId && e.to[1] === port) return e
    }
    return null
}

function edgesFrom(graph, nodeId, port) {
    return graph.edges.filter(function(e) { return e.from[0] === nodeId && e.from[1] === port })
}

function removeEdge(graph, edge) {
    graph.edges = graph.edges.filter(function(e) { return e !== edge })
}

// Is `target` reachable from `start` following wires forward?
function reaches(graph, start, target) {
    var seen = {}
    var stack = [start]
    while (stack.length) {
        var cur = stack.pop()
        if (cur === target) return true
        if (seen[cur]) continue
        seen[cur] = true
        for (var i = 0; i < graph.edges.length; i++)
            if (graph.edges[i].from[0] === cur) stack.push(graph.edges[i].to[0])
    }
    return false
}

// Returns "" when from(out) -> to(in) is allowed, else the reason.
function connectError(graph, specs, from, to) {
    if (from.node === to.node) return "A node cannot feed itself"
    var fn = findNode(graph, from.node), tn = findNode(graph, to.node)
    if (!fn || !tn) return "Unknown node"
    var fs = specs[fn.type], ts = specs[tn.type]
    if (!fs || !ts) return "Unknown node type"
    var fi = portIndex(list(fs.outputs), from.port), ti = portIndex(list(ts.inputs), to.port)
    if (fi < 0 || ti < 0) return "Unknown port"
    if (!compatible(fs.outputs[fi], ts.inputs[ti]))
        return types(fs.outputs[fi]).join("|") + " does not fit " + types(ts.inputs[ti]).join("|")
    if (reaches(graph, to.node, from.node)) return "That would make a loop"
    return ""
}

// Wires from -> to, replacing whatever fed that input before.
function connect(graph, from, to) {
    graph.edges = graph.edges.filter(function(e) { return !(e.to[0] === to.node && e.to[1] === to.port) })
    graph.edges.push({ from: [from.node, from.port], to: [to.node, to.port] })
}

function effectiveParam(spec, node, name) {
    if (node.params && node.params[name] !== undefined && node.params[name] !== null) return node.params[name]
    var ps = list(spec.params)
    for (var i = 0; i < ps.length; i++) if (ps[i].name === name) return ps[i].default
    return undefined
}

// show_when: "<input> is <TYPE>" (what feeds that input) or "<param> is <value>".
function paramVisible(graph, specs, node, ps) {
    var w = ps.show_when
    if (!w) return true
    var m = /^(\w+) is (\w+)$/.exec(w)
    if (!m) return true
    var spec = specs[node.type]
    if (!spec) return true
    if (/^[A-Z]+$/.test(m[2]) && portIndex(list(spec.inputs), m[1]) >= 0) {
        var e = edgeInto(graph, node.id, m[1])
        if (!e) return false
        var src = findNode(graph, e.from[0])
        var ss = src && specs[src.type]
        if (!ss) return false
        var oi = portIndex(list(ss.outputs), e.from[1])
        return oi >= 0 && types(ss.outputs[oi]).indexOf(m[2]) >= 0
    }
    return String(effectiveParam(spec, node, m[1])) === m[2]
}

// Does editing this param change which params or sockets are shown?
function affectsLayout(spec, name) {
    var ps = list(spec.params)
    for (var i = 0; i < ps.length; i++) {
        var w = ps[i].show_when
        if (!w) continue
        var m = /^(\w+) is (\w+)$/.exec(w)
        if (m && m[1] === name) return true
    }
    return false
}

function isImageGen(m) {
    if (!m) return false
    var meta = m.metadata || {}
    if (meta.is_component || meta.component) return false
    if (m.modality === "diffusion" || meta.modality === "diffusion") return true
    return false
}

function fileUrlToLocalPath(fileUrl) {
    var s = String(fileUrl)
    if (s.indexOf("file://") !== 0) return s
    s = s.substring(7)
    try { s = decodeURIComponent(s) } catch (e) {}
    if (/^\/[A-Za-z]:\//.test(s)) s = s.substring(1)
    return s
}

function localPathToFileUrl(path) {
    var s = String(path || "").replace(/\\/g, "/")
    if (!s || s.indexOf("file://") === 0) return s
    if (/^[A-Za-z]:\//.test(s)) s = "/" + s
    if (s.indexOf("//") === 0) return "file:" + encodeURI(s).replace(/#/g, "%23").replace(/\?/g, "%3F")
    return "file://" + s.split("/").map(function(p) { return encodeURIComponent(p) }).join("/")
}

// Editor history contains document snapshots, never executor state. Adjacent
// typing in the same field shares one undo step until a pause or another edit.
function newHistory() { return { past: [], future: [] } }
function recordEdit(history, before, after, label, key, now) {
    if (JSON.stringify(before) === JSON.stringify(after)) return false
    var last = history.past[history.past.length - 1]
    if (key && last && last.key === key && !history.future.length && now - last.time < 750) {
        last.after = clone(after)
        last.time = now
    } else {
        history.past.push({ before: clone(before), after: clone(after), label: label,
                            key: key || "", time: now })
        if (history.past.length > 100) history.past.shift()
    }
    history.future = []
    return true
}
function undo(history) {
    var edit = history.past.pop()
    if (!edit) return null
    history.future.push(edit)
    return clone(edit.before)
}
function redo(history) {
    var edit = history.future.pop()
    if (!edit) return null
    history.past.push(edit)
    return clone(edit.after)
}

function idSet(ids) {
    var out = Object.create(null)
    for (var i = 0; i < ids.length; i++) out[ids[i]] = true
    return out
}
function connectedSection(graph, ids) {
    var seen = idSet(ids), todo = ids.slice()
    while (todo.length) {
        var id = todo.pop()
        for (var i = 0; i < graph.edges.length; i++) {
            var e = graph.edges[i], other = e.from[0] === id ? e.to[0] : e.to[0] === id ? e.from[0] : null
            if (other && !seen[other]) { seen[other] = true; todo.push(other) }
        }
    }
    return graph.nodes.filter(function(n) { return seen[n.id] }).map(function(n) { return n.id })
}
function copySection(graph, ids) {
    var selected = idSet(ids)
    return clone({ version: 1,
        nodes: graph.nodes.filter(function(n) { return selected[n.id] }),
        edges: graph.edges.filter(function(e) { return selected[e.from[0]] && selected[e.to[0]] }),
        groups: list(graph.groups).filter(function(g) {
            return list(g.nodes).length && g.nodes.every(function(id) { return selected[id] })
        }) })
}
// Accept only an editor fragment; external workflow JSON goes through the
// backend import API. Reject malformed/oversize clipboard input before edits.
function parseSection(text, specs) {
    if (text.length > 4 * 1024 * 1024) throw new Error("Clipboard section is too large")
    var s = JSON.parse(text)
    if (!s || s.version !== 1 || !Array.isArray(s.nodes) || !Array.isArray(s.edges)
            || !s.nodes.length || s.nodes.length > 1000 || s.edges.length > 4000)
        throw new Error("Copy some workflow nodes first")
    var seen = Object.create(null)
    s.nodes.forEach(function(n) {
        if (!n || typeof n.id !== "string" || seen[n.id] || !specs[n.type]
                || !Array.isArray(n.pos) || n.pos.length !== 2
                || !n.pos.every(function(v) { return typeof v === "number" && isFinite(v) })
                || (n.params && (typeof n.params !== "object" || Array.isArray(n.params))))
            throw new Error("Clipboard contains an unsupported node")
        seen[n.id] = true
    })
    s.edges.forEach(function(e) {
        if (!e || !Array.isArray(e.from) || !Array.isArray(e.to) || e.from.length !== 2 || e.to.length !== 2
                || !seen[e.from[0]] || !seen[e.to[0]]) throw new Error("Clipboard contains a broken wire")
    })
    if (s.groups !== undefined && !Array.isArray(s.groups)) throw new Error("Invalid clipboard frames")
    list(s.groups).forEach(function(g) {
        if (!g || typeof g.title !== "string" || !Array.isArray(g.nodes)
                || !g.nodes.every(function(id) { return seen[id] })) throw new Error("Invalid clipboard frame")
    })
    return s
}
function pasteSection(graph, section, x, y) {
    var x0 = Infinity, y0 = Infinity, map = Object.create(null), ids = []
    section.nodes.forEach(function(n) { x0 = Math.min(x0, n.pos[0]); y0 = Math.min(y0, n.pos[1]) })
    section.nodes.forEach(function(n) {
        var copy = clone(n)
        copy.id = nextId(graph)
        copy.pos = [Math.round(n.pos[0] - x0 + x), Math.round(n.pos[1] - y0 + y)]
        map[n.id] = copy.id
        ids.push(copy.id)
        graph.nodes.push(copy)
    })
    section.edges.forEach(function(e) { graph.edges.push({ from: [map[e.from[0]], e.from[1]], to: [map[e.to[0]], e.to[1]] }) })
    graph.groups = list(graph.groups)
    list(section.groups).forEach(function(g) {
        var copy = clone(g)
        copy.nodes = g.nodes.map(function(id) { return map[id] })
        graph.groups.push(copy)
    })
    return ids
}

function nodeRect(node, specs, sizes) {
    var size = sizes && sizes[node.id]
    return { x: node.pos[0], y: node.pos[1], w: nodeWidth(node.type),
             h: size ? size.h : (node.collapsed && specs[node.type] ? HEADER + 16 + ROW * Math.max(list(specs[node.type].inputs).length, list(specs[node.type].outputs).length)
                 : specs[node.type] ? estimateHeight(specs[node.type]) : 200) }
}
function selectionInRect(graph, specs, rect, sizes) {
    return graph.nodes.filter(function(n) {
        var b = nodeRect(n, specs, sizes)
        return b.x < rect.x + rect.w && b.x + b.w > rect.x && b.y < rect.y + rect.h && b.y + b.h > rect.y
    }).map(function(n) { return n.id })
}
function sectionBounds(graph, specs, ids, sizes) {
    var set = idSet(ids), nodes = graph.nodes.filter(function(n) { return set[n.id] })
    if (!nodes.length) return null
    var x = Infinity, y = Infinity, right = -Infinity, bottom = -Infinity
    nodes.forEach(function(n) {
        var b = nodeRect(n, specs, sizes)
        x = Math.min(x, b.x); y = Math.min(y, b.y)
        right = Math.max(right, b.x + b.w); bottom = Math.max(bottom, b.y + b.h)
    })
    return { x: x, y: y, w: right - x, h: bottom - y }
}
function alignNodes(graph, specs, ids, mode, sizes) {
    var set = idSet(ids), b = sectionBounds(graph, specs, ids, sizes)
    if (!b) return
    graph.nodes.forEach(function(n) {
        if (!set[n.id]) return
        var r = nodeRect(n, specs, sizes)
        if (mode === "left") n.pos[0] = b.x
        if (mode === "top") n.pos[1] = b.y
        if (mode === "right") n.pos[0] = b.x + b.w - r.w
        if (mode === "bottom") n.pos[1] = b.y + b.h - r.h
    })
}
// Position, labels, notes and viewport do not change a run's computation.
// Node IDs alone cannot establish that a run belongs on the editor canvas.
function stableValue(value) {
    if (Array.isArray(value)) return value.map(stableValue)
    if (value && typeof value === "object") {
        var out = {}
        Object.keys(value).sort().forEach(function(k) { out[k] = stableValue(value[k]) })
        return out
    }
    return value
}
function executionKey(graph) {
    if (!graph) return ""
    var nodes = list(graph.nodes).map(function(n) { return { id: n.id, type: n.type, params: n.params || {} } })
    nodes.sort(function(a, b) { return a.id.localeCompare(b.id) })
    var edges = list(graph.edges).map(function(e) { return { from: e.from, to: e.to } })
    edges.sort(function(a, b) { return JSON.stringify(a).localeCompare(JSON.stringify(b)) })
    return JSON.stringify(stableValue({ version: graph.version, nodes: nodes, edges: edges }))
}

// Rough size of a node for fitting the view before it has been laid out.
function estimateHeight(spec) {
    var rows = Math.max(list(spec.inputs).length, list(spec.outputs).length)
    var h = HEADER + 8 + rows * ROW + 14
    var ps = list(spec.params)
    for (var i = 0; i < ps.length; i++) h += (ps[i].kind === "text" ? 100 : 32)
    return h
}

// Bounding box of all nodes: {x, y, w, h}, or null for an empty graph.
function bounds(graph, specs) {
    if (!graph.nodes.length) return null
    var x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9
    for (var i = 0; i < graph.nodes.length; i++) {
        var n = graph.nodes[i]
        var sp = specs[n.type]
        var w = nodeWidth(n.type), h = sp ? estimateHeight(sp) : 200
        x0 = Math.min(x0, n.pos[0]); y0 = Math.min(y0, n.pos[1])
        x1 = Math.max(x1, n.pos[0] + w); y1 = Math.max(y1, n.pos[1] + h)
    }
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
}

// ---- starter graphs -------------------------------------------------------

var SAMPLE_PROMPT = "a cozy wooden cabin in a snowy forest at golden hour, detailed, cinematic light"
var SAMPLE_NEGATIVE = "blurry, low quality, watermark"

function templateList() {
    return [
        { id: "txt2img", title: "Text to image", hint: "Checkpoint, prompts, size, Sample, Save" },
        { id: "img2img", title: "Image to image", hint: "Start from a picture you choose" },
        { id: "hires", title: "Hires fix", hint: "Sample, Resize ×1.5, then a refining Sample" },
        { id: "txt2video", title: "Text to video", hint: "Video models such as Wan" }
    ]
}

function template(kind, specs, modelRef) {
    var g = newGraph()
    delete g.view            // a new graph opens fitted to the window
    var add = function(type, x, y, params) { return addNode(g, specs[type], x, y, params) }
    var wire = function(a, ap, b, bp) { connect(g, { node: a.id, port: ap }, { node: b.id, port: bp }) }
    var ckpt = add("checkpoint.load", 20, 24, { model: modelRef })
    var pos = add("prompt", 284, 24, { text: SAMPLE_PROMPT })
    pos.title = "Positive prompt"
    var neg = add("prompt", 284, 232, { text: SAMPLE_NEGATIVE })
    neg.title = "Negative prompt"

    if (kind === "txt2video") {
        var vl = add("latent.empty", 20, 250, { width: 832, height: 480 })
        var vs = add("sample.video", 576, 24, { frames: 33, fps: 16, steps: 20, cfg: 6 })
        var vv = add("video.save", 868, 24, { prefix: "video_" })
        wire(ckpt, "model", vs, "model"); wire(ckpt, "clip", vs, "clip"); wire(ckpt, "vae", vs, "vae")
        wire(pos, "cond", vs, "positive"); wire(neg, "cond", vs, "negative")
        wire(vl, "size", vs, "start"); wire(vs, "video", vv, "video")
        g.name = "Text to video"
        return g
    }

    var s1
    if (kind === "img2img") {
        var img = add("image.load", 20, 250, {})
        s1 = add("sample", 576, 24, { strength: 0.6 })
        wire(img, "image", s1, "start")
        g.name = "Image to image"
    } else {
        var lat = add("latent.empty", 20, 250, { width: 1024, height: 1024 })
        s1 = add("sample", 576, 24, {})
        wire(lat, "size", s1, "start")
        g.name = kind === "hires" ? "Hires fix" : "Text to image"
    }
    wire(ckpt, "model", s1, "model"); wire(ckpt, "clip", s1, "clip"); wire(ckpt, "vae", s1, "vae")
    wire(pos, "cond", s1, "positive"); wire(neg, "cond", s1, "negative")

    if (kind === "hires") {
        var rz = add("image.resize", 868, 24, { scale: 1.5 })
        var s2 = add("sample", 1132, 24, { strength: 0.35, steps: 15 })
        var sv = add("image.save", 1424, 24, {})
        wire(s1, "image", rz, "image"); wire(rz, "image", s2, "start")
        wire(ckpt, "model", s2, "model"); wire(ckpt, "clip", s2, "clip"); wire(ckpt, "vae", s2, "vae")
        wire(pos, "cond", s2, "positive"); wire(neg, "cond", s2, "negative")
        wire(s2, "image", sv, "image")
    } else {
        var save = add("image.save", 868, 24, {})
        wire(s1, "image", save, "image")
    }
    return g
}

// The document the API stores: strips nothing, but makes sure the shape is
// exactly what the backend's ParseGraph expects.
function forApi(graph, view) {
    var out = clone(graph)
    out.version = 1
    if (view) out.view = view
    return out
}

// Group issues by node id: { n3: [issue, ...] }; graph-level ones under "".
function issuesByNode(issues) {
    var out = {}
    for (var i = 0; i < issues.length; i++) {
        var k = issues[i].node || ""
        if (!out[k]) out[k] = []
        out[k].push(issues[i])
    }
    return out
}
