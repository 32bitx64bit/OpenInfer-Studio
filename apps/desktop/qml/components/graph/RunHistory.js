.pragma library

// Presentation of durable run snapshots only. No inference, cache decisions,
// progress estimates, or inferred run state belong in the history UI.
function active(run) {
    return run && (run.state === "queued" || run.state === "running")
}

function nodeRows(run) {
    var rows = [], seen = {}, states = (run && run.nodes) || {}
    var graphNodes = (run && run.graph && run.graph.nodes) || []
    for (var i = 0; i < graphNodes.length; ++i) {
        var node = graphNodes[i]
        if (!states[node.id]) continue
        rows.push({ id: node.id, title: node.title || node.type || node.id,
                    type: node.type || "", params: node.params || {}, status: states[node.id] })
        seen[node.id] = true
    }
    Object.keys(states).sort().forEach(function(id) {
        if (!seen[id]) rows.push({ id: id, title: id, type: "", params: {}, status: states[id] })
    })
    return rows
}

function pathKey(value) {
    var key = String(value || "")
    try { key = decodeURIComponent(key) } catch (e) {}
    // localhost is the same local file authority as an empty authority.
    return key.replace(/^file:\/\/localhost\//i, "file:///")
}

function imageUrl(value) {
    return /^file:\/\//i.test(String(value || ""))
        && /\.(png|jpe?g|webp|gif|bmp|tiff?|avif|svg)$/i.test(pathKey(value))
}

function resultImages(run) {
    var rows = nodeRows(run), seen = {}, images = []
    // Saved results lead the batch; identical paths emitted by downstream
    // preview/save nodes appear once. Keep distinct branch/intermediate files.
    rows = rows.filter(function(row) { return row.type === "image.save" })
        .concat(rows.filter(function(row) { return row.type !== "image.save" }))
    rows.forEach(function(row) {
        var urls = row.status.file_urls || [], outputs = row.status.outputs || []
        urls.forEach(function(url, index) {
            if (!imageUrl(url)) return
            var key = pathKey(url)
            if (seen[key]) return
            seen[key] = true
            images.push({ fileUrl: String(url), nodeId: row.id, nodeTitle: row.title,
                          outputPath: outputs[index] || "", saved: row.type === "image.save" })
        })
    })
    return images
}

function prompts(run) {
    var nodes = (run && run.graph && run.graph.nodes) || [], result = []
    nodes.forEach(function(node) {
        var params = node.params || {}
        if (node.type === "prompt" && params.text)
            result.push({ title: node.title || "Prompt", text: String(params.text) })
        // Accommodate snapshots that also retain resolved generation params.
        if (params.prompt) result.push({ title: node.title || "Prompt", text: String(params.prompt) })
        if (params.negative_prompt)
            result.push({ title: "Negative prompt", text: String(params.negative_prompt) })
    })
    return result
}

function title(run) {
    return (run && run.graph && run.graph.name) || (run && run.id) || "Run"
}

function matches(run, query) {
    var terms = String(query || "").trim().toLowerCase().split(/\s+/)
    var text = [title(run), run.id || "", run.state || "", run.error || ""]
    prompts(run).forEach(function(prompt) { text.push(prompt.title, prompt.text) })
    nodeRows(run).forEach(function(row) { text.push(row.title, row.status.message || "") })
    var haystack = text.join(" ").toLowerCase()
    return terms.every(function(term) { return haystack.indexOf(term) !== -1 })
}

function seed(status) {
    if (!status) return ""
    var value = status.resolved_seed
    if (value === undefined || value === null) value = status.seed
    if (value !== undefined && value !== null && Number(value) >= 0) return String(value)
    var found = String(status.message || "").match(/\bseed\s*[:=]?\s*(\d+)\b/i)
    return found ? found[1] : ""
}

function seedSummary(run) {
    var seeds = []
    nodeRows(run).forEach(function(row) {
        var value = seed(row.status)
        if (value !== "") seeds.push(row.title + ": " + value)
    })
    return seeds.join(" · ")
}

function duration(ms) {
    if (!(ms >= 0)) return ""
    return ms < 1000 ? Math.round(ms) + " ms" : (ms / 1000).toFixed(1) + " s"
}

function timestamp(value) {
    if (!value) return ""
    var date = new Date(value)
    return isNaN(date.getTime()) ? String(value) : date.toLocaleString()
}

function progressText(status) {
    var p = status.progress
    var text = status.message || (p && p.message) || status.state || ""
    if (p && p.total > 0 && !(p.message || status.message))
        text += " · " + (p.current || 0) + "/" + p.total + (p.unit ? " " + p.unit : "")
    return text
}
