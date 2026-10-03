import QtQuick
import QtTest
import "../../apps/desktop/qml/js/graphModel.js" as GM

TestCase {
    name: "GraphEditorModel"
    function fixture() {
        return { version: 1, nodes: [
            { id: "n1", type: "prompt", pos: [10, 20], params: { text: "a" } },
            { id: "n2", type: "prompt", pos: [310, 120], params: { text: "b" }, collapsed: true },
            { id: "n3", type: "prompt", pos: [610, 220], params: {} }
        ], edges: [ { from: ["n1", "cond"], to: ["n2", "input"] },
                    { from: ["n2", "cond"], to: ["n3", "input"] } ],
            groups: [{ title: "Prompts", note: "Keep this note", color: "#6aa8e0", nodes: ["n1", "n2"] }] }
    }
    readonly property var specs: ({ prompt: { params: [], inputs: [], outputs: [] } })

    function test_typingCoalescesAndBranches() {
        var h = GM.newHistory(), a = { text: "a" }, b = { text: "ab" }, c = { text: "abc" }
        verify(GM.recordEdit(h, a, b, "Type", "n1:text", 100))
        verify(GM.recordEdit(h, b, c, "Type", "n1:text", 300))
        compare(h.past.length, 1)
        compare(GM.undo(h), a)
        compare(GM.redo(h), c)
        GM.recordEdit(h, c, { text: "pause" }, "Type", "n1:text", 1200)
        compare(h.past.length, 2)
        GM.undo(h)
        GM.recordEdit(h, c, { text: "branch" }, "Type", "n1:text", 1300)
        compare(h.future.length, 0)
        compare(h.past.length, 2)
    }
    function test_copyPasteRemapsInternalWiresAndFrames() {
        var g = fixture(), section = GM.copySection(g, ["n1", "n2"])
        compare(section.edges.length, 1)
        var ids = GM.pasteSection(g, GM.parseSection(JSON.stringify(section), specs), 1000, 500)
        compare(ids, ["n4", "n5"])
        compare(g.nodes[3].pos, [1000, 500])
        compare(g.nodes[4].pos, [1300, 600])
        verify(g.nodes[4].collapsed)
        compare(g.edges[2], { from: ["n4", "cond"], to: ["n5", "input"] })
        compare(g.groups[1].nodes, ["n4", "n5"])
        compare(g.groups[1].note, "Keep this note")
        g.nodes[3].params.text = "edited copy"
        compare(g.nodes[0].params.text, "a")
    }
    function test_clipboardRejectsMalformedAndExternalNodes() {
        var malformed = fixture()
        malformed.nodes[1].id = "n1"
        var rejected = false
        try { GM.parseSection(JSON.stringify(malformed), specs) } catch (e) { rejected = true }
        verify(rejected)
        malformed = fixture(); malformed.edges[0].to[0] = "missing"
        rejected = false
        try { GM.parseSection(JSON.stringify(malformed), specs) } catch (e) { rejected = true }
        verify(rejected)
    }
    function test_connectivitySelectionDeletionAndAlignment() {
        var g = fixture()
        compare(GM.connectedSection(g, ["n2"]), ["n1", "n2", "n3"])
        compare(GM.selectionInRect(g, specs, { x: 0, y: 0, w: 280, h: 90 }), ["n1"])
        GM.alignNodes(g, specs, ["n1", "n2"], "left")
        compare(g.nodes[1].pos[0], 10)
        compare(g.nodes[2].pos[0], 610)
        GM.removeNode(g, "n1")
        compare(g.groups[0].nodes, ["n2"])
        compare(g.edges.length, 1)
    }
    function test_executionIdentityIgnoresOnlyPresentation() {
        var a = fixture(), b = GM.clone(a)
        b.nodes.reverse(); b.edges.reverse(); b.nodes[0].pos = [-100, 30]
        b.groups[0].note = "Different note"; b.name = "Renamed"
        b.nodes[0].collapsed = true
        compare(GM.executionKey(a), GM.executionKey(b))
        b.nodes[0].params.text = "different input"
        verify(GM.executionKey(a) !== GM.executionKey(b))
    }
    function test_fileUrlsRoundtripSpecialCharacters() {
        var p = "/tmp/a #image? 100%.png"
        var url = GM.localPathToFileUrl(p)
        verify(url.indexOf("%23") > 0)
        verify(url.indexOf("%3F") > 0)
        compare(GM.fileUrlToLocalPath(url), p)
    }
}
