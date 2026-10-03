pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../.."
import ".."
import "RunHistory.js" as History

Rectangle {
    id: root
    property var api
    property var events
    // An empty id shows all history. Queue always shows all workflows so every
    // waiting/running job remains individually cancelable.
    property string workflowId: ""
    property var runs: []
    property bool refreshing: false
    property string refreshError: ""
    property bool _ready: false
    property bool _refreshAgain: false
    property int _requestSerial: 0
    property int _cancelSerial: 0
    property var _cancelPending: ({})
    property var _cancelErrors: ({})
    property string _expandedRunId: ""
    property string _comparisonSource: ""
    property var queueRuns: []
    property var _historyRecords: []
    property var historyRuns: _historyRecords.filter(function(run) {
        return (!workflowFilter.checked || root.workflowId === "" || run.workflow_id === root.workflowId)
            && History.matches(run, search.text)
    })
    signal inspectRun(var run)
    signal restoreRun(var run)
    signal useImage(string fileUrl)
    signal compareImage(string fileUrl)

    implicitWidth: 380
    implicitHeight: 640
    color: AppTheme.bgAlt
    border.color: AppTheme.border

    function reload() {
        if (!api) return
        if (refreshing) { _refreshAgain = true; return }
        refreshDelay.stop()
        refreshing = true
        _refreshAgain = false
        var serial = ++_requestSerial
        var client = api
        client.get("/api/v1/workflow/runs", function(status, data) {
            // Ignore callbacks from a replaced API client and duplicate timeout
            // callbacks. Never let an older response roll history back.
            if (serial !== root._requestSerial || client !== root.api || !root.refreshing) return
            root.refreshing = false
            if (status === 200 && data && Array.isArray(data.runs)) {
                // Avoid recreating image delegates for identical idle snapshots.
                if (JSON.stringify(root.runs) !== JSON.stringify(data.runs)) root.runs = data.runs
                var queue = data.runs.filter(function(run) { return History.active(run) })
                var history = data.runs.filter(function(run) { return !History.active(run) })
                // Progress updates must not remount the entire finished gallery.
                if (JSON.stringify(root.queueRuns) !== JSON.stringify(queue)) root.queueRuns = queue
                if (JSON.stringify(root._historyRecords) !== JSON.stringify(history)) root._historyRecords = history
                root.refreshError = ""
                root._ready = true
            } else {
                root.refreshError = (data && (data.detail || data.error)) || (status ? "Could not load runs (HTTP " + status + ")" : "Backend unreachable")
            }
            if (root._refreshAgain) refreshDelay.start()
        })
    }
    function scheduleRefresh() {
        if (refreshing) _refreshAgain = true
        else if (!refreshDelay.running) refreshDelay.start()
    }
    function updateMap(current, key, value) {
        var next = {}
        Object.keys(current).forEach(function(id) { next[id] = current[id] })
        if (value === undefined) delete next[key]
        else next[key] = value
        return next
    }
    function cancelRun(run) {
        if (!api || !History.active(run) || _cancelPending[run.id]) return
        var id = run.id, client = api, serial = ++_cancelSerial
        _cancelPending = updateMap(_cancelPending, id, serial)
        _cancelErrors = updateMap(_cancelErrors, id, undefined)
        client.post("/api/v1/workflow/runs/" + encodeURIComponent(id) + "/cancel", {}, function(status, data) {
            if (client !== root.api || root._cancelPending[id] !== serial) return
            root._cancelPending = root.updateMap(root._cancelPending, id, undefined)
            if (status < 200 || status >= 300)
                root._cancelErrors = root.updateMap(root._cancelErrors, id, (data && (data.detail || data.error)) || "Could not cancel this run")
            root.reload()
        })
    }
    function viewResults(run, index) {
        viewer.sources = History.resultImages(run).map(function(image) { return image.fileUrl })
        viewer.currentIndex = index
        viewer.open()
    }
    function selectComparison(fileUrl) {
        _comparisonSource = fileUrl
        compareImage(fileUrl)
    }
    function toggleRunDetails(id) { _expandedRunId = _expandedRunId === id ? "" : id }

    Component.onCompleted: reload()
    onApiChanged: {
        ++_requestSerial
        refreshing = false; _refreshAgain = false
        _cancelPending = ({}); _cancelErrors = ({})
        runs = []; queueRuns = []; _historyRecords = []; _ready = false; refreshError = ""
        if (api) scheduleRefresh()
    }
    onVisibleChanged: if (visible) scheduleRefresh()
    Connections {
        target: root.events || null
        function onEventReceived(name, payload) {
            if (name.indexOf("workflow.") === 0) root.scheduleRefresh()
        }
        function onReconnected() { root.scheduleRefresh() }
    }
    Timer { id: refreshDelay; interval: 180; onTriggered: root.reload() }
    Timer {
        interval: root.queueRuns.length ? 3000 : 12000
        running: root.visible && !!root.api
        repeat: true
        onTriggered: root.reload()
    }
    ImageViewer { id: viewer; objectName: "historyImageViewer"; comparisonSource: root._comparisonSource }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: AppTheme.padSmall
        spacing: AppTheme.gapTight
        RowLayout {
            Layout.fillWidth: true
            Label { text: "Results & history"; color: AppTheme.text; font.pixelSize: AppTheme.fontTitle; font.weight: Font.DemiBold; Layout.fillWidth: true }
            AppButton { objectName: "refreshRunsButton"; text: root.refreshing ? "Refreshing…" : "Refresh"; flat: true; enabled: !!root.api && !root.refreshing; onClicked: root.reload() }
        }
        Label {
            visible: root.refreshError !== ""
            Layout.fillWidth: true
            text: root.refreshError
            textFormat: Text.PlainText
            color: AppTheme.danger
            wrapMode: Text.Wrap
            font.pixelSize: AppTheme.fontSmall
        }
        SearchField { id: search; objectName: "runHistorySearch"; Layout.fillWidth: true; placeholderText: "Search prompts, names, or run IDs"; searchLabel: "Search run history" }
        AppCheckBox { id: workflowFilter; objectName: "runWorkflowFilter"; visible: root.workflowId !== ""; checked: true; text: "History for this workflow" }
        ScrollView {
            id: scroll
            objectName: "runHistoryScroll"
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            contentWidth: availableWidth
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
            ColumnLayout {
                width: scroll.availableWidth
                spacing: AppTheme.gapTight
                Label {
                    Layout.fillWidth: true
                    text: "Queue · " + root.queueRuns.length + " · All workflows"
                    color: AppTheme.textDim
                    font.weight: Font.DemiBold
                    font.pixelSize: AppTheme.fontSmall
                }
                Label {
                    Layout.fillWidth: true
                    visible: root._ready && !root.queueRuns.length
                    text: "No queued or running jobs"
                    color: AppTheme.textFaint
                    font.pixelSize: AppTheme.fontSmall
                }
                Repeater {
                    model: root.queueRuns
                    delegate: RunHistoryCard {
                        required property var modelData
                        objectName: "queueRun_" + modelData.id
                        Layout.fillWidth: true
                        runData: modelData
                        expanded: root._expandedRunId === modelData.id
                        cancelPending: !!root._cancelPending[modelData.id]
                        cancelError: root._cancelErrors[modelData.id] || ""
                        onInspectRequested: root.inspectRun(modelData)
                        onRestoreRequested: root.restoreRun(modelData)
                        onCancelRequested: root.cancelRun(modelData)
                        onToggleDetails: root.toggleRunDetails(modelData.id)
                        onViewRequested: function(imageIndex) { root.viewResults(modelData, imageIndex) }
                        onUseRequested: function(fileUrl) { root.useImage(fileUrl) }
                        onCompareRequested: function(fileUrl) { root.selectComparison(fileUrl) }
                    }
                }
                Label { Layout.fillWidth: true; text: "History · " + root.historyRuns.length; color: AppTheme.textDim; font.weight: Font.DemiBold; font.pixelSize: AppTheme.fontSmall; topPadding: 8 }
                Repeater {
                    model: root.historyRuns
                    delegate: RunHistoryCard {
                        required property var modelData
                        objectName: "historyRun_" + modelData.id
                        Layout.fillWidth: true
                        runData: modelData
                        expanded: root._expandedRunId === modelData.id
                        onInspectRequested: root.inspectRun(modelData)
                        onRestoreRequested: root.restoreRun(modelData)
                        onToggleDetails: root.toggleRunDetails(modelData.id)
                        onViewRequested: function(imageIndex) { root.viewResults(modelData, imageIndex) }
                        onUseRequested: function(fileUrl) { root.useImage(fileUrl) }
                        onCompareRequested: function(fileUrl) { root.selectComparison(fileUrl) }
                    }
                }
                Label {
                    Layout.fillWidth: true
                    visible: !root.historyRuns.length
                    text: !root._ready ? (root.refreshing ? "Loading run history…" : "Run history unavailable")
                        : root.runs.length ? "No finished runs match this history filter" : "Run a workflow to see its results here"
                    color: AppTheme.textFaint
                    font.pixelSize: AppTheme.fontSmall
                    wrapMode: Text.Wrap
                }
            }
        }
    }
}
