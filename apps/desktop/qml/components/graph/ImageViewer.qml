import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import "../.."
import ".."

// A separate native window. open() works independently of popup parenting.
Window {
    id: root
    property var sources: []
    property int currentIndex: 0
    property string comparisonSource: ""
    property string currentSource: sources && currentIndex >= 0 && currentIndex < sources.length
        ? String(sources[currentIndex]) : ""
    property bool comparing: comparisonSource !== ""
    property bool fullscreen: false
    property bool _wasMaximized: false

    title: "Image Studio · " + (currentSource ? (currentIndex + 1) + " / " + sources.length : "Results")
    width: 1180
    height: 800
    minimumWidth: 640
    minimumHeight: 420
    color: AppTheme.bg
    visible: false

    function open() {
        currentIndex = sources && sources.length ? Math.max(0, Math.min(sources.length - 1, currentIndex)) : 0
        mainImage.resetView(); referenceImage.resetView()
        show()
        raise()
        requestActivate()
    }
    function previous() { if (currentIndex > 0) --currentIndex }
    function next() { if (sources && currentIndex + 1 < sources.length) ++currentIndex }
    function toggleFullscreen() {
        if (visibility === Window.FullScreen) {
            if (_wasMaximized) showMaximized()
            else showNormal()
        } else {
            _wasMaximized = visibility === Window.Maximized
            showFullScreen()
        }
    }
    onVisibilityChanged: function(visibility) { fullscreen = visibility === Window.FullScreen }
    onSourcesChanged: {
        currentIndex = sources && sources.length ? Math.max(0, Math.min(sources.length - 1, currentIndex)) : 0
    }
    Component.onCompleted: AppTheme.applyPalette(root)

    Shortcut { sequence: "Left"; enabled: root.visible; onActivated: root.previous() }
    Shortcut { sequence: "Right"; enabled: root.visible; onActivated: root.next() }
    Shortcut { sequence: "F11"; enabled: root.visible; onActivated: root.toggleFullscreen() }
    Shortcut {
        sequence: "Escape"; enabled: root.visible
        onActivated: { if (root.fullscreen) root.toggleFullscreen(); else root.close() }
    }
    Shortcut { sequence: "+"; enabled: root.visible; onActivated: mainImage.zoomAt(mainImage.zoomFactor * 1.25, mainImage.width / 2, mainImage.height / 2) }
    Shortcut { sequence: "-"; enabled: root.visible; onActivated: mainImage.zoomAt(mainImage.zoomFactor / 1.25, mainImage.width / 2, mainImage.height / 2) }
    Shortcut { sequence: "0"; enabled: root.visible; onActivated: { mainImage.resetView(); referenceImage.resetView() } }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: AppTheme.padSmall
        spacing: AppTheme.gapTight
        RowLayout {
            Layout.fillWidth: true
            AppButton { objectName: "previousImageButton"; text: "←"; accessibleDescription: "Previous image"; enabled: root.currentIndex > 0; onClicked: root.previous() }
            Label { text: root.currentSource ? (root.currentIndex + 1) + " / " + root.sources.length : "No results"; color: AppTheme.text }
            AppButton { objectName: "nextImageButton"; text: "→"; accessibleDescription: "Next image"; enabled: root.sources && root.currentIndex + 1 < root.sources.length; onClicked: root.next() }
            Item { Layout.fillWidth: true }
            AppButton { text: "−"; accessibleDescription: "Zoom out"; enabled: mainImage.zoomFactor > 1; onClicked: mainImage.zoomAt(mainImage.zoomFactor / 1.25, mainImage.width / 2, mainImage.height / 2) }
            Label { text: mainImage.zoomFactor.toFixed(1) + "×"; color: AppTheme.textDim }
            AppButton { text: "+"; accessibleDescription: "Zoom in"; enabled: mainImage.zoomFactor < 16; onClicked: mainImage.zoomAt(mainImage.zoomFactor * 1.25, mainImage.width / 2, mainImage.height / 2) }
            AppButton { objectName: "fitImageButton"; text: "Fit"; onClicked: { mainImage.resetView(); referenceImage.resetView() } }
            AppButton { text: root.fullscreen ? "Window" : "Fullscreen"; onClicked: root.toggleFullscreen() }
            AppButton { text: "Close"; onClicked: root.close() }
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: AppTheme.gap
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Label { visible: root.comparing; text: "Current result"; color: AppTheme.textDim }
                ImageViewport { id: mainImage; objectName: "mainImageViewport"; Layout.fillWidth: true; Layout.fillHeight: true; fileUrl: root.currentSource }
            }
            ColumnLayout {
                visible: root.comparing
                Layout.fillWidth: true
                Layout.fillHeight: true
                Label { text: "Comparison"; color: AppTheme.textDim }
                ImageViewport { id: referenceImage; objectName: "comparisonImageViewport"; Layout.fillWidth: true; Layout.fillHeight: true; fileUrl: root.comparisonSource }
            }
        }
        Label {
            Layout.fillWidth: true
            text: root.currentSource
            textFormat: Text.PlainText
            elide: Text.ElideMiddle
            color: AppTheme.textDim
            font.pixelSize: AppTheme.fontSmall
        }
        Label {
            Layout.fillWidth: true
            text: "Wheel to zoom · Drag to pan · ← → for batch · Double-click to fit"
            color: AppTheme.textFaint
            font.pixelSize: AppTheme.fontSmall
            wrapMode: Text.Wrap
        }
    }
}
