import QtQuick
import QtQuick.Controls
import "../.."

// A fit-to-view image with pointer-centred wheel zoom and bounded drag pan.
Rectangle {
    id: root
    property string fileUrl: ""
    property real zoomFactor: 1
    property real panX: 0
    property real panY: 0
    property size imageSize: picture.sourceSize
    property int imageStatus: picture.status
    property real fitWidth: picture.implicitWidth > 0 && picture.implicitHeight > 0
        ? Math.min(width, height * picture.implicitWidth / picture.implicitHeight) : width
    property real fitHeight: picture.implicitWidth > 0 && picture.implicitHeight > 0
        ? Math.min(height, width * picture.implicitHeight / picture.implicitWidth) : height
    property real maxPanX: Math.max(0, (fitWidth * zoomFactor - width) / 2)
    property real maxPanY: Math.max(0, (fitHeight * zoomFactor - height) / 2)

    color: AppTheme.bgAlt
    clip: true
    border.color: AppTheme.border
    Accessible.name: "Image preview"
    Accessible.description: fileUrl

    function resetView() { zoomFactor = 1; panX = 0; panY = 0 }
    function clampPan() {
        panX = Math.max(-maxPanX, Math.min(maxPanX, panX))
        panY = Math.max(-maxPanY, Math.min(maxPanY, panY))
    }
    function zoomAt(factor, pointX, pointY) {
        var next = Math.max(1, Math.min(16, factor))
        var ratio = next / zoomFactor
        panX = (pointX - width / 2) * (1 - ratio) + panX * ratio
        panY = (pointY - height / 2) * (1 - ratio) + panY * ratio
        zoomFactor = next
        clampPan()
    }
    onFileUrlChanged: resetView()
    onWidthChanged: clampPan()
    onHeightChanged: clampPan()

    Image {
        id: picture
        objectName: "viewerImage"
        source: root.fileUrl
        asynchronous: true
        // Retain original pixels for zoom inspection, rather than a thumbnail.
        cache: false
        fillMode: Image.PreserveAspectFit
        width: root.fitWidth * root.zoomFactor
        height: root.fitHeight * root.zoomFactor
        x: (root.width - width) / 2 + root.panX
        y: (root.height - height) / 2 + root.panY
        smooth: root.zoomFactor < 4
        onStatusChanged: if (status === Image.Ready) root.clampPan()
    }

    BusyIndicator { anchors.centerIn: parent; running: picture.status === Image.Loading }
    Label {
        anchors.centerIn: parent
        width: Math.max(0, parent.width - 32)
        horizontalAlignment: Text.AlignHCenter
        wrapMode: Text.Wrap
        text: root.fileUrl === "" ? "No image selected" : "Image unavailable\nThe result file may have moved or been removed."
        visible: root.fileUrl === "" || picture.status === Image.Error
        color: AppTheme.textDim
        font.pixelSize: AppTheme.fontBody
    }
    MouseArea {
        id: pointer
        objectName: "imagePanArea"
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton
        cursorShape: root.zoomFactor > 1 ? (pressed ? Qt.ClosedHandCursor : Qt.OpenHandCursor) : Qt.ArrowCursor
        property real lastX: 0
        property real lastY: 0
        onPressed: function(mouse) { lastX = mouse.x; lastY = mouse.y }
        onPositionChanged: function(mouse) {
            if (!pressed) return
            root.panX += mouse.x - lastX
            root.panY += mouse.y - lastY
            lastX = mouse.x; lastY = mouse.y
            root.clampPan()
        }
        onWheel: function(wheel) {
            var delta = wheel.angleDelta.y || wheel.pixelDelta.y
            root.zoomAt(root.zoomFactor * Math.pow(1.2, delta / 120), wheel.x, wheel.y)
            wheel.accepted = true
        }
        onDoubleClicked: root.resetView()
    }
}
