import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".."
import "../js/codeHighlight.js" as CodeHL

// Renders chat Markdown with fenced-code syntax highlighting.
// While streaming, falls back to Qt MarkdownText (incomplete fences).
ColumnLayout {
    id: root
    property string content: ""
    property bool streaming: false
    spacing: 8

    readonly property var tokenColors: ({
        "keyword": String(AppTheme.info),
        "string": String(AppTheme.success),
        "comment": String(AppTheme.textFaint),
        "number": String(AppTheme.warning)
    })

    readonly property var parts: {
        if (root.streaming || root.content === "")
            return []
        return CodeHL.splitFences(root.content)
    }

    readonly property bool useSplit: {
        if (root.streaming || root.content === "")
            return false
        var p = root.parts
        for (var i = 0; i < p.length; i++) {
            if (p[i].type === "code")
                return true
        }
        return false
    }

    // Streaming uses plain text so each token paints; MarkdownText often
    // does not relayout while the string is still growing.
    TextEdit {
        Layout.fillWidth: true
        Layout.preferredHeight: contentHeight
        visible: !root.useSplit && root.content !== ""
        text: root.content
        textFormat: root.streaming ? TextEdit.PlainText : TextEdit.MarkdownText
        readOnly: true
        selectByMouse: true
        wrapMode: TextEdit.Wrap
        color: AppTheme.text
        font.pixelSize: AppTheme.fontBody
    }

    Repeater {
        model: root.useSplit ? root.parts : []
        delegate: Item {
            Layout.fillWidth: true
            implicitHeight: modelData.type === "code" ? codeBlock.implicitHeight
                : (mdEdit.visible ? mdEdit.implicitHeight : 0)

            TextEdit {
                id: mdEdit
                anchors.left: parent.left
                anchors.right: parent.right
                visible: modelData.type === "md" && modelData.text.trim() !== ""
                text: modelData.text
                textFormat: TextEdit.MarkdownText
                readOnly: true
                selectByMouse: true
                wrapMode: TextEdit.Wrap
                color: AppTheme.text
                font.pixelSize: AppTheme.fontBody
            }

            Rectangle {
                id: codeBlock
                anchors.left: parent.left
                anchors.right: parent.right
                visible: modelData.type === "code"
                implicitHeight: visible ? codeCol.implicitHeight + 16 : 0
                radius: AppTheme.radiusSmall
                color: AppTheme.surfaceHi
                border.color: AppTheme.border

                ColumnLayout {
                    id: codeCol
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.margins: 8
                    spacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: modelData.lang !== "" ? modelData.lang : "code"
                            color: AppTheme.textFaint
                            font.pixelSize: AppTheme.fontSmall
                            font.family: "monospace"
                        }
                        Item { Layout.fillWidth: true }
                        AppButton {
                            flat: true
                            text: "Copy"
                            onClicked: {
                                codeCopy.text = modelData.text
                                codeCopy.selectAll()
                                codeCopy.copy()
                            }
                        }
                    }

                    TextEdit {
                        Layout.fillWidth: true
                        text: CodeHL.highlight(modelData.text, modelData.lang, root.tokenColors)
                        textFormat: TextEdit.RichText
                        readOnly: true
                        selectByMouse: true
                        wrapMode: TextEdit.Wrap
                        color: AppTheme.text
                        font.family: "monospace"
                        font.pixelSize: AppTheme.fontSmall
                    }
                }
            }
        }
    }

    TextEdit {
        id: codeCopy
        visible: false
        text: ""
    }
}
