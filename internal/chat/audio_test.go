package chat

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildOAIMessagesWithAudio(t *testing.T) {
	s := testService(t)
	dir := t.TempDir()
	s.SetAttachDir(dir)

	c, err := s.CreateConversation("m", "", "sys")
	if err != nil {
		t.Fatal(err)
	}
	wav := []byte("RIFF....WAVEfmt ") // minimal placeholder bytes
	um, err := s.addMessage(c.ID, "", "user", "Transcribe this\n[Audio attached]")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.persistAudio(um.ID, &AudioInput{
		Data:   base64.StdEncoding.EncodeToString(wav),
		Format: "wav",
		Name:   "sample.wav",
	}); err != nil {
		t.Fatal(err)
	}
	am, _ := s.addMessage(c.ID, um.ID, "assistant", "hello")
	chain, err := s.chain(am.ID)
	if err != nil {
		t.Fatal(err)
	}
	msgs, err := s.buildOAIMessages(*c, chain)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 3 { // system + user + assistant
		t.Fatalf("want 3 messages, got %d", len(msgs))
	}
	if msgs[0].Role != "system" || msgs[0].Content != "sys" {
		t.Fatalf("system: %+v", msgs[0])
	}
	parts, ok := msgs[1].Content.([]map[string]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("user content should be multipart: %#v", msgs[1].Content)
	}
	if parts[0]["type"] != "text" {
		t.Fatalf("first part: %#v", parts[0])
	}
	if parts[1]["type"] != "input_audio" {
		t.Fatalf("second part: %#v", parts[1])
	}
	ia, _ := parts[1]["input_audio"].(map[string]any)
	if ia["format"] != "wav" {
		t.Fatalf("format: %#v", ia)
	}
	decoded, err := base64.StdEncoding.DecodeString(ia["data"].(string))
	if err != nil || string(decoded) != string(wav) {
		t.Fatalf("audio data mismatch")
	}
	if msgs[2].Content != "hello" {
		t.Fatalf("assistant: %#v", msgs[2].Content)
	}
}

func TestBuildOAIMessagesTextOnly(t *testing.T) {
	s := testService(t)
	c, _ := s.CreateConversation("m", "", "")
	um, _ := s.addMessage(c.ID, "", "user", "hi")
	am, _ := s.addMessage(c.ID, um.ID, "assistant", "yo")
	chain, _ := s.chain(am.ID)
	msgs, err := s.buildOAIMessages(*c, chain)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 {
		t.Fatalf("got %d", len(msgs))
	}
	if msgs[0].Content != "hi" {
		t.Fatalf("%#v", msgs[0].Content)
	}
}

func TestBuildOAIMessagesIncludesReasoning(t *testing.T) {
	s := testService(t)
	c, _ := s.CreateConversation("m", "", "")
	um, _ := s.addMessage(c.ID, "", "user", "hi")
	am, _ := s.addMessage(c.ID, um.ID, "assistant", "yo")
	_, err := s.db.Exec(`UPDATE conversation_messages SET reasoning=? WHERE id=?`, "because 2+2", am.ID)
	if err != nil {
		t.Fatal(err)
	}
	chain, _ := s.chain(am.ID)
	// chain() re-reads from DB, so reasoning is populated.
	msgs, err := s.buildOAIMessages(*c, chain)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 {
		t.Fatalf("got %d", len(msgs))
	}
	if msgs[1].ReasoningContent != "because 2+2" {
		t.Fatalf("reasoning_content = %#v", msgs[1].ReasoningContent)
	}
	if msgs[0].ReasoningContent != "" {
		t.Fatalf("user must not carry reasoning_content: %#v", msgs[0])
	}
}

func TestResolveAudioBytesRejectsBadFormat(t *testing.T) {
	_, _, _, err := resolveAudioBytes(&AudioInput{Data: base64.StdEncoding.EncodeToString([]byte("x")), Format: "exe"})
	if err == nil {
		t.Fatal("expected format error")
	}
}

func TestPersistAudioFromPath(t *testing.T) {
	s := testService(t)
	dir := t.TempDir()
	s.SetAttachDir(dir)
	src := filepath.Join(dir, "in.wav")
	if err := os.WriteFile(src, []byte("audio-bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	c, _ := s.CreateConversation("m", "", "")
	um, _ := s.addMessage(c.ID, "", "user", "[Audio attached]")
	if err := s.persistAudio(um.ID, &AudioInput{Path: src}); err != nil {
		t.Fatal(err)
	}
	att, err := s.audioAttachment(um.ID)
	if err != nil || att == nil {
		t.Fatalf("att=%v err=%v", att, err)
	}
	if att.format != "wav" {
		t.Fatalf("format %q", att.format)
	}
	raw, _ := os.ReadFile(att.path)
	if string(raw) != "audio-bytes" {
		t.Fatalf("copied %q", raw)
	}
	// Ensure DB row exists.
	var n int
	_ = s.db.QueryRow(`SELECT COUNT(1) FROM attachments WHERE message_id=?`, um.ID).Scan(&n)
	if n != 1 {
		t.Fatalf("attachment rows=%d", n)
	}
	if !strings.HasPrefix(att.path, dir) {
		t.Fatalf("path outside attach dir: %s", att.path)
	}
}
