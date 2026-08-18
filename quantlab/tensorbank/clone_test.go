package tensorbank

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestCloneFileCopiesBytes(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src.bin")
	dst := filepath.Join(dir, "sub", "dst.bin")
	payload := bytes.Repeat([]byte("gguf"), 1024)
	if err := os.WriteFile(src, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := CloneFile(src, dst); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("clone bytes = %d, want %d matching payload", len(got), len(payload))
	}
}

func TestCloneFileSamePathNoOp(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src.bin")
	if err := os.WriteFile(src, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := CloneFile(src, src); err != nil {
		t.Fatal(err)
	}
}

func TestCloneFileEmptyPath(t *testing.T) {
	if err := CloneFile("", "dst"); err == nil {
		t.Fatal("empty source accepted")
	}
	if err := CloneFile("src", ""); err == nil {
		t.Fatal("empty dest accepted")
	}
}
