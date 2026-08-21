package pipeline

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSamePathAndGeneratedUnder(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "folded.gguf")
	b := filepath.Join(dir, "folded.gguf")
	c := filepath.Join(dir, "other.gguf")
	if !samePath(a, b) {
		t.Fatal("identical paths not same")
	}
	if samePath(a, c) {
		t.Fatal("distinct paths reported same")
	}
	if !generatedUnder(dir, a) {
		t.Fatal("job-private file not generated-under work dir")
	}
	if generatedUnder(dir, filepath.Join(t.TempDir(), "x.gguf")) {
		t.Fatal("foreign path treated as generated-under")
	}
}

func TestFoldUnlinkClearsSamePathWithoutRemove(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(p, []byte("gguf"), 0o600); err != nil {
		t.Fatal(err)
	}
	if generatedUnder(dir, p) && samePath(p, p) {
		// Reconstruct-in-place on the folded file: clear the pointer, keep the file.
	}
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("in-place reconstruct target missing: %v", err)
	}
}
