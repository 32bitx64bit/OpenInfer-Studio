package storage

import (
	"os"
	"path/filepath"
	"testing"
)

func TestValidateInsideTempNestedOutput(t *testing.T) {
	root := filepath.Join(t.TempDir(), "models")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(root, "local--Quant-Test-Q4_K_M", "files", "Quant-Test-Q4_K_M.gguf")
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := ValidateInside(root, dest)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(got) != "Quant-Test-Q4_K_M.gguf" {
		t.Fatalf("got %q", got)
	}
}

func TestValidateInsideSymlinkRoot(t *testing.T) {
	real := filepath.Join(t.TempDir(), "real-models")
	if err := os.MkdirAll(real, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(t.TempDir(), "models")
	if err := os.Symlink(real, link); err != nil {
		t.Skip(err)
	}
	dest := filepath.Join(link, "local--x", "files", "x.gguf")
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := ValidateInside(link, dest); err != nil {
		t.Fatal(err)
	}
}

func TestValidateInsideRelative(t *testing.T) {
	root := filepath.Join(t.TempDir(), "models")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := ValidateInside(root, filepath.Join("local--x", "files", "x.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(got) != "x.gguf" {
		t.Fatalf("got %q", got)
	}
}

func TestValidateInsideRejectsEscape(t *testing.T) {
	root := filepath.Join(t.TempDir(), "models")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "evil.gguf")
	if _, err := ValidateInside(root, outside); err == nil {
		t.Fatal("expected escape")
	}
	if _, err := ValidateInside(root, filepath.Join("..", "evil.gguf")); err == nil {
		t.Fatal("expected .. escape")
	}
	if _, err := ValidateInside(root, ""); err == nil {
		t.Fatal("expected empty")
	}
}

func TestSafeJoin(t *testing.T) {
	root := t.TempDir()
	got, err := SafeJoin(root, "a", "b.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(got) != "b.gguf" {
		t.Fatalf("got %q", got)
	}
	if _, err := SafeJoin(root, "..", "b.gguf"); err == nil {
		t.Fatal("expected escape")
	}
}
