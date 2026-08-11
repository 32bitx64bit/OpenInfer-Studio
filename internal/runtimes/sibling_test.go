package runtimes

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestFindDiffusionServer(t *testing.T) {
	dir := t.TempDir()
	server := filepath.Join(dir, "llama-server")
	if runtime.GOOS == "windows" {
		server += ".exe"
	}
	if err := os.WriteFile(server, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := FindDiffusionServer(server); err == nil {
		t.Fatal("expected error when visual server missing")
	}
	vis := filepath.Join(dir, "llama-diffusion-gemma-visual-server")
	if runtime.GOOS == "windows" {
		vis += ".exe"
	}
	if err := os.WriteFile(vis, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := FindDiffusionServer(server)
	if err != nil {
		t.Fatal(err)
	}
	if got != vis {
		t.Fatalf("got %q want %q", got, vis)
	}
}
