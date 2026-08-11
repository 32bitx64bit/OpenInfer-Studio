package runtimes

import (
	"archive/zip"
	"bytes"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/migrations"
)

func buildProbeFake(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	src := filepath.Join(dir, "main.go")
	body := `package main
import ("fmt"; "os")
func main() {
	if len(os.Args) > 1 && os.Args[1] == "--version" {
		fmt.Println("version fake-1")
		return
	}
	fmt.Println("usage: llama-server")
	fmt.Println("--ctx-size N")
	fmt.Println("--host HOST")
}
`
	if err := os.WriteFile(src, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	name := "llama-server"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	out := filepath.Join(dir, name)
	cmd := exec.Command("go", "build", "-o", out, src)
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build fake: %v\n%s", err, b)
	}
	return out
}

func makeZipBytes(t *testing.T, entries map[string][]byte) string {
	t.Helper()
	return makeZipBytesNamed(t, "runtime.zip", entries)
}

func makeZipBytesNamed(t *testing.T, name string, entries map[string][]byte) string {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for entry, body := range entries {
		w, err := zw.Create(entry)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write(body); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func testRuntimeManager(t *testing.T) *Manager {
	t.Helper()
	root := t.TempDir()
	rtDir := filepath.Join(root, "runtimes")
	dbDir := filepath.Join(root, "database")
	for _, d := range []string{rtDir, dbDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	db, err := database.Open(dbDir, migrations.FS)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	return NewManager(db.DB, rtDir, nil, nil, log)
}

func TestImportCustomArchive(t *testing.T) {
	fake := buildProbeFake(t)
	bin, err := os.ReadFile(fake)
	if err != nil {
		t.Fatal(err)
	}
	entry := "bin/llama-server"
	if runtime.GOOS == "windows" {
		entry = "bin/llama-server.exe"
	}
	archive := makeZipBytes(t, map[string][]byte{entry: bin})

	m := testRuntimeManager(t)
	id, err := m.ImportCustom(archive)
	if err != nil {
		t.Fatalf("ImportCustom archive: %v", err)
	}
	if !strings.HasPrefix(id, "custom-") {
		t.Fatalf("id = %q, want custom-*", id)
	}

	rt, err := m.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if rt.Source != "custom-import" {
		t.Errorf("source = %q", rt.Source)
	}
	if rt.InstallDir != filepath.Join(m.dir, id) {
		t.Errorf("install_dir = %q", rt.InstallDir)
	}
	if _, err := os.Stat(rt.ExecutablePath); err != nil {
		t.Errorf("executable missing: %v", err)
	}
	wantExe := filepath.Join(rt.InstallDir, "bin", filepath.Base(entry))
	if rt.ExecutablePath != wantExe {
		t.Errorf("executable_path = %q, want %q", rt.ExecutablePath, wantExe)
	}
	if !strings.Contains(rt.VersionOutput, "fake-1") {
		t.Errorf("version_output = %q", rt.VersionOutput)
	}
	found := false
	for _, c := range rt.Capabilities {
		if c == "ctx-size" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("capabilities missing ctx-size: %v", rt.Capabilities)
	}
	// Original archive left in place.
	if _, err := os.Stat(archive); err != nil {
		t.Errorf("archive should be untouched: %v", err)
	}
}

func TestImportCustomExecutable(t *testing.T) {
	fake := buildProbeFake(t)
	m := testRuntimeManager(t)
	id, err := m.ImportCustom(fake)
	if err != nil {
		t.Fatalf("ImportCustom exe: %v", err)
	}
	rt, err := m.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if rt.ExecutablePath != fake {
		t.Errorf("executable_path = %q, want original %q", rt.ExecutablePath, fake)
	}
	if _, err := os.Stat(fake); err != nil {
		t.Errorf("original exe should remain: %v", err)
	}
}

func TestImportCustomArchiveDetectsVulkanFromName(t *testing.T) {
	fake := buildProbeFake(t)
	bin, err := os.ReadFile(fake)
	if err != nil {
		t.Fatal(err)
	}
	entry := "bin/llama-server"
	if runtime.GOOS == "windows" {
		entry = "bin/llama-server.exe"
	}
	archive := makeZipBytesNamed(t, "llama-b6801-bin-ubuntu-vulkan-x64.zip", map[string][]byte{entry: bin})

	m := testRuntimeManager(t)
	id, err := m.ImportCustom(archive)
	if err != nil {
		t.Fatalf("ImportCustom: %v", err)
	}
	rt, err := m.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if rt.Backend != BackendVulkan {
		t.Fatalf("backend = %q, want vulkan", rt.Backend)
	}
}

func TestImportCustomArchiveMissingServer(t *testing.T) {
	archive := makeZipBytes(t, map[string][]byte{"README.md": []byte("no binary")})
	m := testRuntimeManager(t)
	if _, err := m.ImportCustom(archive); err == nil {
		t.Fatal("expected error for archive without llama-server")
	}
}
