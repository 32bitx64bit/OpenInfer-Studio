package models

import (
	"bytes"
	"encoding/binary"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/migrations"
)

func writeTestGGUF(t *testing.T, dir, name, displayName string) string {
	t.Helper()
	var b bytes.Buffer
	w32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	w64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	wstr := func(s string) { w64(uint64(len(s))); b.WriteString(s) }
	binary.Write(&b, binary.LittleEndian, uint32(0x46554747))
	w32(3)
	w64(0)
	w64(4)
	wstr("general.name")
	w32(8)
	wstr(displayName)
	wstr("general.architecture")
	w32(8)
	wstr("llama")
	wstr("general.file_type")
	w32(4)
	w32(15)
	wstr("llama.context_length")
	w32(4)
	w32(8192)
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func testLibrary(t *testing.T) *Library {
	t.Helper()
	root := t.TempDir()
	dbDir := filepath.Join(root, "database")
	managed := filepath.Join(root, "models")
	for _, d := range []string{dbDir, managed} {
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
	return NewLibrary(db.DB, managed, nil, log)
}

func TestImportFileCopiesIntoManaged(t *testing.T) {
	lib := testLibrary(t)
	srcDir := t.TempDir()
	src := writeTestGGUF(t, srcDir, "Cool-Model-Q4_K_M.gguf", "Cool Model")

	id, err := lib.ImportFile(src)
	if err != nil {
		t.Fatalf("ImportFile: %v", err)
	}
	m, err := lib.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(filepath.Clean(m.PrimaryPath), filepath.Clean(lib.managed)+string(os.PathSeparator)) {
		t.Fatalf("primary_path %q not under managed %q", m.PrimaryPath, lib.managed)
	}
	if !strings.Contains(m.PrimaryPath, string(filepath.Separator)+"local--") {
		t.Fatalf("expected local-- layout, got %q", m.PrimaryPath)
	}
	if !strings.HasSuffix(m.PrimaryPath, "Cool-Model-Q4_K_M.gguf") {
		t.Fatalf("unexpected basename in %q", m.PrimaryPath)
	}
	if m.SourceRepo == "" || !strings.HasPrefix(m.SourceRepo, "local/") {
		t.Fatalf("source_repo = %q", m.SourceRepo)
	}
	if m.Alias != "Cool Model Q4_K_M" {
		t.Fatalf("alias = %q, want source name + quant", m.Alias)
	}
	// Original left in place.
	if _, err := os.Stat(src); err != nil {
		t.Fatalf("source should remain: %v", err)
	}
	// Managed copy exists and is readable.
	if _, err := os.Stat(m.PrimaryPath); err != nil {
		t.Fatal(err)
	}
	// Delete with files should remove the managed copy only.
	if _, err := lib.Delete(id, true); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := os.Stat(m.PrimaryPath); !os.IsNotExist(err) {
		t.Fatalf("managed copy should be gone, err=%v", err)
	}
	if _, err := os.Stat(src); err != nil {
		t.Fatalf("source should still exist: %v", err)
	}
}

func TestImportFileCopiesSplitAndMmproj(t *testing.T) {
	lib := testLibrary(t)
	srcDir := t.TempDir()
	a := writeTestGGUF(t, srcDir, "Big-00001-of-00002.gguf", "Big")
	writeTestGGUF(t, srcDir, "Big-00002-of-00002.gguf", "Big")
	writeTestGGUF(t, srcDir, "mmproj-Big.gguf", "mmproj")

	id, err := lib.ImportFile(a)
	if err != nil {
		t.Fatalf("ImportFile: %v", err)
	}
	m, err := lib.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Files) < 3 {
		t.Fatalf("files = %v, want shards + mmproj", m.Files)
	}
	if m.ProjectorPath == "" {
		t.Fatal("expected projector path")
	}
}

func TestImportFileAlreadyManaged(t *testing.T) {
	lib := testLibrary(t)
	destDir := filepath.Join(lib.managed, "local--Already", "files")
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		t.Fatal(err)
	}
	src := writeTestGGUF(t, destDir, "Already.gguf", "Already Here")
	id, err := lib.ImportFile(src)
	if err != nil {
		t.Fatal(err)
	}
	m, err := lib.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if m.PrimaryPath != src {
		t.Fatalf("should register in place, got %q want %q", m.PrimaryPath, src)
	}
}

func TestScanRefreshesUncustomizedLocalAlias(t *testing.T) {
	lib := testLibrary(t)
	destDir := filepath.Join(lib.managed, "local--Assistant-Q4_K_M", "files")
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := writeTestGGUF(t, destDir, "Assistant-Q4_K_M.gguf", "Muse Glimmer 30B Assistant")
	id, err := lib.ImportFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Pre-fix rows kept general.name as the alias; a rescan should append the quant.
	if _, err := lib.db.Exec(`UPDATE models SET alias=? WHERE id=?`, "Muse Glimmer 30B Assistant", id); err != nil {
		t.Fatal(err)
	}
	if _, err := lib.Scan(); err != nil {
		t.Fatal(err)
	}
	m, err := lib.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if m.Alias != "Muse Glimmer 30B Assistant Q4_K_M" {
		t.Fatalf("alias = %q, want source name + quant", m.Alias)
	}
}

func TestUpdateRejectsEmptyDisplayName(t *testing.T) {
	lib := testLibrary(t)
	srcDir := t.TempDir()
	src := writeTestGGUF(t, srcDir, "Cool-Model-Q4_K_M.gguf", "Cool Model")
	id, err := lib.ImportFile(src)
	if err != nil {
		t.Fatal(err)
	}
	empty := "   "
	if err := lib.Update(id, &empty, nil, nil, nil, nil); err != ErrEmptyDisplayName {
		t.Fatalf("Update empty alias: %v, want ErrEmptyDisplayName", err)
	}
	m, err := lib.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(m.Alias) == "" {
		t.Fatal("empty PATCH must not clear the display name")
	}
}

func TestScanKeepsCustomDisplayName(t *testing.T) {
	lib := testLibrary(t)
	destDir := filepath.Join(lib.managed, "local--Assistant-Q4_K_M", "files")
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := writeTestGGUF(t, destDir, "Assistant-Q4_K_M.gguf", "Muse Glimmer 30B Assistant")
	id, err := lib.ImportFile(path)
	if err != nil {
		t.Fatal(err)
	}
	custom := "My 27B Q3"
	if err := lib.Update(id, &custom, nil, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := lib.Scan(); err != nil {
		t.Fatal(err)
	}
	m, err := lib.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if m.Alias != custom {
		t.Fatalf("alias = %q, want custom display name kept across scan", m.Alias)
	}
}

func writeVocabLayoutGGUF(t *testing.T, dir, name string, fileType uint32, tokens []string, embd, vocabRows uint64) string {
	t.Helper()
	var b bytes.Buffer
	w32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	w64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	wstr := func(s string) { w64(uint64(len(s))); b.WriteString(s) }
	const (
		tUint32 = 4
		tString = 8
		tArray  = 9
	)
	binary.Write(&b, binary.LittleEndian, uint32(0x46554747))
	w32(3)
	w64(1)
	w64(4)
	wstr("general.architecture")
	w32(tString)
	wstr("llama")
	wstr("general.file_type")
	w32(tUint32)
	w32(fileType)
	wstr("llama.embedding_length")
	w32(tUint32)
	w32(uint32(embd))
	wstr("tokenizer.ggml.tokens")
	w32(tArray)
	w32(tString)
	w64(uint64(len(tokens)))
	for _, s := range tokens {
		wstr(s)
	}
	wstr("token_embd.weight")
	w32(2)
	w64(embd)
	w64(vocabRows)
	w32(0)
	w64(0)
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestHighPrecisionFromRepoSkipsInconsistentVocab(t *testing.T) {
	lib := testLibrary(t)
	srcDir := t.TempDir()
	src := writeVocabLayoutGGUF(t, srcDir, "pad-F16.gguf", 1, []string{"a", "b", "c", "d"}, 8, 6)
	id, err := lib.ImportFile(src)
	if err != nil {
		t.Fatal(err)
	}
	if err := lib.SetSourceRepo(id, "org/padded-embed"); err != nil {
		t.Fatal(err)
	}
	if got := lib.HighPrecisionFromRepo("org/padded-embed"); got != nil {
		t.Fatalf("inconsistent vocab GGUF must not be reused, got %s", got.PrimaryPath)
	}

	ok := writeVocabLayoutGGUF(t, srcDir, "ok-F16.gguf", 1, []string{"a", "b", "c", "d"}, 8, 4)
	okID, err := lib.ImportFile(ok)
	if err != nil {
		t.Fatal(err)
	}
	if err := lib.SetSourceRepo(okID, "org/aligned-embed"); err != nil {
		t.Fatal(err)
	}
	found := lib.HighPrecisionFromRepo("org/aligned-embed")
	if found == nil || found.ID != okID {
		t.Fatalf("aligned GGUF should be reused, got %+v", found)
	}
}
