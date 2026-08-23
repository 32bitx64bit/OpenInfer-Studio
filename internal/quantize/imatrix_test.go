package quantize

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func writeIMatrixGGUF(t *testing.T, dir, name string, chunkCount uint32) string {
	t.Helper()
	var b []byte
	put32 := func(v uint32) {
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], v)
		b = append(b, buf[:]...)
	}
	put64 := func(v uint64) {
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], v)
		b = append(b, buf[:]...)
	}
	putStr := func(s string) {
		put64(uint64(len(s)))
		b = append(b, s...)
	}
	put32(0x46554747)
	put32(3)
	put64(0)
	put64(2)
	putStr("general.type")
	put32(8)
	putStr("imatrix")
	putStr("imatrix.chunk_count")
	put32(4)
	put32(chunkCount)
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestIMatrixChunkCount(t *testing.T) {
	if imatrixChunkCount("/missing.gguf") != 0 {
		t.Fatal("missing file should read as 0 chunks")
	}
	path := writeIMatrixGGUF(t, t.TempDir(), "partial.gguf", 10)
	if got := imatrixChunkCount(path); got != 10 {
		t.Fatalf("chunk_count=%d want 10", got)
	}
}

func TestPrepareIMatrixContinue(t *testing.T) {
	dir := t.TempDir()
	out := writeIMatrixGGUF(t, dir, "job.gguf", 10)
	done, skip, inFile, cleanup, err := prepareIMatrixContinue(out, 612)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	if done || skip != 10 || inFile == "" {
		t.Fatalf("done=%v skip=%d in=%q", done, skip, inFile)
	}
	if imatrixChunkCount(inFile) != 10 {
		t.Fatal("continue copy lost chunk_count")
	}
	cleanup()
	if _, err := os.Stat(inFile); !os.IsNotExist(err) {
		t.Fatalf("continue copy not removed: %v", err)
	}

	done, skip, inFile, cleanup, err = prepareIMatrixContinue(out, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	if !done || skip != 10 || inFile != "" {
		t.Fatalf("complete partial: done=%v skip=%d in=%q", done, skip, inFile)
	}
}
