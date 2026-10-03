package mediagen

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

func writeLTXBroadcastGGUF(t *testing.T, path string) []byte {
	t.Helper()
	names := []string{"keyframes_abs_pos_embedding", "audio_scale_shift_table", "transformer_blocks.0.scale_shift_table"}
	b := binary.LittleEndian.AppendUint32(nil, 0x46554747)
	b = binary.LittleEndian.AppendUint32(b, 3)
	b = binary.LittleEndian.AppendUint64(b, uint64(len(names)))
	b = binary.LittleEndian.AppendUint64(b, 0)
	for i, name := range names {
		b = binary.LittleEndian.AppendUint64(b, uint64(len(name)))
		b = append(b, name...)
		b = binary.LittleEndian.AppendUint32(b, 1)
		b = binary.LittleEndian.AppendUint64(b, 16)
		b = binary.LittleEndian.AppendUint32(b, 1)
		b = binary.LittleEndian.AppendUint64(b, uint64(i*32))
	}
	b = append(b, make([]byte, (32-len(b)%32)%32)...)
	b = append(b, make([]byte, len(names)*32)...)
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return b
}

func TestLTXGPUPreparationPreservesSourceAndReusesValidatedCache(t *testing.T) {
	m, models := newManagedManager(t)
	src := filepath.Join(models, "ltx.gguf")
	original := writeLTXBroadcastGGUF(t, src)
	args := []string{"--diffusion-model", src, "--llm", src}
	var log bytes.Buffer
	var progress []string
	if err := m.prepareLTXBroadcast(context.Background(), args, runtimes.BackendHIP, &log, func(s string) { progress = append(progress, s) }); err != nil {
		t.Fatal(err)
	}
	dst := args[1]
	if dst == src || args[3] != src || filepath.Dir(dst) != filepath.Join(m.MediaDir(), "restored") {
		t.Fatalf("incorrect launch arguments: %v", args)
	}
	if after, _ := os.ReadFile(src); !bytes.Equal(original, after) {
		t.Fatal("managed source was modified")
	}
	if len(progress) == 0 || !strings.Contains(progress[len(progress)-1], "100%") {
		t.Fatalf("missing preparation progress: %v", progress)
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(dst, old, old); err != nil {
		t.Fatal(err)
	}
	args = []string{"--diffusion-model", src}
	if err := m.prepareLTXBroadcast(context.Background(), args, runtimes.BackendHIP, &log, nil); err != nil {
		t.Fatal(err)
	}
	fi, _ := os.Stat(dst)
	if args[1] != dst || !fi.ModTime().Equal(old) {
		t.Fatal("valid cache was rebuilt")
	}
	// Corrupt the cached tensor table without changing the file length. A
	// size-only cache check would incorrectly reuse this invalid file.
	b, _ := os.ReadFile(dst)
	b[0] = 0
	if err := os.WriteFile(dst, b, 0o644); err != nil {
		t.Fatal(err)
	}
	args = []string{"--diffusion-model", src}
	if err := m.prepareLTXBroadcast(context.Background(), args, runtimes.BackendHIP, &log, nil); err != nil {
		t.Fatal(err)
	}
	if issues, _, err := gguf.ValidateFile(args[1]); err != nil || len(issues) > 0 {
		t.Fatalf("corrupt cache was reused: %v %v", issues, err)
	}
}

func TestLTXGPUPreparationScopeAndCancellation(t *testing.T) {
	m := newRestoreManager(t)
	src := filepath.Join(t.TempDir(), "ltx.gguf")
	writeLTXBroadcastGGUF(t, src)
	args := []string{"--diffusion-model", src}
	if got, err := ltxBroadcastWarnings(args, runtimes.BackendHIP); err != nil || len(got) != 1 || !strings.Contains(got[0], "source file stay unchanged") {
		t.Fatalf("warnings = %v %v", got, err)
	}
	if err := m.prepareLTXBroadcast(context.Background(), args, runtimes.BackendCPU, &bytes.Buffer{}, nil); err != nil || args[1] != src {
		t.Fatalf("CPU preparation changed argv: %v %v", args, err)
	}
	if got, err := ltxBroadcastWarnings([]string{"--llm", src}, runtimes.BackendHIP); err != nil || len(got) != 0 {
		t.Fatalf("LLM preparation warned: %v %v", got, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := m.prepareLTXBroadcast(ctx, args, runtimes.BackendHIP, &bytes.Buffer{}, nil); err == nil || args[1] != src {
		t.Fatalf("canceled preparation = %v %v", args, err)
	}
	if files := restoredFiles(t, m); len(files) != 0 {
		t.Fatalf("canceled preparation left files: %v", files)
	}
}
