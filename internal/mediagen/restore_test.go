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

	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/gguf"
)

// writeReshapedGGUF writes a tiny GGUF the way ComfyUI-GGUF's converter does:
// an F32 tensor flattened to one dimension, its real [rows, cols] shape in a
// comfy.gguf.orig_shape key. With reshaped=false the tensor is stored as is.
func writeReshapedGGUF(t *testing.T, path string, reshaped bool) []byte {
	t.Helper()
	const rows, cols = 4, 6
	data := make([]byte, rows*cols*4)
	for i := range data {
		data[i] = byte(i*13 + 5)
	}
	str := func(b []byte, s string) []byte {
		b = binary.LittleEndian.AppendUint64(b, uint64(len(s)))
		return append(b, s...)
	}
	var kv []byte
	nKV := 0
	if reshaped {
		kv = str(kv, "comfy.gguf.orig_shape.blk.0.weight")
		kv = binary.LittleEndian.AppendUint32(kv, 9) // array
		kv = binary.LittleEndian.AppendUint32(kv, 5) // of int32
		kv = binary.LittleEndian.AppendUint64(kv, 2)
		kv = binary.LittleEndian.AppendUint32(kv, rows)
		kv = binary.LittleEndian.AppendUint32(kv, cols)
		nKV = 1
	}
	dims := []uint64{rows * cols}
	if !reshaped {
		dims = []uint64{cols, rows}
	}
	b := binary.LittleEndian.AppendUint32(nil, 0x46554747) // "GGUF"
	b = binary.LittleEndian.AppendUint32(b, 3)
	b = binary.LittleEndian.AppendUint64(b, 1)
	b = binary.LittleEndian.AppendUint64(b, uint64(nKV))
	b = append(b, kv...)
	b = str(b, "blk.0.weight")
	b = binary.LittleEndian.AppendUint32(b, uint32(len(dims)))
	for _, d := range dims {
		b = binary.LittleEndian.AppendUint64(b, d)
	}
	b = binary.LittleEndian.AppendUint32(b, 0) // F32
	b = binary.LittleEndian.AppendUint64(b, 0)
	b = append(b, make([]byte, (32-len(b)%32)%32)...)
	b = append(b, data...)
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return data
}

func newRestoreManager(t *testing.T) *Manager {
	t.Helper()
	return NewManager(nil, &config.Layout{DataDir: t.TempDir()}, nil, nil, nil, nil)
}

func TestGGUFArgIndexes(t *testing.T) {
	args := []string{"--diffusion-model", "/m/a.gguf", "--vae", "/m/v.safetensors", "--llm", "/m/L.GGUF", "--diffusion-fa", "--vae-tiling"}
	got := ggufArgIndexes(args)
	if len(got) != 2 || got[0] != 1 || got[1] != 5 {
		t.Errorf("ggufArgIndexes = %v", got)
	}
}

func TestRestoredNameTracksSourceState(t *testing.T) {
	prefix, a := restoredName("/models/My Model.gguf", 100, 5)
	_, same := restoredName("/models/My Model.gguf", 100, 5)
	_, resized := restoredName("/models/My Model.gguf", 101, 5)
	_, touched := restoredName("/models/My Model.gguf", 100, 6)
	otherPrefix, other := restoredName("/elsewhere/My Model.gguf", 100, 5)
	if a != same || a == resized || a == touched || a == other {
		t.Errorf("names do not follow path+size+mtime: %s %s %s %s %s", a, same, resized, touched, other)
	}
	if !strings.HasPrefix(a, prefix) || strings.HasPrefix(resized, otherPrefix) || strings.ContainsAny(a, " /") {
		t.Errorf("bad prefixes: %s (%s) vs %s (%s)", a, prefix, other, otherPrefix)
	}
}

func TestRestoreReshapedGGUFsSwapsAndCaches(t *testing.T) {
	m := newRestoreManager(t)
	src := filepath.Join(t.TempDir(), "enc.gguf")
	data := writeReshapedGGUF(t, src, true)
	plain := filepath.Join(t.TempDir(), "plain.gguf")
	writeReshapedGGUF(t, plain, false)
	vae := filepath.Join(t.TempDir(), "vae.safetensors")
	if err := os.WriteFile(vae, []byte("not a gguf"), 0o644); err != nil {
		t.Fatal(err)
	}

	args := []string{"--diffusion-model", plain, "--llm", src, "--vae", vae, "--diffusion-fa"}
	var log bytes.Buffer
	var progress []string
	if err := m.restoreReshapedGGUFs(context.Background(), args, &log, func(s string) { progress = append(progress, s) }); err != nil {
		t.Fatal(err)
	}
	if args[1] != plain || args[5] != vae || args[0] != "--diffusion-model" || args[6] != "--diffusion-fa" {
		t.Errorf("untouched arguments changed: %v", args)
	}
	restored := args[3]
	if restored == src || filepath.Dir(restored) != filepath.Join(m.MediaDir(), "restored") {
		t.Fatalf("--llm = %s, want a file under media/restored", restored)
	}
	if st, err := gguf.InspectComfyShapes(restored); err != nil || st.Needed() {
		t.Errorf("restored copy still needs restoring: %+v %v", st, err)
	}
	tensors, _, err := gguf.ListTensors(restored)
	if err != nil || len(tensors) != 1 || len(tensors[0].Shape) != 2 || tensors[0].Shape[0] != 6 || tensors[0].Shape[1] != 4 {
		t.Fatalf("restored tensors = %+v %v", tensors, err)
	}
	if raw, _ := os.ReadFile(restored); !bytes.HasSuffix(raw, data) {
		t.Error("restored payload differs from the source's")
	}
	if len(progress) == 0 || !strings.Contains(log.String(), "restoring tensor shapes of enc.gguf") {
		t.Errorf("no progress reported: %v / %q", progress, log.String())
	}
	if _, err := os.Stat(restored + ".part"); err == nil {
		t.Error("partial file left behind")
	}

	// A second launch reuses the copy instead of rewriting it.
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(restored, old, old); err != nil {
		t.Fatal(err)
	}
	args2 := []string{"--llm", src}
	log.Reset()
	if err := m.restoreReshapedGGUFs(context.Background(), args2, &log, nil); err != nil {
		t.Fatal(err)
	}
	if fi, _ := os.Stat(restored); args2[1] != restored || fi.ModTime().After(old.Add(time.Minute)) {
		t.Errorf("second launch did not reuse the cached copy: %s %v", args2[1], fi.ModTime())
	}
	if !strings.Contains(log.String(), "found a restored copy") {
		t.Errorf("log = %q", log.String())
	}

	// Changing the source replaces the cached copy rather than adding to it.
	writeReshapedGGUF(t, src, true)
	later := time.Now().Add(time.Minute)
	if err := os.Chtimes(src, later, later); err != nil {
		t.Fatal(err)
	}
	args3 := []string{"--llm", src}
	if err := m.restoreReshapedGGUFs(context.Background(), args3, &log, nil); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(filepath.Dir(restored))
	if args3[1] == restored || len(entries) != 1 {
		t.Errorf("stale copy kept: %s, %d files in cache", args3[1], len(entries))
	}
}

func TestRestoreReshapedGGUFsStopsWhenCanceled(t *testing.T) {
	m := newRestoreManager(t)
	src := filepath.Join(t.TempDir(), "enc.gguf")
	writeReshapedGGUF(t, src, true)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	args := []string{"--llm", src}
	if err := m.restoreReshapedGGUFs(ctx, args, &bytes.Buffer{}, nil); err == nil {
		t.Fatal("canceled restore succeeded")
	}
	if args[1] != src {
		t.Errorf("argv changed despite the failure: %v", args)
	}
	entries, _ := os.ReadDir(filepath.Join(m.MediaDir(), "restored"))
	if len(entries) != 0 {
		t.Errorf("files left behind: %v", entries)
	}
}

func TestReshapedWarnings(t *testing.T) {
	src := filepath.Join(t.TempDir(), "enc.gguf")
	writeReshapedGGUF(t, src, true)
	plain := filepath.Join(t.TempDir(), "plain.gguf")
	writeReshapedGGUF(t, plain, false)
	got := reshapedWarnings(plain, LoadSettings{LLM: src})
	if len(got) != 1 || !strings.Contains(got[0], "enc.gguf") || !strings.Contains(got[0], "LLM text encoder") {
		t.Errorf("warnings = %v", got)
	}
	if got := reshapedWarnings(plain, LoadSettings{}); len(got) != 0 {
		t.Errorf("plain model warned: %v", got)
	}
}

// newManagedManager is a manager whose models folder is <data>/models, so a
// GGUF placed there is one Studio manages.
func newManagedManager(t *testing.T) (*Manager, string) {
	t.Helper()
	dir := t.TempDir()
	models := filepath.Join(dir, "models")
	if err := os.MkdirAll(models, 0o755); err != nil {
		t.Fatal(err)
	}
	return NewManager(nil, &config.Layout{DataDir: dir, Models: models}, nil, nil, nil, nil), models
}

func restoredFiles(t *testing.T, m *Manager) []string {
	t.Helper()
	entries, _ := os.ReadDir(filepath.Join(m.MediaDir(), "restored"))
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names
}

func TestRestoreReshapedGGUFsReplacesManagedFile(t *testing.T) {
	m, models := newManagedManager(t)
	src := filepath.Join(models, "repo", "enc.gguf")
	if err := os.MkdirAll(filepath.Dir(src), 0o755); err != nil {
		t.Fatal(err)
	}
	data := writeReshapedGGUF(t, src, true)

	args := []string{"--llm", src}
	var log bytes.Buffer
	if err := m.restoreReshapedGGUFs(context.Background(), args, &log, nil); err != nil {
		t.Fatal(err)
	}
	if args[1] != src {
		t.Errorf("--llm = %s, want the original path", args[1])
	}
	if st, err := gguf.InspectComfyShapes(src); err != nil || st.Needed() {
		t.Errorf("file still reshaped: %+v %v", st, err)
	}
	if raw, _ := os.ReadFile(src); !bytes.HasSuffix(raw, data) {
		t.Error("payload changed")
	}
	if left := restoredFiles(t, m); len(left) != 0 {
		t.Errorf("a second copy was left behind: %v", left)
	}
	if !strings.Contains(log.String(), "replaced enc.gguf") {
		t.Errorf("log = %q", log.String())
	}

	// Nothing left to do on the next launch.
	args = []string{"--llm", src}
	log.Reset()
	if err := m.restoreReshapedGGUFs(context.Background(), args, &log, nil); err != nil {
		t.Fatal(err)
	}
	if args[1] != src || log.Len() != 0 {
		t.Errorf("second launch: %v %q", args, log.String())
	}
}

func TestRestoreReshapedGGUFsLeavesSymlinkedFileAlone(t *testing.T) {
	m, models := newManagedManager(t)
	outside := filepath.Join(t.TempDir(), "elsewhere.gguf")
	writeReshapedGGUF(t, outside, true)
	before, _ := os.ReadFile(outside)
	link := filepath.Join(models, "linked.gguf")
	if err := os.Symlink(outside, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	args := []string{"--llm", link}
	if err := m.restoreReshapedGGUFs(context.Background(), args, &bytes.Buffer{}, nil); err != nil {
		t.Fatal(err)
	}
	after, _ := os.ReadFile(outside)
	if !bytes.Equal(before, after) {
		t.Error("the file behind a symlink was modified")
	}
	if fi, err := os.Lstat(link); err != nil || fi.Mode()&os.ModeSymlink == 0 {
		t.Error("the symlink was replaced")
	}
	if args[1] == link || filepath.Dir(args[1]) != filepath.Join(m.MediaDir(), "restored") {
		t.Errorf("--llm = %s, want a copy under media/restored", args[1])
	}
}

func TestRestoreReshapedGGUFsPromotesCachedCopy(t *testing.T) {
	m, models := newManagedManager(t)
	src := filepath.Join(models, "enc.gguf")
	writeReshapedGGUF(t, src, true)

	// An earlier launch that did not own the file left a cached copy.
	unmanaged := NewManager(nil, &config.Layout{DataDir: m.layout.DataDir}, nil, nil, nil, nil)
	args := []string{"--llm", src}
	if err := unmanaged.restoreReshapedGGUFs(context.Background(), args, &bytes.Buffer{}, nil); err != nil {
		t.Fatal(err)
	}
	cached := args[1]
	if cached == src {
		t.Fatal("setup: expected a cached copy")
	}
	cachedBytes, _ := os.ReadFile(cached)

	args = []string{"--llm", src}
	var log bytes.Buffer
	if err := m.restoreReshapedGGUFs(context.Background(), args, &log, nil); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(src); args[1] != src || !bytes.Equal(got, cachedBytes) {
		t.Errorf("cached copy was not moved into place: %v", args)
	}
	if left := restoredFiles(t, m); len(left) != 0 {
		t.Errorf("cache not emptied: %v", left)
	}
	if strings.Contains(log.String(), "restoring tensor shapes of") {
		t.Errorf("rewrote the file instead of promoting the cached copy: %q", log.String())
	}
}
