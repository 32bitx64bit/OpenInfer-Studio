package mediagen

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/hardware"
	"github.com/openinfer/openinfer-studio/internal/storage"
)

// ComfyUI-GGUF stores some tensors reshaped (flattened, or in rows of 256 for
// the k-quants) and keeps the real shape in comfy.gguf.orig_shape.* metadata.
// stable-diffusion.cpp reads the stored dims as they are, so it rejects such
// a file ("tensor … has wrong shape in model metadata") or even builds the
// wrong network from them. Before a launch, each such GGUF in the argv is
// restored (see gguf.RestoreComfyShapes):
//   - a file Studio manages (inside the models folder) is replaced by its
//     restored form, so only one copy stays on disk. The reshaped original is
//     of no use to anything Studio runs, and the swap is an atomic rename of a
//     validated file;
//   - any other file (an extra model directory, a symlink out of the models
//     folder) is never modified: its restored copy is kept under media/restored
//     and reused until the source changes.

// comfyRestoreVersion invalidates cached copies when the restore changes.
const comfyRestoreVersion = 1

// restoreSlack is the free space kept beyond the restored file itself.
const restoreSlack = 256 << 20

// ggufArgIndexes returns the positions in args of flag values that name a
// .gguf file.
func ggufArgIndexes(args []string) []int {
	var out []int
	for i := 1; i < len(args); i++ {
		if strings.HasPrefix(args[i-1], "-") && !strings.HasPrefix(args[i], "-") &&
			strings.EqualFold(filepath.Ext(args[i]), ".gguf") {
			out = append(out, i)
		}
	}
	return out
}

// reshapedWarnings are the pre-launch notes for every launch file that needs
// its shapes restored: shown in the load dialog and written to the log.
func reshapedWarnings(modelPath string, s LoadSettings) []string {
	var out []string
	for _, f := range launchFiles(modelPath, s) {
		if !strings.EqualFold(filepath.Ext(f.Path), ".gguf") {
			continue
		}
		st, err := gguf.InspectComfyShapes(f.Path)
		if err != nil || !st.Needed() {
			continue
		}
		out = append(out, fmt.Sprintf(
			"%s %s was converted with ComfyUI-GGUF, which stores %d tensors reshaped, and stable-diffusion.cpp cannot read that. Studio rewrites it with the original shapes the first time it loads (one time, needs about %s free while it works). A file outside Studio's models folder is left untouched and a corrected copy is kept in media/restored instead.",
			f.Role, filepath.Base(f.Path), st.Reshaped, humanBytes(st.OutputBytes)))
	}
	return out
}

func humanBytes(n int64) string {
	const unit = 1 << 10
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

// restoredName is the cache file name for a source: its base name, a hash of
// its path (so a changed copy of the same file replaces the old one) and a
// hash of its size, modification time and the restore version (so an edited
// source is not served a stale copy).
func restoredName(src string, size, mtimeNano int64) (prefix, name string) {
	abs, err := filepath.Abs(src)
	if err != nil {
		abs = src
	}
	pathSum := sha256.Sum256([]byte(abs))
	stateSum := sha256.Sum256([]byte(fmt.Sprintf("%d|%d|%d", size, mtimeNano, comfyRestoreVersion)))
	base := sanitizeID(strings.TrimSuffix(filepath.Base(src), filepath.Ext(src)))
	prefix = fmt.Sprintf("%s.%x.", base, pathSum[:4])
	return prefix, fmt.Sprintf("%s%x.gguf", prefix, stateSum[:6])
}

// managesFile reports whether path is a regular file inside Studio's own
// models folder (symlinks are resolved: a link out of it does not count), i.e.
// one Studio downloaded or imported and may rewrite.
func (m *Manager) managesFile(path string) bool {
	if m.layout == nil || m.layout.Models == "" {
		return false
	}
	if fi, err := os.Lstat(path); err != nil || !fi.Mode().IsRegular() {
		return false
	}
	_, err := storage.ValidateInside(m.layout.Models, path)
	return err == nil
}

// restoreReshapedGGUFs fixes, in args, every ComfyUI-GGUF-reshaped GGUF: a
// managed file is replaced in place (its path stays), any other is swapped
// for a restored copy. Files that are not GGUFs, cannot be parsed, or need
// nothing are left for sd-server to deal with. logw gets a line per file (with
// the file name); onProgress gets a short status text, e.g. "restoring tensor
// shapes (45%)", while a file is being written.
func (m *Manager) restoreReshapedGGUFs(ctx context.Context, args []string, logw io.Writer, onProgress func(string)) error {
	replaced := false
	for _, i := range ggufArgIndexes(args) {
		src := args[i]
		st, err := gguf.InspectComfyShapes(src)
		if err != nil || !st.Needed() {
			continue
		}
		path, didReplace, err := m.ensureRestored(ctx, src, st, logw, onProgress)
		if err != nil {
			return fmt.Errorf("could not restore the tensor shapes of %s (converted with ComfyUI-GGUF): %w", filepath.Base(src), err)
		}
		args[i] = path
		replaced = replaced || didReplace
	}
	if replaced && m.lib != nil {
		// File sizes changed under the library's rows; refresh them (and tell
		// the UI) without holding up the launch.
		go func() {
			if _, err := m.lib.Scan(); err != nil {
				m.log.Warn("rescanning the library after restoring model files failed", "err", err)
			}
		}()
	}
	return nil
}

// ensureRestored returns the path to launch with for a reshaped GGUF and
// whether it is src itself, replaced by its restored form.
func (m *Manager) ensureRestored(ctx context.Context, src string, st gguf.ComfyShapeStatus, logw io.Writer, onProgress func(string)) (path string, replaced bool, err error) {
	// One at a time: several launches can share a companion, and two writers
	// (or two swaps) would race on the same files.
	m.restoreMu.Lock()
	defer m.restoreMu.Unlock()
	if err := ctx.Err(); err != nil {
		return "", false, err
	}
	// Another launch may have restored it while this one waited.
	cur, err := gguf.InspectComfyShapes(src)
	if err != nil {
		return "", false, err
	}
	if !cur.Needed() {
		return src, false, nil
	}
	st = cur

	fi, err := os.Stat(src)
	if err != nil {
		return "", false, err
	}
	dir := filepath.Join(m.MediaDir(), "restored")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", false, err
	}
	prefix, name := restoredName(src, fi.Size(), fi.ModTime().UnixNano())
	dst := filepath.Join(dir, name)
	if rfi, err := os.Stat(dst); err == nil && rfi.Size() == st.OutputBytes {
		fmt.Fprintf(logw, "=== found a restored copy of %s: %s ===\n", filepath.Base(src), dst)
	} else {
		removeStaleRestored(dir, prefix, name)
		if free := hardware.DiskFree(dir); free != 0 && free < uint64(st.OutputBytes)+restoreSlack {
			return "", false, fmt.Errorf("not enough free disk space in %s: restoring needs %s, %s is free",
				dir, humanBytes(st.OutputBytes), humanBytes(int64(free)))
		}
		fmt.Fprintf(logw, "=== restoring tensor shapes of %s (%d tensors reshaped, %d re-encoded as F16, %s) ===\n",
			filepath.Base(src), st.Reshaped, st.Widened, humanBytes(st.OutputBytes))
		part := dst + ".part"
		lastPct := -10
		_, err = gguf.RestoreComfyShapes(ctx, src, part, func(done, total int64) {
			pct := 0
			if total > 0 {
				pct = int(100 * done / total)
			}
			if pct < lastPct+5 && pct != 100 {
				return
			}
			lastPct = pct
			if onProgress != nil {
				onProgress(fmt.Sprintf("restoring tensor shapes (%d%%)", pct))
			}
		})
		if err != nil {
			return "", false, err
		}
		if err := os.Rename(part, dst); err != nil {
			_ = os.Remove(part)
			return "", false, err
		}
	}

	if !m.managesFile(src) {
		fmt.Fprintf(logw, "=== %s is not in Studio's models folder, so it is left as it is; using the restored copy %s ===\n", filepath.Base(src), dst)
		return dst, false, nil
	}
	// Replace the reshaped original: nothing Studio runs can use it, and
	// keeping both would double the disk use. Same-volume rename is atomic.
	if err := os.Rename(dst, src); err != nil {
		fmt.Fprintf(logw, "=== could not replace %s with its restored form (%v); using the copy %s ===\n", filepath.Base(src), err, dst)
		return dst, false, nil
	}
	fmt.Fprintf(logw, "=== replaced %s with its restored form ===\n", filepath.Base(src))
	return src, true, nil
}

// removeStaleRestored deletes earlier restored copies (and abandoned partial
// writes) of the same source, which only differ from keep in their state hash.
func removeStaleRestored(dir, prefix, keep string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, e := range entries {
		n := e.Name()
		if strings.HasPrefix(n, prefix) && n != keep {
			_ = os.Remove(filepath.Join(dir, n))
		}
	}
}
