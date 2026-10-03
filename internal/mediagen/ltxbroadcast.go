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
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

const ltxBroadcastVersion = 1

func ltxGPUBackend(backend string) bool {
	return backend == runtimes.BackendHIP || backend == runtimes.BackendCUDA
}

func diffusionGGUFIndexes(args []string) []int {
	var out []int
	for _, i := range ggufArgIndexes(args) {
		switch args[i-1] {
		case "--model", "--diffusion-model", "--high-noise-diffusion-model":
			out = append(out, i)
		}
	}
	return out
}

func ltxBroadcastWarnings(args []string, backend string) ([]string, error) {
	if !ltxGPUBackend(backend) {
		return nil, nil
	}
	var out []string
	for _, i := range diffusionGGUFIndexes(args) {
		st, err := gguf.InspectLTXBroadcast(args[i])
		if err != nil && st.Detected {
			return out, fmt.Errorf("cannot prepare LTX GPU compatibility for %s: %w", filepath.Base(args[i]), err)
		}
		if err == nil && st.Needed() {
			out = append(out, fmt.Sprintf("%s has %d half-precision LTX embedding/modulation tables that this GPU backend cannot broadcast safely. Studio prepares a cached compatibility copy with those small tables losslessly converted to F32; the other weights and source file stay unchanged. First preparation needs about %s free disk space.", filepath.Base(args[i]), st.TensorCount, humanBytes(st.OutputBytes)))
		}
	}
	return out, nil
}

func ltxBroadcastName(src string, size, mtimeNano int64) (prefix, name string) {
	abs, err := filepath.Abs(src)
	if err != nil {
		abs = src
	}
	pathSum := sha256.Sum256([]byte(abs))
	stateSum := sha256.Sum256([]byte(fmt.Sprintf("%d|%d|%d", size, mtimeNano, ltxBroadcastVersion)))
	base := sanitizeID(strings.TrimSuffix(filepath.Base(src), filepath.Ext(src)))
	prefix = fmt.Sprintf("ltx-f32.%s.%x.", base, pathSum[:4])
	return prefix, fmt.Sprintf("%s%x.gguf", prefix, stateSum[:6])
}

// prepareLTXBroadcast runs after shape restoration. Only diffusion weights on
// CUDA/HIP need this workaround: sd.cpp excludes these parameters from its
// --tensor-type-rules conversion, so setting that flag does not fix them.
func (m *Manager) prepareLTXBroadcast(ctx context.Context, args []string, backend string, logw io.Writer, onProgress func(string)) error {
	if !ltxGPUBackend(backend) {
		return nil
	}
	for _, i := range diffusionGGUFIndexes(args) {
		st, err := gguf.InspectLTXBroadcast(args[i])
		if err != nil && st.Detected {
			return fmt.Errorf("cannot prepare LTX GPU compatibility for %s: %w", filepath.Base(args[i]), err)
		}
		if err != nil || !st.Needed() {
			continue
		}
		path, err := m.ensureLTXBroadcast(ctx, args[i], logw, onProgress)
		if err != nil {
			return fmt.Errorf("preparing LTX GPU compatibility for %s: %w", filepath.Base(args[i]), err)
		}
		args[i] = path
	}
	return nil
}

func (m *Manager) ensureLTXBroadcast(ctx context.Context, src string, logw io.Writer, onProgress func(string)) (string, error) {
	m.restoreMu.Lock()
	defer m.restoreMu.Unlock()
	if err := ctx.Err(); err != nil {
		return "", err
	}
	st, err := gguf.InspectLTXBroadcast(src)
	if err != nil {
		return "", err
	}
	if !st.Needed() {
		return src, nil
	}
	fi, err := os.Stat(src)
	if err != nil {
		return "", err
	}
	dir := filepath.Join(m.MediaDir(), "restored")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	prefix, name := ltxBroadcastName(src, fi.Size(), fi.ModTime().UnixNano())
	dst := filepath.Join(dir, name)
	if rfi, err := os.Stat(dst); err == nil && rfi.Size() == st.OutputBytes {
		issues, _, verr := gguf.ValidateFile(dst)
		repaired, rerr := gguf.InspectLTXBroadcast(dst)
		if verr == nil && rerr == nil && len(issues) == 0 && !repaired.Needed() && repaired.OutputBytes == st.OutputBytes {
			fmt.Fprintf(logw, "=== reusing LTX GPU compatibility copy of %s: %s ===\n", filepath.Base(src), dst)
			return dst, nil
		}
	}
	part := dst + ".part"
	_ = os.Remove(part) // An interrupted preparation is never used as a model.
	if free := hardware.DiskFree(dir); free != 0 && free < uint64(st.OutputBytes)+restoreSlack {
		return "", fmt.Errorf("not enough free disk space in %s: LTX compatibility preparation needs %s, %s is free", dir, humanBytes(st.OutputBytes), humanBytes(int64(free)))
	}
	fmt.Fprintf(logw, "=== preparing LTX GPU compatibility copy of %s (%d tables to F32, %s added, source preserved) ===\n", filepath.Base(src), st.TensorCount, humanBytes(st.AddedBytes))
	lastPct := -5
	_, err = gguf.RepairLTXBroadcast(ctx, src, part, func(done, total int64) {
		pct := int(100 * done / total)
		if pct < lastPct+5 && pct != 100 {
			return
		}
		lastPct = pct
		if onProgress != nil {
			onProgress(fmt.Sprintf("preparing LTX GPU compatibility (%d%%)", pct))
		}
	})
	if err != nil {
		return "", err
	}
	if err := os.Rename(part, dst); err != nil {
		_ = os.Remove(part)
		return "", err
	}
	removeStaleRestored(dir, prefix, name)
	fmt.Fprintf(logw, "=== LTX GPU compatibility copy validated: %s ===\n", dst)
	return dst, nil
}
