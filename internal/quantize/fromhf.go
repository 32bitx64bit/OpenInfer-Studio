package quantize

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/uuid"
	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/convert"
	"github.com/openinfer/openinfer-studio/internal/downloads"
	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/huggingface"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
	"github.com/openinfer/openinfer-studio/internal/storage"
)

const KindFromHF = "from_hf"

// SnapshotFn writes a snapshot instead of downloading (tests).
type SnapshotFn func(ctx context.Context, destDir string, files []downloads.FileSpec) error

func (m *Manager) SetConvertDeps(hf *huggingface.Client, dl *downloads.Manager) {
	m.hf = hf
	m.dl = dl
}

func (m *Manager) SetSnapshotFn(fn SnapshotFn) { m.snapshotFn = fn }

func (m *Manager) SetProbeFn(fn func(ctx context.Context, repo string) (*FromHFPreview, error)) {
	m.probeFn = fn
}

// FromHFPreview is GET /quantize/from-hf/preview.
type FromHFPreview struct {
	convert.ProbeResult
	Repo          string `json:"repo"`
	SHA           string `json:"sha,omitempty"`
	ReusedModelID string `json:"reused_model_id,omitempty"`
	ReusedAlias   string `json:"reused_alias,omitempty"`
	DiskFree      uint64 `json:"disk_free"`
	DiskPeakBytes int64  `json:"disk_peak_bytes"`
	TightDisk     bool   `json:"tight_disk,omitempty"`
}

func (m *Manager) ProbeFromHF(ctx context.Context, repo string) (*FromHFPreview, error) {
	if m.probeFn != nil {
		return m.probeFn(ctx, repo)
	}
	repo, err := convert.NormalizeRepoID(repo)
	if err != nil {
		return nil, err
	}
	if m.hf == nil {
		return nil, fmt.Errorf("Hugging Face client is not configured")
	}
	info, err := m.hf.Repo(ctx, repo)
	if err != nil {
		return nil, err
	}
	files := make([]convert.NeededFile, 0, len(info.Files))
	for _, f := range info.Files {
		files = append(files, convert.NeededFile{Path: f.Path, Size: f.Size})
	}
	in := convert.ProbeInput{
		RepoID: info.ID,
		Tags:   info.Tags,
		Files:  files,
		DTypes: info.SafetensorsParameters,
	}
	if raw, ferr := m.hf.FetchFile(ctx, repo, "config.json", 4<<20); ferr == nil {
		cfg, err := convert.ParseConfig(raw)
		if err != nil {
			return nil, err
		}
		in.Config = cfg
		in.HasJSON = true
	}
	pr := convert.Evaluate(in)
	out := &FromHFPreview{ProbeResult: pr, Repo: info.ID, SHA: info.SHA}
	if m.lib != nil {
		if existing := m.lib.HighPrecisionFromRepo(info.ID); existing != nil {
			out.ReusedModelID = existing.ID
			out.ReusedAlias = existing.Alias
		}
	}
	ggufBytes := pr.EstimatedGGUFBytes
	snap := pr.SnapshotBytes
	if out.ReusedModelID != "" {
		out.DiskPeakBytes = ggufBytes + ggufBytes/10
	} else {
		peak := snap + ggufBytes
		out.DiskPeakBytes = peak + peak/10
	}
	if m.diskFree != nil && m.layout != nil {
		out.DiskFree = m.diskFree(m.layout.HFCache)
		snapshotNeed := uint64(snap) + uint64(snap)/10
		if out.DiskFree > 0 && snapshotNeed > out.DiskFree {
			out.Compatible = false
			if out.Reason == "" {
				out.Reason = "not enough free disk for Hugging Face snapshot cache"
			}
		} else if out.DiskFree > 0 && snapshotNeed*11/10 > out.DiskFree {
			out.TightDisk = true
			out.Warnings = append(out.Warnings, "disk space is tight for this conversion")
		}
	}
	return out, nil
}

// Dynamic From-HF preview reserves use the streaming-variant-bank model the
// authoritative scratch gate (checkQuantlabDisk → pipeline.EstimateScratch)
// enforces: at most one full-model anchor is in flight at a time (trimmed into
// variants ≈ the output size after each dtype), plus candidate/checkpoint/
// publication copies, imatrix, and held-out baseline logits.
const (
	// previewBytesPerSourceElem is the BF16 source storage per weight.
	previewBytesPerSourceElem = 2
	// previewMaxAnchorBPW is the largest dtype in every effort lattice (Q8_0),
	// so it anchors the largest possible full-model quantization.
	previewMaxAnchorBPW         = 8.5
	previewAnchorMetadataFactor = 1.02
	previewCandidateCopies      = 3
	previewIMatrixReserve       = int64(256) << 20
	previewSlackBytes           = int64(2) << 20
)

// previewLogitsReserve bounds held-out KLD baseline logits per effort
// (ctx × chunks × vocab × 4 without the exact vocab at preview time).
var previewLogitsReserve = map[string]int64{
	"fast":     int64(512) << 20,
	"profiled": int64(1) << 30,
	"deep":     int64(2) << 30,
}

// ProbeFromHFForRequest applies request-specific peak requirements to a normal
// repository preview. Dynamic conversion keeps the snapshot, converted source,
// one in-flight anchor plus promotion allowance, trimmed variants, and the
// candidate/final copies live at once — not one full anchor per dtype.
func (m *Manager) ProbeFromHFForRequest(ctx context.Context, repo string, req Request) (*FromHFPreview, error) {
	out, err := m.ProbeFromHF(ctx, repo)
	if err != nil || !usesQuantlab(req) {
		return out, err
	}
	resolved := req
	resolveQuantTier(&resolved)
	effort := resolveAdaptiveEffort(resolved)

	elems := float64(out.EstimatedGGUFBytes) / previewBytesPerSourceElem
	if elems <= 0 {
		elems = float64(out.SnapshotBytes) / previewBytesPerSourceElem
	}
	anchorEst := int64(elems * previewMaxAnchorBPW / 8 * previewAnchorMetadataFactor)
	var ladder int64
	switch effort {
	case "deep":
		ladder = anchorEst
	case "profiled":
		ladder = anchorEst / 2
	}
	var outputEst int64
	switch {
	case resolved.TargetBytes > 0:
		outputEst = resolved.TargetBytes
	case resolved.TargetBPW > 0:
		outputEst = int64(resolved.TargetBPW * elems / 8)
	default:
		outputEst = int64(defaultAdaptiveTargetBPW * elems / 8)
	}
	var imatrix int64
	if resolved.IMatrixID == "" {
		imatrix = previewIMatrixReserve
	}
	peak := saturatingPreviewAdd(anchorEst, ladder,
		saturatingPreviewMul(outputEst, previewCandidateCopies),
		imatrix, previewLogitsReserve[effort], previewSlackBytes)
	if out.ReusedModelID == "" {
		peak = saturatingPreviewAdd(peak, out.SnapshotBytes)
	}
	out.DiskPeakBytes = peak
	return out, nil
}

func saturatingPreviewAdd(values ...int64) int64 {
	var total int64
	for _, value := range values {
		if value > 0 && total > math.MaxInt64-value {
			return math.MaxInt64
		}
		if value < 0 && total < math.MinInt64-value {
			return math.MinInt64
		}
		total += value
	}
	return total
}

func saturatingPreviewMul(value, multiplier int64) int64 {
	if value <= 0 || multiplier <= 0 {
		return 0
	}
	if value > math.MaxInt64/multiplier {
		return math.MaxInt64
	}
	return value * multiplier
}

// canSkipFromHFProbe is true when a Dynamic from-HF job already has a library
// source and a loadable quantlab checkpoint. Resume must not hit the Hub or
// re-parse the high-precision GGUF just to publish an emitted candidate.
func canSkipFromHFProbe(layout *config.Layout, jobID string, req Request) bool {
	if layout == nil || strings.TrimSpace(jobID) == "" {
		return false
	}
	if !usesQuantlab(req) || strings.TrimSpace(req.SourceModelID) == "" {
		return false
	}
	stateDir := filepath.Join(layout.QuantJobs, jobID, quantlabDirName, "state")
	return quantlabCheckpointValid(stateDir, jobID)
}

func (m *Manager) runFromHF(ctx context.Context, j *Job, req Request, rt *runtimes.Runtime, tools runtimes.ToolsSnapshot) error {
	if strings.TrimSpace(req.HFRepo) == "" {
		return fmt.Errorf("hf_repo is required for kind=from_hf")
	}
	if canSkipFromHFProbe(m.layout, j.ID, req) {
		src, err := m.lib.Get(req.SourceModelID)
		if err != nil {
			return err
		}
		if !models.IsSpeculativeDraft(*src) && req.IMatrixID == "" {
			req.GenerateIMatrix = true
		}
		return m.runQuantlabAdaptive(ctx, j, req, src, rt, tools)
	}
	_ = m.setStage(j.ID, "probe", 0.02)
	prev, err := m.ProbeFromHF(ctx, req.HFRepo)
	if err != nil {
		return err
	}
	if !prev.Compatible && prev.ReusedModelID == "" {
		reason := prev.Reason
		if reason == "" {
			reason = "repository is not convertible"
		}
		return fmt.Errorf("%s", reason)
	}
	var src *models.Model
	if prev.ReusedModelID != "" {
		src, err = m.lib.Get(prev.ReusedModelID)
		if err != nil {
			return err
		}
		if err := gguf.CheckVocabLayout(src.PrimaryPath); err != nil {
			m.log.Info("existing GGUF vocab layout is inconsistent; reconverting", "path", src.PrimaryPath, "err", err)
			old := src.PrimaryPath
			src, err = m.downloadAndConvert(ctx, j, prev)
			if err != nil {
				return err
			}
			if old != "" && old != src.PrimaryPath {
				_ = os.Remove(old)
				if _, serr := m.lib.Scan(); serr == nil {
					if id := m.lib.IDForPath(src.PrimaryPath); id != "" {
						if refreshed, gerr := m.lib.Get(id); gerr == nil {
							src = refreshed
						}
					}
				}
			}
		}
		_, _ = m.db.Exec(`UPDATE quant_jobs SET source_model_id=?, updated_at=? WHERE id=?`, src.ID, now(), j.ID)
	} else {
		src, err = m.downloadAndConvert(ctx, j, prev)
		if err != nil {
			return err
		}
	}
	req.SourceModelID = src.ID
	j.Request = req
	if err := m.persistRequest(j.ID, req); err != nil {
		return err
	}
	if usesQuantlab(req) {
		if !models.IsSpeculativeDraft(*src) && req.IMatrixID == "" {
			req.GenerateIMatrix = true
		}
		if err := m.checkQuantlabDisk(src, req); err != nil {
			return err
		}
		return m.runQuantlabAdaptive(ctx, j, req, src, rt, tools)
	}
	return m.runQuantize(ctx, j, req, src, rt, tools, "")
}

func (m *Manager) downloadAndConvert(ctx context.Context, j *Job, prev *FromHFPreview) (*models.Model, error) {
	destDir := cacheDirFor(m.layout.HFCache, prev.Repo, prev.SHA)
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return nil, err
	}
	specs := make([]downloads.FileSpec, 0, len(prev.Files))
	for _, f := range prev.Files {
		if strings.Contains(f.Path, "..") {
			continue
		}
		url := ""
		if m.hf != nil {
			url = m.hf.DownloadURL(prev.Repo, f.Path)
		}
		specs = append(specs, downloads.FileSpec{
			URL:      url,
			DestPath: filepath.Join(destDir, filepath.FromSlash(f.Path)),
			Size:     f.Size,
		})
	}
	if len(specs) == 0 {
		return nil, fmt.Errorf("probe listed no files to download")
	}

	_ = m.setStage(j.ID, "download", 0.05)
	if m.snapshotFn != nil {
		if err := m.snapshotFn(ctx, destDir, specs); err != nil {
			return nil, err
		}
	} else {
		if m.dl == nil {
			return nil, fmt.Errorf("download manager is not configured")
		}
		id, err := m.dl.Enqueue("hf_snapshot", "HF "+prev.Repo, destDir, specs,
			map[string]string{"repo": prev.Repo, "kind": "hf_snapshot", "quant_job": j.ID})
		if err != nil {
			return nil, err
		}
		m.dlIDs.Store(j.ID, id)
		state, err := m.dl.WaitComplete(ctx, id)
		m.dlIDs.Delete(j.ID)
		if m.pauseIntent(j.ID) {
			_ = m.dl.Pause(id)
			if err != nil {
				return nil, err
			}
			if state != "complete" {
				return nil, context.Canceled
			}
		} else if err != nil {
			_ = m.dl.Cancel(id)
			return nil, fmt.Errorf("huggingface snapshot: %w", err)
		}
		if state != "complete" {
			return nil, fmt.Errorf("huggingface snapshot %s", state)
		}
	}

	dtype := prev.WeightDType
	if dtype == "" {
		dtype = "BF16"
	}
	ggufPath, err := m.destForConvert(prev.Repo, dtype)
	if err != nil {
		return nil, err
	}
	_ = m.setStage(j.ID, "convert", 0.35)
	ggufName := models.DisplayNameFromRepo(prev.Repo)
	if _, err := convert.ConvertDir(destDir, ggufPath, convert.ConvertOptions{Name: ggufName}); err != nil {
		_ = os.Remove(ggufPath)
		return nil, fmt.Errorf("convert: %w", err)
	}
	_ = os.RemoveAll(destDir)

	_ = m.setStage(j.ID, "scan", 0.55)
	if _, err := m.lib.Scan(); err != nil {
		return nil, err
	}
	id := m.lib.IDForPath(ggufPath)
	if id == "" {
		return nil, fmt.Errorf("converted GGUF was not registered by library scan")
	}
	if err := m.lib.SetSourceRepo(id, prev.Repo); err != nil {
		m.log.Warn("stamp source_repo after convert", "err", err)
	}
	src, err := m.lib.Get(id)
	if err != nil {
		return nil, err
	}
	if src.Quantization != "" && ggufName != "" {
		if err := m.lib.SetAlias(id, models.QuantizedAlias(ggufName, src.Quantization)); err != nil {
			m.log.Warn("stamp alias after convert", "err", err)
		} else if renamed, gerr := m.lib.Get(id); gerr == nil {
			src = renamed
		}
	}
	_, _ = m.db.Exec(`UPDATE quant_jobs SET source_model_id=?, updated_at=? WHERE id=?`, id, now(), j.ID)
	return src, nil
}

func cacheDirFor(layoutHFCache, repo, sha string) string {
	safe := strings.NewReplacer("/", "--", "..", "").Replace(repo)
	if sha == "" {
		sha = "main"
	}
	sha = strings.NewReplacer("/", "", "..", "").Replace(sha)
	if len(sha) > 16 {
		sha = sha[:16]
	}
	return filepath.Join(layoutHFCache, safe+"--"+sha)
}

func (m *Manager) destForConvert(repo, dtype string) (string, error) {
	base := safeName(strings.ReplaceAll(repo, "/", "-") + "-" + dtype)
	dir := filepath.Join(m.layout.Models, "local--"+base, "files")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	dest := filepath.Join(dir, base+".gguf")
	if _, err := os.Stat(dest); err == nil {
		if gguf.CheckVocabLayout(dest) == nil {
			dir = filepath.Join(m.layout.Models, "local--"+base+"-"+uuid.NewString()[:8], "files")
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return "", err
			}
			dest = filepath.Join(dir, base+".gguf")
		} else if m.log != nil {
			m.log.Info("replacing GGUF with inconsistent vocab layout", "path", dest)
		}
	}
	if _, err := storage.ValidateInside(m.layout.Models, dest); err != nil {
		return "", err
	}
	return dest, nil
}
