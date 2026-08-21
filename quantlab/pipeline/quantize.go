package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"quantlab/core"
	"quantlab/encode"
	"quantlab/orchestrate"
	"quantlab/profile"
	"quantlab/tensorbank"
)

// anchorMeta records which floor dtypes already have a trimmed anchor in a
// given anchor subdirectory, so interrupted runs extend rather than rebuild.
type anchorMeta struct {
	DTypes []core.DType `json:"dtypes,omitempty"`
}

func (m *anchorMeta) has(d core.DType) bool {
	for _, x := range m.DTypes {
		if x == d {
			return true
		}
	}
	return false
}

func (e *Engine) anchorMetaPath(name string) string {
	return filepath.Join(e.anchorDir(), name)
}

func (e *Engine) loadAnchorMeta(name string) *anchorMeta {
	m := &anchorMeta{}
	if data, err := os.ReadFile(e.anchorMetaPath(name)); err == nil {
		_ = json.Unmarshal(data, m)
	}
	return m
}

func (e *Engine) saveAnchorMeta(name string, m *anchorMeta) error {
	return e.writeJSON(e.anchorMetaPath(name), m)
}

// variantsDir is the quantize-stage trimmed variant anchor directory: each
// file holds only the tensors whose manifest option is that dtype.
func (e *Engine) variantsDir() string { return filepath.Join(e.anchorDir(), "variants") }

// stageQuantize processes distinct dtypes (ordered by estimated anchor-file
// size DESCENDING so the largest job starts first): trim the source to the
// keep-set, run llama-quantize --pure on that subset, and place the result
// as variants/anchor-NNN-DTYPE.gguf. Independent dtypes run in a worker pool
// of at most 2, gated on free disk. Assembly (tensorbank.Build) then
// consumes only the trimmed variant files. Kept tensor payloads and hashes
// are identical to quantize-full-then-trim; only wall-clock and scratch
// change.
func (e *Engine) stageQuantize(ctx context.Context) error {
	manifest := e.Run.Manifest
	if manifest == nil {
		return fmt.Errorf("pipeline: no manifest; solve stage incomplete")
	}
	if err := e.bindManifestSource(manifest); err != nil {
		return err
	}
	seen := map[core.DType]bool{}
	var dtypes []core.DType
	for _, o := range manifest.Options {
		d := o.Target.BaseTensorType()
		if d.IsQuant() && !seen[d] {
			seen[d] = true
			dtypes = append(dtypes, d)
		}
	}
	// Dry-run preserves the original dtype-sorted plan order.
	sort.Slice(dtypes, func(i, j int) bool { return dtypes[i] < dtypes[j] })
	if e.DryRun {
		if err := e.planAnchorJobs(ctx, dtypes); err != nil {
			return err
		}
		return e.complete(core.StageQuantize, "")
	}
	if err := e.ensureVariantAnchors(ctx); err != nil {
		return err
	}
	srcs, closeSrcs, err := e.openAnchorSources()
	if err != nil {
		return err
	}
	candidate := filepath.Join(e.workDir(), "candidate.gguf")
	err = tensorbank.NewAssembler().Build(ctx, srcs, manifest, candidate,
		e.progressFunc(core.StageQuantize, "assembling candidate"))
	closeSrcs()
	if err != nil {
		return err
	}
	e.printf("  candidate: %s (%d bytes planned)\n", candidate, manifest.TotalBytes)
	return e.complete(core.StageQuantize, candidate)
}

// ensureVariantAnchors materializes trimmed per-dtype variant files for the
// solved assignment. Search calls this too: emit scratch cleanup can remove
// anchors while quantize stays marked complete, and slop-prefilter skip still
// needs variants to assemble final.gguf. Parsed GGUF tensor coverage, not
// meta.json dtype labels, decides what must be rebuilt.
func (e *Engine) ensureVariantAnchors(ctx context.Context) error {
	manifest := e.Run.Manifest
	if manifest == nil {
		return fmt.Errorf("pipeline: no manifest; quantize stage incomplete")
	}
	seen := map[core.DType]bool{}
	var dtypes []core.DType
	for _, o := range manifest.Options {
		d := o.Target.BaseTensorType()
		if d.IsQuant() && !seen[d] {
			seen[d] = true
			dtypes = append(dtypes, d)
		}
	}
	sort.Slice(dtypes, func(i, j int) bool {
		return estimatedArtifact(e.Run.Bank, dtypes[i]) > estimatedArtifact(e.Run.Bank, dtypes[j])
	})
	keepFor := func(d core.DType) (map[string]struct{}, error) {
		keep := map[string]struct{}{}
		for _, o := range manifest.Options {
			if o.Target.BaseTensorType() == d {
				keep[o.TensorName] = struct{}{}
			}
		}
		return keep, nil
	}
	return e.runTrimmedAnchorJobs(ctx, dtypes, keepFor, e.variantsDir(), "meta.json")
}

// anchorsForDTypes materializes synthetic per-dtype anchors for
// orchestrate.BuildAnchorBatch (dry-run planning only).
func anchorsForDTypes(dtypes []core.DType) []core.Anchor {
	out := make([]core.Anchor, 0, len(dtypes))
	for _, d := range dtypes {
		out = append(out, core.Anchor{
			Kind:     core.AnchorExplicit,
			Pattern:  "*",
			MinDType: d,
			Reason:   "manifest dtype anchor",
		})
	}
	return out
}

// planAnchorJobs prints (without executing) the anchor quantize invocations.
func (e *Engine) planAnchorJobs(ctx context.Context, dtypes []core.DType) error {
	caps, err := e.caps(ctx, orchestrate.ToolLlamaQuantize)
	if err != nil {
		return err
	}
	batch, err := orchestrate.BuildAnchorBatch(e.Run.Config.SourcePath,
		anchorsForDTypes(dtypes), e.anchorDir(), e.Run.Manifest.ProfileID)
	if err != nil {
		return err
	}
	for _, job := range batch.Jobs {
		req := job.Request
		e.fillQuantizeRequest(&req)
		if caps.Has("--dry-run") {
			req.DryRun = true
		}
		iv, err := orchestrate.PlanQuantize(req, caps, e.Run.Config.Tools.LlamaQuantize)
		if err != nil {
			return err
		}
		e.printf("plan: %s %s\n", iv.Path, argvString(iv.Argv))
	}
	return nil
}

func (e *Engine) fillQuantizeRequest(req *orchestrate.QuantizeRequest) {
	cfg := e.Run.Config
	req.Threads = cfg.Threads
	if ip := e.imatrixPath(); ip != "" {
		req.ImatrixPath = ip
	}
	req.SourceQuantized = orchestrate.SourceIsQuantized(e.Run.Bank)
}

// splitThreads divides cfg.Threads across in-flight llama-quantize jobs.
// Zero total is left as zero (omit --threads). Each worker gets at least 1
// when total > 0.
func splitThreads(total, workers int) int {
	if total <= 0 {
		return total
	}
	if workers <= 1 {
		return total
	}
	n := total / workers
	if n < 1 {
		return 1
	}
	return n
}

// anchorJobWorkers returns the worker-pool size for missing dtypes: at most
// two, and one when free disk cannot safely hold two high-precision source
// subsets, two quantized outputs, accumulated variants, and headroom.
func (e *Engine) anchorJobWorkers(jobs []trimmedAnchorJob) int {
	if len(jobs) <= 1 {
		return 1
	}
	n := 2
	var maxAnchor uint64
	if e.Run.Bank != nil {
		for _, j := range jobs {
			if sz := estimatedArtifact(e.Run.Bank, j.d); sz > maxAnchor {
				maxAnchor = sz
			}
		}
	}
	var variants uint64
	if e.Run.Manifest != nil {
		variants = e.Run.Manifest.TotalBytes
	}
	sourceArtifact := estimatedSourceArtifact(e.Run.Bank)
	need := saturatingAdd(sourceArtifact, sourceArtifact, maxAnchor, maxAnchor, variants)
	need = saturatingAdd(need, need/10)
	if free, ok := tensorbank.DiskFree(e.anchorDir()); ok && need > 0 && free < need {
		return 1
	}
	return n
}

type trimmedAnchorJob struct {
	idx  int
	d    core.DType
	keep map[string]struct{}
}

// runTrimmedAnchorJobs quantizes each missing dtype independently: trim the
// source to that dtype's keep-set (IQ and K-quant alike — IQ still needs
// this to avoid GGML_ASSERT on uncovered 2D tensors), --pure quantize the
// subset, verify exact dtypes, and place the result as the variant. Jobs
// share no tmp paths; a mutex serializes meta.json / ladder-meta.json so
// crash-safe resume still records a dtype only after its variant is on
// disk. Peak scratch is up to N in-flight subset-anchors plus accumulated
// variants (N ≤ 2).
//
// keepFor returns the exact set of tensor names to keep for a given dtype.
// Empty keep-set is an error. outDir is variants/ or ladder/; metaName
// tracks completed dtypes.
func (e *Engine) runTrimmedAnchorJobs(ctx context.Context, dtypes []core.DType, keepFor func(core.DType) (map[string]struct{}, error), outDir, metaName string) error {
	return e.runTrimmedAnchorJobsMode(ctx, dtypes, keepFor, outDir, metaName, false)
}

// runSparseTrimmedAnchorJobs records exact per-dtype tensor coverage instead
// of treating one shard as a complete model-wide dtype rung.
func (e *Engine) runSparseTrimmedAnchorJobs(ctx context.Context, dtypes []core.DType,
	keepFor func(core.DType) (map[string]struct{}, error), outDir, metaName string) error {
	return e.runTrimmedAnchorJobsMode(ctx, dtypes, keepFor, outDir, metaName, true)
}

func (e *Engine) runTrimmedAnchorJobsMode(ctx context.Context, dtypes []core.DType,
	keepFor func(core.DType) (map[string]struct{}, error), outDir, metaName string, sparse bool) error {
	caps, err := e.caps(ctx, orchestrate.ToolLlamaQuantize)
	if err != nil {
		return err
	}
	meta := e.loadAnchorMeta(metaName)
	onDisk, err := tensorCoverage(outDir)
	if err != nil {
		return err
	}
	var jobs []trimmedAnchorJob
	for idx, d := range dtypes {
		keep, err := keepFor(d)
		if err != nil {
			return err
		}
		if len(keep) == 0 {
			return fmt.Errorf("pipeline: no tensors to keep for anchor %s", d)
		}
		for name := range keep {
			if _, ok := onDisk[d.BaseTensorType()][name]; ok {
				delete(keep, name)
			}
		}
		if len(keep) == 0 {
			continue
		}
		jobs = append(jobs, trimmedAnchorJob{idx: idx, d: d, keep: keep})
	}
	if len(jobs) == 0 {
		return nil
	}
	if err := os.MkdirAll(e.anchorDir(), 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	nworkers := e.anchorJobWorkers(jobs)
	threads := splitThreads(e.Run.Config.Threads, nworkers)

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		metaMu   sync.Mutex
		logMu    sync.Mutex
		firstErr error
		errOnce  sync.Once
		done     atomic.Int64
		wg       sync.WaitGroup
	)
	fail := func(err error) {
		if err == nil {
			return
		}
		errOnce.Do(func() {
			firstErr = err
			cancel()
		})
	}

	sem := make(chan struct{}, nworkers)
loop:
	for _, job := range jobs {
		select {
		case <-ctx.Done():
			fail(ctx.Err())
			break loop
		case sem <- struct{}{}:
		}
		wg.Add(1)
		go func(job trimmedAnchorJob) {
			defer wg.Done()
			defer func() { <-sem }()
			if err := ctx.Err(); err != nil {
				fail(err)
				return
			}
			if err := e.runOneTrimmedAnchor(ctx, caps, job, outDir, threads); err != nil {
				fail(err)
				return
			}
			metaMu.Lock()
			if !meta.has(job.d) {
				meta.DTypes = append(meta.DTypes, job.d)
				sort.Slice(meta.DTypes, func(i, j int) bool { return meta.DTypes[i] < meta.DTypes[j] })
			}
			saveErr := e.saveAnchorMeta(metaName, meta)
			metaMu.Unlock()
			if saveErr != nil {
				fail(saveErr)
				return
			}
			n := done.Add(1)
			logMu.Lock()
			e.printf("  anchor %s -> %s\n", job.d, fmt.Sprintf("anchor-%03d-%s.gguf", job.idx, job.d.BaseTensorType()))
			e.obsProgress(e.stage, float64(n)/float64(len(jobs)), fmt.Sprintf("anchor %s", job.d))
			logMu.Unlock()
		}(job)
	}
	wg.Wait()
	return firstErr
}

func (e *Engine) runOneTrimmedAnchor(ctx context.Context, caps *orchestrate.Capabilities, job trimmedAnchorJob, outDir string, threads int) error {
	d := job.d
	keep := job.keep
	tmpPath := filepath.Join(e.anchorDir(), fmt.Sprintf("tmp-%s.gguf", d.BaseTensorType()))
	os.Remove(tmpPath)
	subsetPath := filepath.Join(e.anchorDir(), fmt.Sprintf("tmp-subset-%s.gguf", d.BaseTensorType()))
	os.Remove(subsetPath)
	// Always trim to the keep-set before --pure so K-quant jobs cost
	// ~|keep|/|all| of a full-model quant. IQ kernels abort
	// (GGML_ASSERT(imatrix != NULL)) on any 2D tensor without an imatrix
	// row; the subset also preserves that safety.
	if err := (&tensorbank.Assembler{Scratch: true}).Trim(ctx, e.payloadSource(), keep, subsetPath, nil); err != nil {
		os.Remove(subsetPath)
		return fmt.Errorf("trim source for anchor %s: %w", d, err)
	}
	if e.encodeEnabled() && encode.Supported(d) {
		var imatrix map[string]profile.ImatrixStats
		if ip := e.imatrixPath(); ip != "" {
			stats, err := profile.LoadImatrix(ip)
			if err != nil {
				os.Remove(subsetPath)
				return fmt.Errorf("encode imatrix for anchor %s: %w", d, err)
			}
			imatrix = stats
		}
		err := encode.WriteAnchor(subsetPath, tmpPath, encode.Options{
			DType:   d,
			Imatrix: imatrix,
			Viterbi: e.viterbiEnabled(),
			GPTQ:    e.gptqEnabled(),
			Context: ctx,
		})
		os.Remove(subsetPath)
		if err != nil {
			os.Remove(tmpPath)
			return fmt.Errorf("encode anchor %s: %w", d, err)
		}
	} else {
		req := orchestrate.QuantizeRequest{
			ProfileID:  e.Run.Manifest.ProfileID,
			SourcePath: subsetPath,
			OutputPath: tmpPath,
			Type:       d,
			Pure:       true,
		}
		e.fillQuantizeRequest(&req)
		req.Threads = threads
		iv, err := orchestrate.PlanQuantize(req, caps, e.Run.Config.Tools.LlamaQuantize)
		if err != nil {
			os.Remove(subsetPath)
			return err
		}
		if _, err := runOK(ctx, e.Runner, iv); err != nil {
			os.Remove(tmpPath)
			os.Remove(subsetPath)
			return fmt.Errorf("anchor %s: %w", d, err)
		}
		os.Remove(subsetPath)
	}
	if err := e.verifyAnchor(tmpPath, d); err != nil {
		os.Remove(tmpPath)
		return err
	}
	variantPath, err := uniqueAnchorPath(outDir, job.idx, d)
	if err != nil {
		os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, variantPath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("place anchor %s: %w", d, err)
	}
	return nil
}

func uniqueAnchorPath(outDir string, idx int, d core.DType) (string, error) {
	stem := fmt.Sprintf("anchor-%03d-%s", idx, d.BaseTensorType())
	for suffix := 0; ; suffix++ {
		name := stem + ".gguf"
		if suffix > 0 {
			name = fmt.Sprintf("%s-%d.gguf", stem, suffix)
		}
		path := filepath.Join(outDir, name)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return path, nil
		} else if err != nil {
			return "", fmt.Errorf("pipeline: inspect anchor destination %s: %w", path, err)
		}
	}
}

// tensorsProvided derives tensor coverage from parsed GGUF headers. Metadata
// is only an audit log and can lag the files after a crash, so it must never
// be used as the source of truth for sparse coverage.
func tensorsProvided(dir string, d core.DType) (map[string]struct{}, error) {
	coverage, err := tensorCoverage(dir)
	if err != nil {
		return nil, err
	}
	if coverage[d.BaseTensorType()] == nil {
		return map[string]struct{}{}, nil
	}
	return coverage[d.BaseTensorType()], nil
}

func tensorCoverage(dir string) (map[core.DType]map[string]struct{}, error) {
	out := map[core.DType]map[string]struct{}{}
	ents, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return out, nil
	}
	if err != nil {
		return nil, err
	}
	for _, ent := range ents {
		if ent.IsDir() || filepath.Ext(ent.Name()) != ".gguf" {
			continue
		}
		path := filepath.Join(dir, ent.Name())
		src, err := tensorbank.OpenSource(path)
		if err != nil {
			return nil, fmt.Errorf("pipeline: inspect anchor coverage %s: %w", path, err)
		}
		file, parseErr := tensorbank.Parse(src)
		_ = src.Close()
		if parseErr != nil {
			return nil, fmt.Errorf("pipeline: inspect anchor coverage %s: %w", path, parseErr)
		}
		for _, tensor := range file.Tensors {
			dtype := tensor.DType.BaseTensorType()
			if out[dtype] == nil {
				out[dtype] = map[string]struct{}{}
			}
			out[dtype][tensor.Name] = struct{}{}
		}
	}
	return out, nil
}

// verifyAnchor rejects recipe-style fallback before an invalid anchor can be
// recorded and later make tensorbank assembly fail. Float and otherwise
// non-quantizable source tensors are intentionally preserved as-is. The
// shape check mirrors core.TensorDesc.Quantizable (2D matrices and 3D
// expert stacks, block-aligned contiguous dimension).
func (e *Engine) verifyAnchor(path string, dtype core.DType) error {
	s, err := tensorbank.OpenSource(path)
	if err != nil {
		return fmt.Errorf("pipeline: open anchor %s: %w", dtype, err)
	}
	defer s.Close()
	f, err := tensorbank.Parse(s)
	if err != nil {
		return fmt.Errorf("pipeline: parse anchor %s: %w", dtype, err)
	}
	want := dtype.BaseTensorType()
	var mismatches []string
	for _, t := range f.Tensors {
		if len(t.Shape) < 2 || len(t.Shape) > 3 {
			continue
		}
		if t.Shape[0]%256 != 0 && t.Shape[0]%32 != 0 {
			continue
		}
		if _, ok := want.ExactBytes(t.Elements); !ok {
			continue
		}
		if t.DType != want {
			mismatches = append(mismatches, fmt.Sprintf("%s=%s (want %s)", t.Name, t.DType, want))
		}
	}
	if len(mismatches) > 0 {
		sort.Strings(mismatches)
		return fmt.Errorf("pipeline: anchor %s did not preserve exact tensor types; llama-quantize must support --pure: %s", dtype, strings.Join(mismatches, ", "))
	}
	return nil
}

// anchorGGUFs lists trimmed variant and ladder anchor files available for
// assembly, sorted by path. The source GGUF is opened separately by
// openAnchorSources as the primary (index 0) provider for float tensors.
func (e *Engine) anchorGGUFs() ([]string, error) {
	var out []string
	var collect func(string) error
	collect = func(dir string) error {
		ents, err := os.ReadDir(dir)
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		for _, ent := range ents {
			path := filepath.Join(dir, ent.Name())
			if ent.IsDir() {
				if err := collect(path); err != nil {
					return err
				}
				continue
			}
			if filepath.Ext(ent.Name()) == ".gguf" {
				out = append(out, path)
			}
		}
		return nil
	}
	for _, sub := range []string{"variants", "ladder"} {
		dir := filepath.Join(e.anchorDir(), sub)
		if err := collect(dir); err != nil {
			return nil, err
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("pipeline: no anchor files at %s; quantize stage incomplete", e.anchorDir())
	}
	sort.Strings(out)
	return out, nil
}

// openAnchorSources opens the primary source plus every anchor GGUF.
func (e *Engine) openAnchorSources() ([]*tensorbank.Source, func(), error) {
	paths := []string{e.payloadSource()}
	anchors, err := e.anchorGGUFs()
	if err != nil {
		return nil, nil, err
	}
	paths = append(paths, anchors...)
	var srcs []*tensorbank.Source
	for _, p := range paths {
		s, err := tensorbank.OpenSource(p)
		if err != nil {
			for _, opened := range srcs {
				opened.Close()
			}
			return nil, nil, err
		}
		srcs = append(srcs, s)
	}
	closeAll := func() {
		for _, s := range srcs {
			s.Close()
		}
	}
	return srcs, closeAll, nil
}

func argvString(argv []string) string {
	out := ""
	for i, a := range argv {
		if i > 0 {
			out += " "
		}
		out += fmt.Sprintf("%q", a)
	}
	return out
}
