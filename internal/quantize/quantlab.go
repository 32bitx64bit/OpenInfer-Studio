package quantize

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/runtimes"

	"quantlab/core"
	"quantlab/orchestrate"
	"quantlab/pipeline"
	"quantlab/state"
	"quantlab/tensorbank"
)

// quantlabDirName is the per-job subdirectory holding every quantlab
// artifact: pipeline checkpoint, work dir (anchors, candidates), calibration
// corpora staging, and the emitted GGUF before adoption.
const quantlabDirName = "ql"

// adaptivePresetBPW translates the deprecated adaptive_preset names to a
// target bits-per-weight heuristic.
var adaptivePresetBPW = map[string]float64{
	"quality":    6.0,
	"balanced":   4.5,
	"compact":    3.8,
	"aggressive": 3.0,
}

// Quant tier values selectable for OpenInfer Dynamic compression.
const (
	QuantTierQ5     = "q5"
	QuantTierQ4     = "q4"
	QuantTierQ3     = "q3"
	QuantTierQ2     = "q2"
	QuantTierCustom = "custom"
)

// tierBPW maps each named quant tier to its target bits-per-weight.
var tierBPW = map[string]float64{
	QuantTierQ5: 5.5,
	QuantTierQ4: 4.5,
	QuantTierQ3: 3.5,
	QuantTierQ2: 2.5,
}

// TierBPW returns the bits-per-weight target for a named quant tier and true.
// The "custom" tier (and any unknown/empty tier) returns 0, false because it
// targets an explicit byte budget instead of a BPW goal.
func TierBPW(tier string) (float64, bool) {
	bpw, ok := tierBPW[strings.ToLower(strings.TrimSpace(tier))]
	return bpw, ok
}

// resolveQuantTier normalizes the quant tier on the request and applies named
// tiers to the target fields. A named tier overrides target_bpw with the
// tier's BPW and clears target_bytes. The "custom" tier keeps the caller's
// target_bytes. An empty tier leaves explicit target_bpw/target_bytes intact
// for backward compatibility; normalizeAdaptiveTarget supplies the default
// (q4) BPW when nothing is set.
func resolveQuantTier(req *Request) {
	tier := strings.ToLower(strings.TrimSpace(req.QuantTier))
	if tier == "" {
		return
	}
	req.QuantTier = tier
	if bpw, ok := TierBPW(tier); ok {
		req.TargetBPW = bpw
		req.TargetBytes = 0
	}
	// "custom" and any unknown tier: leave target_bytes as the caller set it.
}

// defaultAdaptiveTargetBPW is the bits-per-weight goal when a Dynamic job
// specifies no tier and no explicit target. It matches QuantTierQ4.
const defaultAdaptiveTargetBPW = 4.5

// normalizeAdaptiveTarget persists the effective target so queued, From-HF,
// and restart-resumed runs all solve the same request.
func normalizeAdaptiveTarget(req *Request) ([]string, error) {
	resolveQuantTier(req)
	if req.TargetBPW > 0 || req.TargetBytes > 0 {
		return nil, nil
	}
	preset := strings.ToLower(strings.TrimSpace(req.AdaptivePreset))
	if preset == "" {
		req.TargetBPW = defaultAdaptiveTargetBPW
		return nil, nil
	}
	bpw, ok := adaptivePresetBPW[preset]
	if !ok {
		return nil, fmt.Errorf("unknown adaptive_preset %q (want quality, balanced, compact, or aggressive)", req.AdaptivePreset)
	}
	req.TargetBPW = bpw
	return []string{fmt.Sprintf("adaptive_preset is deprecated and will be removed; preset %q was translated to target_bpw %.1f; use target_bpw or target_bytes with effort instead", preset, bpw)}, nil
}

func (m *Manager) quantlabScratchEstimate(src *models.Model, req Request) (scratch, output uint64, err error) {
	resolveQuantTier(&req)
	if src == nil {
		return 0, 0, fmt.Errorf("quantlab source is missing")
	}
	source, err := tensorbank.OpenSource(src.PrimaryPath)
	if err != nil {
		return 0, 0, err
	}
	defer source.Close()
	bank, err := tensorbank.NewAssembler().Assemble(source, src.PrimaryPath, src.ID)
	if err != nil {
		return 0, 0, err
	}
	var repair uint64
	if status, err := gguf.InspectSSMConv1d(src.PrimaryPath); err == nil && status.Required && status.Repairable {
		// The repaired copy coexists with anchors. Reserve a margin for anchor
		// geometry changing from the legacy storage type to required F32.
		repair = uint64(status.OutputBytes) + uint64(status.OutputBytes)/10
	}
	var vocab uint64
	if md, err := gguf.ParseFile(src.PrimaryPath); err == nil && md != nil {
		vocab = uint64(md.VocabSize)
	}
	var reusedIMatrix uint64
	generateIMatrix := req.GenerateIMatrix
	effort := resolveAdaptiveEffort(req)
	reqEst := req
	applyEffortCalibration(&reqEst, effort)
	if !generateIMatrix && reqEst.IMatrixID != "" {
		if im, err := m.getIMatrix(reqEst.IMatrixID); err == nil {
			if size := fileSize(im.Path); size > 0 {
				reusedIMatrix = uint64(size)
			}
		}
	}
	if generateIMatrix && reqEst.IMatrixID == "" {
		if im := m.findReusableIMatrix(src, reqEst); im != nil {
			generateIMatrix = false
			if size := fileSize(im.Path); size > 0 {
				reusedIMatrix = uint64(size)
			}
		}
	}
	profile, _ := pipeline.EffortFor(pipeline.Effort(effort))
	budget := uint64(req.TargetBytes)
	if budget == 0 {
		var elems uint64
		for _, tensor := range bank.Tensors {
			elems += tensor.Elements
		}
		budget = uint64(req.TargetBPW * float64(elems) / 8)
	}
	ctxSize := 512
	if profile.EvalCtx > 0 {
		ctxSize = profile.EvalCtx
	}
	// Domain validation is sequential and overwrites one shared logits file.
	// Evaluation retains the main baseline plus that shared domain file.
	scratch = pipeline.EstimateScratch(bank, pipeline.Effort(effort), budget, pipeline.EvalScratchConfig{
		CtxSize: ctxSize, Chunks: profile.EvalChunks, VocabSize: vocab, SourceRepairBytes: repair,
		GenerateIMatrix: generateIMatrix, ReusedImatrixBytes: reusedIMatrix,
		LogitsFiles: 2,
	})
	return scratch, budget, nil
}

func (m *Manager) checkQuantlabDisk(src *models.Model, req Request) error {
	if m.diskFree == nil {
		return nil
	}
	need, outputBudget, err := m.quantlabScratchEstimate(src, req)
	if err != nil {
		return fmt.Errorf("estimate Dynamic scratch disk: %w", err)
	}
	free := m.diskFree(m.layout.QuantJobs)
	// Publication copies instead of moving the checkpointed emitted artifact,
	// so Models needs a complete second artifact until result persistence. The
	// scratch estimate above already reserves the checkpoint artifact.
	finalNeed := outputBudget + outputBudget/10
	modelFree := m.diskFree(m.layout.Models)
	if same, known := sameFilesystem(m.layout.QuantJobs, m.layout.Models); !known || same {
		available := free
		if available == 0 || modelFree > 0 && modelFree < available {
			available = modelFree
		}
		combined := saturatingDiskAdd(need, finalNeed)
		if available > 0 && combined > available {
			return fmt.Errorf("not enough free disk for OpenInfer Dynamic scratch and publication: %.1f GiB required on the shared or conservatively-assumed filesystem; %.1f GiB available", float64(combined)/(1<<30), float64(available)/(1<<30))
		}
		return nil
	}
	if free > 0 && need > free {
		return fmt.Errorf("not enough free disk for OpenInfer Dynamic scratch: %.1f GiB required on %s; %.1f GiB available", float64(need)/(1<<30), m.layout.QuantJobs, float64(free)/(1<<30))
	}
	if modelFree > 0 && finalNeed > modelFree {
		return fmt.Errorf("not enough free disk for OpenInfer Dynamic output: %.1f GiB required on %s; %.1f GiB available", float64(finalNeed)/(1<<30), m.layout.Models, float64(modelFree)/(1<<30))
	}
	return nil
}

func saturatingDiskAdd(a, b uint64) uint64 {
	if ^uint64(0)-a < b {
		return ^uint64(0)
	}
	return a + b
}

func quantlabCSKWorkingSet(available uint64) uint64 {
	if available == 0 {
		return 1 << 30
	}
	limit := available / 4
	if limit > 8<<30 {
		limit = 8 << 30
	}
	return limit
}

// resolveAdaptiveEffort applies the Effort/AdaptiveMode alias rules: Effort
// wins, an empty Effort falls back to AdaptiveMode, and adaptive jobs default
// to profiled. Callers validate the value beforehand.
func resolveAdaptiveEffort(req Request) string {
	e := strings.ToLower(strings.TrimSpace(req.Effort))
	if e == "" {
		e = strings.ToLower(strings.TrimSpace(req.AdaptiveMode))
	}
	if e == "" {
		e = string(state.EffortProfiled)
	}
	return e
}

func effortCalibrationPreset(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case string(state.EffortFast):
		return "quick"
	case string(state.EffortDeep):
		return "research"
	default:
		return "thorough"
	}
}

func applyEffortCalibration(req *Request, effort string) {
	if req == nil {
		return
	}
	if strings.TrimSpace(req.CalibrationPreset) == "" && req.Chunks == 0 {
		req.CalibrationPreset = effortCalibrationPreset(effort)
	}
}

func usesQuantlab(req Request) bool {
	return normalizeKind(req.Kind) == KindAdaptiveQuantize || (normalizeKind(req.Kind) == KindFromHF &&
		(strings.TrimSpace(req.AdaptivePreset) != "" || strings.TrimSpace(req.Effort) != "" ||
			strings.TrimSpace(req.AdaptiveMode) != "" || strings.TrimSpace(req.QuantTier) != "" ||
			req.TargetBPW > 0 || req.TargetBytes > 0))
}

// quantlabStageRange maps a quantlab pipeline stage onto the job-facing stage
// name and its overall-progress range.
type quantlabStageRange struct {
	name       string
	start, end float64
}

var quantlabStageRanges = map[core.Stage]quantlabStageRange{
	core.StageAssemble: {"analyze", 0.05, 0.10},
	core.StageAnchor:   {"anchor", 0.10, 0.15},
	core.StageSolve:    {"solve", 0.15, 0.20},
	core.StageQuantize: {"quantize", 0.20, 0.65},
	core.StageEvaluate: {"validate", 0.65, 0.92},
	core.StageSearch:   {"search", 0.92, 0.93},
	core.StageEmit:     {"finalize", 0.93, 1.0},
}

// quantlabObserver adapts pipeline.Observer to job progress, stage events,
// and the job log. It tracks the active stage so measurement callbacks can
// surface per-candidate feedback (which quantlab otherwise only logs), and
// remaps the quantize stage's anchor/assembly sub-phases so neither is hidden
// by the monotonic-progress clamp.
type quantlabObserver struct {
	m     *Manager
	jobID string

	mu           sync.Mutex
	log          *os.File
	curStage     core.Stage
	curStageName string
	stepCount    int
	measurements int
	evalBudget   int
	stageStarted time.Time
	lastProgress float64
	lastCurrent  int
	lastTotal    int
	lastMessage  string
}

var (
	_ pipeline.Observer   = (*quantlabObserver)(nil)
	_ pipeline.StageTimer = (*quantlabObserver)(nil)
)

func (o *quantlabObserver) StageStarted(stage core.Stage) {
	r, ok := quantlabStageRanges[stage]
	if !ok {
		return
	}
	o.mu.Lock()
	o.curStage = stage
	o.curStageName = r.name
	o.stepCount = 0
	o.measurements = 0
	o.stageStarted = time.Now()
	o.lastProgress = 0
	o.lastCurrent = 0
	o.lastTotal = 0
	o.lastMessage = "starting " + r.name
	o.mu.Unlock()
	_ = o.m.setStage(o.jobID, r.name, r.start)
}

func (o *quantlabObserver) StageProgress(stage core.Stage, progress float64, message string) {
	r, ok := quantlabStageRanges[stage]
	if !ok {
		return
	}
	frac := progress
	current, total := 0, 0
	switch stage {
	case core.StageQuantize:
		frac = remapQuantizeProgress(message, progress)
		if strings.HasPrefix(message, "anchor") {
			o.mu.Lock()
			o.stepCount++
			current, total = stepCounter(o.stepCount, progress)
			o.mu.Unlock()
		}
	case core.StageSearch:
		o.mu.Lock()
		meas := o.measurements
		budget := o.evalBudget
		o.mu.Unlock()
		if budget <= 0 {
			budget = 7
		}
		frac = measurementStageFractionBudget(meas, budget, progress)
	}
	o.mu.Lock()
	o.lastProgress = frac
	o.lastCurrent = current
	o.lastTotal = total
	o.lastMessage = message
	o.mu.Unlock()
	o.m.emitQuantlabProgress(o.jobID, r.name, frac, current, total, message)
}

func (o *quantlabObserver) StageCompleted(stage core.Stage, artifact string) {
	o.logf("stage %s complete: %s", stage, artifact)
}

func (o *quantlabObserver) StageDuration(stage core.Stage, d time.Duration) {
	o.logf("stage %s wall-clock %s", stage, d.Round(time.Millisecond))
}

func (o *quantlabObserver) Measurement(mm core.Measurement) {
	o.mu.Lock()
	o.measurements++
	count := o.measurements
	budget := o.evalBudget
	stage := o.curStage
	name := o.curStageName
	o.mu.Unlock()
	o.logf("measurement %s %s = %.6f (baseline %.6f, delta %.6f)",
		mm.ProfileID, mm.Metric, mm.Value, mm.Baseline, mm.Delta)
	if name == "" {
		return
	}
	msg := fmt.Sprintf("measurement %d: %s %.6f", count, mm.Metric, mm.Value)
	if stage == core.StageEvaluate || stage == core.StageSearch {
		if budget <= 0 {
			budget = 7
		}
		frac := measurementStageFractionBudget(count, budget, 0)
		o.mu.Lock()
		o.lastProgress = frac
		o.lastCurrent = count
		o.lastTotal = 0
		o.lastMessage = msg
		o.mu.Unlock()
		o.m.emitQuantlabProgress(o.jobID, name, frac, count, 0, msg)
	}
}

func (o *quantlabObserver) Heartbeat() {
	o.mu.Lock()
	name := o.curStageName
	frac := o.lastProgress
	current, total := o.lastCurrent, o.lastTotal
	message := o.lastMessage
	started := o.stageStarted
	o.mu.Unlock()
	if name == "" {
		return
	}
	if message == "" {
		message = "working"
	}
	if !started.IsZero() {
		message += fmt.Sprintf(" · active %s", time.Since(started).Round(time.Second))
	}
	if o.m.diskFree != nil {
		if free := o.m.diskFree(o.m.layout.QuantJobs); free > 0 {
			message += fmt.Sprintf(" · disk free %.1f GiB", float64(free)/(1<<30))
		}
	}
	if o.m.hardware != nil {
		if hw := o.m.hardware(); hw != nil && hw.RAMAvailable > 0 {
			message += fmt.Sprintf(" · RAM free %.1f GiB", float64(hw.RAMAvailable)/(1<<30))
		}
	}
	o.m.emitQuantlabProgress(o.jobID, name, frac, current, total, message)
	o.logf("heartbeat: %s", message)
}

func (o *quantlabObserver) ToolOutput(ev orchestrate.OutputEvent) {
	line := strings.TrimSpace(ev.Line)
	if line == "" {
		return
	}
	if len(line) > 512 {
		line = line[:512] + "…"
	}
	o.logf("%s %s: %s", ev.Tool, ev.Stream, line)
	o.mu.Lock()
	name := o.curStageName
	frac := o.lastProgress
	current, total := o.lastCurrent, o.lastTotal
	if c, t, ok := ParseProgressLine(line); ok {
		current, total = c, t
	}
	message := fmt.Sprintf("%s: %s", ev.Tool, line)
	o.lastCurrent, o.lastTotal, o.lastMessage = current, total, message
	o.mu.Unlock()
	if name != "" {
		o.m.emitQuantlabProgress(o.jobID, name, frac, current, total, message)
	}
}

func (o *quantlabObserver) logf(format string, args ...any) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.log != nil {
		fmt.Fprintf(o.log, format+"\n", args...)
	}
}

// emitQuantlabProgress scales a stage-local fraction into the stage's overall
// range and persists it, mirroring emitProgressSample with an explicit range.
// The range is resolved through stageRange (job-aware, so from_hf Dynamic jobs
// scale correctly) and stage_progress/overall are clamped to never decrease.
// current/total carry an optional step counter (e.g. anchor N of M).
func (m *Manager) emitQuantlabProgress(id, stage string, frac float64, current, total int, msg string) {
	j, err := m.Get(id)
	if err != nil {
		return
	}
	frac = clampProgress(frac)
	if frac < j.StageProgress && j.Stage == stage {
		frac = j.StageProgress
	}
	start, end := stageRange(j, stage)
	if start < 0 || end <= start {
		start, end = j.Progress, 1
	}
	overall := start + (end-start)*frac
	if overall < j.Progress {
		overall = j.Progress
	}
	_, _ = m.db.Exec(`UPDATE quant_jobs SET progress=?,stage_progress=?,progress_current=?,progress_total=?,progress_message=?,updated_at=? WHERE id=?`,
		overall, frac, current, total, msg, now(), id)
	if m.events != nil {
		m.events.Publish("quant.progress", Progress{
			ID: id, State: j.State, Stage: stage, Current: current, Total: total,
			Progress: overall, StageProgress: frac, Message: msg,
		})
	}
}

// quantlabDirs returns the quantlab working directories for a job.
func (m *Manager) quantlabDirs(jobID string) (root, outDir, workDir, stateDir, calibDir string) {
	root = filepath.Join(m.layout.QuantJobs, jobID, quantlabDirName)
	return root,
		filepath.Join(root, "out"),
		filepath.Join(root, "work"),
		filepath.Join(root, "state"),
		filepath.Join(root, "calib")
}

func (m *Manager) modelLossCachePath(modelID string) string {
	if m == nil || m.layout == nil || strings.TrimSpace(modelID) == "" {
		return ""
	}
	return filepath.Join(m.layout.QuantIMatrices, safeName(modelID)+".loss-cache.json")
}

// seedQuantlabLossCache copies a prior job's measured-loss cache into this
// run's work dir (and store sidecar) so Solve can load it. Missing source or
// an already-present work-dir file is a no-op.
func (m *Manager) seedQuantlabLossCache(modelID, workDir, stateDir, runID string) {
	src := m.modelLossCachePath(modelID)
	if fileSize(src) <= 0 {
		return
	}
	if work := filepath.Join(workDir, "loss-cache.json"); fileSize(work) <= 0 {
		_ = copyFile(src, work)
	}
	if runID != "" {
		if side := filepath.Join(stateDir, runID+".loss-cache.json"); fileSize(side) <= 0 {
			_ = copyFile(src, side)
		}
	}
}

// adoptQuantlabLossCache publishes the run sidecar to the stable per-source
// path so the next Dynamic job on the same model can seed from it. Errors
// are logged, never fatal: a missing cache only falls back to the heuristic.
func (m *Manager) adoptQuantlabLossCache(src *models.Model, stateDir, runID string) {
	if src == nil || runID == "" {
		return
	}
	dest := m.modelLossCachePath(src.ID)
	if dest == "" {
		return
	}
	side := filepath.Join(stateDir, runID+".loss-cache.json")
	if fileSize(side) <= 0 {
		return
	}
	if err := copyFileAtomic(side, dest); err != nil && m.log != nil {
		m.log.Warn("adopt Dynamic loss cache", "err", err)
	}
}

// stageQuantlabCalibration materializes the quantlab calibration directory:
// the job's prepared calibration text as calibration.txt and the held-out
// partition as evaluation.txt, consumed verbatim (manifest.json present).
func (m *Manager) stageQuantlabCalibration(j *Job, calibDir string) error {
	jobCal := filepath.Join(m.layout.QuantJobs, j.ID, "calibration")
	heldOut := filepath.Join(jobCal, "validation.txt")
	if st, err := os.Stat(heldOut); err != nil || st.IsDir() || st.Size() == 0 {
		return fmt.Errorf("Dynamic validation corpus is unavailable")
	}
	cal := filepath.Join(jobCal, "all.txt")
	if st, err := os.Stat(cal); err != nil || st.IsDir() || st.Size() == 0 {
		// Split default calibration lands in per-domain files; concatenate
		// them into all.txt in deterministic domain order.
		var parts []string
		for _, name := range []string{"prose.txt", "facts.txt", "code.txt"} {
			p := filepath.Join(jobCal, name)
			if st, err := os.Stat(p); err == nil && !st.IsDir() && st.Size() > 0 {
				parts = append(parts, p)
			}
		}
		if len(parts) == 0 {
			return fmt.Errorf("Dynamic calibration corpus is unavailable")
		}
		var b strings.Builder
		for i, p := range parts {
			raw, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			if i > 0 {
				b.WriteString("\n\n")
			}
			b.Write(raw)
		}
		if err := os.WriteFile(cal, []byte(b.String()), 0o644); err != nil {
			return err
		}
	}
	if err := os.MkdirAll(calibDir, 0o755); err != nil {
		return err
	}
	if err := copyFile(cal, filepath.Join(calibDir, "calibration.txt")); err != nil {
		return err
	}
	outputs := map[string]string{
		"calibration.txt": "calibration",
		"evaluation.txt":  "validation",
	}
	if err := copyFile(heldOut, filepath.Join(calibDir, "evaluation.txt")); err != nil {
		return err
	}
	search := filepath.Join(jobCal, "search.txt")
	if st, err := os.Stat(search); err == nil && !st.IsDir() && st.Size() > 0 {
		if err := copyFile(search, filepath.Join(calibDir, "search.txt")); err != nil {
			return err
		}
		outputs["search.txt"] = "search"
	}
	domainFiles, err := filepath.Glob(filepath.Join(jobCal, "validation-*.txt"))
	if err != nil {
		return err
	}
	for _, src := range domainFiles {
		name := "evaluation-" + strings.TrimPrefix(filepath.Base(src), "validation-")
		if err := copyFile(src, filepath.Join(calibDir, name)); err != nil {
			return err
		}
		outputs[name] = "validation-domain"
	}
	raw, err := json.Marshal(struct {
		Version int               `json:"version"`
		Outputs map[string]string `json:"outputs"`
	}{Version: 1, Outputs: outputs})
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(calibDir, "manifest.json"), append(raw, '\n'), 0o644)
}

// quantlabLabel derives the OID ftype label for the custom tier (or any
// request without a named tier) by mapping the solved manifest's achieved
// bits-per-weight to the nearest tier. It replaces the earlier dominant-dtype
// derivation so every OID output is named after its compression tier.
func quantlabLabel(manifest *core.SelectionManifest) string {
	if manifest == nil {
		return "OID-Q4_K_XL"
	}
	return "OID-" + tierPrefixFromBPW(manifestAchievedBPW(manifest)) + "_K_XL"
}

// quantlabTierLabel derives the OID ftype label from the selected quant tier.
// Named tiers map directly to OID-Q{N}_K_XL; the custom tier (and an empty
// tier) maps the manifest's achieved bits-per-weight to the nearest tier
// (>=5 -> Q5, >=4 -> Q4, >=3 -> Q3, else Q2).
func quantlabTierLabel(tier string, manifest *core.SelectionManifest) string {
	switch strings.ToLower(strings.TrimSpace(tier)) {
	case QuantTierQ5:
		return "OID-Q5_K_XL"
	case QuantTierQ4:
		return "OID-Q4_K_XL"
	case QuantTierQ3:
		return "OID-Q3_K_XL"
	case QuantTierQ2:
		return "OID-Q2_K_XL"
	default:
		return quantlabLabel(manifest)
	}
}

// manifestAchievedBPW returns the achieved average bits-per-weight implied by
// the solved manifest. It is the byte-weighted harmonic mean of each option's
// dtype BPW (bytes*8/elements), which equals total_bits/total_elements
// without needing the tensor bank.
func manifestAchievedBPW(manifest *core.SelectionManifest) float64 {
	if manifest == nil {
		return 0
	}
	var bytesSum, weighted float64
	for _, o := range manifest.Options {
		if o.Bytes == 0 {
			continue
		}
		bpw, ok := o.Target.BitsPerWeight()
		if !ok || bpw <= 0 {
			continue
		}
		bytesSum += float64(o.Bytes)
		weighted += float64(o.Bytes) / bpw
	}
	if bytesSum == 0 || weighted == 0 {
		return 0
	}
	return bytesSum / weighted
}

// tierPrefixFromBPW maps a bits-per-weight figure to the nearest quant tier
// prefix used in OID labels.
func tierPrefixFromBPW(bpw float64) string {
	switch {
	case bpw >= 5:
		return "Q5"
	case bpw >= 4:
		return "Q4"
	case bpw >= 3:
		return "Q3"
	default:
		return "Q2"
	}
}

// quantlabGateEvidence renders gate outcomes for the job result from the
// quantlab run report.
func quantlabGateEvidence(report pipeline.RunReport) (gates []map[string]any, causes []string) {
	for _, g := range report.Gates {
		entry := map[string]any{
			"metric":   string(g.Metric),
			"maxDelta": g.MaxDelta,
			"measured": g.Measured,
			"pass":     g.Pass,
		}
		if g.MaxAbsolute > 0 {
			entry["maxAbsolute"] = g.MaxAbsolute
		}
		if g.Measured {
			entry["value"] = g.Value
			entry["delta"] = g.Delta
		}
		gates = append(gates, entry)
		if !g.Pass {
			if g.Measured {
				causes = append(causes, fmt.Sprintf("%s %.6f exceeds gate (maxDelta %.4f, maxAbsolute %.4f)",
					g.Metric, g.Value, g.MaxDelta, g.MaxAbsolute))
			} else {
				causes = append(causes, fmt.Sprintf("%s was not measured", g.Metric))
			}
		}
	}
	return gates, causes
}

// runQuantlabAdaptive executes an adaptive (OpenInfer Dynamic) quantization
// through the quantlab pipeline: plan a crash-resumable run, drive the
// engine with an observer, record quality-gate outcomes on the result,
// and adopt the published GGUF. A failing quality check is reported
// (gates_pass=false) as a warning; the model is still added to the library.
func (m *Manager) runQuantlabAdaptive(ctx context.Context, j *Job, req Request, src *models.Model, rt *runtimes.Runtime, tools runtimes.ToolsSnapshot) error {
	qlRoot, outDir, workDir, stateDir, calibDir := m.quantlabDirs(j.ID)
	store := state.Store{Dir: stateDir}
	run, loadErr := store.Load(j.ID)
	resuming := loadErr == nil && run != nil
	workingSrc := src
	repairedSource := false
	cleanupSource := func() {}
	var err error
	if resuming {
		repairedSource = src != nil && run.Config.SourcePath != src.PrimaryPath
		if repairedSource {
			cleanupSource = func() { _ = os.Remove(run.Config.SourcePath) }
		}
	} else {
		workingSrc, repairedSource, cleanupSource, err = m.prepareQuantlabSource(ctx, j, src)
		if err != nil {
			return err
		}
	}
	defer cleanupSource()
	if src != nil && !models.IsSpeculativeDraft(*src) && req.IMatrixID == "" && tools.IMatrix.Present {
		req.GenerateIMatrix = true
	}
	req.ParseSpecial = true
	req.ProcessOutput = true
	effort := resolveAdaptiveEffort(req)
	applyEffortCalibration(&req, effort)
	imatrixPath := ""
	if !resuming {
		imatrixPath, err = m.resolveIMatrixPath(ctx, j, req, workingSrc, tools)
		if err != nil {
			return err
		}
		if err := m.ensureQuantlabCorpus(j, req, workingSrc); err != nil {
			return err
		}
	}
	j.Request = req

	warnings, err := normalizeAdaptiveTarget(&req)
	if err != nil {
		return err
	}
	if len(warnings) == 0 && strings.TrimSpace(req.QuantTier) == "" {
		if preset := strings.ToLower(strings.TrimSpace(req.AdaptivePreset)); preset != "" {
			if bpw, ok := adaptivePresetBPW[preset]; ok && req.TargetBytes == 0 && req.TargetBPW == bpw {
				warnings = append(warnings, fmt.Sprintf("adaptive_preset is deprecated and will be removed; preset %q was translated to target_bpw %.1f; use target_bpw or target_bytes with effort instead", preset, bpw))
			}
		}
	}
	targetBPW := req.TargetBPW
	targetBytes := req.TargetBytes
	_ = m.persistRequest(j.ID, req)
	j.Request = req

	if !resuming {
		if err := m.stageQuantlabCalibration(j, calibDir); err != nil {
			return err
		}
	}

	logFile, err := os.OpenFile(j.LogPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer logFile.Close()
	effortProfile, _ := pipeline.EffortFor(pipeline.Effort(effort))
	obs := &quantlabObserver{
		m: m, jobID: j.ID, log: logFile,
		evalBudget: effortProfile.EvalChunks,
	}

	budget := uint64(0)
	if targetBytes > 0 {
		budget = uint64(targetBytes)
	}
	var cskWorkingSet uint64
	if m.hardware != nil {
		if hw := m.hardware(); hw != nil {
			cskWorkingSet = quantlabCSKWorkingSet(hw.RAMAvailable)
		}
	}
	if cskWorkingSet == 0 {
		cskWorkingSet = quantlabCSKWorkingSet(0)
	}
	if !resuming {
		run, err = pipeline.Plan(pipeline.PlanOptions{
			SourcePath:            workingSrc.PrimaryPath,
			OutputDir:             outDir,
			WorkDir:               workDir,
			StateDir:              stateDir,
			BudgetBytes:           budget,
			TargetBPW:             targetBPW,
			LlamaQuantize:         tools.Quantize.Path,
			LlamaPerplexity:       tools.Perplexity.Path,
			LlamaImatrix:          tools.IMatrix.Path,
			ImatrixFile:           imatrixPath,
			CalibrationDir:        calibDir,
			Threads:               req.Threads,
			CtxSize:               0,
			Effort:                effort,
			CSKMaxWorkingSetBytes: cskWorkingSet,
			RunID:                 j.ID,
			Stdout:                logFile,
		})
		if err != nil {
			return err
		}
	}
	if workingSrc != nil {
		m.seedQuantlabLossCache(workingSrc.ID, workDir, stateDir, j.ID)
	}

	env := append(os.Environ(), runtimes.LibPathEnv(tools.Quantize.Path)...)
	runner := orchestrate.OSRunner{
		Env: env, IdleTimeout: time.Hour, OnOutput: obs.ToolOutput,
	}
	eng, err := pipeline.NewEngine(store, run, runner, logFile)
	if err != nil {
		return err
	}
	eng.Obs = pipeline.WrapTimer(obs)
	stopHeartbeat := make(chan struct{})
	heartbeatDone := make(chan struct{})
	go func() {
		defer close(heartbeatDone)
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				obs.Heartbeat()
			case <-stopHeartbeat:
				return
			case <-ctx.Done():
				return
			}
		}
	}()
	resumeErr := eng.Resume(ctx)
	close(stopHeartbeat)
	<-heartbeatDone
	if resumeErr != nil {
		m.adoptQuantlabLossCache(workingSrc, stateDir, j.ID)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		_ = m.setResult(j.ID, m.quantlabFailureEvidence(j.ID, stateDir, effort, warnings, repairedSource, resumeErr))
		return fmt.Errorf("Dynamic quantization failed: %w", resumeErr)
	}
	m.adoptQuantlabLossCache(workingSrc, stateDir, j.ID)
	if err := ctx.Err(); err != nil {
		return err
	}

	emitted := run.Artifacts[core.StageEmit]
	if emitted == "" {
		return fmt.Errorf("quantlab run completed without an emitted artifact")
	}
	base := strings.TrimSuffix(filepath.Base(run.Config.SourcePath), filepath.Ext(run.Config.SourcePath))
	reportPath := filepath.Join(outDir, base+"-"+j.ID+".report.json")
	var report pipeline.RunReport
	if data, rerr := os.ReadFile(reportPath); rerr == nil {
		if uerr := json.Unmarshal(data, &report); uerr != nil {
			return fmt.Errorf("decode quantlab report: %w", uerr)
		}
	} else {
		return fmt.Errorf("read quantlab report: %w", rerr)
	}

	label := quantlabTierLabel(req.QuantTier, run.Manifest)
	gates, gateCauses := quantlabGateEvidence(report)
	res := map[string]any{
		"adaptive":        label,
		"oid":             label,
		"display":         "OpenInfer Dynamic " + strings.TrimPrefix(label, "OID-"),
		"effort":          effort,
		"quant_tier":      req.QuantTier,
		"run_id":          j.ID,
		"target_bpw":      targetBPW,
		"budget_bytes":    run.Config.BudgetBytes,
		"estimated_bytes": report.Output.Bytes,
		"gates":           gates,
		"gates_pass":      report.GatesPass,
		"measurements":    report.Measurements,
		"profile_id":      report.ProfileID,
		"source_repaired": repairedSource,
		"artifact_path":   emitted,
	}
	if len(warnings) > 0 {
		res["warnings"] = warnings
	}
	if !report.GatesPass {
		res["quality_gate"] = map[string]any{"passed": false, "rejection_cause": gateCauses}
		warnings = append(warnings, "Quality check exceeded a limit: "+strings.Join(gateCauses, "; "))
		res["warnings"] = warnings
	} else {
		res["quality_gate"] = map[string]any{"passed": true, "rejection_cause": []string{}}
	}
	if err := validateQuantlabOutput(emitted); err != nil {
		return fmt.Errorf("validate emitted Dynamic artifact: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	dest := j.DestPath
	if dest == "" {
		dest, err = m.destFor(workingSrc, label, req.OutputName)
		if err != nil {
			return err
		}
		_, _ = m.db.Exec(`UPDATE quant_jobs SET dest_path=?, updated_at=? WHERE id=?`, dest, now(), j.ID)
	}
	if err := publishQuantlabArtifact(emitted, dest); err != nil {
		return fmt.Errorf("publish validated Dynamic candidate: %w", err)
	}
	if err := ctx.Err(); err != nil {
		discardValidatedCandidate(dest, dest)
		return err
	}
	if err := validateQuantlabOutput(dest); err != nil {
		discardValidatedCandidate(dest, dest)
		return fmt.Errorf("validate published Dynamic artifact: %w", err)
	}
	destBase := strings.TrimSuffix(dest, filepath.Ext(dest))
	emittedBase := strings.TrimSuffix(emitted, filepath.Ext(emitted))
	if err := ctx.Err(); err != nil {
		discardValidatedCandidate(dest, dest)
		return err
	}
	if err := copyFileAtomic(emittedBase+".oid-plan.json", destBase+".oid-plan.json"); err != nil {
		discardValidatedCandidate(dest, dest)
		return fmt.Errorf("adopt quantlab recipe sidecar: %w", err)
	}
	res["recipe_path"] = destBase + ".oid-plan.json"
	if err := rewriteQuantlabReport(reportPath, destBase+".quantlab-report.json", dest); err != nil {
		discardValidatedCandidate(dest, dest)
		return fmt.Errorf("adopt quantlab final report: %w", err)
	}
	res["report_path"] = destBase + ".quantlab-report.json"
	if _, err := os.Stat(emittedBase + ".tensor-types.txt"); err == nil {
		if err := copyFileAtomic(emittedBase+".tensor-types.txt", destBase+".tensor-types.txt"); err != nil {
			discardValidatedCandidate(dest, dest)
			return fmt.Errorf("adopt quantlab tensor types: %w", err)
		}
	}
	if err := ctx.Err(); err != nil {
		discardValidatedCandidate(dest, dest)
		return err
	}

	extras, err := m.afterQuantize(ctx, j, req, workingSrc, dest, tools, rt)
	if err != nil {
		discardValidatedCandidate(dest, dest)
		return err
	}
	if err := ctx.Err(); err != nil {
		discardValidatedCandidate(dest, dest)
		return err
	}
	if err := m.scanAndResult(j.ID, dest, req, workingSrc, extras, label); err != nil {
		discardValidatedCandidate(dest, dest)
		return err
	}
	if err := ctx.Err(); err != nil {
		discardValidatedCandidate(dest, dest)
		_, _ = m.lib.Scan()
		return err
	}
	cur, _ := m.Get(j.ID)
	if cur != nil && len(cur.Result) > 0 {
		var prev map[string]any
		_ = json.Unmarshal(cur.Result, &prev)
		for k, v := range prev {
			if _, ok := res[k]; !ok {
				res[k] = v
			}
		}
		res["ftype"] = label
	}
	if err := ctx.Err(); err != nil {
		discardValidatedCandidate(dest, dest)
		return err
	}
	if err := m.setResult(j.ID, res); err != nil {
		return err
	}
	// Keep ql/out intact through model adoption and the durable final result.
	// This is the resume checkpoint if the process dies during publication.
	_ = os.RemoveAll(qlRoot)
	// A successful run settles its own decisions into the durable library
	// artifacts; the shared per-model measured-loss cache is consumed input,
	// not an output. Leaving it in place would replay this run's profile into
	// every later Dynamic quantization of the same model even after code or
	// estimator improvements.
	if c := m.modelLossCachePath(workingSrc.ID); c != "" {
		if err := os.Remove(c); err != nil && !os.IsNotExist(err) && m.log != nil {
			m.log.Warn("clear Dynamic loss cache", "err", err)
		}
	}
	return nil
}

// publishQuantlabArtifact copies the pipeline checkpoint artifact into the
// library. A pre-existing destination is only reusable when it is byte-for-
// byte identical; otherwise it is the job-owned untrusted partial and is
// removed without touching any unrelated model.
func publishQuantlabArtifact(source, dest string) error {
	sourceSHA, sourceBytes, err := hashFile(source)
	if err != nil {
		return err
	}
	if _, err := os.Stat(dest); err == nil {
		destSHA, destBytes, hashErr := hashFile(dest)
		if hashErr == nil && destBytes == sourceBytes && destSHA == sourceSHA {
			return nil
		}
		if removeErr := os.Remove(dest); removeErr != nil && !os.IsNotExist(removeErr) {
			return fmt.Errorf("remove mismatched job destination: %w", removeErr)
		}
		return fmt.Errorf("existing job destination does not match checkpoint artifact (expected %d bytes sha256 %s); removed it and refused publication", sourceBytes, sourceSHA)
	} else if !os.IsNotExist(err) {
		return err
	}
	if err := copyFileAtomic(source, dest); err != nil {
		return err
	}
	destSHA, destBytes, err := hashFile(dest)
	if err != nil || destBytes != sourceBytes || destSHA != sourceSHA {
		_ = os.Remove(dest)
		if err != nil {
			return err
		}
		return fmt.Errorf("published destination failed checkpoint integrity verification")
	}
	return nil
}

// rewriteQuantlabReport copies the report then updates its adopted output
// identity without discarding fields produced by newer pipeline versions.
func rewriteQuantlabReport(source, dest, outputPath string) error {
	data, err := os.ReadFile(source)
	if err != nil {
		return err
	}
	var report map[string]any
	if err := json.Unmarshal(data, &report); err != nil {
		return err
	}
	output, ok := report["output"].(map[string]any)
	if !ok {
		return fmt.Errorf("quantlab report has no output object")
	}
	sha, bytes, err := hashFile(outputPath)
	if err != nil {
		return err
	}
	output["path"] = outputPath
	if _, exists := output["bytes"]; exists {
		output["bytes"] = bytes
	}
	for _, key := range []string{"sha256", "sha", "hash"} {
		if _, exists := output[key]; exists {
			output[key] = sha
		}
	}
	data, err = json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(dest), "."+filepath.Base(dest)+".tmp-")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, dest)
}

func validateQuantlabOutput(path string) error {
	issues, _, err := gguf.ValidateFile(path)
	if err != nil {
		return err
	}
	if len(issues) > 0 {
		return fmt.Errorf("%s", issues[0])
	}
	return nil
}

// quantlabFailureEvidence builds the failed-job result from the run
// checkpoint: measurements recorded before the failure plus gate outcomes
// evaluated against them, so a rejection is auditable without the emitted
// report (which only exists once emit publishes).
func (m *Manager) quantlabFailureEvidence(jobID, stateDir, effort string, warnings []string, repaired bool, runErr error) map[string]any {
	res := map[string]any{
		"effort":          effort,
		"run_id":          jobID,
		"source_repaired": repaired,
		"error":           runErr.Error(),
	}
	if len(warnings) > 0 {
		res["warnings"] = warnings
	}
	run, err := (state.Store{Dir: stateDir}).Load(jobID)
	if err != nil {
		res["quality_gate"] = map[string]any{"passed": false, "rejection_cause": []string{runErr.Error()}}
		return res
	}
	if len(run.Measurements) > 0 {
		res["measurements"] = run.Measurements
	}
	gateSet := pipeline.GatesForTargetBPW(run.Config.TargetBPW)
	var gates []map[string]any
	var causes []string
	for _, g := range gateSet {
		entry := map[string]any{
			"metric":   string(g.Metric),
			"maxDelta": g.MaxDelta,
			"measured": false,
			"pass":     false,
		}
		if g.MaxAbsolute > 0 {
			entry["maxAbsolute"] = g.MaxAbsolute
		}
		// Latest measurement for this metric wins, matching emit's
		// best-profile gating.
		for i := len(run.Measurements) - 1; i >= 0; i-- {
			mm := run.Measurements[i]
			if mm.Metric != g.Metric {
				continue
			}
			entry["measured"] = true
			entry["value"] = mm.Value
			entry["delta"] = mm.Delta
			entry["pass"] = g.Passes(mm)
			break
		}
		gates = append(gates, entry)
	}
	res["gates"] = gates
	gatesPass := len(gates) > 0
	for _, g := range gates {
		pass, _ := g["pass"].(bool)
		if !pass {
			gatesPass = false
			break
		}
	}
	res["gates_pass"] = gatesPass
	res["quality_gate"] = map[string]any{"passed": gatesPass, "rejection_cause": causes}
	return res
}
