package pipeline

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"quantlab/anchor"
	"quantlab/core"
	"quantlab/orchestrate"
	"quantlab/profile"
	"quantlab/tensorbank"
)

// Sensitivity probes measure, on this model and this corpus, how much mean
// KLD each tensor role costs when the whole role is quantized to a common
// aggressive rung while everything else stays near-lossless (Q8_0). The
// solver then trades bytes between roles on those measurements instead of
// on hand-tuned priors; see profile.Sensitivity for the loss model.
//
// Cost: one Q8_0 quantization of the model, one short baseline eval, and
// per role one trimmed llama-quantize of that role plus one short KLD eval
// (roles below minProbeShare of the payload are pinned to top fidelity and
// skipped). Every probe result is checkpointed, so a resumed run only
// measures what is missing.

const (
	// maxProbeRoles bounds the probe count on exotic architectures; the
	// smallest roles beyond it are merged into one "other" probe.
	maxProbeRoles = 24
	// minProbeShare is the element share below which a role is pinned to
	// its highest-fidelity rung instead of probed: its bytes cannot move
	// the budget, so any allocation would give it top fidelity anyway.
	minProbeShare = 0.002
	// backgroundDType is the near-lossless rung every non-probed tensor
	// takes in a probe model.
	backgroundDType = core.DTypeQ8_0
)

// sensitivityState is the resumable probe record (work dir
// sensitivity.json). Raw mean KLDs are stored; the background subtraction
// happens when the profile.Sensitivity is built.
type sensitivityState struct {
	Version    int                         `json:"version"`
	Signature  string                      `json:"signature"`
	Background *float64                    `json:"background,omitempty"`
	Probes     map[string]sensitivityProbe `json:"probes"`
	Pinned     []string                    `json:"pinned,omitempty"`
}

type sensitivityProbe struct {
	Role       string     `json:"role"`
	ProbeDType core.DType `json:"probeDType"`
	KLD        float64    `json:"kld"`
	Perplexity float64    `json:"perplexity,omitempty"`
	Tensors    []string   `json:"tensors"`
	Elements   uint64     `json:"elements"`
}

func (e *Engine) sensitivityPath() string {
	return filepath.Join(e.workDir(), "sensitivity.json")
}

func (e *Engine) probeDir() string { return filepath.Join(e.anchorDir(), "probes") }

func (e *Engine) probeLogitsPath() string {
	return filepath.Join(e.workDir(), "baseline-logits-probe.bin")
}

// sensitivityEnabled reports whether solve calibrates role sensitivities.
// Probes need the exact loss table (the per-tensor rung shape), a KLD
// harness (llama-perplexity + corpus), and the effort or Extra opt-in.
func (e *Engine) sensitivityEnabled() bool {
	if e.Extra.NoSensitivity {
		return false
	}
	if !e.exactEstimatorEnabled() {
		return false
	}
	cfg := e.Run.Config
	if cfg.EvalCorpus == "" || cfg.Tools.LlamaPerplexity == "" || cfg.Tools.LlamaQuantize == "" {
		return false
	}
	if e.Extra.Sensitivity {
		return true
	}
	return e.effortProfile().SensitivityProbes
}

// probeEvalConfig is the short KLD evaluation used for every probe: the
// run's corpus and context with the effort's probe chunk count, never more
// chunks than the final validation uses.
func (e *Engine) probeEvalConfig() orchestrate.EvalConfig {
	cfg := e.Run.Config
	chunks := e.effortProfile().ProbeChunks
	if chunks <= 0 {
		chunks = 2
	}
	if e.Extra.Chunks > 0 && e.Extra.Chunks < chunks {
		chunks = e.Extra.Chunks
	}
	return orchestrate.EvalConfig{
		CorpusPath: cfg.EvalCorpus,
		CtxSize:    cfg.CtxSize,
		Chunks:     chunks,
		Threads:    cfg.Threads,
		NGPULayers: -1,
	}
}

// sensitivitySignature identifies what the probe results depend on: the
// exact-table identity (payload, imatrix, candidates), the probe eval
// configuration, the tool binaries, and the probe policy constants.
func (e *Engine) sensitivitySignature(bank *core.TensorBank, evalCfg orchestrate.EvalConfig) (string, error) {
	exactSig, err := e.exactLossSignature(bank)
	if err != nil {
		return "", err
	}
	pplSHA, err := e.cachedFileSHA(e.Run.Config.Tools.LlamaPerplexity)
	if err != nil {
		return "", err
	}
	qSHA, err := e.cachedFileSHA(e.Run.Config.Tools.LlamaQuantize)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	_, _ = fmt.Fprintf(h, "v1\x00%s\x00%s\x00%s\x00corpus=%s\x00ctx=%d\x00chunks=%d\x00bg=%s\x00share=%g\x00max=%d\x00",
		exactSig, pplSHA, qSHA, evalCfg.CorpusPath, evalCfg.CtxSize, evalCfg.Chunks,
		backgroundDType, minProbeShare, maxProbeRoles)
	for _, d := range profile.DefaultProbeDTypes {
		_, _ = fmt.Fprintf(h, "%s\x00", d)
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

func (e *Engine) loadSensitivityState(signature string) *sensitivityState {
	data, err := os.ReadFile(e.sensitivityPath())
	if err != nil {
		return nil
	}
	var st sensitivityState
	if json.Unmarshal(data, &st) != nil || st.Version != 1 || st.Signature != signature {
		return nil
	}
	if st.Probes == nil {
		st.Probes = map[string]sensitivityProbe{}
	}
	return &st
}

// calibrateSensitivity runs (or resumes) the probes and returns the
// calibrated model. It returns (nil, nil) when nothing is worth probing.
func (e *Engine) calibrateSensitivity(ctx context.Context, bank *core.TensorBank,
	set *anchor.Set, table map[string]map[core.DType]float64) (*profile.Sensitivity, error) {
	evalCfg := e.probeEvalConfig()
	capsP, err := e.caps(ctx, orchestrate.ToolPerplexity)
	if err != nil {
		return nil, err
	}
	if !capsP.Has("--kl-divergence") {
		e.printf("  sensitivity: skipped (%s lacks --kl-divergence)\n", e.Run.Config.Tools.LlamaPerplexity)
		return nil, nil
	}
	if _, err := e.caps(ctx, orchestrate.ToolLlamaQuantize); err != nil {
		return nil, err
	}
	signature, err := e.sensitivitySignature(bank, evalCfg)
	if err != nil {
		return nil, err
	}
	st := e.loadSensitivityState(signature)
	if st == nil {
		st = &sensitivityState{Version: 1, Signature: signature, Probes: map[string]sensitivityProbe{}}
	}

	include := func(t core.TensorDesc) bool { return !set.Preserved(t) }
	groups := profile.GroupByRole(bank, maxProbeRoles, include)
	var total uint64
	for _, g := range groups {
		total += g.Elements
	}
	var probes []profile.RoleGroup
	var pinned []string
	for _, g := range groups {
		if float64(g.Elements) < minProbeShare*float64(total) {
			pinned = append(pinned, g.Role)
			continue
		}
		probes = append(probes, g)
	}
	if len(probes) == 0 {
		return nil, nil
	}
	sort.Strings(pinned)
	st.Pinned = pinned

	// Resolve each role's probe rung up front so a role that cannot take
	// any preferred rung is pinned rather than half-measured.
	kind := map[string]core.DType{}
	var todo []profile.RoleGroup
	for _, g := range probes {
		d, ok := profile.ProbeDTypeFor(bank, g.Tensors, profile.DefaultProbeDTypes, table)
		if !ok {
			st.Pinned = append(st.Pinned, g.Role)
			continue
		}
		kind[g.Role] = d
		todo = append(todo, g)
	}
	sort.Strings(st.Pinned)
	if len(todo) == 0 {
		return nil, nil
	}

	missing := 0
	for _, g := range todo {
		if p, ok := st.Probes[g.Role]; !ok || p.ProbeDType != kind[g.Role] || len(p.Tensors) != len(g.Tensors) {
			missing++
		}
	}
	if st.Background == nil {
		missing++
	}
	if missing > 0 {
		// Probe scratch peaks at the Q8_0 background anchor plus one
		// assembled probe model. Without room, fall back to the
		// uncalibrated solver rather than fail the run.
		if err := os.MkdirAll(e.probeDir(), 0o755); err != nil {
			return nil, err
		}
		need := saturatingAdd(estimatedArtifact(bank, backgroundDType), estimatedArtifact(bank, backgroundDType))
		need = saturatingAdd(need, need/10)
		if free, ok := tensorbank.DiskFree(e.probeDir()); ok && free < need {
			e.printf("  sensitivity: skipped (need ~%d MiB free for probe scratch, have %d MiB)\n", need>>20, free>>20)
			return nil, nil
		}
		if err := e.runSensitivityProbes(ctx, bank, set, evalCfg, capsP, st, todo, kind); err != nil {
			return nil, err
		}
	}
	// Probe scratch is model-sized; free it before the quantize stage.
	os.RemoveAll(e.probeDir())

	sens := &profile.Sensitivity{Background: *st.Background, Roles: map[string]profile.RoleSensitivity{}, Pinned: st.Pinned}
	for _, g := range todo {
		p := st.Probes[g.Role]
		kld := p.KLD - sens.Background
		if floor := 0.1 * sens.Background; kld < floor {
			kld = floor
		}
		if kld < 1e-6 {
			kld = 1e-6
		}
		var sum float64
		for _, name := range g.Tensors {
			sum += table[name][p.ProbeDType]
		}
		sens.Roles[g.Role] = profile.RoleSensitivity{
			Role: g.Role, ProbeDType: p.ProbeDType, KLD: kld, SumWSSE: sum,
			Elements: g.Elements, Tensors: g.Tensors,
		}
	}
	if err := sens.Validate(); err != nil {
		return nil, err
	}
	return sens, nil
}

// runSensitivityProbes materializes the background anchor, the probe
// baseline logits, and every missing probe, checkpointing after each.
func (e *Engine) runSensitivityProbes(ctx context.Context, bank *core.TensorBank, set *anchor.Set,
	evalCfg orchestrate.EvalConfig, capsP *orchestrate.Capabilities, st *sensitivityState,
	todo []profile.RoleGroup, kind map[string]core.DType) error {
	if err := os.MkdirAll(e.probeDir(), 0o755); err != nil {
		return err
	}
	save := func() error { return e.writeJSON(e.sensitivityPath(), st) }

	// Background anchor: every non-preserved quantizable tensor at Q8_0.
	// One file serves every probe.
	bgKeep := map[string]struct{}{}
	for _, t := range bank.Tensors {
		if t.Quantizable() && !set.Preserved(t) {
			bgKeep[t.Name] = struct{}{}
		}
	}
	bgDir := filepath.Join(e.probeDir(), "background")
	e.obsProgress(core.StageSolve, 0, "sensitivity: quantizing Q8_0 background")
	if err := e.runTrimmedAnchorJobs(ctx, []core.DType{backgroundDType},
		func(core.DType) (map[string]struct{}, error) { return copyKeep(bgKeep), nil }, bgDir, "meta.json"); err != nil {
		return fmt.Errorf("sensitivity background: %w", err)
	}

	logits := e.probeLogitsPath()
	if !e.recordedLogits(logits, evalCfg, capsP) {
		e.obsProgress(core.StageSolve, 0, "sensitivity: baseline logits")
		if _, _, err := e.captureBaselineLogits(ctx, evalCfg, capsP, logits, "sensitivity baseline eval"); err != nil {
			return err
		}
	}

	steps := len(todo) + 1
	done := 0
	if st.Background == nil {
		e.obsProgress(core.StageSolve, float64(done)/float64(steps), "sensitivity: background KLD")
		m, err := e.evalProbeModel(ctx, bank, set, evalCfg, capsP, logits, "background", nil, "", []string{bgDir})
		if err != nil {
			return err
		}
		bg := m.MeanKLD
		st.Background = &bg
		if err := save(); err != nil {
			return err
		}
		e.printf("  sensitivity: background (all %s) kld %.5f\n", backgroundDType, bg)
	}
	done++

	for _, g := range todo {
		d := kind[g.Role]
		if p, ok := st.Probes[g.Role]; ok && p.ProbeDType == d && len(p.Tensors) == len(g.Tensors) {
			done++
			continue
		}
		e.obsProgress(core.StageSolve, float64(done)/float64(steps), fmt.Sprintf("sensitivity: probing %s", g.Role))
		roleDir := filepath.Join(e.probeDir(), "role-"+sanitizeRole(g.Role))
		keep := map[string]struct{}{}
		for _, name := range g.Tensors {
			keep[name] = struct{}{}
		}
		if err := e.runSparseTrimmedAnchorJobs(ctx, []core.DType{d},
			func(core.DType) (map[string]struct{}, error) { return copyKeep(keep), nil }, roleDir, "meta.json"); err != nil {
			return fmt.Errorf("sensitivity probe %s: %w", g.Role, err)
		}
		m, err := e.evalProbeModel(ctx, bank, set, evalCfg, capsP, logits, g.Role, keep, d, []string{bgDir, roleDir})
		if err != nil {
			return err
		}
		os.RemoveAll(roleDir)
		st.Probes[g.Role] = sensitivityProbe{
			Role: g.Role, ProbeDType: d, KLD: m.MeanKLD, Perplexity: m.Perplexity,
			Tensors: g.Tensors, Elements: g.Elements,
		}
		if err := save(); err != nil {
			return err
		}
		e.printf("  sensitivity: %-16s %s kld %.5f (%d tensors, %.1f%% of weights)\n",
			g.Role, d, m.MeanKLD, len(g.Tensors), 100*float64(g.Elements)/float64(totalProbeElements(todo)))
		done++
	}
	return nil
}

// evalProbeModel assembles a probe GGUF (group tensors at probeDType,
// other quantizable tensors at the background rung, preserved tensors as
// stored) from the given anchor directories and measures its KLD against
// the probe baseline logits. The probe file is removed afterwards.
func (e *Engine) evalProbeModel(ctx context.Context, bank *core.TensorBank, set *anchor.Set,
	evalCfg orchestrate.EvalConfig, capsP *orchestrate.Capabilities, logits, label string,
	group map[string]struct{}, probeDType core.DType, anchorDirs []string) (orchestrate.EvalMetrics, error) {
	assignments := make([]core.QuantAssignment, 0, len(bank.Tensors))
	for _, t := range bank.Tensors {
		target := t.DType
		if t.Quantizable() && !set.Preserved(t) {
			target = backgroundDType
			if _, ok := group[t.Name]; ok {
				target = probeDType
			}
		}
		assignments = append(assignments, core.QuantAssignment{TensorName: t.Name, Target: target})
	}
	prof := &core.Profile{ID: "probe-" + sanitizeRole(label), BaseModel: bank.ModelID, Assignments: assignments}
	manifest, err := e.manifestFor(prof)
	if err != nil {
		return orchestrate.EvalMetrics{}, fmt.Errorf("sensitivity probe %s: %w", label, err)
	}
	var paths []string
	for _, dir := range anchorDirs {
		ents, err := os.ReadDir(dir)
		if err != nil {
			return orchestrate.EvalMetrics{}, err
		}
		for _, ent := range ents {
			if !ent.IsDir() && filepath.Ext(ent.Name()) == ".gguf" {
				paths = append(paths, filepath.Join(dir, ent.Name()))
			}
		}
	}
	sort.Strings(paths)
	srcs := []*tensorbank.Source{}
	closeAll := func() {
		for _, s := range srcs {
			s.Close()
		}
	}
	for _, p := range append([]string{e.payloadSource()}, paths...) {
		s, err := tensorbank.OpenSource(p)
		if err != nil {
			closeAll()
			return orchestrate.EvalMetrics{}, err
		}
		srcs = append(srcs, s)
	}
	out := filepath.Join(e.probeDir(), "probe-"+sanitizeRole(label)+".gguf")
	err = tensorbank.NewAssembler().Build(ctx, srcs, manifest, out, nil)
	closeAll()
	if err != nil {
		os.Remove(out)
		return orchestrate.EvalMetrics{}, fmt.Errorf("sensitivity probe %s: assemble: %w", label, err)
	}
	defer os.Remove(out)
	m, err := e.evalModel(ctx, evalCfg, capsP, out, logits)
	if err != nil {
		return orchestrate.EvalMetrics{}, fmt.Errorf("sensitivity probe %s: %w", label, err)
	}
	if !m.HasMeanKLD {
		return orchestrate.EvalMetrics{}, fmt.Errorf("sensitivity probe %s: tool produced no KLD", label)
	}
	return m, nil
}

// printSensitivity logs the calibrated model: per role, the background-
// corrected KLD at the probe rung and the implied KLD per unit of exact
// wSSE (how far raw wSSE is from a commensurable loss).
func (e *Engine) printSensitivity(s *profile.Sensitivity) {
	roles := make([]string, 0, len(s.Roles))
	for r := range s.Roles {
		roles = append(roles, r)
	}
	sort.Slice(roles, func(i, j int) bool { return s.Roles[roles[i]].KLD > s.Roles[roles[j]].KLD })
	e.printf("  sensitivity: %d roles calibrated (background kld %.5f)\n", len(roles), s.Background)
	for _, r := range roles {
		rs := s.Roles[r]
		perW := 0.0
		if rs.SumWSSE > 0 {
			perW = rs.KLD / rs.SumWSSE
		}
		e.printf("    %-18s %-7s kld %.5f  kld/wsse %.3g\n", r, rs.ProbeDType, rs.KLD, perW)
	}
	if len(s.Pinned) > 0 {
		e.printf("    pinned to top fidelity: %v\n", s.Pinned)
	}
}

func copyKeep(m map[string]struct{}) map[string]struct{} {
	out := make(map[string]struct{}, len(m))
	for k := range m {
		out[k] = struct{}{}
	}
	return out
}

func totalProbeElements(groups []profile.RoleGroup) uint64 {
	var n uint64
	for _, g := range groups {
		n += g.Elements
	}
	return n
}

// sanitizeRole makes a role key safe as a file-name component.
func sanitizeRole(role string) string {
	b := make([]byte, 0, len(role))
	for i := 0; i < len(role); i++ {
		c := role[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '-', c == '_':
			b = append(b, c)
		default:
			b = append(b, '_')
		}
	}
	return string(b)
}
