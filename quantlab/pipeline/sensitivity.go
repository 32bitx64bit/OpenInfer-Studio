package pipeline

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
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
	// minProbeMargin is the smallest background-corrected role KLD a probe
	// must clear to be trusted. Below it the measurement is noise (probes
	// routinely report roles at or under the background), so the role keeps
	// this conservative share instead of looking free to compress.
	minProbeMargin = 0.5
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
	// Probes2 holds the optional second-rung probe per role (two-rung
	// exponent fit). A record whose ProbeDType no longer matches the role's
	// planned second rung is re-measured.
	Probes2 map[string]sensitivityProbe `json:"probes2,omitempty"`
	Pinned  []string                    `json:"pinned,omitempty"`
	// DepthVersion and Buckets track depth-bucket probes separately so
	// changing the depth policy does not invalidate role probes.
	DepthVersion int                         `json:"depthVersion,omitempty"`
	Buckets      map[string]sensitivityProbe `json:"buckets,omitempty"`
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

// dropProbeScratch removes probe scratch after a failed probe run. Pause and
// resume use cancellation and keep the scratch resumable; any real failure
// must not leave gigabytes behind.
func (e *Engine) dropProbeScratch(err error) {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return
	}
	os.RemoveAll(e.probeDir())
}

func (e *Engine) probeLogitsPath() string {
	return filepath.Join(e.workDir(), "baseline-logits-probe.bin")
}

// sensitivityEnabled reports whether solve calibrates role sensitivities.
// Probes need the exact loss table (the per-tensor rung shape), a KLD
// harness (llama-perplexity + corpus), and the effort or Extra opt-in.
// Either the evaluation corpus or the disjoint search holdout satisfies the
// corpus requirement.
func (e *Engine) sensitivityEnabled() bool {
	if e.Extra.NoSensitivity {
		return false
	}
	if !e.exactEstimatorEnabled() {
		return false
	}
	cfg := e.Run.Config
	if cfg.Tools.LlamaPerplexity == "" || cfg.Tools.LlamaQuantize == "" {
		return false
	}
	if cfg.EvalCorpus == "" {
		if _, ok := e.searchCorpusPath(); !ok {
			return false
		}
	}
	if e.Extra.Sensitivity {
		return true
	}
	return e.effortProfile().SensitivityProbes
}

// probeEvalConfig is the short KLD evaluation used for every probe: the
// tuning-only search holdout when one exists (so probes never tune on the
// final evaluation corpus), else the run's evaluation corpus, with the
// context and the effort's probe chunk count — never more chunks than the
// final validation uses.
func (e *Engine) probeEvalConfig() orchestrate.EvalConfig {
	cfg := e.Run.Config
	chunks := e.effortProfile().ProbeChunks
	if chunks <= 0 {
		chunks = 2
	}
	if e.Extra.Chunks > 0 && e.Extra.Chunks < chunks {
		chunks = e.Extra.Chunks
	}
	corpus := cfg.EvalCorpus
	if path, ok := e.searchCorpusPath(); ok {
		corpus = path
	} else {
		e.warnProbeSharedCorpus()
	}
	return orchestrate.EvalConfig{
		CorpusPath: corpus,
		CtxSize:    cfg.CtxSize,
		Chunks:     chunks,
		Threads:    cfg.Threads,
		NGPULayers: -1,
	}
}

// warnProbeSharedCorpus prints the shared-corpus warning once per engine.
func (e *Engine) warnProbeSharedCorpus() {
	if e.probeSharedCorpusWarned {
		return
	}
	e.probeSharedCorpusWarned = true
	e.printf("  sensitivity: no distinct search corpus; probes share the evaluation corpus\n")
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
	// The stored probe values change meaning with the tail term.
	if e.tailProbeEnabled() {
		_, _ = fmt.Fprintf(h, "tail=%g\x00", probeTailWeight)
	}
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
	if st.Probes2 == nil {
		st.Probes2 = map[string]sensitivityProbe{}
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
		st = newSensitivityState(signature)
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
	kind2 := e.secondProbeRungs(bank, todo, kind, table)

	missing := 0
	for _, g := range todo {
		if !probeRecorded(st.Probes, g, kind[g.Role]) {
			missing++
		}
		if d2, ok := kind2[g.Role]; ok && !probeRecorded(st.Probes2, g, d2) {
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
		if err := e.runSensitivityProbes(ctx, bank, set, evalCfg, capsP, st, todo, kind, kind2); err != nil {
			// A too-short search holdout only surfaces from the tool: fall
			// back to the evaluation corpus and re-probe under a fresh
			// signature so the measurements stay corpus-consistent.
			if errors.Is(err, errEvalCorpusTooShort) && evalCfg.CorpusPath != e.Run.Config.EvalCorpus {
				e.warnProbeSharedCorpus()
				os.RemoveAll(e.probeDir())
				evalCfg.CorpusPath = e.Run.Config.EvalCorpus
				signature, serr := e.sensitivitySignature(bank, evalCfg)
				if serr != nil {
					return nil, serr
				}
				st = newSensitivityState(signature)
				if err := e.runSensitivityProbes(ctx, bank, set, evalCfg, capsP, st, todo, kind, kind2); err != nil {
					e.dropProbeScratch(err)
					return nil, err
				}
			} else {
				// Probe scratch is model-sized; drop it on failure too
				// (cancellation keeps it so pause/resume can pick up).
				e.dropProbeScratch(err)
				return nil, err
			}
		}
	}
	sens := &profile.Sensitivity{Background: *st.Background, Roles: map[string]profile.RoleSensitivity{}, Pinned: st.Pinned}
	for _, g := range todo {
		p := st.Probes[g.Role]
		kld := calibratedRoleKLD(p.KLD, sens.Background)
		var sum float64
		for _, name := range g.Tensors {
			sum += table[name][p.ProbeDType]
		}
		rs := profile.RoleSensitivity{
			Role: g.Role, ProbeDType: p.ProbeDType, KLD: kld, SumWSSE: sum,
			Elements: g.Elements, Tensors: g.Tensors,
		}
		if d2, ok := kind2[g.Role]; ok && probeRecorded(st.Probes2, g, d2) {
			p2 := st.Probes2[g.Role]
			var sum2 float64
			for _, name := range g.Tensors {
				sum2 += table[name][d2]
			}
			rs.ProbeDType2, rs.KLD2, rs.SumWSSE2 = d2, calibratedRoleKLD(p2.KLD, sens.Background), sum2
			// Fit only from measurements clear of probe noise; otherwise
			// the floored KLDs would fabricate a slope.
			if probeReliable(p.KLD, sens.Background) && probeReliable(p2.KLD, sens.Background) {
				if b, ok := profile.FitRungExponent(kld, sum, rs.KLD2, sum2); ok {
					rs.Exponent = b
				}
			}
		}
		sens.Roles[g.Role] = rs
	}
	if err := sens.Validate(); err != nil {
		return nil, err
	}
	// Depth-bucket probes: redistribute each role's KLD across layers.
	// Runs before probe-dir cleanup because it reuses the background anchor.
	if e.depthProbesEnabled() {
		saveFn := func() error { return e.writeJSON(e.sensitivityPath(), st) }
		dm := e.runDepthProbes(ctx, bank, set, evalCfg, capsP, st, sens, todo, kind, saveFn)
		if dm != nil {
			sens.Depth = dm
			if err := sens.Validate(); err != nil {
				return nil, err
			}
		}
	}
	// Probe scratch is model-sized; free it before the quantize stage.
	os.RemoveAll(e.probeDir())
	return sens, nil
}

// depthProbesEnabled reports whether depth-bucket probes run alongside
// role probes.
func (e *Engine) depthProbesEnabled() bool {
	if e.Extra.NoDepthProbes {
		return false
	}
	return e.effortProfile().DepthProbes
}

// depthStateVersion versions the depth-probe record. v2 probes token-mixer
// and FFN families separately (keys from profile.DepthKey); v1 records are
// discarded on load.
const depthStateVersion = 2

// depthSplitEnabled reports whether depth probes measure the token-mixer and
// FFN families separately (two probes per bucket) instead of one combined
// probe per bucket.
func (e *Engine) depthSplitEnabled() bool { return !e.Extra.NoDepthSplit }

// depthProbe is one planned depth probe: a bucket, the family it covers
// ("" = every family), and the probed tensors with their probe dtypes.
type depthProbe struct {
	key    string
	bucket [2]int
	family string
	dtypes map[string]core.DType
}

// planDepthProbes lists the depth probes for the calibrated tensors.
func planDepthProbes(buckets [][2]int, probeDTypeOf map[string]core.DType, split bool) []depthProbe {
	families := []string{""}
	if split {
		families = profile.DepthFamilies
	}
	var out []depthProbe
	for _, b := range buckets {
		for _, fam := range families {
			dts := map[string]core.DType{}
			for name, d := range probeDTypeOf {
				l := profile.LayerIndex(name)
				if l < b[0] || l > b[1] {
					continue
				}
				if fam != "" && profile.DepthFamily(name) != fam {
					continue
				}
				dts[name] = d
			}
			if len(dts) == 0 {
				continue
			}
			out = append(out, depthProbe{key: profile.DepthKey(fam, b), bucket: b, family: fam, dtypes: dts})
		}
	}
	return out
}

// runDepthProbes evaluates the depth probes (one per bucket, or one per
// bucket and family when split) and returns the fitted depth model.
// Returns nil when the model has too few layers or the probes fail
// (fail-open: the flat model is used).
func (e *Engine) runDepthProbes(ctx context.Context, bank *core.TensorBank, set *anchor.Set,
	evalCfg orchestrate.EvalConfig, capsP *orchestrate.Capabilities,
	st *sensitivityState, sens *profile.Sensitivity,
	todo []profile.RoleGroup, kind map[string]core.DType,
	save func() error) *profile.DepthModel {

	// Determine layer count from calibrated tensors.
	maxLayer := -1
	for _, g := range todo {
		for _, name := range g.Tensors {
			if l := profile.LayerIndex(name); l > maxLayer {
				maxLayer = l
			}
		}
	}
	n := maxLayer + 1
	buckets := profile.DepthBuckets(n)
	if len(buckets) == 0 {
		return nil
	}
	// Map every calibrated tensor name → role probe dtype.
	probeDTypeOf := map[string]core.DType{}
	for _, g := range todo {
		d := kind[g.Role]
		for _, name := range g.Tensors {
			probeDTypeOf[name] = d
		}
	}
	plan := planDepthProbes(buckets, probeDTypeOf, e.depthSplitEnabled())
	if len(plan) == 0 {
		return nil
	}
	// Clear stale depth state if the version or the probe set changed.
	needClear := st.DepthVersion != depthStateVersion
	if !needClear && st.Buckets != nil {
		want := map[string]bool{}
		for _, p := range plan {
			want[p.key] = true
		}
		for key := range st.Buckets {
			if !want[key] {
				needClear = true
				break
			}
		}
	}
	if needClear {
		st.DepthVersion = depthStateVersion
		st.Buckets = map[string]sensitivityProbe{}
	}
	if st.Buckets == nil {
		st.Buckets = map[string]sensitivityProbe{}
	}

	bgDir := filepath.Join(e.probeDir(), "background")
	missing := 0
	for _, p := range plan {
		if _, ok := st.Buckets[p.key]; !ok {
			missing++
		}
	}
	if missing > 0 {
		if err := os.MkdirAll(e.probeDir(), 0o755); err != nil {
			return nil
		}
		for done, p := range plan {
			if _, ok := st.Buckets[p.key]; ok {
				continue
			}
			label := fmt.Sprintf("depth %d-%d", p.bucket[0], p.bucket[1])
			if p.family != "" {
				label = fmt.Sprintf("depth %s %d-%d", p.family, p.bucket[0], p.bucket[1])
			}
			e.obsProgress(core.StageSolve, float64(done)/float64(len(plan)), "sensitivity: "+label)
			// One keep-set per distinct probe dtype.
			byDType := map[core.DType]map[string]struct{}{}
			for name, d := range p.dtypes {
				if byDType[d] == nil {
					byDType[d] = map[string]struct{}{}
				}
				byDType[d][name] = struct{}{}
			}
			var dts []core.DType
			for d := range byDType {
				dts = append(dts, d)
			}
			sort.Slice(dts, func(i, j int) bool { return dts[i] < dts[j] })
			depthDir := filepath.Join(e.probeDir(), p.key)
			if err := e.runSparseTrimmedAnchorJobs(ctx, dts,
				func(d core.DType) (map[string]struct{}, error) { return copyKeep(byDType[d]), nil },
				depthDir, "meta.json"); err != nil {
				e.printf("  sensitivity: %s probe failed: %v\n", label, err)
				continue
			}
			m, err := e.evalProbeModel(ctx, bank, set, evalCfg, capsP, e.probeLogitsPath(), p.key, p.dtypes,
				[]string{bgDir, depthDir})
			os.RemoveAll(depthDir)
			if err != nil {
				e.printf("  sensitivity: %s eval failed: %v\n", label, err)
				continue
			}
			st.Buckets[p.key] = sensitivityProbe{
				Role: p.key, KLD: e.probeValue(m), Perplexity: m.Perplexity,
			}
			if err := save(); err != nil {
				return nil
			}
		}
	}

	// ComputeDepthModel wants background-corrected KLD, like role probes.
	measured := map[string]float64{}
	for key, p := range st.Buckets {
		measured[key] = calibratedRoleKLD(p.KLD, sens.Background)
	}
	return profile.ComputeDepthModel(bank, buckets, sens.Roles, measured, sens.Background)
}

func newSensitivityState(signature string) *sensitivityState {
	return &sensitivityState{Version: 1, Signature: signature,
		Probes: map[string]sensitivityProbe{}, Probes2: map[string]sensitivityProbe{}}
}

// probeRecorded reports whether probes holds a usable record of role g at
// dtype d.
func probeRecorded(probes map[string]sensitivityProbe, g profile.RoleGroup, d core.DType) bool {
	p, ok := probes[g.Role]
	return ok && p.ProbeDType == d && len(p.Tensors) == len(g.Tensors)
}

// probeReliable reports whether a raw probe value clears the background by
// at least the background itself: below that, 2–4 chunk probes are
// dominated by noise and must not drive an exponent fit.
func probeReliable(raw, background float64) bool {
	return raw-background >= background && raw-background > 1e-6
}

// twoRungEnabled reports whether roles get a second probe rung near the
// compression target for the exponent fit.
func (e *Engine) twoRungEnabled() bool {
	return !e.Extra.NoTwoRung && e.effectiveTargetBPW() > 0
}

// effectiveTargetBPW is the configured target, or the bits per weight the
// payload budget implies when only a byte budget was given (0 = unbounded).
func (e *Engine) effectiveTargetBPW() float64 {
	cfg := e.Run.Config
	if cfg.TargetBPW > 0 {
		return cfg.TargetBPW
	}
	if cfg.BudgetBytes == 0 || e.Run.Bank == nil {
		return 0
	}
	var elems uint64
	for _, t := range e.Run.Bank.Tensors {
		elems += t.Elements
	}
	if elems == 0 {
		return 0
	}
	return float64(e.payloadBudget()) * 8 / float64(elems)
}

// secondProbeRungs plans each role's second probe rung (role → dtype). Roles
// whose members cannot all take a distinct second rung are left out and keep
// the linear single-rung model.
func (e *Engine) secondProbeRungs(bank *core.TensorBank, todo []profile.RoleGroup,
	kind map[string]core.DType, table map[string]map[core.DType]float64) map[string]core.DType {
	out := map[string]core.DType{}
	if !e.twoRungEnabled() {
		return out
	}
	prefs := profile.SecondProbeDTypes(e.effectiveTargetBPW())
	for _, g := range todo {
		var list []core.DType
		for _, d := range prefs {
			if d.BaseTensorType() != kind[g.Role].BaseTensorType() {
				list = append(list, d)
			}
		}
		if d, ok := profile.ProbeDTypeFor(bank, g.Tensors, list, table); ok {
			out[g.Role] = d
		}
	}
	return out
}

// runSensitivityProbes materializes the background anchor, the probe
// baseline logits, and every missing probe (first rung per role, then the
// optional second rung), checkpointing after each.
func (e *Engine) runSensitivityProbes(ctx context.Context, bank *core.TensorBank, set *anchor.Set,
	evalCfg orchestrate.EvalConfig, capsP *orchestrate.Capabilities, st *sensitivityState,
	todo []profile.RoleGroup, kind, kind2 map[string]core.DType) error {
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

	steps := len(todo) + len(kind2) + 1
	done := 0
	if st.Background == nil {
		e.obsProgress(core.StageSolve, float64(done)/float64(steps), "sensitivity: background KLD")
		m, err := e.evalProbeModel(ctx, bank, set, evalCfg, capsP, logits, "background", nil, []string{bgDir})
		if err != nil {
			return err
		}
		bg := e.probeValue(m)
		st.Background = &bg
		if err := save(); err != nil {
			return err
		}
		e.printf("  sensitivity: background (all %s) kld %.5f\n", backgroundDType, bg)
	}
	done++

	probe := func(g profile.RoleGroup, d core.DType, probes map[string]sensitivityProbe, prefix string) error {
		if probeRecorded(probes, g, d) {
			done++
			return nil
		}
		e.obsProgress(core.StageSolve, float64(done)/float64(steps), fmt.Sprintf("sensitivity: probing %s %s", g.Role, d))
		roleDir := filepath.Join(e.probeDir(), prefix+sanitizeRole(g.Role))
		keep := map[string]struct{}{}
		probedDTypes := map[string]core.DType{}
		for _, name := range g.Tensors {
			keep[name] = struct{}{}
			probedDTypes[name] = d
		}
		if err := e.runSparseTrimmedAnchorJobs(ctx, []core.DType{d},
			func(core.DType) (map[string]struct{}, error) { return copyKeep(keep), nil }, roleDir, "meta.json"); err != nil {
			return fmt.Errorf("sensitivity probe %s %s: %w", g.Role, d, err)
		}
		label := g.Role
		if prefix != "role-" {
			label = prefix + g.Role
		}
		m, err := e.evalProbeModel(ctx, bank, set, evalCfg, capsP, logits, label, probedDTypes, []string{bgDir, roleDir})
		if err != nil {
			return err
		}
		os.RemoveAll(roleDir)
		probes[g.Role] = sensitivityProbe{
			Role: g.Role, ProbeDType: d, KLD: e.probeValue(m), Perplexity: m.Perplexity,
			Tensors: g.Tensors, Elements: g.Elements,
		}
		if err := save(); err != nil {
			return err
		}
		e.printf("  sensitivity: %-16s %-7s kld %.5f (%d tensors, %.1f%% of weights)\n",
			g.Role, d, e.probeValue(m), len(g.Tensors), 100*float64(g.Elements)/float64(totalProbeElements(todo)))
		done++
		return nil
	}
	for _, g := range todo {
		if err := probe(g, kind[g.Role], st.Probes, "role-"); err != nil {
			return err
		}
	}
	if st.Probes2 == nil {
		st.Probes2 = map[string]sensitivityProbe{}
	}
	for _, g := range todo {
		if d2, ok := kind2[g.Role]; ok {
			if err := probe(g, d2, st.Probes2, "rung2-"); err != nil {
				return err
			}
		}
	}
	return nil
}

// evalProbeModel assembles a probe GGUF (probed tensors at their target
// dtype, other quantizable tensors at the background rung, preserved
// tensors as stored) from the given anchor directories and measures its
// KLD against the probe baseline logits. probedDTypes maps tensor name →
// quant target; nil means no probed tensors (pure background).
func (e *Engine) evalProbeModel(ctx context.Context, bank *core.TensorBank, set *anchor.Set,
	evalCfg orchestrate.EvalConfig, capsP *orchestrate.Capabilities, logits, label string,
	probedDTypes map[string]core.DType, anchorDirs []string) (orchestrate.EvalMetrics, error) {
	assignments := make([]core.QuantAssignment, 0, len(bank.Tensors))
	for _, t := range bank.Tensors {
		target := t.DType
		if t.Quantizable() && !set.Preserved(t) {
			target = backgroundDType
			if d, ok := probedDTypes[t.Name]; ok {
				target = d
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

// probeTailWeight scales the p99 KLD term of a probe score. Per-token p99
// typically sits ~10x above the mean, so 0.1 gives the tail roughly the
// mean's weight: a role whose damage is spiky (rare tokens, logit tails)
// prices higher than one with the same mean spread evenly, which is what the
// p95 gate and generation quality care about.
const probeTailWeight = 0.1

// tailProbeEnabled reports whether probes score mean + tail.
func (e *Engine) tailProbeEnabled() bool { return !e.Extra.NoTailProbe }

// probeValue is the scalar a probe evaluation contributes to the
// sensitivity model: mean KLD, plus probeTailWeight × p99 KLD when the tool
// reports it. Background and role probes share the scoring, so background
// subtraction stays consistent.
func (e *Engine) probeValue(m orchestrate.EvalMetrics) float64 {
	if e.tailProbeEnabled() && m.HasP99 && m.P99KLD >= 0 {
		return m.MeanKLD + probeTailWeight*m.P99KLD
	}
	return m.MeanKLD
}

// calibratedRoleKLD is the background-corrected role KLD the solver
// consumes. Measurements at or under the background are probe noise (a
// 4-chunk run routinely reports roles below it), so they keep a conservative
// share instead of making the role look free to harvest.
func calibratedRoleKLD(measured, background float64) float64 {
	kld := measured - background
	if kld < minProbeMargin*background {
		kld = minProbeMargin * background
	}
	if kld < 1e-6 {
		kld = 1e-6
	}
	return kld
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
		rung2 := ""
		if rs.ProbeDType2 != "" {
			b := rs.Exponent
			fit := "linear"
			if b > 0 {
				fit = fmt.Sprintf("b=%.2f", b)
			}
			rung2 = fmt.Sprintf("  %s kld %.5f  %s", rs.ProbeDType2, rs.KLD2, fit)
		}
		e.printf("    %-18s %-7s kld %.5f  kld/wsse %.3g%s\n", r, rs.ProbeDType, rs.KLD, perW, rung2)
	}
	if len(s.Pinned) > 0 {
		e.printf("    pinned to top fidelity: %v\n", s.Pinned)
	}
	if s.Depth != nil && len(s.Depth.Buckets) > 0 {
		for _, b := range s.Depth.Buckets {
			fam, clamp := b.Family, ""
			if fam == "" {
				fam = "all"
			}
			if b.Clamped {
				clamp = "  [clamped]"
			}
			e.printf("    %-3s layers %d-%d  rho=%.2f (measured %.5f, predicted %.5f)%s\n",
				fam, b.First, b.Last, b.Factor, b.MeasuredKLD, b.PredictedKLD, clamp)
		}
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
