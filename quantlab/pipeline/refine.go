package pipeline

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"quantlab/anchor"
	"quantlab/core"
	"quantlab/orchestrate"
	"quantlab/profile"
	"quantlab/tensorbank"
)

// In-context refinement. Sensitivity probes measure each role against an
// all-Q8_0 background; in the real mix every role is degraded at once, so
// errors compound and the marginal value of a role's bits differs from its
// isolated probe — and not uniformly by role. Refinement measures that
// marginal value where the solver actually allocates: for the largest
// roles, move every tensor one rung up and one rung down inside the solved
// mix, compare the measured KLD change with the model's prediction, fold
// the per-role ratio into the sensitivity model and re-solve. A re-solved
// profile is kept only when it measures better on the tuning holdout; the
// evaluation corpus is never used to choose, so final scores stay honest.
//
// Every variant assembles from the per-dtype variant anchors quantize left
// on disk; only (tensor, rung) pairs not yet materialized cost a trimmed
// llama-quantize. Measurements are checkpointed in refine.json, so a
// resumed run never remeasures a variant.

const (
	refineStateVersion = 1
	// refineMinGain is the relative holdout improvement a re-solved profile
	// must show to replace the current one (paired evaluation on identical
	// tokens; smaller differences are not trusted).
	refineMinGain = 0.01
	// refineLambdaMin / refineLambdaMax bound a role's in-context
	// correction against probe noise.
	refineLambdaMin = 0.25
	refineLambdaMax = 4.0
)

type refineVariant struct {
	Role      string  `json:"role"`
	Dir       int     `json:"dir"` // +1 one rung up, -1 one rung down
	ProfileID string  `json:"profileID"`
	Moved     int     `json:"moved"`
	PredDelta float64 `json:"predDelta"` // predicted |Δ| (sensitivity units)
	Value     float64 `json:"value"`
}

type refineRound struct {
	BaseProfile  string             `json:"baseProfile"`
	BaseValue    float64            `json:"baseValue"`
	BaseMeanKLD  float64            `json:"baseMeanKLD"`
	Variants     []refineVariant    `json:"variants"`
	Lambdas      map[string]float64 `json:"lambdas,omitempty"`
	DefaultScale float64            `json:"defaultScale,omitempty"`
	Resolved     string             `json:"resolved,omitempty"`
	ResolvedVal  float64            `json:"resolvedValue,omitempty"`
	ResolvedKLD  float64            `json:"resolvedMeanKLD,omitempty"`
	Accepted     bool               `json:"accepted"`
	Note         string             `json:"note,omitempty"`
}

// refineMeasure is one holdout evaluation: the probe score the sensitivity
// model is calibrated in, plus the plain mean KLD the acceptance also
// guards.
type refineMeasure struct {
	Value   float64 `json:"value"`
	MeanKLD float64 `json:"meanKLD"`
}

type refineState struct {
	Version   int                      `json:"version"`
	Signature string                   `json:"signature"`
	Measured  map[string]refineMeasure `json:"measured"`
	Rounds    []refineRound            `json:"rounds,omitempty"`
	Done      bool                     `json:"done"`
	Final     string                   `json:"final,omitempty"`
}

func (e *Engine) refinePath() string { return filepath.Join(e.workDir(), "refine.json") }
func (e *Engine) refineDir() string  { return filepath.Join(e.workDir(), "refine") }
func (e *Engine) refineLogitsPath() string {
	return filepath.Join(e.workDir(), "baseline-logits-refine.bin")
}
func (e *Engine) refinedCandidatePath(profileID string) string {
	return filepath.Join(e.workDir(), "candidate-refined-"+profileID+".gguf")
}

// refineEnabled reports whether the search stage runs refinement rounds.
func (e *Engine) refineEnabled() bool {
	if e.Extra.NoRefine || e.DryRun {
		return false
	}
	return e.effortProfile().Refine && e.Run.Config.Tools.LlamaPerplexity != "" &&
		e.Run.Config.Tools.LlamaQuantize != ""
}

// refineRounds is the number of measure/re-solve rounds.
func (e *Engine) refineRounds() int {
	if Effort(e.Run.Config.Effort) == EffortDeep {
		return 2
	}
	return 1
}

// refineEvalConfig evaluates on the tuning holdout with the evaluation
// chunk count. ok=false when no holdout distinct from the evaluation corpus
// exists: refinement then refuses to choose on the corpus that scores it.
func (e *Engine) refineEvalConfig() (orchestrate.EvalConfig, bool) {
	cfg := e.Run.Config
	path, ok := e.searchCorpusPath()
	if !ok {
		return orchestrate.EvalConfig{}, false
	}
	chunks := e.Extra.Chunks
	if chunks <= 0 {
		chunks = e.effortProfile().EvalChunks
	}
	return orchestrate.EvalConfig{
		CorpusPath: path, CtxSize: cfg.CtxSize, Chunks: chunks,
		Threads: cfg.Threads, NGPULayers: -1,
	}, true
}

func (e *Engine) loadSolveArtifact() (*solveArtifact, error) {
	p := e.Run.Artifacts[core.StageSolve]
	if p == "" {
		return nil, fmt.Errorf("no solve artifact")
	}
	data, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var a solveArtifact
	if err := json.Unmarshal(data, &a); err != nil {
		return nil, err
	}
	return &a, nil
}

func (e *Engine) refineSignature(solved *solveArtifact, evalCfg orchestrate.EvalConfig) (string, error) {
	sensJSON, err := json.Marshal(solved.Sensitivity)
	if err != nil {
		return "", err
	}
	pplSHA, err := e.cachedFileSHA(e.Run.Config.Tools.LlamaPerplexity)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	fmt.Fprintf(h, "v%d\x00%s\x00%x\x00corpus=%s\x00ctx=%d\x00chunks=%d\x00ppl=%s\x00roles=%d\x00rounds=%d\x00tail=%v\x00",
		refineStateVersion, solved.Profile.ID, sha256.Sum256(sensJSON), evalCfg.CorpusPath, evalCfg.CtxSize,
		evalCfg.Chunks, pplSHA, e.effortProfile().RefineRoles, e.refineRounds(), e.tailProbeEnabled())
	return hex.EncodeToString(h.Sum(nil)), nil
}

func (e *Engine) loadRefineState(sig string) *refineState {
	fresh := &refineState{Version: refineStateVersion, Signature: sig, Measured: map[string]refineMeasure{}}
	data, err := os.ReadFile(e.refinePath())
	if err != nil {
		return fresh
	}
	var st refineState
	if json.Unmarshal(data, &st) != nil || st.Version != refineStateVersion || st.Signature != sig {
		return fresh
	}
	if st.Measured == nil {
		st.Measured = map[string]refineMeasure{}
	}
	return &st
}

// installedError marks a failure after a refined profile was installed
// (manifest, best profile and candidate artifact swapped and checkpointed).
// From then on the run must not fail open: the search stage stays
// incomplete so a resume finishes scoring the installed profile.
type installedError struct{ err error }

func (e installedError) Error() string { return e.err.Error() }
func (e installedError) Unwrap() error { return e.err }

// refine runs the refinement rounds. Failures before a refined profile is
// installed are fail-open (the solved profile stands); cancellation and
// failures after installation propagate.
func (e *Engine) refine(ctx context.Context) error {
	if !e.refineEnabled() {
		return nil
	}
	err := e.runRefine(ctx)
	if err == nil {
		return nil
	}
	var inst installedError
	if isCancel(err) || errors.As(err, &inst) {
		return err
	}
	e.printf("  refine: stopped (%v); keeping %s\n", err, e.Run.BestProfileID)
	os.RemoveAll(e.refineDir())
	return nil
}

func (e *Engine) runRefine(ctx context.Context) error {
	bank := e.Run.Bank
	if bank == nil || e.Run.Manifest == nil {
		return nil
	}
	solved, err := e.loadSolveArtifact()
	if err != nil || solved.Profile == nil || solved.Sensitivity == nil {
		e.printf("  refine: skipped (no calibrated sensitivity model)\n")
		return nil
	}
	sens := solved.Sensitivity
	if err := sens.Validate(); err != nil {
		return err
	}
	table := e.loadExactLoss(bank)
	if len(table) == 0 {
		e.printf("  refine: skipped (no exact loss table)\n")
		return nil
	}
	evalCfg, ok := e.refineEvalConfig()
	if !ok {
		e.printf("  refine: skipped (no search holdout distinct from the evaluation corpus)\n")
		return nil
	}
	capsP, err := e.caps(ctx, orchestrate.ToolPerplexity)
	if err != nil {
		return err
	}
	if !capsP.Has("--kl-divergence") {
		e.printf("  refine: skipped (%s lacks --kl-divergence)\n", e.Run.Config.Tools.LlamaPerplexity)
		return nil
	}
	sig, err := e.refineSignature(solved, evalCfg)
	if err != nil {
		return err
	}
	st := e.loadRefineState(sig)
	save := func() error { return e.writeJSON(e.refinePath(), st) }
	if st.Done {
		return e.finishRefine(ctx, st)
	}
	// Refinement is optional and outside the scratch estimate's bound: it
	// runs only with room for one assembled variant, the extra ±1-rung
	// anchors and the refined candidate (each ≤ the manifest), plus its
	// holdout logits (sized like the evaluation logits).
	if free, ok := tensorbank.DiskFree(e.workDir()); ok {
		need := saturatingAdd(e.Run.Manifest.TotalBytes, e.Run.Manifest.TotalBytes, e.Run.Manifest.TotalBytes)
		if st, err := os.Stat(e.logitsPath()); err == nil && !e.recordedLogits(e.refineLogitsPath(), evalCfg, capsP) {
			need = saturatingAdd(need, uint64(st.Size()))
		}
		if free < need {
			e.printf("  refine: skipped (need ~%d MiB free scratch, have %d MiB)\n", need>>20, free>>20)
			return nil
		}
	}
	set, err := e.deriveAnchors(bank)
	if err != nil {
		return err
	}
	logits := e.refineLogitsPath()
	if !e.recordedLogits(logits, evalCfg, capsP) {
		e.obsProgress(core.StageSearch, 0, "refine: baseline logits")
		if _, _, err := e.captureBaselineLogits(ctx, evalCfg, capsP, logits, "refine baseline eval"); err != nil {
			return err
		}
	}
	measure := func(p *core.Profile, artifact string) (refineMeasure, error) {
		if m, ok := st.Measured[p.ID]; ok {
			return m, nil
		}
		path := artifact
		if path == "" {
			var err error
			if path, err = e.assembleRefineProfile(ctx, p); err != nil {
				return refineMeasure{}, err
			}
			defer os.Remove(path)
		}
		m, err := e.evalModel(ctx, evalCfg, capsP, path, logits)
		if err != nil {
			return refineMeasure{}, err
		}
		if !m.HasMeanKLD {
			return refineMeasure{}, fmt.Errorf("refine eval of %s produced no KLD", p.ID)
		}
		rm := refineMeasure{Value: e.probeValue(m), MeanKLD: m.MeanKLD}
		st.Measured[p.ID] = rm
		return rm, save()
	}

	base := profileFromManifest(e.Run.Manifest, bank)
	baseArtifact := e.Run.Artifacts[core.StageQuantize]
	cands := solved.Candidates
	if len(cands) == 0 {
		cands = e.candidateDTypes()
	}
	// Resuming between rounds: the next round starts from the last
	// accepted re-solve (installed only at the end), and a recorded round
	// that stopped the search ends it.
	if n := len(st.Rounds); n > 0 {
		last := st.Rounds[n-1]
		if !last.Accepted {
			st.Final, st.Done = last.BaseProfile, true
			if err := save(); err != nil {
				return err
			}
			return e.finishRefine(ctx, st)
		}
		p, err := e.refineSolve(bank, set, sens.Scaled(last.Lambdas, last.DefaultScale), table, cands)
		if err != nil {
			return err
		}
		if p.ID != last.Resolved {
			return fmt.Errorf("refine: re-solve of round %d gave %s, recorded %s", n, p.ID, last.Resolved)
		}
		base, baseArtifact = p, ""
	}
	for round := len(st.Rounds); round < e.refineRounds(); round++ {
		rr := refineRound{BaseProfile: base.ID}
		e.obsProgress(core.StageSearch, 0, fmt.Sprintf("refine round %d: base", round+1))
		bm, err := measure(base, baseArtifact)
		if err != nil {
			return err
		}
		rr.BaseValue, rr.BaseMeanKLD = bm.Value, bm.MeanKLD

		roles := refineRoles(base, bank, sens, e.effortProfile().RefineRoles)
		lambdas := map[string]float64{}
		for i, role := range roles {
			var dm, dp float64
			for _, dir := range []int{+1, -1} {
				v, moved, pred := moveRole(base, bank, set, sens, table, cands, role, dir)
				if moved == 0 || !(pred > 0) {
					continue
				}
				e.obsProgress(core.StageSearch, float64(i)/float64(len(roles)),
					fmt.Sprintf("refine round %d: %s %+d", round+1, role, dir))
				vm, err := measure(v, "")
				if err != nil {
					return err
				}
				rv := refineVariant{Role: role, Dir: dir, ProfileID: v.ID, Moved: moved, PredDelta: pred, Value: vm.Value}
				rr.Variants = append(rr.Variants, rv)
				// Up: KLD should fall by pred; down: rise by pred.
				dm += float64(dir) * (bm.Value - vm.Value)
				dp += pred
			}
			if dp > 0 && dm > 0 {
				l := dm / dp
				lambdas[role] = math.Min(refineLambdaMax, math.Max(refineLambdaMin, l))
			}
		}
		if len(lambdas) == 0 {
			rr.Note = "no role produced a usable in-context measurement"
			st.Rounds = append(st.Rounds, rr)
			break
		}
		// Unmeasured roles share the measured roles' common inflation, so a
		// uniform compounding factor does not tilt bytes toward the roles
		// that happened to be measured.
		var logSum float64
		for _, l := range lambdas {
			logSum += math.Log(l)
		}
		def := math.Exp(logSum / float64(len(lambdas)))
		rr.Lambdas, rr.DefaultScale = lambdas, def
		e.printRefineLambdas(round+1, lambdas, def)

		next, err := e.refineSolve(bank, set, sens.Scaled(lambdas, def), table, cands)
		if err != nil {
			return err
		}
		rr.Resolved = next.ID
		if next.ID == base.ID {
			rr.Note = "re-solve reproduced the current profile"
			st.Rounds = append(st.Rounds, rr)
			break
		}
		nm, err := measure(next, "")
		if err != nil {
			return err
		}
		rr.ResolvedVal, rr.ResolvedKLD = nm.Value, nm.MeanKLD
		rr.Accepted = nm.Value < bm.Value*(1-refineMinGain) && nm.MeanKLD <= bm.MeanKLD
		e.printf("  refine round %d: %s value %.5f (mean KLD %.5f) vs %s %.5f (%.5f): %s\n",
			round+1, next.ID, nm.Value, nm.MeanKLD, base.ID, bm.Value, bm.MeanKLD, acceptWord(rr.Accepted))
		st.Rounds = append(st.Rounds, rr)
		if err := save(); err != nil {
			return err
		}
		if !rr.Accepted {
			break
		}
		base, baseArtifact = next, ""
	}
	st.Final = base.ID
	st.Done = true
	if err := save(); err != nil {
		return err
	}
	return e.finishRefine(ctx, st)
}

func acceptWord(ok bool) string {
	if ok {
		return "accepted"
	}
	return "rejected"
}

// finishRefine installs the refined profile (when one was accepted and is
// not installed yet) and measures it on the evaluation corpus.
func (e *Engine) finishRefine(ctx context.Context, st *refineState) error {
	defer func() {
		os.RemoveAll(e.refineDir())
		os.Remove(e.refineLogitsPath())
		os.Remove(e.refineLogitsPath() + ".complete.json")
	}()
	if st.Final == "" || st.Final == e.Run.BestProfileID {
		if st.Final != "" && len(st.Rounds) > 0 && st.Final != st.Rounds[0].BaseProfile {
			// Installed on an earlier attempt; make sure it is scored
			// (evaluateProfile skips what is already recorded).
			art, err := e.evaluateProfile(ctx)
			if err != nil {
				return installedError{err}
			}
			if art != "" {
				e.Run.Artifacts[core.StageEvaluate] = art
			}
		}
		return nil
	}
	prof, err := e.refineFinalProfile(st)
	if err != nil {
		return err
	}
	manifest, err := e.manifestFor(prof)
	if err != nil {
		return err
	}
	if err := e.ensureAnchorsFor(ctx, manifest); err != nil {
		return err
	}
	srcs, closeSrcs, err := e.openAnchorSources()
	if err != nil {
		return err
	}
	// A per-profile path never aliases the live artifact, so a failed
	// build (which removes only its own temp file) cannot take it down.
	out := e.refinedCandidatePath(prof.ID)
	err = tensorbank.NewAssembler().Build(ctx, srcs, manifest, out, e.progressFunc(core.StageSearch, "assembling refined candidate"))
	closeSrcs()
	if err != nil {
		return err
	}
	prev, prevArtifact := e.Run.BestProfileID, e.Run.Artifacts[core.StageQuantize]
	e.Run.Manifest = manifest
	e.Run.BestProfileID = prof.ID
	e.Run.Artifacts[core.StageQuantize] = out
	// Checkpoint before scoring so a crash never pairs the new artifact
	// with the old manifest.
	if err := e.Store.Save(e.Run); err != nil {
		return installedError{err}
	}
	e.printf("  refine: %s replaces %s (%d bytes planned)\n", prof.ID, prev, manifest.TotalBytes)
	// The replaced candidate is model-sized scratch nothing references now.
	if prevArtifact != "" && prevArtifact != out && generatedUnder(e.workDir(), prevArtifact) {
		os.Remove(prevArtifact)
	}
	art, err := e.evaluateProfile(ctx)
	if err != nil {
		return installedError{err}
	}
	if art != "" {
		e.Run.Artifacts[core.StageEvaluate] = art
	}
	return nil
}

// refineFinalProfile rebuilds the accepted profile from the recorded
// rounds: the last accepted round's re-solve.
func (e *Engine) refineFinalProfile(st *refineState) (*core.Profile, error) {
	solved, err := e.loadSolveArtifact()
	if err != nil {
		return nil, err
	}
	bank := e.Run.Bank
	set, err := e.deriveAnchors(bank)
	if err != nil {
		return nil, err
	}
	table := e.loadExactLoss(bank)
	cands := solved.Candidates
	if len(cands) == 0 {
		cands = e.candidateDTypes()
	}
	for i := len(st.Rounds) - 1; i >= 0; i-- {
		rr := st.Rounds[i]
		if !rr.Accepted || rr.Resolved != st.Final {
			continue
		}
		p, err := e.refineSolve(bank, set, solved.Sensitivity.Scaled(rr.Lambdas, rr.DefaultScale), table, cands)
		if err != nil {
			return nil, err
		}
		if p.ID != st.Final {
			return nil, fmt.Errorf("refine: re-solve gave %s, recorded %s", p.ID, st.Final)
		}
		return p, nil
	}
	return nil, fmt.Errorf("refine: no accepted round produced %s", st.Final)
}

// refineSolve re-runs the calibrated solver with a corrected model.
func (e *Engine) refineSolve(bank *core.TensorBank, set *anchor.Set, sens *profile.Sensitivity,
	table map[string]map[core.DType]float64, cands []core.DType) (*core.Profile, error) {
	var imatrix map[string]profile.ImatrixStats
	if e.imatrixPath() != "" {
		stats, err := e.loadSolverImatrix(bank)
		if err != nil {
			return nil, err
		}
		imatrix = stats
	}
	res, err := profile.Solve(profile.Request{
		Bank: bank, Anchors: set, Candidates: cands,
		BudgetBytes: e.solveBudget(), TargetBPW: e.Run.Config.TargetBPW,
		Imatrix: imatrix, ExactLoss: table, Sensitivity: sens,
		Calibration: e.loadCalibration(bank), PinUnprobed: e.Extra.NoPricedPins,
	})
	if err != nil {
		return nil, err
	}
	return res.Profile, nil
}

// ensureAnchorsFor materializes every (tensor, dtype) pair of manifest in
// the variants directory that no anchor shard covers yet.
func (e *Engine) ensureAnchorsFor(ctx context.Context, manifest *core.SelectionManifest) error {
	byDType := map[core.DType]map[string]struct{}{}
	for _, o := range manifest.Options {
		d := o.Target.BaseTensorType()
		if !d.IsQuant() || e.nativeOption(o) {
			continue
		}
		if byDType[d] == nil {
			byDType[d] = map[string]struct{}{}
		}
		byDType[d][o.TensorName] = struct{}{}
	}
	var dtypes []core.DType
	for d := range byDType {
		dtypes = append(dtypes, d)
	}
	sort.Slice(dtypes, func(i, j int) bool { return dtypes[i] < dtypes[j] })
	return e.runSparseTrimmedAnchorJobs(ctx, dtypes,
		func(d core.DType) (map[string]struct{}, error) { return copyKeep(byDType[d]), nil },
		e.variantsDir(), "meta.json")
}

// assembleRefineProfile builds a scratch GGUF for profile p from the
// variant anchors (quantizing any missing pair first).
func (e *Engine) assembleRefineProfile(ctx context.Context, p *core.Profile) (string, error) {
	manifest, err := e.manifestFor(p)
	if err != nil {
		return "", err
	}
	if err := e.ensureAnchorsFor(ctx, manifest); err != nil {
		return "", err
	}
	if err := os.MkdirAll(e.refineDir(), 0o755); err != nil {
		return "", err
	}
	srcs, closeSrcs, err := e.openAnchorSources()
	if err != nil {
		return "", err
	}
	out := filepath.Join(e.refineDir(), p.ID+".gguf")
	err = tensorbank.NewAssembler().Build(ctx, srcs, manifest, out, nil)
	closeSrcs()
	if err != nil {
		os.Remove(out)
		return "", err
	}
	return out, nil
}

// profileFromManifest rebuilds the profile a manifest freezes.
func profileFromManifest(m *core.SelectionManifest, bank *core.TensorBank) *core.Profile {
	elems := make(map[string]uint64, len(bank.Tensors))
	for _, t := range bank.Tensors {
		elems[t.Name] = t.Elements
	}
	p := &core.Profile{ID: m.ProfileID, BaseModel: bank.ModelID}
	for _, o := range m.Options {
		bpw := 0.0
		if n := elems[o.TensorName]; n > 0 {
			bpw = float64(o.Bytes) * 8 / float64(n)
		}
		p.Assignments = append(p.Assignments, core.QuantAssignment{TensorName: o.TensorName, Target: o.Target, BitsPerWeight: bpw})
	}
	return p
}

// profileIDFor derives the solver's deterministic profile ID for a set of
// assignments, so a re-solve identical to a measured variant reuses its
// measurement.
func profileIDFor(assignments []core.QuantAssignment) string {
	h := sha256.New()
	for _, qa := range assignments {
		fmt.Fprintf(h, "%s:%s\n", qa.TensorName, qa.Target)
	}
	return "prof-" + hex.EncodeToString(h.Sum(nil))[:16]
}

// refineRoles picks up to n calibrated roles holding the most bytes in p.
func refineRoles(p *core.Profile, bank *core.TensorBank, sens *profile.Sensitivity, n int) []string {
	if n <= 0 {
		n = 4
	}
	bytes := map[string]uint64{}
	for _, qa := range p.Assignments {
		if !qa.Target.IsQuant() {
			continue
		}
		role, ok := sens.RoleOf(qa.TensorName)
		if !ok {
			continue
		}
		t, ok := bank.Find(qa.TensorName)
		if !ok {
			continue
		}
		b, _ := qa.Target.ExactBytes(t.Elements)
		bytes[role] += b
	}
	roles := make([]string, 0, len(bytes))
	for r := range bytes {
		roles = append(roles, r)
	}
	sort.Slice(roles, func(i, j int) bool {
		if bytes[roles[i]] != bytes[roles[j]] {
			return bytes[roles[i]] > bytes[roles[j]]
		}
		return roles[i] < roles[j]
	})
	if len(roles) > n {
		roles = roles[:n]
	}
	return roles
}

// neighborRung returns tensor t's next rung from cur in direction dir among
// the legal candidates: up = the cheapest strictly larger rung with lower
// exact error; down = the most expensive strictly smaller rung. Floors and
// block geometry are honored; ok=false at either end of the ladder.
func neighborRung(t core.TensorDesc, cur core.DType, dir int, set *anchor.Set,
	row map[core.DType]float64, cands []core.DType) (core.DType, bool) {
	curB, ok := cur.ExactBytes(t.Elements)
	if !ok {
		return "", false
	}
	curW, hasCur := row[cur.BaseTensorType()]
	floor, hasFloor := set.Floor(t.Name)
	var best core.DType
	var bestB uint64
	for _, d := range cands {
		d = d.BaseTensorType()
		if !d.IsQuant() || d == cur.BaseTensorType() {
			continue
		}
		g, ok := d.Geometry()
		if !ok || len(t.Shape) == 0 || t.Shape[0]%g.BlockSize != 0 {
			continue
		}
		if hasFloor && anchor.Rank(d) > anchor.Rank(floor) {
			continue
		}
		w, ok := row[d]
		if !ok || math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
			continue
		}
		b, _ := d.ExactBytes(t.Elements)
		switch {
		case dir > 0 && b > curB:
			if hasCur && !(w < curW) {
				continue
			}
			if best == "" || b < bestB || (b == bestB && d < best) {
				best, bestB = d, b
			}
		case dir < 0 && b < curB:
			if best == "" || b > bestB || (b == bestB && d < best) {
				best, bestB = d, b
			}
		}
	}
	return best, best != ""
}

// moveRole builds the variant of p with every tensor of role moved one rung
// in dir. It returns the variant, the number of tensors moved, and the
// predicted |ΔKLD| of the move under sens (sensitivity units).
func moveRole(p *core.Profile, bank *core.TensorBank, set *anchor.Set, sens *profile.Sensitivity,
	table map[string]map[core.DType]float64, cands []core.DType, role string, dir int) (*core.Profile, int, float64) {
	v := &core.Profile{BaseModel: p.BaseModel, Assignments: make([]core.QuantAssignment, len(p.Assignments))}
	copy(v.Assignments, p.Assignments)
	moved := 0
	var pred float64
	for i, qa := range v.Assignments {
		if r, ok := sens.RoleOf(qa.TensorName); !ok || r != role || !qa.Target.IsQuant() {
			continue
		}
		t, ok := bank.Find(qa.TensorName)
		if !ok || set.Preserved(t) {
			continue
		}
		row := table[t.Name]
		next, ok := neighborRung(t, qa.Target, dir, set, row, cands)
		if !ok {
			continue
		}
		lc, ok1 := sens.Loss(t, qa.Target, row)
		ln, ok2 := sens.Loss(t, next, row)
		if !ok1 || !ok2 {
			continue
		}
		b, _ := next.ExactBytes(t.Elements)
		v.Assignments[i] = core.QuantAssignment{TensorName: t.Name, Target: next, BitsPerWeight: float64(b) * 8 / float64(t.Elements)}
		pred += math.Abs(lc - ln)
		moved++
	}
	v.ID = profileIDFor(v.Assignments)
	return v, moved, pred
}

func (e *Engine) printRefineLambdas(round int, lambdas map[string]float64, def float64) {
	roles := make([]string, 0, len(lambdas))
	for r := range lambdas {
		roles = append(roles, r)
	}
	sort.Strings(roles)
	e.printf("  refine round %d: in-context corrections (others x%.2f)\n", round, def)
	for _, r := range roles {
		e.printf("    %-18s x%.2f\n", r, lambdas[r])
	}
}
