package pipeline

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"quantlab/calibrate"
	"quantlab/core"
	"quantlab/state"
	"quantlab/tensorbank"
)

// PlanOptions is the full set of `quantlab plan` inputs. All paths are
// absolutized; validation errors are actionable (missing files, bad GGUF,
// missing budget, unusable corpora).
type PlanOptions struct {
	SourcePath      string
	OutputDir       string
	WorkDir         string
	StateDir        string
	BudgetBytes     uint64
	TargetBPW       float64
	LlamaQuantize   string
	LlamaPerplexity string
	LlamaImatrix    string
	ImatrixFile     string
	CalibrationDir  string
	Threads         int
	CtxSize         int
	Chunks          int
	Gates           []core.QualityGate
	// GatesOptOut is the explicit "-gates none": no gates, and no
	// effort-profile defaults.
	GatesOptOut bool
	// Effort names the preset ("fast", "profiled", "deep"; empty = profiled)
	// supplying defaults for every knob not explicitly set.
	Effort string
	// ChunksSet records that -chunks was explicitly passed, so it overrides
	// the effort preset (an explicit Chunks of 0 means unlimited).
	ChunksSet bool
	RunID     string
	Now       time.Time
	Stdout    io.Writer
	DryRun    bool
	// ExactEstimatorOff disables the solve-time exact loss table.
	ExactEstimatorOff     bool
	ScaleFold             bool
	NoScaleFold           bool
	Hadamard              bool
	NoHadamard            bool
	CSK                   bool
	NoCSK                 bool
	CSKMaxWorkingSetBytes uint64
	FTI                   bool
	NoFTI                 bool
	ProbeKLD              bool
	NoProbeKLD            bool
}

// Plan validates everything, builds calibration corpora when needed, derives
// the budget from -target-bpw when required, and writes the checkpoint. With
// DryRun it validates and prints the plan without writing the checkpoint.
func Plan(opts PlanOptions) (*state.Run, error) {
	if opts.Now.IsZero() {
		opts.Now = time.Now()
	}
	if opts.RunID == "" {
		opts.RunID = fmt.Sprintf("run-%d", opts.Now.Unix())
	}
	effort := Effort(opts.Effort)
	profile, err := EffortFor(effort)
	if err != nil {
		return nil, err
	}
	if effort == "" {
		effort = EffortProfiled
	}
	abs := func(name, p string) (string, error) {
		if p == "" {
			return "", fmt.Errorf("pipeline: -%s is required", name)
		}
		a, err := filepath.Abs(p)
		if err != nil {
			return "", fmt.Errorf("pipeline: resolve -%s path %q: %w", name, p, err)
		}
		return a, nil
	}
	if opts.SourcePath, err = abs("src", opts.SourcePath); err != nil {
		return nil, err
	}
	if opts.StateDir, err = abs("state-dir", opts.StateDir); err != nil {
		return nil, err
	}
	if opts.CalibrationDir, err = abs("calibration-dir", opts.CalibrationDir); err != nil {
		return nil, err
	}
	if opts.OutputDir == "" {
		opts.OutputDir = filepath.Dir(opts.SourcePath)
	}
	if opts.OutputDir, err = filepath.Abs(opts.OutputDir); err != nil {
		return nil, err
	}
	if opts.WorkDir == "" {
		opts.WorkDir = filepath.Join(opts.StateDir, opts.RunID+"-work")
	}
	if opts.WorkDir, err = filepath.Abs(opts.WorkDir); err != nil {
		return nil, err
	}
	if opts.LlamaQuantize == "" {
		return nil, fmt.Errorf("pipeline: -quantize binary path is required")
	}
	quantizePath, err := filepath.Abs(opts.LlamaQuantize)
	if err != nil {
		return nil, fmt.Errorf("pipeline: resolve -quantize: %w", err)
	}
	if _, err := os.Stat(quantizePath); err != nil {
		return nil, fmt.Errorf("pipeline: -quantize binary not found: %s", quantizePath)
	}
	opts.LlamaQuantize = quantizePath
	if opts.LlamaPerplexity == "" {
		return nil, fmt.Errorf("pipeline: -perplexity binary path is required")
	}
	perplexityPath, err := filepath.Abs(opts.LlamaPerplexity)
	if err != nil {
		return nil, fmt.Errorf("pipeline: resolve -perplexity: %w", err)
	}
	if _, err := os.Stat(perplexityPath); err != nil {
		return nil, fmt.Errorf("pipeline: -perplexity binary not found: %s", perplexityPath)
	}
	opts.LlamaPerplexity = perplexityPath
	if opts.LlamaImatrix != "" {
		if opts.LlamaImatrix, err = filepath.Abs(opts.LlamaImatrix); err != nil {
			return nil, err
		}
		if _, err := os.Stat(opts.LlamaImatrix); err != nil {
			return nil, fmt.Errorf("pipeline: -imatrix binary not found: %s", opts.LlamaImatrix)
		}
	}
	if opts.ImatrixFile != "" {
		if opts.ImatrixFile, err = filepath.Abs(opts.ImatrixFile); err != nil {
			return nil, err
		}
		if _, err := os.Stat(opts.ImatrixFile); err != nil {
			return nil, fmt.Errorf("pipeline: -imatrix-file not found: %s", opts.ImatrixFile)
		}
	}
	if st, err := os.Stat(opts.SourcePath); err != nil {
		return nil, fmt.Errorf("pipeline: -src not found: %s", opts.SourcePath)
	} else if st.IsDir() {
		return nil, fmt.Errorf("pipeline: -src is a directory, want a GGUF file: %s", opts.SourcePath)
	}

	// Validate the source parses as GGUF v2/v3 now, with an actionable error,
	// and derive the budget from -target-bpw when -budget-bytes is absent.
	elems, err := ggufElements(opts.SourcePath)
	if err != nil {
		return nil, err
	}
	budget := opts.BudgetBytes
	if budget == 0 {
		if opts.TargetBPW <= 0 {
			return nil, fmt.Errorf("pipeline: provide -budget-bytes or -target-bpw")
		}
		budget = uint64(opts.TargetBPW * float64(elems) / 8.0)
		if budget == 0 {
			return nil, fmt.Errorf("pipeline: -target-bpw %.3f yields a zero byte budget", opts.TargetBPW)
		}
	} else if opts.TargetBPW < 0 {
		return nil, fmt.Errorf("pipeline: -target-bpw must be >= 0")
	}

	threads := opts.Threads
	if threads == 0 {
		threads = 4
	}
	ctxSize := opts.CtxSize
	if ctxSize == 0 {
		ctxSize = 2048
		if profile.EvalCtx > 0 {
			ctxSize = profile.EvalCtx
		}
	}
	calibCorpus, searchCorpus, evalCorpus, err := prepareCorpora(opts.CalibrationDir, false)
	if err != nil {
		return nil, err
	}
	domainCorpora, err := domainEvalPaths(filepath.Dir(evalCorpus))
	if err != nil {
		return nil, err
	}
	chunks := opts.Chunks
	if !opts.ChunksSet {
		chunks = profile.EvalChunks
	}
	if chunks < 0 {
		return nil, fmt.Errorf("pipeline: -chunks must be >= 0")
	}
	cfg := state.RunConfig{
		SourcePath: opts.SourcePath,
		OutputDir:  opts.OutputDir,
		WorkDir:    opts.WorkDir,
		Tools: state.ToolPaths{
			LlamaQuantize:   opts.LlamaQuantize,
			LlamaPerplexity: opts.LlamaPerplexity,
			LlamaImatrix:    opts.LlamaImatrix,
		},
		ImatrixPath:         opts.ImatrixFile,
		CalibrationCorpus:   calibCorpus,
		SearchCorpus:        searchCorpus,
		EvalCorpus:          evalCorpus,
		DomainEvalCorpora:   domainCorpora,
		BudgetBytes:         budget,
		TargetBPW:           opts.TargetBPW,
		Threads:             threads,
		CtxSize:             ctxSize,
		Gates:               opts.Gates,
		GatesOptOut:         opts.GatesOptOut,
		Effort:              state.Effort(effort),
		SearchEnabled:       false,
		MaxSearchIterations: 0,
	}
	r, err := state.NewRun(opts.RunID, opts.Now, cfg)
	if err != nil {
		return nil, err
	}
	w := opts.Stdout
	if w == nil {
		w = os.Stdout
	}
	fmt.Fprintf(w, "run %s planned: src=%s budget=%d threads=%d ctx=%d effort=%s search=%v\n",
		r.RunID, cfg.SourcePath, cfg.BudgetBytes, cfg.Threads, cfg.CtxSize, cfg.Effort, cfg.SearchEnabled)
	if opts.DryRun {
		fmt.Fprintf(w, "dry-run: checkpoint not written (state dir %s)\n", opts.StateDir)
		return r, nil
	}
	store := state.Store{Dir: opts.StateDir}
	if err := store.Save(r); err != nil {
		return nil, err
	}
	if chunks > 0 || opts.CSKMaxWorkingSetBytes > 0 ||
		opts.ExactEstimatorOff || opts.ScaleFold || opts.NoScaleFold ||
		opts.Hadamard || opts.NoHadamard || opts.CSK || opts.NoCSK ||
		opts.FTI || opts.NoFTI || opts.ProbeKLD || opts.NoProbeKLD {
		e := &Engine{Store: store, Run: r, Extra: ExtraConfig{
			Chunks:                chunks,
			ExactEstimatorOff:     opts.ExactEstimatorOff,
			ScaleFold:             opts.ScaleFold,
			NoScaleFold:           opts.NoScaleFold,
			Hadamard:              opts.Hadamard,
			NoHadamard:            opts.NoHadamard,
			CSK:                   opts.CSK,
			NoCSK:                 opts.NoCSK,
			CSKMaxWorkingSetBytes: opts.CSKMaxWorkingSetBytes,
			FTI:                   opts.FTI,
			NoFTI:                 opts.NoFTI,
			ProbeKLD:              opts.ProbeKLD,
			NoProbeKLD:            opts.NoProbeKLD,
		}}
		if err := e.saveExtra(); err != nil {
			return nil, fmt.Errorf("pipeline: write extra config: %w", err)
		}
	}
	return r, nil
}

// ggufElements parses the GGUF metadata (no hashing) and sums tensor element
// counts, for budget derivation from a bits-per-weight target.
func ggufElements(path string) (uint64, error) {
	s, err := tensorbank.OpenSource(path)
	if err != nil {
		return 0, fmt.Errorf("pipeline: -src: %w", err)
	}
	defer s.Close()
	f, err := tensorbank.Parse(s)
	if err != nil {
		return 0, fmt.Errorf("pipeline: -src is not a usable GGUF v2/v3 file: %w", err)
	}
	var elems uint64
	for _, t := range f.Tensors {
		elems += t.Elements
	}
	return elems, nil
}

// prepareCorpora resolves the calibration directory: an existing
// calibrate.Manifest plus corpora is used verbatim; otherwise the directory's
// .txt files are built into calibration/evaluation corpora via calibrate.
func prepareCorpora(dir string, needSearch bool) (calib, search, eval string, err error) {
	if _, statErr := os.Stat(filepath.Join(dir, "manifest.json")); statErr == nil {
		calib = filepath.Join(dir, "calibration.txt")
		searchPath := filepath.Join(dir, "search.txt")
		eval = filepath.Join(dir, "evaluation.txt")
		for name, p := range map[string]string{"calibration.txt": calib, "evaluation.txt": eval} {
			if _, err := os.Stat(p); err != nil {
				return "", "", "", fmt.Errorf("pipeline: calibration dir manifest exists but %s is missing", name)
			}
		}
		if st, err := os.Stat(searchPath); err == nil && !st.IsDir() && st.Size() > 0 {
			search = searchPath
		}
		return calib, search, eval, nil
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return "", "", "", fmt.Errorf("pipeline: -calibration-dir: %w", err)
	}
	var texts []string
	for _, ent := range ents {
		if ent.IsDir() || filepath.Ext(ent.Name()) != ".txt" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, ent.Name()))
		if err != nil {
			return "", "", "", err
		}
		if len(data) > 0 {
			texts = append(texts, splitCorpusText(string(data))...)
		}
	}
	if len(texts) == 0 {
		return "", "", "", fmt.Errorf("pipeline: -calibration-dir %s has no manifest.json and no .txt source files", dir)
	}
	cfg := &calibrate.Config{
		Domains: []calibrate.DomainSpec{
			{Source: calibrate.SliceSource(calibrate.DomainGeneral, texts)},
		},
		Seed:           42,
		CalibPercent:   80,
		MaxRecordBytes: 32 << 20,
		MinRecords:     1,
		MinTokens:      1,
	}
	if needSearch {
		cfg.SearchPercent = 10
	}
	if _, _, err := calibrate.Build(context.Background(), dir, cfg); err != nil {
		return "", "", "", fmt.Errorf("pipeline: building calibration corpora: %w", err)
	}
	search = filepath.Join(dir, "search.txt")
	if st, err := os.Stat(search); err != nil || st.IsDir() || st.Size() == 0 {
		search = ""
	}
	return filepath.Join(dir, "calibration.txt"), search, filepath.Join(dir, "evaluation.txt"), nil
}

func splitCorpusText(text string) []string {
	var records []string
	for _, paragraph := range strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n\n") {
		if paragraph = strings.TrimSpace(paragraph); paragraph != "" {
			records = append(records, paragraph)
		}
	}
	if len(records) >= 3 {
		return records
	}
	fields := strings.Fields(text)
	if len(fields) >= 3 {
		parts := len(fields) / 128
		if parts < 3 {
			parts = 3
		}
		if parts > 32 {
			parts = 32
		}
		records = records[:0]
		for i := 0; i < parts; i++ {
			start := i * len(fields) / parts
			end := (i + 1) * len(fields) / parts
			if start < end {
				records = append(records, strings.Join(fields[start:end], " "))
			}
		}
		return records
	}
	return records
}
