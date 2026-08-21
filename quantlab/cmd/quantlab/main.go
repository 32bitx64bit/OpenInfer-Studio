// quantlab is the CLI entrypoint for the surgical quantization optimizer.
// It owns argument parsing and delegates all orchestration to the pipeline
// package; library packages own the logic.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"time"

	"quantlab/pipeline"
	"quantlab/state"
)

const usage = `quantlab - surgical GGUF quantization optimizer

usage:
  quantlab plan    -src <model.gguf> -state-dir <dir> -calibration-dir <dir>
                   -quantize <bin> -perplexity <bin>
                   [-out <dir>] [-work <dir>] [-budget-bytes <n> | -target-bpw <f>]
                   [-imatrix <bin>] [-imatrix-file <file>]
                   [-threads <n>] [-ctx <n>] [-chunks <n>]
                   [-effort fast|profiled|deep]
                   [-gates mean-kld=<f>[,p95-kld=<f>] | -gates none]
                   [-run <id>] [-dry-run]
                   [-scale-fold] [-no-scale-fold] [-hadamard] [-no-hadamard]
                   [-csk] [-no-csk] [-fti] [-no-fti] [-probe-kld] [-no-probe-kld]
  quantlab resume  -state-dir <dir> -run <id> [-stage-limit <n>] [-dry-run]
  quantlab status  -state-dir <dir> -run <id>
`

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "quantlab:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout io.Writer) error {
	if len(args) == 0 {
		return fmt.Errorf("missing command\n%s", usage)
	}
	switch args[0] {
	case "plan":
		return cmdPlan(args[1:], stdout)
	case "resume":
		return cmdResume(args[1:], stdout)
	case "status":
		return cmdStatus(args[1:], stdout)
	case "help", "-h", "--help":
		fmt.Fprint(stdout, usage)
		return nil
	default:
		return fmt.Errorf("unknown command %q\n%s", args[0], usage)
	}
}

func cmdPlan(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("plan", flag.ContinueOnError)
	src := fs.String("src", "", "source GGUF model path")
	out := fs.String("out", "", "output directory for final artifacts (default: beside source)")
	work := fs.String("work", "", "work directory for intermediates (default: <state-dir>/<run>-work)")
	stateDir := fs.String("state-dir", "", "checkpoint directory")
	budget := fs.Uint64("budget-bytes", 0, "hard size budget in bytes for the complete final GGUF artifact (header, metadata, and padding included)")
	targetBPW := fs.Float64("target-bpw", 0, "target average bits-per-weight (alternative to -budget-bytes)")
	quantize := fs.String("quantize", "", "llama-quantize binary path")
	perplexity := fs.String("perplexity", "", "llama-perplexity binary path")
	imatrixBin := fs.String("imatrix", "", "llama-imatrix binary path")
	imatrixFile := fs.String("imatrix-file", "", "precomputed importance matrix file")
	calibrationDir := fs.String("calibration-dir", "", "corpus directory: existing manifest+corpora, or .txt sources to build from")
	threads := fs.Int("threads", 0, "worker threads (default 4)")
	ctxSize := fs.Int("ctx", 0, "evaluation context size (default 2048)")
	chunks := fs.Int("chunks", 0, "evaluation chunk limit (default: effort preset; explicit 0 = unlimited)")
	gatesSpec := fs.String("gates", "", "quality gates: mean-kld=<f>,p95-kld=<f>, or 'none' to opt out (default: scaled to -target-bpw; Q5 0.15/1.0 when unset)")
	effort := fs.String("effort", "", "effort preset: fast, profiled (default), or deep")
	noExact := fs.Bool("no-exact", false, "disable the solve-time exact loss table")
	scaleFold := fs.Bool("scale-fold", false, "force AWQ-style scale folding")
	noScaleFold := fs.Bool("no-scale-fold", false, "disable scale folding")
	hadamard := fs.Bool("hadamard", false, "force residual Hadamard")
	noHadamard := fs.Bool("no-hadamard", false, "disable residual Hadamard")
	csk := fs.Bool("csk", false, "force SwiGLU down-proj compensation")
	noCSK := fs.Bool("no-csk", false, "disable SwiGLU down-proj compensation")
	fti := fs.Bool("fti", false, "write a sharpened imatrix GGUF (opt-in; profiled/deep already sharpen in memory for the solver)")
	noFTI := fs.Bool("no-fti", false, "disable imatrix sharpening")
	probeKLD := fs.Bool("probe-kld", false, "force probe-KLD in the exact loss table (default-on for profiled/deep)")
	noProbeKLD := fs.Bool("no-probe-kld", false, "disable probe-KLD in the exact loss table")
	runID := fs.String("run", "", "run id (default: run-<unixtime>)")
	dryRun := fs.Bool("dry-run", false, "validate and plan without writing artifacts")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected argument %q\n%s", fs.Arg(0), usage)
	}
	setFlags := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { setFlags[f.Name] = true })
	gatesOptOut := false
	if strings.TrimSpace(*gatesSpec) == "none" {
		gatesOptOut = true
		*gatesSpec = ""
	}
	gates, err := pipeline.ParseGates(*gatesSpec)
	if err != nil {
		return err
	}
	_, err = pipeline.Plan(pipeline.PlanOptions{
		SourcePath:        *src,
		OutputDir:         *out,
		WorkDir:           *work,
		StateDir:          *stateDir,
		BudgetBytes:       *budget,
		TargetBPW:         *targetBPW,
		LlamaQuantize:     *quantize,
		LlamaPerplexity:   *perplexity,
		LlamaImatrix:      *imatrixBin,
		ImatrixFile:       *imatrixFile,
		CalibrationDir:    *calibrationDir,
		Threads:           *threads,
		CtxSize:           *ctxSize,
		Chunks:            *chunks,
		Gates:             gates,
		GatesOptOut:       gatesOptOut,
		Effort:            *effort,
		ChunksSet:         setFlags["chunks"],
		RunID:             *runID,
		Now:               time.Now(),
		Stdout:            stdout,
		DryRun:            *dryRun,
		ExactEstimatorOff: *noExact,
		ScaleFold:         *scaleFold,
		NoScaleFold:       *noScaleFold,
		Hadamard:          *hadamard,
		NoHadamard:        *noHadamard,
		CSK:               *csk,
		NoCSK:             *noCSK,
		FTI:               *fti,
		NoFTI:             *noFTI,
		ProbeKLD:          *probeKLD,
		NoProbeKLD:        *noProbeKLD,
	})
	return err
}

func cmdResume(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("resume", flag.ContinueOnError)
	stateDir := fs.String("state-dir", "", "checkpoint directory")
	runID := fs.String("run", "", "run id")
	stageLimit := fs.Int("stage-limit", 0, "execute at most N stages (0 = all remaining)")
	dryRun := fs.Bool("dry-run", false, "plan remaining stages without executing tools or writing artifacts")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected argument %q\n%s", fs.Arg(0), usage)
	}
	if *stateDir == "" || *runID == "" {
		return fmt.Errorf("resume requires -state-dir and -run\n%s", usage)
	}
	store := state.Store{Dir: *stateDir}
	r, err := store.Load(*runID)
	if err != nil {
		return err
	}
	eng, err := pipeline.NewEngine(store, r, pipeline.OSRunner(), stdout)
	if err != nil {
		return err
	}
	eng.DryRun = *dryRun
	eng.StageLimit = *stageLimit
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	return eng.Resume(ctx)
}

func cmdStatus(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("status", flag.ContinueOnError)
	stateDir := fs.String("state-dir", "", "checkpoint directory")
	runID := fs.String("run", "", "run id")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected argument %q\n%s", fs.Arg(0), usage)
	}
	if *stateDir == "" || *runID == "" {
		return fmt.Errorf("status requires -state-dir and -run\n%s", usage)
	}
	r, err := (state.Store{Dir: *stateDir}).Load(*runID)
	if err != nil {
		return err
	}
	pipeline.PrintStatus(stdout, r)
	return nil
}
