package quantize

import (
	"bytes"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

// Request is the JSON body for POST /quantize/jobs. Unknown fields are
// rejected by the API decoder.
type Request struct {
	Kind               string   `json:"kind"`
	RuntimeID          string   `json:"runtime_id"`
	SourceModelID      string   `json:"source_model_id"`
	FType              string   `json:"ftype"`
	OutputName         string   `json:"output_name"`
	Threads            int      `json:"threads"`
	AllowRequantize    bool     `json:"allow_requantize"`
	LeaveOutputTensor  bool     `json:"leave_output_tensor"`
	Pure               bool     `json:"pure"`
	KeepSplit          bool     `json:"keep_split"`
	OutputTensorType   string   `json:"output_tensor_type"`
	TokenEmbeddingType string   `json:"token_embedding_type"`
	TensorTypes        []string `json:"tensor_types"`
	TensorTypeFile     string   `json:"tensor_type_file"`
	IMatrixID          string   `json:"imatrix_id"`
	GenerateIMatrix    bool     `json:"generate_imatrix"`
	CalibrationPath    string   `json:"calibration_path"`
	CalibrationPreset  string   `json:"calibration_preset"`
	Chunks             int      `json:"chunks"`
	ChunkSkip          int      `json:"chunk_skip"`
	// IMatrixInFile is a previous llama-imatrix GGUF to continue from.
	// Not part of the public request body.
	IMatrixInFile       string   `json:"-"`
	GPULayers           int      `json:"gpu_layers"`
	ParseSpecial        bool     `json:"parse_special"`
	ProcessOutput       bool     `json:"process_output"`
	CombineIMatrixIDs   []string `json:"combine_imatrix_ids"`
	DeleteIntermediates bool     `json:"delete_intermediates"`
	KeepIMatrix         bool     `json:"keep_imatrix"`
	QuantizeProjector   bool     `json:"quantize_projector"`
	ProjectorFType      string   `json:"projector_ftype"`
	CopyProjector       bool     `json:"copy_projector"`
	DraftModelID        string   `json:"draft_model_id"`
	QuantizeDraft       bool     `json:"quantize_draft"`
	DraftFType          string   `json:"draft_ftype"`
	AdaptivePreset      string   `json:"adaptive_preset"`
	AdaptiveMode        string   `json:"adaptive_mode"`
	// Effort selects the quantlab search/evaluation effort for adaptive jobs:
	// "fast", "profiled", or "deep" (empty means profiled). AdaptiveMode is a
	// deprecated alias used when Effort is empty. AdaptivePreset is
	// deprecated: it translates to a TargetBPW heuristic (quality 6.0,
	// balanced 4.5, compact 3.8, aggressive 3.0) only when neither TargetBPW
	// nor TargetBytes is set. QuantTier selects the OpenInfer Dynamic
	// compression tier ("q5", "q4" default, "q3", "q2", "custom"): a named
	// tier overrides target_bpw; "custom" targets target_bytes; empty defers
	// to target_bpw/target_bytes for backward compatibility.
	Effort                  string   `json:"effort"`
	TargetBPW               float64  `json:"target_bpw"`
	TargetBytes             int64    `json:"target_bytes"`
	QuantTier               string   `json:"quant_tier"`
	PriorWeight             *float64 `json:"prior_weight"`
	AcknowledgeRequantize   bool     `json:"acknowledge_requantize"`
	AcknowledgeExperimental bool     `json:"acknowledge_experimental"`
	// AllowFailedQualityGates is ignored. Quality-check misses are warnings;
	// a finished GGUF is always published. Kept so older clients still decode.
	AllowFailedQualityGates bool   `json:"allow_failed_quality_gates"`
	UnloadFirst             bool   `json:"unload_first"`
	HFRepo                  string `json:"hf_repo"`
}

const (
	KindQuantize         = "quantize"
	KindIMatrix          = "imatrix"
	KindCombineIMatrix   = "combine_imatrix"
	KindAdaptiveQuantize = "adaptive_quantize"
)

func normalizeKind(k string) string {
	k = strings.TrimSpace(k)
	if k == "" {
		return KindQuantize
	}
	return k
}

func flagOK(flags []string, long string, shorts ...string) bool {
	if runtimes.ToolHasFlag(flags, long) {
		return true
	}
	for _, s := range shorts {
		if runtimes.ToolHasFlag(flags, s) {
			return true
		}
	}
	return false
}

func addFlag(args *[]string, flags []string, long string, shorts ...string) bool {
	if flagOK(flags, long, shorts...) {
		*args = append(*args, long)
		return true
	}
	return false
}

func addFlagVal(args *[]string, flags []string, long, value string, shorts ...string) bool {
	if value == "" {
		return false
	}
	if flagOK(flags, long, shorts...) {
		*args = append(*args, long, value)
		return true
	}
	return false
}

// PlanQuantize builds llama-quantize argv (excluding the executable).
func PlanQuantize(req Request, flags []string, inPath, outPath, imatrixPath, tensorFile string) ([]string, error) {
	return planQuantize(req, flags, inPath, outPath, imatrixPath, tensorFile, false)
}

// PlanQuantizeDryRun builds argv for an exact output-size calculation. The
// dry-run syntax intentionally omits the output path for compatibility with
// llama.cpp versions that implement this option.
func PlanQuantizeDryRun(req Request, flags []string, inPath, imatrixPath, tensorFile string) ([]string, error) {
	if !flagOK(flags, "--dry-run") {
		return nil, fmt.Errorf("this runtime's llama-quantize does not advertise --dry-run")
	}
	return planQuantize(req, flags, inPath, "", imatrixPath, tensorFile, true)
}

func planQuantize(req Request, flags []string, inPath, outPath, imatrixPath, tensorFile string, dryRun bool) ([]string, error) {
	ftype := CanonicalFType(req.FType)
	if ftype == "" {
		return nil, fmt.Errorf("missing quantization type")
	}
	var args []string
	if dryRun {
		args = append(args, "--dry-run")
	}
	if req.AllowRequantize {
		addFlag(&args, flags, "--allow-requantize")
	}
	if req.LeaveOutputTensor {
		addFlag(&args, flags, "--leave-output-tensor")
	}
	if req.Pure {
		addFlag(&args, flags, "--pure")
	}
	if imatrixPath != "" {
		if !addFlagVal(&args, flags, "--imatrix", imatrixPath) {
			return nil, fmt.Errorf("this runtime's llama-quantize does not advertise --imatrix")
		}
	}
	if req.PriorWeight != nil {
		if *req.PriorWeight < 0 {
			return nil, fmt.Errorf("prior_weight must be non-negative")
		}
		addFlagVal(&args, flags, "--prior-weight", strconv.FormatFloat(*req.PriorWeight, 'g', -1, 64))
	}
	addFlagVal(&args, flags, "--output-tensor-type", strings.ToLower(req.OutputTensorType))
	addFlagVal(&args, flags, "--token-embedding-type", strings.ToLower(req.TokenEmbeddingType))
	for _, tt := range req.TensorTypes {
		tt = strings.TrimSpace(tt)
		if tt == "" {
			continue
		}
		if !addFlagVal(&args, flags, "--tensor-type", tt) {
			return nil, fmt.Errorf("this runtime's llama-quantize does not advertise --tensor-type")
		}
	}
	if tensorFile != "" {
		if !addFlagVal(&args, flags, "--tensor-type-file", tensorFile) {
			if !flagOK(flags, "--tensor-type") {
				return nil, fmt.Errorf("this runtime's llama-quantize advertises neither --tensor-type-file nor --tensor-type")
			}
			entries, err := readTensorTypeEntries(tensorFile)
			if err != nil {
				return nil, err
			}
			for _, entry := range entries {
				args = append(args, "--tensor-type", entry)
			}
		}
	}
	if req.KeepSplit {
		addFlag(&args, flags, "--keep-split")
	}
	args = append(args, inPath)
	if !dryRun {
		args = append(args, outPath)
	}
	args = append(args, ftype)
	if req.Threads > 0 {
		args = append(args, fmt.Sprintf("%d", req.Threads))
	}
	return args, nil
}

func readTensorTypeEntries(path string) ([]string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("read tensor type file: %w", err)
	}
	if info.IsDir() || info.Size() > 8<<20 {
		return nil, fmt.Errorf("tensor type file must be a regular file no larger than 8 MiB")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tensor type file: %w", err)
	}
	var entries []string
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if !strings.Contains(line, "=") {
			return nil, fmt.Errorf("invalid tensor type entry %q", line)
		}
		entries = append(entries, line)
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("tensor type file is empty")
	}
	return entries, nil
}

// PlanIMatrix builds llama-imatrix argv for generating a matrix from a model.
func PlanIMatrix(req Request, flags []string, modelPath, calPath, outPath string) ([]string, error) {
	if modelPath == "" || calPath == "" {
		return nil, fmt.Errorf("imatrix requires a model and calibration file")
	}
	var args []string
	if !addFlagVal(&args, flags, "--model", modelPath, "-m") {
		// Fall back to short -m even when help parsing missed it: every
		// llama-imatrix build uses -m. Still prefer advertised names.
		if flagOK(flags, "-m") {
			args = append(args, "-m", modelPath)
		} else {
			args = append(args, "-m", modelPath)
		}
	}
	if !addFlagVal(&args, flags, "--file", calPath, "-f") {
		args = append(args, "-f", calPath)
	}
	if !addFlagVal(&args, flags, "--output-file", outPath, "-o") {
		args = append(args, "-o", outPath)
	}
	if req.GPULayers > 0 {
		addFlagVal(&args, flags, "--n-gpu-layers", fmt.Sprintf("%d", req.GPULayers), "-ngl")
	}
	chunks := req.Chunks
	if chunks == 0 {
		chunks = presetChunks(req.CalibrationPreset)
	}
	if chunks > 0 {
		ctx := 512
		if wantIMatrixCtx(req) {
			ctx = imatrixCtxForFile(calPath, 4096)
			if ctx < 512 {
				ctx = 512
			}
		}
		chunks = clampIMatrixChunks(calPath, ctx, chunks)
		addFlagVal(&args, flags, "--chunks", fmt.Sprintf("%d", chunks))
	}
	if req.ChunkSkip > 0 {
		addFlagVal(&args, flags, "--chunk", fmt.Sprintf("%d", req.ChunkSkip), "--from-chunk")
	}
	if in := strings.TrimSpace(req.IMatrixInFile); in != "" {
		if !addFlagVal(&args, flags, "--in-file", in) {
			return nil, fmt.Errorf("partial imatrix exists but this runtime's llama-imatrix does not advertise --in-file")
		}
	}
	if req.ParseSpecial {
		addFlag(&args, flags, "--parse-special")
	}
	if req.ProcessOutput {
		addFlag(&args, flags, "--process-output")
	}
	if wantIMatrixCtx(req) {
		ctx := imatrixCtxForFile(calPath, 4096)
		if ctx > 0 {
			addFlagVal(&args, flags, "--ctx-size", strconv.Itoa(ctx), "-c")
		}
	}
	addFlag(&args, flags, "--no-ppl")
	if req.Threads > 0 {
		addFlagVal(&args, flags, "--threads", fmt.Sprintf("%d", req.Threads), "-t")
	}
	return args, nil
}

// PlanCombine builds llama-imatrix argv that merges existing matrices.
// modelPath is required by recent llama.cpp argparse even though combine
// returns before loading weights. Pass the source GGUF, not an imatrix.
func PlanCombine(flags []string, inputs []string, outPath, modelPath string) ([]string, error) {
	if len(inputs) < 2 {
		return nil, fmt.Errorf("combining imatrices requires at least two input files")
	}
	if !flagOK(flags, "--in-file") {
		return nil, fmt.Errorf("this runtime's llama-imatrix does not advertise --in-file")
	}
	if strings.TrimSpace(modelPath) == "" {
		return nil, fmt.Errorf("combining imatrices requires a source model path")
	}
	var args []string
	// llama.cpp common args treat a repeated flag as "keep last" and want
	// comma-separated values instead.
	args = append(args, "--in-file", strings.Join(inputs, ","))
	if !addFlagVal(&args, flags, "--model", modelPath, "-m") {
		args = append(args, "-m", modelPath)
	}
	if !addFlagVal(&args, flags, "--output-file", outPath, "-o") {
		args = append(args, "-o", outPath)
	}
	return args, nil
}

func presetChunks(preset string) int {
	switch strings.ToLower(strings.TrimSpace(preset)) {
	case "quick":
		return 20
	case "thorough":
		return 500
	case "research":
		return 750
	default:
		return 200
	}
}

func wantIMatrixCtx(req Request) bool {
	if usesQuantlab(req) {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(req.CalibrationPreset)) {
	case "thorough", "research":
		return true
	default:
		return false
	}
}

// imatrixCtxForFile picks a llama-imatrix --ctx-size that the calibration
// file can actually fill. Prefer a context that still yields several unique
// windows: looping two 4096-token windows 750 times overfits the matrix.
// llama.cpp refuses to start unless the file tokenizes to at least 2*ctx
// tokens (4096 ctx needs 8192).
func imatrixCtxForFile(path string, want int) int {
	if want <= 0 {
		want = 4096
	}
	tokens := fileTokenEstimate(path)
	if tokens <= 0 {
		return want
	}
	pick := func(minWindows int) int {
		for _, c := range []int{4096, 2048, 1024, 512} {
			if c <= want && tokens >= minWindows*c {
				return c
			}
		}
		return 0
	}
	if c := pick(4); c > 0 {
		return c
	}
	if c := pick(2); c > 0 {
		return c
	}
	return want
}

func fileTokenEstimate(path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	b = bytes.TrimSpace(b)
	// ASCII prose/code averages above four bytes per token. Count non-ASCII
	// runes separately because CJK commonly approaches one token per rune;
	// dividing UTF-8 bytes by five would dangerously overestimate coverage.
	ascii, nonASCII := 0, 0
	for len(b) > 0 {
		r, n := utf8.DecodeRune(b)
		if r == utf8.RuneError && n == 1 {
			ascii++
		} else if r < utf8.RuneSelf {
			ascii++
		} else {
			nonASCII++
		}
		b = b[n:]
	}
	return ascii/5 + nonASCII
}

// clampIMatrixChunks caps --chunks so a short file is not replayed dozens
// of times. A few wraps help alignment; 40× the unique windows overfits.
func clampIMatrixChunks(path string, ctx, want int) int {
	if want <= 0 {
		return want
	}
	tokens := fileTokenEstimate(path)
	if tokens <= 0 {
		return want
	}
	if ctx < 1 {
		ctx = 512
	}
	unique := tokens / ctx
	if unique < 2 {
		unique = 2
	}
	max := unique * 4
	if want > max {
		return max
	}
	return want
}
