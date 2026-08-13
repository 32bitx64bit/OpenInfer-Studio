package quantize

import (
	"fmt"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

// Request is the JSON body for POST /quantize/jobs. Unknown fields are
// rejected by the API decoder.
type Request struct {
	Kind                    string   `json:"kind"`
	RuntimeID               string   `json:"runtime_id"`
	SourceModelID           string   `json:"source_model_id"`
	FType                   string   `json:"ftype"`
	OutputName              string   `json:"output_name"`
	Threads                 int      `json:"threads"`
	AllowRequantize         bool     `json:"allow_requantize"`
	LeaveOutputTensor       bool     `json:"leave_output_tensor"`
	Pure                    bool     `json:"pure"`
	KeepSplit               bool     `json:"keep_split"`
	OutputTensorType        string   `json:"output_tensor_type"`
	TokenEmbeddingType      string   `json:"token_embedding_type"`
	TensorTypes             []string `json:"tensor_types"`
	TensorTypeFile          string   `json:"tensor_type_file"`
	IMatrixID               string   `json:"imatrix_id"`
	GenerateIMatrix         bool     `json:"generate_imatrix"`
	CalibrationPath         string   `json:"calibration_path"`
	CalibrationPreset       string   `json:"calibration_preset"`
	Chunks                  int      `json:"chunks"`
	ChunkSkip               int      `json:"chunk_skip"`
	GPULayers               int      `json:"gpu_layers"`
	ParseSpecial            bool     `json:"parse_special"`
	ProcessOutput           bool     `json:"process_output"`
	CombineIMatrixIDs       []string `json:"combine_imatrix_ids"`
	DeleteIntermediates     bool     `json:"delete_intermediates"`
	KeepIMatrix             bool     `json:"keep_imatrix"`
	QuantizeProjector       bool     `json:"quantize_projector"`
	ProjectorFType          string   `json:"projector_ftype"`
	CopyProjector           bool     `json:"copy_projector"`
	DraftModelID            string   `json:"draft_model_id"`
	QuantizeDraft           bool     `json:"quantize_draft"`
	DraftFType              string   `json:"draft_ftype"`
	AdaptivePreset          string   `json:"adaptive_preset"`
	TargetBPW               float64  `json:"target_bpw"`
	TargetBytes             int64    `json:"target_bytes"`
	AcknowledgeRequantize   bool     `json:"acknowledge_requantize"`
	AcknowledgeExperimental bool     `json:"acknowledge_experimental"`
	UnloadFirst             bool     `json:"unload_first"`
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
	ftype := CanonicalFType(req.FType)
	if ftype == "" {
		return nil, fmt.Errorf("missing quantization type")
	}
	var args []string
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
			return nil, fmt.Errorf("this runtime's llama-quantize does not advertise --tensor-type-file")
		}
	}
	if req.KeepSplit {
		addFlag(&args, flags, "--keep-split")
	}
	args = append(args, inPath, outPath, ftype)
	if req.Threads > 0 {
		args = append(args, fmt.Sprintf("%d", req.Threads))
	}
	return args, nil
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
		addFlagVal(&args, flags, "--chunks", fmt.Sprintf("%d", chunks))
	}
	if req.ChunkSkip > 0 {
		addFlagVal(&args, flags, "--chunk", fmt.Sprintf("%d", req.ChunkSkip), "--from-chunk")
	}
	if req.ParseSpecial {
		addFlag(&args, flags, "--parse-special")
	}
	if req.ProcessOutput {
		addFlag(&args, flags, "--process-output")
	}
	addFlag(&args, flags, "--no-ppl")
	if req.Threads > 0 {
		addFlagVal(&args, flags, "--threads", fmt.Sprintf("%d", req.Threads), "-t")
	}
	return args, nil
}

// PlanCombine builds llama-imatrix argv that merges existing matrices.
func PlanCombine(flags []string, inputs []string, outPath string) ([]string, error) {
	if len(inputs) < 2 {
		return nil, fmt.Errorf("combining imatrices requires at least two input files")
	}
	if !flagOK(flags, "--in-file") {
		return nil, fmt.Errorf("this runtime's llama-imatrix does not advertise --in-file")
	}
	var args []string
	for _, in := range inputs {
		args = append(args, "--in-file", in)
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
		return 250
	default:
		return 100
	}
}
