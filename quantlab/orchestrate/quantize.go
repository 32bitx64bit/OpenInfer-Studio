package orchestrate

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"sort"
	"strings"

	"quantlab/core"
)

// QuantizeRequest fully describes one llama-quantize execution. It supports
// both uniform recipes (Type only) and surgical mixed quantization (Type as
// the base plus a tensor-type file carrying per-tensor overrides).
type QuantizeRequest struct {
	ProfileID  string     `json:"profileID"`
	SourcePath string     `json:"sourcePath"`
	OutputPath string     `json:"outputPath"`
	Type       core.DType `json:"type"` // recipe label or per-tensor base type
	// TensorTypeFile, when set, is a "name TYPE" file materialized via
	// WriteTensorTypeFile and passed as --tensor-type.
	TensorTypeFile string `json:"tensorTypeFile,omitempty"`
	// ImatrixPath is required when Type.RequiresImatrix().
	ImatrixPath string `json:"imatrixPath,omitempty"`
	// OutputType / EmbeddingType override the output-head and token-embedding
	// tensor storage types (float types only).
	OutputType    core.DType `json:"outputType,omitempty"`
	EmbeddingType core.DType `json:"embeddingType,omitempty"`
	// Pure quantizes all output-class tensors too (no default float keep).
	Pure bool `json:"pure,omitempty"`
	// KeepSplit emits split GGUFs matching the source layout.
	KeepSplit bool `json:"keepSplit,omitempty"`
	// DryRun prints the planned per-tensor types without writing output.
	DryRun bool `json:"dryRun,omitempty"`
	// Threads is the positional nthread argument; <= 0 omits it.
	Threads int `json:"threads,omitempty"`
	// SourceQuantized records that the source GGUF already holds quantized
	// tensors. Requantization is refused unless AllowRequantize is set.
	SourceQuantized bool `json:"sourceQuantized,omitempty"`
	AllowRequantize bool `json:"allowRequantize,omitempty"`
}

// flagSpec binds one optional request feature to the llama-quantize flag
// that enables it, so the planner can gate emission on probed capabilities.
type quantFlag struct {
	flag string
	when func(QuantizeRequest) bool
	emit func(QuantizeRequest) ([]string, error)
}

func quantFlagSpecs() []quantFlag {
	return []quantFlag{
		{"--imatrix",
			func(r QuantizeRequest) bool { return r.ImatrixPath != "" },
			func(r QuantizeRequest) ([]string, error) { return []string{"--imatrix", r.ImatrixPath}, nil }},
		{"--tensor-type",
			func(r QuantizeRequest) bool { return r.TensorTypeFile != "" },
			func(r QuantizeRequest) ([]string, error) { return []string{"--tensor-type", r.TensorTypeFile}, nil }},
		{"--output-tensor-type",
			func(r QuantizeRequest) bool { return r.OutputType != "" },
			func(r QuantizeRequest) ([]string, error) {
				if !r.OutputType.IsFloat() {
					return nil, fmt.Errorf("orchestrate: output tensor type %q is not a float dtype", r.OutputType)
				}
				return []string{"--output-tensor-type", string(r.OutputType)}, nil
			}},
		{"--token-embedding-type",
			func(r QuantizeRequest) bool { return r.EmbeddingType != "" },
			func(r QuantizeRequest) ([]string, error) {
				if !r.EmbeddingType.IsFloat() {
					return nil, fmt.Errorf("orchestrate: embedding type %q is not a float dtype", r.EmbeddingType)
				}
				return []string{"--token-embedding-type", string(r.EmbeddingType)}, nil
			}},
		{"--pure",
			func(r QuantizeRequest) bool { return r.Pure },
			func(r QuantizeRequest) ([]string, error) { return []string{"--pure"}, nil }},
		{"--keep-split",
			func(r QuantizeRequest) bool { return r.KeepSplit },
			func(r QuantizeRequest) ([]string, error) { return []string{"--keep-split"}, nil }},
		{"--dry-run",
			func(r QuantizeRequest) bool { return r.DryRun },
			func(r QuantizeRequest) ([]string, error) { return []string{"--dry-run"}, nil }},
	}
}

func (r QuantizeRequest) Validate() error {
	if r.ProfileID == "" || r.SourcePath == "" || r.OutputPath == "" {
		return fmt.Errorf("orchestrate: quantize request missing id/paths")
	}
	if r.SourcePath == r.OutputPath && !r.DryRun {
		return fmt.Errorf("orchestrate: quantize source/output paths must differ")
	}
	if !r.Type.IsQuant() {
		return fmt.Errorf("orchestrate: quantize type %q is not a quant dtype", r.Type)
	}
	if r.Type.RequiresImatrix() && r.ImatrixPath == "" {
		return fmt.Errorf("orchestrate: type %q requires an imatrix", r.Type)
	}
	if r.SourceQuantized && !r.AllowRequantize {
		return fmt.Errorf("orchestrate: refusing to requantize already-quantized source %q (set AllowRequantize to override)", r.SourcePath)
	}
	if r.Threads < 0 {
		return fmt.Errorf("orchestrate: negative thread count")
	}
	return nil
}

// PlanQuantize builds the llama-quantize invocation for r, emitting only
// flags that caps advertises:
//
//	llama-quantize [flags...] <source> <output> <type> [nthreads]
//
// Flags appear in the fixed canonical order of quantFlagSpecs; positional
// arguments follow current llama.cpp usage with the thread count as the
// trailing positional. A nil caps permits the full current flag set; a
// non-nil caps rejects any flag the binary did not advertise. When caps
// advertises a non-empty quant type list, the positional `type` argument
// must appear in it (fail-closed against tools that do not support the
// requested dtype); an empty list (older tools) disables type gating.
func PlanQuantize(r QuantizeRequest, caps *Capabilities, binaryPath string) (Invocation, error) {
	if err := r.Validate(); err != nil {
		return Invocation{}, err
	}
	if caps != nil && caps.Tool != ToolLlamaQuantize {
		return Invocation{}, fmt.Errorf("orchestrate: capabilities are for %s, not %s", caps.Tool, ToolLlamaQuantize)
	}
	ftype := r.Type
	if r.Pure {
		mapped := r.Type.PureFType()
		if mapped != ftype && (caps == nil || caps.HasType(string(mapped))) {
			ftype = mapped
		}
	}
	if caps != nil && len(caps.Types) > 0 && !caps.HasType(string(ftype)) {
		return Invocation{}, fmt.Errorf("orchestrate: %s does not advertise quant type %s; cannot honor request", binaryPath, ftype)
	}
	var argv []string
	for _, spec := range quantFlagSpecs() {
		if !spec.when(r) {
			continue
		}
		if caps != nil && !caps.Has(spec.flag) {
			return Invocation{}, fmt.Errorf("orchestrate: %s does not advertise %s; cannot honor request", binaryPath, spec.flag)
		}
		args, err := spec.emit(r)
		if err != nil {
			return Invocation{}, err
		}
		argv = append(argv, args...)
	}
	argv = append(argv, r.SourcePath, r.OutputPath, string(ftype))
	if r.Threads > 0 {
		argv = append(argv, fmt.Sprintf("%d", r.Threads))
	}
	iv := Invocation{Tool: ToolLlamaQuantize, Path: binaryPath, Argv: argv}
	if err := iv.Validate(); err != nil {
		return Invocation{}, err
	}
	return iv, nil
}

// SourceIsQuantized reports whether any tensor in bank already uses a quant
// dtype, i.e. the source is a requantization candidate to refuse by default.
func SourceIsQuantized(bank *core.TensorBank) bool {
	if bank == nil {
		return false
	}
	for _, t := range bank.Tensors {
		if t.DType.IsQuant() {
			return true
		}
	}
	return false
}

// WriteTensorTypeFile writes the per-tensor override file consumed by
// llama-quantize --tensor-type: one "name TYPE" line per option, sorted by
// tensor name for byte-for-byte determinism across runs. Tensor names come
// from untrusted GGUF headers, so they are re-validated here (no whitespace,
// control characters, or empty names) to make override-line injection via a
// crafted tensor name impossible even if an earlier validation layer was
// bypassed.
func WriteTensorTypeFile(w io.Writer, opts []core.TensorOption) error {
	sorted := make([]core.TensorOption, len(opts))
	copy(sorted, opts)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].TensorName < sorted[j].TensorName })
	seen := map[string]bool{}
	for _, o := range sorted {
		if err := core.ValidateTensorName(o.TensorName); err != nil {
			return fmt.Errorf("orchestrate: tensor-type file: %w", err)
		}
		if err := o.Validate(); err != nil {
			return err
		}
		if seen[o.TensorName] {
			return fmt.Errorf("orchestrate: duplicate tensor %q in type file", o.TensorName)
		}
		seen[o.TensorName] = true
		if _, err := fmt.Fprintf(w, "%s %s\n", o.TensorName, o.Target.BaseTensorType()); err != nil {
			return err
		}
	}
	return nil
}

// TensorTypeFileBytes returns the canonical deterministic file contents.
func TensorTypeFileBytes(opts []core.TensorOption) ([]byte, error) {
	var sb strings.Builder
	if err := WriteTensorTypeFile(&sb, opts); err != nil {
		return nil, err
	}
	return []byte(sb.String()), nil
}

// TensorTypeFileSHA256 returns the hex digest of the canonical file contents.
func TensorTypeFileSHA256(opts []core.TensorOption) (string, error) {
	b, err := TensorTypeFileBytes(opts)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}
