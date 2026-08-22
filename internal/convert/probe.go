package convert

import (
	"fmt"
	"path/filepath"
	"strings"
)

// NeededFile is one snapshot path the converter needs.
type NeededFile struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// ProbeInput is Hub metadata plus parsed config.json.
type ProbeInput struct {
	RepoID  string
	Tags    []string
	Files   []NeededFile
	DTypes  map[string]int64 // Hub safetensors.parameters
	Config  map[string]any
	HasJSON bool // config.json was fetched
}

// ProbeResult is the compatibility check shown on the Quantization page.
type ProbeResult struct {
	Compatible         bool         `json:"compatible"`
	Reason             string       `json:"reason,omitempty"`
	Architecture       string       `json:"architecture,omitempty"`
	Adapter            string       `json:"adapter,omitempty"`
	WeightDType        string       `json:"weight_dtype,omitempty"`
	SnapshotBytes      int64        `json:"snapshot_bytes"`
	EstimatedGGUFBytes int64        `json:"estimated_gguf_bytes"`
	Warnings           []string     `json:"warnings,omitempty"`
	Files              []NeededFile `json:"files,omitempty"`
	VisionWarning      bool         `json:"vision_warning"`
}

var allowedDTypes = map[string]bool{
	"F32": true, "F16": true, "BF16": true,
	"FLOAT32": true, "FLOAT16": true, "BFLOAT16": true,
}

func Evaluate(in ProbeInput) ProbeResult {
	out := ProbeResult{}
	if !in.HasJSON || in.Config == nil {
		out.Reason = "repository has no config.json"
		return out
	}
	if looksQuantizedRepo(in.RepoID, in.Tags, in.Files) {
		out.Reason = "repository looks like GPTQ, AWQ, bitsandbytes, or NVFP4 — convert needs BF16/F16/F32 safetensors"
		return out
	}
	var st []NeededFile
	var stBytes int64
	for _, f := range in.Files {
		if strings.HasSuffix(strings.ToLower(f.Path), ".safetensors") &&
			!strings.Contains(strings.ToLower(filepath.Base(f.Path)), "optimizer") {
			st = append(st, f)
			stBytes += f.Size
		}
	}
	if len(st) == 0 {
		out.Reason = "repository has no .safetensors weights (GGUF-only or .bin-only repos cannot be converted here)"
		return out
	}
	dtype, err := hubWeightDType(in.DTypes)
	if err != nil {
		out.Reason = err.Error()
		return out
	}
	out.WeightDType = dtype
	ad, err := FindAdapter(in.Config)
	if err != nil {
		out.Reason = err.Error()
		return out
	}
	out.Adapter = ad.ID()
	out.Architecture = ad.ID()

	keep := SelectSnapshotFiles(in.Files)
	var snap int64
	for _, f := range keep {
		snap += f.Size
	}
	out.Files = keep
	out.SnapshotBytes = snap
	out.EstimatedGGUFBytes = stBytes
	if strings.EqualFold(dtype, "F32") {
		out.EstimatedGGUFBytes = stBytes / 2
	}
	if hasVision(in.Config, in.Files) {
		out.VisionWarning = true
		out.Warnings = append(out.Warnings, "this repo includes vision weights; conversion writes a language GGUF only. Chat still works; keep using an existing mmproj for images.")
	}
	out.Compatible = true
	return out
}

func hubWeightDType(params map[string]int64) (string, error) {
	if len(params) == 0 {
		return "", fmt.Errorf("Hub did not report safetensors dtypes; refusing to download an unknown precision")
	}
	best := ""
	var bestN int64
	for k, n := range params {
		u := strings.ToUpper(strings.TrimSpace(k))
		if !allowedDTypes[u] {
			return "", fmt.Errorf("safetensors dtype %s is not convertible (need BF16, F16, or F32)", k)
		}
		if n >= bestN {
			bestN = n
			best = canonicalDType(u)
		}
	}
	if best == "" {
		return "", fmt.Errorf("Hub safetensors.parameters is empty")
	}
	return best, nil
}

func canonicalDType(u string) string {
	switch u {
	case "FLOAT32", "F32":
		return "F32"
	case "FLOAT16", "F16":
		return "F16"
	default:
		return "BF16"
	}
}

func looksQuantizedRepo(id string, tags []string, files []NeededFile) bool {
	blob := strings.ToLower(id)
	for _, t := range tags {
		blob += " " + strings.ToLower(t)
	}
	for _, f := range files {
		blob += " " + strings.ToLower(f.Path)
	}
	needles := []string{"gptq", "awq", "bitsandbytes", "bnb-4bit", "bnb_4bit", "nvfp4", "fp8", "nf4", "int4", "int8", "w8a16", "w4a16"}
	for _, n := range needles {
		if strings.Contains(blob, n) {
			return true
		}
	}
	return false
}

func hasVision(cfg map[string]any, files []NeededFile) bool {
	if _, ok := cfg["vision_config"]; ok {
		return true
	}
	for _, a := range stringSlice(cfg["architectures"]) {
		if strings.Contains(strings.ToLower(a), "conditionalgeneration") ||
			strings.Contains(strings.ToLower(a), "forvision") {
			return true
		}
	}
	for _, f := range files {
		b := strings.ToLower(filepath.Base(f.Path))
		if b == "processor_config.json" || b == "preprocessor_config.json" {
			return true
		}
		if strings.Contains(strings.ToLower(f.Path), "vision") {
			return true
		}
	}
	return false
}

// SelectSnapshotFiles keeps weights + tokenizer/config needed to convert.
func SelectSnapshotFiles(files []NeededFile) []NeededFile {
	var out []NeededFile
	for _, f := range files {
		if KeepSnapshotPath(f.Path) {
			out = append(out, f)
		}
	}
	return out
}

func KeepSnapshotPath(path string) bool {
	if path == "" || strings.Contains(path, "..") {
		return false
	}
	base := strings.ToLower(filepath.Base(path))
	if strings.HasSuffix(base, ".safetensors") {
		return !strings.Contains(base, "optimizer")
	}
	switch base {
	case "config.json", "tokenizer.json", "tokenizer_config.json",
		"special_tokens_map.json", "generation_config.json",
		"processor_config.json", "preprocessor_config.json",
		"added_tokens.json", "chat_template.jinja", "chat_template.json",
		"vocab.json", "merges.txt", "model.safetensors.index.json":
		return true
	}
	if strings.HasSuffix(base, ".jinja") && strings.Contains(base, "chat") {
		return true
	}
	if strings.HasSuffix(base, ".json") && strings.Contains(base, "tokenizer") {
		return true
	}
	if strings.HasSuffix(base, "index.json") && strings.Contains(base, "safetensors") {
		return true
	}
	return false
}

// NormalizeRepoID accepts author/name or a huggingface.co URL.
func NormalizeRepoID(s string) (string, error) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "https://huggingface.co/")
	s = strings.TrimPrefix(s, "http://huggingface.co/")
	s = strings.TrimPrefix(s, "huggingface.co/")
	s = strings.Trim(s, "/")
	if i := strings.Index(s, "?"); i >= 0 {
		s = s[:i]
	}
	parts := strings.Split(s, "/")
	if len(parts) < 2 || parts[0] == "" || parts[1] == "" {
		return "", fmt.Errorf("expected author/model")
	}
	if strings.Contains(parts[0], "..") || strings.Contains(parts[1], "..") {
		return "", fmt.Errorf("invalid repository id")
	}
	return parts[0] + "/" + parts[1], nil
}
