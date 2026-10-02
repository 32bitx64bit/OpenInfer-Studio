package sdmodel

import (
	"path/filepath"
	"regexp"
	"strings"
)

// Quantization packs.
//
// ComfyUI repositories publish weights in schemes that need more than the
// tensor itself to decode: per-tensor or per-block scales, activation
// scales, a rotation, packed 4/6-bit integers (scaled FP8, INT8 "convrot",
// NVFP4, AWQ, W6A8, …). stable-diffusion.cpp reads plain float, plain FP8
// and GGUF; on these packs its loader aborts the whole process while it is
// still building the model (ggml_abort in Linear::init_params). Spotting
// them up front turns a cryptic abort into an explanation.

// weightActivationRe matches names like w6a8, w8a8, w4a16.
var weightActivationRe = regexp.MustCompile(`^w\d{1,2}a\d{1,2}$`)

// packedTensorSuffixes are tensor-name endings that only quantized packs
// carry: scales, activation scales, ComfyUI's per-layer quant config, and
// AutoAWQ / GPTQ packed weights.
var packedTensorSuffixes = []string{
	".comfy_quant", ".weight_scale", ".weight_scale_2", ".scale_weight", ".scale_input",
	".input_scale", ".qweight", ".qzeros",
}

// PackedName reports the quantization pack a file name states ("INT8 with
// rotation (convrot)", "NVFP4", "W6A8", …), or "" when it states none.
func PackedName(path string) string {
	words := nameWords(strings.ToLower(filepath.Base(path)))
	var found []string
	add := func(s string) { found = append(found, s) }
	if words["convrot"] {
		add("INT8 with rotation (convrot)")
	} else if words["int8"] || words["i8"] {
		add("INT8")
	}
	if words["nvfp4"] {
		add("NVFP4")
	} else if words["fp4"] {
		add("FP4")
	}
	if words["nf4"] {
		add("NF4")
	}
	if words["int4"] || words["svdq"] {
		add("INT4")
	}
	for w := range words {
		if weightActivationRe.MatchString(w) {
			add(strings.ToUpper(w) + " weight/activation quantization")
		}
	}
	if words["awq"] {
		add("AWQ")
	}
	if words["gptq"] {
		add("GPTQ")
	}
	if words["scaled"] {
		add("scaled FP8")
	}
	return strings.Join(found, ", ")
}

// PackedReason says why a weight file is stored in a pack stable-diffusion.cpp
// is known to abort on, or "" when it looks loadable. It reads the file name,
// and for safetensors the tensor names too: scale tensors and ComfyUI's
// .comfy_quant entries identify a pack whatever the file is called. GGUF
// quantizations are native and always fine.
func PackedReason(path string) string {
	if strings.HasSuffix(strings.ToLower(path), ".gguf") {
		return ""
	}
	if why := PackedName(path); why != "" {
		return why
	}
	if !strings.HasSuffix(strings.ToLower(path), ".safetensors") {
		return ""
	}
	names, err := TensorNames(path)
	if err != nil {
		return ""
	}
	for _, n := range names {
		lower := strings.ToLower(n)
		for _, suffix := range packedTensorSuffixes {
			if strings.HasSuffix(lower, suffix) {
				return "quantized with scale tensors (" + strings.TrimPrefix(suffix, ".") + ")"
			}
		}
	}
	return ""
}
