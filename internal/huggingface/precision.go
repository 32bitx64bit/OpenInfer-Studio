package huggingface

import (
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// Precision classes group storage formats the way users think about them.
const (
	ClassFull  = "full"  // fp32 / bf16 / fp16
	ClassFP8   = "fp8"   // 8-bit float
	ClassInt8  = "int8"  // 8-bit integer packs
	ClassQuant = "quant" // GGUF block quantizations (Q4_K_M, IQ3_XS, Q8_0, …)
	Class4Bit  = "4bit"  // nf4 / fp4 / int4 packs outside GGUF
	// ClassPacked is weight/activation quantization (W6A8, AWQ, GPTQ): integer
	// weights that need per-layer scales to decode.
	ClassPacked = "packed"
)

// Precision describes how a weight file stores its numbers, read from the
// file name. The zero value means "not stated in the name".
type Precision struct {
	ID     string  `json:"id"`               // canonical id, e.g. "q4_k_m", "bf16", "fp8"
	Label  string  `json:"label"`            // display label, e.g. "Q4_K_M", "FP8"
	Bits   float64 `json:"bits"`             // nominal bits per weight; 0 when unknown
	Class  string  `json:"class,omitempty"`  // Class* constant
	Packed bool    `json:"packed,omitempty"` // needs per-tensor scales or rotation to decode
}

// Known reports whether the file name stated a precision.
func (p Precision) Known() bool { return p.ID != "" }

// effectiveBits is Bits, with an unstated precision assumed to be a 16-bit
// float (what checkpoint files usually hold) so options still order sensibly.
func (p Precision) effectiveBits() float64 {
	if p.Bits > 0 {
		return p.Bits
	}
	return 16
}

// quantBits holds nominal bits per weight of the GGUF quantizations (block
// scales included, as llama.cpp reports them). Used only to order options
// and to aim presets; they are not size predictions.
var quantBits = map[string]float64{
	"IQ1_S": 1.56, "IQ1_M": 1.75,
	"IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5, "IQ2_M": 2.7,
	"Q2_K": 2.96, "Q2_K_S": 2.8, "Q2_K_L": 3.1, "Q2_K_XL": 3.2,
	"IQ3_XXS": 3.06, "IQ3_XS": 3.3, "IQ3_S": 3.44, "IQ3_M": 3.66,
	"Q3_K_S": 3.41, "Q3_K_M": 3.74, "Q3_K_L": 4.03, "Q3_K_XL": 4.2,
	"IQ4_XS": 4.25, "IQ4_NL": 4.5, "MXFP4": 4.25,
	"Q4_0": 4.34, "Q4_K_S": 4.37, "Q4_K_M": 4.58, "Q4_K_L": 4.8, "Q4_K_XL": 4.9, "Q4_1": 5.0,
	"Q5_0": 5.21, "Q5_K_S": 5.21, "Q5_K_M": 5.33, "Q5_K_L": 5.5, "Q5_K_XL": 5.6, "Q5_1": 6.0,
	"Q6_K": 6.14, "Q6_K_L": 6.4, "Q6_K_XL": 6.5,
	"Q8_0": 8.5, "Q8_K_L": 8.7, "Q8_K_XL": 8.8,
	"TQ1_0": 1.69, "TQ2_0": 2.06,
	"F16": 16, "BF16": 16, "F32": 32,
}

var quantFamilyRe = regexp.MustCompile(`^(IQ|TQ|Q)(\d)`)

// ggufPrecision builds the Precision of a GGUF quantization token (as
// returned by quantOf, possibly prefixed UD- / OID-).
func ggufPrecision(quant string) Precision {
	label := strings.ToUpper(strings.TrimSpace(quant))
	if label == "" {
		return Precision{}
	}
	bare := strings.TrimPrefix(strings.TrimPrefix(label, "UD-"), "OID-")
	p := Precision{ID: strings.ToLower(label), Label: label, Class: ClassQuant}
	switch bare {
	case "F16":
		p.ID, p.Class = "fp16", ClassFull
	case "BF16":
		p.ID, p.Class = "bf16", ClassFull
	case "F32":
		p.ID, p.Class = "fp32", ClassFull
	}
	if b, ok := quantBits[bare]; ok {
		p.Bits = b
	} else if m := quantFamilyRe.FindStringSubmatch(bare); m != nil {
		// Unlisted variant (Q4_K_XXL, …): digit plus typical block overhead.
		d := float64(m[2][0] - '0')
		if m[1] == "IQ" {
			p.Bits = d + 0.3
		} else {
			p.Bits = d + 0.6
		}
	}
	return p
}

// precisionTokens are the file-name tokens that state a storage format.
// Matching is per whole token (names are split on non-alphanumerics), so
// "flux1" never reads as a float format.
var precisionTokens = map[string]bool{
	"fp32": true, "f32": true, "float32": true,
	"fp16": true, "f16": true, "float16": true,
	"bf16": true, "bfloat16": true,
	"fp8": true, "float8": true, "e4m3": true, "e4m3fn": true, "e4m3fnuz": true, "e5m2": true,
	"scaled": true, "int8": true, "i8": true, "w8a8": true, "convrot": true,
	"nf4": true, "fp4": true, "nvfp4": true, "int4": true, "svdq": true,
	"awq": true, "gptq": true,
	// GGUF dynamic-quant prefixes that stay behind once the quant is removed.
	"ud": true, "oid": true,
}

var (
	// weightActivationRe matches w6a8, w8a8, w4a16: weight/activation bit widths.
	weightActivationRe = regexp.MustCompile(`^w(\d{1,2})a\d{1,2}$`)
	nameTokenRe        = regexp.MustCompile(`[A-Za-z0-9]+`)
	shardRe            = regexp.MustCompile(`(?i)[-_.]?\d{5}-of-\d{5}`)
)

// isPrecisionToken reports whether a lower-case name token states a storage
// format (fp16, e4m3fn, Q-less names like w6a8, awq, …).
func isPrecisionToken(t string) bool {
	return precisionTokens[t] || weightActivationRe.MatchString(t)
}

// nameTokens lower-cases a file stem and splits it into alphanumeric tokens.
func nameTokens(stem string) []string {
	raw := nameTokenRe.FindAllString(stem, -1)
	for i := range raw {
		raw[i] = strings.ToLower(raw[i])
	}
	return raw
}

// PrecisionOf reads a file's precision from its name. GGUF files use the
// quantization token; safetensors/ckpt files use float / fp8 / int8 / 4-bit
// tokens (t5xxl_fp8_e4m3fn_scaled.safetensors, flux1-dev-fp8.safetensors,
// diffusion_pytorch_model.fp16.safetensors).
func PrecisionOf(path string) Precision {
	base := filepathBase(path)
	lower := strings.ToLower(base)
	if strings.HasSuffix(lower, ".gguf") {
		return ggufPrecision(quantOf(path))
	}
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	toks := map[string]bool{}
	for _, t := range nameTokens(shardRe.ReplaceAllString(stem, "")) {
		toks[t] = true
	}
	has := func(names ...string) bool {
		for _, n := range names {
			if toks[n] {
				return true
			}
		}
		return false
	}
	scaled := has("scaled")
	// w6a8 / w8a4 / w4a16: weight and activation bit widths. Hardware-specific
	// integer packs; the weight width is the bits per weight.
	for tok := range toks {
		if m := weightActivationRe.FindStringSubmatch(tok); m != nil && tok != "w8a8" {
			bits, _ := strconv.Atoi(m[1])
			return Precision{ID: tok, Label: strings.ToUpper(tok), Bits: float64(bits), Class: ClassPacked, Packed: true}
		}
	}
	switch {
	case has("nf4"):
		return Precision{ID: "nf4", Label: "NF4", Bits: 4, Class: Class4Bit, Packed: true}
	case has("nvfp4", "fp4"):
		return Precision{ID: "fp4", Label: "FP4", Bits: 4, Class: Class4Bit, Packed: true}
	case has("int4", "svdq"):
		return Precision{ID: "int4", Label: "INT4", Bits: 4, Class: Class4Bit, Packed: true}
	case has("awq"):
		return Precision{ID: "awq", Label: "AWQ", Bits: 4, Class: ClassPacked, Packed: true}
	case has("gptq"):
		return Precision{ID: "gptq", Label: "GPTQ", Bits: 4, Class: ClassPacked, Packed: true}
	case has("int8", "i8", "w8a8"):
		// There is no plain int8 weight layout in safetensors: these files
		// carry scales (and sometimes a rotation) that loaders must apply.
		return Precision{ID: "int8", Label: "INT8", Bits: 8, Class: ClassInt8, Packed: true}
	case has("fp8", "float8", "e4m3", "e4m3fn", "e4m3fnuz", "e5m2"):
		detail := ""
		switch {
		case has("e5m2"):
			detail = "e5m2"
		case has("e4m3fn"):
			detail = "e4m3fn"
		case has("e4m3fnuz"):
			detail = "e4m3fnuz"
		case has("e4m3"):
			detail = "e4m3"
		}
		p := Precision{ID: "fp8", Label: "FP8", Bits: 8, Class: ClassFP8, Packed: scaled || has("convrot")}
		if detail != "" {
			p.ID += "_" + detail
			p.Label += " " + detail
		}
		if scaled {
			p.ID += "_scaled"
			p.Label += " scaled"
		}
		return p
	case has("bf16", "bfloat16"):
		return Precision{ID: "bf16", Label: "BF16", Bits: 16, Class: ClassFull}
	case has("fp16", "f16", "float16"):
		return Precision{ID: "fp16", Label: "FP16", Bits: 16, Class: ClassFull}
	case has("fp32", "f32", "float32"):
		return Precision{ID: "fp32", Label: "FP32", Bits: 32, Class: ClassFull}
	case scaled || has("convrot"):
		return Precision{ID: "scaled", Label: "Scaled", Bits: 8, Class: ClassFP8, Packed: true}
	}
	return Precision{}
}

// stemSansPrecision is the file stem with precision tokens, GGUF quant
// tokens and shard markers removed: files that differ only in precision
// share it, files that are different models do not.
func stemSansPrecision(path string) string {
	base := filepathBase(path)
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	stem = shardRe.ReplaceAllString(stem, "")
	if strings.HasSuffix(strings.ToLower(base), ".gguf") {
		if q := quantOf(path); q != "" {
			// quantRe wants a separator in front of the token.
			stem = quantRe.ReplaceAllString(stem, "")
		}
	}
	return dropTokens(stem, isPrecisionToken)
}

// dropTokens removes the alphanumeric tokens drop() accepts (given the token
// in lower case) from a stem, keeping the separators between the rest.
func dropTokens(stem string, drop func(lower string) bool) string {
	locs := nameTokenRe.FindAllStringIndex(stem, -1)
	var b strings.Builder
	prevEnd := 0
	for _, loc := range locs {
		tok := stem[loc[0]:loc[1]]
		sep := stem[prevEnd:loc[0]] // separator right before this token
		prevEnd = loc[1]
		if drop(strings.ToLower(tok)) {
			continue
		}
		if b.Len() > 0 {
			b.WriteString(sep)
		}
		b.WriteString(tok)
	}
	return strings.Trim(b.String(), "-_. ")
}
