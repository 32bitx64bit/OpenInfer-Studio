// Package quantize drives llama-quantize / llama-imatrix from the selected
// llama.cpp runtime as managed background jobs.
package quantize

import (
	"regexp"
	"strings"
)

// IMatrixPolicy describes whether a ftype needs an importance matrix.
type IMatrixPolicy string

const (
	IMatrixRequired    IMatrixPolicy = "required"
	IMatrixRecommended IMatrixPolicy = "recommended"
	IMatrixOptional    IMatrixPolicy = "optional"
	IMatrixNone        IMatrixPolicy = "none"
)

// Band is a UI grouping for quantization types.
type Band string

const (
	BandNearLossless Band = "near_lossless"
	BandBalanced     Band = "balanced"
	BandCompact      Band = "compact"
	BandAggressive   Band = "aggressive"
	BandExperimental Band = "experimental"
	BandRepack       Band = "repack"
)

// FType is one llama-quantize output type with OpenInfer UX metadata.
type FType struct {
	Name         string        `json:"name"`
	Description  string        `json:"description"`
	BPW          float64       `json:"bpw"`
	Band         Band          `json:"band"`
	Simple       bool          `json:"simple"`
	IMatrix      IMatrixPolicy `json:"imatrix"`
	AliasOf      string        `json:"alias_of,omitempty"`
	Experimental bool          `json:"experimental"`
	MoEOnly      bool          `json:"moe_only,omitempty"`
	Advertised   bool          `json:"advertised"` // present in this runtime's --help
}

// knownFTypes is the UX overlay. A runtime catalog only exposes types that
// appear in llama-quantize --help (Advertised=true), plus aliases hidden from
// the simple UI.
var knownFTypes = []FType{
	{Name: "Q8_0", Description: "Near-full quality 8-bit. Fast to produce, large file.", BPW: 8.50, Band: BandNearLossless, Simple: true, IMatrix: IMatrixOptional},
	{Name: "Q6_K", Description: "Excellent quality k-quant. Small loss versus Q8.", BPW: 6.56, Band: BandNearLossless, Simple: true, IMatrix: IMatrixOptional},
	{Name: "Q5_K_M", Description: "Very good quality with mixed higher-bit tensors.", BPW: 5.70, Band: BandBalanced, Simple: true, IMatrix: IMatrixRecommended},
	{Name: "Q5_K_S", Description: "5-bit k-quant, slightly smaller than Q5_K_M.", BPW: 5.57, Band: BandBalanced, Simple: false, IMatrix: IMatrixRecommended},
	{Name: "Q5_K", Description: "Alias for Q5_K_M.", BPW: 5.70, Band: BandBalanced, Simple: false, IMatrix: IMatrixRecommended, AliasOf: "Q5_K_M"},
	{Name: "Q5_1", Description: "Legacy 5-bit.", BPW: 5.50, Band: BandBalanced, Simple: false, IMatrix: IMatrixOptional},
	{Name: "Q5_0", Description: "Legacy 5-bit.", BPW: 5.21, Band: BandBalanced, Simple: false, IMatrix: IMatrixOptional},
	{Name: "Q4_K_M", Description: "Best balance of size, speed, and quality for most users.", BPW: 4.89, Band: BandBalanced, Simple: true, IMatrix: IMatrixRecommended},
	{Name: "Q4_K_S", Description: "Smaller 4-bit k-quant than Q4_K_M.", BPW: 4.67, Band: BandCompact, Simple: true, IMatrix: IMatrixRecommended},
	{Name: "Q4_K", Description: "Alias for Q4_K_M.", BPW: 4.89, Band: BandBalanced, Simple: false, IMatrix: IMatrixRecommended, AliasOf: "Q4_K_M"},
	{Name: "Q4_1", Description: "Legacy 4-bit.", BPW: 5.00, Band: BandCompact, Simple: false, IMatrix: IMatrixOptional},
	{Name: "Q4_0", Description: "Legacy 4-bit.", BPW: 4.50, Band: BandCompact, Simple: false, IMatrix: IMatrixOptional},
	{Name: "IQ4_NL", Description: "4.5 bpw non-linear. Competitive with Q4_K; benefits from an imatrix.", BPW: 4.50, Band: BandCompact, Simple: true, IMatrix: IMatrixRecommended},
	{Name: "IQ4_XS", Description: "4.25 bpw non-linear. Strong compact choice with an imatrix.", BPW: 4.25, Band: BandCompact, Simple: true, IMatrix: IMatrixRecommended},
	{Name: "Q3_K_L", Description: "3-bit k-quant, larger mix.", BPW: 4.03, Band: BandCompact, Simple: false, IMatrix: IMatrixRecommended},
	{Name: "Q3_K_M", Description: "Balanced 3-bit k-quant.", BPW: 3.74, Band: BandCompact, Simple: true, IMatrix: IMatrixRecommended},
	{Name: "Q3_K_S", Description: "Small 3-bit k-quant.", BPW: 3.44, Band: BandAggressive, Simple: false, IMatrix: IMatrixRecommended},
	{Name: "Q3_K", Description: "Alias for Q3_K_M.", BPW: 3.74, Band: BandCompact, Simple: false, IMatrix: IMatrixRecommended, AliasOf: "Q3_K_M"},
	{Name: "IQ3_M", Description: "3.66 bpw importance-aware mix.", BPW: 3.66, Band: BandAggressive, Simple: false, IMatrix: IMatrixRequired},
	{Name: "IQ3_S", Description: "3.44 bpw importance-aware.", BPW: 3.44, Band: BandAggressive, Simple: false, IMatrix: IMatrixRequired},
	{Name: "IQ3_XS", Description: "3.3 bpw importance-aware.", BPW: 3.30, Band: BandAggressive, Simple: false, IMatrix: IMatrixRequired},
	{Name: "IQ3_XXS", Description: "3.06 bpw. Requires an imatrix.", BPW: 3.06, Band: BandAggressive, Simple: false, IMatrix: IMatrixRequired},
	{Name: "Q2_K", Description: "2-bit k-quant. Quality drops without an imatrix.", BPW: 3.16, Band: BandAggressive, Simple: false, IMatrix: IMatrixRequired},
	{Name: "Q2_K_S", Description: "2-bit k-quant small. Requires an imatrix.", BPW: 2.97, Band: BandAggressive, Simple: false, IMatrix: IMatrixRequired},
	{Name: "IQ2_M", Description: "2.7 bpw. Requires an imatrix.", BPW: 2.70, Band: BandAggressive, Simple: false, IMatrix: IMatrixRequired},
	{Name: "IQ2_S", Description: "2.5 bpw. Requires an imatrix.", BPW: 2.50, Band: BandAggressive, Simple: false, IMatrix: IMatrixRequired},
	{Name: "IQ2_XS", Description: "2.31 bpw. Requires an imatrix.", BPW: 2.31, Band: BandAggressive, Simple: false, IMatrix: IMatrixRequired},
	{Name: "IQ2_XXS", Description: "2.06 bpw. Requires an imatrix.", BPW: 2.06, Band: BandAggressive, Simple: false, IMatrix: IMatrixRequired},
	{Name: "IQ1_M", Description: "1.75 bpw. Experimental; requires an imatrix.", BPW: 1.75, Band: BandExperimental, Simple: false, IMatrix: IMatrixRequired, Experimental: true},
	{Name: "IQ1_S", Description: "1.56 bpw. Experimental; requires an imatrix.", BPW: 1.56, Band: BandExperimental, Simple: false, IMatrix: IMatrixRequired, Experimental: true},
	{Name: "Q2_0", Description: "2.25 bpw group-64. Experimental.", BPW: 2.25, Band: BandExperimental, Simple: false, IMatrix: IMatrixRecommended, Experimental: true},
	{Name: "Q1_0", Description: "1.125 bpw. Experimental.", BPW: 1.13, Band: BandExperimental, Simple: false, IMatrix: IMatrixRequired, Experimental: true},
	{Name: "TQ1_0", Description: "Ternary 1.69 bpw. Experimental.", BPW: 1.69, Band: BandExperimental, Simple: false, IMatrix: IMatrixRequired, Experimental: true},
	{Name: "TQ2_0", Description: "Ternary 2.06 bpw. Experimental.", BPW: 2.06, Band: BandExperimental, Simple: false, IMatrix: IMatrixRequired, Experimental: true},
	{Name: "MXFP4_MOE", Description: "MXFP4 for Mixture-of-Experts models.", BPW: 4.25, Band: BandExperimental, Simple: false, IMatrix: IMatrixOptional, Experimental: true, MoEOnly: true},
	{Name: "F16", Description: "Half precision. Not a compression target.", BPW: 16.00, Band: BandRepack, Simple: false, IMatrix: IMatrixNone},
	{Name: "BF16", Description: "BFloat16. Not a compression target.", BPW: 16.00, Band: BandRepack, Simple: false, IMatrix: IMatrixNone},
	{Name: "F32", Description: "Full precision. Not a compression target.", BPW: 32.00, Band: BandRepack, Simple: false, IMatrix: IMatrixNone},
	{Name: "COPY", Description: "Copy tensors without quantizing.", BPW: 0, Band: BandRepack, Simple: false, IMatrix: IMatrixNone},
}

var ftypeTokenRe = regexp.MustCompile(`(?i)\b((?:IQ[1-4]_[A-Z0-9]+)|(?:Q[1-8](?:_[A-Z0-9_]+)?)|(?:TQ[12]_0)|(?:MXFP4_MOE)|(?:MXFP4)|(?:BF16)|(?:F16)|(?:F32)|(?:COPY))\b`)

var orTypeRe = regexp.MustCompile(`(?i)\bor\s+((?:IQ[1-4]_[A-Z0-9]+)|(?:Q[1-8](?:_[A-Z0-9_]+)?)|(?:TQ[12]_0)|(?:MXFP4_MOE)|(?:BF16)|(?:F16)|(?:F32)|(?:COPY))\b`)

func knownByName() map[string]FType {
	m := map[string]FType{}
	for _, t := range knownFTypes {
		m[strings.ToUpper(t.Name)] = t
	}
	return m
}

// LookupFType returns known metadata for name (case-insensitive).
func LookupFType(name string) (FType, bool) {
	t, ok := knownByName()[strings.ToUpper(strings.TrimSpace(name))]
	return t, ok
}

// ParseQuantizeTypes extracts ftypes advertised in llama-quantize --help.
// Unknown tokens still listed in help are returned with conservative metadata.
func ParseQuantizeTypes(help string) []FType {
	seen := map[string]bool{}
	var names []string
	for _, m := range orTypeRe.FindAllStringSubmatch(help, -1) {
		n := strings.ToUpper(m[1])
		if !seen[n] {
			seen[n] = true
			names = append(names, n)
		}
	}
	if len(names) == 0 {
		for _, m := range ftypeTokenRe.FindAllStringSubmatch(help, -1) {
			n := strings.ToUpper(m[1])
			if n == "COPY" && !strings.Contains(strings.ToUpper(help), "COPY") {
				continue
			}
			if !seen[n] {
				seen[n] = true
				names = append(names, n)
			}
		}
	}
	known := knownByName()
	var out []FType
	for _, n := range names {
		if t, ok := known[n]; ok {
			t.Advertised = true
			out = append(out, t)
			continue
		}
		out = append(out, FType{
			Name: n, Description: "Advertised by this runtime.",
			BPW: 4, Band: BandExperimental, IMatrix: IMatrixRecommended, Advertised: true, Experimental: true,
		})
	}
	return out
}

// HighPrecision reports whether src is a suitable quantization source
// without a scary requantize confirmation. Q8_0 is included: it is the
// usual "full quality" download. llama-quantize still needs
// --allow-requantize for Q8 (see needsRequantizeFlag).
func HighPrecision(quant string) bool {
	switch stripDynamicPrefix(quant) {
	case "", "F32", "F16", "BF16", "Q8_0":
		return true
	}
	return false
}

func stripDynamicPrefix(quant string) string {
	u := strings.ToUpper(strings.TrimSpace(quant))
	u = strings.TrimPrefix(u, "UD-")
	u = strings.TrimPrefix(u, "OID-")
	return u
}

// needsRequantizeFlag reports whether llama-quantize will refuse the source
// unless --allow-requantize is passed. Float types are allowed; Q8 and
// below are not.
func needsRequantizeFlag(quant string) bool {
	switch strings.ToUpper(strings.TrimSpace(quant)) {
	case "", "F32", "F16", "BF16":
		return false
	default:
		return true
	}
}

// CanonicalFType resolves aliases (Q4_K → Q4_K_M).
func CanonicalFType(name string) string {
	t, ok := LookupFType(name)
	if !ok {
		return strings.ToUpper(strings.TrimSpace(name))
	}
	if t.AliasOf != "" {
		return t.AliasOf
	}
	return t.Name
}

func RequiresIMatrix(name string) bool {
	t, ok := LookupFType(name)
	if !ok {
		u := strings.ToUpper(name)
		return strings.HasPrefix(u, "IQ1") || strings.HasPrefix(u, "IQ2") || u == "IQ3_XXS" || u == "Q2_K_S"
	}
	return t.IMatrix == IMatrixRequired
}

func RecommendsIMatrix(name string) bool {
	t, ok := LookupFType(name)
	if !ok {
		return true
	}
	return t.IMatrix == IMatrixRequired || t.IMatrix == IMatrixRecommended
}
