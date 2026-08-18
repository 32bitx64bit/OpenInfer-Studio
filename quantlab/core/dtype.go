package core

// DType identifies a GGUF tensor element type as used by llama.cpp. Two
// families share this enum:
//
//   - Per-tensor GGML types (Q4_K, Q6_K, IQ2_XXS, ...): what an individual
//     tensor is stored as. These have exact block geometry (see Geometry).
//   - Whole-model recipe labels (Q4_K_M, Q3_K_L, ...): what llama-quantize
//     accepts as its type argument, encoding a base per-tensor type plus
//     upgrade rules for sensitive tensors. IsRecipeLabel distinguishes them.
//
// Legacy constants DTypeQ4_K ("Q4_K_M") and DTypeQ5_K ("Q5_K_M") retain their
// recipe-label values for compatibility; new code should prefer the explicit
// constants and the Recipe type.
type DType string

// Float (unquantized) per-tensor types.
const (
	DTypeF32  DType = "F32"
	DTypeF16  DType = "F16"
	DTypeBF16 DType = "BF16"
	DTypeF64  DType = "F64"
)

// Integer per-tensor GGML types. These occur in GGUF payloads but are not
// quantization targets.
const (
	DTypeI8  DType = "I8"
	DTypeI16 DType = "I16"
	DTypeI32 DType = "I32"
	DTypeI64 DType = "I64"
)

// Integer K-quants: per-tensor base types and recipe labels.
const (
	DTypeQ8_0   DType = "Q8_0"
	DTypeQ8_1   DType = "Q8_1"
	DTypeQ8_K   DType = "Q8_K"
	DTypeQ6_K   DType = "Q6_K"
	DTypeQ5_K   DType = "Q5_K_M" // legacy alias: recipe label (compat)
	DTypeQ5_K_T DType = "Q5_K"   // explicit per-tensor base type
	DTypeQ5_K_M DType = "Q5_K_M" // recipe label
	DTypeQ5_K_S DType = "Q5_K_S" // recipe label
	DTypeQ5_1   DType = "Q5_1"
	DTypeQ5_0   DType = "Q5_0"
	DTypeQ4_K   DType = "Q4_K_M" // legacy alias: recipe label (compat)
	DTypeQ4_K_T DType = "Q4_K"   // explicit per-tensor base type
	DTypeQ4_K_M DType = "Q4_K_M" // recipe label
	DTypeQ4_K_S DType = "Q4_K_S" // recipe label
	DTypeQ4_1   DType = "Q4_1"
	DTypeQ4_0   DType = "Q4_0"
	DTypeQ3_K   DType = "Q3_K"   // per-tensor base type
	DTypeQ3_K_L DType = "Q3_K_L" // recipe label
	DTypeQ3_K_M DType = "Q3_K_M" // recipe label
	DTypeQ3_K_S DType = "Q3_K_S" // recipe label
	DTypeQ2_K   DType = "Q2_K"
)

// IQ (importance-matrix) per-tensor types. These require an imatrix for
// acceptable quality.
const (
	DTypeIQ4_NL  DType = "IQ4_NL"
	DTypeIQ4_XS  DType = "IQ4_XS"
	DTypeIQ3_XXS DType = "IQ3_XXS"
	// DTypeIQ3_XS is retained only to reject stale requests. Current ggml has
	// no raw IQ3_XS tensor type ID.
	DTypeIQ3_XS  DType = "IQ3_XS"
	DTypeIQ3_S   DType = "IQ3_S"
	DTypeIQ2_XXS DType = "IQ2_XXS"
	DTypeIQ2_XS  DType = "IQ2_XS"
	DTypeIQ2_S   DType = "IQ2_S"
	// DTypeIQ2_M is a llama-quantize ftype recipe, not a ggml tensor type.
	// llama_ftype_get_default_type(MOSTLY_IQ2_S) is IQ2_XS; MOSTLY_IQ2_M
	// writes IQ2_S tensors. Solver targets stay IQ2_S; PureFType maps the
	// positional --pure ftype.
	DTypeIQ2_M DType = "IQ2_M"
	DTypeIQ1_S DType = "IQ1_S"
	DTypeIQ1_M DType = "IQ1_M"
)

// Deprecated aliases retained for packages written against the initial
// scaffold. New code uses the explicit underscore-suffixed constants.
const (
	DTypeIQ2XXS = DTypeIQ2_XXS
	DTypeIQ3XXS = DTypeIQ3_XXS
)

// QuantDTypes lists quant dtypes from highest to lowest fidelity (by typical
// bits-per-weight). Recipe labels appear next to their base type.
var QuantDTypes = []DType{
	DTypeQ8_0,
	DTypeQ6_K,
	DTypeQ5_K_M, DTypeQ5_K_S, DTypeQ5_K_T, DTypeQ5_1, DTypeQ5_0,
	DTypeIQ4_NL,
	DTypeQ4_K_M, DTypeQ4_K_S, DTypeQ4_K_T, DTypeQ4_1, DTypeIQ4_XS, DTypeQ4_0,
	DTypeQ3_K_L, DTypeQ3_K_M, DTypeIQ3_S, DTypeQ3_K_S, DTypeIQ3_XXS, DTypeQ3_K,
	DTypeIQ2_S, DTypeQ2_K, DTypeIQ2_XS, DTypeIQ2_XXS,
	DTypeIQ1_M, DTypeIQ1_S,
}

func (d DType) Valid() bool {
	switch d {
	case DTypeF32, DTypeF16, DTypeBF16, DTypeF64,
		DTypeI8, DTypeI16, DTypeI32, DTypeI64,
		DTypeQ4_0, DTypeQ4_1, DTypeQ5_0, DTypeQ5_1, DTypeQ8_0, DTypeQ8_1,
		DTypeQ2_K, DTypeQ3_K, DTypeQ4_K_T, DTypeQ5_K_T, DTypeQ6_K, DTypeQ8_K,
		DTypeIQ2_XXS, DTypeIQ2_XS, DTypeIQ3_XXS, DTypeIQ1_S, DTypeIQ4_NL,
		DTypeIQ3_S, DTypeIQ2_S, DTypeIQ4_XS, DTypeIQ1_M:
		return true
	}
	return d.IsRecipeLabel()
}

func (d DType) IsFloat() bool {
	switch d {
	case DTypeF32, DTypeF16, DTypeBF16, DTypeF64:
		return true
	}
	return false
}

func (d DType) IsQuant() bool {
	switch d {
	case DTypeQ4_0, DTypeQ4_1, DTypeQ5_0, DTypeQ5_1, DTypeQ8_0, DTypeQ8_1,
		DTypeQ2_K, DTypeQ3_K, DTypeQ4_K_T, DTypeQ5_K_T, DTypeQ6_K, DTypeQ8_K,
		DTypeIQ2_XXS, DTypeIQ2_XS, DTypeIQ3_XXS, DTypeIQ1_S, DTypeIQ4_NL,
		DTypeIQ3_S, DTypeIQ2_S, DTypeIQ4_XS, DTypeIQ1_M:
		return true
	}
	return d.IsRecipeLabel()
}

// IsRecipeLabel reports whether d is a whole-model llama-quantize recipe
// label rather than a single per-tensor GGML type.
func (d DType) IsRecipeLabel() bool {
	switch d {
	case DTypeQ5_K_M, DTypeQ5_K_S, DTypeQ4_K_M, DTypeQ4_K_S,
		DTypeQ3_K_L, DTypeQ3_K_M, DTypeQ3_K_S, DTypeIQ2_M:
		return true
	}
	return false
}

// RequiresImatrix reports whether d is an IQ type whose quality depends on an
// importance matrix. Recipe labels follow their base per-tensor type.
func (d DType) RequiresImatrix() bool {
	switch d.BaseTensorType() {
	case DTypeIQ4_NL, DTypeIQ4_XS, DTypeIQ3_XXS, DTypeIQ3_S,
		DTypeIQ2_XXS, DTypeIQ2_XS, DTypeIQ2_S, DTypeIQ1_S, DTypeIQ1_M:
		return true
	}
	return false
}

// BaseTensorType maps a recipe label to its dominant per-tensor type; it
// returns d unchanged when d is already a per-tensor type.
func (d DType) BaseTensorType() DType {
	switch d {
	case DTypeQ5_K_M, DTypeQ5_K_S:
		return DTypeQ5_K_T
	case DTypeQ4_K_M, DTypeQ4_K_S:
		return DTypeQ4_K_T
	case DTypeQ3_K_L, DTypeQ3_K_M, DTypeQ3_K_S:
		return DTypeQ3_K
	case DTypeIQ2_M:
		return DTypeIQ2_S
	default:
		return d
	}
}

// PureFType is the llama-quantize positional type that --pure must use to
// emit tensors of type d. IQ2_S is a file-type recipe whose --pure default
// tensor type is IQ2_XS; IQ2_M is the ftype that writes IQ2_S payloads.
func (d DType) PureFType() DType {
	if d == DTypeIQ2_S {
		return DTypeIQ2_M
	}
	return d.BaseTensorType()
}

// BlockGeometry mirrors ggml type traits: tensors are quantized in blocks of
// BlockSize elements occupying TypeSize bytes. ExactBytes derives exact
// on-disk tensor costs from it.
type BlockGeometry struct {
	BlockSize uint64 `json:"blockSize"`
	TypeSize  uint64 `json:"typeSize"`
}

// BitsPerWeight returns BlockSize-normalized bpw (TypeSize*8/BlockSize).
func (g BlockGeometry) BitsPerWeight() float64 {
	return float64(g.TypeSize) * 8.0 / float64(g.BlockSize)
}

// ExactBytes returns the exact byte cost of storing elements weights in this
// geometry, padding the final block. Elements must be a multiple of nothing;
// partial final blocks round up.
func (g BlockGeometry) ExactBytes(elements uint64) uint64 {
	blocks := (elements + g.BlockSize - 1) / g.BlockSize
	return blocks * g.TypeSize
}

// geometryTable mirrors ggml type traits for every supported dtype. Recipe
// labels resolve through their base per-tensor type.
var geometryTable = map[DType]BlockGeometry{
	DTypeF32:  {1, 4},
	DTypeF16:  {1, 2},
	DTypeBF16: {1, 2},
	DTypeF64:  {1, 8},
	DTypeI8:   {1, 1},
	DTypeI16:  {1, 2},
	DTypeI32:  {1, 4},
	DTypeI64:  {1, 8},

	DTypeQ8_0:   {32, 34},
	DTypeQ8_1:   {32, 40},
	DTypeQ8_K:   {256, 292},
	DTypeQ6_K:   {256, 210},
	DTypeQ5_K_T: {256, 176},
	DTypeQ5_1:   {32, 24},
	DTypeQ5_0:   {32, 22},
	DTypeQ4_K_T: {256, 144},
	DTypeQ4_1:   {32, 20},
	DTypeQ4_0:   {32, 18},
	DTypeQ3_K:   {256, 110},
	DTypeQ2_K:   {256, 84},

	DTypeIQ4_NL:  {32, 18},
	DTypeIQ4_XS:  {32, 17},
	DTypeIQ3_XXS: {256, 98},
	DTypeIQ3_S:   {256, 110},
	DTypeIQ2_XXS: {256, 66},
	DTypeIQ2_XS:  {256, 74},
	DTypeIQ2_S:   {256, 82},
	DTypeIQ1_S:   {256, 50},
	DTypeIQ1_M:   {256, 56},
}

// Geometry returns the block geometry of d, resolving recipe labels to their
// base per-tensor type, and whether d has known geometry.
func (d DType) Geometry() (BlockGeometry, bool) {
	g, ok := geometryTable[d.BaseTensorType()]
	return g, ok
}

// ExactBytes returns the exact on-disk byte cost of elements weights stored
// as d. Recipe labels cost as their base type; upgrade mixing is accounted by
// the solver via TensorOption, not here.
func (d DType) ExactBytes(elements uint64) (uint64, bool) {
	g, ok := d.Geometry()
	if !ok {
		return 0, false
	}
	return g.ExactBytes(elements), true
}

// BitsPerWeight returns the exact bpw implied by block geometry.
func (d DType) BitsPerWeight() (float64, bool) {
	g, ok := d.Geometry()
	if !ok {
		return 0, false
	}
	return g.BitsPerWeight(), true
}

// Recipe is a whole-model quantization recipe label (the type argument to
// llama-quantize) together with its per-tensor resolution rules.
type Recipe struct {
	Label DType `json:"label"`
	// Base is the default per-tensor type for unclassed tensors.
	Base DType `json:"base"`
	// UpgradeV, when valid, is the per-tensor type used for attention V and
	// FFN down projections (the classic _M upgrade rule).
	UpgradeV DType `json:"upgradeV,omitempty"`
}

// RecipeFor returns the canonical recipe rules for a recipe label.
func RecipeFor(label DType) (Recipe, bool) {
	switch label {
	case DTypeQ4_K_M:
		return Recipe{Label: label, Base: DTypeQ4_K_T, UpgradeV: DTypeQ6_K}, true
	case DTypeQ4_K_S:
		return Recipe{Label: label, Base: DTypeQ4_K_T}, true
	case DTypeQ5_K_M:
		return Recipe{Label: label, Base: DTypeQ5_K_T, UpgradeV: DTypeQ6_K}, true
	case DTypeQ5_K_S:
		return Recipe{Label: label, Base: DTypeQ5_K_T}, true
	case DTypeQ3_K_L:
		return Recipe{Label: label, Base: DTypeQ3_K, UpgradeV: DTypeQ5_K_T}, true
	case DTypeQ3_K_M:
		return Recipe{Label: label, Base: DTypeQ3_K, UpgradeV: DTypeQ4_K_T}, true
	case DTypeQ3_K_S:
		return Recipe{Label: label, Base: DTypeQ3_K}, true
	case DTypeIQ2_M:
		return Recipe{Label: label, Base: DTypeIQ2_S}, true
	}
	// Non-label quant dtypes are uniform recipes.
	if label.IsQuant() && !label.IsRecipeLabel() {
		return Recipe{Label: label, Base: label.BaseTensorType()}, true
	}
	return Recipe{}, false
}
