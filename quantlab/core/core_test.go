package core

import (
	"strings"
	"testing"
)

func validTensor() TensorDesc {
	return TensorDesc{
		Name:     "blk.0.ffn_down.weight",
		DType:    DTypeF16,
		Shape:    []uint64{4096, 11008},
		Offset:   0,
		Length:   4096 * 11008 * 2,
		Elements: 4096 * 11008,
	}
}

func TestTensorDescValidate(t *testing.T) {
	if err := validTensor().Validate(); err != nil {
		t.Fatalf("valid tensor rejected: %v", err)
	}
	cases := []struct {
		name string
		mut  func(*TensorDesc)
	}{
		{"empty name", func(d *TensorDesc) { d.Name = "" }},
		{"bad dtype", func(d *TensorDesc) { d.DType = "Q9_X" }},
		{"empty shape", func(d *TensorDesc) { d.Shape = nil }},
		{"zero dim", func(d *TensorDesc) { d.Shape = []uint64{0, 8} }},
		{"element mismatch", func(d *TensorDesc) { d.Elements++ }},
		{"zero length", func(d *TensorDesc) { d.Length = 0 }},
	}
	for _, c := range cases {
		d := validTensor()
		c.mut(&d)
		if err := d.Validate(); err == nil {
			t.Errorf("%s: expected error", c.name)
		}
	}
}

func TestValidateTensorName(t *testing.T) {
	if err := ValidateTensorName("blk.0.attn_q.weight"); err != nil {
		t.Fatalf("valid name rejected: %v", err)
	}
	bad := map[string]string{
		"empty":             "",
		"newline injection": "blk.0.weight\nfake.tensor Q8_0",
		"cr injection":      "blk.0.weight\rQ8_0",
		"leading space":     " blk.0.weight",
		"trailing space":    "blk.0.weight ",
		"embedded space":    "blk.0 weight",
		"tab":               "blk.0\tweight",
		"bell control":      "blk.0\aweight",
		"del control":       "blk.0\x7fweight",
		"unicode line sep":  "blk.0 weight",
		"unicode nel":       "blk.0weight",
		"zero-width joiner": "blk.0‍weight",
		"too long":          strings.Repeat("a", MaxTensorNameLen+1),
	}
	for name, tn := range bad {
		if err := ValidateTensorName(tn); err == nil {
			t.Errorf("%s: name %q accepted", name, tn)
		}
		d := validTensor()
		d.Name = tn
		if err := d.Validate(); err == nil {
			t.Errorf("%s: TensorDesc with name %q accepted", name, tn)
		}
		o := TensorOption{TensorName: tn, Target: DTypeQ8_0, Bytes: 64}
		if err := o.Validate(); err == nil {
			t.Errorf("%s: TensorOption with name %q accepted", name, tn)
		}
	}
}

func TestTensorBankValidateDuplicate(t *testing.T) {
	b := &TensorBank{SourcePath: "/m.gguf", Tensors: []TensorDesc{validTensor(), validTensor()}}
	if err := b.Validate(); err == nil {
		t.Fatal("duplicate tensor names accepted")
	}
}

func TestQuantizable(t *testing.T) {
	m2d := validTensor()
	if !m2d.Quantizable() {
		t.Error("2D weight not quantizable")
	}
	norm := TensorDesc{Name: "n", DType: DTypeF32, Shape: []uint64{4096}, Length: 4096 * 4, Elements: 4096}
	if norm.Quantizable() {
		t.Error("1D norm marked quantizable")
	}
	odd := TensorDesc{Name: "o", DType: DTypeF16, Shape: []uint64{17, 8}, Length: 17 * 8 * 2, Elements: 136}
	if odd.Quantizable() {
		t.Error("unaligned 2D tensor marked quantizable")
	}
}

func TestAnchorMatches(t *testing.T) {
	cases := []struct {
		a    Anchor
		name string
		want bool
	}{
		{Anchor{Kind: AnchorExplicit, Name: "blk.0.attn_q.weight", MinDType: DTypeQ8_0}, "blk.0.attn_q.weight", true},
		{Anchor{Kind: AnchorNorm, Pattern: "*_norm.weight", MinDType: DTypeQ8_0}, "blk.3.ffn_norm.weight", true},
		{Anchor{Kind: AnchorNorm, Pattern: "*_norm.weight", MinDType: DTypeQ8_0}, "blk.3.attn_q.weight", false},
		{Anchor{Kind: AnchorEmbedding, Pattern: "token_embd*", MinDType: DTypeQ8_0}, "token_embd.weight", true},
		{Anchor{Kind: AnchorEmbedding, Pattern: "*attn*", MinDType: DTypeQ8_0}, "blk.1.attn_v.weight", true},
		{Anchor{Kind: AnchorEmbedding, Pattern: "*", MinDType: DTypeQ8_0}, "anything", true},
	}
	for _, c := range cases {
		if got := c.a.Matches(c.name); got != c.want {
			t.Errorf("pattern %q vs %q: got %v want %v", c.a.Pattern+c.a.Name, c.name, got, c.want)
		}
	}
}

func TestAnchorValidate(t *testing.T) {
	if err := (Anchor{Kind: AnchorNorm, Pattern: "*_norm*", MinDType: DTypeQ8_0}).Validate(); err != nil {
		t.Fatalf("valid anchor rejected: %v", err)
	}
	for _, a := range []Anchor{
		{Kind: AnchorNorm, MinDType: DTypeQ8_0},                // no name/pattern
		{Kind: "bogus", Name: "x", MinDType: DTypeQ8_0},        // bad kind
		{Kind: AnchorNorm, Name: "x", MinDType: DType("NOPE")}, // bad dtype
	} {
		if err := a.Validate(); err == nil {
			t.Errorf("expected error for %+v", a)
		}
	}
}

func TestProfileValidate(t *testing.T) {
	bank := &TensorBank{SourcePath: "/m.gguf", Tensors: []TensorDesc{validTensor()}}
	p := &Profile{
		ID: "p1",
		Assignments: []QuantAssignment{
			{TensorName: "blk.0.ffn_down.weight", Target: DTypeQ4_K, BitsPerWeight: 4.5},
		},
	}
	if err := p.Validate(bank); err != nil {
		t.Fatalf("valid profile rejected: %v", err)
	}
	p.Assignments[0].TensorName = "missing.tensor"
	if err := p.Validate(bank); err == nil {
		t.Fatal("unknown tensor accepted")
	}
	p.Assignments[0].TensorName = "blk.0.ffn_down.weight"
	p.Assignments[0].BitsPerWeight = 0
	if err := p.Validate(bank); err == nil {
		t.Fatal("zero bpw accepted")
	}
}

func TestStageOrderAndValidity(t *testing.T) {
	if len(StageOrder) != 7 {
		t.Fatalf("expected 7 stages, got %d", len(StageOrder))
	}
	for i, s := range StageOrder {
		if StageIndex(s) != i || !s.Valid() {
			t.Errorf("stage %q index/valid mismatch", s)
		}
	}
	if Stage("bogus").Valid() {
		t.Error("bogus stage valid")
	}
}

func TestDTypeClassification(t *testing.T) {
	if !DTypeQ4_K.IsQuant() || DTypeF16.IsQuant() {
		t.Error("IsQuant misclassified")
	}
	if DType("Q3_K_X").Valid() {
		t.Error("unlisted dtype accepted")
	}
	for _, d := range QuantDTypes {
		if !d.IsQuant() {
			t.Errorf("%q in QuantDTypes but not quant", d)
		}
	}
}

func TestDTypeRecipeDistinction(t *testing.T) {
	if !DTypeQ4_K_M.IsRecipeLabel() || !DTypeQ3_K_L.IsRecipeLabel() || !DTypeIQ2_M.IsRecipeLabel() {
		t.Error("recipe labels not detected")
	}
	if DTypeQ6_K.IsRecipeLabel() || DTypeQ8_0.IsRecipeLabel() || DTypeIQ2_XXS.IsRecipeLabel() || DTypeIQ2_S.IsRecipeLabel() {
		t.Error("per-tensor type misclassified as recipe label")
	}
	if got := DTypeQ4_K_M.BaseTensorType(); got != DTypeQ4_K_T {
		t.Errorf("Q4_K_M base = %q, want Q4_K", got)
	}
	if got := DTypeQ3_K_S.BaseTensorType(); got != DTypeQ3_K {
		t.Errorf("Q3_K_S base = %q, want Q3_K", got)
	}
	if got := DTypeIQ2_XS.BaseTensorType(); got != DTypeIQ2_XS {
		t.Errorf("IQ2_XS base = %q, want identity", got)
	}
	if got := DTypeIQ2_M.BaseTensorType(); got != DTypeIQ2_S {
		t.Errorf("IQ2_M base = %q, want IQ2_S", got)
	}
}

func TestPureFType(t *testing.T) {
	if got := DTypeIQ2_S.PureFType(); got != DTypeIQ2_M {
		t.Fatalf("IQ2_S PureFType = %q, want IQ2_M", got)
	}
	if got := DTypeQ4_K_T.PureFType(); got != DTypeQ4_K_T {
		t.Fatalf("Q4_K PureFType = %q, want Q4_K", got)
	}
	if got := DTypeIQ2_XS.PureFType(); got != DTypeIQ2_XS {
		t.Fatalf("IQ2_XS PureFType = %q, want IQ2_XS", got)
	}
	if got := DTypeIQ2_M.PureFType(); got != DTypeIQ2_S {
		t.Fatalf("IQ2_M PureFType = %q, want IQ2_S (base)", got)
	}
}

func TestDTypeGeometryAndExactBytes(t *testing.T) {
	for _, tc := range []struct {
		d DType
		g BlockGeometry
	}{
		{DTypeF32, BlockGeometry{1, 4}}, {DTypeF16, BlockGeometry{1, 2}}, {DTypeBF16, BlockGeometry{1, 2}}, {DTypeF64, BlockGeometry{1, 8}},
		{DTypeI8, BlockGeometry{1, 1}}, {DTypeI16, BlockGeometry{1, 2}}, {DTypeI32, BlockGeometry{1, 4}}, {DTypeI64, BlockGeometry{1, 8}},
		{DTypeQ4_0, BlockGeometry{32, 18}}, {DTypeQ4_1, BlockGeometry{32, 20}}, {DTypeQ5_0, BlockGeometry{32, 22}}, {DTypeQ5_1, BlockGeometry{32, 24}},
		{DTypeQ8_0, BlockGeometry{32, 34}}, {DTypeQ8_1, BlockGeometry{32, 40}}, {DTypeQ2_K, BlockGeometry{256, 84}}, {DTypeQ3_K, BlockGeometry{256, 110}},
		{DTypeQ4_K_T, BlockGeometry{256, 144}}, {DTypeQ5_K_T, BlockGeometry{256, 176}}, {DTypeQ6_K, BlockGeometry{256, 210}}, {DTypeQ8_K, BlockGeometry{256, 292}},
		{DTypeIQ2_XXS, BlockGeometry{256, 66}}, {DTypeIQ2_XS, BlockGeometry{256, 74}}, {DTypeIQ3_XXS, BlockGeometry{256, 98}}, {DTypeIQ1_S, BlockGeometry{256, 50}},
		{DTypeIQ4_NL, BlockGeometry{32, 18}}, {DTypeIQ3_S, BlockGeometry{256, 110}}, {DTypeIQ2_S, BlockGeometry{256, 82}}, {DTypeIQ4_XS, BlockGeometry{32, 17}}, {DTypeIQ1_M, BlockGeometry{256, 56}},
	} {
		got, ok := tc.d.Geometry()
		if !ok || got != tc.g {
			t.Errorf("%s geometry = %+v,%v want %+v", tc.d, got, ok, tc.g)
		}
	}
	if DTypeIQ3_XS.Valid() {
		t.Error("stale IQ3_XS accepted without a raw GGML type ID")
	}
	// Q4_K: 256-element blocks of 144 bytes.
	b, ok := DTypeQ4_K_T.ExactBytes(512)
	if !ok || b != 288 {
		t.Fatalf("Q4_K 512 elems = %d bytes (ok=%v), want 288", b, ok)
	}
	// Partial final block rounds up.
	b, _ = DTypeQ4_K_T.ExactBytes(257)
	if b != 288 {
		t.Fatalf("Q4_K 257 elems = %d, want 288 (2 blocks)", b)
	}
	// Q8_0: 32/34.
	b, _ = DTypeQ8_0.ExactBytes(32)
	if b != 34 {
		t.Fatalf("Q8_0 block = %d, want 34", b)
	}
	// F16 exactness.
	b, _ = DTypeF16.ExactBytes(1000)
	if b != 2000 {
		t.Fatalf("F16 1000 elems = %d, want 2000", b)
	}
	// Recipe labels cost as base type.
	b, _ = DTypeQ4_K_M.ExactBytes(256)
	if b != 144 {
		t.Fatalf("Q4_K_M 256 elems = %d, want 144", b)
	}
	bpw, ok := DTypeQ6_K.BitsPerWeight()
	if !ok || bpw != 6.5625 {
		t.Fatalf("Q6_K bpw = %v, want 6.5625", bpw)
	}
}

func TestRecipeFor(t *testing.T) {
	r, ok := RecipeFor(DTypeQ4_K_M)
	if !ok || r.Base != DTypeQ4_K_T || r.UpgradeV != DTypeQ6_K {
		t.Fatalf("Q4_K_M recipe = %+v", r)
	}
	r, ok = RecipeFor(DTypeQ4_K_S)
	if !ok || r.UpgradeV != "" {
		t.Fatalf("Q4_K_S recipe should have no upgrade: %+v", r)
	}
	r, ok = RecipeFor(DTypeQ8_0)
	if !ok || r.Base != DTypeQ8_0 {
		t.Fatalf("Q8_0 uniform recipe = %+v", r)
	}
	if _, ok := RecipeFor(DTypeF16); ok {
		t.Fatal("float recipe accepted")
	}
}

func TestRequiresImatrix(t *testing.T) {
	if !DTypeIQ4_XS.RequiresImatrix() || !DTypeIQ1_S.RequiresImatrix() || !DTypeIQ2_M.RequiresImatrix() {
		t.Error("IQ types must require imatrix")
	}
	if DTypeQ4_K_T.RequiresImatrix() || DTypeF16.RequiresImatrix() {
		t.Error("K/float types must not require imatrix")
	}
}
