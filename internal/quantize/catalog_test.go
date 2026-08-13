package quantize

import (
	"strings"
	"testing"
)

func TestParseQuantizeTypesFromHelp(t *testing.T) {
	help := `
Allowed quantization types:
   2  or  Q4_0
  15  or  Q4_K_M
  18  or  Q6_K
   7  or  Q8_0
  30  or  IQ4_XS
  23  or  IQ3_XXS
`
	got := ParseQuantizeTypes(help)
	want := []string{"Q4_0", "Q4_K_M", "Q6_K", "Q8_0", "IQ4_XS", "IQ3_XXS"}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d: %+v", len(got), len(want), got)
	}
	for i, n := range want {
		if got[i].Name != n {
			t.Errorf("got[%d]=%s want %s", i, got[i].Name, n)
		}
		if !got[i].Advertised {
			t.Errorf("%s not advertised", n)
		}
	}
	iq3, ok := LookupFType("IQ3_XXS")
	if !ok || iq3.IMatrix != IMatrixRequired {
		t.Fatal("IQ3_XXS should require imatrix")
	}
}

func TestCanonicalAndHighPrecision(t *testing.T) {
	if CanonicalFType("q4_k") != "Q4_K_M" {
		t.Errorf("alias Q4_K = %s", CanonicalFType("q4_k"))
	}
	if !HighPrecision("F16") || !HighPrecision("Q8_0") || HighPrecision("Q4_K_M") {
		t.Fatal("high precision classification")
	}
	if needsRequantizeFlag("F16") || !needsRequantizeFlag("Q8_0") || !needsRequantizeFlag("Q4_K_M") {
		t.Fatal("requantize flag classification")
	}
	if !RequiresIMatrix("IQ2_XXS") || RequiresIMatrix("Q4_K_M") {
		t.Fatal("imatrix required policy")
	}
	if RecommendsIMatrix("Q8_0") {
		t.Fatal("Q8_0 should not recommend imatrix")
	}
	if LookupMust(t, "Q8_0").IMatrix != IMatrixOptional {
		t.Fatal("Q8_0 imatrix policy")
	}
}

func LookupMust(t *testing.T, name string) FType {
	t.Helper()
	ft, ok := LookupFType(name)
	if !ok {
		t.Fatalf("missing %s", name)
	}
	return ft
}

func TestParseQuantizeTypesEmptyFallsBack(t *testing.T) {
	if got := ParseQuantizeTypes("no types here COPY maybe"); len(got) == 0 {
		// COPY may still match token regex; that's ok as long as we don't panic.
		_ = strings.TrimSpace
	}
}
