package mediagen

import "testing"

func TestNativeRequestPreservesLiteralSeedAndZeroGuidance(t *testing.T) {
	p := GenerateParams{Prompt: "a tree", Seed: 0, CFGScaleExplicit: true, GuidanceExplicit: true}
	ValidateGenerateParams(&p)
	body := SDRequestBody(p)
	if body["seed"] != int64(0) {
		t.Fatal("literal zero seed became random")
	}
	guidance := body["sample_params"].(map[string]any)["guidance"].(map[string]any)
	if guidance["txt_cfg"] != float64(0) || guidance["distilled_guidance"] != float64(0) {
		t.Fatalf("zero guidance dropped: %v", guidance)
	}
	if canFallbackGeneration(p) {
		t.Fatal("OpenAI fallback cannot represent explicit native sampling fields")
	}
	unset := SDRequestBody(GenerateParams{Prompt: "a tree", Seed: -1})
	if unset["sample_params"] != nil {
		t.Fatal("omitted guidance must preserve runtime defaults")
	}
}
