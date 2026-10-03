package mediagen

import (
	"encoding/base64"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenerationInputValidationRejectsInvalidOrAmbiguousInputs(t *testing.T) {
	raw, path := pngInput(t)
	inline := base64.StdEncoding.EncodeToString(raw)
	for _, p := range []GenerateParams{
		{MaskImagePath: path}, {InitImagePath: path, InitImage: inline}, {ControlImage: "invalid-base64"},
		{InitImage: base64.StdEncoding.EncodeToString([]byte("not an image"))},
		{RefImages: []string{inline}, RefImagePaths: []string{path}}, {RefImagePaths: []string{""}},
		{RefImages: make([]string, maxReferenceImages+1)}, {InitImagePath: "relative.png"},
		{InitImage: strings.Repeat("A", maxInitImageInlineBytes+1)},
		{Guidance: math.NaN()}, {ControlStrength: math.Inf(1)}, {ControlStrength: 0.8},
		{ControlImagePath: path, ControlStrength: 11}, {Lora: []Lora{{Path: "model", Multiplier: math.NaN()}}},
	} {
		if err := validateGenerationInputs(p); err == nil {
			t.Fatalf("invalid inputs accepted: %+v", p)
		}
	}
	bad := filepath.Join(t.TempDir(), "wrong.jpg")
	if err := os.WriteFile(bad, raw, 0600); err != nil {
		t.Fatal(err)
	}
	if err := validateGenerationInputs(GenerateParams{InitImagePath: bad}); err == nil {
		t.Fatal("extension/content mismatch accepted")
	}
	large := filepath.Join(t.TempDir(), "large.png")
	f, err := os.Create(large)
	if err != nil {
		t.Fatal(err)
	}
	if err = f.Truncate(maxInitImagePathBytes + 1); err != nil {
		t.Fatal(err)
	}
	_ = f.Close()
	if err = validateGenerationInputs(GenerateParams{ControlImagePath: large}); err == nil {
		t.Fatal("oversized local file accepted")
	}
}

func TestSDRequestBodyPreservesConditioningZeroesAndGuidance(t *testing.T) {
	p := GenerateParams{InitImage: "init", MaskImage: "mask", ControlImage: "control", RefImages: []string{"ref"}, Guidance: 3.5, Lora: []Lora{{Path: "film", Multiplier: 0}}}
	body := SDRequestBody(p)
	if v, ok := body["strength"]; !ok || v != float64(0) {
		t.Fatal("zero denoising strength dropped")
	}
	if v, ok := body["control_strength"]; !ok || v != float64(0) {
		t.Fatal("zero control strength dropped")
	}
	if body["sample_params"].(map[string]any)["guidance"].(map[string]any)["distilled_guidance"] != 3.5 {
		t.Fatal("guidance dropped")
	}
	if body["lora"].([]Lora)[0].Multiplier != 0 {
		t.Fatal("zero lora multiplier changed")
	}
	p.Kind = KindVideo
	body = SDRequestBody(p)
	if body["init_image"] == nil {
		t.Fatal("video init_image dropped")
	}
	for _, field := range []string{"mask_image", "control_image", "ref_images", "batch_count"} {
		if _, ok := body[field]; ok {
			t.Fatalf("image-only field %s forwarded to vid_gen", field)
		}
	}
}

func TestPromptLoraConversionValidatesLocalFilesAndTags(t *testing.T) {
	path := filepath.Join(t.TempDir(), "film.safetensors")
	if err := os.WriteFile(path, []byte("weights"), 0600); err != nil {
		t.Fatal(err)
	}
	c := normalizeAPICapabilities(nil, "m", "")
	c.Loras = []LoraEntry{{Name: "film", Path: "film.safetensors"}}
	p, err := convertPromptLora(GenerateParams{Prompt: "portrait <lora:film:0>"}, c, LoadSettings{LoraModelDir: filepath.Dir(path)})
	if err != nil || len(p.Lora) != 1 || p.Lora[0].Path != "film.safetensors" || p.Lora[0].Multiplier != 0 || p.Prompt != "portrait" {
		t.Fatalf("params=%+v err=%v", p, err)
	}
	for _, prompt := range []string{"p <lora:film>", "p <lora:film:NaN>", "p <lora:film:no>", "p <lora:film:1", "p <lora:../film:1>"} {
		if _, err := convertPromptLora(GenerateParams{Prompt: prompt}, c, LoadSettings{LoraModelDir: filepath.Dir(path)}); err == nil {
			t.Fatalf("invalid tag accepted: %s", prompt)
		}
	}
	if _, err := convertPromptLora(GenerateParams{Prompt: "p", Lora: []Lora{{Path: path + "missing", Multiplier: 1}}}, c, LoadSettings{}); err == nil {
		t.Fatal("missing local lora accepted")
	}
}
