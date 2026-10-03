package mediagen

import (
	"fmt"
	"strings"
)

// ValidateGeneration rejects inputs the loaded-model API has not explicitly
// advertised. Missing discovery permits only the existing basic image path;
// it does not advertise that path as a capability.
func (c *APICapabilities) ValidateGeneration(p GenerateParams) error {
	mode := "img_gen"
	if p.Kind == KindVideo {
		mode = "vid_gen"
	}
	if c.Known && !c.SupportsMode(mode) {
		return fmt.Errorf("loaded-model API does not advertise mode %s", mode)
	}
	if !c.Known && mode == "vid_gen" {
		return fmt.Errorf("video generation requires advertised vid_gen support: %s", c.Reason)
	}
	requested := []struct {
		feature string
		active  bool
	}{
		{"init_image", p.InitImage != "" || p.InitImagePath != ""},
		{"mask_image", p.MaskImage != "" || p.MaskImagePath != ""},
		{"control_image", p.ControlImage != "" || p.ControlImagePath != "" || p.ControlStrength != 0},
		{"ref_images", len(p.RefImages)+len(p.RefImagePaths) > 0},
		{"lora", len(p.Lora) > 0 || strings.Contains(strings.ToLower(p.Prompt), "<lora:") || strings.Contains(strings.ToLower(p.NegativePrompt), "<lora:")},
	}
	for _, f := range requested {
		if !f.active {
			continue
		}
		if mode == "vid_gen" && (f.feature == "mask_image" || f.feature == "control_image" || f.feature == "ref_images") {
			return fmt.Errorf("%s is not part of the vid_gen API", f.feature)
		}
		if !c.SupportsFeature(mode, f.feature) {
			if !c.Known {
				return fmt.Errorf("%s requires loaded-model API capability advertisement: %s", f.feature, c.Reason)
			}
			return fmt.Errorf("loaded-model API does not advertise %s for %s", f.feature, mode)
		}
	}
	for _, choice := range []struct {
		name, value string
		choices     []string
	}{{"sampler", p.Sampler, c.Samplers}, {"scheduler", p.Scheduler, c.Schedulers}} {
		if choice.value != "" && c.Known && !containsChoice(choice.choices, choice.value) {
			return fmt.Errorf("%s %q is not advertised by the loaded-model API", choice.name, choice.value)
		}
	}
	if formats, ok := c.OutputFormatsByMode[mode]; ok && p.OutputFormat != "" && !containsChoice(formats, p.OutputFormat) {
		return fmt.Errorf("output_format %q is not advertised for %s", p.OutputFormat, mode)
	}
	l := c.Limits
	for _, d := range []struct {
		name            string
		value, min, max int
	}{{"width", p.Width, l.MinWidth, l.MaxWidth}, {"height", p.Height, l.MinHeight, l.MaxHeight}} {
		if d.value > 0 && ((d.min > 0 && d.value < d.min) || (d.max > 0 && d.value > d.max)) {
			return fmt.Errorf("%s exceeds advertised loaded-model limits", d.name)
		}
	}
	if l.MaxBatchCount > 0 && p.BatchCount > l.MaxBatchCount {
		return fmt.Errorf("batch_count exceeds advertised loaded-model limit")
	}
	return nil
}

func containsChoice(choices []string, value string) bool {
	for _, choice := range choices {
		if strings.EqualFold(choice, value) {
			return true
		}
	}
	return false
}

func canFallbackGeneration(p GenerateParams) bool {
	return p.Kind != KindVideo && p.InitImage == "" && p.InitImagePath == "" && p.MaskImage == "" && p.MaskImagePath == "" &&
		p.ControlImage == "" && p.ControlImagePath == "" && len(p.RefImages)+len(p.RefImagePaths) == 0 && len(p.Lora) == 0 &&
		p.Guidance == 0 && !p.GuidanceExplicit && p.Steps == 0 && p.CFGScale == 0 && !p.CFGScaleExplicit && p.Sampler == "" && p.Scheduler == "" && p.NegativePrompt == "" && p.Seed < 0
}
