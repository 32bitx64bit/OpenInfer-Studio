package workflow

import (
	"testing"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

type capableEnv struct {
	*fakeEnv
	caps *mediagen.APICapabilities
}

func (e capableEnv) GenerationCapabilities(string) (*mediagen.APICapabilities, error) {
	return e.caps, nil
}

func imageCapabilityEnv(features map[string]bool) capableEnv {
	return capableEnv{newFakeEnv(), &mediagen.APICapabilities{Known: true, SupportedModes: []string{"img_gen"}, FeaturesByMode: map[string]map[string]bool{"img_gen": features}, Samplers: []string{"euler"}, Schedulers: []string{"karras"}}}
}

func maskGraph() Graph {
	g := txt2img()
	g.Nodes = append(g.Nodes, node("n9", "image.load", map[string]any{"path": "/img/fisherman.png"}), node("n10", "mask.load", map[string]any{"path": "/img/mask.png"}))
	g.Edges[5] = wire("n9", "image", "n5", "start")
	g.Edges = append(g.Edges, wire("n10", "mask", "n5", "mask"))
	return g
}

func TestInpaintingPlansOnlyWithAdvertisedMaskSupport(t *testing.T) {
	env := imageCapabilityEnv(map[string]bool{"init_image": true, "mask_image": true})
	env.files["/img/mask.png"] = FileMeta{Size: 200, Width: 640, Height: 800}
	plan, issues := Build(maskGraph(), NewRegistry(), env, allCaps, Options{})
	if plan == nil || HasErrors(issues) {
		t.Fatalf("supported inpainting: %+v", issues)
	}
	gs := plan.Stages[0].Generate
	if gs.Mask == nil || gs.Params.MaskImagePath != "/img/mask.png" || gs.Init == nil {
		t.Fatalf("mask not forwarded: %+v", gs)
	}
	env.caps.FeaturesByMode["img_gen"]["mask_image"] = false
	if plan, issues := Build(maskGraph(), NewRegistry(), env, allCaps, Options{}); plan != nil || !hasIssue(issues, "plan.api_capability") {
		t.Fatalf("unsupported mask accepted: %+v", issues)
	}
	env.caps.FeaturesByMode["img_gen"]["mask_image"] = true
	env.files["/img/mask.png"] = FileMeta{Size: 200, Width: 32, Height: 32}
	if plan, issues := Build(maskGraph(), NewRegistry(), env, allCaps, Options{}); plan != nil || !hasIssue(issues, "plan.mask_size") {
		t.Fatalf("mismatched mask accepted: %+v", issues)
	}
}

func TestControlAndReferenceInputsUseNativeFields(t *testing.T) {
	env := imageCapabilityEnv(map[string]bool{"control_image": true, "ref_images": true})
	g := txt2img()
	g.Nodes = append(g.Nodes, node("n9", "image.load", map[string]any{"path": "/img/fisherman.png"}))
	g.Edges = append(g.Edges, wire("n9", "image", "n5", "control"), wire("n9", "image", "n5", "reference"))
	plan, issues := Build(g, NewRegistry(), env, allCaps, Options{})
	if plan == nil {
		t.Fatalf("supported inputs rejected: %+v", issues)
	}
	gs := plan.Stages[0].Generate
	if gs.Control == nil || gs.Reference == nil || gs.Params.ControlStrength != 0.9 || len(gs.Params.RefImagePaths) != 1 {
		t.Fatalf("native input missing: %+v", gs)
	}
}

func TestLoadedModelRejectsUnadvertisedSamplerBeforeQueue(t *testing.T) {
	env := imageCapabilityEnv(map[string]bool{})
	g := txt2img()
	g.Nodes[4].Params["sampler"] = "unsupported"
	plan, issues := Build(g, NewRegistry(), env, allCaps, Options{})
	if plan != nil || !hasIssue(issues, "plan.api_capability") {
		t.Fatalf("unadvertised sampler accepted: %+v", issues)
	}
}

func TestNativeUpscaleAcceptsOnlyAdvertisedRGBModels(t *testing.T) {
	env := imageCapabilityEnv(map[string]bool{})
	env.caps.Upscale = true
	env.caps.Upscalers = []mediagen.UpscalerEntry{{Name: "rgb", Model: true, ImageUpscale: true}, {Name: "latent", Model: true}}
	g := txt2img()
	g.Nodes = append(g.Nodes, node("n9", "image.upscale", map[string]any{"upscaler": "rgb"}))
	g.Edges[6] = wire("n9", "image", "n6", "image")
	g.Edges = append(g.Edges, wire("n1", "model", "n9", "model"), wire("n5", "image", "n9", "image"))
	plan, issues := Build(g, NewRegistry(), env, allCaps, Options{})
	if plan == nil || len(plan.Stages) != 3 || plan.Stages[1].Upscale == nil {
		t.Fatalf("native upscale rejected: %+v", issues)
	}
	g.Nodes[len(g.Nodes)-1].Params["upscaler"] = "latent"
	if plan, issues := Build(g, NewRegistry(), env, allCaps, Options{}); plan != nil || !hasIssue(issues, "plan.upscaler") {
		t.Fatalf("latent upscaler accepted: %+v", issues)
	}
}
