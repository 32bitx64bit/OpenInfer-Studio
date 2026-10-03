package workflow

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

type apiEnv interface {
	GenerationCapabilities(string) (*mediagen.APICapabilities, error)
}

type capabilityResult struct {
	caps *mediagen.APICapabilities
	err  error
}

// A planning pass queries each loaded model once, including failed discovery.
func (b *builder) modelAPICapabilities(model string) (*mediagen.APICapabilities, error) {
	if result, ok := b.apiCaps[model]; ok {
		return result.caps, result.err
	}
	provider, ok := b.env.(apiEnv)
	if !ok {
		return nil, fmt.Errorf("loaded-model API capability discovery unavailable")
	}
	caps, err := provider.GenerationCapabilities(model)
	if b.apiCaps == nil {
		b.apiCaps = map[string]capabilityResult{}
	}
	b.apiCaps[model] = capabilityResult{caps, err}
	return caps, err
}

func (b *builder) requireFeature(n Node, model, mode, feature string) bool {
	_, ok := b.env.(apiEnv)
	if !ok {
		b.errorf("plan.api_capability", n.ID, "", "%s requires loaded-model API capability discovery", feature)
		return false
	}
	caps, err := b.modelAPICapabilities(model)
	if err != nil {
		b.errorf("plan.api_capability", n.ID, "", "cannot discover %s: %v", feature, err)
		return false
	}
	if caps == nil || !caps.SupportsFeature(mode, feature) {
		reason := "the loaded model does not advertise this feature"
		if caps != nil && caps.Reason != "" {
			reason = caps.Reason
		}
		b.errorf("plan.api_capability", n.ID, "", "%s is unavailable for %s: %s. Load a compatible model first.", feature, mode, reason)
		return false
	}
	return true
}

func (b *builder) sampleInputs(n Node, st *Stage, model string, video bool) {
	gs := st.Generate
	mode := "img_gen"
	if video {
		mode = "vid_gen"
	}
	if video && gs.Init != nil {
		b.requireFeature(n, model, mode, "init_image")
	}
	for _, entry := range []struct {
		port, feature string
		target        **ImageRef
	}{{"mask", "mask_image", &gs.Mask}, {"control", "control_image", &gs.Control}, {"reference", "ref_images", &gs.Reference}} {
		up, _, ok := b.upstream(n, entry.port)
		if !ok {
			continue
		}
		src, ok := b.imgs[up.ID]
		if !ok {
			continue
		}
		if !b.requireFeature(n, model, mode, entry.feature) {
			continue
		}
		if entry.port == "mask" {
			if gs.Init == nil {
				b.errorf("plan.mask_start", n.ID, "mask", "inpainting requires an image connected to start")
				continue
			}
			if src.w > 0 && gs.Params.Width > 0 && (src.w != gs.Params.Width || src.h != gs.Params.Height) {
				b.errorf("plan.mask_size", n.ID, "mask", "mask dimensions must match the starting image")
				continue
			}
		}
		ref := src.ref
		*entry.target = &ref
		if ref.Stage != "" {
			st.Deps = append(st.Deps, ref.Stage)
		}
		switch entry.port {
		case "mask":
			gs.Params.MaskImagePath = ref.Path
		case "control":
			gs.Params.ControlImagePath = ref.Path
			gs.Params.ControlStrength = b.effective(n).num("control_strength")
		case "reference":
			if ref.Path != "" {
				gs.Params.RefImagePaths = []string{ref.Path}
			}
		}
	}
	if _, ok := b.env.(apiEnv); ok {
		caps, err := b.modelAPICapabilities(model)
		if err != nil {
			b.errorf("plan.api_capability", n.ID, "", "capability discovery failed: %v", err)
		} else if caps != nil && caps.Known {
			if err := caps.ValidateGeneration(gs.Params); err != nil {
				b.errorf("plan.api_capability", n.ID, "", "%v", err)
			}
		} else if len(gs.Params.Lora) > 0 {
			b.errorf("plan.api_capability", n.ID, "model", "load a model that advertises structured LoRA support first")
		}
	}
}

type UpscaleStage struct {
	Server string                 `json:"server"`
	Src    ImageRef               `json:"src"`
	Params mediagen.UpscaleParams `json:"params"`
}

type upscaleRunner interface {
	RunUpscale(context.Context, string, mediagen.UpscaleParams, *mediagen.LoadOverrides) (*mediagen.Job, error)
}

func upscaleSpec() NodeSpec {
	return NodeSpec{Type: "image.upscale", Title: "Model Upscale", Category: CatImage, APIFeature: "upscale", Description: "Use a compatible RGB upscaler advertised by the loaded model. Follow with Resize to set explicit dimensions.", Inputs: []Port{{Name: "model", Type: PortTypes{TypeModel}, Required: true}, {Name: "image", Type: PortTypes{TypeImage}, Required: true}}, Outputs: []Port{{Name: "image", Type: PortTypes{TypeImage}}}, Params: []ParamSpec{{Name: "upscaler", Kind: ParamEnum, Required: true, From: "capabilities.upscalers"}, {Name: "repeats", Kind: ParamInt, Default: 1, Min: ptr(1), Max: ptr(4)}, {Name: "tile_size", Kind: ParamInt, Default: 0, Min: ptr(0), Max: ptr(2048)}}}
}

func (b *builder) upscale(n Node) {
	ck, _, ok := b.upstream(n, "model")
	if !ok {
		return
	}
	for ck.Type == "lora.load" {
		ck, _, _ = b.upstream(ck, "model")
	}
	if ck.Type != "checkpoint.load" {
		b.errorf("plan.model_chain", n.ID, "model", "Model Upscale needs a checkpoint model")
		return
	}
	ref, _ := b.effective(ck)["model"].(map[string]any)
	id, _ := ref["library_id"].(string)
	info := b.model(ck.ID, id)
	if info == nil {
		return
	}
	up, _, ok := b.upstream(n, "image")
	if !ok {
		return
	}
	src, ok := b.imgs[up.ID]
	if !ok {
		return
	}
	_, ok = b.env.(apiEnv)
	if !ok {
		b.errorf("plan.api_capability", n.ID, "", "load a model with native RGB upscaling support first")
		return
	}
	caps, err := b.modelAPICapabilities(id)
	if err != nil || caps == nil || !caps.Known || !caps.Upscale {
		b.errorf("plan.api_capability", n.ID, "", "the loaded model does not advertise native RGB upscaling")
		return
	}
	p := b.effective(n)
	name := p.str("upscaler")
	if name != "" {
		allowed := false
		for _, u := range caps.Upscalers {
			allowed = allowed || (u.Name == name && u.Model && u.ImageUpscale)
		}
		if !allowed {
			b.errorf("plan.upscaler", n.ID, "", "upscaler %q is not advertised as a compatible RGB model", name)
			return
		}
	}
	sig := signature(id, mediagen.LoadOverrides{})
	b.useServer(ServerNeed{Signature: sig, ModelID: id, ModelName: info.Name})
	st := b.newStage(n, StageUpscale)
	st.Upscale = &UpscaleStage{Server: sig, Src: src.ref, Params: mediagen.UpscaleParams{Upscaler: name, Repeats: p.integer("repeats"), TileSize: p.integer("tile_size"), OutputFormat: "png"}}
	if src.ref.Stage != "" {
		st.Deps = []string{src.ref.Stage}
	}
	// Scale is not advertised by every runtime. Unknown dimensions stay unknown.
	b.imgs[n.ID] = imgInfo{ref: ImageRef{Stage: st.ID}}
}

func (e *Executor) runUpscale(r *run, st Stage, servers map[string]ServerNeed) ([]string, string, string, error) {
	runner, ok := e.runner.(upscaleRunner)
	if !ok {
		return nil, "", "", fmt.Errorf("native upscaling unavailable")
	}
	need, ok := servers[st.Upscale.Server]
	if !ok {
		return nil, "", "", fmt.Errorf("unknown upscale server")
	}
	path, err := e.resolveImage(r, st.Upscale.Src)
	if err != nil {
		return nil, "", "", err
	}
	p := st.Upscale.Params
	p.ImagePath = path
	job, err := runner.RunUpscale(r.ctx, need.ModelID, p, &need.Overrides)
	if err != nil {
		return nil, "", "", err
	}
	if job == nil || len(job.OutputPaths) == 0 {
		return nil, "", "", fmt.Errorf("upscale returned no image")
	}
	return job.OutputPaths, job.ID, "upscaled", nil
}

func (b *builder) structuredLoras(n Node, model string, loras []Node) []mediagen.Lora {
	if len(loras) == 0 {
		return nil
	}
	// The native server contract uses structured LoRA fields. The media manager
	// validates them again after loading, when the model-specific catalog is known.
	result := []mediagen.Lora{}
	for i := len(loras) - 1; i >= 0; i-- {
		p := b.effective(loras[i])
		result = append(result, mediagen.Lora{Path: p.str("path"), Multiplier: p.num("strength")})
	}
	return result
}

// Native APIs resolve structured entries through a catalog fixed at launch.
// Keep a compatible existing catalog, or use the selected files' common tree.
func (b *builder) configureLoraCatalog(n Node, model string, loras []mediagen.Lora, ov *mediagen.LoadOverrides) {
	if len(loras) == 0 {
		return
	}
	inside := func(root, path string) bool {
		if !filepath.IsAbs(root) {
			return false
		}
		rel, err := filepath.Rel(root, path)
		return err == nil && filepath.IsLocal(rel)
	}
	settings, running := b.env.RunningServer(model)
	allInside := running && filepath.IsAbs(settings.LoraModelDir)
	for _, lora := range loras {
		allInside = allInside && inside(settings.LoraModelDir, lora.Path)
	}
	if allInside {
		ov.LoraModelDir = settings.LoraModelDir
		return
	}
	// Custom runtimes may advertise absolute catalog paths directly.
	if caps, err := b.modelAPICapabilities(model); err == nil && caps != nil && caps.Known {
		allListed := true
		for _, lora := range loras {
			listed := false
			for _, entry := range caps.Loras {
				listed = listed || (filepath.IsAbs(entry.Path) && filepath.Clean(entry.Path) == filepath.Clean(lora.Path))
			}
			allListed = allListed && listed
		}
		if allListed {
			return
		}
	}
	if !b.launchCaps.Has("lora-model-dir") {
		b.errorf("plan.lora_catalog", n.ID, "model", "the runtime cannot configure a LoRA catalog; select files already advertised by the loaded model")
		return
	}
	root := filepath.Dir(loras[0].Path)
	for _, lora := range loras {
		for !inside(root, lora.Path) {
			parent := filepath.Dir(root)
			if parent == root {
				b.errorf("plan.lora_catalog", n.ID, "model", "place the selected LoRAs under one shared model directory")
				return
			}
			root = parent
		}
	}
	if filepath.Dir(root) == root {
		b.errorf("plan.lora_catalog", n.ID, "model", "place the selected LoRAs under one shared model directory; a filesystem root cannot be used as a catalog")
		return
	}
	ov.LoraModelDir = root
}
