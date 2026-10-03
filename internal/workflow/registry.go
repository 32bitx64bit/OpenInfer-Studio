package workflow

import (
	"fmt"
	"slices"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

// Caps is what the selected sd.cpp runtime advertises (sd-server --help).
// Same rule as the llama.cpp side: a node that needs a flag the runtime does
// not list is shown disabled, and a graph using it is rejected.
type Caps struct {
	API *mediagen.APICapabilities
	// Known is false when no stable-diffusion.cpp runtime is installed.
	Known bool
	// Flags are capability ids as parsed by mediagen.ParseSDCapabilities.
	Flags []string
}

// Has reports whether the runtime advertises the capability.
func (c Caps) Has(id string) bool { return slices.Contains(c.Flags, id) }

// NodeTypeView is a node spec plus whether the current runtime can run it.
// GET /api/v1/workflow/node-types returns these.
type NodeTypeView struct {
	NodeSpec
	Available bool   `json:"available"`
	Reason    string `json:"reason,omitempty"`
}

// Registry holds the node types this build ships. The set is closed: nodes
// are Go values compiled into the app, not loaded plugins.
type Registry struct {
	specs []NodeSpec
	index map[string]int
}

// NewRegistry returns the registry of built-in node types.
func NewRegistry() *Registry {
	r := &Registry{index: map[string]int{}}
	for _, s := range append(append(builtinSpecs(), imageSpecs()...), upscaleSpec()) {
		// Descriptors are a JSON contract: lists are always arrays, never null.
		if s.Inputs == nil {
			s.Inputs = []Port{}
		}
		if s.Outputs == nil {
			s.Outputs = []Port{}
		}
		if s.Params == nil {
			s.Params = []ParamSpec{}
		}
		r.index[s.Type] = len(r.specs)
		r.specs = append(r.specs, s)
	}
	return r
}

// Get returns the spec for a node type.
func (r *Registry) Get(typ string) (NodeSpec, bool) {
	i, ok := r.index[typ]
	if !ok {
		return NodeSpec{}, false
	}
	return r.specs[i], true
}

// Specs returns every spec in palette order.
func (r *Registry) Specs() []NodeSpec { return slices.Clone(r.specs) }

// View annotates every spec with availability under caps.
func (r *Registry) View(caps Caps) []NodeTypeView {
	out := make([]NodeTypeView, 0, len(r.specs))
	for _, s := range r.specs {
		v := NodeTypeView{NodeSpec: s, Available: true}
		switch s.APIFeature {
		case "upscale":
			rgb := false
			if caps.API != nil {
				for _, upscaler := range caps.API.Upscalers {
					rgb = rgb || (upscaler.Model && upscaler.ImageUpscale && upscaler.Name != "")
				}
			}
			if caps.API == nil || !caps.API.Known || !caps.API.Upscale || !rgb {
				v.Available = false
				v.Reason = "Load a compatible model to discover native image upscaling support"
			}
		case "lora":
			if caps.API == nil || !(caps.API.SupportsFeature("img_gen", "lora") || caps.API.SupportsFeature("vid_gen", "lora")) {
				v.Available = false
				v.Reason = "Load a compatible model to discover structured LoRA support"
			}
		}
		if reason := missingCapability(s.Requires, caps); reason != "" {
			v.Available = false
			v.Reason = reason
		}
		out = append(out, v)
	}
	return out
}

// missingCapability returns a human reason when caps cannot satisfy req.
func missingCapability(req []string, caps Caps) string {
	if len(req) == 0 {
		return ""
	}
	if !caps.Known {
		return "no stable-diffusion.cpp runtime installed"
	}
	var missing []string
	for _, id := range req {
		if !caps.Has(id) {
			missing = append(missing, "--"+id)
		}
	}
	if len(missing) == 0 {
		return ""
	}
	return fmt.Sprintf("this runtime does not advertise %s", strings.Join(missing, ", "))
}

func ptr(v float64) *float64 { return &v }

// Category ids. The palette groups and colours by these.
const (
	CatLoaders = "loaders"
	CatPrompt  = "prompt"
	CatInput   = "input"
	CatSample  = "sampling"
	CatImage   = "image"
	CatOutput  = "output"
)

// encoderRoleCaps maps a Load Text Encoder role to the sd-server flag it
// needs, so the node is checked against the role the user picked.
var encoderRoleCaps = map[string]string{
	"llm": "llm", "t5xxl": "t5xxl", "clip_l": "clip-l", "clip_g": "clip-g",
}

func samplerParams(video bool) []ParamSpec {
	ps := []ParamSpec{
		{Name: "seed", Kind: ParamSeed, Default: -1},
		{Name: "seed_mode", Kind: ParamEnum, Options: []string{"fixed", "increment", "random"}, Default: "fixed"},
		{Name: "steps", Label: "Steps (0 = auto)", Kind: ParamInt, Min: ptr(0), Max: ptr(300), Default: 20},
		{Name: "cfg", Kind: ParamFloat, Min: ptr(0), Max: ptr(30), Default: 7.0},
		{Name: "guidance", Kind: ParamFloat, Min: ptr(0), Max: ptr(30), Default: 0.0},
		{Name: "sampler", Kind: ParamEnum, From: "capabilities.samplers"},
		{Name: "scheduler", Kind: ParamEnum, From: "capabilities.schedulers"},
	}
	if video {
		return append(ps,
			ParamSpec{Name: "frames", Kind: ParamInt, Min: ptr(1), Max: ptr(512), Default: 33},
			ParamSpec{Name: "fps", Kind: ParamInt, Min: ptr(1), Max: ptr(60), Default: 16},
			ParamSpec{Name: "output_format", Kind: ParamEnum, Options: []string{"webm", "avi"}, Default: "webm"},
		)
	}
	return append(ps,
		ParamSpec{Name: "control_strength", Kind: ParamFloat, Min: ptr(0), Max: ptr(10), Default: 0.9, ShowWhen: "control is IMAGE"},
		ParamSpec{Name: "strength", Kind: ParamFloat, Min: ptr(0), Max: ptr(1), Default: 0.75, ShowWhen: "start is IMAGE"},
		ParamSpec{Name: "output_format", Kind: ParamEnum, Options: []string{"png", "jpeg", "webp"}, Default: "png"},
	)
}

func samplerInputs() []Port {
	return []Port{
		{Name: "model", Type: PortTypes{TypeModel}, Required: true},
		{Name: "clip", Type: PortTypes{TypeClip}, Required: true},
		{Name: "vae", Type: PortTypes{TypeVAE}},
		{Name: "positive", Type: PortTypes{TypeCond}, Required: true},
		{Name: "negative", Type: PortTypes{TypeCond}},
		{Name: "mask", Type: PortTypes{TypeMask}},
		{Name: "control", Type: PortTypes{TypeImage}},
		{Name: "reference", Type: PortTypes{TypeImage}},
		{Name: "start", Type: PortTypes{TypeSize, TypeImage}, Required: true},
	}
}

func videoInputs() []Port {
	out := []Port{}
	for _, p := range samplerInputs() {
		if p.Name != "mask" && p.Name != "control" && p.Name != "reference" {
			out = append(out, p)
		}
	}
	return out
}

func builtinSpecs() []NodeSpec {
	modelParam := ParamSpec{Name: "model", Kind: ParamModel, Required: true}
	return []NodeSpec{
		{
			Type: "checkpoint.load", Category: CatLoaders, Title: "Load Checkpoint",
			Description: "A diffusion model from your library. Text encoder and VAE are paired automatically from its folder.",
			Outputs: []Port{
				{Name: "model", Type: PortTypes{TypeModel}},
				{Name: "clip", Type: PortTypes{TypeClip}},
				{Name: "vae", Type: PortTypes{TypeVAE}},
			},
			Params: []ParamSpec{modelParam},
		},
		{
			Type: "textencoder.load", Category: CatLoaders, Title: "Load Text Encoder",
			Description: "Use a specific text encoder instead of the one paired with the checkpoint.",
			Outputs:     []Port{{Name: "clip", Type: PortTypes{TypeClip}}},
			Params: []ParamSpec{
				{Name: "path", Kind: ParamPath, Required: true},
				{Name: "role", Kind: ParamEnum, Options: []string{"llm", "t5xxl", "clip_l", "clip_g"}, Default: "llm"},
			},
			extraRequires: func(p map[string]any) []string {
				role, _ := p["role"].(string)
				if role == "" {
					role = "llm"
				}
				if c, ok := encoderRoleCaps[role]; ok {
					return []string{c}
				}
				return nil
			},
		},
		{
			Type: "vae.load", Category: CatLoaders, Title: "Load VAE",
			Description: "Use a specific VAE instead of the one paired with the checkpoint. Changing it restarts the model.",
			Outputs:     []Port{{Name: "vae", Type: PortTypes{TypeVAE}}},
			Params:      []ParamSpec{{Name: "path", Kind: ParamPath, Required: true}},
			Requires:    []string{"vae"},
		},
		{
			Type: "lora.load", Category: CatLoaders, Title: "Load LoRA",
			Description: "Applies a local LoRA through the runtime’s structured API when the loaded model supports it.",
			Inputs:      []Port{{Name: "model", Type: PortTypes{TypeModel}, Required: true}},
			Outputs:     []Port{{Name: "model", Type: PortTypes{TypeModel}}},
			Params: []ParamSpec{
				{Name: "path", Kind: ParamPath, Required: true},
				{Name: "strength", Kind: ParamFloat, Min: ptr(-2), Max: ptr(2), Default: 1.0},
			},
			APIFeature: "lora",
		},
		{
			Type: "prompt", Category: CatPrompt, Title: "Prompt",
			Outputs: []Port{{Name: "cond", Type: PortTypes{TypeCond}}},
			Params:  []ParamSpec{{Name: "text", Kind: ParamText, Default: ""}},
		},
		{
			Type: "latent.empty", Category: CatInput, Title: "Empty Latent",
			Description: "Output size and batch for text-to-image. A recipe, not a latent tensor.",
			Outputs:     []Port{{Name: "size", Type: PortTypes{TypeSize}}},
			Params: []ParamSpec{
				{Name: "width", Kind: ParamInt, Min: ptr(64), Max: ptr(4096), Default: 1024},
				{Name: "height", Kind: ParamInt, Min: ptr(64), Max: ptr(4096), Default: 1024},
				{Name: "batch", Kind: ParamInt, Min: ptr(1), Max: ptr(8), Default: 1},
			},
		},
		{
			Type: "image.load", Category: CatInput, Title: "Load Image",
			Outputs: []Port{{Name: "image", Type: PortTypes{TypeImage}}},
			Params:  []ParamSpec{{Name: "path", Kind: ParamPath, Required: true}},
		},
		{
			Type: "sample", Category: CatSample, Title: "Sample",
			Description: "Sampling and VAE decode as one call. Wire an image into start for image-to-image.",
			Inputs:      samplerInputs(),
			Outputs:     []Port{{Name: "image", Type: PortTypes{TypeImage}}},
			Params:      samplerParams(false),
		},
		{
			Type: "sample.video", Category: CatSample, Title: "Sample (video)",
			Description: "Text-to-video, or image-to-video when the loaded model advertises it.",
			Inputs:      videoInputs(),
			Outputs:     []Port{{Name: "video", Type: PortTypes{TypeVideo}}},
			Params:      samplerParams(true),
		},
		{
			Type: "image.resize", Category: CatImage, Title: "Resize",
			Description: "Scale an image or fit it to a size. Runs on the CPU in the backend, no model needed.",
			Inputs:      []Port{{Name: "image", Type: PortTypes{TypeImage}, Required: true}},
			Outputs:     []Port{{Name: "image", Type: PortTypes{TypeImage}}},
			Params: []ParamSpec{
				{Name: "mode", Kind: ParamEnum, Options: []string{"scale", "size"}, Default: "scale"},
				{Name: "scale", Kind: ParamFloat, Min: ptr(0.1), Max: ptr(8), Default: 1.5, ShowWhen: "mode is scale"},
				{Name: "width", Kind: ParamInt, Min: ptr(16), Max: ptr(4096), Default: 1024, ShowWhen: "mode is size"},
				{Name: "height", Kind: ParamInt, Min: ptr(16), Max: ptr(4096), Default: 1024, ShowWhen: "mode is size"},
				{Name: "filter", Kind: ParamEnum, Options: []string{"lanczos", "bilinear"}, Default: "lanczos"},
			},
		},
		{
			Type: "image.save", Category: CatOutput, Title: "Save Image", Output: true,
			Inputs: []Port{{Name: "image", Type: PortTypes{TypeImage}, Required: true}},
			Params: []ParamSpec{{Name: "prefix", Kind: ParamString, Default: "image_"}},
		},
		{
			Type: "video.save", Category: CatOutput, Title: "Save Video", Output: true,
			Inputs: []Port{{Name: "video", Type: PortTypes{TypeVideo}, Required: true}},
			Params: []ParamSpec{{Name: "prefix", Kind: ParamString, Default: "video_"}},
		},
	}
}
