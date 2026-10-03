package mediagen

// LoadOverrides are the sd-server launch inputs a node graph controls.
// sd-server fixes these when it starts, so a graph that needs different ones
// than the running server has must restart it. Empty fields mean "the graph
// does not care": they neither override a launch setting nor force a restart.
type LoadOverrides struct {
	VAE               string `json:"vae,omitempty"`
	LLM               string `json:"llm,omitempty"`
	T5XXL             string `json:"t5xxl,omitempty"`
	ClipL             string `json:"clip_l,omitempty"`
	ClipG             string `json:"clip_g,omitempty"`
	LoraModelDir      string `json:"lora_model_dir,omitempty"`
	ControlNet        string `json:"control_net,omitempty"`
	ESRGAN            string `json:"esrgan,omitempty"`
	HiresUpscalersDir string `json:"hires_upscalers_dir,omitempty"`
}

// IsZero reports whether the overrides leave every launch setting alone.
func (o LoadOverrides) IsZero() bool { return o == LoadOverrides{} }

// Apply writes the non-empty overrides onto launch settings.
func (o LoadOverrides) Apply(s *LoadSettings) {
	set := func(src string, dst *string) {
		if src != "" {
			*dst = src
		}
	}
	set(o.VAE, &s.VAE)
	set(o.LLM, &s.LLM)
	set(o.T5XXL, &s.T5XXL)
	set(o.ClipL, &s.ClipL)
	set(o.ClipG, &s.ClipG)
	set(o.LoraModelDir, &s.LoraModelDir)
	set(o.ControlNet, &s.ControlNet)
	set(o.ESRGAN, &s.ESRGAN)
	if o.ESRGAN != "" && o.HiresUpscalersDir == "" {
		s.HiresUpscalersDir = ""
	} // rederive from the selected model
	set(o.HiresUpscalersDir, &s.HiresUpscalersDir)
}

// SatisfiedBy reports whether a server launched with s already has every
// non-empty override in effect. s should be the settings after companion
// pairing (what actually ran), so a graph that names the file the launcher
// would have auto-paired anyway does not trigger a pointless restart.
func (o LoadOverrides) SatisfiedBy(s LoadSettings) bool {
	match := func(want, have string) bool { return want == "" || want == have }
	return match(o.VAE, s.VAE) && match(o.LLM, s.LLM) && match(o.T5XXL, s.T5XXL) &&
		match(o.ClipL, s.ClipL) && match(o.ClipG, s.ClipG) && match(o.LoraModelDir, s.LoraModelDir) &&
		match(o.ControlNet, s.ControlNet) && match(o.ESRGAN, s.ESRGAN) && match(o.HiresUpscalersDir, s.HiresUpscalersDir)
}
