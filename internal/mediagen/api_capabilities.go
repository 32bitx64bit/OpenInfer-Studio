package mediagen

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

var (
	ErrServerNotRunning        = errors.New("diffusion server is not ready")
	ErrCapabilitiesUnavailable = errors.New("runtime does not expose /sdcpp/v1/capabilities")
)

// Lora is the native server API's structured LoRA entry. Relative paths
// must be advertised by the server or resolved under its configured LoRA dir.
type Lora struct {
	Path        string  `json:"path"`
	Multiplier  float64 `json:"multiplier"`
	IsHighNoise bool    `json:"is_high_noise,omitempty"`
}

type LoraEntry struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

type UpscalerEntry struct {
	Name         string `json:"name"`
	Model        bool   `json:"model"`
	ImageUpscale bool   `json:"image_upscale"`
}

type APILimits struct {
	MinWidth         int `json:"min_width,omitempty"`
	MaxWidth         int `json:"max_width,omitempty"`
	MinHeight        int `json:"min_height,omitempty"`
	MaxHeight        int `json:"max_height,omitempty"`
	MaxBatchCount    int `json:"max_batch_count,omitempty"`
	MaxUpscaleWidth  int `json:"max_upscale_width,omitempty"`
	MaxUpscaleHeight int `json:"max_upscale_height,omitempty"`
}

// WorkflowDefaults uses the editor's workflow keys. Presence is retained
// internally so JSON preserves explicit zero/empty defaults while omitting
// fields that the runtime did not return. The exported Go fields are values.
type WorkflowDefaults struct {
	Width      int     `json:"width"`
	Height     int     `json:"height"`
	BatchCount int     `json:"batch"`
	Steps      int     `json:"steps"`
	Sampler    string  `json:"sampler"`
	Scheduler  string  `json:"scheduler"`
	CFG        float64 `json:"cfg"`
	Guidance   float64 `json:"guidance"`
	Strength   float64 `json:"strength"`
	Frames     int     `json:"frames"`
	FPS        int     `json:"fps"`
	present    map[string]bool
}

func (d WorkflowDefaults) MarshalJSON() ([]byte, error) {
	fields := map[string]any{"width": d.Width, "height": d.Height, "batch": d.BatchCount, "steps": d.Steps, "sampler": d.Sampler, "scheduler": d.Scheduler, "cfg": d.CFG, "guidance": d.Guidance, "strength": d.Strength, "frames": d.Frames, "fps": d.FPS}
	if d.present != nil {
		for field := range fields {
			if !d.present[field] {
				delete(fields, field)
			}
		}
	}
	return json.Marshal(fields)
}

func normalizeWorkflowDefaults(doc map[string]any) WorkflowDefaults {
	d := WorkflowDefaults{present: map[string]bool{}}
	integer := func(source map[string]any, native, key string, target *int) {
		if value, ok := source[native].(float64); ok && value >= -2147483648 && value <= 2147483647 && value == float64(int(value)) {
			*target = int(value)
			d.present[key] = true
		}
	}
	number := func(source map[string]any, native, key string, target *float64) {
		if value, ok := source[native].(float64); ok {
			*target = value
			d.present[key] = true
		}
	}
	str := func(source map[string]any, native, key string, target *string) {
		if value, ok := source[native].(string); ok {
			*target = value
			d.present[key] = true
		}
	}
	integer(doc, "width", "width", &d.Width)
	integer(doc, "height", "height", &d.Height)
	integer(doc, "batch_count", "batch", &d.BatchCount)
	integer(doc, "video_frames", "frames", &d.Frames)
	integer(doc, "fps", "fps", &d.FPS)
	number(doc, "strength", "strength", &d.Strength)
	sp, _ := doc["sample_params"].(map[string]any)
	integer(sp, "sample_steps", "steps", &d.Steps)
	str(sp, "sample_method", "sampler", &d.Sampler)
	str(sp, "scheduler", "scheduler", &d.Scheduler)
	// Native "default" is an automatic sentinel, not an advertised enum
	// choice. The workflow's empty choice leaves the server default in effect.
	if d.Sampler == "default" {
		d.Sampler = ""
	}
	if d.Scheduler == "default" {
		d.Scheduler = ""
	}
	guidance, _ := sp["guidance"].(map[string]any)
	number(guidance, "txt_cfg", "cfg", &d.CFG)
	number(guidance, "distilled_guidance", "guidance", &d.Guidance)
	return d
}

// APICapabilities describes only the HTTP API advertised by this loaded
// model's live runtime. CLI flags are intentionally a separate interface.
// Unknown metadata never enables advanced inputs. No runtime-wide cache is
// used: model reloads and component changes can change these features.
type APICapabilities struct {
	Known               bool                        `json:"known"`
	Reason              string                      `json:"reason,omitempty"`
	ModelID             string                      `json:"model_id"`
	RuntimeID           string                      `json:"runtime_id"`
	CurrentMode         string                      `json:"current_mode,omitempty"`
	SupportedModes      []string                    `json:"supported_modes"`
	FeaturesByMode      map[string]map[string]bool  `json:"features_by_mode"`
	DefaultsByMode      map[string]WorkflowDefaults `json:"defaults_by_mode"`
	Samplers            []string                    `json:"samplers"`
	Schedulers          []string                    `json:"schedulers"`
	OutputFormatsByMode map[string][]string         `json:"output_formats_by_mode"`
	Loras               []LoraEntry                 `json:"loras"`
	Upscale             bool                        `json:"upscale"`
	Upscalers           []UpscalerEntry             `json:"upscalers"`
	Limits              APILimits                   `json:"limits"`
}

func (c *APICapabilities) SupportsMode(mode string) bool {
	if c == nil || !c.Known {
		return false
	}
	for _, m := range c.SupportedModes {
		if m == mode {
			return true
		}
	}
	return false
}

func (c *APICapabilities) SupportsFeature(mode, feature string) bool {
	return c.SupportsMode(mode) && c.FeaturesByMode[mode][feature]
}

// GenerationCapabilities discovers mode/feature/choice metadata without
// starting or reloading a model. A stopped model returns Known:false and a
// reason, allowing planning to distinguish unavailable from supported.
func (m *Manager) GenerationCapabilities(modelID string) (*APICapabilities, error) {
	return m.generationCapabilities(context.Background(), modelID)
}

func (m *Manager) generationCapabilities(ctx context.Context, modelID string) (*APICapabilities, error) {
	runtimeID, _, _ := m.serverMeta(modelID)
	doc, err := m.Capabilities(ctx, modelID)
	if err != nil {
		if errors.Is(err, ErrServerNotRunning) || errors.Is(err, ErrCapabilitiesUnavailable) {
			c := normalizeAPICapabilities(nil, modelID, runtimeID)
			c.Reason = err.Error()
			return c, nil
		}
		return nil, err
	}
	return normalizeAPICapabilities(doc, modelID, runtimeID), nil
}

func normalizeAPICapabilities(doc map[string]any, modelID, runtimeID string) *APICapabilities {
	c := &APICapabilities{
		ModelID: modelID, RuntimeID: runtimeID,
		SupportedModes: []string{}, FeaturesByMode: map[string]map[string]bool{},
		Samplers: stringList(doc["samplers"]), Schedulers: stringList(doc["schedulers"]),
		OutputFormatsByMode: map[string][]string{}, Loras: []LoraEntry{}, Upscalers: []UpscalerEntry{},
		DefaultsByMode: map[string]WorkflowDefaults{},
	}
	c.CurrentMode, _ = doc["current_mode"].(string)
	// A legacy current_mode is explicit evidence of one mode. Never apply
	// its top-level feature mirror to another mode, or override a modern
	// explicitly empty supported_modes list.
	modes, hasModes := doc["supported_modes"]
	if !hasModes && (c.CurrentMode == "img_gen" || c.CurrentMode == "vid_gen") {
		modes = []any{c.CurrentMode}
	}
	featureValue, hasFeatures := doc["features_by_mode"]
	features, _ := featureValue.(map[string]any)
	formatValue, hasFormats := doc["output_formats_by_mode"]
	formats, _ := formatValue.(map[string]any)
	defaultValue, hasDefaults := doc["defaults_by_mode"]
	defaults, _ := defaultValue.(map[string]any)
	for _, mode := range stringList(modes) {
		if mode != "img_gen" && mode != "vid_gen" {
			continue
		}
		c.SupportedModes = append(c.SupportedModes, mode)
		f, _ := features[mode].(map[string]any)
		if !hasFeatures && mode == c.CurrentMode {
			f, _ = doc["features"].(map[string]any)
		}
		flags := map[string]bool{}
		for key, value := range f {
			if b, ok := value.(bool); ok {
				flags[key] = b
			}
		}
		c.FeaturesByMode[mode] = flags
		modeDefaults, _ := defaults[mode].(map[string]any)
		if !hasDefaults && mode == c.CurrentMode {
			modeDefaults, _ = doc["defaults"].(map[string]any)
		}
		if modeDefaults != nil {
			c.DefaultsByMode[mode] = normalizeWorkflowDefaults(modeDefaults)
		}
		if hasFormats {
			c.OutputFormatsByMode[mode] = stringList(formats[mode])
		} else if mode == c.CurrentMode {
			c.OutputFormatsByMode[mode] = stringList(doc["output_formats"])
		}
	}
	c.Upscale, _ = doc["upscale"].(bool)
	decodeField := func(key string, dst any) {
		if raw, err := json.Marshal(doc[key]); err == nil {
			_ = json.Unmarshal(raw, dst)
		}
	}
	decodeField("limits", &c.Limits)
	// Decode each catalog entry independently so malformed entries cannot
	// partially set booleans and accidentally advertise support.
	if list, ok := doc["upscalers"].([]any); ok {
		for _, entry := range list {
			var up UpscalerEntry
			raw, err := json.Marshal(entry)
			if err == nil && json.Unmarshal(raw, &up) == nil && up.Name != "" {
				c.Upscalers = append(c.Upscalers, up)
			}
		}
	}
	if list, ok := doc["loras"].([]any); ok {
		for _, entry := range list {
			var l LoraEntry
			raw, err := json.Marshal(entry)
			if err == nil && json.Unmarshal(raw, &l) == nil && l.Name != "" && l.Path != "" {
				c.Loras = append(c.Loras, l)
			}
		}
	}
	c.Known = len(c.SupportedModes) > 0 || c.Upscale
	if !c.Known {
		c.Reason = "loaded-model capabilities contain no explicit supported mode; advanced features are unavailable"
	}
	return c
}

func stringList(value any) []string {
	out := []string{}
	seen := map[string]bool{}
	if values, ok := value.([]any); ok {
		for _, value := range values {
			if s, ok := value.(string); ok && strings.TrimSpace(s) != "" && !seen[s] {
				out = append(out, s)
				seen[s] = true
			}
		}
	}
	return out
}

const maxCapabilityResponseBytes = 1 << 20

// readSDJSON bounds response bodies and rejects multiple JSON documents.
func readSDJSON(r io.Reader, limit int64, dst any) error {
	raw, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return err
	}
	if int64(len(raw)) > limit {
		return fmt.Errorf("sd-server response exceeds %d byte limit", limit)
	}
	return json.Unmarshal(raw, dst)
}

func (m *Manager) fetchCapabilities(ctx context.Context, modelID string) (map[string]any, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	port, ok := m.ServerPort(modelID)
	if !ok {
		return nil, fmt.Errorf("%w for model %s", ErrServerNotRunning, modelID)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		fmt.Sprintf("http://127.0.0.1:%d/sdcpp/v1/capabilities", port), nil)
	if err != nil {
		return nil, err
	}
	resp, err := m.doSD(req, port)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusMethodNotAllowed {
		return nil, ErrCapabilitiesUnavailable
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("sd-server capabilities HTTP %d", resp.StatusCode)
	}
	var out map[string]any
	if err := readSDJSON(resp.Body, maxCapabilityResponseBytes, &out); err != nil {
		return nil, fmt.Errorf("reading sd-server capabilities: %w", err)
	}
	if out == nil {
		return nil, errors.New("sd-server capabilities must be a JSON object")
	}
	if !m.serverMatchesPort(modelID, port) {
		return nil, ErrServerNotRunning
	}
	return out, nil
}
