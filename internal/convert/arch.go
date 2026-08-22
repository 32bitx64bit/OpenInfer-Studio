package convert

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Adapter converts one HF architecture into GGUF tensors + metadata.
type Adapter interface {
	ID() string
	Match(cfg map[string]any) bool
	Convert(dir string, tensors []TensorRef, cfg map[string]any, w *Writer) (*ConvertStats, error)
}

type ConvertStats struct {
	Architecture string
	Tensors      int
	Skipped      int
	Warnings     []string
	GGUFType     int // GGMLF16 or GGMLBF16
}

func FindAdapter(cfg map[string]any) (Adapter, error) {
	return resolveAdapter(cfg, nil)
}

func AdapterRegistered(cfg map[string]any) bool {
	_, err := FindAdapter(cfg)
	return err == nil
}

func LoadConfig(dir string) (map[string]any, error) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	return ParseConfig(b)
}

func ParseConfig(b []byte) (map[string]any, error) {
	var cfg map[string]any
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("config.json: %w", err)
	}
	if tc, ok := cfg["text_config"].(map[string]any); ok {
		for k, v := range tc {
			// Keep root keys (model_type=muse_glimmer, architectures).
			// text_config.model_type is muse_glimmer_text.
			if _, exists := cfg[k]; exists {
				continue
			}
			cfg[k] = v
		}
	}
	return cfg, nil
}

func stringSlice(v any) []string {
	switch x := v.(type) {
	case []string:
		return x
	case []any:
		out := make([]string, 0, len(x))
		for _, e := range x {
			if s, ok := e.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func cfgInt(cfg map[string]any, keys ...string) int {
	for _, k := range keys {
		if n, ok := asInt(cfg[k]); ok {
			return n
		}
	}
	return 0
}

func cfgFloat(cfg map[string]any, keys ...string) float64 {
	for _, k := range keys {
		if n, ok := asFloat(cfg[k]); ok {
			return n
		}
	}
	return 0
}

func cfgString(cfg map[string]any, keys ...string) string {
	for _, k := range keys {
		if s, ok := cfg[k].(string); ok && s != "" {
			return s
		}
	}
	return ""
}

func cfgMap(cfg map[string]any, key string) map[string]any {
	if cfg == nil {
		return nil
	}
	if m, ok := cfg[key].(map[string]any); ok {
		return m
	}
	return nil
}

func cfgFloatSlice(v any) []float64 {
	switch x := v.(type) {
	case []float64:
		return append([]float64(nil), x...)
	case []any:
		out := make([]float64, 0, len(x))
		for _, e := range x {
			if n, ok := asFloat(e); ok {
				out = append(out, n)
			}
		}
		return out
	default:
		return nil
	}
}

func asInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case json.Number:
		i, err := n.Int64()
		return int(i), err == nil
	case int:
		return n, true
	case int64:
		return int(n), true
	default:
		return 0, false
	}
}

func asFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case float32:
		return float64(n), true
	default:
		return 0, false
	}
}
