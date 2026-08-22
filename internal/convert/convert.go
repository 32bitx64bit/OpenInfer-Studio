package convert

import (
	"fmt"
	"os"
	"path/filepath"
)

type ConvertOptions struct {
	Name       string
	OnProgress func(done, total int)
}

// ConvertDir reads a Hugging Face snapshot directory and writes a GGUF.
func ConvertDir(dir, dest string, opts ConvertOptions) (*ConvertStats, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, fmt.Errorf("config.json: %w", err)
	}
	if opts.Name != "" {
		cfg["name"] = opts.Name
	}
	tensors, err := IndexDir(dir)
	if err != nil {
		return nil, err
	}
	ad, err := resolveAdapter(cfg, tensorNames(tensors))
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return nil, err
	}
	w, err := NewWriter(dest)
	if err != nil {
		return nil, err
	}
	stats, err := ad.Convert(dir, tensors, cfg, w)
	cerr := w.Close()
	if err != nil {
		_ = os.Remove(dest)
		return stats, err
	}
	if cerr != nil {
		_ = os.Remove(dest)
		return stats, cerr
	}
	return stats, nil
}
