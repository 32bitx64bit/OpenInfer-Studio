package workflow

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

type workflowFile struct {
	Format  string          `json:"format"`
	Version int             `json:"version"`
	Name    string          `json:"name"`
	Graph   json.RawMessage `json:"graph"`
}

func (s *Store) Import(path string) (*Record, error) {
	if !filepath.IsAbs(path) {
		return nil, invalid("choose an absolute workflow file path")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, invalid("read workflow: %v", err)
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil || !st.Mode().IsRegular() || st.Size() > maxGraphBytes+4096 {
		return nil, invalid("workflow must be a regular JSON file of at most 2 MiB")
	}
	raw, err := io.ReadAll(io.LimitReader(f, maxGraphBytes+4097))
	if err != nil {
		return nil, err
	}
	var doc workflowFile
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, invalid("invalid workflow JSON: %v", err)
	}
	graph := json.RawMessage(raw)
	name := doc.Name
	if len(doc.Graph) > 0 {
		if doc.Format != "openinfer-workflow" || doc.Version != 1 {
			return nil, invalid("unsupported workflow file format")
		}
		graph = doc.Graph
	}
	if name == "" {
		name = filepath.Base(path)
	}
	return s.Create(name, graph)
}

func (s *Store) Export(id, path string) error {
	if !filepath.IsAbs(path) || filepath.Ext(path) != ".json" {
		return invalid("choose an absolute .json workflow path")
	}
	rec, err := s.Get(id)
	if err != nil {
		return err
	}
	raw, err := json.MarshalIndent(workflowFile{Format: "openinfer-workflow", Version: 1, Name: rec.Name, Graph: rec.Graph}, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".openinfer-workflow-*.json")
	if err != nil {
		return fmt.Errorf("create export: %w", err)
	}
	temp := f.Name()
	defer os.Remove(temp)
	if _, err = f.Write(append(raw, '\n')); err != nil {
		f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(temp, path)
}
