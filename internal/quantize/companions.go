package quantize

import (
	"path/filepath"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/models"
)

// Companion is a related GGUF offered alongside the source model.
type Companion struct {
	Kind          string `json:"kind"` // projector|draft|shard
	ModelID       string `json:"model_id,omitempty"`
	Path          string `json:"path"`
	SizeBytes     int64  `json:"size_bytes"`
	Quantization  string `json:"quantization,omitempty"`
	Alias         string `json:"alias,omitempty"`
	DefaultAction string `json:"default_action"` // quantize|copy|skip
	DefaultFType  string `json:"default_ftype,omitempty"`
}

// ListCompanions finds mmproj, split shards, and speculative drafts for src.
func ListCompanions(src models.Model, library []models.Model, mainFType string) []Companion {
	var out []Companion
	if src.ProjectorPath != "" {
		out = append(out, Companion{
			Kind:          "projector",
			Path:          src.ProjectorPath,
			SizeBytes:     fileSize(src.ProjectorPath),
			Alias:         filepath.Base(src.ProjectorPath),
			DefaultAction: "copy",
			DefaultFType:  "Q8_0",
		})
	}
	primary := filepath.Clean(src.PrimaryPath)
	proj := filepath.Clean(src.ProjectorPath)
	for _, f := range src.Files {
		cf := filepath.Clean(f)
		if cf == primary || (proj != "." && cf == proj) {
			continue
		}
		base := strings.ToLower(filepath.Base(f))
		if strings.Contains(base, "mmproj") || strings.Contains(base, "mm-proj") {
			continue
		}
		action := "skip"
		if isSplitPath(f) {
			action = "quantize" // keep-split handles these
		}
		out = append(out, Companion{
			Kind:          "shard",
			Path:          f,
			SizeBytes:     fileSize(f),
			Alias:         filepath.Base(f),
			DefaultAction: action,
			DefaultFType:  mainFType,
		})
	}
	drafts := models.FilterDraftCandidates(src, library, true)
	for _, d := range drafts {
		if !d.Compatible {
			continue
		}
		out = append(out, Companion{
			Kind:          "draft",
			ModelID:       d.ID,
			Path:          d.PrimaryPath,
			SizeBytes:     d.SizeBytes,
			Quantization:  d.Quantization,
			Alias:         d.Alias,
			DefaultAction: "skip",
			DefaultFType:  moreAggressive(mainFType),
		})
	}
	return out
}
