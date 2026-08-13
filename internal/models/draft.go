package models

import (
	"encoding/json"
	"path/filepath"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/gguf"
)

// DraftCandidate is a library model offered as a speculative-decoding draft,
// with an optional incompatibility reason when filtering is disabled.
type DraftCandidate struct {
	Model
	Compatible bool   `json:"compatible"`
	Reason     string `json:"reason,omitempty"`
}

type draftMeta struct {
	Tokenizer        string `json:"tokenizer"`
	Projector        bool   `json:"projector"`
	Multimodal       bool   `json:"multimodal"`
	SpeculativeDraft bool   `json:"speculative_draft"`
	HasMTP           bool   `json:"has_mtp"`
	SpecType         string `json:"spec_type"`
}

func parseDraftMeta(raw json.RawMessage) draftMeta {
	var m draftMeta
	_ = json.Unmarshal(raw, &m)
	return m
}

func looksLikeProjector(m Model) bool {
	meta := parseDraftMeta(m.Metadata)
	if meta.Projector {
		return true
	}
	base := strings.ToLower(filepath.Base(m.PrimaryPath))
	if strings.Contains(base, "mmproj") || strings.Contains(base, "mm-proj") {
		return true
	}
	// Projector-only rows sometimes lack architecture and have tiny param counts.
	if m.Architecture == "" && meta.Multimodal && m.Parameters == 0 {
		return true
	}
	return false
}

func isSpeculativeDraftModel(m Model) bool {
	meta := parseDraftMeta(m.Metadata)
	if meta.SpeculativeDraft {
		return true
	}
	ok, _ := gguf.DetectSpeculativeDraft(m.Architecture, m.Alias, m.PrimaryPath, nil)
	return ok
}

// IsSpeculativeDraft reports whether m is a draft sidecar (DFlash, EAGLE, MTP, …),
// not a standalone chat model.
func IsSpeculativeDraft(m Model) bool {
	return isSpeculativeDraftModel(m)
}

// IsProjectorFile reports whether m is an mmproj / projector GGUF.
func IsProjectorFile(m Model) bool {
	return looksLikeProjector(m)
}

func draftSpecType(m Model) gguf.SpecType {
	meta := parseDraftMeta(m.Metadata)
	if meta.SpecType != "" {
		return gguf.SpecType(meta.SpecType)
	}
	_, st := gguf.DetectSpeculativeDraft(m.Architecture, m.Alias, m.PrimaryPath, nil)
	return st
}

// DraftCompatible reports whether draft is a plausible speculative-decoding
// partner for target. Heuristics are deliberately conservative: architecture
// and tokenizer must match when known, the draft must be smaller, and
// projectors / the target itself are excluded. Specialized drafts (EAGLE,
// DFlash, …) may fail these checks — users can disable filtering in Settings.
func DraftCompatible(target, draft Model) (ok bool, reason string) {
	if draft.ID == "" || draft.ID == target.ID {
		return false, "same as target model"
	}
	if draft.PrimaryPath != "" && draft.PrimaryPath == target.PrimaryPath {
		return false, "same file as target"
	}
	if looksLikeProjector(draft) {
		return false, "multimodal projector, not a draft model"
	}

	specialized := isSpeculativeDraftModel(draft)
	draftSpec := draftSpecType(draft)

	tArch := strings.ToLower(strings.TrimSpace(target.Architecture))
	dArch := strings.ToLower(strings.TrimSpace(draft.Architecture))
	// Specialized draft arches / sidecars (eagle3, dflash, mtp-, …) intentionally
	// differ from or share the target arch; skip equality for those.
	if !specialized && draftSpec == "" && tArch != "" && dArch != "" && tArch != dArch {
		return false, "architecture mismatch (" + draft.Architecture + " ≠ " + target.Architecture + ")"
	}

	tTok := strings.ToLower(strings.TrimSpace(parseDraftMeta(target.Metadata).Tokenizer))
	dTok := strings.ToLower(strings.TrimSpace(parseDraftMeta(draft.Metadata).Tokenizer))
	if tTok != "" && dTok != "" && tTok != dTok {
		return false, "tokenizer mismatch (" + dTok + " ≠ " + tTok + ")"
	}

	// Draft should be smaller than the target. Prefer parameter counts; fall
	// back to on-disk size when params are unknown. Specialized drafts are
	// almost always tiny; still enforce when both sizes are known.
	switch {
	case target.Parameters > 0 && draft.Parameters > 0:
		if draft.Parameters >= target.Parameters {
			return false, "draft is not smaller than the target (by parameter count)"
		}
	case target.SizeBytes > 0 && draft.SizeBytes > 0:
		if draft.SizeBytes >= target.SizeBytes {
			return false, "draft is not smaller than the target (by file size)"
		}
	}

	return true, ""
}

// FilterDraftCandidates returns library models usable as drafts for target.
//
// When filter is true (default / load.filter_incompatible_drafts): only
// detected speculative-decoding sidecars (mtp-, gemma4-assistant, eagle3-,
// dflash-, dspark-, dedicated draft arches, …) are listed. When filter is
// false, every other library model is returned so draft-simple companions
// can be picked manually.
func FilterDraftCandidates(target Model, library []Model, filter bool) []DraftCandidate {
	out := make([]DraftCandidate, 0, len(library))
	for _, m := range library {
		if m.ID == target.ID {
			continue
		}
		if looksLikeProjector(m) {
			continue
		}
		specDraft := isSpeculativeDraftModel(m)
		if filter && !specDraft {
			continue
		}
		ok, reason := DraftCompatible(target, m)
		if filter && !ok {
			continue
		}
		out = append(out, DraftCandidate{Model: m, Compatible: ok, Reason: reason})
	}
	return out
}
