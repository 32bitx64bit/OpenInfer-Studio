package gguf

import (
	"path/filepath"
	"strings"
)

// SpecType is a llama.cpp --spec-type value for draft-based speculation.
// See docs/speculative.md and common/speculative.cpp type table.
type SpecType string

const (
	SpecNone   SpecType = ""
	SpecSimple SpecType = "draft-simple"
	SpecEagle3 SpecType = "draft-eagle3"
	SpecDFlash SpecType = "draft-dflash"
	SpecDSpark SpecType = "draft-dspark"
	SpecMTP    SpecType = "draft-mtp"
)

// Official HF/converter sidecar prefixes used by llama.cpp's download planner
// (common/download.cpp find_best_{mtp,eagle3,dflash,dspark}).
var sidecarPrefixes = []struct {
	prefix string
	spec   SpecType
}{
	{"mtp-", SpecMTP},
	{"eagle3-", SpecEagle3},
	{"dflash-", SpecDFlash},
	{"dspark-", SpecDSpark},
}

// Dedicated draft architectures (LLM_ARCH_* in llama-arch.cpp).
var speculativeArchs = map[string]SpecType{
	"eagle3":       SpecEagle3,
	"dflash":       SpecDFlash,
	"dflash-draft": SpecDFlash,
	"dspark":       SpecDSpark,
	// Official Gemma 4 MTP assistants (Google / AtomicChat). Filenames are often
	// *-assistant*.gguf without an mtp- prefix; architecture is the signal.
	"gemma4-assistant": SpecMTP,
	"gemma4_assistant": SpecMTP,
	// Muse Glimmer DFlash drafter (Meta). Transformers arch is
	// muse_glimmer_assistant; GGUF may use hyphen or underscore.
	"muse-glimmer-assistant": SpecDFlash,
	"muse_glimmer_assistant": SpecDFlash,
	"museglimmer-assistant":  SpecDFlash,
}

// first-stem-token → spec. DFlash/DSpark must not match mid-name feature
// tags on a trunk (e.g. Muse-Glimmer-30B-DFlash-ROCmFP4.gguf).
var speculativeFirstTokens = map[string]SpecType{
	"mtp":     SpecMTP,
	"eagle3":  SpecEagle3,
	"dflash":  SpecDFlash,
	"dspark":  SpecDSpark,
	"d-flash": SpecDFlash,
	"d-spark": SpecDSpark,
}

// Extra name tokens (community GGUFs that don't use the official prefix).
// These are matched as delimited tokens, except dflash/dspark which are
// prefix / first-token only (see LooksLikeSpeculativeDraftName).
var speculativeNameHints = []struct {
	token string
	spec  SpecType
}{
	{"eagle3", SpecEagle3},
	{"eagle-3", SpecEagle3},
	{"eagle_3", SpecEagle3},
	{"speculator.eagle3", SpecEagle3},
	{"speculator-eagle3", SpecEagle3},
	{"speculator", SpecSimple}, // generic; refine via arch/keys when possible
	{"spec-draft", SpecSimple},
	{"draft-eagle", SpecEagle3},
	{"draft-dflash", SpecDFlash},
	{"draft-dspark", SpecDSpark},
	{"draft-mtp", SpecMTP},
}

// IsSpeculativeDraftArch reports whether general.architecture is a dedicated
// speculative-decoding draft architecture.
func IsSpeculativeDraftArch(arch string) bool {
	_, ok := speculativeArchs[strings.ToLower(strings.TrimSpace(arch))]
	return ok
}

// SpecTypeForArch returns the --spec-type for a dedicated draft architecture.
func SpecTypeForArch(arch string) SpecType {
	return speculativeArchs[strings.ToLower(strings.TrimSpace(arch))]
}

// SidecarSpecType returns the speculative type when basename uses an official
// llama.cpp sidecar prefix (mtp-, eagle3-, dflash-, dspark-).
func SidecarSpecType(pathOrName string) SpecType {
	base := strings.ToLower(filepath.Base(pathOrName))
	for _, s := range sidecarPrefixes {
		if strings.HasPrefix(base, s.prefix) {
			return s.spec
		}
	}
	return SpecNone
}

// NextnPredictLayers reads {arch}.nextn_predict_layers from GGUF KV.
// Non-zero means the file carries Multi-Token Prediction / NextN heads
// (fused into a trunk or exported as a separate mtp- draft).
func NextnPredictLayers(arch string, raw map[string]any) uint32 {
	if raw == nil {
		return 0
	}
	arch = strings.TrimSpace(arch)
	keys := []string{}
	if arch != "" {
		keys = append(keys, arch+".nextn_predict_layers")
	}
	// Be tolerant of odd writers that omit the arch prefix.
	keys = append(keys, "nextn_predict_layers", "general.nextn_predict_layers")
	for _, k := range keys {
		if v, ok := raw[k]; ok {
			if n, ok := toUint32(v); ok {
				return n
			}
		}
	}
	// Suffix scan — some forks use slightly different arch spellings.
	for k, v := range raw {
		lk := strings.ToLower(k)
		if strings.HasSuffix(lk, ".nextn_predict_layers") || lk == "nextn_predict_layers" {
			if n, ok := toUint32(v); ok {
				return n
			}
		}
	}
	return 0
}

// HasNextnTensors reports NextN/MTP tensor-family keys in metadata (rare) or
// well-known nextn.* parameter keys some converters emit alongside tensors.
// nextn_predict_layers is intentionally excluded — use NextnPredictLayers so a
// value of 0 (MTP disabled) does not flip HasMTP.
func HasNextnTensors(raw map[string]any) bool {
	if raw == nil {
		return false
	}
	for k := range raw {
		lk := strings.ToLower(k)
		if strings.HasSuffix(lk, ".nextn_predict_layers") || lk == "nextn_predict_layers" {
			continue
		}
		if strings.HasPrefix(lk, "nextn.") || strings.Contains(lk, ".nextn.") {
			return true
		}
	}
	return false
}

// HasSpeculativeDraftKeys detects draft-only GGUF metadata keys written by
// convert_hf_to_gguf for EAGLE/DFlash/DSpark (not plain NextN-on-trunk).
func HasSpeculativeDraftKeys(raw map[string]any) bool {
	if raw == nil {
		return false
	}
	for k := range raw {
		lk := strings.ToLower(k)
		switch {
		case strings.HasPrefix(lk, "eagle3."),
			strings.HasPrefix(lk, "dflash."),
			strings.HasPrefix(lk, "dflash-draft."),
			strings.HasPrefix(lk, "dspark."),
			strings.HasSuffix(lk, ".target_layers"),
			strings.HasSuffix(lk, ".target_layer_ids"),
			lk == "general.speculative" || lk == "general.is_draft":
			return true
		}
	}
	return false
}

// LooksLikeSpeculativeDraftName reports draft/speculator naming outside the
// official sidecar prefixes (community uploads, older converters).
//
// DFlash / DSpark are prefix- or first-token-only: a trunk named
// Muse-Glimmer-30B-DFlash-….gguf is not a sidecar. EAGLE3 / speculator
// tokens may appear later in the basename.
func LooksLikeSpeculativeDraftName(s string) (bool, SpecType) {
	if st := SidecarSpecType(s); st != SpecNone {
		return true, st
	}
	base := strings.ToLower(filepath.Base(s))
	if strings.Contains(base, "mmproj") || strings.Contains(base, "mm-proj") ||
		strings.Contains(base, "projector") {
		return false, SpecNone
	}
	stem := strings.TrimSuffix(base, ".gguf")
	first := firstPathToken(stem)
	if st, ok := speculativeFirstTokens[first]; ok {
		return true, st
	}
	if st := assistantSidecarSpec(base); st != SpecNone {
		return true, st
	}
	for _, h := range speculativeNameHints {
		if delimitedToken(base, h.token) || strings.HasPrefix(stem, h.token) {
			return true, h.spec
		}
	}
	// Trailing draft markers for draft-simple companions.
	if strings.Contains(base, "-draft.") || strings.Contains(base, "_draft.") ||
		strings.Contains(base, "-draft-") || strings.Contains(base, "_draft_") ||
		strings.HasSuffix(stem, "-draft") || strings.HasSuffix(stem, "_draft") {
		return true, SpecSimple
	}
	return false, SpecNone
}

func firstPathToken(stem string) string {
	stem = strings.ToLower(strings.TrimSpace(stem))
	for i := 0; i < len(stem); i++ {
		c := stem[i]
		if c == '-' || c == '_' || c == '.' {
			return stem[:i]
		}
	}
	return stem
}

func delimitedToken(s, tok string) bool {
	if tok == "" || !strings.Contains(s, tok) {
		return false
	}
	for i := 0; i+len(tok) <= len(s); i++ {
		if s[i:i+len(tok)] != tok {
			continue
		}
		leftOK := i == 0 || !isDraftAlnum(s[i-1])
		rightOK := i+len(tok) == len(s) || !isDraftAlnum(s[i+len(tok)])
		if leftOK && rightOK {
			return true
		}
	}
	return false
}

func isDraftAlnum(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') || (b >= 'A' && b <= 'Z')
}

// assistantSidecarSpec maps *-assistant*.gguf companions that omit an
// official mtp-/dflash- prefix (Gemma 4 MTP, Muse Glimmer DFlash).
func assistantSidecarSpec(base string) SpecType {
	if !strings.Contains(base, "assistant") {
		return SpecNone
	}
	if strings.Contains(base, "glimmer") || strings.Contains(base, "museglimmer") {
		return SpecDFlash
	}
	if strings.Contains(base, "gemma-4") || strings.Contains(base, "gemma4") {
		return SpecMTP
	}
	return SpecNone
}

// SpecShortLabel is a compact UI tag for a --spec-type (dflash, eagle3, …).
func SpecShortLabel(st SpecType) string {
	switch st {
	case SpecDFlash:
		return "dflash"
	case SpecEagle3:
		return "eagle3"
	case SpecDSpark:
		return "dspark"
	case SpecMTP:
		return "mtp-draft"
	case SpecSimple:
		return "draft"
	default:
		return ""
	}
}

// DetectSpeculativeDraft combines architecture, official sidecar prefixes,
// draft-only metadata keys, and name heuristics. Draft sidecars must not be
// labeled as vision/audio chat models.
//
// Note: a trunk GGUF that merely embeds NextN/MTP heads (*.nextn_predict_layers)
// is NOT a speculative draft — that is HasMTP on a normal (possibly multimodal)
// chat model. Only mtp-/eagle3-/dflash-/dspark- sidecars and dedicated draft
// arches are treated as drafts.
func DetectSpeculativeDraft(arch, name, path string, raw map[string]any) (isDraft bool, spec SpecType) {
	if st := SpecTypeForArch(arch); st != SpecNone {
		return true, st
	}
	if st := SidecarSpecType(path); st != SpecNone {
		return true, st
	}
	if st := SidecarSpecType(name); st != SpecNone {
		return true, st
	}
	if HasSpeculativeDraftKeys(raw) {
		// Prefer arch/name refinement; default to simple.
		if ok, st := LooksLikeSpeculativeDraftName(name); ok {
			return true, st
		}
		if ok, st := LooksLikeSpeculativeDraftName(path); ok {
			return true, st
		}
		return true, SpecSimple
	}
	if ok, st := LooksLikeSpeculativeDraftName(name); ok {
		return true, st
	}
	if ok, st := LooksLikeSpeculativeDraftName(path); ok {
		return true, st
	}
	return false, SpecNone
}

// InferSpecType chooses --spec-type when a draft model is in use:
//  1. explicit user value wins
//  2. else draft architecture / official sidecar prefix / name
//  3. else draft-simple
//
// When there is no draft path, returns SpecNone unless explicit is set.
// Fused-trunk MTP (HasMTP without a sidecar) is enabled by the UI setting
// SpecType to draft-mtp explicitly — it must not be inferred just because
// nextn_predict_layers is present.
func InferSpecType(explicit, draftPath string, draftSpec SpecType) SpecType {
	if e := SpecType(strings.TrimSpace(explicit)); e != SpecNone {
		return e
	}
	if draftPath == "" {
		return SpecNone
	}
	if draftSpec != SpecNone {
		return draftSpec
	}
	if st := SidecarSpecType(draftPath); st != SpecNone {
		return st
	}
	if ok, st := LooksLikeSpeculativeDraftName(draftPath); ok {
		return st
	}
	return SpecSimple
}

// ClearMultimodal strips vision/audio/projector flags from a draft model.
func (md *Metadata) ClearMultimodal() {
	md.HasVision = false
	md.HasAudio = false
	md.Multimodal = false
	md.Projector = false
}

// ApplySpeculativeFlags sets SpeculativeDraft / HasMTP / SpecType on md using
// architecture, KV, and optional path/name. Safe to call after extract().
func (md *Metadata) ApplySpeculativeFlags(path string) {
	md.NextnPredictLayers = NextnPredictLayers(md.Architecture, md.Raw)
	md.HasMTP = md.NextnPredictLayers > 0 || HasNextnTensors(md.Raw)

	isDraft, spec := DetectSpeculativeDraft(md.Architecture, md.Name, path, md.Raw)
	md.SpeculativeDraft = isDraft
	md.SpecType = spec
	if isDraft {
		md.ClearMultimodal()
		// Standalone mtp- drafts always carry NextN heads.
		if spec == SpecMTP {
			md.HasMTP = true
		}
	}
}
