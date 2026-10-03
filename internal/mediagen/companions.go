package mediagen

import (
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/sdmodel"
)

// Pipeline companion roles understood by sd.cpp. A transformer-only
// checkpoint (ComfyUI-style GGUF dump or split diffusers weights) needs
// these as separate files; full single-file checkpoints embed them.
const (
	CompanionVAE        = "vae"
	CompanionTAESD      = "taesd"
	CompanionLLM        = "llm"
	CompanionT5XXL      = "t5xxl"
	CompanionClipL      = "clip_l"
	CompanionClipG      = "clip_g"
	CompanionClipVision = "clip_vision"
	CompanionTokenizer  = "tokenizer"
	CompanionUpscaler   = "esrgan"
	CompanionControlNet = "control_net"
)

// Companion describes one pipeline component: found locally (Status
// "ready", Path set) or required from the source repo (Status "missing",
// RepoPath + Size set).
type Companion struct {
	Role     string `json:"role"`
	Label    string `json:"label"`
	Status   string `json:"status"` // ready | missing | incompatible
	Reason   string `json:"reason,omitempty"`
	Path     string `json:"path,omitempty"`
	RepoPath string `json:"repo_path,omitempty"`
	Size     int64  `json:"size"`
}

var companionLabels = map[string]string{
	CompanionVAE:        "VAE",
	CompanionTAESD:      "Tiny AutoEncoder (TAESD)",
	CompanionLLM:        "LLM text encoder",
	CompanionT5XXL:      "T5-XXL text encoder",
	CompanionClipL:      "CLIP-L text encoder",
	CompanionClipG:      "CLIP-G text encoder",
	CompanionClipVision: "CLIP vision encoder",
	CompanionTokenizer:  "tokenizer.json",
	CompanionUpscaler:   "Upscaler (ESRGAN)",
	CompanionControlNet: "ControlNet",
}

// CompanionLabel returns the display name for a role.
func CompanionLabel(role string) string {
	if l, ok := companionLabels[role]; ok {
		return l
	}
	return role
}

// IsComponentDiffusionFile reports whether a checkpoint is a transformer-only
// component (ComfyUI-style split weights) rather than an all-in-one
// checkpoint. sd.cpp cannot version-detect transformer-only files through
// --model ("get sd version from file failed"); --diffusion-model classifies
// them from the tensor layout instead.
//
// Rule: true (use --diffusion-model) unless the tensor table shows an
// all-in-one checkpoint (diffusion weights AND an embedded VAE). This covers
// KV-less GGUF dumps, city96 ComfyUI-GGUFs that carry general.architecture
// KV but no VAE, and transformer-only .safetensors. When tensors cannot be
// read, fall back to the KV-less GGUF heuristic.
func IsComponentDiffusionFile(path string) bool {
	if tensors, err := sdmodel.TensorNames(path); err == nil && len(tensors) > 0 {
		return !sdmodel.IsFullCheckpoint(tensors)
	}
	if !strings.HasSuffix(strings.ToLower(path), ".gguf") {
		return false
	}
	md, err := gguf.ParseFile(path)
	if err != nil || md == nil {
		return false
	}
	// KV-less dump (e.g. ComfyUI-GGUF conversions): no architecture, no name.
	return strings.TrimSpace(md.Architecture) == "" && strings.TrimSpace(md.Name) == ""
}

// SourceRepoFor derives the Hugging Face id a managed download came from:
// <models>/<owner>--<repo>/<group>/<file> → "owner/repo". The library's
// source_repo column is preferred when set (see the API layer).
func SourceRepoFor(modelsDir, primaryPath string) string {
	rel, err := filepath.Rel(modelsDir, primaryPath)
	if err != nil || rel == "" || strings.HasPrefix(rel, "..") {
		return ""
	}
	top := strings.SplitN(rel, string(os.PathSeparator), 2)[0]
	if top == "" {
		return ""
	}
	if i := strings.Index(top, "--"); i > 0 && i+2 < len(top) {
		return top[:i] + "/" + top[i+2:]
	}
	return ""
}

// classifyCompanion maps a component file path to its sd.cpp role, or "".
// Both directory names (vae/, text_encoders/, clip_vision/) and basenames
// (qwen3vl_…, t5xxl_…, clip_l_…, tokenizer.json) count, so repo-mirrored
// and flat download layouts classify the same way. Delegates to
// sdmodel.ComponentRole so the library scanner and the launcher agree.
func classifyCompanion(path string) string {
	// What the download recorded for this file beats guessing from its name:
	// the model itself is never a companion, and a role the user ticked it as
	// is its role whatever it is called.
	role, recorded := sdmodel.DeclaredRole(path)
	if recorded && role == sdmodel.RoleModel {
		return ""
	}
	tensors, _ := sdmodel.TensorNames(path)
	if !recorded || role == sdmodel.RoleComponent {
		role = sdmodel.ComponentRole(path, tensors)
	}
	switch role {
	case sdmodel.RoleVAE:
		return CompanionVAE
	case sdmodel.RoleTAESD:
		return CompanionTAESD
	case sdmodel.RoleLLM:
		return CompanionLLM
	case sdmodel.RoleT5XXL:
		return CompanionT5XXL
	case sdmodel.RoleClipL:
		return CompanionClipL
	case sdmodel.RoleClipG:
		return CompanionClipG
	case sdmodel.RoleClipVision:
		return CompanionClipVision
	case sdmodel.RoleTokenizer:
		return CompanionTokenizer
	case sdmodel.RoleESRGAN:
		return CompanionUpscaler
	case sdmodel.RoleControlNet:
		return CompanionControlNet
	}
	return ""
}

func isCompanionWeight(path string) bool {
	lower := strings.ToLower(path)
	for _, ext := range []string{".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".json"} {
		if strings.HasSuffix(lower, ext) {
			return true
		}
	}
	return false
}

// companionPreference ranks how loadable a companion file is for
// sd.cpp (higher is better). Delegates to sdmodel.Preference so the
// launcher and the Hugging Face download grouper pick the same file.
func companionPreference(path string) int {
	return sdmodel.Preference(path)
}

// DiscoverCompanions walks the model's managed tree and returns the best
// local file per companion role. The primary checkpoint is skipped. Both
// repo-mirrored (…/vae/x.safetensors) and flat (…/components/x.safetensors)
// download layouts are understood. When several files map to one role the
// most sd.cpp-loadable wins (see companionPreference).
func DiscoverCompanions(modelRoot, primaryPath string) map[string]string {
	out := map[string]string{}
	best := map[string]int{}
	if modelRoot == "" {
		return out
	}
	primaryAbs, _ := filepath.Abs(primaryPath)
	_ = filepath.WalkDir(modelRoot, func(p string, e os.DirEntry, err error) error {
		if err != nil || e.IsDir() {
			return nil
		}
		if abs, err := filepath.Abs(p); err == nil && abs == primaryAbs {
			return nil
		}
		if !isCompanionWeight(e.Name()) {
			return nil
		}
		role := classifyCompanion(p)
		if role == "" {
			return nil
		}
		score := companionPreference(p)
		if role == CompanionVAE && vaeIncompatibility(p) != "" {
			// Keep it for diagnostics if it is the only candidate, but never
			// let a known incompatible file beat a usable local alternative.
			score = 1
		}
		if score > best[role] {
			best[role] = score
			out[role] = p
		}
		return nil
	})
	return out
}

// ApplyCompanions fills empty LoadSettings companion fields from locally
// discovered files. Explicit user choices always win.
func ApplyCompanions(s *LoadSettings, modelRoot, primaryPath string) {
	local := DiscoverCompanions(modelRoot, primaryPath)
	set := func(role string, dst *string) {
		if *dst == "" {
			if p, ok := local[role]; ok {
				*dst = p
			}
		}
	}
	set(CompanionVAE, &s.VAE)
	set(CompanionTAESD, &s.TAESD)
	set(CompanionLLM, &s.LLM)
	set(CompanionT5XXL, &s.T5XXL)
	set(CompanionClipL, &s.ClipL)
	set(CompanionClipG, &s.ClipG)
	set(CompanionClipVision, &s.ClipVision)
	set(CompanionTokenizer, &s.Tokenizer)
	set(CompanionUpscaler, &s.ESRGAN)
	// ControlNet is intentionally NOT auto-applied: wiring one silently
	// changes generation behaviour. It stays listed in ComponentReport so
	// the user can opt in explicitly. LoRA files are likewise never
	// auto-paired (they are not pipeline components).
}

// ModelRoot returns the managed model root directory for a checkpoint
// (the models/<owner>--<repo> tree holding it).
func ModelRoot(primaryPath string) string {
	dir := filepath.Dir(primaryPath)
	for i := 0; i < 4; i++ {
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		// The managed layout is <models>/<owner>--<repo>/<group>/<file>.
		if strings.Contains(filepath.Base(dir), "--") {
			return dir
		}
		dir = parent
	}
	return filepath.Dir(primaryPath)
}

// ComponentReport merges local discovery with the source-repo file list to
// produce the ready/missing view shown in the load dialog. repoFiles paths
// are repo-relative (vae/x.safetensors, text_encoders/y.safetensors). When
// a repo ships several files per role the most sd.cpp-loadable candidate is
// the one offered for download (see companionPreference).
func ComponentReport(modelRoot, primaryPath string, repoFiles []string, repoSizes map[string]int64) []Companion {
	local := DiscoverCompanions(modelRoot, primaryPath)
	bestPath := map[string]string{}
	bestScore := map[string]int{}
	for _, p := range repoFiles {
		role := classifyCompanion(p)
		if role == "" {
			continue
		}
		if score := companionPreference(p); score > bestScore[role] {
			bestScore[role] = score
			bestPath[role] = p
		}
	}
	seen := map[string]bool{}
	var out []Companion
	for role, p := range bestPath {
		seen[role] = true
		if lp, ok := local[role]; ok {
			out = append(out, Companion{
				Role: role, Label: CompanionLabel(role), Status: "ready",
				Path: lp, RepoPath: p, Size: repoSizes[p],
			})
			continue
		}
		out = append(out, Companion{
			Role: role, Label: CompanionLabel(role), Status: "missing",
			RepoPath: p, Size: repoSizes[p],
		})
	}
	// Local-only extras (e.g. offline, no repo known).
	for role, lp := range local {
		if seen[role] {
			continue
		}
		var size int64
		if st, err := os.Stat(lp); err == nil {
			size = st.Size()
		}
		out = append(out, Companion{
			Role: role, Label: CompanionLabel(role), Status: "ready",
			Path: lp, Size: size,
		})
	}
	for i := range out {
		if out[i].Role == CompanionVAE && out[i].Path != "" {
			if why := vaeIncompatibility(out[i].Path); why != "" {
				out[i].Status = "incompatible"
				out[i].Reason = why
			}
		}
	}
	// Stable order: companions that decide startup first.
	sort.Slice(out, func(i, j int) bool { return out[i].Role < out[j].Role })
	return out
}

// MissingCompanions filters a report down to the "missing" entries.
func MissingCompanions(report []Companion) []Companion {
	var out []Companion
	for _, c := range report {
		if c.Status == "missing" {
			out = append(out, c)
		}
	}
	return out
}
