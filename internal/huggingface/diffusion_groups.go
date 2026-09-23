package huggingface

import (
	"path/filepath"
	"sort"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/sdmodel"
)

// DiffusionGroup is one downloadable unit of a generator repo: a single-file
// checkpoint (SD1.5/FLUX .safetensors) or a diffusers bundle directory worth
// of weights (model_index.json + weight files). Checkpoint groups auto-attach
// the pipeline companions sd.cpp needs (VAE + text encoders) the way the LLM
// flow attaches an mmproj — a transformer-only checkpoint is not loadable
// without them. Optional extras (LoRA / upscaler / ControlNet) and unused
// alternatives stay in the separate components group.
type DiffusionGroup struct {
	ID         string          `json:"id"` // stable download group id
	Label      string          `json:"label"`
	Kind       string          `json:"kind"` // checkpoint|bundle|component
	TotalBytes int64           `json:"total_bytes"`
	Files      []DiffusionFile `json:"files"`
}

// DiffusionFile is one file inside a diffusion download group.
type DiffusionFile struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
	Role string `json:"role"` // checkpoint|vae|encoder|lora|upscaler|bundle|other
}

// roleFromSDRole maps an sdmodel component role onto the download-group
// vocabulary.
func roleFromSDRole(role string) string {
	switch role {
	case sdmodel.RoleVAE, sdmodel.RoleTAESD:
		return "vae"
	case sdmodel.RoleLLM, sdmodel.RoleT5XXL, sdmodel.RoleClipL, sdmodel.RoleClipG,
		sdmodel.RoleClipVision, sdmodel.RoleTokenizer:
		return "encoder"
	case sdmodel.RoleLoRA:
		return "lora"
	case sdmodel.RoleESRGAN, sdmodel.RoleControlNet:
		return "upscaler"
	}
	return ""
}

// diffusionRole classifies a weight/config file by its full repo path —
// directory AND basename — so `text_encoders/qwen3vl_8b_bf16.safetensors`
// reads as an encoder, never as a second "checkpoint". Delegates to
// sdmodel.ComponentRole so Discover offers the same roles the library
// scanner and the sd-server launcher agree on.
func diffusionRole(path string) string {
	base := strings.ToLower(filepath.Base(path))
	if strings.Contains(base, "model_index.json") {
		return "bundle"
	}
	if r := roleFromSDRole(sdmodel.ComponentRole(path, nil)); r != "" {
		return r
	}
	if isDiffusionWeight(path) {
		return "checkpoint"
	}
	return "other"
}

// requiredCompanionRoles are the pipeline parts sd.cpp cannot generate
// without once the weights are transformer-only: a VAE and at least one
// text encoder. They are attached to every checkpoint download.
var requiredCompanionRoles = []string{"vae", "encoder"}

// pickBestCompanions selects the most loadable file per required role
// (sdmodel.Preference: GGUF > bf16/f16/f32 > exotic quants) so a repo
// offering int8 and bf16 encoders downloads the one sd.cpp can actually
// load, not every alternative.
func pickBestCompanions(comps []DiffusionFile) (picked []DiffusionFile, rest []DiffusionFile) {
	best := map[string]DiffusionFile{}
	bestScore := map[string]int{}
	for _, c := range comps {
		required := false
		for _, r := range requiredCompanionRoles {
			if c.Role == r {
				required = true
				break
			}
		}
		if !required {
			rest = append(rest, c)
			continue
		}
		if s := sdmodel.Preference(c.Path); s > bestScore[c.Role] {
			bestScore[c.Role] = s
			best[c.Role] = c
		}
	}
	for _, c := range comps {
		required := false
		for _, r := range requiredCompanionRoles {
			if c.Role == r {
				required = true
				break
			}
		}
		if !required {
			continue
		}
		if b, ok := best[c.Role]; ok && b.Path == c.Path {
			picked = append(picked, c)
		} else {
			rest = append(rest, c)
		}
	}
	sort.Slice(picked, func(i, j int) bool { return picked[i].Path < picked[j].Path })
	sort.Slice(rest, func(i, j int) bool { return rest[i].Path < rest[j].Path })
	return picked, rest
}

// GroupDiffusionFiles organizes a generator repo's files into download
// groups. Single-file checkpoints (.safetensors/.ckpt) are one group each,
// with the required VAE/text-encoder companions auto-attached; diffusers
// bundles (model_index.json present) are one "bundle weights" group with
// every weight file plus a separate config group when present. Non-weight
// files (READMEs, images) are excluded.
func GroupDiffusionFiles(files []FileEntry) []DiffusionGroup {
	var out []DiffusionGroup
	hasIndex := false
	for _, f := range files {
		if strings.ToLower(filepath.Base(f.Path)) == "model_index.json" {
			hasIndex = true
			break
		}
	}

	seen := map[string]DiffusionFile{}
	add := func(path string, size int64) {
		if _, ok := seen[path]; ok {
			return
		}
		role := diffusionRole(path)
		if role == "other" {
			return
		}
		seen[path] = DiffusionFile{Path: path, Size: size, Role: role}
	}
	for _, f := range files {
		add(f.Path, f.Size)
	}
	if len(seen) == 0 {
		return nil
	}

	if !hasIndex {
		// Single-file repo: one group per checkpoint so mixed quants stay
		// distinct; the required pipeline companions ride along with each
		// checkpoint, the way the LLM flow includes the mmproj.
		var ckpts []DiffusionFile
		var comps []DiffusionFile
		for _, df := range seen {
			if df.Role == "checkpoint" {
				ckpts = append(ckpts, df)
			} else {
				comps = append(comps, df)
			}
		}
		sort.Slice(ckpts, func(i, j int) bool { return ckpts[i].Path < ckpts[j].Path })
		picked, rest := pickBestCompanions(comps)
		for _, ck := range ckpts {
			fs := append([]DiffusionFile{ck}, picked...)
			g := DiffusionGroup{
				ID:    "ckpt-" + strings.ToLower(safeGroupID(ck.Path)),
				Label: checkpointLabel(ck.Path),
				Kind:  "checkpoint",
				Files: fs,
			}
			for _, f := range fs {
				g.TotalBytes += f.Size
			}
			out = append(out, g)
		}
		if len(rest) > 0 {
			g := DiffusionGroup{ID: "components", Label: "LoRA / upscaler / ControlNet / alternatives", Kind: "component", Files: rest}
			for _, f := range rest {
				g.TotalBytes += f.Size
			}
			out = append(out, g)
		}
		return out
	}

	// Diffusers bundle: weights group + bundle config file.
	var weights []DiffusionFile
	var total int64
	for _, df := range seen {
		if df.Role == "bundle" {
			continue
		}
		weights = append(weights, df)
		total += df.Size
	}
	sort.Slice(weights, func(i, j int) bool { return weights[i].Path < weights[j].Path })
	if len(weights) > 0 {
		out = append(out, DiffusionGroup{
			ID: "bundle-weights", Label: "Diffusers weights", Kind: "bundle",
			TotalBytes: total, Files: weights,
		})
	}
	for _, df := range seen {
		if df.Role == "bundle" {
			out = append(out, DiffusionGroup{
				ID: "bundle-config", Label: "Bundle config (model_index.json)",
				Kind: "bundle", TotalBytes: df.Size, Files: []DiffusionFile{df},
			})
		}
	}
	return out
}

func safeGroupID(path string) string {
	r := strings.NewReplacer("/", "-", " ", "-", ".", "-")
	return strings.Trim(r.Replace(strings.TrimSuffix(path, filepath.Ext(path))), "-")
}

func checkpointLabel(path string) string {
	base := filepath.Base(path)
	base = strings.TrimSuffix(base, filepath.Ext(base))
	if base == "" {
		return "Checkpoint"
	}
	// Shorten common verbose prefixes but keep the stem recognizable.
	for _, p := range []string{"v1-5-pruned-emaonly", "sd_xl_base_1.0"} {
		if strings.EqualFold(base, p) {
			return base
		}
	}
	if len(base) > 64 {
		base = base[:64]
	}
	return base
}
