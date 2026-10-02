package huggingface

import (
	"math"
	"path"
	"sort"
	"strconv"
	"strings"
)

// Download plan.
//
// A repository is presented as a set of components (the model, a vision
// projector, a drafter, a VAE, text encoders, …). Each component offers one
// option per stored precision; the user includes or leaves out components and
// picks one option for each, then confirms a single download. Nothing here
// downloads anything: the plan only describes what the repository offers and
// which choice is the sensible default.

// Plan kinds.
const (
	PlanChat      = "chat"      // GGUF language model repository
	PlanGenerator = "generator" // image / video generator repository
)

// PlanFile is one repository file inside an option.
type PlanFile struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
	Part int    `json:"part,omitempty"`
	// Dest keeps the repository folder when the file's name alone does not
	// say what it is (unet/diffusion_pytorch_model.safetensors); empty means
	// the file is saved flat next to the others.
	Dest string `json:"dest,omitempty"`
}

// PlanOption is one selectable precision of a component: a single file or a
// complete split set.
type PlanOption struct {
	ID          string     `json:"id"`
	Label       string     `json:"label"`             // "Q4_K_M", "BF16", "Default"
	Variant     string     `json:"variant,omitempty"` // which build, when a component has several
	Precision   string     `json:"precision"`         // canonical precision id ("" when unstated)
	Bits        float64    `json:"bits"`              // nominal bits per weight (16 when unstated)
	Class       string     `json:"class,omitempty"`
	Kind        string     `json:"kind,omitempty"` // drafter speculative type
	Tags        []string   `json:"tags,omitempty"`
	Files       []PlanFile `json:"files"`
	TotalBytes  int64      `json:"total_bytes"`
	EstMemBytes int64      `json:"est_memory_bytes,omitempty"`
	Recommended bool       `json:"recommended,omitempty"`
	// Warn is set when the option is unlikely to load in the engine that
	// runs it; it is still selectable.
	Warn string `json:"warn,omitempty"`
}

// PlanComponent is one part of a model the user can include or leave out.
type PlanComponent struct {
	ID    string `json:"id"`
	Role  string `json:"role"`
	Label string `json:"label"`
	// Short names the component inside a download label ("mmproj", "VAE").
	// Empty for the main model, which the label leads with.
	Short string `json:"short,omitempty"`
	Hint  string `json:"hint,omitempty"`
	// Selected is whether the component is included by default.
	Selected bool `json:"selected"`
	// Experimental components are not enabled unless the user opted in to
	// the feature in Settings; the client decides.
	Experimental bool `json:"experimental,omitempty"`
	// Default is the option picked for this component.
	Default string `json:"default"`
	// FollowPrecisionOf names a component whose chosen precision this one
	// should track when the user changes it (a drafter follows the model).
	FollowPrecisionOf string       `json:"follow_precision_of,omitempty"`
	Options           []PlanOption `json:"options"`
}

// Plan is everything a repository offers, as components.
type Plan struct {
	Kind       string          `json:"kind"`
	Components []PlanComponent `json:"components"`
	Notes      []string        `json:"notes,omitempty"`
}

// BuildPlan describes what a repository offers to download.
func BuildPlan(info *RepoInfo) Plan {
	paths := make([]string, 0, len(info.Files))
	for _, f := range info.Files {
		paths = append(paths, f.Path)
	}
	if kind := DetectDiffusion(info.ID, info.PipelineTag, info.Tags, paths); kind != DiffusionNone {
		return buildGeneratorPlan(info.Files)
	}
	return buildChatPlan(info.Files, DetectModalities(info.ID, info.PipelineTag, info.Tags, paths))
}

// pickByBits returns the index of the option nearest target bits per weight.
// Options flagged unlikely-to-load are skipped while a sound one exists, and
// 32-bit floats are skipped while a 16-bit-or-smaller option exists (they
// double the download for no visible gain). Ties go to the smaller file.
func pickByBits(opts []PlanOption, target float64) int {
	if len(opts) == 0 {
		return -1
	}
	var pool []int
	for i, o := range opts {
		if o.Warn == "" {
			pool = append(pool, i)
		}
	}
	if len(pool) == 0 {
		for i := range opts {
			pool = append(pool, i)
		}
	}
	var small []int
	for _, i := range pool {
		if opts[i].Bits <= 16 {
			small = append(small, i)
		}
	}
	if len(small) > 0 {
		pool = small
	}
	best := pool[0]
	for _, i := range pool[1:] {
		if optionCloser(opts[i], opts[best], target) {
			best = i
		}
	}
	return best
}

func optionCloser(a, b PlanOption, target float64) bool {
	da, db := math.Abs(a.Bits-target), math.Abs(b.Bits-target)
	if da != db {
		return da < db
	}
	if a.TotalBytes != b.TotalBytes {
		return a.TotalBytes < b.TotalBytes
	}
	// Same size: FP16 over BF16 — it is the reference format and the one
	// every backend accelerates.
	return a.Precision == "fp16" && b.Precision != "fp16"
}

// sortOptions orders options smallest precision first, then smallest file.
func sortOptions(opts []PlanOption) {
	sort.SliceStable(opts, func(i, j int) bool {
		if opts[i].Bits != opts[j].Bits {
			return opts[i].Bits < opts[j].Bits
		}
		if opts[i].TotalBytes != opts[j].TotalBytes {
			return opts[i].TotalBytes < opts[j].TotalBytes
		}
		if opts[i].Variant != opts[j].Variant {
			return opts[i].Variant < opts[j].Variant
		}
		return opts[i].Label < opts[j].Label
	})
}

// uniqueOptionIDs makes option ids unique within a component.
func uniqueOptionIDs(opts []PlanOption) {
	seen := map[string]int{}
	for i := range opts {
		id := opts[i].ID
		seen[id]++
		if n := seen[id]; n > 1 {
			opts[i].ID = id + "-" + strconv.Itoa(n)
		}
	}
}

// setDefault records the default option of a component by index and marks it
// recommended.
func setDefault(c *PlanComponent, idx int) {
	if idx < 0 || idx >= len(c.Options) {
		return
	}
	c.Default = c.Options[idx].ID
	c.Options[idx].Recommended = true
}

// optionIndex finds an option by id, or -1.
func optionIndex(opts []PlanOption, id string) int {
	for i, o := range opts {
		if o.ID == id {
			return i
		}
	}
	return -1
}

// fileCensus says what a repository holds when none of it is usable, so an
// empty picker explains itself ("14 files: .json x9, .md x1") instead of
// leaving the user guessing.
func fileCensus(files []FileEntry) string {
	if len(files) == 0 {
		return "Hugging Face returned no file list for this repository. It may be gated or private: accept its terms on Hugging Face and add your access token in Settings."
	}
	counts := map[string]int{}
	for _, f := range files {
		ext := strings.ToLower(path.Ext(f.Path))
		if ext == "" {
			ext = "(no extension)"
		}
		counts[ext]++
	}
	type kv struct {
		ext string
		n   int
	}
	var list []kv
	for e, n := range counts {
		list = append(list, kv{e, n})
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].n != list[j].n {
			return list[i].n > list[j].n
		}
		return list[i].ext < list[j].ext
	})
	parts := make([]string, 0, 6)
	for i, e := range list {
		if i == 5 {
			parts = append(parts, "…")
			break
		}
		parts = append(parts, e.ext+" ×"+strconv.Itoa(e.n))
	}
	return "The repository holds " + strconv.Itoa(len(files)) + " files (" + strings.Join(parts, ", ") + ")."
}

func hasModality(mods []string, m string) bool {
	for _, x := range mods {
		if strings.EqualFold(x, m) {
			return true
		}
	}
	return false
}
