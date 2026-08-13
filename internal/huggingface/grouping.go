package huggingface

import (
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/gguf"
)

// FileKind classifies a repository file.
type FileKind string

const (
	KindGGUF      FileKind = "gguf"
	KindProjector FileKind = "projector"
	KindDraft     FileKind = "draft"
	KindOther     FileKind = "other"
)

// GroupedFile is one downloadable file inside a group.
type GroupedFile struct {
	Path     string `json:"path"`
	Size     int64  `json:"size"`
	Kind     string `json:"kind"` // gguf|projector|draft
	Part     int    `json:"part,omitempty"`
	Quant    string `json:"quant,omitempty"`
	SpecType string `json:"spec_type,omitempty"`
}

// FileGroup is one logical download unit: a single GGUF, a complete split
// set, or a vision set. Sets are only offered whole — never per-shard.
type FileGroup struct {
	ID          string        `json:"id"`    // stable, derived from base name + quant
	Label       string        `json:"label"` // e.g. "Q4_K_M" or "IQ4_XS · MTP"
	Quant       string        `json:"quant"`
	Split       bool          `json:"split"`
	Parts       int           `json:"parts"`
	Vision      bool          `json:"vision"`              // includes an mmproj file
	Draft       bool          `json:"draft"`               // speculative sidecar, not a chat model
	MTP         string        `json:"mtp,omitempty"`       // "" | "mtp" | "mtp-draft"
	SpecType    string        `json:"spec_type,omitempty"` // llama.cpp --spec-type when Draft
	TotalBytes  int64         `json:"total_bytes"`
	Files       []GroupedFile `json:"files"`
	EstMemBytes int64         `json:"est_memory_bytes"` // rough estimate, clearly marked
}

var (
	// name-Q4_K_M.gguf, name.IQ4_XS.gguf, name.q8_0.gguf
	quantRe = regexp.MustCompile(`(?i)[.\-_]((?:IQ[1-4]_[A-Z0-9]+|Q[1-8]_[A-Z0-9_]+(?:_[SMXL])?|F16|F32|BF16|TQ[12]_0|MXFP4))`)
	// name-00001-of-00003.gguf
	splitRe = regexp.MustCompile(`(?i)-(\d{5})-of-(\d{5})\.gguf$`)
	// mmproj-model-f16.gguf, model.mmproj-Q8_0.gguf
	projRe = regexp.MustCompile(`(?i)(mmproj|mm-proj|projector)`)
)

// classifyFile returns the kind for a repo path.
func classifyFile(path string) FileKind {
	lower := strings.ToLower(path)
	if !strings.HasSuffix(lower, ".gguf") {
		return KindOther
	}
	base := lower[strings.LastIndex(lower, "/")+1:]
	if projRe.MatchString(base) {
		return KindProjector
	}
	if ok, _ := gguf.LooksLikeSpeculativeDraftName(base); ok {
		return KindDraft
	}
	return KindGGUF
}

func fileSpecType(path string) string {
	ok, st := gguf.LooksLikeSpeculativeDraftName(path)
	if !ok {
		return ""
	}
	return string(st)
}

// quantOf extracts the quantization token from a filename.
// Prefer the rightmost match in the basename so names like
// "model.f16.gguf.Q4_K_M.gguf" (common for requantized F16 uploads)
// resolve to Q4_K_M rather than the leftover F16 fragment. When the
// basename has no token, fall back to a parent directory that is itself
// a known quant label (e.g. Q4_K_M/model.gguf).
func quantOf(path string) string {
	if q := gguf.UnslothDynamicQuant(path, ""); q != "" {
		return q
	}
	base := path
	dir := ""
	if i := strings.LastIndex(base, "/"); i >= 0 {
		dir = base[:i]
		base = base[i+1:]
	}
	if m := quantRe.FindAllStringSubmatch(base, -1); len(m) > 0 {
		return strings.ToUpper(m[len(m)-1][1])
	}
	if dir != "" {
		folder := dir
		if i := strings.LastIndex(folder, "/"); i >= 0 {
			folder = folder[i+1:]
		}
		folder = strings.ToUpper(folder)
		if rest, ok := strings.CutPrefix(folder, "UD-"); ok {
			if _, known := quantRanks[rest]; known {
				return folder
			}
		}
		if _, ok := quantRanks[folder]; ok {
			return folder
		}
	}
	return ""
}

// GroupFiles organizes repository files into logical download units:
// split shards are merged into one set, and non-GGUF files are excluded.
// Non-split GGUFs are one group per file so mixed MTP / non-MTP quants in the
// same repo (same quant token, different filenames) stay distinct.
// Projector (mmproj) files and speculative sidecars (dflash-/eagle3-/mtp-/…)
// are returned separately: the UI offers include-vision / include-drafter
// toggles that append them to a chosen trunk group, instead of listing
// companions as fake quants (or pairing mmproj onto a drafter).
func GroupFiles(files []FileEntry) (groups []FileGroup, projectors []GroupedFile, drafts []GroupedFile) {
	files = dedupeAliasFiles(files)

	type key struct{ stem, quant, mtp string }
	regular := map[key][]GroupedFile{}
	splits := map[key][]GroupedFile{}
	splitDeclared := map[key]int{} // declared shard count from filename
	projByQuant := map[string][]GroupedFile{}
	var draftFiles []GroupedFile

	for _, f := range files {
		kind := classifyFile(f.Path)
		if kind == KindOther {
			continue
		}
		q := quantOf(f.Path)
		mtp := FileMTP(f.Path)
		spec := fileSpecType(f.Path)
		gf := GroupedFile{Path: f.Path, Size: f.Size, Kind: string(kind), Quant: q, SpecType: spec}
		if kind == KindProjector {
			projByQuant[q] = append(projByQuant[q], gf)
			continue
		}
		if kind == KindDraft {
			draftFiles = append(draftFiles, gf)
			continue
		}
		if m := splitRe.FindStringSubmatch(f.Path); len(m) == 3 {
			part, _ := strconv.Atoi(m[1])
			declared, _ := strconv.Atoi(m[2])
			gf.Part = part
			k := key{stem: splitStem(f.Path), quant: q, mtp: mtp}
			splits[k] = append(splits[k], gf)
			if declared > splitDeclared[k] {
				splitDeclared[k] = declared
			}
			continue
		}
		// One group per non-split GGUF. Key includes the full relative path so
		// identical basenames in different folders stay separate, and so MTP /
		// non-MTP siblings that share a quant never merge.
		k := key{stem: strings.TrimSuffix(f.Path, ".gguf"), quant: q, mtp: mtp}
		regular[k] = append(regular[k], gf)
	}

	groups = []FileGroup{}
	seenID := map[string]int{}

	uniqID := func(stem, quant, mtp string) string {
		id := strings.ToLower(strings.NewReplacer("/", "-", " ", "-", ".", "-").Replace(stem))
		if quant != "" {
			id += "-" + strings.ToLower(quant)
		}
		if mtp != "" {
			id += "-" + mtp
		}
		if n := seenID[id]; n > 0 {
			seenID[id] = n + 1
			return id + "-" + strconv.Itoa(n+1)
		}
		seenID[id] = 1
		return id
	}

	for k, fs := range regular {
		g := FileGroup{
			ID:    uniqID(k.stem, k.quant, k.mtp),
			Label: quantLabel(k.quant, k.mtp, "", fs[0].Path),
			Quant: k.quant,
			MTP:   k.mtp,
			Files: sortedFiles(fs),
		}
		for _, f := range fs {
			g.TotalBytes += f.Size
		}
		groups = append(groups, g)
	}
	for k, fs := range splits {
		parts := splitDeclared[k]
		var total int64
		for _, f := range fs {
			if f.Part > parts {
				parts = f.Part
			}
			total += f.Size
		}
		g := FileGroup{
			ID:         uniqID(k.stem, k.quant, k.mtp),
			Label:      quantLabel(k.quant, k.mtp, "", fs[0].Path) + " split set",
			Quant:      k.quant,
			MTP:        k.mtp,
			Split:      true,
			Parts:      parts,
			Files:      sortedFiles(fs),
			TotalBytes: total,
		}
		groups = append(groups, g)
	}

	// Projectors (vision) are reported once for the whole repo; they pair
	// with any quantization at download time via the UI toggle.
	var allProjectors []GroupedFile
	for _, ps := range projByQuant {
		allProjectors = append(allProjectors, ps...)
	}
	allProjectors = sortedFiles(allProjectors)
	allDrafts := sortedFiles(draftFiles)

	if len(groups) == 0 && len(allProjectors) > 0 {
		// Projector-only repository corner case: offer the set directly and
		// suppress the separate return so it is not added twice.
		g := FileGroup{ID: uniqID("projectors", "", ""), Label: "Projector files", Vision: true, Files: allProjectors}
		for _, f := range allProjectors {
			g.TotalBytes += f.Size
		}
		groups = append(groups, g)
		allProjectors = nil
	}
	if len(groups) == 0 && len(allDrafts) > 0 {
		// Draft-only repository: each sidecar is a downloadable group, never
		// paired with vision.
		for _, d := range allDrafts {
			mtp := ""
			if d.SpecType == string(gguf.SpecMTP) {
				mtp = "mtp-draft"
			}
			g := FileGroup{
				ID:       uniqID(strings.TrimSuffix(d.Path, ".gguf"), d.Quant, mtp),
				Label:    quantLabel(d.Quant, mtp, d.SpecType, d.Path),
				Quant:    d.Quant,
				Draft:    true,
				MTP:      mtp,
				SpecType: d.SpecType,
				Files:    []GroupedFile{d},
			}
			g.TotalBytes = d.Size
			groups = append(groups, g)
		}
		allDrafts = nil
	}

	disambiguateGroupLabels(groups)

	// Memory estimate: file size + ~25% overhead for context/KV at defaults.
	for i := range groups {
		groups[i].EstMemBytes = groups[i].TotalBytes + groups[i].TotalBytes/4
	}

	// Sort by quantization rank: smallest/heaviest-compressed quants first,
	// full-precision last. Within a quant, plain before MTP, then by size.
	sort.SliceStable(groups, func(a, b int) bool {
		ra, rb := quantRank(groups[a]), quantRank(groups[b])
		if ra != rb {
			return ra < rb
		}
		if groups[a].Quant == groups[b].Quant && groups[a].MTP != groups[b].MTP {
			return groups[a].MTP < groups[b].MTP
		}
		if groups[a].Draft != groups[b].Draft {
			return !groups[a].Draft && groups[b].Draft
		}
		if groups[a].TotalBytes != groups[b].TotalBytes {
			return groups[a].TotalBytes < groups[b].TotalBytes
		}
		return groups[a].Label < groups[b].Label
	})
	return groups, allProjectors, allDrafts
}

// aliasSizeSlop is how close two large GGUF sizes must be to treat the
// shorter / generic name as a duplicate alias (Muse Glimmer kquant copies
// differ by ~3 KiB). Small files are never slop-matched — test fixtures and
// tiny GGUFs would otherwise collapse into one group.
const (
	aliasSizeSlop int64 = 16 << 10
	aliasMinSize  int64 = 32 << 20
)

// aliasItem is one GGUF considered by alias dedupe.
type aliasItem struct {
	kind  FileKind
	idx   int
	score int
	size  int64
}

// dedupeAliasFiles drops near-duplicate GGUFs that are alias names of a
// more specific sibling (mmproj-kquant.gguf next to mmproj-Model-Q4_K_M.gguf).
// Size-unknown files (0) are never clustered.
func dedupeAliasFiles(files []FileEntry) []FileEntry {
	drop := map[int]bool{}
	var ggufs []aliasItem
	for i, f := range files {
		k := classifyFile(f.Path)
		if k == KindOther || f.Size <= 0 {
			continue
		}
		ggufs = append(ggufs, aliasItem{kind: k, idx: i, score: aliasScore(f.Path), size: f.Size})
	}
	sort.Slice(ggufs, func(i, j int) bool {
		if ggufs[i].kind != ggufs[j].kind {
			return ggufs[i].kind < ggufs[j].kind
		}
		if ggufs[i].size != ggufs[j].size {
			return ggufs[i].size < ggufs[j].size
		}
		return ggufs[i].score > ggufs[j].score
	})
	for i := 0; i < len(ggufs); i++ {
		if drop[ggufs[i].idx] {
			continue
		}
		for j := i + 1; j < len(ggufs); j++ {
			if ggufs[j].kind != ggufs[i].kind {
				break
			}
			if !isAliasPair(ggufs[i], ggufs[j], files) {
				if ggufs[j].size-ggufs[i].size > aliasSizeSlop {
					break
				}
				continue
			}
			if ggufs[j].score > ggufs[i].score {
				drop[ggufs[i].idx] = true
				break
			}
			drop[ggufs[j].idx] = true
		}
	}
	out := make([]FileEntry, 0, len(files))
	for i, f := range files {
		if drop[i] {
			continue
		}
		out = append(out, f)
	}
	return out
}

func isAliasPair(a, b aliasItem, files []FileEntry) bool {
	if a.kind != b.kind {
		return false
	}
	delta := a.size - b.size
	if delta < 0 {
		delta = -delta
	}
	scoreGap := a.score - b.score
	if scoreGap < 0 {
		scoreGap = -scoreGap
	}
	if scoreGap < 20 {
		return false
	}
	large := a.size >= aliasMinSize && b.size >= aliasMinSize
	if large && delta <= aliasSizeSlop {
		return true
	}
	if delta == 0 && (looksLikeGenericAlias(files[a.idx].Path) || looksLikeGenericAlias(files[b.idx].Path)) {
		return true
	}
	return false
}

func looksLikeGenericAlias(path string) bool {
	stem := strings.TrimSuffix(strings.ToLower(filepathBase(path)), ".gguf")
	if stem == "dflash-kquant" || stem == "mmproj-kquant" {
		return true
	}
	return strings.Contains(stem, "kquant") && quantOf(path) == ""
}

func aliasScore(path string) int {
	base := strings.ToLower(filepathBase(path))
	stem := strings.TrimSuffix(base, ".gguf")
	score := 0
	if quantOf(path) != "" {
		score += 100
	}
	score += len(stem)
	if stem == "dflash-kquant" || stem == "mmproj-kquant" ||
		(strings.HasSuffix(stem, "-kquant") && quantOf(path) == "") {
		score -= 80
	}
	if strings.Contains(stem, "kquant") && quantOf(path) == "" {
		score -= 20
	}
	return score
}

func filepathBase(path string) string {
	if i := strings.LastIndex(path, "/"); i >= 0 {
		return path[i+1:]
	}
	return path
}

// quantLabel builds the Discover download-row title for a quant (+ MTP / draft).
func quantLabel(quant, mtp, spec, path string) string {
	label := orDefault(quant, "GGUF")
	if hint := mtpVariantHint(path); hint != "" && mtp == "mtp" {
		label += " · " + hint
	}
	switch mtp {
	case "mtp":
		label += " · MTP"
	case "mtp-draft":
		label += " · MTP draft"
	}
	if mtp != "mtp-draft" {
		switch spec {
		case string(gguf.SpecDFlash):
			label += " · DFlash draft"
		case string(gguf.SpecEagle3):
			label += " · EAGLE3 draft"
		case string(gguf.SpecDSpark):
			label += " · DSpark draft"
		case string(gguf.SpecMTP):
			label += " · MTP draft"
		case string(gguf.SpecSimple):
			label += " · draft"
		}
	}
	return label
}

// disambiguateGroupLabels appends a short stem hint when two groups would
// otherwise share a label (e.g. two Q4_K_M trunks from different builds).
func disambiguateGroupLabels(groups []FileGroup) {
	byLabel := map[string][]int{}
	for i, g := range groups {
		byLabel[g.Label] = append(byLabel[g.Label], i)
	}
	for label, idxs := range byLabel {
		if len(idxs) < 2 {
			continue
		}
		hints := make([]string, len(idxs))
		used := map[string]int{}
		for n, i := range idxs {
			h := distinctiveStem(groups[i])
			hints[n] = h
			used[h]++
		}
		for n, i := range idxs {
			h := hints[n]
			if h == "" || used[h] == len(idxs) {
				continue // identical hint is not distinctive
			}
			groups[i].Label = label + " · " + h
		}
	}
}

func distinctiveStem(g FileGroup) string {
	if len(g.Files) == 0 {
		return ""
	}
	stem := strings.ToUpper(stemOf(g.Files[0].Path))
	parts := strings.FieldsFunc(stem, func(r rune) bool {
		return r == '-' || r == '_' || r == '.'
	})
	quant := strings.ToUpper(g.Quant)
	skip := map[string]bool{
		"GGUF": true, "GGML": true, "MODEL": true, "KQUANT": true,
		quant: true,
	}
	var keep []string
	for _, p := range parts {
		if skip[p] || p == "" {
			continue
		}
		if quant != "" && strings.HasPrefix(p, quant) {
			continue
		}
		switch p {
		case "17GB", "26GB", "32GB", "DYNAMIC", "XL", "L", "S", "M",
			"AMD", "LOW", "HIGH", "ULTRA", "FAST", "SPARSE", "DENSE", "UD":
			keep = append(keep, p)
		}
	}
	if len(keep) > 0 {
		if len(keep) > 2 {
			keep = keep[:2]
		}
		return strings.Join(keep, " ")
	}
	for i := len(parts) - 1; i >= 0; i-- {
		p := parts[i]
		if skip[p] || p == "" {
			continue
		}
		if len(p) >= 2 && len(p) <= 16 {
			return p
		}
	}
	return ""
}

// mtpVariantHint pulls a short build tag immediately before MTP in the
// filename (e.g. AMD / LOW in …-AMD-MTP-IQ4_XS), so same-quant MTP builds
// stay distinguishable in the UI.
func mtpVariantHint(path string) string {
	base := strings.ToUpper(stemOf(path))
	parts := strings.FieldsFunc(base, func(r rune) bool {
		return r == '-' || r == '_'
	})
	for i, p := range parts {
		if p != "MTP" || i == 0 {
			continue
		}
		prev := parts[i-1]
		switch prev {
		case "AMD", "LOW", "HIGH", "ULTRA", "FAST", "SPARSE", "DENSE":
			return prev
		}
	}
	return ""
}

// Known quantization sort order (smallest → largest).
var quantRanks = map[string]int{
	"IQ1_S": 1, "IQ1_M": 2,
	"IQ2_XXS": 3, "IQ2_XS": 4, "Q2_K_S": 5, "Q2_K": 6, "Q2_K_L": 7, "Q2_K_XL": 8, "IQ2_S": 9, "IQ2_M": 10,
	"IQ3_XXS": 11, "IQ3_XS": 12, "Q3_K_S": 13, "IQ3_S": 14, "IQ3_M": 15,
	"Q3_K_M": 16, "Q3_K_L": 17, "Q3_K_XL": 18,
	"IQ4_NL": 19, "IQ4_XS": 20, "Q4_0": 21, "Q4_K_S": 22, "Q4_K_M": 23, "Q4_K_L": 24, "Q4_K_XL": 25, "Q4_1": 26,
	"Q5_0": 27, "Q5_K_S": 28, "Q5_K_M": 29, "Q5_K_L": 30, "Q5_K_XL": 31, "Q5_1": 32,
	"Q6_K": 33, "Q6_K_L": 34, "Q6_K_XL": 35, "Q8_0": 36, "Q8_K_L": 37, "Q8_K_XL": 38,
	"TQ1_0": 39, "TQ2_0": 40, "MXFP4": 41,
	"F16": 50, "BF16": 51, "F32": 52,
}

// quantRank returns the sort rank of a group; unknown quants rank by a
// size-derived estimate between Q8_0 and F16.
func quantRank(g FileGroup) int {
	q := strings.TrimPrefix(g.Quant, "UD-")
	if r, ok := quantRanks[q]; ok {
		return r
	}
	return 45
}

func sortedFiles(fs []GroupedFile) []GroupedFile {
	out := append([]GroupedFile{}, fs...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Part != out[j].Part {
			return out[i].Part < out[j].Part
		}
		return out[i].Path < out[j].Path
	})
	return out
}

// stemOf returns the filename without directory and .gguf extension.
func stemOf(path string) string {
	base := path
	if i := strings.LastIndex(base, "/"); i >= 0 {
		base = base[i+1:]
	}
	return strings.TrimSuffix(base, ".gguf")
}

// splitStem strips the -00001-of-00003 suffix.
func splitStem(path string) string {
	return splitRe.ReplaceAllString(path, "")
}

func orDefault(s, d string) string {
	if s == "" {
		return d
	}
	return s
}
