// Package calibrate builds deterministic calibration and held-out evaluation
// corpora for quantization workflows. It accepts multiple domain sources
// (general prose, chat, code, reasoning/task), streams bounded records,
// normalizes them safely, deduplicates by content hash, applies domain
// quotas/weights with deterministic seeded interleaving, and partitions
// records into calibration vs held-out evaluation sets by stable content
// hash so no record can ever appear in both.
//
// Outputs are plain-text corpora plus a versioned JSON manifest containing
// per-source hashes and counts, the domain mix, the seed, and output hashes.
// All writes are atomic (temp file + rename). No network access.
package calibrate

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Well-known domain names. Sources may use any domain string, but these are
// the conventional ones and General is special: mixtures must include it.
const (
	DomainGeneral   = "general"
	DomainChat      = "chat"
	DomainCode      = "code"
	DomainReasoning = "reasoning"
)

// ManifestVersion is the schema version embedded in every manifest.
const ManifestVersion = 1

// recordSep separates records in plain-text corpus files.
const recordSep = "\n\n====\n\n"

// Record is a single normalized corpus entry.
type Record struct {
	Domain string
	Text   string
	Hash   string // hex sha256 of normalized text
}

// Source streams records for one domain. Next returns the next raw record
// text, or io.EOF when exhausted. Sources must be deterministic: given the
// same construction they must yield the same sequence.
type Source interface {
	Domain() string
	Next(ctx context.Context) (string, error)
}

// SourceFunc adapts a function to Source.
type SourceFunc struct {
	DomainName string
	NextFn     func(ctx context.Context) (string, error)
}

func (s SourceFunc) Domain() string { return s.DomainName }
func (s SourceFunc) Next(ctx context.Context) (string, error) {
	return s.NextFn(ctx)
}

// SliceSource is a deterministic Source over an in-memory slice.
func SliceSource(domain string, items []string) Source {
	i := 0
	return SourceFunc{
		DomainName: domain,
		NextFn: func(ctx context.Context) (string, error) {
			if err := ctx.Err(); err != nil {
				return "", err
			}
			if i >= len(items) {
				return "", io.EOF
			}
			s := items[i]
			i++
			return s, nil
		},
	}
}

// DomainSpec configures one domain's contribution.
type DomainSpec struct {
	Source Source
	// Quota caps records taken from this domain (0 = unlimited).
	Quota int
	// Weight biases interleaving frequency relative to other domains.
	// <= 0 is treated as 1.
	Weight float64
}

// Config controls corpus construction.
type Config struct {
	Domains []DomainSpec
	Seed    int64

	// CalibPercent is the percentage of records (0-100) assigned to the
	// calibration corpus. SearchPercent reserves a tuning-only holdout; the
	// remainder goes to final evaluation. SearchPercent zero preserves the
	// legacy two-way split.
	CalibPercent  int
	SearchPercent int

	// MaxRecordBytes bounds a single normalized record; larger records are
	// skipped. 0 defaults to 64 KiB.
	MaxRecordBytes int
	// MaxTotalRecords bounds the corpus size; 0 defaults to 1,000,000.
	MaxTotalRecords int

	// MinRecords, MinDomains and MinTokens are validation floors on the
	// final corpus.
	MinRecords int
	MinDomains int
	MinTokens  int

	// ChatWrap, if non-empty, wraps records whose domain is DomainChat.
	// It must contain exactly one {{content}} placeholder and is otherwise
	// model-agnostic; callers supply the template string.
	ChatWrap string

	// PartitionSalt domains the calib/eval split hash; changing it reshuffles
	// the partition deterministically.
	PartitionSalt string
}

// Stats summarizes a built corpus.
type Stats struct {
	TotalRecords   int
	CalibRecords   int
	SearchRecords  int
	EvalRecords    int
	TotalTokens    int
	DomainCounts   map[string]int
	DistinctHashes int
}

// tokenEstimate approximates tokens as words + punctuation-ish overhead.
func tokenEstimate(text string) int {
	words := 0
	inWord := false
	for _, r := range text {
		if unicode.IsSpace(r) {
			inWord = false
		} else if !inWord {
			words++
			inWord = true
		}
	}
	return words + len(text)/16
}

// normalize trims, validates UTF-8, strips control characters (except
// newline/tab), and collapses 3+ consecutive newlines. Returns ok=false for
// records that are malformed or empty after normalization.
func normalize(raw string, maxBytes int) (string, bool) {
	if !utf8.ValidString(raw) {
		return "", false
	}
	var b strings.Builder
	b.Grow(len(raw))
	nl := 0
	for _, r := range raw {
		if r == '\n' {
			nl++
			if nl > 2 {
				continue
			}
			b.WriteRune(r)
			continue
		}
		nl = 0
		if unicode.IsControl(r) && r != '\t' {
			continue
		}
		b.WriteRune(r)
	}
	s := strings.TrimSpace(b.String())
	if s == "" || len(s) > maxBytes {
		return "", false
	}
	return s, true
}

// wrapChat applies a chat template string around content.
func wrapChat(template, content string) (string, error) {
	const ph = "{{content}}"
	if strings.Count(template, ph) != 1 {
		return "", fmt.Errorf("calibrate: chat template must contain exactly one %s placeholder", ph)
	}
	return strings.Replace(template, ph, content, 1), nil
}

func hashText(parts ...string) string {
	h := sha256.New()
	for _, p := range parts {
		h.Write([]byte(p))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// partitionFor returns true if the record belongs in the calibration set.
// The split key hashes the content hash plus salt, so a record's partition is
// stable, content-driven, and identical text can never land in both sets.
func partitionFor(contentHash, salt string, calibPercent int) bool {
	return partitionBucket(contentHash, salt) < uint64(calibPercent)
}

func partitionBucket(contentHash, salt string) uint64 {
	sum := sha256.Sum256([]byte(salt + "|" + contentHash))
	v := uint64(sum[0])<<56 | uint64(sum[1])<<48 | uint64(sum[2])<<40 |
		uint64(sum[3])<<32 | uint64(sum[4])<<24 | uint64(sum[5])<<16 |
		uint64(sum[6])<<8 | uint64(sum[7])
	return v % 100
}

// validateConfig checks static config errors before any I/O.
func validateConfig(cfg *Config) error {
	if len(cfg.Domains) == 0 {
		return errors.New("calibrate: at least one domain source is required")
	}
	if cfg.CalibPercent < 0 || cfg.CalibPercent > 100 {
		return errors.New("calibrate: CalibPercent must be in [0,100]")
	}
	if cfg.SearchPercent < 0 || cfg.CalibPercent+cfg.SearchPercent > 100 {
		return errors.New("calibrate: SearchPercent must be >= 0 and CalibPercent+SearchPercent must be <= 100")
	}
	seen := map[string]bool{}
	for i, d := range cfg.Domains {
		if d.Source == nil {
			return fmt.Errorf("calibrate: domain spec %d has nil source", i)
		}
		name := d.Source.Domain()
		if name == "" {
			return fmt.Errorf("calibrate: domain spec %d has empty domain name", i)
		}
		if seen[name] {
			return fmt.Errorf("calibrate: duplicate domain %q", name)
		}
		seen[name] = true
	}
	if cfg.ChatWrap != "" {
		if _, err := wrapChat(cfg.ChatWrap, "x"); err != nil {
			return err
		}
	}
	return nil
}

// BuildResult describes the built corpus.
type BuildResult struct {
	Stats         Stats
	CalibRecords  []Record
	SearchRecords []Record
	EvalRecords   []Record
}

// collect streams all sources, normalizes, dedupes, applies quotas, then
// interleaves deterministically and partitions.
func collect(ctx context.Context, cfg *Config) (*BuildResult, error) {
	if err := validateConfig(cfg); err != nil {
		return nil, err
	}
	maxRec := cfg.MaxRecordBytes
	if maxRec <= 0 {
		maxRec = 64 << 10
	}
	maxTotal := cfg.MaxTotalRecords
	if maxTotal <= 0 {
		maxTotal = 1_000_000
	}

	seen := make(map[string]struct{})
	perDomain := make(map[string][]Record, len(cfg.Domains))

	for _, d := range cfg.Domains {
		dom := d.Source.Domain()
		count := 0
		for {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if d.Quota > 0 && count >= d.Quota {
				break
			}
			raw, err := d.Source.Next(ctx)
			if err == io.EOF {
				break
			}
			if err != nil {
				return nil, fmt.Errorf("calibrate: source %q: %w", dom, err)
			}
			text, ok := normalize(raw, maxRec)
			if !ok {
				continue
			}
			if dom == DomainChat && cfg.ChatWrap != "" {
				w, err := wrapChat(cfg.ChatWrap, text)
				if err != nil {
					return nil, err
				}
				if len(w) > maxRec {
					continue
				}
				text = w
			}
			h := hashText(text)
			if _, dup := seen[h]; dup {
				continue
			}
			seen[h] = struct{}{}
			perDomain[dom] = append(perDomain[dom], Record{Domain: dom, Text: text, Hash: h})
			count++
			if count >= maxTotal {
				break
			}
		}
	}

	order := interleave(cfg, perDomain, maxTotal)

	res := &BuildResult{Stats: Stats{DomainCounts: map[string]int{}, DistinctHashes: len(seen)}}
	for _, r := range order {
		bucket := partitionBucket(r.Hash, cfg.PartitionSalt)
		switch {
		case bucket < uint64(cfg.CalibPercent):
			res.CalibRecords = append(res.CalibRecords, r)
			res.Stats.CalibRecords++
		case bucket < uint64(cfg.CalibPercent+cfg.SearchPercent):
			res.SearchRecords = append(res.SearchRecords, r)
			res.Stats.SearchRecords++
		default:
			res.EvalRecords = append(res.EvalRecords, r)
			res.Stats.EvalRecords++
		}
		res.Stats.TotalRecords++
		res.Stats.DomainCounts[r.Domain]++
		res.Stats.TotalTokens += tokenEstimate(r.Text)
	}
	// Tiny corpora can hash entirely into one bucket. Preserve strict
	// disjointness while moving one deterministic record so every requested
	// partition exists; larger corpora retain the content-hash split exactly.
	moveLast := func(from *[]Record, to *[]Record) bool {
		if len(*from) <= 1 {
			return false
		}
		n := len(*from) - 1
		*to = append(*to, (*from)[n])
		*from = (*from)[:n]
		return true
	}
	if cfg.SearchPercent > 0 && len(res.SearchRecords) == 0 {
		if !moveLast(&res.CalibRecords, &res.SearchRecords) {
			moveLast(&res.EvalRecords, &res.SearchRecords)
		}
	}
	if cfg.CalibPercent+cfg.SearchPercent < 100 && len(res.EvalRecords) == 0 {
		if !moveLast(&res.CalibRecords, &res.EvalRecords) {
			moveLast(&res.SearchRecords, &res.EvalRecords)
		}
	}
	if cfg.CalibPercent > 0 && len(res.CalibRecords) == 0 {
		if !moveLast(&res.EvalRecords, &res.CalibRecords) {
			moveLast(&res.SearchRecords, &res.CalibRecords)
		}
	}
	res.Stats.CalibRecords = len(res.CalibRecords)
	res.Stats.SearchRecords = len(res.SearchRecords)
	res.Stats.EvalRecords = len(res.EvalRecords)
	if err := res.Stats.validate(cfg); err != nil {
		return nil, err
	}
	return res, nil
}

func (s Stats) validate(cfg *Config) error {
	if s.TotalRecords < cfg.MinRecords {
		return fmt.Errorf("calibrate: corpus has %d records, below minimum %d", s.TotalRecords, cfg.MinRecords)
	}
	if len(s.DomainCounts) < cfg.MinDomains {
		return fmt.Errorf("calibrate: corpus spans %d domains, below minimum %d", len(s.DomainCounts), cfg.MinDomains)
	}
	if s.TotalTokens < cfg.MinTokens {
		return fmt.Errorf("calibrate: corpus has ~%d tokens, below minimum %d", s.TotalTokens, cfg.MinTokens)
	}
	if cfg.SearchPercent > 0 && s.SearchRecords == 0 {
		return fmt.Errorf("calibrate: tuning holdout is empty; provide more source records")
	}
	if cfg.CalibPercent+cfg.SearchPercent < 100 && s.EvalRecords == 0 {
		return fmt.Errorf("calibrate: evaluation holdout is empty; provide more source records")
	}
	return nil
}

// interleave merges per-domain record lists into one deterministic sequence
// using a seeded weighted pick. Domains keep their internal source order.
func interleave(cfg *Config, perDomain map[string][]Record, maxTotal int) []Record {
	type slot struct {
		dom    string
		weight float64
	}
	var slots []slot
	var total int
	for _, d := range cfg.Domains {
		dom := d.Source.Domain()
		recs := perDomain[dom]
		if len(recs) == 0 {
			continue
		}
		w := d.Weight
		if w <= 0 {
			w = 1
		}
		slots = append(slots, slot{dom: dom, weight: w})
		total += len(recs)
	}
	if total > maxTotal {
		total = maxTotal
	}
	sort.Slice(slots, func(i, j int) bool { return slots[i].dom < slots[j].dom })

	rng := rand.New(rand.NewSource(cfg.Seed))
	idx := make(map[string]int, len(slots))
	out := make([]Record, 0, total)
	for len(out) < total && len(slots) > 0 {
		var wsum float64
		for _, s := range slots {
			wsum += s.weight
		}
		pick := rng.Float64() * wsum
		si := 0
		for i, s := range slots {
			pick -= s.weight
			if pick <= 0 {
				si = i
				break
			}
		}
		dom := slots[si].dom
		recs := perDomain[dom]
		out = append(out, recs[idx[dom]])
		idx[dom]++
		if idx[dom] >= len(recs) {
			slots = append(slots[:si], slots[si+1:]...)
		}
	}
	return out
}

// Mixture describes a task-aware domain mixture. Weights are relative;
// the general domain must carry a nonzero weight.
type Mixture struct {
	Weights map[string]float64
	// MinGeneralWeight is the required floor for DomainGeneral
	// (default check is simply > 0).
	MinGeneralWeight float64
}

// Validate enforces the nonzero general-domain floor.
func (m Mixture) Validate() error {
	w, ok := m.Weights[DomainGeneral]
	if !ok || w <= 0 || w < m.MinGeneralWeight {
		return fmt.Errorf("calibrate: mixture requires a nonzero %q domain weight (floor %v)", DomainGeneral, m.MinGeneralWeight)
	}
	for dom, w := range m.Weights {
		if w < 0 {
			return fmt.Errorf("calibrate: mixture has negative weight for domain %q", dom)
		}
	}
	return nil
}

// TaskSpec names a task and its domain mixture.
type TaskSpec struct {
	Name    string
	Mixture Mixture
}

// TaskCorpora builds one corpus per task sharing the same sources. Each
// source may only be consumed once, so callers pass a factory that returns
// fresh sources per task.
func TaskCorpora(ctx context.Context, tasks []TaskSpec, seed int64,
	factory func(task string) []DomainSpec, cfgBase Config) (map[string]*BuildResult, error) {
	if len(tasks) == 0 {
		return nil, errors.New("calibrate: at least one task is required")
	}
	out := make(map[string]*BuildResult, len(tasks))
	for i, t := range tasks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := t.Mixture.Validate(); err != nil {
			return nil, fmt.Errorf("task %q: %w", t.Name, err)
		}
		cfg := cfgBase
		cfg.Seed = seed + int64(i) // stable per-task seed derivation
		cfg.PartitionSalt = cfgBase.PartitionSalt + "|" + t.Name
		specs := factory(t.Name)
		for j := range specs {
			if w, ok := t.Mixture.Weights[specs[j].Source.Domain()]; ok {
				specs[j].Weight = w
			} else {
				specs[j].Weight = 0 // excluded from this task's mixture
			}
		}
		// weight 0 domains must not be dropped entirely: keep only weighted ones
		kept := specs[:0]
		for _, s := range specs {
			if s.Weight > 0 {
				kept = append(kept, s)
			}
		}
		cfg.Domains = kept
		res, err := collect(ctx, &cfg)
		if err != nil {
			return nil, fmt.Errorf("task %q: %w", t.Name, err)
		}
		out[t.Name] = res
	}
	return out, nil
}

// SourceInfo is the manifest entry for one domain source.
type SourceInfo struct {
	Domain string `json:"domain"`
	Count  int    `json:"count"`
	// Hash chains the content hashes of accepted records in source order.
	Hash string `json:"hash"`
}

// Manifest is the versioned build descriptor written alongside corpora.
type Manifest struct {
	Version       int            `json:"version"`
	Seed          int64          `json:"seed"`
	PartitionSalt string         `json:"partitionSalt,omitempty"`
	CalibPercent  int            `json:"calibPercent"`
	SearchPercent int            `json:"searchPercent,omitempty"`
	Sources       []SourceInfo   `json:"sources"`
	DomainMix     map[string]int `json:"domainMix"`
	Stats         struct {
		TotalRecords  int `json:"totalRecords"`
		CalibRecords  int `json:"calibRecords"`
		SearchRecords int `json:"searchRecords,omitempty"`
		EvalRecords   int `json:"evalRecords"`
		TotalTokens   int `json:"totalTokens"`
	} `json:"stats"`
	// Outputs maps output basename to its hex sha256.
	Outputs map[string]string `json:"outputs"`
}

// WriteCorpus writes the calibration corpus, evaluation corpus, and manifest
// into dir atomically. It returns the manifest with output hashes filled in.
func WriteCorpus(ctx context.Context, dir string, res *BuildResult, cfg *Config) (*Manifest, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	calibHash, err := writeRecordsAtomic(ctx, filepath.Join(dir, "calibration.txt"), res.CalibRecords)
	if err != nil {
		return nil, err
	}
	evalHash, err := writeRecordsAtomic(ctx, filepath.Join(dir, "evaluation.txt"), res.EvalRecords)
	if err != nil {
		return nil, err
	}

	m := &Manifest{
		Version:       ManifestVersion,
		Seed:          cfg.Seed,
		PartitionSalt: cfg.PartitionSalt,
		CalibPercent:  cfg.CalibPercent,
		SearchPercent: cfg.SearchPercent,
		DomainMix:     res.Stats.DomainCounts,
		Outputs: map[string]string{
			"calibration.txt": calibHash,
			"evaluation.txt":  evalHash,
		},
	}
	if cfg.SearchPercent > 0 {
		searchHash, err := writeRecordsAtomic(ctx, filepath.Join(dir, "search.txt"), res.SearchRecords)
		if err != nil {
			return nil, err
		}
		m.Outputs["search.txt"] = searchHash
	}
	// Per-domain evaluation corpora: with more than one domain, emit
	// evaluation-<domain>.txt files so the pipeline can stratify KLD gates
	// per domain. Domain names are file-safe (letters/digits/./_/- only;
	// anything else is replaced).
	if distinctDomains(res) > 1 {
		if err := writeDomainEvals(ctx, dir, res.EvalRecords, m); err != nil {
			return nil, err
		}
	}
	m.Stats.TotalRecords = res.Stats.TotalRecords
	m.Stats.CalibRecords = res.Stats.CalibRecords
	m.Stats.SearchRecords = res.Stats.SearchRecords
	m.Stats.EvalRecords = res.Stats.EvalRecords
	m.Stats.TotalTokens = res.Stats.TotalTokens

	// Per-source info: count and chained hash, computed in source order from
	// the combined record list (per-source order is preserved by interleave).
	chains := map[string]hash.Hash{}
	counts := map[string]int{}
	allRecords := append(append([]Record{}, res.CalibRecords...), res.SearchRecords...)
	allRecords = append(allRecords, res.EvalRecords...)
	for _, r := range allRecords {
		h, ok := chains[r.Domain]
		if !ok {
			h = sha256.New()
			chains[r.Domain] = h
		}
		h.Write([]byte(r.Hash))
		h.Write([]byte{0})
		counts[r.Domain]++
	}
	for _, d := range cfg.Domains {
		dom := d.Source.Domain()
		if h, ok := chains[dom]; ok {
			m.Sources = append(m.Sources, SourceInfo{
				Domain: dom,
				Count:  counts[dom],
				Hash:   hex.EncodeToString(h.Sum(nil)),
			})
		}
	}

	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return nil, err
	}
	if err := writeFileAtomic(filepath.Join(dir, "manifest.json"), data); err != nil {
		return nil, err
	}
	return m, nil
}

func writeRecordsAtomic(ctx context.Context, path string, recs []Record) (string, error) {
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	w := bufio.NewWriter(io.MultiWriter(f, h))
	for i, r := range recs {
		if err := ctx.Err(); err != nil {
			f.Close()
			os.Remove(tmp)
			return "", err
		}
		if i > 0 {
			if _, err := w.WriteString(recordSep); err != nil {
				f.Close()
				os.Remove(tmp)
				return "", err
			}
		}
		if _, err := w.WriteString(r.Text); err != nil {
			f.Close()
			os.Remove(tmp)
			return "", err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		os.Remove(tmp)
		return "", err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return "", err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return "", err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func writeFileAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// Build collects records and writes corpus + manifest in one call.
func Build(ctx context.Context, dir string, cfg *Config) (*BuildResult, *Manifest, error) {
	res, err := collect(ctx, cfg)
	if err != nil {
		return nil, nil, err
	}
	m, err := WriteCorpus(ctx, dir, res, cfg)
	if err != nil {
		return nil, nil, err
	}
	return res, m, nil
}

// distinctDomains counts the domains present in the eval records.
func distinctDomains(res *BuildResult) int {
	seen := map[string]bool{}
	for _, r := range res.EvalRecords {
		seen[r.Domain] = true
	}
	return len(seen)
}

// domainFileSafe returns a file-safe form of a domain name.
func domainFileSafe(name string) string {
	out := make([]rune, 0, len(name))
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			out = append(out, r)
		case r == '.', r == '_', r == '-':
			out = append(out, r)
		default:
			out = append(out, '_')
		}
	}
	return string(out)
}

// writeDomainEvals emits evaluation-<domain>.txt files and registers them
// in the manifest outputs. The single-domain case is skipped by the caller.
func writeDomainEvals(ctx context.Context, dir string, evalRecords []Record, m *Manifest) error {
	var domains []string
	seen := map[string]bool{}
	for _, r := range evalRecords {
		if !seen[r.Domain] {
			seen[r.Domain] = true
			domains = append(domains, r.Domain)
		}
	}
	sort.Strings(domains)
	for _, dom := range domains {
		var recs []Record
		for _, r := range evalRecords {
			if r.Domain == dom {
				recs = append(recs, r)
			}
		}
		if len(recs) == 0 {
			continue
		}
		name := "evaluation-" + domainFileSafe(dom) + ".txt"
		hash, err := writeRecordsAtomic(ctx, filepath.Join(dir, name), recs)
		if err != nil {
			return err
		}
		m.Outputs[name] = hash
	}
	return nil
}
