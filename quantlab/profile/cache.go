package profile

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"

	"quantlab/core"
)

// CacheVersion is the on-disk schema version. Loading any other version is a
// hard error; caches are never silently migrated.
const CacheVersion = 1

// CacheEntry is one measured candidate loss with full provenance.
type CacheEntry struct {
	TensorName string          `json:"tensorName"`
	Target     core.DType      `json:"target"`
	Loss       float64         `json:"loss"`
	Confidence float64         `json:"confidence,omitempty"`
	Prov       core.Provenance `json:"prov"`
}

// Cache is a strict, versioned store of measured candidate losses bound to
// one model identity (ModelID + source SHA256). It refuses to load data for
// any other model so stale measurements can never leak across runs.
type Cache struct {
	Version  int          `json:"version"`
	ModelID  string       `json:"modelID"`
	ModelSHA string       `json:"modelSHA"`
	Entries  []CacheEntry `json:"entries"`

	index map[string]CacheEntry
}

// NewCache returns an empty cache bound to a model identity.
func NewCache(modelID, modelSHA string) *Cache {
	return &Cache{
		Version: CacheVersion, ModelID: modelID, ModelSHA: modelSHA,
		index: make(map[string]CacheEntry),
	}
}

func cacheKey(tensor string, target core.DType) string {
	return tensor + "\x00" + string(target)
}

func (c *Cache) reindex() {
	c.index = make(map[string]CacheEntry, len(c.Entries))
	for _, e := range c.Entries {
		c.index[cacheKey(e.TensorName, e.Target)] = e
	}
}

// IdentityMatches reports whether the cache is bound to the given model
// identity. An empty expected SHA matches any recorded SHA only when the
// cache itself recorded none.
func (c *Cache) IdentityMatches(modelID, modelSHA string) bool {
	return c.ModelID == modelID && c.ModelSHA == modelSHA
}

func entryFrom(cl CandidateLoss) CacheEntry {
	conf := cl.Confidence
	if conf == 0 {
		conf = 1
	}
	return CacheEntry{
		TensorName: cl.TensorName, Target: cl.Target, Loss: cl.Loss,
		Confidence: conf, Prov: *cl.Prov,
	}
}

func entryConfidence(e CacheEntry) float64 {
	if e.Confidence <= 0 {
		return 1
	}
	return e.Confidence
}

// Put records a measured loss. Provenance must validate. Duplicate
// (tensor, target) keys are rejected; use PutReplace to overwrite.
func (c *Cache) Put(cl CandidateLoss) error {
	if cl.Evidence != EvidenceMeasured {
		return fmt.Errorf("cache: only measured evidence may be stored, got %q", cl.Evidence)
	}
	if err := cl.Validate(); err != nil {
		return err
	}
	if c.index == nil {
		c.reindex()
	}
	e := entryFrom(cl)
	k := cacheKey(e.TensorName, e.Target)
	if _, dup := c.index[k]; dup {
		return fmt.Errorf("cache: duplicate entry %q/%q", e.TensorName, e.Target)
	}
	c.index[k] = e
	c.Entries = append(c.Entries, e)
	return nil
}

// PutReplace records a measured loss, overwriting an existing (tensor, target)
// entry. Last write wins so search ingest is deterministic when history is
// walked in order.
func (c *Cache) PutReplace(cl CandidateLoss) error {
	if cl.Evidence != EvidenceMeasured {
		return fmt.Errorf("cache: only measured evidence may be stored, got %q", cl.Evidence)
	}
	if err := cl.Validate(); err != nil {
		return err
	}
	if c.index == nil {
		c.reindex()
	}
	e := entryFrom(cl)
	k := cacheKey(e.TensorName, e.Target)
	if _, dup := c.index[k]; dup {
		for i := range c.Entries {
			if cacheKey(c.Entries[i].TensorName, c.Entries[i].Target) == k {
				c.Entries[i] = e
				break
			}
		}
	} else {
		c.Entries = append(c.Entries, e)
	}
	c.index[k] = e
	return nil
}

// Get returns the measured loss for (tensor, target), if present.
func (c *Cache) Get(tensor string, target core.DType) (CandidateLoss, bool) {
	if c == nil || c.index == nil {
		return CandidateLoss{}, false
	}
	e, ok := c.index[cacheKey(tensor, target)]
	if !ok {
		return CandidateLoss{}, false
	}
	prov := e.Prov
	return CandidateLoss{
		TensorName: e.TensorName, Target: e.Target, Loss: e.Loss,
		Evidence: EvidenceMeasured, Confidence: entryConfidence(e), Prov: &prov,
	}, true
}

// LoadCache strictly decodes a cache from r and binds it to the expected
// model identity. Version mismatch, identity mismatch, malformed entries,
// and duplicate keys are all hard errors.
func LoadCache(r io.Reader, modelID, modelSHA string) (*Cache, error) {
	var c Cache
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("cache: decode: %w", err)
	}
	if c.Version != CacheVersion {
		return nil, fmt.Errorf("cache: unsupported version %d (want %d)", c.Version, CacheVersion)
	}
	if !c.IdentityMatches(modelID, modelSHA) {
		return nil, fmt.Errorf("cache: identity mismatch: cache is for %q/%q, want %q/%q",
			c.ModelID, c.ModelSHA, modelID, modelSHA)
	}
	seen := make(map[string]struct{}, len(c.Entries))
	for _, e := range c.Entries {
		cl := CandidateLoss{
			TensorName: e.TensorName, Target: e.Target, Loss: e.Loss,
			Evidence: EvidenceMeasured, Confidence: entryConfidence(e), Prov: &e.Prov,
		}
		if err := cl.Validate(); err != nil {
			return nil, fmt.Errorf("cache: invalid entry: %w", err)
		}
		k := cacheKey(e.TensorName, e.Target)
		if _, dup := seen[k]; dup {
			return nil, fmt.Errorf("cache: duplicate entry %q/%q", e.TensorName, e.Target)
		}
		seen[k] = struct{}{}
	}
	c.reindex()
	return &c, nil
}

// Save writes the cache deterministically: entries sorted by (tensor, target).
func (c *Cache) Save(w io.Writer) error {
	out := *c
	out.Entries = append([]CacheEntry(nil), c.Entries...)
	sort.Slice(out.Entries, func(i, j int) bool {
		if out.Entries[i].TensorName != out.Entries[j].TensorName {
			return out.Entries[i].TensorName < out.Entries[j].TensorName
		}
		return out.Entries[i].Target < out.Entries[j].Target
	})
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(&out)
}
