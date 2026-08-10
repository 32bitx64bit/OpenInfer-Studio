package gguf

import (
	"path/filepath"
	"strings"
)

// PoolingType is a llama.cpp --pooling value.
type PoolingType string

const (
	PoolingUnset PoolingType = ""
	PoolingNone  PoolingType = "none"
	PoolingMean  PoolingType = "mean"
	PoolingCLS   PoolingType = "cls"
	PoolingLast  PoolingType = "last"
	PoolingRank  PoolingType = "rank"
)

// Dedicated embedding / encoder architectures (LLM_ARCH_* in llama-arch.cpp).
var embeddingArchs = map[string]bool{
	"bert":            true,
	"modern-bert":     true,
	"nomic-bert":      true,
	"nomic-bert-moe":  true,
	"neo-bert":        true,
	"jina-bert-v2":    true,
	"jina-bert-v3":    true,
	"eurobert":        true,
	"gemma-embedding": true,
}

// Filename / alias tokens that strongly suggest an embedding model.
// Matched as substrings on a lowercased basename (no path).
var embeddingNameHints = []string{
	"embeddinggemma",
	"qwen3-embedding",
	"qwen2-embedding",
	"nomic-embed",
	"snowflake-arctic-embed",
	"bge-m3",
	"bge-large",
	"bge-base",
	"bge-small",
	"e5-mistral",
	"e5-large",
	"e5-base",
	"e5-small",
	"gte-large",
	"gte-base",
	"gte-small",
	"gte-qwen",
	"mxbai-embed",
	"jina-embeddings",
	"jina-embed",
}

// IsEmbeddingArch reports whether general.architecture is a dedicated
// embedding / BERT-family architecture.
func IsEmbeddingArch(arch string) bool {
	return embeddingArchs[strings.ToLower(strings.TrimSpace(arch))]
}

// ParsePoolingType maps GGUF pooling_type (int or string) to a --pooling value.
func ParsePoolingType(v any) PoolingType {
	switch t := v.(type) {
	case string:
		switch strings.ToLower(strings.TrimSpace(t)) {
		case "none", "0":
			return PoolingNone
		case "mean", "1":
			return PoolingMean
		case "cls", "2":
			return PoolingCLS
		case "last", "3":
			return PoolingLast
		case "rank", "4":
			return PoolingRank
		}
	case float64:
		return poolingFromInt(int(t))
	case int:
		return poolingFromInt(t)
	case int32:
		return poolingFromInt(int(t))
	case int64:
		return poolingFromInt(int(t))
	case uint32:
		return poolingFromInt(int(t))
	case uint64:
		return poolingFromInt(int(t))
	}
	if n, ok := toUint32(v); ok {
		return poolingFromInt(int(n))
	}
	return PoolingUnset
}

func poolingFromInt(n int) PoolingType {
	switch n {
	case 0:
		return PoolingNone
	case 1:
		return PoolingMean
	case 2:
		return PoolingCLS
	case 3:
		return PoolingLast
	case 4:
		return PoolingRank
	default:
		return PoolingUnset
	}
}

// PoolingTypeFromKV reads {arch}.pooling_type (or unprefixed pooling_type).
func PoolingTypeFromKV(arch string, raw map[string]any) PoolingType {
	if raw == nil {
		return PoolingUnset
	}
	arch = strings.TrimSpace(arch)
	keys := []string{}
	if arch != "" {
		keys = append(keys, arch+".pooling_type")
	}
	keys = append(keys, "pooling_type", "general.pooling_type")
	for _, k := range keys {
		if v, ok := raw[k]; ok {
			if p := ParsePoolingType(v); p != PoolingUnset {
				return p
			}
		}
	}
	for k, v := range raw {
		lk := strings.ToLower(k)
		if strings.HasSuffix(lk, ".pooling_type") || lk == "pooling_type" {
			if p := ParsePoolingType(v); p != PoolingUnset {
				return p
			}
		}
	}
	return PoolingUnset
}

// EmbeddingLengthOut reads {arch}.embedding_length_out when present.
func EmbeddingLengthOut(arch string, raw map[string]any) uint32 {
	if raw == nil {
		return 0
	}
	arch = strings.TrimSpace(arch)
	keys := []string{}
	if arch != "" {
		keys = append(keys, arch+".embedding_length_out")
	}
	keys = append(keys, "embedding_length_out")
	for _, k := range keys {
		if v, ok := raw[k]; ok {
			if n, ok := toUint32(v); ok {
				return n
			}
		}
	}
	return 0
}

func generalType(raw map[string]any) string {
	if raw == nil {
		return ""
	}
	if v, ok := raw["general.type"]; ok {
		if s, ok := v.(string); ok {
			return strings.ToLower(strings.TrimSpace(s))
		}
	}
	return ""
}

func looksLikeRerankerName(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, "rerank") || strings.Contains(s, "re-rank")
}

func looksLikeEmbeddingName(pathOrName string) bool {
	base := strings.ToLower(filepath.Base(pathOrName))
	base = strings.TrimSuffix(base, ".gguf")
	for _, h := range embeddingNameHints {
		if strings.Contains(base, h) {
			return true
		}
	}
	// Token-ish matches: -embed-, _embed_, embed-, -embedding-
	for _, tok := range []string{"-embed-", "_embed_", ".embed.", "-embedding-", "_embedding_", ".embedding."} {
		if strings.Contains(base, tok) {
			return true
		}
	}
	if strings.HasPrefix(base, "embed-") || strings.HasPrefix(base, "embedding-") ||
		strings.HasPrefix(base, "bge-") || strings.HasPrefix(base, "e5-") || strings.HasPrefix(base, "gte-") {
		return true
	}
	if strings.HasSuffix(base, "-embed") || strings.HasSuffix(base, "-embedding") ||
		strings.Contains(base, "-embed-") {
		return true
	}
	return false
}

// chatArchConflict is true when the architecture is a normal generative LLM
// family — filename "embed" hints alone must not reclassify these.
func chatArchConflict(arch string) bool {
	a := strings.ToLower(strings.TrimSpace(arch))
	if a == "" || IsEmbeddingArch(a) {
		return false
	}
	// Known generative families (non-exhaustive; anything not embedding-arch
	// is treated as potential chat when we only have a weak name hint).
	prefixes := []string{
		"llama", "qwen", "mistral", "gemma", "phi", "gpt", "falcon", "mpt",
		"bloom", "stablelm", "command", "deepseek", "yi", "internlm", "baichuan",
		"cohere", "granite", "olmo", "dbrx", "mixtral", "arctic",
	}
	for _, p := range prefixes {
		if a == p || strings.HasPrefix(a, p) {
			// gemma-embedding is already in embeddingArchs
			if strings.Contains(a, "embed") {
				return false
			}
			return true
		}
	}
	return false
}

// DetectEmbedding decides whether a GGUF is a dedicated embedding or reranker
// model. Order: general.type → pooling_type / embedding_length_out → arch →
// rerank name → conservative filename heuristics.
func DetectEmbedding(arch, name, path string, raw map[string]any) (isEmbedding, isReranker bool, pooling PoolingType, outDim uint32) {
	pooling = PoolingTypeFromKV(arch, raw)
	outDim = EmbeddingLengthOut(arch, raw)
	gt := generalType(raw)
	combinedName := strings.ToLower(name + " " + filepath.Base(path) + " " + arch)

	switch gt {
	case "embedding", "embed", "embeddings":
		isEmbedding = true
	case "reranker", "rerank":
		isEmbedding, isReranker = true, true
	}

	if pooling != PoolingUnset || outDim > 0 {
		isEmbedding = true
	}
	if IsEmbeddingArch(arch) {
		isEmbedding = true
	}
	if pooling == PoolingRank || looksLikeRerankerName(combinedName) {
		isEmbedding, isReranker = true, true
	}
	if !isEmbedding && looksLikeEmbeddingName(path) && !chatArchConflict(arch) {
		isEmbedding = true
	}
	if !isEmbedding && looksLikeEmbeddingName(name) && !chatArchConflict(arch) {
		isEmbedding = true
	}

	if isReranker && pooling == PoolingUnset {
		pooling = PoolingRank
	}
	return isEmbedding, isReranker, pooling, outDim
}

// ApplyEmbeddingFlags sets IsEmbedding / IsReranker / PoolingType /
// EmbeddingLengthOut on md. Safe to call after extract(). Skips speculative
// draft sidecars — those are never embedders.
func (md *Metadata) ApplyEmbeddingFlags(path string) {
	if md.SpeculativeDraft {
		md.IsEmbedding = false
		md.IsReranker = false
		md.PoolingType = PoolingUnset
		md.EmbeddingLengthOut = 0
		return
	}
	isEmb, isRR, pooling, outDim := DetectEmbedding(md.Architecture, md.Name, path, md.Raw)
	md.IsEmbedding = isEmb
	md.IsReranker = isRR
	md.PoolingType = pooling
	md.EmbeddingLengthOut = outDim
	if isEmb {
		// Embedding models are not multimodal chat targets.
		md.ClearMultimodal()
	}
}
