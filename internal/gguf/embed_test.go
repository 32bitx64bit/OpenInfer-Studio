package gguf

import "testing"

func TestDetectEmbeddingArch(t *testing.T) {
	ok, rr, pooling, _ := DetectEmbedding("nomic-bert", "Nomic", "nomic-embed-text-v1.5.Q4_K_M.gguf", nil)
	if !ok || rr {
		t.Fatalf("nomic-bert: emb=%v rerank=%v", ok, rr)
	}
	_ = pooling
	ok, _, _, _ = DetectEmbedding("llama", "Chat", "meta-llama-3-8b.Q4_K_M.gguf", nil)
	if ok {
		t.Fatal("plain llama must not be embedding")
	}
}

func TestDetectEmbeddingPoolingType(t *testing.T) {
	raw := map[string]any{"qwen3.pooling_type": uint32(3)} // LAST
	ok, _, pooling, _ := DetectEmbedding("qwen3", "Qwen3-Embedding", "Qwen3-Embedding-0.6B-Q8_0.gguf", raw)
	if !ok {
		t.Fatal("pooling_type should mark embedder")
	}
	if pooling != PoolingLast {
		t.Fatalf("pooling = %q, want last", pooling)
	}
}

func TestDetectEmbeddingGeneralType(t *testing.T) {
	raw := map[string]any{"general.type": "embedding"}
	ok, _, _, _ := DetectEmbedding("llama", "x", "x.gguf", raw)
	if !ok {
		t.Fatal("general.type=embedding")
	}
}

func TestDetectEmbeddingLengthOut(t *testing.T) {
	raw := map[string]any{"bert.embedding_length_out": uint32(768)}
	ok, _, _, out := DetectEmbedding("bert", "x", "x.gguf", raw)
	if !ok || out != 768 {
		t.Fatalf("emb=%v out=%d", ok, out)
	}
}

func TestDetectReranker(t *testing.T) {
	ok, rr, pooling, _ := DetectEmbedding("bert", "bge-reranker-v2-m3", "bge-reranker-v2-m3.Q4_K_M.gguf", nil)
	if !ok || !rr {
		t.Fatalf("reranker: emb=%v rr=%v", ok, rr)
	}
	if pooling != PoolingRank {
		t.Fatalf("pooling = %q, want rank", pooling)
	}
	raw := map[string]any{"bert.pooling_type": uint32(4)}
	ok, rr, pooling, _ = DetectEmbedding("bert", "model", "model.gguf", raw)
	if !ok || !rr || pooling != PoolingRank {
		t.Fatalf("rank pooling: emb=%v rr=%v pool=%q", ok, rr, pooling)
	}
}

func TestDetectEmbeddingNameHintDoesNotStealChat(t *testing.T) {
	// Filename contains "embed" but architecture is a chat family with no KV signal.
	ok, _, _, _ := DetectEmbedding("llama", "something", "my-embed-helper-notes.gguf", nil)
	if ok {
		t.Fatal("chat arch + weak name must not classify as embedding")
	}
	ok, _, _, _ = DetectEmbedding("", "nomic-embed-text", "nomic-embed-text-v1.Q4_K_M.gguf", nil)
	if !ok {
		t.Fatal("strong nomic-embed name with empty arch should classify")
	}
}

func TestDetectEmbeddingIgnoresHiddenSizeAlone(t *testing.T) {
	// embedding_length is present on every LLM — must not trigger.
	raw := map[string]any{"llama.embedding_length": uint32(4096)}
	ok, _, _, _ := DetectEmbedding("llama", "Llama-3", "Llama-3-8B.Q4_K_M.gguf", raw)
	if ok {
		t.Fatal("embedding_length alone must not mark embedder")
	}
}

func TestApplyEmbeddingFlagsSkipsDraft(t *testing.T) {
	md := &Metadata{
		Architecture:     "eagle3",
		SpeculativeDraft: true,
		Raw:              map[string]any{"general.type": "embedding"},
	}
	md.ApplyEmbeddingFlags("/models/eagle3-draft.gguf")
	if md.IsEmbedding {
		t.Fatal("speculative draft must not become embedding")
	}
}

func TestParsePoolingType(t *testing.T) {
	cases := map[any]PoolingType{
		uint32(0): "none", uint32(1): "mean", uint32(2): "cls", uint32(3): "last", uint32(4): "rank",
		"CLS": "cls", "mean": "mean", "LAST": "last",
	}
	for in, want := range cases {
		if got := ParsePoolingType(in); got != want {
			t.Errorf("ParsePoolingType(%v) = %q, want %q", in, got, want)
		}
	}
}
