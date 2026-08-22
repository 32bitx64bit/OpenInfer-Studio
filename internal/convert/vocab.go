package convert

import (
	"fmt"
	"strings"
)

func isVocabWeight(ggufName string) bool {
	return ggufName == "token_embd.weight" || ggufName == "output.weight"
}

func vocabRows(t TensorRef) int {
	n := strings.ReplaceAll(t.Name, "language_model.", "")
	stem := strings.TrimSuffix(n, ".weight")
	switch {
	case strings.HasSuffix(stem, "embed_tokens") || stem == "embed_tokens" || stem == "model.embed_tokens":
	case stem == "lm_head" || strings.HasSuffix(stem, ".lm_head"):
	default:
		return 0
	}
	if len(t.Shape) == 0 {
		return 0
	}
	return int(t.Shape[0])
}

// alignVocab makes tokenizer length, embedding rows, and lm_head rows agree.
// Hugging Face configs often under-report vocab_size while the embedding
// matrix is padded (Qwen and others). llama.cpp checks token_embd against
// n_vocab from the tokenizer / vocab_size KV — they must match.
func alignVocab(tok *ggmlTokenizer, tensors []TensorRef, cfgVocab int) int {
	n := cfgVocab
	if tok != nil && len(tok.Tokens) > n {
		n = len(tok.Tokens)
	}
	for _, t := range tensors {
		if v := vocabRows(t); v > n {
			n = v
		}
	}
	if tok != nil {
		tok.padTo(n)
	}
	return n
}

func (t *ggmlTokenizer) padTo(n int) {
	if t == nil || n <= len(t.Tokens) {
		return
	}
	for len(t.Tokens) < n {
		i := len(t.Tokens)
		t.Tokens = append(t.Tokens, fmt.Sprintf("[PAD%d]", i))
		t.TokenType = append(t.TokenType, tokenTypeUnused)
	}
}

func padVocabLeading(hf []int64, vocab int) []int64 {
	out := append([]int64(nil), hf...)
	if vocab <= 0 || len(out) == 0 || int(out[0]) >= vocab {
		return out
	}
	out[0] = int64(vocab)
	return out
}

// padLeadingDim grows a row-major tensor along dim 0 with zeros.
func padLeadingDim(raw []byte, src, dst []int64, elem int) ([]byte, error) {
	if elem <= 0 {
		return nil, fmt.Errorf("invalid element size")
	}
	if len(src) == 0 || len(dst) == 0 || src[0] == dst[0] {
		return raw, nil
	}
	if src[0] > dst[0] {
		return nil, fmt.Errorf("cannot shrink leading dim %d → %d", src[0], dst[0])
	}
	row := 1
	for i := 1; i < len(src); i++ {
		row *= int(src[i])
	}
	if row <= 0 {
		return raw, nil
	}
	rowBytes := row * elem
	wantSrc := int(src[0]) * rowBytes
	if len(raw) < wantSrc {
		return nil, fmt.Errorf("short tensor: have %d bytes, want %d", len(raw), wantSrc)
	}
	out := make([]byte, int(dst[0])*rowBytes)
	copy(out, raw[:wantSrc])
	return out, nil
}
