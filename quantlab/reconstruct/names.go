package reconstruct

import (
	"strings"

	"quantlab/tensorbank"
)

func layerPrefix(name string) string {
	for _, pre := range []string{"blk.", "layers.", "layer."} {
		if i := strings.Index(name, pre); i >= 0 {
			rest := name[i+len(pre):]
			j := strings.IndexByte(rest, '.')
			if j >= 0 {
				return pre + rest[:j]
			}
		}
	}
	return ""
}

func localStem(name string) string {
	n := strings.ToLower(name)
	for _, pre := range []string{"blk.", "layers.", "layer."} {
		i := strings.Index(n, pre)
		if i < 0 {
			continue
		}
		rest := n[i+len(pre):]
		j := 0
		for j < len(rest) && rest[j] >= '0' && rest[j] <= '9' {
			j++
		}
		if j < len(rest) && rest[j] == '.' {
			n = rest[j+1:]
			break
		}
	}
	return strings.TrimSuffix(n, ".weight")
}

func isNormName(name string) bool {
	low := strings.ToLower(name)
	return strings.Contains(low, "norm") && strings.HasSuffix(low, ".weight")
}

func isEmbedding(name string) bool {
	n := strings.ToLower(name)
	return strings.Contains(n, "token_embd") || strings.Contains(n, "tok_embeddings") ||
		strings.Contains(n, "word_embd") || strings.Contains(n, "embed_tokens")
}

func isOutputHead(name string) bool {
	n := strings.ToLower(name)
	if strings.Contains(n, "attn") {
		return false
	}
	return strings.HasSuffix(n, "output.weight") || n == "output.weight" ||
		strings.Contains(n, "lm_head")
}

func isAttnOut(stem string) bool {
	if strings.Contains(stem, "qkv") {
		return false
	}
	return strings.HasPrefix(stem, "attn_output") || strings.HasPrefix(stem, "attn_o") ||
		strings.Contains(stem, "o_proj") || strings.Contains(stem, "wo") ||
		strings.HasPrefix(stem, "out_proj")
}

func isFFNDown(stem string) bool {
	return strings.Contains(stem, "ffn_down") || strings.Contains(stem, "down_proj") ||
		strings.HasSuffix(stem, ".w2")
}

func isFFNGate(stem string) bool {
	if strings.Contains(stem, "gate_inp") {
		return false
	}
	return strings.Contains(stem, "ffn_gate") || strings.Contains(stem, "gate_proj") ||
		strings.HasSuffix(stem, ".w1")
}

func isFFNUp(stem string) bool {
	return strings.Contains(stem, "ffn_up") || strings.Contains(stem, "up_proj") ||
		strings.HasSuffix(stem, ".w3")
}

func isReadResidual(name string) bool {
	stem := localStem(name)
	if isAttnOut(stem) || isFFNDown(stem) || isNormName(name) || isEmbedding(name) || isOutputHead(name) {
		return false
	}
	low := strings.ToLower(name)
	switch {
	case strings.Contains(low, "attn_q."), strings.HasSuffix(low, "attn_q.weight"),
		strings.Contains(low, "attn_k."), strings.HasSuffix(low, "attn_k.weight"),
		strings.Contains(low, "attn_v."), strings.HasSuffix(low, "attn_v.weight"),
		strings.Contains(low, "attn_qkv"), strings.Contains(low, "qkv_proj"),
		strings.Contains(low, "q_proj"), strings.Contains(low, "k_proj"),
		strings.Contains(low, "v_proj"):
		return true
	case isFFNGate(stem), isFFNUp(stem):
		return true
	}
	return false
}

func isWriteResidual(name string) bool {
	stem := localStem(name)
	return isAttnOut(stem) || isFFNDown(stem)
}

// ReadsResidual reports whether a model tensor (or its imatrix base name)
// reads the residual stream and should have its input-channel importance
// flattened after a residual Hadamard.
func ReadsResidual(name string) bool {
	name = strings.TrimSuffix(name, ".in_sum2")
	name = strings.TrimSuffix(name, ".counts")
	return isReadResidual(name)
}

func residualWidth(file *tensorbank.File) int {
	counts := map[uint64]int{}
	for _, t := range file.Tensors {
		if len(t.Shape) != 1 || !isFloatDType(t.DType) || !isNormName(t.Name) {
			continue
		}
		counts[t.Shape[0]]++
	}
	var best uint64
	bestN := 0
	for d, n := range counts {
		if n > bestN || (n == bestN && d > best) {
			best, bestN = d, n
		}
	}
	return int(best)
}

func ne0Of(t tensorbank.TensorInfo) uint64 {
	if len(t.Shape) == 0 {
		return t.Elements
	}
	return t.Shape[0]
}

func ne1Of(t tensorbank.TensorInfo) uint64 {
	if len(t.Shape) < 2 {
		return 1
	}
	return t.Shape[1]
}
