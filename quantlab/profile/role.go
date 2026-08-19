package profile

import (
	"strings"
)

// layerIndex parses a GGUF-style "blk.N." (or layers.N / layer.N) index.
// Tensors without a layer number return -1.
func layerIndex(name string) int {
	for _, prefix := range []string{"blk.", "layers.", "layer."} {
		i := strings.Index(name, prefix)
		if i < 0 {
			continue
		}
		rest := name[i+len(prefix):]
		d, n := 0, 0
		for n < len(rest) && rest[n] >= '0' && rest[n] <= '9' {
			d = d*10 + int(rest[n]-'0')
			n++
		}
		if n > 0 {
			return d
		}
	}
	return -1
}

// localName is the tensor identity after a blk.N. / layers.N. prefix.
func localName(name string) string {
	for _, prefix := range []string{"blk.", "layers.", "layer."} {
		i := strings.Index(name, prefix)
		if i < 0 {
			continue
		}
		rest := name[i+len(prefix):]
		n := 0
		for n < len(rest) && rest[n] >= '0' && rest[n] <= '9' {
			n++
		}
		if n < len(rest) && rest[n] == '.' {
			return rest[n+1:]
		}
	}
	return name
}

func localStem(name string) string {
	loc := strings.ToLower(localName(name))
	return strings.TrimSuffix(loc, ".weight")
}

func maxLayerIndex(names []string) int {
	max := -1
	for _, n := range names {
		if d := layerIndex(n); d > max {
			max = d
		}
	}
	return max
}

func isMoEExpert(name string) bool {
	n := strings.ToLower(name)
	if strings.Contains(n, "gate_inp") || strings.Contains(n, "router") {
		return false
	}
	return strings.Contains(n, "expert") || strings.Contains(n, "_exps") ||
		strings.Contains(n, ".exps.")
}

func hasToken(stem, tok string) bool {
	if stem == tok {
		return true
	}
	return strings.HasSuffix(stem, "."+tok) || strings.HasPrefix(stem, tok+".") ||
		strings.Contains(stem, "."+tok+".")
}

func isAttnV(stem string) bool {
	if strings.Contains(stem, "qkv") {
		return false
	}
	return strings.HasPrefix(stem, "attn_v") || hasToken(stem, "v_proj") ||
		hasToken(stem, "wv") || strings.HasSuffix(stem, ".v")
}

func isAttnOut(stem string) bool {
	if strings.Contains(stem, "qkv") {
		return false
	}
	return strings.HasPrefix(stem, "attn_output") || strings.HasPrefix(stem, "attn_o") ||
		hasToken(stem, "o_proj") || hasToken(stem, "wo") ||
		strings.HasPrefix(stem, "out_proj")
}

func isAttnQ(stem string) bool {
	if strings.Contains(stem, "qkv") {
		return false
	}
	return (strings.HasPrefix(stem, "attn_q") && !strings.HasPrefix(stem, "attn_qkv")) ||
		hasToken(stem, "q_proj") || hasToken(stem, "wq")
}

func isAttnK(stem string) bool {
	if strings.Contains(stem, "qkv") {
		return false
	}
	return strings.HasPrefix(stem, "attn_k") || hasToken(stem, "k_proj") || hasToken(stem, "wk")
}

func isAttnGate(stem string) bool {
	return strings.HasPrefix(stem, "attn_gate")
}

func isFFNDown(stem string) bool {
	return strings.Contains(stem, "ffn_down") || hasToken(stem, "down_proj") ||
		hasToken(stem, "w2")
}

func isFFNUp(stem string) bool {
	return strings.Contains(stem, "ffn_up") || hasToken(stem, "up_proj") ||
		hasToken(stem, "w3")
}

func isFFNGate(stem string) bool {
	if strings.Contains(stem, "gate_inp") {
		return false
	}
	return strings.Contains(stem, "ffn_gate") || hasToken(stem, "gate_proj") ||
		hasToken(stem, "w1")
}

func isOutputHead(name, stem string) bool {
	n := strings.ToLower(name)
	if strings.Contains(n, "attn") {
		return false
	}
	if stem == "output" || stem == "output.weight" || strings.HasSuffix(n, "output.weight") ||
		n == "output.weight" || n == "output" {
		return true
	}
	return strings.Contains(n, "lm_head")
}

func isEmbedding(name string) bool {
	n := strings.ToLower(name)
	return strings.Contains(n, "token_embd") || strings.Contains(n, "tok_embeddings") ||
		strings.Contains(n, "word_embd") || strings.Contains(n, "embed_tokens")
}

func isBlockAttnOrFFN(name string) bool {
	n := strings.ToLower(name)
	stem := localStem(name)
	if isEmbedding(name) || isOutputHead(name, stem) {
		return false
	}
	return strings.Contains(n, "attn") || strings.Contains(n, "ffn") || strings.Contains(n, "mlp") ||
		strings.Contains(n, "ssm_") || strings.Contains(n, "attn_qkv") ||
		hasToken(stem, "q_proj") || hasToken(stem, "k_proj") || hasToken(stem, "v_proj") ||
		hasToken(stem, "o_proj") || hasToken(stem, "gate_proj") || hasToken(stem, "up_proj") ||
		hasToken(stem, "down_proj")
}

// roleBase is an architecture-agnostic sensitivity multiplier from GGUF
// convert name classes (embed / attn / ffn / MoE). It never encodes a
// model family. Named keepers (embed, attn_v, attn_out) outrank dull FFN
// and MoE experts. Soft only: the solver can outvote them under a tight
// budget. HEURISTIC.
func roleBase(name string) float64 {
	n := strings.ToLower(name)
	stem := localStem(name)
	expert := isMoEExpert(name)

	switch {
	case isEmbedding(name), isOutputHead(name, stem):
		return 1.5
	case isAttnV(stem), isAttnOut(stem):
		return 1.35
	case isFFNDown(stem):
		if expert {
			return 0.9
		}
		return 1.05
	case isAttnQ(stem), isAttnK(stem), isAttnGate(stem):
		return 1.05
	case isFFNUp(stem), isFFNGate(stem):
		if expert {
			return 0.75
		}
		return 0.85
	case strings.Contains(n, "attn"):
		return 1.1
	case expert && (strings.Contains(n, "ffn") || strings.Contains(n, "mlp")):
		return 0.95
	case strings.Contains(n, "ffn"), strings.Contains(n, "mlp"):
		return 1.0
	default:
		return 1.05
	}
}

// roleFactor classifies the tensor by name and applies a soft first/last-block
// bump on attention and FFN weights. First two and last two transformer
// blocks are bumped; mid-depth is not. HEURISTIC.
func roleFactor(name string, maxLayer int) float64 {
	f := roleBase(name)
	if !isBlockAttnOrFFN(name) {
		return f
	}
	d := layerIndex(name)
	if d < 0 {
		return f
	}
	if d <= 1 || (maxLayer >= 0 && d >= maxLayer-1) {
		return f * 1.2
	}
	return f
}

// roleBPWScale further harvests dull FFN at low target bpw so named keepers
// (embed / attn_v / attn_out) can hold bits against element-weighted
// greedy lossDec. 1 at unset or Q4+ budgets. HEURISTIC.
func roleBPWScale(name string, bpw float64) float64 {
	if bpw <= 0 || bpw >= 4.5 {
		return 1
	}
	stem := localStem(name)
	expert := isMoEExpert(name)
	low := bpw <= 4.0
	switch {
	case isFFNUp(stem), isFFNGate(stem):
		if expert {
			if low {
				return 0.4
			}
			return 0.7
		}
		if low {
			return 0.4
		}
		return 0.75
	case isFFNDown(stem):
		if expert {
			if low {
				return 0.65
			}
			return 0.85
		}
		if low {
			return 0.85
		}
		return 0.95
	}
	return 1
}

// swigluHalf reports whether name is the SwiGLU gate or up projection.
func swigluHalf(name string) string {
	stem := localStem(name)
	if strings.Contains(stem, "gate_inp") {
		return ""
	}
	switch {
	case isFFNGate(stem):
		return "gate"
	case isFFNUp(stem):
		return "up"
	}
	return ""
}
