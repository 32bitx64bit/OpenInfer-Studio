package anchor

import (
	"strings"

	"quantlab/core"
)

// tensorRole is the GGUF-derived quantization role of one tensor. Names are
// classified the way convert's workKind/layout encode them: RMS/conv1d/A_log
// stay F32, linear-attn projections are not softmax-attention, MoE routers
// get a hard floor, and unknown names fail closed (never silently quantized).
type tensorRole byte

const (
	roleUnknown tensorRole = iota
	roleNorm
	roleConv1d
	roleSSMState // ssm_a (A_log) and ssm_dt
	roleLinearAttn
	roleAttention
	roleRouter
	roleEmbed
	roleOutput
	roleFFN
)

func keepFloat(r tensorRole) bool {
	switch r {
	case roleNorm, roleConv1d, roleSSMState, roleUnknown:
		return true
	}
	return false
}

func layerIndex(name string) int {
	i := strings.Index(name, "blk.")
	if i < 0 {
		return -1
	}
	rest := name[i+4:]
	n, digits := 0, 0
	for digits < len(rest) && rest[digits] >= '0' && rest[digits] <= '9' {
		n = n*10 + int(rest[digits]-'0')
		digits++
	}
	if digits == 0 {
		return -1
	}
	return n
}

// localName is the tensor identity after a "blk.N." prefix, or the full name.
func localName(name string) string {
	if i := strings.Index(name, "blk."); i >= 0 {
		rest := name[i+4:]
		if j := strings.IndexByte(rest, '.'); j >= 0 {
			return rest[j+1:]
		}
	}
	return name
}

func linearAttnLayers(bank *core.TensorBank) map[int]struct{} {
	out := map[int]struct{}{}
	for _, t := range bank.Tensors {
		loc := strings.ToLower(localName(t.Name))
		if strings.HasPrefix(loc, "ssm_") || strings.Contains(loc, ".ssm_") {
			if layer := layerIndex(t.Name); layer >= 0 {
				out[layer] = struct{}{}
			}
		}
	}
	return out
}

func classify(name string, linear map[int]struct{}) tensorRole {
	n := strings.ToLower(name)
	loc := strings.ToLower(localName(name))

	switch {
	case strings.Contains(n, "token_embd"), strings.Contains(n, "tok_embeddings"),
		strings.Contains(n, "word_embd"):
		return roleEmbed
	case loc == "output.weight", n == "output.weight", n == "output":
		return roleOutput
	}

	// Convert stores these as F32 (RMS, conv1d, A_log, 1D dt). Match before
	// any attn_* / ssm_out classification.
	if strings.Contains(loc, "ssm_conv1d") {
		return roleConv1d
	}
	if loc == "ssm_a" || strings.HasPrefix(loc, "ssm_a.") {
		return roleSSMState
	}
	if loc == "ssm_dt" || strings.HasPrefix(loc, "ssm_dt.") || strings.HasPrefix(loc, "ssm_dt_") {
		return roleSSMState
	}
	if strings.Contains(loc, "_norm") || strings.HasSuffix(loc, "norm.weight") ||
		loc == "output_norm.weight" || strings.Contains(loc, "ssm_norm") {
		return roleNorm
	}

	if strings.Contains(loc, "ffn_gate_inp") {
		return roleRouter
	}

	layer := layerIndex(name)
	_, lin := linear[layer]

	if strings.HasPrefix(loc, "ssm_") {
		return roleLinearAttn
	}
	if lin && (strings.HasPrefix(loc, "attn_qkv") || strings.HasPrefix(loc, "attn_gate")) {
		return roleLinearAttn
	}

	switch {
	case strings.HasPrefix(loc, "attn_qkv"), strings.HasPrefix(loc, "attn_gate"):
		return roleAttention
	case strings.HasPrefix(loc, "attn_q."), loc == "attn_q.weight":
		return roleAttention
	case strings.HasPrefix(loc, "attn_k."), loc == "attn_k.weight":
		return roleAttention
	case strings.HasPrefix(loc, "attn_v."), loc == "attn_v.weight":
		return roleAttention
	case strings.HasPrefix(loc, "attn_output"), loc == "attn_output.weight":
		return roleAttention
	}

	if strings.HasPrefix(loc, "ffn_") || strings.Contains(loc, ".ffn_") {
		return roleFFN
	}
	return roleUnknown
}

func localStem(name string) string {
	return strings.TrimSuffix(strings.ToLower(localName(name)), ".weight")
}

func hasTok(stem, tok string) bool {
	if stem == tok {
		return true
	}
	return strings.HasSuffix(stem, "."+tok) || strings.Contains(stem, "."+tok+".")
}

// isAttnV reports attn_v / v_proj / wv names that receive ValuePrior.
// Fused qkv is not V: convert stores one tensor, and a V prior would pin
// the whole Q/K/V stack.
func isAttnV(name string) bool {
	stem := localStem(name)
	if strings.Contains(stem, "qkv") {
		return false
	}
	return strings.HasPrefix(stem, "attn_v") || hasTok(stem, "v_proj") || hasTok(stem, "wv")
}

// isFFNDown reports ffn_down / down_proj / w2 names that receive DownPrior.
func isFFNDown(name string) bool {
	stem := localStem(name)
	return strings.Contains(stem, "ffn_down") || hasTok(stem, "down_proj") || hasTok(stem, "w2")
}

// isMoEExpert reports expert-stacked FFN names (ffn_*_exps, experts.N).
// Routers (gate_inp) are not experts.
func isMoEExpert(name string) bool {
	n := strings.ToLower(name)
	if strings.Contains(n, "gate_inp") || strings.Contains(n, "router") {
		return false
	}
	return strings.Contains(n, "expert") || strings.Contains(n, "_exps") ||
		strings.Contains(n, ".exps.")
}
