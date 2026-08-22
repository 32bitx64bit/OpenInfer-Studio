package convert

import (
	"regexp"
	"strconv"
	"strings"
)

func overlay(base map[string]string, extra map[string]string) map[string]string {
	out := make(map[string]string, len(base)+len(extra))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range extra {
		out[k] = v
	}
	return out
}

var llamaLayers = map[string]string{
	"self_attn.q_proj":         "attn_q.weight",
	"self_attn.k_proj":         "attn_k.weight",
	"self_attn.v_proj":         "attn_v.weight",
	"self_attn.o_proj":         "attn_output.weight",
	"mlp.gate_proj":            "ffn_gate.weight",
	"mlp.up_proj":              "ffn_up.weight",
	"mlp.down_proj":            "ffn_down.weight",
	"input_layernorm":          "attn_norm.weight",
	"post_attention_layernorm": "ffn_norm.weight",
}

var internlmAliases = map[string]string{
	"attention.wq":    "attn_q.weight",
	"attention.wk":    "attn_k.weight",
	"attention.wv":    "attn_v.weight",
	"attention.wo":    "attn_output.weight",
	"attention.wqkv":  "attn_qkv.weight",
	"feed_forward.w1": "ffn_gate.weight",
	"feed_forward.w3": "ffn_up.weight",
	"feed_forward.w2": "ffn_down.weight",
	"attention_norm":  "attn_norm.weight",
	"ffn_norm":        "ffn_norm.weight",
}

var linearAttnLayers = map[string]string{
	"post_attention_layernorm": "post_attention_norm.weight",
	"linear_attn.in_proj_qkv":  "attn_qkv.weight",
	"linear_attn.in_proj_z":    "attn_gate.weight",
	"linear_attn.in_proj_a":    "ssm_alpha.weight",
	"linear_attn.in_proj_b":    "ssm_beta.weight",
	"linear_attn.out_proj":     "ssm_out.weight",
	"linear_attn.conv1d":       "ssm_conv1d.weight",
	"linear_attn.A_log":        "ssm_a",
	"linear_attn.dt_bias":      "ssm_dt.bias",
	"linear_attn.norm":         "ssm_norm.weight",
}

var moeLayers = map[string]string{
	"mlp.gate":                    "ffn_gate_inp.weight",
	"block_sparse_moe.gate":       "ffn_gate_inp.weight",
	"mlp.shared_expert.gate_proj": "ffn_gate_shexp.weight",
	"mlp.shared_expert.up_proj":   "ffn_up_shexp.weight",
	"mlp.shared_expert.down_proj": "ffn_down_shexp.weight",
	"mlp.shared_expert_gate":      "ffn_gate_inp_shexp.weight",
}

var gemmaFFLayers = map[string]string{
	"post_attention_layernorm":   "post_attention_norm.weight",
	"pre_feedforward_layernorm":  "ffn_norm.weight",
	"post_feedforward_layernorm": "post_ffw_norm.weight",
}

func layersFor(feat layout) map[string]string {
	m := overlay(llamaLayers, internlmAliases)
	if feat.GemmaFF {
		m = overlay(m, gemmaFFLayers)
	} else if feat.LinearAttn {
		m = overlay(m, linearAttnLayers)
	}
	if feat.QKNorm {
		m["self_attn.q_norm"] = "attn_q_norm.weight"
		m["self_attn.k_norm"] = "attn_k_norm.weight"
	}
	if feat.MoE {
		m = overlay(m, moeLayers)
	}
	if feat.PackedQKV {
		m["self_attn.qkv_proj"] = "attn_qkv.weight"
		m["mlp.gate_up_proj"] = "ffn_up.weight"
	}
	if feat.AttnGate {
		m["self_attn.gate_proj"] = "attn_gate.weight"
	}
	return m
}

func familyFor(arch string, feat layout) Family {
	f := Family{
		ID:          arch,
		GGUFArch:    arch,
		DefaultName: arch,
		Layers:      layersFor(feat),
		TieOutput:   true,
		LinearAttn:  feat.LinearAttn,
		MoE:         feat.MoE,
		KeepQKNorm:  feat.QKNorm,
		RopePartial: feat.LinearAttn,
		SkipSSMNorm: feat.LinearAttn,
	}
	if feat.GemmaFF || feat.LinearAttn || strings.HasPrefix(arch, "gemma") {
		f.RMSPlus = rmsPlusAll
	}
	return f
}

var (
	layerRe          = regexp.MustCompile(`(?:^|\.)layers\.(\d+)\.(.+)$`)
	expertRe         = regexp.MustCompile(`^(?:mlp|feed_forward|block_sparse_moe)\.experts\.(\d+)\.(.+)$`)
	expertKindToGGUF = map[string]string{
		"gate_proj": "ffn_gate_exps.weight",
		"up_proj":   "ffn_up_exps.weight",
		"down_proj": "ffn_down_exps.weight",
		"w1":        "ffn_gate_exps.weight",
		"w3":        "ffn_up_exps.weight",
		"w2":        "ffn_down_exps.weight",
	}
)

type mappedTensor struct {
	GGUF    string
	Skip    bool
	Vision  bool
	Kind    workKind
	Reorder string
	Expert  int // -1 = not an expert shard
	NHeads  int
}

func (f Family) MapName(hfName string) mappedTensor {
	if isVisionTensor(hfName) {
		return mappedTensor{Skip: true, Vision: true, Expert: -1}
	}
	n := strings.ReplaceAll(hfName, "language_model.", "")
	nl := strings.ToLower(n)
	if strings.Contains(nl, "mtp.") || strings.HasPrefix(nl, "mtp.") ||
		strings.Contains(nl, ".mtp") {
		return mappedTensor{Skip: true, Expert: -1}
	}
	if strings.Contains(n, "rotary_emb") || strings.Contains(n, "inv_freq") ||
		strings.Contains(n, "cos_cached") || strings.Contains(n, "sin_cached") {
		return mappedTensor{Skip: true, Expert: -1}
	}
	if strings.HasSuffix(n, ".bias") && !strings.HasSuffix(n, "dt_bias") {
		return mappedTensor{Skip: true, Expert: -1}
	}

	stem := n
	if strings.HasSuffix(stem, ".weight") {
		stem = strings.TrimSuffix(stem, ".weight")
	}

	switch {
	case strings.HasSuffix(stem, "embed_tokens") || stem == "model.embed_tokens" || stem == "embed_tokens":
		return mappedTensor{GGUF: "token_embd.weight", Kind: kindCopy, Expert: -1}
	case stem == "lm_head" || strings.HasSuffix(stem, ".lm_head"):
		return mappedTensor{GGUF: "output.weight", Kind: kindCopy, Expert: -1}
	case stem == "model.norm" || stem == "norm" || strings.HasSuffix(stem, ".model.norm"):
		return mappedTensor{GGUF: "output_norm.weight", Kind: f.normKind("output_norm.weight"), Expert: -1}
	}

	m := layerRe.FindStringSubmatch(stem)
	if m == nil {
		return mappedTensor{Expert: -1} // unknown
	}
	bid, rest := m[1], m[2]

	if f.MoE {
		if em := expertRe.FindStringSubmatch(rest); em != nil {
			idx, _ := strconv.Atoi(em[1])
			kindName := strings.TrimSuffix(em[2], ".weight")
			gg, ok := expertKindToGGUF[kindName]
			if !ok {
				return mappedTensor{Expert: -1}
			}
			return mappedTensor{
				GGUF:   "blk." + bid + "." + gg,
				Kind:   kindStack,
				Expert: idx,
			}
		}
	}

	gg, ok := f.Layers[rest]
	if !ok {
		return mappedTensor{Expert: -1}
	}
	ggufName := "blk." + bid + "." + gg
	out := mappedTensor{GGUF: ggufName, Expert: -1, Kind: kindCopy}
	out.Kind = f.tensorKind(ggufName, rest)
	out.Reorder = f.reorderOf(rest)
	return out
}

func (f Family) tensorKind(ggufName, rest string) workKind {
	if rest == "linear_attn.A_log" {
		return kindNegExp
	}
	if rest == "linear_attn.conv1d" {
		return kindConv1d
	}
	if f.UnpermuteQK {
		switch {
		case strings.Contains(ggufName, "attn_q.weight"):
			return kindUnpermuteQ
		case strings.Contains(ggufName, "attn_k.weight"):
			return kindUnpermuteK
		}
	}
	return f.normKind(ggufName)
}

func (f Family) normKind(ggufName string) workKind {
	isNorm := strings.Contains(ggufName, "_norm") || strings.HasSuffix(ggufName, "norm.weight") ||
		ggufName == "output_norm.weight" || strings.Contains(ggufName, "ssm_norm")
	if !isNorm {
		return kindCopy
	}
	if f.SkipSSMNorm && strings.Contains(ggufName, "ssm_norm") {
		return kindRMS
	}
	switch f.RMSPlus {
	case rmsPlusAll:
		return kindRMSPlus
	case rmsPlusLayers:
		if ggufName == "output_norm.weight" {
			return kindRMS
		}
		return kindRMSPlus
	default:
		if strings.Contains(ggufName, "_norm") || ggufName == "output_norm.weight" {
			return kindRMS
		}
		return kindCopy
	}
}

func (f Family) reorderOf(rest string) string {
	if !f.LinearAttn || !strings.HasPrefix(rest, "linear_attn.") {
		return ""
	}
	switch rest {
	case "linear_attn.in_proj_qkv":
		return "v_qkv"
	case "linear_attn.in_proj_z":
		return "v_rows"
	case "linear_attn.in_proj_a", "linear_attn.in_proj_b":
		return "v_rows1"
	case "linear_attn.A_log", "linear_attn.dt_bias":
		return "v_1d"
	case "linear_attn.conv1d":
		return "v_conv"
	case "linear_attn.out_proj":
		return "v_cols"
	}
	return ""
}

func isVisionTensor(name string) bool {
	n := strings.ToLower(name)
	return strings.Contains(n, "vision_tower") ||
		strings.Contains(n, "vision_adapter") ||
		strings.Contains(n, "vision_projection") ||
		strings.Contains(n, "vision_model") ||
		strings.Contains(n, "visual.") ||
		strings.Contains(n, "model.visual") ||
		strings.Contains(n, "mm_projector") ||
		strings.Contains(n, "multi_modal") ||
		strings.Contains(n, "audio_tower") ||
		strings.Contains(n, "audio_model")
}
