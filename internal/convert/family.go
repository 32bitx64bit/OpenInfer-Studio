package convert

import (
	"strings"
)

type rmsPlusMode int

const (
	rmsPlusNone   rmsPlusMode = iota
	rmsPlusLayers             // every *norm except output_norm
	rmsPlusAll                // including output_norm
)

// Family is the GGUF graph the generic converter emits for one inferred
// llama.cpp architecture.
type Family struct {
	ID, GGUFArch string
	TokenizerPre string
	DefaultName  string
	DefaultRope  float64

	UnpermuteQK bool
	KeepQKNorm  bool
	RMSPlus     rmsPlusMode
	SkipSSMNorm bool // linear_attn.norm stays as stored (Qwen3.5)
	TieOutput   bool
	LinearAttn  bool
	MoE         bool
	RopePartial bool // rope.dimension_count = head_dim * partial_rotary_factor
	Layers      map[string]string
}

type hyper struct {
	nHead, nKV, nEmbd, nFF, nLayer, nCtx, headDim, vocab int
	rms, rope, partialRotary                             float64
	mrope                                                []int32
	linConv, linKeyDim, linValDim                        int
	linKeyHeads, linValHeads, fullAttnEvery              int
	nExpert, nExpertUsed, nExpertFF, nSharedFF           int
	qkScale                                              float32
	softcap, logitScale                                  float32
	swa                                                  uint32
}

func parseHyper(cfg map[string]any, f Family) hyper {
	h := hyper{
		nHead:   cfgInt(cfg, "num_attention_heads", "n_head", "num_heads"),
		nKV:     cfgInt(cfg, "num_key_value_heads", "n_kv_heads"),
		nEmbd:   cfgInt(cfg, "hidden_size", "n_embd", "d_model"),
		nFF:     cfgInt(cfg, "intermediate_size", "n_inner", "ffn_dim"),
		nLayer:  cfgInt(cfg, "num_hidden_layers", "n_layer", "num_layers"),
		nCtx:    cfgInt(cfg, "max_position_embeddings", "n_ctx", "max_length", "max_sequence_length"),
		headDim: cfgInt(cfg, "head_dim"),
		vocab:   cfgInt(cfg, "vocab_size"),
		rms:     cfgFloat(cfg, "rms_norm_eps", "layer_norm_eps"),
	}
	if h.nKV <= 0 {
		h.nKV = h.nHead
	}
	if h.headDim <= 0 && h.nHead > 0 && h.nEmbd > 0 {
		h.headDim = h.nEmbd / h.nHead
	}
	if h.rms == 0 {
		h.rms = 1e-5
	}
	h.rope = ropeTheta(cfg, f.DefaultRope)
	h.partialRotary = 1
	if rp := cfgMap(cfg, "rope_parameters"); rp != nil {
		if p := cfgFloat(rp, "partial_rotary_factor"); p > 0 {
			h.partialRotary = p
		}
		if sec := cfgIntSlice(rp["mrope_section"]); len(sec) > 0 {
			h.mrope = sec
		}
	}
	if p := cfgFloat(cfg, "partial_rotary_factor"); p > 0 {
		h.partialRotary = p
	}
	if f.RopePartial && h.partialRotary == 1 {
		h.partialRotary = 0.25
	}
	if f.LinearAttn {
		h.linConv = cfgInt(cfg, "linear_conv_kernel_dim")
		h.linKeyDim = cfgInt(cfg, "linear_key_head_dim")
		h.linValDim = cfgInt(cfg, "linear_value_head_dim")
		h.linKeyHeads = cfgInt(cfg, "linear_num_key_heads")
		h.linValHeads = cfgInt(cfg, "linear_num_value_heads")
		h.fullAttnEvery = cfgInt(cfg, "full_attention_interval")
		if h.fullAttnEvery <= 0 {
			h.fullAttnEvery = 4
		}
		if h.linConv <= 0 {
			h.linConv = 4
		}
		if len(h.mrope) == 0 {
			h.mrope = []int32{11, 11, 10, 0}
		} else if len(h.mrope) == 3 {
			h.mrope = append(h.mrope, 0)
		}
	}
	if f.MoE {
		h.nExpert = cfgInt(cfg, "num_experts", "num_local_experts", "n_routed_experts")
		h.nExpertUsed = cfgInt(cfg, "num_experts_per_tok", "num_experts_per_token", "moe_topk")
		h.nExpertFF = cfgInt(cfg, "moe_intermediate_size", "expert_intermediate_size")
		if h.nExpertFF <= 0 {
			h.nExpertFF = h.nFF
		}
		h.nSharedFF = cfgInt(cfg, "shared_expert_intermediate_size")
		if h.nSharedFF <= 0 {
			h.nSharedFF = h.nFF
		}
	}
	h.qkScale = float32(cfgFloat(cfg, "qk_scale_factor"))
	h.softcap = float32(cfgFloat(cfg, "final_logit_softcapping", "final_logit_softcapping_scale"))
	h.logitScale = float32(cfgFloat(cfg, "output_multiplier", "logits_scaling"))
	if h.logitScale == 0 {
		h.logitScale = float32(cfgFloat(cfg, "query_pre_attn_scalar"))
	}
	h.swa = uint32(cfgInt(cfg, "sliding_window"))
	return h
}

func ropeTheta(cfg map[string]any, def float64) float64 {
	if rp := cfgMap(cfg, "rope_parameters"); rp != nil {
		if t := cfgFloat(rp, "rope_theta"); t > 0 {
			return t
		}
	}
	if t := cfgFloat(cfg, "rope_theta", "rotary_emb_base", "global_rope_theta"); t > 0 {
		return t
	}
	if def > 0 {
		return def
	}
	return 10000
}

func cfgIntSlice(v any) []int32 {
	switch x := v.(type) {
	case []int32:
		return append([]int32(nil), x...)
	case []any:
		out := make([]int32, 0, len(x))
		for _, e := range x {
			if n, ok := asInt(e); ok {
				out = append(out, int32(n))
			}
		}
		return out
	default:
		return nil
	}
}

func writeStandardKV(w *Writer, f Family, name string, h hyper, fileType uint32) {
	arch := f.GGUFArch
	w.AddKV("general.architecture", arch)
	w.AddKV("general.name", name)
	w.AddKV("general.type", "model")
	w.AddKV("general.file_type", fileType)
	w.AddKV("general.quantization_version", uint32(2))
	w.AddKV(arch+".block_count", uint32(h.nLayer))
	if h.nCtx > 0 {
		w.AddKV(arch+".context_length", uint32(h.nCtx))
	}
	if h.nEmbd > 0 {
		w.AddKV(arch+".embedding_length", uint32(h.nEmbd))
	}
	if h.vocab > 0 {
		w.AddKV(arch+".vocab_size", uint32(h.vocab))
	}
	if h.nFF > 0 {
		w.AddKV(arch+".feed_forward_length", uint32(h.nFF))
	}
	if h.nHead > 0 {
		w.AddKV(arch+".attention.head_count", uint32(h.nHead))
	}
	if h.nKV > 0 {
		w.AddKV(arch+".attention.head_count_kv", uint32(h.nKV))
	}
	if h.headDim > 0 {
		w.AddKV(arch+".attention.key_length", uint32(h.headDim))
		w.AddKV(arch+".attention.value_length", uint32(h.headDim))
		ropeDim := h.headDim
		if f.RopePartial && h.partialRotary > 0 {
			ropeDim = int(float64(h.headDim) * h.partialRotary)
		}
		if ropeDim > 0 {
			w.AddKV(arch+".rope.dimension_count", uint32(ropeDim))
		}
	}
	w.AddKV(arch+".attention.layer_norm_rms_epsilon", float32(h.rms))
	if h.rope > 0 {
		w.AddKV(arch+".rope.freq_base", float32(h.rope))
	}
	if len(h.mrope) > 0 {
		w.AddKV(arch+".rope.dimension_sections", h.mrope)
	}
	if h.nExpert > 0 {
		w.AddKV(arch+".expert_count", uint32(h.nExpert))
		if h.nExpertUsed > 0 {
			w.AddKV(arch+".expert_used_count", uint32(h.nExpertUsed))
		}
		if h.nExpertFF > 0 {
			w.AddKV(arch+".expert_feed_forward_length", uint32(h.nExpertFF))
		}
		if h.nSharedFF > 0 && h.nSharedFF != h.nExpertFF {
			w.AddKV(arch+".expert_shared_feed_forward_length", uint32(h.nSharedFF))
		}
	}
	if f.LinearAttn {
		if h.linConv > 0 {
			w.AddKV(arch+".ssm.conv_kernel", uint32(h.linConv))
		}
		if h.linKeyDim > 0 {
			w.AddKV(arch+".ssm.state_size", uint32(h.linKeyDim))
		}
		if h.linKeyHeads > 0 {
			w.AddKV(arch+".ssm.group_count", uint32(h.linKeyHeads))
		}
		if h.linValHeads > 0 {
			w.AddKV(arch+".ssm.time_step_rank", uint32(h.linValHeads))
		}
		if h.linValDim > 0 && h.linValHeads > 0 {
			w.AddKV(arch+".ssm.inner_size", uint32(h.linValDim*h.linValHeads))
		}
		if h.fullAttnEvery > 0 {
			w.AddKV(arch+".full_attention_interval", uint32(h.fullAttnEvery))
		}
	}
}

func writeFamilyKV(w *Writer, f Family, name string, cfg map[string]any, h hyper, fileType uint32) {
	writeStandardKV(w, f, name, h, fileType)
	writeRopeScaling(w, f.GGUFArch, cfg)
	if h.swa != 0 {
		w.AddKV(f.GGUFArch+".attention.sliding_window", h.swa)
	}
	if pat := slidingPattern(cfg, h.nLayer); slidingPatternUseful(pat) {
		w.AddKV(f.GGUFArch+".attention.sliding_window_pattern", pat)
	}
	if h.softcap != 0 {
		w.AddKV(f.GGUFArch+".final_logit_softcapping", h.softcap)
	}
	if attnCap := float32(cfgFloat(cfg, "attn_logit_softcapping")); attnCap != 0 {
		w.AddKV(f.GGUFArch+".attn_logit_softcapping", attnCap)
	}
	if local := cfgFloat(cfg, "rope_local_base_freq"); local > 0 {
		w.AddKV(f.GGUFArch+".rope.freq_base_swa", float32(local))
	}
	if f.GGUFArch == "gemma2" && h.nLayer > 0 && !slidingPatternUseful(slidingPattern(cfg, h.nLayer)) {
		pat := make([]bool, h.nLayer)
		for i := range pat {
			pat[i] = (i % 2) == 0 // layer 0 sliding, 1 full, …
		}
		w.AddKV(f.GGUFArch+".attention.sliding_window_pattern", pat)
	}
}

func writeRopeScaling(w *Writer, arch string, cfg map[string]any) {
	if cfg == nil {
		return
	}
	rs := cfgMap(cfg, "rope_scaling")
	if rs == nil {
		if rp := cfgMap(cfg, "rope_parameters"); rp != nil {
			rs = rp
		}
	}
	if rs == nil {
		return
	}
	typ := cfgString(rs, "rope_type", "type")
	if typ == "" || strings.EqualFold(typ, "default") || strings.EqualFold(typ, "mrope") {
		return
	}
	w.AddKV(arch+".rope.scaling.type", typ)
	if fac := cfgFloat(rs, "factor"); fac > 0 {
		w.AddKV(arch+".rope.scaling.factor", float32(fac))
	}
	if orig := cfgInt(rs, "original_max_position_embeddings", "original_context_length"); orig > 0 {
		w.AddKV(arch+".rope.scaling.original_context_length", uint32(orig))
	}
	if a := cfgFloat(rs, "attn_factor"); a > 0 {
		w.AddKV(arch+".rope.scaling.attn_factor", float32(a))
	}
}

func sniffTokenizerPre(tok *ggmlTokenizer, fallback string) string {
	if tok == nil {
		return fallback
	}
	has := func(s string) bool {
		for _, t := range tok.Tokens {
			if t == s {
				return true
			}
		}
		return false
	}
	switch {
	case has("<|start|>") && has("<|message|>"):
		return "llama4"
	case has("<|eot_id|>"):
		return "llama-bpe"
	case has("<start_of_turn>"):
		if fallback == "gemma3" {
			return "gemma3"
		}
		return "gemma"
	case has("<|im_start|>"):
		return "qwen2"
	case has("<|end|>") && (has("<|assistant|>") || has("<|user|>")):
		return "phi3"
	default:
		if fallback != "" {
			return fallback
		}
		return "default"
	}
}
