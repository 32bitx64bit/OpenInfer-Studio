package convert

import (
	"fmt"
	"regexp"
	"strings"
)

type glimmerAdapter struct{}

func (glimmerAdapter) ID() string { return "muse-glimmer" }

func (glimmerAdapter) Match(cfg map[string]any) bool {
	feat := detectLayout(cfg, nil)
	arch, err := inferArch(cfg, feat)
	return err == nil && arch == "muse-glimmer"
}

var glimmerLayerRe = regexp.MustCompile(`(?:^|\.)layers\.(\d+)\.(.+)$`)

// MapGlimmerName maps an HF safetensors name to a GGUF tensor name.
// skip=true means drop the tensor. unknown=true means it is not a known
// language weight (caller should fail closed unless skip).
func MapGlimmerName(hfName string) (ggufName string, skip, vision bool) {
	if isVisionTensor(hfName) {
		return "", true, true
	}
	n := strings.ReplaceAll(hfName, "language_model.", "")
	if strings.HasSuffix(n, ".bias") {
		return "", true, false
	}
	if strings.Contains(n, "rotary_emb") || strings.Contains(n, "inv_freq") ||
		strings.Contains(n, "cos_cached") || strings.Contains(n, "sin_cached") {
		return "", true, false
	}
	// HF Q/K RMSNorm is replaced by synthesized qk-norm tensors.
	if strings.Contains(n, "self_attn.q_norm") || strings.Contains(n, "self_attn.k_norm") {
		return "", true, false
	}
	stem := strings.TrimSuffix(n, ".weight")
	switch {
	case strings.HasSuffix(stem, "embed_tokens") || stem == "model.embed_tokens" || stem == "embed_tokens":
		return "token_embd.weight", false, false
	case stem == "lm_head" || strings.HasSuffix(stem, ".lm_head"):
		return "output.weight", false, false
	case stem == "model.norm" || stem == "norm":
		return "output_norm.weight", false, false
	}
	m := glimmerLayerRe.FindStringSubmatch(stem)
	if m == nil {
		return "", false, false
	}
	bid, rest := m[1], m[2]
	var blk string
	switch rest {
	case "self_attn.q_proj":
		blk = "attn_q"
	case "self_attn.k_proj":
		blk = "attn_k"
	case "self_attn.v_proj":
		blk = "attn_v"
	case "self_attn.o_proj":
		blk = "attn_output"
	case "self_attn.gate_proj":
		blk = "attn_gate"
	case "mlp.gate_proj":
		blk = "ffn_gate"
	case "mlp.up_proj":
		blk = "ffn_up"
	case "mlp.down_proj":
		blk = "ffn_down"
	case "input_layernorm":
		blk = "attn_norm"
	case "post_attention_layernorm":
		blk = "post_attention_norm"
	case "pre_feedforward_layernorm":
		blk = "ffn_norm"
	case "post_feedforward_layernorm":
		blk = "post_ffw_norm"
	default:
		return "", false, false
	}
	return "blk." + bid + "." + blk + ".weight", false, false
}

type glimmerWork struct {
	GGUF   string
	Shape  []int64 // GGUF dims (reversed vs HF for rank-2)
	DType  int
	Src    *TensorRef
	Kind   string // copy, unpermute_q, unpermute_k, rms_plus, rms, qk_q, qk_k, tie
	NHeads int
}

func (glimmerAdapter) Convert(dir string, tensors []TensorRef, cfg map[string]any, w *Writer) (*ConvertStats, error) {
	tok, err := loadTokenizer(dir)
	if err != nil {
		return nil, err
	}
	nHead := cfgInt(cfg, "num_attention_heads", "n_head")
	nKV := cfgInt(cfg, "num_key_value_heads", "n_kv_heads")
	if nKV <= 0 {
		nKV = nHead
	}
	nEmbd := cfgInt(cfg, "hidden_size", "n_embd")
	nFF := cfgInt(cfg, "intermediate_size", "n_inner")
	nLayer := cfgInt(cfg, "num_hidden_layers", "n_layer")
	nCtx := cfgInt(cfg, "max_position_embeddings", "n_ctx", "max_length")
	headDim := cfgInt(cfg, "head_dim")
	if headDim <= 0 && nHead > 0 {
		headDim = nEmbd / nHead
	}
	vocab := alignVocab(tok, tensors, cfgInt(cfg, "vocab_size"))
	rms := cfgFloat(cfg, "rms_norm_eps", "layer_norm_eps")
	if rms == 0 {
		rms = 1e-5
	}
	rope := glimmerRopeTheta(cfg)
	qkScale := float32(cfgFloat(cfg, "qk_scale_factor"))
	if qkScale == 0 {
		qkScale = 1
	}
	softcap := float32(cfgFloat(cfg, "final_logit_softcapping"))
	logitScale := float32(cfgFloat(cfg, "output_multiplier"))
	swa := uint32(cfgInt(cfg, "sliding_window"))

	weightDType := inferWeightDType(tensors)
	store, err := storeType(weightDType)
	if err != nil {
		return nil, err
	}
	fileType := uint32(1) // F16
	if store == GGMLBF16 {
		fileType = 32
	}

	name := cfgString(cfg, "name")
	if name == "" {
		name = "Muse-Glimmer"
	}

	w.AddKV("general.architecture", "muse-glimmer")
	w.AddKV("general.name", name)
	w.AddKV("general.type", "model")
	w.AddKV("general.file_type", fileType)
	w.AddKV("general.quantization_version", uint32(2))
	w.AddKV("muse-glimmer.block_count", uint32(nLayer))
	w.AddKV("muse-glimmer.context_length", uint32(nCtx))
	w.AddKV("muse-glimmer.embedding_length", uint32(nEmbd))
	if vocab > 0 {
		w.AddKV("muse-glimmer.vocab_size", uint32(vocab))
	}
	w.AddKV("muse-glimmer.feed_forward_length", uint32(nFF))
	w.AddKV("muse-glimmer.attention.head_count", uint32(nHead))
	w.AddKV("muse-glimmer.attention.head_count_kv", uint32(nKV))
	w.AddKV("muse-glimmer.attention.key_length", uint32(headDim))
	w.AddKV("muse-glimmer.attention.value_length", uint32(headDim))
	w.AddKV("muse-glimmer.attention.layer_norm_rms_epsilon", float32(rms))
	w.AddKV("muse-glimmer.rope.freq_base", float32(rope))
	if softcap != 0 {
		w.AddKV("muse-glimmer.final_logit_softcapping", softcap)
	}
	if logitScale != 0 {
		w.AddKV("muse-glimmer.logit_scale", logitScale)
	}
	if swa != 0 {
		w.AddKV("muse-glimmer.attention.sliding_window", swa)
	}
	if pat := slidingPattern(cfg, nLayer); len(pat) > 0 {
		w.AddKV("muse-glimmer.attention.sliding_window_pattern", pat)
	}
	w.addTokenizer(tok, "llama4")

	stats := &ConvertStats{Architecture: "muse-glimmer", GGUFType: store}
	var work []glimmerWork
	haveOutput := false
	var embed *TensorRef
	seenQ := map[int]bool{}

	for i := range tensors {
		t := tensors[i]
		ggufName, skip, vision := MapGlimmerName(t.Name)
		if vision {
			stats.Skipped++
			if len(stats.Warnings) == 0 {
				stats.Warnings = append(stats.Warnings, "vision tensors were skipped; this GGUF is language-only (pair an existing mmproj for images)")
			}
			continue
		}
		if skip {
			stats.Skipped++
			continue
		}
		if ggufName == "" {
			return stats, fmt.Errorf("cannot map tensor %q to muse-glimmer GGUF (not convertible)", t.Name)
		}
		if ggufName == "output.weight" {
			haveOutput = true
		}
		if ggufName == "token_embd.weight" {
			embed = &tensors[i]
		}
		kind := "copy"
		nHeads := 0
		switch {
		case strings.Contains(ggufName, "attn_q.weight"):
			kind = "unpermute_q"
			nHeads = nHead
		case strings.Contains(ggufName, "attn_k.weight"):
			kind = "unpermute_k"
			nHeads = nKV
		case strings.HasSuffix(ggufName, "_norm.weight") && ggufName != "output_norm.weight":
			kind = "rms_plus"
		case ggufName == "output_norm.weight":
			kind = "rms"
		}
		hfShape := t.Shape
		if isVocabWeight(ggufName) {
			hfShape = padVocabLeading(t.Shape, vocab)
		}
		ggufShape := ggufDims(hfShape)
		dtype := store
		if kind == "rms_plus" || kind == "rms" {
			dtype = GGMLF32
			ggufShape = append([]int64(nil), t.Shape...)
		}
		work = append(work, glimmerWork{GGUF: ggufName, Shape: ggufShape, DType: dtype, Src: &tensors[i], Kind: kind, NHeads: nHeads})
		if kind == "unpermute_q" {
			bid := layerIndex(ggufName)
			if bid >= 0 && !seenQ[bid] {
				seenQ[bid] = true
				qshape := []int64{int64(headDim)}
				work = append(work,
					glimmerWork{GGUF: fmt.Sprintf("blk.%d.attn_q_norm.weight", bid), Shape: qshape, DType: GGMLF32, Kind: "qk_q"},
					glimmerWork{GGUF: fmt.Sprintf("blk.%d.attn_k_norm.weight", bid), Shape: qshape, DType: GGMLF32, Kind: "qk_k"},
				)
			}
		}
	}
	if !haveOutput {
		if embed == nil {
			return stats, fmt.Errorf("model has neither lm_head nor embed_tokens")
		}
		work = append(work, glimmerWork{GGUF: "output.weight", Shape: ggufDims(padVocabLeading(embed.Shape, vocab)), DType: store, Src: embed, Kind: "tie"})
	}

	for _, item := range work {
		if err := w.PlanTensor(item.GGUF, item.Shape, item.DType); err != nil {
			return stats, err
		}
	}
	if err := w.WriteHeader(); err != nil {
		return stats, err
	}
	elem := 2
	if strings.EqualFold(weightDType, "F32") {
		elem = 4
	}
	for _, item := range work {
		payload, err := glimmerPayload(item, weightDType, elem, qkScale, headDim)
		if err != nil {
			return stats, err
		}
		if err := w.WriteTensor(payload); err != nil {
			return stats, err
		}
		stats.Tensors++
	}
	return stats, nil
}

// glimmerRopeTheta is the language-tower RoPE base (local layers, θ=500000).
// HF stores it under text_config.rope_parameters.rope_theta (flattened by
// ParseConfig). The vision tower's rope_parameters.rope_theta is 10000 and
// must not leak into the GGUF — that default is what llama.cpp uses when
// muse-glimmer.rope.freq_base is missing, and it tanks long-context quality.
func glimmerRopeTheta(cfg map[string]any) float64 {
	if rp := cfgMap(cfg, "rope_parameters"); rp != nil {
		if t := cfgFloat(rp, "rope_theta"); t > 0 {
			return t
		}
	}
	if t := cfgFloat(cfg, "rope_theta", "rotary_emb_base", "global_rope_theta"); t > 0 {
		return t
	}
	for _, t := range cfgFloatSlice(cfg["layer_rope_theta"]) {
		if t > 0 {
			return t
		}
	}
	return 500000
}

func inferWeightDType(tensors []TensorRef) string {
	counts := map[string]int{}
	for _, t := range tensors {
		if isVisionTensor(t.Name) || len(t.Shape) < 2 {
			continue
		}
		counts[strings.ToUpper(t.DType)]++
	}
	best, n := "BF16", 0
	for k, v := range counts {
		if v > n {
			best, n = k, v
		}
	}
	if n == 0 {
		return "BF16"
	}
	return best
}

func slidingPattern(cfg map[string]any, nLayer int) []bool {
	raw := cfg["layer_types"]
	var types []string
	switch x := raw.(type) {
	case []any:
		for _, e := range x {
			if s, ok := e.(string); ok {
				types = append(types, s)
			}
		}
	case []string:
		types = x
	}
	if len(types) == 0 {
		return nil
	}
	out := make([]bool, 0, nLayer)
	for i := 0; i < nLayer; i++ {
		t := types[i%len(types)]
		out = append(out, t == "sliding_attention")
	}
	return out
}

func slidingPatternUseful(pat []bool) bool {
	for _, p := range pat {
		if p {
			return true
		}
	}
	return false
}

func layerIndex(ggufName string) int {
	var n int
	if _, err := fmt.Sscanf(ggufName, "blk.%d.", &n); err != nil {
		return -1
	}
	return n
}

func ggufDims(hf []int64) []int64 {
	if len(hf) == 2 {
		return []int64{hf[1], hf[0]}
	}
	out := make([]int64, len(hf))
	for i := range hf {
		out[i] = hf[len(hf)-1-i]
	}
	return out
}

func glimmerPayload(item glimmerWork, srcDType string, elem int, qkScale float32, headDim int) ([]byte, error) {
	switch item.Kind {
	case "qk_q":
		return f32Bytes(filledF32(headDim, qkScale)), nil
	case "qk_k":
		return f32Bytes(onesF32(headDim)), nil
	}
	if item.Src == nil {
		return nil, fmt.Errorf("tensor %s has no source", item.GGUF)
	}
	raw, err := ReadPayload(*item.Src)
	if err != nil {
		return nil, err
	}
	if isVocabWeight(item.GGUF) {
		srcElem := elemSize(item.Src.DType)
		if srcElem <= 0 {
			srcElem = elem
		}
		padded, err := padLeadingDim(raw, item.Src.Shape, ggufDims(item.Shape), srcElem)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", item.GGUF, err)
		}
		raw = padded
	}
	switch item.Kind {
	case "rms_plus":
		f32, err := toF32(raw, item.Src.DType)
		if err != nil {
			return nil, err
		}
		return addOneF32(f32), nil
	case "rms":
		return toF32(raw, item.Src.DType)
	case "unpermute_q", "unpermute_k":
		if len(item.Src.Shape) != 2 {
			return nil, fmt.Errorf("%s: expected rank-2", item.GGUF)
		}
		dim1 := int(item.Src.Shape[0])
		dim2 := int(item.Src.Shape[1])
		unp, err := unpermuteHF(raw, item.NHeads, dim1, dim2, elem)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", item.GGUF, err)
		}
		return convertPayload(unp, srcDType, item.DType)
	case "copy", "tie":
		return convertPayload(raw, item.Src.DType, item.DType)
	default:
		return nil, fmt.Errorf("unknown work kind %s", item.Kind)
	}
}
