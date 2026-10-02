package convert

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
)

// Gemma 3n and Gemma 4 ("E" models) in llama.cpp's gemma3n / gemma4
// architectures. Both add per-layer embeddings (PLE): a second, per-layer
// token embedding table plus a projection of the main embedding, gated into
// every decoder layer. Gemma 3n additionally carries AltUp (four parallel
// residual streams with learned predict/correct coefficients) and LAuReL
// (a low-rank residual branch). The tensor naming, stacking and metadata
// below follow llama.cpp's own converter (conversion/gemma.py) and loader
// (src/models/gemma3n.cpp, gemma4.cpp).

// Per-layer tensors shared by both families (per-layer embedding path).
var gemmaPLELayers = map[string]string{
	"per_layer_input_gate":      "inp_gate.weight",
	"per_layer_projection":      "proj.weight",
	"post_per_layer_input_norm": "post_norm.weight",
}

// Gemma 3n only: AltUp and LAuReL.
var gemma3nLayers = map[string]string{
	"altup.correction_coefs":     "altup_correct_coef.weight",
	"altup.correct_output_scale": "altup_correct_scale.weight",
	"altup.prediction_coefs":     "altup_predict_coef.weight",
	"altup.modality_router":      "altup_router.weight",
	"altup.router_norm":          "altup_router_norm.weight",
	"laurel.linear_left":         "laurel_l.weight",
	"laurel.linear_right":        "laurel_r.weight",
	"laurel.post_laurel_norm":    "laurel_post_norm.weight",
}

// Gemma 4 only: a learned scalar on each layer's output.
var gemma4Layers = map[string]string{
	"layer_scalar": "layer_output_scale.weight",
}

// configureGemmaN specializes a family for gemma3n / gemma4.
func configureGemmaN(f *Family, arch string) {
	f.Gemma = arch
	// These models apply their RMSNorm weight as stored (no Gemma 1–3 "+1").
	f.RMSPlus = rmsPlusNone
	f.KeepQKNorm = true
	// The output head is tied to token_embd; llama.cpp falls back to it when
	// no output tensor exists, so a duplicate would only cost disk.
	f.TieOutput = false
	f.RopePartial = false
	extra := map[string]string{
		"self_attn.q_norm": "attn_q_norm.weight",
		"self_attn.k_norm": "attn_k_norm.weight",
	}
	f.Layers = overlay(overlay(f.Layers, extra), gemmaPLELayers)
	switch arch {
	case "gemma3n":
		f.Layers = overlay(f.Layers, gemma3nLayers)
	case "gemma4":
		f.Layers = overlay(f.Layers, gemma4Layers)
	}
}

var altupStackRe = regexp.MustCompile(`^(altup_projections|altup_unembed_projections)\.(\d+)$`)

// gemmaGlobal maps the model-level (non-layer) PLE and AltUp tensors.
func (f Family) gemmaGlobal(stem string) (mappedTensor, bool) {
	base := strings.TrimPrefix(stem, "model.")
	switch base {
	case "embed_tokens_per_layer":
		return mappedTensor{GGUF: "per_layer_token_embd.weight", Kind: kindCopy, Expert: -1}, true
	case "per_layer_model_projection":
		return mappedTensor{GGUF: "per_layer_model_proj.weight", Kind: kindCopy, Expert: -1}, true
	case "per_layer_projection_norm":
		return mappedTensor{GGUF: "per_layer_proj_norm.weight", Kind: f.normKind("per_layer_proj_norm.weight"), Expert: -1}, true
	}
	if f.Gemma == "gemma3n" {
		if m := altupStackRe.FindStringSubmatch(base); m != nil {
			gg := "altup_proj.weight"
			if m[1] == "altup_unembed_projections" {
				gg = "altup_unembd_proj.weight"
			}
			var idx int
			fmt.Sscanf(m[2], "%d", &idx)
			return mappedTensor{GGUF: gg, Kind: kindStackIdx, Expert: idx}, true
		}
	}
	return mappedTensor{}, false
}

// gemmaForcesF32 lists tensors llama.cpp keeps in F32 even when they are
// matrices (they are tiny and read by float-only ops).
func gemmaForcesF32(ggufName string) bool {
	return strings.HasSuffix(ggufName, ".altup_correct_coef.weight") ||
		strings.HasSuffix(ggufName, ".altup_predict_coef.weight")
}

// stackSrc is one source of an index-stacked tensor.
type stackSrc struct {
	idx int
	ref TensorRef
}

// orderStack sorts srcs by index and checks the indices are 0..n-1 with
// identical shapes. It returns the ordered refs and the stacked HF shape
// ([n] + shape).
func orderStack(name string, srcs []stackSrc, want int) ([]TensorRef, []int64, error) {
	if len(srcs) == 0 {
		return nil, nil, fmt.Errorf("%s: no sources", name)
	}
	sort.Slice(srcs, func(i, j int) bool { return srcs[i].idx < srcs[j].idx })
	if want > 0 && len(srcs) != want {
		return nil, nil, fmt.Errorf("%s: got %d stacked tensors, want %d", name, len(srcs), want)
	}
	refs := make([]TensorRef, len(srcs))
	for i, s := range srcs {
		if s.idx != i {
			return nil, nil, fmt.Errorf("%s: missing stacked tensor %d (have index %d)", name, i, s.idx)
		}
		if !sameShape(s.ref.Shape, srcs[0].ref.Shape) {
			return nil, nil, fmt.Errorf("%s: tensor %q shape %v != %v", name, s.ref.Name, s.ref.Shape, srcs[0].ref.Shape)
		}
		refs[i] = s.ref
	}
	return refs, append([]int64{int64(len(refs))}, srcs[0].ref.Shape...), nil
}

// stackByOrder concatenates the payloads of refs (already ordered).
func stackByOrder(refs []TensorRef) ([]byte, error) {
	var out []byte
	for _, r := range refs {
		raw, err := ReadPayload(r)
		if err != nil {
			return nil, err
		}
		out = append(out, raw...)
	}
	return out, nil
}

// --- hyperparameters --------------------------------------------------

// gemmaRope returns the rope parameter block of a layer type, or the flat
// block when the config does not split by layer type.
func gemmaRope(cfg map[string]any, layerType string) map[string]any {
	rp := cfgMap(cfg, "rope_parameters")
	if m := cfgMap(rp, layerType); m != nil {
		return m
	}
	return rp
}

// gemmaRopeBases resolves the global and sliding-window rope bases. swa is
// 0 when the config states none (llama.cpp then uses its default).
func gemmaRopeBases(cfg map[string]any) (global, swa float64) {
	global = cfgFloat(gemmaRope(cfg, "full_attention"), "rope_theta")
	if global <= 0 {
		global = cfgFloat(cfg, "rope_theta", "global_rope_theta")
	}
	if global <= 0 {
		global = 1_000_000
	}
	swa = cfgFloat(cfgMap(cfgMap(cfg, "rope_parameters"), "sliding_attention"), "rope_theta")
	if swa <= 0 {
		swa = cfgFloat(cfg, "rope_local_base_freq")
	}
	return global, swa
}

// gemmaFFLengths returns the per-layer FFN width, or a single value.
func gemmaFFLengths(cfg map[string]any, nLayer int) (scalar int, arr []int32) {
	switch x := cfg["intermediate_size"].(type) {
	case []any:
		for _, e := range x {
			if n, ok := asInt(e); ok {
				arr = append(arr, int32(n))
			}
		}
		return 0, arr
	default:
		n := cfgInt(cfg, "intermediate_size")
		return n, nil
	}
}

// gemma4HeadDims resolves the full-attention (global) head dimension and
// KV head count; the config may carry them top-level or per layer.
func gemma4HeadDims(cfg map[string]any) (headDimFull, kvHeadsFull int) {
	headDimFull = cfgInt(cfg, "global_head_dim")
	kvHeadsFull = cfgInt(cfg, "num_global_key_value_heads")
	if headDimFull > 0 && kvHeadsFull > 0 {
		return
	}
	types := stringSlice(cfg["layer_types"])
	plc := cfgMap(cfg, "per_layer_config")
	for i := 0; i < len(types); i++ {
		if types[i] != "full_attention" {
			continue
		}
		lc := cfgMap(plc, fmt.Sprintf("%d", i))
		if headDimFull <= 0 {
			headDimFull = cfgInt(lc, "head_dim")
		}
		if kvHeadsFull <= 0 {
			kvHeadsFull = cfgInt(lc, "num_key_value_heads")
		}
		if headDimFull > 0 && kvHeadsFull > 0 {
			break
		}
	}
	return
}

// validateGemmaN rejects configs the llama.cpp loaders cannot represent.
func validateGemmaN(f Family, cfg map[string]any, h hyper) error {
	switch f.Gemma {
	case "gemma3n":
		if n := cfgInt(cfg, "altup_num_inputs"); n != 0 && n != 4 {
			return fmt.Errorf("gemma3n: altup_num_inputs=%d; llama.cpp supports exactly 4", n)
		}
		if cfgInt(cfg, "hidden_size_per_layer_input") <= 0 {
			return fmt.Errorf("gemma3n: config has no hidden_size_per_layer_input")
		}
	case "gemma4":
		hd, _ := gemma4HeadDims(cfg)
		if hd <= 0 {
			return fmt.Errorf("gemma4: cannot determine the full-attention head dimension (global_head_dim / per_layer_config)")
		}
		if len(stringSlice(cfg["layer_types"])) == 0 {
			return fmt.Errorf("gemma4: config has no layer_types")
		}
	}
	if h.nLayer <= 0 || h.nEmbd <= 0 {
		return fmt.Errorf("%s: config has no layer count / hidden size", f.Gemma)
	}
	return nil
}

// gemmaActivationSparsity is the standard-normal inverse CDF (in float32,
// like the reference) of each layer's activation sparsity level. A level of
// 0 maps to -Inf, which llama.cpp treats as "no sparsity".
func gemmaActivationSparsity(pattern []float64) []float32 {
	out := make([]float32, len(pattern))
	for i, p := range pattern {
		out[i] = float32(math.Sqrt2 * math.Erfinv(2*p-1))
	}
	return out
}

// writeGemmaNKV writes (or replaces) the metadata the gemma3n / gemma4
// loaders read on top of the standard Gemma keys.
func writeGemmaNKV(w *Writer, f Family, cfg map[string]any, h hyper) {
	arch := f.GGUFArch
	global, swa := gemmaRopeBases(cfg)
	w.SetKV(arch+".rope.freq_base", float32(global))
	if swa > 0 {
		w.SetKV(arch+".rope.freq_base_swa", float32(swa))
	}

	// sliding_window_pattern is required by the gemma4 loader and used by
	// gemma3n: always write it, even when no layer is sliding.
	if pat := slidingPattern(cfg, h.nLayer); len(pat) > 0 {
		w.SetKV(arch+".attention.sliding_window_pattern", pat)
	}

	w.SetKV(arch+".embedding_length_per_layer_input", uint32(cfgInt(cfg, "hidden_size_per_layer_input")))
	w.SetKV(arch+".attention.shared_kv_layers", uint32(cfgInt(cfg, "num_kv_shared_layers")))

	if scalar, arr := gemmaFFLengths(cfg, h.nLayer); len(arr) > 0 {
		w.SetKV(arch+".feed_forward_length", arr)
	} else if scalar > 0 {
		w.SetKV(arch+".feed_forward_length", uint32(scalar))
	}

	switch f.Gemma {
	case "gemma3n":
		w.SetKV(arch+".altup.active_idx", uint32(cfgInt(cfg, "altup_active_idx")))
		w.SetKV(arch+".altup.num_inputs", uint32(cfgInt(cfg, "altup_num_inputs")))
		if pat := cfgFloatSlice(cfg["activation_sparsity_pattern"]); len(pat) > 0 {
			w.SetKV(arch+".activation_sparsity_scale", gemmaActivationSparsity(pat))
		}
	case "gemma4":
		writeGemma4KV(w, arch, cfg, h)
	}
}

func writeGemma4KV(w *Writer, arch string, cfg map[string]any, h hyper) {
	headDimFull, kvFull := gemma4HeadDims(cfg)
	headDimSWA := cfgInt(cfg, "head_dim")
	if headDimSWA <= 0 {
		headDimSWA = h.headDim
	}
	w.SetKV(arch+".attention.key_length", uint32(headDimFull))
	w.SetKV(arch+".attention.value_length", uint32(headDimFull))
	w.SetKV(arch+".attention.key_length_swa", uint32(headDimSWA))
	w.SetKV(arch+".attention.value_length_swa", uint32(headDimSWA))

	// Full-attention layers use "proportional" rope: all head_dim dims are
	// declared rotary and rope_freqs.weight zeroes the unrotated ones.
	w.SetKV(arch+".rope.dimension_count", uint32(headDimFull))
	w.SetKV(arch+".rope.dimension_count_swa", uint32(float64(headDimSWA)*h.partialRotary))

	if swaKV := cfgInt(cfg, "num_key_value_heads"); kvFull > 0 && swaKV > 0 {
		pat := slidingPattern(cfg, h.nLayer)
		arr := make([]int32, h.nLayer)
		for i := range arr {
			if i < len(pat) && pat[i] {
				arr[i] = int32(swaKV)
			} else {
				arr[i] = int32(kvFull)
			}
		}
		w.SetKV(arch+".attention.head_count_kv", arr)
	}

	// use_double_wide_mlp doubles the FFN width of the KV-shared layers.
	if cfgBool(cfg, "use_double_wide_mlp") {
		n := cfgInt(cfg, "intermediate_size")
		first := h.nLayer - cfgInt(cfg, "num_kv_shared_layers")
		arr := make([]int32, h.nLayer)
		for i := range arr {
			arr[i] = int32(n)
			if i >= first {
				arr[i] = int32(2 * n)
			}
		}
		w.SetKV(arch+".feed_forward_length", arr)
	}
	if ff := cfgInt(cfg, "expert_intermediate_size", "moe_intermediate_size"); ff > 0 {
		w.SetKV(arch+".expert_feed_forward_length", uint32(ff))
	}
}

// gemmaExtraWork returns the tensors a family generates rather than reads.
func gemmaExtraWork(f Family, cfg map[string]any) ([]workItem, error) {
	if f.Gemma != "gemma4" {
		return nil, nil
	}
	// Full-attention layers use "proportional" rope with a partial rotary
	// factor: the rotated frequencies are 1, the rest effectively infinite.
	rp := gemmaRope(cfg, "full_attention")
	if typ := cfgString(rp, "rope_type", "type"); typ != "" && typ != "proportional" {
		return nil, fmt.Errorf("gemma4: unsupported full-attention rope type %q", typ)
	}
	headDimFull, _ := gemma4HeadDims(cfg)
	partial := cfgFloat(rp, "partial_rotary_factor")
	if partial <= 0 {
		partial = 1
	}
	nRot := int(float64(headDimFull) * partial / 2)
	nUnrot := headDimFull/2 - nRot
	vals := make([]float32, 0, headDimFull/2)
	for i := 0; i < nRot; i++ {
		vals = append(vals, 1)
	}
	for i := 0; i < nUnrot; i++ {
		vals = append(vals, 1e30)
	}
	return []workItem{{
		GGUF:  "rope_freqs.weight",
		Shape: []int64{int64(len(vals))},
		DType: GGMLF32,
		Kind:  kindData,
		Data:  f32Bytes(vals),
	}}, nil
}
