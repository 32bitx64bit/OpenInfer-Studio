package convert

import (
	_ "embed"
	"fmt"
	"sort"
	"strings"
	"unicode"
)

// llama.cpp MODEL_ARCH_NAMES — the only architecture allowlist.
//
//go:embed llamacpp_archs.txt
var llamaCPPArchFile string

var llamaCPPArch map[string]bool

func init() {
	llamaCPPArch = make(map[string]bool)
	for _, line := range strings.Split(llamaCPPArchFile, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		llamaCPPArch[line] = true
	}
}

// layout is inferred from config.json and/or safetensors names — not from
// Hugging Face class-name allowlists.
type layout struct {
	Decoder     bool
	LinearAttn  bool
	SplitLinear bool // in_proj_qkv / in_proj_z (Qwen3.5), not packed qkvz
	MoE         bool
	GemmaFF     bool
	QKNorm      bool
	PackedQKV   bool
	AttnGate    bool
	InternLM    bool
	MLA         bool
	block       string
}

func tensorNames(tensors []TensorRef) []string {
	out := make([]string, len(tensors))
	for i, t := range tensors {
		out[i] = t.Name
	}
	return out
}

func detectLayout(cfg map[string]any, names []string) layout {
	var f layout
	blob := strings.ToLower(strings.Join(names, "\n"))
	has := func(s string) bool {
		return strings.Contains(blob, strings.ToLower(s))
	}

	if cfgInt(cfg, "kv_lora_rank", "q_lora_rank", "qk_nope_head_dim") > 0 ||
		cfgBool(cfg, "use_mla", "multi_latent_attention") ||
		has("kv_a_proj") || has("kv_b_proj") || has("q_a_proj") || has("q_b_proj") {
		f.MLA = true
		f.block = "this checkpoint uses MLA / latent attention; those safetensors are not convertible here"
		return f
	}
	if cfgInt(cfg, "altup_num_inputs", "hidden_size_per_layer_input") > 0 ||
		has("altup") || has("per_layer_input") {
		f.block = "altup / per-layer embeddings are not convertible"
		return f
	}
	if has("in_proj_qkvz") || has("in_proj_ba") && !has("in_proj_qkv") {
		f.block = "packed linear-attention weights (Qwen3-Next style) are not convertible"
		return f
	}

	mt := strings.ToLower(cfgString(cfg, "model_type"))
	archBlob := mt
	for _, a := range stringSlice(cfg["architectures"]) {
		archBlob += " " + strings.ToLower(a)
	}
	if strings.Contains(archBlob, "rwkv") || has("time_mix") || has("time_decay") || has(".rwkv") {
		f.block = "RWKV weights are not convertible"
		return f
	}
	if (strings.Contains(archBlob, "mamba") || has("mixer.A_log") || has("mixer.conv1d")) &&
		!has("linear_attn.") && cfgInt(cfg, "linear_num_key_heads", "linear_conv_kernel_dim") == 0 {
		f.block = "Mamba weights are not convertible"
		return f
	}

	f.LinearAttn = cfgInt(cfg, "linear_num_key_heads", "linear_conv_kernel_dim", "linear_key_head_dim") > 0 ||
		has("linear_attn.") || hasLayerType(cfg, "linear")
	f.SplitLinear = has("linear_attn.in_proj_qkv") || has("linear_attn.in_proj_z")
	f.MoE = cfgInt(cfg, "num_experts", "num_local_experts", "n_routed_experts") > 0 ||
		has(".experts.") || has("block_sparse_moe")
	f.GemmaFF = has("pre_feedforward_layernorm") || has("post_feedforward_layernorm")
	f.QKNorm = has("self_attn.q_norm") || has("self_attn.k_norm") ||
		cfgBool(cfg, "use_qk_norm", "qk_norm")
	f.PackedQKV = has("qkv_proj") || has("gate_up_proj")
	f.AttnGate = has("self_attn.gate_proj")
	f.InternLM = has("attention.wq") || has("feed_forward.w1")
	f.Decoder = cfgInt(cfg, "hidden_size", "n_embd", "d_model") > 0 &&
		cfgInt(cfg, "num_attention_heads", "n_head", "num_heads") > 0 &&
		cfgInt(cfg, "num_hidden_layers", "n_layer", "num_layers") > 0
	return f
}

func hasLayerType(cfg map[string]any, needle string) bool {
	needle = strings.ToLower(needle)
	for _, s := range stringSlice(cfg["layer_types"]) {
		if strings.Contains(strings.ToLower(s), needle) {
			return true
		}
	}
	return false
}

func cfgBool(cfg map[string]any, keys ...string) bool {
	for _, k := range keys {
		switch v := cfg[k].(type) {
		case bool:
			if v {
				return true
			}
		case string:
			if strings.EqualFold(v, "true") {
				return true
			}
		}
	}
	return false
}

func inferArch(cfg map[string]any, feat layout) (string, error) {
	if feat.block != "" {
		return "", fmt.Errorf("%s", feat.block)
	}
	cands := expandNames(cfg)
	sort.Slice(cands, func(i, j int) bool {
		if len(cands[i]) != len(cands[j]) {
			return len(cands[i]) > len(cands[j])
		}
		return cands[i] < cands[j]
	})

	for _, c := range cands {
		if !llamaCPPArch[c] {
			continue
		}
		return withMoESuffix(c, feat), nil
	}

	// HF's model_type is often not a GGUF architecture (mistral, mixtral).
	// Match the graph, never prefix-fuzzy (mistral must not become mistral3).
	if feat.LinearAttn {
		if feat.MoE && llamaCPPArch["qwen35moe"] {
			return "qwen35moe", nil
		}
		if llamaCPPArch["qwen35"] {
			return "qwen35", nil
		}
	}
	if feat.AttnGate && feat.GemmaFF && llamaCPPArch["muse-glimmer"] {
		return "muse-glimmer", nil
	}
	if feat.InternLM && llamaCPPArch["internlm2"] {
		return "internlm2", nil
	}
	if feat.GemmaFF && feat.QKNorm && llamaCPPArch["gemma3"] {
		return "gemma3", nil
	}
	if feat.GemmaFF && llamaCPPArch["gemma2"] {
		return "gemma2", nil
	}
	if feat.PackedQKV && llamaCPPArch["phi3"] {
		return "phi3", nil
	}
	if feat.MoE && llamaCPPArch["llama"] {
		return "llama", nil
	}
	if feat.Decoder && llamaCPPArch["llama"] {
		return "llama", nil
	}

	mt, _ := cfg["model_type"].(string)
	tried := strings.Join(cands, ", ")
	if tried == "" {
		tried = mt
	}
	return "", fmt.Errorf("llama.cpp has no loader for architecture %q (tried %s)", mt, tried)
}

func withMoESuffix(c string, feat layout) string {
	if !feat.MoE || strings.Contains(c, "moe") {
		return c
	}
	for _, suf := range []string{"moe", "-moe", "_moe"} {
		if llamaCPPArch[c+suf] {
			return c + suf
		}
	}
	return c
}

func requireConvertible(arch string, feat layout) error {
	if feat.block != "" {
		return fmt.Errorf("%s", feat.block)
	}
	if strings.Contains(arch, "bert") || strings.Contains(arch, "embed") ||
		arch == "t5" || arch == "t5encoder" || arch == "clip" {
		return fmt.Errorf("llama.cpp architecture %q is not a causal LM this converter can write", arch)
	}
	if strings.HasPrefix(arch, "rwkv") || strings.HasPrefix(arch, "arwkv") ||
		strings.HasPrefix(arch, "mamba") || arch == "jamba" {
		return fmt.Errorf("llama.cpp can load %s, but this converter cannot emit those tensors", arch)
	}
	// llama.cpp's qwen3next loader is the packed graph; we only emit split Qwen3.5 projections.
	if arch == "qwen3next" && !feat.SplitLinear {
		return fmt.Errorf("llama.cpp can load qwen3next, but packed linear-attention safetensors are not convertible")
	}
	return nil
}

var hfClassSuffixes = []string{
	"ForConditionalGeneration",
	"ForCausalLM",
	"ForSequenceClassification",
	"TextModel",
	"Model",
}

func expandNames(cfg map[string]any) []string {
	var raw []string
	if mt, _ := cfg["model_type"].(string); mt != "" {
		raw = append(raw, mt)
	}
	for _, a := range stringSlice(cfg["architectures"]) {
		raw = append(raw, a)
		stripped := a
		for _, suf := range hfClassSuffixes {
			stripped = strings.TrimSuffix(stripped, suf)
		}
		if stripped != a && stripped != "" {
			raw = append(raw, stripped)
		}
	}
	seen := map[string]bool{}
	var out []string
	add := func(s string) {
		s = strings.ToLower(strings.TrimSpace(s))
		if s == "" || seen[s] {
			return
		}
		seen[s] = true
		out = append(out, s)
	}
	for _, r := range raw {
		for _, v := range nameVariants(r) {
			add(v)
		}
	}
	return out
}

func nameVariants(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	bases := []string{s}
	lower := strings.ToLower(s)
	for _, suf := range []string{"_text", "-text", "_vision", "-vision"} {
		if strings.HasSuffix(lower, suf) {
			bases = append(bases, s[:len(s)-len(suf)])
		}
	}
	var out []string
	for _, b := range bases {
		kebab := camelToKebab(b)
		out = append(out,
			b,
			strings.ReplaceAll(b, "_", "-"),
			kebab,
			strings.NewReplacer("_", "", "-", "").Replace(strings.ToLower(b)),
			strings.NewReplacer("_", "", "-", "").Replace(kebab),
		)
	}
	return out
}

func camelToKebab(s string) string {
	var b strings.Builder
	rs := []rune(s)
	for i, r := range rs {
		if r == '_' {
			b.WriteByte('-')
			continue
		}
		if i > 0 && unicode.IsUpper(r) {
			prev := rs[i-1]
			nextLower := i+1 < len(rs) && unicode.IsLower(rs[i+1])
			if unicode.IsLower(prev) || unicode.IsDigit(prev) || (unicode.IsUpper(prev) && nextLower) {
				b.WriteByte('-')
			}
		}
		b.WriteRune(unicode.ToLower(r))
	}
	return b.String()
}

type autoAdapter struct {
	arch string
	feat layout
}

func (a autoAdapter) ID() string { return a.arch }

func (a autoAdapter) Match(cfg map[string]any) bool {
	feat := detectLayout(cfg, nil)
	arch, err := inferArch(cfg, feat)
	return err == nil && requireConvertible(arch, feat) == nil && arch != "muse-glimmer"
}

func (a autoAdapter) Convert(dir string, tensors []TensorRef, cfg map[string]any, w *Writer) (*ConvertStats, error) {
	feat := detectLayout(cfg, tensorNames(tensors))
	arch, err := inferArch(cfg, feat)
	if err != nil {
		return nil, err
	}
	if err := requireConvertible(arch, feat); err != nil {
		return nil, err
	}
	if arch == "muse-glimmer" {
		return glimmerAdapter{}.Convert(dir, tensors, cfg, w)
	}
	return convertFamily(familyFor(arch, feat), dir, tensors, cfg, w)
}

func resolveAdapter(cfg map[string]any, names []string) (Adapter, error) {
	feat := detectLayout(cfg, names)
	arch, err := inferArch(cfg, feat)
	if err != nil {
		return nil, err
	}
	if err := requireConvertible(arch, feat); err != nil {
		return nil, err
	}
	if arch == "muse-glimmer" {
		return glimmerAdapter{}, nil
	}
	return autoAdapter{arch: arch, feat: feat}, nil
}
