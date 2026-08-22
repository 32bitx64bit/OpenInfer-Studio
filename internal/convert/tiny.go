package convert

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
)

// WriteTinyGlimmerSnapshot writes a one-layer muse-glimmer snapshot for tests.
func WriteTinyGlimmerSnapshot(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	cfg := map[string]any{
		"architectures": []any{"MuseGlimmerForConditionalGeneration"},
		"model_type":    "muse_glimmer",
		"text_config": map[string]any{
			"hidden_size":             8,
			"intermediate_size":       16,
			"num_hidden_layers":       1,
			"num_attention_heads":     2,
			"num_key_value_heads":     1,
			"head_dim":                4,
			"vocab_size":              8,
			"max_position_embeddings": 32,
			"rms_norm_eps":            1e-5,
			"sliding_window":          8,
			"layer_types":             []any{"sliding_attention"},
			"layer_rope_theta":        []any{500000.0},
			"qk_scale_factor":         0.5,
			"final_logit_softcapping": 20,
			"output_multiplier":       0.2,
			"rope_parameters": map[string]any{
				"rope_theta": 500000.0,
				"rope_type":  "default",
			},
		},
		// Vision RoPE is 10000; convert must not pick this up as the LM base.
		"vision_config": map[string]any{
			"rope_parameters": map[string]any{
				"rope_theta": 10000.0,
				"rope_type":  "default",
			},
		},
	}
	b, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), b, 0o644); err != nil {
		return err
	}
	tok := map[string]any{
		"model": map[string]any{
			"type": "BPE",
			"vocab": map[string]int{
				"a": 0, "b": 1,
				"<|begin_of_text|>": 2,
				"<|end_of_text|>":   3,
				"<|eot|>":           4,
				"<|start|>":         5,
				"<|message|>":       6,
				"<|eom|>":           7,
			},
			"merges": []any{},
		},
		"added_tokens": []any{
			map[string]any{"id": 2, "content": "<|begin_of_text|>", "special": true},
			map[string]any{"id": 3, "content": "<|end_of_text|>", "special": true},
			map[string]any{"id": 4, "content": "<|eot|>", "special": true},
			map[string]any{"id": 5, "content": "<|start|>", "special": true},
			map[string]any{"id": 6, "content": "<|message|>", "special": true},
			map[string]any{"id": 7, "content": "<|eom|>", "special": true},
		},
	}
	tb, _ := json.Marshal(tok)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), tb, 0o644); err != nil {
		return err
	}
	tc := map[string]any{
		"bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>",
		"add_bos_token": true, "chat_template": "<|start|>user<|message|>{{ content }}<|eot|>",
	}
	tcb, _ := json.Marshal(tc)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), tcb, 0o644); err != nil {
		return err
	}
	gc := map[string]any{
		"bos_token_id": 2,
		"eos_token_id": []any{3, 4}, // <|end_of_text|>, <|eot|> — never <|eom|>
	}
	gcb, _ := json.Marshal(gc)
	if err := os.WriteFile(filepath.Join(dir, "generation_config.json"), gcb, 0o644); err != nil {
		return err
	}
	bf16 := func(n int) []byte {
		out := make([]byte, n*2)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint16(out[i*2:], 0x3f80)
		}
		return out
	}
	tensors := map[string]stTensor{
		"model.embed_tokens.weight":                        {DType: "BF16", Shape: []int64{8, 8}, Data: bf16(64)},
		"lm_head.weight":                                   {DType: "BF16", Shape: []int64{8, 8}, Data: bf16(64)},
		"model.norm.weight":                                {DType: "BF16", Shape: []int64{8}, Data: bf16(8)},
		"model.layers.0.self_attn.q_proj.weight":           {DType: "BF16", Shape: []int64{8, 8}, Data: bf16(64)},
		"model.layers.0.self_attn.k_proj.weight":           {DType: "BF16", Shape: []int64{4, 8}, Data: bf16(32)},
		"model.layers.0.self_attn.v_proj.weight":           {DType: "BF16", Shape: []int64{4, 8}, Data: bf16(32)},
		"model.layers.0.self_attn.o_proj.weight":           {DType: "BF16", Shape: []int64{8, 8}, Data: bf16(64)},
		"model.layers.0.self_attn.gate_proj.weight":        {DType: "BF16", Shape: []int64{8, 8}, Data: bf16(64)},
		"model.layers.0.mlp.gate_proj.weight":              {DType: "BF16", Shape: []int64{16, 8}, Data: bf16(128)},
		"model.layers.0.mlp.up_proj.weight":                {DType: "BF16", Shape: []int64{16, 8}, Data: bf16(128)},
		"model.layers.0.mlp.down_proj.weight":              {DType: "BF16", Shape: []int64{8, 16}, Data: bf16(128)},
		"model.layers.0.input_layernorm.weight":            {DType: "BF16", Shape: []int64{8}, Data: bf16(8)},
		"model.layers.0.post_attention_layernorm.weight":   {DType: "BF16", Shape: []int64{8}, Data: bf16(8)},
		"model.layers.0.pre_feedforward_layernorm.weight":  {DType: "BF16", Shape: []int64{8}, Data: bf16(8)},
		"model.layers.0.post_feedforward_layernorm.weight": {DType: "BF16", Shape: []int64{8}, Data: bf16(8)},
		"model.vision_tower.dummy.weight":                  {DType: "BF16", Shape: []int64{4}, Data: bf16(4)},
	}
	return writeSafetensors(filepath.Join(dir, "model.safetensors"), tensors)
}

func bf16Ones(n int) []byte {
	out := make([]byte, n*2)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(out[i*2:], 0x3f80)
	}
	return out
}

func writeJSON(dir, name string, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, name), b, 0o644)
}

func writeTinyBPETokenizer(dir string, vocab map[string]int, added []any, bos, eos string, tmpl string) error {
	tok := map[string]any{
		"model": map[string]any{
			"type":   "BPE",
			"vocab":  vocab,
			"merges": []any{},
		},
		"added_tokens": added,
	}
	if err := writeJSON(dir, "tokenizer.json", tok); err != nil {
		return err
	}
	return writeJSON(dir, "tokenizer_config.json", map[string]any{
		"bos_token": bos, "eos_token": eos,
		"add_bos_token": true, "chat_template": tmpl,
	})
}

// WriteTinyLlamaSnapshot writes a one-layer Llama-style snapshot for tests.
func WriteTinyLlamaSnapshot(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	cfg := map[string]any{
		"architectures":           []any{"LlamaForCausalLM"},
		"model_type":              "llama",
		"hidden_size":             8,
		"intermediate_size":       16,
		"num_hidden_layers":       1,
		"num_attention_heads":     2,
		"num_key_value_heads":     1,
		"head_dim":                4,
		"vocab_size":              8,
		"max_position_embeddings": 32,
		"rms_norm_eps":            1e-5,
		"rope_theta":              500000.0,
		"tie_word_embeddings":     true,
	}
	if err := writeJSON(dir, "config.json", cfg); err != nil {
		return err
	}
	if err := writeTinyBPETokenizer(dir, map[string]int{
		"a": 0, "b": 1,
		"<|begin_of_text|>":   2,
		"<|end_of_text|>":     3,
		"<|eot_id|>":          4,
		"<|start_header_id|>": 5,
		"<|end_header_id|>":   6,
		"<|eom_id|>":          7,
	}, []any{
		map[string]any{"id": 2, "content": "<|begin_of_text|>", "special": true},
		map[string]any{"id": 3, "content": "<|end_of_text|>", "special": true},
		map[string]any{"id": 4, "content": "<|eot_id|>", "special": true},
	}, "<|begin_of_text|>", "<|end_of_text|>", "{{ bos_token }}{{ message }}"); err != nil {
		return err
	}
	tensors := map[string]stTensor{
		"model.embed_tokens.weight":                      {DType: "BF16", Shape: []int64{8, 8}, Data: bf16Ones(64)},
		"model.norm.weight":                              {DType: "BF16", Shape: []int64{8}, Data: bf16Ones(8)},
		"model.layers.0.self_attn.q_proj.weight":         {DType: "BF16", Shape: []int64{8, 8}, Data: bf16Ones(64)},
		"model.layers.0.self_attn.k_proj.weight":         {DType: "BF16", Shape: []int64{4, 8}, Data: bf16Ones(32)},
		"model.layers.0.self_attn.v_proj.weight":         {DType: "BF16", Shape: []int64{4, 8}, Data: bf16Ones(32)},
		"model.layers.0.self_attn.o_proj.weight":         {DType: "BF16", Shape: []int64{8, 8}, Data: bf16Ones(64)},
		"model.layers.0.mlp.gate_proj.weight":            {DType: "BF16", Shape: []int64{16, 8}, Data: bf16Ones(128)},
		"model.layers.0.mlp.up_proj.weight":              {DType: "BF16", Shape: []int64{16, 8}, Data: bf16Ones(128)},
		"model.layers.0.mlp.down_proj.weight":            {DType: "BF16", Shape: []int64{8, 16}, Data: bf16Ones(128)},
		"model.layers.0.input_layernorm.weight":          {DType: "BF16", Shape: []int64{8}, Data: bf16Ones(8)},
		"model.layers.0.post_attention_layernorm.weight": {DType: "BF16", Shape: []int64{8}, Data: bf16Ones(8)},
	}
	return writeSafetensors(filepath.Join(dir, "model.safetensors"), tensors)
}

// WriteTinyQwen35Snapshot writes a 2-layer hybrid Qwen3.5 snapshot (linear + full attn).
func WriteTinyQwen35Snapshot(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	cfg := map[string]any{
		"architectures": []any{"Qwen3_5ForConditionalGeneration"},
		"model_type":    "qwen3_5",
		"text_config": map[string]any{
			"hidden_size":             8,
			"intermediate_size":       16,
			"num_hidden_layers":       2,
			"num_attention_heads":     2,
			"num_key_value_heads":     1,
			"head_dim":                4,
			"vocab_size":              8,
			"max_position_embeddings": 32,
			"rms_norm_eps":            1e-6,
			"full_attention_interval": 2,
			"linear_conv_kernel_dim":  2,
			"linear_key_head_dim":     2,
			"linear_value_head_dim":   2,
			"linear_num_key_heads":    2,
			"linear_num_value_heads":  4,
			"tie_word_embeddings":     true,
			"layer_types":             []any{"linear_attention", "full_attention"},
			"rope_parameters": map[string]any{
				"rope_theta":            10000000.0,
				"rope_type":             "default",
				"partial_rotary_factor": 0.25,
				"mrope_section":         []any{11, 11, 10},
			},
		},
		"vision_config": map[string]any{"hidden_size": 4},
	}
	if err := writeJSON(dir, "config.json", cfg); err != nil {
		return err
	}
	if err := writeTinyBPETokenizer(dir, map[string]int{
		"a": 0, "b": 1,
		"<|im_start|>":     2,
		"<|im_end|>":       3,
		"<|endoftext|>":    4,
		"<|vision_start|>": 5,
		"<|vision_end|>":   6,
		"c":                7,
	}, []any{
		map[string]any{"id": 2, "content": "<|im_start|>", "special": true},
		map[string]any{"id": 3, "content": "<|im_end|>", "special": true},
	}, "<|endoftext|>", "<|im_end|>", "{% for m in messages %}{{ m.content }}{% endfor %}"); err != nil {
		return err
	}
	// linear: key_dim=4, value_dim=8, conv_dim=8+2*2*2=16, kernel=2
	tensors := map[string]stTensor{
		"model.embed_tokens.weight": {DType: "BF16", Shape: []int64{8, 8}, Data: bf16Ones(64)},
		"model.norm.weight":         {DType: "BF16", Shape: []int64{8}, Data: bf16Ones(8)},
		"model.visual.patch.weight": {DType: "BF16", Shape: []int64{4}, Data: bf16Ones(4)},

		"model.layers.0.linear_attn.in_proj_qkv.weight":  {DType: "BF16", Shape: []int64{16, 8}, Data: bf16Ones(128)},
		"model.layers.0.linear_attn.in_proj_z.weight":    {DType: "BF16", Shape: []int64{8, 8}, Data: bf16Ones(64)},
		"model.layers.0.linear_attn.in_proj_a.weight":    {DType: "BF16", Shape: []int64{4, 8}, Data: bf16Ones(32)},
		"model.layers.0.linear_attn.in_proj_b.weight":    {DType: "BF16", Shape: []int64{4, 8}, Data: bf16Ones(32)},
		"model.layers.0.linear_attn.out_proj.weight":     {DType: "BF16", Shape: []int64{8, 8}, Data: bf16Ones(64)},
		"model.layers.0.linear_attn.conv1d.weight":       {DType: "BF16", Shape: []int64{16, 1, 2}, Data: bf16Ones(32)},
		"model.layers.0.linear_attn.A_log":               {DType: "BF16", Shape: []int64{4}, Data: bf16Ones(4)},
		"model.layers.0.linear_attn.dt_bias":             {DType: "BF16", Shape: []int64{4}, Data: bf16Ones(4)},
		"model.layers.0.linear_attn.norm.weight":         {DType: "BF16", Shape: []int64{2}, Data: bf16Ones(2)},
		"model.layers.0.mlp.gate_proj.weight":            {DType: "BF16", Shape: []int64{16, 8}, Data: bf16Ones(128)},
		"model.layers.0.mlp.up_proj.weight":              {DType: "BF16", Shape: []int64{16, 8}, Data: bf16Ones(128)},
		"model.layers.0.mlp.down_proj.weight":            {DType: "BF16", Shape: []int64{8, 16}, Data: bf16Ones(128)},
		"model.layers.0.input_layernorm.weight":          {DType: "BF16", Shape: []int64{8}, Data: bf16Ones(8)},
		"model.layers.0.post_attention_layernorm.weight": {DType: "BF16", Shape: []int64{8}, Data: bf16Ones(8)},

		"model.layers.1.self_attn.q_proj.weight":         {DType: "BF16", Shape: []int64{8, 8}, Data: bf16Ones(64)},
		"model.layers.1.self_attn.k_proj.weight":         {DType: "BF16", Shape: []int64{4, 8}, Data: bf16Ones(32)},
		"model.layers.1.self_attn.v_proj.weight":         {DType: "BF16", Shape: []int64{4, 8}, Data: bf16Ones(32)},
		"model.layers.1.self_attn.o_proj.weight":         {DType: "BF16", Shape: []int64{8, 8}, Data: bf16Ones(64)},
		"model.layers.1.self_attn.q_norm.weight":         {DType: "BF16", Shape: []int64{4}, Data: bf16Ones(4)},
		"model.layers.1.self_attn.k_norm.weight":         {DType: "BF16", Shape: []int64{4}, Data: bf16Ones(4)},
		"model.layers.1.mlp.gate_proj.weight":            {DType: "BF16", Shape: []int64{16, 8}, Data: bf16Ones(128)},
		"model.layers.1.mlp.up_proj.weight":              {DType: "BF16", Shape: []int64{16, 8}, Data: bf16Ones(128)},
		"model.layers.1.mlp.down_proj.weight":            {DType: "BF16", Shape: []int64{8, 16}, Data: bf16Ones(128)},
		"model.layers.1.input_layernorm.weight":          {DType: "BF16", Shape: []int64{8}, Data: bf16Ones(8)},
		"model.layers.1.post_attention_layernorm.weight": {DType: "BF16", Shape: []int64{8}, Data: bf16Ones(8)},
	}
	return writeSafetensors(filepath.Join(dir, "model.safetensors"), tensors)
}

// WriteTinyQwen3MoeSnapshot writes a one-layer Qwen3-MoE snapshot for tests.
func WriteTinyQwen3MoeSnapshot(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	cfg := map[string]any{
		"architectures":           []any{"Qwen3MoeForCausalLM"},
		"model_type":              "qwen3_moe",
		"hidden_size":             8,
		"intermediate_size":       16,
		"moe_intermediate_size":   16,
		"num_hidden_layers":       1,
		"num_attention_heads":     2,
		"num_key_value_heads":     1,
		"head_dim":                4,
		"vocab_size":              8,
		"max_position_embeddings": 32,
		"rms_norm_eps":            1e-6,
		"num_experts":             2,
		"num_experts_per_tok":     1,
		"rope_theta":              1000000.0,
		"tie_word_embeddings":     true,
	}
	if err := writeJSON(dir, "config.json", cfg); err != nil {
		return err
	}
	if err := writeTinyBPETokenizer(dir, map[string]int{
		"a": 0, "b": 1,
		"<|im_start|>":  2,
		"<|im_end|>":    3,
		"<|endoftext|>": 4,
		"c":             5, "d": 6, "e": 7,
	}, []any{
		map[string]any{"id": 2, "content": "<|im_start|>", "special": true},
		map[string]any{"id": 3, "content": "<|im_end|>", "special": true},
	}, "<|endoftext|>", "<|im_end|>", "{% for m in messages %}{{ m.content }}{% endfor %}"); err != nil {
		return err
	}
	tensors := map[string]stTensor{
		"model.embed_tokens.weight":                      {DType: "BF16", Shape: []int64{8, 8}, Data: bf16Ones(64)},
		"model.norm.weight":                              {DType: "BF16", Shape: []int64{8}, Data: bf16Ones(8)},
		"model.layers.0.self_attn.q_proj.weight":         {DType: "BF16", Shape: []int64{8, 8}, Data: bf16Ones(64)},
		"model.layers.0.self_attn.k_proj.weight":         {DType: "BF16", Shape: []int64{4, 8}, Data: bf16Ones(32)},
		"model.layers.0.self_attn.v_proj.weight":         {DType: "BF16", Shape: []int64{4, 8}, Data: bf16Ones(32)},
		"model.layers.0.self_attn.o_proj.weight":         {DType: "BF16", Shape: []int64{8, 8}, Data: bf16Ones(64)},
		"model.layers.0.self_attn.q_norm.weight":         {DType: "BF16", Shape: []int64{4}, Data: bf16Ones(4)},
		"model.layers.0.self_attn.k_norm.weight":         {DType: "BF16", Shape: []int64{4}, Data: bf16Ones(4)},
		"model.layers.0.input_layernorm.weight":          {DType: "BF16", Shape: []int64{8}, Data: bf16Ones(8)},
		"model.layers.0.post_attention_layernorm.weight": {DType: "BF16", Shape: []int64{8}, Data: bf16Ones(8)},
		"model.layers.0.mlp.gate.weight":                 {DType: "BF16", Shape: []int64{2, 8}, Data: bf16Ones(16)},
		"model.layers.0.mlp.experts.0.gate_proj.weight":  {DType: "BF16", Shape: []int64{16, 8}, Data: bf16Ones(128)},
		"model.layers.0.mlp.experts.0.up_proj.weight":    {DType: "BF16", Shape: []int64{16, 8}, Data: bf16Ones(128)},
		"model.layers.0.mlp.experts.0.down_proj.weight":  {DType: "BF16", Shape: []int64{8, 16}, Data: bf16Ones(128)},
		"model.layers.0.mlp.experts.1.gate_proj.weight":  {DType: "BF16", Shape: []int64{16, 8}, Data: bf16Ones(128)},
		"model.layers.0.mlp.experts.1.up_proj.weight":    {DType: "BF16", Shape: []int64{16, 8}, Data: bf16Ones(128)},
		"model.layers.0.mlp.experts.1.down_proj.weight":  {DType: "BF16", Shape: []int64{8, 16}, Data: bf16Ones(128)},
	}
	return writeSafetensors(filepath.Join(dir, "model.safetensors"), tensors)
}
