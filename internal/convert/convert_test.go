package convert

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/gguf"
)

func TestMapGlimmerName(t *testing.T) {
	cases := []struct {
		hf           string
		want         string
		skip, vision bool
	}{
		{"model.embed_tokens.weight", "token_embd.weight", false, false},
		{"model.language_model.embed_tokens.weight", "token_embd.weight", false, false},
		{"lm_head.weight", "output.weight", false, false},
		{"model.norm.weight", "output_norm.weight", false, false},
		{"model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight", false, false},
		{"model.language_model.layers.3.mlp.down_proj.weight", "blk.3.ffn_down.weight", false, false},
		{"model.layers.2.self_attn.gate_proj.weight", "blk.2.attn_gate.weight", false, false},
		{"model.layers.1.input_layernorm.weight", "blk.1.attn_norm.weight", false, false},
		{"model.layers.1.post_attention_layernorm.weight", "blk.1.post_attention_norm.weight", false, false},
		{"model.layers.1.pre_feedforward_layernorm.weight", "blk.1.ffn_norm.weight", false, false},
		{"model.layers.1.post_feedforward_layernorm.weight", "blk.1.post_ffw_norm.weight", false, false},
		{"model.vision_tower.patch_embedder.weight", "", true, true},
		{"model.layers.0.self_attn.q_norm.weight", "", true, false},
	}
	for _, c := range cases {
		got, skip, vis := MapGlimmerName(c.hf)
		if got != c.want || skip != c.skip || vis != c.vision {
			t.Errorf("MapGlimmerName(%q) = %q skip=%v vis=%v, want %q skip=%v vis=%v",
				c.hf, got, skip, vis, c.want, c.skip, c.vision)
		}
	}
}

func TestNormalizeRepoID(t *testing.T) {
	got, err := NormalizeRepoID("https://huggingface.co/Blackfrost-AI/Muse-Glimmer-30B-Abliterated-BF16")
	if err != nil || got != "Blackfrost-AI/Muse-Glimmer-30B-Abliterated-BF16" {
		t.Fatalf("got %q err %v", got, err)
	}
	if _, err := NormalizeRepoID("nopath"); err == nil {
		t.Fatal("expected error")
	}
}

func TestEvaluateProbe(t *testing.T) {
	cfg := map[string]any{
		"architectures": []any{"MuseGlimmerForConditionalGeneration"},
		"model_type":    "muse_glimmer",
		"hidden_size":   float64(8),
	}
	files := []NeededFile{
		{Path: "config.json", Size: 100},
		{Path: "model-00001-of-00002.safetensors", Size: 1000},
		{Path: "tokenizer.json", Size: 50},
	}
	ok := Evaluate(ProbeInput{
		RepoID:  "Blackfrost-AI/Muse-Glimmer-30B-Abliterated-BF16",
		Files:   files,
		DTypes:  map[string]int64{"BF16": 100},
		Config:  cfg,
		HasJSON: true,
	})
	if !ok.Compatible {
		t.Fatalf("expected compatible: %s", ok.Reason)
	}
	if ok.WeightDType != "BF16" {
		t.Fatalf("dtype %s", ok.WeightDType)
	}

	nv := Evaluate(ProbeInput{
		RepoID:  "org/model-NVFP4",
		Files:   files,
		DTypes:  map[string]int64{"NVFP4": 100},
		Config:  cfg,
		HasJSON: true,
	})
	if nv.Compatible {
		t.Fatal("NVFP4 must fail")
	}

	missing := Evaluate(ProbeInput{RepoID: "a/b", Files: files, HasJSON: false})
	if missing.Compatible || missing.Reason == "" {
		t.Fatalf("missing config: %+v", missing)
	}

	unknown := Evaluate(ProbeInput{
		RepoID:  "a/deepseek-bf16",
		Files:   files,
		DTypes:  map[string]int64{"BF16": 1},
		Config:  map[string]any{"architectures": []any{"DeepseekV3ForCausalLM"}, "model_type": "deepseek_v3", "kv_lora_rank": float64(512)},
		HasJSON: true,
	})
	if unknown.Compatible {
		t.Fatal("unknown arch must fail")
	}

	llama := Evaluate(ProbeInput{
		RepoID:  "meta-llama/Llama-3.1-8B",
		Files:   files,
		DTypes:  map[string]int64{"BF16": 1},
		Config:  map[string]any{"architectures": []any{"LlamaForCausalLM"}, "model_type": "llama"},
		HasJSON: true,
	})
	if !llama.Compatible {
		t.Fatalf("llama should convert: %s", llama.Reason)
	}
	if llama.Architecture != "llama" {
		t.Fatalf("llama architecture %s", llama.Architecture)
	}

	qwen := Evaluate(ProbeInput{
		RepoID:  "Qwen/Qwen3.5-0.8B",
		Files:   files,
		DTypes:  map[string]int64{"BF16": 1},
		Config:  map[string]any{"architectures": []any{"Qwen3_5ForConditionalGeneration"}, "model_type": "qwen3_5"},
		HasJSON: true,
	})
	if !qwen.Compatible {
		t.Fatalf("qwen3.5 should convert: %s", qwen.Reason)
	}
	if qwen.Architecture != "qwen35" {
		t.Fatalf("qwen3.5 architecture %s want qwen35", qwen.Architecture)
	}

	mistral := Evaluate(ProbeInput{
		RepoID: "mistralai/Mistral-7B-v0.1",
		Files:  files,
		DTypes: map[string]int64{"BF16": 1},
		Config: map[string]any{
			"architectures":       []any{"MistralForCausalLM"},
			"model_type":          "mistral",
			"hidden_size":         float64(8),
			"num_attention_heads": float64(2),
			"num_hidden_layers":   float64(1),
		},
		HasJSON: true,
	})
	if !mistral.Compatible {
		t.Fatalf("mistral should map to llama: %s", mistral.Reason)
	}
	if mistral.Architecture != "llama" {
		t.Fatalf("mistral architecture %s want llama", mistral.Architecture)
	}

	apertus := Evaluate(ProbeInput{
		RepoID:  "swiss-ai/Apertus-8B",
		Files:   files,
		DTypes:  map[string]int64{"BF16": 1},
		Config:  map[string]any{"architectures": []any{"ApertusForCausalLM"}, "model_type": "apertus"},
		HasJSON: true,
	})
	if !apertus.Compatible {
		t.Fatalf("apertus is a llama.cpp arch: %s", apertus.Reason)
	}
	if apertus.Architecture != "apertus" {
		t.Fatalf("apertus architecture %s", apertus.Architecture)
	}

	next := Evaluate(ProbeInput{
		RepoID:  "Qwen/Qwen3-Next",
		Files:   files,
		DTypes:  map[string]int64{"BF16": 1},
		Config:  map[string]any{"architectures": []any{"Qwen3NextForCausalLM"}, "model_type": "qwen3_next"},
		HasJSON: true,
	})
	if next.Compatible {
		t.Fatal("qwen3next packed linear-attn must fail")
	}
}

func TestSafetensorsRoundTripAndGGUFTranspose(t *testing.T) {
	dir := t.TempDir()
	// HF [out=2, in=3] BF16. GGUF shape should be reversed [3, 2]; bytes stay row-major.
	shape := []int64{2, 3}
	data := []byte{
		0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, // 1, 2, 3 as BF16 (approx)
		0x00, 0x44, 0x00, 0x45, 0x00, 0x46,
	}
	path := filepath.Join(dir, "model.safetensors")
	if err := writeSafetensors(path, map[string]stTensor{
		"linear.weight": {DType: "BF16", Shape: shape, Data: data},
	}); err != nil {
		t.Fatal(err)
	}
	refs, err := IndexDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) != 1 || refs[0].Name != "linear.weight" {
		t.Fatalf("refs %+v", refs)
	}
	got, err := ReadPayload(refs[0])
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(data) {
		t.Fatalf("payload mismatch")
	}

	out := filepath.Join(dir, "t.gguf")
	w, err := NewWriter(out)
	if err != nil {
		t.Fatal(err)
	}
	w.AddKV("general.architecture", "test")
	w.AddKV("general.file_type", uint32(32))
	if err := w.PlanTensor("blk.0.attn_q.weight", ggufDims(shape), GGMLBF16); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteHeader(); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteTensor(data); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	tensors, md, err := gguf.ListTensors(out)
	if err != nil {
		t.Fatal(err)
	}
	if md.Architecture != "test" {
		t.Fatalf("arch %s", md.Architecture)
	}
	if len(tensors) != 1 {
		t.Fatalf("tensors %d", len(tensors))
	}
	if tensors[0].Name != "blk.0.attn_q.weight" {
		t.Fatalf("name %s", tensors[0].Name)
	}
	if len(tensors[0].Shape) != 2 || tensors[0].Shape[0] != 3 || tensors[0].Shape[1] != 2 {
		t.Fatalf("shape %v want [3 2] (GGUF reverse of HF [2 3])", tensors[0].Shape)
	}
}

func TestConvertTinyGlimmer(t *testing.T) {
	dir := t.TempDir()
	if err := WriteTinyGlimmerSnapshot(dir); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(dir, "out.gguf")
	stats, err := ConvertDir(dir, dest, ConvertOptions{Name: "tiny-glimmer"})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Tensors < 10 {
		t.Fatalf("too few tensors: %d", stats.Tensors)
	}
	md, err := gguf.ParseFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if md.Architecture != "muse-glimmer" {
		t.Fatalf("arch %s", md.Architecture)
	}
	if md.Quantization != "BF16" {
		t.Fatalf("quant %s", md.Quantization)
	}
	tensors, _, err := gguf.ListTensors(dest)
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]gguf.Tensor{}
	for _, tn := range tensors {
		names[tn.Name] = tn
	}
	for _, want := range []string{
		"token_embd.weight", "output.weight", "output_norm.weight",
		"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
		"blk.0.attn_output.weight", "blk.0.attn_gate.weight",
		"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
		"blk.0.attn_norm.weight", "blk.0.post_attention_norm.weight",
		"blk.0.ffn_norm.weight", "blk.0.post_ffw_norm.weight",
		"blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight",
	} {
		if _, ok := names[want]; !ok {
			t.Errorf("missing %s", want)
		}
	}
	// HF q_proj [8, 8] → GGUF [8, 8]; k_proj [4, 8] → [8, 4]
	if k := names["blk.0.attn_k.weight"]; len(k.Shape) != 2 || k.Shape[0] != 8 || k.Shape[1] != 4 {
		t.Errorf("attn_k shape %v", k.Shape)
	}

	if v, ok := md.Raw["muse-glimmer.rope.freq_base"]; !ok {
		t.Fatal("missing muse-glimmer.rope.freq_base")
	} else if f, ok := asTestFloat(v); !ok || f != 500000 {
		t.Fatalf("rope.freq_base=%v want 500000 (not vision 10000)", v)
	}
	if v, ok := md.Raw["tokenizer.ggml.pre"]; !ok || v != "llama4" {
		t.Fatalf("tokenizer.pre=%v want llama4", md.Raw["tokenizer.ggml.pre"])
	}
	if v, ok := md.Raw["tokenizer.ggml.eot_token_id"]; !ok {
		t.Fatal("missing eot_token_id")
	} else if n, ok := asTestUint(v); !ok || n != 4 {
		t.Fatalf("eot_token_id=%v want 4 (<|eot|>)", v)
	}
	if v, ok := md.Raw["tokenizer.ggml.eos_token_id"]; !ok {
		t.Fatal("missing eos_token_id")
	} else if n, ok := asTestUint(v); !ok || n != 3 {
		t.Fatalf("eos_token_id=%v want 3 (<|end_of_text|>)", v)
	}
	eogs, ok := md.Raw["tokenizer.ggml.eos_token_ids"].([]uint32)
	if !ok {
		t.Fatalf("eos_token_ids type %T", md.Raw["tokenizer.ggml.eos_token_ids"])
	}
	if len(eogs) != 2 || eogs[0] != 3 || eogs[1] != 4 {
		t.Fatalf("eos_token_ids=%v want [end_of_text=3, eot=4] without eom", eogs)
	}
	for _, id := range eogs {
		if id == 7 {
			t.Fatal("eog must not include <|eom|>")
		}
	}
}

func asTestFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float32:
		return float64(n), true
	case float64:
		return n, true
	default:
		return 0, false
	}
}

func asTestUint(v any) (uint32, bool) {
	switch n := v.(type) {
	case uint32:
		return n, true
	case uint64:
		return uint32(n), true
	case int32:
		if n >= 0 {
			return uint32(n), true
		}
	}
	return 0, false
}

func TestGlimmerRopeThetaIgnoresVision(t *testing.T) {
	raw := []byte(`{
		"architectures": ["MuseGlimmerForConditionalGeneration"],
		"model_type": "muse_glimmer",
		"text_config": {
			"hidden_size": 8,
			"rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
			"layer_rope_theta": [500000.0, 500000.0, 500000.0, 0]
		},
		"vision_config": {
			"rope_parameters": {"rope_theta": 10000.0}
		}
	}`)
	cfg, err := ParseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	if cfgFloat(cfg, "rope_theta") != 0 {
		t.Fatalf("top-level rope_theta leaked: %v", cfg["rope_theta"])
	}
	got := glimmerRopeTheta(cfg)
	if got != 500000 {
		t.Fatalf("glimmerRopeTheta=%v want 500000", got)
	}
	// Missing rope_parameters still must not fall back to llama.cpp's 10000.
	if glimmerRopeTheta(map[string]any{}) != 500000 {
		t.Fatal("empty cfg should default to 500000, not 10000")
	}
}

func TestLoadTokenizerHarmonySpecials(t *testing.T) {
	dir := t.TempDir()
	tok := map[string]any{
		"model": map[string]any{
			"type": "BPE",
			"vocab": map[string]int{
				"hi":                0,
				"<|begin_of_text|>": 1,
				"<|end_of_text|>":   2,
				"<|eot|>":           3,
				"<|start|>":         4,
				"<|message|>":       5,
				"<|channel|>":       6, // added, special=false, still CONTROL (looks special)
				"<|eom|>":           7,
				"userword":          8, // added, not special → USER_DEFINED
			},
			"merges": []any{},
		},
		"added_tokens": []any{
			map[string]any{"id": 1, "content": "<|begin_of_text|>", "special": true},
			map[string]any{"id": 2, "content": "<|end_of_text|>", "special": true},
			map[string]any{"id": 3, "content": "<|eot|>", "special": true},
			map[string]any{"id": 4, "content": "<|start|>", "special": true},
			map[string]any{"id": 5, "content": "<|message|>", "special": true},
			map[string]any{"id": 6, "content": "<|channel|>", "special": false},
			map[string]any{"id": 7, "content": "<|eom|>", "special": true},
			map[string]any{"id": 8, "content": "userword", "special": false},
		},
	}
	b, _ := json.Marshal(tok)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	tc := map[string]any{
		"bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>",
		"add_bos_token": true, "chat_template": "<|start|>user<|message|>x<|eot|>",
	}
	tcb, _ := json.Marshal(tc)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), tcb, 0o644); err != nil {
		t.Fatal(err)
	}
	gc := map[string]any{"eos_token_id": []any{2, 3}}
	gcb, _ := json.Marshal(gc)
	if err := os.WriteFile(filepath.Join(dir, "generation_config.json"), gcb, 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := loadTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got.Eos != 2 || got.Eot != 3 || got.Bos != 1 {
		t.Fatalf("bos=%d eos=%d eot=%d", got.Bos, got.Eos, got.Eot)
	}
	if len(got.Eog) != 2 || got.Eog[0] != 2 || got.Eog[1] != 3 {
		t.Fatalf("eog=%v want eos+eot", got.Eog)
	}
	for _, id := range got.Eog {
		if id == 7 {
			t.Fatal("eog must not include <|eom|>")
		}
	}
	wantType := map[int]int32{
		0: tokenTypeNormal,
		1: tokenTypeControl, 2: tokenTypeControl, 3: tokenTypeControl,
		4: tokenTypeControl, 5: tokenTypeControl, 6: tokenTypeControl, 7: tokenTypeControl,
		8: tokenTypeUserDefined,
	}
	for id, want := range wantType {
		if got.TokenType[id] != want {
			t.Errorf("token %d %q type=%d want %d", id, got.Tokens[id], got.TokenType[id], want)
		}
	}
}

func TestGlimmerRMSNormPlusOneAndQK(t *testing.T) {
	dir := t.TempDir()
	ones := make([]byte, 8*2)
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint16(ones[i*2:], 0x3f80) // BF16 1.0
	}
	path := filepath.Join(dir, "model.safetensors")
	if err := writeSafetensors(path, map[string]stTensor{
		"n.weight": {DType: "BF16", Shape: []int64{8}, Data: ones},
	}); err != nil {
		t.Fatal(err)
	}
	refs, err := IndexDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	plus, err := glimmerPayload(glimmerWork{
		GGUF: "blk.0.attn_norm.weight", Shape: []int64{8}, DType: GGMLF32, Src: &refs[0], Kind: "rms_plus",
	}, "BF16", 2, 0.5, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(plus) != 32 {
		t.Fatalf("rms_plus bytes %d", len(plus))
	}
	for i := 0; i < 8; i++ {
		f := math.Float32frombits(binary.LittleEndian.Uint32(plus[i*4:]))
		if f != 2 {
			t.Fatalf("layernorm +1: [%d]=%v want 2", i, f)
		}
	}
	plain, err := glimmerPayload(glimmerWork{
		GGUF: "output_norm.weight", Shape: []int64{8}, DType: GGMLF32, Src: &refs[0], Kind: "rms",
	}, "BF16", 2, 0.5, 4)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		f := math.Float32frombits(binary.LittleEndian.Uint32(plain[i*4:]))
		if f != 1 {
			t.Fatalf("output_norm no +1: [%d]=%v want 1", i, f)
		}
	}
	q, err := glimmerPayload(glimmerWork{GGUF: "blk.0.attn_q_norm.weight", Kind: "qk_q"}, "BF16", 2, 0.5, 4)
	if err != nil {
		t.Fatal(err)
	}
	k, err := glimmerPayload(glimmerWork{GGUF: "blk.0.attn_k_norm.weight", Kind: "qk_k"}, "BF16", 2, 0.5, 4)
	if err != nil {
		t.Fatal(err)
	}
	if math.Float32frombits(binary.LittleEndian.Uint32(q[:4])) != 0.5 {
		t.Fatalf("q_norm want qk_scale_factor")
	}
	if math.Float32frombits(binary.LittleEndian.Uint32(k[:4])) != 1 {
		t.Fatalf("k_norm want ones")
	}
}

func TestUnpermuteHFInterleavesPairs(t *testing.T) {
	// HF [out=8, in=2], 2 heads → headHalf=2. Bytes are row-major u8 for simplicity (elem=1).
	// Rows: head0-part0, head0-part1, head1-part0, head1-part1, each part 2 rows.
	src := []byte{
		0, 1, // h0 p0 d0
		2, 3, // h0 p0 d1
		4, 5, // h0 p1 d0
		6, 7, // h0 p1 d1
		8, 9, // h1 p0 d0
		10, 11, // h1 p0 d1
		12, 13, // h1 p1 d0
		14, 15, // h1 p1 d1
	}
	got, err := unpermuteHF(src, 2, 8, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	// dst row = (h*headHalf+d)*2+p  → interleave parts inside each head.
	want := []byte{
		0, 1, // h0 d0 p0
		4, 5, // h0 d0 p1
		2, 3, // h0 d1 p0
		6, 7, // h0 d1 p1
		8, 9, // h1 d0 p0
		12, 13, // h1 d0 p1
		10, 11, // h1 d1 p0
		14, 15, // h1 d1 p1
	}
	if string(got) != string(want) {
		t.Fatalf("unpermute\n got %v\nwant %v", got, want)
	}
}

func TestMapQwen35Name(t *testing.T) {
	cases := []struct {
		hf, want     string
		skip, vision bool
	}{
		{"model.embed_tokens.weight", "token_embd.weight", false, false},
		{"lm_head.weight", "output.weight", false, false},
		{"model.norm.weight", "output_norm.weight", false, false},
		{"model.layers.0.linear_attn.in_proj_qkv.weight", "blk.0.attn_qkv.weight", false, false},
		{"model.layers.0.linear_attn.A_log", "blk.0.ssm_a", false, false},
		{"model.layers.0.linear_attn.dt_bias", "blk.0.ssm_dt.bias", false, false},
		{"model.layers.0.linear_attn.conv1d.weight", "blk.0.ssm_conv1d.weight", false, false},
		{"model.layers.0.linear_attn.norm.weight", "blk.0.ssm_norm.weight", false, false},
		{"model.layers.1.self_attn.q_proj.weight", "blk.1.attn_q.weight", false, false},
		{"model.layers.1.self_attn.q_norm.weight", "blk.1.attn_q_norm.weight", false, false},
		{"model.layers.1.post_attention_layernorm.weight", "blk.1.post_attention_norm.weight", false, false},
		{"model.visual.patch_embed.weight", "", true, true},
		{"mtp.layers.0.mlp.gate_proj.weight", "", true, false},
	}
	for _, c := range cases {
		got := familyFor("qwen35", layout{LinearAttn: true, QKNorm: true}).MapName(c.hf)
		if got.GGUF != c.want || got.Skip != c.skip || got.Vision != c.vision {
			t.Errorf("MapName(%q) = %q skip=%v vis=%v, want %q skip=%v vis=%v",
				c.hf, got.GGUF, got.Skip, got.Vision, c.want, c.skip, c.vision)
		}
	}
}

func TestVHeadReorderGroupedToTiled(t *testing.T) {
	// nK=2, nVPerK=2, headDim=1: grouped [G0v0, G0v1, G1v0, G1v1] → tiled [G0v0, G1v0, G0v1, G1v1]
	src := []byte{10, 11, 12, 13}
	got, err := permuteDim0(src, 4, 1, vHeadPerm(2, 2, 1))
	if err != nil {
		t.Fatal(err)
	}
	want := []byte{10, 12, 11, 13}
	if string(got) != string(want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestConvertTinyLlama(t *testing.T) {
	dir := t.TempDir()
	if err := WriteTinyLlamaSnapshot(dir); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(dir, "out.gguf")
	stats, err := ConvertDir(dir, dest, ConvertOptions{Name: "tiny-llama"})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Architecture != "llama" {
		t.Fatalf("arch %s", stats.Architecture)
	}
	md, err := gguf.ParseFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if md.Architecture != "llama" {
		t.Fatalf("arch %s", md.Architecture)
	}
	if v, ok := md.Raw["tokenizer.ggml.pre"]; !ok || v != "llama-bpe" {
		t.Fatalf("tokenizer.pre=%v want llama-bpe", md.Raw["tokenizer.ggml.pre"])
	}
	if v, ok := md.Raw["llama.vocab_size"]; !ok {
		t.Fatal("missing llama.vocab_size")
	} else if n, ok := asTestUint(v); !ok || n != 8 {
		t.Fatalf("vocab_size=%v want 8", v)
	}
	tensors, _, err := gguf.ListTensors(dest)
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{}
	for _, tn := range tensors {
		names[tn.Name] = true
	}
	for _, want := range []string{
		"token_embd.weight", "output.weight", "output_norm.weight",
		"blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_norm.weight",
	} {
		if !names[want] {
			t.Errorf("missing %s", want)
		}
	}
}

func TestPadLeadingDim(t *testing.T) {
	src := []byte{1, 2, 3, 4}
	got, err := padLeadingDim(src, []int64{2, 2}, []int64{4, 2}, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []byte{1, 2, 3, 4, 0, 0, 0, 0}
	if string(got) != string(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	same, err := padLeadingDim(src, []int64{2, 2}, []int64{2, 2}, 1)
	if err != nil || string(same) != string(src) {
		t.Fatalf("no-op pad: %v %v", same, err)
	}
}

func TestAlignVocab(t *testing.T) {
	tok := &ggmlTokenizer{Tokens: []string{"a", "b", "c", "d"}, TokenType: []int32{1, 1, 1, 1}}
	tensors := []TensorRef{
		{Name: "model.embed_tokens.weight", Shape: []int64{8, 4}},
		{Name: "lm_head.weight", Shape: []int64{6, 4}},
	}
	n := alignVocab(tok, tensors, 4)
	if n != 8 {
		t.Fatalf("vocab %d want 8", n)
	}
	if len(tok.Tokens) != 8 {
		t.Fatalf("tokens %d", len(tok.Tokens))
	}
	if tok.Tokens[4] != "[PAD4]" || tok.TokenType[4] != tokenTypeUnused {
		t.Fatalf("pad token %q type %d", tok.Tokens[4], tok.TokenType[4])
	}
}

func TestConvertPadsTokenizerToEmbed(t *testing.T) {
	dir := t.TempDir()
	cfg := map[string]any{
		"architectures":           []any{"LlamaForCausalLM"},
		"model_type":              "llama",
		"hidden_size":             8,
		"intermediate_size":       16,
		"num_hidden_layers":       1,
		"num_attention_heads":     2,
		"num_key_value_heads":     1,
		"head_dim":                4,
		"vocab_size":              4,
		"max_position_embeddings": 32,
		"rms_norm_eps":            1e-5,
		"rope_theta":              500000.0,
		"tie_word_embeddings":     true,
	}
	if err := writeJSON(dir, "config.json", cfg); err != nil {
		t.Fatal(err)
	}
	if err := writeTinyBPETokenizer(dir, map[string]int{
		"a": 0, "b": 1,
		"<|begin_of_text|>": 2,
		"<|eot_id|>":        3,
	}, []any{
		map[string]any{"id": 2, "content": "<|begin_of_text|>", "special": true},
		map[string]any{"id": 3, "content": "<|eot_id|>", "special": true},
	}, "<|begin_of_text|>", "<|eot_id|>", "{{ bos_token }}{{ message }}"); err != nil {
		t.Fatal(err)
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
	if err := writeSafetensors(filepath.Join(dir, "model.safetensors"), tensors); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(dir, "out.gguf")
	if _, err := ConvertDir(dir, dest, ConvertOptions{Name: "pad-vocab"}); err != nil {
		t.Fatal(err)
	}
	md, err := gguf.ParseFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := md.Raw["llama.vocab_size"]; !ok {
		t.Fatal("missing vocab_size")
	} else if n, ok := asTestUint(v); !ok || n != 8 {
		t.Fatalf("vocab_size=%v want 8 (embed rows, not config 4)", v)
	}
	toks, ok := md.Raw["tokenizer.ggml.tokens"].([]string)
	if !ok || len(toks) != 8 {
		t.Fatalf("tokenizer tokens %d (%T)", len(toks), md.Raw["tokenizer.ggml.tokens"])
	}
	listed, _, err := gguf.ListTensors(dest)
	if err != nil {
		t.Fatal(err)
	}
	for _, tn := range listed {
		if tn.Name == "token_embd.weight" {
			if len(tn.Shape) != 2 || tn.Shape[0] != 8 || tn.Shape[1] != 8 {
				t.Fatalf("token_embd shape %v want [8 8] (GGUF reverse of padded HF [8 8])", tn.Shape)
			}
		}
	}
	if err := gguf.CheckVocabLayout(dest); err != nil {
		t.Fatalf("padded convert still fails vocab layout: %v", err)
	}
}

func TestConvertTinyQwen35(t *testing.T) {
	dir := t.TempDir()
	if err := WriteTinyQwen35Snapshot(dir); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(dir, "out.gguf")
	stats, err := ConvertDir(dir, dest, ConvertOptions{Name: "tiny-qwen35"})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Architecture != "qwen35" {
		t.Fatalf("arch %s", stats.Architecture)
	}
	if stats.Skipped == 0 {
		t.Fatal("expected vision tensors skipped")
	}
	md, err := gguf.ParseFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if md.Architecture != "qwen35" {
		t.Fatalf("arch %s", md.Architecture)
	}
	if md.FullAttentionInterval != 2 {
		t.Fatalf("full_attention_interval %d", md.FullAttentionInterval)
	}
	if v, ok := md.Raw["tokenizer.ggml.pre"]; !ok || v != "qwen2" {
		t.Fatalf("tokenizer.pre=%v want qwen2", md.Raw["tokenizer.ggml.pre"])
	}
	if v, ok := md.Raw["qwen35.ssm.state_size"]; !ok {
		t.Fatal("missing ssm.state_size")
	} else if n, ok := asTestUint(v); !ok || n != 2 {
		t.Fatalf("ssm.state_size=%v", v)
	}
	if v, ok := md.Raw["qwen35.rope.dimension_count"]; !ok {
		t.Fatal("missing rope.dimension_count")
	} else if n, ok := asTestUint(v); !ok || n != 1 {
		t.Fatalf("rope.dimension_count=%v want 4*0.25=1", v)
	}
	sec, ok := md.Raw["qwen35.rope.dimension_sections"].([]uint32)
	if !ok || len(sec) != 4 || sec[0] != 11 || sec[3] != 0 {
		t.Fatalf("dimension_sections=%v (%T)", md.Raw["qwen35.rope.dimension_sections"], md.Raw["qwen35.rope.dimension_sections"])
	}
	tensors, _, err := gguf.ListTensors(dest)
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]gguf.Tensor{}
	for _, tn := range tensors {
		names[tn.Name] = tn
	}
	for _, want := range []string{
		"token_embd.weight", "output.weight", "output_norm.weight",
		"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
		"blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_conv1d.weight",
		"blk.0.ssm_norm.weight", "blk.0.ssm_out.weight",
		"blk.0.post_attention_norm.weight",
		"blk.1.attn_q.weight", "blk.1.attn_q_norm.weight",
		"blk.1.post_attention_norm.weight",
	} {
		if _, ok := names[want]; !ok {
			t.Errorf("missing %s", want)
		}
	}
	if _, ok := names["blk.0.attn_q.weight"]; ok {
		t.Error("linear layer should not have attn_q")
	}
	// llama.cpp's SSM conv kernels require F32 weights (ggml asserts
	// src1->nb[0] == sizeof(float)); bf16 aborts or garbles.
	if names["blk.0.ssm_conv1d.weight"].TypeName != "f32" {
		t.Fatalf("ssm_conv1d must be f32, got %s", names["blk.0.ssm_conv1d.weight"].TypeName)
	}
}

func TestConvertTinyQwen3Moe(t *testing.T) {
	dir := t.TempDir()
	if err := WriteTinyQwen3MoeSnapshot(dir); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(dir, "out.gguf")
	stats, err := ConvertDir(dir, dest, ConvertOptions{Name: "tiny-qwen3moe"})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Architecture != "qwen3moe" {
		t.Fatalf("arch %s", stats.Architecture)
	}
	md, err := gguf.ParseFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := md.Raw["qwen3moe.expert_count"]; !ok {
		t.Fatal("missing expert_count")
	} else if n, ok := asTestUint(v); !ok || n != 2 {
		t.Fatalf("expert_count=%v", v)
	}
	tensors, _, err := gguf.ListTensors(dest)
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]gguf.Tensor{}
	for _, tn := range tensors {
		names[tn.Name] = tn
	}
	for _, want := range []string{
		"blk.0.ffn_gate_inp.weight",
		"blk.0.ffn_gate_exps.weight",
		"blk.0.ffn_up_exps.weight",
		"blk.0.ffn_down_exps.weight",
		"blk.0.attn_q_norm.weight",
	} {
		if _, ok := names[want]; !ok {
			t.Errorf("missing %s", want)
		}
	}
	if g := names["blk.0.ffn_gate_exps.weight"]; len(g.Shape) != 3 || g.Shape[0] != 8 || g.Shape[1] != 16 || g.Shape[2] != 2 {
		t.Fatalf("ffn_gate_exps shape %v want [8 16 2] (GGUF reverse of [2 16 8])", g.Shape)
	}
}

func TestNegExpALog(t *testing.T) {
	src := f32Bytes([]float32{0, 1})
	got := negExpF32(src)
	z := math.Float32frombits(binary.LittleEndian.Uint32(got[:4]))
	one := math.Float32frombits(binary.LittleEndian.Uint32(got[4:]))
	if z != -1 {
		t.Fatalf("-exp(0)=%v want -1", z)
	}
	if math.Abs(float64(one)-(-math.E)) > 1e-5 {
		t.Fatalf("-exp(1)=%v want %v", one, -math.E)
	}
}

func TestInferArch(t *testing.T) {
	cases := []struct {
		cfg     map[string]any
		names   []string
		want    string
		wantErr string
	}{
		{
			cfg:  map[string]any{"model_type": "llama", "architectures": []any{"LlamaForCausalLM"}},
			want: "llama",
		},
		{
			cfg:  map[string]any{"model_type": "qwen3_5", "architectures": []any{"Qwen3_5ForConditionalGeneration"}},
			want: "qwen35",
		},
		{
			cfg:  map[string]any{"model_type": "qwen3_moe", "architectures": []any{"Qwen3MoeForCausalLM"}, "num_experts": float64(2)},
			want: "qwen3moe",
		},
		{
			cfg: map[string]any{
				"model_type": "mistral", "architectures": []any{"MistralForCausalLM"},
				"hidden_size": float64(8), "num_attention_heads": float64(2), "num_hidden_layers": float64(1),
			},
			want: "llama",
		},
		{
			cfg: map[string]any{
				"model_type": "mixtral", "architectures": []any{"MixtralForCausalLM"},
				"hidden_size": float64(8), "num_attention_heads": float64(2), "num_hidden_layers": float64(1),
				"num_local_experts": float64(8),
			},
			want: "llama",
		},
		{
			cfg:  map[string]any{"model_type": "apertus", "architectures": []any{"ApertusForCausalLM"}},
			want: "apertus",
		},
		{
			cfg:  map[string]any{"model_type": "smollm3", "architectures": []any{"SmolLM3ForCausalLM"}},
			want: "smollm3",
		},
		{
			cfg:     map[string]any{"model_type": "deepseek_v3", "architectures": []any{"DeepseekV3ForCausalLM"}, "kv_lora_rank": float64(512)},
			wantErr: "MLA",
		},
		{
			cfg:     map[string]any{"model_type": "qwen3_next", "architectures": []any{"Qwen3NextForCausalLM"}},
			wantErr: "qwen3next",
		},
		{
			cfg:  map[string]any{"model_type": "muse_glimmer", "architectures": []any{"MuseGlimmerForConditionalGeneration"}},
			want: "muse-glimmer",
		},
	}
	for _, c := range cases {
		feat := detectLayout(c.cfg, c.names)
		arch, err := inferArch(c.cfg, feat)
		if err == nil {
			err = requireConvertible(arch, feat)
		}
		if c.wantErr != "" {
			if err == nil {
				t.Errorf("%v: want error %q, got arch %s", c.cfg["model_type"], c.wantErr, arch)
				continue
			}
			if !strings.Contains(err.Error(), c.wantErr) {
				t.Errorf("%v: error %q, want substring %q", c.cfg["model_type"], err, c.wantErr)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v: %v", c.cfg["model_type"], err)
			continue
		}
		if arch != c.want {
			t.Errorf("%v: got %s want %s", c.cfg["model_type"], arch, c.want)
		}
	}
}

func TestFamilyStoreF32Contract(t *testing.T) {
	f := familyFor("qwen35", layout{LinearAttn: true, QKNorm: true})
	cases := []struct {
		hf       string
		hfShape  []int64
		wantKind workKind
		wantF32  bool
	}{
		{"model.layers.0.linear_attn.A_log", []int64{8}, kindNegExp, true},
		{"model.layers.0.linear_attn.dt_bias", []int64{8}, kindCopy, true},
		{"model.layers.0.linear_attn.conv1d.weight", []int64{16, 1, 4}, kindConv1d, true},
		{"model.layers.0.linear_attn.norm.weight", []int64{8}, kindRMS, true},
		{"model.layers.0.input_layernorm.weight", []int64{8}, kindRMSPlus, true},
		{"model.layers.0.linear_attn.out_proj.weight", []int64{8, 8}, kindCopy, false},
		{"model.layers.1.self_attn.q_proj.weight", []int64{8, 8}, kindCopy, false},
	}
	for _, c := range cases {
		m := f.MapName(c.hf)
		if m.Kind != c.wantKind {
			t.Errorf("%s: kind %s, want %s", c.hf, m.Kind, c.wantKind)
		}
		shape := convPlanShape(m.Kind, c.hfShape)
		dt, _ := familyStore(m.Kind, shape, GGMLF16)
		if c.wantF32 && dt != GGMLF32 {
			t.Errorf("%s: store dtype %d, want F32", c.hf, dt)
		}
		if !c.wantF32 && dt == GGMLF32 {
			t.Errorf("%s: stored F32, want weight dtype (2D copy)", c.hf)
		}
	}
}

func TestUnknownTensorFailsClosed(t *testing.T) {
	f := familyFor("llama", layout{})
	m := f.MapName("model.layers.0.not_a_real_tensor.weight")
	if m.GGUF != "" || m.Skip {
		t.Fatalf("unknown tensor must be unmapped (fail closed), got %+v", m)
	}
	moe := familyFor("qwen3moe", layout{MoE: true})
	if got := moe.MapName("model.layers.0.mlp.gate.weight"); got.GGUF != "blk.0.ffn_gate_inp.weight" {
		t.Fatalf("moe router map = %+v", got)
	}
}
