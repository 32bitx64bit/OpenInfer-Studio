package convert

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/gguf"
)

// --- a small, strict GGUF reader for assertions -------------------------

type testGGUFTensor struct {
	Name   string
	Dims   []uint64 // GGUF order (ne0 first)
	Type   uint32
	Offset uint64
}

type testGGUF struct {
	KV      map[string]any
	Tensors map[string]testGGUFTensor
	Data    []byte // whole file
	DataAt  uint64
}

func readTestGGUF(t *testing.T, path string) *testGGUF {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	r := bytes.NewReader(b)
	rd := func(v any) {
		if err := binary.Read(r, binary.LittleEndian, v); err != nil {
			t.Fatalf("read: %v", err)
		}
	}
	str := func() string {
		var n uint64
		rd(&n)
		buf := make([]byte, n)
		if _, err := r.Read(buf); err != nil && n > 0 {
			t.Fatal(err)
		}
		return string(buf)
	}
	var magic, ver uint32
	var nT, nKV uint64
	rd(&magic)
	rd(&ver)
	rd(&nT)
	rd(&nKV)
	if magic != ggufMagic {
		t.Fatalf("bad magic %x", magic)
	}
	var scalar func(typ uint32) any
	scalar = func(typ uint32) any {
		switch typ {
		case ggufTypeUint8:
			var v uint8
			rd(&v)
			return v
		case ggufTypeInt32:
			var v int32
			rd(&v)
			return v
		case ggufTypeUint32:
			var v uint32
			rd(&v)
			return v
		case ggufTypeFloat32:
			var v float32
			rd(&v)
			return v
		case ggufTypeBool:
			var v uint8
			rd(&v)
			return v != 0
		case ggufTypeString:
			return str()
		case ggufTypeUint64:
			var v uint64
			rd(&v)
			return v
		}
		t.Fatalf("unsupported value type %d", typ)
		return nil
	}
	g := &testGGUF{KV: map[string]any{}, Tensors: map[string]testGGUFTensor{}, Data: b}
	for i := uint64(0); i < nKV; i++ {
		key := str()
		var typ uint32
		rd(&typ)
		if _, dup := g.KV[key]; dup {
			t.Fatalf("duplicate GGUF key %q (llama.cpp rejects these)", key)
		}
		if typ == ggufTypeArray {
			var et uint32
			var n uint64
			rd(&et)
			rd(&n)
			switch et {
			case ggufTypeString:
				out := make([]string, n)
				for j := range out {
					out[j] = str()
				}
				g.KV[key] = out
			case ggufTypeBool:
				out := make([]bool, n)
				for j := range out {
					out[j] = scalar(et).(bool)
				}
				g.KV[key] = out
			case ggufTypeInt32:
				out := make([]int32, n)
				for j := range out {
					out[j] = scalar(et).(int32)
				}
				g.KV[key] = out
			case ggufTypeUint32:
				out := make([]uint32, n)
				for j := range out {
					out[j] = scalar(et).(uint32)
				}
				g.KV[key] = out
			case ggufTypeFloat32:
				out := make([]float32, n)
				for j := range out {
					out[j] = scalar(et).(float32)
				}
				g.KV[key] = out
			default:
				t.Fatalf("%s: unsupported array element type %d", key, et)
			}
			continue
		}
		g.KV[key] = scalar(typ)
	}
	for i := uint64(0); i < nT; i++ {
		name := str()
		var nd uint32
		rd(&nd)
		dims := make([]uint64, nd)
		for j := range dims {
			rd(&dims[j])
		}
		var typ uint32
		var off uint64
		rd(&typ)
		rd(&off)
		g.Tensors[name] = testGGUFTensor{Name: name, Dims: dims, Type: typ, Offset: off}
	}
	pos := uint64(len(b)) - uint64(r.Len())
	g.DataAt = alignU64(pos, ggufAlign)
	return g
}

func (g *testGGUF) payload(t *testing.T, name string, nbytes int) []byte {
	t.Helper()
	ti, ok := g.Tensors[name]
	if !ok {
		t.Fatalf("no tensor %s", name)
	}
	at := g.DataAt + ti.Offset
	return g.Data[at : at+uint64(nbytes)]
}

func (g *testGGUF) u32(t *testing.T, key string) uint32 {
	t.Helper()
	v, ok := g.KV[key].(uint32)
	if !ok {
		t.Fatalf("%s = %v (%T), want uint32", key, g.KV[key], g.KV[key])
	}
	return v
}

func (g *testGGUF) f32(t *testing.T, key string) float32 {
	t.Helper()
	v, ok := g.KV[key].(float32)
	if !ok {
		t.Fatalf("%s = %v (%T), want float32", key, g.KV[key], g.KV[key])
	}
	return v
}

func dimsEqual(a []uint64, b ...uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// --- synthetic checkpoints ----------------------------------------------

func bf16Fill(n int, v float32) []byte {
	out := make([]byte, n*2)
	bits := uint16(math.Float32bits(v) >> 16)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(out[i*2:], bits)
	}
	return out
}

func bf16T(v float32, shape ...int64) stTensor {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return stTensor{DType: "BF16", Shape: shape, Data: bf16Fill(int(n), v)}
}

// protoPiece encodes one sentencepiece.ModelProto.pieces entry.
func protoPiece(piece string, score float32, typ int) []byte {
	var msg []byte
	msg = append(msg, 0x0a, byte(len(piece)))
	msg = append(msg, piece...)
	msg = append(msg, 0x15)
	var f [4]byte
	binary.LittleEndian.PutUint32(f[:], math.Float32bits(score))
	msg = append(msg, f[:]...)
	if typ != 1 {
		msg = append(msg, 0x18, byte(typ))
	}
	out := []byte{0x0a, byte(len(msg))}
	return append(out, msg...)
}

// tinySPMModel is a ModelProto with ten pieces plus unrelated trailing
// fields (a trainer_spec-like submessage) the parser must skip.
func tinySPMModel() []byte {
	var b []byte
	b = append(b, protoPiece("<pad>", 0, 3)...)
	b = append(b, protoPiece("<eos>", 0, 3)...)
	b = append(b, protoPiece("<bos>", 0, 3)...)
	b = append(b, protoPiece("<unk>", 0, 2)...)
	b = append(b, protoPiece("▁a", -1, 1)...)
	b = append(b, protoPiece("b", -2, 1)...)
	b = append(b, protoPiece("<0x41>", 0, 6)...)
	b = append(b, protoPiece("<start_of_turn>", 0, 4)...)
	b = append(b, protoPiece("<end_of_turn>", 0, 4)...)
	b = append(b, protoPiece("c", -3, 1)...)
	// field 2 (trainer_spec) length-delimited, field 9 varint.
	b = append(b, 0x12, 0x03, 0x01, 0x02, 0x03)
	b = append(b, 0x48, 0x01)
	return b
}

func writeGemma3nSnapshot(t *testing.T, dir string, mutate func(cfg map[string]any, tensors map[string]stTensor)) {
	t.Helper()
	const H, PL, L, FF, V, VPL = 8, 4, 3, 16, 12, 10
	text := map[string]any{
		"model_type":                  "gemma3n_text",
		"hidden_size":                 H,
		"hidden_size_per_layer_input": PL,
		"intermediate_size":           FF,
		"num_hidden_layers":           L,
		"num_attention_heads":         2,
		"num_key_value_heads":         1,
		"head_dim":                    4,
		"vocab_size":                  V,
		"vocab_size_per_layer_input":  VPL,
		"max_position_embeddings":     64,
		"rms_norm_eps":                1e-6,
		"sliding_window":              8,
		"layer_types":                 []any{"sliding_attention", "sliding_attention", "full_attention"},
		"rope_theta":                  1000000.0,
		"rope_local_base_freq":        10000.0,
		"altup_num_inputs":            4,
		"altup_active_idx":            0,
		"laurel_rank":                 2,
		"num_kv_shared_layers":        1,
		"activation_sparsity_pattern": []any{0.95, 0.5, 0.0},
		"final_logit_softcapping":     30.0,
	}
	cfg := map[string]any{
		"architectures": []any{"Gemma3nForConditionalGeneration"},
		"model_type":    "gemma3n",
		"text_config":   text,
	}
	p := "model.language_model."
	tensors := map[string]stTensor{
		p + "embed_tokens.weight":               bf16T(1, V, H),
		p + "embed_tokens_per_layer.weight":     bf16T(1, VPL, L*PL),
		p + "per_layer_model_projection.weight": bf16T(1, L*PL, H),
		p + "per_layer_projection_norm.weight":  bf16T(1, PL),
		p + "norm.weight":                       bf16T(1, H),
		// multimodal towers: language-only conversion must skip them.
		"model.vision_tower.timm_model.conv.weight": bf16T(1, 2, 2),
		"model.audio_tower.conformer.0.weight":      bf16T(1, 2, 2),
		"model.embed_vision.embedding.weight":       bf16T(1, 4, 4),
		"model.embed_audio.embedding.weight":        bf16T(1, 4, 4),
	}
	for i := 0; i < 3; i++ {
		tensors[fmt.Sprintf("%saltup_projections.%d.weight", p, i)] = bf16T(float32(1+i), H, H)
		tensors[fmt.Sprintf("%saltup_unembed_projections.%d.weight", p, i)] = bf16T(float32(4+i), H, H)
	}
	for l := 0; l < L; l++ {
		lp := fmt.Sprintf("%slayers.%d.", p, l)
		for _, n := range []string{"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
			"post_feedforward_layernorm", "post_per_layer_input_norm", "altup.router_norm", "laurel.post_laurel_norm"} {
			tensors[lp+n+".weight"] = bf16T(1, H)
		}
		tensors[lp+"self_attn.q_proj.weight"] = bf16T(1, 8, H)
		tensors[lp+"self_attn.k_proj.weight"] = bf16T(1, 4, H)
		tensors[lp+"self_attn.v_proj.weight"] = bf16T(1, 4, H)
		tensors[lp+"self_attn.o_proj.weight"] = bf16T(1, H, 8)
		tensors[lp+"self_attn.q_norm.weight"] = bf16T(1, 4)
		tensors[lp+"self_attn.k_norm.weight"] = bf16T(1, 4)
		tensors[lp+"mlp.gate_proj.weight"] = bf16T(1, FF, H)
		tensors[lp+"mlp.up_proj.weight"] = bf16T(1, FF, H)
		tensors[lp+"mlp.down_proj.weight"] = bf16T(1, H, FF)
		tensors[lp+"per_layer_input_gate.weight"] = bf16T(1, PL, H)
		tensors[lp+"per_layer_projection.weight"] = bf16T(1, H, PL)
		tensors[lp+"altup.correct_output_scale"] = bf16T(1, H) // a bare Parameter: no .weight
		tensors[lp+"altup.correction_coefs.weight"] = bf16T(1, 4, 4)
		tensors[lp+"altup.prediction_coefs.weight"] = bf16T(1, 16, 4)
		tensors[lp+"altup.modality_router.weight"] = bf16T(1, 4, H)
		tensors[lp+"laurel.linear_left.weight"] = bf16T(1, 2, H)
		tensors[lp+"laurel.linear_right.weight"] = bf16T(1, H, 2)
	}
	if mutate != nil {
		mutate(cfg, tensors)
	}
	if err := writeJSON(dir, "config.json", cfg); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.model"), tinySPMModel(), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writeJSON(dir, "tokenizer_config.json", map[string]any{
		"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>",
		"add_bos_token": true, "chat_template": "{{ bos_token }}",
		"added_tokens_decoder": map[string]any{
			"7": map[string]any{"content": "<start_of_turn>", "special": true},
		},
	}); err != nil {
		t.Fatal(err)
	}
	if err := writeSafetensors(filepath.Join(dir, "model.safetensors"), tensors); err != nil {
		t.Fatal(err)
	}
}

const (
	testBF16 = 30
	testF32  = 0
)

func TestConvertGemma3nMatchesLlamaCppLayout(t *testing.T) {
	dir := t.TempDir()
	writeGemma3nSnapshot(t, dir, nil)
	dest := filepath.Join(dir, "out.gguf")
	stats, err := ConvertDir(dir, dest, ConvertOptions{Name: "tiny-gemma3n"})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Architecture != "gemma3n" {
		t.Fatalf("arch %q", stats.Architecture)
	}
	joined := strings.Join(stats.Warnings, "\n")
	if !strings.Contains(joined, "language-only") {
		t.Fatalf("expected the language-only warning, got %v", stats.Warnings)
	}
	// This tiny fixture deliberately differs from the real 256 / 64 / 20
	// values the current llama.cpp gemma3n loader assumes: warn, don't block.
	for _, want := range []string{"hidden_size_per_layer_input=4", "laurel_rank=2", "2 layers keep their own KV"} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing loader-assumption warning %q in %v", want, stats.Warnings)
		}
	}
	g := readTestGGUF(t, dest)

	// Tensor names and GGUF dims exactly as llama.cpp's gemma3n loader
	// creates them (src/models/gemma3n.cpp), with n_embd=8, n_embd_altup=4,
	// n_layer=3, n_vocab=12, n_altup=4, laurel_rank=2, n_ff=16.
	want := map[string][]uint64{
		"token_embd.weight":           {8, 12},
		"per_layer_token_embd.weight": {12, 12}, // {n_embd_altup*n_layer, n_vocab}
		"per_layer_model_proj.weight": {8, 12},  // {n_embd, n_embd_altup*n_layer}
		"per_layer_proj_norm.weight":  {4},
		"altup_proj.weight":           {8, 8, 3}, // {n_embd, n_embd, n_altup-1}
		"altup_unembd_proj.weight":    {8, 8, 3},
		"output_norm.weight":          {8},
	}
	for l := 0; l < 3; l++ {
		b := func(n string) string { return fmt.Sprintf("blk.%d.%s", l, n) }
		for n, d := range map[string][]uint64{
			"attn_norm.weight": {8}, "attn_q.weight": {8, 8}, "attn_k.weight": {8, 4},
			"attn_v.weight": {8, 4}, "attn_output.weight": {8, 8},
			"attn_q_norm.weight": {4}, "attn_k_norm.weight": {4},
			"post_attention_norm.weight": {8}, "ffn_norm.weight": {8},
			"ffn_gate.weight": {8, 16}, "ffn_up.weight": {8, 16}, "ffn_down.weight": {16, 8},
			"post_ffw_norm.weight": {8},
			"inp_gate.weight":      {8, 4}, "proj.weight": {4, 8}, "post_norm.weight": {8},
			"altup_correct_coef.weight": {4, 4}, "altup_correct_scale.weight": {8},
			"altup_predict_coef.weight": {4, 16}, "altup_router.weight": {8, 4},
			"altup_router_norm.weight": {8},
			"laurel_l.weight":          {8, 2}, "laurel_r.weight": {2, 8}, "laurel_post_norm.weight": {8},
		} {
			want[b(n)] = d
		}
	}
	for name, dims := range want {
		ti, ok := g.Tensors[name]
		if !ok {
			t.Errorf("missing tensor %s", name)
			continue
		}
		if !dimsEqual(ti.Dims, dims...) {
			t.Errorf("%s dims %v, want %v", name, ti.Dims, dims)
		}
	}
	for name := range g.Tensors {
		if _, ok := want[name]; !ok {
			t.Errorf("unexpected tensor %s (vision/audio must be skipped, nothing else extra)", name)
		}
	}
	if _, ok := g.Tensors["output.weight"]; ok {
		t.Error("tied output must not be duplicated into output.weight")
	}

	// Storage types: norms, 1-D and the AltUp coefficient matrices are F32.
	for name, typ := range map[string]uint32{
		"blk.0.attn_norm.weight":           testF32,
		"blk.0.altup_correct_scale.weight": testF32,
		"blk.0.altup_correct_coef.weight":  testF32,
		"blk.0.altup_predict_coef.weight":  testF32,
		"blk.0.laurel_post_norm.weight":    testF32,
		"per_layer_proj_norm.weight":       testF32,
		"blk.0.attn_q.weight":              testBF16,
		"blk.0.altup_router.weight":        testBF16,
		"blk.0.laurel_l.weight":            testBF16,
		"token_embd.weight":                testBF16,
		"per_layer_token_embd.weight":      testBF16,
		"altup_proj.weight":                testBF16,
	} {
		if got := g.Tensors[name].Type; got != typ {
			t.Errorf("%s stored as ggml type %d, want %d", name, got, typ)
		}
	}

	// AltUp matrices are stacked in index order.
	for name, base := range map[string]float32{"altup_proj.weight": 1, "altup_unembd_proj.weight": 4} {
		raw := g.payload(t, name, 3*64*2)
		for k := 0; k < 3; k++ {
			wantBits := uint16(math.Float32bits(base+float32(k)) >> 16)
			for i := 0; i < 64; i++ {
				if got := binary.LittleEndian.Uint16(raw[(k*64+i)*2:]); got != wantBits {
					t.Fatalf("%s: matrix %d element %d = %#x, want %#x", name, k, i, got, wantBits)
				}
			}
		}
	}

	// The per-layer table is padded with zero rows up to n_vocab (the
	// converter pads Gemma 3n's 262144 → 262400 the same way).
	pl := g.payload(t, "per_layer_token_embd.weight", 12*12*2)
	rowBytes := 12 * 2
	for r := 0; r < 12; r++ {
		nonzero := false
		for _, v := range pl[r*rowBytes : (r+1)*rowBytes] {
			if v != 0 {
				nonzero = true
			}
		}
		if wantData := r < 10; nonzero != wantData {
			t.Errorf("per-layer embedding row %d nonzero=%v, want %v", r, nonzero, wantData)
		}
	}

	// Metadata.
	if g.KV["general.architecture"] != "gemma3n" {
		t.Errorf("architecture %v", g.KV["general.architecture"])
	}
	for k, v := range map[string]uint32{
		"gemma3n.block_count": 3, "gemma3n.embedding_length": 8, "gemma3n.feed_forward_length": 16,
		"gemma3n.attention.head_count": 2, "gemma3n.attention.head_count_kv": 1,
		"gemma3n.attention.key_length": 4, "gemma3n.attention.value_length": 4,
		"gemma3n.attention.sliding_window": 8, "gemma3n.altup.active_idx": 0,
		"gemma3n.altup.num_inputs": 4, "gemma3n.embedding_length_per_layer_input": 4,
		"gemma3n.attention.shared_kv_layers": 1,
	} {
		if got := g.u32(t, k); got != v {
			t.Errorf("%s = %d, want %d", k, got, v)
		}
	}
	if got := g.f32(t, "gemma3n.rope.freq_base"); got != 1e6 {
		t.Errorf("rope.freq_base = %v", got)
	}
	if got := g.f32(t, "gemma3n.rope.freq_base_swa"); got != 10000 {
		t.Errorf("rope.freq_base_swa = %v", got)
	}
	if got := g.f32(t, "gemma3n.final_logit_softcapping"); got != 30 {
		t.Errorf("softcap = %v", got)
	}
	pat, _ := g.KV["gemma3n.attention.sliding_window_pattern"].([]bool)
	if len(pat) != 3 || !pat[0] || !pat[1] || pat[2] {
		t.Errorf("sliding_window_pattern = %v", g.KV["gemma3n.attention.sliding_window_pattern"])
	}
	sp, _ := g.KV["gemma3n.activation_sparsity_scale"].([]float32)
	if len(sp) != 3 || math.Abs(float64(sp[0])-1.6448536) > 1e-6 || sp[1] != 0 || !math.IsInf(float64(sp[2]), -1) {
		t.Errorf("activation_sparsity_scale = %v, want [1.6448536 0 -Inf]", sp)
	}

	// SentencePiece tokenizer, as llama.cpp's converter writes it.
	if g.KV["tokenizer.ggml.model"] != "llama" {
		t.Errorf("tokenizer model %v, want llama (SentencePiece)", g.KV["tokenizer.ggml.model"])
	}
	toks, _ := g.KV["tokenizer.ggml.tokens"].([]string)
	scores, _ := g.KV["tokenizer.ggml.scores"].([]float32)
	types, _ := g.KV["tokenizer.ggml.token_type"].([]int32)
	if len(toks) != 12 || len(scores) != 12 || len(types) != 12 {
		t.Fatalf("tokenizer lengths tokens=%d scores=%d types=%d, want 12 (vocab_size)", len(toks), len(scores), len(types))
	}
	if toks[4] != "▁a" || scores[4] != -1 || types[4] != tokenTypeNormal {
		t.Errorf("piece 4 = %q %v %d", toks[4], scores[4], types[4])
	}
	if types[6] != tokenTypeByte || types[3] != tokenTypeUnknown || types[0] != tokenTypeControl {
		t.Errorf("token types %v", types)
	}
	// Like llama.cpp's converter: a user-defined piece is a plain token
	// unless tokenizer_config.json marks it special (then control).
	if types[7] != tokenTypeControl {
		t.Errorf("<start_of_turn> (special in tokenizer_config) type %d, want CONTROL", types[7])
	}
	if types[8] != tokenTypeNormal {
		t.Errorf("<end_of_turn> (user-defined piece, no override) type %d, want NORMAL", types[8])
	}
	if toks[10] != "[PAD10]" || types[10] != tokenTypeUnused || scores[10] != spmPadScore {
		t.Errorf("padding slot = %q %d %v", toks[10], types[10], scores[10])
	}
	if g.KV["tokenizer.ggml.add_space_prefix"] != false {
		t.Errorf("add_space_prefix = %v, want false", g.KV["tokenizer.ggml.add_space_prefix"])
	}
	if g.u32(t, "tokenizer.ggml.bos_token_id") != 2 || g.u32(t, "tokenizer.ggml.eos_token_id") != 1 ||
		g.u32(t, "tokenizer.ggml.padding_token_id") != 0 || g.u32(t, "tokenizer.ggml.unknown_token_id") != 3 {
		t.Errorf("special ids bos=%v eos=%v pad=%v unk=%v", g.KV["tokenizer.ggml.bos_token_id"],
			g.KV["tokenizer.ggml.eos_token_id"], g.KV["tokenizer.ggml.padding_token_id"], g.KV["tokenizer.ggml.unknown_token_id"])
	}
	if g.KV["tokenizer.ggml.pre"] != "default" {
		t.Errorf("tokenizer.pre = %v, want default", g.KV["tokenizer.ggml.pre"])
	}
	if _, ok := g.KV["tokenizer.ggml.merges"]; ok {
		t.Error("SentencePiece tokenizers carry no BPE merges")
	}
}

func TestConvertGemma3nFeedForwardList(t *testing.T) {
	dir := t.TempDir()
	writeGemma3nSnapshot(t, dir, func(cfg map[string]any, _ map[string]stTensor) {
		cfg["text_config"].(map[string]any)["intermediate_size"] = []any{16, 16, 16}
	})
	dest := filepath.Join(dir, "out.gguf")
	if _, err := ConvertDir(dir, dest, ConvertOptions{}); err != nil {
		t.Fatal(err)
	}
	g := readTestGGUF(t, dest)
	arr, _ := g.KV["gemma3n.feed_forward_length"].([]int32)
	if len(arr) != 3 || arr[0] != 16 {
		t.Fatalf("feed_forward_length = %v, want per-layer [16 16 16]", g.KV["gemma3n.feed_forward_length"])
	}
}

func TestConvertGemma3nRejectsUnsupportedAltup(t *testing.T) {
	dir := t.TempDir()
	writeGemma3nSnapshot(t, dir, func(cfg map[string]any, _ map[string]stTensor) {
		cfg["text_config"].(map[string]any)["altup_num_inputs"] = 3
	})
	_, err := ConvertDir(dir, filepath.Join(dir, "out.gguf"), ConvertOptions{})
	if err == nil || !strings.Contains(err.Error(), "altup_num_inputs=3") {
		t.Fatalf("err = %v, want altup input count rejection", err)
	}
}

func TestConvertGemma3nFailsClosedOnUnknownTensor(t *testing.T) {
	dir := t.TempDir()
	writeGemma3nSnapshot(t, dir, func(_ map[string]any, ts map[string]stTensor) {
		ts["model.language_model.layers.0.mystery.weight"] = bf16T(1, 2, 2)
	})
	_, err := ConvertDir(dir, filepath.Join(dir, "out.gguf"), ConvertOptions{})
	if err == nil || !strings.Contains(err.Error(), "mystery") {
		t.Fatalf("err = %v, want the unmapped tensor named", err)
	}
}

func TestConvertGemma3nIncompleteAltupStackFails(t *testing.T) {
	dir := t.TempDir()
	writeGemma3nSnapshot(t, dir, func(_ map[string]any, ts map[string]stTensor) {
		delete(ts, "model.language_model.altup_projections.2.weight")
	})
	_, err := ConvertDir(dir, filepath.Join(dir, "out.gguf"), ConvertOptions{})
	if err == nil || !strings.Contains(err.Error(), "altup_proj.weight") {
		t.Fatalf("err = %v, want an incomplete altup stack rejection", err)
	}
}

func TestSPMModelParse(t *testing.T) {
	pieces, err := parseSPMModel(tinySPMModel())
	if err != nil {
		t.Fatal(err)
	}
	if len(pieces) != 10 {
		t.Fatalf("pieces = %d, want 10 (trailing fields skipped)", len(pieces))
	}
	if p := pieces[4]; p.Piece != "▁a" || p.Score != -1 || p.Type != 1 {
		t.Errorf("piece 4 = %+v", p)
	}
	if pieces[3].Type != 2 || pieces[6].Type != 6 {
		t.Errorf("types = %d %d", pieces[3].Type, pieces[6].Type)
	}
	if _, err := parseSPMModel([]byte{0x0a, 0x20, 0x01}); err == nil {
		t.Error("truncated model must fail")
	}
	if _, err := parseSPMModel(nil); err == nil {
		t.Error("empty model must fail")
	}
}

func TestGemmaActivationSparsity(t *testing.T) {
	got := gemmaActivationSparsity([]float64{0.95, 0.5, 0, 1})
	if math.Abs(float64(got[0])-1.6448536) > 1e-6 || got[1] != 0 ||
		!math.IsInf(float64(got[2]), -1) || !math.IsInf(float64(got[3]), 1) {
		t.Errorf("got %v", got)
	}
}

// --- Gemma 4 --------------------------------------------------------------

func writeGemma4Snapshot(t *testing.T, dir string, mutate func(cfg map[string]any, tensors map[string]stTensor)) {
	t.Helper()
	const H, PL, L, V = 8, 4, 4, 12
	text := map[string]any{
		"model_type":                  "gemma4_text",
		"hidden_size":                 H,
		"hidden_size_per_layer_input": PL,
		"intermediate_size":           16,
		"num_hidden_layers":           L,
		"num_attention_heads":         2,
		"num_key_value_heads":         1,
		"num_global_key_value_heads":  2,
		"head_dim":                    4,
		"global_head_dim":             8,
		"vocab_size":                  V,
		"vocab_size_per_layer_input":  V,
		"max_position_embeddings":     128,
		"rms_norm_eps":                1e-6,
		"sliding_window":              16,
		"layer_types":                 []any{"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"},
		"num_kv_shared_layers":        1,
		"use_double_wide_mlp":         true,
		"final_logit_softcapping":     30.0,
		"rope_parameters": map[string]any{
			"full_attention":    map[string]any{"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
			"sliding_attention": map[string]any{"rope_type": "default", "rope_theta": 10000.0},
		},
	}
	cfg := map[string]any{
		"architectures": []any{"Gemma4ForConditionalGeneration"},
		"model_type":    "gemma4",
		"text_config":   text,
	}
	p := "model.language_model."
	tensors := map[string]stTensor{
		p + "embed_tokens.weight":               bf16T(1, V, H),
		p + "embed_tokens_per_layer.weight":     bf16T(1, V, L*PL),
		p + "per_layer_model_projection.weight": bf16T(1, L*PL, H),
		p + "per_layer_projection_norm.weight":  bf16T(1, PL),
		p + "norm.weight":                       bf16T(1, H),
		"model.vision_tower.encoder.0.weight":   bf16T(1, 2, 2),
		"model.embed_vision.embedding.weight":   bf16T(1, 4, 4),
	}
	for l := 0; l < L; l++ {
		lp := fmt.Sprintf("%slayers.%d.", p, l)
		full := l == 3
		hd, kv, ff := int64(4), int64(1), int64(16)
		if full {
			hd, kv = 8, 2
		}
		if l == 3 {
			ff = 32 // double-wide MLP on the KV-shared layer
		}
		for _, n := range []string{"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
			"post_feedforward_layernorm", "post_per_layer_input_norm"} {
			tensors[lp+n+".weight"] = bf16T(1, H)
		}
		tensors[lp+"self_attn.q_proj.weight"] = bf16T(1, 2*hd, H)
		tensors[lp+"self_attn.k_proj.weight"] = bf16T(1, kv*hd, H)
		tensors[lp+"self_attn.v_proj.weight"] = bf16T(1, kv*hd, H)
		tensors[lp+"self_attn.o_proj.weight"] = bf16T(1, H, 2*hd)
		tensors[lp+"self_attn.q_norm.weight"] = bf16T(1, hd)
		tensors[lp+"self_attn.k_norm.weight"] = bf16T(1, hd)
		tensors[lp+"mlp.gate_proj.weight"] = bf16T(1, ff, H)
		tensors[lp+"mlp.up_proj.weight"] = bf16T(1, ff, H)
		tensors[lp+"mlp.down_proj.weight"] = bf16T(1, H, ff)
		tensors[lp+"per_layer_input_gate.weight"] = bf16T(1, PL, H)
		tensors[lp+"per_layer_projection.weight"] = bf16T(1, H, PL)
		tensors[lp+"layer_scalar"] = bf16T(0.5, 1)
	}
	if mutate != nil {
		mutate(cfg, tensors)
	}
	if err := writeJSON(dir, "config.json", cfg); err != nil {
		t.Fatal(err)
	}
	vocab := map[string]int{
		"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "▁a": 4, "b": 5,
		"<0x0A>": 6, "<|channel>": 7, "<channel|>": 8, "<turn|>": 9, "ab": 10, "<|think|>": 11,
	}
	tok := map[string]any{
		"model": map[string]any{
			"type": "BPE", "vocab": vocab,
			"merges": []any{[]any{"▁", "a"}, []any{"a", "b"}},
		},
		"added_tokens": []any{
			map[string]any{"id": 0, "content": "<pad>", "special": true},
			map[string]any{"id": 1, "content": "<eos>", "special": true},
			map[string]any{"id": 2, "content": "<bos>", "special": true},
			map[string]any{"id": 7, "content": "<|channel>", "special": true},
			map[string]any{"id": 8, "content": "<channel|>", "special": true},
			map[string]any{"id": 9, "content": "<turn|>", "special": true},
			map[string]any{"id": 11, "content": "<|think|>", "special": true},
		},
	}
	if err := writeJSON(dir, "tokenizer.json", tok); err != nil {
		t.Fatal(err)
	}
	if err := writeJSON(dir, "tokenizer_config.json", map[string]any{
		"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>",
		"add_bos_token": true, "chat_template": "{{ bos_token }}",
	}); err != nil {
		t.Fatal(err)
	}
	if err := writeJSON(dir, "generation_config.json", map[string]any{"eos_token_id": []any{1, 9}}); err != nil {
		t.Fatal(err)
	}
	if err := writeSafetensors(filepath.Join(dir, "model.safetensors"), tensors); err != nil {
		t.Fatal(err)
	}
}

func TestConvertGemma4MatchesLlamaCppLayout(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Snapshot(t, dir, nil)
	dest := filepath.Join(dir, "out.gguf")
	stats, err := ConvertDir(dir, dest, ConvertOptions{Name: "tiny-gemma4"})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Architecture != "gemma4" {
		t.Fatalf("arch %q", stats.Architecture)
	}
	g := readTestGGUF(t, dest)

	// llama.cpp gemma4 loader (src/models/gemma4.cpp): n_embd=8, n_embd_per_layer=4,
	// n_layer=4, n_vocab=12; SWA head dim 4 / 1 KV head, global head dim 8 / 2 KV heads.
	want := map[string][]uint64{
		"token_embd.weight":           {8, 12},
		"per_layer_token_embd.weight": {16, 12},
		"per_layer_model_proj.weight": {8, 16},
		"per_layer_proj_norm.weight":  {4},
		"output_norm.weight":          {8},
		"rope_freqs.weight":           {4}, // {n_embd_head_global / 2}
	}
	for l := 0; l < 4; l++ {
		hd, kv, ff := uint64(4), uint64(1), uint64(16)
		if l == 3 {
			hd, kv, ff = 8, 2, 32
		}
		b := func(n string) string { return fmt.Sprintf("blk.%d.%s", l, n) }
		for n, d := range map[string][]uint64{
			"attn_norm.weight": {8}, "attn_q.weight": {8, 2 * hd}, "attn_k.weight": {8, kv * hd},
			"attn_v.weight": {8, kv * hd}, "attn_output.weight": {2 * hd, 8},
			"attn_q_norm.weight": {hd}, "attn_k_norm.weight": {hd},
			"post_attention_norm.weight": {8}, "ffn_norm.weight": {8},
			"ffn_gate.weight": {8, ff}, "ffn_up.weight": {8, ff}, "ffn_down.weight": {ff, 8},
			"post_ffw_norm.weight": {8}, "layer_output_scale.weight": {1},
			"inp_gate.weight": {8, 4}, "proj.weight": {4, 8}, "post_norm.weight": {8},
		} {
			want[b(n)] = d
		}
	}
	for name, dims := range want {
		ti, ok := g.Tensors[name]
		if !ok {
			t.Errorf("missing tensor %s", name)
			continue
		}
		if !dimsEqual(ti.Dims, dims...) {
			t.Errorf("%s dims %v, want %v", name, ti.Dims, dims)
		}
	}
	for name := range g.Tensors {
		if _, ok := want[name]; !ok {
			t.Errorf("unexpected tensor %s", name)
		}
	}
	for _, name := range []string{"rope_freqs.weight", "blk.0.layer_output_scale.weight", "blk.0.attn_norm.weight"} {
		if g.Tensors[name].Type != testF32 {
			t.Errorf("%s must be F32, got ggml type %d", name, g.Tensors[name].Type)
		}
	}

	// rope_freqs: head_dim_full=8, partial_rotary_factor=0.25 → 1 rotated
	// frequency (1.0) and 3 unrotated ones (1e30).
	rf := g.payload(t, "rope_freqs.weight", 16)
	var got []float32
	for i := 0; i < 4; i++ {
		got = append(got, math.Float32frombits(binary.LittleEndian.Uint32(rf[i*4:])))
	}
	if got[0] != 1 || got[1] != 1e30 || got[2] != 1e30 || got[3] != 1e30 {
		t.Errorf("rope_freqs = %v", got)
	}
	// layer_scalar is copied through (0.5 in F32).
	ls := g.payload(t, "blk.2.layer_output_scale.weight", 4)
	if v := math.Float32frombits(binary.LittleEndian.Uint32(ls)); v != 0.5 {
		t.Errorf("layer_output_scale = %v, want 0.5", v)
	}
	// Norms are applied as stored: no Gemma 1-3 "+1" shift.
	nw := g.payload(t, "blk.0.attn_norm.weight", 4)
	if v := math.Float32frombits(binary.LittleEndian.Uint32(nw)); v != 1 {
		t.Errorf("attn_norm[0] = %v, want 1 (no +1 shift for gemma4)", v)
	}

	// Metadata the gemma4 loader requires or refines.
	for k, v := range map[string]uint32{
		"gemma4.block_count": 4, "gemma4.embedding_length": 8, "gemma4.attention.head_count": 2,
		"gemma4.attention.key_length": 8, "gemma4.attention.value_length": 8,
		"gemma4.attention.key_length_swa": 4, "gemma4.attention.value_length_swa": 4,
		"gemma4.rope.dimension_count": 8, "gemma4.rope.dimension_count_swa": 4,
		"gemma4.embedding_length_per_layer_input": 4, "gemma4.attention.shared_kv_layers": 1,
		"gemma4.attention.sliding_window": 16,
	} {
		if got := g.u32(t, k); got != v {
			t.Errorf("%s = %d, want %d", k, got, v)
		}
	}
	if kv, _ := g.KV["gemma4.attention.head_count_kv"].([]int32); len(kv) != 4 || kv[0] != 1 || kv[2] != 1 || kv[3] != 2 {
		t.Errorf("head_count_kv = %v, want per-layer [1 1 1 2]", g.KV["gemma4.attention.head_count_kv"])
	}
	if ff, _ := g.KV["gemma4.feed_forward_length"].([]int32); len(ff) != 4 || ff[0] != 16 || ff[2] != 16 || ff[3] != 32 {
		t.Errorf("feed_forward_length = %v, want [16 16 16 32] (double-wide on the shared layer)", g.KV["gemma4.feed_forward_length"])
	}
	if pat, _ := g.KV["gemma4.attention.sliding_window_pattern"].([]bool); len(pat) != 4 || !pat[0] || pat[3] {
		t.Errorf("sliding_window_pattern = %v", g.KV["gemma4.attention.sliding_window_pattern"])
	}
	if got := g.f32(t, "gemma4.rope.freq_base"); got != 1e6 {
		t.Errorf("rope.freq_base = %v", got)
	}
	if got := g.f32(t, "gemma4.rope.freq_base_swa"); got != 10000 {
		t.Errorf("rope.freq_base_swa = %v", got)
	}

	// "gemma4" tokenizer: BPE merges, byte fallback typed, chat tokens visible.
	if g.KV["tokenizer.ggml.model"] != "gemma4" {
		t.Errorf("tokenizer model %v, want gemma4", g.KV["tokenizer.ggml.model"])
	}
	merges, _ := g.KV["tokenizer.ggml.merges"].([]string)
	if len(merges) != 2 || merges[1] != "a b" {
		t.Errorf("merges = %v", merges)
	}
	toks, _ := g.KV["tokenizer.ggml.tokens"].([]string)
	types, _ := g.KV["tokenizer.ggml.token_type"].([]int32)
	if len(toks) != 12 || len(types) != 12 {
		t.Fatalf("tokens %d types %d", len(toks), len(types))
	}
	if types[6] != tokenTypeByte {
		t.Errorf("<0x0A> type %d, want BYTE", types[6])
	}
	if types[7] != tokenTypeUserDefined || types[8] != tokenTypeUserDefined {
		t.Errorf("<|channel>/<channel|> types %d %d, want USER_DEFINED", types[7], types[8])
	}
	if types[1] != tokenTypeControl || types[11] != tokenTypeControl {
		t.Errorf("eos/think types %d %d, want CONTROL", types[1], types[11])
	}
	if g.KV["tokenizer.ggml.add_space_prefix"] != false || g.KV["tokenizer.ggml.add_bos_token"] != true {
		t.Errorf("space prefix %v add_bos %v", g.KV["tokenizer.ggml.add_space_prefix"], g.KV["tokenizer.ggml.add_bos_token"])
	}
	if _, ok := g.KV["tokenizer.ggml.scores"]; ok {
		t.Error("BPE gemma4 tokenizer carries no scores")
	}
	eog, _ := g.KV["tokenizer.ggml.eos_token_ids"].([]uint32)
	if len(eog) != 2 || eog[0] != 1 || eog[1] != 9 {
		t.Errorf("eos_token_ids = %v, want [1 9] (<eos>, <turn|>)", eog)
	}
}

func TestConvertGemma4WithoutPerLayerInput(t *testing.T) {
	// The 31B-style dense model has no per-layer embeddings: the PLE keys
	// are written as 0 and no PLE tensors are expected.
	dir := t.TempDir()
	writeGemma4Snapshot(t, dir, func(cfg map[string]any, ts map[string]stTensor) {
		text := cfg["text_config"].(map[string]any)
		delete(text, "hidden_size_per_layer_input")
		for name := range ts {
			if strings.Contains(name, "per_layer") || strings.Contains(name, "post_per_layer") {
				delete(ts, name)
			}
		}
	})
	dest := filepath.Join(dir, "out.gguf")
	if _, err := ConvertDir(dir, dest, ConvertOptions{}); err != nil {
		t.Fatal(err)
	}
	g := readTestGGUF(t, dest)
	if got := g.u32(t, "gemma4.embedding_length_per_layer_input"); got != 0 {
		t.Errorf("embedding_length_per_layer_input = %d, want 0", got)
	}
	if _, ok := g.Tensors["per_layer_token_embd.weight"]; ok {
		t.Error("no per-layer table expected")
	}
}

func TestConvertGemma4FailsClosedOnUnknownTensor(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Snapshot(t, dir, func(_ map[string]any, ts map[string]stTensor) {
		ts["model.language_model.layers.1.self_attn.mystery_norm.weight"] = bf16T(1, 4)
	})
	_, err := ConvertDir(dir, filepath.Join(dir, "out.gguf"), ConvertOptions{})
	if err == nil || !strings.Contains(err.Error(), "mystery_norm") {
		t.Fatalf("err = %v, want the unmapped tensor named", err)
	}
}

func TestConvertGemma4RequiresMerges(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Snapshot(t, dir, nil)
	if err := writeJSON(dir, "tokenizer.json", map[string]any{
		"model": map[string]any{"type": "BPE", "vocab": map[string]int{"a": 0}, "merges": []any{}},
	}); err != nil {
		t.Fatal(err)
	}
	_, err := ConvertDir(dir, filepath.Join(dir, "out.gguf"), ConvertOptions{})
	if err == nil || !strings.Contains(err.Error(), "merges") {
		t.Fatalf("err = %v, want a missing-merges rejection", err)
	}
}

// --- detection ------------------------------------------------------------

func TestInferArchGemmaN(t *testing.T) {
	g3n := map[string]any{
		"model_type": "gemma3n", "architectures": []any{"Gemma3nForConditionalGeneration"},
		"altup_num_inputs": float64(4), "hidden_size_per_layer_input": float64(256),
		"hidden_size": float64(8), "num_attention_heads": float64(2), "num_hidden_layers": float64(2),
	}
	g4 := map[string]any{
		"model_type": "gemma4", "architectures": []any{"Gemma4ForConditionalGeneration"},
		"hidden_size_per_layer_input": float64(256),
		"hidden_size":                 float64(8), "num_attention_heads": float64(2), "num_hidden_layers": float64(2),
	}
	cases := []struct {
		name    string
		cfg     map[string]any
		names   []string
		want    string
		wantErr string
	}{
		{name: "gemma3n", cfg: g3n, want: "gemma3n"},
		{name: "gemma4 E-series", cfg: g4, want: "gemma4"},
		{name: "gemma4 by tensor names", cfg: map[string]any{
			"model_type": "gemma4", "architectures": []any{"Gemma4ForCausalLM"},
			"hidden_size": float64(8), "num_attention_heads": float64(2), "num_hidden_layers": float64(2),
		}, names: []string{"model.language_model.layers.0.per_layer_input_gate.weight"}, want: "gemma4"},
		{name: "gemma4 assistant", cfg: map[string]any{
			"model_type": "gemma4_assistant", "architectures": []any{"Gemma4AssistantForCausalLM"},
		}, wantErr: "assistant"},
		{name: "gemma4 moe", cfg: map[string]any{
			"model_type": "gemma4", "architectures": []any{"Gemma4ForConditionalGeneration"},
			"hidden_size_per_layer_input": float64(0), "num_experts": float64(8),
			"hidden_size": float64(8), "num_attention_heads": float64(2), "num_hidden_layers": float64(2),
		}, wantErr: "mixture-of-experts"},
		{name: "altup on another architecture", cfg: map[string]any{
			"model_type": "llama", "architectures": []any{"LlamaForCausalLM"},
			"altup_num_inputs": float64(4),
		}, wantErr: "only Gemma 3n and Gemma 4"},
		{name: "per-layer tensors on another architecture", cfg: map[string]any{
			"model_type": "llama", "architectures": []any{"LlamaForCausalLM"},
		}, names: []string{"model.layers.0.per_layer_input_gate.weight"}, wantErr: "only Gemma 3n and Gemma 4"},
		{name: "unresolvable gemma4 variant must not become gemma3", cfg: map[string]any{
			"model_type": "gemma4_dspark", "architectures": []any{"Gemma4DSparkModel"},
			"hidden_size_per_layer_input": float64(256), "num_hidden_layers": float64(2),
			"hidden_size": float64(8), "num_attention_heads": float64(2),
		}, names: []string{"model.layers.0.pre_feedforward_layernorm.weight", "model.layers.0.self_attn.q_norm.weight"},
			wantErr: "no loader"},
	}
	for _, c := range cases {
		feat := detectLayout(c.cfg, c.names)
		arch, err := inferArch(c.cfg, feat)
		if err == nil {
			err = requireConvertible(arch, feat)
		}
		if c.wantErr != "" {
			if err == nil || !strings.Contains(err.Error(), c.wantErr) {
				t.Errorf("%s: err = %v (arch %q), want substring %q", c.name, err, arch, c.wantErr)
			}
			continue
		}
		if err != nil || arch != c.want {
			t.Errorf("%s: arch %q err %v, want %q", c.name, arch, err, c.want)
		}
	}
}

func TestGemmaNFamilyMapsNames(t *testing.T) {
	f3 := familyFor("gemma3n", layout{GemmaFF: true, QKNorm: true, GemmaN: "gemma3n"})
	f4 := familyFor("gemma4", layout{GemmaFF: true, QKNorm: true, GemmaN: "gemma4"})
	cases := []struct {
		f    Family
		hf   string
		want string
		kind workKind
	}{
		{f3, "model.language_model.embed_tokens_per_layer.weight", "per_layer_token_embd.weight", kindCopy},
		{f3, "model.language_model.per_layer_projection_norm.weight", "per_layer_proj_norm.weight", kindRMS},
		{f3, "model.language_model.altup_projections.2.weight", "altup_proj.weight", kindStackIdx},
		{f3, "model.language_model.altup_unembed_projections.0.weight", "altup_unembd_proj.weight", kindStackIdx},
		{f3, "model.language_model.layers.7.altup.correct_output_scale", "blk.7.altup_correct_scale.weight", kindCopy},
		{f3, "model.language_model.layers.7.altup.prediction_coefs.weight", "blk.7.altup_predict_coef.weight", kindF32},
		{f3, "model.language_model.layers.7.altup.correction_coefs.weight", "blk.7.altup_correct_coef.weight", kindF32},
		{f3, "model.language_model.layers.7.laurel.post_laurel_norm.weight", "blk.7.laurel_post_norm.weight", kindRMS},
		{f3, "model.language_model.layers.7.post_per_layer_input_norm.weight", "blk.7.post_norm.weight", kindRMS},
		{f3, "model.language_model.layers.7.pre_feedforward_layernorm.weight", "blk.7.ffn_norm.weight", kindRMS},
		{f4, "model.language_model.layers.3.layer_scalar", "blk.3.layer_output_scale.weight", kindCopy},
		{f4, "model.language_model.layers.3.self_attn.k_norm.weight", "blk.3.attn_k_norm.weight", kindRMS},
	}
	for _, c := range cases {
		m := c.f.MapName(c.hf)
		if m.GGUF != c.want || m.Kind != c.kind {
			t.Errorf("%s → %q (%s), want %q (%s)", c.hf, m.GGUF, m.Kind, c.want, c.kind)
		}
	}
	// AltUp is Gemma 3n only.
	if m := f4.MapName("model.language_model.altup_projections.0.weight"); m.GGUF != "" {
		t.Errorf("gemma4 must not map altup tensors, got %+v", m)
	}
	for _, name := range []string{"model.embed_vision.embedding.weight", "model.embed_audio.hard_embedding_norm.weight"} {
		if m := f3.MapName(name); !m.Vision {
			t.Errorf("%s must be treated as a non-text tensor", name)
		}
	}
}

func TestGGUFWriterSetKVReplaces(t *testing.T) {
	w := &Writer{}
	w.AddKV("a", uint32(1))
	w.SetKV("a", uint32(2))
	w.SetKV("b", uint32(3))
	if len(w.kv) != 2 || w.kv[0].Value != uint32(2) || !w.HasKV("b") || w.HasKV("c") {
		t.Fatalf("kv = %+v", w.kv)
	}
}

// The converted files must pass Studio's own import-time checks, not just
// parse: metadata extraction, tensor-table validation and vocab layout.
func TestConvertedGemmaNPassStudioValidation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		write func(*testing.T, string, func(map[string]any, map[string]stTensor))
		arch  string
	}{
		{"gemma3n", writeGemma3nSnapshot, "gemma3n"},
		{"gemma4", writeGemma4Snapshot, "gemma4"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			tc.write(t, dir, nil)
			dest := filepath.Join(dir, "out.gguf")
			if _, err := ConvertDir(dir, dest, ConvertOptions{Name: "tiny"}); err != nil {
				t.Fatal(err)
			}
			md, err := gguf.ParseFile(dest)
			if err != nil {
				t.Fatalf("ParseFile: %v", err)
			}
			if md.Architecture != tc.arch {
				t.Errorf("architecture %q", md.Architecture)
			}
			if issues, _, err := gguf.ValidateFile(dest); err != nil {
				t.Fatalf("ValidateFile: %v", err)
			} else if len(issues) > 0 {
				t.Errorf("validation issues: %v", issues)
			}
			if err := gguf.CheckVocabLayout(dest); err != nil {
				t.Errorf("vocab layout: %v", err)
			}
			if _, _, err := gguf.ListTensors(dest); err != nil {
				t.Errorf("ListTensors: %v", err)
			}
		})
	}
}

// The Quantization page's repo check (config.json only) must accept the
// AltUp / per-layer-embedding Gemma configs the converter now maps.
func TestEvaluateProbeAcceptsGemmaN(t *testing.T) {
	files := []NeededFile{
		{Path: "config.json", Size: 100},
		{Path: "model-00001-of-00004.safetensors", Size: 4000},
		{Path: "tokenizer.json", Size: 50},
	}
	cases := map[string]map[string]any{
		"gemma3n": {
			"architectures": []any{"Gemma3nForConditionalGeneration"}, "model_type": "gemma3n",
			"text_config": map[string]any{
				"model_type": "gemma3n_text", "altup_num_inputs": float64(4), "hidden_size_per_layer_input": float64(256),
				"hidden_size": float64(2048), "num_attention_heads": float64(8), "num_hidden_layers": float64(35),
			},
		},
		"gemma4": {
			"architectures": []any{"Gemma4ForConditionalGeneration"}, "model_type": "gemma4",
			"text_config": map[string]any{
				"model_type": "gemma4_text", "hidden_size_per_layer_input": float64(256),
				"hidden_size": float64(2560), "num_attention_heads": float64(8), "num_hidden_layers": float64(42),
			},
		},
	}
	for want, cfgRaw := range cases {
		b, _ := json.Marshal(cfgRaw)
		cfg, err := ParseConfig(b)
		if err != nil {
			t.Fatal(err)
		}
		res := Evaluate(ProbeInput{RepoID: "google/" + want, Files: files, DTypes: map[string]int64{"BF16": 1}, Config: cfg, HasJSON: true})
		if !res.Compatible || res.Adapter != want {
			t.Errorf("%s: compatible=%v adapter=%q reason=%q", want, res.Compatible, res.Adapter, res.Reason)
		}
	}
}

// --- review fixes -----------------------------------------------------------

func TestSnapshotFilterKeepsSentencePieceModel(t *testing.T) {
	files := []NeededFile{
		{Path: "config.json"}, {Path: "tokenizer.json"}, {Path: "tokenizer.model"},
		{Path: "tokenizer_config.json"}, {Path: "model-00001-of-00004.safetensors"},
		{Path: "README.md"}, {Path: "pytorch_model.bin"},
	}
	kept := map[string]bool{}
	for _, f := range SelectSnapshotFiles(files) {
		kept[f.Path] = true
	}
	for _, want := range []string{"tokenizer.model", "tokenizer.json", "config.json"} {
		if !kept[want] {
			t.Errorf("%s dropped from the download list", want)
		}
	}
	if kept["README.md"] || kept["pytorch_model.bin"] {
		t.Errorf("unrelated files kept: %v", kept)
	}
}

// What the From-HF flow actually downloads must be enough to convert.
func TestConvertGemma3nFromFilteredSnapshot(t *testing.T) {
	full := t.TempDir()
	writeGemma3nSnapshot(t, full, nil)
	ents, err := os.ReadDir(full)
	if err != nil {
		t.Fatal(err)
	}
	var all []NeededFile
	for _, e := range ents {
		all = append(all, NeededFile{Path: e.Name()})
	}
	// Extra files a real repo has that the filter must drop.
	all = append(all, NeededFile{Path: "README.md"}, NeededFile{Path: "pytorch_model.bin"})
	snap := t.TempDir()
	for _, f := range SelectSnapshotFiles(all) {
		b, err := os.ReadFile(filepath.Join(full, f.Path))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(snap, f.Path), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	dest := filepath.Join(snap, "out.gguf")
	if _, err := ConvertDir(snap, dest, ConvertOptions{}); err != nil {
		t.Fatal(err)
	}
	if g := readTestGGUF(t, dest); g.KV["tokenizer.ggml.model"] != "llama" {
		t.Errorf("tokenizer model %v, want llama", g.KV["tokenizer.ggml.model"])
	}
}

func TestConvertGemma3nRequiresTokenizerModel(t *testing.T) {
	dir := t.TempDir()
	writeGemma3nSnapshot(t, dir, nil)
	if err := os.Remove(filepath.Join(dir, "tokenizer.model")); err != nil {
		t.Fatal(err)
	}
	// Even with a tokenizer.json present, falling back would write a BPE
	// tokenizer with a pre-tokenizer name llama.cpp rejects.
	if err := writeJSON(dir, "tokenizer.json", map[string]any{
		"model": map[string]any{"type": "BPE", "vocab": map[string]int{"a": 0}, "merges": []any{}},
	}); err != nil {
		t.Fatal(err)
	}
	_, err := ConvertDir(dir, filepath.Join(dir, "out.gguf"), ConvertOptions{})
	if err == nil || !strings.Contains(err.Error(), "tokenizer.model") {
		t.Fatalf("err = %v, want a missing tokenizer.model rejection", err)
	}
}

// Gemma 1-3 share the root cause (a "gpt2" tokenizer with a Gemma
// pre-tokenizer name): with a tokenizer.model they now get the SPM tokenizer.
func TestConvertGemma3UsesSentencePieceTokenizer(t *testing.T) {
	dir := t.TempDir()
	cfg := map[string]any{
		"architectures": []any{"Gemma3ForCausalLM"}, "model_type": "gemma3_text",
		"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
		"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
		"vocab_size": 10, "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
		"sliding_window": 8, "layer_types": []any{"full_attention"},
	}
	if err := writeJSON(dir, "config.json", cfg); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.model"), tinySPMModel(), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writeJSON(dir, "tokenizer_config.json", map[string]any{"bos_token": "<bos>", "eos_token": "<eos>", "add_bos_token": true}); err != nil {
		t.Fatal(err)
	}
	lp := "model."
	ts := map[string]stTensor{
		lp + "embed_tokens.weight": bf16T(1, 10, 8), lp + "norm.weight": bf16T(1, 8),
		lp + "layers.0.input_layernorm.weight":            bf16T(1, 8),
		lp + "layers.0.post_attention_layernorm.weight":   bf16T(1, 8),
		lp + "layers.0.pre_feedforward_layernorm.weight":  bf16T(1, 8),
		lp + "layers.0.post_feedforward_layernorm.weight": bf16T(1, 8),
		lp + "layers.0.self_attn.q_proj.weight":           bf16T(1, 8, 8),
		lp + "layers.0.self_attn.k_proj.weight":           bf16T(1, 4, 8),
		lp + "layers.0.self_attn.v_proj.weight":           bf16T(1, 4, 8),
		lp + "layers.0.self_attn.o_proj.weight":           bf16T(1, 8, 8),
		lp + "layers.0.self_attn.q_norm.weight":           bf16T(1, 4),
		lp + "layers.0.self_attn.k_norm.weight":           bf16T(1, 4),
		lp + "layers.0.mlp.gate_proj.weight":              bf16T(1, 16, 8),
		lp + "layers.0.mlp.up_proj.weight":                bf16T(1, 16, 8),
		lp + "layers.0.mlp.down_proj.weight":              bf16T(1, 8, 16),
	}
	if err := writeSafetensors(filepath.Join(dir, "model.safetensors"), ts); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(dir, "out.gguf")
	stats, err := ConvertDir(dir, dest, ConvertOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Architecture != "gemma3" {
		t.Fatalf("arch %s", stats.Architecture)
	}
	g := readTestGGUF(t, dest)
	if g.KV["tokenizer.ggml.model"] != "llama" || g.KV["tokenizer.ggml.pre"] != "default" {
		t.Errorf("tokenizer model=%v pre=%v, want llama/default", g.KV["tokenizer.ggml.model"], g.KV["tokenizer.ggml.pre"])
	}
	if scores, _ := g.KV["tokenizer.ggml.scores"].([]float32); len(scores) != 10 {
		t.Errorf("scores = %d, want 10", len(scores))
	}
	// Gemma 1-3 keep their "+1" norm shift (only 3n/4 dropped it).
	nw := g.payload(t, "blk.0.attn_norm.weight", 4)
	if v := math.Float32frombits(binary.LittleEndian.Uint32(nw)); v != 2 {
		t.Errorf("gemma3 attn_norm[0] = %v, want 2 (1 + stored 1)", v)
	}

	// Without tokenizer.model the previous tokenizer.json path is kept.
	if err := os.Remove(filepath.Join(dir, "tokenizer.model")); err != nil {
		t.Fatal(err)
	}
	if err := writeTinyBPETokenizer(dir, map[string]int{"a": 0, "b": 1, "<bos>": 2, "<eos>": 3},
		[]any{map[string]any{"id": 2, "content": "<bos>", "special": true}, map[string]any{"id": 3, "content": "<eos>", "special": true}},
		"<bos>", "<eos>", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := ConvertDir(dir, filepath.Join(dir, "out2.gguf"), ConvertOptions{}); err != nil {
		t.Fatalf("tokenizer.json fallback for gemma3 broke: %v", err)
	}
}

func TestInferArchGemmaNVariantsWithoutHints(t *testing.T) {
	base := map[string]any{"hidden_size": float64(8), "num_attention_heads": float64(2), "num_hidden_layers": float64(2)}
	mk := func(extra map[string]any) map[string]any {
		out := map[string]any{}
		for k, v := range base {
			out[k] = v
		}
		for k, v := range extra {
			out[k] = v
		}
		return out
	}
	names := []string{"model.layers.0.pre_feedforward_layernorm.weight", "model.layers.0.self_attn.q_norm.weight"}
	for _, c := range []struct {
		name string
		cfg  map[string]any
		want string
	}{
		{"dspark without a per-layer hint", mk(map[string]any{"model_type": "gemma4_dspark", "architectures": []any{"Gemma4DSparkModel"}}), "no loader"},
		{"unified without a per-layer hint", mk(map[string]any{"model_type": "gemma4_unified", "architectures": []any{"Gemma4UnifiedForConditionalGeneration"}}), "no loader"},
		{"moe flagged only by enable_moe_block", mk(map[string]any{"model_type": "gemma4", "architectures": []any{"Gemma4ForConditionalGeneration"}, "enable_moe_block": true}), "mixture-of-experts"},
	} {
		feat := detectLayout(c.cfg, names)
		arch, err := inferArch(c.cfg, feat)
		if err == nil {
			err = requireConvertible(arch, feat)
		}
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: arch %q err %v, want %q (must never fall through to gemma3)", c.name, arch, err, c.want)
		}
	}
	// A dense Gemma 4 without per-layer input still resolves to gemma4.
	dense := mk(map[string]any{"model_type": "gemma4", "architectures": []any{"Gemma4ForCausalLM"}})
	if arch, err := inferArch(dense, detectLayout(dense, names)); err != nil || arch != "gemma4" {
		t.Errorf("dense gemma4: arch %q err %v", arch, err)
	}
}

func TestValidateGemmaN(t *testing.T) {
	f3 := familyFor("gemma3n", layout{GemmaN: "gemma3n"})
	f4 := familyFor("gemma4", layout{GemmaN: "gemma4"})
	h := hyper{nLayer: 35, nEmbd: 2048}
	real3n := map[string]any{
		"hidden_size_per_layer_input": float64(256), "laurel_rank": float64(64), "altup_num_inputs": float64(4),
		"altup_active_idx": float64(0), "num_kv_shared_layers": float64(15), "sliding_window": float64(512),
		"intermediate_size": float64(16384),
	}
	if warn, err := validateGemmaN(f3, real3n, h); err != nil || len(warn) != 0 {
		t.Errorf("real E4B-shaped gemma3n config: warnings %v err %v, want none", warn, err)
	}
	noWindow := map[string]any{}
	for k, v := range real3n {
		noWindow[k] = v
	}
	delete(noWindow, "sliding_window")
	if _, err := validateGemmaN(f3, noWindow, h); err == nil || !strings.Contains(err.Error(), "sliding_window") {
		t.Errorf("missing sliding_window: err %v", err)
	}
	badList := map[string]any{}
	for k, v := range real3n {
		badList[k] = v
	}
	badList["intermediate_size"] = []any{float64(16384), float64(16384)}
	if _, err := validateGemmaN(f3, badList, h); err == nil || !strings.Contains(err.Error(), "2 widths for 35 layers") {
		t.Errorf("short intermediate_size list: err %v", err)
	}
	g4 := map[string]any{
		"sliding_window": float64(512), "global_head_dim": float64(512), "layer_types": []any{"full_attention"},
		"rope_parameters": map[string]any{"full_attention": map[string]any{"rope_type": "proportional"}},
	}
	if warn, err := validateGemmaN(f4, g4, hyper{nLayer: 42, nEmbd: 2560}); err != nil || len(warn) != 0 {
		t.Errorf("gemma4: warnings %v err %v", warn, err)
	}
	g4["rope_parameters"] = map[string]any{"full_attention": map[string]any{"rope_type": "dynamic"}}
	if _, err := validateGemmaN(f4, g4, hyper{nLayer: 42, nEmbd: 2560}); err == nil || !strings.Contains(err.Error(), "rope type") {
		t.Errorf("unknown rope type: err %v", err)
	}
}

// llama.cpp aborts on a rope scaling type it does not know; Gemma 4's
// "proportional" rope is carried by rope_freqs.weight instead.
func TestGemmaRopeScalingOnlyLinearOrYarn(t *testing.T) {
	cases := []struct {
		name string
		cfg  map[string]any
		want string // "" = no scaling keys
	}{
		{"nested proportional", map[string]any{"rope_parameters": map[string]any{
			"full_attention": map[string]any{"rope_type": "proportional", "partial_rotary_factor": 0.25}}}, ""},
		{"flat proportional", map[string]any{"rope_parameters": map[string]any{
			"rope_type": "proportional", "partial_rotary_factor": 0.25}}, ""},
		{"default", map[string]any{"rope_parameters": map[string]any{
			"full_attention": map[string]any{"rope_type": "default"}}}, ""},
		{"linear", map[string]any{"rope_parameters": map[string]any{
			"full_attention": map[string]any{"rope_type": "linear", "factor": 8.0}}}, "linear"},
	}
	for _, c := range cases {
		w := &Writer{}
		writeGemmaRopeScaling(w, "gemma4", c.cfg)
		if c.want == "" {
			if len(w.kv) != 0 {
				t.Errorf("%s: wrote %v, want no scaling keys", c.name, w.kv)
			}
			continue
		}
		if len(w.kv) < 2 || w.kv[0].Key != "gemma4.rope.scaling.type" || w.kv[0].Value != c.want {
			t.Errorf("%s: kv = %v", c.name, w.kv)
		}
	}
}

func TestReadPayloadPaddedAndNoCopy(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.safetensors")
	if err := writeSafetensors(path, map[string]stTensor{"t": {DType: "BF16", Shape: []int64{2, 2}, Data: []byte{1, 2, 3, 4, 5, 6, 7, 8}}}); err != nil {
		t.Fatal(err)
	}
	refs, err := IndexDir(dir)
	if err != nil || len(refs) != 1 {
		t.Fatalf("index: %v %v", refs, err)
	}
	got, err := readPayloadPadded(refs[0], 4)
	if err != nil {
		t.Fatal(err)
	}
	if want := []byte{1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0}; !bytes.Equal(got, want) {
		t.Errorf("padded = %v, want %v", got, want)
	}
	src := []byte{1, 2, 3, 4}
	out, err := convertPayload(src, "BF16", GGMLBF16)
	if err != nil || &out[0] != &src[0] {
		t.Errorf("same-dtype convertPayload must hand the buffer through, not copy (err %v)", err)
	}
}
