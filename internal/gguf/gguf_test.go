package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// buildGGUF assembles a minimal valid GGUF header in memory.
func buildGGUF(t *testing.T, kvs map[string]any) []byte {
	t.Helper()
	var b bytes.Buffer
	w32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	w64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	wstr := func(s string) { w64(uint64(len(s))); b.WriteString(s) }

	binary.Write(&b, binary.LittleEndian, uint32(magicGGUF))
	w32(3)                // version
	w64(0)                // tensor count
	w64(uint64(len(kvs))) // kv count
	for k, v := range kvs {
		wstr(k)
		switch val := v.(type) {
		case string:
			w32(tString)
			wstr(val)
		case uint32:
			w32(tUint32)
			w32(val)
		case uint64:
			w32(tUint64)
			w64(val)
		case float32:
			w32(tFloat32)
			binary.Write(&b, binary.LittleEndian, val)
		case []bool:
			w32(tArray)
			w32(tBool)
			w64(uint64(len(val)))
			for _, x := range val {
				if x {
					b.WriteByte(1)
				} else {
					b.WriteByte(0)
				}
			}
		case []uint32:
			w32(tArray)
			w32(tInt32) // gemma4 uses i32 for head_count_kv arrays
			w64(uint64(len(val)))
			for _, x := range val {
				w32(x)
			}
		case []string:
			w32(tArray)
			w32(tString)
			w64(uint64(len(val)))
			for _, s := range val {
				wstr(s)
			}
		default:
			t.Fatalf("unsupported kv type %T for %s", v, k)
		}
	}
	return b.Bytes()
}

func TestParseValid(t *testing.T) {
	data := buildGGUF(t, map[string]any{
		"general.name":            "TestModel",
		"general.architecture":    "llama",
		"general.file_type":       uint32(15), // Q4_K_M
		"general.parameter_count": uint64(7_000_000_000),
		"llama.context_length":    uint32(4096),
		"llama.embedding_length":  uint32(4096),
		"tokenizer.ggml.model":    "llama",
	})
	md, err := parse(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if md.Name != "TestModel" || md.Architecture != "llama" {
		t.Errorf("unexpected name/arch: %+v", md)
	}
	if md.Quantization != "Q4_K_M" {
		t.Errorf("quant = %q, want Q4_K_M", md.Quantization)
	}
	if md.Parameters != 7_000_000_000 {
		t.Errorf("params = %d", md.Parameters)
	}
	if md.ContextLength != 4096 || md.Embedding != 4096 {
		t.Errorf("ctx/emb = %d/%d", md.ContextLength, md.Embedding)
	}
}

func TestTokenizerCount(t *testing.T) {
	data := buildGGUF(t, map[string]any{
		"general.architecture":  "llama",
		"llama.vocab_size":      uint32(5),
		"tokenizer.ggml.tokens": []string{"a", "b", "c", "d", "e"},
		"tokenizer.ggml.model":  "llama",
	})
	md, err := parse(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if md.TokenizerCount != 5 {
		t.Fatalf("tokenizer_count=%d want 5", md.TokenizerCount)
	}
	if md.VocabSize != 5 {
		t.Fatalf("vocab_size=%d want 5", md.VocabSize)
	}
}

func TestParseBadMagic(t *testing.T) {
	_, err := parse(bytes.NewReader([]byte("NOPE........")), 12)
	if err != ErrBadMagic {
		t.Fatalf("want ErrBadMagic, got %v", err)
	}
}

func TestParseBadVersion(t *testing.T) {
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint32(magicGGUF))
	binary.Write(&b, binary.LittleEndian, uint32(99))
	_, err := parse(bytes.NewReader(b.Bytes()), int64(b.Len()))
	if err == nil || !bytes.Contains([]byte(err.Error()), []byte("version")) {
		t.Fatalf("want version error, got %v", err)
	}
}

func TestParseTruncated(t *testing.T) {
	data := buildGGUF(t, map[string]any{"general.name": "x"})
	// Cut the file in half — must fail cleanly, not panic.
	_, err := parse(bytes.NewReader(data[:len(data)/2]), int64(len(data)/2))
	if err == nil {
		t.Fatal("expected truncation error")
	}
}

func TestParseMaliciousCounts(t *testing.T) {
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint32(magicGGUF))
	binary.Write(&b, binary.LittleEndian, uint32(3))
	binary.Write(&b, binary.LittleEndian, uint64(1<<40)) // absurd tensor count
	binary.Write(&b, binary.LittleEndian, uint64(1<<40)) // absurd kv count
	_, err := parse(bytes.NewReader(b.Bytes()), int64(b.Len()))
	if err == nil {
		t.Fatal("expected bounds error")
	}
}

func TestParseHugeStringLength(t *testing.T) {
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint32(magicGGUF))
	binary.Write(&b, binary.LittleEndian, uint32(3))
	binary.Write(&b, binary.LittleEndian, uint64(0))
	binary.Write(&b, binary.LittleEndian, uint64(1))
	// Key with impossible length.
	binary.Write(&b, binary.LittleEndian, uint64(1<<62))
	_, err := parse(bytes.NewReader(b.Bytes()), int64(b.Len()))
	if err == nil {
		t.Fatal("expected bounds error for oversized string")
	}
}

func TestParseSlidingWindowMetadata(t *testing.T) {
	pattern := []bool{true, true, true, true, true, false}
	heads := []uint32{8, 8, 8, 8, 8, 1}
	data := buildGGUF(t, map[string]any{
		"general.architecture":                    "gemma4",
		"gemma4.block_count":                      uint32(6),
		"gemma4.embedding_length":                 uint32(3840),
		"gemma4.attention.head_count":             uint32(16),
		"gemma4.attention.head_count_kv":          heads,
		"gemma4.attention.key_length":             uint32(512),
		"gemma4.attention.value_length":           uint32(512),
		"gemma4.attention.key_length_swa":         uint32(256),
		"gemma4.attention.value_length_swa":       uint32(256),
		"gemma4.attention.sliding_window":         uint32(1024),
		"gemma4.attention.sliding_window_pattern": pattern,
		"gemma4.attention.shared_kv_layers":       uint32(0),
	})
	md, err := parse(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if md.HeadCountKV != 8 {
		t.Errorf("head_count_kv max = %d, want 8", md.HeadCountKV)
	}
	if len(md.HeadCountKVLayers) != 6 || md.HeadCountKVLayers[5] != 1 {
		t.Errorf("per-layer kv heads = %v", md.HeadCountKVLayers)
	}
	if md.SlidingWindow != 1024 || md.HeadDimSWA != 256 {
		t.Errorf("swa window/dim = %d/%d", md.SlidingWindow, md.HeadDimSWA)
	}
	if len(md.SlidingWindowPattern) != 6 || md.SlidingWindowPattern[5] {
		t.Errorf("pattern = %v", md.SlidingWindowPattern)
	}
}

func TestParseFullAttentionInterval(t *testing.T) {
	data := buildGGUF(t, map[string]any{
		"general.architecture":           "qwen35",
		"qwen35.block_count":             uint32(24),
		"qwen35.attention.head_count":    uint32(8),
		"qwen35.attention.head_count_kv": uint32(2),
		"qwen35.attention.key_length":    uint32(256),
		"qwen35.full_attention_interval": uint32(4),
		"qwen35.ssm.state_size":          uint32(128),
		"qwen35.ssm.inner_size":          uint32(2048),
	})
	md, err := parse(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if md.FullAttentionInterval != 4 || md.SSMStateSize != 128 || md.SSMInnerSize != 2048 {
		t.Errorf("hybrid meta = interval=%d state=%d inner=%d",
			md.FullAttentionInterval, md.SSMStateSize, md.SSMInnerSize)
	}
}

func TestBytesForType(t *testing.T) {
	if got := BytesForType(256, "q4_k"); got != 144 {
		t.Errorf("q4_k 256 = %d, want 144", got)
	}
	if got := BytesForType(256, "q8_0"); got != 272 {
		t.Errorf("q8_0 256 = %d, want 272", got)
	}
}
