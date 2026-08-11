// Package gguf reads GGUF metadata with bounded allocations. It validates
// magic/version and does not load tensor payloads into memory.
package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

const (
	magicGGUF = 0x46554747 // "GGUF" little-endian

	maxKVCount      = 1 << 20
	maxTensorCount  = 1 << 22
	maxStringLen    = 16 << 20 // 16 MiB string cap
	maxArrayLen     = 1 << 24  // element cap per array
	maxCaptureArray = 4096     // numeric/bool arrays retained for architecture metadata
	maxVersion      = 3
	minVersion      = 2
)

// Metadata is the extracted, UI-relevant subset of a GGUF header.
type Metadata struct {
	Version           uint32   `json:"version"`
	TensorCount       uint64   `json:"tensor_count"`
	Name              string   `json:"name,omitempty"`
	Architecture      string   `json:"architecture,omitempty"`
	FileType          uint32   `json:"file_type"`                // general.file_type
	Quantization      string   `json:"quantization,omitempty"`   // resolved name of FileType
	Parameters        uint64   `json:"parameters,omitempty"`     // general.parameter_count
	ContextLength     uint32   `json:"context_length,omitempty"` // <arch>.context_length
	Embedding         uint32   `json:"embedding_length,omitempty"`
	BlockCount        uint32   `json:"block_count,omitempty"`          // <arch>.block_count (layers)
	HeadCount         uint32   `json:"head_count,omitempty"`           // attention heads
	HeadCountKV       uint32   `json:"head_count_kv,omitempty"`        // KV heads (GQA); max if per-layer
	HeadCountKVLayers []uint32 `json:"head_count_kv_layers,omitempty"` // per-layer KV heads (Gemma 4)
	HeadDim           uint32   `json:"head_dim,omitempty"`             // key dim (derived or key_length)
	ValueDim          uint32   `json:"value_dim,omitempty"`            // value dim when ≠ key
	HeadDimSWA        uint32   `json:"head_dim_swa,omitempty"`         // SWA key dim
	ValueDimSWA       uint32   `json:"value_dim_swa,omitempty"`        // SWA value dim
	SlidingWindow     uint32   `json:"sliding_window,omitempty"`
	// SlidingWindowPattern: true = sliding-window layer, false = full-context.
	SlidingWindowPattern []bool `json:"sliding_window_pattern,omitempty"`
	SharedKVLayers       uint32 `json:"shared_kv_layers,omitempty"` // trailing layers reuse KV
	// FullAttentionInterval: hybrid models (Qwen3.5) — only every Nth layer has dense KV.
	FullAttentionInterval uint32         `json:"full_attention_interval,omitempty"`
	SSMStateSize          uint32         `json:"ssm_state_size,omitempty"`
	SSMInnerSize          uint32         `json:"ssm_inner_size,omitempty"`
	ChatTemplate          string         `json:"chat_template,omitempty"`
	Tokenizer             string         `json:"tokenizer,omitempty"`
	Multimodal            bool           `json:"multimodal"`
	HasVision             bool           `json:"has_vision"`
	HasAudio              bool           `json:"has_audio"`
	Projector             bool           `json:"projector"`         // looks like an mmproj file
	SpeculativeDraft      bool           `json:"speculative_draft"` // sidecar draft, not a chat model
	HasMTP                bool           `json:"has_mtp"`           // NextN / MTP heads present
	NextnPredictLayers    uint32         `json:"nextn_predict_layers,omitempty"`
	SpecType              SpecType       `json:"spec_type,omitempty"`    // preferred --spec-type when used as draft
	IsEmbedding           bool           `json:"is_embedding"`           // dedicated embedding / encoder model
	IsReranker            bool           `json:"is_reranker"`            // embedding subtype for cross-encoder rank
	PoolingType           PoolingType    `json:"pooling_type,omitempty"` // none|mean|cls|last|rank
	EmbeddingLengthOut    uint32         `json:"embedding_length_out,omitempty"`
	IsDiffusion           bool           `json:"is_diffusion"`            // block-diffusion LM (not autoregressive)
	CanvasLength          uint32         `json:"canvas_length,omitempty"` // diffusion canvas size (tokens)
	Raw                   map[string]any `json:"-"`                       // full kv for future use, not serialized to UI
}

// Errors that callers can classify.
var (
	ErrBadMagic     = errors.New("not a GGUF file (bad magic)")
	ErrBadVersion   = errors.New("unsupported GGUF version")
	ErrTruncated    = errors.New("truncated GGUF header")
	ErrBoundsUnsafe = errors.New("declared size exceeds safety bound")
)

// Known llama.cpp file-type → quantization label (subset; unknown → "").
var fileTypeNames = map[uint32]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0",
	8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
	14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
	19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
	24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
	29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
	34: "TQ1_0", 35: "TQ2_0", 36: "MXFP4",
}

// reader wraps io.ReaderAt with a hard cursor and error stickiness.
type reader struct {
	r   io.ReaderAt
	off int64
	err error
}

func (r *reader) read(p []byte) {
	if r.err != nil {
		return
	}
	n, err := r.r.ReadAt(p, r.off)
	r.off += int64(n)
	if err != nil {
		r.err = ErrTruncated
	}
}

func (r *reader) u32() uint32 {
	var b [4]byte
	r.read(b[:])
	return binary.LittleEndian.Uint32(b[:])
}

func (r *reader) u64() uint64 {
	var b [8]byte
	r.read(b[:])
	return binary.LittleEndian.Uint64(b[:])
}

func (r *reader) str() string {
	n := r.u64()
	if r.err != nil {
		return ""
	}
	if n > maxStringLen {
		r.err = fmt.Errorf("%w: string length %d", ErrBoundsUnsafe, n)
		return ""
	}
	buf := make([]byte, n)
	r.read(buf)
	return string(buf)
}

// skip advances the cursor n bytes (bounds-checked against remaining file).
func (r *reader) skip(n uint64, fileSize int64) {
	if r.err != nil {
		return
	}
	if n > math.MaxInt64 || r.off+int64(n) > fileSize {
		r.err = ErrTruncated
		return
	}
	r.off += int64(n)
}

// value type tags
const (
	tUint8, tInt8, tUint16, tInt16   = 0, 1, 2, 3
	tUint32, tInt32, tFloat32, tBool = 4, 5, 6, 7
	tString, tArray, tUint64, tInt64 = 8, 9, 10, 11
	tFloat64                         = 12
)

func fixedSize(t uint32) (uint64, bool) {
	switch t {
	case tUint8, tInt8, tBool:
		return 1, true
	case tUint16, tInt16:
		return 2, true
	case tUint32, tInt32, tFloat32:
		return 4, true
	case tUint64, tInt64, tFloat64:
		return 8, true
	}
	return 0, false
}

// value reads one metadata value; strings and small scalar arrays are
// captured, everything else is skipped.
func (r *reader) value(fileSize int64) any {
	t := r.u32()
	if r.err != nil {
		return nil
	}
	switch t {
	case tString:
		return r.str()
	case tArray:
		et := r.u32()
		n := r.u64()
		if r.err != nil {
			return nil
		}
		if n > maxArrayLen {
			r.err = fmt.Errorf("%w: array length %d", ErrBoundsUnsafe, n)
			return nil
		}
		if et == tString {
			// Read up to 64 strings for inspection, skip the rest.
			out := make([]string, 0, 8)
			var i uint64
			for ; i < n; i++ {
				s := r.str()
				if r.err != nil {
					return nil
				}
				if i < 64 {
					out = append(out, s)
				}
			}
			return out
		}
		sz, ok := fixedSize(et)
		if !ok {
			r.err = fmt.Errorf("unknown array element type %d", et)
			return nil
		}
		// Keep small numeric/bool arrays (per-layer heads, SWA patterns).
		if n <= maxCaptureArray {
			switch et {
			case tBool:
				raw := make([]byte, n)
				r.read(raw)
				out := make([]bool, n)
				for i, b := range raw {
					out[i] = b != 0
				}
				return out
			case tUint8, tInt8:
				raw := make([]byte, n)
				r.read(raw)
				out := make([]uint32, n)
				for i, b := range raw {
					if et == tInt8 {
						out[i] = uint32(int8(b))
					} else {
						out[i] = uint32(b)
					}
				}
				return out
			case tUint16, tInt16:
				out := make([]uint32, n)
				for i := uint64(0); i < n; i++ {
					var b [2]byte
					r.read(b[:])
					v := binary.LittleEndian.Uint16(b[:])
					if et == tInt16 {
						out[i] = uint32(int16(v))
					} else {
						out[i] = uint32(v)
					}
				}
				return out
			case tUint32, tInt32:
				out := make([]uint32, n)
				for i := uint64(0); i < n; i++ {
					v := r.u32()
					if et == tInt32 {
						out[i] = uint32(int32(v))
					} else {
						out[i] = v
					}
				}
				return out
			}
		}
		r.skip(sz*n, fileSize)
		return nil
	default:
		switch t {
		case tUint8:
			var b [1]byte
			r.read(b[:])
			return uint8(b[0])
		case tInt8:
			var b [1]byte
			r.read(b[:])
			return int8(b[0])
		case tUint16:
			var b [2]byte
			r.read(b[:])
			return binary.LittleEndian.Uint16(b[:])
		case tInt16:
			var b [2]byte
			r.read(b[:])
			return int16(binary.LittleEndian.Uint16(b[:]))
		case tUint32:
			return r.u32()
		case tInt32:
			return int32(r.u32())
		case tFloat32:
			return math.Float32frombits(r.u32())
		case tBool:
			var b [1]byte
			r.read(b[:])
			return b[0] != 0
		case tUint64:
			return r.u64()
		case tInt64:
			return int64(r.u64())
		case tFloat64:
			return math.Float64frombits(r.u64())
		}
		r.err = fmt.Errorf("unknown value type %d", t)
		return nil
	}
}

// ParseFile reads the GGUF header of path. It opens, validates and closes the
// file; tensor payloads are never touched.
func ParseFile(path string) (*Metadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	return parse(f, st.Size())
}

func parse(ra io.ReaderAt, fileSize int64) (*Metadata, error) {
	r := &reader{r: ra}
	if magic := r.u32(); magic != magicGGUF {
		return nil, ErrBadMagic
	}
	version := r.u32()
	if r.err != nil {
		return nil, r.err
	}
	if version < minVersion || version > maxVersion {
		return nil, fmt.Errorf("%w: %d", ErrBadVersion, version)
	}
	tensors := r.u64()
	kvCount := r.u64()
	if r.err != nil {
		return nil, r.err
	}
	if tensors > maxTensorCount || kvCount > maxKVCount {
		return nil, fmt.Errorf("%w: tensors=%d kv=%d", ErrBoundsUnsafe, tensors, kvCount)
	}

	md := &Metadata{Version: version, TensorCount: tensors, Raw: map[string]any{}}
	for i := uint64(0); i < kvCount; i++ {
		key := r.str()
		val := r.value(fileSize)
		if r.err != nil {
			return nil, r.err
		}
		md.Raw[key] = val
	}
	md.extract()
	return md, nil
}

// extract lifts well-known keys into typed fields.
func (md *Metadata) extract() {
	get := func(k string) (any, bool) { v, ok := md.Raw[k]; return v, ok }
	if v, ok := get("general.name"); ok {
		if s, ok := v.(string); ok {
			md.Name = s
		}
	}
	if v, ok := get("general.architecture"); ok {
		if s, ok := v.(string); ok {
			md.Architecture = s
		}
	}
	if v, ok := get("general.file_type"); ok {
		if n, ok := toUint32(v); ok {
			md.FileType = n
			md.Quantization = fileTypeNames[n]
		}
	}
	if v, ok := get("general.parameter_count"); ok {
		if n, ok := toUint64(v); ok {
			md.Parameters = n
		}
	}
	if md.Architecture != "" {
		if v, ok := get(md.Architecture + ".context_length"); ok {
			if n, ok := toUint32(v); ok {
				md.ContextLength = n
			}
		}
		if v, ok := get(md.Architecture + ".embedding_length"); ok {
			if n, ok := toUint32(v); ok {
				md.Embedding = n
			}
		}
		if v, ok := get(md.Architecture + ".block_count"); ok {
			if n, ok := toUint32(v); ok {
				md.BlockCount = n
			}
		}
		if v, ok := get(md.Architecture + ".attention.head_count"); ok {
			if n, ok := toUint32(v); ok {
				md.HeadCount = n
			}
		}
		if v, ok := get(md.Architecture + ".attention.head_count_kv"); ok {
			if layers, ok := toUint32Slice(v); ok && len(layers) > 0 {
				md.HeadCountKVLayers = layers
				var max uint32
				for _, h := range layers {
					if h > max {
						max = h
					}
				}
				md.HeadCountKV = max
			} else if n, ok := toUint32(v); ok {
				md.HeadCountKV = n
			}
		}
	}
	// Head dimension: explicit key wins, else derive from embedding / heads.
	if v, ok := get(md.Architecture + ".attention.key_length"); ok {
		if n, ok := toUint32(v); ok {
			md.HeadDim = n
		}
	}
	if v, ok := get(md.Architecture + ".attention.value_length"); ok {
		if n, ok := toUint32(v); ok {
			md.ValueDim = n
		}
	}
	if v, ok := get(md.Architecture + ".attention.key_length_swa"); ok {
		if n, ok := toUint32(v); ok {
			md.HeadDimSWA = n
		}
	}
	if v, ok := get(md.Architecture + ".attention.value_length_swa"); ok {
		if n, ok := toUint32(v); ok {
			md.ValueDimSWA = n
		}
	}
	if v, ok := get(md.Architecture + ".attention.sliding_window"); ok {
		if n, ok := toUint32(v); ok {
			md.SlidingWindow = n
		}
	}
	if v, ok := get(md.Architecture + ".attention.sliding_window_pattern"); ok {
		if p, ok := toBoolSlice(v); ok {
			md.SlidingWindowPattern = p
		}
	}
	if v, ok := get(md.Architecture + ".attention.shared_kv_layers"); ok {
		if n, ok := toUint32(v); ok {
			md.SharedKVLayers = n
		}
	}
	if v, ok := get(md.Architecture + ".full_attention_interval"); ok {
		if n, ok := toUint32(v); ok {
			md.FullAttentionInterval = n
		}
	}
	if v, ok := get(md.Architecture + ".ssm.state_size"); ok {
		if n, ok := toUint32(v); ok {
			md.SSMStateSize = n
		}
	}
	if v, ok := get(md.Architecture + ".ssm.inner_size"); ok {
		if n, ok := toUint32(v); ok {
			md.SSMInnerSize = n
		}
	}
	if md.HeadDim == 0 && md.HeadCount > 0 && md.Embedding > 0 {
		md.HeadDim = md.Embedding / md.HeadCount
	}
	if md.ValueDim == 0 {
		md.ValueDim = md.HeadDim
	}
	if md.HeadCountKV == 0 {
		md.HeadCountKV = md.HeadCount // MHA fallback
	}
	if v, ok := get("tokenizer.chat_template"); ok {
		if s, ok := v.(string); ok {
			md.ChatTemplate = s
		}
	}
	if v, ok := get("tokenizer.ggml.model"); ok {
		if s, ok := v.(string); ok {
			md.Tokenizer = s
		}
	}
	// Multimodal indicators (vision / audio encoders live in mmproj or fused GGUF).
	// Boolean encoder flags must be truthy — vision-only mmproj files often
	// include clip.has_audio_encoder=false (and the reverse for audio-only).
	if rawBoolTrue(md.Raw, "clip.has_vision_encoder") {
		md.HasVision = true
		md.Multimodal = true
	}
	if rawBoolTrue(md.Raw, "clip.has_audio_encoder") {
		md.HasAudio = true
		md.Multimodal = true
	}
	for k := range md.Raw {
		lk := strings.ToLower(k)
		switch {
		case k == "clip.has_vision_encoder" || k == "clip.has_audio_encoder":
			// Handled above via value, not presence.
		case k == "clip.vision.patch_size" || k == "gemma3.mm.scale_emb" ||
			strings.HasPrefix(lk, "clip.vision.") || strings.HasPrefix(lk, "gemma3.mm."):
			md.HasVision = true
			md.Multimodal = true
		case k == "audio.block_count" || strings.HasPrefix(lk, "clip.audio.") ||
			strings.HasPrefix(lk, "audio."):
			md.HasAudio = true
			md.Multimodal = true
		}
	}
	if rawBoolTrue(md.Raw, "clip.has_vision_encoder") || md.HasVision {
		if md.Architecture == "clip" || md.Name == "" {
			md.Projector = md.Architecture == "clip"
		}
	}
	if md.Architecture == "clip" {
		md.Projector = true
	}

	// Speculative draft / MTP detection (path filled in by callers that have one).
	md.ApplySpeculativeFlags("")
}

// rawBoolTrue reports whether key exists in raw and is a true boolean (or a
// non-zero numeric stand-in). Missing keys and explicit false are false.
func rawBoolTrue(raw map[string]any, key string) bool {
	v, ok := raw[key]
	if !ok || v == nil {
		return false
	}
	switch b := v.(type) {
	case bool:
		return b
	case uint8:
		return b != 0
	case uint32:
		return b != 0
	case uint64:
		return b != 0
	case int:
		return b != 0
	case int32:
		return b != 0
	case int64:
		return b != 0
	default:
		return false
	}
}

func toUint32(v any) (uint32, bool) {
	switch n := v.(type) {
	case uint32:
		return n, true
	case uint64:
		if n <= math.MaxUint32 {
			return uint32(n), true
		}
	case int64:
		if n >= 0 && uint64(n) <= math.MaxUint32 {
			return uint32(n), true
		}
	case int32:
		if n >= 0 {
			return uint32(n), true
		}
	case int:
		if n >= 0 {
			return uint32(n), true
		}
	case uint16:
		return uint32(n), true
	case uint8:
		return uint32(n), true
	case float64:
		// JSON-unmarshaled numbers land as float64.
		if n >= 0 && n <= float64(math.MaxUint32) && n == math.Trunc(n) {
			return uint32(n), true
		}
	case float32:
		if n >= 0 && n <= float32(math.MaxUint32) && float64(n) == math.Trunc(float64(n)) {
			return uint32(n), true
		}
	}
	return 0, false
}

func toUint32Slice(v any) ([]uint32, bool) {
	switch a := v.(type) {
	case []uint32:
		if len(a) == 0 {
			return nil, false
		}
		return a, true
	case []any:
		out := make([]uint32, 0, len(a))
		for _, x := range a {
			n, ok := toUint32(x)
			if !ok {
				return nil, false
			}
			out = append(out, n)
		}
		return out, len(out) > 0
	}
	return nil, false
}

func toBoolSlice(v any) ([]bool, bool) {
	switch a := v.(type) {
	case []bool:
		if len(a) == 0 {
			return nil, false
		}
		return a, true
	case []any:
		out := make([]bool, len(a))
		for i, x := range a {
			switch b := x.(type) {
			case bool:
				out[i] = b
			case uint8:
				out[i] = b != 0
			case uint32:
				out[i] = b != 0
			default:
				return nil, false
			}
		}
		return out, true
	}
	return nil, false
}

func toUint64(v any) (uint64, bool) {
	switch n := v.(type) {
	case uint64:
		return n, true
	case uint32:
		return uint64(n), true
	case int64:
		if n >= 0 {
			return uint64(n), true
		}
	case uint8:
		return uint64(n), true
	}
	return 0, false
}
