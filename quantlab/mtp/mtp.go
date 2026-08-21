// Package mtp detects fused Multi-Token Prediction / NextN tensors in a
// trunk GGUF so tiny (Q2-class) quants can omit them. Dedicated MTP
// sidecar files are left untouched.
package mtp

import (
	"strconv"
	"strings"

	"quantlab/core"
	"quantlab/tensorbank"
)

// DropBelowBPW is the exclusive upper bound at which fused MTP heads are
// stripped. Unsloth Dynamic 3.0 keeps MTP on UD-Q2_K_XL (~2.9 bpw) and
// drops it under that class.
const DropBelowBPW = 2.75

// ShouldDrop reports whether a job at bpw should omit fused MTP / NextN
// tensors. Non-positive bpw is a no-op so Q4 jobs with only a byte budget
// must compute an implied bpw first.
func ShouldDrop(bpw float64) bool {
	return bpw > 0 && bpw < DropBelowBPW
}

// ImpliedBPW is the bits-per-weight a byte budget would buy across every
// tensor in bank (the inverse of profile.BPWToBudget). Returns 0 when the
// estimate is not defined.
func ImpliedBPW(bank *core.TensorBank, budgetBytes uint64) float64 {
	if bank == nil || budgetBytes == 0 {
		return 0
	}
	var elems uint64
	for _, t := range bank.Tensors {
		elems += t.Elements
	}
	if elems == 0 {
		return 0
	}
	return float64(budgetBytes) * 8.0 / float64(elems)
}

// Select returns the fused NextN / MTP tensor names that should be omitted
// from a tiny quant. ok is false when the file has no fused MTP, is a
// dedicated sidecar/draft, or dropping would empty the tensor set.
func Select(bank *core.TensorBank, kvs []tensorbank.KV, arch string) (drop []string, ok bool) {
	if bank == nil || len(bank.Tensors) == 0 {
		return nil, false
	}
	arch = strings.TrimSpace(arch)
	if arch == "" {
		arch = kvString(kvs, "general.architecture")
	}
	nextn := NextnPredictLayers(kvs, arch)
	blockCount := BlockCount(kvs, arch)

	var names []string
	hasNextnName := false
	for _, t := range bank.Tensors {
		names = append(names, t.Name)
		if isNextnName(t.Name) {
			hasNextnName = true
		}
	}
	if nextn == 0 && !hasNextnName {
		return nil, false
	}
	if blockCount > 0 && !hasTrunkLayer(names, blockCount) {
		return nil, false
	}

	seen := make(map[string]struct{})
	for _, name := range names {
		if isNextnName(name) {
			seen[name] = struct{}{}
			continue
		}
		if nextn > 0 && blockCount > 0 {
			if n, ok := layerIndex(name); ok && n >= uint64(blockCount) {
				seen[name] = struct{}{}
			}
		}
	}
	if len(seen) == 0 || len(seen) >= len(names) {
		return nil, false
	}
	drop = make([]string, 0, len(seen))
	for _, name := range names {
		if _, ok := seen[name]; ok {
			drop = append(drop, name)
		}
	}
	return drop, true
}

// NextnPredictLayers reads {arch}.nextn_predict_layers (and aliases) from
// GGUF KV. Reimplemented here so quantlab does not import internal/gguf.
func NextnPredictLayers(kvs []tensorbank.KV, arch string) uint32 {
	arch = strings.TrimSpace(arch)
	keys := make([]string, 0, 3)
	if arch != "" {
		keys = append(keys, arch+".nextn_predict_layers")
	}
	keys = append(keys, "nextn_predict_layers", "general.nextn_predict_layers")
	for _, k := range keys {
		if n, ok := kvUint32(kvs, k); ok {
			return n
		}
	}
	for _, kv := range kvs {
		lk := strings.ToLower(kv.Key)
		if strings.HasSuffix(lk, ".nextn_predict_layers") || lk == "nextn_predict_layers" {
			if n, ok := valueUint32(kv.Value); ok {
				return n
			}
		}
	}
	return 0
}

// BlockCount reads {arch}.block_count (trunk depth) from GGUF KV.
func BlockCount(kvs []tensorbank.KV, arch string) uint32 {
	arch = strings.TrimSpace(arch)
	if arch != "" {
		if n, ok := kvUint32(kvs, arch+".block_count"); ok {
			return n
		}
	}
	if n, ok := kvUint32(kvs, "block_count"); ok {
		return n
	}
	if n, ok := kvUint32(kvs, "general.block_count"); ok {
		return n
	}
	return 0
}

// ZeroNextnLayers sets existing nextn_predict_layers keys to 0. Keys that
// were never present are not added.
func ZeroNextnLayers(kvs []tensorbank.KV, arch string) []tensorbank.KV {
	arch = strings.TrimSpace(arch)
	keys := make([]string, 0, 4)
	if arch != "" {
		keys = append(keys, arch+".nextn_predict_layers")
	}
	keys = append(keys, "nextn_predict_layers", "general.nextn_predict_layers")
	out := kvs
	patched := make(map[string]struct{}, len(keys))
	for _, k := range keys {
		out = tensorbank.SetScalar(out, k, 0)
		patched[k] = struct{}{}
	}
	for _, kv := range out {
		if _, done := patched[kv.Key]; done {
			continue
		}
		lk := strings.ToLower(kv.Key)
		if strings.HasSuffix(lk, ".nextn_predict_layers") || lk == "nextn_predict_layers" {
			out = tensorbank.SetScalar(out, kv.Key, 0)
		}
	}
	return out
}

func isNextnName(name string) bool {
	n := strings.ToLower(name)
	return strings.Contains(n, "nextn") || strings.Contains(n, ".mtp.")
}

func hasTrunkLayer(names []string, blockCount uint32) bool {
	for _, name := range names {
		n, ok := layerIndex(name)
		if ok && n < uint64(blockCount) {
			return true
		}
	}
	return false
}

func layerIndex(name string) (uint64, bool) {
	parts := strings.Split(strings.ToLower(name), ".")
	for i := 0; i+1 < len(parts); i++ {
		if parts[i] != "blk" && parts[i] != "layers" {
			continue
		}
		n, err := strconv.ParseUint(parts[i+1], 10, 64)
		if err != nil {
			continue
		}
		return n, true
	}
	return 0, false
}

func kvString(kvs []tensorbank.KV, key string) string {
	for _, kv := range kvs {
		if kv.Key != key {
			continue
		}
		s, _ := kv.Value.Scalar.(string)
		return s
	}
	return ""
}

func kvUint32(kvs []tensorbank.KV, key string) (uint32, bool) {
	for _, kv := range kvs {
		if kv.Key == key {
			return valueUint32(kv.Value)
		}
	}
	return 0, false
}

func valueUint32(v tensorbank.Value) (uint32, bool) {
	switch n := v.Scalar.(type) {
	case uint8:
		return uint32(n), true
	case uint16:
		return uint32(n), true
	case uint32:
		return n, true
	case uint64:
		if n > uint64(^uint32(0)) {
			return 0, false
		}
		return uint32(n), true
	case int8:
		if n < 0 {
			return 0, false
		}
		return uint32(n), true
	case int16:
		if n < 0 {
			return 0, false
		}
		return uint32(n), true
	case int32:
		if n < 0 {
			return 0, false
		}
		return uint32(n), true
	case int64:
		if n < 0 || n > int64(^uint32(0)) {
			return 0, false
		}
		return uint32(n), true
	case float32:
		if n < 0 || n > float32(^uint32(0)) {
			return 0, false
		}
		return uint32(n), true
	case float64:
		if n < 0 || n > float64(^uint32(0)) {
			return 0, false
		}
		return uint32(n), true
	}
	return 0, false
}
