package tensorbank

import (
	"quantlab/core"
)

// DefaultAlignment is assumed when a bank lacks alignment metadata (older
// checkpoints); it is the GGUF default llama.cpp writes.
const DefaultAlignment uint64 = 32

// bankAlignment returns the bank's alignment, defaulting when unknown.
func bankAlignment(bank *core.TensorBank) uint64 {
	if bank == nil || bank.Alignment == 0 {
		return DefaultAlignment
	}
	return bank.Alignment
}

// PlannedArtifactSize computes the exact final size, in bytes, of the GGUF
// artifact that the assembler produces for manifest over the bank's source:
// fixed header + KV metadata + tensor-info section, padded to the source's
// general.alignment, plus every tensor payload with per-tensor alignment
// padding. This matches Assembler.Build's layout byte for byte, so
// BudgetBytes can be enforced against the true on-disk artifact size, not
// just the tensor payload sum.
//
// It returns false when the bank lacks KV metadata length (older
// checkpoints); callers then fall back to payload accounting plus a
// conservative reserve.
func PlannedArtifactSize(bank *core.TensorBank, m *core.SelectionManifest) (uint64, bool) {
	if bank == nil || m == nil {
		return 0, false
	}
	if bank.KVMetadataLen == 0 {
		return 0, false
	}
	payload := make(map[string]uint64, len(m.Options))
	for _, o := range m.Options {
		if _, dup := payload[o.TensorName]; dup {
			return 0, false
		}
		payload[o.TensorName] = o.Bytes
	}
	al := bankAlignment(bank)
	// Header + KV + tensor-info section, padded to alignment. The assembler
	// writes tensors in primary-source (bank) order, so the per-tensor
	// alignment walk mirrors it exactly.
	infos := tensorInfoSize(bank.Tensors)
	meta := 24 + bank.KVMetadataLen + infos
	dataStart := alignUp(meta, al)
	var cur uint64
	for _, t := range bank.Tensors {
		length, ok := payload[t.Name]
		if !ok {
			return 0, false
		}
		cur = alignUp(cur, al)
		cur += length
	}
	return dataStart + cur, true
}

// tensorInfoSize sums the serialized tensor-info descriptor sizes.
func tensorInfoSize(tensors []core.TensorDesc) uint64 {
	var n uint64
	for _, t := range tensors {
		n += 8 + uint64(len(t.Name)) + 4 + 8*uint64(len(t.Shape)) + 4 + 8
	}
	return n
}

// OverheadReserve returns a conservative upper bound on the TOTAL
// non-payload overhead of any artifact assembled from this bank: the fixed
// 24-byte header, the KV metadata section, the tensor-info section, pre-data
// padding (bounded by one alignment unit), and per-tensor alignment padding
// (each tensor contributes strictly less than one alignment unit). The
// solver and KLD search deduct this reserve from BudgetBytes so payload
// plans never overflow the final artifact budget; emit additionally
// hard-checks the exact artifact size (PlannedArtifactSize).
//
// When the bank lacks KV metadata length (older checkpoints) the reserve
// under-counts by the true KV size; such banks only arise from old
// checkpoints, and emit's exact check remains the hard backstop.
func OverheadReserve(bank *core.TensorBank) uint64 {
	if bank == nil || len(bank.Tensors) == 0 {
		return 0
	}
	al := bankAlignment(bank)
	kv := bank.KVMetadataLen
	meta := 24 + kv + tensorInfoSize(bank.Tensors)
	pad0 := alignUp(meta, al) - meta
	return meta + pad0 + uint64(len(bank.Tensors))*(al-1)
}
