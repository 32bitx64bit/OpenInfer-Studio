// Package scalefold implements AWQ-style equivalent scaling for weight-only
// quantization: for each whitelisted projection feeding from a gated RMS
// norm (attention Q/K/V/fused-QKV sharing an attention norm; SwiGLU
// gate/up sharing an FFN norm), it picks per-input-channel scales s
// minimizing the projections' importance-weighted quantization error, then
// folds them into the GGUF payload as W' = W·diag(s) and g' = g/s — an
// exactly lossless transform in floating point that moves activation-scale
// difficulty into the much more boxable weight channels.
//
// ffn_down (post-nonlinearity input), attention output, embedding, and the
// output head are never folded: their inputs are not diagonal-gated norm
// outputs. Only float source payloads (F32/F16/BF16) are transformed; a
// consumer row that is already quantized skips the whole cluster.
//
// The transform rewrites both the source GGUF and the imatrix
// (per-row in_sum2 entries divided by s_r², matching the activation
// rescale), so every downstream stage — quantization anchors, the exact
// loss table, KLD search — sees a consistent folded model.
package scalefold

import (
	"sort"
	"strings"

	"quantlab/core"
	"quantlab/profile"
)

// Cluster is one norm tensor with its whitelisted consumers, the chosen
// alpha, and the final per-channel scales.
type Cluster struct {
	Norm      string     `json:"norm"`
	Consumers []string   `json:"consumers"`
	Alpha     float64    `json:"alpha"`
	Scales    []float32  `json:"scales"`
	ErrBefore float64    `json:"errBefore,omitempty"`
	ErrAfter  float64    `json:"errAfter,omitempty"`
	Probe     core.DType `json:"probeDType"`
	Skipped   string     `json:"skipped,omitempty"`
}

// AlphaGrid is the search set for the fold exponent. 0.0 is the no-op fold,
// so a cluster never comes out worse than unfolded.
var AlphaGrid = []float64{0, 0.25, 0.5, 0.75, 1.0}

// scaleClamp bounds individual scales so degenerate channels cannot
// explode the norm gain.
const scaleClamp = 64.0

// maxClusterRows bounds rows decoded per consumer during alpha selection.
const maxClusterRows = 1 << 12

// layerPrefix extracts the "blk.N" style layer key (or "" for non-layer
// tensors) from a GGUF tensor name.
func layerPrefix(name string) string {
	for _, pre := range []string{"blk.", "layers.", "layer."} {
		if i := strings.Index(name, pre); i >= 0 {
			rest := name[i+len(pre):]
			j := strings.IndexByte(rest, '.')
			if j >= 0 {
				return pre + rest[:j]
			}
		}
	}
	return ""
}

// isNormTensor reports a gated RMS norm weight: a 1-D float tensor whose
// local name advertises norm(orm) status.
func isNormTensor(t core.TensorDesc) bool {
	if len(t.Shape) != 1 || !t.DType.IsFloat() {
		return false
	}
	low := strings.ToLower(t.Name)
	if !strings.Contains(low, "norm") {
		return false
	}
	return strings.HasSuffix(low, ".weight")
}

// clusterKind identifies which norm family a name belongs to: "attn" feeds
// attention projections, "ffn"/"mlp" feed SwiGLU gate/up.
func clusterKind(name string) string {
	low := strings.ToLower(name)
	switch {
	case strings.Contains(low, "attn"), strings.Contains(low, "attention"):
		return "attn"
	case strings.Contains(low, "ffn"), strings.Contains(low, "mlp"):
		return "ffn"
	}
	// Neutral names (norm1, ln_1, ...): cannot attribute safely.
	return ""
}

// whitelistedConsumer matches the AWQ-safe projection set by name (model
// agnostic, spanning llama/qwen/mistral styles).
func whitelistedConsumer(name string) bool {
	low := strings.ToLower(name)
	if strings.Contains(low, "norm") || strings.Contains(low, "embd") ||
		strings.Contains(low, "output.weight") || strings.Contains(low, "lm_head") {
		return false
	}
	switch {
	case strings.Contains(low, "attn_q."), strings.HasSuffix(low, "attn_q.weight"),
		strings.Contains(low, "attn_k."), strings.HasSuffix(low, "attn_k.weight"),
		strings.Contains(low, "attn_v."), strings.HasSuffix(low, "attn_v.weight"),
		strings.Contains(low, "attn_qkv"), strings.Contains(low, "qkv_proj"),
		strings.Contains(low, "q_proj"), strings.Contains(low, "k_proj"),
		strings.Contains(low, "v_proj"):
		return true
	}
	switch {
	case strings.Contains(low, "ffn_gate"), strings.Contains(low, "ffn_up"),
		strings.Contains(low, "gate_proj"), strings.Contains(low, "up_proj"),
		strings.HasSuffix(low, ".w1.weight"), strings.HasSuffix(low, ".w3.weight"):
		return !strings.Contains(low, "gate_inp") && !strings.Contains(low, "_exps")
	}
	return !strings.Contains(low, "gate_inp") && strings.Contains(low, "up_exps")
}

// consumerKind maps a whitelisted consumer to its norm family.
func consumerKind(name string) string {
	low := strings.ToLower(name)
	if strings.Contains(low, "attn_") || strings.Contains(low, "qkv") ||
		strings.Contains(low, "q_proj") || strings.Contains(low, "k_proj") ||
		strings.Contains(low, "v_proj") {
		return "attn"
	}
	return "ffn"
}

// Discover finds fold clusters in the bank: for each 1-D norm weight with a
// name-attributable family, all same-layer whitelisted projections whose
// contiguous (input) dimension equals the norm length and whose payloads
// are float. A consumer pair missing its norm (or vice versa) simply yields
// no cluster.
//
// The result is independent of architecture details discovered by name
// alone; no per-model configuration is required.
func Discover(bank *core.TensorBank) []Cluster {
	if bank == nil {
		return nil
	}
	byLayer := map[string][]core.TensorDesc{}
	for _, t := range bank.Tensors {
		byLayer[layerPrefix(t.Name)] = append(byLayer[layerPrefix(t.Name)], t)
	}
	var out []Cluster
	var keys []string
	for k := range byLayer {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		ts := byLayer[k]
		var norms []core.TensorDesc
		consumers := map[string][]core.TensorDesc{}
		for _, t := range ts {
			if isNormTensor(t) {
				norms = append(norms, t)
			} else if len(t.Shape) == 2 && t.DType.IsFloat() && whitelistedConsumer(t.Name) {
				c := consumerKind(t.Name)
				if c != "" {
					consumers[c] = append(consumers[c], t)
				}
			}
		}
		for _, n := range norms {
			kind := clusterKind(n.Name)
			if kind == "" {
				continue
			}
			var cl Cluster
			cl.Norm = n.Name
			for _, c := range consumers[kind] {
				if c.Shape[0] != n.Shape[0] {
					continue
				}
				cl.Consumers = append(cl.Consumers, c.Name)
			}
			if len(cl.Consumers) == 0 {
				continue
			}
			sort.Strings(cl.Consumers)
			out = append(out, cl)
		}
	}
	return out
}

// channelImportanceOne builds the per-input-channel importance vector for
// one consumer from its imatrix entries: llama-imatrix records, for every
// row of a projection, the chunk-coarse activation powers of that
// projection's input vector, so the mean across rows reduces row-wise
// sampling noise. Chunk granularity coarser than channels is spread
// uniformly over covered channels. Nil when the entry lacks vector data.
func channelImportanceOne(st profile.ImatrixStats, ne0, rows uint64) []float64 {
	if ne0 == 0 || rows == 0 || len(st.Values) == 0 {
		return nil
	}
	rowChunks := uint64(len(st.Values)) / rows
	if rowChunks == 0 {
		return nil
	}
	sum := make([]float64, ne0)
	hits := make([]float64, ne0)
	sz := ne0 / rowChunks
	if sz == 0 {
		sz = 1
	}
	nr := rows
	if uint64(len(st.Values))/rowChunks < nr {
		nr = uint64(len(st.Values)) / rowChunks
	}
	for r := uint64(0); r < nr; r++ {
		for c := uint64(0); c < rowChunks; c++ {
			v := float64(st.Values[r*rowChunks+c])
			for j := c * sz; j < (c+1)*sz && j < ne0; j++ {
				sum[j] += v
				hits[j]++
			}
		}
	}
	var total float64
	for i := range sum {
		if hits[i] > 0 {
			sum[i] /= hits[i]
		}
		total += sum[i]
	}
	if total <= 0 {
		return nil
	}
	return sum
}

// ChannelImportance pools per-consumer importance vectors for the cluster,
// averaging with weights proportional to consumer row counts.
func ChannelImportance(imatrix map[string]profile.ImatrixStats, cl Cluster, shapes map[string]core.TensorDesc) []float64 {
	var sum []float64
	var weight []float64
	touched := false
	for _, name := range cl.Consumers {
		t, ok := shapes[name]
		if !ok || len(t.Shape) != 2 {
			continue
		}
		st, ok := imatrix[name]
		if !ok {
			continue
		}
		v := channelImportanceOne(st, t.Shape[0], t.Shape[1])
		if v == nil {
			continue
		}
		if sum == nil {
			sum = make([]float64, len(v))
			weight = make([]float64, len(v))
		}
		for i := range sum {
			sum[i] += v[i]
			weight[i]++
		}
		touched = true
	}
	if !touched {
		return nil
	}
	for i := range sum {
		if weight[i] > 0 {
			sum[i] /= weight[i]
		}
	}
	// Normalize to mean 1 so the scale search sees relative importance.
	var mean float64
	for _, v := range sum {
		mean += v
	}
	mean /= float64(len(sum))
	if mean <= 0 {
		return nil
	}
	for i := range sum {
		sum[i] /= mean
	}
	return sum
}
