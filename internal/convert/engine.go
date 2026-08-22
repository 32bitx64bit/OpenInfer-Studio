package convert

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

type workKind string

const (
	kindCopy       workKind = "copy"
	kindTie        workKind = "tie"
	kindRMS        workKind = "rms"
	kindRMSPlus    workKind = "rms_plus"
	kindUnpermuteQ workKind = "unpermute_q"
	kindUnpermuteK workKind = "unpermute_k"
	kindNegExp     workKind = "neg_exp"
	kindConv1d     workKind = "conv1d"
	kindStack      workKind = "stack"
)

type workItem struct {
	GGUF    string
	Shape   []int64 // GGUF dims
	HFShape []int64 // HF dims after squeeze (for reorder)
	DType   int
	Src     *TensorRef
	Srcs    []TensorRef
	Kind    workKind
	Reorder string
	NHeads  int
	NExpert int
}

func convertFamily(f Family, dir string, tensors []TensorRef, cfg map[string]any, w *Writer) (*ConvertStats, error) {
	tok, err := loadTokenizer(dir)
	if err != nil {
		return nil, err
	}
	h := parseHyper(cfg, f)
	h.vocab = alignVocab(tok, tensors, h.vocab)
	weightDType := inferWeightDType(tensors)
	store, err := storeType(weightDType)
	if err != nil {
		return nil, err
	}
	fileType := uint32(1)
	if store == GGMLBF16 {
		fileType = 32
	}
	name := cfgString(cfg, "name")
	if name == "" {
		name = f.DefaultName
	}

	writeFamilyKV(w, f, name, cfg, h, fileType)
	pre := sniffTokenizerPre(tok, tokenizerFallback(f.GGUFArch))
	f.UnpermuteQK = shouldUnpermute(f.GGUFArch, pre)
	w.addTokenizer(tok, pre)

	stats := &ConvertStats{Architecture: f.GGUFArch, GGUFType: store}
	items, err := planFamilyWork(f, tensors, h, store, stats)
	if err != nil {
		return stats, err
	}

	for _, item := range items {
		if err := w.PlanTensor(item.GGUF, item.Shape, item.DType); err != nil {
			return stats, err
		}
	}
	if err := w.WriteHeader(); err != nil {
		return stats, err
	}
	elem := 2
	if strings.EqualFold(weightDType, "F32") {
		elem = 4
	}
	for _, item := range items {
		payload, err := familyPayload(item, weightDType, elem, h)
		if err != nil {
			return stats, err
		}
		if err := w.WriteTensor(payload); err != nil {
			return stats, err
		}
		stats.Tensors++
	}
	return stats, nil
}

func tokenizerFallback(arch string) string {
	switch {
	case strings.HasPrefix(arch, "qwen"):
		return "qwen2"
	case arch == "gemma3" || strings.HasPrefix(arch, "gemma3"):
		return "gemma3"
	case strings.HasPrefix(arch, "gemma"):
		return "gemma"
	case strings.HasPrefix(arch, "phi"):
		return "phi3"
	case arch == "llama4" || arch == "muse-glimmer":
		return "llama4"
	case arch == "llama" || arch == "internlm2":
		return "llama-bpe"
	default:
		return ""
	}
}

func shouldUnpermute(arch, pre string) bool {
	if strings.HasPrefix(arch, "qwen") || strings.HasPrefix(arch, "gemma") || strings.HasPrefix(arch, "phi") {
		return false
	}
	if pre == "llama-bpe" || pre == "llama4" {
		return true
	}
	switch arch {
	case "llama", "llama4", "internlm2":
		return true
	}
	return false
}

func planFamilyWork(f Family, tensors []TensorRef, h hyper, store int, stats *ConvertStats) ([]workItem, error) {
	var work []workItem
	haveOutput := false
	var embed *TensorRef
	stacked := map[string][]TensorRef{}

	for i := range tensors {
		t := tensors[i]
		m := f.MapName(t.Name)
		if m.Vision {
			stats.Skipped++
			if len(stats.Warnings) == 0 {
				stats.Warnings = append(stats.Warnings, "vision tensors were skipped; this GGUF is language-only (pair an existing mmproj for images)")
			}
			continue
		}
		if m.Skip {
			stats.Skipped++
			continue
		}
		if m.GGUF == "" {
			return nil, fmt.Errorf("cannot map tensor %q to %s GGUF (not convertible)", t.Name, f.GGUFArch)
		}
		if m.Kind == kindStack {
			stacked[m.GGUF] = append(stacked[m.GGUF], tensors[i])
			continue
		}
		if m.GGUF == "output.weight" {
			haveOutput = true
		}
		if m.GGUF == "token_embd.weight" {
			embed = &tensors[i]
		}
		hfShape := convPlanShape(m.Kind, t.Shape)
		if isVocabWeight(m.GGUF) {
			hfShape = padVocabLeading(hfShape, h.vocab)
		}
		item := workItem{
			GGUF:    m.GGUF,
			Src:     &tensors[i],
			Kind:    m.Kind,
			Reorder: m.Reorder,
			HFShape: hfShape,
		}
		switch m.Kind {
		case kindUnpermuteQ:
			item.NHeads = h.nHead
		case kindUnpermuteK:
			item.NHeads = h.nKV
		}
		item.DType, item.Shape = familyStore(m.Kind, hfShape, store)
		work = append(work, item)
	}

	for ggufName, srcs := range stacked {
		hfShape, err := expertPlanShape(srcs, h.nExpert)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", ggufName, err)
		}
		work = append(work, workItem{
			GGUF:    ggufName,
			Shape:   ggufDims(hfShape),
			HFShape: hfShape,
			DType:   store,
			Srcs:    srcs,
			Kind:    kindStack,
			NExpert: h.nExpert,
		})
	}

	if !haveOutput && f.TieOutput {
		if embed == nil {
			return nil, fmt.Errorf("model has neither lm_head nor embed_tokens")
		}
		hf := padVocabLeading(embed.Shape, h.vocab)
		work = append(work, workItem{
			GGUF: "output.weight", Shape: ggufDims(hf), HFShape: hf, DType: store, Src: embed, Kind: kindTie,
		})
	}
	return work, nil
}

func convPlanShape(kind workKind, hf []int64) []int64 {
	if kind == kindConv1d && len(hf) == 3 {
		if hf[1] == 1 {
			return []int64{hf[0], hf[2]}
		}
		if hf[2] == 1 {
			return []int64{hf[0], hf[1]}
		}
	}
	return append([]int64(nil), hf...)
}

func familyStore(kind workKind, hfShape []int64, store int) (dtype int, shape []int64) {
	shape = ggufDims(hfShape)
	switch kind {
	case kindRMS, kindRMSPlus, kindNegExp:
		return GGMLF32, append([]int64(nil), hfShape...)
	case kindConv1d:
		// llama.cpp's SSM conv kernels read the conv weights as raw F32
		// (ggml asserts src1->nb[0] == sizeof(float) on CPU and CUDA).
		// Official qwen35-family GGUFs store ssm_conv1d as F32 even when
		// every other tensor is quantized; storing bf16 aborts or garbles.
		return GGMLF32, shape
	}
	if len(hfShape) <= 1 {
		return GGMLF32, append([]int64(nil), hfShape...)
	}
	return store, shape
}

func familyPayload(item workItem, srcDType string, elem int, h hyper) ([]byte, error) {
	if item.Kind == kindStack {
		raw, _, err := stackExperts(item.Srcs, item.NExpert)
		if err != nil {
			return nil, err
		}
		return convertPayload(raw, item.Srcs[0].DType, item.DType)
	}
	if item.Src == nil {
		return nil, fmt.Errorf("tensor %s has no source", item.GGUF)
	}
	raw, err := ReadPayload(*item.Src)
	if err != nil {
		return nil, err
	}
	hfShape := append([]int64(nil), item.Src.Shape...)
	srcElem := elemSize(item.Src.DType)
	if srcElem <= 0 {
		srcElem = elem
	}

	if isVocabWeight(item.GGUF) && item.Src != nil && len(item.HFShape) > 0 {
		padded, err := padLeadingDim(raw, item.Src.Shape, item.HFShape, srcElem)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", item.GGUF, err)
		}
		raw = padded
	}

	if item.Kind == kindConv1d {
		sq, data, err := squeezeConv(hfShape, raw)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", item.GGUF, err)
		}
		raw, hfShape = data, sq
	}

	if item.Reorder != "" {
		re, _, err := applyReorder(raw, hfShape, srcElem, item.Reorder, h)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", item.GGUF, err)
		}
		raw = re
	}

	switch item.Kind {
	case kindRMSPlus:
		f32, err := toF32(raw, item.Src.DType)
		if err != nil {
			return nil, err
		}
		return addOneF32(f32), nil
	case kindRMS:
		return toF32(raw, item.Src.DType)
	case kindNegExp:
		f32, err := toF32(raw, item.Src.DType)
		if err != nil {
			return nil, err
		}
		return negExpF32(f32), nil
	case kindUnpermuteQ, kindUnpermuteK:
		if len(item.Src.Shape) != 2 {
			return nil, fmt.Errorf("%s: expected rank-2", item.GGUF)
		}
		dim1 := int(item.Src.Shape[0])
		dim2 := int(item.Src.Shape[1])
		unp, err := unpermuteHF(raw, item.NHeads, dim1, dim2, elemSize(item.Src.DType))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", item.GGUF, err)
		}
		return convertPayload(unp, srcDType, item.DType)
	case kindCopy, kindTie, kindConv1d:
		if item.DType == GGMLF32 && strings.ToUpper(item.Src.DType) != "F32" {
			return toF32(raw, item.Src.DType)
		}
		return convertPayload(raw, item.Src.DType, item.DType)
	default:
		return nil, fmt.Errorf("unknown work kind %s", item.Kind)
	}
}

func negExpF32(src []byte) []byte {
	out := make([]byte, len(src))
	n := len(src) / 4
	for i := 0; i < n; i++ {
		f := math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(float32(-math.Exp(float64(f)))))
	}
	return out
}
