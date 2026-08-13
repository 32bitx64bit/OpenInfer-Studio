package quantize

import (
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/gguf"
)

// AdaptivePlan is a mixed-precision assignment labeled OpenInfer Adaptive.
type AdaptivePlan struct {
	Label       string            `json:"label"`
	TargetBPW   float64           `json:"target_bpw"`
	TargetBytes int64             `json:"target_bytes"`
	Assignments map[string]string `json:"assignments"` // tensor name → ggml type
	Estimated   int64             `json:"estimated_bytes"`
}

var blkIndexRe = regexp.MustCompile(`blk\.(\d+)\.`)

func adaptiveTargetBPW(preset string, explicit float64) float64 {
	if explicit > 0 {
		return explicit
	}
	switch strings.ToLower(strings.TrimSpace(preset)) {
	case "quality":
		return 6.2
	case "compact":
		return 3.9
	default:
		return 4.9
	}
}

func tensorImportance(name string, nLayers int) float64 {
	lname := strings.ToLower(name)
	score := 1.0
	switch {
	case strings.Contains(lname, "token_embd") || strings.Contains(lname, "tok_embd"):
		score = 10
	case strings.Contains(lname, "output.weight") || strings.HasSuffix(lname, "output"):
		score = 10
	case strings.Contains(lname, "attn_v") || strings.Contains(lname, "attn.v"):
		score = 8
	case strings.Contains(lname, "ffn_down") || strings.Contains(lname, "ffn.down") || strings.Contains(lname, "w2"):
		score = 7
	case strings.Contains(lname, "attn_out") || strings.Contains(lname, "attn_output") || strings.Contains(lname, "attn.o"):
		score = 6.5
	case strings.Contains(lname, "attn_q") || strings.Contains(lname, "attn_k"):
		score = 5
	case strings.Contains(lname, "ffn_gate") || strings.Contains(lname, "ffn_up") || strings.Contains(lname, "exps"):
		score = 3
	case strings.Contains(lname, "norm"):
		score = 9
	}
	if m := blkIndexRe.FindStringSubmatch(lname); len(m) == 2 && nLayers > 0 {
		i, _ := strconv.Atoi(m[1])
		if i <= 1 || i >= nLayers-2 {
			score += 2
		}
	}
	return score
}

func paletteForPreset(preset string) []string {
	switch strings.ToLower(strings.TrimSpace(preset)) {
	case "quality":
		return []string{"q8_0", "q6_k", "q5_k"}
	case "compact":
		return []string{"q5_k", "q4_k", "q3_k", "q2_k"}
	default:
		return []string{"q6_k", "q5_k", "q4_k", "q3_k"}
	}
}

func ggmlBPW(t string) float64 {
	switch strings.ToLower(t) {
	case "q8_0":
		return 8.5
	case "q6_k":
		return 6.5625
	case "q5_k":
		return 5.5
	case "q4_k":
		return 4.5
	case "q3_k":
		return 3.4375
	case "q2_k":
		return 2.625
	case "f16", "bf16":
		return 16
	default:
		return 4.5
	}
}

func minTypeForName(name string) string {
	lname := strings.ToLower(name)
	if strings.Contains(lname, "token_embd") || strings.Contains(lname, "output.weight") || strings.Contains(lname, "norm") {
		return "q5_k"
	}
	return ""
}

type scoredTensor struct {
	gguf.Tensor
	Score float64
}

// PlanAdaptive assigns per-tensor ggml types to hit a size/bpw budget.
// This is OpenInfer Adaptive, not Unsloth Dynamic 2.0.
func PlanAdaptive(path string, preset string, targetBPW float64, targetBytes int64, stats map[string]float64) (*AdaptivePlan, error) {
	tensors, md, err := gguf.ListTensors(path)
	if err != nil {
		return nil, err
	}
	nLayers := int(md.BlockCount)
	var params uint64
	for _, t := range tensors {
		params += t.Elements
	}
	bpw := adaptiveTargetBPW(preset, targetBPW)
	budget := targetBytes
	if budget <= 0 && params > 0 {
		budget = int64(float64(params) * bpw / 8)
	}
	pal := paletteForPreset(preset)
	scored := make([]scoredTensor, 0, len(tensors))
	for _, t := range tensors {
		s := tensorImportance(t.Name, nLayers)
		if stats != nil {
			if z, ok := stats[t.Name]; ok && z > 0 {
				s += z
			}
		}
		scored = append(scored, scoredTensor{Tensor: t, Score: s})
	}
	sort.Slice(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })

	assign := map[string]string{}
	// Start everyone at the lowest palette type, then upgrade important tensors.
	low := pal[len(pal)-1]
	for _, t := range scored {
		typ := low
		if min := minTypeForName(t.Name); min != "" && ggmlBPW(min) > ggmlBPW(typ) {
			typ = min
		}
		assign[t.Name] = typ
	}
	est := func() int64 {
		var n int64
		for _, t := range scored {
			n += int64(float64(t.Elements) * ggmlBPW(assign[t.Name]) / 8)
		}
		return n
	}
	for _, t := range scored {
		cur := assign[t.Name]
		for i := 0; i < len(pal); i++ {
			if pal[i] == cur {
				break
			}
			next := pal[i]
			if ggmlBPW(next) <= ggmlBPW(cur) {
				continue
			}
			assign[t.Name] = next
			if budget > 0 && est() > budget {
				assign[t.Name] = cur
				break
			}
			cur = next
			break // upgrade one step at a time per tensor, then continue to next tensor
		}
	}
	// Second pass: keep upgrading highest-score tensors while under budget.
	changed := true
	for changed {
		changed = false
		for _, t := range scored {
			cur := assign[t.Name]
			idx := indexOf(pal, cur)
			if idx <= 0 {
				continue
			}
			next := pal[idx-1]
			assign[t.Name] = next
			if budget > 0 && est() > budget {
				assign[t.Name] = cur
				continue
			}
			changed = true
		}
	}

	label := "OpenInfer Adaptive"
	switch strings.ToLower(strings.TrimSpace(preset)) {
	case "quality":
		label = "OpenInfer Adaptive Quality"
	case "compact":
		label = "OpenInfer Adaptive Compact"
	default:
		label = "OpenInfer Adaptive Balanced"
	}
	return &AdaptivePlan{
		Label: label, TargetBPW: bpw, TargetBytes: budget,
		Assignments: assign, Estimated: est(),
	}, nil
}

func indexOf(ss []string, v string) int {
	for i, s := range ss {
		if s == v {
			return i
		}
	}
	return -1
}

func writeTensorTypeFile(path string, assign map[string]string) error {
	var b strings.Builder
	names := make([]string, 0, len(assign))
	for n := range assign {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		fmt.Fprintf(&b, "%s=%s\n", n, assign[n])
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

var zdRe = regexp.MustCompile(`(?i)(\S+)\s+.*\bzd(?:\s*score)?\s*[:=]?\s*([0-9.]+)`)

// ParseIMatrixStats extracts per-tensor ZD-like scores from --show-statistics output.
func ParseIMatrixStats(text string) map[string]float64 {
	out := map[string]float64{}
	for _, line := range strings.Split(text, "\n") {
		m := zdRe.FindStringSubmatch(line)
		if len(m) != 3 {
			continue
		}
		v, err := strconv.ParseFloat(m[2], 64)
		if err != nil {
			continue
		}
		out[m[1]] = v
	}
	return out
}
