package quantize

import (
	"regexp"
	"strconv"
	"strings"
)

var (
	quantizeProgRe = regexp.MustCompile(`\[\s*(\d+)\s*/\s*(\d+)\s*\]`)
	imatrixChunkRe = regexp.MustCompile(`(?i)(?:chunk|chunks)\s+(\d+)\s*/\s*(\d+)`)
	imatrixSaveRe  = regexp.MustCompile(`(?i)saved.*\b(\d+)\s+chunks`)
)

// Progress is the event payload for quant.progress.
type Progress struct {
	ID       string  `json:"id"`
	State    string  `json:"state"`
	Stage    string  `json:"stage"`
	Current  int     `json:"current"`
	Total    int     `json:"total"`
	Progress float64 `json:"progress"`
	Message  string  `json:"message,omitempty"`
}

// ParseProgressLine extracts a 0–1 fraction from llama-quantize / llama-imatrix logs.
func ParseProgressLine(line string) (current, total int, ok bool) {
	if m := quantizeProgRe.FindStringSubmatch(line); len(m) == 3 {
		c, _ := strconv.Atoi(m[1])
		t, _ := strconv.Atoi(m[2])
		if t > 0 {
			return c, t, true
		}
	}
	if m := imatrixChunkRe.FindStringSubmatch(line); len(m) == 3 {
		c, _ := strconv.Atoi(m[1])
		t, _ := strconv.Atoi(m[2])
		if t > 0 {
			return c, t, true
		}
	}
	if m := imatrixSaveRe.FindStringSubmatch(line); len(m) == 2 {
		c, _ := strconv.Atoi(m[1])
		if c > 0 {
			return c, 0, true
		}
	}
	_ = strings.TrimSpace(line)
	return 0, 0, false
}

func fraction(current, total int, fallback float64) float64 {
	if total > 0 {
		f := float64(current) / float64(total)
		if f < 0 {
			return 0
		}
		if f > 1 {
			return 1
		}
		return f
	}
	return fallback
}
