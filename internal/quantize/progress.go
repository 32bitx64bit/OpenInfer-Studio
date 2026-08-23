package quantize

import (
	"errors"
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

var errTerminalLineTooLong = errors.New("terminal progress line is too long")

var (
	quantizeProgRe = regexp.MustCompile(`\[\s*(\d+)\s*/\s*(\d+)\s*\]`)
	imatrixChunkRe = regexp.MustCompile(`(?i)(?:chunk|chunks)\s+(\d+)\s*/\s*(\d+)`)
	imatrixSaveRe  = regexp.MustCompile(`(?i)saved.*\b(\d+)\s+chunks`)
	imatrixTotalRe = regexp.MustCompile(`(?i)computing over\s+(\d+)\s+chunks`)
	secondsPassRe  = regexp.MustCompile(`(?i)([0-9]+(?:\.[0-9]+)?)\s+seconds per pass`)
)

// Progress is the event payload for quant.progress.
type Progress struct {
	ID              string  `json:"id"`
	State           string  `json:"state"`
	Stage           string  `json:"stage"`
	Current         int     `json:"current"`
	Total           int     `json:"total"`
	Progress        float64 `json:"progress"`
	StageProgress   float64 `json:"stage_progress"`
	StageETASeconds int64   `json:"stage_eta_seconds,omitempty"`
	ETASeconds      int64   `json:"eta_seconds,omitempty"`
	Message         string  `json:"message,omitempty"`
	Estimated       bool    `json:"estimated,omitempty"`
}

type commandProgressSample struct {
	Current    int
	Total      int
	Progress   float64
	ETASeconds int64
	Message    string
	Estimated  bool
}

// commandProgressTracker combines explicit tool counters with llama-imatrix's
// one-shot seconds-per-pass estimate. Its Estimate method advances long stages
// even when the tool only redraws an unparseable terminal spinner.
type commandProgressTracker struct {
	mu             sync.Mutex
	started        time.Time
	sampleAt       time.Time
	current        int
	total          int
	secondsPerUnit float64
	message        string
}

func newCommandProgressTracker(started time.Time) *commandProgressTracker {
	return &commandProgressTracker{started: started, sampleAt: started}
}

func (t *commandProgressTracker) Observe(line string, at time.Time) (commandProgressSample, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	line = strings.TrimSpace(line)
	changed := false
	if m := imatrixTotalRe.FindStringSubmatch(line); len(m) == 2 {
		if total, err := strconv.Atoi(m[1]); err == nil && total > 0 {
			t.total = total
			changed = true
		}
	}
	if m := secondsPassRe.FindStringSubmatch(line); len(m) == 2 {
		if seconds, err := strconv.ParseFloat(m[1], 64); err == nil && seconds > 0 {
			t.secondsPerUnit = seconds
			if t.current == 0 {
				t.current = 1
			}
			t.sampleAt = at
			changed = true
		}
	}
	if current, total, ok := ParseProgressLine(line); ok {
		if total > 0 {
			t.total = total
		}
		if current >= 0 {
			if current > t.current {
				elapsed := at.Sub(t.sampleAt).Seconds()
				delta := current - t.current
				if t.current == 0 {
					elapsed = at.Sub(t.started).Seconds()
				}
				if elapsed > 0 && delta > 0 {
					observed := elapsed / float64(delta)
					if t.secondsPerUnit == 0 {
						t.secondsPerUnit = observed
					} else {
						t.secondsPerUnit = 0.7*t.secondsPerUnit + 0.3*observed
					}
				}
			}
			t.current = current
			t.sampleAt = at
		}
		changed = true
	}
	if line != "" && changed {
		t.message = line
	}
	return t.estimateLocked(at), changed
}

func (t *commandProgressTracker) Estimate(at time.Time) (commandProgressSample, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.total <= 0 {
		return commandProgressSample{}, false
	}
	return t.estimateLocked(at), true
}

func (t *commandProgressTracker) estimateLocked(at time.Time) commandProgressSample {
	current := float64(t.current)
	estimated := false
	if t.secondsPerUnit > 0 && current < float64(t.total) {
		current += math.Max(0, at.Sub(t.sampleAt).Seconds()) / t.secondsPerUnit
		if current >= float64(t.total) {
			current = math.Nextafter(float64(t.total), 0)
		}
		estimated = true
	}
	progress := 0.0
	eta := int64(0)
	if t.total > 0 {
		progress = clampProgress(current / float64(t.total))
		if t.secondsPerUnit > 0 {
			eta = int64(math.Ceil((float64(t.total) - current) * t.secondsPerUnit))
		}
	}
	return commandProgressSample{
		Current: int(math.Floor(current)), Total: t.total, Progress: progress,
		ETASeconds: eta, Message: t.message, Estimated: estimated,
	}
}

func clampProgress(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// stageRange maps stage-local progress to the full pipeline. The weights are
// time-oriented estimates, so overall ETA remains useful for imatrix-heavy
// quantlab jobs while every stage still exposes its own independent 0..1 fraction.
func stageRange(j *Job, stage string) (float64, float64) {
	quantlab := j.Kind == KindAdaptiveQuantize || usesQuantlab(j.Request)
	fromHF := j.Kind == KindFromHF
	if fromHF {
		switch stage {
		case "probe":
			return 0, 0.01
		case "download":
			return 0.01, 0.25
		case "convert":
			return 0.25, 0.40
		case "scan":
			return 0.40, 0.41
		}
	}
	base := 0.0
	if fromHF {
		base = 0.41
	}
	if j.Kind == KindIMatrix {
		return 0, 1
	}
	if j.Kind == KindCombineIMatrix {
		return 0, 1
	}
	if quantlab {
		scale := func(start, end float64) (float64, float64) {
			return base + (1-base)*start, base + (1-base)*end
		}
		switch stage {
		case "repair_source":
			return scale(0, 0.02)
		case "imatrix":
			return scale(0.02, 0.05)
		case "analyze":
			return scale(0.05, 0.10)
		case "anchor":
			return scale(0.10, 0.15)
		case "solve":
			return scale(0.15, 0.20)
		case "quantize":
			return scale(0.20, 0.65)
		case "validate":
			return scale(0.65, 0.92)
		case "search":
			return scale(0.92, 0.93)
		case "finalize":
			return scale(0.93, 1.0)
		case "quantize_projector":
			return scale(0.95, 0.98)
		case "quantize_draft":
			return scale(0.98, 0.995)
		}
	}
	if stage == "imatrix" {
		return base, 0.82
	}
	if stage == "quantize" {
		if j.Request.GenerateIMatrix {
			return 0.82, 0.96
		}
		return base, 0.96
	}
	if stage == "quantize_projector" {
		return 0.96, 0.98
	}
	if stage == "quantize_draft" {
		return 0.98, 0.995
	}
	return -1, -1
}

func overallETA(stageETA int64, overall, stageEnd float64) int64 {
	if stageETA <= 0 || overall >= 1 || stageEnd <= overall {
		return stageETA
	}
	eta := float64(stageETA) * (1 - overall) / (stageEnd - overall)
	if eta > 7*24*60*60 {
		eta = 7 * 24 * 60 * 60
	}
	return int64(math.Ceil(eta))
}

// StageText maps a job-facing stage name to a human-readable label. Dynamic
// (adaptive) jobs build per-dtype anchors during the shared "quantize" stage,
// so that label is specialized by kind; every other quantlab stage has a
// unique label so a long quiet stage still tells the user what is happening.
func StageText(kind, stage string) string {
	switch stage {
	case "imatrix":
		return "Building importance matrix"
	case "combine_imatrix":
		return "Combining importance matrices"
	case "repair_source":
		return "Repairing source tensors"
	case "probe":
		return "Inspecting source"
	case "download":
		return "Downloading source"
	case "convert":
		return "Converting to GGUF"
	case "scan":
		return "Scanning model"
	case "analyze":
		return "Analyzing source tensors"
	case "anchor":
		return "Planning anchors"
	case "solve":
		return "Solving bit allocation"
	case "evaluate", "validate":
		return "Validating against source (KLD)"
	case "search":
		return "Preparing output"
	case "emit", "finalize":
		return "Publishing model"
	case "quantize":
		if kind == KindAdaptiveQuantize {
			return "Quantizing anchors"
		}
		return "Quantizing weights"
	case "quantize_projector":
		return "Quantizing projector"
	case "quantize_draft":
		return "Quantizing draft model"
	case "validate_baseline":
		return "Evaluating baseline"
	case "validate_candidate":
		return "Evaluating quantized model"
	case "":
		return "Running"
	}
	return strings.ReplaceAll(stage, "_", " ")
}

// stepCounter derives a current/total step pair from the count of discrete
// StageProgress callbacks and the stage-local fraction they reported. quantlab
// reports each anchor build as (i+1)/M, so total = round(count / frac).
func stepCounter(count int, frac float64) (current, total int) {
	if count <= 0 {
		return 0, 0
	}
	if frac <= 0 {
		return count, 0
	}
	total = int(math.Round(float64(count) / frac))
	if total < count {
		total = count
	}
	return count, total
}

// remapQuantizeProgress splits the quantize stage's two sub-phases — discrete
// per-dtype anchor builds, then continuous candidate assembly — across the 0..1
// stage fraction. Without this, the monotonic clamp in emitQuantlabProgress
// hides the (often long) assembly phase behind a completed anchor bar.
func remapQuantizeProgress(message string, frac float64) float64 {
	if strings.HasPrefix(message, "anchor") {
		return 0.5 * clampProgress(frac)
	}
	return 0.5 + 0.5*clampProgress(frac)
}

// measurementStageFraction advances the evaluate/search stage bar per measured
// candidate so long optimization runs do not appear frozen. Assembly
// byte-progress fills the first sliver; each fresh measurement advances the bar
// toward 0.9, leaving headroom until the stage actually completes.
func measurementStageFraction(measurements int, assemblyFrac float64) float64 {
	return measurementStageFractionBudget(measurements, 7, assemblyFrac)
}

func measurementStageFractionBudget(measurements, budget int, assemblyFrac float64) float64 {
	frac := 0.05 + 0.10*clampProgress(assemblyFrac)
	if measurements > 0 && budget > 0 {
		ratio := float64(measurements) / float64(budget)
		if ratio > 1 {
			ratio = 1
		}
		if v := 0.2 + 0.7*ratio; v > frac {
			frac = v
		}
	}
	if frac > 0.9 {
		frac = 0.9
	}
	return frac
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

// splitTerminalLines accepts both newline logs and carriage-return terminal
// redraws used by several llama.cpp progress reporters.
func splitTerminalLines(data []byte, atEOF bool) (advance int, token []byte, err error) {
	for i, b := range data {
		if b != '\n' && b != '\r' {
			continue
		}
		advance = i + 1
		for advance < len(data) && (data[advance] == '\n' || data[advance] == '\r') {
			advance++
		}
		return advance, data[:i], nil
	}
	if atEOF && len(data) > 0 {
		return len(data), data, nil
	}
	if len(data) >= 1<<20 {
		return 0, nil, errTerminalLineTooLong
	}
	return 0, nil, nil
}
