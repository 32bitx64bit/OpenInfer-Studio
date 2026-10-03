package mediagen

import (
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Progress describes observed work, never an estimate based on elapsed time.
// Units keep weight-loading counters separate from completed sampling steps.
type Progress struct {
	JobID            string `json:"job_id"`
	Phase            string `json:"phase"`
	Message          string `json:"message"`
	Current          int    `json:"current,omitempty"`
	Total            int    `json:"total,omitempty"`
	Unit             string `json:"unit,omitempty"` // tensors | steps | tiles
	SamplingCurrent  int    `json:"sampling_current,omitempty"`
	SamplingTotal    int    `json:"sampling_total,omitempty"`
	ElapsedMS        int64  `json:"elapsed_ms"`
	QuietMS          int64  `json:"quiet_ms"`
	ServerResponding *bool  `json:"server_responding,omitempty"` // nil before the first native job poll
}

var (
	ansiEscapeRe = regexp.MustCompile(`\x1b\[[0-9;]*[A-Za-z]`)
	// sd.cpp util.cpp prints CR-delimited bars, including in redirected logs.
	// MB/s and GB/s count tensors. Iteration bars can describe either
	// sampling or VAE tiles, depending on the operation logged before them.
	workCounterRe     = regexp.MustCompile(`\|[^\r\n]*\|\s+(\d+)/(\d+)\s+-\s+[\d.]+\s*(GB/s|MB/s|s/it|it/s)`)
	generationStartRe = regexp.MustCompile(`generate_(?:video|image) \d+x\d+`)
)

type logProgress struct {
	path                string
	offset              int64
	partial             string
	started, lastOutput time.Time
	progress            Progress
	fatal               string
	decoding            bool
}

func newLogProgress(path string, offset int64) *logProgress {
	t := time.Now()
	return &logProgress{path: path, offset: offset, started: t, lastOutput: t,
		progress: Progress{Phase: "preparing", Message: "Preparing generation"}}
}

func (p *logProgress) phase(phase, message string) {
	if p.progress.Phase != phase {
		p.progress.Current, p.progress.Total, p.progress.Unit = 0, 0, ""
	}
	p.progress.Phase, p.progress.Message = phase, message
}

func (p *logProgress) line(line string) {
	line = strings.TrimSpace(ansiEscapeRe.ReplaceAllString(line, ""))
	if strings.Contains(line, "GGML_ASSERT") || strings.Contains(line, "GGML_ABORT") {
		p.fatal = line
		p.phase("failed", "Runtime assertion; generation cannot continue")
		return
	}
	if c := workCounterRe.FindStringSubmatch(line); c != nil {
		current, _ := strconv.Atoi(c[1])
		total, _ := strconv.Atoi(c[2])
		if total <= 0 || current < 0 || current > total {
			return
		}
		if c[3] == "MB/s" || c[3] == "GB/s" {
			p.phase("loading_weights", fmt.Sprintf("Loading weights: %d/%d tensors", current, total))
			p.progress.Unit = "tensors"
		} else if p.decoding {
			p.phase("decoding", fmt.Sprintf("Decoding: tile %d/%d", current, total))
			p.progress.Unit = "tiles"
		} else {
			p.phase("sampling", fmt.Sprintf("Sampling: step %d/%d", current, total))
			p.progress.Unit = "steps"
			p.progress.SamplingCurrent, p.progress.SamplingTotal = current, total
		}
		p.progress.Current, p.progress.Total = current, total
		return
	}
	switch {
	case strings.Contains(line, "generate_video completed"):
		p.phase("saving", "Encoding video")
	case strings.Contains(line, "generate_image completed"):
		p.phase("saving", "Saving output")
	case strings.Contains(line, "sampling completed"), strings.Contains(line, "generating latent video completed"),
		strings.Contains(line, "ltx_video_vae build cached graph cut plan"),
		strings.Contains(line, "decoding"), strings.Contains(line, "decode_first_stage"), strings.Contains(line, "vae decode"):
		p.decoding = true
		p.phase("decoding", "Decoding output")
	case strings.Contains(line, "get_learned_condition completed"):
		p.phase("preparing", "Text encoded; preparing sampling")
	case generationStartRe.MatchString(line):
		p.decoding = false
		p.progress.SamplingCurrent, p.progress.SamplingTotal = 0, 0
		p.phase("preparing", "Preparing sampling")
	case strings.Contains(line, "build cached graph cut plan"):
		if !p.decoding {
			p.phase("preparing", "Preparing compute graph")
		}
	case strings.Contains(line, "loading tensors completed"):
		if p.decoding {
			p.phase("decoding", "Decoding output")
		} else {
			p.phase("computing", "Weights loaded; computing")
		}
	case strings.Contains(line, "encoding video"), strings.Contains(line, "saving video"):
		p.phase("saving", "Encoding video")
	}
}

// read consumes new bytes only, preserving partial lines between polls. Long
// lines are bounded and log truncation restarts parsing rather than hanging.
func (p *logProgress) read() {
	f, err := os.Open(p.path)
	if err != nil {
		return
	}
	defer f.Close()
	if st, err := f.Stat(); err == nil && st.Size() < p.offset {
		p.offset, p.partial = 0, ""
	}
	if _, err := f.Seek(p.offset, io.SeekStart); err != nil {
		return
	}
	b, _ := io.ReadAll(io.LimitReader(f, 256<<10))
	if len(b) == 0 {
		return
	}
	p.offset += int64(len(b))
	p.lastOutput = time.Now()
	text := p.partial + string(b)
	last := strings.LastIndexAny(text, "\r\n")
	if last >= 0 {
		for _, line := range strings.FieldsFunc(text[:last], func(r rune) bool { return r == '\r' || r == '\n' }) {
			p.line(line)
		}
		text = text[last+1:]
	}
	// A bar's current update is often never newline-terminated until the
	// operation finishes. Parse it now, while retaining partial byte writes.
	p.line(text)
	if len(text) > 16<<10 {
		text = text[len(text)-(16<<10):]
	}
	p.partial = text
}

func (p *logProgress) snapshot(jobID, status string) Progress {
	p.read()
	if status == "queued" {
		p.phase("queued", "Waiting in server queue")
	} else if status == "completed" {
		p.phase("saving", "Saving generated output")
	}
	v := p.progress
	// Offloaded weights are loaded between sampler steps. Keep the last
	// observed step count visible while those tensor bars replace the
	// current-operation counter.
	if v.SamplingTotal > 0 && !p.decoding && (v.Phase == "loading_weights" || v.Phase == "computing" || v.Phase == "preparing") {
		v.Message = fmt.Sprintf("Sampling: step %d/%d; %s", v.SamplingCurrent, v.SamplingTotal, v.Message)
	}
	v.JobID = jobID
	v.ElapsedMS = time.Since(p.started).Milliseconds()
	v.QuietMS = time.Since(p.lastOutput).Milliseconds()
	responding := status != "unresponsive"
	v.ServerResponding = &responding
	return v
}

func runtimeFailure(logText, logPath string) string {
	logText = latestStartupLog(logText)
	for _, line := range strings.FieldsFunc(logText, func(r rune) bool { return r == '\r' || r == '\n' }) {
		if strings.Contains(line, "GGML_ASSERT") || strings.Contains(line, "GGML_ABORT") {
			cause := "sd-server aborted during generation: " + strings.TrimSpace(line)
			if strings.Contains(line, "binbcast.cu") && strings.Contains(line, "sizeof(src1_t)") {
				cause = "The GPU backend rejected a tensor layout during generation (binbcast.cu broadcast assertion).\n" + strings.TrimSpace(line)
			}
			if logPath != "" {
				cause += "\nFull log: " + logPath
			}
			return cause
		}
	}
	return "sd-server exited unexpectedly:\n" + cleanLogTail(logText, 12)
}

func coreDumpFailure(logPath string) string {
	return "sd-server crashed and Linux is writing a core dump. Generation has stopped; the runtime is being terminated.\nFull log: " + logPath
}

// abortServer reacts to a runtime assertion or a crash confirmed by the OS.
// Silence alone is not evidence of a crash. Stop the affected process group
// promptly rather than leave its GPU allocations alive while gdb/core dumping
// runs, and retain the failure in the server list for inspection.
func (m *Manager) abortServer(modelID string, port int, reason string) {
	m.mu.Lock()
	sv := m.servers[modelID]
	if sv == nil || sv.port != port {
		m.mu.Unlock()
		return
	}
	sv.ready, sv.state, sv.errMsg, sv.port = false, ServerFailed, reason, 0
	sv.updatedAt = now()
	h := sv.handle
	m.mu.Unlock()
	if h != nil {
		_ = h.KillTree()
	}
	m.log.Error("sd-server fatal failure", "model_id", modelID, "error", reason)
	m.publish("media.server_error", map[string]any{"model_id": modelID, "error": reason})
	m.publish("media.server_state", map[string]any{"model_id": modelID, "state": ServerFailed, "error": reason})
	m.failJobsForModel(modelID, reason)
}
