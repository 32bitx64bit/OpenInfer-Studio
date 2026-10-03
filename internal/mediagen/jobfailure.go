package mediagen

import (
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// sd-server reports a failed job with a bare message ("generate_video returned
// no results"); the reason is only in its log. These helpers read the part of
// the log written while the job ran and put the cause into the job's error.

var (
	// "[WARN   ] model_manager.cpp:1914 - model manager cannot make enough memory
	// available on ROCm0: need 8640.31 MB device / 8129.17 MB budget, available
	// 4654.00 MB device / 4617.55 MB budget". sd.cpp's "MB" is MiB.
	noMemoryRe = regexp.MustCompile(`cannot make enough memory available on (\S+): need ([\d.]+) MB device / [\d.]+ MB budget, available ([\d.]+) MB device`)
	// "model manager memory on ROCm0: reported free 4654.00 MB / total 16368.00 MB,
	// tracked weights 11028.45 MB / other runtime …"
	trackedWeightsRe = regexp.MustCompile(`tracked weights ([\d.]+) MB`)
	// "video.cpp:1612 - generate_video 832x480x243"
	generateVideoRe = regexp.MustCompile(`generate_video (\d+)x(\d+)x(\d+)`)
	// "[ERROR  ] video.cpp:1709 - sampling failed after 0.81s"
	// ggml uses a logger name instead of a source location:
	// "[ERROR  ] ggml - alloc_tensor_range: failed to allocate ROCm0 buffer ...".
	logErrorRe         = regexp.MustCompile(`^\[ERROR\s*\]\s+.+?\s+-\s+(.+)$`)
	bufferAllocationRe = regexp.MustCompile(`failed to allocate (\S+) buffer of size (\d+)`)
	oomRe              = regexp.MustCompile(`(?i)out of memory|failed to allocate|alloc(?:ate|ation)? (?:of )?.*(?:failed|error)`)
)

// logSize is the current size of a log file, 0 when it cannot be read.
func logSize(path string) int64 {
	if path == "" {
		return 0
	}
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return fi.Size()
}

// readLogFrom returns what was appended to path after offset (at most the last
// 512 KiB of it). A log that shrank is read from the start.
func readLogFrom(path string, offset int64) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	size := logSize(path)
	if offset < 0 || offset > size {
		offset = 0
	}
	if size-offset > 512<<10 {
		offset = size - 512<<10
	}
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return ""
	}
	b, _ := io.ReadAll(f)
	return string(b)
}

func mibToGiB(mb string) float64 {
	var v float64
	_, _ = fmt.Sscanf(mb, "%f", &v)
	return v / 1024
}

// explainJobFailure returns msg with the cause found in the log the server
// wrote during the job. A message that already says something specific, and a
// log with nothing in it, come back unchanged.
func explainJobFailure(logSegment, msg string) string {
	lines := strings.FieldsFunc(logSegment, func(r rune) bool { return r == '\n' || r == '\r' })

	var noMem []string
	var weights, request string
	var errs []string
	oomLine := ""
	for _, line := range lines {
		line = strings.TrimSpace(ansiEscapeRe.ReplaceAllString(line, ""))
		if m := noMemoryRe.FindStringSubmatch(line); m != nil {
			noMem = m
		}
		if m := trackedWeightsRe.FindStringSubmatch(line); m != nil {
			weights = m[1]
		}
		if m := generateVideoRe.FindStringSubmatch(line); m != nil {
			request = fmt.Sprintf("%s×%s, %s frames", m[1], m[2], m[3])
		}
		if m := logErrorRe.FindStringSubmatch(line); m != nil {
			errs = append(errs, m[1])
			if oomLine == "" && oomRe.MatchString(m[1]) {
				oomLine = m[1]
			}
		}
	}

	if noMem != nil {
		var b strings.Builder
		fmt.Fprintf(&b, "Ran out of GPU memory on %s", noMem[1])
		if request != "" {
			fmt.Fprintf(&b, " (%s)", request)
		}
		fmt.Fprintf(&b, ": a step needs about %.1f GiB of working memory but only %.1f GiB is free", mibToGiB(noMem[2]), mibToGiB(noMem[3]))
		if weights != "" {
			fmt.Fprintf(&b, ", because the model's weights already take %.1f GiB", mibToGiB(weights))
		}
		b.WriteString(". Use fewer frames or a lower resolution, or load the model with \"Offload weights to CPU (slow, saves VRAM)\" turned on.")
		return b.String() + " (sd-server: " + msg + ")"
	}
	if oomLine != "" {
		if allocation := bufferAllocationRe.FindStringSubmatch(oomLine); allocation != nil {
			bytes, _ := strconv.ParseUint(allocation[2], 10, 64)
			kind := "GPU"
			if strings.EqualFold(allocation[1], "cpu") {
				kind = "system"
			}
			cause := fmt.Sprintf("Ran out of %s memory on %s while allocating %.1f MiB", kind, allocation[1], float64(bytes)/(1<<20))
			if request != "" {
				cause += " (" + request + ")"
			}
			if strings.Contains(logSegment, "failed during output caching") || strings.Contains(logSegment, "failed to capture graph cut tensor") {
				cause += ". The allocation was for an intermediate tensor cache; offloading model weights alone does not guarantee enough working memory"
			}
			if kind == "GPU" {
				return cause + ". Reload with more VRAM headroom (a lower Max VRAM budget), or use fewer frames / lower resolution. (sd-server: " + msg + ")"
			}
			return cause + ". Free system RAM or use a smaller request. (sd-server: " + msg + ")"
		}
		return "Ran out of memory: " + oomLine + ". Use a smaller request or leave more working-memory headroom when loading the model. (sd-server: " + msg + ")"
	}
	// A bare message with errors in the log: show the last of them.
	if len(errs) > 0 && isBareJobError(msg) {
		if len(errs) > 3 {
			errs = errs[len(errs)-3:]
		}
		return msg + " (sd-server log: " + strings.Join(errs, "; ") + ")"
	}
	return msg
}

// isBareJobError reports whether msg is one of sd-server's contentless
// failure messages.
func isBareJobError(msg string) bool {
	m := strings.ToLower(msg)
	return m == "generation failed" || strings.Contains(m, "returned no results")
}
