package quantize

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
)

func jsonUnmarshal(data []byte, dest any) error {
	if len(data) == 0 {
		return nil
	}
	return json.Unmarshal(data, dest)
}

func fileSize(path string) int64 {
	if path == "" {
		return 0
	}
	st, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return st.Size()
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	if _, err := out.ReadFrom(in); err != nil {
		return err
	}
	return out.Close()
}

// copyFileAtomic never exposes a partially copied publication artifact.
func copyFileAtomic(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), "."+filepath.Base(dst)+".tmp-")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.ReadFrom(in); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, dst)
}

func hashFile(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), n, nil
}

var unsafeNameRe = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

func safeName(s string) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, " ", "-")
	s = unsafeNameRe.ReplaceAllString(s, "-")
	s = regexp.MustCompile(`-{2,}`).ReplaceAllString(s, "-")
	s = strings.Trim(s, ".-_")
	if s == "" {
		s = "model"
	}
	if len(s) > 80 {
		s = s[:80]
	}
	return s
}

func sidecarFilePrefix(path string) string {
	base := filepath.Base(path)
	for _, p := range []string{"dflash-", "mtp-", "eagle3-", "dspark-"} {
		if len(base) >= len(p) && strings.EqualFold(base[:len(p)], p) {
			return base[:len(p)]
		}
	}
	return ""
}

func isSplitPath(path string) bool {
	base := filepath.Base(path)
	return regexp.MustCompile(`(?i)-\d{5}-of-\d{5}\.gguf$`).MatchString(base)
}

func lastNRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[len(r)-n:])
}

func printable(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsControl(r) && r != '\n' && r != '\t' {
			return -1
		}
		return r
	}, s)
}

var (
	llamaErrLineRe = regexp.MustCompile(`(?:^|\s)E\s+(\S+:\s*.+)$`)
	imatrixNeedRe  = regexp.MustCompile(`(?i)need at least (\d+) tokens`)
	imatrixHaveRe  = regexp.MustCompile(`(?i)tokenizes to only (\d+) tokens`)
	tensorShapeRe  = regexp.MustCompile(`(?i)tensor '([^']+)' has wrong shape;\s*expected\s+([0-9, ]+),\s*got\s+([0-9, ]+)`)
)

func llamaErrorLines(tail string) string {
	var errs []string
	seen := map[string]bool{}
	for _, line := range strings.Split(tail, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		rest := ""
		if m := llamaErrLineRe.FindStringSubmatch(line); len(m) >= 2 {
			rest = strings.TrimSpace(m[1])
		} else {
			lower := strings.ToLower(line)
			if !strings.Contains(lower, "failed to quantize") &&
				!strings.Contains(lower, "failed to create context") &&
				!strings.HasPrefix(lower, "error:") {
				continue
			}
			rest = line
			if _, msg, ok := strings.Cut(rest, ": "); ok && strings.TrimSpace(msg) != "" {
				rest = strings.TrimSpace(msg)
			}
		}
		if rest == "" || seen[rest] {
			continue
		}
		seen[rest] = true
		errs = append(errs, rest)
	}
	if len(errs) == 0 {
		return ""
	}
	if len(errs) > 4 {
		errs = errs[len(errs)-4:]
	}
	return strings.Join(errs, " ")
}

func friendlyToolError(msg string) string {
	need := imatrixNeedRe.FindStringSubmatch(msg)
	have := imatrixHaveRe.FindStringSubmatch(msg)
	if len(need) == 2 && len(have) == 2 {
		return fmt.Sprintf("Calibration text is too short for llama-imatrix (need at least %s tokens; this file tokenized to %s). Use a larger calibration file.", need[1], have[1])
	}
	if len(need) == 2 {
		return fmt.Sprintf("Calibration text is too short for llama-imatrix (need at least %s tokens). Use a larger calibration file.", need[1])
	}
	lower := strings.ToLower(msg)
	if strings.Contains(lower, "dflash requires ctx_other") {
		return "llama-imatrix cannot initialize this speculative assistant GGUF without the main model."
	}
	if m := tensorShapeRe.FindStringSubmatch(msg); len(m) == 4 {
		name := m[1]
		exp, got := compactDims(m[2]), compactDims(m[3])
		if strings.Contains(name, "token_embd") || strings.Contains(name, "output") {
			return fmt.Sprintf("%s shape does not match the architecture vocab (expected %s, file has %s). Embedding rows, output rows, and tokenizer length must agree — reconvert from the Hugging Face weights.", name, exp, got)
		}
		return fmt.Sprintf("Tensor %s has the wrong shape for this architecture (expected %s, file has %s). The GGUF metadata does not match the weight layout.", name, exp, got)
	}
	if strings.Contains(lower, "check_tensor_dims") || strings.Contains(lower, "has wrong shape") {
		return "A GGUF tensor has the wrong shape for this architecture. Reconvert from the Hugging Face weights, or use a GGUF built for this llama.cpp version."
	}
	if strings.Contains(lower, "requantizing") && strings.Contains(lower, "disabled") {
		return "Source is already quantized. llama-quantize refused to requantize it. Start from F16/BF16/F32, or enable Allow requantize in Advanced for Q6 and below."
	}
	return ""
}

func compactDims(s string) string {
	fields := strings.Fields(strings.ReplaceAll(s, ",", " "))
	for len(fields) > 1 && fields[len(fields)-1] == "1" {
		fields = fields[:len(fields)-1]
	}
	return strings.Join(fields, "×")
}

func summarizeToolFailure(waitErr error, tail string) error {
	if waitErr == nil {
		return nil
	}
	if msg := llamaErrorLines(tail); msg != "" {
		if friendly := friendlyToolError(msg); friendly != "" {
			return fmt.Errorf("%s", friendly)
		}
		return fmt.Errorf("%w: %s", waitErr, printable(msg))
	}
	if strings.TrimSpace(tail) != "" {
		return fmt.Errorf("%w: %s", waitErr, printable(lastNRunes(tail, 280)))
	}
	return waitErr
}
