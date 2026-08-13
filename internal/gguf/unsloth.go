package gguf

import (
	"path/filepath"
	"regexp"
	"strings"
)

// Unsloth Dynamic 2.0 GGUFs are named with a UD- prefix on the quant token
// (e.g. Qwen3-8B-UD-Q4_K_XL.gguf). The GGUF general.file_type is still a
// standard llama.cpp type, so callers must overlay this label from the path
// or general.name. OpenInfer does not generate these files.
var unslothDynamicRe = regexp.MustCompile(`(?i)(?:^|[.\-_])UD-((?:IQ[1-4]_[A-Z0-9]+|Q[1-8](?:_[A-Z0-9]+)+|TQ[12]_0|MXFP4(?:_MOE)?))`)

// UnslothDynamicQuant returns "UD-Q4_K_XL" (etc.) when path or name is an
// Unsloth Dynamic 2.0 GGUF, otherwise "".
func UnslothDynamicQuant(path, name string) string {
	seen := map[string]bool{}
	for _, s := range []string{name, path, filepath.Base(path), filepath.Base(filepath.Dir(path))} {
		s = strings.TrimSpace(s)
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		if m := unslothDynamicRe.FindStringSubmatch(s); len(m) == 2 {
			return "UD-" + strings.ToUpper(m[1])
		}
	}
	return ""
}
