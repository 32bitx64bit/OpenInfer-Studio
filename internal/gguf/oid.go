package gguf

import (
	"path/filepath"
	"regexp"
	"strings"
)

// OpenInfer Dynamic GGUFs are named with an OID- prefix on the quant token
// (e.g. Qwen3-8B-OID-Q4_K_XL.gguf). The GGUF general.file_type is still a
// standard llama.cpp type, so callers overlay this label from the path or
// general.name — same pattern as Unsloth UD- files.
var openInferDynamicRe = regexp.MustCompile(`(?i)(?:^|[.\-_])OID-((?:IQ[1-4](?:_[A-Z0-9]+)+|Q[1-8](?:_[A-Z0-9]+)+|TQ[12]_0|MXFP4(?:_MOE)?))`)

// OpenInferDynamicQuant returns "OID-Q4_K_XL" (etc.) when path or name is an
// OpenInfer Dynamic GGUF, otherwise "".
func OpenInferDynamicQuant(path, name string) string {
	seen := map[string]bool{}
	for _, s := range []string{name, path, filepath.Base(path), filepath.Base(filepath.Dir(path))} {
		s = strings.TrimSpace(s)
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		if m := openInferDynamicRe.FindStringSubmatch(s); len(m) == 2 {
			return "OID-" + strings.ToUpper(m[1])
		}
	}
	return ""
}

// OverlayDynamicQuant prefers OpenInfer OID- then Unsloth UD- labels over
// the GGUF general.file_type, which cannot represent mixed-precision recipes.
func OverlayDynamicQuant(path, name, current string) string {
	if q := OpenInferDynamicQuant(path, name); q != "" {
		return q
	}
	if q := UnslothDynamicQuant(path, name); q != "" {
		return q
	}
	return current
}
