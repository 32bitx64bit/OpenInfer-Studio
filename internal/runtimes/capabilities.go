package runtimes

import (
	"regexp"
	"strings"
)

// knownFlags maps llama.cpp CLI flags to OpenInfer capability identifiers.
// Only flags present in a runtime's --help output are considered supported,
// and only supported flags are ever passed to that runtime.
var knownFlags = map[string]string{
	"--ctx-size":              "ctx-size",
	"--n-gpu-layers":          "gpu-layers",
	"--threads":               "threads",
	"--threads-batch":         "threads-batch",
	"--flash-attn":            "flash-attn",
	"--parallel":              "parallel",
	"--batch-size":            "batch-size",
	"--ubatch-size":           "ubatch-size",
	"--cache-type-k":          "cache-type-k",
	"--cache-type-v":          "cache-type-v",
	"--no-mmap":               "no-mmap",
	"--mlock":                 "mlock",
	"--numa":                  "numa",
	"--main-gpu":              "main-gpu",
	"--device":                "device",
	"--split-mode":            "split-mode",
	"--tensor-split":          "tensor-split",
	"--cont-batching":         "cont-batching",
	"--no-cont-batching":      "no-cont-batching",
	"--rope-scaling":          "rope-scaling",
	"--rope-freq-base":        "rope-freq-base",
	"--rope-freq-scale":       "rope-freq-scale",
	"--swa-full":              "swa-full",
	"--kv-unified":            "kv-unified",
	"--no-kv-unified":         "no-kv-unified",
	"--kv-offload":            "kv-offload",
	"--no-kv-offload":         "no-kv-offload",
	"--op-offload":            "op-offload",
	"--no-op-offload":         "no-op-offload",
	"--cpu-moe":               "cpu-moe",
	"--n-cpu-moe":             "n-cpu-moe",
	"--prio":                  "prio",
	"--poll":                  "poll",
	"--fit":                   "fit",
	"--no-warmup":             "no-warmup",
	"--warmup":                "warmup",
	"--alias":                 "alias",
	"--mmproj":                "mmproj",
	"--no-mmproj":             "no-mmproj",
	"--no-mmproj-offload":     "no-mmproj-offload",
	"--chat-template":         "chat-template",
	"--chat-template-file":    "chat-template-file",
	"--jinja":                 "jinja",
	"--lora":                  "lora",
	"--lora-scaled":           "lora-scaled",
	"--model-draft":           "model-draft",
	"--spec-draft-model":      "model-draft",
	"--draft-max":             "draft-max",
	"--spec-draft-n-max":      "draft-max",
	"--draft-min":             "draft-min",
	"--spec-draft-n-min":      "draft-min",
	"--spec-type":             "spec-type",
	"--sleep-idle-seconds":    "sleep-idle-seconds",
	"--embedding":             "embedding",
	"--embeddings":            "embedding",
	"--pooling":               "pooling",
	"--reranking":             "reranking",
	"--api-key":               "api-key",
	"--host":                  "host",
	"--port":                  "port",
	"--timeout-read":          "timeout-read",
	"--timeout-write":         "timeout-write",
	"--defrag-thold":          "defrag-thold",
	"--log-file":              "log-file",
	"--verbose":               "verbose",
	"--grammar":               "grammar",
	"--json-schema":           "json-schema",
	"--no-webui":              "no-webui",
	"--slots":                 "slots",
	"--props":                 "props",
	"--metrics":               "metrics",
	"--cache-reuse":           "cache-reuse",
	"--media-path":            "media-path",
	"--chat-template-kwargs":  "chat-template-kwargs",
	"--reasoning":             "reasoning",
	"--reasoning-format":      "reasoning-format",
	"--reasoning-budget":      "reasoning-budget",
	"--reasoning-preserve":    "reasoning-preserve",
	"--no-reasoning-preserve": "no-reasoning-preserve",
	"--speculative-draft":     "speculative-draft",
}

// flagRe finds long flags anywhere in help text (help formats vary wildly
// across llama.cpp builds: "-c, --ctx-size N", "  --ctx-size N", etc).
var flagRe = regexp.MustCompile(`(--[a-z0-9][a-z0-9\-]+)`)

// removedRe marks help lines for flags that still appear in --help but error
// when used ("--draft-max N ... the argument has been removed"). Merely
// DEPRECATED flags (e.g. --mlock) still work, so only hard removals match.
var removedRe = regexp.MustCompile(`(?i)has been removed|no longer (supported|available)`)

// flagRemoved reports whether flag appears in the definition column of a
// help line declaring it removed. The description column is excluded because
// removal notes name the replacement ("use --spec-draft-n-max instead").
func flagRemoved(help, flag string) bool {
	for _, line := range strings.Split(help, "\n") {
		idx := strings.Index(line, flag)
		if idx < 0 {
			continue
		}
		// The removal notice must come after the flag itself. Replacement
		// mentions ("use --spec-draft-n-max instead") appear after the
		// notice, so they are not misread as removed.
		loc := removedRe.FindStringIndex(line)
		if loc != nil && loc[0] > idx {
			return true
		}
	}
	return false
}

// ParseCapabilities extracts supported capability identifiers from
// `llama-server --help` output. Unknown flags are ignored conservatively;
// flags documented as removed are excluded even though they appear in help.
func ParseCapabilities(help string) []string {
	seen := map[string]bool{}
	var out []string
	for _, m := range flagRe.FindAllStringSubmatch(help, -1) {
		flag := m[1]
		if flagRemoved(help, flag) {
			continue
		}
		if cap, ok := knownFlags[flag]; ok && !seen[cap] {
			seen[cap] = true
			out = append(out, cap)
		}
	}
	return out
}

var descColumnRe = regexp.MustCompile(`\s{2,}|\t`)

// valueTokenAfter returns the text between flag and the description column
// on this help line ("" when the flag is absent or ends the definition).
// llama.cpp help pads between the definition and description with 2+ spaces;
// the definition column may itself contain single spaces and aliases.
func valueTokenAfter(line, flag string) string {
	idx := strings.Index(line, flag)
	if idx < 0 {
		return ""
	}
	rest := line[idx+len(flag):]
	if cut := descColumnRe.FindStringIndex(rest); cut != nil {
		rest = rest[:cut[0]]
	}
	return strings.TrimSpace(rest)
}

// FlagTakesValue inspects the raw --help text to decide whether a flag
// expects a value. llama.cpp changed --flash-attn from a boolean switch to
// "--flash-attn [on|off|auto]", so this must be discovered per runtime.
//
// Rule: within the flag-definition column, a token after the flag starting
// with '[', '<', '{' or written as an ALL-CAPS placeholder (N, FNAME, TYPE…)
// means the flag takes a value. The description column is ignored, since it
// contains all-caps words like DEPRECATED.
func FlagTakesValue(help, flag string) bool {
	for _, line := range strings.Split(help, "\n") {
		rest := valueTokenAfter(line, flag)
		if rest == "" {
			continue
		}
		next := strings.Fields(rest)[0]
		if strings.HasPrefix(next, "[") || strings.HasPrefix(next, "<") || strings.HasPrefix(next, "{") {
			return true
		}
		trimmed := strings.TrimRight(next, ",;.")
		if len(trimmed) > 0 && trimmed == strings.ToUpper(trimmed) {
			return true
		}
	}
	return false
}

// SupportsFlag reports whether the raw CLI flag is known and supported by a
// runtime with the given capability list. Capability rows are snapshotted at
// install time, so newly recognized flags may be missing from old snapshots —
// when that happens we fall back to the runtime's captured --help text.
func SupportsFlag(caps []string, help, flag string) bool {
	if flagRemoved(help, flag) {
		return false
	}
	cap, known := knownFlags[flag]
	if known {
		for _, c := range caps {
			if c == cap {
				return true
			}
		}
		if help != "" && strings.Contains(help, flag) {
			return true
		}
		return false
	}
	// Unknown to us: permit only if the runtime's own help mentions it.
	return strings.Contains(help, flag)
}
