package gguf

import (
	"encoding/json"
	"regexp"
	"strings"
)

// Reasoning styles taken from GGUF chat templates (Jinja kwargs).
const (
	ReasoningEffort   = "reasoning_effort"   // gpt-oss / Harmony
	ReasoningStrength = "reasoning_strength" // Muse Glimmer
	EnableThinking    = "enable_thinking"    // Qwen3, Gemma 4, GLM
	Thinking          = "thinking"           // DeepSeek V3.x, Kimi
)

const EffortOff = "off"

// Canonical discrete levels, weakest → strongest. "off"/"none" are disable
// tokens; "on" is the boolean-on stand-in.
var knownEfforts = []string{
	"off", "none", "minimal", "low", "medium", "high", "xhigh", "max", "on",
}

var (
	identEffort   = regexp.MustCompile(`\breasoning_effort\b`)
	identStrength = regexp.MustCompile(`\breasoning_strength\b`)
	identEnable   = regexp.MustCompile(`\benable_thinking\b`)
	identThinking = regexp.MustCompile(`(?i)(?:thinking\s+is\s+defined|\bif\s+thinking\b|\bset\s+thinking\b)`)
	// Template kwargs llama.cpp's --reasoning-preserve aliases onto.
	identPreserveReasoning = regexp.MustCompile(`\bpreserve_reasoning\b`)
	identPreserveThinking  = regexp.MustCompile(`\bpreserve_thinking\b`)
	identClearThinking     = regexp.MustCompile(`\bclear_thinking\b`)
	identTruncateThinking  = regexp.MustCompile(`\btruncate_history_thinking\b`)
	identDropThinking      = regexp.MustCompile(`\bdrop_thinking\b`)
	quotedEffort           = regexp.MustCompile(`['"](off|none|minimal|low|medium|high|xhigh|max)['"]`)
	defaultEffort          = regexp.MustCompile(`(?i)(?:else|defaults?\s+to)\s*['"](off|none|minimal|low|medium|high|xhigh|max)['"]`)
	setDefault             = regexp.MustCompile(`(?i)set\s+(reasoning_effort|reasoning_strength)\s*=\s*['"](off|none|minimal|low|medium|high|xhigh|max)['"]`)
	enableIsFalse          = regexp.MustCompile(`\benable_thinking\b[^.\n]{0,80}\bis\s+false\b`)
	enableIfTruthy         = regexp.MustCompile(`\bif\s+enable_thinking\b(?:\s+is\s+defined\s+and\s+enable_thinking)?\s*(?:%}|-?%})`)
	slashLevels            = regexp.MustCompile(`(?i)\b(low)\s*/\s*(medium)\s*/\s*(high)(?:\s*/\s*(xhigh))?`)
)

// Reasoning is the compact, UI-facing description of how a GGUF lets callers
// control thinking. Empty Style means the template has no control (or no
// thinking at all) — hide the picker.
type Reasoning struct {
	// Style is the primary chat_template_kwargs key.
	Style string `json:"style,omitempty"`
	// Efforts are the values the UI/API should offer, including "off" when
	// the template can disable thinking. Order is weakest → strongest.
	Efforts []string `json:"efforts,omitempty"`
	// Default is the effort to use when the caller omits one. Muse Glimmer
	// is forced to "low" so chat can stream an answer before max_tokens.
	Default string `json:"default_effort,omitempty"`
	// CanDisable is true when "off" is a real template path, not just the
	// weakest effort.
	CanDisable bool `json:"can_disable,omitempty"`
	// Toggle is an extra boolean kwarg (enable_thinking / thinking) that
	// gates reasoning on effort-style templates that also support off.
	Toggle string `json:"toggle,omitempty"`
	// CanPreserve is true when the template can keep prior-turn reasoning
	// in the prompt (preserve_thinking / preserve_reasoning / clear_thinking).
	CanPreserve bool `json:"can_preserve,omitempty"`
}

// Controllable reports whether the chat UI should show a reasoning picker.
func (r Reasoning) Controllable() bool {
	return r.Style != "" && len(r.Efforts) > 0
}

// Allows reports whether effort is one of the advertised values ("off" and
// "none" are interchangeable when disable is supported).
func (r Reasoning) Allows(effort string) bool {
	e := normalizeEffort(effort)
	if e == "" {
		return false
	}
	for _, v := range r.Efforts {
		if normalizeEffort(v) == e {
			return true
		}
	}
	return false
}

// DetectReasoning sniffs a Jinja chat template (and architecture fallbacks)
// for reasoning controls. The model supplies the levels — we only advertise
// what the template actually understands, plus "off" when thinking can be
// turned off.
func DetectReasoning(template, arch string) Reasoning {
	if IsMuseGlimmerChat(arch) {
		r := detectFromTemplate(template)
		if r.Style != ReasoningStrength {
			r.Style = ReasoningStrength
			if len(r.Efforts) == 0 {
				r.Efforts = []string{"low", "medium", "high", "xhigh"}
			}
		}
		r.CanDisable = false
		r.Toggle = ""
		r.Efforts = dropDisable(r.Efforts)
		if len(r.Efforts) == 0 {
			r.Efforts = []string{"low", "medium", "high", "xhigh"}
		}
		r.Default = "low"
		return r
	}
	return detectFromTemplate(template)
}

func detectFromTemplate(template string) Reasoning {
	tmpl := template
	if tmpl == "" {
		return Reasoning{}
	}

	hasEffort := identEffort.MatchString(tmpl)
	hasStrength := identStrength.MatchString(tmpl)
	hasEnable := identEnable.MatchString(tmpl)
	hasThinking := identThinking.MatchString(tmpl)
	preserve := detectPreserve(tmpl)

	var r Reasoning
	switch {
	case hasStrength:
		r.Style = ReasoningStrength
		r.Efforts = extractEfforts(tmpl, ReasoningStrength)
		if len(r.Efforts) == 0 {
			r.Efforts = []string{"low", "medium", "high", "xhigh"}
		}
		r.Default = extractDefault(tmpl, ReasoningStrength)
		if r.Default == "" {
			r.Default = "high"
		}
	case hasEffort:
		r.Style = ReasoningEffort
		r.Efforts = extractEfforts(tmpl, ReasoningEffort)
		if len(r.Efforts) == 0 {
			r.Efforts = []string{"low", "medium", "high"}
		}
		r.Default = extractDefault(tmpl, ReasoningEffort)
		if r.Default == "" {
			r.Default = "medium"
		}
		if hasEnable {
			r.Toggle = EnableThinking
			r.CanDisable = true
		}
	case hasEnable:
		r.Style = EnableThinking
		r.Toggle = EnableThinking
		r.CanDisable = true
		r.Efforts = []string{EffortOff, "on"}
		r.Default = "on"
		if enableDefaultOff(tmpl) {
			r.Default = EffortOff
		}
	case hasThinking:
		r.Style = Thinking
		r.Toggle = Thinking
		r.CanDisable = true
		r.Efforts = []string{EffortOff, "on"}
		r.Default = "on"
	default:
		if preserve {
			return Reasoning{CanPreserve: true}
		}
		return Reasoning{}
	}

	if containsEffort(r.Efforts, "none") || containsEffort(r.Efforts, "off") {
		r.CanDisable = true
	}
	r.Efforts = normalizeEffortList(r.Efforts, r.CanDisable)
	if r.Default == "none" {
		r.Default = EffortOff
	}
	if r.Default == "" || !r.Allows(r.Default) {
		r.Default = firstNonOff(r.Efforts)
	}
	r.CanPreserve = preserve
	return r
}

func detectPreserve(tmpl string) bool {
	return identPreserveReasoning.MatchString(tmpl) ||
		identPreserveThinking.MatchString(tmpl) ||
		identClearThinking.MatchString(tmpl) ||
		identTruncateThinking.MatchString(tmpl) ||
		identDropThinking.MatchString(tmpl)
}

func extractEfforts(tmpl, ident string) []string {
	var found []string
	seen := map[string]bool{}
	add := func(v string) {
		v = strings.ToLower(strings.TrimSpace(v))
		if v == "" || seen[v] || !isKnownEffort(v) {
			return
		}
		seen[v] = true
		found = append(found, v)
	}

	windows := identWindows(tmpl, ident)
	if len(windows) == 0 {
		windows = []string{tmpl}
	}
	for _, w := range windows {
		for _, m := range quotedEffort.FindAllStringSubmatch(w, -1) {
			add(m[1])
		}
		if m := slashLevels.FindStringSubmatch(w); m != nil {
			add(m[1])
			add(m[2])
			add(m[3])
			if m[4] != "" {
				add(m[4])
			}
		}
	}
	ordered := orderEfforts(found)
	// A single quoted token is almost always the default ("medium"), not
	// an enumeration. Family fallbacks fill the real set.
	if len(ordered) < 2 {
		return nil
	}
	return ordered
}

func extractDefault(tmpl, ident string) string {
	windows := identWindows(tmpl, ident)
	if len(windows) == 0 {
		windows = []string{tmpl}
	}
	for _, w := range windows {
		if m := setDefault.FindStringSubmatch(w); m != nil && m[1] == ident {
			return strings.ToLower(m[2])
		}
		if m := defaultEffort.FindStringSubmatch(w); m != nil {
			return strings.ToLower(m[1])
		}
	}
	if m := defaultEffort.FindStringSubmatch(tmpl); m != nil {
		return strings.ToLower(m[1])
	}
	return ""
}

func identWindows(tmpl, ident string) []string {
	re := regexp.MustCompile(`\b` + regexp.QuoteMeta(ident) + `\b`)
	idxs := re.FindAllStringIndex(tmpl, -1)
	if idxs == nil {
		return nil
	}
	out := make([]string, 0, len(idxs))
	for _, idx := range idxs {
		start := idx[0] - 160
		if start < 0 {
			start = 0
		}
		end := idx[1] + 220
		if end > len(tmpl) {
			end = len(tmpl)
		}
		out = append(out, tmpl[start:end])
	}
	return out
}

func enableDefaultOff(tmpl string) bool {
	// Qwen: `{% if enable_thinking is defined and enable_thinking is false %}`
	// means missing → thinking on. Gemma: `{% if enable_thinking %}` means
	// missing → thinking off.
	if enableIsFalse.MatchString(tmpl) {
		return false
	}
	return enableIfTruthy.MatchString(tmpl)
}

func isKnownEffort(v string) bool {
	for _, k := range knownEfforts {
		if k == v {
			return true
		}
	}
	return false
}

func containsEffort(list []string, want string) bool {
	want = normalizeEffort(want)
	for _, v := range list {
		if normalizeEffort(v) == want {
			return true
		}
	}
	return false
}

func orderEfforts(in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, k := range knownEfforts {
		for _, v := range in {
			if strings.ToLower(v) == k && !seen[k] {
				seen[k] = true
				out = append(out, k)
			}
		}
	}
	return out
}

func dropDisable(in []string) []string {
	var out []string
	for _, v := range in {
		switch normalizeEffort(v) {
		case EffortOff, "on":
			continue
		default:
			out = append(out, v)
		}
	}
	return orderEfforts(out)
}

func normalizeEffortList(in []string, canDisable bool) []string {
	var levels []string
	hasNone := false
	for _, v := range in {
		switch n := normalizeEffort(v); n {
		case EffortOff:
			hasNone = true
		case "on":
			// keep for boolean styles; drop from effort-style lists
			if len(in) <= 2 {
				levels = append(levels, n)
			}
		default:
			levels = append(levels, n)
		}
	}
	levels = orderEfforts(levels)
	if canDisable || hasNone {
		if !containsEffort(levels, EffortOff) {
			levels = append([]string{EffortOff}, levels...)
		}
	}
	return levels
}

func firstNonOff(efforts []string) string {
	for _, v := range efforts {
		if normalizeEffort(v) != EffortOff {
			return v
		}
	}
	if len(efforts) > 0 {
		return efforts[0]
	}
	return ""
}

func normalizeEffort(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "none", "false", "no", "0":
		return EffortOff
	case "true", "yes", "1":
		return "on"
	case "extra", "extra-high", "extra_high", "highest":
		return "xhigh"
	default:
		return s
	}
}

// ReasoningFromMetadata reads a library metadata JSON blob, falling back to
// architecture defaults (Muse Glimmer) when the compact reasoning object is
// missing — e.g. a library that has not been rescanned yet.
func ReasoningFromMetadata(meta json.RawMessage, arch string) Reasoning {
	var wrap struct {
		Reasoning Reasoning `json:"reasoning"`
	}
	if len(meta) > 0 {
		_ = json.Unmarshal(meta, &wrap)
	}
	if wrap.Reasoning.Controllable() || wrap.Reasoning.CanPreserve {
		return wrap.Reasoning
	}
	return DetectReasoning("", arch)
}

// Kwargs is the chat_template_kwargs object llama.cpp should receive for
// effort. Unknown or disallowed values yield nil (caller keeps existing
// kwargs). "off" becomes the template's native disable path.
func (r Reasoning) Kwargs(effort string) map[string]any {
	if !r.Controllable() {
		return nil
	}
	e := normalizeEffort(effort)
	if e == "" || !r.Allows(e) {
		return nil
	}
	out := map[string]any{}
	if e == EffortOff {
		if !r.CanDisable {
			return nil
		}
		if r.Toggle != "" {
			out[r.Toggle] = false
		}
		switch r.Style {
		case EnableThinking, Thinking:
			out[r.Style] = false
		case ReasoningEffort, ReasoningStrength:
			if r.Toggle == "" {
				out[r.Style] = "none"
			}
		}
		return out
	}
	if r.Toggle != "" {
		out[r.Toggle] = true
	}
	switch r.Style {
	case EnableThinking, Thinking:
		out[r.Style] = true
	case ReasoningEffort, ReasoningStrength:
		out[r.Style] = e
	}
	return out
}

// OpenAIEffort is the top-level reasoning_effort value llama.cpp / OpenAI
// clients understand. Empty means do not set the field (Glimmer rejects it;
// boolean "on" has no OpenAI equivalent).
func (r Reasoning) OpenAIEffort(effort string) string {
	if r.Style != ReasoningEffort {
		return ""
	}
	e := normalizeEffort(effort)
	if e == EffortOff {
		if r.CanDisable {
			return "none"
		}
		return ""
	}
	switch e {
	case "minimal", "low", "medium", "high", "xhigh":
		return e
	default:
		return ""
	}
}

// ApplyToRequest merges native reasoning kwargs into an OpenAI-style body.
// It reads reasoning_effort or reasoning.effort when effort is empty.
func ApplyToRequest(body map[string]any, r Reasoning, effort string) {
	if body == nil || !r.Controllable() {
		return
	}
	if strings.TrimSpace(effort) == "" {
		effort, _ = requestEffort(body)
	}
	kwargs := r.Kwargs(effort)
	if kwargs == nil {
		return
	}
	existing := map[string]any{}
	switch v := body["chat_template_kwargs"].(type) {
	case map[string]any:
		existing = v
	case json.RawMessage:
		_ = json.Unmarshal(v, &existing)
	case string:
		_ = json.Unmarshal([]byte(v), &existing)
	}
	if existing == nil {
		existing = map[string]any{}
	}
	for k, val := range kwargs {
		existing[k] = val
	}
	body["chat_template_kwargs"] = existing
	if oa := r.OpenAIEffort(effort); oa != "" {
		body["reasoning_effort"] = oa
	} else if r.Style != ReasoningEffort {
		// Glimmer (and boolean templates) must not send a top-level
		// reasoning_effort — llama.cpp / the template may reject it.
		delete(body, "reasoning_effort")
	}
}

func requestEffort(body map[string]any) (string, bool) {
	if v, ok := body["reasoning_effort"].(string); ok && strings.TrimSpace(v) != "" {
		return v, true
	}
	switch rec := body["reasoning"].(type) {
	case map[string]any:
		if v, ok := rec["effort"].(string); ok && strings.TrimSpace(v) != "" {
			return v, true
		}
	}
	return "", false
}

// PatchJSONReasoning applies native chat_template_kwargs for a request that
// already carries reasoning_effort / reasoning.effort. Unrelated bodies are
// returned unchanged.
func PatchJSONReasoning(body []byte, r Reasoning) []byte {
	if !r.Controllable() || len(body) == 0 {
		return body
	}
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return body
	}
	if _, ok := requestEffort(m); !ok {
		return body
	}
	ApplyToRequest(m, r, "")
	out, err := json.Marshal(m)
	if err != nil {
		return body
	}
	return out
}
