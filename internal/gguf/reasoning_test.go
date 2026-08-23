package gguf

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"
)

const gptOSSTemplate = `{#-
  In addition to the normal inputs of ` + "`messages`" + ` and ` + "`tools`" + `, this template also accepts:
  - "reasoning_effort": A string that describes the reasoning effort, defaults to "medium".
 #}
{%- if reasoning_effort is not defined %}
    {%- set reasoning_effort = "medium" %}
{%- endif %}
{{- "Reasoning: " + reasoning_effort + "\n\n" }}
{%- if "thinking" in message %}
    {{- message.thinking }}
{%- endif %}
`

const qwenThinkTemplate = `{%- if enable_thinking is defined and enable_thinking is false %}
    {%- set thinking = false %}
{%- else %}
    {%- set thinking = true %}
{%- endif %}
{%- if thinking %}
    {{- '<think>' }}
{%- endif %}
`

const gemmaThinkTemplate = `{%- if enable_thinking is defined and enable_thinking -%}
{{- '<|think|>' }}
{%- endif -%}
`

const glimmerTemplate = `{%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high' -%}
Reasoning strength: {{ rs }}.
`

const glimmerListedTemplate = `{%- set reasoning_strength = reasoning_strength if reasoning_strength is defined else "high" -%}
{# levels: low / medium / high / xhigh #}
`

const deepseekThinkTemplate = `{%- if thinking is defined and thinking %}
{{- '<think>' }}
{%- endif %}
`

const effortPlusToggle = `{%- if enable_thinking is defined and enable_thinking is false %}
{%- else %}
{%- if reasoning_effort not in ["low", "medium", "high", "none"] %}
{%- set reasoning_effort = "medium" %}
{%- endif %}
{{- reasoning_effort }}
{%- endif %}
`

func TestDetectReasoningGPTOss(t *testing.T) {
	r := DetectReasoning(gptOSSTemplate, "gpt-oss")
	if r.Style != ReasoningEffort {
		t.Fatalf("style = %q", r.Style)
	}
	if r.CanDisable {
		t.Fatal("gpt-oss cannot disable reasoning")
	}
	if !reflect.DeepEqual(r.Efforts, []string{"low", "medium", "high"}) {
		t.Fatalf("efforts = %v", r.Efforts)
	}
	if r.Default != "medium" {
		t.Fatalf("default = %q", r.Default)
	}
	if r.Toggle != "" {
		t.Fatalf("toggle = %q, message.thinking must not look like a kwarg", r.Toggle)
	}
}

func TestDetectReasoningQwenEnableThinking(t *testing.T) {
	r := DetectReasoning(qwenThinkTemplate, "qwen3")
	if r.Style != EnableThinking || !r.CanDisable {
		t.Fatalf("got %+v", r)
	}
	if !reflect.DeepEqual(r.Efforts, []string{"off", "on"}) {
		t.Fatalf("efforts = %v", r.Efforts)
	}
	if r.Default != "on" {
		t.Fatalf("default = %q, want on", r.Default)
	}
}

func TestDetectReasoningGemmaDefaultOff(t *testing.T) {
	r := DetectReasoning(gemmaThinkTemplate, "gemma4")
	if r.Style != EnableThinking || !r.CanDisable {
		t.Fatalf("got %+v", r)
	}
	if r.Default != EffortOff {
		t.Fatalf("default = %q, want off", r.Default)
	}
}

func TestDetectReasoningGlimmer(t *testing.T) {
	r := DetectReasoning(glimmerTemplate, "muse-glimmer")
	if r.Style != ReasoningStrength {
		t.Fatalf("style = %q", r.Style)
	}
	if r.CanDisable {
		t.Fatal("Glimmer cannot disable reasoning")
	}
	if !reflect.DeepEqual(r.Efforts, []string{"low", "medium", "high", "xhigh"}) {
		t.Fatalf("efforts = %v", r.Efforts)
	}
	if r.Default != "low" {
		t.Fatalf("default = %q, want low (stream-friendly override)", r.Default)
	}

	listed := DetectReasoning(glimmerListedTemplate, "muse-glimmer")
	if !reflect.DeepEqual(listed.Efforts, []string{"low", "medium", "high", "xhigh"}) {
		t.Fatalf("listed efforts = %v", listed.Efforts)
	}

	fallback := DetectReasoning("", "muse-glimmer")
	if !fallback.Controllable() || fallback.Default != "low" {
		t.Fatalf("arch fallback = %+v", fallback)
	}
	if DetectReasoning("", "muse-glimmer-assistant").Controllable() {
		t.Fatal("draft assistant must not advertise chat reasoning")
	}
}

func TestDetectReasoningDeepSeekThinking(t *testing.T) {
	r := DetectReasoning(deepseekThinkTemplate, "deepseek2")
	if r.Style != Thinking || !r.CanDisable {
		t.Fatalf("got %+v", r)
	}
}

func TestDetectReasoningEffortPlusOff(t *testing.T) {
	r := DetectReasoning(effortPlusToggle, "qwen3")
	if r.Style != ReasoningEffort {
		t.Fatalf("style = %q", r.Style)
	}
	if !r.CanDisable || r.Toggle != EnableThinking {
		t.Fatalf("disable path missing: %+v", r)
	}
	if !reflect.DeepEqual(r.Efforts, []string{"off", "low", "medium", "high"}) {
		t.Fatalf("efforts = %v", r.Efforts)
	}
}

func TestDetectReasoningPlainLlama(t *testing.T) {
	r := DetectReasoning("{{ messages }} hello thinking world", "llama")
	if r.Controllable() {
		t.Fatalf("plain template must not look controllable: %+v", r)
	}
	if r.CanPreserve {
		t.Fatal("plain template must not advertise preserve")
	}
}

func TestDetectPreserveReasoning(t *testing.T) {
	qwen := `{%- set preserve_thinking = preserve_thinking | default(false) -%}
{%- if enable_thinking is defined and enable_thinking is false -%}{%- endif -%}`
	r := DetectReasoning(qwen, "qwen3")
	if !r.CanPreserve {
		t.Fatalf("preserve_thinking must set can_preserve: %+v", r)
	}
	if r.Style != EnableThinking {
		t.Fatalf("style = %q", r.Style)
	}

	glm := `{%- set clear_thinking = clear_thinking | default(true) -%}
{%- if enable_thinking is defined and enable_thinking is false -%}{%- endif -%}`
	if !DetectReasoning(glm, "glm4").CanPreserve {
		t.Fatal("clear_thinking must set can_preserve")
	}

	onlyPreserve := `{%- if preserve_reasoning -%}{{ message.reasoning_content }}{%- endif -%}`
	p := DetectReasoning(onlyPreserve, "llama")
	if p.Controllable() || !p.CanPreserve {
		t.Fatalf("preserve-only template: %+v", p)
	}
}

func TestReasoningKwargs(t *testing.T) {
	gpt := DetectReasoning(gptOSSTemplate, "gpt-oss")
	got := gpt.Kwargs("high")
	if got[ReasoningEffort] != "high" {
		t.Fatalf("gpt kwargs = %v", got)
	}
	if gpt.Kwargs("off") != nil {
		t.Fatal("gpt-oss off must be ignored")
	}

	qwen := DetectReasoning(qwenThinkTemplate, "qwen3")
	off := qwen.Kwargs("off")
	if off[EnableThinking] != false {
		t.Fatalf("qwen off = %v", off)
	}
	on := qwen.Kwargs("on")
	if on[EnableThinking] != true {
		t.Fatalf("qwen on = %v", on)
	}

	glim := DetectReasoning("", "muse-glimmer")
	ks := glim.Kwargs("xhigh")
	if ks[ReasoningStrength] != "xhigh" {
		t.Fatalf("glimmer kwargs = %v", ks)
	}
	if glim.OpenAIEffort("high") != "" {
		t.Fatal("glimmer must not emit top-level reasoning_effort")
	}
	if gpt.OpenAIEffort("high") != "high" {
		t.Fatalf("gpt openai effort = %q", gpt.OpenAIEffort("high"))
	}
}

func TestApplyToRequestGlimmerStripsTopLevel(t *testing.T) {
	body := map[string]any{"reasoning_effort": "high", "model": "g"}
	ApplyToRequest(body, DetectReasoning("", "muse-glimmer"), "")
	if _, ok := body["reasoning_effort"]; ok {
		t.Fatalf("top-level reasoning_effort leaked: %+v", body)
	}
	kw, _ := body["chat_template_kwargs"].(map[string]any)
	if kw[ReasoningStrength] != "high" {
		t.Fatalf("kwargs = %v", kw)
	}
}

func TestApplyToRequestMergesExistingKwargs(t *testing.T) {
	body := map[string]any{
		"chat_template_kwargs": map[string]any{"preserve_thinking": true},
		"reasoning_effort":     "low",
	}
	ApplyToRequest(body, DetectReasoning(gptOSSTemplate, "gpt-oss"), "")
	kw := body["chat_template_kwargs"].(map[string]any)
	if kw[ReasoningEffort] != "low" || kw["preserve_thinking"] != true {
		t.Fatalf("merged kwargs = %v", kw)
	}
	if body["reasoning_effort"] != "low" {
		t.Fatalf("top-level = %v", body["reasoning_effort"])
	}
}

func TestReasoningFromMetadata(t *testing.T) {
	raw, _ := json.Marshal(map[string]any{
		"reasoning": Reasoning{Style: ReasoningEffort, Efforts: []string{"low", "high"}, Default: "low"},
	})
	r := ReasoningFromMetadata(raw, "llama")
	if r.Style != ReasoningEffort || r.Default != "low" {
		t.Fatalf("got %+v", r)
	}
	fallback := ReasoningFromMetadata([]byte(`{}`), "muse-glimmer")
	if fallback.Style != ReasoningStrength {
		t.Fatalf("arch fallback missing: %+v", fallback)
	}
	preserveOnly, _ := json.Marshal(map[string]any{
		"reasoning": Reasoning{CanPreserve: true},
	})
	got := ReasoningFromMetadata(preserveOnly, "llama")
	if !got.CanPreserve {
		t.Fatalf("preserve-only metadata dropped: %+v", got)
	}
}

func TestPatchJSONReasoning(t *testing.T) {
	in := []byte(`{"model":"g","reasoning_effort":"medium"}`)
	out := PatchJSONReasoning(in, DetectReasoning("", "muse-glimmer"))
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["reasoning_effort"]; ok {
		t.Fatal("glimmer must drop top-level reasoning_effort")
	}
	kw := m["chat_template_kwargs"].(map[string]any)
	if kw[ReasoningStrength] != "medium" {
		t.Fatalf("kwargs = %v", kw)
	}
	unchanged := []byte(`{"model":"g","messages":[]}`)
	if string(PatchJSONReasoning(unchanged, DetectReasoning("", "muse-glimmer"))) != string(unchanged) {
		t.Fatal("bodies without effort must pass through")
	}
}

func TestParseFileDetectsReasoning(t *testing.T) {
	data := buildGGUF(t, map[string]any{
		"general.architecture":    "qwen3",
		"tokenizer.chat_template": qwenThinkTemplate,
	})
	md, err := parse(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if !md.Reasoning.Controllable() || md.Reasoning.Style != EnableThinking {
		t.Fatalf("parsed reasoning = %+v", md.Reasoning)
	}
}
