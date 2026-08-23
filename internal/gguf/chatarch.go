package gguf

import "strings"

// GlimmerChatTemplateKwargs is the llama.cpp --chat-template-kwargs default
// for Muse Glimmer. The embedded template defaults to reasoning_strength=high,
// which spends max_tokens on thought and can return empty content — chat
// looks like it never streamed. "low" still reasons, but the answer starts
// in time to stream.
const GlimmerChatTemplateKwargs = `{"reasoning_strength":"low"}`

// IsMuseGlimmerChat reports a Muse Glimmer *chat* GGUF (not the DFlash
// muse-glimmer-assistant drafter).
func IsMuseGlimmerChat(arch string) bool {
	a := strings.ToLower(strings.TrimSpace(arch))
	if a == "" || strings.Contains(a, "assistant") {
		return false
	}
	return strings.Contains(a, "muse-glimmer") ||
		strings.Contains(a, "muse_glimmer") ||
		strings.Contains(a, "museglimmer")
}

// NeedsJinja is true when llama-server must run --jinja for correct stop
// tokens and reasoning_content. Muse Glimmer requires it even without an
// mmproj (text-only OID quants land in a new folder and lose the projector).
func NeedsJinja(arch string, multimodal bool) bool {
	return multimodal || IsMuseGlimmerChat(arch)
}
