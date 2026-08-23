package gguf

import "testing"

func TestIsMuseGlimmerChat(t *testing.T) {
	cases := []struct {
		arch string
		want bool
	}{
		{"muse-glimmer", true},
		{"muse_glimmer", true},
		{"Muse-Glimmer", true},
		{"muse-glimmer-assistant", false},
		{"muse_glimmer_assistant", false},
		{"llama", false},
		{"", false},
	}
	for _, c := range cases {
		if got := IsMuseGlimmerChat(c.arch); got != c.want {
			t.Errorf("IsMuseGlimmerChat(%q) = %v, want %v", c.arch, got, c.want)
		}
	}
}

func TestNeedsJinja(t *testing.T) {
	if !NeedsJinja("muse-glimmer", false) {
		t.Fatal("text-only Muse Glimmer must still use jinja")
	}
	if NeedsJinja("muse-glimmer-assistant", false) {
		t.Fatal("DFlash assistant is not a chat jinja target")
	}
	if !NeedsJinja("llama", true) {
		t.Fatal("multimodal models need jinja")
	}
	if NeedsJinja("llama", false) {
		t.Fatal("plain llama without a projector does not force jinja")
	}
}
