package instances

import (
	"strings"
	"testing"
)

func TestDiffusionBlocks(t *testing.T) {
	cases := []struct {
		maxTok int
		canvas uint32
		want   int
	}{
		{0, 256, 8},
		{-1, 256, 8},
		{256, 256, 1},
		{257, 256, 2},
		{2048, 256, 8},
		{2048, 0, 8},
		{100000, 256, 64}, // clamped
	}
	for _, c := range cases {
		if got := diffusionBlocks(c.maxTok, c.canvas); got != c.want {
			t.Errorf("diffusionBlocks(%d,%d)=%d want %d", c.maxTok, c.canvas, got, c.want)
		}
	}
}

func TestSplitThoughtChannels(t *testing.T) {
	cases := []struct {
		name string
		in   string
		r, c string
	}{
		{
			name: "empty thought then answer",
			in:   "<|channel>thought\n<channel|>Hello!",
			r:    "",
			c:    "Hello!",
		},
		{
			name: "thought with body",
			in:   "<|channel>thought\nplan\n<channel|>Answer",
			r:    "plan",
			c:    "Answer",
		},
		{
			name: "no markers",
			in:   "just text",
			r:    "",
			c:    "just text",
		},
		{
			name: "unterminated thought",
			in:   "<|channel>thought\nstill thinking",
			r:    "still thinking",
			c:    "",
		},
	}
	for _, c := range cases {
		r, content := splitThoughtChannels(c.in)
		if r != c.r || content != c.c {
			t.Errorf("%s: got (%q,%q) want (%q,%q)", c.name, r, content, c.r, c.c)
		}
	}
}

func TestFormatDiffusionErrToolong(t *testing.T) {
	got := formatDiffusionErr("ERR toolong 5000 4096")
	if !strings.Contains(got, "5000") || !strings.Contains(got, "4096") {
		t.Fatalf("unexpected: %q", got)
	}
}

func TestLooksTruncated(t *testing.T) {
	if !looksTruncated("toss the fish with:") {
		t.Fatal("expected truncated on trailing colon")
	}
	if looksTruncated("Done.") {
		t.Fatal("expected complete on period")
	}
}

func TestShouldContinueDiffusion(t *testing.T) {
	// Early EOS after 2 of 8, text cut mid-sentence → continue.
	if !shouldContinueDiffusion(2, 8, 322, 2048, 8, 2, "toss the fish with:") {
		t.Fatal("expected continue on early EOS + truncated text")
	}
	// Filled the full request → stop.
	if shouldContinueDiffusion(8, 8, 2000, 2048, 8, 8, "Done.") {
		t.Fatal("expected stop when budget exhausted")
	}
	// Clean short reply must NOT keep spinning just because predicted << max_tokens.
	if shouldContinueDiffusion(1, 8, 13, 2048, 8, 1, "Hello! How can I help you today?") {
		t.Fatal("expected stop on clean short reply")
	}
}

func TestLongestCommonPrefix(t *testing.T) {
	if got := longestCommonPrefix("hello world", "hello there"); got != "hello " {
		t.Fatalf("got %q", got)
	}
	if got := longestCommonPrefix("abc", "xyz"); got != "" {
		t.Fatalf("got %q", got)
	}
	// Full-string match used to panic (a[len(a)]).
	if got := longestCommonPrefix("same", "same"); got != "same" {
		t.Fatalf("got %q", got)
	}
	if got := longestCommonPrefix("short", "shorter"); got != "short" {
		t.Fatalf("got %q", got)
	}
	if got := longestCommonPrefix("shorter", "short"); got != "short" {
		t.Fatalf("got %q", got)
	}
}

func TestWordStablePrefix(t *testing.T) {
	if got := wordStablePrefix("Hello world partial", false); got != "Hello world " {
		t.Fatalf("got %q", got)
	}
	if got := wordStablePrefix("Hello", false); got != "" {
		t.Fatalf("expected empty for single partial word, got %q", got)
	}
	if got := wordStablePrefix("Hello", true); got != "Hello" {
		t.Fatalf("flush got %q", got)
	}
}

func TestStableCanvasTrackerMonotonic(t *testing.T) {
	var tr stableCanvasTracker
	// Frame 1: seed only
	if d := tr.onFrame("The quick brown fox jumps"); d != "" {
		t.Fatalf("first frame should not emit, got %q", d)
	}
	// Frame 2: agrees on prefix → pending set, nothing promoted yet
	if d := tr.onFrame("The quick brown fox ZZ"); d != "" {
		t.Fatalf("pending only on first agreement, got %q", d)
	}
	// Frame 3: still agrees on "The quick brown fox " → promote pending, emit words
	d := tr.onFrame("The quick brown fox yes")
	if d == "" {
		t.Fatal("expected stable word emit after two agreeing steps")
	}
	if !strings.HasPrefix(tr.streamed, "The quick") {
		t.Fatalf("streamed=%q", tr.streamed)
	}
	// Never retract: noisy frame that diverges early should not shrink streamed
	before := tr.streamed
	_ = tr.onFrame("Totally different canvas noise")
	if tr.streamed != before {
		t.Fatalf("retracted from %q to %q", before, tr.streamed)
	}
	// Commit locks final text via append
	delta, snap, useSnap := tr.setConfirmed("The quick brown fox jumps over the lazy dog.")
	if useSnap {
		t.Fatalf("unexpected snapshot %q", snap)
	}
	if !strings.HasPrefix(tr.streamed, "The quick brown fox") {
		t.Fatalf("after commit streamed=%q delta=%q", tr.streamed, delta)
	}
}
