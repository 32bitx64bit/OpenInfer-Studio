package reasoning

import "testing"

func TestSplit(t *testing.T) {
	cases := []struct {
		name string
		in   string
		r, c string
	}{
		{"plain", "hello", "", "hello"},
		{"think", "<think>plan</think>Answer", "plan", "Answer"},
		{"channel empty", "<|channel>thought\n<channel|>Hello!", "", "Hello!"},
		{"channel body", "<|channel>thought\nplan\n<channel|>Answer", "plan", "Answer"},
		{"unterminated", "<think>still thinking", "still thinking", ""},
		{"reasoning tag", "<reasoning>x</reasoning>y", "x", "y"},
	}
	for _, c := range cases {
		r, content := Split(c.in)
		if r != c.r || content != c.c {
			t.Errorf("%s: got (%q,%q) want (%q,%q)", c.name, r, content, c.r, c.c)
		}
	}
}

func TestSplitterStreaming(t *testing.T) {
	var s Splitter
	// Split across chunks mid-tag and mid-body.
	chunks := []string{"He", "llo <th", "ink>pla", "n</thi", "nk> wor", "ld"}
	var r, c string
	for _, ch := range chunks {
		rd, cd := s.Push(ch)
		r += rd
		c += cd
	}
	rd, cd := s.Flush()
	r += rd
	c += cd
	if r != "plan" || c != "Hello  world" {
		t.Fatalf("got reasoning=%q content=%q", r, c)
	}
}

func TestSplitterHoldbackPartialStart(t *testing.T) {
	var s Splitter
	rd, cd := s.Push("abc <thi")
	if rd != "" || cd != "abc " {
		t.Fatalf("holdback: got r=%q c=%q", rd, cd)
	}
	rd, cd = s.Push("nk>secret</think>out")
	if rd != "secret" || cd != "out" {
		t.Fatalf("complete: got r=%q c=%q", rd, cd)
	}
}
