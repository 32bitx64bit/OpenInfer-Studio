// Package reasoning splits model thought/reasoning markup out of assistant
// text into OpenAI-style reasoning_content, for both streaming deltas and
// complete messages.
package reasoning

import (
	"strings"
	"unicode/utf8"
)

// Marker pairs recognized as reasoning channels. First match (earliest start)
// wins when several appear.
var Markers = [][2]string{
	{"<|channel>thought", "<channel|>"},
	{"<think>", "</think>"},
	{"<reasoning>", "</reasoning>"},
}

// Split extracts all closed reasoning blocks from a full string. Unterminated
// thought (length-truncated generation) becomes reasoning; content is the rest
// with markers removed.
func Split(full string) (reasoningOut, content string) {
	var rParts, cParts []string
	rest := full
	for {
		best := -1
		bestStart, bestEnd := "", ""
		for _, pair := range Markers {
			i := strings.Index(rest, pair[0])
			if i >= 0 && (best < 0 || i < best) {
				best = i
				bestStart, bestEnd = pair[0], pair[1]
			}
		}
		if best < 0 {
			cParts = append(cParts, rest)
			break
		}
		cParts = append(cParts, rest[:best])
		body := rest[best+len(bestStart):]
		if j := strings.Index(body, bestEnd); j >= 0 {
			rParts = append(rParts, body[:j])
			rest = body[j+len(bestEnd):]
			continue
		}
		rParts = append(rParts, body)
		break
	}
	return strings.Trim(strings.Join(rParts, ""), "\n"), strings.Join(cParts, "")
}

// Splitter incrementally routes streamed content deltas into reasoning vs
// answer text. Partial tag prefixes at the buffer end are held back until the
// next chunk disambiguates them.
type Splitter struct {
	buf         strings.Builder
	inReasoning bool
	endTag      string
}

// Push consumes a content delta and returns newly confirmed reasoning and
// content fragments (either may be empty). Markers themselves are never emitted.
func (s *Splitter) Push(delta string) (reasoningDelta, contentDelta string) {
	if delta == "" {
		return "", ""
	}
	s.buf.WriteString(delta)
	return s.drain(false)
}

// Flush releases any held partial tag as content (or reasoning if inside a
// block). Call at end of stream.
func (s *Splitter) Flush() (reasoningDelta, contentDelta string) {
	return s.drain(true)
}

func (s *Splitter) drain(flush bool) (reasoningDelta, contentDelta string) {
	var rOut, cOut strings.Builder
	for {
		raw := s.buf.String()
		if raw == "" {
			break
		}
		if s.inReasoning {
			if i := strings.Index(raw, s.endTag); i >= 0 {
				rOut.WriteString(raw[:i])
				s.buf.Reset()
				s.buf.WriteString(raw[i+len(s.endTag):])
				s.inReasoning = false
				s.endTag = ""
				continue
			}
			hold := 0
			if !flush {
				hold = trailingPartial(raw, s.endTag)
			}
			emit := raw[:len(raw)-hold]
			rOut.WriteString(emit)
			s.buf.Reset()
			if hold > 0 {
				s.buf.WriteString(raw[len(raw)-hold:])
			}
			break
		}

		best := -1
		bestStart, bestEnd := "", ""
		for _, pair := range Markers {
			i := strings.Index(raw, pair[0])
			if i >= 0 && (best < 0 || i < best) {
				best = i
				bestStart, bestEnd = pair[0], pair[1]
			}
		}
		if best >= 0 {
			cOut.WriteString(raw[:best])
			s.buf.Reset()
			s.buf.WriteString(raw[best+len(bestStart):])
			s.inReasoning = true
			s.endTag = bestEnd
			continue
		}
		hold := 0
		if !flush {
			hold = trailingPartialAnyStart(raw)
		}
		emit := raw[:len(raw)-hold]
		cOut.WriteString(emit)
		s.buf.Reset()
		if hold > 0 {
			s.buf.WriteString(raw[len(raw)-hold:])
		}
		break
	}
	return rOut.String(), cOut.String()
}

func trailingPartial(s, marker string) int {
	upper := min(len(marker)-1, len(s))
	for k := upper; k > 0; k-- {
		if strings.HasSuffix(s, marker[:k]) && utf8.ValidString(marker[:k]) {
			return k
		}
	}
	return 0
}

func trailingPartialAnyStart(s string) int {
	longest := 0
	for _, pair := range Markers {
		if n := trailingPartial(s, pair[0]); n > longest {
			longest = n
		}
	}
	return longest
}
