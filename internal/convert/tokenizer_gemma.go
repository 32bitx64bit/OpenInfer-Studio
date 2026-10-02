package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Gemma tokenizers.
//
// Gemma 1–3 and Gemma 3n ship a SentencePiece model (tokenizer.model).
// llama.cpp reads those as tokenizer "llama" (SPM) with per-piece scores and
// types, exactly as its own converter writes them; the generic BPE loader
// (tokenizer "gpt2" + a Gemma pre-tokenizer name) is not something
// llama.cpp loads. Gemma 4 is SPM-style BPE and uses llama.cpp's "gemma4"
// tokenizer model with merges from tokenizer.json.

// spmPiece is one entry of a SentencePiece ModelProto.
type spmPiece struct {
	Piece string
	Score float32
	Type  int32 // sentencepiece Type == llama token type
}

// SentencePiece Type values equal llama_token_type values.
const spmTypeNormal int32 = 1

// parseSPMModel decodes the pieces of a SentencePiece ModelProto:
//
//	ModelProto      { repeated SentencePiece pieces = 1; ... }
//	SentencePiece   { string piece = 1; float score = 2; Type type = 3; }
//
// Every other field is skipped by wire type.
func parseSPMModel(b []byte) ([]spmPiece, error) {
	var pieces []spmPiece
	for len(b) > 0 {
		tag, n := binary.Uvarint(b)
		if n <= 0 {
			return nil, fmt.Errorf("tokenizer.model: truncated field tag")
		}
		b = b[n:]
		field, wire := tag>>3, tag&7
		if field == 1 && wire == 2 {
			l, n := binary.Uvarint(b)
			if n <= 0 || uint64(len(b)-n) < l {
				return nil, fmt.Errorf("tokenizer.model: truncated piece")
			}
			p, err := parseSPMPiece(b[n : n+int(l)])
			if err != nil {
				return nil, err
			}
			pieces = append(pieces, p)
			b = b[n+int(l):]
			continue
		}
		rest, err := skipProtoField(b, wire)
		if err != nil {
			return nil, err
		}
		b = rest
	}
	if len(pieces) == 0 {
		return nil, fmt.Errorf("tokenizer.model: no pieces")
	}
	return pieces, nil
}

func parseSPMPiece(b []byte) (spmPiece, error) {
	p := spmPiece{Type: spmTypeNormal}
	for len(b) > 0 {
		tag, n := binary.Uvarint(b)
		if n <= 0 {
			return p, fmt.Errorf("tokenizer.model: truncated piece tag")
		}
		b = b[n:]
		field, wire := tag>>3, tag&7
		switch {
		case field == 1 && wire == 2:
			l, n := binary.Uvarint(b)
			if n <= 0 || uint64(len(b)-n) < l {
				return p, fmt.Errorf("tokenizer.model: truncated piece text")
			}
			p.Piece = string(b[n : n+int(l)])
			b = b[n+int(l):]
		case field == 2 && wire == 5:
			if len(b) < 4 {
				return p, fmt.Errorf("tokenizer.model: truncated piece score")
			}
			p.Score = math.Float32frombits(binary.LittleEndian.Uint32(b))
			b = b[4:]
		case field == 3 && wire == 0:
			v, n := binary.Uvarint(b)
			if n <= 0 {
				return p, fmt.Errorf("tokenizer.model: truncated piece type")
			}
			p.Type = int32(v)
			b = b[n:]
		default:
			rest, err := skipProtoField(b, wire)
			if err != nil {
				return p, err
			}
			b = rest
		}
	}
	return p, nil
}

func skipProtoField(b []byte, wire uint64) ([]byte, error) {
	switch wire {
	case 0:
		_, n := binary.Uvarint(b)
		if n <= 0 {
			return nil, fmt.Errorf("tokenizer.model: truncated varint")
		}
		return b[n:], nil
	case 1:
		if len(b) < 8 {
			return nil, fmt.Errorf("tokenizer.model: truncated fixed64")
		}
		return b[8:], nil
	case 2:
		l, n := binary.Uvarint(b)
		if n <= 0 || uint64(len(b)-n) < l {
			return nil, fmt.Errorf("tokenizer.model: truncated length-delimited field")
		}
		return b[n+int(l):], nil
	case 5:
		if len(b) < 4 {
			return nil, fmt.Errorf("tokenizer.model: truncated fixed32")
		}
		return b[4:], nil
	}
	return nil, fmt.Errorf("tokenizer.model: unsupported wire type %d", wire)
}

// spmPadScore is the score of a vocabulary slot no piece fills.
const spmPadScore float32 = -10000

// loadSPMTokenizer builds the llama.cpp "llama" (SPM) tokenizer from a
// SentencePiece tokenizer.model, the way llama.cpp's converter does: pieces
// keep their score and type, ids past the model up to vocabSize become
// unused [PADn] slots, and added_tokens.json / tokenizer_config.json
// overrides become user-defined or control tokens.
func loadSPMTokenizer(dir string, vocabSize int) (*ggmlTokenizer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer.model"))
	if err != nil {
		return nil, err
	}
	pieces, err := parseSPMModel(raw)
	if err != nil {
		return nil, err
	}
	n := len(pieces)
	if vocabSize > n {
		n = vocabSize
	}
	tokens := make([]string, n)
	scores := make([]float32, n)
	types := make([]int32, n)
	for i := range tokens {
		tokens[i] = fmt.Sprintf("[PAD%d]", i)
		scores[i] = spmPadScore
		types[i] = tokenTypeUnused
	}
	for i, p := range pieces {
		tokens[i], scores[i], types[i] = p.Piece, p.Score, p.Type
	}

	if b, err := os.ReadFile(filepath.Join(dir, "added_tokens.json")); err == nil {
		var added map[string]int
		if json.Unmarshal(b, &added) == nil {
			for tok, id := range added {
				if id < 0 || id >= n {
					continue
				}
				tokens[id], scores[id], types[id] = tok, -1000, tokenTypeUserDefined
			}
		}
	}
	if b, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json")); err == nil {
		var tc struct {
			AddedTokensDecoder map[string]struct {
				Content string `json:"content"`
				Special bool   `json:"special"`
			} `json:"added_tokens_decoder"`
		}
		if json.Unmarshal(b, &tc) == nil {
			for idStr, d := range tc.AddedTokensDecoder {
				var id int
				if _, err := fmt.Sscanf(idStr, "%d", &id); err != nil || id < 0 || id >= n {
					continue
				}
				content := d.Content
				if d.Special || tokenLooksSpecial(content) {
					types[id] = tokenTypeControl
				} else {
					// user-defined spaces are stored pre-normalized
					content = strings.ReplaceAll(content, "▁", " ")
					types[id] = tokenTypeUserDefined
				}
				scores[id], tokens[id] = -1000, content
			}
		}
	}

	out := &ggmlTokenizer{
		Model:     "llama",
		Tokens:    tokens,
		Scores:    scores,
		TokenType: types,
		Bos:       -1, Eos: -1, Unk: -1, Pad: -1, Eot: -1,
		AddBos:        true,
		NoSpacePrefix: true,
	}
	finishTokenizer(dir, out)
	return out, nil
}

var byteTokenRe = regexp.MustCompile(`^<0x[0-9A-Fa-f]{2}>$`)

// gemma4VisibleTokens are control-looking tokens llama.cpp's converter keeps
// user-defined so the chat parser can read them in the output.
var gemma4VisibleTokens = map[string]bool{
	"<|channel>": true, "<channel|>": true,
	"<|tool_call>": true, "<tool_call|>": true,
	"<|tool_response>": true, "<tool_response|>": true,
	`<|"|>`: true,
}

// gemma4Tokenizer adapts the generic BPE tokenizer.json load to llama.cpp's
// "gemma4" tokenizer model: SPM-style byte fallback tokens are byte-typed,
// the chat-visible specials are user-defined, there is no space prefix, and
// a BOS is always added.
func gemma4Tokenizer(tok *ggmlTokenizer) *ggmlTokenizer {
	tok.Model = "gemma4"
	tok.NoSpacePrefix = true
	tok.AddBos = true
	for i, t := range tok.Tokens {
		switch {
		case gemma4VisibleTokens[t]:
			tok.TokenType[i] = tokenTypeUserDefined
		case byteTokenRe.MatchString(t):
			tok.TokenType[i] = tokenTypeByte
		}
	}
	return tok
}

// loadGemmaTokenizer picks the tokenizer for a Gemma family: Gemma 4 uses
// the "gemma4" BPE model from tokenizer.json; the SentencePiece families use
// tokenizer.model when present (what llama.cpp writes) and otherwise the
// generic tokenizer.json path.
func loadGemmaTokenizer(dir, family string, vocabSize int) (*ggmlTokenizer, error) {
	if family == "gemma4" {
		tok, err := loadTokenizer(dir)
		if err != nil {
			return nil, err
		}
		if len(tok.Merges) == 0 {
			return nil, fmt.Errorf("gemma4 tokenizer.json has no BPE merges")
		}
		return gemma4Tokenizer(tok), nil
	}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.model")); err == nil {
		return loadSPMTokenizer(dir, vocabSize)
	}
	return loadTokenizer(dir)
}
