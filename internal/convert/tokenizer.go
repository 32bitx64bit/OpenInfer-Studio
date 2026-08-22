package convert

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type hfTokenizerJSON struct {
	Model struct {
		Type   string          `json:"type"`
		Vocab  map[string]int  `json:"vocab"`
		Merges json.RawMessage `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID         int    `json:"id"`
		Content    string `json:"content"`
		Special    bool   `json:"special"`
		SingleWord bool   `json:"single_word"`
	} `json:"added_tokens"`
}

type hfTokenizerConfig struct {
	BosToken       any    `json:"bos_token"`
	EosToken       any    `json:"eos_token"`
	PadToken       any    `json:"pad_token"`
	UnkToken       any    `json:"unk_token"`
	ChatTemplate   string `json:"chat_template"`
	AddBosToken    *bool  `json:"add_bos_token"`
	AddEosToken    *bool  `json:"add_eos_token"`
	ModelMaxLength any    `json:"model_max_length"`
}

// GGUF tokenizer.ggml.token_type (llama_token_type).
const (
	tokenTypeNormal      int32 = 1
	tokenTypeUnknown     int32 = 2
	tokenTypeControl     int32 = 3
	tokenTypeUserDefined int32 = 4
	tokenTypeUnused      int32 = 5
	tokenTypeByte        int32 = 6
)

type ggmlTokenizer struct {
	Model                   string
	Tokens                  []string
	Merges                  []string
	TokenType               []int32
	Bos, Eos, Unk, Pad, Eot int32
	Eog                     []int32 // eos + eot; never <|eom|>
	AddBos, AddEos          bool
	ChatTemplate            string
}

func loadTokenizer(dir string) (*ggmlTokenizer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	var tj hfTokenizerJSON
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	if len(tj.Model.Vocab) == 0 {
		return nil, fmt.Errorf("tokenizer.json: empty vocab")
	}
	maxID := 0
	for _, id := range tj.Model.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	for _, a := range tj.AddedTokens {
		if a.ID > maxID {
			maxID = a.ID
		}
	}
	tokens := make([]string, maxID+1)
	types := make([]int32, maxID+1)
	for i := range types {
		types[i] = tokenTypeNormal
	}
	for tok, id := range tj.Model.Vocab {
		if id < 0 || id >= len(tokens) {
			continue
		}
		tokens[id] = tok
	}
	for _, a := range tj.AddedTokens {
		if a.ID < 0 || a.ID >= len(tokens) {
			continue
		}
		tokens[a.ID] = a.Content
		// llama.cpp get_vocab_base: added + (special or looks like <|…|>) → CONTROL,
		// other added tokens → USER_DEFINED. Harmony specials with special=false
		// still have to be CONTROL so --parse-special matches them as whole tokens.
		if a.Special || tokenLooksSpecial(a.Content) {
			types[a.ID] = tokenTypeControl
		} else {
			types[a.ID] = tokenTypeUserDefined
		}
	}
	for i, t := range tokens {
		if t == "" {
			tokens[i] = fmt.Sprintf("[MISSING_%d]", i)
		}
	}
	merges, err := parseMerges(tj.Model.Merges)
	if err != nil {
		return nil, err
	}
	out := &ggmlTokenizer{
		Model:     "gpt2",
		Tokens:    tokens,
		Merges:    merges,
		TokenType: types,
		Bos:       -1, Eos: -1, Unk: -1, Pad: -1, Eot: -1,
		AddBos: true,
	}
	if strings.EqualFold(tj.Model.Type, "BPE") || tj.Model.Type == "" {
		out.Model = "gpt2"
	} else {
		out.Model = strings.ToLower(tj.Model.Type)
	}

	var tc hfTokenizerConfig
	if b, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json")); err == nil {
		_ = json.Unmarshal(b, &tc)
	}
	if tc.ChatTemplate == "" {
		if b, err := os.ReadFile(filepath.Join(dir, "chat_template.jinja")); err == nil {
			tc.ChatTemplate = string(b)
		}
	}
	out.ChatTemplate = strings.TrimSpace(tc.ChatTemplate)
	if tc.AddBosToken != nil {
		out.AddBos = *tc.AddBosToken
	}
	if tc.AddEosToken != nil {
		out.AddEos = *tc.AddEosToken
	}

	lookup := func(tok any) int32 {
		s := tokenString(tok)
		if s == "" {
			return -1
		}
		for id, t := range tokens {
			if t == s {
				return int32(id)
			}
		}
		return -1
	}
	out.Bos = lookup(tc.BosToken)
	out.Eos = lookup(tc.EosToken)
	out.Pad = lookup(tc.PadToken)
	out.Unk = lookup(tc.UnkToken)
	for id, t := range tokens {
		if isEOTToken(t) && out.Eot < 0 {
			out.Eot = int32(id)
		}
	}
	if out.Bos < 0 {
		out.Bos = lookup("<|begin_of_text|>")
		if out.Bos < 0 {
			out.Bos = lookup("<|startoftext|>")
		}
	}
	if out.Eos < 0 {
		out.Eos = lookup("<|end_of_text|>")
		if out.Eos < 0 {
			out.Eos = lookup("<|endoftext|>")
		}
	}
	applyGenerationConfig(dir, out, tokens)
	out.Eog = eogIDs(out, tokens)
	return out, nil
}

func isEOTToken(t string) bool {
	switch t {
	case "<|eot|>", "<|eot_id|>", "<|im_end|>":
		return true
	default:
		return false
	}
}

func isEOMToken(t string) bool {
	switch t {
	case "<|eom|>", "<|eom_id|>":
		return true
	default:
		return false
	}
}

// tokenLooksSpecial mirrors llama.cpp TextModel.does_token_look_special.
func tokenLooksSpecial(t string) bool {
	switch t {
	case "<pad>", "<mask>", "<2mass>", "[@BOS@]":
		return true
	}
	if strings.HasPrefix(t, "<|") && strings.HasSuffix(t, "|>") {
		return true
	}
	if strings.HasPrefix(t, "<｜") && strings.HasSuffix(t, "｜>") {
		return true
	}
	if strings.HasPrefix(t, "<unused") && strings.HasSuffix(t, ">") {
		return true
	}
	return false
}

func applyGenerationConfig(dir string, out *ggmlTokenizer, tokens []string) {
	b, err := os.ReadFile(filepath.Join(dir, "generation_config.json"))
	if err != nil {
		return
	}
	var gc struct {
		BosTokenID any `json:"bos_token_id"`
		EosTokenID any `json:"eos_token_id"`
		PadTokenID any `json:"pad_token_id"`
	}
	if json.Unmarshal(b, &gc) != nil {
		return
	}
	ids := tokenIDList(gc.EosTokenID)
	if out.Eos < 0 && len(ids) > 0 {
		out.Eos = ids[0]
	}
	for _, id := range ids {
		if id < 0 || int(id) >= len(tokens) {
			continue
		}
		if isEOMToken(tokens[id]) {
			continue
		}
		if isEOTToken(tokens[id]) && out.Eot < 0 {
			out.Eot = id
		}
	}
	if out.Bos < 0 {
		if bos := tokenIDList(gc.BosTokenID); len(bos) > 0 {
			out.Bos = bos[0]
		}
	}
	if out.Pad < 0 {
		if pad := tokenIDList(gc.PadTokenID); len(pad) > 0 {
			out.Pad = pad[0]
		}
	}
}

func tokenIDList(v any) []int32 {
	switch x := v.(type) {
	case float64:
		return []int32{int32(x)}
	case json.Number:
		n, err := x.Int64()
		if err != nil {
			return nil
		}
		return []int32{int32(n)}
	case int:
		return []int32{int32(x)}
	case int32:
		return []int32{x}
	case int64:
		return []int32{int32(x)}
	case []any:
		out := make([]int32, 0, len(x))
		for _, e := range x {
			ids := tokenIDList(e)
			out = append(out, ids...)
		}
		return out
	default:
		return nil
	}
}

func eogIDs(t *ggmlTokenizer, tokens []string) []int32 {
	seen := map[int32]bool{}
	var out []int32
	add := func(id int32) {
		if id < 0 || int(id) >= len(tokens) || seen[id] {
			return
		}
		if isEOMToken(tokens[id]) {
			return
		}
		seen[id] = true
		out = append(out, id)
	}
	add(t.Eos)
	add(t.Eot)
	return out
}

func tokenString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case map[string]any:
		if c, ok := x["content"].(string); ok {
			return c
		}
	}
	return ""
}

func parseMerges(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var asPairs [][]string
	if err := json.Unmarshal(raw, &asPairs); err == nil {
		out := make([]string, 0, len(asPairs))
		for _, p := range asPairs {
			if len(p) >= 2 {
				out = append(out, p[0]+" "+p[1])
			}
		}
		return out, nil
	}
	var asStr []string
	if err := json.Unmarshal(raw, &asStr); err != nil {
		return nil, fmt.Errorf("tokenizer merges: %w", err)
	}
	return asStr, nil
}

func (w *Writer) addTokenizer(t *ggmlTokenizer, pre string) {
	if t == nil {
		return
	}
	if pre == "" {
		pre = "default"
	}
	w.AddKV("tokenizer.ggml.model", t.Model)
	w.AddKV("tokenizer.ggml.pre", pre)
	w.AddKV("tokenizer.ggml.tokens", t.Tokens)
	if len(t.Merges) > 0 {
		w.AddKV("tokenizer.ggml.merges", t.Merges)
	}
	w.AddKV("tokenizer.ggml.token_type", t.TokenType)
	w.AddKV("tokenizer.ggml.add_bos_token", t.AddBos)
	w.AddKV("tokenizer.ggml.add_eos_token", t.AddEos)
	w.AddKV("tokenizer.ggml.add_sep_token", false)
	if t.Bos >= 0 {
		w.AddKV("tokenizer.ggml.bos_token_id", uint32(t.Bos))
	}
	if t.Eos >= 0 {
		w.AddKV("tokenizer.ggml.eos_token_id", uint32(t.Eos))
	}
	if t.Unk >= 0 {
		w.AddKV("tokenizer.ggml.unknown_token_id", uint32(t.Unk))
	}
	if t.Pad >= 0 {
		w.AddKV("tokenizer.ggml.padding_token_id", uint32(t.Pad))
	}
	if t.Eot >= 0 {
		w.AddKV("tokenizer.ggml.eot_token_id", uint32(t.Eot))
	}
	if len(t.Eog) > 0 {
		ids := make([]uint32, len(t.Eog))
		for i, id := range t.Eog {
			ids[i] = uint32(id)
		}
		w.AddKV("tokenizer.ggml.eos_token_ids", ids)
	}
	if t.ChatTemplate != "" {
		w.AddKV("tokenizer.chat_template", t.ChatTemplate)
	}
}
