package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func writeVocabLayoutGGUF(t *testing.T, tokens []string, embd, vocabRows uint64, vocabSizeKV uint32) string {
	t.Helper()
	var b bytes.Buffer
	w32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	w64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	wstr := func(s string) { w64(uint64(len(s))); b.WriteString(s) }

	kvCount := uint64(3) // arch, embedding, tokens
	if vocabSizeKV > 0 {
		kvCount++
	}
	binary.Write(&b, binary.LittleEndian, uint32(magicGGUF))
	w32(3)
	w64(1) // one tensor
	w64(kvCount)

	wstr("general.architecture")
	w32(tString)
	wstr("llama")

	wstr("llama.embedding_length")
	w32(tUint32)
	w32(uint32(embd))

	if vocabSizeKV > 0 {
		wstr("llama.vocab_size")
		w32(tUint32)
		w32(vocabSizeKV)
	}

	wstr("tokenizer.ggml.tokens")
	w32(tArray)
	w32(tString)
	w64(uint64(len(tokens)))
	for _, s := range tokens {
		wstr(s)
	}

	wstr("token_embd.weight")
	w32(2)
	w64(embd)
	w64(vocabRows)
	w32(0) // F32
	w64(0)

	path := filepath.Join(t.TempDir(), "v.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestCheckVocabLayoutOK(t *testing.T) {
	toks := []string{"a", "b", "c", "d"}
	path := writeVocabLayoutGGUF(t, toks, 8, 4, 4)
	if err := CheckVocabLayout(path); err != nil {
		t.Fatal(err)
	}
}

func TestCheckVocabLayoutTokenizerVsEmbed(t *testing.T) {
	toks := []string{"a", "b", "c", "d"}
	path := writeVocabLayoutGGUF(t, toks, 8, 6, 0)
	err := CheckVocabLayout(path)
	if err == nil {
		t.Fatal("expected mismatch")
	}
	if got := err.Error(); got == "" {
		t.Fatal("empty error")
	}
}

func TestCheckVocabLayoutVocabSizeKV(t *testing.T) {
	toks := []string{"a", "b", "c", "d", "e", "f"}
	path := writeVocabLayoutGGUF(t, toks, 8, 6, 4)
	if err := CheckVocabLayout(path); err == nil {
		t.Fatal("expected vocab_size KV mismatch")
	}
}
