package gguf

import "fmt"

// CheckVocabLayout reports when tokenizer length, {arch}.vocab_size, and the
// vocab dimension of token_embd / output disagree. llama.cpp check_tensor_dims
// uses that n_vocab; a mismatch is architecture-agnostic and means reconvert.
func CheckVocabLayout(path string) error {
	tensors, md, err := ListTensors(path)
	if err != nil {
		return err
	}
	var nEmbd, nOut uint64
	for _, t := range tensors {
		switch t.Name {
		case "token_embd.weight":
			nEmbd = vocabDim(t.Shape, md.Embedding)
		case "output.weight":
			nOut = vocabDim(t.Shape, md.Embedding)
		}
	}
	type named struct {
		name string
		n    uint64
	}
	var have []named
	if md.TokenizerCount > 0 {
		have = append(have, named{"tokenizer", uint64(md.TokenizerCount)})
	}
	if n := vocabSizeKV(md); n > 0 {
		have = append(have, named{md.Architecture + ".vocab_size", uint64(n)})
	}
	if nEmbd > 0 {
		have = append(have, named{"token_embd.weight", nEmbd})
	}
	if nOut > 0 {
		have = append(have, named{"output.weight", nOut})
	}
	if len(have) < 2 {
		return nil
	}
	want := have[0]
	for _, c := range have[1:] {
		if c.n != want.n {
			return fmt.Errorf("vocab layout mismatch: %s is %d but %s is %d", want.name, want.n, c.name, c.n)
		}
	}
	return nil
}

func vocabSizeKV(md *Metadata) uint32 {
	if md == nil || md.Architecture == "" || md.Raw == nil {
		return 0
	}
	v, ok := md.Raw[md.Architecture+".vocab_size"]
	if !ok {
		return 0
	}
	n, ok := toUint32(v)
	if !ok {
		return 0
	}
	return n
}

// vocabDim is the GGUF dimension that is not embedding_length.
// token_embd is stored as [n_embd, n_vocab].
func vocabDim(shape []uint64, embd uint32) uint64 {
	switch len(shape) {
	case 0:
		return 0
	case 1:
		return shape[0]
	default:
		if embd > 0 {
			if uint32(shape[0]) == embd {
				return shape[1]
			}
			if uint32(shape[1]) == embd {
				return shape[0]
			}
		}
		if shape[0] >= shape[1] {
			return shape[0]
		}
		return shape[1]
	}
}
