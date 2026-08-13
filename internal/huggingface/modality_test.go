package huggingface

import (
	"reflect"
	"testing"
)

func TestDetectModalities(t *testing.T) {
	cases := []struct {
		id, pipe string
		tags     []string
		files    []string
		want     []string
	}{
		{"ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF", "audio-text-to-text",
			[]string{"gguf", "audio-text-to-text"}, nil, []string{"audio"}},
		{"ggml-org/Voxtral-Mini-3B-2507-GGUF", "",
			[]string{"gguf", "conversational"}, nil, []string{"audio"}},
		{"ggml-org/Qwen3-ASR-0.6B-GGUF", "", nil, nil, []string{"audio"}},
		{"ggml-org/llava-v1.6-mistral-7b-GGUF", "image-text-to-text",
			[]string{"image-text-to-text"}, nil, []string{"vision"}},
		{"ggml-org/Qwen2.5-Omni-3B-GGUF", "any-to-any",
			[]string{"multimodal", "any-to-any"}, nil, []string{"audio", "vision"}},
		{"bartowski/Llama-3.2-3B-Instruct-GGUF", "",
			[]string{"gguf", "conversational"}, nil, nil},
		// Text-only Gemma 4 quants must NOT inherit audio/vision from the name.
		{"someone/gemma-4-E2B-it-GGUF", "", []string{"gguf"}, nil, nil},
		{"SC117/gemma-4-12B-it-heretic-QAT-GGUF", "", []string{"gguf"}, nil, nil},
		// Official-style Gemma 4 with mmproj: both encoders.
		{"google/gemma-4-E2B-it-qat-q4_0-gguf", "", []string{"gguf"},
			[]string{"gemma-4-E2B_q4_0-it.gguf", "gemma-4-E2B-it-mmproj.gguf"},
			[]string{"audio", "vision"}},
		// Vision-only projector repo: no audio from name alone.
		{"openbmb/MiniCPM-V-4.6-gguf", "", []string{"gguf"},
			[]string{"MiniCPM-V-4_6-Q6_K.gguf", "mmproj-model-f16.gguf"},
			[]string{"vision"}},
		// Speculative draft / speculator repos must not inherit VL/audio labels.
		{"williamliao/gemma-4-31B-it-EAGLE3-Speculator-GGUF", "image-text-to-text",
			[]string{"gguf", "image-text-to-text"},
			[]string{"eagle3.gguf"}, nil},
		{"z-lab/Qwen3-4B-DFlash-GGUF", "", []string{"gguf", "speculative-decoding"}, nil, nil},
		{"org/Qwen2.5-VL-3B-Instruct-eagle3-GGUF", "", []string{"gguf"}, nil, nil},
		{"someone/mtp-Qwen3.6-27B-GGUF", "", []string{"gguf"}, nil, nil},
		// Mixed VL + DFlash sidecar: vision stays, even with dflash tags.
		{"meta-models/Muse-Glimmer-30B-GGUF", "image-text-to-text",
			[]string{"gguf", "image-text-to-text", "dflash", "speculative-decoding"},
			[]string{
				"Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
				"mmproj-Muse-Glimmer-30B-Q4_K_M.gguf",
				"dflash-Muse-Glimmer-30B-Q4_K_M.gguf",
			},
			[]string{"vision"}},
		{"Blackfrost-AI/Muse-Glimmer-30B-Abliterated-GGUF", "image-text-to-text",
			[]string{"gguf", "multimodal", "dflash"},
			[]string{
				"Muse-Glimmer-30B-Abliterated-Q4_K_M.gguf",
				"dflash-Muse-Glimmer-30B-Abliterated-F16.gguf",
				"mmproj-Muse-Glimmer-30B-Abliterated-F16.gguf",
			},
			[]string{"vision"}},
	}
	for _, tc := range cases {
		got := DetectModalities(tc.id, tc.pipe, tc.tags, tc.files)
		if len(got) == 0 && len(tc.want) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("DetectModalities(%q) = %v, want %v", tc.id, got, tc.want)
		}
	}
}

func TestDetectMTP(t *testing.T) {
	cases := []struct {
		id    string
		tags  []string
		files []string
		want  string
	}{
		// Fused trunk MTP GGUF (DavidAU-style).
		{"DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF",
			[]string{"gguf"}, nil, "mtp"},
		{"unsloth/Qwen3.6-27B-MTP-GGUF", []string{"gguf"}, nil, "mtp"},
		{"someone/Qwen3.5-4B-MTP-IQ4_XS.gguf", nil,
			[]string{"Qwen3.5-4B-MTP-IQ4_XS.gguf"}, "mtp"},
		{"org/model", []string{"mtp", "gguf"}, nil, "mtp"},
		// Sidecar draft packages.
		{"someone/mtp-Qwen3.6-27B-GGUF", []string{"gguf"}, nil, "mtp-draft"},
		{"org/repo", nil, []string{"mtp-Qwen3.6-27B-Q4_K_M.gguf"}, "mtp-draft"},
		// Non-MTP.
		{"bartowski/Llama-3.2-3B-Instruct-GGUF", []string{"gguf"}, nil, ""},
		{"z-lab/Qwen3-4B-DFlash-GGUF", []string{"gguf", "speculative-decoding"}, nil, ""},
	}
	for _, tc := range cases {
		if got := DetectMTP(tc.id, tc.tags, tc.files); got != tc.want {
			t.Errorf("DetectMTP(%q) = %q, want %q", tc.id, got, tc.want)
		}
	}
}

func TestDetectEmbedding(t *testing.T) {
	cases := []struct {
		id, pipe string
		tags     []string
		files    []string
		want     string
	}{
		{"nomic-ai/nomic-embed-text-v1.5-GGUF", "feature-extraction",
			[]string{"gguf", "feature-extraction"}, nil, "embedding"},
		{"BAAI/bge-base-en-v1.5-gguf", "", []string{"gguf"}, nil, "embedding"},
		{"someone/bge-reranker-v2-m3-GGUF", "", []string{"gguf", "reranker"}, nil, "reranker"},
		{"google/embeddinggemma-300m-GGUF", "", []string{"gguf"},
			[]string{"embeddinggemma-300m-Q8_0.gguf"}, "embedding"},
		{"bartowski/Llama-3.2-3B-Instruct-GGUF", "", []string{"gguf"}, nil, ""},
		{"williamliao/gemma-4-31B-it-EAGLE3-Speculator-GGUF", "",
			[]string{"gguf", "speculative-decoding"}, []string{"eagle3.gguf"}, ""},
	}
	for _, tc := range cases {
		if got := DetectEmbedding(tc.id, tc.pipe, tc.tags, tc.files); got != tc.want {
			t.Errorf("DetectEmbedding(%q) = %q, want %q", tc.id, got, tc.want)
		}
	}
}

func TestModalityLabel(t *testing.T) {
	if ModalityLabel([]string{"audio"}) != "audio" {
		t.Fatal("audio")
	}
	if ModalityLabel([]string{"vision"}) != "vision" {
		t.Fatal("vision")
	}
	if ModalityLabel([]string{"audio", "vision"}) != "audio+vision" {
		t.Fatal("mixed")
	}
	if ModalityLabel(nil) != "" {
		t.Fatal("empty")
	}
}

func TestDetectDraftSidecar(t *testing.T) {
	cases := []struct {
		id    string
		tags  []string
		files []string
		want  string
	}{
		{"meta-models/Muse-Glimmer-30B-GGUF", nil,
			[]string{
				"Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
				"mmproj-Muse-Glimmer-30B-Q4_K_M.gguf",
				"dflash-Muse-Glimmer-30B-Q4_K_M.gguf",
			}, "dflash"},
		{"z-lab/Qwen3-4B-DFlash-GGUF", []string{"speculative-decoding"}, nil, "dflash"},
		{"bartowski/Llama-3.2-3B-Instruct-GGUF", nil, nil, ""},
		{"org/repo", nil, []string{"eagle3.gguf"}, "eagle3"},
		{"org/repo", nil, []string{"mtp-Qwen3.6-27B-Q4_K_M.gguf"}, "mtp-draft"},
	}
	for _, tc := range cases {
		if got := DetectDraftSidecar(tc.id, tc.tags, tc.files); got != tc.want {
			t.Errorf("DetectDraftSidecar(%q) = %q, want %q", tc.id, got, tc.want)
		}
	}
}
