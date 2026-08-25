package diagnostics

import (
	"strings"
	"testing"
)

func TestRedact(t *testing.T) {
	cases := map[string]string{
		"Authorization: Bearer hf_abcdefghijklmnopqrstuvwxyz123456": "hf_",
		"api-key=sk-oi-abcdef1234567890":                            "sk-oi-abcdef1234567890",
		"token=eyJhbGciOiJIUzI1NiJ9.abcdef":                         "eyJhbGciOiJIUzI1NiJ9",
		"normal log line without secrets":                           "normal log line without secrets",
	}
	for in, mustNotContain := range cases {
		out := Redact(in)
		if mustNotContain == in {
			if out != in {
				t.Errorf("plain log line modified: %q → %q", in, out)
			}
			continue
		}
		if strings.Contains(out, mustNotContain) {
			t.Errorf("secret not redacted: %q → %q", in, out)
		}
		if !strings.Contains(out, "[REDACTED]") {
			t.Errorf("no redaction marker in %q", out)
		}
	}
}

func TestClassify(t *testing.T) {
	cases := []struct {
		log  string
		code int
		want FailureClass
	}{
		{"llama_model_load: error loading model: unknown model architecture: bailing", 1, ClassUnsupportedArch},
		{"CUDA error: out of memory when allocating buffer", 1, ClassInsufficientVRAM},
		{"failed to allocate kv cache", 1, ClassContextAlloc},
		{"error: unrecognized argument: --bogus-flag", 1, ClassInvalidFlag},
		{"bind(): address already in use", 1, ClassPortConflict},
		{"std::bad_alloc", 1, ClassInsufficientRAM},
		{"failed to find a memory slot for batch of size 66", -1, ClassRuntimeCrash},
		{"slot operator(): failed to decode mtmd chunk, idx = 4127, res = 1", -1, ClassRuntimeCrash},
		{"check_tensor_dims: tensor 'token_embd.weight' has wrong shape; expected   5120, 248077, got   5120, 248320,      1,      1", 1, ClassTensorShape},
		{"completely novel error nobody has seen", 1, ClassUnknown},
	}
	for _, c := range cases {
		got := Classify(c.log, c.code, false)
		if got.Class != c.want {
			t.Errorf("Classify(%q) = %s, want %s", c.log, got.Class, c.want)
		}
		if got.Summary == "" || got.Suggestion == "" {
			t.Errorf("classification lacks summary/suggestion for %q", c.log)
		}
	}
	if got := Classify("", -1, true); got.Class != ClassTimeout {
		t.Errorf("timeout not classified: %s", got.Class)
	}
}

func TestRedactPaths(t *testing.T) {
	out := RedactPaths("/home/alice/models/m.gguf error", "/home/alice")
	if !strings.HasPrefix(out, "~/") {
		t.Errorf("home not redacted: %q", out)
	}
}
