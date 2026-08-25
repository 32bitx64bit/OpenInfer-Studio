package runtimes

import "testing"

const fakeHelp = `
usage: llama-server [options]

options:
  -m,    --model FNAME            model path
  -c,    --ctx-size N             size of the prompt context (default: 4096)
  -ngl,  --n-gpu-layers N         number of layers to store in VRAM
  -t,    --threads N              number of threads to use during generation
  -fa,   --flash-attn             enable Flash Attention
  -np,   --parallel N             number of slots for process requests
  -b,    --batch-size N           logical maximum batch size
         --cache-type-k TYPE      KV cache data type for K
         --mmproj FILE            path to a multimodal projector file
         --host HOST              ip address to listen on
         --port PORT              port to listen on
         --api-key KEY            API key
         --jinja                  use jinja chat template
`

func TestParseCapabilities(t *testing.T) {
	caps := ParseCapabilities(fakeHelp)
	want := []string{"ctx-size", "gpu-layers", "threads", "flash-attn", "parallel",
		"batch-size", "cache-type-k", "mmproj", "host", "port", "api-key", "jinja"}
	for _, w := range want {
		found := false
		for _, c := range caps {
			if c == w {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("capability %q not parsed from %q", w, caps)
		}
	}
}

func TestParseCapabilitiesReasoningPreserve(t *testing.T) {
	help := fakeHelp + `
         --reasoning-preserve, --no-reasoning-preserve
         -rea, --reasoning [on|off|auto]
`
	caps := ParseCapabilities(help)
	want := []string{"reasoning-preserve", "no-reasoning-preserve", "reasoning"}
	for _, w := range want {
		found := false
		for _, c := range caps {
			if c == w {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("capability %q not parsed from %q", w, caps)
		}
	}
	if !SupportsFlag(caps, help, "--reasoning-preserve") {
		t.Error("--reasoning-preserve should be supported")
	}
	if !SupportsFlag(caps, help, "--no-reasoning-preserve") {
		t.Error("--no-reasoning-preserve should be supported")
	}
}

func TestSupportsFlag(t *testing.T) {
	caps := ParseCapabilities(fakeHelp)
	if !SupportsFlag(caps, fakeHelp, "--ctx-size") {
		t.Error("--ctx-size should be supported")
	}
	if SupportsFlag(caps, fakeHelp, "--tensor-split") {
		t.Error("--tensor-split not in help, should be unsupported")
	}
	// Unknown-to-us flag absent from help must be rejected.
	if SupportsFlag(caps, fakeHelp, "--some-future-flag") {
		t.Error("future flag not in help should be unsupported")
	}
	helpWithFuture := fakeHelp + "\n  --some-future-flag X  future\n"
	if !SupportsFlag(caps, helpWithFuture, "--some-future-flag") {
		t.Error("future flag present in help should be permitted")
	}
}
