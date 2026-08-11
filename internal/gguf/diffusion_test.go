package gguf

import "testing"

func TestDetectDiffusion(t *testing.T) {
	cases := []struct {
		name   string
		arch   string
		gguf   string
		path   string
		raw    map[string]any
		want   bool
		canvas uint32
	}{
		{
			name:   "arch diffusion-gemma",
			arch:   "diffusion-gemma",
			path:   "/models/foo.gguf",
			want:   true,
			canvas: 256,
		},
		{
			name:   "canvas kv",
			arch:   "llama",
			raw:    map[string]any{"diffusion.canvas_length": uint32(256)},
			path:   "/models/foo.gguf",
			want:   true,
			canvas: 256,
		},
		{
			name:   "filename diffusiongemma",
			arch:   "",
			path:   "/models/DiffusionGemma-26B-A4B-it-Q4_K_M.gguf",
			want:   true,
			canvas: 256,
		},
		{
			name:   "filename diffusion-gemma",
			arch:   "",
			path:   "/x/diffusion-gemma-q8.gguf",
			want:   true,
			canvas: 256,
		},
		{
			name: "normal gemma4",
			arch: "gemma4",
			path: "/models/gemma-4-12B-it-Q4_K_M.gguf",
			want: false,
		},
		{
			name: "mmproj ignored",
			arch: "",
			path: "/models/mmproj-diffusion-something.gguf",
			want: false,
		},
		{
			name:   "custom canvas",
			arch:   "diffusion-gemma",
			raw:    map[string]any{"diffusion.canvas_length": uint32(512)},
			path:   "/m.gguf",
			want:   true,
			canvas: 512,
		},
	}
	for _, c := range cases {
		got, canvas := DetectDiffusion(c.arch, c.gguf, c.path, c.raw)
		if got != c.want || (c.want && canvas != c.canvas) {
			t.Errorf("%s: got (%v,%d) want (%v,%d)", c.name, got, canvas, c.want, c.canvas)
		}
	}
}

func TestApplyDiffusionFlagsClearsDraftEmbed(t *testing.T) {
	md := &Metadata{
		Architecture:     "diffusion-gemma",
		SpeculativeDraft: true,
		IsEmbedding:      true,
		Raw:              map[string]any{"diffusion.canvas_length": uint32(256)},
	}
	md.ApplyDiffusionFlags("/x/diffusiongemma.gguf")
	if !md.IsDiffusion || md.CanvasLength != 256 {
		t.Fatalf("flags = %+v", md)
	}
	if md.SpeculativeDraft || md.IsEmbedding {
		t.Fatalf("draft/embed should be cleared: %+v", md)
	}
}
