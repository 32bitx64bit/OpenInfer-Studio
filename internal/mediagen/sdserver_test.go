package mediagen

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const sdHelpSample = `
Usage: sd-server [options]

Svr Options:
  -l,    --listen-ip TEXT           server listen ip (default: 127.0.0.1)
         --listen-port N            server listen port (default: 1234)

Context Options:
  -m,    --model FNAME              path to full model
         --vae FNAME                path to standalone vae model
         --taesd FNAME              path to taesd. Using Tiny AutoEncoder
         --control-net FNAME        path to control net model
  -t,    --threads N                number of threads to use during computation
         --backend PARAM            runtime backend assignment
         --max-vram N               optional per-device budget in GiB
         --offload-to-cpu           place the weights in RAM to save VRAM
         --diffusion-fa             use flash attention in the diffusion model only
         --vae-tiling               process vae in tiles to reduce memory usage
         --type TYPE                weight type (examples: f32, f16, q8_0)

Default Generation Options:
  -p,    --prompt TEXT              the prompt to render
  -W,    --width N                  image width, in pixel space (default: 512)
  -H,    --height N                 image height, in pixel space (default: 512)
         --steps N                  number of sample steps (default: 20)
         --cfg-scale N              unconditional guidance scale: (default: 7.0)
         --seed N                   RNG seed
         --video-frames N           video frames (default: 1)
`

func TestParseSDCapabilities(t *testing.T) {
	caps := ParseSDCapabilities(sdHelpSample)
	for _, want := range []string{"listen-ip", "listen-port", "vae", "taesd", "threads", "diffusion-fa", "vae-tiling", "video-frames"} {
		found := false
		for _, c := range caps {
			if c == want || strings.HasSuffix(c, want) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("capability %q missing in %v", want, caps)
		}
	}
}

func TestSupportsSDFlag(t *testing.T) {
	caps := ParseSDCapabilities(sdHelpSample)
	if !SupportsSDFlag(caps, sdHelpSample, "--vae") {
		t.Error("--vae should be supported")
	}
	if SupportsSDFlag(caps, sdHelpSample, "--temporal-tiling") {
		t.Error("--temporal-tiling absent from help, should be unsupported")
	}
}

func TestRawArgsRetainNegativeVRAMBudgetAndRejectShortFlags(t *testing.T) {
	caps := ParseSDCapabilities(sdHelpSample)
	args, warnings := parseSDRawArgs("--max-vram -2 --backend ROCm0", caps, sdHelpSample)
	if strings.Join(args, " ") != "--max-vram -2 --backend ROCm0" || len(warnings) != 0 {
		t.Fatalf("negative budget lost: %v %v", args, warnings)
	}
	args, warnings = parseSDRawArgs("--max-vram -x", caps, sdHelpSample)
	if strings.Contains(strings.Join(args, " "), "-x") || len(warnings) == 0 {
		t.Fatalf("short flag accepted as budget: %v %v", args, warnings)
	}
}

func TestBuildServerArgsGating(t *testing.T) {
	caps := ParseSDCapabilities(sdHelpSample)
	dir := t.TempDir()
	model := filepath.Join(dir, "sd15.safetensors")
	vaePath := filepath.Join(dir, "vae.safetensors")
	ctrlPath := filepath.Join(dir, "control.safetensors")
	for _, p := range []string{model, vaePath, ctrlPath} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	s := DefaultLoadSettings()
	s.VAE = vaePath
	s.ControlNet = ctrlPath // advertised, kept
	s.TemporalTiling = true // NOT advertised, dropped
	s.Threads = 8
	args, err := BuildServerArgs(s, model, false, caps, sdHelpSample, "127.0.0.1", 8123)
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{"--model " + model, "--vae " + vaePath, "--threads 8", "--listen-port 8123"} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in %q", want, joined)
		}
	}
	if strings.Contains(joined, "--temporal-tiling") {
		t.Errorf("unsupported --temporal-tiling leaked: %q", joined)
	}
}

func TestSDRequestBody(t *testing.T) {
	p := GenerateParams{Kind: KindImage, Prompt: "a cat", Width: 512, Height: 512, Steps: 20, CFGScale: 7, Seed: -1, Sampler: "Euler_A", Scheduler: "Discrete"}
	body := SDRequestBody(p)
	if body["prompt"] != "a cat" {
		t.Fatalf("prompt = %v", body["prompt"])
	}
	sp, _ := body["sample_params"].(map[string]any)
	if sp == nil || sp["sample_steps"] != 20 || sp["sample_method"] != "euler_a" || sp["scheduler"] != "discrete" {
		t.Fatalf("sample_params = %v", sp)
	}
	pv := GenerateParams{Kind: KindVideo, Prompt: "rain", VideoFrames: 33, FPS: 16}
	vb := SDRequestBody(pv)
	if vb["video_frames"] != 33 || vb["fps"] != 16 {
		t.Fatalf("video body = %v", vb)
	}
	if _, ok := vb["batch_count"]; ok {
		t.Fatalf("vid_gen must not carry batch_count: %v", vb)
	}
}

func TestValidateGenerateParams(t *testing.T) {
	p := GenerateParams{Width: 99999, Steps: 999, CFGScale: 99, BatchCount: 99, VideoFrames: 9999, FPS: 999}
	ValidateGenerateParams(&p)
	if p.Width != 4096 || p.Steps != 300 || p.CFGScale != 30 || p.BatchCount != 8 || p.VideoFrames != 512 || p.FPS != 60 {
		t.Fatalf("clamps not applied: %+v", p)
	}
	if p.Seed != -1 {
		t.Fatalf("seed default = %d, want -1", p.Seed)
	}
}
