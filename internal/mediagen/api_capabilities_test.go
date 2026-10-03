package mediagen

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func decodeCapabilityFixture(t *testing.T, fixture string) map[string]any {
	t.Helper()
	var doc map[string]any
	if err := json.Unmarshal([]byte(fixture), &doc); err != nil {
		t.Fatal(err)
	}
	return doc
}

func TestAPICapabilitiesModeAndLegacyBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name, fixture                 string
		known, mask, video, videoInit bool
	}{
		{"modern", `{"supported_modes":["img_gen","vid_gen"],"current_mode":"img_gen","features_by_mode":{"img_gen":{"mask_image":true,"ref_images":"true"},"vid_gen":{"init_image":false}},"features":{"init_image":true}}`, true, true, true, false},
		{"legacy current mode", `{"current_mode":"img_gen","features":{"mask_image":true,"init_image":true}}`, true, true, false, false},
		{"legacy video", `{"current_mode":"vid_gen","features":{"init_image":true}}`, true, false, true, true},
		{"legacy ambiguous", `{"features":{"mask_image":true,"init_image":true},"samplers":["euler"]}`, false, false, false, false},
		{"empty modern defeats mirror", `{"supported_modes":[],"current_mode":"img_gen","features":{"mask_image":true}}`, false, false, false, false},
		{"modern false defeats mirror", `{"supported_modes":["img_gen"],"current_mode":"img_gen","features_by_mode":{"img_gen":{"mask_image":false}},"features":{"mask_image":true}}`, true, false, false, false},
		{"malformed modern defeats mirror", `{"supported_modes":["img_gen"],"current_mode":"img_gen","features_by_mode":null,"features":{"mask_image":true}}`, true, false, false, false},
		{"unadvertised feature mode", `{"supported_modes":["img_gen"],"features_by_mode":{"vid_gen":{"init_image":true}}}`, true, false, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := normalizeAPICapabilities(decodeCapabilityFixture(t, tc.fixture), "m", "r")
			if c.Known != tc.known || c.SupportsFeature("img_gen", "mask_image") != tc.mask || c.SupportsMode("vid_gen") != tc.video || c.SupportsFeature("vid_gen", "init_image") != tc.videoInit {
				t.Fatalf("incorrect advertisement: %+v", c)
			}
			if c.SupportsFeature("img_gen", "ref_images") {
				t.Fatal("a string must not enable a feature")
			}
		})
	}
}

func TestAPICapabilitiesChoicesAndUnsupportedFeatures(t *testing.T) {
	c := normalizeAPICapabilities(decodeCapabilityFixture(t, `{"supported_modes":["img_gen"],"samplers":["euler",null,"euler",5],"schedulers":["discrete"],"features_by_mode":{"img_gen":{"init_image":true}},"output_formats_by_mode":{"img_gen":["png"]},"limits":{"max_width":512}}`), "m", "r")
	if !reflect.DeepEqual(c.Samplers, []string{"euler"}) {
		t.Fatalf("samplers=%v", c.Samplers)
	}
	for _, p := range []GenerateParams{{Kind: KindVideo}, {MaskImage: "mask"}, {ControlImage: "control"}, {RefImages: []string{"ref"}}, {Lora: []Lora{{Path: "lora"}}}, {Sampler: "invented"}, {Scheduler: "karras"}, {Width: 1024}, {OutputFormat: "webp"}} {
		if err := c.ValidateGeneration(p); err == nil {
			t.Fatalf("unadvertised request accepted: %+v", p)
		}
	}
	if err := c.ValidateGeneration(GenerateParams{InitImage: "input", Sampler: "Euler", Scheduler: "Discrete", OutputFormat: "png", Width: 512}); err != nil {
		t.Fatal(err)
	}
	unknown := normalizeAPICapabilities(nil, "m", "")
	for _, p := range []GenerateParams{{InitImage: "input"}, {Kind: KindVideo}, {MaskImage: "input"}, {Prompt: "p <lora:demo:1>"}} {
		if err := unknown.ValidateGeneration(p); err == nil || !strings.Contains(err.Error(), unknown.Reason) {
			t.Fatalf("unknown capability did not explain rejection: %v", err)
		}
	}
}

func newCapabilitySD(t *testing.T, doc map[string]any, extra func(http.ResponseWriter, *http.Request) bool) *fakeSD {
	t.Helper()
	f := &fakeSD{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if extra != nil && extra(w, r) {
			return
		}
		switch {
		case r.URL.Path == "/sdcpp/v1/capabilities":
			if doc == nil {
				http.NotFound(w, r)
				return
			}
			_ = json.NewEncoder(w).Encode(doc)
		case r.Method == http.MethodPost && (r.URL.Path == "/sdcpp/v1/img_gen" || r.URL.Path == "/sdcpp/v1/vid_gen"):
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				http.Error(w, err.Error(), 400)
				return
			}
			f.requests = append(f.requests, body)
			f.poll = okPoll()
			if r.URL.Path == "/sdcpp/v1/vid_gen" {
				f.poll = map[string]any{"status": "completed", "result": map[string]any{"output_format": "webm", "b64_json": base64.StdEncoding.EncodeToString([]byte("VIDEO"))}}
			}
			w.WriteHeader(202)
			_ = json.NewEncoder(w).Encode(map[string]string{"id": "sd-1"})
		case r.URL.Path == "/sdcpp/v1/jobs/sd-1":
			_ = json.NewEncoder(w).Encode(f.poll)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func pngInput(t *testing.T) ([]byte, string) {
	t.Helper()
	var b bytes.Buffer
	if err := png.Encode(&b, image.NewNRGBA(image.Rect(0, 0, 8, 8))); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "input.png")
	if err := os.WriteFile(path, b.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return b.Bytes(), path
}

func TestNativeGenerationConditioningAndStructuredLora(t *testing.T) {
	doc := decodeCapabilityFixture(t, `{"supported_modes":["img_gen"],"features_by_mode":{"img_gen":{"init_image":true,"mask_image":true,"control_image":true,"ref_images":true,"lora":true}},"samplers":["euler"],"schedulers":["discrete"],"loras":[{"name":"film","path":"film.safetensors"}]}`)
	f := newCapabilitySD(t, doc, nil)
	m := newJobManager(t, f)
	raw, path := pngInput(t)
	p := GenerateParams{Prompt: "portrait <lora:film:0.65>", InitImagePath: path, MaskImagePath: path, ControlImagePath: path, ControlStrength: 0.8, RefImagePaths: []string{path}, Sampler: "euler", Scheduler: "discrete", Guidance: 3.5, CFGScale: 7, Strength: 0.7}
	j, err := m.RunStage(context.Background(), "model-1", p, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(f.requests) != 1 {
		t.Fatalf("requests=%v", f.requests)
	}
	body := f.requests[0]
	if body["prompt"] != "portrait" || body["control_strength"] != 0.8 {
		t.Fatalf("body=%v", body)
	}
	for _, field := range []string{"init_image", "mask_image", "control_image"} {
		got, _ := body[field].(string)
		if !strings.HasSuffix(got, base64.StdEncoding.EncodeToString(raw)) {
			t.Fatalf("%s was not encoded", field)
		}
	}
	refs, _ := body["ref_images"].([]any)
	if len(refs) != 1 {
		t.Fatalf("refs=%v", refs)
	}
	loras, _ := body["lora"].([]any)
	if len(loras) != 1 || loras[0].(map[string]any)["path"] != "film.safetensors" || loras[0].(map[string]any)["multiplier"] != 0.65 {
		t.Fatalf("loras=%v", loras)
	}
	guidance := body["sample_params"].(map[string]any)["guidance"].(map[string]any)
	if guidance["distilled_guidance"] != 3.5 || guidance["txt_cfg"] != float64(7) {
		t.Fatalf("guidance=%v", guidance)
	}
	if bytes.Contains(j.Params, []byte("data:image")) || strings.Contains(string(j.Params), base64.StdEncoding.EncodeToString(raw)) {
		t.Fatal("local image payload leaked into persisted params")
	}
	if p.Prompt != "portrait <lora:film:0.65>" || len(p.Lora) != 0 {
		t.Fatal("conversion mutated caller params")
	}
}

func TestUnknownLoadedModelRejectsBeforeSubmittingAdvancedRequest(t *testing.T) {
	f := newCapabilitySD(t, nil, nil)
	m := newJobManager(t, f)
	_, path := pngInput(t)
	for _, p := range []GenerateParams{{Prompt: "p", InitImagePath: path}, {Prompt: "p <lora:film:1>"}, {Prompt: "p", Kind: KindVideo}} {
		job, err := m.RunStage(context.Background(), "model-1", p, nil)
		if err == nil || job == nil || job.State != StateFailed || !strings.Contains(err.Error(), "advertis") {
			t.Fatalf("job=%+v err=%v", job, err)
		}
	}
	if len(f.requests) != 0 {
		t.Fatal("advanced input reached unsupported runtime")
	}
}

func TestImageToVideoRequiresModeSpecificAdvertisement(t *testing.T) {
	_, path := pngInput(t)
	for _, enabled := range []bool{false, true} {
		t.Run(fmtBool(enabled), func(t *testing.T) {
			doc := map[string]any{"supported_modes": []any{"vid_gen"}, "features_by_mode": map[string]any{"vid_gen": map[string]any{"init_image": enabled}}}
			f := newCapabilitySD(t, doc, nil)
			m := newJobManager(t, f)
			_, err := m.RunStage(context.Background(), "model-1", GenerateParams{Kind: KindVideo, Prompt: "p", InitImagePath: path, VideoFrames: 17, FPS: 8}, nil)
			if enabled {
				if err != nil {
					t.Fatal(err)
				}
				if f.requests[0]["init_image"] == nil {
					t.Fatal("video init_image dropped")
				}
			} else if err == nil || len(f.requests) != 0 {
				t.Fatal("unadvertised i2v submitted")
			}
		})
	}
}

func fmtBool(b bool) string {
	if b {
		return "advertised"
	}
	return "unsupported"
}

func TestCapabilityDiscoveryBoundsAndAuthentication(t *testing.T) {
	for _, tc := range []struct {
		name    string
		code    int
		body    string
		wantErr bool
		known   bool
	}{
		{"missing", 404, "", false, false}, {"failed", 500, `{"supported_modes":["img_gen"]}`, true, false},
		{"null", 200, "null", true, false}, {"trailing", 200, `{} {}`, true, false},
		{"oversized", 200, strings.Repeat(" ", maxCapabilityResponseBytes+1), true, false},
		{"valid", 200, `{"supported_modes":["img_gen"]}`, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newCapabilitySD(t, nil, func(w http.ResponseWriter, r *http.Request) bool {
				if r.Header.Get("Authorization") != "Bearer secret" {
					http.Error(w, "missing key", 401)
					return true
				}
				w.WriteHeader(tc.code)
				_, _ = w.Write([]byte(tc.body))
				return true
			})
			m := newJobManager(t, f)
			m.servers["model-1"].apiKey = "secret"
			c, err := m.GenerationCapabilities("model-1")
			if (err != nil) != tc.wantErr || (err == nil && c.Known != tc.known) {
				t.Fatalf("caps=%+v err=%v", c, err)
			}
		})
	}
	m := NewManager(nil, nil, nil, nil, nil, nil)
	c, err := m.GenerationCapabilities("stopped")
	if err != nil || c.Known || c.Reason == "" {
		t.Fatalf("stopped=%+v err=%v", c, err)
	}
}

func TestCapabilitiesContextBoundsIncompleteBody(t *testing.T) {
	f := newCapabilitySD(t, nil, func(w http.ResponseWriter, r *http.Request) bool {
		_, _ = w.Write([]byte(`{"`))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		return true
	})
	m := newJobManager(t, f)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := m.Capabilities(ctx, "model-1"); err == nil || time.Since(start) > time.Second {
		t.Fatalf("unbounded capability response: %v", err)
	}
}
