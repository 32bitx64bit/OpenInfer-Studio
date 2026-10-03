package mediagen

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func upscaleCapabilities(t *testing.T) map[string]any {
	return decodeCapabilityFixture(t, `{"supported_modes":["img_gen"],"upscale":true,"upscalers":[{"name":"RGB","model":true,"image_upscale":true},{"name":"Latent","model":true,"image_upscale":false},{"name":"Nearest","model":false,"image_upscale":false}],"limits":{"max_upscale_width":8192,"max_upscale_height":8192}}`)
}

func TestNativeUpscaleStoresManagedOutputAndDimensions(t *testing.T) {
	raw, path := pngInput(t)
	var request map[string]any
	f := newCapabilitySD(t, upscaleCapabilities(t), func(w http.ResponseWriter, r *http.Request) bool {
		if r.URL.Path != "/sdcpp/v1/upscale" {
			return false
		}
		if r.Header.Get("Authorization") != "Bearer secret" {
			http.Error(w, "missing key", 401)
			return true
		}
		_ = json.NewDecoder(r.Body).Decode(&request)
		_ = json.NewEncoder(w).Encode(map[string]any{"images": []any{map[string]any{"b64_json": base64.StdEncoding.EncodeToString(raw)}}, "output_format": "png", "width": 8, "height": 8, "scale": 2, "repeats": 1})
		return true
	})
	m := newJobManager(t, f)
	m.servers["model-1"].apiKey = "secret"
	j, err := m.UpscaleStage(context.Background(), "model-1", UpscaleParams{ImagePath: path, Upscaler: "RGB", TileSize: 256})
	if err != nil {
		t.Fatal(err)
	}
	if j.State != StateComplete || j.Width != 8 || j.Height != 8 || len(j.OutputPaths) != 1 {
		t.Fatalf("job=%+v", j)
	}
	if filepath.IsAbs(j.OutputPaths[0]) {
		t.Fatal("output path must be managed relative path")
	}
	got, err := os.ReadFile(filepath.Join(m.MediaDir(), j.OutputPaths[0]))
	if err != nil || string(got) != string(raw) {
		t.Fatalf("stored=%v err=%v", got, err)
	}
	if request["upscaler"] != "RGB" || request["repeats"] != float64(1) || request["tile_size"] != float64(256) {
		t.Fatalf("request=%v", request)
	}
	var params UpscaleParams
	_ = json.Unmarshal(j.Params, &params)
	if params.Image != "" || params.ImagePath != path {
		t.Fatal("local input was persisted as image bytes")
	}
}

func TestUpscaleRequiresExplicitRuntimeAndRGBCatalogSupport(t *testing.T) {
	_, path := pngInput(t)
	for _, tc := range []struct {
		name     string
		doc      map[string]any
		upscaler string
	}{
		{"unknown", nil, ""},
		{"unadvertised endpoint", decodeCapabilityFixture(t, `{"supported_modes":["img_gen"],"upscalers":[{"name":"RGB","model":true,"image_upscale":true}]}`), "RGB"},
		{"legacy catalog", decodeCapabilityFixture(t, `{"upscale":true,"upscalers":[{"name":"RGB"}]}`), "RGB"},
		{"latent model", upscaleCapabilities(t), "Latent"},
		{"builtin resize", upscaleCapabilities(t), "Nearest"},
		{"no model selected", upscaleCapabilities(t), ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			called := false
			f := newCapabilitySD(t, tc.doc, func(w http.ResponseWriter, r *http.Request) bool {
				if r.URL.Path == "/sdcpp/v1/upscale" {
					called = true
				}
				return false
			})
			m := newJobManager(t, f)
			if _, err := m.UpscaleStage(context.Background(), "model-1", UpscaleParams{ImagePath: path, Upscaler: tc.upscaler}); err == nil || called {
				t.Fatalf("unadvertised upscale submitted: err=%v called=%v", err, called)
			}
		})
	}
}

func TestUpscaleCancellationBoundsHTTPAndDoesNotStoreOutput(t *testing.T) {
	_, path := pngInput(t)
	f := newCapabilitySD(t, upscaleCapabilities(t), func(w http.ResponseWriter, r *http.Request) bool {
		if r.URL.Path != "/sdcpp/v1/upscale" {
			return false
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		<-r.Context().Done()
		return true
	})
	m := newJobManager(t, f)
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	start := time.Now()
	j, err := m.UpscaleStage(ctx, "model-1", UpscaleParams{ImagePath: path, Upscaler: "RGB"})
	if !errors.Is(err, context.DeadlineExceeded) || j == nil || j.State != StateCanceled || len(j.OutputPaths) != 0 || time.Since(start) > time.Second {
		t.Fatalf("job=%+v err=%v elapsed=%v", j, err, time.Since(start))
	}
}

func TestUpscaleRejectsBadRequestBounds(t *testing.T) {
	_, path := pngInput(t)
	m := NewManager(nil, nil, nil, nil, nil, nil)
	for _, p := range []UpscaleParams{{}, {ImagePath: path, Image: "also"}, {ImagePath: path, Repeats: 5}, {ImagePath: path, Repeats: -1}, {ImagePath: path, TileSize: 3000}, {ImagePath: path, OutputFormat: "../../bad"}} {
		if _, err := m.UpscaleStage(context.Background(), "model", p); err == nil {
			t.Fatalf("bad request accepted: %+v", p)
		}
	}
}
