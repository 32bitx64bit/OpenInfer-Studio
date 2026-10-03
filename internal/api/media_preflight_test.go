package api

import (
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/mediagen"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

func TestDiffusionPreviewPreservesIncompatibleComponentReport(t *testing.T) {
	h, db, _, dir := newSaveTest(t)
	exe := filepath.Join(dir, "sd-server")
	help := "--model --diffusion-model --vae --listen-ip --listen-port"
	for path, data := range map[string][]byte{exe: nil, filepath.Join(dir, "help.txt"): []byte(help)} {
		if err := os.WriteFile(path, data, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.Exec(`INSERT INTO runtimes (id, installed_at, install_dir, executable_path) VALUES ('sd', '', ?, ?)`, dir, exe); err != nil {
		t.Fatal(err)
	}
	h.d.Layout = &config.Layout{DataDir: dir, Models: dir}
	h.d.RT = runtimes.NewManager(db, dir, nil, nil, nil)
	h.d.Media = mediagen.NewManager(db, h.d.Layout, h.d.RT, nil, nil, nil)

	writeHeader := func(path string, names ...string) {
		t.Helper()
		header := map[string]any{}
		for _, name := range names {
			header[name] = map[string]any{"dtype": "BF16", "shape": []int{1}, "data_offsets": []int{0, 0}}
		}
		b, err := json.Marshal(header)
		if err != nil {
			t.Fatal(err)
		}
		data := make([]byte, 8)
		binary.LittleEndian.PutUint64(data, uint64(len(b)))
		if err := os.WriteFile(path, append(data, b...), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	model := &models.Model{ID: "ltx", PrimaryPath: filepath.Join(dir, "model.safetensors")}
	vae := filepath.Join(dir, "video_vae.safetensors")
	writeHeader(model.PrimaryPath, "transformer_blocks.0.attn1.to_q.weight")
	writeHeader(vae, "decoder.conv_in_x_t.weight", "decoder.diff_blocks.0.attn.qkv.weight")

	preview := func() map[string]any {
		t.Helper()
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/preview", strings.NewReader(`{"runtime_id":"sd"}`))
		h.previewDiffusionLoad(w, r, model)
		if w.Code != http.StatusOK {
			t.Fatalf("status %d: %s", w.Code, w.Body.String())
		}
		var got map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		return got
	}
	got := preview()
	if got["can_load"] != false || got["command"] != "" {
		t.Fatalf("incompatible launch allowed: %v", got)
	}
	components := got["components"].([]any)
	if len(components) != 1 || components[0].(map[string]any)["status"] != "incompatible" {
		t.Fatalf("component report lost: %v", got)
	}
	if !strings.Contains(got["warnings"].([]any)[0].(string), "diffusion-decoder") {
		t.Fatalf("correction not displayed: %v", got)
	}

	// A corrected file at the same path must clear the blocker; do not cache
	// the old header or leave the UI permanently unable to load this model.
	writeHeader(vae, "decoder.conv_in.conv.weight", "decoder.conv_out.conv.weight")
	got = preview()
	if got["can_load"] != true || !strings.Contains(got["command"].(string), "--vae "+vae) {
		t.Fatalf("corrected pipeline still blocked: %v", got)
	}
}
