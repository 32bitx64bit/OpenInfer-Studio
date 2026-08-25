package huggingface

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRepoPopulatesSHAAndSafetensors(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/model", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":  "org/model",
			"sha": "deadbeefcafebabe",
			"safetensors": map[string]any{
				"parameters": map[string]int64{"BF16": 42},
				"total":      42,
			},
			"siblings": []any{},
			"tags":     []string{"image-text-to-text"},
		})
	})
	mux.HandleFunc("/api/models/org/model/tree/main", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"path": "config.json", "size": 12, "type": "file"},
			{"path": "model.safetensors", "size": 100, "type": "file"},
		})
	})
	mux.HandleFunc("/org/model/raw/main/README.md", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(404)
	})
	mux.HandleFunc("/org/model/resolve/main/config.json", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"model_type":"muse_glimmer"}`))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	c := NewClient()
	c.SetBaseURL(srv.URL)
	info, err := c.Repo(context.Background(), "org/model")
	if err != nil {
		t.Fatal(err)
	}
	if info.SHA != "deadbeefcafebabe" {
		t.Fatalf("sha %q", info.SHA)
	}
	if info.SafetensorsParameters["BF16"] != 42 {
		t.Fatalf("params %+v", info.SafetensorsParameters)
	}
	if len(info.Files) != 2 {
		t.Fatalf("files %d", len(info.Files))
	}
	raw, err := c.FetchFile(context.Background(), "org/model", "config.json", 4096)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != `{"model_type":"muse_glimmer"}` {
		t.Fatalf("config %s", raw)
	}
}
