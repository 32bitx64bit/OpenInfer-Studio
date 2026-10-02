package huggingface

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
)

// searchServer answers /api/models by the query's filter / pipeline_tag
// source and records every query it saw.
type searchServer struct {
	mu      sync.Mutex
	queries []string
	rows    map[string][]map[string]any // source key → rows
	fail    map[string]int              // source key → HTTP status
}

func sourceKey(r *http.Request) string {
	q := r.URL.Query()
	if p := q.Get("pipeline_tag"); p != "" {
		return "pipeline_tag=" + p
	}
	return "filter=" + q.Get("filter")
}

func (s *searchServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	key := sourceKey(r)
	s.mu.Lock()
	s.queries = append(s.queries, key)
	status := s.fail[key]
	rows := s.rows[key]
	s.mu.Unlock()
	if status != 0 {
		http.Error(w, "boom", status)
		return
	}
	_ = json.NewEncoder(w).Encode(rows)
}

func (s *searchServer) seen() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := append([]string(nil), s.queries...)
	sort.Strings(out)
	return out
}

func newSearchClient(t *testing.T, s *searchServer) *Client {
	t.Helper()
	mux := http.NewServeMux()
	mux.Handle("/api/models", s)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	c := NewClient()
	c.SetBaseURL(srv.URL)
	return c
}

func row(id string, downloads, likes int64, pipe string, tags ...string) map[string]any {
	return map[string]any{
		"id": id, "downloads": downloads, "likes": likes,
		"pipeline_tag": pipe, "tags": tags,
		"lastModified": "2026-01-01T00:00:00Z",
	}
}

func ids(rs []SearchResult) string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.ID
	}
	return strings.Join(out, ",")
}

func TestSearchKindAllQueriesEverySource(t *testing.T) {
	s := &searchServer{rows: map[string][]map[string]any{}}
	c := newSearchClient(t, s)
	if _, err := c.SearchKind(context.Background(), "q", "", 10, SearchKindAll); err != nil {
		t.Fatal(err)
	}
	// Browsing with no query stays on the tagged sources.
	s2 := &searchServer{rows: map[string][]map[string]any{}}
	if _, err := newSearchClient(t, s2).SearchKind(context.Background(), "", "", 10, SearchKindAll); err != nil {
		t.Fatal(err)
	}
	if got := s2.seen(); len(got) != 7 || got[0] == "filter=" {
		t.Fatalf("empty query should not run the untagged source: %v", got)
	}
	want := []string{
		"filter=", // untagged: a query is there to match repository names against
		"filter=comfyui", "filter=diffusers", "filter=diffusion-single-file", "filter=gguf",
		"pipeline_tag=image-to-video", "pipeline_tag=text-to-image", "pipeline_tag=text-to-video",
	}
	if got := s.seen(); strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("sources queried = %v, want %v", got, want)
	}
}

func TestSearchKindLegacyKindsStaySingleSource(t *testing.T) {
	for kind, want := range map[string]string{
		"": "filter=gguf", "llm": "filter=gguf",
		"diffusion": "filter=comfyui filter=diffusers filter=diffusion-single-file",
	} {
		s := &searchServer{rows: map[string][]map[string]any{}}
		c := newSearchClient(t, s)
		if _, err := c.SearchKind(context.Background(), "q", "", 10, kind); err != nil {
			t.Fatal(err)
		}
		if got := strings.Join(s.seen(), " "); got != want {
			t.Errorf("kind %q queried %v, want %s", kind, got, want)
		}
	}
}

func TestSearchKindAllDedupesAndSortsAcrossSources(t *testing.T) {
	s := &searchServer{rows: map[string][]map[string]any{
		"filter=gguf": {
			row("org/chat-gguf", 500, 5, "text-generation", "gguf"),
			// Also surfaces through pipeline_tag below.
			row("city96/FLUX.1-dev-gguf", 900, 9, "text-to-image", "gguf", "text-to-image"),
		},
		"filter=diffusers": {
			row("black-forest-labs/FLUX.1-dev", 5000, 50, "text-to-image", "diffusers"),
		},
		"pipeline_tag=text-to-image": {
			row("city96/FLUX.1-dev-gguf", 900, 9, "text-to-image", "gguf", "text-to-image"),
			row("Comfy-Org/flux1-dev", 700, 7, "text-to-image"),
		},
		"pipeline_tag=text-to-video": {
			row("Wan-AI/Wan2.1-T2V-1.3B", 300, 3, "text-to-video", "diffusers"),
		},
	}}
	c := newSearchClient(t, s)
	got, err := c.SearchKind(context.Background(), "flux", "downloads", 10, SearchKindAll)
	if err != nil {
		t.Fatal(err)
	}
	want := "black-forest-labs/FLUX.1-dev,city96/FLUX.1-dev-gguf,Comfy-Org/flux1-dev,org/chat-gguf,Wan-AI/Wan2.1-T2V-1.3B"
	if ids(got) != want {
		t.Fatalf("merged = %s\nwant    %s", ids(got), want)
	}
	for _, r := range got {
		isGen := r.ID != "org/chat-gguf"
		if isGen && r.Diffusion == "" {
			t.Errorf("%s not labelled as a generator", r.ID)
		}
		if !isGen && r.Diffusion != "" {
			t.Errorf("chat model %s labelled %q", r.ID, r.Diffusion)
		}
	}
}

func TestSearchKindAllRelevanceInterleavesSources(t *testing.T) {
	s := &searchServer{rows: map[string][]map[string]any{
		"filter=gguf":      {row("a/g1", 1, 1, ""), row("a/g2", 1, 1, ""), row("a/g3", 1, 1, "")},
		"filter=diffusers": {row("b/d1", 1, 1, "text-to-image", "diffusers"), row("b/d2", 1, 1, "text-to-image", "diffusers")},
	}}
	c := newSearchClient(t, s)
	got, err := c.SearchKind(context.Background(), "q", "", 4, SearchKindAll)
	if err != nil {
		t.Fatal(err)
	}
	// Sources keep their own order and alternate, so neither side starves
	// the other; the limit truncates the interleaved list.
	if want := "a/g1,b/d1,a/g2,b/d2"; ids(got) != want {
		t.Fatalf("interleaved = %s, want %s", ids(got), want)
	}
}

func TestSearchKindAllToleratesPartialFailure(t *testing.T) {
	s := &searchServer{
		rows: map[string][]map[string]any{
			"filter=gguf": {row("a/g1", 1, 1, "")},
		},
		fail: map[string]int{
			"filter=diffusers":           http.StatusInternalServerError,
			"pipeline_tag=text-to-image": http.StatusTooManyRequests,
		},
	}
	c := newSearchClient(t, s)
	got, err := c.SearchKind(context.Background(), "q", "", 10, SearchKindAll)
	if err != nil {
		t.Fatalf("partial failure must not fail the search: %v", err)
	}
	if ids(got) != "a/g1" {
		t.Fatalf("results = %s", ids(got))
	}
}

func TestSearchKindAllFailsWhenEverySourceFails(t *testing.T) {
	s := &searchServer{fail: map[string]int{
		"filter=gguf": 500, "filter=diffusers": 500,
		"pipeline_tag=text-to-image": 500, "pipeline_tag=text-to-video": 500, "pipeline_tag=image-to-video": 500,
		"filter=": 500, "filter=comfyui": 500, "filter=diffusion-single-file": 500,
	}}
	c := newSearchClient(t, s)
	_, err := c.SearchKind(context.Background(), "q", "", 10, SearchKindAll)
	if err == nil {
		t.Fatal("expected an error when every source fails")
	}
	if _, ok := err.(*APIError); !ok {
		t.Fatalf("error %T should stay an *APIError so the API reports the upstream status", err)
	}
}

func TestSearchKindAllEmptyResultIsNonNil(t *testing.T) {
	s := &searchServer{rows: map[string][]map[string]any{}}
	c := newSearchClient(t, s)
	got, err := c.SearchKind(context.Background(), "q", "likes", 10, SearchKindAll)
	if err != nil {
		t.Fatal(err)
	}
	if got == nil {
		t.Fatal("empty search must encode as [] not null")
	}
}

func TestRepoRef(t *testing.T) {
	cases := map[string]string{
		"https://huggingface.co/Comfy-Org/MiniMax-H3":                       "Comfy-Org/MiniMax-H3",
		"https://huggingface.co/Comfy-Org/MiniMax-H3/":                      "Comfy-Org/MiniMax-H3",
		"https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/split_files": "Comfy-Org/MiniMax-H3",
		"https://huggingface.co/Comfy-Org/MiniMax-H3?library=comfyui":       "Comfy-Org/MiniMax-H3",
		"  http://www.huggingface.co/Comfy-Org/MiniMax-H3#files ":           "Comfy-Org/MiniMax-H3",
		"huggingface.co/Comfy-Org/MiniMax-H3":                               "Comfy-Org/MiniMax-H3",
		"HTTPS://HuggingFace.co/Comfy-Org/MiniMax-H3":                       "Comfy-Org/MiniMax-H3",
		"hf.co/unsloth/Qwen3-8B-GGUF":                                       "unsloth/Qwen3-8B-GGUF",
		"Comfy-Org/MiniMax-H3":                                              "Comfy-Org/MiniMax-H3",
		"city96/FLUX.1-dev-gguf":                                            "city96/FLUX.1-dev-gguf",
		// Not a repository.
		"":                     "",
		"flux":                 "",
		"flux gguf":            "",
		"comfy org/minimax h3": "",
		"https://huggingface.co/models?search=wan": "",
		"https://huggingface.co/datasets/a/b":      "",
		"https://huggingface.co/Comfy-Org":         "",
		"https://example.com/Comfy-Org/MiniMax-H3": "",
		"a/b/c":       "",
		"/etc/passwd": "",
	}
	for in, want := range cases {
		if got := RepoRef(in); got != want {
			t.Errorf("RepoRef(%q) = %q, want %q", in, got, want)
		}
	}
}

// exactServer is a Hub that also answers /api/models/<owner>/<name>.
func exactServer(t *testing.T, s *searchServer, exact map[string]map[string]any) *Client {
	t.Helper()
	mux := http.NewServeMux()
	mux.Handle("/api/models", s)
	mux.HandleFunc("/api/models/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/api/models/")
		row, ok := exact[id]
		if !ok {
			http.Error(w, "nope", http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(row)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	c := NewClient()
	c.SetBaseURL(srv.URL)
	return c
}

func minimaxRow() map[string]any {
	r := row("Comfy-Org/MiniMax-H3", 10, 1, "")
	r["siblings"] = []map[string]any{
		{"rfilename": "split_files/diffusion_models/minimax_h3_bf16.safetensors"},
		{"rfilename": "split_files/vae/minimax_vae.safetensors"},
	}
	return r
}

// A repository with no gguf / diffusers / task tag cannot be reached by any
// tagged source; pasting its URL still finds it.
func TestSearchPastedURLFindsUntaggedRepo(t *testing.T) {
	s := &searchServer{rows: map[string][]map[string]any{}}
	c := exactServer(t, s, map[string]map[string]any{"Comfy-Org/MiniMax-H3": minimaxRow()})
	got, err := c.SearchKind(context.Background(), "https://huggingface.co/Comfy-Org/MiniMax-H3", "downloads", 10, SearchKindAll)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != "Comfy-Org/MiniMax-H3" {
		t.Fatalf("results = %s", ids(got))
	}
	if got[0].Diffusion != DiffusionVideo {
		t.Errorf("untagged comfy-org safetensors repo should read as a video generator, got %q", got[0].Diffusion)
	}
}

func TestSearchExactRepoIsPinnedFirstAndNotDuplicated(t *testing.T) {
	s := &searchServer{rows: map[string][]map[string]any{
		"filter=gguf":                {row("org/other-gguf", 9000, 90, "", "gguf")},
		"pipeline_tag=text-to-video": {minimaxRowWithTag()},
	}}
	c := exactServer(t, s, map[string]map[string]any{"Comfy-Org/MiniMax-H3": minimaxRow()})
	got, err := c.SearchKind(context.Background(), "Comfy-Org/MiniMax-H3", "downloads", 10, SearchKindAll)
	if err != nil {
		t.Fatal(err)
	}
	if ids(got) != "Comfy-Org/MiniMax-H3,org/other-gguf" {
		t.Fatalf("results = %s (the asked-for repo goes first even though another row has more downloads, and appears once)", ids(got))
	}
}

func minimaxRowWithTag() map[string]any {
	r := minimaxRow()
	r["pipeline_tag"] = "text-to-video"
	return r
}

func TestSearchExactRepoMissingFallsBackToSearch(t *testing.T) {
	s := &searchServer{rows: map[string][]map[string]any{
		"filter=gguf": {row("someone/thing-gguf", 5, 5, "", "gguf")},
	}}
	c := exactServer(t, s, nil) // the lookup answers 404
	got, err := c.SearchKind(context.Background(), "someone/thing", "", 10, SearchKindAll)
	if err != nil {
		t.Fatalf("a missing exact repo must not fail the search: %v", err)
	}
	if ids(got) != "someone/thing-gguf" {
		t.Fatalf("results = %s", ids(got))
	}
}

func TestSearchExactRepoSurvivesEverySourceFailing(t *testing.T) {
	s := &searchServer{fail: map[string]int{
		"filter=gguf": 500, "filter=diffusers": 500, "filter=": 500,
		"filter=comfyui": 500, "filter=diffusion-single-file": 500,
		"pipeline_tag=text-to-image": 500, "pipeline_tag=text-to-video": 500, "pipeline_tag=image-to-video": 500,
	}}
	c := exactServer(t, s, map[string]map[string]any{"Comfy-Org/MiniMax-H3": minimaxRow()})
	got, err := c.SearchKind(context.Background(), "Comfy-Org/MiniMax-H3", "", 10, SearchKindAll)
	if err != nil || ids(got) != "Comfy-Org/MiniMax-H3" {
		t.Fatalf("got %s, err %v", ids(got), err)
	}
}

// The untagged source sees every kind of model on the Hub; only loadable
// ones may come through.
func TestSearchUntaggedSourceKeepsOnlyLoadableRepos(t *testing.T) {
	llm := row("meta/Llama-safetensors", 100, 1, "text-generation", "transformers")
	llm["siblings"] = []map[string]any{{"rfilename": "model-00001-of-00002.safetensors"}}
	ggufNoTag := row("someone/model-quants", 50, 1, "")
	ggufNoTag["siblings"] = []map[string]any{{"rfilename": "model-Q4_K_M.gguf"}}
	s := &searchServer{rows: map[string][]map[string]any{
		"filter=": {llm, minimaxRow(), ggufNoTag, row("x/dataset-ish", 1, 1, "")},
	}}
	c := newSearchClient(t, s)
	got, err := c.SearchKind(context.Background(), "model", "downloads", 10, SearchKindAll)
	if err != nil {
		t.Fatal(err)
	}
	if ids(got) != "someone/model-quants,Comfy-Org/MiniMax-H3" {
		t.Fatalf("results = %s", ids(got))
	}
}

func TestSearchPlainWordDoesNotLookUpARepo(t *testing.T) {
	var lookups int
	mux := http.NewServeMux()
	mux.Handle("/api/models", &searchServer{rows: map[string][]map[string]any{}})
	mux.HandleFunc("/api/models/", func(w http.ResponseWriter, r *http.Request) {
		lookups++
		http.Error(w, "nope", 404)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	c := NewClient()
	c.SetBaseURL(srv.URL)
	if _, err := c.SearchKind(context.Background(), "flux gguf", "", 10, SearchKindAll); err != nil {
		t.Fatal(err)
	}
	if lookups != 0 {
		t.Errorf("a plain query triggered %d repo lookups", lookups)
	}
}

// The reported repository: tagged only diffusion-single-file + comfyui (and a
// license), with no pipeline tag and no diffusers tag. It is found by the
// single-file and ComfyUI sources even with no query, and reads as video.
func TestSearchFindsSingleFileComfyRepoByItsTags(t *testing.T) {
	r := row("Comfy-Org/MiniMax-H3", 10, 1, "", "diffusion-single-file", "comfyui",
		"license:minimax-h3-community-license-agreement")
	r["siblings"] = []map[string]any{{"rfilename": "split_files/diffusion_models/minimax_h3_bf16.safetensors"}}
	s := &searchServer{rows: map[string][]map[string]any{"filter=diffusion-single-file": {r}, "filter=comfyui": {r}}}
	c := newSearchClient(t, s)
	for _, q := range []string{"", "minimax"} {
		got, err := c.SearchKind(context.Background(), q, "", 10, SearchKindAll)
		if err != nil {
			t.Fatal(err)
		}
		if ids(got) != "Comfy-Org/MiniMax-H3" {
			t.Fatalf("query %q: results = %s (listed by two sources, shown once)", q, ids(got))
		}
		if got[0].Diffusion != DiffusionVideo {
			t.Errorf("query %q: diffusion = %q, want video", q, got[0].Diffusion)
		}
	}
}
