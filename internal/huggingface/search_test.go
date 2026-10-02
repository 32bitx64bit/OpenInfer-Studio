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
	want := []string{
		"filter=diffusers", "filter=gguf",
		"pipeline_tag=image-to-video", "pipeline_tag=text-to-image", "pipeline_tag=text-to-video",
	}
	if got := s.seen(); strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("sources queried = %v, want %v", got, want)
	}
}

func TestSearchKindLegacyKindsStaySingleSource(t *testing.T) {
	for kind, want := range map[string]string{
		"": "filter=gguf", "llm": "filter=gguf", "diffusion": "filter=diffusers",
	} {
		s := &searchServer{rows: map[string][]map[string]any{}}
		c := newSearchClient(t, s)
		if _, err := c.SearchKind(context.Background(), "q", "", 10, kind); err != nil {
			t.Fatal(err)
		}
		if got := s.seen(); len(got) != 1 || got[0] != want {
			t.Errorf("kind %q queried %v, want [%s]", kind, got, want)
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
