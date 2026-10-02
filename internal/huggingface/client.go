// Package huggingface is a client for the Hugging Face Hub REST API.
package huggingface

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	defaultBase = "https://huggingface.co"
	maxBody     = 32 << 20
)

// Client is a small authenticated HF API client.
type Client struct {
	http  *http.Client
	base  string
	mu    sync.RWMutex
	token string
}

func NewClient() *Client {
	return &Client{
		http: &http.Client{Timeout: 30 * time.Second},
		base: defaultBase,
	}
}

// SetBaseURL overrides the API root (tests).
func (c *Client) SetBaseURL(u string) { c.base = strings.TrimSuffix(u, "/") }

// SetToken sets/clears the HF access token. It is held in memory and in the
// OS keychain (when available) — never in logs or the database.
func (c *Client) SetToken(t string) {
	c.mu.Lock()
	c.token = t
	c.mu.Unlock()
}

// HasToken reports whether a token is configured.
func (c *Client) HasToken() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.token != ""
}

func (c *Client) do(ctx context.Context, path string, q url.Values, dst any) error {
	u := c.base + path
	if len(q) > 0 {
		u += "?" + q.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	c.mu.RLock()
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	c.mu.RUnlock()
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "openinfer-studio/0.1")

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("huggingface request failed: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBody))
	if err != nil {
		return err
	}
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return &APIError{Status: resp.StatusCode,
			Message: "authentication required or repository gated; grant access on Hugging Face and configure a token"}
	}
	if resp.StatusCode == http.StatusNotFound {
		return &APIError{Status: resp.StatusCode, Message: "repository not found"}
	}
	if resp.StatusCode != http.StatusOK {
		return &APIError{Status: resp.StatusCode, Message: string(body[:min(len(body), 512)])}
	}
	if dst == nil {
		return nil
	}
	return json.Unmarshal(body, dst)
}

// APIError preserves the upstream status and message so the UI can show the
// real failure (gated repo, bad token, rate limit) instead of a generic one.
type APIError struct {
	Status  int
	Message string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("huggingface: HTTP %d: %s", e.Status, e.Message)
}

func (e *APIError) HTTPStatus() int {
	switch e.Status {
	case http.StatusNotFound:
		return http.StatusNotFound
	case http.StatusUnauthorized, http.StatusForbidden:
		return http.StatusForbidden
	default:
		return http.StatusBadGateway
	}
}

// SearchResult is one row of the Discover page.
type SearchResult struct {
	ID          string    `json:"id"`
	Author      string    `json:"author"`
	Downloads   int64     `json:"downloads"`
	Likes       int64     `json:"likes"`
	Trending    float64   `json:"trending_score"`
	UpdatedAt   time.Time `json:"last_modified"`
	Tags        []string  `json:"tags"`
	Private     bool      `json:"private"`
	Gated       any       `json:"gated"` // false | "auto" | "manual"
	PipelineTag string    `json:"pipeline_tag,omitempty"`
	Modalities  []string  `json:"modalities,omitempty"` // audio | vision
	MTP         string    `json:"mtp,omitempty"`        // "" | "mtp" | "mtp-draft"
	Draft       string    `json:"draft,omitempty"`      // "" | dflash | eagle3 | dspark | mtp-draft | draft
	Embedding   string    `json:"embedding,omitempty"`  // "" | "embedding" | "reranker"
	Diffusion   string    `json:"diffusion,omitempty"`  // "" | "image" | "video" | "both"
}

// hfModel mirrors the /api/models payload fields we use.
type hfModel struct {
	ID            string    `json:"id"`
	Author        string    `json:"author"`
	Downloads     int64     `json:"downloads"`
	Likes         int64     `json:"likes"`
	TrendingScore float64   `json:"trendingScore"`
	LastModified  time.Time `json:"lastModified"`
	Tags          []string  `json:"tags"`
	Private       bool      `json:"private"`
	Gated         any       `json:"gated"`
	PipelineTag   string    `json:"pipeline_tag"`
	Siblings      []struct {
		RFileName string `json:"rfilename"`
	} `json:"siblings"`
	CardData    map[string]any `json:"cardData"`
	SHA         string         `json:"sha"`
	Safetensors *hfSafetensors `json:"safetensors"`
}

type hfSafetensors struct {
	Parameters map[string]int64 `json:"parameters"`
	Total      int64            `json:"total"`
}

// Search kinds accepted by SearchKind.
const (
	// SearchKindLLM ("" or "llm") queries GGUF repositories only (the legacy
	// default for API callers).
	SearchKindLLM = "llm"
	// SearchKindDiffusion queries image/video generators only.
	SearchKindDiffusion = "diffusion"
	// SearchKindAll is the Discover page's single corpus: GGUF repositories
	// plus image/video generators in one ranked list.
	SearchKindAll = "all"
)

// searchSpec is one Hub /api/models query that contributes to a search.
type searchSpec struct {
	filter      string // library/format tag ("gguf", "diffusers")
	pipelineTag string // task tag; the Hub accepts only one per request
	// untagged runs the query with no tag filter and keeps only rows that
	// look loadable (GGUF files or an image/video generator). Many ComfyUI
	// repackages carry no gguf, diffusers or task tag, so no tagged source
	// can find them by name.
	untagged bool
}

// searchSpecs returns the Hub queries a kind expands to. The unified search
// asks for every source that can hold a loadable model: GGUF repositories
// (chat models and GGUF-quantized diffusion transformers), diffusers
// bundles, single-file / ComfyUI safetensors repositories, repositories
// tagged with an image/video generation task, and, when there is a query to
// match names against, untagged repositories.
func searchSpecs(kind, query string) []searchSpec {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case SearchKindDiffusion:
		return []searchSpec{{filter: "diffusers"}, {filter: "diffusion-single-file"}, {filter: "comfyui"}}
	case SearchKindAll:
		specs := []searchSpec{
			{filter: "gguf"},
			{filter: "diffusers"},
			// Single-file checkpoints and ComfyUI repackages (safetensors) are
			// tagged with these libraries, not with diffusers.
			{filter: "diffusion-single-file"},
			{filter: "comfyui"},
			{pipelineTag: "text-to-image"},
			{pipelineTag: "text-to-video"},
			{pipelineTag: "image-to-video"},
		}
		if strings.TrimSpace(query) != "" {
			specs = append(specs, searchSpec{untagged: true})
		}
		return specs
	default:
		return []searchSpec{{filter: "gguf"}}
	}
}

var repoRefRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)

// huggingFacePages are first path segments of huggingface.co that are not an
// owner.
var huggingFacePages = map[string]bool{
	"models": true, "datasets": true, "spaces": true, "docs": true, "blog": true,
	"papers": true, "collections": true, "organizations": true, "settings": true,
	"api": true, "join": true, "login": true, "pricing": true, "tasks": true,
}

// RepoRef returns "owner/name" when q is a pasted Hugging Face repository URL
// (https://huggingface.co/owner/name, …/tree/main, hf.co/owner/name, …) or a
// bare owner/name id, else "".
func RepoRef(q string) string {
	q = strings.TrimSpace(q)
	if q == "" || strings.ContainsAny(q, " \t\n") {
		return ""
	}
	scheme := false
	for _, prefix := range []string{"https://", "http://"} {
		if rest, ok := strings.CutPrefix(strings.ToLower(q), prefix); ok {
			q = q[len(q)-len(rest):]
			scheme = true
			break
		}
	}
	host := false
	for _, h := range []string{"www.huggingface.co/", "huggingface.co/", "hf.co/"} {
		if len(q) > len(h) && strings.EqualFold(q[:len(h)], h) {
			q = q[len(h):]
			host = true
			break
		}
	}
	if scheme && !host {
		return "" // a URL, but not a Hugging Face one
	}
	if i := strings.IndexAny(q, "?#"); i >= 0 {
		q = q[:i]
	}
	if !host {
		// A bare id is exactly owner/name: no leading or trailing slash.
		if repoRefRe.MatchString(q) {
			return q
		}
		return ""
	}
	q = strings.Trim(q, "/")
	parts := strings.Split(q, "/")
	if len(parts) < 2 || huggingFacePages[strings.ToLower(parts[0])] {
		return ""
	}
	ref := parts[0] + "/" + parts[1]
	if !repoRefRe.MatchString(ref) {
		return ""
	}
	return ref
}

// Search queries model repositories (GGUF text models; see SearchKind).
func (c *Client) Search(ctx context.Context, query, sort string, limit int) ([]SearchResult, error) {
	return c.SearchKind(ctx, query, sort, limit, "")
}

// SearchKind is Search with an explicit corpus selector. "" | "llm" →
// GGUF text models, "diffusion" → diffusers image/video generators, "all" →
// both, merged into one list.
func (c *Client) SearchKind(ctx context.Context, query, sort string, limit int, kind string) ([]SearchResult, error) {
	if limit <= 0 || limit > 100 {
		limit = 30
	}

	// A pasted URL or owner/name id means one specific repository: look it
	// up directly (tags or no tags) and search for its id besides.
	var ref string
	if strings.EqualFold(strings.TrimSpace(kind), SearchKindAll) {
		if ref = RepoRef(query); ref != "" {
			query = ref
		}
	}

	specs := searchSpecs(kind, query)
	if len(specs) == 1 {
		return c.searchOnce(ctx, query, sort, limit, kind, specs[0])
	}

	type outcome struct {
		rows []SearchResult
		err  error
	}
	outs := make([]outcome, len(specs))
	var exact *SearchResult
	var wg sync.WaitGroup
	for i, spec := range specs {
		wg.Add(1)
		go func(i int, spec searchSpec) {
			defer wg.Done()
			outs[i].rows, outs[i].err = c.searchOnce(ctx, query, sort, limit, kind, spec)
		}(i, spec)
	}
	if ref != "" {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var m hfModel
			if err := c.do(ctx, "/api/models/"+ref, url.Values{"full": {"true"}}, &m); err == nil && m.ID != "" {
				r, _ := resultOf(m)
				exact = &r
			}
		}()
	}
	wg.Wait()

	// A source that fails (rate limit, one bad filter) must not blank the
	// others; the search only fails when every source does.
	lists := make([][]SearchResult, 0, len(outs))
	var firstErr error
	for _, o := range outs {
		if o.err != nil {
			if firstErr == nil {
				firstErr = o.err
			}
			continue
		}
		lists = append(lists, o.rows)
	}
	if len(lists) == 0 && exact == nil {
		return nil, firstErr
	}
	merged := mergeSearchResults(lists, sort, limit)
	if exact != nil {
		// The repository that was asked for by name goes first.
		out := []SearchResult{*exact}
		for _, r := range merged {
			if !strings.EqualFold(r.ID, exact.ID) {
				out = append(out, r)
			}
		}
		if len(out) > limit {
			out = out[:limit]
		}
		merged = out
	}
	return merged, nil
}

// searchOnce runs one Hub query and converts its rows.
func (c *Client) searchOnce(ctx context.Context, query, sort string, limit int, kind string, spec searchSpec) ([]SearchResult, error) {
	q := url.Values{}
	q.Set("search", query)
	q.Set("limit", strconv.Itoa(limit))
	q.Set("full", "true")
	if spec.filter != "" {
		q.Set("filter", spec.filter)
	}
	if spec.pipelineTag != "" {
		q.Set("pipeline_tag", spec.pipelineTag)
	}
	if strings.EqualFold(kind, SearchKindDiffusion) && strings.TrimSpace(query) == "" {
		// Diffusers-only browsing with no query: seed with the known
		// families so the result is not dominated by adapters.
		q.Set("search", "stable diffusion flux sdxl wan ltx")
	}
	switch sort {
	case "downloads", "likes", "lastModified", "trending":
		q.Set("sort", sort)
		if sort != "trending" {
			q.Set("direction", "-1")
		}
	default:
		// HF default relevance ordering
	}
	var rows []hfModel
	if err := c.do(ctx, "/api/models", q, &rows); err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(rows))
	for _, m := range rows {
		r, files := resultOf(m)
		if spec.untagged && !looksLoadable(m.Tags, files, r.Diffusion) {
			continue
		}
		out = append(out, r)
	}
	return out, nil
}

// resultOf converts a Hub model row into a Discover row, with its file names.
func resultOf(m hfModel) (SearchResult, []string) {
	author := m.Author
	if author == "" {
		author, _, _ = strings.Cut(m.ID, "/")
	}
	files := make([]string, 0, len(m.Siblings))
	for _, s := range m.Siblings {
		files = append(files, s.RFileName)
	}
	return SearchResult{
		ID: m.ID, Author: author, Downloads: m.Downloads, Likes: m.Likes,
		Trending: m.TrendingScore, UpdatedAt: m.LastModified, Tags: m.Tags,
		Private: m.Private, Gated: m.Gated, PipelineTag: m.PipelineTag,
		Modalities: DetectModalities(m.ID, m.PipelineTag, m.Tags, files),
		MTP:        DetectMTP(m.ID, m.Tags, files),
		Draft:      DetectDraftSidecar(m.ID, m.Tags, files),
		Embedding:  DetectEmbedding(m.ID, m.PipelineTag, m.Tags, files),
		Diffusion:  DetectDiffusion(m.ID, m.PipelineTag, m.Tags, files),
	}, files
}

// looksLoadable reports whether an untagged search hit is something this app
// can download for use: a GGUF repository or an image/video generator.
func looksLoadable(tags, files []string, diffusion string) bool {
	if diffusion != DiffusionNone {
		return true
	}
	for _, t := range tags {
		if strings.EqualFold(t, "gguf") {
			return true
		}
	}
	for _, f := range files {
		if strings.HasSuffix(strings.ToLower(f), ".gguf") {
			return true
		}
	}
	return false
}

// mergeSearchResults folds per-source result lists into one list of at most
// limit rows with each repository once. A sorted search (downloads, likes,
// trending, lastModified) orders by that key across sources; relevance has
// no comparable score, so sources are interleaved round-robin and each
// keeps its own ranking.
func mergeSearchResults(lists [][]SearchResult, sort string, limit int) []SearchResult {
	seen := map[string]bool{}
	var merged []SearchResult
	add := func(r SearchResult) {
		key := strings.ToLower(r.ID)
		if r.ID == "" || seen[key] {
			return
		}
		seen[key] = true
		merged = append(merged, r)
	}

	less := searchLess(sort)
	if less == nil {
		for rank := 0; ; rank++ {
			more := false
			for _, l := range lists {
				if rank < len(l) {
					more = true
					add(l[rank])
				}
			}
			if !more {
				break
			}
		}
	} else {
		for _, l := range lists {
			for _, r := range l {
				add(r)
			}
		}
		slices.SortStableFunc(merged, func(a, b SearchResult) int {
			switch {
			case less(a, b):
				return -1
			case less(b, a):
				return 1
			}
			return 0
		})
	}
	if len(merged) > limit {
		merged = merged[:limit]
	}
	if merged == nil {
		merged = []SearchResult{}
	}
	return merged
}

// searchLess orders results for a Hub sort key, best first; nil means the
// sort has no cross-source key (relevance).
func searchLess(sort string) func(a, b SearchResult) bool {
	switch sort {
	case "downloads":
		return func(a, b SearchResult) bool { return a.Downloads > b.Downloads }
	case "likes":
		return func(a, b SearchResult) bool { return a.Likes > b.Likes }
	case "trending":
		return func(a, b SearchResult) bool { return a.Trending > b.Trending }
	case "lastModified":
		return func(a, b SearchResult) bool { return a.UpdatedAt.After(b.UpdatedAt) }
	}
	return nil
}

// FileEntry is one file in a repository tree.
type FileEntry struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// RepoInfo is the repository detail payload.
type RepoInfo struct {
	ID                    string           `json:"id"`
	Author                string           `json:"author"`
	Downloads             int64            `json:"downloads"`
	Likes                 int64            `json:"likes"`
	Tags                  []string         `json:"tags"`
	Gated                 any              `json:"gated"`
	PipelineTag           string           `json:"pipeline_tag,omitempty"`
	Card                  string           `json:"card"`
	CardData              map[string]any   `json:"card_data"`
	Files                 []FileEntry      `json:"files"`
	SHA                   string           `json:"sha"`
	SafetensorsParameters map[string]int64 `json:"safetensors_parameters,omitempty"`
	SafetensorsTotal      int64            `json:"safetensors_total,omitempty"`
}

// Repo fetches repository metadata, the recursive file tree and the model
// card markdown.
func (c *Client) Repo(ctx context.Context, repo string) (*RepoInfo, error) {
	var m hfModel
	if err := c.do(ctx, "/api/models/"+repo, url.Values{"full": {"true"}}, &m); err != nil {
		return nil, err
	}

	var tree []struct {
		Path string `json:"path"`
		Size int64  `json:"size"`
		Type string `json:"type"`
	}
	err := c.do(ctx, "/api/models/"+repo+"/tree/main", url.Values{"recursive": {"true"}}, &tree)
	if err != nil {
		// Fall back to siblings when tree is unavailable.
		for _, s := range m.Siblings {
			tree = append(tree, struct {
				Path string `json:"path"`
				Size int64  `json:"size"`
				Type string `json:"type"`
			}{Path: s.RFileName, Type: "file"})
		}
	}

	info := &RepoInfo{
		ID: m.ID, Author: m.Author, Downloads: m.Downloads, Likes: m.Likes,
		Tags: m.Tags, Gated: m.Gated, CardData: m.CardData, PipelineTag: m.PipelineTag,
		SHA: m.SHA,
	}
	if m.Safetensors != nil {
		info.SafetensorsParameters = m.Safetensors.Parameters
		info.SafetensorsTotal = m.Safetensors.Total
	}
	for _, f := range tree {
		if f.Type == "directory" {
			continue
		}
		info.Files = append(info.Files, FileEntry{Path: f.Path, Size: f.Size})
	}

	// Model card markdown (best-effort).
	var card []byte
	u := c.base + "/" + repo + "/raw/main/README.md"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err == nil {
		c.mu.RLock()
		if c.token != "" {
			req.Header.Set("Authorization", "Bearer "+c.token)
		}
		c.mu.RUnlock()
		if resp, err := c.http.Do(req); err == nil && resp.StatusCode == http.StatusOK {
			card, _ = io.ReadAll(io.LimitReader(resp.Body, 1<<20))
			resp.Body.Close()
			info.Card = string(card)
		} else if err == nil {
			resp.Body.Close()
		}
	}
	return info, nil
}

// DownloadURL builds the resolve URL for a repository file.
func (c *Client) DownloadURL(repo, path string) string {
	// ?download=true nudges the Hub toward a direct CDN response suitable
	// for multi-connection Range downloads.
	return fmt.Sprintf("%s/%s/resolve/main/%s?download=true", c.base, repo, path)
}

// FetchFile downloads a small repository file (config.json, tokenizer, …).
func (c *Client) FetchFile(ctx context.Context, repo, path string, max int64) ([]byte, error) {
	if max <= 0 || max > maxBody {
		max = maxBody
	}
	u := fmt.Sprintf("%s/%s/resolve/main/%s", c.base, repo, path)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	c.mu.RLock()
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	c.mu.RUnlock()
	req.Header.Set("User-Agent", "openinfer-studio/0.1")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("huggingface fetch %s: %w", path, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, max+1))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return nil, &APIError{Status: resp.StatusCode,
			Message: "authentication required or repository gated; grant access on Hugging Face and configure a token"}
	}
	if resp.StatusCode == http.StatusNotFound {
		return nil, &APIError{Status: resp.StatusCode, Message: path + " not found"}
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &APIError{Status: resp.StatusCode, Message: string(body[:min(len(body), 512)])}
	}
	if int64(len(body)) > max {
		return nil, fmt.Errorf("huggingface: %s exceeds %d bytes", path, max)
	}
	return body, nil
}
