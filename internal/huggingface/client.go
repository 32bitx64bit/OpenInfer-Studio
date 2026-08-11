// Package huggingface is a client for the Hugging Face Hub REST API.
package huggingface

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
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
	Embedding   string    `json:"embedding,omitempty"`  // "" | "embedding" | "reranker"
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
	CardData map[string]any `json:"cardData"`
}

// Search queries GGUF model repositories.
func (c *Client) Search(ctx context.Context, query, sort string, limit int) ([]SearchResult, error) {
	if limit <= 0 || limit > 100 {
		limit = 30
	}
	q := url.Values{}
	q.Set("search", query)
	q.Set("filter", "gguf")
	q.Set("limit", strconv.Itoa(limit))
	q.Set("full", "true")
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
		author := m.Author
		if author == "" {
			author, _, _ = strings.Cut(m.ID, "/")
		}
		files := make([]string, 0, len(m.Siblings))
		for _, s := range m.Siblings {
			files = append(files, s.RFileName)
		}
		out = append(out, SearchResult{
			ID: m.ID, Author: author, Downloads: m.Downloads, Likes: m.Likes,
			Trending: m.TrendingScore, UpdatedAt: m.LastModified, Tags: m.Tags,
			Private: m.Private, Gated: m.Gated, PipelineTag: m.PipelineTag,
			Modalities: DetectModalities(m.ID, m.PipelineTag, m.Tags, files),
			MTP:        DetectMTP(m.ID, m.Tags, files),
			Embedding:  DetectEmbedding(m.ID, m.PipelineTag, m.Tags, files),
		})
	}
	return out, nil
}

// FileEntry is one file in a repository tree.
type FileEntry struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// RepoInfo is the repository detail payload.
type RepoInfo struct {
	ID          string         `json:"id"`
	Author      string         `json:"author"`
	Downloads   int64          `json:"downloads"`
	Likes       int64          `json:"likes"`
	Tags        []string       `json:"tags"`
	Gated       any            `json:"gated"`
	PipelineTag string         `json:"pipeline_tag,omitempty"`
	Card        string         `json:"card"`
	CardData    map[string]any `json:"card_data"`
	Files       []FileEntry    `json:"files"`
	SHA         string         `json:"sha"`
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
