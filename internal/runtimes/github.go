package runtimes

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Release is one upstream release (llama.cpp or stable-diffusion.cpp).
type Release struct {
	Tag         string    `json:"tag"` // e.g. b5678
	Name        string    `json:"name"`
	PublishedAt time.Time `json:"published_at"`
	Prerelease  bool      `json:"prerelease"`
	Assets      []Asset   `json:"assets"`
	Kind        string    `json:"kind,omitempty"` // llm|diffusion (feed source)
}

// Asset is one downloadable build artifact.
type Asset struct {
	Name        string `json:"name"`
	URL         string `json:"url"`
	Size        int64  `json:"size"`
	DownloadURL string `json:"download_url"` // browser_download_url
}

type ghRelease struct {
	TagName     string    `json:"tag_name"`
	Name        string    `json:"name"`
	PublishedAt time.Time `json:"published_at"`
	Prerelease  bool      `json:"prerelease"`
	Assets      []struct {
		Name               string `json:"name"`
		URL                string `json:"url"`
		BrowserDownloadURL string `json:"browser_download_url"`
		Size               int64  `json:"size"`
	} `json:"assets"`
}

// ReleaseFeed fetches official ggml-org/llama.cpp releases via the GitHub
// API. BaseURL is overridable for tests.
type ReleaseFeed struct {
	BaseURL string
	http    *http.Client
	// DiffusionBaseURL optionally points at the stable-diffusion.cpp feed so
	// one feed object can serve both runtimes pages. Empty = llama.cpp only.
	DiffusionBaseURL string
	DiffusionRepo    string // default leejet/stable-diffusion.cpp
}

func NewReleaseFeed() *ReleaseFeed {
	return &ReleaseFeed{BaseURL: "https://api.github.com", http: &http.Client{Timeout: 30 * time.Second}}
}

// Latest returns the newest releases (default page of 20).
func (f *ReleaseFeed) Latest(ctx context.Context) ([]Release, error) {
	return f.latestKind(ctx, f.BaseURL, "ggml-org", "llama.cpp", "llm")
}

// LatestDiffusion returns the newest stable-diffusion.cpp releases.
func (f *ReleaseFeed) LatestDiffusion(ctx context.Context) ([]Release, error) {
	base := f.DiffusionBaseURL
	if base == "" {
		base = f.BaseURL
	}
	repo := f.DiffusionRepo
	owner, name, ok := splitRepo(repo)
	if !ok {
		owner, name = "leejet", "stable-diffusion.cpp"
	}
	return f.latestKind(ctx, base, owner, name, "diffusion")
}

func splitRepo(repo string) (string, string, bool) {
	parts := strings.Split(strings.TrimSpace(repo), "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", "", false
	}
	return parts[0], parts[1], true
}

func (f *ReleaseFeed) latestKind(ctx context.Context, base, owner, repo, kind string) ([]Release, error) {
	u, err := url.JoinPath(base, "repos", owner, repo, "releases")
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u+"?per_page=20", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", "openinfer-studio/0.1")
	resp, err := f.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("github releases request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("github releases: HTTP %d: %s", resp.StatusCode, body)
	}
	var raw []ghRelease
	if err := json.NewDecoder(io.LimitReader(resp.Body, 8<<20)).Decode(&raw); err != nil {
		return nil, fmt.Errorf("decoding releases: %w", err)
	}
	out := make([]Release, 0, len(raw))
	for _, r := range raw {
		rel := Release{Tag: r.TagName, Name: r.Name, PublishedAt: r.PublishedAt, Prerelease: r.Prerelease, Kind: kind}
		for _, a := range r.Assets {
			rel.Assets = append(rel.Assets, Asset{
				Name: a.Name, URL: a.URL, Size: a.Size, DownloadURL: a.BrowserDownloadURL,
			})
		}
		out = append(out, rel)
	}
	return out, nil
}
