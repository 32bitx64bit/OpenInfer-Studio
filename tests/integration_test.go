// Integration tests: run the real backend services against a fake
// llama-server binary (tests/fakeserver) and httptest-based Hugging Face and
// GitHub API fakes. No network and no real model downloads are involved.
package tests

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/openinfer/openinfer-studio/internal/api"
	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/internal/downloads"
	"github.com/openinfer/openinfer-studio/internal/huggingface"
	"github.com/openinfer/openinfer-studio/internal/instances"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/proxy"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
	"github.com/openinfer/openinfer-studio/migrations"
)

// buildFakeServer compiles tests/fakeserver into a temp binary.
func buildFakeServer(t *testing.T) string {
	t.Helper()
	exe := filepath.Join(t.TempDir(), "llama-server")
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	cmd := exec.Command("go", "build", "-o", exe, "./fakeserver")
	cmd.Dir = "."
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("building fake server: %v\n%s", err, out)
	}
	return exe
}

// writeFakeGGUF creates a minimal valid GGUF file.
func writeFakeGGUF(t *testing.T, dir, name string) string {
	t.Helper()
	var b bytes.Buffer
	w32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	w64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	wstr := func(s string) { w64(uint64(len(s))); b.WriteString(s) }
	binary.Write(&b, binary.LittleEndian, uint32(0x46554747)) // GGUF
	w32(3)
	w64(0)
	w64(4)
	wstr("general.name")
	w32(8)
	wstr("Integration Test Model")
	wstr("general.architecture")
	w32(8)
	wstr("llama")
	wstr("general.file_type")
	w32(4)
	w32(15)
	wstr("llama.context_length")
	w32(4)
	w32(8192)
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

type testEnv struct {
	dir string
	db  *database.DB
	rt  *runtimes.Manager
	lib *models.Library
	im  *instances.Manager
	dl  *downloads.Manager
	hub *api.Hub
}

func newTestEnv(t *testing.T, fakeExe string) *testEnv {
	t.Helper()
	dir := t.TempDir()
	for _, d := range []string{"database", "runtimes", "models", "partial", "logs", "temp", "cache"} {
		os.MkdirAll(filepath.Join(dir, d), 0o755)
	}
	db, err := database.Open(filepath.Join(dir, "database"), migrations.FS)
	if err != nil {
		t.Fatal(err)
	}

	hub := api.NewHub()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	dl := downloads.NewManager(db.DB, filepath.Join(dir, "partial"), hub, log,
		func(string) uint64 { return 1 << 40 })
	rt := runtimes.NewManager(db.DB, filepath.Join(dir, "runtimes"), dl, hub, log)
	lib := models.NewLibrary(db.DB, filepath.Join(dir, "models"), hub, log)
	im := instances.NewManager(db.DB, rt, lib, hub, log,
		filepath.Join(dir, "logs"), filepath.Join(dir, "temp"), filepath.Join(dir, "cache"))
	t.Cleanup(func() {
		im.StopAll()
		deadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(deadline) {
			if len(im.List()) == 0 {
				break
			}
			time.Sleep(50 * time.Millisecond)
		}
		// Windows keeps an exclusive lock on a running .exe until the
		// process handle is fully released; give the OS a moment so
		// t.TempDir cleanup can delete the fake llama-server binary.
		if runtime.GOOS == "windows" {
			time.Sleep(200 * time.Millisecond)
		}
		db.Close()
	})

	// Register the fake runtime through the real import path (probes the exe).
	rtID, err := rt.ImportCustom(fakeExe)
	if err != nil {
		t.Fatalf("importing fake runtime: %v", err)
	}
	if err := rt.SetPreferred(rtID); err != nil {
		t.Fatal(err)
	}
	return &testEnv{dir: dir, db: db, rt: rt, lib: lib, im: im, dl: dl, hub: hub}
}

func waitState(t *testing.T, im *instances.Manager, modelID string, want ...string) instances.Instance {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if inst, ok := im.Get(modelID); ok {
			for _, w := range want {
				if inst.State == w {
					return *inst
				}
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	inst, _ := im.Get(modelID)
	logTail, _ := im.Logs(modelID, 8192)
	t.Fatalf("model never reached %v (state=%v)\nlog:\n%s", want, inst, logTail)
	return instances.Instance{}
}

func TestLoadChatUnloadCycle(t *testing.T) {
	env := newTestEnv(t, buildFakeServer(t))
	writeFakeGGUF(t, filepath.Join(env.dir, "models"), "testmodel.gguf")
	if n, err := env.lib.Scan(); err != nil || n != 1 {
		t.Fatalf("scan: n=%d err=%v", n, err)
	}
	all, _ := env.lib.List()
	if len(all) != 1 {
		t.Fatalf("library size = %d", len(all))
	}
	m := all[0]
	if m.Architecture != "llama" || m.Quantization != "Q4_K_M" || m.ContextLength != 8192 {
		t.Errorf("metadata wrong: %+v", m)
	}

	// Command preview must contain model path and host.
	preview, err := env.im.Preview(m.ID, instances.DefaultSettings())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(preview.Command, "--model") || !strings.Contains(preview.Command, "testmodel.gguf") {
		t.Errorf("preview missing model: %q", preview.Command)
	}

	// Load.
	if _, err := env.im.Start(m.ID, instances.DefaultSettings()); err != nil {
		t.Fatalf("start: %v", err)
	}
	inst := waitState(t, env.im, m.ID, instances.StateReady)
	if inst.PID <= 0 || inst.Port <= 0 {
		t.Errorf("instance missing pid/port: %+v", inst)
	}

	// Chat with the fake server through the instance endpoint.
	ep, err := env.im.EndpointFor(m.ID)
	if err != nil {
		t.Fatal(err)
	}
	body := `{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req, _ := http.NewRequest("POST", ep.URL+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+ep.APIKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	streamed, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(streamed), "Hello") {
		t.Errorf("streamed response missing content: %q", streamed)
	}

	// Per-process API key must be enforced.
	req2, _ := http.NewRequest("POST", ep.URL+"/v1/chat/completions", strings.NewReader(body))
	resp2, _ := http.DefaultClient.Do(req2)
	if resp2.StatusCode != http.StatusUnauthorized {
		t.Errorf("instance accepted request without API key: %d", resp2.StatusCode)
	}
	resp2.Body.Close()

	// Unload: process tree must be gone.
	if err := env.im.Stop(m.ID, false); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if _, ok := env.im.Get(m.ID); !ok {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if _, ok := env.im.Get(m.ID); ok {
		t.Error("instance still present after unload")
	}
	if _, err := env.im.EndpointFor(m.ID); err == nil {
		t.Error("endpoint still reachable after unload")
	}
}

func TestCrashClassification(t *testing.T) {
	env := newTestEnv(t, buildFakeServer(t))
	writeFakeGGUF(t, filepath.Join(env.dir, "models"), "crashme.gguf")
	if _, err := env.lib.Scan(); err != nil {
		t.Fatal(err)
	}
	all, _ := env.lib.List()
	m := all[0]
	if _, err := env.im.Start(m.ID, instances.DefaultSettings()); err != nil {
		t.Fatal(err)
	}
	inst := waitState(t, env.im, m.ID, instances.StateCrashed, instances.StateFailed)
	if inst.FailureClass == "" {
		t.Errorf("crash not classified: %+v", inst)
	}
	// Simulated stderr mentions out of memory → VRAM/RAM classification.
	if inst.FailureClass != "insufficient-vram" && inst.FailureClass != "insufficient-ram" {
		t.Errorf("unexpected class %q", inst.FailureClass)
	}
}

func TestProxyEndToEnd(t *testing.T) {
	env := newTestEnv(t, buildFakeServer(t))
	writeFakeGGUF(t, filepath.Join(env.dir, "models"), "proxymodel.gguf")
	env.lib.Scan()
	all, _ := env.lib.List()
	m := all[0]

	s := instances.DefaultSettings()
	s.Alias = "test-alias"
	if _, err := env.im.Start(m.ID, s); err != nil {
		t.Fatal(err)
	}
	waitState(t, env.im, m.ID, instances.StateReady)

	// Wire the proxy through the same adapters the app uses.
	px := proxy.NewServer(&proxyAdapter{env.im}, env.db.DB,
		slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err := px.LoadProfile(); err != nil {
		t.Fatal(err)
	}
	cfg := px.Config()
	cfg.Port = pickPort(t)
	if err := px.UpdateConfig(cfg); err != nil {
		t.Fatal(err)
	}
	if err := px.Start(); err != nil {
		t.Fatal(err)
	}
	defer px.Stop()

	base := fmt.Sprintf("http://127.0.0.1:%d", cfg.Port)

	// Wrong key → 401.
	resp, _ := http.Get(base + "/v1/models")
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("proxy without key: %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Correct key → alias listed.
	req, _ := http.NewRequest("GET", base+"/v1/models", nil)
	req.Header.Set("Authorization", "Bearer "+px.Config().APIKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	modelsBody, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(modelsBody), "test-alias") {
		t.Errorf("alias missing from /v1/models: %s", modelsBody)
	}

	// Chat completions stream through the proxy.
	cbody := `{"model":"test-alias","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req, _ = http.NewRequest("POST", base+"/v1/chat/completions", strings.NewReader(cbody))
	req.Header.Set("Authorization", "Bearer "+px.Config().APIKey)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	chatBody, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(chatBody), "Hello") {
		t.Errorf("proxied stream missing content: %q", chatBody)
	}

	// Request log recorded.
	if len(px.RecentRequests()) == 0 {
		t.Error("proxy request log empty")
	}
}

// proxyAdapter mirrors apps/core/adapters.go (kept separate to avoid
// importing the main package from tests).
type proxyAdapter struct{ im *instances.Manager }

func (a *proxyAdapter) EndpointFor(modelID string) (proxy.Endpoint, error) {
	ep, err := a.im.EndpointFor(modelID)
	return proxy.Endpoint{URL: ep.URL, APIKey: ep.APIKey, Alias: ep.Alias}, err
}
func (a *proxyAdapter) Touch(modelID string) { a.im.Touch(modelID) }
func (a *proxyAdapter) ResolveModelID(name string) (string, error) {
	return a.im.ResolveModelID(name)
}
func (a *proxyAdapter) LoadedModels() []string { return a.im.LoadedModels() }

func pickPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}

// TestHuggingFaceAgainstFakeAPI runs the real HF client against a fake API.
func TestHuggingFaceAgainstFakeAPI(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode([]map[string]any{
			{"id": "test/model", "author": "test", "downloads": 42, "likes": 7},
		})
	})
	mux.HandleFunc("/api/models/test/model", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"id": "test/model", "author": "test", "downloads": 42, "likes": 7,
		})
	})
	mux.HandleFunc("/api/models/test/model/tree/main", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode([]map[string]any{
			{"path": "model-Q4_K_M.gguf", "size": 4000000000, "type": "file"},
			{"path": "model-Q8_0.gguf", "size": 8000000000, "type": "file"},
		})
	})
	mux.HandleFunc("/test/model/raw/main/README.md", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "# Test Model\nA fake card.")
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := huggingface.NewClient()
	c.SetBaseURL(srv.URL)

	results, err := c.Search(context.Background(), "test", "downloads", 10)
	if err != nil || len(results) != 1 {
		t.Fatalf("search: %v %+v", err, results)
	}
	repo, err := c.Repo(context.Background(), "test/model")
	if err != nil {
		t.Fatal(err)
	}
	if len(repo.Files) != 2 {
		t.Errorf("files = %+v", repo.Files)
	}
	if !strings.Contains(repo.Card, "Test Model") {
		t.Errorf("card missing: %q", repo.Card)
	}
	groups, _, _ := huggingface.GroupFiles(repo.Files)
	if len(groups) != 2 {
		t.Errorf("groups = %+v", groups)
	}
}

// TestGitHubFeedAgainstFakeAPI validates release parsing + asset resolution.
func TestGitHubFeedAgainstFakeAPI(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "ggml-org/llama.cpp/releases") {
			w.WriteHeader(404)
			return
		}
		json.NewEncoder(w).Encode([]map[string]any{
			{
				"tag_name": "b9999", "name": "build 9999",
				"published_at": "2026-07-01T00:00:00Z",
				"assets": []map[string]any{
					{"name": "llama-b9999-bin-ubuntu-x64.zip", "browser_download_url": "https://example/ub.zip", "size": 1000},
					{"name": "llama-b9999-bin-ubuntu-vulkan-x64.zip", "browser_download_url": "https://example/vk.zip", "size": 2000},
					{"name": "llama-b9999-bin-win-x64.zip", "browser_download_url": "https://example/win.zip", "size": 3000},
				},
			},
		})
	}))
	defer srv.Close()

	feed := runtimes.NewReleaseFeed()
	feed.BaseURL = srv.URL
	rels, err := feed.Latest(context.Background())
	if err != nil || len(rels) != 1 {
		t.Fatalf("feed: %v %+v", err, rels)
	}
	matches := runtimes.ResolveAssets(rels[0], runtimes.MachineProfile{
		OS: "linux", Arch: "amd64", Vulkan: true, GPUVendor: "amd",
	}, "")
	if len(matches) != 2 { // windows asset excluded
		t.Fatalf("matches = %+v", matches)
	}
	if matches[0].Backend != "vulkan" {
		t.Errorf("want vulkan first for AMD/Vulkan machine, got %s", matches[0].Backend)
	}
}
