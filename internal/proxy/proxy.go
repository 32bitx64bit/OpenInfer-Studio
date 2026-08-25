// Package proxy is the user-facing OpenAI-compatible HTTP API. It is separate
// from the desktop control API (own port, bind, key, CORS) and routes to
// managed llama-server instances without exposing admin endpoints.
package proxy

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/reasoning"
)

// EndpointProvider resolves model names (ID or alias) to ready instances.
type EndpointProvider interface {
	EndpointFor(modelID string) (Endpoint, error)
	Touch(modelID string)
	ResolveModelID(name string) (string, error)
}

// Endpoint mirrors instances.Endpoint.
type Endpoint struct {
	URL    string
	APIKey string
	Alias  string
}

// Config for the public server.
type Config struct {
	Port      int    `json:"port"`
	Bind      string `json:"bind"`      // 127.0.0.1 (default) or 0.0.0.0 with explicit LAN opt-in
	AllowLAN  bool   `json:"allow_lan"` // requires Bind 0.0.0.0 and shows a warning in UI
	CORS      string `json:"cors"`      // "" = disabled, or comma-separated origins
	APIKey    string `json:"api_key"`
	Autostart bool   `json:"autostart"`
}

// RequestLog records recent public API traffic for the Developer page.
type RequestLog struct {
	Time       time.Time `json:"time"`
	Method     string    `json:"method"`
	Path       string    `json:"path"`
	Model      string    `json:"model"`
	Status     int       `json:"status"`
	DurationMS int64     `json:"duration_ms"`
	Client     string    `json:"client"`
}

type Server struct {
	eps    EndpointProvider
	db     *sql.DB
	log    *slog.Logger
	cfg    Config
	http   *http.Server
	client *http.Client

	mu      sync.Mutex
	running bool
	reqs    []RequestLog
	clients map[string]time.Time
}

func NewServer(eps EndpointProvider, db *sql.DB, log *slog.Logger) *Server {
	return &Server{
		eps: eps, db: db, log: log,
		client:  &http.Client{Timeout: 0},
		cfg:     Config{Port: 1235, Bind: "127.0.0.1"},
		clients: map[string]time.Time{},
	}
}

// GenerateAPIKey creates a new random public API key.
func GenerateAPIKey() string {
	var b [24]byte
	_, _ = rand.Read(b[:])
	return "sk-oi-" + hex.EncodeToString(b[:])
}

// Config returns the current configuration.
func (s *Server) Config() Config { return s.cfg }

// LoadProfile restores the persisted server profile, generating an API key
// on first run.
func (s *Server) LoadProfile() error {
	var id, bind, cors, apiKey string
	var port, lan, autostart int
	err := s.db.QueryRow(`SELECT id,port,bind,allow_lan,cors,autostart,api_key FROM server_profiles ORDER BY created_at LIMIT 1`).
		Scan(&id, &port, &bind, &lan, &cors, &autostart, &apiKey)
	if err == sql.ErrNoRows {
		apiKey = GenerateAPIKey()
		_, err = s.db.Exec(`INSERT INTO server_profiles(id,name,port,bind,allow_lan,cors,autostart,api_key,created_at)
			VALUES ('default','default',1235,'127.0.0.1',0,'',0,?,?)`, apiKey, time.Now().UTC().Format(time.RFC3339Nano))
		if err != nil {
			return err
		}
		s.cfg = Config{Port: 1235, Bind: "127.0.0.1", APIKey: apiKey}
		return nil
	}
	if err != nil {
		return err
	}
	s.cfg = Config{Port: port, Bind: bind, AllowLAN: lan == 1, CORS: cors, APIKey: apiKey, Autostart: autostart == 1}
	if s.cfg.APIKey == "" {
		s.cfg.APIKey = GenerateAPIKey()
		s.persist()
	}
	return nil
}

func (s *Server) persist() {
	lan, auto := 0, 0
	if s.cfg.AllowLAN {
		lan = 1
	}
	if s.cfg.Autostart {
		auto = 1
	}
	_, _ = s.db.Exec(`UPDATE server_profiles SET port=?, bind=?, allow_lan=?, cors=?, autostart=?, api_key=? WHERE id='default'`,
		s.cfg.Port, s.cfg.Bind, lan, s.cfg.CORS, auto, s.cfg.APIKey)
}

// UpdateConfig validates and persists new settings. Restart required to bind.
func (s *Server) UpdateConfig(cfg Config) error {
	if cfg.Port < 1 || cfg.Port > 65535 {
		return fmt.Errorf("invalid port %d", cfg.Port)
	}
	if cfg.AllowLAN && cfg.Bind == "127.0.0.1" {
		cfg.Bind = "0.0.0.0"
	}
	if !cfg.AllowLAN {
		cfg.Bind = "127.0.0.1"
	}
	if cfg.APIKey == "" {
		cfg.APIKey = s.cfg.APIKey
	}
	s.cfg = cfg
	s.persist()
	return nil
}

// RegenerateKey replaces the public API key.
func (s *Server) RegenerateKey() string {
	s.cfg.APIKey = GenerateAPIKey()
	s.persist()
	return s.cfg.APIKey
}

// Running reports server state.
func (s *Server) Running() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.running
}

// RecentRequests returns the last N logged requests (newest first).
func (s *Server) RecentRequests() []RequestLog {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]RequestLog, len(s.reqs))
	for i, r := range s.reqs {
		out[len(s.reqs)-1-i] = r
	}
	return out
}

// Clients returns recently active client IPs.
func (s *Server) Clients() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := []string{}
	for c := range s.clients {
		out = append(out, c)
	}
	return out
}

// Start binds and serves. LAN binding requires AllowLAN (defense in depth).
func (s *Server) Start() error {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return nil
	}
	s.mu.Unlock()

	bind := s.cfg.Bind
	if bind != "127.0.0.1" && !s.cfg.AllowLAN {
		return fmt.Errorf("refusing non-loopback bind without explicit LAN opt-in")
	}
	ln, err := net.Listen("tcp", net.JoinHostPort(bind, fmt.Sprint(s.cfg.Port)))
	if err != nil {
		return fmt.Errorf("binding %s:%d: %w", bind, s.cfg.Port, err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", s.wrap(s.handleModels))
	mux.HandleFunc("POST /v1/chat/completions", s.wrap(s.handleChatCompletions))
	mux.HandleFunc("POST /v1/completions", s.wrap(s.handlePassthrough))
	mux.HandleFunc("POST /v1/embeddings", s.wrap(s.handlePassthrough))
	mux.HandleFunc("POST /v1/responses", s.wrap(s.handleResponses))

	s.http = &http.Server{Handler: mux, ReadHeaderTimeout: 30 * time.Second}
	s.mu.Lock()
	s.running = true
	s.mu.Unlock()
	s.log.Info("public API started", "bind", bind, "port", s.cfg.Port, "lan", s.cfg.AllowLAN)
	go func() {
		if err := s.http.Serve(ln); err != nil && err != http.ErrServerClosed {
			s.log.Error("public API failed", "err", err)
		}
		s.mu.Lock()
		s.running = false
		s.mu.Unlock()
	}()
	return nil
}

// Stop shuts the server down.
func (s *Server) Stop() {
	if s.http == nil {
		return
	}
	_ = s.http.Close()
	s.mu.Lock()
	s.running = false
	s.mu.Unlock()
}

// wrap enforces auth, CORS and request logging.
func (s *Server) wrap(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		if s.cfg.CORS != "" {
			w.Header().Set("Access-Control-Allow-Origin", s.cfg.CORS)
			w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if !s.authorized(r) {
			s.logReq(r, "", http.StatusUnauthorized, start)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"error":{"message":"invalid or missing API key","type":"authentication_error"}}`)
			return
		}
		sw := &statusWriter{ResponseWriter: w, status: 200}
		h(sw, r)
		s.logReq(r, modelOf(r), sw.status, start)
	}
}

func (s *Server) authorized(r *http.Request) bool {
	h := r.Header.Get("Authorization")
	if !strings.HasPrefix(h, "Bearer ") {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(strings.TrimPrefix(h, "Bearer ")), []byte(s.cfg.APIKey)) == 1
}

func (s *Server) logReq(r *http.Request, model string, status int, start time.Time) {
	host, _, _ := net.SplitHostPort(r.RemoteAddr)
	s.mu.Lock()
	s.clients[host] = time.Now()
	s.reqs = append(s.reqs, RequestLog{
		Time: start.UTC(), Method: r.Method, Path: r.URL.Path, Model: model,
		Status: status, DurationMS: time.Since(start).Milliseconds(), Client: host,
	})
	if len(s.reqs) > 200 {
		s.reqs = s.reqs[len(s.reqs)-200:]
	}
	s.mu.Unlock()
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

// Flush supports SSE streaming through the wrapper.
func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// modelOf extracts the model field from a request body (best-effort).
func modelOf(r *http.Request) string {
	if v := r.Context().Value(ctxKeyModel); v != nil {
		return v.(string)
	}
	return ""
}

type ctxKey struct{ name string }

var ctxKeyModel = ctxKey{"model"}

// resolveModel maps a request model name to a ready instance endpoint.
func (s *Server) resolveModel(r *http.Request, body []byte) (Endpoint, []byte, error) {
	var probe struct {
		Model string `json:"model"`
	}
	_ = json.Unmarshal(body, &probe)
	if probe.Model == "" {
		return Endpoint{}, body, fmt.Errorf("missing model field")
	}
	id, err := s.eps.ResolveModelID(probe.Model)
	if err != nil {
		return Endpoint{}, body, err
	}
	ep, err := s.eps.EndpointFor(id)
	if err != nil {
		return Endpoint{}, body, err
	}
	return ep, gguf.PatchJSONReasoning(body, s.reasoningFor(id)), nil
}

func (s *Server) reasoningFor(modelID string) gguf.Reasoning {
	if s.db == nil || modelID == "" {
		return gguf.Reasoning{}
	}
	var arch, meta string
	err := s.db.QueryRow(`SELECT architecture, COALESCE(metadata_json,'') FROM models WHERE id=?`, modelID).
		Scan(&arch, &meta)
	if err != nil {
		return gguf.Reasoning{}
	}
	return gguf.ReasoningFromMetadata(json.RawMessage(meta), arch)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	// llama-server exposes /v1/models per instance; aggregate across instances
	// via the endpoint provider instead.
	type modelEntry struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	list := map[string]any{"object": "list", "data": []modelEntry{}}
	if lp, ok := s.eps.(LoadedLister); ok {
		data := []modelEntry{}
		for _, m := range lp.LoadedModels() {
			data = append(data, modelEntry{ID: m, Object: "model", Created: time.Now().Unix(), OwnedBy: "openinfer-studio"})
		}
		list["data"] = data
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(list)
}

// LoadedLister lists ready model names for /v1/models.
type LoadedLister interface {
	LoadedModels() []string
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.forward(w, r, "/v1/chat/completions")
}

func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	// llama-server serves /v1/responses on recent builds; older runtimes
	// return 404 which we surface transparently.
	s.forward(w, r, "/v1/responses")
}

func (s *Server) handlePassthrough(w http.ResponseWriter, r *http.Request) {
	s.forward(w, r, r.URL.Path)
}

// forward proxies a request to the resolved instance, streaming the response
// (including SSE) back to the client.
func (s *Server) forward(w http.ResponseWriter, r *http.Request, path string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if err != nil {
		writeOAIError(w, http.StatusBadRequest, "reading request body failed")
		return
	}
	ep, newBody, err := s.resolveModel(r, body)
	if err != nil {
		writeOAIError(w, http.StatusNotFound, err.Error())
		return
	}
	var probe struct {
		Model string `json:"model"`
	}
	_ = json.Unmarshal(newBody, &probe)
	ctx := contextWithModel(r.Context(), probe.Model)

	upstream, err := http.NewRequestWithContext(ctx, r.Method, ep.URL+path, bytes.NewReader(newBody))
	if err != nil {
		writeOAIError(w, http.StatusInternalServerError, "building upstream request failed")
		return
	}
	upstream.Header = r.Header.Clone()
	upstream.Header.Set("Authorization", "Bearer "+ep.APIKey)
	upstream.Header.Del("Accept-Encoding")

	resp, err := s.client.Do(upstream)
	if err != nil {
		writeOAIError(w, http.StatusBadGateway, "model instance unreachable: "+err.Error())
		return
	}
	defer resp.Body.Close()
	s.eps.Touch(probe.Model)

	for k, vs := range resp.Header {
		lk := strings.ToLower(k)
		if lk == "content-length" || lk == "transfer-encoding" || lk == "connection" {
			continue
		}
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)

	flusher, _ := w.(http.Flusher)
	ct := strings.ToLower(resp.Header.Get("Content-Type"))
	if path == "/v1/chat/completions" && strings.Contains(ct, "text/event-stream") {
		rewriteChatSSE(w, resp.Body, flusher)
		return
	}
	if path == "/v1/chat/completions" && strings.Contains(ct, "json") {
		rewriteChatJSON(w, resp.Body)
		return
	}

	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

// rewriteChatSSE peels thought markup out of streamed content deltas into
// OpenAI reasoning_content so external clients do not see raw <think> tags.
func rewriteChatSSE(w http.ResponseWriter, body io.Reader, flusher http.Flusher) {
	var split reasoning.Splitter
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 64<<10), 4<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			fmt.Fprintf(w, "%s\n", line)
			if flusher != nil {
				flusher.Flush()
			}
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			if rDelta, cDelta := split.Flush(); rDelta != "" || cDelta != "" {
				writeReasoningChunk(w, flusher, rDelta, cDelta)
			}
			fmt.Fprintf(w, "data: [DONE]\n\n")
			if flusher != nil {
				flusher.Flush()
			}
			return
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			fmt.Fprintf(w, "data: %s\n\n", payload)
			if flusher != nil {
				flusher.Flush()
			}
			continue
		}
		choices, _ := chunk["choices"].([]any)
		rewrote := false
		for _, chAny := range choices {
			ch, ok := chAny.(map[string]any)
			if !ok {
				continue
			}
			delta, _ := ch["delta"].(map[string]any)
			if delta == nil {
				continue
			}
			content, _ := delta["content"].(string)
			if content == "" {
				continue
			}
			rDelta, cDelta := split.Push(content)
			rewrote = true
			if rDelta == "" && cDelta == content {
				continue
			}
			if cDelta == "" {
				delete(delta, "content")
			} else {
				delta["content"] = cDelta
			}
			if rDelta != "" {
				if existing, _ := delta["reasoning_content"].(string); existing != "" {
					delta["reasoning_content"] = existing + rDelta
				} else {
					delta["reasoning_content"] = rDelta
				}
			}
			if len(delta) == 0 {
				delta["content"] = ""
			}
		}
		if !rewrote {
			fmt.Fprintf(w, "data: %s\n\n", payload)
		} else {
			b, err := json.Marshal(chunk)
			if err != nil {
				fmt.Fprintf(w, "data: %s\n\n", payload)
			} else {
				fmt.Fprintf(w, "data: %s\n\n", b)
			}
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
}

func writeReasoningChunk(w http.ResponseWriter, flusher http.Flusher, reasoningDelta, contentDelta string) {
	delta := map[string]any{}
	if contentDelta != "" {
		delta["content"] = contentDelta
	}
	if reasoningDelta != "" {
		delta["reasoning_content"] = reasoningDelta
	}
	if len(delta) == 0 {
		return
	}
	chunk := map[string]any{
		"choices": []map[string]any{{
			"index": 0, "delta": delta, "finish_reason": nil,
		}},
	}
	b, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", b)
	if flusher != nil {
		flusher.Flush()
	}
}

func rewriteChatJSON(w http.ResponseWriter, body io.Reader) {
	raw, err := io.ReadAll(io.LimitReader(body, 32<<20))
	if err != nil {
		return
	}
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		_, _ = w.Write(raw)
		return
	}
	choices, _ := obj["choices"].([]any)
	for _, chAny := range choices {
		ch, ok := chAny.(map[string]any)
		if !ok {
			continue
		}
		msg, _ := ch["message"].(map[string]any)
		if msg == nil {
			continue
		}
		content, _ := msg["content"].(string)
		existing, _ := msg["reasoning_content"].(string)
		taggedR, taggedC := reasoning.Split(content)
		if taggedR == "" && taggedC == content {
			continue
		}
		msg["content"] = taggedC
		if existing != "" && taggedR != "" {
			msg["reasoning_content"] = existing + "\n" + taggedR
		} else if taggedR != "" {
			msg["reasoning_content"] = taggedR
		}
	}
	b, err := json.Marshal(obj)
	if err != nil {
		_, _ = w.Write(raw)
		return
	}
	_, _ = w.Write(b)
}

func writeOAIError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": msg, "type": "openinfer_error"},
	})
}

func contextWithModel(ctx context.Context, model string) context.Context {
	return context.WithValue(ctx, ctxKeyModel, model)
}
