package instances

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/openinfer/openinfer-studio/internal/processes"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

// DiffusionEngine drives llama-diffusion-gemma-visual-server over stdin/stdout.
// Protocol (Unsloth / llama.cpp DiffusionGemma PR):
//
//	startup → scan until "READY <n_vocab> [maxtok]"
//	request → write JSON file, send path on stdin
//	stream  → F … / C … / STATS … / DONE / ERR …
//	shutdown → "QUIT"
type DiffusionEngine struct {
	handle  *processes.Handle
	stdin   io.WriteCloser
	stdout  *bufio.Reader
	log     io.Writer
	reqPath string
	NVocab  int
	MaxTok  int
	Canvas  uint32

	mu sync.Mutex
}

// DiffusionLaunch configures the visual-server child.
type DiffusionLaunch struct {
	Exe       string
	ModelPath string
	WorkDir   string
	Env       map[string]string // includes LD_LIBRARY_PATH, NGL, MAXTOK
	NGL       int
	MaxTok    int // 0 = auto; clamped ≤8192 when explicit
	Canvas    uint32
	Log       io.Writer
	TempDir   string
}

func canvasMaxTok(n int) int {
	if n <= 0 || n > 8192 {
		return 0 // let the server auto-size to VRAM
	}
	return n
}

func resolveNGL(s LoadSettings) int {
	switch s.GPUOffload {
	case "none":
		return 0
	case "custom":
		if s.GPULayers > 0 {
			return s.GPULayers
		}
		return 0
	case "all", "auto", "":
		return 999
	default:
		return 999
	}
}

// StartDiffusionEngine launches the visual server and waits for READY.
func StartDiffusionEngine(launch DiffusionLaunch) (*DiffusionEngine, error) {
	env := map[string]string{}
	for k, v := range launch.Env {
		env[k] = v
	}
	env["NGL"] = strconv.Itoa(launch.NGL)
	env["MAXTOK"] = strconv.Itoa(canvasMaxTok(launch.MaxTok))

	// Mirror load logs to the instance log (stderr); protocol is on stdout.
	spec := processes.Spec{
		Exe:  launch.Exe,
		Args: []string{launch.ModelPath},
		Dir:  launch.WorkDir,
		Env:  env,
	}
	h, stdin, stdout, err := processes.StartPiped(spec, launch.Log)
	if err != nil {
		return nil, err
	}

	reqDir := launch.TempDir
	if reqDir == "" {
		reqDir = os.TempDir()
	}
	if st, err := os.Stat("/dev/shm"); err == nil && st.IsDir() {
		reqDir = "/dev/shm"
	}
	reqPath := filepath.Join(reqDir, fmt.Sprintf("openinfer-diffusion-%d.req", os.Getpid()))

	eng := &DiffusionEngine{
		handle:  h,
		stdin:   stdin,
		stdout:  bufio.NewReaderSize(stdout, 1<<20),
		log:     launch.Log,
		reqPath: reqPath,
		Canvas:  launch.Canvas,
	}
	if eng.Canvas == 0 {
		eng.Canvas = 256
	}

	if err := eng.waitReady(10 * time.Minute); err != nil {
		eng.forceKill()
		return nil, err
	}
	return eng, nil
}

func (e *DiffusionEngine) logf(format string, args ...any) {
	if e.log == nil {
		return
	}
	fmt.Fprintf(e.log, format, args...)
}

func (e *DiffusionEngine) waitReady(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		if time.Now().After(deadline) {
			return fmt.Errorf("diffusion visual server did not become READY within %s", timeout)
		}
		line, err := e.readLine(deadline)
		if err != nil {
			return fmt.Errorf("waiting for READY: %w", err)
		}
		if strings.HasPrefix(line, "READY") {
			parts := strings.Fields(line)
			if len(parts) > 1 {
				e.NVocab, _ = strconv.Atoi(parts[1])
			}
			if len(parts) > 2 {
				e.MaxTok, _ = strconv.Atoi(parts[2])
			}
			e.logf("diffusion visual server READY n_vocab=%d maxtok=%d\n", e.NVocab, e.MaxTok)
			return nil
		}
		// Pre-READY chatter (model load logs) — keep in the instance log.
		e.logf("%s\n", line)
	}
}

func (e *DiffusionEngine) readLine(deadline time.Time) (string, error) {
	type result struct {
		line string
		err  error
	}
	ch := make(chan result, 1)
	go func() {
		s, err := e.stdout.ReadString('\n')
		ch <- result{strings.TrimRight(s, "\r\n"), err}
	}()
	remain := time.Until(deadline)
	if remain < 0 {
		remain = 0
	}
	select {
	case r := <-ch:
		return r.line, r.err
	case <-time.After(remain):
		return "", fmt.Errorf("read timeout")
	}
}

// DiffusionMessage is one chat turn for the visual server.
type DiffusionMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// DiffusionEvent is one streamed protocol event.
type DiffusionEvent struct {
	Kind   string // frame|commit|stats|done|error
	Frame  string
	Commit string
	Stats  map[string]any
	Error  string
	Block  int
	Step   int
	Total  int
}

// Generate runs one diffusion turn. Only one Generate may run at a time.
func (e *DiffusionEngine) Generate(ctx context.Context, messages []DiffusionMessage, nBlocks, seed int) (<-chan DiffusionEvent, error) {
	e.mu.Lock()
	if e.handle == nil || e.handle.Cmd.ProcessState != nil {
		e.mu.Unlock()
		return nil, fmt.Errorf("diffusion visual server is not running")
	}
	if nBlocks < 1 {
		nBlocks = 1
	}
	if seed == 0 {
		seed = 3407
	}
	req := map[string]any{
		"seed":     seed,
		"n_blocks": nBlocks,
		"messages": messages,
	}
	body, err := json.Marshal(req)
	if err != nil {
		e.mu.Unlock()
		return nil, err
	}
	if err := os.WriteFile(e.reqPath, body, 0o600); err != nil {
		e.mu.Unlock()
		return nil, err
	}
	e.logf("diffusion request n_blocks=%d seed=%d messages=%d bytes=%d\n",
		nBlocks, seed, len(messages), len(body))
	if _, err := fmt.Fprintf(e.stdin, "%s\n", e.reqPath); err != nil {
		e.mu.Unlock()
		return nil, fmt.Errorf("sending request to visual server: %w", err)
	}

	out := make(chan DiffusionEvent, 16)
	go func() {
		defer close(out)
		defer e.mu.Unlock()
		for {
			if err := ctx.Err(); err != nil {
				out <- DiffusionEvent{Kind: "error", Error: err.Error()}
				return
			}
			line, err := e.stdout.ReadString('\n')
			if err != nil {
				out <- DiffusionEvent{Kind: "error", Error: fmt.Sprintf("visual server closed: %v", err)}
				return
			}
			line = strings.TrimRight(line, "\r\n")
			switch {
			case line == "DONE":
				e.logf("diffusion DONE\n")
				out <- DiffusionEvent{Kind: "done"}
				return
			case strings.HasPrefix(line, "ERR"):
				e.logf("diffusion %s\n", line)
				out <- DiffusionEvent{Kind: "error", Error: formatDiffusionErr(line)}
				return
			case strings.HasPrefix(line, "STATS"):
				stats := parseDiffusionStats(line)
				e.logf("diffusion %s\n", line)
				out <- DiffusionEvent{Kind: "stats", Stats: stats}
			case strings.HasPrefix(line, "F "):
				// F <block> <step> <total> <json-text>
				parts := strings.SplitN(line, " ", 5)
				if len(parts) < 5 {
					continue
				}
				block, _ := strconv.Atoi(parts[1])
				step, _ := strconv.Atoi(parts[2])
				total, _ := strconv.Atoi(parts[3])
				var text string
				_ = json.Unmarshal([]byte(parts[4]), &text)
				out <- DiffusionEvent{Kind: "frame", Frame: text, Block: block, Step: step, Total: total}
			case strings.HasPrefix(line, "C "):
				// Protocol: C <block> <json-string> (Unsloth / visual-server.cpp).
				parts := strings.SplitN(line, " ", 3)
				if len(parts) < 3 {
					continue
				}
				var text string
				if err := json.Unmarshal([]byte(parts[2]), &text); err != nil {
					e.logf("diffusion commit parse error: %v payload=%q\n", err, truncLog(parts[2], 120))
					continue
				}
				e.logf("diffusion commit block=%s chars=%d\n", parts[1], len(text))
				out <- DiffusionEvent{Kind: "commit", Commit: text}
			default:
				// Non-protocol chatter during generate — log only.
				e.logf("%s\n", line)
			}
		}
	}()
	return out, nil
}

func truncLog(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func formatDiffusionErr(line string) string {
	parts := strings.Fields(line)
	if len(parts) >= 2 && parts[1] == "toolong" {
		needed, budget := 0, 0
		if len(parts) > 2 {
			needed, _ = strconv.Atoi(parts[2])
		}
		if len(parts) > 3 {
			budget, _ = strconv.Atoi(parts[3])
		}
		return fmt.Sprintf(
			"conversation needs %d tokens but DiffusionGemma's context budget is %d; start a new chat or shorten the message",
			needed, budget)
	}
	return line
}

func parseDiffusionStats(line string) map[string]any {
	stats := map[string]any{}
	for _, tok := range strings.Fields(line)[1:] {
		k, v, ok := strings.Cut(tok, "=")
		if !ok {
			continue
		}
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			stats[k] = n
			continue
		}
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			stats[k] = f
			continue
		}
		stats[k] = v
	}
	return stats
}

// Close asks the visual server to exit, then kills the tree if needed.
func (e *DiffusionEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.stdin != nil {
		_, _ = fmt.Fprintln(e.stdin, "QUIT")
		_ = e.stdin.Close()
	}
	os.Remove(e.reqPath)
	if e.handle == nil {
		return nil
	}
	done := make(chan struct{})
	go func() {
		_, _ = e.handle.Wait()
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-time.After(8 * time.Second):
		e.forceKill()
		<-done
		return nil
	}
}

func (e *DiffusionEngine) forceKill() {
	if e.handle != nil {
		_ = e.handle.KillTree()
	}
	if e.stdin != nil {
		_ = e.stdin.Close()
	}
	os.Remove(e.reqPath)
}

// PID returns the visual-server process id, or 0.
func (e *DiffusionEngine) PID() int {
	if e == nil || e.handle == nil || e.handle.Cmd == nil || e.handle.Cmd.Process == nil {
		return 0
	}
	return e.handle.Cmd.Process.Pid
}

// Wait blocks until the visual server exits.
func (e *DiffusionEngine) Wait() (int, error) {
	if e == nil || e.handle == nil {
		return -1, fmt.Errorf("no diffusion engine")
	}
	return e.handle.Wait()
}

// DiffusionShim exposes OpenAI-compatible /health + /v1/chat/completions on a
// loopback port, translating to the visual-server protocol so Studio chat and
// readiness probing work unchanged.
type DiffusionShim struct {
	engine *DiffusionEngine
	apiKey string
	alias  string
	canvas uint32

	ln     net.Listener
	server *http.Server
}

// StartDiffusionShim serves HTTP on the given listener.
func StartDiffusionShim(ln net.Listener, engine *DiffusionEngine, apiKey, alias string) *DiffusionShim {
	s := &DiffusionShim{
		engine: engine,
		apiKey: apiKey,
		alias:  alias,
		canvas: engine.Canvas,
		ln:     ln,
	}
	if s.canvas == 0 {
		s.canvas = 256
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	s.server = &http.Server{Handler: s.auth(mux)}
	go func() { _ = s.server.Serve(ln) }()
	return s
}

func (s *DiffusionShim) auth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.apiKey != "" {
			got := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
			got = strings.TrimSpace(got)
			if got != s.apiKey {
				http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

func (s *DiffusionShim) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func (s *DiffusionShim) handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id": s.alias, "object": "model", "owned_by": "openinfer",
		}},
	})
}

func (s *DiffusionShim) handleChat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Messages []struct {
			Role    string `json:"role"`
			Content any    `json:"content"`
		} `json:"messages"`
		Stream              bool `json:"stream"`
		MaxTokens           int  `json:"max_tokens"`
		MaxCompletionTokens int  `json:"max_completion_tokens"`
		Seed                *int `json:"seed"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 8<<20)).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid json"}`, http.StatusBadRequest)
		return
	}
	msgs := make([]DiffusionMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		content := flattenContent(m.Content)
		// Prior turns may still contain raw thought-channel markers; strip them
		// so the visual server does not see Studio's display scaffolding.
		if strings.EqualFold(m.Role, "assistant") {
			_, content = splitThoughtChannels(content)
		}
		msgs = append(msgs, DiffusionMessage{
			Role:    m.Role,
			Content: content,
		})
	}
	maxTok := req.MaxTokens
	if req.MaxCompletionTokens > maxTok {
		maxTok = req.MaxCompletionTokens
	}
	nBlocks := diffusionBlocks(maxTok, s.canvas)
	seed := int(time.Now().UnixNano() & 0x7fffffff)
	if req.Seed != nil {
		seed = *req.Seed
	}
	s.engine.logf("diffusion chat max_tokens=%d canvas=%d → n_blocks=%d stream=%v seed=%d\n",
		maxTok, s.canvas, nBlocks, req.Stream, seed)

	if req.Stream {
		s.streamChatContinue(w, r.Context(), msgs, nBlocks, maxTok, seed)
		return
	}
	s.nonStreamChatContinue(w, r.Context(), msgs, nBlocks, maxTok, seed)
}

func flattenContent(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case []any:
		var b strings.Builder
		for _, part := range t {
			m, ok := part.(map[string]any)
			if !ok {
				continue
			}
			if m["type"] == "text" {
				if s, ok := m["text"].(string); ok {
					b.WriteString(s)
				}
			}
		}
		return b.String()
	default:
		return fmt.Sprint(v)
	}
}

func diffusionBlocks(maxTokens int, canvas uint32) int {
	if canvas == 0 {
		canvas = 256
	}
	if maxTokens <= 0 {
		return 8 // sensible default ~2k tokens at canvas 256
	}
	n := (maxTokens + int(canvas) - 1) / int(canvas)
	if n < 1 {
		n = 1
	}
	if n > 64 {
		n = 64
	}
	return n
}

// Thought-channel markers used by DiffusionGemma (and DeepSeek-style fallbacks).
var thoughtMarkers = [][2]string{
	{"<|channel>thought", "<channel|>"},
	{"<think>", "</think>"},
}

func splitThoughtChannels(full string) (reasoning, content string) {
	var rParts, cParts []string
	rest := full
	for {
		best := -1
		bestStart, bestEnd := "", ""
		for _, pair := range thoughtMarkers {
			i := strings.Index(rest, pair[0])
			if i >= 0 && (best < 0 || i < best) {
				best = i
				bestStart, bestEnd = pair[0], pair[1]
			}
		}
		if best < 0 {
			cParts = append(cParts, rest)
			break
		}
		cParts = append(cParts, rest[:best])
		body := rest[best+len(bestStart):]
		if j := strings.Index(body, bestEnd); j >= 0 {
			rParts = append(rParts, body[:j])
			rest = body[j+len(bestEnd):]
			continue
		}
		// Unterminated thought (length-truncated generation).
		rParts = append(rParts, body)
		break
	}
	return strings.Trim(strings.Join(rParts, ""), "\n"), strings.Join(cParts, "")
}

func markerHoldback(full string) int {
	longest := 0
	for _, pair := range thoughtMarkers {
		for _, marker := range pair {
			upper := min(len(marker)-1, len(full))
			for k := upper; k > 0; k-- {
				if strings.HasSuffix(full, marker[:k]) {
					longest = max(longest, k)
					break
				}
			}
		}
	}
	return longest
}

func statsInt(stats map[string]any, keys ...string) int {
	for _, k := range keys {
		switch v := stats[k].(type) {
		case int64:
			return int(v)
		case float64:
			return int(v)
		case int:
			return v
		}
	}
	return 0
}

func timingsFromStats(stats map[string]any) (timings map[string]any, promptN, completionN int) {
	if stats == nil {
		return nil, 0, 0
	}
	num := func(keys ...string) float64 {
		for _, k := range keys {
			switch v := stats[k].(type) {
			case int64:
				return float64(v)
			case float64:
				return v
			case int:
				return float64(v)
			}
		}
		return 0
	}
	rate := func(n, ms float64) float64 {
		if ms <= 0 {
			return 0
		}
		return n / ms * 1000
	}
	P := int(num("prompt_n"))
	G := int(num("predicted_n"))
	wall := num("wall_ms", "predicted_ms")
	steps := num("steps")
	blocks := num("blocks")
	canvas := num("canvas")
	prep := num("prompt_prepare_ms", "prompt_ms")
	decode := num("decode_ms")
	par := rate(canvas*steps, wall)
	eff := rate(canvas*blocks, wall)
	out := rate(float64(G), wall)
	timings = map[string]any{
		"predicted_n":                 G,
		"predicted_ms":                wall,
		"predicted_per_second":        par,
		"prompt_n":                    P,
		"diffusion":                   true,
		"diffusion_blocks":            int(blocks),
		"diffusion_steps":             int(steps),
		"diffusion_canvas":            int(canvas),
		"diffusion_prompt_n":          P,
		"diffusion_prompt_prepare_ms": prep,
		"diffusion_decode_ms":         decode,
		"diffusion_wall_ms":           wall,
		"diffusion_effective_tok_s":   eff,
		"diffusion_parallel_tok_s":    par,
		"diffusion_output_tok_s":      out,
	}
	return timings, P, G
}

// mergeStats accumulates wall-clock / block counts across auto-continue rounds.
func mergeStats(dst, src map[string]any) map[string]any {
	if src == nil {
		return dst
	}
	if dst == nil {
		out := make(map[string]any, len(src))
		for k, v := range src {
			out[k] = v
		}
		return out
	}
	dst["blocks"] = statsInt(dst, "blocks") + statsInt(src, "blocks")
	dst["steps"] = statsInt(dst, "steps") + statsInt(src, "steps")
	dst["predicted_n"] = statsInt(dst, "predicted_n") + statsInt(src, "predicted_n")
	dst["wall_ms"] = numAsFloat(dst["wall_ms"]) + numAsFloat(src["wall_ms"])
	dst["decode_ms"] = numAsFloat(dst["decode_ms"]) + numAsFloat(src["decode_ms"])
	if p := statsInt(src, "prompt_n"); p > statsInt(dst, "prompt_n") {
		dst["prompt_n"] = p
	}
	if c := statsInt(src, "canvas"); c > 0 {
		dst["canvas"] = c
	}
	return dst
}

func numAsFloat(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int64:
		return float64(t)
	case int:
		return float64(t)
	default:
		return 0
	}
}

type diffusionRound struct {
	rawCommit string // last cumulative commit from this round (may include thought markers)
	content   string // thought-stripped answer text for this round only
	stats     map[string]any
	err       string
}

func consumeDiffusionRound(events <-chan DiffusionEvent) diffusionRound {
	var r diffusionRound
	for ev := range events {
		switch ev.Kind {
		case "commit":
			r.rawCommit = ev.Commit
		case "stats":
			r.stats = ev.Stats
		case "error":
			r.err = ev.Error
		}
	}
	_, r.content = splitThoughtChannels(r.rawCommit)
	return r
}

func looksTruncated(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return true
	}
	// Mid-list / mid-clause endings the EB sampler often leaves after early EOS.
	if strings.HasSuffix(s, ":") || strings.HasSuffix(s, ",") || strings.HasSuffix(s, "(") {
		return true
	}
	r, _ := utf8.DecodeLastRuneInString(s)
	switch r {
	case '.', '!', '?', '"', '\'', ')', ']', '\u201d', '\u2019':
		return false
	default:
		return true
	}
}

func shouldContinueDiffusion(blocksUsed, blocksBudget, predicted, maxTok, remRequested, roundBlocks int, content string) bool {
	if blocksUsed >= blocksBudget {
		return false
	}
	if maxTok > 0 && predicted >= maxTok {
		return false
	}
	// Completed the full allotment this round (no early EOS from the visual server).
	if remRequested > 0 && roundBlocks >= remRequested {
		return false
	}
	// Only continue when the model early-stopped mid-thought. Do NOT keep
	// generating just because predicted_n << max_tokens — that turns a finished
	// "Hello!" into seven extra diffusion rounds on Vulkan.
	return looksTruncated(content)
}

func continueMessages(base []DiffusionMessage, soFar string) []DiffusionMessage {
	out := make([]DiffusionMessage, 0, len(base)+2)
	out = append(out, base...)
	out = append(out,
		DiffusionMessage{Role: "assistant", Content: soFar},
		DiffusionMessage{Role: "user", Content: "Continue exactly where you left off. Do not repeat earlier text."},
	)
	return out
}

// longestCommonPrefix returns the shared leading bytes of a and b (UTF-8 safe by
// truncating a broken trailing rune).
func longestCommonPrefix(a, b string) string {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	// Back up if we landed mid-rune (i == len(a) is already a boundary).
	for i > 0 && i < len(a) && !utf8.RuneStart(a[i]) {
		i--
	}
	return a[:i]
}

// wordStablePrefix holds back a trailing partial word so the UI only advances on
// complete words (spaces / punctuation). flush=true returns the full string.
func wordStablePrefix(s string, flush bool) string {
	if flush || s == "" {
		return s
	}
	// Find last word boundary; keep through that boundary.
	cut := -1
	for i, r := range s {
		switch r {
		case ' ', '\n', '\t', '\r', '.', ',', ';', ':', '!', '?', ')', ']', '"', '\'':
			cut = i + utf8.RuneLen(r)
		}
	}
	if cut < 0 {
		return ""
	}
	return s[:cut]
}

// stableCanvasTracker turns noisy denoising frames into a monotonically growing
// confirmed prefix: only text that agreed across consecutive frames, and only
// through complete word boundaries.
type stableCanvasTracker struct {
	prevFrame string
	stable    string // prefix locked for ≥2 agreeing steps
	pending   string // prefix that agreed on the latest step (not yet shown)
	streamed  string // full answer already sent to the client
	confirmed string // answerSoFar + committedRound (100% from C lines)
}

func (t *stableCanvasTracker) setConfirmed(confirmed string) (deltaContent string, snapshot string, useSnapshot bool) {
	t.confirmed = confirmed
	t.prevFrame = ""
	t.stable = ""
	t.pending = ""
	full := confirmed
	if strings.HasPrefix(full, t.streamed) {
		d := full[len(t.streamed):]
		t.streamed = full
		return d, "", false
	}
	// Rare: commit corrected earlier stable guesses — one clean replace.
	t.streamed = full
	return "", full, true
}

func (t *stableCanvasTracker) onFrame(frameBody string) (deltaContent string) {
	if frameBody == "" {
		return ""
	}
	if t.prevFrame == "" {
		t.prevFrame = frameBody
		return ""
	}
	lcp := longestCommonPrefix(t.prevFrame, frameBody)
	t.prevFrame = frameBody

	// pending = prefix that agreed on the latest step. Promote to stable only
	// when the next step still agrees (pending ⊆ lcp), so we never stream text
	// that flips on the following denoising step.
	if t.pending != "" && strings.HasPrefix(lcp, t.pending) {
		t.stable = t.pending
	}
	if strings.HasPrefix(lcp, t.stable) {
		t.pending = lcp
	} else if !strings.HasPrefix(t.stable, lcp) {
		// Hard divergence — wait for commit rather than guessing.
		t.pending = ""
		return ""
	} else {
		// lcp is a shorter prefix of stable; keep stable, clear pending.
		t.pending = ""
		return ""
	}

	reveal := wordStablePrefix(t.stable, false)
	full := t.confirmed + reveal
	if !strings.HasPrefix(full, t.streamed) {
		return ""
	}
	d := full[len(t.streamed):]
	if d == "" {
		return ""
	}
	t.streamed = full
	return d
}

func (t *stableCanvasTracker) flushCanvas() (deltaContent string) {
	// End of block: promote pending → stable and release held partial words.
	if t.pending != "" && strings.HasPrefix(t.pending, t.stable) {
		t.stable = t.pending
	}
	reveal := wordStablePrefix(t.stable, true)
	full := t.confirmed + reveal
	if !strings.HasPrefix(full, t.streamed) {
		return ""
	}
	d := full[len(t.streamed):]
	t.streamed = full
	return d
}

func (s *DiffusionShim) streamChatContinue(w http.ResponseWriter, ctx context.Context, base []DiffusionMessage, nBlocks, maxTok, seed int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, `{"error":"streaming unsupported"}`, http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher.Flush()

	id := fmt.Sprintf("chatcmpl-diff-%d", time.Now().UnixNano())
	writeSSE := func(obj map[string]any) {
		b, _ := json.Marshal(obj)
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}
	writeDelta := func(delta map[string]any, finish any) {
		writeSSE(map[string]any{
			"id": id, "object": "chat.completion.chunk",
			"choices": []map[string]any{{
				"index": 0, "delta": delta, "finish_reason": finish,
			}},
		})
	}
	writeSnapshot := func(snapshot, reasoning string) {
		writeSSE(map[string]any{
			"id": id, "object": "chat.completion.chunk",
			"type": "diffusion_snapshot", "text": snapshot, "reasoning": reasoning,
			"choices": []map[string]any{{
				"index": 0, "delta": map[string]any{}, "finish_reason": nil,
			}},
		})
	}
	emitContent := func(delta string) {
		if delta == "" {
			return
		}
		writeDelta(map[string]any{"content": delta}, nil)
	}

	var aggStats map[string]any
	blocksUsed, predicted := 0, 0
	answerSoFar := ""
	reasoningSoFar := ""
	msgs := base
	roundSeed := seed
	tracker := stableCanvasTracker{}

	emitReasoning := func(next string) {
		if next == "" || next == reasoningSoFar {
			return
		}
		if strings.HasPrefix(next, reasoningSoFar) {
			if d := next[len(reasoningSoFar):]; d != "" {
				writeDelta(map[string]any{"reasoning_content": d}, nil)
			}
			reasoningSoFar = next
			return
		}
		writeSnapshot(tracker.streamed, next)
		reasoningSoFar = next
	}

	for round := 0; round < 8; round++ {
		rem := nBlocks - blocksUsed
		if rem < 1 {
			break
		}
		events, err := s.engine.Generate(ctx, msgs, rem, roundSeed)
		if err != nil {
			writeDelta(map[string]any{"content": "\n[" + err.Error() + "]"}, "stop")
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}

		lastCommit := ""
		committedRound := ""
		var roundStats map[string]any
		errMsg := ""
		tracker.confirmed = answerSoFar
		tracker.prevFrame = ""
		tracker.stable = ""
		tracker.pending = ""

		for ev := range events {
			switch ev.Kind {
			case "stats":
				roundStats = ev.Stats
			case "frame":
				_, frameBody := splitThoughtChannels(ev.Frame)
				if frameBody == "" {
					frameBody = strings.TrimSpace(ev.Frame)
				}
				emitContent(tracker.onFrame(frameBody))
			case "commit":
				lastCommit = ev.Commit
				// Flush any held partial word from the canvas, then lock the commit.
				emitContent(tracker.flushCanvas())
				cut := markerHoldback(ev.Commit)
				safe := ev.Commit
				if cut > 0 {
					safe = ev.Commit[:len(ev.Commit)-cut]
				}
				rr, rc := splitThoughtChannels(safe)
				committedRound = rc
				if rr != "" {
					emitReasoning(rr)
				}
				delta, snap, useSnap := tracker.setConfirmed(answerSoFar + committedRound)
				if useSnap {
					writeSnapshot(snap, reasoningSoFar)
				} else {
					emitContent(delta)
				}
			case "error":
				errMsg = ev.Error
			case "done":
				if lastCommit != "" {
					emitContent(tracker.flushCanvas())
					rr, rc := splitThoughtChannels(lastCommit)
					committedRound = rc
					if rr != "" {
						emitReasoning(rr)
					}
					delta, snap, useSnap := tracker.setConfirmed(answerSoFar + committedRound)
					if useSnap {
						writeSnapshot(snap, reasoningSoFar)
					} else {
						emitContent(delta)
					}
				}
			}
		}
		if errMsg != "" {
			writeDelta(map[string]any{"content": "\n[" + errMsg + "]"}, "stop")
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}

		_, roundContent := splitThoughtChannels(lastCommit)
		roundBlocks := statsInt(roundStats, "blocks")
		roundPred := statsInt(roundStats, "predicted_n")
		aggStats = mergeStats(aggStats, roundStats)
		blocksUsed += roundBlocks
		predicted += roundPred

		if round == 0 {
			answerSoFar = roundContent
		} else {
			answerSoFar += roundContent
		}

		s.engine.logf("diffusion round=%d rem=%d got_blocks=%d predicted=%d total_blocks=%d/%d truncated=%v\n",
			round, rem, roundBlocks, roundPred, blocksUsed, nBlocks, looksTruncated(answerSoFar))

		if !shouldContinueDiffusion(blocksUsed, nBlocks, predicted, maxTok, rem, roundBlocks, answerSoFar) {
			break
		}
		msgs = continueMessages(base, answerSoFar)
		roundSeed = seed + round + 1
	}

	timings, P, G := timingsFromStats(aggStats)
	if G == 0 {
		G = predicted
	}
	if timings != nil {
		writeSSE(map[string]any{
			"id": id, "object": "chat.completion.chunk",
			"choices": []any{},
			"usage": map[string]any{
				"prompt_tokens": P, "completion_tokens": G, "total_tokens": P + G,
			},
			"timings": timings,
		})
	}
	writeDelta(map[string]any{}, "stop")
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *DiffusionShim) nonStreamChatContinue(w http.ResponseWriter, ctx context.Context, base []DiffusionMessage, nBlocks, maxTok, seed int) {
	var aggStats map[string]any
	blocksUsed, predicted := 0, 0
	answerSoFar := ""
	reasoningSoFar := ""
	msgs := base
	roundSeed := seed

	for round := 0; round < 8; round++ {
		rem := nBlocks - blocksUsed
		if rem < 1 {
			break
		}
		events, err := s.engine.Generate(ctx, msgs, rem, roundSeed)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":{"message":%q}}`, err.Error()), http.StatusInternalServerError)
			return
		}
		r := consumeDiffusionRound(events)
		if r.err != "" {
			http.Error(w, fmt.Sprintf(`{"error":{"message":%q}}`, r.err), http.StatusInternalServerError)
			return
		}
		rr, rc := splitThoughtChannels(r.rawCommit)
		roundBlocks := statsInt(r.stats, "blocks")
		roundPred := statsInt(r.stats, "predicted_n")
		aggStats = mergeStats(aggStats, r.stats)
		blocksUsed += roundBlocks
		predicted += roundPred
		if round == 0 {
			answerSoFar = rc
			reasoningSoFar = rr
		} else {
			answerSoFar += rc
			if rr != "" {
				if reasoningSoFar != "" {
					reasoningSoFar += "\n"
				}
				reasoningSoFar += rr
			}
		}
		s.engine.logf("diffusion round=%d rem=%d got_blocks=%d predicted=%d total_blocks=%d/%d truncated=%v\n",
			round, rem, roundBlocks, roundPred, blocksUsed, nBlocks, looksTruncated(answerSoFar))
		if !shouldContinueDiffusion(blocksUsed, nBlocks, predicted, maxTok, rem, roundBlocks, answerSoFar) {
			break
		}
		msgs = continueMessages(base, answerSoFar)
		roundSeed = seed + round + 1
	}

	msg := map[string]any{"role": "assistant", "content": answerSoFar}
	if reasoningSoFar != "" {
		msg["reasoning_content"] = reasoningSoFar
	}
	timings, P, G := timingsFromStats(aggStats)
	if G == 0 {
		G = predicted
	}
	out := map[string]any{
		"id":     fmt.Sprintf("chatcmpl-diff-%d", time.Now().UnixNano()),
		"object": "chat.completion",
		"model":  s.alias,
		"choices": []map[string]any{{
			"index": 0, "message": msg, "finish_reason": "stop",
		}},
		"usage": map[string]any{
			"prompt_tokens": P, "completion_tokens": G, "total_tokens": P + G,
		},
	}
	if timings != nil {
		out["timings"] = timings
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

// Close shuts down the HTTP listener.
func (s *DiffusionShim) Close() error {
	if s.server != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = s.server.Shutdown(ctx)
	}
	if s.ln != nil {
		_ = s.ln.Close()
	}
	return nil
}

// ResolveDiffusionBinary finds the visual server next to a runtime's llama-server.
func ResolveDiffusionBinary(llamaServer string) (string, error) {
	return runtimes.FindDiffusionServer(llamaServer)
}
