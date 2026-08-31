// fakeserver is a test fixture standing in for llama-server. It implements
// just enough surface for process-management integration tests: --version,
// --help, --port/--api-key/--model/--host flags, /health readiness, and
// streaming /v1/chat/completions. If the model path contains "crashme", it
// exits(1) after binding, simulating a load crash.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

func main() {
	fs := flag.NewFlagSet("llama-server", flag.ContinueOnError)
	var (
		version    = fs.Bool("version", false, "print version")
		port       = fs.Int("port", 8080, "")
		host       = fs.String("host", "127.0.0.1", "")
		apiKey     = fs.String("api-key", "", "")
		model      = fs.String("model", "", "")
		alias      = fs.String("alias", "", "")
		ctxSize    = fs.Int("ctx-size", 4096, "")
		ngl        = fs.Int("n-gpu-layers", 0, "")
		threads    = fs.Int("threads", 4, "")
		fa         = fs.Bool("flash-attn", false, "")
		mmproj     = fs.String("mmproj", "", "")
		noMmproj   = fs.Bool("no-mmproj", false, "")
		modelDraft = fs.String("model-draft", "", "")
		draftMax   = fs.Int("draft-max", 0, "")
		specType   = fs.String("spec-type", "", "")
		parallel   = fs.Int("parallel", 1, "")
		batch      = fs.Int("batch-size", 2048, "")
		cacheK     = fs.String("cache-type-k", "", "")
	)
	// llama.cpp-style --help exits 0 after printing usage.
	fs.Usage = func() {
		fmt.Println(`usage: llama-server [options]
  -m,    --model FNAME
  -c,    --ctx-size N
  -ngl,  --n-gpu-layers N
  -t,    --threads N
  -fa,   --flash-attn
  -np,   --parallel N
  -b,    --batch-size N
         --cache-type-k TYPE
         --mmproj FILE
         --no-mmproj
         --model-draft FILE
         --spec-draft-model FILE
         --draft-max N
         --spec-draft-n-max N
         --spec-type TYPE
         --host HOST
         --port PORT
         --api-key KEY
         --alias NAME`)
	}
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(1)
	}
	_ = parallel
	_ = batch
	_ = cacheK
	_ = fa
	_ = threads
	_ = ngl
	_ = ctxSize
	_ = mmproj
	_ = noMmproj
	_ = modelDraft
	_ = draftMax
	_ = specType

	// Manual scan for --help since flag package treats it as error by default.
	for _, a := range os.Args[1:] {
		if a == "--help" || a == "-h" {
			fs.Usage()
			os.Exit(0)
		}
	}
	if *version {
		fmt.Println("version: 9999 (deadbeef)")
		fmt.Println("built with fake-cc for openinfer tests")
		return
	}

	mux := http.NewServeMux()
	var (
		slotsMu    sync.Mutex
		activeReqs int
		decoded    int
	)
	check := func(w http.ResponseWriter, r *http.Request) bool {
		if *apiKey != "" && r.Header.Get("Authorization") != "Bearer "+*apiKey {
			w.WriteHeader(http.StatusUnauthorized)
			return false
		}
		return true
	}
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if !check(w, r) {
			return
		}
		name := *alias
		if name == "" {
			name = *model
		}
		json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{
			map[string]any{"id": name, "object": "model"}}})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if !check(w, r) {
			return
		}
		slotsMu.Lock()
		activeReqs++
		decoded = 0
		slotsMu.Unlock()
		defer func() {
			slotsMu.Lock()
			activeReqs--
			slotsMu.Unlock()
		}()
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		chunks := []string{
			`{"choices":[{"delta":{"role":"assistant"}}]}`,
			`{"choices":[{"delta":{"content":"Hello"}}]}`,
			`{"choices":[{"delta":{"content":" from fake llama-server"}}]}`,
			`{"choices":[{"delta":{}}],"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10},"timings":{"prompt_per_second":100.0,"predicted_per_second":42.0,"prompt_n":5,"predicted_n":5}}`,
		}
		for i, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			if fl != nil {
				fl.Flush()
			}
			slotsMu.Lock()
			decoded = i + 1
			slotsMu.Unlock()
			time.Sleep(10 * time.Millisecond)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	mux.HandleFunc("/slots", func(w http.ResponseWriter, r *http.Request) {
		if !check(w, r) {
			return
		}
		slotsMu.Lock()
		n, d := activeReqs, decoded
		slotsMu.Unlock()
		slots := []map[string]any{}
		for i := 0; i < n; i++ {
			slots = append(slots, map[string]any{
				"id": i, "id_task": 100 + i, "is_processing": true,
				"n_prompt_tokens": 5,
				"next_token":      []map[string]any{{"has_next_token": true, "n_decoded": d}},
			})
		}
		json.NewEncoder(w).Encode(slots)
	})

	addr := fmt.Sprintf("%s:%d", *host, *port)
	fmt.Fprintf(os.Stderr, "fake llama-server listening on %s\n", addr)
	srv := &http.Server{Addr: addr, Handler: mux}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bind failed: %v\n", err)
		os.Exit(1)
	}
	go func() {
		// Simulate a load crash after readiness when requested.
		if strings.Contains(*model, "crashme") {
			time.Sleep(500 * time.Millisecond)
			fmt.Fprintln(os.Stderr, "ggml_cuda: out of memory — simulated crash")
			os.Exit(1)
		}
	}()
	_ = srv.Serve(ln)
}
