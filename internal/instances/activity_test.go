package instances

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

type captureSink struct {
	mu     sync.Mutex
	events []string
	acts   []Activity
	states []string
}

func (c *captureSink) Publish(event string, payload any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, event)
	switch event {
	case "instance.activity":
		if a, ok := payload.(Activity); ok {
			c.acts = append(c.acts, a)
		}
	case "instance.state_changed":
		if m, ok := payload.(map[string]any); ok {
			c.states = append(c.states, fmt.Sprint(m["state"]))
		}
	}
}

func (c *captureSink) lastActivity() (Activity, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.acts) == 0 {
		return Activity{}, false
	}
	return c.acts[len(c.acts)-1], true
}

func (c *captureSink) hasState(s string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, st := range c.states {
		if st == s {
			return true
		}
	}
	return false
}

func TestMonitorActivityBusyTransitions(t *testing.T) {
	old := activityPollInterval
	activityPollInterval = 30 * time.Millisecond
	defer func() { activityPollInterval = old }()

	var mu sync.Mutex
	processing := true
	decoded := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/slots" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		mu.Lock()
		p, d := processing, decoded
		if p {
			decoded += 3
		}
		mu.Unlock()
		json.NewEncoder(w).Encode([]map[string]any{
			{"id": 0, "id_task": 7, "is_processing": p, "n_decoded": d, "n_prompt_tokens": 5},
		})
	}))
	defer srv.Close()

	port := 0
	if _, err := fmt.Sscanf(strings.TrimPrefix(srv.URL, "http://127.0.0.1:"), "%d", &port); err != nil {
		t.Fatalf("parse port: %v", err)
	}

	sink := &captureSink{}
	m := &Manager{events: sink, log: slog.Default()}
	li := &liveInstance{
		Instance: Instance{ModelID: "m1", Port: port, State: StateReady},
		apiKey:   "k",
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.monitorActivity(ctx, li)

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if a, ok := sink.lastActivity(); ok && a.Busy && a.DecodedTotal > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	act, ok := sink.lastActivity()
	if !ok {
		t.Fatal("no activity events published")
	}
	if !act.Busy || act.ActiveRequests != 1 {
		t.Fatalf("expected busy with 1 active request, got %+v", act)
	}
	if act.DecodedTotal == 0 || act.Slots[0].Decoded == 0 {
		t.Fatalf("expected token counts, got %+v", act)
	}
	if !sink.hasState(StateBusy) {
		t.Fatal("expected ready → busy state transition")
	}
	if act.TokensPerSec <= 0 {
		t.Fatalf("expected positive tok/s, got %+v", act)
	}

	mu.Lock()
	processing = false
	mu.Unlock()
	deadline = time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if a, ok := sink.lastActivity(); ok && !a.Busy {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	act, _ = sink.lastActivity()
	if act.Busy || act.ActiveRequests != 0 {
		t.Fatalf("expected idle after completion, got %+v", act)
	}
	liState := func() string {
		m.mu.Lock()
		defer m.mu.Unlock()
		return li.State
	}()
	if liState != StateReady {
		t.Fatalf("expected state ready after completion, got %s", liState)
	}
}

func TestParseLlamaSlotDecoded(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want int
	}{
		{
			name: "top-level legacy",
			raw:  `{"id":0,"id_task":7,"is_processing":true,"n_decoded":12,"n_prompt_tokens":5}`,
			want: 12,
		},
		{
			name: "next_token object",
			raw:  `{"id":0,"id_task":7,"is_processing":true,"n_prompt_tokens":5,"next_token":{"has_next_token":true,"n_decoded":34}}`,
			want: 34,
		},
		{
			name: "next_token array",
			raw:  `{"id":0,"id_task":7,"is_processing":true,"n_prompt_tokens":5,"next_token":[{"has_next_token":true,"n_decoded":56}]}`,
			want: 56,
		},
		{
			name: "nested wins over stale top-level zero",
			raw:  `{"id":0,"is_processing":true,"n_decoded":0,"next_token":{"n_decoded":9}}`,
			want: 9,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var s llamaSlot
			if err := json.Unmarshal([]byte(tc.raw), &s); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if s.Decoded != tc.want {
				t.Fatalf("decoded=%d want %d (processing=%v prompt=%d)", s.Decoded, tc.want, s.Processing, s.PromptTokens)
			}
			if !s.Processing {
				t.Fatal("expected is_processing")
			}
		})
	}
}

func TestMonitorActivityNestedNextToken(t *testing.T) {
	old := activityPollInterval
	activityPollInterval = 30 * time.Millisecond
	defer func() { activityPollInterval = old }()

	var mu sync.Mutex
	decoded := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/slots" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		mu.Lock()
		d := decoded
		decoded += 4
		mu.Unlock()
		json.NewEncoder(w).Encode([]map[string]any{
			{
				"id": 0, "id_task": 7, "is_processing": true,
				"n_prompt_tokens": 8,
				"next_token":      []map[string]any{{"has_next_token": true, "n_decoded": d}},
			},
		})
	}))
	defer srv.Close()

	port := 0
	if _, err := fmt.Sscanf(strings.TrimPrefix(srv.URL, "http://127.0.0.1:"), "%d", &port); err != nil {
		t.Fatalf("parse port: %v", err)
	}

	sink := &captureSink{}
	m := &Manager{events: sink, log: slog.Default()}
	li := &liveInstance{
		Instance: Instance{ModelID: "m1", Port: port, State: StateReady},
		apiKey:   "k",
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.monitorActivity(ctx, li)

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if a, ok := sink.lastActivity(); ok && a.Busy && a.DecodedTotal > 0 && a.TokensPerSec > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	act, ok := sink.lastActivity()
	if !ok {
		t.Fatal("no activity events published")
	}
	if !act.Busy || act.DecodedTotal == 0 {
		t.Fatalf("expected nested n_decoded token counts, got %+v", act)
	}
	if act.TokensPerSec <= 0 {
		t.Fatalf("expected positive tok/s from nested n_decoded, got %+v", act)
	}
}

func TestMonitorActivityGivesUpWithoutSlots(t *testing.T) {
	old := activityPollInterval
	activityPollInterval = 20 * time.Millisecond
	defer func() { activityPollInterval = old }()

	srv := httptest.NewServer(http.NotFoundHandler())
	defer srv.Close()

	port := 0
	if _, err := fmt.Sscanf(strings.TrimPrefix(srv.URL, "http://127.0.0.1:"), "%d", &port); err != nil {
		t.Fatalf("parse port: %v", err)
	}

	sink := &captureSink{}
	m := &Manager{events: sink, log: slog.Default()}
	li := &liveInstance{
		Instance: Instance{ModelID: "m1", Port: port, State: StateReady},
		apiKey:   "k",
	}
	done := make(chan struct{})
	go func() {
		m.monitorActivity(context.Background(), li)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("monitor did not exit on 404")
	}
	if len(sink.events) != 0 {
		t.Fatalf("expected no events, got %v", sink.events)
	}
}
