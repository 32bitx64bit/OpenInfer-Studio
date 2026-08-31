package instances

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Activity is a live snapshot of llama-server /slots for one loaded model.
type Activity struct {
	ModelID        string         `json:"model_id"`
	Busy           bool           `json:"busy"`
	ActiveRequests int            `json:"active_requests"`
	DecodedTotal   int            `json:"decoded_total"`
	TokensPerSec   float64        `json:"tokens_per_second"`
	Slots          []SlotActivity `json:"slots"`
	SampledAt      time.Time      `json:"sampled_at"`
}

// SlotActivity is one in-flight request on the llama-server instance.
type SlotActivity struct {
	ID           int     `json:"id"`
	TaskID       int     `json:"task_id"`
	Processing   bool    `json:"processing"`
	PromptTokens int     `json:"prompt_tokens"`
	Decoded      int     `json:"decoded"`
	TokensPerSec float64 `json:"tokens_per_second"`
}

// llamaSlot mirrors the /slots response shape of llama-server. Missing
// fields decode as zero, which keeps us compatible across versions.
// n_decoded lives under next_token in current llama.cpp (object historically,
// array of one object on recent builds). Older fakes / builds put it at the
// top level.
type llamaSlot struct {
	ID           int  `json:"id"`
	TaskID       int  `json:"id_task"`
	Processing   bool `json:"is_processing"`
	PromptTokens int  `json:"n_prompt_tokens"`
	Decoded      int  `json:"n_decoded"`
}

type llamaNextToken struct {
	Decoded int `json:"n_decoded"`
}

func (s *llamaSlot) UnmarshalJSON(data []byte) error {
	var raw struct {
		ID           int             `json:"id"`
		TaskID       int             `json:"id_task"`
		Processing   bool            `json:"is_processing"`
		PromptTokens int             `json:"n_prompt_tokens"`
		Decoded      int             `json:"n_decoded"`
		NextToken    json.RawMessage `json:"next_token"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	s.ID = raw.ID
	s.TaskID = raw.TaskID
	s.Processing = raw.Processing
	s.PromptTokens = raw.PromptTokens
	s.Decoded = raw.Decoded
	if n, ok := decodedFromNextToken(raw.NextToken); ok {
		s.Decoded = n
	}
	return nil
}

func decodedFromNextToken(raw json.RawMessage) (int, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return 0, false
	}
	var obj llamaNextToken
	if err := json.Unmarshal(raw, &obj); err == nil {
		return obj.Decoded, true
	}
	var arr []llamaNextToken
	if err := json.Unmarshal(raw, &arr); err == nil && len(arr) > 0 {
		return arr[0].Decoded, true
	}
	return 0, false
}

// activityPollInterval is a var so tests can shorten it.
var activityPollInterval = time.Second

// Activity returns the last sampled activity snapshot for a model.
func (m *Manager) Activity(modelID string) (*Activity, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if li, ok := m.instances[modelID]; ok && li.lastActivity != nil {
		cp := *li.lastActivity
		return &cp, true
	}
	return nil, false
}

// monitorActivity polls /slots and drives ready ⇄ busy transitions until
// ctx is canceled or the server does not expose /slots.
func (m *Manager) monitorActivity(ctx context.Context, li *liveInstance) {
	client := &http.Client{Timeout: 3 * time.Second}
	url := fmt.Sprintf("http://127.0.0.1:%d/slots", li.Port)
	prev := map[int]int{} // slot ID → last decoded count
	prevAt := time.Time{}

	t := time.NewTicker(activityPollInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return
		}
		req.Header.Set("Authorization", "Bearer "+li.apiKey)
		resp, err := client.Do(req)
		if err != nil {
			continue // transient; process exit will cancel ctx
		}
		if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusForbidden ||
			resp.StatusCode == http.StatusMethodNotAllowed {
			resp.Body.Close()
			return // /slots unsupported or disabled; stop monitoring
		}
		var slots []llamaSlot
		if resp.StatusCode == http.StatusOK {
			err = json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&slots)
		} else {
			err = fmt.Errorf("slots returned %d", resp.StatusCode)
		}
		resp.Body.Close()
		if err != nil {
			continue
		}

		now := time.Now()
		act := Activity{
			ModelID:   li.ModelID,
			Slots:     make([]SlotActivity, 0, len(slots)),
			SampledAt: now.UTC(),
		}
		elapsed := now.Sub(prevAt).Seconds()
		for _, s := range slots {
			sa := SlotActivity{
				ID: s.ID, TaskID: s.TaskID, Processing: s.Processing,
				PromptTokens: s.PromptTokens, Decoded: s.Decoded,
			}
			if s.Processing {
				act.Busy = true
				act.ActiveRequests++
				act.DecodedTotal += s.Decoded
				if elapsed > 0 && prevAt != (time.Time{}) {
					if d := s.Decoded - prev[s.ID]; d > 0 {
						sa.TokensPerSec = float64(d) / elapsed
						act.TokensPerSec += sa.TokensPerSec
					}
				}
			}
			prev[s.ID] = s.Decoded
			act.Slots = append(act.Slots, sa)
		}
		prevAt = now

		m.mu.Lock()
		li.lastActivity = &act
		st := li.State
		m.mu.Unlock()
		m.events.Publish("instance.activity", act)

		// Drive busy transitions without touching load/unload states.
		if act.Busy && st == StateReady {
			m.setState(li, StateBusy)
			m.emit(li)
		} else if !act.Busy && st == StateBusy {
			m.setState(li, StateReady)
			m.emit(li)
		}
	}
}

// streamLogs publishes appended instance-log lines as instance.log events.
func (m *Manager) streamLogs(ctx context.Context, li *liveInstance) {
	lines := make(chan string, 128)
	go func() { _ = m.scanLogLines(ctx, li, lines) }()

	var batch []string
	flush := time.NewTicker(300 * time.Millisecond)
	defer flush.Stop()
	send := func() {
		if len(batch) == 0 {
			return
		}
		m.events.Publish("instance.log", map[string]any{
			"model_id": li.ModelID, "lines": batch,
		})
		batch = nil
	}
	for {
		select {
		case <-ctx.Done():
			send()
			return
		case line := <-lines:
			batch = append(batch, line)
			if len(batch) >= 50 {
				send()
			}
		case <-flush.C:
			send()
		}
	}
}
