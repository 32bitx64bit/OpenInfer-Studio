// Package chat persists conversations as message trees and streams
// completions from the selected model's managed llama-server instance.
package chat

import (
	"bufio"
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

type EventSink interface {
	Publish(event string, payload any)
}

// Conversation metadata.
type Conversation struct {
	ID        string          `json:"id"`
	Title     string          `json:"title"`
	ModelID   string          `json:"model_id"`
	System    string          `json:"system"`
	Params    json.RawMessage `json:"params"`
	Archived  bool            `json:"archived"`
	CreatedAt string          `json:"created_at"`
	UpdatedAt string          `json:"updated_at"`
}

// Message is one node of the conversation tree.
type Message struct {
	ID        string          `json:"id"`
	ConvID    string          `json:"conv_id"`
	ParentID  string          `json:"parent_id"`
	Role      string          `json:"role"`
	Content   string          `json:"content"`
	Reasoning string          `json:"reasoning,omitempty"`
	Stats     json.RawMessage `json:"stats,omitempty"`
	Error     string          `json:"error,omitempty"`
	CreatedAt string          `json:"created_at"`
}

// GenParams are per-request generation settings (validated bounds in API).
type GenParams struct {
	Temperature        *float64 `json:"temperature,omitempty"`
	TopP               *float64 `json:"top_p,omitempty"`
	TopK               *int     `json:"top_k,omitempty"`
	MinP               *float64 `json:"min_p,omitempty"`
	TypicalP           *float64 `json:"typical_p,omitempty"`
	RepeatPenalty      *float64 `json:"repeat_penalty,omitempty"`
	RepeatLastN        *int     `json:"repeat_last_n,omitempty"`
	FrequencyPenalty   *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty    *float64 `json:"presence_penalty,omitempty"`
	Seed               *int64   `json:"seed,omitempty"`
	MaxTokens          *int     `json:"max_tokens,omitempty"`
	Stop               []string `json:"stop,omitempty"`
	JSONSchema         string   `json:"json_schema,omitempty"`
	Grammar            string   `json:"grammar,omitempty"`
	ChatTemplateKwargs string   `json:"chat_template_kwargs,omitempty"`
}

// mergeGenParams applies explicit request settings over persisted conversation
// settings. Pointer fields preserve an intentional zero value (for example
// temperature 0) while omitted fields inherit the conversation default.
func mergeGenParams(saved, overrides GenParams) GenParams {
	out := saved
	if overrides.Temperature != nil {
		out.Temperature = overrides.Temperature
	}
	if overrides.TopP != nil {
		out.TopP = overrides.TopP
	}
	if overrides.TopK != nil {
		out.TopK = overrides.TopK
	}
	if overrides.MinP != nil {
		out.MinP = overrides.MinP
	}
	if overrides.TypicalP != nil {
		out.TypicalP = overrides.TypicalP
	}
	if overrides.RepeatPenalty != nil {
		out.RepeatPenalty = overrides.RepeatPenalty
	}
	if overrides.RepeatLastN != nil {
		out.RepeatLastN = overrides.RepeatLastN
	}
	if overrides.FrequencyPenalty != nil {
		out.FrequencyPenalty = overrides.FrequencyPenalty
	}
	if overrides.PresencePenalty != nil {
		out.PresencePenalty = overrides.PresencePenalty
	}
	if overrides.Seed != nil {
		out.Seed = overrides.Seed
	}
	if overrides.MaxTokens != nil {
		out.MaxTokens = overrides.MaxTokens
	}
	if overrides.Stop != nil {
		out.Stop = overrides.Stop
	}
	if overrides.JSONSchema != "" {
		out.JSONSchema = overrides.JSONSchema
	}
	if overrides.Grammar != "" {
		out.Grammar = overrides.Grammar
	}
	if overrides.ChatTemplateKwargs != "" {
		out.ChatTemplateKwargs = overrides.ChatTemplateKwargs
	}
	return out
}

// AudioInput is optional audio for a user turn (experimental llama.cpp path).
type AudioInput struct {
	Path   string `json:"path,omitempty"`   // local file from the desktop picker
	Data   string `json:"data,omitempty"`   // base64-encoded bytes (tests / alternate clients)
	Format string `json:"format,omitempty"` // wav, mp3, …
	Name   string `json:"name,omitempty"`
}

const maxAudioBytes = 25 << 20

var allowedAudioExt = map[string]bool{
	"wav": true, "mp3": true, "flac": true, "ogg": true, "m4a": true, "webm": true,
}

var now = func() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// EndpointProvider resolves model IDs to ready instance endpoints.
type EndpointProvider interface {
	EndpointFor(modelID string) (endpoint Endpoint, err error)
	Touch(modelID string)
}

// Endpoint mirrors instances.Endpoint without an import cycle.
type Endpoint struct {
	URL    string
	APIKey string
	Alias  string
}

type Service struct {
	db     *sql.DB
	events EventSink
	eps    EndpointProvider
	http   *http.Client

	// Active generations: conv ID → cancel.
	cancels map[string]context.CancelFunc

	streaming atomic.Bool
	attachDir string
}

func NewService(db *sql.DB, events EventSink, eps EndpointProvider) *Service {
	s := &Service{
		db: db, events: events, eps: eps,
		http:    &http.Client{Timeout: 0},
		cancels: map[string]context.CancelFunc{},
	}
	s.streaming.Store(true) // streaming is the default
	return s
}

// SetAttachDir sets where chat audio attachments are stored on disk.
func (s *Service) SetAttachDir(dir string) { s.attachDir = dir }

// SetStreaming toggles token streaming. When disabled, requests use
// stream:false and the full response is emitted as one event.
func (s *Service) SetStreaming(on bool) { s.streaming.Store(on) }

func (s *Service) CreateConversation(modelID, title, system string) (*Conversation, error) {
	id := uuid.NewString()
	if title == "" {
		title = "New chat"
	}
	_, err := s.db.Exec(`INSERT INTO conversations(id,title,model_id,system,created_at,updated_at)
		VALUES (?,?,?,?,?,?)`, id, title, modelID, system, now(), now())
	if err != nil {
		return nil, err
	}
	return &Conversation{ID: id, Title: title, ModelID: modelID, System: system,
		Params: json.RawMessage(`{}`), CreatedAt: now(), UpdatedAt: now()}, nil
}

func (s *Service) ListConversations(includeArchived bool) ([]Conversation, error) {
	q := `SELECT id,title,model_id,system,params_json,archived,created_at,updated_at FROM conversations`
	if !includeArchived {
		q += ` WHERE archived = 0`
	}
	q += ` ORDER BY updated_at DESC`
	rows, err := s.db.Query(q)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Conversation
	for rows.Next() {
		var c Conversation
		var params string
		var arch int
		if err := rows.Scan(&c.ID, &c.Title, &c.ModelID, &c.System, &params, &arch, &c.CreatedAt, &c.UpdatedAt); err != nil {
			return nil, err
		}
		c.Params = json.RawMessage(params)
		c.Archived = arch == 1
		out = append(out, c)
	}
	return out, nil
}

func (s *Service) GetConversation(id string) (*Conversation, error) {
	var c Conversation
	var params string
	var arch int
	err := s.db.QueryRow(`SELECT id,title,model_id,system,params_json,archived,created_at,updated_at FROM conversations WHERE id=?`, id).
		Scan(&c.ID, &c.Title, &c.ModelID, &c.System, &params, &arch, &c.CreatedAt, &c.UpdatedAt)
	if err != nil {
		return nil, err
	}
	c.Params = json.RawMessage(params)
	c.Archived = arch == 1
	return &c, nil
}

func (s *Service) RenameConversation(id, title string) error {
	_, err := s.db.Exec(`UPDATE conversations SET title=?, updated_at=? WHERE id=?`, title, now(), id)
	return err
}

func (s *Service) SetConversationModel(id, modelID string) error {
	_, err := s.db.Exec(`UPDATE conversations SET model_id=?, updated_at=? WHERE id=?`, modelID, now(), id)
	return err
}

func (s *Service) ArchiveConversation(id string, archived bool) error {
	a := 0
	if archived {
		a = 1
	}
	_, err := s.db.Exec(`UPDATE conversations SET archived=?, updated_at=? WHERE id=?`, a, now(), id)
	return err
}

func (s *Service) DeleteConversation(id string) error {
	s.Stop(id)
	_, err := s.db.Exec(`DELETE FROM conversations WHERE id=?`, id)
	return err
}

func (s *Service) SetSystemPrompt(id, system string) error {
	_, err := s.db.Exec(`UPDATE conversations SET system=?, updated_at=? WHERE id=?`, system, now(), id)
	return err
}

func (s *Service) SetParams(id string, params json.RawMessage) error {
	var v any
	if err := json.Unmarshal(params, &v); err != nil {
		return fmt.Errorf("invalid params JSON: %w", err)
	}
	_, err := s.db.Exec(`UPDATE conversations SET params_json=?, updated_at=? WHERE id=?`, string(params), now(), id)
	return err
}

func (s *Service) Messages(convID string) ([]Message, error) {
	rows, err := s.db.Query(`SELECT id,conv_id,parent_id,role,content,reasoning,stats_json,error,created_at
		FROM conversation_messages WHERE conv_id=? ORDER BY created_at`, convID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Message
	for rows.Next() {
		var m Message
		var stats string
		if err := rows.Scan(&m.ID, &m.ConvID, &m.ParentID, &m.Role, &m.Content, &m.Reasoning, &stats, &m.Error, &m.CreatedAt); err != nil {
			return nil, err
		}
		m.Stats = json.RawMessage(stats)
		out = append(out, m)
	}
	return out, nil
}

func (s *Service) addMessage(convID, parentID, role, content string) (*Message, error) {
	m := &Message{ID: uuid.NewString(), ConvID: convID, ParentID: parentID,
		Role: role, Content: content, CreatedAt: now()}
	_, err := s.db.Exec(`INSERT INTO conversation_messages(id,conv_id,parent_id,role,content,created_at)
		VALUES (?,?,?,?,?,?)`, m.ID, convID, parentID, role, content, m.CreatedAt)
	if err != nil {
		return nil, err
	}
	_, _ = s.db.Exec(`UPDATE conversations SET updated_at=? WHERE id=?`, now(), convID)
	return m, nil
}

// chain walks parent links from leaf to root, returning root→leaf order.
func (s *Service) chain(leafID string) ([]Message, error) {
	var rev []Message
	cur := leafID
	seen := map[string]bool{}
	for cur != "" {
		if seen[cur] {
			return nil, errors.New("conversation tree contains a cycle")
		}
		seen[cur] = true
		var m Message
		var stats string
		err := s.db.QueryRow(`SELECT id,conv_id,parent_id,role,content,reasoning,stats_json,error,created_at
			FROM conversation_messages WHERE id=?`, cur).
			Scan(&m.ID, &m.ConvID, &m.ParentID, &m.Role, &m.Content, &m.Reasoning, &stats, &m.Error, &m.CreatedAt)
		if err != nil {
			return nil, err
		}
		m.Stats = json.RawMessage(stats)
		rev = append(rev, m)
		cur = m.ParentID
	}
	out := make([]Message, len(rev))
	for i, m := range rev {
		out[len(rev)-1-i] = m
	}
	return out, nil
}

// latestLeaf returns the deepest descendant of the conversation root.
func (s *Service) latestLeaf(convID string) (string, error) {
	var count int
	if err := s.db.QueryRow(`SELECT COUNT(1) FROM conversation_messages WHERE conv_id=?`, convID).Scan(&count); err != nil {
		return "", err
	}
	if count == 0 {
		return "", nil
	}
	// Leaves = messages with no children; pick newest.
	var leaf string
	err := s.db.QueryRow(`SELECT m.id FROM conversation_messages m
		WHERE m.conv_id=? AND NOT EXISTS
		  (SELECT 1 FROM conversation_messages c WHERE c.parent_id = m.id)
		ORDER BY m.created_at DESC LIMIT 1`, convID).Scan(&leaf)
	return leaf, err
}

// TokenEvent is streamed to the UI.
type TokenEvent struct {
	ConvID            string `json:"conv_id"`
	MessageID         string `json:"message_id"`
	Delta             string `json:"delta,omitempty"`
	Reasoning         string `json:"reasoning_delta,omitempty"`
	Snapshot          string `json:"snapshot,omitempty"`           // full content replace (diffusion live canvas)
	ReasoningSnapshot string `json:"reasoning_snapshot,omitempty"` // full reasoning replace
	Replace           bool   `json:"replace,omitempty"`            // when true, Snapshot/ReasoningSnapshot replace
	Done              bool   `json:"done"`
	Error             string `json:"error,omitempty"`
	Stats             any    `json:"stats,omitempty"`
}

// Generate streams an assistant reply for the chain ending at parentID
// ("" = continue from the latest leaf). The user message is persisted first
// when content is non-empty or audio is attached. Events: chat.token.
func (s *Service) Generate(ctx context.Context, convID, parentID, userContent string, params GenParams, audio *AudioInput) (string, error) {
	var conv Conversation
	var paramsJSON string
	var arch int
	err := s.db.QueryRow(`SELECT id,title,model_id,system,params_json,archived FROM conversations WHERE id=?`, convID).
		Scan(&conv.ID, &conv.Title, &conv.ModelID, &conv.System, &paramsJSON, &arch)
	if err != nil {
		return "", fmt.Errorf("conversation not found: %w", err)
	}
	var savedParams GenParams
	if strings.TrimSpace(paramsJSON) != "" {
		if err := json.Unmarshal([]byte(paramsJSON), &savedParams); err != nil {
			return "", fmt.Errorf("invalid saved generation parameters: %w", err)
		}
	}
	params = mergeGenParams(savedParams, params)

	// Branch point: explicit parent (edit/regenerate) or latest leaf.
	if parentID == "" && (userContent != "" || audio != nil) {
		parentID, _ = s.latestLeaf(convID)
	}
	if userContent != "" || audio != nil {
		display := userContent
		if audio != nil && strings.TrimSpace(display) == "" {
			display = "[Audio attached]"
		} else if audio != nil {
			display = userContent + "\n[Audio attached]"
		}
		um, err := s.addMessage(convID, parentID, "user", display)
		if err != nil {
			return "", err
		}
		if audio != nil {
			if err := s.persistAudio(um.ID, audio); err != nil {
				return "", err
			}
		}
		parentID = um.ID
		s.events.Publish("chat.message", map[string]any{"conv_id": convID, "message": um})
	}
	if parentID == "" {
		return "", errors.New("nothing to respond to")
	}

	chain, err := s.chain(parentID)
	if err != nil {
		return "", err
	}

	ep, err := s.eps.EndpointFor(conv.ModelID)
	if err != nil {
		return "", err
	}

	// Placeholder assistant message; content accumulates.
	am, err := s.addMessage(convID, parentID, "assistant", "")
	if err != nil {
		return "", err
	}

	// The stream runs in the background after the HTTP handler returns 202;
	// it must not inherit the request context, which is canceled on return.
	ctx, cancel := context.WithCancel(context.Background())
	s.cancels[convID] = cancel

	go func() {
		defer cancel()
		defer delete(s.cancels, convID)
		stats, err := s.stream(ctx, conv, ep, chain, am.ID, params)
		if err != nil {
			_, _ = s.db.Exec(`UPDATE conversation_messages SET error=? WHERE id=?`, err.Error(), am.ID)
			s.events.Publish("chat.token", TokenEvent{ConvID: convID, MessageID: am.ID, Done: true, Error: err.Error()})
			return
		}
		statsJSON, _ := json.Marshal(stats)
		_, _ = s.db.Exec(`UPDATE conversation_messages SET stats_json=? WHERE id=?`, string(statsJSON), am.ID)
		s.events.Publish("chat.token", TokenEvent{ConvID: convID, MessageID: am.ID, Done: true, Stats: stats})
	}()
	return am.ID, nil
}

// Stop cancels an in-flight generation.
func (s *Service) Stop(convID string) {
	if c, ok := s.cancels[convID]; ok {
		c()
	}
}

// Stats are extracted from llama-server timings.
type Stats struct {
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	TimeToFirstToken float64 `json:"ttft_seconds"`
	TokensPerSecond  float64 `json:"tokens_per_second"`
	PromptPerSecond  float64 `json:"prompt_per_second"`
	TotalSeconds     float64 `json:"total_seconds"`
}

// stream posts the chat completion request and consumes SSE.
func (s *Service) stream(ctx context.Context, conv Conversation, ep Endpoint,
	chain []Message, assistantID string, params GenParams) (*Stats, error) {

	msgs, err := s.buildOAIMessages(conv, chain)
	if err != nil {
		return nil, err
	}

	stream := s.streaming.Load()
	body := map[string]any{
		"model":             ep.Alias,
		"messages":          msgs,
		"stream":            stream,
		"timings_per_token": true,
	}
	if stream {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	applyParams(body, params)

	payload, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ep.URL+"/v1/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+ep.APIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	start := time.Now()
	resp, err := s.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		return nil, fmt.Errorf("server returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}

	s.eps.Touch(conv.ModelID)

	if !stream {
		return s.nonStreamingResponse(resp, conv.ID, assistantID, start)
	}

	var content strings.Builder
	var reasoning strings.Builder
	stats := &Stats{}
	firstToken := false

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64<<10), 4<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk struct {
			Type      string `json:"type"`
			Text      string `json:"text"`
			Reasoning string `json:"reasoning"`
			Choices   []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage"`
			Timings *struct {
				PromptPerSecond    float64 `json:"prompt_per_second"`
				PredictedPerSecond float64 `json:"predicted_per_second"`
				PromptN            int     `json:"prompt_n"`
				PredictedN         int     `json:"predicted_n"`
			} `json:"timings"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue // tolerate non-JSON SSE lines
		}
		// Diffusion live canvas / commit snapshots replace the bubble in place.
		if (chunk.Type == "diffusion_frame" || chunk.Type == "diffusion_snapshot") && chunk.Text != "" {
			if !firstToken {
				stats.TimeToFirstToken = time.Since(start).Seconds()
				firstToken = true
			}
			content.Reset()
			content.WriteString(chunk.Text)
			reasoning.Reset()
			reasoning.WriteString(chunk.Reasoning)
			s.events.Publish("chat.token", TokenEvent{
				ConvID:            conv.ID,
				MessageID:         assistantID,
				Snapshot:          chunk.Text,
				ReasoningSnapshot: chunk.Reasoning,
				Replace:           true,
			})
			continue
		}
		for _, ch := range chunk.Choices {
			if ch.Delta.Content != "" {
				if !firstToken {
					stats.TimeToFirstToken = time.Since(start).Seconds()
					firstToken = true
				}
				content.WriteString(ch.Delta.Content)
				s.events.Publish("chat.token", TokenEvent{ConvID: conv.ID, MessageID: assistantID, Delta: ch.Delta.Content})
			}
			if ch.Delta.ReasoningContent != "" {
				reasoning.WriteString(ch.Delta.ReasoningContent)
				s.events.Publish("chat.token", TokenEvent{ConvID: conv.ID, MessageID: assistantID, Reasoning: ch.Delta.ReasoningContent})
			}
		}
		if chunk.Usage != nil {
			stats.PromptTokens = chunk.Usage.PromptTokens
			stats.CompletionTokens = chunk.Usage.CompletionTokens
			stats.TotalTokens = chunk.Usage.TotalTokens
		}
		if chunk.Timings != nil {
			stats.PromptPerSecond = chunk.Timings.PromptPerSecond
			stats.TokensPerSecond = chunk.Timings.PredictedPerSecond
			if chunk.Timings.PredictedN > 0 {
				stats.CompletionTokens = chunk.Timings.PredictedN
			}
			if chunk.Timings.PromptN > 0 {
				stats.PromptTokens = chunk.Timings.PromptN
			}
		}
	}
	if err := sc.Err(); err != nil && !errors.Is(err, context.Canceled) {
		// Persist partial content on stream errors.
		_, _ = s.db.Exec(`UPDATE conversation_messages SET content=?, reasoning=? WHERE id=?`,
			content.String(), reasoning.String(), assistantID)
		return stats, fmt.Errorf("stream interrupted: %w", err)
	}

	stats.TotalSeconds = time.Since(start).Seconds()
	if stats.TokensPerSecond == 0 && stats.CompletionTokens > 0 && stats.TotalSeconds > 0 {
		stats.TokensPerSecond = float64(stats.CompletionTokens) / stats.TotalSeconds
	}
	_, err = s.db.Exec(`UPDATE conversation_messages SET content=?, reasoning=? WHERE id=?`,
		content.String(), reasoning.String(), assistantID)
	return stats, err
}

// nonStreamingResponse handles a stream:false completion: the whole reply is
// parsed at once and emitted as a single delta before the done event.
func (s *Service) nonStreamingResponse(resp *http.Response, convID, assistantID string, start time.Time) (*Stats, error) {
	body, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if err != nil {
		return nil, err
	}
	var full struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
		Timings *struct {
			PromptPerSecond    float64 `json:"prompt_per_second"`
			PredictedPerSecond float64 `json:"predicted_per_second"`
			PromptN            int     `json:"prompt_n"`
			PredictedN         int     `json:"predicted_n"`
		} `json:"timings"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &full); err != nil {
		return nil, fmt.Errorf("invalid response: %w", err)
	}
	if full.Error != nil {
		return nil, fmt.Errorf("server error: %s", full.Error.Message)
	}
	var content, reasoning string
	if len(full.Choices) > 0 {
		content = full.Choices[0].Message.Content
		reasoning = full.Choices[0].Message.ReasoningContent
	}
	stats := &Stats{TotalSeconds: time.Since(start).Seconds()}
	if full.Usage != nil {
		stats.PromptTokens = full.Usage.PromptTokens
		stats.CompletionTokens = full.Usage.CompletionTokens
		stats.TotalTokens = full.Usage.TotalTokens
	}
	if full.Timings != nil {
		stats.PromptPerSecond = full.Timings.PromptPerSecond
		stats.TokensPerSecond = full.Timings.PredictedPerSecond
	}
	if stats.TokensPerSecond == 0 && stats.CompletionTokens > 0 && stats.TotalSeconds > 0 {
		stats.TokensPerSecond = float64(stats.CompletionTokens) / stats.TotalSeconds
	}

	_, err = s.db.Exec(`UPDATE conversation_messages SET content=?, reasoning=? WHERE id=?`,
		content, reasoning, assistantID)
	if err != nil {
		return stats, err
	}
	// One consolidated event so the UI updates identically to streaming.
	s.events.Publish("chat.token", TokenEvent{ConvID: convID, MessageID: assistantID, Delta: content, Reasoning: reasoning})
	return stats, nil
}

type oaiMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

// buildOAIMessages constructs OpenAI-compatible messages, expanding audio
// attachments into input_audio content parts when present.
func (s *Service) buildOAIMessages(conv Conversation, chain []Message) ([]oaiMessage, error) {
	msgs := []oaiMessage{}
	if conv.System != "" {
		msgs = append(msgs, oaiMessage{Role: "system", Content: conv.System})
	}
	for _, m := range chain {
		if m.Error != "" || (m.Role == "assistant" && m.Content == "") {
			continue
		}
		content, err := s.messageContent(m)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, oaiMessage{Role: m.Role, Content: content})
	}
	return msgs, nil
}

func (s *Service) messageContent(m Message) (any, error) {
	att, err := s.audioAttachment(m.ID)
	if err != nil {
		return nil, err
	}
	if att == nil {
		return m.Content, nil
	}
	data, err := os.ReadFile(att.path)
	if err != nil {
		return nil, fmt.Errorf("reading audio attachment: %w", err)
	}
	text := strings.TrimSuffix(m.Content, "\n[Audio attached]")
	text = strings.TrimSpace(text)
	if text == "" || text == "[Audio attached]" {
		text = "Transcribe or respond to this audio."
	}
	parts := []map[string]any{
		{"type": "text", "text": text},
		{"type": "input_audio", "input_audio": map[string]any{
			"data":   base64.StdEncoding.EncodeToString(data),
			"format": att.format,
		}},
	}
	return parts, nil
}

type audioAtt struct {
	path   string
	format string
}

func (s *Service) audioAttachment(messageID string) (*audioAtt, error) {
	var path, mime, name string
	err := s.db.QueryRow(`SELECT path, mime, name FROM attachments WHERE message_id=? AND kind='audio' LIMIT 1`, messageID).
		Scan(&path, &mime, &name)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	format := strings.TrimPrefix(strings.ToLower(filepath.Ext(path)), ".")
	if format == "" {
		format = strings.TrimPrefix(strings.ToLower(filepath.Ext(name)), ".")
	}
	if format == "" && strings.Contains(mime, "/") {
		format = strings.TrimPrefix(mime, "audio/")
	}
	if format == "" {
		format = "wav"
	}
	return &audioAtt{path: path, format: format}, nil
}

func (s *Service) persistAudio(messageID string, audio *AudioInput) error {
	if audio == nil {
		return nil
	}
	if s.attachDir == "" {
		return errors.New("attachments directory not configured")
	}
	raw, format, name, err := resolveAudioBytes(audio)
	if err != nil {
		return err
	}
	if len(raw) == 0 {
		return errors.New("empty audio attachment")
	}
	if len(raw) > maxAudioBytes {
		return fmt.Errorf("audio attachment exceeds %d byte limit", maxAudioBytes)
	}
	if err := os.MkdirAll(s.attachDir, 0o755); err != nil {
		return err
	}
	id := uuid.NewString()
	dest := filepath.Join(s.attachDir, id+"."+format)
	if err := os.WriteFile(dest, raw, 0o600); err != nil {
		return err
	}
	mime := "audio/" + format
	_, err = s.db.Exec(`INSERT INTO attachments(id,message_id,kind,path,mime,name,size_bytes) VALUES (?,?,?,?,?,?,?)`,
		id, messageID, "audio", dest, mime, name, len(raw))
	return err
}

func resolveAudioBytes(audio *AudioInput) (raw []byte, format, name string, err error) {
	format = strings.ToLower(strings.TrimPrefix(audio.Format, "."))
	name = audio.Name
	if audio.Data != "" {
		raw, err = base64.StdEncoding.DecodeString(audio.Data)
		if err != nil {
			return nil, "", "", fmt.Errorf("invalid audio base64: %w", err)
		}
		if format == "" {
			format = "wav"
		}
		if name == "" {
			name = "audio." + format
		}
	} else if audio.Path != "" {
		raw, err = os.ReadFile(audio.Path)
		if err != nil {
			return nil, "", "", fmt.Errorf("reading audio file: %w", err)
		}
		if name == "" {
			name = filepath.Base(audio.Path)
		}
		if format == "" {
			format = strings.TrimPrefix(strings.ToLower(filepath.Ext(audio.Path)), ".")
		}
	} else {
		return nil, "", "", errors.New("audio requires path or data")
	}
	if format == "" {
		format = "wav"
	}
	if !allowedAudioExt[format] {
		return nil, "", "", fmt.Errorf("unsupported audio format %q (use wav/mp3/flac/ogg/m4a/webm)", format)
	}
	return raw, format, name, nil
}

// applyParams merges validated generation params into the request body.
func applyParams(body map[string]any, p GenParams) {
	if p.Temperature != nil {
		body["temperature"] = *p.Temperature
	}
	if p.TopP != nil {
		body["top_p"] = *p.TopP
	}
	if p.TopK != nil {
		body["top_k"] = *p.TopK
	}
	if p.MinP != nil {
		body["min_p"] = *p.MinP
	}
	if p.TypicalP != nil {
		body["typical_p"] = *p.TypicalP
	}
	if p.RepeatPenalty != nil {
		body["repeat_penalty"] = *p.RepeatPenalty
	}
	if p.RepeatLastN != nil {
		body["repeat_last_n"] = *p.RepeatLastN
	}
	if p.FrequencyPenalty != nil {
		body["frequency_penalty"] = *p.FrequencyPenalty
	}
	if p.PresencePenalty != nil {
		body["presence_penalty"] = *p.PresencePenalty
	}
	if p.Seed != nil {
		body["seed"] = *p.Seed
	}
	if p.MaxTokens != nil {
		body["max_tokens"] = *p.MaxTokens
	}
	if len(p.Stop) > 0 {
		body["stop"] = p.Stop
	}
	if p.JSONSchema != "" {
		body["response_format"] = map[string]any{
			"type":        "json_schema",
			"json_schema": map[string]any{"name": "output", "schema": json.RawMessage(p.JSONSchema)},
		}
	}
	if p.Grammar != "" {
		body["grammar"] = p.Grammar
	}
	if p.ChatTemplateKwargs != "" {
		body["chat_template_kwargs"] = json.RawMessage(p.ChatTemplateKwargs)
	}
}
