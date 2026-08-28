package chat

import (
	"io"
	"log/slog"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/migrations"
)

type nullSink struct{}

func (nullSink) Publish(string, any) {}

func testService(t *testing.T) *Service {
	t.Helper()
	db, err := database.Open(t.TempDir(), migrations.FS)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	_ = slog.New(slog.NewTextHandler(io.Discard, nil))
	return NewService(db.DB, nullSink{}, nil)
}

func TestConversationCRUD(t *testing.T) {
	s := testService(t)
	c, err := s.CreateConversation("model-1", "Test chat", "You are helpful.")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.RenameConversation(c.ID, "Renamed"); err != nil {
		t.Fatal(err)
	}
	if err := s.ArchiveConversation(c.ID, true); err != nil {
		t.Fatal(err)
	}
	list, _ := s.ListConversations(false)
	if len(list) != 0 {
		t.Error("archived conversation should be hidden by default")
	}
	list, _ = s.ListConversations(true)
	if len(list) != 1 || list[0].Title != "Renamed" {
		t.Errorf("unexpected list: %+v", list)
	}
	if err := s.DeleteConversation(c.ID); err != nil {
		t.Fatal(err)
	}
	list, _ = s.ListConversations(true)
	if len(list) != 0 {
		t.Error("delete failed")
	}
}

// TestBranching verifies the message-tree behavior that powers edit and
// regenerate: two children of one parent form two branches.
func TestBranching(t *testing.T) {
	s := testService(t)
	c, _ := s.CreateConversation("m", "", "")

	u1, _ := s.addMessage(c.ID, "", "user", "hello")
	a1, _ := s.addMessage(c.ID, u1.ID, "assistant", "hi there")
	// Branch: regenerate from the same user message.
	a2, _ := s.addMessage(c.ID, u1.ID, "assistant", "alternative answer")
	// Continue the first branch.
	u2, _ := s.addMessage(c.ID, a1.ID, "user", "follow up")

	chain, err := s.chain(u2.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(chain) != 3 || chain[1].ID != a1.ID {
		t.Fatalf("branch 1 chain wrong: %+v", chain)
	}

	chain2, err := s.chain(a2.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(chain2) != 2 || chain2[1].ID != a2.ID {
		t.Fatalf("branch 2 chain wrong: %+v", chain2)
	}

	// Both branches must survive in the full message list.
	msgs, _ := s.Messages(c.ID)
	if len(msgs) != 4 {
		t.Errorf("want 4 messages, got %d", len(msgs))
	}
}

func TestLatestLeaf(t *testing.T) {
	s := testService(t)
	c, _ := s.CreateConversation("m", "", "")
	if leaf, _ := s.latestLeaf(c.ID); leaf != "" {
		t.Error("empty conversation must have no leaf")
	}
	u1, _ := s.addMessage(c.ID, "", "user", "one")
	a1, _ := s.addMessage(c.ID, u1.ID, "assistant", "two")
	leaf, err := s.latestLeaf(c.ID)
	if err != nil || leaf != a1.ID {
		t.Errorf("leaf = %q, want %q", leaf, a1.ID)
	}
}

func TestParamsMerge(t *testing.T) {
	body := map[string]any{"model": "x"}
	temp := 0.5
	maxTok := 100
	applyParams(body, GenParams{Temperature: &temp, MaxTokens: &maxTok, Stop: []string{"END"}}, gguf.Reasoning{})
	if body["temperature"] != 0.5 || body["max_tokens"] != 100 {
		t.Errorf("params not applied: %+v", body)
	}
	stops, ok := body["stop"].([]string)
	if !ok || stops[0] != "END" {
		t.Errorf("stop not applied: %+v", body["stop"])
	}
	if _, present := body["top_k"]; present {
		t.Error("unset params must not appear in the request")
	}
}

func TestMergeGenParamsUsesSavedDefaultsAndRequestOverrides(t *testing.T) {
	savedTemp := 0.35
	savedMax := 512
	requestTemp := 0.8
	merged := mergeGenParams(
		GenParams{Temperature: &savedTemp, MaxTokens: &savedMax, Stop: []string{"saved"}},
		GenParams{Temperature: &requestTemp},
	)
	if merged.Temperature == nil || *merged.Temperature != requestTemp {
		t.Fatalf("temperature = %v, want request override %v", merged.Temperature, requestTemp)
	}
	if merged.MaxTokens == nil || *merged.MaxTokens != savedMax {
		t.Fatalf("max tokens = %v, want saved default %v", merged.MaxTokens, savedMax)
	}
	if len(merged.Stop) != 1 || merged.Stop[0] != "saved" {
		t.Fatalf("stop = %v, want saved stop", merged.Stop)
	}
}

func TestJSONSchemaParam(t *testing.T) {
	body := map[string]any{}
	applyParams(body, GenParams{JSONSchema: `{"type":"object"}`}, gguf.Reasoning{})
	rf, ok := body["response_format"].(map[string]any)
	if !ok || rf["type"] != "json_schema" {
		t.Errorf("response_format malformed: %+v", body)
	}
}

func TestMergeReasoningEffort(t *testing.T) {
	merged := mergeGenParams(
		GenParams{ReasoningEffort: "medium"},
		GenParams{ReasoningEffort: "off"},
	)
	if merged.ReasoningEffort != "off" {
		t.Fatalf("got %q", merged.ReasoningEffort)
	}
	merged = mergeGenParams(GenParams{ReasoningEffort: "high"}, GenParams{})
	if merged.ReasoningEffort != "high" {
		t.Fatalf("omitted override dropped saved effort: %q", merged.ReasoningEffort)
	}
}

func TestApplyReasoningEffortParams(t *testing.T) {
	qwen := gguf.DetectReasoning(
		`{%- if enable_thinking is defined and enable_thinking is false %}{% endif %}`,
		"qwen3",
	)
	body := map[string]any{}
	applyParams(body, GenParams{ReasoningEffort: "off"}, qwen)
	kw, _ := body["chat_template_kwargs"].(map[string]any)
	if kw["enable_thinking"] != false {
		t.Fatalf("kwargs = %v", body["chat_template_kwargs"])
	}

	glim := gguf.DetectReasoning("", "muse-glimmer")
	body = map[string]any{}
	applyParams(body, GenParams{
		ChatTemplateKwargs: `{"reasoning_strength":"low"}`,
		ReasoningEffort:    "xhigh",
	}, glim)
	kw, _ = body["chat_template_kwargs"].(map[string]any)
	if kw["reasoning_strength"] != "xhigh" {
		t.Fatalf("merged kwargs = %v", kw)
	}
	if _, ok := body["reasoning_effort"]; ok {
		t.Fatal("glimmer must not send top-level reasoning_effort")
	}
}

func TestMergeReasoningBudget(t *testing.T) {
	saved := 2048
	override := 512
	merged := mergeGenParams(
		GenParams{ReasoningBudget: &saved},
		GenParams{ReasoningBudget: &override},
	)
	if merged.ReasoningBudget == nil || *merged.ReasoningBudget != override {
		t.Fatalf("got %v, want override %d", merged.ReasoningBudget, override)
	}
	merged = mergeGenParams(GenParams{ReasoningBudget: &saved}, GenParams{})
	if merged.ReasoningBudget == nil || *merged.ReasoningBudget != saved {
		t.Fatalf("omitted override dropped saved budget: %v", merged.ReasoningBudget)
	}
	zero := 0
	merged = mergeGenParams(GenParams{ReasoningBudget: &saved}, GenParams{ReasoningBudget: &zero})
	if merged.ReasoningBudget == nil || *merged.ReasoningBudget != 0 {
		t.Fatalf("zero budget must be preserved: %v", merged.ReasoningBudget)
	}
}

func TestApplyReasoningBudgetParams(t *testing.T) {
	n := 2048
	body := map[string]any{}
	applyParams(body, GenParams{ReasoningBudget: &n}, gguf.Reasoning{})
	if body["reasoning_budget_tokens"] != 2048 {
		t.Fatalf("reasoning_budget_tokens = %v", body["reasoning_budget_tokens"])
	}
	if body["thinking_budget_tokens"] != 2048 {
		t.Fatalf("thinking_budget_tokens = %v", body["thinking_budget_tokens"])
	}

	off := 0
	body = map[string]any{}
	applyParams(body, GenParams{ReasoningBudget: &off}, gguf.Reasoning{})
	if body["reasoning_budget_tokens"] != 0 || body["thinking_budget_tokens"] != 0 {
		t.Fatalf("zero budget dropped: %+v", body)
	}

	unlimited := -1
	body = map[string]any{}
	applyParams(body, GenParams{ReasoningBudget: &unlimited}, gguf.Reasoning{})
	if body["reasoning_budget_tokens"] != -1 {
		t.Fatalf("unlimited = %v", body["reasoning_budget_tokens"])
	}

	body = map[string]any{}
	applyParams(body, GenParams{}, gguf.Reasoning{})
	if _, ok := body["reasoning_budget_tokens"]; ok {
		t.Fatal("unset budget must not appear in the request")
	}
	if _, ok := body["thinking_budget_tokens"]; ok {
		t.Fatal("unset thinking_budget_tokens leaked")
	}
}

func TestNormalizeGenParamsLegacyMaxAndBudgetRoom(t *testing.T) {
	legacy := 2048
	p := GenParams{MaxTokens: &legacy}
	normalizeGenParams(&p)
	if p.MaxTokens == nil || *p.MaxTokens != defaultMaxTokens {
		t.Fatalf("legacy 2048 max_tokens = %v, want %d", p.MaxTokens, defaultMaxTokens)
	}

	p = GenParams{}
	normalizeGenParams(&p)
	if p.MaxTokens == nil || *p.MaxTokens != defaultMaxTokens {
		t.Fatalf("omitted max_tokens = %v, want %d", p.MaxTokens, defaultMaxTokens)
	}

	budget := 8192
	maxTok := 8192
	p = GenParams{MaxTokens: &maxTok, ReasoningBudget: &budget}
	normalizeGenParams(&p)
	if p.MaxTokens == nil || *p.MaxTokens != 8192+minAnswerTokens {
		t.Fatalf("budget equal to max_tokens = %v, want %d so the answer still fits",
			p.MaxTokens, 8192+minAnswerTokens)
	}

	budget = 2048
	maxTok = 8192
	p = GenParams{MaxTokens: &maxTok, ReasoningBudget: &budget}
	normalizeGenParams(&p)
	if p.MaxTokens == nil || *p.MaxTokens != 8192 {
		t.Fatalf("roomy max_tokens overwritten: %v", p.MaxTokens)
	}

	unlimited := -1
	maxTok = 2048
	p = GenParams{MaxTokens: &maxTok, ReasoningBudget: &unlimited}
	normalizeGenParams(&p)
	if p.MaxTokens == nil || *p.MaxTokens != defaultMaxTokens {
		t.Fatalf("unlimited budget should still lift legacy max: %v", p.MaxTokens)
	}
}
