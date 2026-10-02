package workflow

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/migrations"
)

func newStore(t *testing.T) *Store {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "database")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	db, err := database.Open(dir, migrations.FS)
	if err != nil {
		t.Fatalf("open database: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return NewStore(db.DB)
}

func graphJSON(t *testing.T, g Graph) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(g)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestStoreRoundTrip(t *testing.T) {
	s := newStore(t)
	rec, err := s.Create("  Portrait  ", graphJSON(t, txt2img()))
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if rec.ID == "" || rec.Name != "Portrait" || rec.CreatedAt == "" {
		t.Fatalf("record = %+v", rec)
	}

	got, err := s.Get(rec.ID)
	if err != nil {
		t.Fatal(err)
	}
	back, err := ParseGraph(got.Graph)
	if err != nil || len(back.Nodes) != 6 || len(back.Edges) != 7 {
		t.Fatalf("stored graph: %d nodes %d edges, %v", len(back.Nodes), len(back.Edges), err)
	}
	if plan, issues := Build(back, NewRegistry(), newFakeEnv(), allCaps, Options{}); plan == nil {
		t.Fatalf("a stored graph must still plan: %+v", issues)
	}
}

func TestStoreKeepsInvalidDrafts(t *testing.T) {
	s := newStore(t)
	draft := Graph{Version: 1, Nodes: []Node{node("n1", "sample", nil)}} // required inputs unwired
	if _, err := s.Create("Draft", graphJSON(t, draft)); err != nil {
		t.Fatalf("a half-built draft must save: %v", err)
	}
}

func TestStoreRejectsBadDocuments(t *testing.T) {
	s := newStore(t)
	for name, tc := range map[string]struct {
		name  string
		graph json.RawMessage
		want  string
	}{
		"empty name":     {"   ", graphJSON(t, txt2img()), "name is required"},
		"long name":      {strings.Repeat("x", 201), graphJSON(t, txt2img()), "name exceeds"},
		"not json":       {"ok", json.RawMessage(`{`), "invalid graph JSON"},
		"future version": {"ok", json.RawMessage(`{"version":9}`), "version"},
		"empty graph":    {"ok", nil, "empty"},
		"too many nodes": {"ok", bigGraph(t, MaxNodes+1), "nodes"},
	} {
		if _, err := s.Create(tc.name, tc.graph); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want substring %q", name, err, tc.want)
		}
	}
}

func bigGraph(t *testing.T, n int) json.RawMessage {
	t.Helper()
	g := Graph{Version: 1}
	for i := 0; i < n; i++ {
		g.Nodes = append(g.Nodes, node("n", "prompt", nil))
	}
	return graphJSON(t, g)
}

func TestStoreUpdateListDelete(t *testing.T) {
	s := newStore(t)
	a, _ := s.Create("A", graphJSON(t, txt2img()))
	b, _ := s.Create("B", graphJSON(t, hiresFix()))

	newName := "A renamed"
	up, err := s.Update(a.ID, &newName, nil)
	if err != nil || up.Name != newName || string(up.Graph) != string(a.Graph) {
		t.Fatalf("rename only: %+v %v", up, err)
	}
	up, err = s.Update(a.ID, nil, graphJSON(t, hiresFix()))
	if err != nil || up.Name != newName {
		t.Fatalf("graph only: %+v %v", up, err)
	}
	if _, err := s.Update(a.ID, nil, json.RawMessage(`{`)); err == nil {
		t.Fatal("an invalid graph document must be rejected on update")
	}
	if kept, _ := s.Get(a.ID); string(kept.Graph) != string(up.Graph) {
		t.Fatal("a rejected update must leave the stored graph untouched")
	}

	list, err := s.List()
	if err != nil || len(list) != 2 {
		t.Fatalf("list = %+v %v", list, err)
	}
	if list[0].ID != a.ID { // A was updated last
		t.Fatalf("list order: %+v", list)
	}
	if list[0].NodeCount != 8 || list[1].NodeCount != 8 {
		t.Fatalf("node counts = %d, %d", list[0].NodeCount, list[1].NodeCount)
	}

	if err := s.Delete(b.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Get(b.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Get after delete: %v", err)
	}
	if err := s.Delete(b.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("second delete: %v", err)
	}
	if _, err := s.Update("nope", &newName, nil); !errors.Is(err, ErrNotFound) {
		t.Fatalf("update missing: %v", err)
	}
}
