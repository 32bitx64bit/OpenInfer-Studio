package workflow

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
)

// ErrNotFound is returned when a workflow id does not exist.
var ErrNotFound = errors.New("workflow not found")

// ErrInvalid marks a rejected document or name (a client error, not a
// storage failure). Use errors.Is(err, ErrInvalid).
var ErrInvalid = errors.New("invalid workflow")

type invalidError struct{ err error }

func (e invalidError) Error() string        { return e.err.Error() }
func (e invalidError) Unwrap() error        { return e.err }
func (e invalidError) Is(target error) bool { return target == ErrInvalid }

func invalid(format string, args ...any) error {
	return invalidError{fmt.Errorf(format, args...)}
}

// Store limits.
const (
	MaxWorkflows = 500
	maxNameLen   = 200
)

// Record is one stored workflow.
type Record struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Graph     json.RawMessage `json:"graph"`
	CreatedAt string          `json:"created_at"`
	UpdatedAt string          `json:"updated_at"`
}

// Summary is a Record without its graph, for list views.
type Summary struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	NodeCount int    `json:"node_count"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
}

// Store persists workflows in SQLite. Saving never requires a valid graph:
// a half-built draft is a normal thing to keep. Only structure (version,
// size, node and edge limits) is checked; Validate and Build do the rest.
type Store struct {
	db *sql.DB
}

// NewStore returns a store over an open database with migration 0005 applied.
func NewStore(db *sql.DB) *Store { return &Store{db: db} }

func now() string { return time.Now().UTC().Format(time.RFC3339Nano) }

func cleanName(name string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return "", invalid("workflow name is required")
	}
	if len(name) > maxNameLen {
		return "", invalid("workflow name exceeds %d bytes", maxNameLen)
	}
	return name, nil
}

// canonical parses raw as a graph and re-encodes it, dropping fields this
// build does not know so stored documents stay within the contract.
func canonical(raw json.RawMessage) (json.RawMessage, error) {
	g, err := ParseGraph(raw)
	if err != nil {
		return nil, invalidError{err}
	}
	out, err := json.Marshal(g)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// Create stores a new workflow.
func (s *Store) Create(name string, graph json.RawMessage) (*Record, error) {
	name, err := cleanName(name)
	if err != nil {
		return nil, err
	}
	doc, err := canonical(graph)
	if err != nil {
		return nil, err
	}
	var n int
	if err := s.db.QueryRow(`SELECT COUNT(1) FROM workflows`).Scan(&n); err != nil {
		return nil, err
	}
	if n >= MaxWorkflows {
		return nil, invalid("workflow limit reached (%d); delete one first", MaxWorkflows)
	}
	id, ts := uuid.NewString(), now()
	if _, err := s.db.Exec(`INSERT INTO workflows(id,name,graph_json,created_at,updated_at) VALUES (?,?,?,?,?)`,
		id, name, string(doc), ts, ts); err != nil {
		return nil, err
	}
	return s.Get(id)
}

// Get returns one workflow with its graph.
func (s *Store) Get(id string) (*Record, error) {
	var r Record
	var graph string
	err := s.db.QueryRow(`SELECT id,name,graph_json,created_at,updated_at FROM workflows WHERE id=?`, id).
		Scan(&r.ID, &r.Name, &graph, &r.CreatedAt, &r.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	r.Graph = json.RawMessage(graph)
	return &r, nil
}

// List returns every workflow, most recently changed first.
func (s *Store) List() ([]Summary, error) {
	rows, err := s.db.Query(`SELECT id,name,graph_json,created_at,updated_at FROM workflows ORDER BY updated_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Summary{}
	for rows.Next() {
		var sm Summary
		var graph string
		if err := rows.Scan(&sm.ID, &sm.Name, &graph, &sm.CreatedAt, &sm.UpdatedAt); err != nil {
			return nil, err
		}
		var doc struct {
			Nodes []json.RawMessage `json:"nodes"`
		}
		if json.Unmarshal([]byte(graph), &doc) == nil {
			sm.NodeCount = len(doc.Nodes)
		}
		out = append(out, sm)
	}
	return out, rows.Err()
}

// Update changes a workflow's name and/or graph. A nil argument leaves that
// part unchanged.
func (s *Store) Update(id string, name *string, graph json.RawMessage) (*Record, error) {
	cur, err := s.Get(id)
	if err != nil {
		return nil, err
	}
	newName, newGraph := cur.Name, string(cur.Graph)
	if name != nil {
		if newName, err = cleanName(*name); err != nil {
			return nil, err
		}
	}
	if graph != nil {
		doc, err := canonical(graph)
		if err != nil {
			return nil, err
		}
		newGraph = string(doc)
	}
	if _, err := s.db.Exec(`UPDATE workflows SET name=?, graph_json=?, updated_at=? WHERE id=?`,
		newName, newGraph, now(), id); err != nil {
		return nil, err
	}
	return s.Get(id)
}

// Delete removes a workflow.
func (s *Store) Delete(id string) error {
	res, err := s.db.Exec(`DELETE FROM workflows WHERE id=?`, id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return ErrNotFound
	}
	return nil
}
