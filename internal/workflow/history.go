package workflow

import (
	"database/sql"
	"encoding/json"
	"fmt"
)

// RunOptions is captured at submission, independently of subsequent editor changes.
type RunOptions struct {
	Graph      *Graph
	WorkflowID string
	Only       string
	Force      bool
}

func (e *Executor) persistLocked(r *run) error {
	if e.db == nil {
		return nil
	}
	raw, err := json.Marshal(r.view)
	if err != nil {
		return err
	}
	_, err = e.db.Exec(`INSERT INTO workflow_runs(id,state,created_at,view_json) VALUES(?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET state=excluded.state,view_json=excluded.view_json`,
		r.view.ID, r.view.State, r.view.CreatedAt, string(raw))
	return err
}

func (e *Executor) persist(r *run) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.persistLocked(r); err != nil {
		e.log.Error("saving workflow run", "run", r.view.ID, "err", err)
	}
}

func (e *Executor) storedRun(id string) (RunView, error) {
	var v RunView
	if e.db == nil {
		return v, sql.ErrNoRows
	}
	var raw string
	if err := e.db.QueryRow(`SELECT view_json FROM workflow_runs WHERE id=?`, id).Scan(&raw); err != nil {
		return v, err
	}
	err := json.Unmarshal([]byte(raw), &v)
	return v, err
}

// History merges durable results with authoritative in-flight state.
func (e *Executor) History() ([]RunView, error) {
	if e.db == nil {
		return e.List(), nil
	}
	rows, err := e.db.Query(`SELECT view_json FROM workflow_runs ORDER BY created_at DESC LIMIT 200`)
	if err != nil {
		return nil, err
	}
	out := []RunView{}
	for rows.Next() {
		var raw string
		var v RunView
		if err := rows.Scan(&raw); err != nil {
			rows.Close()
			return nil, err
		}
		if err := json.Unmarshal([]byte(raw), &v); err != nil {
			rows.Close()
			return nil, err
		}
		out = append(out, v)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return nil, err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	for i := range out {
		if r := e.runs[out[i].ID]; r != nil {
			out[i] = r.snapshotLocked()
		}
	}
	return out, nil
}

// Interrupted work is recorded as canceled after restart, never silently replayed.
func (e *Executor) recoverRuns() error {
	rows, err := e.db.Query(`SELECT view_json FROM workflow_runs WHERE state IN ('queued','running')`)
	if err != nil {
		return err
	}
	views := []RunView{}
	for rows.Next() {
		var raw string
		var v RunView
		if err := rows.Scan(&raw); err != nil {
			rows.Close()
			return err
		}
		if err := json.Unmarshal([]byte(raw), &v); err != nil {
			rows.Close()
			return err
		}
		views = append(views, v)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return err
	}
	for _, v := range views {
		v.State, v.Error, v.FinishedAt = RunCanceled, "Interrupted when the backend stopped", ts()
		for id, ns := range v.Nodes {
			if ns.State == NodePending || ns.State == NodeRunning {
				ns.State = NodeCanceled
				v.Nodes[id] = ns
			}
		}
		if err := e.persistLocked(&run{view: v}); err != nil {
			return fmt.Errorf("recover run %s: %w", v.ID, err)
		}
	}
	return nil
}

func (e *Executor) recordSeed(r *run, nodeID string, seed *int64) {
	if seed == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	value := *seed
	ns := r.view.Nodes[nodeID]
	ns.Seed = &value
	r.view.Nodes[nodeID] = ns
	if r.view.Graph != nil {
		for i := range r.view.Graph.Nodes {
			n := &r.view.Graph.Nodes[i]
			if n.ID == nodeID {
				if n.Params == nil {
					n.Params = map[string]any{}
				}
				n.Params["seed"] = value
			}
		}
	}
	// Record the resolved seed before inference starts, so even interrupted
	// runs can be inspected and restored reproducibly after a restart.
	if err := e.persistLocked(r); err != nil {
		e.log.Error("saving resolved workflow seed", "run", r.view.ID, "err", err)
	}
}
