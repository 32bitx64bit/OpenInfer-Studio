-- Durable graph snapshots and stage-result reuse. Outputs remain in the media store.
CREATE TABLE workflow_runs (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    view_json TEXT NOT NULL
);
CREATE INDEX idx_workflow_runs_created ON workflow_runs(created_at DESC);
CREATE TABLE workflow_cache (
    key TEXT PRIMARY KEY,
    entry_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
