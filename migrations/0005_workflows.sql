-- OpenInfer Studio node-graph workflows (Image Studio Graph view).
-- A workflow is a JSON graph document edited by the QML canvas. Runs, the
-- node cache and media_jobs linkage arrive with the executor in their own
-- migration; this one only stores what the user draws.

CREATE TABLE IF NOT EXISTS workflows (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    graph_json  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workflows_updated ON workflows(updated_at);
