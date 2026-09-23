-- OpenInfer Studio image/video generation schema.
-- Diffusion checkpoints (.safetensors/.ckpt/.pt bundles) are library rows
-- with modality='diffusion', alongside the existing GGUF LLM rows.

CREATE TABLE IF NOT EXISTS media_jobs (
    id             TEXT PRIMARY KEY,
    model_id       TEXT NOT NULL DEFAULT '',
    kind           TEXT NOT NULL DEFAULT 'image', -- image|video
    state          TEXT NOT NULL DEFAULT 'queued', -- queued|running|complete|failed|canceled
    prompt         TEXT NOT NULL DEFAULT '',
    params_json    TEXT NOT NULL DEFAULT '{}',
    output_path    TEXT NOT NULL DEFAULT '',
    output_format  TEXT NOT NULL DEFAULT '',
    width          INTEGER NOT NULL DEFAULT 0,
    height         INTEGER NOT NULL DEFAULT 0,
    frames         INTEGER NOT NULL DEFAULT 0,
    seed           INTEGER NOT NULL DEFAULT 0,
    runtime_id     TEXT NOT NULL DEFAULT '',
    pid            INTEGER NOT NULL DEFAULT 0,
    log_path       TEXT NOT NULL DEFAULT '',
    result_json    TEXT NOT NULL DEFAULT '{}',
    error          TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    finished_at    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_media_jobs_state ON media_jobs(state);
CREATE INDEX IF NOT EXISTS idx_media_jobs_model ON media_jobs(model_id);
