-- Quantization jobs and reusable importance matrices.

CREATE TABLE IF NOT EXISTS quant_jobs (
    id               TEXT PRIMARY KEY,
    kind             TEXT NOT NULL DEFAULT 'quantize', -- quantize|imatrix|combine_imatrix|adaptive_quantize
    state            TEXT NOT NULL DEFAULT 'queued',   -- queued|running|canceling|complete|failed|canceled
    stage            TEXT NOT NULL DEFAULT '',
    progress         REAL NOT NULL DEFAULT 0,
    runtime_id       TEXT NOT NULL DEFAULT '',
    source_model_id  TEXT NOT NULL DEFAULT '',
    dest_path        TEXT NOT NULL DEFAULT '',
    pid              INTEGER NOT NULL DEFAULT 0,
    log_path         TEXT NOT NULL DEFAULT '',
    request_json     TEXT NOT NULL DEFAULT '{}',
    result_json      TEXT NOT NULL DEFAULT '{}',
    error            TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    finished_at      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_quant_jobs_state ON quant_jobs(state);
CREATE INDEX IF NOT EXISTS idx_quant_jobs_created ON quant_jobs(created_at);

CREATE TABLE IF NOT EXISTS imatrices (
    id              TEXT PRIMARY KEY,
    source_model_id TEXT NOT NULL DEFAULT '',
    path            TEXT NOT NULL,
    format          TEXT NOT NULL DEFAULT 'gguf', -- gguf|dat
    dataset_label   TEXT NOT NULL DEFAULT '',
    n_chunks        INTEGER NOT NULL DEFAULT 0,
    origin          TEXT NOT NULL DEFAULT 'generated', -- generated|imported|combined
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_imatrices_model ON imatrices(source_model_id);
