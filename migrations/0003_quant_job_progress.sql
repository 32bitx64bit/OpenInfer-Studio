-- Stage-local quantization progress and ETA details.
ALTER TABLE quant_jobs ADD COLUMN stage_progress REAL NOT NULL DEFAULT 0;
ALTER TABLE quant_jobs ADD COLUMN progress_current INTEGER NOT NULL DEFAULT 0;
ALTER TABLE quant_jobs ADD COLUMN progress_total INTEGER NOT NULL DEFAULT 0;
ALTER TABLE quant_jobs ADD COLUMN stage_eta_seconds INTEGER NOT NULL DEFAULT 0;
ALTER TABLE quant_jobs ADD COLUMN eta_seconds INTEGER NOT NULL DEFAULT 0;
ALTER TABLE quant_jobs ADD COLUMN progress_message TEXT NOT NULL DEFAULT '';
ALTER TABLE quant_jobs ADD COLUMN stage_started_at TEXT NOT NULL DEFAULT '';
