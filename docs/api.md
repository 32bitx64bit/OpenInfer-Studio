# Control API

Base: `http://127.0.0.1:<port>/api/v1` — loopback only. Every request needs
`Authorization: Bearer <session-token>`; the WebSocket
(`GET /api/v1/events`) takes `{"token":"…"}` as its first frame.

Errors: `{"error": "message", "detail": "debug detail"}`.

## Core

| Method & path | Purpose |
|---|---|
| GET `/status` | app identity: `version`, `commit`, `date`, `goos`/`goarch`, `api`, `data_dir` |
| GET `/hardware[?refresh=1]` | detected hardware + backend recommendation |
| GET `/settings` / PUT `/settings/{key}` | key-value settings (`{"value":"…"}`) |
| GET `/logs/files`, GET `/logs/tail?name=` | log files, redacted tails |

## Hugging Face

| Method & path | Purpose |
|---|---|
| GET `/hf/search?q=&sort=&limit=` | GGUF repo search (sort: downloads, likes, trending, lastModified); results include `modalities`, `mtp`, and `embedding` (`embedding`\|`reranker`) when detectable |
| GET `/hf/repo/{author}/{name}` | repo detail + grouped file sets + `modalities` + `mtp` + `embedding` + model card |
| GET/PUT/DELETE `/hf/token` | token status / store in OS keychain / remove |

## Downloads

| Method & path | Purpose |
|---|---|
| GET `/downloads` | queue with per-file progress |
| POST `/downloads` | enqueue `{kind,label,repo,group,files:[{path,size,url?}]}` |
| POST `/downloads/{id}/pause|resume|cancel|retry` | control |
| POST `/downloads/{id}/reorder` | `{"position":n}` |
| DELETE `/downloads/{id}` | remove record + partials |

Events: `download.progress` (bytes, speed, ETA), `download.state_changed`.

## Library & models

| Method & path | Purpose |
|---|---|
| GET `/models` | all models (favorites first) |
| POST `/models/scan` | rescan registered directories |
| POST `/models/import` | `{"path": "/…/file.gguf"}` copy into managed `models/local--…/files/` (plus same-dir shards/mmproj), then scan |
| GET `/models/{id}` | model + presets + live instance |
| PATCH `/models/{id}` | alias, favorite, notes, pinned_runtime, pinned_backend |
| DELETE `/models/{id}` | two-phase: first call returns `paths`; re-call with `?confirmed=1&delete_files=1` |
| GET/POST `/models/{id}/presets`, PUT/DELETE `…/presets/{pid}` | load presets |
| GET/POST/DELETE `/directories[/{id}]` | extra model directories |

## Loading & instances

| Method & path | Purpose |
|---|---|
| POST `/models/{id}/preview` | resolved command, resolutions, warnings — nothing started |
| POST `/models/{id}/estimate` | projected memory: weights, draft, mmproj, KV, compute, media, overhead; independent `gpu_bytes`/`cpu_bytes` vs `gpu_budget_bytes`/`cpu_budget_bytes` (`fits_gpu`/`fits_cpu`/`fits`); `offload_fraction` for custom layer offload; `budget_kind` is `VRAM+RAM`, `VRAM`, `RAM`, or `unified RAM` |
| GET `/models/{id}/draft-candidates[?filter=0\|1]` | draft picker list; filter on (default / `load.filter_incompatible_drafts`) returns only speculative sidecars (mtp-/gemma4-assistant/eagle3-/dflash-/dspark-); filter off returns all other library models |
| POST `/models/{id}/load` | start (LoadSettings JSON; all fields optional). Speculative: `draft_model`, `draft_max`, `draft_min`, `spec_type`. Embedders: `embedding` (bool; auto-true for detected embedders), `pooling` (`none\|mean\|cls\|last\|rank`; empty = model default; rerankers prefer `rank`). Emits `--embedding` / `--pooling` when the runtime advertises them. Block-diffusion models (`is_diffusion`, e.g. DiffusionGemma) launch `llama-diffusion-gemma-visual-server` beside the runtime with an OpenAI-compatible shim instead of `llama-server`. Expert/perf: `threads_batch`, `cont_batching`, `cache_reuse`, `prio`, `poll`, `numa`, `fit`, `kv_offload`, `op_offload`, `kv_unified`, `swa_full`, `cpu_moe`, `n_cpu_moe`, `main_gpu`, `device`, `split_mode`, `tensor_split`, `no_warmup`, `raw_args` |
| POST `/models/{id}/unload[?force=1]` | graceful / forced stop |
| POST `/models/{id}/restart` | reload with settings |
| GET `/models/{id}/logs` | redacted log tail |
| GET `/models/{id}/activity` | latest `/slots` snapshot (busy, tokens, tok/s) |
| GET `/models/{id}/diagnostics[?redact_home=1]` | full failure report |
| GET `/instances` | live instances with state machine state |

Events: `instance.state_changed`, `instance.updated`, `instance.activity`,
`instance.log`.

## Runtimes

| Method & path | Purpose |
|---|---|
| GET `/runtimes` | installed builds + capabilities + pinning |
| GET `/runtimes/releases[?backend=]` | official releases with scored asset matches |
| POST `/runtimes/install` | `{tag, asset, backend}` — async, progress via events |
| POST `/runtimes/import` | `{path}` custom llama-server executable or archive (`.zip` / `.tar.gz` / `.tgz`) |
| POST `/runtimes/{id}/preferred`, `/health` | |
| GET `/runtimes/{id}/capabilities` | parsed caps + raw help + version output |
| DELETE `/runtimes/{id}` | refused while pinned by models |

## Chat

| Method & path | Purpose |
|---|---|
| GET/POST `/chat` | list / create conversations |
| PATCH `/chat/{id}` | title, model_id, system, archived, params |
| DELETE `/chat/{id}` | delete |
| GET `/chat/{id}/messages` | full message tree (branches included) |
| POST `/chat/{id}/generate` | `{parent_id?, content?, params?, audio?}` → streams `chat.token`; `audio` requires Settings `experimental.audio_models=1` and a `has_audio` model (`{path\|data, format?, name?}`) |
| POST `/chat/{id}/stop` | cancel generation |

Setting keys of note: `experimental.audio_models` (`"0"`/`"1"`, default off) gates audio discovery labels and chat audio attachments (stays experimental while upstream llama.cpp marks audio as such). `onboarding.completed` (`"0"`/`"1"`) tracks whether the first-run setup wizard has been finished or skipped.

## Public server

| Method & path | Purpose |
|---|---|
| GET/PUT `/server` | config (port, bind, allow_lan, cors, autostart) + running state |
| POST `/server/start|stop|regenerate-key` | |
| GET `/server/requests` | recent public requests |

The public OpenAI-compatible server itself (separate port, separate key):
`GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`,
`POST /v1/embeddings`, `POST /v1/responses` (subject to runtime support).

Dedicated GGUF embedders (and rerankers) are detected on library scan
(`metadata.is_embedding` / `is_reranker`, optional `pooling_type`). Loading
them auto-enables embedding mode so the instance serves
`POST /v1/embeddings` rather than chat. They are omitted from the Chat model
picker; load from Library and call the Developer API.
