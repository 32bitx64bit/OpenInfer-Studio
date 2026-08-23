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
| PATCH `/models/{id}` | `alias` (library display name; empty rejected), favorite, notes, pinned_runtime, pinned_backend |
| DELETE `/models/{id}` | two-phase: first call returns `paths`; re-call with `?confirmed=1&delete_files=1` |
| GET/POST `/models/{id}/presets`, PUT/DELETE `…/presets/{pid}` | load presets |
| GET/POST/DELETE `/directories[/{id}]` | extra model directories |

Events: `library.scanned`, `library.model_imported`, `library.model_updated` (`id`, `alias`).

## Loading & instances

| Method & path | Purpose |
|---|---|
| POST `/models/{id}/preview` | resolved command, resolutions, warnings — nothing started |
| POST `/models/{id}/estimate` | projected memory: weights, draft, mmproj, KV, compute, media, overhead; independent `gpu_bytes`/`cpu_bytes` vs `gpu_budget_bytes`/`cpu_budget_bytes` (`fits_gpu`/`fits_cpu`/`fits`); `offload_fraction` for custom layer offload; `budget_kind` is `VRAM+RAM`, `VRAM`, `RAM`, or `unified RAM` |
| GET `/models/{id}/draft-candidates[?filter=0\|1]` | draft picker list; filter on (default / `load.filter_incompatible_drafts`) returns only speculative sidecars (mtp-/gemma4-assistant/eagle3-/dflash-/dspark-); filter off returns all other library models |
| POST `/models/{id}/load` | start (LoadSettings JSON; all fields optional). Speculative: `draft_model`, `draft_max`, `draft_min`, `spec_type`. Embedders: `embedding` (bool; auto-true for detected embedders), `pooling` (`none\|mean\|cls\|last\|rank`; empty = model default; rerankers prefer `rank`). Emits `--embedding` / `--pooling` when the runtime advertises them. Block-diffusion models (`is_diffusion`, e.g. DiffusionGemma) launch `llama-diffusion-gemma-visual-server` beside the runtime with an OpenAI-compatible shim instead of `llama-server`. Muse Glimmer (`muse-glimmer`, including text-only OID quants): auto `--jinja` and `--chat-template-kwargs {"reasoning_strength":"low"}` so thought splits into `reasoning_content` and the answer can stream before `max_tokens` is spent. Override with `jinja` / `chat_template_kwargs`. Chat templates that can keep prior-turn reasoning (`metadata.reasoning.can_preserve`: `preserve_thinking` / `preserve_reasoning` / `clear_thinking`) default `reasoning_preserve` on (`--reasoning-preserve`; older runtimes get the equivalent `chat_template_kwargs`). Override with `reasoning_preserve`. Expert/perf: `threads_batch`, `cont_batching`, `cache_reuse`, `prio`, `poll`, `numa`, `fit`, `kv_offload`, `op_offload`, `kv_unified`, `swa_full`, `cpu_moe`, `n_cpu_moe`, `main_gpu`, `device`, `split_mode`, `tensor_split`, `no_warmup`, `raw_args` |
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
| GET `/runtimes/{id}/tools` | sibling binaries (`llama-quantize`, `llama-imatrix`, `llama-gguf-split`) + parsed flags + advertised ftypes |
| DELETE `/runtimes/{id}` | refused while pinned by models |

## Quantization

GGUF → GGUF jobs driven by the selected runtime’s `llama-quantize` /
`llama-imatrix`. HTTP is async: `POST /quantize/jobs` returns **202** and a
job id. Preview remains a cheap internal estimate; queued jobs verify output
size with `llama-quantize --dry-run` when the runtime advertises it.

| Method & path | Purpose |
|---|---|
| GET `/quantize/types?runtime_id=` | ftypes with band, bpw, imatrix policy, plus tool catalog |
| GET `/quantize/from-hf/preview?repo=` | probe a Hugging Face safetensors repo: compatible, architecture, weight dtype, download/GGUF/disk estimates, reuse if a matching high-precision GGUF is already in the library |
| POST `/quantize/preview` | estimated size, VRAM/RAM/disk fit, warnings, companions, recommended ftype. Body is a `Request` |
| POST `/quantize/jobs` | start `quantize` \| `imatrix` \| `combine_imatrix` \| `adaptive_quantize` \| `from_hf`. Returns 202 `{id, job}` |
| GET `/quantize/jobs` | recent jobs |
| GET `/quantize/jobs/{id}` | job + redacted log tail |
| POST `/quantize/jobs/{id}/pause` | stop tools and free the GPU; Dynamic checkpoints and the repaired source copy are kept. Queued jobs pause immediately |
| POST `/quantize/jobs/{id}/resume` | re-queue a paused job. Dynamic work continues from the last checkpoint; classic `llama-quantize` restarts that tool |
| POST `/quantize/jobs/{id}/cancel` | KillTree; incomplete dest GGUF is deleted. Paused jobs are abandoned without starting |
| DELETE `/quantize/jobs/{id}` | remove a queued, paused, or finished job from history (not running/pausing). Library GGUFs are kept |
| POST `/quantize/jobs/clear-history` | delete complete/failed/canceled jobs; paused and queued jobs are kept; returns `{removed}` |
| GET `/quantize/imatrices?model_id=` | reusable importance matrices |
| POST `/quantize/imatrices/import` | `{path, source_model_id, dataset_label}` copies into managed storage |
| DELETE `/quantize/imatrices/{id}[?delete_file=1]` | forget (+ optional file delete) |

`Request` fields (unknown JSON keys are rejected): `kind`, `runtime_id`,
`source_model_id`, `hf_repo`, `ftype`, `output_name`, `threads`, `allow_requantize`,
`leave_output_tensor`, `pure`, `keep_split`, `output_tensor_type`,
`token_embedding_type`, `tensor_types`, `tensor_type_file`, `imatrix_id`,
`generate_imatrix`, `calibration_path`, `calibration_preset`
(`quick`\|`standard`\|`thorough`\|`research`), `chunks`, `chunk_skip`, `gpu_layers`,
`parse_special`, `process_output`, `combine_imatrix_ids`,
`delete_intermediates`, `keep_imatrix`, `quantize_projector`,
`projector_ftype`, `copy_projector`, `draft_model_id`, `quantize_draft`,
`draft_ftype`, `effort` (`fast`\|`profiled`\|`deep`, default `profiled`),
`quant_tier` (`q5`\|`q4`\|`q3`\|`q2`\|`custom`, default `q4`),
`adaptive_mode` (deprecated alias for `effort`), `adaptive_preset` (deprecated;
translated to `target_bpw`), `target_bpw`, `target_bytes`,
`prior_weight`, `acknowledge_requantize`,
`acknowledge_experimental`, `unload_first`.

`quant_tier` selects the OpenInfer Dynamic compression tier. A named tier
overrides `target_bpw` with the tier's bits-per-weight and clears
`target_bytes`; `quant_tier=custom` targets an explicit `target_bytes` (which
must be positive). An empty `quant_tier` defers to `target_bpw`/`target_bytes`
for backward compatibility. Tier → BPW:

| `quant_tier` | target BPW | notes |
|---|---|---|
| `q5` | 5.5 | highest Dynamic quality |
| `q4` (default) | 4.5 | balanced |
| `q3` | 3.5 | compact |
| `q2` | 2.5 | smallest |
| `custom` | — | explicit `target_bytes` |

For `kind=adaptive_quantize`, omitting `quant_tier`, `target_bpw`,
`target_bytes`, and `adaptive_preset` uses the default size goal of **4.5 BPW**
(the `q4` tier). `target_bytes`, when supplied (or `quant_tier=custom`),
overrides the BPW target.

`kind=from_hf` converts a Hugging Face **BF16/F16/F32 safetensors** repo to a
library GGUF. Convert detects the graph from `config.json` and tensor names,
maps it onto a llama.cpp `general.architecture` (from llama.cpp
`MODEL_ARCH_NAMES`), and fail-closes if llama.cpp has no loader or the weight
layout cannot be emitted (MLA, packed Qwen3-Next, RWKV/Mamba, altup). Hugging
Face class names are not allowlisted — Mistral/Mixtral write GGUF `llama`
because that is the loader llama.cpp uses. Vision weights are skipped so the
GGUF is language-only. Then the same quantize / OID path runs as a library
source. `hf_repo` is
`author/model`. Probe with `GET /quantize/from-hf/preview?repo=` before
downloading: NVFP4 / GPTQ / AWQ / unknown architectures fail closed. The
high-precision GGUF is kept if the later quantize step fails. Snapshot files
are stored under the Hugging Face cache (not `models/`) and deleted after a
successful convert. If the library already has an F32/F16/BF16 (or Q8) with
`source_repo` equal to that id, download+convert is skipped.

I-quants that require an imatrix: `IQ1_*`, `IQ2_*`, `IQ3_XXS`, `Q2_K`,
`Q2_K_S`. Mixed-precision jobs use `kind=adaptive_quantize` and are labeled
after their compression tier: **OID-Q5_K_XL** (q5, ~5.5 bpw) /
**OID-Q4_K_XL** (q4, ~4.5 bpw, IQ4_XS-class size) /
**OID-Q3_K_XL** (q3, ~3.5 bpw, Unsloth UD-Q3_K_XL-class size) /
**OID-Q2_K_XL** (q2, ~2.5 bpw) — OpenInfer Dynamic, not Unsloth `UD-`.
`quant_tier` selects the size goal and the OID label is derived from it: named
tiers map directly (`q4` → `OID-Q4_K_XL`), and `quant_tier=custom` maps the
achieved bits-per-weight to the nearest tier (≥5 → Q5, ≥4 → Q4, ≥3 → Q3, else
Q2). The result JSON carries `quant_tier`. The mix is architecture-general: dense
FFN up/gate/down on compact stays `IQ3_S` (same bytes as Unsloth `Q3_K`);
`attn_gate` is treated as FFN-like bulk, not attention. Compact typically
keeps `IQ3_S` on `attn_q` / `attn_gate` (Unsloth used `Q2_K` on a middle
band). The aggressive SKU (**OID-Q2_K_XL**, q2)
keeps dense `ffn_gate` at `IQ3_S` (Unsloth never goes below `Q3_K` there),
uses confidence-weighted per-model activation evidence to place `IQ4_XS` on
sensitive `ffn_down` layers, and only harvests `attn_q` / `attn_gate` to
`Q2_K` if the byte budget requires it. ZD drives cliff protection while
adjacent-layer CosSim only smooths noisy FFN evidence. Never `IQ2_*` on dense FFN. High-importance and
first/last layers keep `IQ3_S` or higher on dense FFN. Embeddings and output
stay `Q6_K`. MoE expert bulk still floors at `IQ3_XXS`. attn_v / output / MLA
`attn_*_b` stay high. Each OID GGUF writes a sidecar `.oid-plan.json`
(inspectable tensor map). `llama-imatrix` uses `--ctx-size` up to 4096 when
advertised (adaptive / thorough / research) if the calibration file fills at
least four unique windows; shorter files drop to 2048/1024/512 so the matrix
is not two windows looped hundreds of times. `--chunks` is also capped at
4× unique windows. The bundled deterministic corpus is a mixed, original set of several million
estimated tokens across prose, facts, code, multilingual, structured
tool-call, long-context, chat, and refusal-adjacent/direct-answer coverage,
with expanded lexicons and per-domain templates so nearby records do not
share the same 5-gram frames. It is
split before rendering into disjoint calibration, search, and validation
partitions with a provenance manifest. Calibration plus original fact cards is wrapped
in the source GGUF’s chat special tokens (ChatML, Llama 3, Gemma, Mistral,
Phi-3, or Harmony/Glimmer `<|start|>user<|message|>…`) so `--parse-special`
sees the same role tokens as Chat. Calibration records are round-robined by
domain into one mixed `llama-imatrix` pass — averaging three overfitted
domain matrices is not the same as mixed activations. Custom
`calibration_path` is wrapped as one file.
The planner combines confidence-weighted evidence with architecture-neutral
fallbacks and enforces SwiGLU within the byte budget. Preview
includes an `adaptive` object with the tensor mix and estimated size. Dynamic
jobs always generate an importance matrix when `llama-imatrix` is present
and force `parse_special` and `process_output` so the output tensor is
included in the matrix. Standalone imatrix jobs still leave `process_output`
to the request.

`effort` selects the optimization depth of a Dynamic
(`kind=adaptive_quantize`) job:

| `effort` | What it does |
|---|---|
| `fast` | heuristic solve + mandatory 2-chunk KLD gate |
| `profiled` (default) | exact-loss solve with ProbeKLD + 4-chunk KLD validation |
| `deep` | exact-loss solve with ProbeKLD + 8-chunk KLD validation |

Every effort validates the quantized model against the source before it is
published. Default KLD gates scale with `target_bpw` / the named quant tier:
a Q5 job still uses mean ≤ 0.15 and p95 ≤ 1.0; a Q3 job uses a Q3 bar
(mean ≤ 2.0, p95 ≤ 16). Exceeding a gate is a **warning** on the job
(`gates_pass=false`); the GGUF is still added to the library. Only a failed
quantization (corrupt output, missing tools, cancelled job) fails the job.

Dynamic runs checkpoint each completed stage and can resume from the first
unfinished stage without repeating recorded measurements. They retain working
artifacts until a successful publish; allow substantial scratch disk for
anchors, candidate GGUFs, calibration data, and (for `deep`) baseline logits.

Deprecated compatibility fields: `adaptive_mode` is an alias for `effort`
(same three values; an explicit `effort` wins). `adaptive_preset` is
translated to `target_bpw` and the result carries a deprecation warning:

| `adaptive_preset` | `target_bpw` | OID label (achieved-BPW tier) |
|---|---|---|
| `quality` | 6.0 | OID-Q5_K_XL |
| `balanced` | 4.5 | OID-Q4_K_XL |
| `compact` | 3.8 | OID-Q3_K_XL |
| `aggressive` | 3.0 | OID-Q3_K_XL |

Adaptive job results report the effort, gate outcomes with measured values,
the measurements, the sidecar recipe path, and any warnings:

```json
{
  "model_id": "local--my-model",
  "dest_path": "/…/models/local--my-model/files/my-model-OID-Q4_K_XL.gguf",
  "ftype": "OID-Q4_K_XL",
  "effort": "profiled",
  "quant_tier": "q4",
  "gates": [
    {"metric": "kld", "pass": true, "measured": true, "value": 0.09, "maxDelta": 0.40},
    {"metric": "p95-kld", "pass": true, "measured": true, "value": 0.62, "maxAbsolute": 3.0}
  ],
  "measurements": {"…": "…"},
  "recipe_path": "/…/models/local--my-model/files/my-model.oid-plan.json",
  "report_path": "/…/models/local--my-model/files/my-model.quantlab-report.json",
  "warnings": ["adaptive_preset is deprecated; translated to target_bpw=4.5"]
}
```

`report_path` points to the emitted `.quantlab-report.json` summary, including
the effective budget, gate outcomes, measurements, and search details.

Events: `quant.progress` (`id`, `stage`, `current`, `total`, `message`,
`progress` for the full pipeline, `stage_progress`, `stage_eta_seconds`,
`eta_seconds`, and `estimated`). Job rows persist the same details as
`progress_current`, `progress_total`, and `progress_message`, so reconnecting
the frontend does not lose ETA or stage state. Long `llama-imatrix` passes are
interpolated from the runtime's reported seconds-per-pass between checkpoints.
`quant.state_changed` (`id, state, kind, source_model_id, dest_path, model_id, ftype, error`).

## Chat

| Method & path | Purpose |
|---|---|
| GET/POST `/chat` | list / create conversations |
| PATCH `/chat/{id}` | title, model_id, system, archived, params |
| DELETE `/chat/{id}` | delete |
| GET `/chat/{id}/messages` | full message tree (branches included) |
| POST `/chat/{id}/generate` | `{parent_id?, content?, params?, audio?}` → streams `chat.token`. `params.reasoning_effort` is a template-native level (`low`/`medium`/`high`/`xhigh`/…) or `off` when the model can disable thinking. Translated to llama.cpp `chat_template_kwargs` from GGUF chat-template detection (`metadata.reasoning`). `audio` requires Settings `experimental.audio_models=1` and a `has_audio` model (`{path\|data, format?, name?}`) |
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
`reasoning_effort` (or Responses `reasoning.effort`) is mapped to the loaded
model's native `chat_template_kwargs` (for example Muse Glimmer
`reasoning_strength`, Qwen `enable_thinking`). Models that cannot disable
thinking ignore `off` / `none`.

Library models include `metadata.reasoning` when the GGUF chat template
exposes a control: `style`, `efforts` (including `off` when thinking can be
turned off), `default_effort`, `can_disable`, and `can_preserve` (the
template can keep prior-turn reasoning via `--reasoning-preserve`).

Dedicated GGUF embedders (and rerankers) are detected on library scan
(`metadata.is_embedding` / `is_reranker`, optional `pooling_type`). Loading
them auto-enables embedding mode so the instance serves
`POST /v1/embeddings` rather than chat. They are omitted from the Chat model
picker; load from Library and call the Developer API.
