# AGENTS.md — OpenInfer Studio

## What this is

OpenInfer Studio: an open-source desktop app for running GGUF models via
llama.cpp. Qt 6/QML frontend, Go backend, SQLite state, one managed
`llama-server` process per loaded model.

## Build / test

- `./scripts/build.sh` — backend + desktop
- `./scripts/test.sh` — all Go tests (root module + quantlab) + backend self-test
- `go test ./...` — Studio packages; `go test -C quantlab ./...` — Quantlab
- `gofmt -w . && go vet ./... && go vet -C quantlab ./...` before committing

## Hard rules

- C++ (`apps/desktop/main.cpp`) is a bootstrap only: launch backend, pass
  token/port to QML, load Main.qml, kill backend on exit. No logic.
- All logic in `internal/` (Go). UI in `apps/desktop/qml/`.
- No shell command strings anywhere; `exec.Command` argv only.
- Only pass llama.cpp flags the selected runtime advertises in `--help`
  (see `internal/runtimes/capabilities.go`).
- Secrets (HF token, API keys) never in logs or SQLite; HF token lives in the
  OS keychain via `internal/auth/keychain.go`.
- Inference processes: loopback only, random port, per-process API key,
  process-group/Job-Object cleanup.
- SQLite schema changes = new file in `migrations/NNNN_name.sql`; never edit
  applied migrations.
- Event envelopes: `{version:1, event, timestamp, payload}`.
- App release version lives in `internal/version/VERSION` (embedded by Go,
  read by CMake/packaging). Bump that file to cut a release.

## Layout

- `apps/core` — backend main (+ watchdog, adapters)
- `internal/*` — api, auth, chat, config, database, diagnostics, downloads,
  gguf, hardware, huggingface, instances, models, processes, proxy, runtimes,
  storage
- `migrations/` — embedded SQL
- `tests/` — integration tests; `tests/fakeserver` is the fake llama-server
- `packaging/`, `scripts/`, `docs/` (control API: `docs/api.md`)

## Data dirs

Linux `~/.local/share/openinfer-studio`, macOS `~/Library/Application
Support/OpenInfer Studio`, Windows `%LOCALAPPDATA%\OpenInfer Studio`.
