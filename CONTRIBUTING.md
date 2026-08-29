# Contributing

## Ground rules

- Contributions are accepted under the project's
  [AGPL-3.0](LICENSE) license.
- The C++ bootstrap stays minimal. Logic belongs in Go or QML.
- `gofmt -w .`, `go vet ./...`, and `go vet -C quantlab ./...` must pass; add tests for new behavior.
- No telemetry, no hidden network calls, no new heavyweight dependencies.
- Do not invent llama.cpp capabilities: expose what the runtime's `--help`
  advertises.
- Keep user-facing errors honest: show the underlying error.

## Workflow

1. Open an issue or pick an open one.
2. Branch from `main`, keep changes focused.
3. Run `./scripts/test.sh` before opening a PR.
4. Update `docs/api.md` when endpoints change.

## Where things live

See `README.md` for build and layout, `AGENTS.md` for agent/contributor rules.
