# OpenInfer Studio

Desktop app for finding, downloading, configuring, and running GGUF models
locally with llama.cpp. Search Hugging Face, manage llama.cpp builds, load
models with an explicit memory estimate, chat with streaming and reasoning
controls, quantize (including OpenInfer Dynamic), and serve loaded models
through a local OpenAI-compatible API.

**Status:** 1.4.1. First-run setup, discover → download → load → chat →
quantize → serve works end to end. See *Known limitations* below.

## Features

**Get started.** A first-run wizard installs a llama.cpp runtime matched to
detected hardware (Vulkan, CUDA, HIP, or Metal). Theme follows the OS; dark
and light are available.

**Find and download.** Browse Hugging Face for GGUF repositories. Results are
grouped by quantization and split set, with tags for vision, audio, MTP,
embeddings, and speculative drafts. Downloads resume, support multiple
connections per file, and land in the library automatically. A Hugging Face
token (OS keychain only) unlocks gated or private repos.

**Library.** Import local GGUFs, register extra model directories, favorite
and rename models, and pin a runtime per model. Load uses a live memory
estimate (weights, KV cache, projector, draft, overhead) against VRAM and
RAM. Context, GPU offload, Flash Attention, KV cache type, speculative
decoding (MTP / EAGLE3 / DFlash / DSpark / draft-simple), embedding/rerank
mode, and expert llama.cpp flags are all available — only flags the selected
runtime advertises in `--help` are passed.

**Chat.** Streaming replies, conversation branches and regenerate, system
prompt, and per-chat parameters. Reasoning models expose template-native
effort (including off when the chat template allows it) and a thinking-token
budget. Dedicated embedders and rerankers stay out of the chat picker; load
them from the library and call the Developer API.

**Quantize.** Background jobs driven by the selected runtime’s
`llama-quantize` / `llama-imatrix`: named ftypes, importance matrices, and
**OpenInfer Dynamic** mixed-precision (OID-Q5/Q4/Q3/Q2_K_XL) with size
tiers, effort (`fast` / `profiled` / `deep`), and KLD gates against the
source. Convert a Hugging Face BF16/F16/F32 safetensors repo to GGUF, then
quantize. Pause keeps Dynamic checkpoints; resume continues from the last
stage.

**Serve.** An optional OpenAI-compatible endpoint (`/v1/models`,
chat/completions, completions, embeddings, responses) with its own API key.
Loopback by default; LAN bind is opt-in. Optional [HostIt](https://github.com/32bitx64bit/HostIt)
registration can expose that endpoint through a local agent.

**Operate.** Hardware report and backend recommendation, runtime install from
official llama.cpp releases or a custom binary/archive, redacted logs, and
classified load failures with the generated command and a retry/CPU fallback.

## Platforms

| Platform | Status | Notes |
|---|---|---|
| Linux x86_64 | primary | AppImage via `packaging/linux/build-appimage.sh x86_64` |
| Linux aarch64 | supported | AppImage via `packaging/linux/build-appimage.sh aarch64` (native arm64 host) |
| Windows x86_64 | supported | `.exe` installer + portable zip |
| macOS arm64 | supported | `.dmg` via `packaging/macos/build-bundle.sh arm64` |
| macOS x86_64 | supported | `.dmg` via `packaging/macos/build-bundle.sh x86_64` |

## Architecture

```
openinfer-studio (Qt 6 / QML)
    │  REST + WebSocket, loopback only
    ▼
openinfer-core (Go)
    │  one managed process per loaded model
    ▼
llama-server (official llama.cpp builds, pinable per model)
```

- **C++ bootstrap** launches the backend and loads QML — no app logic.
- **Go backend** owns Hugging Face browsing, downloads, runtimes, process
  supervision, chat, quantization, convert, and the OpenAI-compatible proxy.
- **QML** is the UI; it talks to the backend over an authenticated local API.

Control API reference: [`docs/api.md`](docs/api.md).

## Build

Requirements: Go ≥ 1.26, CMake ≥ 3.24, Qt ≥ 6.5 (Quick, QuickControls2,
WebSockets, Widgets), C++17.

```bash
./scripts/build.sh          # debug → build/
./scripts/build.sh release  # optimized
./scripts/test.sh           # Go tests + backend self-test
./build/openinfer-studio    # run
```

Backend alone (development):

```bash
go run ./apps/core --token dev-token --port 0 --data-dir /tmp/oi-dev
```

## Data directories

| Platform | Path |
|---|---|
| Linux | `~/.local/share/openinfer-studio` (config `~/.config/…`, cache `~/.cache/…`) |
| Windows | `%LOCALAPPDATA%\OpenInfer Studio` |
| macOS | `~/Library/Application Support/OpenInfer Studio` |

Contains `database/`, `runtimes/`, `models/`, `downloads/`, `cache/`, `logs/`,
`presets/`, `sessions/`, `temp/`. Nothing else is written without an explicit
user action.

## Privacy

Network use is limited to Hugging Face, llama.cpp release/runtime downloads,
and links you open. No telemetry, accounts, or cloud inference. Chats, prompts,
models, and hardware info stay local. Offline once models and runtimes are
installed.

The control API is loopback-only with a session token. The optional public
OpenAI-compatible server is a separate listener with its own key; LAN bind is
opt-in. Inference processes bind loopback with per-process keys. Hugging Face
tokens live in the OS keychain, never in logs or SQLite.

## Troubleshooting

- **Backend did not become ready** — run
  `openinfer-core --selftest --token t --data-dir /tmp/oi` and check
  `logs/application/core.log`.
- **Model fails to load** — the dialog shows the classified cause, generated
  command, and log tail. Retry with safe settings or CPU fallback from there.
- **Download stuck** — Downloads page shows resume state; partials under
  `downloads/partial/` resume automatically.
- **GPU unused** — Settings → Hardware for the detected backend, then install a
  matching runtime (Vulkan / CUDA / HIP / Metal).
- **Quantize job paused or interrupted** — Dynamic jobs resume from the last
  checkpoint. Classic `llama-quantize` restarts that tool.

## Known limitations

- Image and PDF chat attachments are not wired yet. Experimental **audio**
  attachments can be enabled under Settings → Experimental → Audio models
  (mirrors llama.cpp’s experimental libmtmd audio input; quality may vary —
  remains gated until upstream treats audio as stable).
- Hugging Face convert supports architectures llama.cpp can load as GGUF from
  BF16/F16/F32 safetensors. NVFP4 / GPTQ / AWQ and some layouts (MLA, packed
  Qwen3-Next, RWKV/Mamba, altup) fail closed. Vision weights are skipped, so
  the converted GGUF is language-only.

## License

OpenInfer Studio is free software licensed under the
[GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).

## Releases

Release notes: [`CHANGELOG.md`](CHANGELOG.md). Version lives in
`internal/version/VERSION`. Tag `vX.Y.Z` (matching that file) to trigger
`.github/workflows/release.yml`, which publishes:

- Linux: `OpenInferStudio-*-linux-x86_64.AppImage` and `OpenInferStudio-*-linux-aarch64.AppImage`
- Windows: `OpenInferStudio-*-windows-x86_64-setup.exe` (+ portable `.zip`)
- macOS: `OpenInferStudio-*-macos-arm64.dmg` and `OpenInferStudio-*-macos-x86_64.dmg`

Local packaging (on each OS):

```bash
./packaging/linux/build-appimage.sh x86_64
./packaging/linux/build-appimage.sh aarch64   # on an arm64 Linux host
./packaging/macos/build-bundle.sh arm64
./packaging/macos/build-bundle.sh x86_64
pwsh ./packaging/windows/build-installer.ps1
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
