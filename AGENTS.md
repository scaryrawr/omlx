# oMLX agent guide

## Project overview

oMLX is a Python/FastAPI inference server for Apple Silicon Macs. It serves OpenAI-, Anthropic-, and Responses-compatible APIs over MLX models, with continuous batching, multi-model loading, VLM/audio/embedding/reranker engines, MCP tools, and tiered KV caching.

Keep durable shared guidance in this file. Add nested `AGENTS.md` files only for directories with meaningfully different rules, and avoid scoped Copilot instruction files unless they add targeted value beyond these repo-wide rules.

Key areas:

- `omlx/server.py` wires the FastAPI app, route handlers, auth, streaming responses, and server state.
- `omlx/cli.py`, `omlx/settings.py`, and `omlx/config.py` handle startup, persisted settings, CLI/env/file precedence, and user-visible defaults.
- `omlx/engine_pool.py` discovers models, selects engine types, tracks loaded engines, and enforces LRU/pinning behavior.
- `omlx/scheduler.py`, `omlx/engine/`, and `omlx/engine_core.py` own request scheduling and MLX generation.
- `omlx/cache/` owns paged, prefix, hot, SSD, hybrid, vision-feature, and recovery cache behavior. Treat cache serialization and MLX buffer lifetime carefully.
- `omlx/api/` contains API models, protocol adapters, route helpers, tool calling, thinking parsing, structured output, and SSE formatting.
- `omlx/adapter/` contains model-family output parsers and Harmony/Gemma-specific adaptation; keep model quirks there instead of in generic server paths.
- `omlx/admin/` contains the web admin dashboard routes and utilities; templates/static assets/i18n files are packaged as package data.
- `omlx/integrations/`, `omlx/mcp/`, and `omlx/eval/` cover editor/client setup, MCP execution, and evaluation helpers.
- `packaging/` builds the macOS menubar app bundle and DMG using venvstacks.

## Setup and validation

Use `uv run` for Python tooling and tests. If dependencies are missing, start with `uv sync --dev`.

- Default fast test run: `uv run pytest`
- Explicit fast test selection: `uv run pytest -m "not slow and not integration"`
- Narrow test run: `uv run pytest tests/test_config.py -v`
- Lint: `uv run ruff check .`
- Type check: `uv run mypy omlx`
- Format when needed: `uv run black .`
- macOS app build checks live under `packaging/`; see `packaging/README.md` before running `python build.py`.

`pytest.ini` defaults to verbose tests excluding `slow` and `integration`. Mark tests that require real model files with `@pytest.mark.slow`, and tests that require a live server with `@pytest.mark.integration`.

## Code conventions

- Keep the Apache 2.0 SPDX header at the top of Python source and test files.
- Prefer dataclasses for configuration and request/response state where surrounding code already uses them.
- Keep protocol-specific conversion in `omlx/api/` or `omlx/api/adapters/`; avoid mixing API wire-format logic into scheduler/cache internals.
- Preserve async boundaries: FastAPI handlers and engine orchestration are async, while MLX generation is isolated through scheduler/engine abstractions.
- Keep optional dependency behavior explicit. Extras such as `mcp`, `audio`, `grammar`, `image`, `modelscope`, and `paroquant` are intentionally separated; use availability helpers such as `omlx/utils/optional_deps.py` rather than importing heavy optional packages in core paths.
- For model-family patches under `omlx/patches/`, keep changes narrow and covered by focused regression tests. These files mirror upstream behavior and can be brittle across dependency updates.
- When changing cache, scheduler, streaming, adapters, or tool-call behavior, add focused tests for token accounting, finish reasons, cancellation/disconnect behavior, cache reuse/regression paths, and protocol output shape.

## Safety notes

- The project targets Apple Silicon and MLX. Avoid adding torch-heavy dependencies to core paths unless they are guarded by an optional extra.
- Do not enable Hugging Face `trust_remote_code` by default; tests assert the safer default.
- Do not swallow MLX, GPU synchronization, cache corruption, model-loading, or settings-persistence errors with broad fallbacks. Surface errors through existing exception types in `omlx/exceptions.py`, route-level HTTP error mapping, or explicit user-facing warnings consistent with nearby code.
- Do not run slow/integration/model-loading tests unless the task needs them and the environment has the required models or server.
- Be careful with commands that build the macOS app or DMG; they can be expensive and may write large artifacts under `packaging/build/` and `packaging/dist/`.
