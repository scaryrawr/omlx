# Instructions for oMLX

These rules are for coding agents that work in this repository. The guide for people is [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md). Read it before you open a pull request.

These rules apply to the lines you add or change. Older code in this repository does not always follow them, and it is not a model to copy.

## Commands

- `make dev` installs the server, the web UI and the dev tools, then builds the native custom kernels. On a Mac without full Xcode, use `make dev-no-kernels`.
- `python -m pytest tests/test_<area>.py` runs the tests for the code you changed. Run those first, then `python -m pytest`. The default run skips tests marked `slow` and `integration`.
- `make web` rebuilds the web UI CSS and normalizes the translation files. Run it after you edit a template, JavaScript or `apps/omlx-web/omlx_web/i18n/en.json`.
- `make app` builds the macOS app. Run it when you change `apps/omlx-mac/`.

## Layout

- `omlx/` is the server package. It runs without `apps/` and then serves the API only.
- `apps/omlx-web/` is the web UI. It reaches the server only through the admin API.
- `apps/omlx-mac/` is the SwiftUI app. It starts the server as a subprocess and uses the same admin API.
- `omlx/patches/` holds runtime patches for the pinned mlx-lm and mlx-vlm. A fix for their behavior goes there.

## Before you write code

- Read the callers and the tests of the code you change before you edit it.
- Find the cause first. Fix the place where the behavior first goes wrong, not the place where the symptom shows.
- Do not ship a fix for a cause you have not confirmed. A change that might help does not belong in the diff.
- Make the smallest change that fixes the cause. Reuse the code path that already exists.
- Do not fix a bug by turning a feature off, skipping it for one model, or adding a switch that disables it.
- Do not fix a bug by tuning a constant, such as a chunk size, a retry count, a timeout or a buffer size, unless you have confirmed that the constant is the cause.
- Do not add a setting, a fallback, a retry or a cooldown as extra defense around a fix. Do not add a guard for a state that cannot occur.
- Do not add an `OMLX_*` environment variable that switches behavior. If the user needs a toggle, make it a setting with an entry in the settings UI.
- Do not rewrite a hot path for a small gain.
- Do not remove an existing recovery or fallback branch before you read the commit that added it.
- Do not mix unrelated cleanup into the change.
- CONTRIBUTING.md asks for an issue or a discussion before a change over 1,000 lines or a new subsystem. Tell the user before you start one.

## Comments

- Do not add a comment that restates the code or explains your change. Keep a comment only for behavior that an external dependency forces, or for an invariant the code cannot check. Keep it to one or two lines.
- The comments already in a file are not a model to copy. Match the file in naming and idiom, not in comment density.
- Do not write "do not remove" or "keep in sync" in a comment. Enforce that constraint with a test or an assert.
- Do not mention the conversation, the request or the review in code, names or messages.
- Add a docstring only to a public function or class.
- Use ASCII punctuation in comments and docstrings.

## Code

- Put imports at the top of the file. An optional dependency goes there too, inside `try` and `except ImportError`.
- Validate input where it enters, such as an API request, a config file or the CLI. Inside, do not check again. Write no `None` check, `hasattr` or `getattr` default for a value the caller always sets.
- Do not catch an exception to hide it. Write no `except: pass`, and no `except Exception` around code that cannot raise.
- Do not use `# noqa`, `# type: ignore`, `cast` or `Any` to get a line you wrote past the linter or the type checker. Fix the line.
- Return early instead of nesting.
- Do not add a wrapper with one caller, a helper that only forwards a call, or a second copy of a helper that exists.
- When you replace an internal function, move its callers and delete the old one. Leave no compatibility layer and no parallel old and new path.
- Keep one rule or value in one place.
- Leave no `TODO` or `FIXME` marker, commented-out code, unused import, debug print or scratch file in the diff.
- Do not reformat lines you did not change.
- Do not copy code or text from another project unless its license allows it. Keep the license notice and add the source to an Acknowledgements section in the pull request.

## Tests

- A bug fix gets one regression test where a test is practical. It fails without the fix and passes with it.
- Put the test in the existing test file for that code, and reuse its fixtures and helpers. Do not create a test file when one already covers the module.
- Assert what a caller can observe, with the exact value. Do not assert only that something is set or was called, and do not copy the implementation into the test.
- Do not add tests for trivial behavior, or several cases that check the same thing.
- Do not skip, loosen or delete an existing test to make your change pass. Do not add `sleep` or retries.
- A bug fix that has to rework the test harness or restructure how modules import each other is too big. Look for a smaller one.

## Pull requests

- Search open pull requests for the same fix before you start. If one exists, tell the user and stop.
- Keep one problem in one pull request.
- A performance change needs numbers from `main` and from the change, on the same Mac, model and settings.
- A UI change needs before and after screenshots of the real screen.
- Do not commit work logs, personal test notes or benchmark scripts.

## Pull request text

Start the description from `.github/pull_request_template.md` and keep its headings.

The first section, `TL;DR`, belongs to the user.

- Never write this section.
- Never suggest wording for it, draft it for the user to edit, or rewrite what the user wrote.
- Never fill it from the diff, the commits or the rest of the description.
- You may translate the user's text into English, sentence for sentence. Add nothing and drop nothing. If the user wrote it in English, paste it unchanged.
- If the user gives you no text for this section, do not open the pull request.
- Never tick a box in the checklist. The user ticks them.

The maintainer closes a pull request without review when this section is missing, empty or written by an AI.

You may write the other sections. Under Validation, list what was run, with the Mac and the model, and what remains untested. Leave out test counts, lint results and CI status. Put numbers in a table and add no diagram.

### Examples

User: Open the PR for me.
Agent: I need your TL;DR first. Write one to three sentences on what this change fixes and why. Any language works, and I can translate it. This project does not accept an AI-written TL;DR.

User: Just write the TL;DR, I will check it.
Agent: I cannot write or suggest that section. Tell me in your own words and I will put your text in.

User: 캐시 히트인데 TTFT가 두 배로 늘어나는 문제를 고쳤습니다.
Agent: Here is your text in English, sentence for sentence. "I fixed the problem where TTFT doubles on a cache hit." I added nothing.

User: Make my TL;DR sound better.
Agent: I cannot edit that section. I can translate it as written, or you can change it yourself.

## Fork maintenance and project overview

oMLX is a Python/FastAPI inference server for Apple Silicon Macs. It serves OpenAI-, Anthropic-, and Responses-compatible APIs over MLX models, with continuous batching, multi-model loading, VLM/audio/embedding/reranker engines, MCP tools, and tiered KV caching.

Keep durable shared guidance in this file. Add nested `AGENTS.md` files only for directories with meaningfully different rules, and avoid scoped Copilot instruction files unless they add targeted value beyond these repo-wide rules.

This is a fork maintained by rebasing onto `upstream/main`; its history is intentionally not kept stable. Squashing/rewriting the fork's commits since the `upstream/main` merge-base and force-pushing (`git push --force-with-lease`) is acceptable when cleaning up. Always create a `backup/*` tag before a destructive rewrite, and verify the rewrite changed no code (`git diff <backup> HEAD` is empty) before pushing. When regrouping heavily interleaved commits (features scattered among dependency bumps and rebase-cleanup chores), prefer `git reset --soft <merge-base>` then re-stage per file into logical commits over an interactive reorder; the soft-reset approach keeps the final tree byte-identical and avoids reorder conflicts on shared files such as `pyproject.toml`, `README.md`, and `AGENTS.md`. The regroup pattern that stays byte-identical is a stable file-to-group mapping (image engine/registry/routes + admin Imagine + i18n; audio/video API + VLM; server/admin-chat; upstream-rebase reconciliation; git-pinned deps; Copilot integration). Because `git reset --soft`/mixed reset does NOT scrub the working tree, run the regroup only from a verified-clean checkout of the original tip (`git status` empty and `git rev-parse HEAD^{tree}` equal to the backup tree); a dirty tree left over from diagnostics silently commits stale file content and breaks tree identity. If a run drifts, `git reset --hard <backup>` and re-run rather than patching in place. **Never use `git stash` during a merge** — it can discard `MERGE_HEAD` and break the index, leaving you stranded without merge state or staged conflict resolutions. Pushing requires the `scaryrawr` account: the default git credential helper (gcm-core / a `copilot` host helper) can resolve to a different account and 403; push with `gh auth switch --user scaryrawr` active plus an inline helper override, e.g. `git -c credential.helper= -c 'credential.https://github.com.helper=!gh auth git-credential' push --force-with-lease origin main`, then unset any local `credential.helper` entries you added.

Key areas:

- `omlx/server.py` wires the FastAPI app, route handlers, auth, streaming responses, and server state.
- `omlx/cli.py`, `omlx/settings.py`, and `omlx/config.py` handle startup, persisted settings, CLI/env/file precedence, and user-visible defaults.
- `omlx/engine_pool.py` discovers models, selects engine types, tracks loaded engines, and enforces LRU/pinning behavior.
- `omlx/scheduler.py`, `omlx/engine/`, and `omlx/engine_core.py` own request scheduling and MLX generation.
- `omlx/utils/model_loading.py` and `omlx/model_discovery.py` handle model loading, tokenizer setup, model classification, and image-manifest discovery. Keep `model_discovery.py`'s image exports aligned with `engine_pool.py`; validate `import omlx.server` after discovery refactors so missing cross-module exports fail fast.
- `omlx/cache/` owns paged, prefix, hot, SSD, hybrid, vision-feature, and recovery cache behavior. Treat cache serialization and MLX buffer lifetime carefully.
- `omlx/api/` contains API models, protocol adapters, route helpers, tool calling, thinking parsing, structured output, and SSE formatting.
- `omlx/adapter/` contains model-family output parsers and Harmony/Gemma-specific adaptation; keep model quirks there instead of in generic server paths.
- `omlx/admin/` contains the admin HTTP API and utilities; `apps/omlx-web/omlx_web/` owns browser routes, templates, static assets, and i18n files, packaged in the same wheel. Keep web rendering independent of `omlx` imports through `WebUIHost`.
- `omlx/integrations/`, `omlx/mcp/`, and `omlx/eval/` cover editor/client setup, MCP execution, and evaluation helpers.
- `packaging/` builds the macOS menubar app bundle and DMG using venvstacks.

## Setup and validation

Use `uv run` for Python tooling and tests. If dependencies are missing, start with `uv sync --dev`. Pin validation commands to Python 3.12 (for example, `UV_PYTHON=python3.12 uv run --python python3.12 pytest ...`): `onnxruntime==1.20.1` does not support CPython 3.14. Rebuild local custom kernels after MLX upgrades with `OMLX_WITH_CUSTOM_KERNEL=1 uv pip install --python .venv/bin/python --reinstall --no-deps -e .`. Native floating-point QMM wrappers must match MLX's template parameters and global-scale argument, even when global scaling is disabled.

`uv.lock` is intentionally ignored in this repository; dependency pin changes should be tracked in `pyproject.toml` unless a task explicitly changes the lockfile policy.

When bumping git-sourced dependencies, audit `[project].dependencies`, `[project.optional-dependencies]`, and `tool.uv.override-dependencies`; compare the upstream dependency metadata at the new commit (`pyproject.toml`, `requirements.txt`, or equivalent) for changed version constraints that must be mirrored locally; also check mirrors in `packaging/build.py`, `omlx/admin/routes.py` engine metadata, `Formula/omlx.rb`, `tests/test_admin_engine_info.py`, and user-facing install examples such as `README.md`. Some packaging paths intentionally duplicate pins from `pyproject.toml`; if an optional extra pin changes, run targeted validation with that extra enabled (for example `uv sync --dev --extra audio` for `mlx-audio`). For `mlx-lm` bumps, validate the scheduler and MTP BatchGenerator integration against upstream API shape changes such as stop matching (`StopSequenceMatcher` / `stop_matchers` in current pins, replacing older `SequenceStateMachine` / `state_machines`) and `GenerationBatch.Response` fields; keep every MTP response construction routed through `_generation_response`, including unfinished singleton emissions. For `mlx-vlm` bumps, smoke Qwen3.5 dense and MoE VLM generation when available: model loading alone does not exercise oMLX patches whose upstream helper APIs or call signatures may have changed. When native MTP is enabled, include a multi-token verify/rejection cycle and confirm rollback states are returned without cache-rebuild warnings; a one-token smoke does not exercise the dedicated speculative verifier.

- Default fast test run: `uv run pytest`
- Explicit fast test selection: `uv run pytest -m "not slow and not integration"`
- Narrow test run: `uv run pytest tests/test_config.py -v`
- Lint: `uv run ruff check .` is the intended Python lint command, but the current tree may have pre-existing full-repo ruff diagnostics; when a task is not a lint cleanup, prefer `uv run ruff check <touched Python paths>` plus focused tests and call out any baseline failures separately. Do not include non-Python files such as `README.md` or `Formula/*.rb` in Ruff path lists; use `git diff --check` for whitespace on mixed-file changes. Ruff's SIM118 suggestion is unsafe for `safetensors.safe_open` handles in this environment; keep `f.keys()` there and add a targeted `# noqa: SIM118` if needed.
- Type check: `uv run mypy omlx`
- Format when needed: `uv run black .`
- macOS app build checks live under `packaging/`; see `packaging/README.md` before running `python build.py`.

`pytest.ini` defaults to verbose tests excluding `slow` and `integration`. Mark tests that require real model files with `@pytest.mark.slow`, and tests that require a live server with `@pytest.mark.integration`.

For manual server smoke tests, remember `omlx serve` persists non-default CLI flags to `~/.omlx/settings.json`. Prefer a disposable base path such as `--base-path "$(mktemp -d /tmp/omlx-smoke.XXXXXX)"` for temporary runs with flags such as `--model-dir`, `--memory-guard`, or `--max-concurrent-requests`; use `--memory-guard aggressive` for light smoke runs rather than disabling the guard with `off`. If the Homebrew service is already running, prefer an isolated temporary server on an alternate port rather than stopping the service unless you specifically need to validate the service instance. Treat a temporary server as a persistent readiness-gated process, not a bounded command: inspect startup output immediately, probe its health or target endpoint as soon as it reports ready, run the dependent smoke request, and then stop it; never wait for the server process to exit or for a completion notification. When `brew services` is slow or hangs, manage the service directly with the user launch agent label/path (`homebrew.mxcl.omlx`, `~/Library/LaunchAgents/homebrew.mxcl.omlx.plist`) via `launchctl list`, `launchctl bootout`, and `launchctl bootstrap`. For unauthenticated smoke servers, unset `OMLX_API_KEY` before launching; if auth is enabled, the generated key is stored under `auth.api_key` in that temporary settings file. Restart any stopped Homebrew service afterward, and verify it on the configured `server.port` from `~/.omlx/settings.json` rather than assuming the default 8000. Query `/v1/models` before smoke requests and use the returned IDs; models stored under nested `owner/model` directories may be served by basename. For Responses API smoke tests, validate nested `.output[].content[].text` in addition to any convenience `.output_text` field. For lightweight image-generation smoke tests with local mlx-vlm models, use the core mlx-vlm dependency and at least two steps because `steps=1` can fail inside backend progress math rather than validating route behavior. For local image-edit smoke tests with large inputs, prefer multipart `/v1/images/edits` uploads (`-F image=@...`, plus `prompt`, `model`, `steps`, `guidance`, and `image_strength` fields) over JSON data URIs to avoid shell/base64 argument-size limits.

## Code conventions

- Keep the Apache 2.0 SPDX header at the top of Python source and test files.
- Prefer dataclasses for configuration and request/response state where surrounding code already uses them.
- Keep protocol-specific conversion in `omlx/api/` or `omlx/api/adapters/`; avoid mixing API wire-format logic into scheduler/cache internals.
- Preserve async boundaries: FastAPI handlers and engine orchestration are async, while MLX generation is isolated through scheduler/engine abstractions.
- Keep optional dependency behavior explicit. Extras such as `mcp`, `audio`, `grammar`, `image`, `modelscope`, and `paroquant` are intentionally separated; use availability helpers such as `omlx/utils/optional_deps.py` rather than importing heavy optional packages in core paths.
- For Copilot CLI integration launches, keep `COPILOT_OFFLINE=true` so subagents do not fall back to GitHub-hosted services. The launcher persists the omlx provider into `~/.copilot/providers.json` (honoring `COPILOT_PROVIDERS_CONFIG`): merge into the existing file, clobber only entries whose provider name/id is `omlx`, preserve all foreign providers/models and unknown top-level keys, keep the file `0600`, and back up before rewriting (skip the write when content is unchanged).
- When adding admin UI i18n keys, update every file under `apps/omlx-web/omlx_web/i18n/*.json` (including `pt-BR.json`) and run focused tests or JSON parsing checks that cover missing locale keys.
- Admin templates ship with precompiled `apps/omlx-web/omlx_web/static/css/tailwind.css`; if you use new arbitrary Tailwind classes (for example `w-[...]`, `h-[...]`, `z-[...]`, or dynamic responsive grid classes), either rebuild and commit the CSS or prefer local semantic CSS in the template so installed pages do not lose layout.
- For image generation/edit defaults, keep request-level `steps`/`guidance`/`image_strength` overrides ahead of manifest `default_*` values, and manifest values ahead of `omlx/image_registry.py` family defaults; resolve family defaults through canonical base-model aliases so every supported alias gets the same behavior, and cover changes with focused `tests/test_image_engine.py` cases. Family-level defaults must be backed by mlx-vlm model configuration or manifests rather than subjective manual results.
- For mlx-vlm image editing, preserve each family’s documented input contract: FLUX.2 and Mage-Flow edit support multi-image requests, while Z-Image and ERNIE-Image require exactly one source image. Current mlx-vlm image families do not support masks. Route `image_strength` through request extras as both `strength` and `image_strength` so Z-Image and ERNIE-Image receive their respective names.
- For VLM/chat media handling, prefer `omlx/utils/image.py`'s `extract_media_from_messages()` in engine paths so image, audio, and video handling stays aligned; keep `extract_images_from_messages()` as compatibility-only unless a three-value API is specifically required.
- Keep Pixtral serving preprocessing MLX-resident via `return_tensors="mlx"` while preserving the image processor's default NumPy output for compatibility.
- For VLM speculative decoding, keep standalone MTP drafter artifacts such as `gemma4_assistant`, `gemma4_unified_assistant`, and `qwen3_5_mtp` out of normal model discovery; they are loaded through parent VLM drafter paths and are not chat-capable models.
- In SpecPrefill chunk selection, reduce chunk scores in one MLX operation and transfer the score vector once; per-chunk `.item()` calls serialize GPU work. Preserve deterministic Python tie ordering.
- For model-family patches under `omlx/patches/`, keep changes narrow and covered by focused regression tests. These files mirror upstream behavior and can be brittle across dependency updates.
- When a model-family patch changes checkpoint sanitization or quantization metadata, test a complete tiny synthetic checkpoint through the real loader with `strict=True`; helper-only conversion tests do not catch config normalization, module replacement, or unexpected/missing binding keys.
- For affine PLE dequantization, promote mixed floating-point scales and biases to their shared MLX result dtype before calling `mx.dequantize`; its Metal kernel reads both buffers using one scalar type. Keep MXFP scales as `uint8` and preserve the precision of checkpoint metadata.
- When changing cache, scheduler, streaming, adapters, or tool-call behavior, add focused tests for token accounting, finish reasons, cancellation/disconnect behavior, cache reuse/regression paths, and protocol output shape.
- Keep streaming `RequestOutput.output_token_ids` empty until the terminal output to avoid quadratic cumulative-list copies; streaming consumers should use `new_token_ids`, while terminal and non-streaming consumers receive the complete token sequence.

## Safety notes

- The project targets Apple Silicon and MLX. Avoid adding torch-heavy dependencies to core paths unless they are guarded by an optional extra.
- Do not enable Hugging Face `trust_remote_code` by default; tests assert the safer default.
- Do not swallow MLX, GPU synchronization, cache corruption, model-loading, or settings-persistence errors with broad fallbacks. Surface errors through existing exception types in `omlx/exceptions.py`, route-level HTTP error mapping, or explicit user-facing warnings consistent with nearby code.
- Do not run slow/integration/model-loading tests unless the task needs them and the environment has the required models or server.
- Be careful with commands that build the macOS app or DMG; they can be expensive and may write large artifacts under `packaging/build/` and `packaging/dist/`.
