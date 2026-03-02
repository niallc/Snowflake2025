# AGENTS.md

This file captures project-specific guidance for Codex agents working in this repository.

## Guidance Freshness
- Use timestamped entries in this file.
- Treat entries older than 3 months as lower-priority guidance unless still clearly applicable.
- Prefer newer entries when guidance conflicts.

## Entries

### 2026-02-16 - Temporary Characterization Tests
- Temporary characterization tests are allowed for local refactor safety.
- These tests are intentionally kept in git-ignored paths and should generally not be committed.
- Use them to validate behavior during refactors, then update/remove as needed.

### 2026-02-16 - Local Test Commitment Expectations
- Git-ignored tests are intentionally used for short-term regression checks after refactors.
- Do not treat untracked/ignored tests as a required deliverable unless the user explicitly asks to commit them.
- Prefer focused local validation and fail-fast runtime checks over broad permanent test expansion by default.

### 2026-02-16 - Compatibility and Fallback Policy
- This is a single-developer project with no external API compatibility requirements.
- Prefer one clear contract and fail-fast behavior over backward-compatibility shims.
- Do not add fallback paths for legacy field names/semantics unless explicitly requested.

### 2026-02-18 - model_config.py Churn During Training
- `hex_ai/inference/model_config.py` is expected to change frequently as training identifies new best models.
- Typical updates are append-only additions to `MODEL_GENERATIONS`.
- Do not treat these updates as unexpected or concerning during unrelated cleanup/refactor work.
- Ignore unrelated `model_config.py` changes unless the task explicitly requires editing that file.
- If hex_ai/inference/model_config.py is dirty and not part of the task, continue without pausing or asking; treat it as expected background churn.

### 2026-02-25 - Fail-Fast and Explicit Override Policy
- Prefer fail-fast behavior when restart/state assumptions are violated (missing files, incompatible state, unexpected invariants).
- Do not add silent fallbacks that continue execution with ambiguous semantics unless explicitly requested.
- If a non-fail-fast path is needed for exceptional recovery, gate it behind an explicit manual override and log it clearly.
- Prioritize surfacing potential bugs over masking them with automatic fallback behavior.

### 2026-02-27 - Virtualenv Activation Guard for `hex_ai`
- `hex_ai/__init__.py` enforces environment validation via `VIRTUAL_ENV`; calling `hex_ai_env/bin/python` directly may still fail if `VIRTUAL_ENV` is unset.
- Preferred invocation for scripts/tests: `source hex_ai_env/bin/activate && python ...` from the repository root.
- For subprocess launches that do not source a shell profile, set both `VIRTUAL_ENV=<repo>/hex_ai_env` and prepend `<repo>/hex_ai_env/bin` to `PATH`.
- If you see `ImportError: hex_ai requires hex_ai_env virtual environment`, treat it as an environment setup issue first (not a code/runtime bug).

### 2026-03-02 - Codex Sandbox vs GPU/MPS Availability
- Observed behavior on this machine: when launched inside Codex sandbox, PyTorch may report `torch.backends.mps.is_built() == True` but `torch.backends.mps.is_available() == False`, which causes normal auto-device logic to fall back to CPU.
- The same interpreter/venv launched from a normal terminal outside sandbox can report `mps_available == True` and use GPU/MPS without special tournament flags.
- There is no project-wide rule to always prefer one context; choose based on task safety/performance needs.
- If GPU usage needs to be explicit/verified for long runs, launch from a normal terminal (`source hex_ai_env/bin/activate`) and sanity-check with a short probe:
  - `python -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"`
