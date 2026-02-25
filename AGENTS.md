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

### 2026-02-25 - Fail-Fast and Explicit Override Policy
- Prefer fail-fast behavior when restart/state assumptions are violated (missing files, incompatible state, unexpected invariants).
- Do not add silent fallbacks that continue execution with ambiguous semantics unless explicitly requested.
- If a non-fail-fast path is needed for exceptional recovery, gate it behind an explicit manual override and log it clearly.
- Prioritize surfacing potential bugs over masking them with automatic fallback behavior.
