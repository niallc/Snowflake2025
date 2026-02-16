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
