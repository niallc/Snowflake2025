"""
Utilities for resumable run state persistence.

This module provides a small, reusable JSON state store with atomic writes and
config fingerprint validation for long-running scripts that need restart/resume
behavior.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


STATE_FILE_VERSION = 1


class RunStateError(RuntimeError):
    """Base error for run-state operations."""


class RunStateMismatchError(RunStateError):
    """Raised when an existing state file doesn't match the expected run config."""


def utc_now_iso() -> str:
    """Return current UTC time as an ISO 8601 string without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compute_config_fingerprint(config: Mapping[str, Any]) -> str:
    """
    Compute a stable hash for a run configuration.

    Args:
        config: Mapping with JSON-serializable run configuration values.

    Returns:
        SHA-256 hex digest of the canonicalized config.
    """
    canonical_json = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class JsonRunStateStore:
    """Persistent JSON state store with atomic writes."""

    def __init__(self, state_file: str | Path):
        self.state_file = Path(state_file)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load existing state, or return None if the file does not exist."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except json.JSONDecodeError as e:
            raise RunStateError(f"Invalid JSON in state file {self.state_file}: {e}") from e
        except OSError as e:
            raise RunStateError(f"Failed to read state file {self.state_file}: {e}") from e

        if not isinstance(state, dict):
            raise RunStateError(
                f"State file {self.state_file} must contain a top-level JSON object"
            )
        return state

    def save(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Save state atomically and return the saved payload with refreshed timestamp.

        Args:
            state: JSON-serializable state mapping.

        Returns:
            The saved state dictionary (copy of input with updated `updated_at`).
        """
        payload = dict(state)
        payload["updated_at"] = utc_now_iso()

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.state_file.with_suffix(f"{self.state_file.suffix}.tmp")

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True, default=str)
                f.write("\n")
            temp_file.replace(self.state_file)
        except OSError as e:
            raise RunStateError(f"Failed to write state file {self.state_file}: {e}") from e

        return payload

    def delete(self) -> None:
        """Delete the state file if it exists."""
        self.state_file.unlink(missing_ok=True)

    def create(
        self,
        *,
        run_type: str,
        config_snapshot: Mapping[str, Any],
        progress: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create and persist a new state record."""
        now = utc_now_iso()
        state = {
            "version": STATE_FILE_VERSION,
            "run_type": run_type,
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "config_fingerprint": compute_config_fingerprint(config_snapshot),
            "config_snapshot": dict(config_snapshot),
            "metadata": dict(metadata) if metadata else {},
            "progress": dict(progress),
        }
        return self.save(state)

    def assert_compatible(
        self,
        state: Mapping[str, Any],
        *,
        run_type: str,
        config_fingerprint: str,
    ) -> None:
        """Validate that an existing state file matches expected run context."""
        existing_run_type = state.get("run_type")
        if existing_run_type != run_type:
            raise RunStateMismatchError(
                f"State run_type mismatch: expected '{run_type}', found '{existing_run_type}'"
            )

        existing_fingerprint = state.get("config_fingerprint")
        if existing_fingerprint != config_fingerprint:
            raise RunStateMismatchError(
                "State configuration mismatch. Use matching options or pass a reset flag "
                "to start a fresh run state."
            )
