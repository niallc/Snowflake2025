"""
Move provenance sidecar helpers for self-play and TRMPH preprocessing.

Authoritative semantics and end-to-end usage notes for policy targets live in:
  write_ups/search_policy_target_design_2026_03_04.md

Current policy-target source-code semantics:
  V: usually visit-count distribution target; currently also used for externally
     injected opening-book moves (stored as one-hot on the opening move)
  G: softmax(score_without_gumbel) on final scored/ranked subset
  C: policy-masked row (unused for policy loss)
  T: currently visit-count distribution in searched terminal-move cases; one-hot
     only in terminal-shortcut/no-visit cases

Schema v1:
- One JSON object per line
- Required fields:
  - schema_version (int, must be 1)
  - game_index (int, 0-based)
  - move_count (int, number of played moves in game)
  - move_codes (str, length == move_count; chars in V/G/C/T)
  - policy_train_mask (str, length == move_count; chars in 0/1)

Schema v2 (additive, backward-compatible with v1 readers that ignore unknown fields):
- All v1 required fields, plus:
  - policy_target_encoding (str; currently "dense_fp16_zlib_base64")
  - policy_targets_blob (str; base64-encoded, zlib-compressed fp16 matrix bytes)
  - policy_target_size (int; number of action logits per row, e.g. 169)
  - policy_target_source_codes (str, length == move_count; chars in V/G/C/T)
  - policy_target_version (int; target-construction contract version)
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union
import zlib

import numpy as np


MOVE_PROVENANCE_SCHEMA_VERSION = 1
MOVE_PROVENANCE_SCHEMA_VERSION_V2 = 2
MOVE_PROVENANCE_SCHEMA_VERSION_LATEST = MOVE_PROVENANCE_SCHEMA_VERSION_V2
SUPPORTED_MOVE_PROVENANCE_SCHEMA_VERSIONS = frozenset(
    {MOVE_PROVENANCE_SCHEMA_VERSION, MOVE_PROVENANCE_SCHEMA_VERSION_V2}
)

POLICY_TARGET_ENCODING_DENSE_FP16_ZLIB_BASE64 = "dense_fp16_zlib_base64"
VALID_POLICY_TARGET_ENCODINGS = frozenset(
    {POLICY_TARGET_ENCODING_DENSE_FP16_ZLIB_BASE64}
)

MOVE_CODE_VISIT_COUNT = "V"
MOVE_CODE_GUMBEL_ROOT = "G"
MOVE_CODE_CONFIDENCE_TERMINATION = "C"
MOVE_CODE_TERMINAL_TERMINATION = "T"

VALID_MOVE_CODES = frozenset(
    {
        MOVE_CODE_VISIT_COUNT,
        MOVE_CODE_GUMBEL_ROOT,
        MOVE_CODE_CONFIDENCE_TERMINATION,
        MOVE_CODE_TERMINAL_TERMINATION,
    }
)

POLICY_TRAINABLE_MOVE_CODES = frozenset(
    {
        MOVE_CODE_VISIT_COUNT,
        MOVE_CODE_GUMBEL_ROOT,
        MOVE_CODE_TERMINAL_TERMINATION,
    }
)

MASK_TRAIN = "1"
MASK_SKIP = "0"
VALID_POLICY_MASK_BITS = frozenset({MASK_TRAIN, MASK_SKIP})

PathLike = Union[str, Path]


def sidecar_path_for_trmph(trmph_path: PathLike) -> Path:
    """Return sidecar path for a .trmph file."""
    return Path(trmph_path).with_suffix(".provenance.jsonl")


def build_policy_train_mask(move_codes: str) -> str:
    """Build deterministic policy train mask from move codes."""
    if not isinstance(move_codes, str):
        raise TypeError(f"move_codes must be str, got {type(move_codes)}")
    invalid_codes = sorted(set(move_codes) - VALID_MOVE_CODES)
    if invalid_codes:
        raise ValueError(
            f"Invalid move codes {invalid_codes}. Expected subset of {sorted(VALID_MOVE_CODES)}."
        )
    return "".join(
        MASK_TRAIN if code in POLICY_TRAINABLE_MOVE_CODES else MASK_SKIP
        for code in move_codes
    )


def _coerce_nonnegative_int(value: Any, field_name: str, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: {field_name} must be int, got bool")
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field_name} must be int, got {type(value)}") from exc
    if coerced < 0:
        raise ValueError(f"{context}: {field_name} must be >= 0, got {coerced}")
    return coerced


def encode_policy_targets_blob(policy_targets: np.ndarray) -> tuple[str, int]:
    """
    Encode dense policy-target matrix for sidecar storage.

    The on-wire format is:
    - float16 row-major bytes
    - zlib compressed
    - base64 ASCII
    """
    arr = np.asarray(policy_targets, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"policy_targets must be 2D, got shape {getattr(arr, 'shape', None)}"
        )
    if arr.shape[1] <= 0:
        raise ValueError(
            f"policy_targets second dimension must be positive, got shape {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("policy_targets contains non-finite values")

    arr16 = np.ascontiguousarray(arr, dtype=np.float16)
    raw = arr16.tobytes(order="C")
    compressed = zlib.compress(raw, level=6)
    blob = base64.b64encode(compressed).decode("ascii")
    return blob, int(arr16.shape[1])


def decode_policy_targets_blob(
    blob: str,
    *,
    move_count: int,
    policy_target_size: int,
    encoding: str,
) -> np.ndarray:
    """Decode policy-target matrix from sidecar blob into float32 array."""
    if encoding not in VALID_POLICY_TARGET_ENCODINGS:
        raise ValueError(
            f"Unsupported policy_target_encoding {encoding!r}; "
            f"expected one of {sorted(VALID_POLICY_TARGET_ENCODINGS)}"
        )
    if encoding != POLICY_TARGET_ENCODING_DENSE_FP16_ZLIB_BASE64:
        raise RuntimeError(
            f"Encoding {encoding!r} is declared valid but has no decoder implementation."
        )
    if not isinstance(blob, str):
        raise TypeError(f"policy_targets_blob must be str, got {type(blob)}")
    if not blob:
        raise ValueError("policy_targets_blob must be non-empty")
    move_count_int = _coerce_nonnegative_int(
        move_count, "move_count", "decode_policy_targets_blob"
    )
    policy_target_size_int = _coerce_nonnegative_int(
        policy_target_size, "policy_target_size", "decode_policy_targets_blob"
    )
    if policy_target_size_int <= 0:
        raise ValueError(
            f"decode_policy_targets_blob: policy_target_size must be > 0, got {policy_target_size_int}"
        )

    try:
        compressed = base64.b64decode(blob.encode("ascii"), validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("policy_targets_blob is not valid base64-encoded data") from exc

    try:
        raw = zlib.decompress(compressed)
    except zlib.error as exc:
        raise ValueError("policy_targets_blob is not valid zlib-compressed payload") from exc

    arr16 = np.frombuffer(raw, dtype=np.float16)
    expected_size = move_count_int * policy_target_size_int
    if arr16.size != expected_size:
        raise ValueError(
            "Decoded policy-target payload size mismatch: "
            f"expected {expected_size} float16 values, got {arr16.size}"
        )
    return arr16.astype(np.float32, copy=False).reshape(
        (move_count_int, policy_target_size_int)
    )


@dataclass(frozen=True)
class MoveProvenanceRecord:
    schema_version: int
    game_index: int
    move_count: int
    move_codes: str
    policy_train_mask: str
    policy_target_encoding: Optional[str] = None
    policy_targets_blob: Optional[str] = None
    policy_target_size: Optional[int] = None
    policy_target_source_codes: Optional[str] = None
    policy_target_version: Optional[int] = None

    def __post_init__(self) -> None:
        context = "MoveProvenanceRecord"

        schema_version = _coerce_nonnegative_int(
            self.schema_version, "schema_version", context
        )
        if schema_version not in SUPPORTED_MOVE_PROVENANCE_SCHEMA_VERSIONS:
            raise ValueError(
                f"{context}: unsupported schema_version {schema_version}; "
                f"expected one of {sorted(SUPPORTED_MOVE_PROVENANCE_SCHEMA_VERSIONS)}"
            )

        game_index = _coerce_nonnegative_int(self.game_index, "game_index", context)
        move_count = _coerce_nonnegative_int(self.move_count, "move_count", context)

        if not isinstance(self.move_codes, str):
            raise TypeError(f"{context}: move_codes must be str, got {type(self.move_codes)}")
        if not isinstance(self.policy_train_mask, str):
            raise TypeError(
                f"{context}: policy_train_mask must be str, got {type(self.policy_train_mask)}"
            )

        if len(self.move_codes) != move_count:
            raise ValueError(
                f"{context}: move_codes length {len(self.move_codes)} must equal move_count {move_count}"
            )
        if len(self.policy_train_mask) != move_count:
            raise ValueError(
                f"{context}: policy_train_mask length {len(self.policy_train_mask)} must equal move_count {move_count}"
            )

        invalid_codes = sorted(set(self.move_codes) - VALID_MOVE_CODES)
        if invalid_codes:
            raise ValueError(
                f"{context}: invalid move codes {invalid_codes}; expected subset of {sorted(VALID_MOVE_CODES)}"
            )

        invalid_mask_bits = sorted(set(self.policy_train_mask) - VALID_POLICY_MASK_BITS)
        if invalid_mask_bits:
            raise ValueError(
                f"{context}: invalid policy_train_mask bits {invalid_mask_bits}; expected only 0/1"
            )

        expected_mask = build_policy_train_mask(self.move_codes)
        if self.policy_train_mask != expected_mask:
            raise ValueError(
                f"{context}: policy_train_mask {self.policy_train_mask!r} does not match expected {expected_mask!r} from move_codes"
            )

        has_policy_targets_payload = any(
            field is not None
            for field in (
                self.policy_target_encoding,
                self.policy_targets_blob,
                self.policy_target_size,
                self.policy_target_source_codes,
                self.policy_target_version,
            )
        )
        if schema_version == MOVE_PROVENANCE_SCHEMA_VERSION:
            if has_policy_targets_payload:
                raise ValueError(
                    f"{context}: schema_version {MOVE_PROVENANCE_SCHEMA_VERSION} "
                    "cannot include policy-target payload fields"
                )
        elif schema_version == MOVE_PROVENANCE_SCHEMA_VERSION_V2:
            missing_v2_fields = [
                name
                for name, value in (
                    ("policy_target_encoding", self.policy_target_encoding),
                    ("policy_targets_blob", self.policy_targets_blob),
                    ("policy_target_size", self.policy_target_size),
                    ("policy_target_source_codes", self.policy_target_source_codes),
                    ("policy_target_version", self.policy_target_version),
                )
                if value is None
            ]
            if missing_v2_fields:
                raise ValueError(
                    f"{context}: schema_version {MOVE_PROVENANCE_SCHEMA_VERSION_V2} "
                    f"missing required v2 fields: {missing_v2_fields}"
                )

            encoding = str(self.policy_target_encoding)
            if encoding not in VALID_POLICY_TARGET_ENCODINGS:
                raise ValueError(
                    f"{context}: invalid policy_target_encoding {encoding!r}; "
                    f"expected one of {sorted(VALID_POLICY_TARGET_ENCODINGS)}"
                )

            if not isinstance(self.policy_targets_blob, str):
                raise TypeError(
                    f"{context}: policy_targets_blob must be str, got {type(self.policy_targets_blob)}"
                )
            if not self.policy_targets_blob:
                raise ValueError(f"{context}: policy_targets_blob must be non-empty")

            policy_target_size = _coerce_nonnegative_int(
                self.policy_target_size, "policy_target_size", context
            )
            if policy_target_size <= 0:
                raise ValueError(
                    f"{context}: policy_target_size must be > 0, got {policy_target_size}"
                )

            if not isinstance(self.policy_target_source_codes, str):
                raise TypeError(
                    f"{context}: policy_target_source_codes must be str, got {type(self.policy_target_source_codes)}"
                )
            if len(self.policy_target_source_codes) != move_count:
                raise ValueError(
                    f"{context}: policy_target_source_codes length {len(self.policy_target_source_codes)} "
                    f"must equal move_count {move_count}"
                )
            invalid_source_codes = sorted(
                set(self.policy_target_source_codes) - VALID_MOVE_CODES
            )
            if invalid_source_codes:
                raise ValueError(
                    f"{context}: invalid policy_target_source_codes values {invalid_source_codes}; "
                    f"expected subset of {sorted(VALID_MOVE_CODES)}"
                )

            policy_target_version = _coerce_nonnegative_int(
                self.policy_target_version, "policy_target_version", context
            )

            object.__setattr__(self, "policy_target_encoding", encoding)
            object.__setattr__(self, "policy_target_size", policy_target_size)
            object.__setattr__(self, "policy_target_version", policy_target_version)

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "game_index", game_index)
        object.__setattr__(self, "move_count", move_count)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "game_index": self.game_index,
            "move_count": self.move_count,
            "move_codes": self.move_codes,
            "policy_train_mask": self.policy_train_mask,
        }
        if self.schema_version >= MOVE_PROVENANCE_SCHEMA_VERSION_V2:
            payload["policy_target_encoding"] = self.policy_target_encoding
            payload["policy_targets_blob"] = self.policy_targets_blob
            payload["policy_target_size"] = self.policy_target_size
            payload["policy_target_source_codes"] = self.policy_target_source_codes
            payload["policy_target_version"] = self.policy_target_version
        return payload

    def to_json_line(self) -> str:
        # Sorted keys keeps sidecar deterministic for easier diffs/debugging.
        return json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":"))

    def has_policy_targets(self) -> bool:
        return self.schema_version >= MOVE_PROVENANCE_SCHEMA_VERSION_V2

    def decode_policy_targets(self) -> Optional[np.ndarray]:
        if not self.has_policy_targets():
            return None
        if self.policy_target_encoding is None:
            raise RuntimeError("policy_target_encoding unexpectedly missing on v2 record")
        if self.policy_targets_blob is None:
            raise RuntimeError("policy_targets_blob unexpectedly missing on v2 record")
        if self.policy_target_size is None:
            raise RuntimeError("policy_target_size unexpectedly missing on v2 record")
        return decode_policy_targets_blob(
            self.policy_targets_blob,
            move_count=self.move_count,
            policy_target_size=self.policy_target_size,
            encoding=self.policy_target_encoding,
        )

    def validate_policy_targets_payload(self) -> None:
        """
        Validate v2 payload integrity by decoding blob shape/encoding checks.

        Callers that immediately decode policy targets can skip this to avoid
        duplicated decode work.
        """
        if not self.has_policy_targets():
            return
        if self.policy_target_encoding is None:
            raise RuntimeError("policy_target_encoding unexpectedly missing on v2 record")
        if self.policy_targets_blob is None:
            raise RuntimeError("policy_targets_blob unexpectedly missing on v2 record")
        if self.policy_target_size is None:
            raise RuntimeError("policy_target_size unexpectedly missing on v2 record")
        decode_policy_targets_blob(
            self.policy_targets_blob,
            move_count=self.move_count,
            policy_target_size=self.policy_target_size,
            encoding=self.policy_target_encoding,
        )

    def with_game_index(self, game_index: int) -> "MoveProvenanceRecord":
        """Clone record with rewritten game_index while preserving payload."""
        return MoveProvenanceRecord(
            schema_version=self.schema_version,
            game_index=game_index,
            move_count=self.move_count,
            move_codes=self.move_codes,
            policy_train_mask=self.policy_train_mask,
            policy_target_encoding=self.policy_target_encoding,
            policy_targets_blob=self.policy_targets_blob,
            policy_target_size=self.policy_target_size,
            policy_target_source_codes=self.policy_target_source_codes,
            policy_target_version=self.policy_target_version,
        )

    def equivalent_for_game_content(self, other: "MoveProvenanceRecord") -> bool:
        """
        Return whether two records are equivalent for game-level dedupe semantics.

        Intentionally ignores optional v2 payload fields so duplicate-game handling
        remains aligned with pre-v2 behavior (move-level provenance only).
        """
        if not isinstance(other, MoveProvenanceRecord):
            return False
        return (
            self.move_count == other.move_count
            and self.move_codes == other.move_codes
            and self.policy_train_mask == other.policy_train_mask
        )

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any], context: str) -> "MoveProvenanceRecord":
        required_fields = {
            "schema_version",
            "game_index",
            "move_count",
            "move_codes",
            "policy_train_mask",
        }
        missing_fields = sorted(required_fields - set(payload.keys()))
        if missing_fields:
            raise ValueError(f"{context}: missing required fields: {missing_fields}")
        return cls(
            schema_version=payload["schema_version"],
            game_index=payload["game_index"],
            move_count=payload["move_count"],
            move_codes=payload["move_codes"],
            policy_train_mask=payload["policy_train_mask"],
            policy_target_encoding=payload.get("policy_target_encoding"),
            policy_targets_blob=payload.get("policy_targets_blob"),
            policy_target_size=payload.get("policy_target_size"),
            policy_target_source_codes=payload.get("policy_target_source_codes"),
            policy_target_version=payload.get("policy_target_version"),
        )


def make_move_provenance_record(
    game_index: int,
    move_codes: str,
    *,
    policy_targets: Optional[np.ndarray] = None,
    policy_target_source_codes: Optional[str] = None,
    policy_target_version: Optional[int] = None,
) -> MoveProvenanceRecord:
    """Create a schema-valid move provenance record for one game."""
    policy_train_mask = build_policy_train_mask(move_codes)
    schema_version = MOVE_PROVENANCE_SCHEMA_VERSION
    policy_target_encoding = None
    policy_targets_blob = None
    policy_target_size = None
    source_codes = None
    target_version = None

    if policy_targets is not None:
        targets_arr = np.asarray(policy_targets, dtype=np.float32)
        if targets_arr.ndim != 2:
            raise ValueError(
                f"policy_targets must be 2D when provided, got shape {getattr(targets_arr, 'shape', None)}"
            )
        if targets_arr.shape[0] != len(move_codes):
            raise ValueError(
                "policy_targets row count must equal move count "
                f"(got {targets_arr.shape[0]} vs {len(move_codes)})"
            )

        if policy_target_source_codes is None:
            source_codes = move_codes
        else:
            if not isinstance(policy_target_source_codes, str):
                raise TypeError(
                    "policy_target_source_codes must be str when provided, "
                    f"got {type(policy_target_source_codes)}"
                )
            if len(policy_target_source_codes) != len(move_codes):
                raise ValueError(
                    "policy_target_source_codes length must match move count "
                    f"(got {len(policy_target_source_codes)} vs {len(move_codes)})"
                )
            invalid_source_codes = sorted(
                set(policy_target_source_codes) - VALID_MOVE_CODES
            )
            if invalid_source_codes:
                raise ValueError(
                    "policy_target_source_codes contains invalid values: "
                    f"{invalid_source_codes}"
                )
            source_codes = policy_target_source_codes

        policy_targets_blob, policy_target_size = encode_policy_targets_blob(targets_arr)
        policy_target_encoding = POLICY_TARGET_ENCODING_DENSE_FP16_ZLIB_BASE64
        target_version = (
            1 if policy_target_version is None else int(policy_target_version)
        )
        if target_version < 0:
            raise ValueError(
                f"policy_target_version must be >= 0, got {target_version}"
            )
        schema_version = MOVE_PROVENANCE_SCHEMA_VERSION_V2
    else:
        if policy_target_source_codes is not None:
            raise ValueError(
                "policy_target_source_codes cannot be provided when policy_targets is None"
            )
        if policy_target_version is not None:
            raise ValueError(
                "policy_target_version cannot be provided when policy_targets is None"
            )

    return MoveProvenanceRecord(
        schema_version=schema_version,
        game_index=game_index,
        move_count=len(move_codes),
        move_codes=move_codes,
        policy_train_mask=policy_train_mask,
        policy_target_encoding=policy_target_encoding,
        policy_targets_blob=policy_targets_blob,
        policy_target_size=policy_target_size,
        policy_target_source_codes=source_codes,
        policy_target_version=target_version,
    )


def parse_move_provenance_line(
    line: str, *, source: PathLike, line_number: int
) -> MoveProvenanceRecord:
    """Parse one JSONL line into a validated provenance record."""
    normalized = line.strip()
    if not normalized:
        raise ValueError(f"{source}: blank provenance line at {line_number}")

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{source}: invalid JSON at provenance line {line_number}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"{source}: provenance line {line_number} must decode to an object, got {type(payload)}"
        )

    context = f"{source}: provenance line {line_number}"
    return MoveProvenanceRecord.from_json_dict(payload, context=context)


def load_move_provenance_sidecar(
    sidecar_path: PathLike,
    *,
    validate_policy_targets_payload: bool = True,
) -> list[MoveProvenanceRecord]:
    """
    Load provenance records from a sidecar JSONL file.

    Args:
        sidecar_path: Sidecar JSONL path.
        validate_policy_targets_payload: If True, decode-validate each v2 blob
            during load. Set False when callers immediately decode each record
            anyway to avoid duplicate decode work.
    """
    path = Path(sidecar_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required move provenance sidecar: {path}"
        )
    if not path.is_file():
        raise ValueError(f"Provenance sidecar path is not a file: {path}")

    records: list[MoveProvenanceRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = parse_move_provenance_line(
                line, source=path, line_number=line_number
            )
            if validate_policy_targets_payload and record.has_policy_targets():
                record.validate_policy_targets_payload()
            records.append(record)

    return records


def write_move_provenance_sidecar(
    sidecar_path: PathLike, records: Iterable[MoveProvenanceRecord]
) -> None:
    """Write validated records to sidecar JSONL file."""
    path = Path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            if not isinstance(record, MoveProvenanceRecord):
                raise TypeError(
                    f"Expected MoveProvenanceRecord, got {type(record)}"
                )
            handle.write(record.to_json_line())
            handle.write("\n")
