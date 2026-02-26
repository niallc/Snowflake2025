"""
Move provenance sidecar helpers for self-play and TRMPH preprocessing.

Schema v1:
- One JSON object per line
- Required fields:
  - schema_version (int, must be 1)
  - game_index (int, 0-based)
  - move_count (int, number of played moves in game)
  - move_codes (str, length == move_count; chars in V/G/C/T)
  - policy_train_mask (str, length == move_count; chars in 0/1)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Union


MOVE_PROVENANCE_SCHEMA_VERSION = 1

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


@dataclass(frozen=True)
class MoveProvenanceRecord:
    schema_version: int
    game_index: int
    move_count: int
    move_codes: str
    policy_train_mask: str

    def __post_init__(self) -> None:
        context = "MoveProvenanceRecord"

        schema_version = _coerce_nonnegative_int(
            self.schema_version, "schema_version", context
        )
        if schema_version != MOVE_PROVENANCE_SCHEMA_VERSION:
            raise ValueError(
                f"{context}: schema_version must be {MOVE_PROVENANCE_SCHEMA_VERSION}, got {schema_version}"
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

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "game_index", game_index)
        object.__setattr__(self, "move_count", move_count)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "game_index": self.game_index,
            "move_count": self.move_count,
            "move_codes": self.move_codes,
            "policy_train_mask": self.policy_train_mask,
        }

    def to_json_line(self) -> str:
        # Sorted keys keeps sidecar deterministic for easier diffs/debugging.
        return json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":"))

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
        )


def make_move_provenance_record(game_index: int, move_codes: str) -> MoveProvenanceRecord:
    """Create a schema-valid move provenance record for one game."""
    policy_train_mask = build_policy_train_mask(move_codes)
    return MoveProvenanceRecord(
        schema_version=MOVE_PROVENANCE_SCHEMA_VERSION,
        game_index=game_index,
        move_count=len(move_codes),
        move_codes=move_codes,
        policy_train_mask=policy_train_mask,
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


def load_move_provenance_sidecar(sidecar_path: PathLike) -> list[MoveProvenanceRecord]:
    """Load and validate all provenance records from a sidecar JSONL file."""
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
