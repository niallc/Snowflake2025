"""Sidecar helpers for per-game ladder-certificate label sequences."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import zlib

import numpy as np

from hex_ai.data_processing import parse_trmph_line_flexible
from hex_ai.data_utils import load_trmph_file

from .sequence_labeler import build_ladder_certificate_label_sequence_for_trmph_game
from .library_loader import LoadedLadderTemplate

PathLike = str | Path

LADDER_CERTIFICATE_SIDECAR_SCHEMA_VERSION = 1
LADDER_CERTIFICATE_MAP_ENCODING_DENSE_U8_ZLIB_BASE64 = "dense_u8_zlib_base64"
VALID_LADDER_CERTIFICATE_ENCODINGS = frozenset(
    {LADDER_CERTIFICATE_MAP_ENCODING_DENSE_U8_ZLIB_BASE64}
)


def ladder_certificate_sidecar_path_for_trmph(trmph_path: PathLike) -> Path:
    """Return the default ladder-certificate sidecar path for a TRMPH file."""
    return Path(trmph_path).with_suffix(".ladder_certificates.jsonl")


def encode_ladder_certificate_maps_blob(maps: np.ndarray) -> str:
    """Encode dense ladder maps as zlib-compressed base64 uint8 bytes."""
    arr = np.asarray(maps, dtype=np.uint8)
    if arr.ndim != 4:
        raise ValueError(
            f"ladder certificate maps must be 4D, got shape {getattr(arr, 'shape', None)}"
        )
    raw = np.ascontiguousarray(arr).tobytes(order="C")
    compressed = zlib.compress(raw, level=6)
    return base64.b64encode(compressed).decode("ascii")


def decode_ladder_certificate_maps_blob(
    blob: str,
    *,
    position_count: int,
    plane_count: int,
    board_size: int,
    encoding: str,
) -> np.ndarray:
    """Decode dense ladder maps from sidecar storage into uint8 arrays."""
    if encoding not in VALID_LADDER_CERTIFICATE_ENCODINGS:
        raise ValueError(
            f"Unsupported ladder-certificate encoding {encoding!r}; "
            f"expected one of {sorted(VALID_LADDER_CERTIFICATE_ENCODINGS)}"
        )
    if encoding != LADDER_CERTIFICATE_MAP_ENCODING_DENSE_U8_ZLIB_BASE64:
        raise RuntimeError(
            f"Encoding {encoding!r} is declared valid but has no decoder implementation."
        )
    if not isinstance(blob, str) or not blob:
        raise ValueError("ladder-certificate blob must be a non-empty base64 string")

    try:
        compressed = base64.b64decode(blob.encode("ascii"), validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("ladder-certificate blob is not valid base64 data") from exc

    try:
        raw = zlib.decompress(compressed)
    except zlib.error as exc:
        raise ValueError("ladder-certificate blob is not valid zlib data") from exc

    expected_size = position_count * plane_count * board_size * board_size
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != expected_size:
        raise ValueError(
            "Decoded ladder-certificate payload size mismatch: "
            f"expected {expected_size} bytes, got {arr.size}"
        )
    return arr.reshape((position_count, plane_count, board_size, board_size))


@dataclass(frozen=True)
class LadderCertificateSidecarRecord:
    """One per-game ladder-certificate payload stored in JSONL sidecars."""

    schema_version: int
    game_index: int
    move_count: int
    board_size: int
    plane_names: tuple[str, ...]
    template_origin_encoding: str
    template_origin_blob: str
    carrier_encoding: str
    carrier_blob: str
    match_counts_by_position: tuple[int, ...]
    embeddings_considered_by_position: tuple[int, ...]
    elapsed_ms_by_position: tuple[float, ...]
    used_must_include_filter_by_position: str

    def __post_init__(self) -> None:
        context = "LadderCertificateSidecarRecord"
        if int(self.schema_version) != LADDER_CERTIFICATE_SIDECAR_SCHEMA_VERSION:
            raise ValueError(
                f"{context}: unsupported schema_version {self.schema_version}; "
                f"expected {LADDER_CERTIFICATE_SIDECAR_SCHEMA_VERSION}"
            )
        if int(self.game_index) < 0:
            raise ValueError(f"{context}: game_index must be >= 0, got {self.game_index}")
        if int(self.move_count) < 0:
            raise ValueError(f"{context}: move_count must be >= 0, got {self.move_count}")
        if int(self.board_size) <= 0:
            raise ValueError(f"{context}: board_size must be > 0, got {self.board_size}")
        if not self.plane_names:
            raise ValueError(f"{context}: plane_names must be non-empty")

        position_count = self.position_count
        if len(self.match_counts_by_position) != position_count:
            raise ValueError(
                f"{context}: match_counts_by_position length must equal position_count "
                f"({len(self.match_counts_by_position)} vs {position_count})"
            )
        if len(self.embeddings_considered_by_position) != position_count:
            raise ValueError(
                f"{context}: embeddings_considered_by_position length must equal position_count "
                f"({len(self.embeddings_considered_by_position)} vs {position_count})"
            )
        if len(self.elapsed_ms_by_position) != position_count:
            raise ValueError(
                f"{context}: elapsed_ms_by_position length must equal position_count "
                f"({len(self.elapsed_ms_by_position)} vs {position_count})"
            )
        if len(self.used_must_include_filter_by_position) != position_count:
            raise ValueError(
                f"{context}: used_must_include_filter_by_position length must equal position_count "
                f"({len(self.used_must_include_filter_by_position)} vs {position_count})"
            )
        invalid_filter_bits = sorted(
            set(self.used_must_include_filter_by_position) - {"0", "1"}
        )
        if invalid_filter_bits:
            raise ValueError(
                f"{context}: invalid filter bits {invalid_filter_bits}; expected only 0/1"
            )
        if self.template_origin_encoding not in VALID_LADDER_CERTIFICATE_ENCODINGS:
            raise ValueError(
                f"{context}: invalid template_origin_encoding {self.template_origin_encoding!r}"
            )
        if self.carrier_encoding not in VALID_LADDER_CERTIFICATE_ENCODINGS:
            raise ValueError(
                f"{context}: invalid carrier_encoding {self.carrier_encoding!r}"
            )
        if not isinstance(self.template_origin_blob, str) or not self.template_origin_blob:
            raise ValueError(f"{context}: template_origin_blob must be a non-empty string")
        if not isinstance(self.carrier_blob, str) or not self.carrier_blob:
            raise ValueError(f"{context}: carrier_blob must be a non-empty string")

    @property
    def position_count(self) -> int:
        return int(self.move_count) + 1

    @property
    def plane_count(self) -> int:
        return len(self.plane_names)

    def decode_template_origin_maps(self) -> np.ndarray:
        return decode_ladder_certificate_maps_blob(
            self.template_origin_blob,
            position_count=self.position_count,
            plane_count=self.plane_count,
            board_size=int(self.board_size),
            encoding=self.template_origin_encoding,
        )

    def decode_carrier_maps(self) -> np.ndarray:
        return decode_ladder_certificate_maps_blob(
            self.carrier_blob,
            position_count=self.position_count,
            plane_count=self.plane_count,
            board_size=int(self.board_size),
            encoding=self.carrier_encoding,
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": int(self.schema_version),
            "game_index": int(self.game_index),
            "move_count": int(self.move_count),
            "board_size": int(self.board_size),
            "plane_names": list(self.plane_names),
            "template_origin_encoding": self.template_origin_encoding,
            "template_origin_blob": self.template_origin_blob,
            "carrier_encoding": self.carrier_encoding,
            "carrier_blob": self.carrier_blob,
            "match_counts_by_position": list(self.match_counts_by_position),
            "embeddings_considered_by_position": list(
                self.embeddings_considered_by_position
            ),
            "elapsed_ms_by_position": list(self.elapsed_ms_by_position),
            "used_must_include_filter_by_position": self.used_must_include_filter_by_position,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict) -> "LadderCertificateSidecarRecord":
        if not isinstance(payload, dict):
            raise TypeError(
                f"LadderCertificateSidecarRecord payload must be a dict, got {type(payload)!r}"
            )
        return cls(
            schema_version=int(payload["schema_version"]),
            game_index=int(payload["game_index"]),
            move_count=int(payload["move_count"]),
            board_size=int(payload["board_size"]),
            plane_names=tuple(str(name) for name in payload["plane_names"]),
            template_origin_encoding=str(payload["template_origin_encoding"]),
            template_origin_blob=str(payload["template_origin_blob"]),
            carrier_encoding=str(payload["carrier_encoding"]),
            carrier_blob=str(payload["carrier_blob"]),
            match_counts_by_position=tuple(
                int(value) for value in payload["match_counts_by_position"]
            ),
            embeddings_considered_by_position=tuple(
                int(value) for value in payload["embeddings_considered_by_position"]
            ),
            elapsed_ms_by_position=tuple(
                float(value) for value in payload["elapsed_ms_by_position"]
            ),
            used_must_include_filter_by_position=str(
                payload["used_must_include_filter_by_position"]
            ),
        )

    @classmethod
    def from_json_line(cls, line: str) -> "LadderCertificateSidecarRecord":
        return cls.from_dict(json.loads(line))


def build_ladder_certificate_sidecar_record_for_trmph_game(
    *,
    game_index: int,
    trmph_text: str,
    templates: tuple[LoadedLadderTemplate, ...] | list[LoadedLadderTemplate],
    board_size: int | None = None,
    orientations: tuple[str, ...] = ("red_bottom", "blue_right"),
    allow_attacker_superset_on_empty: bool = True,
) -> LadderCertificateSidecarRecord:
    """Build one JSONL sidecar record from a TRMPH game string."""
    sequence = build_ladder_certificate_label_sequence_for_trmph_game(
        trmph_text,
        templates,
        board_size=board_size,
        orientations=orientations,
        allow_attacker_superset_on_empty=allow_attacker_superset_on_empty,
    )
    move_count = sequence.template_origin_maps.shape[0] - 1
    board_size_int = int(sequence.template_origin_maps.shape[-1])
    return LadderCertificateSidecarRecord(
        schema_version=LADDER_CERTIFICATE_SIDECAR_SCHEMA_VERSION,
        game_index=int(game_index),
        move_count=int(move_count),
        board_size=board_size_int,
        plane_names=tuple(sequence.plane_names),
        template_origin_encoding=LADDER_CERTIFICATE_MAP_ENCODING_DENSE_U8_ZLIB_BASE64,
        template_origin_blob=encode_ladder_certificate_maps_blob(
            sequence.template_origin_maps
        ),
        carrier_encoding=LADDER_CERTIFICATE_MAP_ENCODING_DENSE_U8_ZLIB_BASE64,
        carrier_blob=encode_ladder_certificate_maps_blob(sequence.carrier_maps),
        match_counts_by_position=tuple(
            int(stat.active_matches_after_update) for stat in sequence.position_stats
        ),
        embeddings_considered_by_position=tuple(
            int(stat.embeddings_considered) for stat in sequence.position_stats
        ),
        elapsed_ms_by_position=tuple(
            float(stat.elapsed_ms) for stat in sequence.position_stats
        ),
        used_must_include_filter_by_position="".join(
            "1" if stat.used_must_include_filter else "0"
            for stat in sequence.position_stats
        ),
    )


def extract_trmph_games_for_ladder_sidecar(
    file_path: PathLike,
    *,
    include_winnerless: bool = False,
) -> tuple[str, ...]:
    """
    Extract parseable TRMPH games from a file for ladder-sidecar generation.

    By default this mirrors the current training-preprocessing expectation and
    skips winnerless lines.
    """
    games: list[str] = []
    for line in load_trmph_file(Path(file_path)):
        try:
            trmph_text, winner = parse_trmph_line_flexible(line)
        except ValueError:
            continue
        if winner is None and not include_winnerless:
            continue
        games.append(trmph_text)
    return tuple(games)


def build_ladder_certificate_sidecar_records_for_trmph_file(
    file_path: PathLike,
    templates: tuple[LoadedLadderTemplate, ...] | list[LoadedLadderTemplate],
    *,
    include_winnerless: bool = False,
    orientations: tuple[str, ...] = ("red_bottom", "blue_right"),
    allow_attacker_superset_on_empty: bool = True,
) -> tuple[LadderCertificateSidecarRecord, ...]:
    """Build ladder-certificate sidecar records for every eligible game in a file."""
    games = extract_trmph_games_for_ladder_sidecar(
        file_path,
        include_winnerless=include_winnerless,
    )
    return tuple(
        build_ladder_certificate_sidecar_record_for_trmph_game(
            game_index=game_index,
            trmph_text=trmph_text,
            templates=templates,
            orientations=orientations,
            allow_attacker_superset_on_empty=allow_attacker_superset_on_empty,
        )
        for game_index, trmph_text in enumerate(games)
    )


def load_ladder_certificate_sidecar(
    sidecar_path: PathLike,
) -> tuple[LadderCertificateSidecarRecord, ...]:
    """Load and validate all ladder-certificate sidecar records from disk."""
    path = Path(sidecar_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing ladder-certificate sidecar: {path}")
    if not path.is_file():
        raise ValueError(f"Ladder-certificate sidecar path is not a file: {path}")

    records: list[LadderCertificateSidecarRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(LadderCertificateSidecarRecord.from_json_line(stripped))
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse ladder-certificate sidecar line {line_number} in {path}: {exc}"
                ) from exc
    return tuple(records)


def write_ladder_certificate_sidecar(
    sidecar_path: PathLike,
    records: tuple[LadderCertificateSidecarRecord, ...]
    | list[LadderCertificateSidecarRecord],
) -> Path:
    """Write validated ladder-certificate sidecar records to disk atomically."""
    path = Path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if not isinstance(record, LadderCertificateSidecarRecord):
                raise TypeError(
                    "write_ladder_certificate_sidecar expects LadderCertificateSidecarRecord "
                    f"instances, got {type(record)!r}"
                )
            handle.write(record.to_json_line())
            handle.write("\n")
    temp_path.replace(path)
    return path
