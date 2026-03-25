"""Load ladder-template libraries for certificate matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .core import LadderTemplateAnnotation, load_ladder_template_annotation

TemplateCoord = tuple[int, int]

_ROW_CLASS_PREFIXES: tuple[tuple[str, int], ...] = (
    ("second_row_escape", 2),
    ("third_row_escape", 3),
    ("fourth_row_escape", 4),
    ("fifth_row_escape", 5),
)
_CARRIER_STATES = frozenset({"red", "blue", "empty"})


@dataclass(frozen=True)
class LoadedLadderTemplate:
    """Pre-indexed ladder template ready for matching."""

    template_id: str
    row_class: int
    source_path: Path
    source_set: str
    annotation: LadderTemplateAnnotation
    local_rows: int
    local_cols: int
    carrier_local: tuple[TemplateCoord, ...]
    attacker_required_local: tuple[TemplateCoord, ...]
    defender_required_local: tuple[TemplateCoord, ...]
    empty_carrier_local: tuple[TemplateCoord, ...]
    shaded_local: tuple[TemplateCoord, ...]
    left_boundary_local: tuple[TemplateCoord, ...]
    right_boundary_local: tuple[TemplateCoord, ...]

    @property
    def attacker(self) -> str:
        return self.annotation.metadata.attacker

    @property
    def target_edge(self) -> str:
        return self.annotation.metadata.target_edge


def default_hexwiki_generated_library_dir() -> Path:
    """Return the default generated HexWiki ladder-template directory."""
    return (
        Path(__file__).resolve().parent
        / "library"
        / "hexwiki"
        / "theory_of_ladder_escapes"
        / "generated"
    )


def infer_row_class_from_template_name(template_name: str) -> int:
    """Infer ladder row class from the current HexWiki filename convention."""
    for prefix, row_class in _ROW_CLASS_PREFIXES:
        if template_name.startswith(prefix):
            return row_class
    raise ValueError(
        f"Could not infer row class from template name {template_name!r}. "
        f"Expected one of {[prefix for prefix, _ in _ROW_CLASS_PREFIXES]}."
    )


def build_loaded_ladder_template(
    annotation: LadderTemplateAnnotation,
    *,
    source_path: Path,
    source_set: str,
) -> LoadedLadderTemplate:
    """Convert a reviewed/generated annotation into a pre-indexed record."""
    template_id = annotation.metadata.name or source_path.stem
    row_class = infer_row_class_from_template_name(template_id)

    attacker_required_local: list[TemplateCoord] = []
    defender_required_local: list[TemplateCoord] = []
    empty_carrier_local: list[TemplateCoord] = []
    shaded_local: list[TemplateCoord] = []
    left_boundary_local: list[TemplateCoord] = []
    right_boundary_local: list[TemplateCoord] = []
    carrier_local: list[TemplateCoord] = []

    for cell in annotation.cells:
        coord = (cell.row, cell.col)
        if cell.state == annotation.metadata.attacker:
            attacker_required_local.append(coord)
            carrier_local.append(coord)
        elif cell.state in {"red", "blue"}:
            defender_required_local.append(coord)
            carrier_local.append(coord)
        elif cell.state == "empty":
            empty_carrier_local.append(coord)
            carrier_local.append(coord)
        elif cell.state == "plus":
            left_boundary_local.append(coord)
        elif cell.state == "minus":
            right_boundary_local.append(coord)
        elif cell.state == "shaded":
            shaded_local.append(coord)
        else:
            raise ValueError(
                f"Unsupported ladder-template cell state {cell.state!r} "
                f"in {source_path}"
            )

    if annotation.metadata.open_left != bool(left_boundary_local):
        raise ValueError(
            "Template open_left metadata does not match presence of '+' boundary "
            f"cells in {source_path}"
        )
    if annotation.metadata.open_right != bool(right_boundary_local):
        raise ValueError(
            "Template open_right metadata does not match presence of '-' boundary "
            f"cells in {source_path}"
        )
    if len(left_boundary_local) not in {0, row_class}:
        raise ValueError(
            f"Expected left boundary size 0 or {row_class} for {source_path}, "
            f"got {len(left_boundary_local)}"
        )

    return LoadedLadderTemplate(
        template_id=template_id,
        row_class=row_class,
        source_path=source_path,
        source_set=source_set,
        annotation=annotation,
        local_rows=int(annotation.grid.rows),
        local_cols=int(annotation.grid.cols),
        carrier_local=tuple(sorted(carrier_local)),
        attacker_required_local=tuple(sorted(attacker_required_local)),
        defender_required_local=tuple(sorted(defender_required_local)),
        empty_carrier_local=tuple(sorted(empty_carrier_local)),
        shaded_local=tuple(sorted(shaded_local)),
        left_boundary_local=tuple(sorted(left_boundary_local)),
        right_boundary_local=tuple(sorted(right_boundary_local)),
    )


def load_ladder_template_library(
    library_dir: str | Path,
    *,
    source_set: str,
) -> tuple[LoadedLadderTemplate, ...]:
    """Load a ladder-template directory into pre-indexed template records."""
    root = Path(library_dir)
    if not root.exists():
        raise FileNotFoundError(f"Ladder-template directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Ladder-template path is not a directory: {root}")

    records: list[LoadedLadderTemplate] = []
    for path in sorted(root.glob("*.json")):
        if path.name == "manifest.json":
            continue
        annotation = load_ladder_template_annotation(path)
        records.append(
            build_loaded_ladder_template(
                annotation,
                source_path=path,
                source_set=source_set,
            )
        )
    if not records:
        raise ValueError(f"No ladder-template JSON files found in {root}")
    return tuple(records)


def load_hexwiki_generated_ladder_templates() -> tuple[LoadedLadderTemplate, ...]:
    """Load the current generated HexWiki ladder-escape corpus."""
    return load_ladder_template_library(
        default_hexwiki_generated_library_dir(),
        source_set="generated",
    )


def iter_templates_for_row_class(
    templates: Iterable[LoadedLadderTemplate],
    *,
    row_class: int,
) -> tuple[LoadedLadderTemplate, ...]:
    """Return templates filtered to one ladder row class."""
    return tuple(template for template in templates if template.row_class == row_class)
