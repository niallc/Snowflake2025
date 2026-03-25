"""Utilities for image-driven ladder-template prefill."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class DetectedImageCell:
    center_x: float
    center_y: float
    state: str


@dataclass(frozen=True)
class InferredTemplateCell:
    row: int
    col: int
    center_x: float
    center_y: float
    state: str


@dataclass(frozen=True)
class InferredGridGeometry:
    rows: int
    cols: int
    radius: float
    origin_x: float
    origin_y: float
    horizontal_pitch: float
    vertical_pitch: float
    cells: tuple[InferredTemplateCell, ...]


def _median(values: Sequence[float], *, default: float) -> float:
    if not values:
        return default
    return float(median(values))


def _positive_differences(values: Sequence[float], *, min_gap: float) -> list[float]:
    diffs: list[float] = []
    for left, right in zip(values, values[1:]):
        diff = float(right - left)
        if diff > min_gap:
            diffs.append(diff)
    return diffs


def infer_offset_grid(
    detected_cells: Iterable[DetectedImageCell],
    *,
    radius_hint: float,
) -> InferredGridGeometry:
    """Infer local offset-grid coordinates from detected image centers."""
    cells = tuple(detected_cells)
    if not cells:
        raise ValueError("detected_cells must not be empty")
    if radius_hint <= 0:
        raise ValueError(f"radius_hint must be positive, got {radius_hint}")

    horizontal_pitch = float(radius_hint) * (3 ** 0.5)
    vertical_pitch = float(radius_hint) * 1.5

    min_y = min(cell.center_y for cell in cells)
    provisional_rows = [
        int(round((cell.center_y - min_y) / vertical_pitch))
        for cell in cells
    ]
    row_offset = min(provisional_rows)
    normalized_rows = [row - row_offset for row in provisional_rows]

    row_adjusted_x = [
        cell.center_x - 0.5 * horizontal_pitch * row
        for cell, row in zip(cells, normalized_rows)
    ]
    min_base_x = min(row_adjusted_x)
    provisional_cols = [
        int(round((base_x - min_base_x) / horizontal_pitch))
        for base_x in row_adjusted_x
    ]
    col_offset = min(provisional_cols)
    normalized_cols = [col - col_offset for col in provisional_cols]

    origin_x = _median(
        [
            cell.center_x - horizontal_pitch * (col + row / 2.0)
            for cell, row, col in zip(cells, normalized_rows, normalized_cols)
        ],
        default=min_base_x,
    )
    origin_y = _median(
        [
            cell.center_y - vertical_pitch * row
            for cell, row in zip(cells, normalized_rows)
        ],
        default=min_y,
    )

    inferred_cells = tuple(
        sorted(
            (
                InferredTemplateCell(
                    row=row,
                    col=col,
                    center_x=cell.center_x,
                    center_y=cell.center_y,
                    state=cell.state,
                )
                for cell, row, col in zip(cells, normalized_rows, normalized_cols)
            ),
            key=lambda item: (item.row, item.col, item.state),
        )
    )

    return InferredGridGeometry(
        rows=max(item.row for item in inferred_cells) + 1,
        cols=max(item.col for item in inferred_cells) + 1,
        radius=float(radius_hint),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        horizontal_pitch=float(horizontal_pitch),
        vertical_pitch=float(vertical_pitch),
        cells=inferred_cells,
    )


def build_prefill_annotation_payload(
    geometry: InferredGridGeometry,
    *,
    image_name: str,
    image_width: int,
    image_height: int,
    family: str,
    template_name: str,
    attacker: str = "red",
    target_edge: str = "red_bottom",
    open_left: bool = False,
    open_right: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    """Build annotator-compatible JSON from inferred image geometry."""
    return {
        "schema_version": 1,
        "kind": "ladder_template_annotation",
        "coord_system": "row_col_offset",
        "metadata": {
            "name": template_name,
            "family": family,
            "attacker": attacker,
            "target_edge": target_edge,
            "open_left": open_left,
            "open_right": open_right,
            "notes": notes,
        },
        "grid": {
            "rows": geometry.rows,
            "cols": geometry.cols,
            "radius": geometry.radius,
            "origin_x": geometry.origin_x,
            "origin_y": geometry.origin_y,
            "grid_opacity": 1,
            "show_coords": False,
        },
        "image_overlay": {
            "image_name": image_name,
            "image_width": image_width,
            "image_height": image_height,
            "image_opacity": 0.8,
            "image_scale": 1.0,
            "image_offset_x": 0.0,
            "image_offset_y": 0.0,
        },
        "cells": [
            {
                "row": cell.row,
                "col": cell.col,
                "state": cell.state,
            }
            for cell in geometry.cells
        ],
    }
