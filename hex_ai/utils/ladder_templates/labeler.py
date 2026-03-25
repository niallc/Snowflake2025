"""Convert ladder-template matches into dense spatial supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matcher import LadderTemplateMatch

SUPPORTED_CERTIFICATE_ORIENTATIONS: tuple[str, ...] = ("red_bottom", "blue_right")
SUPPORTED_ROW_CLASSES: tuple[int, ...] = (2, 3, 4, 5)


def ladder_certificate_plane_specs() -> tuple[tuple[str, int], ...]:
    """Return the canonical plane order."""
    return tuple(
        (orientation, row_class)
        for orientation in SUPPORTED_CERTIFICATE_ORIENTATIONS
        for row_class in SUPPORTED_ROW_CLASSES
    )


def ladder_certificate_plane_names(*, suffix: str) -> tuple[str, ...]:
    """Return canonical plane names for one spatial target family."""
    return tuple(
        f"{orientation}_row{row_class}_{suffix}"
        for orientation, row_class in ladder_certificate_plane_specs()
    )


def ladder_certificate_plane_index(*, orientation: str, row_class: int) -> int:
    """Return the canonical plane index for an orientation/row-class pair."""
    specs = ladder_certificate_plane_specs()
    try:
        return specs.index((orientation, row_class))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported ladder certificate plane ({orientation!r}, {row_class!r})"
        ) from exc


@dataclass(frozen=True)
class LadderCertificateLabels:
    """Dense label tensors derived from ladder-template matches."""

    plane_names: tuple[str, ...]
    template_origin_maps: np.ndarray
    carrier_maps: np.ndarray


def build_ladder_certificate_labels(
    matches: list[LadderTemplateMatch] | tuple[LadderTemplateMatch, ...],
    *,
    board_size: int,
) -> LadderCertificateLabels:
    """Build canonical template-origin and carrier maps from match records."""
    plane_count = len(ladder_certificate_plane_specs())
    template_origin_maps = np.zeros((plane_count, board_size, board_size), dtype=np.uint8)
    carrier_maps = np.zeros((plane_count, board_size, board_size), dtype=np.uint8)

    for match in matches:
        plane_index = ladder_certificate_plane_index(
            orientation=match.orientation,
            row_class=match.row_class,
        )
        origin_row, origin_col = match.template_origin
        template_origin_maps[plane_index, origin_row, origin_col] = 1
        for row, col in match.carrier_cells:
            carrier_maps[plane_index, row, col] = 1

    return LadderCertificateLabels(
        plane_names=ladder_certificate_plane_names(suffix="template_origin"),
        template_origin_maps=template_origin_maps,
        carrier_maps=carrier_maps,
    )
