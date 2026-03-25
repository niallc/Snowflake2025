"""Direct SVG-to-annotation conversion for ladder-template diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable
from xml.etree import ElementTree as ET

from hex_ai.utils.ladder_templates.prefill_geometry import (
    DetectedImageCell,
    build_prefill_annotation_payload,
    infer_offset_grid,
)


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ParsedLadderTemplateSvg:
    image_width: int
    image_height: int
    radius_hint: float
    detected_cells: tuple[DetectedImageCell, ...]


def _strip_px(value: str) -> float:
    return float(value[:-2]) if value.endswith("px") else float(value)


def _path_numbers(path_data: str) -> list[float]:
    return [float(value) for value in _NUMBER_RE.findall(path_data)]


def _find_hex_path(root: ET.Element) -> ET.Element:
    for element in root.findall(".//{*}path"):
        if element.get("class") == "bodypath" and (element.get("id") or "").startswith("hexes"):
            return element
    raise ValueError("Could not find the main hex path in the SVG.")


def _parse_hex_centers(path_data: str, *, scale_x: float, scale_y: float) -> tuple[tuple[float, float], ...]:
    centers: list[tuple[float, float]] = []
    for subpath in re.findall(r"M[^z]+z", path_data):
        numbers = _path_numbers(subpath)
        if len(numbers) < 5:
            continue
        origin_x = numbers[0]
        origin_y = numbers[1]
        dx_to_center = numbers[3]
        dy_to_center = numbers[4]
        centers.append(
            (
                (origin_x + dx_to_center) * scale_x,
                (origin_y + dy_to_center) * scale_y,
            )
        )
    if not centers:
        raise ValueError("Could not parse any hex centers from the SVG body path.")
    return tuple(centers)


def _parse_radius_hint(path_data: str, *, scale_y: float) -> float:
    first_subpath = re.search(r"M[^z]+z", path_data)
    if first_subpath is None:
        raise ValueError("Could not parse a representative hex subpath from the SVG body path.")
    numbers = _path_numbers(first_subpath.group(0))
    if len(numbers) < 3:
        raise ValueError("Representative hex subpath did not contain enough geometry numbers.")
    return numbers[2] * scale_y


def _actual_circle_class(element: ET.Element) -> str:
    class_name = element.get("class") or ""
    cx = element.get("cx")
    cy = element.get("cy")
    radius = element.get("r")
    if not class_name or cx is None or cy is None or radius is None:
        return ""
    return class_name


def _parse_marker_candidates(root: ET.Element, *, scale_x: float, scale_y: float) -> list[DetectedImageCell]:
    markers: list[DetectedImageCell] = []

    for element in root.findall(".//{*}path"):
        if element.get("stroke") != "black":
            continue
        path_data = element.get("d") or ""
        numbers = _path_numbers(path_data)
        if len(numbers) == 8:
            center_x = ((numbers[0] + numbers[2]) / 2.0) * scale_x
            center_y = ((numbers[1] + numbers[3]) / 2.0) * scale_y
            markers.append(DetectedImageCell(center_x=center_x, center_y=center_y, state="plus"))
        elif len(numbers) == 4:
            center_x = ((numbers[0] + numbers[2]) / 2.0) * scale_x
            center_y = ((numbers[1] + numbers[3]) / 2.0) * scale_y
            markers.append(DetectedImageCell(center_x=center_x, center_y=center_y, state="minus"))

    for element in root.findall(".//{*}circle"):
        class_name = _actual_circle_class(element)
        if class_name not in {"vertcirc", "horizcirc"}:
            continue
        markers.append(
            DetectedImageCell(
                center_x=float(element.get("cx", "0")) * scale_x,
                center_y=float(element.get("cy", "0")) * scale_y,
                state="red" if class_name == "vertcirc" else "blue",
            )
        )
    return markers


def _assign_states_to_hexes(
    centers: Iterable[tuple[float, float]],
    markers: Iterable[DetectedImageCell],
    *,
    radius_hint: float,
) -> tuple[DetectedImageCell, ...]:
    center_list = list(centers)
    assigned_states = ["empty"] * len(center_list)
    max_distance = radius_hint * 0.35

    for marker in markers:
        best_index = min(
            range(len(center_list)),
            key=lambda index: (
                (center_list[index][0] - marker.center_x) ** 2
                + (center_list[index][1] - marker.center_y) ** 2
            ),
        )
        best_center = center_list[best_index]
        distance = ((best_center[0] - marker.center_x) ** 2 + (best_center[1] - marker.center_y) ** 2) ** 0.5
        if distance > max_distance:
            raise ValueError(
                f"Marker {marker.state} at {(marker.center_x, marker.center_y)} did not match a hex center closely enough."
            )
        if assigned_states[best_index] != "empty":
            raise ValueError(
                f"Multiple markers mapped to the same hex center {best_center}: "
                f"{assigned_states[best_index]!r} and {marker.state!r}"
            )
        assigned_states[best_index] = marker.state

    return tuple(
        DetectedImageCell(center_x=center_x, center_y=center_y, state=state)
        for (center_x, center_y), state in zip(center_list, assigned_states)
    )


def parse_ladder_template_svg(path: str | Path) -> ParsedLadderTemplateSvg:
    svg_path = Path(path)
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))

    image_width = int(round(_strip_px(root.get("width", "0"))))
    image_height = int(round(_strip_px(root.get("height", "0"))))
    view_box = root.get("viewBox")
    if not view_box:
        raise ValueError("SVG is missing a viewBox attribute.")
    _, _, view_box_width, view_box_height = [float(part) for part in view_box.split()]
    scale_x = image_width / view_box_width
    scale_y = image_height / view_box_height

    hex_path = _find_hex_path(root)
    hex_path_data = hex_path.get("d") or ""
    centers = _parse_hex_centers(hex_path_data, scale_x=scale_x, scale_y=scale_y)
    radius_hint = _parse_radius_hint(hex_path_data, scale_y=scale_y)
    markers = _parse_marker_candidates(root, scale_x=scale_x, scale_y=scale_y)
    detected_cells = _assign_states_to_hexes(centers, markers, radius_hint=radius_hint)

    return ParsedLadderTemplateSvg(
        image_width=image_width,
        image_height=image_height,
        radius_hint=radius_hint,
        detected_cells=detected_cells,
    )


def build_prefill_annotation_payload_from_svg(
    path: str | Path,
    *,
    template_name: str,
    family: str,
    attacker: str = "red",
    target_edge: str = "red_bottom",
    notes: str = "",
) -> dict[str, object]:
    svg_path = Path(path)
    parsed = parse_ladder_template_svg(svg_path)
    geometry = infer_offset_grid(parsed.detected_cells, radius_hint=parsed.radius_hint)
    open_left = any(cell.state == "plus" for cell in geometry.cells)
    open_right = any(cell.state == "minus" for cell in geometry.cells)
    return build_prefill_annotation_payload(
        geometry,
        image_name=svg_path.name,
        image_width=parsed.image_width,
        image_height=parsed.image_height,
        family=family,
        template_name=template_name,
        attacker=attacker,
        target_edge=target_edge,
        open_left=open_left,
        open_right=open_right,
        notes=notes,
    )
