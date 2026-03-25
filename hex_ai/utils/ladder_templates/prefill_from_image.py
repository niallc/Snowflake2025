#!/usr/bin/env python3
"""Generate a first-pass ladder-template annotation from a diagram image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from hex_ai.utils.ladder_templates.prefill_geometry import (
    DetectedImageCell,
    build_prefill_annotation_payload,
    infer_offset_grid,
)


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for image prefill.\n"
            "Install it in the project venv first, for example:\n"
            "  pip install opencv-python"
        ) from exc
    return cv2


def _dedupe_candidates(candidates: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    accepted: list[tuple[float, float, float, float]] = []
    for center_x, center_y, radius, area in sorted(candidates, key=lambda item: item[3], reverse=True):
        is_duplicate = False
        for accepted_x, accepted_y, accepted_radius, _ in accepted:
            distance = float(((center_x - accepted_x) ** 2 + (center_y - accepted_y) ** 2) ** 0.5)
            if distance < max(radius, accepted_radius) * 0.55:
                is_duplicate = True
                break
        if not is_duplicate:
            accepted.append((center_x, center_y, radius, area))
    accepted.sort(key=lambda item: (item[1], item[0]))
    return accepted


def _detect_hex_candidates(image_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
    cv2 = _require_cv2()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image_area = float(image_bgr.shape[0] * image_bgr.shape[1])
    min_area = max(80.0, image_area / 5000.0)
    max_area = image_area / 8.0

    candidates: list[tuple[float, float, float, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        if len(approx) < 5 or len(approx) > 8:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if h <= 0:
            continue
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:
            continue
        (center_x, center_y), radius = cv2.minEnclosingCircle(approx)
        if radius < 6:
            continue
        candidates.append((float(center_x), float(center_y), float(radius), area))

    return _dedupe_candidates(candidates)


def _patch_with_mask(image: np.ndarray, center_x: float, center_y: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    patch_radius = max(4, int(round(radius * 0.55)))
    cx = int(round(center_x))
    cy = int(round(center_y))
    left = max(0, cx - patch_radius)
    right = min(width, cx + patch_radius + 1)
    top = max(0, cy - patch_radius)
    bottom = min(height, cy + patch_radius + 1)
    patch = image[top:bottom, left:right]

    yy, xx = np.ogrid[top:bottom, left:right]
    mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= (patch_radius * 0.92) ** 2
    return patch, mask


def _classify_detected_cell(image_bgr: np.ndarray, center_x: float, center_y: float, radius: float) -> str:
    cv2 = _require_cv2()
    patch_bgr, mask = _patch_with_mask(image_bgr, center_x, center_y, radius)
    if patch_bgr.size == 0:
        return "empty"

    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

    red_mask = (
        (((patch_hsv[:, :, 0] <= 10) | (patch_hsv[:, :, 0] >= 170)) & (patch_hsv[:, :, 1] >= 90) & (patch_hsv[:, :, 2] >= 90))
        & mask
    )
    blue_mask = (
        ((patch_hsv[:, :, 0] >= 90) & (patch_hsv[:, :, 0] <= 135) & (patch_hsv[:, :, 1] >= 70) & (patch_hsv[:, :, 2] >= 60))
        & mask
    )

    mask_area = max(1, int(mask.sum()))
    red_ratio = float(red_mask.sum()) / mask_area
    blue_ratio = float(blue_mask.sum()) / mask_area
    if red_ratio >= 0.18:
        return "red"
    if blue_ratio >= 0.18:
        return "blue"

    inner_patch, inner_mask = _patch_with_mask(image_bgr, center_x, center_y, radius * 0.65)
    inner_gray = cv2.cvtColor(inner_patch, cv2.COLOR_BGR2GRAY)
    dark_mask = (inner_gray <= 140) & inner_mask
    inner_area = max(1, int(inner_mask.sum()))
    dark_ratio = float(dark_mask.sum()) / inner_area

    if dark_ratio < 0.025:
        return "empty"

    row_scores = dark_mask.sum(axis=1).astype(float) / max(1, dark_mask.shape[1])
    col_scores = dark_mask.sum(axis=0).astype(float) / max(1, dark_mask.shape[0])
    horizontal_score = float(row_scores.max()) if row_scores.size else 0.0
    vertical_score = float(col_scores.max()) if col_scores.size else 0.0

    if horizontal_score >= 0.30 and vertical_score >= 0.20:
        return "plus"
    if horizontal_score >= 0.30 and vertical_score < 0.18:
        return "minus"
    if dark_ratio >= 0.18:
        return "shaded"
    return "empty"


def detect_image_cells(image_bgr: np.ndarray) -> tuple[list[DetectedImageCell], float]:
    """Detect likely template cells and classify their states."""
    candidates = _detect_hex_candidates(image_bgr)
    if not candidates:
        raise ValueError("No candidate hex cells detected.")

    radius_hint = float(np.median([item[2] for item in candidates]))
    detected_cells = [
        DetectedImageCell(
            center_x=center_x,
            center_y=center_y,
            state=_classify_detected_cell(image_bgr, center_x, center_y, radius_hint),
        )
        for center_x, center_y, _, _ in candidates
    ]
    return detected_cells, radius_hint


def _draw_debug_overlay(
    image_bgr: np.ndarray,
    inferred_cells: Iterable[DetectedImageCell],
    *,
    radius: float,
    output_path: Path,
) -> None:
    cv2 = _require_cv2()
    canvas = image_bgr.copy()
    color_by_state = {
        "red": (40, 40, 220),
        "blue": (220, 120, 40),
        "plus": (20, 20, 20),
        "minus": (20, 20, 20),
        "shaded": (120, 80, 180),
        "empty": (90, 170, 90),
    }
    for index, cell in enumerate(inferred_cells, start=1):
        cx = int(round(cell.center_x))
        cy = int(round(cell.center_y))
        color = color_by_state.get(cell.state, (150, 150, 150))
        cv2.circle(canvas, (cx, cy), int(round(radius * 0.65)), color, 2)
        cv2.putText(
            canvas,
            f"{index}:{cell.state}",
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError(f"Failed to write debug overlay to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefill a ladder-template annotation JSON from a diagram image."
    )
    parser.add_argument("image_path", type=Path, help="Path to the source image.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the annotator-compatible JSON output file.",
    )
    parser.add_argument(
        "--template-name",
        type=str,
        default="",
        help="Template name to include in the output JSON.",
    )
    parser.add_argument(
        "--family",
        type=str,
        default="opencv_prefill",
        help="Source/family tag to include in the output JSON.",
    )
    parser.add_argument(
        "--attacker",
        type=str,
        default="red",
        choices=("red", "blue"),
        help="Attacker color metadata for the output JSON.",
    )
    parser.add_argument(
        "--target-edge",
        type=str,
        default="red_bottom",
        choices=("red_bottom", "red_top", "blue_left", "blue_right", "none"),
        help="Target edge metadata for the output JSON.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="Prefill generated from image; review in annotator before use.",
        help="Notes to include in the output JSON.",
    )
    parser.add_argument(
        "--debug-image",
        type=Path,
        default=None,
        help="Optional path to save a labeled detection overlay image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2 = _require_cv2()
    image = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {args.image_path}")

    detected_cells, radius_hint = detect_image_cells(image)
    geometry = infer_offset_grid(detected_cells, radius_hint=radius_hint)

    open_left = any(cell.state == "plus" for cell in geometry.cells)
    open_right = any(cell.state == "minus" for cell in geometry.cells)
    payload = build_prefill_annotation_payload(
        geometry,
        image_name=args.image_path.name,
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
        family=args.family,
        template_name=args.template_name or args.image_path.stem,
        attacker=args.attacker,
        target_edge=args.target_edge,
        open_left=open_left,
        open_right=open_right,
        notes=args.notes,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote annotation prefill to {args.output}")
    print(f"Detected {len(geometry.cells)} cells with radius≈{geometry.radius:.2f}")

    if args.debug_image is not None:
        _draw_debug_overlay(image, detected_cells, radius=geometry.radius, output_path=args.debug_image)
        print(f"Wrote debug overlay to {args.debug_image}")


if __name__ == "__main__":
    main()
