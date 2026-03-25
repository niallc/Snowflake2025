#!/usr/bin/env python3
"""Generate annotator-compatible ladder-template JSON directly from SVG diagrams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hex_ai.utils.ladder_templates.svg_prefill import build_prefill_annotation_payload_from_svg


def _infer_family_from_manifest_item(manifest_item: dict[str, object] | None) -> str:
    if not manifest_item:
        return "hexwiki_svg"
    source_page_url = str(manifest_item.get("source_page_url", ""))
    if "Theory_of_ladder_escapes" in source_page_url:
        return "hexwiki_theory_of_ladder_escapes"
    return "hexwiki_svg"


def _build_notes(manifest_item: dict[str, object] | None) -> str:
    if not manifest_item:
        return "Generated directly from SVG; review in annotator before use."
    source_page_url = str(manifest_item.get("source_page_url", ""))
    section_anchor = str(manifest_item.get("section_anchor", ""))
    parts = ["Generated directly from SVG; review in annotator before use."]
    if source_page_url:
        parts.append(f"Source page: {source_page_url}")
    if section_anchor:
        parts.append(f"Source section: {section_anchor}")
    return " ".join(parts)


def _load_manifest_items(input_path: Path) -> dict[str, dict[str, object]]:
    manifest_path = input_path / "manifest.json" if input_path.is_dir() else input_path.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not isinstance(items, list):
        return {}
    manifest_items: dict[str, dict[str, object]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        if isinstance(filename, str):
            manifest_items[filename] = item
    return manifest_items


def _iter_svg_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.glob("*.svg") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefill ladder-template annotation JSON directly from SVG diagrams."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A single SVG file or a directory of SVG files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write generated JSON files into.",
    )
    parser.add_argument(
        "--family",
        type=str,
        default="",
        help="Optional family override. Defaults to a family inferred from the scrape manifest.",
    )
    parser.add_argument(
        "--attacker",
        type=str,
        default="red",
        choices=("red", "blue"),
        help="Attacker color metadata (default: red).",
    )
    parser.add_argument(
        "--target-edge",
        type=str,
        default="red_bottom",
        choices=("red_bottom", "red_top", "blue_left", "blue_right", "none"),
        help="Target edge metadata (default: red_bottom).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    svg_paths = _iter_svg_paths(args.input_path)
    if not svg_paths:
        raise ValueError(f"No SVG files found in {args.input_path}")

    manifest_items = _load_manifest_items(args.input_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, object]] = []
    for svg_path in svg_paths:
        manifest_item = manifest_items.get(svg_path.name)
        family = args.family or _infer_family_from_manifest_item(manifest_item)
        notes = _build_notes(manifest_item)
        payload = build_prefill_annotation_payload_from_svg(
            svg_path,
            template_name=svg_path.stem,
            family=family,
            attacker=args.attacker,
            target_edge=args.target_edge,
            notes=notes,
        )
        output_path = args.output_dir / f"{svg_path.stem}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        outputs.append(
            {
                "input_svg": str(svg_path.resolve()),
                "output_json": str(output_path.resolve()),
                "cells": len(payload["cells"]),
                "grid_rows": payload["grid"]["rows"],
                "grid_cols": payload["grid"]["cols"],
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "count": len(outputs),
                "items": outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(outputs)} JSON files into {args.output_dir}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
