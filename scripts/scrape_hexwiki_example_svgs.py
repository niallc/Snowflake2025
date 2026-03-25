#!/usr/bin/env python3
"""Scrape inline SVG ladder diagrams from HexWiki example sections."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from hex_ai.hexwiki_svg_scraper import (
    DEFAULT_THEORY_OF_LADDER_ESCAPES_SECTIONS,
    HexWikiSectionSpec,
    extract_hexwiki_svgs,
    write_extracted_hexwiki_svgs,
)


DEFAULT_PAGE_URL = "https://www.hexwiki.net/index.php/Theory_of_ladder_escapes"


def _parse_section_specs(values: list[str]) -> tuple[HexWikiSectionSpec, ...]:
    if not values:
        return DEFAULT_THEORY_OF_LADDER_ESCAPES_SECTIONS
    specs: list[HexWikiSectionSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --section value {value!r}. Expected format anchor_id=filename_prefix."
            )
        anchor_id, filename_prefix = value.split("=", 1)
        normalized_anchor = anchor_id.strip()
        normalized_prefix = filename_prefix.strip()
        if not normalized_anchor or not normalized_prefix:
            raise ValueError(
                f"Invalid --section value {value!r}. Both anchor_id and filename_prefix are required."
            )
        specs.append(
            HexWikiSectionSpec(
                anchor_id=normalized_anchor,
                filename_prefix=normalized_prefix,
            )
        )
    return tuple(specs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape standalone SVG files from HexWiki example sections."
    )
    parser.add_argument(
        "--page-url",
        type=str,
        default=DEFAULT_PAGE_URL,
        help=f"HexWiki page to scrape (default: {DEFAULT_PAGE_URL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the extracted SVG files, manifest, and review HTML.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional explicit manifest path. Defaults to <output-dir>/manifest.json.",
    )
    parser.add_argument(
        "--section",
        action="append",
        default=[],
        help=(
            "Section anchor mapping in the form anchor_id=filename_prefix. "
            "May be repeated. Defaults to the four ladder-example sections on Theory_of_ladder_escapes."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds (default: 20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    section_specs = _parse_section_specs(args.section)
    response = requests.get(
        args.page_url,
        timeout=args.timeout,
        headers={"User-Agent": "Snowflake2025 HexWiki SVG scraper"},
    )
    response.raise_for_status()
    extracted = extract_hexwiki_svgs(
        response.text,
        source_page_url=args.page_url,
        section_specs=section_specs,
    )
    manifest = write_extracted_hexwiki_svgs(
        extracted,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print(f"Scraped {manifest['count']} SVGs into {args.output_dir}")
    for prefix, count in manifest["by_prefix"].items():
        print(f"  {prefix}: {count}")
    print(f"Review page: {(args.output_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
