"""Extract inline SVG diagrams from HexWiki sections."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Sequence


@dataclass(frozen=True)
class HexWikiSectionSpec:
    anchor_id: str
    filename_prefix: str


@dataclass(frozen=True)
class HexWikiHeading:
    anchor_id: str
    level: int
    start: int
    end: int


@dataclass(frozen=True)
class ExtractedHexWikiSvg:
    source_page_url: str
    section_anchor: str
    filename_prefix: str
    index: int
    svg: str
    width: str
    height: str
    view_box: str

    @property
    def filename(self) -> str:
        return f"{self.filename_prefix}_{self.index:02d}.svg"

    def manifest_entry(self, *, output_dir: Path) -> dict[str, object]:
        return {
            "source_page_url": self.source_page_url,
            "section_anchor": self.section_anchor,
            "filename_prefix": self.filename_prefix,
            "index": self.index,
            "filename": self.filename,
            "path": str((output_dir / self.filename).resolve()),
            "width": self.width,
            "height": self.height,
            "view_box": self.view_box,
        }


_HEADING_RE = re.compile(r"<h(?P<level>[1-6])\b[^>]*>(?P<body>.*?)</h(?P=level)>", flags=re.S)
_HEADLINE_ID_RE = re.compile(
    r'<span\b[^>]*\bclass="[^"]*\bmw-headline\b[^"]*"[^>]*\bid="(?P<id_after>[^"]+)"'
    r'|<span\b[^>]*\bid="(?P<id_before>[^"]+)"[^>]*\bclass="[^"]*\bmw-headline\b[^"]*"',
    flags=re.S,
)
_SVG_RE = re.compile(r"<svg\b.*?</svg>", flags=re.S)
_ATTRIBUTE_RE_TEMPLATE = r'\b{attribute}="(?P<value>[^"]*)"'


DEFAULT_THEORY_OF_LADDER_ESCAPES_SECTIONS: tuple[HexWikiSectionSpec, ...] = (
    HexWikiSectionSpec(anchor_id="Examples", filename_prefix="second_row_escape"),
    HexWikiSectionSpec(anchor_id="Examples_2", filename_prefix="third_row_escape"),
    HexWikiSectionSpec(anchor_id="Examples_3", filename_prefix="fourth_row_escape"),
    HexWikiSectionSpec(anchor_id="Examples_4", filename_prefix="fifth_row_escape"),
)


def find_hexwiki_headings(html: str) -> tuple[HexWikiHeading, ...]:
    headings: list[HexWikiHeading] = []
    for match in _HEADING_RE.finditer(html):
        id_match = _HEADLINE_ID_RE.search(match.group("body"))
        if id_match is None:
            continue
        anchor_id = id_match.group("id_after") or id_match.group("id_before")
        headings.append(
            HexWikiHeading(
                anchor_id=anchor_id,
                level=int(match.group("level")),
                start=match.start(),
                end=match.end(),
            )
        )
    return tuple(headings)


def _attribute_value(svg: str, attribute: str) -> str:
    match = re.search(_ATTRIBUTE_RE_TEMPLATE.format(attribute=re.escape(attribute)), svg)
    return match.group("value") if match is not None else ""


def _normalize_svg(svg: str) -> str:
    normalized = svg.strip()
    svg_tag_end = normalized.find(">")
    if svg_tag_end < 0:
        raise ValueError("SVG fragment is missing an opening tag terminator.")
    opening_tag = normalized[:svg_tag_end]
    if 'xmlns="' not in opening_tag:
        normalized = normalized.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    if "xlink:" in normalized and "xmlns:xlink=" not in normalized[: normalized.find(">")]:
        normalized = normalized.replace(
            "<svg",
            '<svg xmlns:xlink="http://www.w3.org/1999/xlink"',
            1,
        )
    return normalized


def extract_hexwiki_svgs(
    html: str,
    *,
    source_page_url: str,
    section_specs: Sequence[HexWikiSectionSpec],
) -> tuple[ExtractedHexWikiSvg, ...]:
    headings = find_hexwiki_headings(html)
    heading_index_by_id = {heading.anchor_id: index for index, heading in enumerate(headings)}

    extracted: list[ExtractedHexWikiSvg] = []
    for spec in section_specs:
        if spec.anchor_id not in heading_index_by_id:
            available = ", ".join(sorted(heading_index_by_id))
            raise ValueError(
                f"Section anchor {spec.anchor_id!r} not found. Available anchors: {available}"
            )
        heading = headings[heading_index_by_id[spec.anchor_id]]
        section_end = len(html)
        for candidate in headings[heading_index_by_id[spec.anchor_id] + 1 :]:
            if candidate.level <= heading.level:
                section_end = candidate.start
                break
        section_html = html[heading.end:section_end]
        for svg_index, svg_match in enumerate(_SVG_RE.finditer(section_html), start=1):
            svg = _normalize_svg(svg_match.group(0))
            extracted.append(
                ExtractedHexWikiSvg(
                    source_page_url=source_page_url,
                    section_anchor=spec.anchor_id,
                    filename_prefix=spec.filename_prefix,
                    index=svg_index,
                    svg=svg,
                    width=_attribute_value(svg, "width"),
                    height=_attribute_value(svg, "height"),
                    view_box=_attribute_value(svg, "viewBox"),
                )
            )
    return tuple(extracted)


def write_extracted_hexwiki_svgs(
    extracted_svgs: Iterable[ExtractedHexWikiSvg],
    *,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    by_prefix: dict[str, int] = {}
    extracted_list = list(extracted_svgs)

    for extracted in extracted_list:
        output_path = resolved_output_dir / extracted.filename
        output_path.write_text(extracted.svg, encoding="utf-8")
        entries.append(extracted.manifest_entry(output_dir=resolved_output_dir))
        by_prefix[extracted.filename_prefix] = by_prefix.get(extracted.filename_prefix, 0) + 1

    manifest = {
        "count": len(extracted_list),
        "by_prefix": dict(sorted(by_prefix.items())),
        "items": entries,
    }
    resolved_manifest_path = Path(manifest_path) if manifest_path is not None else resolved_output_dir / "manifest.json"
    resolved_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_contact_sheet(extracted_list, output_dir=resolved_output_dir)
    return manifest


def _write_contact_sheet(extracted_svgs: Sequence[ExtractedHexWikiSvg], *, output_dir: Path) -> None:
    cards = "\n".join(
        (
            "      <article class=\"card\">"
            f"<h2>{extracted.filename}</h2>"
            f"<p>{extracted.section_anchor}</p>"
            f"<img src=\"{extracted.filename}\" alt=\"{extracted.filename}\" />"
            "      </article>"
        )
        for extracted in extracted_svgs
    )
    html = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <title>HexWiki SVG Scrape Review</title>\n"
        "  <style>\n"
        "    :root { color-scheme: light; }\n"
        "    body { font-family: Georgia, serif; margin: 0; padding: 24px; background: #f4f1ea; color: #1d1d1d; }\n"
        "    h1 { margin: 0 0 8px 0; font-size: 1.8rem; }\n"
        "    p.lede { margin: 0 0 24px 0; max-width: 70ch; }\n"
        "    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px; }\n"
        "    .card { background: #fffdfa; border: 1px solid #d8cfbf; border-radius: 14px; padding: 14px; box-shadow: 0 12px 30px rgba(40, 28, 12, 0.08); }\n"
        "    .card h2 { margin: 0 0 6px 0; font-size: 1rem; }\n"
        "    .card p { margin: 0 0 10px 0; color: #665f52; font-size: 0.9rem; }\n"
        "    .card img { display: block; width: 100%; height: auto; background: white; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>HexWiki SVG Scrape Review</h1>\n"
        "  <p class=\"lede\">Review the extracted diagrams below. Each card links the local filename to the source section anchor.</p>\n"
        "  <section class=\"grid\">\n"
        f"{cards}\n"
        "  </section>\n"
        "</body>\n"
        "</html>\n"
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")
