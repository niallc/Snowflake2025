from pathlib import Path

from hex_ai.utils.ladder_templates.svg_prefill import (
    build_prefill_annotation_payload_from_svg,
    parse_ladder_template_svg,
)


_SAMPLE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="113px" height="94px" viewBox="0 0 421 353" version="1.1">
  <defs>
    <path id="hexes36" class="bodypath" style="fill:#f6f6f6;" d="M33,74v82l71,41l71,-41v-82l-71-41zM175,74v82l71,41l71,-41v-82l-71-41zM104,197v82l71,41l71,-41v-82l-71-41zM246,197v82l71,41l71,-41v-82l-71-41z"/>
  </defs>
  <g>
    <use xlink:href="#hexes36" transform="translate(10,20)" filter="url(#shadow)"/>
    <use xlink:href="#edge00" style="stroke:red" transform="translate(175,238) rotate(180)"/>
    <use xlink:href="#edge00" style="stroke:red" transform="translate(317,238) rotate(180)"/>
    <use xlink:href="#hexes36"/>
  </g>
  <path stroke-linecap="butt" stroke-width="8" stroke="black" d="M79,115L129,115M104,90L104,140"/>
  <path stroke-linecap="butt" stroke-width="8" stroke="black" d="M150,238L200,238M175,213L175,263"/>
  <circle class="vertcirc" cx="317" cy="238" r="50"/>
</svg>
"""


def test_parse_ladder_template_svg_reads_cells_and_radius(tmp_path: Path):
    svg_path = tmp_path / "second_row_escape_01.svg"
    svg_path.write_text(_SAMPLE_SVG, encoding="utf-8")

    parsed = parse_ladder_template_svg(svg_path)

    assert parsed.image_width == 113
    assert parsed.image_height == 94
    assert round(parsed.radius_hint, 3) == round(82 * (94 / 353), 3)
    assert len(parsed.detected_cells) == 4
    assert all(cell.center_x > 0 for cell in parsed.detected_cells)
    assert all(cell.center_y > 0 for cell in parsed.detected_cells)
    assert [cell.state for cell in parsed.detected_cells] == ["plus", "empty", "plus", "red"]


def test_build_prefill_annotation_payload_from_svg_generates_expected_template(tmp_path: Path):
    svg_path = tmp_path / "second_row_escape_01.svg"
    svg_path.write_text(_SAMPLE_SVG, encoding="utf-8")

    payload = build_prefill_annotation_payload_from_svg(
        svg_path,
        template_name="second_row_escape_01",
        family="hexwiki_theory_of_ladder_escapes",
        notes="Generated directly from SVG; review in annotator before use.",
    )

    assert payload["metadata"]["name"] == "second_row_escape_01"
    assert payload["metadata"]["family"] == "hexwiki_theory_of_ladder_escapes"
    assert payload["metadata"]["open_left"] is True
    assert payload["metadata"]["open_right"] is False
    assert payload["grid"]["rows"] == 2
    assert payload["grid"]["cols"] == 2
    assert payload["image_overlay"]["image_name"] == "second_row_escape_01.svg"
    assert payload["cells"] == [
        {"row": 0, "col": 0, "state": "plus"},
        {"row": 0, "col": 1, "state": "empty"},
        {"row": 1, "col": 0, "state": "plus"},
        {"row": 1, "col": 1, "state": "red"},
    ]
