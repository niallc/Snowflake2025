from pathlib import Path
import json

import pytest

from hex_ai.utils.ladder_templates.hexwiki_svg_scraper import (
    HexWikiSectionSpec,
    extract_hexwiki_svgs,
    find_hexwiki_headings,
    write_extracted_hexwiki_svgs,
)


_SAMPLE_HTML = """
<html>
  <body>
    <h2><span class="mw-headline" id="Intro">Intro</span></h2>
    <p>Intro text</p>
    <h3><span class="mw-headline" id="Examples">Examples</span></h3>
    <p>Examples text</p>
    <div><svg width="100px" height="80px" viewBox="0 0 100 80"><defs><path id="hexes0" class="bodypath" d="M0,0z"/></defs><g><use xlink:href="#hexes0" filter="url(#shadow)"/><use xlink:href="#edge00" style="stroke:red"/></g><circle class="vertcirc" cx="10" cy="10" r="5"/></svg></div>
    <h4><span class="mw-headline" id="Nested">Nested</span></h4>
    <div><svg width="120px" height="90px" viewBox="0 0 120 90"><defs><path id="hexes1" class="bodypath" d="M0,0z"/></defs><g><use xlink:href="#hexes1" filter="url(#shadow)"/></g><circle class="vertcirc" cx="10" cy="10" r="5"/></svg></div>
    <h3><span class="mw-headline" id="Examples_2">Examples 2</span></h3>
    <div><svg width="140px" height="110px" viewBox="0 0 140 110"><defs><path id="hexes2" class="bodypath" d="M0,0z"/></defs><g><use xlink:href="#hexes2" filter="url(#shadow)"/><use xlink:href="#edge00" style="stroke:red"/></g></svg></div>
  </body>
</html>
"""


def test_find_hexwiki_headings_reads_headline_ids_and_levels():
    headings = find_hexwiki_headings(_SAMPLE_HTML)

    assert tuple((heading.anchor_id, heading.level) for heading in headings) == (
        ("Intro", 2),
        ("Examples", 3),
        ("Nested", 4),
        ("Examples_2", 3),
    )


def test_extract_hexwiki_svgs_uses_same_or_higher_heading_as_boundary():
    extracted = extract_hexwiki_svgs(
        _SAMPLE_HTML,
        source_page_url="https://example.test/page",
        section_specs=(HexWikiSectionSpec(anchor_id="Examples", filename_prefix="second_row_escape"),),
    )

    assert [item.filename for item in extracted] == [
        "second_row_escape_01.svg",
        "second_row_escape_02.svg",
    ]
    assert extracted[0].width == "100px"
    assert extracted[0].height == "80px"
    assert extracted[0].view_box == "0 0 100 80"
    assert 'xmlns:xlink="http://www.w3.org/1999/xlink"' in extracted[0].svg
    assert ".bodypath" in extracted[0].svg
    assert 'id="edge00"' in extracted[0].svg
    assert 'id="shadow"' in extracted[0].svg


def test_extract_hexwiki_svgs_raises_for_missing_anchor():
    with pytest.raises(ValueError, match="Section anchor 'Missing' not found"):
        extract_hexwiki_svgs(
            _SAMPLE_HTML,
            source_page_url="https://example.test/page",
            section_specs=(HexWikiSectionSpec(anchor_id="Missing", filename_prefix="nope"),),
        )


def test_write_extracted_hexwiki_svgs_writes_files_manifest_and_contact_sheet(tmp_path: Path):
    extracted = extract_hexwiki_svgs(
        _SAMPLE_HTML,
        source_page_url="https://example.test/page",
        section_specs=(
            HexWikiSectionSpec(anchor_id="Examples", filename_prefix="second_row_escape"),
            HexWikiSectionSpec(anchor_id="Examples_2", filename_prefix="third_row_escape"),
        ),
    )

    manifest = write_extracted_hexwiki_svgs(extracted, output_dir=tmp_path)

    assert manifest["count"] == 3
    assert manifest["by_prefix"] == {
        "second_row_escape": 2,
        "third_row_escape": 1,
    }
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["count"] == 3
    assert (tmp_path / "second_row_escape_01.svg").exists()
    assert (tmp_path / "third_row_escape_01.svg").exists()
    contact_sheet = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "HexWiki SVG Scrape Review" in contact_sheet
    assert "second_row_escape_01.svg" in contact_sheet
