#!/usr/bin/env python3
"""Build a static HTML viewer for dead-cell counterfactual root-choice debug JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_records(path: Path, max_records: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
            if payload.get("event_type") != "dead_cell_counterfactual_root_choice":
                continue
            records.append(payload)
            if max_records > 0 and len(records) >= max_records:
                break
    records.sort(key=lambda r: float(r.get("counterfactual_abs_distance_from_0p5", 1.0)))
    return records


def _build_html(records: List[Dict[str, Any]], source_path: Path) -> str:
    records_json = json.dumps(records, ensure_ascii=True)
    source_path_json = json.dumps(str(source_path))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Dead-Cell Counterfactual Viewer</title>
  <style>
    :root {{
      --bg: #f6f5ef;
      --fg: #1d1c19;
      --muted: #6a675f;
      --panel: #fffdf7;
      --line: #d6d2c4;
      --blue: #1f4e8c;
      --red: #8c311f;
      --empty: #f3efe2;
      --counterfactual: #f5be3a;
      --actual: #0d6f5a;
      --accent: #0d6f5a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: \"Avenir Next\", \"Segoe UI\", sans-serif;
      color: var(--fg);
      background: radial-gradient(circle at 20% 0%, #fffdf7 0%, var(--bg) 58%);
    }}
    .layout {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 20px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: 0 10px 24px rgba(24, 28, 26, 0.06);
      padding: 14px 16px;
    }}
    .title {{
      margin: 0 0 8px 0;
      font-size: 1.2rem;
      color: var(--accent);
    }}
    .source {{
      margin: 0 0 12px 0;
      font-size: 0.86rem;
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
    .stats {{
      margin: 0 0 14px 0;
      font-size: 0.9rem;
      color: var(--muted);
    }}
    .control-block {{
      margin-bottom: 14px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }}
    .control-label {{
      margin: 0 0 8px 0;
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .rule-list {{
      display: grid;
      gap: 6px;
      max-height: 200px;
      overflow: auto;
      padding-right: 4px;
    }}
    .rule-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.92rem;
    }}
    select, button, input {{
      font: inherit;
      border-radius: 8px;
      border: 1px solid var(--line);
      padding: 8px 10px;
      background: #fff;
      color: var(--fg);
    }}
    .range-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 10px;
    }}
    .nav-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .nav-row button {{
      min-width: 72px;
      cursor: pointer;
    }}
    .nav-status {{
      font-size: 0.9rem;
      color: var(--muted);
    }}
    .legend {{
      margin: 0 0 10px 0;
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 0.9rem;
      color: var(--muted);
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .chip-mark {{
      width: 13px;
      height: 13px;
      border-radius: 99px;
      border: 1px solid var(--line);
      display: inline-block;
    }}
    .chip-mark.actual {{ outline: 3px solid var(--actual); outline-offset: 1px; }}
    .chip-mark.counterfactual {{ outline: 3px solid var(--counterfactual); outline-offset: 1px; }}
    .board-wrap {{
      overflow: auto;
      padding-bottom: 8px;
    }}
    .board {{
      display: inline-block;
      min-width: 420px;
    }}
    .board-row {{
      display: flex;
      align-items: center;
      margin-bottom: 3px;
    }}
    .row-label {{
      width: 24px;
      text-align: right;
      margin-right: 6px;
      color: var(--muted);
      font-size: 0.76rem;
      font-weight: 600;
    }}
    .cells {{
      display: flex;
      gap: 3px;
    }}
    .cell {{
      width: 24px;
      height: 24px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--empty);
      position: relative;
    }}
    .cell.blue {{ background: color-mix(in srgb, var(--blue) 24%, #ffffff); border-color: var(--blue); }}
    .cell.red {{ background: color-mix(in srgb, var(--red) 24%, #ffffff); border-color: var(--red); }}
    .cell.actual {{ outline: 3px solid var(--actual); outline-offset: 1px; }}
    .cell.counterfactual {{ box-shadow: inset 0 0 0 3px var(--counterfactual); }}
    .meta {{
      margin-top: 14px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
      display: grid;
      gap: 8px;
      font-size: 0.92rem;
    }}
    .meta-line {{ overflow-wrap: anywhere; }}
    .meta-key {{
      color: var(--muted);
      font-weight: 600;
      margin-right: 6px;
    }}
    .empty-state {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    @media (max-width: 980px) {{
      .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class=\"layout\">
    <section class=\"panel\">
      <h1 class=\"title\">Dead-Cell Counterfactual Viewer</h1>
      <p class=\"source\"><strong>Source:</strong> <span id=\"sourcePath\"></span></p>
      <p class=\"stats\" id=\"globalStats\"></p>

      <div class=\"control-block\">
        <p class=\"control-label\">Strategy Filter</p>
        <select id=\"strategyFilter\"></select>
      </div>

      <div class=\"control-block\">
        <p class=\"control-label\">Rule Filter (Any Match)</p>
        <div class=\"rule-list\" id=\"ruleList\"></div>
      </div>

      <div class=\"control-block\">
        <p class=\"control-label\">Near-Neutral Filter</p>
        <div class=\"range-row\">
          <input id=\"neutralRange\" type=\"range\" min=\"0\" max=\"0.5\" step=\"0.005\" value=\"0.5\" />
          <span id=\"neutralValue\">0.500</span>
        </div>
        <p class=\"stats\">Keep records where |counterfactual p - 0.5| <= threshold.</p>
      </div>

      <div class=\"control-block\">
        <p class=\"control-label\">Navigation</p>
        <div class=\"nav-row\">
          <button id=\"prevBtn\" type=\"button\">Prev</button>
          <button id=\"nextBtn\" type=\"button\">Next</button>
          <button id=\"randomBtn\" type=\"button\">Random</button>
        </div>
        <p class=\"nav-status\" id=\"navStatus\"></p>
      </div>
    </section>

    <section class=\"panel\">
      <p class=\"legend\">
        <span class=\"chip\"><span class=\"chip-mark actual\"></span>masked-MCTS selected move</span>
        <span class=\"chip\"><span class=\"chip-mark counterfactual\"></span>unmasked counterfactual move</span>
      </p>
      <div id=\"contentRoot\"></div>
    </section>
  </div>

  <script>
    const sourcePath = {source_path_json};
    const records = {records_json};

    const sourcePathEl = document.getElementById("sourcePath");
    const globalStatsEl = document.getElementById("globalStats");
    const strategyFilterEl = document.getElementById("strategyFilter");
    const ruleListEl = document.getElementById("ruleList");
    const neutralRangeEl = document.getElementById("neutralRange");
    const neutralValueEl = document.getElementById("neutralValue");
    const navStatusEl = document.getElementById("navStatus");
    const contentRootEl = document.getElementById("contentRoot");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const randomBtn = document.getElementById("randomBtn");

    const allRules = Array.from(new Set(records.flatMap((r) => r.counterfactual_move_rules || []))).sort();
    const allStrategies = Array.from(new Set(records.map((r) => r.strategy_label || "(unknown)"))).sort();
    let activeRules = new Set(allRules);
    let activeStrategy = "ALL";
    let maxNeutralDistance = 0.5;
    let filteredRecords = [];
    let index = 0;

    sourcePathEl.textContent = sourcePath;
    globalStatsEl.textContent = `Loaded ${{records.length}} records (sorted by counterfactual |p-0.5| ascending).`;

    function escapeHtml(text) {{
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function parseTrmphState(trmph) {{
      const m = /^#(\\d+),?(.*)$/.exec(trmph || "");
      if (!m) return null;
      const boardSize = Number(m[1]);
      const movesBlob = m[2] || "";
      const moveTokens = movesBlob.match(/[a-z]+\\d+/g) || [];
      const board = Array.from({{ length: boardSize }}, () => Array.from({{ length: boardSize }}, () => "e"));
      const colLabelToIdx = (label) => {{
        let value = 0;
        for (const ch of label) value = value * 26 + (ch.charCodeAt(0) - 96);
        return value - 1;
      }};
      for (let i = 0; i < moveTokens.length; i++) {{
        const token = moveTokens[i];
        const mm = /^([a-z]+)(\\d+)$/.exec(token);
        if (!mm) continue;
        const col = colLabelToIdx(mm[1]);
        const row = Number(mm[2]) - 1;
        if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) continue;
        board[row][col] = (i % 2 === 0) ? "b" : "r";
      }}
      return {{ boardSize, board }};
    }}

    function buildBoardHtml(entry) {{
      const parsed = parseTrmphState(entry.state_trmph);
      if (!parsed) return '<p class="empty-state">Could not parse TRMPH state.</p>';
      const actualRow = Number(entry.masked_strategy_move?.row);
      const actualCol = Number(entry.masked_strategy_move?.col);
      const cfRow = Number(entry.counterfactual_unmasked_move?.row);
      const cfCol = Number(entry.counterfactual_unmasked_move?.col);

      let rowsHtml = "";
      for (let r = 0; r < parsed.boardSize; r++) {{
        let cellsHtml = "";
        for (let c = 0; c < parsed.boardSize; c++) {{
          const piece = parsed.board[r][c];
          const classes = ["cell"];
          if (piece === "b") classes.push("blue");
          if (piece === "r") classes.push("red");
          if (r === actualRow && c === actualCol) classes.push("actual");
          if (r === cfRow && c === cfCol) classes.push("counterfactual");
          cellsHtml += `<div class="${{classes.join(" ")}}" title="(${{r}},${{c}})"></div>`;
        }}
        const rowIndentPx = r * 13;
        rowsHtml += `
          <div class="board-row" style="margin-left:${{rowIndentPx}}px">
            <div class="row-label">${{r + 1}}</div>
            <div class="cells">${{cellsHtml}}</div>
          </div>`;
      }}
      return `<div class="board-wrap"><div class="board">${{rowsHtml}}</div></div>`;
    }}

    function toFixedOrNa(value, digits = 4) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return Number(value).toFixed(digits);
    }}

    function buildMetaHtml(entry) {{
      const rules = (entry.counterfactual_move_rules || []).join(", ");
      const trmphHref = `https://trmph.com/hex/board${{entry.state_trmph || ""}}`;
      return `
        <div class="meta">
          <div class="meta-line"><span class="meta-key">Counterfactual Move (masked):</span>${{escapeHtml(entry.counterfactual_unmasked_move?.trmph || "(unknown)")}}</div>
          <div class="meta-line"><span class="meta-key">Masked-MCTS Move:</span>${{escapeHtml(entry.masked_strategy_move?.trmph || "(unknown)")}}</div>
          <div class="meta-line"><span class="meta-key">Rules:</span>${{escapeHtml(rules || "(none)")}}</div>
          <div class="meta-line"><span class="meta-key">Strategy:</span>${{escapeHtml(entry.strategy_label || "(unknown)")}}</div>
          <div class="meta-line"><span class="meta-key">Player To Move:</span>${{escapeHtml(entry.current_player || "?")}}</div>
          <div class="meta-line"><span class="meta-key">Move Index To Play:</span>${{escapeHtml(entry.move_index_to_play)}}</div>
          <div class="meta-line"><span class="meta-key">Counterfactual p(root):</span>${{toFixedOrNa(entry.counterfactual_root_win_probability, 4)}} (|p-0.5|=${{toFixedOrNa(entry.counterfactual_abs_distance_from_0p5, 4)}})</div>
          <div class="meta-line"><span class="meta-key">Masked p(root):</span>${{toFixedOrNa(entry.masked_root_win_probability, 4)}} (|p-0.5|=${{toFixedOrNa(entry.masked_abs_distance_from_0p5, 4)}})</div>
          <div class="meta-line"><span class="meta-key">Delta (unmasked-masked):</span>${{toFixedOrNa(entry.win_probability_delta_unmasked_minus_masked, 4)}}</div>
          <div class="meta-line"><span class="meta-key">Legal Moves:</span>masked ${{escapeHtml(entry.legal_moves_after_mask_count)}} | unmasked ${{escapeHtml(entry.legal_moves_unmasked_count)}}</div>
          <div class="meta-line"><span class="meta-key">Detected Dead Cells:</span>${{escapeHtml(entry.dead_cells_detected_total)}}</div>
          <div class="meta-line"><span class="meta-key">State TRMPH:</span>${{escapeHtml(entry.state_trmph || "")}}</div>
          <div class="meta-line"><a href="${{trmphHref}}" target="_blank" rel="noreferrer noopener">Open in TRMPH viewer</a></div>
        </div>`;
    }}

    function render() {{
      if (!filteredRecords.length) {{
        navStatusEl.textContent = "0 matches";
        contentRootEl.innerHTML = '<p class="empty-state">No records match current filters.</p>';
        return;
      }}
      const entry = filteredRecords[index];
      navStatusEl.textContent = `${{index + 1}} / ${{filteredRecords.length}}`;
      contentRootEl.innerHTML = `${{buildBoardHtml(entry)}}${{buildMetaHtml(entry)}}`;
    }}

    function recomputeFiltered() {{
      filteredRecords = records.filter((entry) => {{
        const strategyOk = activeStrategy === "ALL" || (entry.strategy_label || "(unknown)") === activeStrategy;
        const rules = entry.counterfactual_move_rules || [];
        const ruleOk = rules.some((r) => activeRules.has(r));
        const neutralOk = Number(entry.counterfactual_abs_distance_from_0p5 || 1.0) <= maxNeutralDistance;
        return strategyOk && ruleOk && neutralOk;
      }});
      if (index >= filteredRecords.length) index = 0;
      render();
    }}

    function buildControls() {{
      strategyFilterEl.innerHTML = `<option value="ALL">All Strategies</option>` +
        allStrategies.map((s) => `<option value="${{escapeHtml(s)}}">${{escapeHtml(s)}}</option>`).join("");
      strategyFilterEl.addEventListener("change", () => {{
        activeStrategy = strategyFilterEl.value;
        recomputeFiltered();
      }});

      ruleListEl.innerHTML = allRules.map((rule) => `
        <label class="rule-item">
          <input type="checkbox" data-rule="${{escapeHtml(rule)}}" checked />
          <span>${{escapeHtml(rule)}}</span>
        </label>
      `).join("");
      ruleListEl.addEventListener("change", () => {{
        activeRules = new Set(
          Array.from(ruleListEl.querySelectorAll("input[type=checkbox]"))
            .filter((el) => el.checked)
            .map((el) => el.dataset.rule)
        );
        recomputeFiltered();
      }});

      neutralRangeEl.addEventListener("input", () => {{
        maxNeutralDistance = Number(neutralRangeEl.value);
        neutralValueEl.textContent = maxNeutralDistance.toFixed(3);
        recomputeFiltered();
      }});

      prevBtn.addEventListener("click", () => {{
        if (!filteredRecords.length) return;
        index = (index - 1 + filteredRecords.length) % filteredRecords.length;
        render();
      }});
      nextBtn.addEventListener("click", () => {{
        if (!filteredRecords.length) return;
        index = (index + 1) % filteredRecords.length;
        render();
      }});
      randomBtn.addEventListener("click", () => {{
        if (!filteredRecords.length) return;
        index = Math.floor(Math.random() * filteredRecords.length);
        render();
      }});
    }}

    buildControls();
    recomputeFiltered();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static HTML viewer from dead-cell counterfactual JSONL logs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to dead-cell counterfactual JSONL log produced by run_tournament.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("temp/dead_cell_debug/deadmask_counterfactual_events.html"),
        help=(
            "Path to output HTML file "
            "(default: temp/dead_cell_debug/deadmask_counterfactual_events.html)."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Maximum records to load from JSONL (0 means all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input log does not exist: {args.input}")
    if args.max_records < 0:
        raise ValueError(f"--max-records must be >= 0, got {args.max_records}")

    records = _load_records(args.input, args.max_records)
    if not records:
        raise ValueError(
            f"No dead-cell counterfactual records found in {args.input}. "
            "Check that counterfactual dead-cell debug logging is enabled."
        )

    html = _build_html(records, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
