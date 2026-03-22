#!/usr/bin/env python3
"""Build a static HTML viewer for root weak-move debug JSONL logs."""

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

            event_type = payload.get("event_type")
            if event_type == "weak_move_filter_root":
                normalized = dict(payload)
                normalized["focus_move"] = normalized.get("filtered_move")
                normalized.setdefault("status", "dead")
                normalized.setdefault("vulnerable_reply_moves", [])
                payload = normalized
            elif event_type == "dead_cell_prune_root":
                normalized = dict(payload)
                normalized["focus_move"] = normalized.get("dead_cell")
                normalized["status"] = "dead"
                normalized["vulnerable_reply_moves"] = []
                payload = normalized
            else:
                continue

            records.append(payload)
            if max_records > 0 and len(records) >= max_records:
                break
    return records


def _build_html(records: List[Dict[str, Any]], source_path: Path) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Weak-Move Debug Viewer</title>
  <style>
    :root {
      --bg: #f6f5ef;
      --fg: #1d1c19;
      --muted: #6a675f;
      --panel: #fffdf7;
      --line: #d6d2c4;
      --blue: #2a73b7;
      --blue-dark: #184f81;
      --red: #c55231;
      --red-dark: #8c311f;
      --empty: #f3efe2;
      --focus-dead: #f5be3a;
      --focus-vulnerable: #0d6f5a;
      --reply: #8a5cf6;
      --accent: #0d6f5a;
      --board-stroke: #6f6a5f;
      --board-bg: #fbf8ee;
      --coord: #6c685e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--fg);
      background: radial-gradient(circle at 20% 0%, #fffdf7 0%, var(--bg) 58%);
    }
    .layout {
      max-width: 1360px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 20px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: 0 10px 24px rgba(24, 28, 26, 0.06);
      padding: 14px 16px;
    }
    .title {
      margin: 0 0 8px 0;
      font-size: 1.2rem;
      color: var(--accent);
    }
    .source {
      margin: 0 0 12px 0;
      font-size: 0.86rem;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .stats {
      margin: 0 0 14px 0;
      font-size: 0.9rem;
      color: var(--muted);
    }
    .control-block {
      margin-bottom: 14px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }
    .control-label {
      margin: 0 0 8px 0;
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }
    .rule-list {
      display: grid;
      gap: 6px;
      max-height: 220px;
      overflow: auto;
      padding-right: 4px;
    }
    .rule-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.92rem;
    }
    select, button {
      font: inherit;
      border-radius: 8px;
      border: 1px solid var(--line);
      padding: 8px 10px;
      background: #fff;
      color: var(--fg);
    }
    .nav-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .nav-row button {
      min-width: 72px;
      cursor: pointer;
    }
    .nav-status {
      font-size: 0.9rem;
      color: var(--muted);
    }
    .legend {
      margin: 0 0 12px 0;
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 0.9rem;
      color: var(--muted);
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .chip-mark {
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid var(--line);
      display: inline-block;
      background: var(--empty);
    }
    .chip-mark.blue {
      background: color-mix(in srgb, var(--blue) 72%, white);
      border-color: var(--blue-dark);
    }
    .chip-mark.red {
      background: color-mix(in srgb, var(--red) 72%, white);
      border-color: var(--red-dark);
    }
    .chip-mark.focus-dead {
      outline: 3px solid var(--focus-dead);
      outline-offset: 1px;
    }
    .chip-mark.focus-vulnerable {
      outline: 3px solid var(--focus-vulnerable);
      outline-offset: 1px;
    }
    .chip-mark.reply {
      box-shadow: inset 0 0 0 3px var(--reply);
    }
    .board-wrap {
      overflow: auto;
      padding-bottom: 8px;
      background: linear-gradient(180deg, rgba(255,255,255,0.45), rgba(243,239,226,0.55));
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }
    .hex-board {
      display: block;
      min-width: 560px;
    }
    .legend-copy {
      margin: 0 0 12px 0;
      font-size: 0.9rem;
      color: var(--muted);
    }
    .meta {
      margin-top: 14px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
      display: grid;
      gap: 8px;
      font-size: 0.92rem;
    }
    .meta-line { overflow-wrap: anywhere; }
    .meta-key {
      color: var(--muted);
      font-weight: 600;
      margin-right: 6px;
    }
    .empty-state {
      color: var(--muted);
      font-size: 0.95rem;
    }
    .svg-board-bg {
      fill: var(--board-bg);
      stroke: rgba(0, 0, 0, 0.04);
    }
    .edge-blue {
      stroke: var(--blue-dark);
      stroke-width: 18;
      stroke-linecap: round;
      opacity: 0.35;
    }
    .edge-red {
      stroke: var(--red-dark);
      stroke-width: 22;
      stroke-linecap: round;
      opacity: 0.35;
    }
    .cell-hex {
      fill: var(--empty);
      stroke: var(--board-stroke);
      stroke-width: 1.1;
    }
    .piece.blue {
      fill: var(--blue);
      stroke: var(--blue-dark);
      stroke-width: 1.8;
    }
    .piece.red {
      fill: var(--red);
      stroke: var(--red-dark);
      stroke-width: 1.8;
    }
    .focus-ring {
      fill: none;
      stroke-width: 4.2;
      stroke-linejoin: round;
    }
    .focus-ring.dead {
      stroke: var(--focus-dead);
    }
    .focus-ring.vulnerable {
      stroke: var(--focus-vulnerable);
    }
    .reply-ring {
      fill: none;
      stroke: var(--reply);
      stroke-width: 3;
      stroke-dasharray: 6 4;
      stroke-linejoin: round;
      opacity: 0.92;
    }
    .axis-label {
      fill: var(--coord);
      font-size: 12px;
      font-weight: 700;
      text-anchor: middle;
      dominant-baseline: middle;
    }
    .axis-label.side {
      text-anchor: end;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .hex-board { min-width: 480px; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <section class="panel">
      <h1 class="title">Weak-Move Debug Viewer</h1>
      <p class="source"><strong>Source:</strong> <span id="sourcePath"></span></p>
      <p class="stats" id="globalStats"></p>

      <div class="control-block">
        <p class="control-label">Strategy Filter</p>
        <select id="strategyFilter"></select>
      </div>

      <div class="control-block">
        <p class="control-label">Status Filter</p>
        <select id="statusFilter"></select>
      </div>

      <div class="control-block">
        <p class="control-label">Rule Filter (Any Match)</p>
        <div class="rule-list" id="ruleList"></div>
      </div>

      <div class="control-block">
        <p class="control-label">Navigation</p>
        <div class="nav-row">
          <button id="prevBtn" type="button">Prev</button>
          <button id="nextBtn" type="button">Next</button>
          <button id="randomBtn" type="button">Random</button>
        </div>
        <p class="nav-status" id="navStatus"></p>
      </div>
    </section>

    <section class="panel">
      <div class="legend">
        <span class="chip"><span class="chip-mark blue"></span>Blue stone</span>
        <span class="chip"><span class="chip-mark red"></span>Red stone</span>
        <span class="chip"><span class="chip-mark focus-dead"></span>Dead focus move</span>
        <span class="chip"><span class="chip-mark focus-vulnerable"></span>Vulnerable focus move</span>
        <span class="chip"><span class="chip-mark reply"></span>Vulnerable reply</span>
      </div>
      <p class="legend-copy">Board layout is now truly hexagonal, using the same neighbor geometry as the main web renderer.</p>
      <div id="contentRoot"></div>
    </section>
  </div>

  <script>
    const sourcePath = __SOURCE_PATH_JSON__;
    const records = __RECORDS_JSON__;

    const sourcePathEl = document.getElementById("sourcePath");
    const globalStatsEl = document.getElementById("globalStats");
    const strategyFilterEl = document.getElementById("strategyFilter");
    const statusFilterEl = document.getElementById("statusFilter");
    const ruleListEl = document.getElementById("ruleList");
    const navStatusEl = document.getElementById("navStatus");
    const contentRootEl = document.getElementById("contentRoot");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const randomBtn = document.getElementById("randomBtn");

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function normalizeRules(entry) {
      return Array.isArray(entry.rules) ? entry.rules : [];
    }

    function normalizeStatus(entry) {
      const raw = String(entry.status || "dead").toLowerCase();
      return raw === "vulnerable" ? "vulnerable" : "dead";
    }

    function parseTrmphState(trmph) {
      const m = /^#(\\d+),?(.*)$/.exec(trmph || "");
      if (!m) return null;
      const boardSize = Number(m[1]);
      const movesBlob = m[2] || "";
      const moveTokens = movesBlob.match(/[a-z]+\\d+/g) || [];
      const board = Array.from({ length: boardSize }, () => Array.from({ length: boardSize }, () => "e"));

      const colLabelToIdx = (label) => {
        let value = 0;
        for (const ch of label) value = value * 26 + (ch.charCodeAt(0) - 96);
        return value - 1;
      };

      for (let i = 0; i < moveTokens.length; i += 1) {
        const token = moveTokens[i];
        const mm = /^([a-z]+)(\\d+)$/.exec(token);
        if (!mm) continue;
        const col = colLabelToIdx(mm[1]);
        const row = Number(mm[2]) - 1;
        if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) continue;
        board[row][col] = (i % 2 === 0) ? "b" : "r";
      }
      return { boardSize, board, moveCount: moveTokens.length };
    }

    function colLabelFromIndex(index) {
      let value = index + 1;
      let label = "";
      while (value > 0) {
        const rem = (value - 1) % 26;
        label = String.fromCharCode(97 + rem) + label;
        value = Math.floor((value - 1) / 26);
      }
      return label;
    }

    function moveLabel(row, col) {
      return `${colLabelFromIndex(col)}${row + 1}`;
    }

    function moveKey(move) {
      if (!move || !Number.isFinite(Number(move.row)) || !Number.isFinite(Number(move.col))) {
        return null;
      }
      return `${Number(move.row)},${Number(move.col)}`;
    }

    function hexRadiusForBoard(boardSize) {
      if (boardSize <= 7) return 30;
      if (boardSize <= 11) return 24;
      if (boardSize <= 15) return 20;
      return 18;
    }

    function makeHexPoints(cx, cy, radius) {
      const points = [];
      for (let i = 0; i < 6; i += 1) {
        const angle = Math.PI / 3 * i + Math.PI / 6;
        points.push(`${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`);
      }
      return points.join(" ");
    }

    function hexVertices(cx, cy, radius) {
      const vertices = [];
      for (let i = 0; i < 6; i += 1) {
        const angle = Math.PI / 3 * i + Math.PI / 6;
        vertices.push({
          x: cx + radius * Math.cos(angle),
          y: cy + radius * Math.sin(angle),
        });
      }
      return vertices;
    }

    function edgeMidpoint(a, b) {
      return {
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
      };
    }

    function buildBoardHtml(entry) {
      const parsed = parseTrmphState(entry.state_trmph);
      if (!parsed) return '<p class="empty-state">Could not parse TRMPH state.</p>';

      const boardSize = parsed.boardSize;
      const board = parsed.board;
      const hexRadius = hexRadiusForBoard(boardSize);
      const dx = hexRadius * Math.sqrt(3);
      const dy = hexRadius * 1.5;
      const originX = hexRadius * 3.1;
      const originY = hexRadius * 2.7;

      function hexCenter(row, col) {
        return {
          x: originX + dx * (col + row / 2),
          y: originY + dy * row,
        };
      }

      const topLeft = hexCenter(0, 0);
      const topRight = hexCenter(0, boardSize - 1);
      const bottomLeft = hexCenter(boardSize - 1, 0);
      const bottomRight = hexCenter(boardSize - 1, boardSize - 1);
      const svgWidth = bottomRight.x + hexRadius * 3.2;
      const svgHeight = bottomRight.y + hexRadius * 3.0;
      const focusKey = moveKey(entry.focus_move);
      const focusStatus = normalizeStatus(entry);
      const replyKeys = new Set((entry.vulnerable_reply_moves || []).map(moveKey).filter(Boolean));

      const tl = hexVertices(topLeft.x, topLeft.y, hexRadius);
      const tr = hexVertices(topRight.x, topRight.y, hexRadius);
      const bl = hexVertices(bottomLeft.x, bottomLeft.y, hexRadius);
      const br = hexVertices(bottomRight.x, bottomRight.y, hexRadius);
      const leftTopMid = edgeMidpoint(tl[2], tl[3]);
      const leftBottomMid = edgeMidpoint(bl[2], bl[3]);
      const rightTopMid = edgeMidpoint(tr[0], tr[5]);
      const rightBottomMid = edgeMidpoint(br[0], br[5]);

      const svgParts = [];
      svgParts.push(`<svg class="hex-board" width="${svgWidth}" height="${svgHeight}" viewBox="0 0 ${svgWidth} ${svgHeight}" aria-label="Hex board">`);
      svgParts.push(`<rect class="svg-board-bg" x="0" y="0" width="${svgWidth}" height="${svgHeight}" rx="18" ry="18"></rect>`);
      svgParts.push(`<line class="edge-blue" x1="${topLeft.x}" y1="${topLeft.y - hexRadius}" x2="${topRight.x}" y2="${topRight.y - hexRadius}"></line>`);
      svgParts.push(`<line class="edge-blue" x1="${bottomLeft.x}" y1="${bottomLeft.y + hexRadius}" x2="${bottomRight.x}" y2="${bottomRight.y + hexRadius}"></line>`);
      svgParts.push(`<line class="edge-red" x1="${leftTopMid.x}" y1="${leftTopMid.y}" x2="${leftBottomMid.x}" y2="${leftBottomMid.y}"></line>`);
      svgParts.push(`<line class="edge-red" x1="${rightTopMid.x}" y1="${rightTopMid.y}" x2="${rightBottomMid.x}" y2="${rightBottomMid.y}"></line>`);

      for (let col = 0; col < boardSize; col += 1) {
        const center = hexCenter(0, col);
        svgParts.push(`<text class="axis-label" x="${center.x}" y="${center.y - hexRadius * 1.55}">${escapeHtml(colLabelFromIndex(col))}</text>`);
      }
      for (let row = 0; row < boardSize; row += 1) {
        const center = hexCenter(row, 0);
        svgParts.push(`<text class="axis-label side" x="${center.x - hexRadius * 1.65}" y="${center.y}">${row + 1}</text>`);
      }

      for (let row = 0; row < boardSize; row += 1) {
        for (let col = 0; col < boardSize; col += 1) {
          const center = hexCenter(row, col);
          const key = `${row},${col}`;
          const piece = board[row][col];
          const title = moveLabel(row, col);
          const basePoints = makeHexPoints(center.x, center.y, hexRadius);
          const focusPoints = makeHexPoints(center.x, center.y, hexRadius * 0.92);
          const replyPoints = makeHexPoints(center.x, center.y, hexRadius * 0.72);

          svgParts.push(`<g>`);
          svgParts.push(`<title>${escapeHtml(title)}</title>`);
          svgParts.push(`<polygon class="cell-hex" points="${basePoints}"></polygon>`);

          if (key === focusKey) {
            svgParts.push(`<polygon class="focus-ring ${focusStatus}" points="${focusPoints}"></polygon>`);
          }
          if (replyKeys.has(key)) {
            svgParts.push(`<polygon class="reply-ring" points="${replyPoints}"></polygon>`);
          }
          if (piece === "b") {
            svgParts.push(`<circle class="piece blue" cx="${center.x}" cy="${center.y}" r="${hexRadius * 0.58}"></circle>`);
          } else if (piece === "r") {
            svgParts.push(`<circle class="piece red" cx="${center.x}" cy="${center.y}" r="${hexRadius * 0.58}"></circle>`);
          }
          svgParts.push(`</g>`);
        }
      }

      svgParts.push(`</svg>`);
      return `<div class="board-wrap">${svgParts.join("")}</div>`;
    }

    function buildCountsSummary(entry) {
      const keepMoves = Number(entry.keep_moves_total);
      const safeMoves = Number(entry.safe_moves_total);
      const vulnerableMoves = Number(entry.vulnerable_moves_total);
      const deadMoves = Number(entry.dead_moves_total);
      const legalBefore = Number(entry.legal_moves_before_count);

      if (
        Number.isFinite(keepMoves) &&
        Number.isFinite(safeMoves) &&
        Number.isFinite(vulnerableMoves) &&
        Number.isFinite(deadMoves) &&
        Number.isFinite(legalBefore)
      ) {
        return `kept ${keepMoves} | safe ${safeMoves} | vulnerable ${vulnerableMoves} | dead ${deadMoves} | legal before ${legalBefore}`;
      }

      const pruned = Number(entry.pruned_moves_total);
      const detected = Number(entry.dead_cells_detected_total);
      if (Number.isFinite(pruned) && Number.isFinite(legalBefore) && Number.isFinite(detected)) {
        return `pruned ${pruned} at root | legal before ${legalBefore} | detected dead cells ${detected}`;
      }

      return "(counts unavailable)";
    }

    function buildMetaHtml(entry) {
      const focusMove = entry.focus_move || null;
      const focusLabel = focusMove?.trmph || (
        Number.isFinite(Number(focusMove?.row)) && Number.isFinite(Number(focusMove?.col))
          ? moveLabel(Number(focusMove.row), Number(focusMove.col))
          : "(unknown)"
      );
      const rules = normalizeRules(entry).join(", ");
      const replies = (entry.vulnerable_reply_moves || [])
        .map((move) => move?.trmph || moveLabel(Number(move.row), Number(move.col)))
        .filter(Boolean)
        .join(", ");
      const trmphHref = `https://trmph.com/hex/board${entry.state_trmph || ""}`;
      return `
        <div class="meta">
          <div class="meta-line"><span class="meta-key">Focus Move:</span>${escapeHtml(focusLabel)}</div>
          <div class="meta-line"><span class="meta-key">Status:</span>${escapeHtml(normalizeStatus(entry))}</div>
          <div class="meta-line"><span class="meta-key">Rules:</span>${escapeHtml(rules || "(none)")}</div>
          <div class="meta-line"><span class="meta-key">Vulnerable Replies:</span>${escapeHtml(replies || "(none)")}</div>
          <div class="meta-line"><span class="meta-key">Strategy:</span>${escapeHtml(entry.strategy_label || "(unknown)")}</div>
          <div class="meta-line"><span class="meta-key">Player To Move:</span>${escapeHtml(entry.current_player || "?")}</div>
          <div class="meta-line"><span class="meta-key">Move Index To Play:</span>${escapeHtml(entry.move_index_to_play)}</div>
          <div class="meta-line"><span class="meta-key">Filter Summary:</span>${escapeHtml(buildCountsSummary(entry))}</div>
          <div class="meta-line"><span class="meta-key">State TRMPH:</span>${escapeHtml(entry.state_trmph || "")}</div>
          <div class="meta-line"><a href="${trmphHref}" target="_blank" rel="noreferrer noopener">Open in TRMPH viewer</a></div>
        </div>`;
    }

    const allRules = Array.from(new Set(records.flatMap((r) => normalizeRules(r)))).sort();
    const allStrategies = Array.from(new Set(records.map((r) => r.strategy_label || "(unknown)"))).sort();
    const allStatuses = Array.from(new Set(records.map((r) => normalizeStatus(r)))).sort();

    let activeRules = new Set(allRules);
    let activeStrategy = "ALL";
    let activeStatus = "ALL";
    let filteredRecords = [];
    let index = 0;

    sourcePathEl.textContent = sourcePath;
    globalStatsEl.textContent = `Loaded ${records.length} records | Statuses: ${allStatuses.join(", ") || "(none)"} | Rules: ${allRules.join(", ") || "(none)"}`;

    function render() {
      if (!filteredRecords.length) {
        navStatusEl.textContent = "0 matches";
        contentRootEl.innerHTML = '<p class="empty-state">No records match current filters.</p>';
        return;
      }
      const entry = filteredRecords[index];
      navStatusEl.textContent = `${index + 1} / ${filteredRecords.length}`;
      contentRootEl.innerHTML = `
        ${buildBoardHtml(entry)}
        ${buildMetaHtml(entry)}
      `;
    }

    function recomputeFiltered() {
      filteredRecords = records.filter((entry) => {
        const strategyOk = activeStrategy === "ALL" || (entry.strategy_label || "(unknown)") === activeStrategy;
        const statusOk = activeStatus === "ALL" || normalizeStatus(entry) === activeStatus;
        const rules = normalizeRules(entry);
        const ruleOk = allRules.length === 0 || rules.some((rule) => activeRules.has(rule));
        return strategyOk && statusOk && ruleOk;
      });
      if (index >= filteredRecords.length) index = 0;
      render();
    }

    function buildControls() {
      strategyFilterEl.innerHTML = `<option value="ALL">All Strategies</option>` +
        allStrategies.map((s) => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`).join("");
      strategyFilterEl.addEventListener("change", () => {
        activeStrategy = strategyFilterEl.value;
        recomputeFiltered();
      });

      statusFilterEl.innerHTML = `<option value="ALL">All Statuses</option>` +
        allStatuses.map((s) => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`).join("");
      statusFilterEl.addEventListener("change", () => {
        activeStatus = statusFilterEl.value;
        recomputeFiltered();
      });

      if (allRules.length === 0) {
        ruleListEl.innerHTML = '<p class="empty-state">No rule labels in this log.</p>';
      } else {
        ruleListEl.innerHTML = allRules.map((rule) => `
          <label class="rule-item">
            <input type="checkbox" data-rule="${escapeHtml(rule)}" checked />
            <span>${escapeHtml(rule)}</span>
          </label>
        `).join("");
        ruleListEl.addEventListener("change", () => {
          activeRules = new Set(
            Array.from(ruleListEl.querySelectorAll("input[type=checkbox]"))
              .filter((el) => el.checked)
              .map((el) => el.dataset.rule)
          );
          recomputeFiltered();
        });
      }

      prevBtn.addEventListener("click", () => {
        if (!filteredRecords.length) return;
        index = (index - 1 + filteredRecords.length) % filteredRecords.length;
        render();
      });
      nextBtn.addEventListener("click", () => {
        if (!filteredRecords.length) return;
        index = (index + 1) % filteredRecords.length;
        render();
      });
      randomBtn.addEventListener("click", () => {
        if (!filteredRecords.length) return;
        index = Math.floor(Math.random() * filteredRecords.length);
        render();
      });
    }

    buildControls();
    recomputeFiltered();
  </script>
</body>
</html>
"""
    return (
        template.replace("__RECORDS_JSON__", json.dumps(records, ensure_ascii=True))
        .replace("__SOURCE_PATH_JSON__", json.dumps(str(source_path)))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static HTML viewer from root weak-move debug JSONL logs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to weak-move/dead-cell debug JSONL log produced by tournament debug mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("temp/dead_cell_debug_viewer.html"),
        help="Path to output HTML file (default: temp/dead_cell_debug_viewer.html).",
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
            f"No root weak-move records found in {args.input}. "
            "Check that dead-cell debug logging is enabled for masked MCTS strategies."
        )

    html = _build_html(records, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
