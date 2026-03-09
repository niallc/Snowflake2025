(function (global) {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const DEFAULT_HEX_RADIUS = 18;

  function readCssToken(name, fallback) {
    if (typeof document === 'undefined' || !document.documentElement) {
      return fallback;
    }
    const raw = getComputedStyle(document.documentElement).getPropertyValue(name);
    const trimmed = typeof raw === 'string' ? raw.trim() : '';
    return trimmed || fallback;
  }

  function createSvgElement(tagName) {
    return document.createElementNS(SVG_NS, tagName);
  }

  function displayColorForInternalSide(globalObject, side) {
    const themeApi = globalObject && globalObject.HexThemePreferences;
    if (themeApi && typeof themeApi.getDisplayColorKeyForInternalSide === 'function') {
      return themeApi.getDisplayColorKeyForInternalSide(side);
    }
    return side === 'red' ? 'red' : 'blue';
  }

  function hexCenter(row, col, hexRadius) {
    const padding = hexRadius * 0.5;
    const extraTopPadding = hexRadius * 0.5;
    const extraLeftPadding = hexRadius * 0.5;
    return {
      x: hexRadius * Math.sqrt(3) * (col + row / 2) + hexRadius + padding + extraLeftPadding,
      y: hexRadius * 1.5 * row + hexRadius + padding + extraTopPadding,
    };
  }

  function hexVertices(cx, cy, r) {
    const vertices = [];
    for (let i = 0; i < 6; i += 1) {
      const angle = Math.PI / 3 * i + Math.PI / 6;
      vertices.push({
        x: cx + r * Math.cos(angle),
        y: cy + r * Math.sin(angle),
      });
    }
    return vertices;
  }

  function edgeMidpoint(vertexA, vertexB) {
    return {
      x: (vertexA.x + vertexB.x) / 2,
      y: (vertexA.y + vertexB.y) / 2,
    };
  }

  function makeHex(cx, cy, r, fill, stroke) {
    const points = [];
    for (let i = 0; i < 6; i += 1) {
      const angle = Math.PI / 3 * i + Math.PI / 6;
      points.push([
        cx + r * Math.cos(angle),
        cy + r * Math.sin(angle),
      ]);
    }
    const polygon = createSvgElement('polygon');
    polygon.setAttribute('points', points.map((point) => point.join(',')).join(' '));
    polygon.setAttribute('fill', fill);
    polygon.setAttribute('stroke', stroke);
    polygon.setAttribute('stroke-width', '1');
    return polygon;
  }

  function makeEdgeLine(x1, y1, x2, y2, color, width) {
    const line = createSvgElement('line');
    line.setAttribute('x1', String(x1));
    line.setAttribute('y1', String(y1));
    line.setAttribute('x2', String(x2));
    line.setAttribute('y2', String(y2));
    line.setAttribute('stroke', color);
    line.setAttribute('stroke-width', String(width));
    line.setAttribute('stroke-linecap', 'round');
    line.setAttribute('opacity', '0.35');
    return line;
  }

  function makePieceDisc(cx, cy, r, fill, stroke) {
    const disc = createSvgElement('circle');
    disc.setAttribute('cx', String(cx));
    disc.setAttribute('cy', String(cy));
    disc.setAttribute('r', String(r * 0.62));
    disc.setAttribute('fill', fill);
    disc.setAttribute('stroke', stroke);
    disc.setAttribute('stroke-width', '1.6');
    return disc;
  }

  class BoardRenderer {
    constructor(container, options = {}) {
      if (!container) {
        throw new Error('BoardRenderer requires a container element');
      }
      this.container = container;
      this.hexRadius = Number.isFinite(options.hexRadius) ? options.hexRadius : DEFAULT_HEX_RADIUS;
    }

    getColors() {
      const blueDisplayColor = displayColorForInternalSide(global, 'blue');
      const redDisplayColor = displayColorForInternalSide(global, 'red');
      return {
        emptyHex: readCssToken('--board-cell-empty', '#f0f0f0'),
        boardBackground: readCssToken('--board-bg', '#f8f8fa'),
        hexStroke: readCssToken('--board-cell-stroke', '#555555'),
        bluePieceFill: readCssToken(`--board-piece-${blueDisplayColor}-fill`, '#0099ff'),
        bluePieceStroke: readCssToken(`--board-piece-${blueDisplayColor}-stroke`, '#0064af'),
        redPieceFill: readCssToken(`--board-piece-${redDisplayColor}-fill`, '#ff4444'),
        redPieceStroke: readCssToken(`--board-piece-${redDisplayColor}-stroke`, '#952500'),
        blueEdge: readCssToken(`--board-edge-${blueDisplayColor}`, '#0099ff'),
        redEdge: readCssToken(`--board-edge-${redDisplayColor}`, '#ff4444'),
        playedRecommendedFill: readCssToken('--review-played-recommended-fill', 'rgba(34, 117, 214, 0.16)'),
        playedRecommendedStroke: readCssToken('--review-played-recommended-stroke', '#2275d6'),
        playedCloseFill: readCssToken('--review-played-close-fill', 'rgba(214, 170, 46, 0.18)'),
        playedCloseStroke: readCssToken('--review-played-close-stroke', '#d6aa2e'),
        playedMistakeFill: readCssToken('--review-played-mistake-fill', 'rgba(191, 76, 83, 0.16)'),
        playedMistakeStroke: readCssToken('--review-played-mistake-stroke', '#bf4c53'),
        suggestionFill: readCssToken('--review-suggestion-fill', 'rgba(47, 153, 101, 0.18)'),
        suggestionStroke: readCssToken('--review-suggestion-stroke', '#2f9965'),
        suggestionText: readCssToken('--review-suggestion-text', '#184f35'),
      };
    }

    render(snapshot) {
      if (!snapshot || !Array.isArray(snapshot.board) || snapshot.board.length === 0) {
        throw new Error('BoardRenderer.render requires a non-empty board array');
      }

      const board = snapshot.board;
      const boardSize = Number.isFinite(snapshot.display_board_size)
        ? snapshot.display_board_size
        : board.length;
      const markers = Array.isArray(snapshot.markers) ? snapshot.markers : [];
      const colors = this.getColors();
      const w = this.hexRadius * Math.sqrt(3);
      const h = this.hexRadius * 1.5;
      const boardWidth = w * (boardSize - 1 + 0.5) + 2 * this.hexRadius;
      const boardHeight = h * (boardSize - 1) + 2 * this.hexRadius;
      const edgeBorderWidth = 17;
      const padding = this.hexRadius * 0.5;
      const diagonalOffset = w * (boardSize - 1) * 0.45;
      const svgWidth = boardWidth + 2 * padding + edgeBorderWidth + diagonalOffset;
      const svgHeight = boardHeight + 2 * padding + edgeBorderWidth;

      const svg = createSvgElement('svg');
      svg.setAttribute('width', String(svgWidth));
      svg.setAttribute('height', String(svgHeight));
      svg.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);
      svg.style.background = colors.boardBackground;

      this.drawEdges(svg, boardSize, colors);
      this.drawCells(svg, board, boardSize, colors);
      this.drawMarkers(svg, markers, colors);

      this.container.innerHTML = '';
      this.container.appendChild(svg);
    }

    drawEdges(svg, boardSize, colors) {
      const topLeft = hexCenter(0, 0, this.hexRadius);
      const topRight = hexCenter(0, boardSize - 1, this.hexRadius);
      const bottomLeft = hexCenter(boardSize - 1, 0, this.hexRadius);
      const bottomRight = hexCenter(boardSize - 1, boardSize - 1, this.hexRadius);

      svg.appendChild(
        makeEdgeLine(
          topLeft.x,
          topLeft.y - this.hexRadius,
          topRight.x,
          topRight.y - this.hexRadius,
          colors.blueEdge,
          18
        )
      );
      svg.appendChild(
        makeEdgeLine(
          bottomLeft.x,
          bottomLeft.y + this.hexRadius,
          bottomRight.x,
          bottomRight.y + this.hexRadius,
          colors.blueEdge,
          18
        )
      );

      const tl = hexVertices(topLeft.x, topLeft.y, this.hexRadius);
      const tr = hexVertices(topRight.x, topRight.y, this.hexRadius);
      const bl = hexVertices(bottomLeft.x, bottomLeft.y, this.hexRadius);
      const br = hexVertices(bottomRight.x, bottomRight.y, this.hexRadius);

      const leftTopMid = edgeMidpoint(tl[2], tl[3]);
      const leftBottomMid = edgeMidpoint(bl[2], bl[3]);
      const rightTopMid = edgeMidpoint(tr[0], tr[5]);
      const rightBottomMid = edgeMidpoint(br[0], br[5]);

      svg.appendChild(
        makeEdgeLine(leftTopMid.x, leftTopMid.y, leftBottomMid.x, leftBottomMid.y, colors.redEdge, 22)
      );
      svg.appendChild(
        makeEdgeLine(rightTopMid.x, rightTopMid.y, rightBottomMid.x, rightBottomMid.y, colors.redEdge, 22)
      );
    }

    drawCells(svg, board, boardSize, colors) {
      for (let row = 0; row < boardSize; row += 1) {
        for (let col = 0; col < boardSize; col += 1) {
          const cell = board[row] && board[row][col] ? String(board[row][col]) : 'e';
          const center = hexCenter(row, col, this.hexRadius);
          let fill = colors.emptyHex;
          if (cell === 'b') {
            fill = colors.emptyHex;
          } else if (cell === 'r') {
            fill = colors.emptyHex;
          }
          svg.appendChild(makeHex(center.x, center.y, this.hexRadius, fill, colors.hexStroke));

          if (cell === 'b') {
            svg.appendChild(
              makePieceDisc(center.x, center.y, this.hexRadius, colors.bluePieceFill, colors.bluePieceStroke)
            );
          } else if (cell === 'r') {
            svg.appendChild(
              makePieceDisc(center.x, center.y, this.hexRadius, colors.redPieceFill, colors.redPieceStroke)
            );
          }
        }
      }
    }

    drawMarkers(svg, markers, colors) {
      markers.forEach((marker) => {
        if (!marker || !Number.isFinite(marker.row) || !Number.isFinite(marker.col)) {
          return;
        }
        if (marker.kind === 'played') {
          this.drawPlayedMarker(svg, marker, colors);
          return;
        }
        this.drawSuggestionMarker(svg, marker, colors);
      });
    }

    drawPlayedMarker(svg, marker, colors) {
      const tone = marker && typeof marker.tone === 'string' ? marker.tone : 'mistake';
      const fillByTone = {
        recommended: colors.playedRecommendedFill,
        close: colors.playedCloseFill,
        mistake: colors.playedMistakeFill,
      };
      const strokeByTone = {
        recommended: colors.playedRecommendedStroke,
        close: colors.playedCloseStroke,
        mistake: colors.playedMistakeStroke,
      };
      const fill = fillByTone[tone] || colors.playedMistakeFill;
      const stroke = strokeByTone[tone] || colors.playedMistakeStroke;
      const center = hexCenter(marker.row, marker.col, this.hexRadius);
      const radius = Math.max(5.5, this.hexRadius * 0.44);
      const outer = createSvgElement('circle');
      outer.setAttribute('cx', String(center.x));
      outer.setAttribute('cy', String(center.y));
      outer.setAttribute('r', String(radius));
      outer.setAttribute('fill', fill);
      outer.setAttribute('stroke', stroke);
      outer.setAttribute('stroke-width', '2.5');
      svg.appendChild(outer);

      const arm = radius * 0.58;
      const lineA = makeEdgeLine(center.x - arm, center.y - arm, center.x + arm, center.y + arm, stroke, 2.6);
      const lineB = makeEdgeLine(center.x - arm, center.y + arm, center.x + arm, center.y - arm, stroke, 2.6);
      lineA.setAttribute('opacity', '1');
      lineB.setAttribute('opacity', '1');
      svg.appendChild(lineA);
      svg.appendChild(lineB);
    }

    drawSuggestionMarker(svg, marker, colors) {
      const center = hexCenter(marker.row, marker.col, this.hexRadius);
      const radius = Math.max(5, this.hexRadius * 0.4);
      const circle = createSvgElement('circle');
      circle.setAttribute('cx', String(center.x));
      circle.setAttribute('cy', String(center.y));
      circle.setAttribute('r', String(radius));
      circle.setAttribute('fill', colors.suggestionFill);
      circle.setAttribute('stroke', colors.suggestionStroke);
      circle.setAttribute('stroke-width', '2');
      svg.appendChild(circle);

      const text = createSvgElement('text');
      text.setAttribute('x', String(center.x));
      text.setAttribute('y', String(center.y + 0.8));
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('dominant-baseline', 'middle');
      text.setAttribute('font-size', String(Math.max(10, this.hexRadius * 0.62)));
      text.setAttribute('font-weight', '700');
      text.setAttribute('font-family', 'system-ui, sans-serif');
      text.setAttribute('fill', colors.suggestionText);
      text.textContent = marker.label || '?';
      svg.appendChild(text);
    }
  }

  global.HexBoardRenderer = {
    BoardRenderer,
  };
})(window);
