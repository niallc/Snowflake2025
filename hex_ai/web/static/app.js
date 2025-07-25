// --- Color palette inspired by Taerim's nimbus ---
const COLORS = {
  LIGHT_BLUE: '#e7fcfc',
  MEDIUM_BLUE: '#bbeeee',
  DARK_BLUE: '#8bd6d6',
  VERY_DARK_BLUE: '#0099ff', // more vivid blue for edge
  LIGHT_RED: '#fff4ea',
  MEDIUM_RED: '#ffe1c8',
  DARK_RED: '#ffcea5',
  VERY_DARK_RED: '#ff6600', // more vivid orange-red for edge
  LIGHT_GRAY: '#cccccc',
  BOARD_BG: '#f8f8fa',
  GRID: '#bbb',
  LAST_MOVE: '#222',
};

const BOARD_SIZE = 13;
const HEX_RADIUS = 16; // px, radius of each hex

// --- State ---
let state = {
  trmph: `#13,`,
  board: [],
  player: 'blue',
  legal_moves: [],
  winner: null,
  last_move: null,
};

// --- Utility: Convert (row, col) to TRMPH move ---
function rowcolToTrmph(row, col) {
  return String.fromCharCode(97 + col) + (row + 1);
}

// --- API Calls ---
async function fetchState(trmph) {
  const resp = await fetch('/api/state', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trmph }),
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

async function fetchMove(trmph, move) {
  const resp = await fetch('/api/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trmph, move }),
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

// --- Board Rendering ---
function drawBoard(container, board, legalMoves, lastMove, winner) {
  container.innerHTML = '';
  // Math for flat-topped hex grid, blue at top/bottom
  const w = HEX_RADIUS * Math.sqrt(3);
  const h = HEX_RADIUS * 1.5;
  // Make the SVG area wider and taller for full edge visibility
  const svgWidth = 1.5 * (w * (BOARD_SIZE - 1 + 0.5) + 2 * HEX_RADIUS);
  const svgHeight = 1.2 * (h * (BOARD_SIZE - 1) + 2 * HEX_RADIUS);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', svgWidth);
  svg.setAttribute('height', svgHeight);
  svg.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);
  svg.style.background = COLORS.BOARD_BG;

  // --- Draw player edge indicators ---
  // Blue: top and bottom (across the topmost and bottommost hexes)
  svg.appendChild(makeEdgeLine(
    hexCenter(0, 0).x, hexCenter(0, 0).y - HEX_RADIUS,
    hexCenter(0, BOARD_SIZE - 1).x, hexCenter(0, BOARD_SIZE - 1).y - HEX_RADIUS,
    COLORS.VERY_DARK_BLUE
  ));
  svg.appendChild(makeEdgeLine(
    hexCenter(BOARD_SIZE - 1, 0).x, hexCenter(BOARD_SIZE - 1, 0).y + HEX_RADIUS,
    hexCenter(BOARD_SIZE - 1, BOARD_SIZE - 1).x, hexCenter(BOARD_SIZE - 1, BOARD_SIZE - 1).y + HEX_RADIUS,
    COLORS.VERY_DARK_BLUE
  ));
  // Red: left and right (use pi/4 for offset)
  const redAngle = Math.PI / 4;
  svg.appendChild(makeEdgeLine(
    hexCenter(0, 0).x - HEX_RADIUS * Math.cos(redAngle), hexCenter(0, 0).y + HEX_RADIUS * Math.sin(redAngle),
    hexCenter(BOARD_SIZE - 1, 0).x - HEX_RADIUS * Math.cos(redAngle), hexCenter(BOARD_SIZE - 1, 0).y - HEX_RADIUS * Math.sin(redAngle),
    COLORS.VERY_DARK_RED
  ));
  svg.appendChild(makeEdgeLine(
    hexCenter(0, BOARD_SIZE - 1).x + HEX_RADIUS * Math.cos(redAngle), hexCenter(0, BOARD_SIZE - 1).y + HEX_RADIUS * Math.sin(redAngle),
    hexCenter(BOARD_SIZE - 1, BOARD_SIZE - 1).x + HEX_RADIUS * Math.cos(redAngle), hexCenter(BOARD_SIZE - 1, BOARD_SIZE - 1).y - HEX_RADIUS * Math.sin(redAngle),
    COLORS.VERY_DARK_RED
  ));

  // Draw hexes
  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      const { x, y } = hexCenter(row, col);
      const cell = board[row]?.[col] || 0;
      let fill = '#fff';
      if (cell === 1) fill = COLORS.DARK_BLUE;
      if (cell === 2) fill = COLORS.DARK_RED;
      if (lastMove && lastMove[0] === row && lastMove[1] === col) fill = COLORS.LAST_MOVE;
      if (winner === 'blue' && cell === 1) fill = COLORS.VERY_DARK_BLUE;
      if (winner === 'red' && cell === 2) fill = COLORS.VERY_DARK_RED;
      const isLegal = legalMoves.includes(rowcolToTrmph(row, col));
      const hex = makeHex(x, y, HEX_RADIUS, fill, isLegal);
      hex.setAttribute('data-row', row);
      hex.setAttribute('data-col', col);
      if (isLegal && !winner) {
        hex.classList.add('clickable');
        hex.addEventListener('click', onCellClick);
      }
      svg.appendChild(hex);
    }
  }
  container.appendChild(svg);
}

function makeHex(cx, cy, r, fill, highlight) {
  const points = [];
  for (let i = 0; i < 6; i++) {
    // Flat-topped: angle starts at 0, increments by 60deg, rotate by pi/6
    const angle = Math.PI / 3 * i + Math.PI / 6;
    points.push([
      cx + r * Math.cos(angle),
      cy + r * Math.sin(angle)
    ]);
  }
  const hex = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
  hex.setAttribute('points', points.map(p => p.join(',')).join(' '));
  hex.setAttribute('fill', fill);
  hex.setAttribute('stroke', highlight ? COLORS.MEDIUM_BLUE : COLORS.GRID);
  hex.setAttribute('stroke-width', highlight ? 4 : 2);
  if (highlight) hex.style.cursor = 'pointer';
  return hex;
}

function makeEdgeLine(x1, y1, x2, y2, color) {
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', x1);
  line.setAttribute('y1', y1);
  line.setAttribute('x2', x2);
  line.setAttribute('y2', y2);
  line.setAttribute('stroke', color);
  line.setAttribute('stroke-width', 10);
  line.setAttribute('stroke-linecap', 'round');
  line.setAttribute('opacity', 0.25);
  return line;
}

function hexCenter(row, col) {
  // Flat-topped, blue at top/bottom: x = HEX_RADIUS * sqrt(3) * (col + row/2) + HEX_RADIUS
  // y = HEX_RADIUS * 1.5 * row + HEX_RADIUS
  const x = HEX_RADIUS * Math.sqrt(3) * (col + row / 2) + HEX_RADIUS + HEX_RADIUS * 0.25; // add margin
  const y = HEX_RADIUS * 1.5 * row + HEX_RADIUS + HEX_RADIUS * 0.25; // add margin
  return { x, y };
}

// --- UI Update ---
function updateUI() {
  const boardContainer = document.getElementById('board-container');
  drawBoard(boardContainer, state.board, state.legal_moves, state.last_move, state.winner);
  // Status
  const status = document.getElementById('status-line');
  if (state.winner) {
    status.textContent = `Game over: ${state.winner} wins!`;
    status.style.color = state.winner === 'blue' ? COLORS.VERY_DARK_BLUE : COLORS.VERY_DARK_RED;
  } else {
    status.textContent = `${state.player[0].toUpperCase() + state.player.slice(1)}'s turn`;
    status.style.color = state.player === 'blue' ? COLORS.VERY_DARK_BLUE : COLORS.VERY_DARK_RED;
  }
  // TRMPH
  document.getElementById('trmph-string').value = state.trmph;
}

// --- Event Handlers ---
async function onCellClick(e) {
  const row = parseInt(e.target.getAttribute('data-row'));
  const col = parseInt(e.target.getAttribute('data-col'));
  try {
    const result = await fetchMove(state.trmph, rowcolToTrmph(row, col));
    state.trmph = result.new_trmph;
    state.board = result.board;
    state.player = result.player;
    state.legal_moves = result.legal_moves;
    state.winner = result.winner;
    state.last_move = getLastMove(result.board, result.legal_moves);
    updateUI();
  } catch (err) {
    alert('Move failed: ' + err.message);
  }
}

function getLastMove(board, legalMoves) {
  // Find the most recent move by comparing board state and legal moves
  // (for now, just return null; can be improved with move history)
  return null;
}

// --- Controls ---
document.addEventListener('DOMContentLoaded', async () => {
  // Initial state fetch
  try {
    const result = await fetchState(state.trmph);
    state.board = result.board;
    state.player = result.player;
    state.legal_moves = result.legal_moves;
    state.winner = result.winner;
    state.last_move = null;
    updateUI();
  } catch (err) {
    document.getElementById('status-line').textContent = 'Failed to load board.';
  }

  // Reset button
  document.getElementById('reset-btn').addEventListener('click', async () => {
    state.trmph = '#13,';
    const result = await fetchState(state.trmph);
    state.board = result.board;
    state.player = result.player;
    state.legal_moves = result.legal_moves;
    state.winner = result.winner;
    state.last_move = null;
    updateUI();
  });

  // Copy TRMPH
  document.getElementById('copy-trmph').addEventListener('click', () => {
    navigator.clipboard.writeText(state.trmph);
  });
}); 