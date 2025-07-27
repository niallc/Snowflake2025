// --- Color palette inspired by Taerim's nimbus ---
const COLORS = {
  LIGHT_BLUE: '#e7fcfc',
  MEDIUM_BLUE: '#bbeeee',
  DARK_BLUE: '#8bd6d6',
  VERY_DARK_BLUE: '#0099ff', // more vivid blue for edge
  DARKER_BLUE: '#0066cc', // even darker blue for last move
  LIGHT_RED: '#fff4ea',
  MEDIUM_RED: '#ffe1c8',
  DARK_RED: '#ffcea5',
  VERY_DARK_RED: '#ff6600', // more vivid orange-red for edge
  DARKER_RED: '#cc3300', // even darker red for last move
  LIGHT_GRAY: '#cccccc',
  BOARD_BG: '#f8f8fa',
  GRID: '#bbb',
  LAST_MOVE: '#222', // Keep for backward compatibility but won't use
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
  last_move_player: null, // Track which player made the last move
  blue_model_id: 'model1',
  red_model_id: 'model2',
  blue_search_widths: [],
  red_search_widths: [],
  blue_temperature: 0.2,
  red_temperature: 0.2,
  auto_step_active: false,
  auto_step_timeout: null,
  available_models: []
};

// --- Utility: Get per-player settings ---
function getCurrentPlayerSettings() {
  if (state.player === 'blue') {
    return {
      model_id: state.blue_model_id,
      search_widths: state.blue_search_widths,
      temperature: state.blue_temperature
    };
  } else {
    return {
      model_id: state.red_model_id,
      search_widths: state.red_search_widths,
      temperature: state.red_temperature
    };
  }
}

// --- Utility: Convert (row, col) to TRMPH move ---
function rowcolToTrmph(row, col) {
  return String.fromCharCode(97 + col) + (row + 1);
}

// --- API Calls ---
async function fetchModels() {
  const resp = await fetch('/api/models', {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

async function fetchState(trmph, model_id = 'model1', temperature = 1.0) {
  const resp = await fetch('/api/state', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trmph, model_id, temperature }),
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

async function fetchMove(trmph, move, model_id = 'model1', search_widths = [], temperature = 1.0, 
                       blue_model_id = 'model1', blue_search_widths = [], blue_temperature = 1.0,
                       red_model_id = 'model2', red_search_widths = [], red_temperature = 1.0) {
  const resp = await fetch('/api/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      trmph, 
      move, 
      model_id, 
      search_widths, 
      temperature,
      blue_model_id,
      blue_search_widths,
      blue_temperature,
      red_model_id,
      red_search_widths,
      red_temperature
    }),
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

async function applyHumanMove(trmph, move, model_id = 'model1', temperature = 1.0) {
  const resp = await fetch('/api/apply_move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      trmph, 
      move, 
      model_id, 
      temperature
    }),
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

async function makeComputerMove(trmph, model_id, search_widths = [], temperature = 1.0) {
  const resp = await fetch('/api/computer_move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trmph, model_id, search_widths, temperature }),
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

// --- Board Rendering ---
function drawBoard(container, board, legalMoves, lastMove, winner, lastMovePlayer) {
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
      
      // Highlight last move with darker color based on player
      if (lastMove && lastMove[0] === row && lastMove[1] === col) {
        if (lastMovePlayer === 'blue') {
          fill = COLORS.DARKER_BLUE;
        } else if (lastMovePlayer === 'red') {
          fill = COLORS.DARKER_RED;
        }
      }
      
      if (winner === 'blue' && cell === 1) fill = COLORS.VERY_DARK_BLUE;
      if (winner === 'red' && cell === 2) fill = COLORS.VERY_DARK_RED;
      const isLegal = legalMoves.includes(rowcolToTrmph(row, col));
      const hex = makeHex(x, y, HEX_RADIUS, fill, isLegal);
      hex.setAttribute('data-row', row);
      hex.setAttribute('data-col', col);
      if (isLegal && !winner && !state.auto_step_active) {
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
  drawBoard(boardContainer, state.board, state.legal_moves, state.last_move, state.winner, state.last_move_player);
  
  // Status
  const status = document.getElementById('status-line');
  if (state.winner) {
    status.textContent = `Game over: ${state.winner} wins!`;
    status.style.color = state.winner === 'blue' ? COLORS.VERY_DARK_BLUE : COLORS.VERY_DARK_RED;
  } else if (state.auto_step_active) {
    status.textContent = `Auto-stepping: ${state.player[0].toUpperCase() + state.player.slice(1)}'s turn`;
    status.style.color = state.player === 'blue' ? COLORS.VERY_DARK_BLUE : COLORS.VERY_DARK_RED;
  } else {
    status.textContent = `${state.player[0].toUpperCase() + state.player.slice(1)}'s turn`;
    status.style.color = state.player === 'blue' ? COLORS.VERY_DARK_BLUE : COLORS.VERY_DARK_RED;
  }
  
  // TRMPH
  document.getElementById('trmph-string').value = state.trmph;
  
  // Update step button
  const stepBtn = document.getElementById('step-btn');
  if (state.winner) {
    stepBtn.textContent = 'Game Over';
    stepBtn.disabled = true;
  } else {
    stepBtn.textContent = 'Step (Computer Move)';
    stepBtn.disabled = false;
  }
  
  // Disable controls during auto-step (but allow interruption)
  const controls = document.querySelectorAll('select, input:not(#auto-step-checkbox), button:not(#step-btn)');
  controls.forEach(control => {
    control.disabled = state.auto_step_active;
  });
}

// --- Event Handlers ---
async function onCellClick(e) {
  if (state.auto_step_active || state.winner) return;
  
  const row = parseInt(e.target.getAttribute('data-row'));
  const col = parseInt(e.target.getAttribute('data-col'));
  const { model_id, search_widths, temperature } = getCurrentPlayerSettings();
  
  // Store the move that was just made
  const moveMade = [row, col];
  const currentPlayer = state.player;
  
  try {
    // Step 1: Immediately apply the human move and show it
    const humanResult = await applyHumanMove(
      state.trmph, 
      rowcolToTrmph(row, col), 
      model_id, 
      temperature
    );
    
    // Update state with human move
    state.trmph = humanResult.new_trmph;
    state.board = humanResult.board;
    state.player = humanResult.player;
    state.legal_moves = humanResult.legal_moves;
    state.winner = humanResult.winner;
    state.last_move = moveMade;
    state.last_move_player = currentPlayer;
    updateUI();
    
    // Step 2: If game is not over, get the computer move
    if (!state.winner) {
      // Determine which player's settings to use for the computer move
      const computerPlayer = state.player; // Current player after human move
      let computerModelId, computerSearchWidths, computerTemperature;
      
      if (computerPlayer === 'blue') {
        computerModelId = state.blue_model_id;
        computerSearchWidths = state.blue_search_widths;
        computerTemperature = state.blue_temperature;
      } else {
        computerModelId = state.red_model_id;
        computerSearchWidths = state.red_search_widths;
        computerTemperature = state.red_temperature;
      }
      
      // Make computer move
      const computerResult = await makeComputerMove(
        state.trmph, 
        computerModelId, 
        computerSearchWidths, 
        computerTemperature
      );
      
      if (computerResult.success) {
        state.trmph = computerResult.new_trmph;
        state.board = computerResult.board;
        state.player = computerResult.player;
        state.legal_moves = computerResult.legal_moves;
        state.winner = computerResult.winner;
        state.last_move = computerResult.move_made ? getLastMoveFromTrmph(computerResult.move_made) : null;
        state.last_move_player = computerPlayer;
        updateUI();
      } else {
        alert('Computer move failed: ' + computerResult.error);
      }
    }
  } catch (err) {
    alert('Move failed: ' + err.message);
  }
}

function getLastMove(board, legalMoves) {
  // This function is no longer used since we track the last move directly
  // in the onCellClick and stepComputerMove functions
  return null;
}

// --- Computer move functionality ---
async function stepComputerMove() {
  if (state.winner) return;
  
  const { model_id, search_widths, temperature } = getCurrentPlayerSettings();
  const currentPlayer = state.player; // Store current player before the move
  
  try {
    const result = await makeComputerMove(state.trmph, model_id, search_widths, temperature);
    
    if (result.success) {
      state.trmph = result.new_trmph;
      state.board = result.board;
      state.player = result.player;
      state.legal_moves = result.legal_moves;
      state.winner = result.winner;
      state.last_move = result.move_made ? getLastMoveFromTrmph(result.move_made) : null;
      state.last_move_player = currentPlayer; // Use the player who made the move
      updateUI();
      
      // If auto-step is active and game isn't over, schedule next move
      if (state.auto_step_active && !state.winner) {
        const delay = parseInt(document.getElementById('step-delay').value);
        state.auto_step_timeout = setTimeout(stepComputerMove, delay);
      }
    } else {
      alert('Computer move failed: ' + result.error);
    }
  } catch (err) {
    alert('Computer move failed: ' + err.message);
  }
}

function getLastMoveFromTrmph(moveTrmph) {
  // Convert TRMPH move to row, col for highlighting
  const col = moveTrmph.charCodeAt(0) - 97; // 'a' = 0, 'b' = 1, etc.
  const row = parseInt(moveTrmph.slice(1)) - 1; // '1' = 0, '2' = 1, etc.
  return [row, col];
}

function stopAutoStep() {
  state.auto_step_active = false;
  if (state.auto_step_timeout) {
    clearTimeout(state.auto_step_timeout);
    state.auto_step_timeout = null;
  }
  updateUI();
}

// --- Controls ---
document.addEventListener('DOMContentLoaded', async () => {
  // Load available models
  try {
    const modelsResult = await fetchModels();
    state.available_models = modelsResult.models;
    
    // Update model selection dropdowns
    const blueSelect = document.getElementById('blue-model');
    const redSelect = document.getElementById('red-model');
    
    blueSelect.innerHTML = '';
    redSelect.innerHTML = '';
    
    state.available_models.forEach(model => {
      const option = document.createElement('option');
      option.value = model.id;
      option.textContent = model.name;
      blueSelect.appendChild(option.cloneNode(true));
      redSelect.appendChild(option);
    });
    
    // Set default selections
    blueSelect.value = state.blue_model_id;
    redSelect.value = state.red_model_id;
  } catch (err) {
    console.error('Failed to load models:', err);
  }

  // Initial state fetch
  try {
    // Use blue's settings for initial fetch
    const result = await fetchState(state.trmph, state.blue_model_id, state.blue_temperature);
    state.board = result.board;
    state.player = result.player;
    state.legal_moves = result.legal_moves;
    state.winner = result.winner;
    state.last_move = null;
    state.last_move_player = null; // Initialize last_move_player
    updateUI();
  } catch (err) {
    document.getElementById('status-line').textContent = 'Failed to load board.';
  }

  // Model selection handlers
  document.getElementById('blue-model').addEventListener('change', (e) => {
    state.blue_model_id = e.target.value;
  });
  document.getElementById('red-model').addEventListener('change', (e) => {
    state.red_model_id = e.target.value;
  });

  // Blue search widths
  document.getElementById('blue-search-widths').addEventListener('input', (e) => {
    const value = e.target.value.trim();
    if (value) {
      state.blue_search_widths = value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    } else {
      state.blue_search_widths = [];
    }
  });
  // Red search widths
  document.getElementById('red-search-widths').addEventListener('input', (e) => {
    const value = e.target.value.trim();
    if (value) {
      state.red_search_widths = value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    } else {
      state.red_search_widths = [];
    }
  });
  // Blue temperature
  document.getElementById('blue-temperature').addEventListener('input', (e) => {
    state.blue_temperature = parseFloat(e.target.value);
  });
  // Red temperature
  document.getElementById('red-temperature').addEventListener('input', (e) => {
    state.red_temperature = parseFloat(e.target.value);
  });

  // Step button handler
  document.getElementById('step-btn').addEventListener('click', async () => {
    await stepComputerMove();
  });

  // Auto-step checkbox handler
  document.getElementById('auto-step-checkbox').addEventListener('change', (e) => {
    if (e.target.checked) {
      state.auto_step_active = true;
      updateUI();
      // Start auto-stepping
      stepComputerMove();
    } else {
      stopAutoStep();
    }
  });

  // Reset button
  document.getElementById('reset-btn').addEventListener('click', async () => {
    if (state.auto_step_active) {
      stopAutoStep();
      document.getElementById('auto-step-checkbox').checked = false;
    }
    
    state.trmph = '#13,';
    // Use blue's settings for reset
    const result = await fetchState(state.trmph, state.blue_model_id, state.blue_temperature);
    state.board = result.board;
    state.player = result.player;
    state.legal_moves = result.legal_moves;
    state.winner = result.winner;
    state.last_move = null;
    state.last_move_player = null; // Reset last_move_player
    updateUI();
  });

  // Copy TRMPH
  document.getElementById('copy-trmph').addEventListener('click', () => {
    navigator.clipboard.writeText(state.trmph);
  });
}); 