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
  red_model_id: 'model1',
  blue_search_widths: [],
  red_search_widths: [],
  blue_temperature: 0.2,
  red_temperature: 0.2,
  auto_step_active: false,
  auto_step_timeout: null,
  available_models: [],
  verbose_level: 3, // Debug output level
  computer_enabled: true, // Whether computer moves are enabled
  move_history: [], // Track move history for undo functionality
  constants: null // Will be populated from backend
};

// --- Game Constants (will be populated from backend) ---
let GAME_CONSTANTS = {
  BOARD_SIZE: 13, // Default fallback
  PIECE_VALUES: {
    EMPTY: 'e',
    BLUE: 'b', 
    RED: 'r'
  }
};

const HEX_RADIUS = 16; // px, radius of each hex

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
async function fetchConstants() {
  const resp = await fetch('/api/constants', {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

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
    body: JSON.stringify({ trmph, model_id, temperature, verbose: state.verbose_level }),
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

async function fetchMove(trmph, move, model_id = 'model1', search_widths = [], temperature = 1.0, 
                       blue_model_id = 'model1', blue_search_widths = [], blue_temperature = 1.0,
                       red_model_id = 'model2', red_search_widths = [], red_temperature = 1.0,
                       verbose = 3) {
  console.log(`fetchMove called with blue_model_id: ${blue_model_id}, red_model_id: ${red_model_id}`);
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
      red_temperature,
      verbose
    }),
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

async function applyHumanMove(trmph, move, model_id = 'model1', temperature = 1.0) {
  const resp = await fetch('/api/apply_move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trmph, move, model_id, temperature, verbose: state.verbose_level }),
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

async function makeComputerMove(trmph, model_id, search_widths = [], temperature = 1.0, verbose = 0) {
  console.log(`makeComputerMove called with model_id: ${model_id}`);
  const resp = await fetch('/api/computer_move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trmph, model_id, search_widths, temperature, verbose }),
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

// --- Board Rendering ---
function drawBoard(container, board, legalMoves, lastMove, winner, lastMovePlayer) {
  // Debug logging (only if verbose level >= 4)
  if (state.verbose_level >= 4) {
    debugBoardState(board, legalMoves, lastMove, winner, lastMovePlayer);
  }
  
  container.innerHTML = '';
  // Math for flat-topped hex grid, blue at top/bottom
  const w = HEX_RADIUS * Math.sqrt(3);
  const h = HEX_RADIUS * 1.5;
  // Make the SVG area wider and taller for full edge visibility
  const svgWidth = 1.5 * (w * (GAME_CONSTANTS.BOARD_SIZE - 1 + 0.5) + 2 * HEX_RADIUS);
  const svgHeight = 1.2 * (h * (GAME_CONSTANTS.BOARD_SIZE - 1) + 2 * HEX_RADIUS);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', svgWidth);
  svg.setAttribute('height', svgHeight);
  svg.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);
  svg.style.background = COLORS.BOARD_BG;

  // --- Draw player edge indicators ---
  // Blue: top and bottom (across the topmost and bottommost hexes)
  svg.appendChild(makeEdgeLine(
    hexCenter(0, 0).x, hexCenter(0, 0).y - HEX_RADIUS,
    hexCenter(0, GAME_CONSTANTS.BOARD_SIZE - 1).x, hexCenter(0, GAME_CONSTANTS.BOARD_SIZE - 1).y - HEX_RADIUS,
    COLORS.VERY_DARK_BLUE
  ));
  svg.appendChild(makeEdgeLine(
    hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, 0).x, hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, 0).y + HEX_RADIUS,
    hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, GAME_CONSTANTS.BOARD_SIZE - 1).x, hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, GAME_CONSTANTS.BOARD_SIZE - 1).y + HEX_RADIUS,
    COLORS.VERY_DARK_BLUE
  ));
  // Red: left and right (use pi/4 for offset)
  const redAngle = Math.PI / 4;
  svg.appendChild(makeEdgeLine(
    hexCenter(0, 0).x - HEX_RADIUS * Math.cos(redAngle), hexCenter(0, 0).y + HEX_RADIUS * Math.sin(redAngle),
    hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, 0).x - HEX_RADIUS * Math.cos(redAngle), hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, 0).y - HEX_RADIUS * Math.sin(redAngle),
    COLORS.VERY_DARK_RED
  ));
  svg.appendChild(makeEdgeLine(
    hexCenter(0, GAME_CONSTANTS.BOARD_SIZE - 1).x + HEX_RADIUS * Math.cos(redAngle), hexCenter(0, GAME_CONSTANTS.BOARD_SIZE - 1).y + HEX_RADIUS * Math.sin(redAngle),
    hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, GAME_CONSTANTS.BOARD_SIZE - 1).x + HEX_RADIUS * Math.cos(redAngle), hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, GAME_CONSTANTS.BOARD_SIZE - 1).y - HEX_RADIUS * Math.sin(redAngle),
    COLORS.VERY_DARK_RED
  ));

  // Draw hexes
  for (let row = 0; row < GAME_CONSTANTS.BOARD_SIZE; row++) {
    for (let col = 0; col < GAME_CONSTANTS.BOARD_SIZE; col++) {
      const { x, y } = hexCenter(row, col);
      const cell = board[row]?.[col] || GAME_CONSTANTS.PIECE_VALUES.EMPTY;
      let fill = '#fff';
      if (cell === GAME_CONSTANTS.PIECE_VALUES.BLUE) fill = COLORS.DARK_BLUE;
      if (cell === GAME_CONSTANTS.PIECE_VALUES.RED) fill = COLORS.DARK_RED;
      
      // Highlight last move with darker color based on player
      if (lastMove && lastMove[0] === row && lastMove[1] === col) {
        if (lastMovePlayer === 'blue') {
          fill = COLORS.DARKER_BLUE;
        } else if (lastMovePlayer === 'red') {
          fill = COLORS.DARKER_RED;
        }
      }
      
      if (winner === 'blue' && cell === GAME_CONSTANTS.PIECE_VALUES.BLUE) fill = COLORS.VERY_DARK_BLUE;
      if (winner === 'red' && cell === GAME_CONSTANTS.PIECE_VALUES.RED) fill = COLORS.VERY_DARK_RED;
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
  
  // Update debug controls
  const verboseLevel = document.getElementById('verbose-level');
  const computerToggle = document.getElementById('computer-toggle');
  if (verboseLevel) verboseLevel.value = state.verbose_level;
  if (computerToggle) computerToggle.textContent = state.computer_enabled ? 'ON' : 'OFF';
  
  // Show/hide debug output based on verbose level
  const debugOutput = document.getElementById('debug-output');
  if (debugOutput) {
    debugOutput.style.display = state.verbose_level > 0 ? 'block' : 'none';
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
  
  // Save state for undo functionality
  saveStateForUndo();
  
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
    
    // Step 2: If game is not over and computer is enabled, get the computer move
    if (!state.winner && state.computer_enabled) {
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
      
      // Make computer move with verbose output
      const computerResult = await makeComputerMove(
        state.trmph, 
        computerModelId, 
        computerSearchWidths, 
        computerTemperature,
        state.verbose_level
      );
      
      if (computerResult.success) {
        state.trmph = computerResult.new_trmph;
        state.board = computerResult.board;
        state.player = computerResult.player;
        state.legal_moves = computerResult.legal_moves;
        state.winner = computerResult.winner;
        state.last_move = computerResult.move_made ? getLastMoveFromTrmph(computerResult.move_made) : null;
        state.last_move_player = computerPlayer;
        
        // Display debug information if available
        if (computerResult.debug_info) {
          displayDebugInfo(computerResult.debug_info);
        }
        
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
    const result = await makeComputerMove(state.trmph, model_id, search_widths, temperature, state.verbose_level);
    
    if (result.success) {
      state.trmph = result.new_trmph;
      state.board = result.board;
      state.player = result.player;
      state.legal_moves = result.legal_moves;
      state.winner = result.winner;
      state.last_move = result.move_made ? getLastMoveFromTrmph(result.move_made) : null;
      state.last_move_player = currentPlayer; // Use the player who made the move
      
      // Display debug information if available
      if (result.debug_info) {
        displayDebugInfo(result.debug_info);
      }
      
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
  // Load game constants from backend
  try {
    const constantsResult = await fetchConstants();
    GAME_CONSTANTS = {
      BOARD_SIZE: constantsResult.BOARD_SIZE,
      PIECE_VALUES: constantsResult.PIECE_VALUES,
      PLAYER_VALUES: constantsResult.PLAYER_VALUES,
      WINNER_VALUES: constantsResult.WINNER_VALUES
    };
    state.constants = constantsResult;
    console.log('Loadedx game constants:', GAME_CONSTANTS);
  } catch (err) {
    console.error('Failed to load constants, using defaults:', err);
  }

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

  // Debug controls
  document.getElementById('verbose-level').addEventListener('change', (e) => {
    state.verbose_level = parseInt(e.target.value);
    updateUI();
  });

  document.getElementById('computer-toggle').addEventListener('click', () => {
    state.computer_enabled = !state.computer_enabled;
    updateUI();
  });

  document.getElementById('undo-btn').addEventListener('click', () => {
    if (state.move_history.length > 0) {
      const previousState = state.move_history.pop();
      Object.assign(state, previousState);
      updateUI();
    }
  });

  document.getElementById('clear-btn').addEventListener('click', async () => {
    state.trmph = '#13,';
    state.move_history = [];
    const result = await fetchState(state.trmph, state.blue_model_id, state.blue_temperature);
    state.board = result.board;
    state.player = result.player;
    state.legal_moves = result.legal_moves;
    state.winner = result.winner;
    state.last_move = null;
    state.last_move_player = null;
    updateUI();
  });
});

// --- Debug Output Functions ---
function displayDebugInfo(debugInfo) {
  if (!debugInfo || state.verbose_level === 0) return;
  
  const debugContent = document.getElementById('debug-content');
  if (!debugContent) return;
  
  let output = '';
  
  // Model information
  if (debugInfo.model_info) {
    output += '=== MODEL INFORMATION ===\n';
    output += `Model ID: ${debugInfo.model_info.model_id}\n`;
    output += `Model Type: ${debugInfo.model_info.model_type}\n`;
    output += `Model Path: ${debugInfo.model_info.model_path}\n`;
    output += '\n';
  }
  
  // Basic information
  if (debugInfo.basic) {
    output += '=== BASIC INFORMATION ===\n';
    output += `Current Player: ${debugInfo.basic.current_player}\n`;
    output += `Game Over: ${debugInfo.basic.game_over}\n`;
    output += `Legal Moves: ${debugInfo.basic.legal_moves_count}\n`;
    output += `Value Logit: ${debugInfo.basic.value_logit.toFixed(4)}\n`;
    output += `Win Probability: ${(debugInfo.basic.win_probability * 100).toFixed(2)}%\n`;
    output += `Temperature: ${debugInfo.basic.temperature}\n`;
    output += `Search Widths: ${debugInfo.basic.search_widths ? debugInfo.basic.search_widths.join(',') : 'None'}\n`;
    if (debugInfo.basic.model_move) {
      output += `Model Move: ${debugInfo.basic.model_move}\n`;
    }
    output += '\n';
  }
  
  // Policy analysis
  if (debugInfo.policy_analysis) {
    output += '=== POLICY ANALYSIS ===\n';
    
    // Post-temperature scaling (current behavior)
    output += `Top ${debugInfo.policy_analysis.top_moves.length} moves (post-temperature scaling):\n`;
    debugInfo.policy_analysis.top_moves.forEach((move, index) => {
      const probPercent = (move.probability * 100).toFixed(2);
      output += `  ${index + 1}. ${move.move} (${move.row},${move.col}): ${probPercent}%\n`;
    });
    
    // Pre-temperature scaling (raw logits)
    if (debugInfo.policy_analysis.raw_top_moves) {
      output += `\nTop ${debugInfo.policy_analysis.raw_top_moves.length} moves (raw logits, pre-temperature):\n`;
      debugInfo.policy_analysis.raw_top_moves.forEach((move, index) => {
        const logitStr = move.raw_logit.toFixed(4);
        output += `  ${index + 1}. ${move.move} (${move.row},${move.col}): ${logitStr}\n`;
      });
    }
    
    output += `Total legal moves: ${debugInfo.policy_analysis.total_legal_moves}\n\n`;
  }
  
  // Tree search analysis
  if (debugInfo.tree_search) {
    output += '=== TREE SEARCH ANALYSIS ===\n';
    if (debugInfo.tree_search.error) {
      output += `Error: ${debugInfo.tree_search.error}\n`;
    } else {
      output += `Search Widths: ${debugInfo.tree_search.search_widths.join(',')}\n`;
      output += `Tree Depth: ${debugInfo.tree_search.tree_depth}\n`;
      output += `Tree Size: ${debugInfo.tree_search.tree_size} nodes\n`;
      output += `Final Value: ${debugInfo.tree_search.final_value.toFixed(4)}\n`;
      output += `Best Move: ${debugInfo.tree_search.best_move || 'None'}\n`;
      
      // Terminal nodes (verbose level 3)
      if (debugInfo.tree_search.terminal_nodes && state.verbose_level >= 3) {
        output += '\nTerminal Nodes:\n';
        debugInfo.tree_search.terminal_nodes.forEach((node, index) => {
          const pathStr = node.path.join(' → ');
          const valueStr = node.value !== null ? node.value.toFixed(4) : 'None';
          output += `  ${index + 1}. Path: ${pathStr} | Value: ${valueStr} | Depth: ${node.depth}\n`;
        });
      }
    }
    output += '\n';
  }
  
  // Policy vs Value comparison
  if (debugInfo.policy_value_comparison) {
    output += '=== POLICY vs VALUE COMPARISON ===\n';
    output += `Policy Top Move: ${debugInfo.policy_value_comparison.policy_top_move}\n`;
    output += `Tree Best Move: ${debugInfo.policy_value_comparison.tree_best_move}\n`;
    output += `Moves Match: ${debugInfo.policy_value_comparison.moves_match ? 'YES' : 'NO'}\n`;
    output += `Policy Top Probability: ${(debugInfo.policy_value_comparison.policy_top_prob * 100).toFixed(2)}%\n`;
    if (!debugInfo.policy_value_comparison.moves_match) {
      output += '⚠️  WARNING: Policy and value networks disagree!\n';
    }
    output += '\n';
  }
  
  debugContent.textContent = output;
}

function saveStateForUndo() {
  // Save current state for undo functionality
  const stateCopy = {
    trmph: state.trmph,
    board: JSON.parse(JSON.stringify(state.board)),
    player: state.player,
    legal_moves: [...state.legal_moves],
    winner: state.winner,
    last_move: state.last_move ? [...state.last_move] : null,
    last_move_player: state.last_move_player
  };
  state.move_history.push(stateCopy);
  
  // Keep only last 10 moves in history
  if (state.move_history.length > 10) {
    state.move_history.shift();
  }
} 

// --- Debug utilities ---
function debugBoardState(board, legalMoves, lastMove, winner, lastMovePlayer) {
  console.log('=== Board Debug Info ===');
  console.log('Board dimensions:', board.length, 'x', board[0]?.length);
  console.log('Legal moves count:', legalMoves.length);
  console.log('Last move:', lastMove);
  console.log('Last move player:', lastMovePlayer);
  console.log('Winner:', winner);
  
  // Log a sample of board values
  console.log('Sample board values:');
  for (let row = 0; row < Math.min(3, board.length); row++) {
    for (let col = 0; col < Math.min(3, board[row]?.length || 0); col++) {
      const cell = board[row]?.[col] || 'e';
      console.log(`  [${row},${col}]: '${cell}' (type: ${typeof cell})`);
    }
  }
  
  // Count pieces
  let blueCount = 0, redCount = 0, emptyCount = 0;
  for (let row = 0; row < board.length; row++) {
    for (let col = 0; col < board[row]?.length || 0; col++) {
      const cell = board[row]?.[col] || 'e';
      if (cell === 'b') blueCount++;
      else if (cell === 'r') redCount++;
      else emptyCount++;
    }
  }
  console.log('Piece counts - Blue:', blueCount, 'Red:', redCount, 'Empty:', emptyCount);
  console.log('========================');
} 

// --- Model Browser Functionality ---

// Model browser state
let modelBrowserState = {
  modalOpen: false,
  selectedModel: null,
  currentPlayer: null, // 'blue' or 'red'
  recentModels: [],
  searchResults: [],
  directoryModels: []
};

// API calls for model browser
async function fetchRecentModels() {
  try {
    const response = await fetch('/api/model-browser/recent');
    if (!response.ok) throw new Error('Failed to fetch recent models');
    const data = await response.json();
    return data.recent_models || [];
  } catch (error) {
    console.error('Error fetching recent models:', error);
    return [];
  }
}

async function fetchModelDirectories() {
  try {
    const response = await fetch('/api/model-browser/directories');
    if (!response.ok) throw new Error('Failed to fetch directories');
    const data = await response.json();
    return data.directories || [];
  } catch (error) {
    console.error('Error fetching directories:', error);
    return [];
  }
}

async function fetchModelsInDirectory(directory) {
  try {
    const response = await fetch(`/api/model-browser/directory/${encodeURIComponent(directory)}`);
    if (!response.ok) throw new Error('Failed to fetch models in directory');
    const data = await response.json();
    return data.models || [];
  } catch (error) {
    console.error('Error fetching models in directory:', error);
    return [];
  }
}

async function searchModels(query) {
  try {
    const response = await fetch(`/api/model-browser/search?q=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error('Failed to search models');
    const data = await response.json();
    return data.models || [];
  } catch (error) {
    console.error('Error searching models:', error);
    return [];
  }
}

async function selectModel(modelPath, modelId) {
  try {
    const response = await fetch('/api/model-browser/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: modelPath, model_id: modelId })
    });
    if (!response.ok) throw new Error('Failed to select model');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error selecting model:', error);
    return { success: false, error: error.message };
  }
}

// Modal management
function openModelBrowser(player) {
  modelBrowserState.modalOpen = true;
  modelBrowserState.currentPlayer = player;
  modelBrowserState.selectedModel = null;
  
  document.getElementById('model-browser-modal').style.display = 'block';
  document.getElementById('select-model-btn').disabled = true;
  
  // Load initial data
  loadRecentModels();
  loadDirectories();
}

function closeModelBrowser() {
  modelBrowserState.modalOpen = false;
  modelBrowserState.selectedModel = null;
  modelBrowserState.currentPlayer = null;
  
  document.getElementById('model-browser-modal').style.display = 'none';
  document.getElementById('search-results').innerHTML = '';
  document.getElementById('directory-models').innerHTML = '';
  document.getElementById('model-search').value = '';
}

// Load and display recent models
async function loadRecentModels() {
  const recentModels = await fetchRecentModels();
  modelBrowserState.recentModels = recentModels;
  
  const container = document.getElementById('recent-models-list');
  if (recentModels.length === 0) {
    container.innerHTML = '<div class="empty-list">No recent models</div>';
  } else {
    container.innerHTML = recentModels.map(model => createModelItem(model)).join('');
  }
}

// Load and display directories
async function loadDirectories() {
  const directories = await fetchModelDirectories();
  const select = document.getElementById('directory-select');
  
  // Clear existing options except the first one
  select.innerHTML = '<option value="">Select directory...</option>';
  
  directories.forEach(dir => {
    const option = document.createElement('option');
    option.value = dir;
    option.textContent = dir;
    select.appendChild(option);
  });
}

// Load models in selected directory
async function loadDirectoryModels(directory) {
  const models = await fetchModelsInDirectory(directory);
  modelBrowserState.directoryModels = models;
  
  const container = document.getElementById('directory-models');
  if (models.length === 0) {
    container.innerHTML = '<div class="empty-list">No models found in this directory</div>';
  } else {
    container.innerHTML = models.map(model => createModelItem(model)).join('');
  }
}

// Search models
async function performSearch(query) {
  if (!query.trim()) {
    document.getElementById('search-results').innerHTML = '';
    return;
  }
  
  const models = await searchModels(query);
  modelBrowserState.searchResults = models;
  
  const container = document.getElementById('search-results');
  if (models.length === 0) {
    container.innerHTML = '<div class="empty-list">No models found matching your search</div>';
  } else {
    container.innerHTML = models.map(model => createModelItem(model)).join('');
  }
}

// Create model item HTML
function createModelItem(model) {
  const epochInfo = model.epoch && model.mini ? 
    `<span class="model-item-epoch">E${model.epoch} M${model.mini}</span>` : '';
  
  return `
    <div class="model-item" data-model-path="${model.relative_path}">
      <div class="model-item-header">
        <div class="model-item-name">${model.filename}${epochInfo}</div>
        <div class="model-item-size">${model.size_mb} MB</div>
      </div>
      <div class="model-item-path">${model.relative_path}</div>
    </div>
  `;
}

// Handle model selection
function selectModelItem(modelPath) {
  modelBrowserState.selectedModel = modelPath;
  document.getElementById('select-model-btn').disabled = false;
  
  // Update visual selection
  document.querySelectorAll('.model-item').forEach(item => {
    item.classList.remove('selected');
  });
  
  const selectedItem = document.querySelector(`[data-model-path="${modelPath}"]`);
  if (selectedItem) {
    selectedItem.classList.add('selected');
  }
}

// Apply selected model
async function applySelectedModel() {
  if (!modelBrowserState.selectedModel || !modelBrowserState.currentPlayer) {
    return;
  }
  
  const modelId = `model_${Date.now()}`; // Generate unique ID
  console.log(`Attempting to select model: ${modelBrowserState.selectedModel} with ID: ${modelId}`);
  
  const result = await selectModel(modelBrowserState.selectedModel, modelId);
  console.log('Model selection result:', result);
  
  if (result.success) {
    // Update the appropriate model dropdown
    const modelName = modelBrowserState.selectedModel.split('/').pop();
    
    if (modelBrowserState.currentPlayer === 'blue') {
      state.blue_model_id = result.model_id;
      updateModelDropdown('blue-model', result.model_id, modelName);
      console.log(`Set blue model to: ${result.model_id}`);
    } else {
      state.red_model_id = result.model_id;
      updateModelDropdown('red-model', result.model_id, modelName);
      console.log(`Set red model to: ${result.model_id}`);
    }
    
    closeModelBrowser();
    updateUI();
  } else {
    console.error('Model selection failed:', result.error);
    alert(`Error selecting model: ${result.error}`);
  }
}

// Update model dropdown
function updateModelDropdown(selectId, modelId, modelName) {
  const select = document.getElementById(selectId);
  
  // Check if option already exists
  let option = select.querySelector(`option[value="${modelId}"]`);
  if (!option) {
    option = document.createElement('option');
    option.value = modelId;
    select.appendChild(option);
  }
  
  option.textContent = modelName;
  select.value = modelId;
}

// Event listeners for model browser
document.addEventListener('DOMContentLoaded', function() {
  // Browse buttons
  document.getElementById('blue-model-browse').addEventListener('click', () => {
    openModelBrowser('blue');
  });
  
  document.getElementById('red-model-browse').addEventListener('click', () => {
    openModelBrowser('red');
  });
  
  // Modal close
  document.querySelector('.close').addEventListener('click', closeModelBrowser);
  document.getElementById('cancel-model-btn').addEventListener('click', closeModelBrowser);
  
  // Click outside modal to close
  document.getElementById('model-browser-modal').addEventListener('click', (e) => {
    if (e.target.id === 'model-browser-modal') {
      closeModelBrowser();
    }
  });
  
  // Model selection
  document.addEventListener('click', (e) => {
    if (e.target.closest('.model-item')) {
      const modelItem = e.target.closest('.model-item');
      const modelPath = modelItem.dataset.modelPath;
      selectModelItem(modelPath);
    }
  });
  
  // Search functionality
  document.getElementById('search-btn').addEventListener('click', () => {
    const query = document.getElementById('model-search').value;
    performSearch(query);
  });
  
  document.getElementById('model-search').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      const query = e.target.value;
      performSearch(query);
    }
  });
  
  // Directory selection
  document.getElementById('directory-select').addEventListener('change', (e) => {
    const directory = e.target.value;
    if (directory) {
      loadDirectoryModels(directory);
    } else {
      document.getElementById('directory-models').innerHTML = '';
    }
  });
  
  // Refresh directories
  document.getElementById('refresh-dirs-btn').addEventListener('click', loadDirectories);
  
  // Select model button
  document.getElementById('select-model-btn').addEventListener('click', applySelectedModel);
}); 