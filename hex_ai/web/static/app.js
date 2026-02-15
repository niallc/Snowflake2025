// --- Color palette inspired by Taerim's nimbus ---
// 
// 🎨 COLOR CUSTOMIZATION GUIDE:
// 
// This color system has three levels:
// 1. BASE_COLORS: Named color constants (MEDIUM_BLUE, DARK_RED, etc.) - change hex values here
// 2. COLORS: Semantic board regions that reference base colors (BLUE_EDGE_BORDER, HEX_GRID_COLOR, etc.) - change which base color they use
// 3. APPLICATION: Where colors are actually used in the code
//
// To customize colors:
// - For base colors: Change the hex values in the BASE_COLORS section
// - For semantic regions: Change which base color they reference in the COLORS section
//
// Examples:
// - To make grid lines darker: Change BASE_COLORS.MEDIUM_BLUE: '#bbeeee' to '#999999'
// - To use a different blue for pieces: Change COLORS.BLUE_PIECE_COLOR: BASE_COLORS.DARK_BLUE to BASE_COLORS.VERY_DARK_BLUE
// - To make empty hexes light blue: Change COLORS.EMPTY_HEX_COLOR: BASE_COLORS.WHITE to BASE_COLORS.LIGHT_BLUE
//
// Key semantic colors you might want to change:
// - HEX_GRID_COLOR: ⭐ Controls the light cyan lines between hexagons
// - EMPTY_HEX_COLOR: ⭐ Controls the fill color of empty hexagons
// - BOARD_BACKGROUND: Background color behind the game board
// - BLUE_EDGE_BORDER: Blue edge borders (top/bottom)
// - RED_EDGE_BORDER: Red edge borders (left/right)
// - BLUE_PIECE_COLOR: Colors of blue pieces
// - RED_PIECE_COLOR: Colors of red pieces
// - BLUE_LAST_MOVE: Colors of blue's last move
// - RED_LAST_MOVE: Colors of red's last move
// - BLUE_WINNING_PIECE: Colors of blue pieces when blue wins
// - RED_WINNING_PIECE: Colors of red pieces when red wins
//
// ===== BASE COLOR PALETTE =====
// Light theme colors
const LIGHT_COLORS = {
  WHITE: '#fff',
  LIGHT_GRAY: '#f8f8fa',
  MEDIUM_GRAY: '#bbb',
  DARK_GRAY: '#222',
  
  // Board colors
  EMPTY_HEX_GRAY: '#f0f0f0',      // ⭐ LIGHT GRAY for empty hexagons
  GRID_WHITE: '#ffffff',           // ⭐ WHITE for grid lines between hexagons
  BOARD_BACKGROUND: '#f8f8fa',     // ⭐ LIGHT GRAY for board background
  
  // Blue palette
  LIGHT_BLUE: '#e7fcfc',
  MEDIUM_BLUE: '#bbeeee',         // ⭐ LIGHT CYAN - used for grid lines
  DARK_BLUE: '#8bd6d6',           // ⭐ MEDIUM CYAN - used for blue pieces
  VERY_DARK_BLUE: '#0099ff',      // ⭐ VIVID BLUE - used for edges and winning pieces
  DARKER_BLUE: '#0066cc',         // ⭐ DARK BLUE - used for last moves
  
  // Red palette
  LIGHT_RED: '#fff4ea',
  MEDIUM_RED: '#ffe1c8',
  DARK_RED: '#ffcea5',            // ⭐ MEDIUM ORANGE - used for red pieces
  VERY_DARK_RED: '#ff6600',       // ⭐ VIVID ORANGE-RED - used for edges and winning pieces
  DARKER_RED: '#cc3300',          // ⭐ DARK RED - used for last moves
};

// Dark theme colors
const DARK_COLORS = {
  WHITE: '#2d2d2d',
  LIGHT_GRAY: '#1a1a1a',
  MEDIUM_GRAY: '#666',
  DARK_GRAY: '#e0e0e0',
  
  // Board colors
  EMPTY_HEX_GRAY: '#3a3a3a',      // ⭐ DARK GRAY for empty hexagons
  GRID_WHITE: '#4a4a4a',          // ⭐ DARK GRAY for grid lines between hexagons
  BOARD_BACKGROUND: '#1a1a1a',     // ⭐ DARK GRAY for board background
  
  // Blue palette (adjusted for dark theme)
  LIGHT_BLUE: '#1a3a4a',
  MEDIUM_BLUE: '#2a5a6a',         // ⭐ DARKER CYAN for grid lines
  DARK_BLUE: '#4a9a9a',           // ⭐ LIGHTER CYAN for blue pieces (more visible on dark)
  VERY_DARK_BLUE: '#4a9aff',      // ⭐ BRIGHTER BLUE for edges and winning pieces
  DARKER_BLUE: '#2a7aff',         // ⭐ BRIGHTER BLUE for last moves
  
  // Red palette (adjusted for dark theme)
  LIGHT_RED: '#4a2a1a',
  MEDIUM_RED: '#6a3a2a',
  DARK_RED: '#ff9a6a',            // ⭐ BRIGHTER ORANGE for red pieces (more visible on dark)
  VERY_DARK_RED: '#ff4a00',       // ⭐ BRIGHTER ORANGE-RED for edges and winning pieces
  DARKER_RED: '#ff2a00',          // ⭐ BRIGHTER RED for last moves
};

// Dynamic color system
function getColors() {
  return state.dark_mode ? DARK_COLORS : LIGHT_COLORS;
}

const COLORS = {
  // ===== SEMANTIC BOARD REGIONS =====
  // These will be dynamically updated based on dark mode
  get BOARD_BACKGROUND() { return getColors().BOARD_BACKGROUND; },
  get EMPTY_HEX_COLOR() { return getColors().EMPTY_HEX_GRAY; },
  get HEX_GRID_COLOR() { return getColors().GRID_WHITE; },
  
  // Edge borders
  get BLUE_EDGE_BORDER() { return getColors().VERY_DARK_BLUE; },
  get RED_EDGE_BORDER() { return getColors().VERY_DARK_RED; },
  
  // Piece colors
  get BLUE_PIECE_COLOR() { return getColors().DARK_BLUE; },
  get RED_PIECE_COLOR() { return getColors().DARK_RED; },
  
  // Last move colors
  get BLUE_LAST_MOVE() { return getColors().DARKER_BLUE; },
  get RED_LAST_MOVE() { return getColors().DARKER_RED; },
  
  // Winning piece colors
  get BLUE_WINNING_PIECE() { return getColors().VERY_DARK_BLUE; },
  get RED_WINNING_PIECE() { return getColors().VERY_DARK_RED; },
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
  blue_model_id: 'best',
  red_model_id: 'best',  // Use current best model for both players by default
  blue_temperature: 1.0,
  red_temperature: 1.0,
  blue_policy_temperature: 0.15,
  red_policy_temperature: 0.15,
  blue_fixed_tree_temperature: 1.0,
  red_fixed_tree_temperature: 1.0,
  // MCTS settings
  blue_num_simulations: 39,
  red_num_simulations: 39,
  blue_exploration_constant: 2.9,
  red_exploration_constant: 2.9,
  // Gumbel settings
  blue_enable_gumbel: true,
  red_enable_gumbel: true,
  blue_gumbel_max_sims: 49999,
  red_gumbel_max_sims: 49999,
  // Move selection method
  blue_move_method: 'mcts',
  red_move_method: 'mcts',
  // Fixed tree search settings (will be set from backend constants)
  blue_search_widths: [],
  red_search_widths: [],
  blue_search_widths_str: '',
  red_search_widths_str: '',
  auto_step_active: false,
  auto_step_timeout: null,
  available_models: [],
  verbose_level: 2, 
  computer_enabled: true, // Whether computer moves are enabled
  move_history: [], // Track move history for undo functionality
  redo_history: [], // Track undone moves for redo functionality
  constants: null, // Will be populated from backend
  dark_mode: false, // Dark mode state
  heatmap_enabled: false,
  heatmap_loading: false,
  heatmap_error: null,
  heatmap_scores: {},
  heatmap_policy_probs: {},
  heatmap_min_score: null,
  heatmap_max_score: null,
  heatmap_opacity: 0.62,
  heatmap_scope: 'policy_top_k',
  heatmap_top_k: 12,
  heatmap_score_type: 'policy_value',
  heatmap_policy_temperature: 1.0
};

let heatmapRequestToken = 0;

// --- User Modification Tracking ---
// Track which settings have been manually modified by the user
// This allows smart auto-updates while preserving user customizations
const userModifiedSettings = {
  blue: {
    temperature: false,
    num_simulations: false,
    gumbel_enabled: false
  },
  red: {
    temperature: false,
    num_simulations: false,
    gumbel_enabled: false
  }
};

// Smart defaults for different modes
const SMART_DEFAULTS = {
  gumbel: {
    num_simulations: 38,
    temperature: 1.0
  },
  mcts: {
    num_simulations: 2048,
    temperature: 0.25
  },
  policy: {
    temperature: 0.15
  }
};

// --- Game Constants (will be populated from backend) ---
let GAME_CONSTANTS = {
  BOARD_SIZE: 13,
  PIECE_VALUES: {
    EMPTY: 'e',
    BLUE: 'b', 
    RED: 'r'
  }
};

const HEX_RADIUS = 22; // px, radius of each hex (increased from 16 for 1.4x larger board)

// --- Utility: Get per-player settings ---
function getCurrentPlayerSettings() {
  if (state.player === 'blue') {
    const move_method = state.blue_move_method;
    return {
      model_id: state.blue_model_id,
      temperature: getPlayerTemperature('blue', move_method),
      num_simulations: state.blue_num_simulations,
      exploration_constant: state.blue_exploration_constant,
      enable_gumbel: state.blue_enable_gumbel,
      gumbel_max_sims: state.blue_gumbel_max_sims,
      move_method
    };
  } else {
    const move_method = state.red_move_method;
    return {
      model_id: state.red_model_id,
      temperature: getPlayerTemperature('red', move_method),
      num_simulations: state.red_num_simulations,
      exploration_constant: state.red_exploration_constant,
      enable_gumbel: state.red_enable_gumbel,
      gumbel_max_sims: state.red_gumbel_max_sims,
      move_method
    };
  }
}

function getPlayerTemperature(player, moveMethod) {
  if (moveMethod === 'policy') {
    return state[`${player}_policy_temperature`];
  }
  if (moveMethod === 'fixed_tree') {
    return state[`${player}_fixed_tree_temperature`];
  }
  return state[`${player}_temperature`];
}

// --- Smart Settings Management ---
// Mark a setting as user-modified
function markSettingAsModified(player, setting) {
  if (userModifiedSettings[player] && userModifiedSettings[player].hasOwnProperty(setting)) {
    userModifiedSettings[player][setting] = true;
    updateSettingVisualState(player, setting, true);
  }
}

// Reset user modification tracking (useful for testing or reset functionality)
function resetUserModifications(player = null) {
  if (player) {
    // Reset specific player
    Object.keys(userModifiedSettings[player]).forEach(setting => {
      userModifiedSettings[player][setting] = false;
      updateSettingVisualState(player, setting, false);
    });
  } else {
    // Reset all players
    Object.keys(userModifiedSettings).forEach(p => {
      Object.keys(userModifiedSettings[p]).forEach(setting => {
        userModifiedSettings[p][setting] = false;
        updateSettingVisualState(p, setting, false);
      });
    });
  }
}

// Update visual state to show if setting is auto-managed or user-modified
function updateSettingVisualState(player, setting, isModified) {
  const elementId = `${player}-${setting.replace('_', '-')}`;
  const element = document.getElementById(elementId);
  if (element) {
    if (isModified) {
      element.classList.add('user-modified');
      element.title = 'This setting has been manually modified by you';
    } else {
      element.classList.remove('user-modified');
      element.title = 'This setting is auto-managed based on Gumbel mode';
    }
  }
}

// Fixed Tree Search constants (loaded from backend)
let FIXED_TREE_MAX_PRODUCT;
let FIXED_TREE_DEFAULT_WIDTH;
let FIXED_TREE_DEFAULT_TEMPERATURE;

// Validate search widths input
function validateSearchWidths(widthsStr) {
  try {
    const widths = widthsStr.split(',').map(s => parseInt(s.trim()));
    
    // Check all are valid integers 1-169
    if (widths.some(w => isNaN(w) || w < 1 || w > FIXED_TREE_DEFAULT_WIDTH)) {
      return { valid: false, error: `All widths must be integers between 1 and ${FIXED_TREE_DEFAULT_WIDTH}` };
    }
    
    // Check product constraint
    const product = widths.reduce((a, b) => a * b, 1);
    if (product > FIXED_TREE_MAX_PRODUCT) {
      return { valid: false, error: `Product ${product} exceeds limit of ${FIXED_TREE_MAX_PRODUCT}` };
    }
    
    return { valid: true, widths };
  } catch (e) {
    return { valid: false, error: "Invalid format. Use comma-separated integers like '169' or '20,10,5'" };
  }
}

// Smart update when Gumbel is toggled
function onGumbelToggle(player, enabled) {
  const mode = enabled ? 'gumbel' : 'mcts';
  const defaults = SMART_DEFAULTS[mode];
  
  // Only update settings that haven't been manually modified
  if (!userModifiedSettings[player].num_simulations) {
    state[`${player}_num_simulations`] = defaults.num_simulations;
    updateUIElement(`${player}-num-simulations`, defaults.num_simulations);
  }
  
  if (!userModifiedSettings[player].temperature) {
    state[`${player}_temperature`] = defaults.temperature;
    updateUIElement(`${player}-temperature`, defaults.temperature);
  }
  
  // Always update the Gumbel setting itself
  state[`${player}_enable_gumbel`] = enabled;
  updateUIElement(`${player}-enable-gumbel`, enabled);
  
  console.log(`Smart update for ${player}: ${mode} mode - sims: ${state[`${player}_num_simulations`]}, temp: ${state[`${player}_temperature`]}`);
}

// Smart update when Policy play is toggled
function onPolicyToggle(player, enabled) {
  if (enabled) {
    // Policy mode: set temperature to 0.15, disable Gumbel
    if (!userModifiedSettings[player].temperature) {
      state[`${player}_temperature`] = SMART_DEFAULTS.policy.temperature;
      updateUIElement(`${player}-temperature`, SMART_DEFAULTS.policy.temperature);
    }
    
    // Disable Gumbel when using policy play
    if (!userModifiedSettings[player].gumbel_enabled) {
      state[`${player}_enable_gumbel`] = false;
      updateUIElement(`${player}-enable-gumbel`, false);
    }
    
    console.log(`Smart update for ${player}: policy mode - temp: ${state[`${player}_temperature`]}, gumbel: ${state[`${player}_enable_gumbel`]}`);
  } else {
    // Return to previous mode (Gumbel or MCTS)
    const gumbelEnabled = state[`${player}_enable_gumbel`];
    const mode = gumbelEnabled ? 'gumbel' : 'mcts';
    const defaults = SMART_DEFAULTS[mode];
    
    if (!userModifiedSettings[player].temperature) {
      state[`${player}_temperature`] = defaults.temperature;
      updateUIElement(`${player}-temperature`, defaults.temperature);
    }
    
    console.log(`Smart update for ${player}: returning to ${mode} mode - temp: ${state[`${player}_temperature`]}`);
  }
  
}

// Smart update when move method is changed
function onMoveMethodChange(player, method) {
  // Show/hide sections based on method
  const mctsSection = document.querySelector(`.${player}-mcts-section`);
  const policySection = document.querySelector(`.${player}-policy-section`);
  const fixedTreeSection = document.querySelector(`.${player}-fixed-tree-section`);
  
  if (mctsSection) mctsSection.style.display = method === 'mcts' ? 'block' : 'none';
  if (policySection) policySection.style.display = method === 'policy' ? 'block' : 'none';
  if (fixedTreeSection) fixedTreeSection.style.display = method === 'fixed_tree' ? 'block' : 'none';
  
  
  console.log(`Move method changed for ${player}: ${method}`);
}

// Helper to update UI element value
function updateUIElement(elementId, value) {
  const element = document.getElementById(elementId);
  if (element) {
    if (element.type === 'checkbox') {
      element.checked = value;
    } else {
      element.value = value;
    }
  }
}

// Initialize visual states for all smart settings
function initializeSmartSettingsVisualStates() {
  // Initialize visual states for all tracked settings
  Object.keys(userModifiedSettings).forEach(player => {
    Object.keys(userModifiedSettings[player]).forEach(setting => {
      updateSettingVisualState(player, setting, userModifiedSettings[player][setting]);
    });
  });
}

// --- Dark Mode Functions ---
function toggleDarkMode() {
  state.dark_mode = !state.dark_mode;
  
  // Update the data-theme attribute on the document
  if (state.dark_mode) {
    document.documentElement.setAttribute('data-theme', 'dark');
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
  
  // Update the toggle button text and icon
  const toggleBtn = document.getElementById('dark-mode-toggle');
  if (toggleBtn) {
    if (state.dark_mode) {
      toggleBtn.textContent = '☀️ Light';
      toggleBtn.title = 'Switch to light mode';
    } else {
      toggleBtn.textContent = '🌙 Dark';
      toggleBtn.title = 'Switch to dark mode';
    }
  }
  
  // Redraw the board with new colors
  updateUI();
  
  // Save preference to localStorage
  localStorage.setItem('hex_ai_dark_mode', state.dark_mode.toString());
}

function initializeDarkMode() {
  // Check localStorage for saved preference
  const savedDarkMode = localStorage.getItem('hex_ai_dark_mode');
  if (savedDarkMode !== null) {
    state.dark_mode = savedDarkMode === 'true';
  } else {
    state.dark_mode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  
  // Apply the theme
  if (state.dark_mode) {
    document.documentElement.setAttribute('data-theme', 'dark');
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
  
  // Update the toggle button
  const toggleBtn = document.getElementById('dark-mode-toggle');
  if (toggleBtn) {
    if (state.dark_mode) {
      toggleBtn.textContent = '☀️ Light';
      toggleBtn.title = 'Switch to light mode';
    } else {
      toggleBtn.textContent = '🌙 Dark';
      toggleBtn.title = 'Switch to dark mode';
    }
  }
}

// --- Utility: Convert (row, col) to TRMPH move ---
function rowcolToTrmph(row, col) {
  return String.fromCharCode(97 + col) + (row + 1);
}

// --- Custom Tooltip Functions ---
let tooltip = null;

function showTooltip(event, text) {
  // Remove existing tooltip
  hideTooltip();
  
  // Create tooltip element
  tooltip = document.createElement('div');
  tooltip.textContent = text;
  tooltip.style.cssText = `
    position: fixed;
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-family: monospace;
    pointer-events: none;
    z-index: 1000;
    left: ${event.clientX + 10}px;
    top: ${event.clientY - 30}px;
  `;
  
  document.body.appendChild(tooltip);
}

function hideTooltip() {
  if (tooltip) {
    tooltip.remove();
    tooltip = null;
  }
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

async function fetchState(trmph, model_id = 'best', temperature = 1.0) {
  const resp = await fetch('/api/state', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trmph, model_id, temperature, verbose: state.verbose_level }),
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

async function fetchMoveHeatmap(
  trmph,
  model_id = 'best',
  scoreType = 'policy_value',
  selectionMode = 'policy_top_k',
  topK = 12,
  policyTemperature = 1.0
) {
  const resp = await fetch('/api/move_heatmap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      trmph,
      model_id,
      score_type: scoreType,
      selection_mode: selectionMode,
      top_k: topK,
      policy_temperature: policyTemperature
    }),
  });
  if (!resp.ok) {
    let message = `API error (${resp.status})`;
    try {
      const err = await resp.json();
      if (err && err.error) {
        message = err.error;
      }
    } catch (_ignore) {
      // Keep default message when body is not JSON.
    }
    throw new Error(message);
  }
  return await resp.json();
}

async function applyHumanMove(trmph, move, model_id = 'best', temperature = 1.0) {
  const resp = await fetch('/api/apply_move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trmph, move, model_id, temperature, verbose: state.verbose_level }),
  });
  if (!resp.ok) throw new Error('API error');
  return await resp.json();
}

async function makeComputerMove(trmph, model_id, temperature, verbose,
                               num_simulations, exploration_constant,
                               enable_gumbel, gumbel_max_sims, 
                               move_method, search_widths) {
  console.log(`makeComputerMove called with model_id: ${model_id}, move_method: ${move_method}`);
  
  if (move_method === 'policy') {
    // Use policy endpoint for fast policy sampling
    const resp = await fetch('/api/policy_move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        trmph, 
        model_id, 
        temperature, 
        verbose
      }),
    });
    if (!resp.ok) throw new Error('API error');
    return await resp.json();
  } else if (move_method === 'fixed_tree') {
    // Use fixed tree search endpoint
    const resp = await fetch('/api/fixed_tree_move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        trmph, 
        model_id, 
        search_widths,
        temperature, 
        verbose
      }),
    });
    if (!resp.ok) throw new Error('API error');
    return await resp.json();
  } else {
    // Use MCTS endpoint (default)
    const resp = await fetch('/api/mcts_move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        trmph, 
        model_id, 
        num_simulations, 
        exploration_constant, 
        temperature, 
        verbose,
        enable_gumbel,
        gumbel_max_sims
      }),
    });
    if (!resp.ok) throw new Error('API error');
    return await resp.json();
  }
}

function clearHeatmapData() {
  state.heatmap_scores = {};
  state.heatmap_policy_probs = {};
  state.heatmap_min_score = null;
  state.heatmap_max_score = null;
  state.heatmap_error = null;
}

function getHeatmapScoreForMove(moveTrmph) {
  if (!state.heatmap_enabled) {
    return null;
  }
  const value = state.heatmap_scores[moveTrmph];
  return Number.isFinite(value) ? value : null;
}

function getHeatmapFillColor(score) {
  if (!Number.isFinite(score)) {
    return COLORS.EMPTY_HEX_COLOR;
  }
  if (window.HexHeatmap && typeof window.HexHeatmap.scoreToColor === 'function') {
    return window.HexHeatmap.scoreToColor(score, {
      alpha: state.heatmap_opacity,
      darkMode: state.dark_mode,
      fallback: COLORS.EMPTY_HEX_COLOR
    });
  }
  return COLORS.EMPTY_HEX_COLOR;
}

function updateHeatmapControls() {
  const enabled = document.getElementById('heatmap-enabled');
  const status = document.getElementById('heatmap-status');
  const opacity = document.getElementById('heatmap-opacity');
  const opacityValue = document.getElementById('heatmap-opacity-value');
  const scope = document.getElementById('heatmap-scope');
  const topK = document.getElementById('heatmap-top-k');
  const refresh = document.getElementById('heatmap-refresh');

  if (enabled) {
    enabled.checked = state.heatmap_enabled;
  }
  if (opacity) {
    opacity.value = state.heatmap_opacity.toFixed(2);
  }
  if (opacityValue) {
    opacityValue.textContent = `${Math.round(state.heatmap_opacity * 100)}%`;
  }
  if (scope) {
    scope.value = state.heatmap_scope;
  }
  if (topK) {
    topK.value = state.heatmap_top_k;
    topK.disabled = state.heatmap_scope === 'all_legal';
  }
  if (refresh) {
    refresh.disabled = !state.heatmap_enabled || state.heatmap_loading;
    refresh.textContent = state.heatmap_loading ? 'Analyzing...' : 'Analyze';
  }
  if (status) {
    if (!state.heatmap_enabled) {
      status.textContent = 'Off';
    } else if (state.heatmap_loading) {
      status.textContent = 'Loading...';
    } else if (state.heatmap_error) {
      status.textContent = `Error: ${state.heatmap_error}`;
    } else if (
      Number.isFinite(state.heatmap_min_score) &&
      Number.isFinite(state.heatmap_max_score)
    ) {
      const selected = Object.keys(state.heatmap_scores).length;
      const scopeLabel = state.heatmap_scope === 'all_legal'
        ? 'all legal'
        : `top ${state.heatmap_top_k}`;
      if (window.HexHeatmap && typeof window.HexHeatmap.formatPercent === 'function') {
        status.textContent = `${scopeLabel}: ${selected} moves, ${window.HexHeatmap.formatPercent(state.heatmap_min_score)} to ${window.HexHeatmap.formatPercent(state.heatmap_max_score)}`;
      } else {
        status.textContent = `${scopeLabel}: ${selected} moves, ${(state.heatmap_min_score * 100).toFixed(1)}% to ${(state.heatmap_max_score * 100).toFixed(1)}%`;
      }
    } else {
      status.textContent = 'No legal moves';
    }
  }
}

async function refreshMoveHeatmap() {
  if (!state.heatmap_enabled || state.winner) {
    if (state.heatmap_loading || Object.keys(state.heatmap_scores).length > 0 || state.heatmap_error) {
      state.heatmap_loading = false;
      clearHeatmapData();
      updateUI();
    } else {
      updateHeatmapControls();
    }
    return;
  }

  const requestToken = ++heatmapRequestToken;
  state.heatmap_loading = true;
  state.heatmap_error = null;
  updateHeatmapControls();

  try {
    const { model_id } = getCurrentPlayerSettings();
    const result = await fetchMoveHeatmap(
      state.trmph,
      model_id,
      state.heatmap_score_type,
      state.heatmap_scope,
      state.heatmap_top_k,
      state.heatmap_policy_temperature
    );
    if (requestToken !== heatmapRequestToken) {
      return;
    }
    state.heatmap_scores = result.scores || {};
    state.heatmap_policy_probs = result.policy_probs || {};
    state.heatmap_min_score = Number.isFinite(result.min_score) ? result.min_score : null;
    state.heatmap_max_score = Number.isFinite(result.max_score) ? result.max_score : null;
    state.heatmap_error = null;
  } catch (err) {
    if (requestToken !== heatmapRequestToken) {
      return;
    }
    clearHeatmapData();
    state.heatmap_error = err.message || 'Failed to load heatmap';
  } finally {
    if (requestToken === heatmapRequestToken) {
      state.heatmap_loading = false;
      updateUI();
    }
  }
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
  // Calculate SVG dimensions with balanced padding
  // Account for full board size including edge borders (which extend beyond hex centers)
  const boardWidth = w * (GAME_CONSTANTS.BOARD_SIZE - 1 + 0.5) + 2 * HEX_RADIUS;
  const boardHeight = h * (GAME_CONSTANTS.BOARD_SIZE - 1) + 2 * HEX_RADIUS;
  const edgeBorderWidth = 17; // 🎨 EDGE BORDER WIDTH - Account for thick red edge borders
  const padding = HEX_RADIUS * 0.5; // 🎨 BALANCED PADDING - Equal padding on all sides
  
  // 🎨 DIAMOND SHAPE COMPENSATION - Account for hex board's diagonal offset
  // The bottom edge extends further right than the top edge due to the diamond shape
  const diagonalOffset = w * (GAME_CONSTANTS.BOARD_SIZE - 1) * 0.45; // Slightly reduced from 0.5 to balance left/right padding
  
  const svgWidth = boardWidth + 2 * padding + edgeBorderWidth + diagonalOffset;
  const svgHeight = boardHeight + 2 * padding + edgeBorderWidth;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', svgWidth);
  svg.setAttribute('height', svgHeight);
  svg.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);
  svg.style.background = COLORS.BOARD_BACKGROUND; /* 🎨 CENTRAL PLAY AREA BACKGROUND - Light gray area containing the actual game board */

  // --- Draw player edge indicators ---
  // Blue: top and bottom (across the topmost and bottommost hexes)
  // Increase stroke width to extend the strips toward the board
  svg.appendChild(makeEdgeLine(
    hexCenter(0, 0).x, hexCenter(0, 0).y - HEX_RADIUS,
    hexCenter(0, GAME_CONSTANTS.BOARD_SIZE - 1).x, hexCenter(0, GAME_CONSTANTS.BOARD_SIZE - 1).y - HEX_RADIUS,
    COLORS.BLUE_EDGE_BORDER,
    18  // Increased stroke width from 10 to 20 to extend down
  ));
  svg.appendChild(makeEdgeLine(
    hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, 0).x, hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, 0).y + HEX_RADIUS,
    hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, GAME_CONSTANTS.BOARD_SIZE - 1).x, hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, GAME_CONSTANTS.BOARD_SIZE - 1).y + HEX_RADIUS,
    COLORS.BLUE_EDGE_BORDER,
    18  // Increased stroke width from 10 to 20 to extend up
  ));
  
  // Red edges: pass through midpoints of the true outer edges
  function edgeMidpoint(vA, vB) {
    return { x: (vA.x + vB.x) / 2, y: (vA.y + vB.y) / 2 };
  }

  const tl = hexVertices(hexCenter(0, 0).x, hexCenter(0, 0).y, HEX_RADIUS);
  const bl = hexVertices(hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, 0).x,
                        hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, 0).y, HEX_RADIUS);

  // left side uses edge between vertices 2 and 3
  const leftTopMid  = edgeMidpoint(tl[2], tl[3]);
  const leftBotMid  = edgeMidpoint(bl[2], bl[3]);
  svg.appendChild(makeEdgeLine(leftTopMid.x, leftTopMid.y, leftBotMid.x, leftBotMid.y, COLORS.RED_EDGE_BORDER, 22));

  const tr = hexVertices(hexCenter(0, GAME_CONSTANTS.BOARD_SIZE - 1).x,
                        hexCenter(0, GAME_CONSTANTS.BOARD_SIZE - 1).y, HEX_RADIUS);
  const br = hexVertices(hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, GAME_CONSTANTS.BOARD_SIZE - 1).x,
                        hexCenter(GAME_CONSTANTS.BOARD_SIZE - 1, GAME_CONSTANTS.BOARD_SIZE - 1).y, HEX_RADIUS);

  // right side uses edge between vertices 0 and 5
  const rightTopMid = edgeMidpoint(tr[0], tr[5]);
  const rightBotMid = edgeMidpoint(br[0], br[5]);
  svg.appendChild(makeEdgeLine(rightTopMid.x, rightTopMid.y, rightBotMid.x, rightBotMid.y, COLORS.RED_EDGE_BORDER, 22));


  // Draw hexes
  for (let row = 0; row < GAME_CONSTANTS.BOARD_SIZE; row++) {
    for (let col = 0; col < GAME_CONSTANTS.BOARD_SIZE; col++) {
      const { x, y } = hexCenter(row, col);
      const cell = board[row]?.[col] || GAME_CONSTANTS.PIECE_VALUES.EMPTY;
      const moveTrmph = rowcolToTrmph(row, col);
      // ⭐ EMPTY HEX COLOR - uses COLORS.EMPTY_HEX_COLOR for empty hexagons
      let fill = COLORS.EMPTY_HEX_COLOR;
      const heatmapScore = getHeatmapScoreForMove(moveTrmph);
      if (cell === GAME_CONSTANTS.PIECE_VALUES.EMPTY && Number.isFinite(heatmapScore)) {
        fill = getHeatmapFillColor(heatmapScore);
      }
      if (cell === GAME_CONSTANTS.PIECE_VALUES.BLUE) fill = COLORS.BLUE_PIECE_COLOR;
      if (cell === GAME_CONSTANTS.PIECE_VALUES.RED) fill = COLORS.RED_PIECE_COLOR;
      
      // Highlight last move with darker color based on player
      if (lastMove && lastMove[0] === row && lastMove[1] === col) {
        if (lastMovePlayer === 'blue') {
          fill = COLORS.BLUE_LAST_MOVE;
        } else if (lastMovePlayer === 'red') {
          fill = COLORS.RED_LAST_MOVE;
        }
      }
      
      if (winner === 'blue' && cell === GAME_CONSTANTS.PIECE_VALUES.BLUE) fill = COLORS.BLUE_WINNING_PIECE;
      if (winner === 'red' && cell === GAME_CONSTANTS.PIECE_VALUES.RED) fill = COLORS.RED_WINNING_PIECE;
      const isLegal = legalMoves.includes(moveTrmph);
      const hex = makeHex(x, y, HEX_RADIUS, fill, isLegal);
      hex.setAttribute('data-row', row);
      hex.setAttribute('data-col', col);
      if (isLegal && !state.auto_step_active) {
        hex.classList.add('clickable');
        hex.addEventListener('click', onCellClick);
      }
      svg.appendChild(hex);
      
      // Add TRMPH label in center of hex
      const trmphLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      trmphLabel.setAttribute('x', x);
      trmphLabel.setAttribute('y', y + 4); // Slight vertical adjustment for centering
      trmphLabel.setAttribute('text-anchor', 'middle');
      trmphLabel.setAttribute('dominant-baseline', 'middle');
      trmphLabel.setAttribute('font-size', '10px');
      trmphLabel.setAttribute('font-family', 'monospace');
      trmphLabel.setAttribute('pointer-events', 'none'); // Make text non-interactive
      
      // Set label color based on piece type and theme
      let labelColor;
      if (state.dark_mode) {
        // Dark mode colors
        labelColor = '#666'; // Default dark grey for empty hexes
        if (cell === GAME_CONSTANTS.PIECE_VALUES.BLUE) {
          if (lastMove && lastMove[0] === row && lastMove[1] === col) {
            labelColor = '#7ab3ff'; // Bright blue for last move
          } else {
            labelColor = '#4a9aff'; // Bright blue for blue pieces
          }
        } else if (cell === GAME_CONSTANTS.PIECE_VALUES.RED) {
          if (lastMove && lastMove[0] === row && lastMove[1] === col) {
            labelColor = '#ff7a4a'; // Bright red for last move
          } else {
            labelColor = '#ff4a00'; // Bright red for red pieces
          }
        }
      } else {
        // Light mode colors
        labelColor = '#e0e0e0'; // Default light grey for empty hexes
        if (cell === GAME_CONSTANTS.PIECE_VALUES.BLUE) {
          if (lastMove && lastMove[0] === row && lastMove[1] === col) {
            labelColor = '#004499'; // Much darker blue for last move
          } else {
            labelColor = '#00eeee'; // trmph label color, blue
          }
        } else if (cell === GAME_CONSTANTS.PIECE_VALUES.RED) {
          if (lastMove && lastMove[0] === row && lastMove[1] === col) {
            labelColor = '#992200'; // Much darker red for last move
          } else {
            labelColor = '#ddcc00'; // trmph label color, red
          }
        }
      }
      
      trmphLabel.setAttribute('fill', labelColor);
      trmphLabel.textContent = moveTrmph;
      svg.appendChild(trmphLabel);
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
  // Grid lines between hexagons - use consistent stroke color and width for all hexagons
  hex.setAttribute('stroke', COLORS.HEX_GRID_COLOR);
  hex.setAttribute('stroke-width', 2);
  if (highlight) hex.style.cursor = 'pointer';
  
  // Add tooltip functionality
  hex.addEventListener('mouseenter', function(e) {
    // console.log('Mouse enter event triggered!');
    const row = parseInt(this.getAttribute('data-row'));
    const col = parseInt(this.getAttribute('data-col'));
    // console.log('Row:', row, 'Col:', col, 'Row type:', typeof row, 'Col type:', typeof col);
    if (!isNaN(row) && !isNaN(col)) {
      const trmph = rowcolToTrmph(row, col);
      const score = getHeatmapScoreForMove(trmph);
      let tooltipText = trmph;
      if (Number.isFinite(score)) {
        const percent = window.HexHeatmap && typeof window.HexHeatmap.formatPercent === 'function'
          ? window.HexHeatmap.formatPercent(score)
          : `${(score * 100).toFixed(1)}%`;
        tooltipText = `${trmph} (${percent} win)`;
      }
      showTooltip(e, tooltipText);
    } else {
      console.log('Invalid row/col values - row:', row, 'col:', col);
    }
  });
  
  hex.addEventListener('mouseleave', function(e) {
    hideTooltip();
  });
  
  return hex;
}

function hexVertices(cx, cy, r) {
  const pts = [];
  for (let i = 0; i < 6; i++) {
    const angle = Math.PI / 3 * i + Math.PI / 6; // same orientation as makeHex
    pts.push({ 
      x: cx + r * Math.cos(angle), 
      y: cy + r * Math.sin(angle) 
    });
  }
  return pts;
}

function makeEdgeLine(x1, y1, x2, y2, color, strokeWidth = 10) {
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', x1);
  line.setAttribute('y1', y1);
  line.setAttribute('x2', x2);
  line.setAttribute('y2', y2);
  line.setAttribute('stroke', color);
  line.setAttribute('stroke-width', strokeWidth);
  line.setAttribute('stroke-linecap', 'round');
  line.setAttribute('opacity', 0.25);
  return line;
}

function hexCenter(row, col) {
  // Flat-topped, blue at top/bottom: x = HEX_RADIUS * sqrt(3) * (col + row/2) + HEX_RADIUS
  // y = HEX_RADIUS * 1.5 * row + HEX_RADIUS
  const padding = HEX_RADIUS * 0.5; // 🎨 BALANCED PADDING - Same as SVG padding
  const extraTopPadding = HEX_RADIUS * 0.5; // 🎨 EXTRA TOP PADDING - Additional space at top
  const extraLeftPadding = HEX_RADIUS * 0.5; // 🎨 EXTRA LEFT PADDING - Reduced to balance right side (was 0.3)
  const x = HEX_RADIUS * Math.sqrt(3) * (col + row / 2) + HEX_RADIUS + padding + extraLeftPadding; // centered with extra left padding
  const y = HEX_RADIUS * 1.5 * row + HEX_RADIUS + padding + extraTopPadding; // centered with extra top padding
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
    status.style.color = state.winner === 'blue' ? COLORS.BLUE_EDGE_BORDER : COLORS.RED_EDGE_BORDER;
  } else if (state.auto_step_active) {
    status.textContent = `Auto-stepping: ${state.player[0].toUpperCase() + state.player.slice(1)}'s turn`;
    status.style.color = state.player === 'blue' ? COLORS.BLUE_EDGE_BORDER : COLORS.RED_EDGE_BORDER;
  } else {
    status.textContent = `${state.player[0].toUpperCase() + state.player.slice(1)}'s turn`;
    status.style.color = state.player === 'blue' ? COLORS.BLUE_EDGE_BORDER : COLORS.RED_EDGE_BORDER;
  }
  
  // TRMPH
  document.getElementById('trmph-string').value = state.trmph;
  
  // Update step button - keep it enabled even when game is over
  const stepBtn = document.getElementById('step-btn');
  if (state.winner) {
    stepBtn.textContent = 'Game Over (Step Still Works)';
  } else {
    stepBtn.textContent = 'Step (Computer Move)';
  }
  stepBtn.disabled = false;
  
  // Update debug controls
  const verboseLevel = document.getElementById('verbose-level');
  const computerToggle = document.getElementById('computer-toggle');
  if (verboseLevel) verboseLevel.value = state.verbose_level;
  if (computerToggle) computerToggle.textContent = state.computer_enabled ? 'ON' : 'OFF';

  // Update MCTS controls
  const blueNumSimulations = document.getElementById('blue-num-simulations');
  const blueExplorationConstant = document.getElementById('blue-exploration-constant');
  const blueTemperature = document.getElementById('blue-temperature');
  const redNumSimulations = document.getElementById('red-num-simulations');
  const redExplorationConstant = document.getElementById('red-exploration-constant');
  const redTemperature = document.getElementById('red-temperature');
  
  if (blueNumSimulations) blueNumSimulations.value = state.blue_num_simulations;
  if (blueExplorationConstant) blueExplorationConstant.value = state.blue_exploration_constant;
  if (blueTemperature) blueTemperature.value = state.blue_temperature;
  if (redNumSimulations) redNumSimulations.value = state.red_num_simulations;
  if (redExplorationConstant) redExplorationConstant.value = state.red_exploration_constant;
  if (redTemperature) redTemperature.value = state.red_temperature;

  const bluePolicyTemperature = document.getElementById('blue-policy-temperature');
  const redPolicyTemperature = document.getElementById('red-policy-temperature');
  if (bluePolicyTemperature) bluePolicyTemperature.value = state.blue_policy_temperature;
  if (redPolicyTemperature) redPolicyTemperature.value = state.red_policy_temperature;

  const blueFixedTreeTemperature = document.getElementById('blue-fixed-tree-temperature');
  const redFixedTreeTemperature = document.getElementById('red-fixed-tree-temperature');
  if (blueFixedTreeTemperature) blueFixedTreeTemperature.value = state.blue_fixed_tree_temperature;
  if (redFixedTreeTemperature) redFixedTreeTemperature.value = state.red_fixed_tree_temperature;
  
  // Update Gumbel controls
  const blueEnableGumbel = document.getElementById('blue-enable-gumbel');
  const blueGumbelMaxSims = document.getElementById('blue-gumbel-max-sims');
  const redEnableGumbel = document.getElementById('red-enable-gumbel');
  const redGumbelMaxSims = document.getElementById('red-gumbel-max-sims');
  
  if (blueEnableGumbel) blueEnableGumbel.checked = state.blue_enable_gumbel;
  if (blueGumbelMaxSims) blueGumbelMaxSims.value = state.blue_gumbel_max_sims;
  if (redEnableGumbel) redEnableGumbel.checked = state.red_enable_gumbel;
  if (redGumbelMaxSims) redGumbelMaxSims.value = state.red_gumbel_max_sims;
  
  
  // Show/hide debug output based on verbose level
  const debugOutput = document.getElementById('debug-output');
  if (debugOutput) {
    debugOutput.style.display = state.verbose_level > 0 ? 'block' : 'none';
  }

  updateHeatmapControls();
  
  // Keep all controls active - no need to disable them during auto-step or game over
}

// --- Event Handlers ---
async function onCellClick(e) {
  if (state.auto_step_active) return;
  
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
    void refreshMoveHeatmap();
    
    // Step 2: If game is not over and computer is enabled, get the computer move
    if (!state.winner && state.computer_enabled) {
      // Save state for undo functionality before computer move
      saveStateForUndo();
      
      // Determine which player's settings to use for the computer move
      const computerPlayer = state.player; // Current player after human move
      const computerMoveMethod = computerPlayer === 'blue' ? state.blue_move_method : state.red_move_method;
      const computerSearchWidths = computerPlayer === 'blue' ? state.blue_search_widths : state.red_search_widths;
      const computerTemperature = getPlayerTemperature(computerPlayer, computerMoveMethod);
      let computerModelId, computerNumSimulations, computerExplorationConstant, computerEnableGumbel, computerGumbelMaxSims;
      
      if (computerPlayer === 'blue') {
        computerModelId = state.blue_model_id;
        computerNumSimulations = state.blue_num_simulations;
        computerExplorationConstant = state.blue_exploration_constant;
        computerEnableGumbel = state.blue_enable_gumbel;
        computerGumbelMaxSims = state.blue_gumbel_max_sims;
      } else {
        computerModelId = state.red_model_id;
        computerNumSimulations = state.red_num_simulations;
        computerExplorationConstant = state.red_exploration_constant;
        computerEnableGumbel = state.red_enable_gumbel;
        computerGumbelMaxSims = state.red_gumbel_max_sims;
      }
      
      // Make computer move with verbose output
      const computerResult = await makeComputerMove(
        state.trmph, 
        computerModelId, 
        computerTemperature,
        state.verbose_level,
        computerNumSimulations,
        computerExplorationConstant,
        computerEnableGumbel,
        computerGumbelMaxSims,
        computerMoveMethod,
        computerSearchWidths
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
      } else if (computerResult.mcts_debug_info) {
        displayMCTSDebugInfo(computerResult.mcts_debug_info);
      } else if (computerResult.fixed_tree_debug_info) {
        displayFixedTreeDebugInfo(computerResult.fixed_tree_debug_info);
      }
      
      // Display policy information if available
      if (computerResult.policy_info) {
        displayPolicyInfo(computerResult.policy_info);
      }
      
      // Display detailed exploration if tree_data exists
      if (computerResult.tree_data) {
        displayDetailedExploration({ tree_data: computerResult.tree_data });
      }
        
        updateUI();
        void refreshMoveHeatmap();
      } else {
        alert('Computer move failed: ' + computerResult.error);
      }
    }
  } catch (err) {
    alert('Move failed: ' + err.message);
  }
}


// --- Computer move functionality ---
async function stepComputerMove() {
  if (state.winner) return;
  
  const { model_id, temperature, num_simulations, exploration_constant, enable_gumbel, gumbel_max_sims } = getCurrentPlayerSettings();
  const currentPlayer = state.player; // Store current player before the move
  
  // Get move method and search widths for current player
  const moveMethod = currentPlayer === 'blue' ? state.blue_move_method : state.red_move_method;
  const searchWidths = currentPlayer === 'blue' ? state.blue_search_widths : state.red_search_widths;
  
  // Save state for undo functionality before computer move
  saveStateForUndo();
  
  try {
    const result = await makeComputerMove(
      state.trmph, 
      model_id, 
      temperature, 
      state.verbose_level,
      num_simulations,
      exploration_constant,
      enable_gumbel,
      gumbel_max_sims,
      moveMethod,
      searchWidths
    );
    
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
      } else if (result.mcts_debug_info) {
        displayMCTSDebugInfo(result.mcts_debug_info);
      } else if (result.fixed_tree_debug_info) {
        displayFixedTreeDebugInfo(result.fixed_tree_debug_info);
      }
      
      // Display policy information if available
      if (result.policy_info) {
        displayPolicyInfo(result.policy_info);
      }
      
      // Display detailed exploration if tree_data exists
      if (result.tree_data) {
        displayDetailedExploration({ tree_data: result.tree_data });
      }
      
      updateUI();
      void refreshMoveHeatmap();
      
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
  // Initialize dark mode first
  initializeDarkMode();
  
  // Verify detailed exploration elements exist (quiet check)
  const explorationDiv = document.getElementById('detailed-exploration');
  const explorationContent = document.getElementById('exploration-content');
  if (!explorationDiv || !explorationContent) {
    console.warn('Detailed exploration elements not found in DOM');
  }
  
  // Load game constants from backend
  try {
    const constantsResult = await fetchConstants();
    GAME_CONSTANTS = {
      BOARD_SIZE: constantsResult.BOARD_SIZE,
      PIECE_VALUES: constantsResult.PIECE_VALUES,
      PLAYER_VALUES: constantsResult.PLAYER_VALUES,
      WINNER_VALUES: constantsResult.WINNER_VALUES
    };
    const maxMoves = GAME_CONSTANTS.BOARD_SIZE * GAME_CONSTANTS.BOARD_SIZE;
    state.heatmap_top_k = Math.min(state.heatmap_top_k, maxMoves);
    const heatmapTopKInput = document.getElementById('heatmap-top-k');
    if (heatmapTopKInput) {
      heatmapTopKInput.max = String(maxMoves);
      heatmapTopKInput.value = String(state.heatmap_top_k);
    }
    
    // Load fixed tree constants from backend
    if (constantsResult.FIXED_TREE) {
      FIXED_TREE_MAX_PRODUCT = constantsResult.FIXED_TREE.MAX_PRODUCT;
      FIXED_TREE_DEFAULT_WIDTH = constantsResult.FIXED_TREE.DEFAULT_WIDTH;
      FIXED_TREE_DEFAULT_TEMPERATURE = constantsResult.FIXED_TREE.DEFAULT_TEMPERATURE;
      console.log('Loaded fixed tree constants:', constantsResult.FIXED_TREE);
      
      // Update state with backend constants
      state.blue_search_widths = [FIXED_TREE_DEFAULT_WIDTH];
      state.red_search_widths = [FIXED_TREE_DEFAULT_WIDTH];
      state.blue_search_widths_str = FIXED_TREE_DEFAULT_WIDTH.toString();
      state.red_search_widths_str = FIXED_TREE_DEFAULT_WIDTH.toString();
      state.blue_fixed_tree_temperature = FIXED_TREE_DEFAULT_TEMPERATURE;
      state.red_fixed_tree_temperature = FIXED_TREE_DEFAULT_TEMPERATURE;
      
      // Update UI with backend constants
      const blueSearchWidths = document.getElementById('blue-search-widths');
      const redSearchWidths = document.getElementById('red-search-widths');
      if (blueSearchWidths) blueSearchWidths.value = FIXED_TREE_DEFAULT_WIDTH.toString();
      if (redSearchWidths) redSearchWidths.value = FIXED_TREE_DEFAULT_WIDTH.toString();
      const blueFixedTreeTemperature = document.getElementById('blue-fixed-tree-temperature');
      const redFixedTreeTemperature = document.getElementById('red-fixed-tree-temperature');
      if (blueFixedTreeTemperature) blueFixedTreeTemperature.value = FIXED_TREE_DEFAULT_TEMPERATURE;
      if (redFixedTreeTemperature) redFixedTreeTemperature.value = FIXED_TREE_DEFAULT_TEMPERATURE;
    }
    
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
    
    // Set default selections - check if the default values exist in the dropdown
    if (state.available_models.some(model => model.id === state.blue_model_id)) {
      blueSelect.value = state.blue_model_id;
    } else if (state.available_models.length > 0) {
      // Fallback to first available model
      state.blue_model_id = state.available_models[0].id;
      blueSelect.value = state.blue_model_id;
      console.log(`Blue model not found, using: ${state.blue_model_id}`);
    }
    
    if (state.available_models.some(model => model.id === state.red_model_id)) {
      redSelect.value = state.red_model_id;
    } else if (state.available_models.length > 0) {
      // Fallback to first available model if red model not found
      const redFallbackIndex = Math.min(1, state.available_models.length - 1);
      state.red_model_id = state.available_models[redFallbackIndex].id;
      redSelect.value = state.red_model_id;
      console.log(`Red model not found, using: ${state.red_model_id}`);
    }
    
    console.log(`Model dropdowns initialized. Available models: ${state.available_models.map(m => m.id).join(', ')}`);
    console.log(`Selected models - Blue: ${state.blue_model_id}, Red: ${state.red_model_id}`);
  } catch (err) {
    console.error('Failed to load models:', err);
  }

  // Initialize visual states for smart settings
  initializeSmartSettingsVisualStates();

  // Initial state fetch
  try {
    // Use blue's settings for initial fetch
    const result = await fetchState(
      state.trmph,
      state.blue_model_id,
      getPlayerTemperature('blue', state.blue_move_method)
    );
    state.board = result.board;
    state.player = result.player;
    state.legal_moves = result.legal_moves;
    state.winner = result.winner;
    state.last_move = null;
    state.last_move_player = null; // Initialize last_move_player
    updateUI();
    void refreshMoveHeatmap();
  } catch (err) {
    document.getElementById('status-line').textContent = 'Failed to load board.';
  }

  // Model selection handlers
  const blueModel = document.getElementById('blue-model');
  const redModel = document.getElementById('red-model');
  if (blueModel) blueModel.addEventListener('change', (e) => {
    state.blue_model_id = e.target.value;
    if (state.player === 'blue') {
      void refreshMoveHeatmap();
    }
  });
  if (redModel) redModel.addEventListener('change', (e) => {
    state.red_model_id = e.target.value;
    if (state.player === 'red') {
      void refreshMoveHeatmap();
    }
  });


  // Blue temperature
  document.getElementById('blue-temperature').addEventListener('input', (e) => {
    markSettingAsModified('blue', 'temperature');
    state.blue_temperature = parseFloat(e.target.value);
  });
  // Red temperature
  document.getElementById('red-temperature').addEventListener('input', (e) => {
    markSettingAsModified('red', 'temperature');
    state.red_temperature = parseFloat(e.target.value);
  });

  // Policy temperatures
  const bluePolicyTemperature = document.getElementById('blue-policy-temperature');
  const redPolicyTemperature = document.getElementById('red-policy-temperature');
  if (bluePolicyTemperature) bluePolicyTemperature.addEventListener('input', (e) => {
    state.blue_policy_temperature = parseFloat(e.target.value);
  });
  if (redPolicyTemperature) redPolicyTemperature.addEventListener('input', (e) => {
    state.red_policy_temperature = parseFloat(e.target.value);
  });

  // Fixed tree temperatures
  const blueFixedTreeTemperature = document.getElementById('blue-fixed-tree-temperature');
  const redFixedTreeTemperature = document.getElementById('red-fixed-tree-temperature');
  if (blueFixedTreeTemperature) blueFixedTreeTemperature.addEventListener('input', (e) => {
    state.blue_fixed_tree_temperature = parseFloat(e.target.value);
  });
  if (redFixedTreeTemperature) redFixedTreeTemperature.addEventListener('input', (e) => {
    state.red_fixed_tree_temperature = parseFloat(e.target.value);
  });

  // MCTS controls

  document.getElementById('blue-num-simulations').addEventListener('input', (e) => {
    markSettingAsModified('blue', 'num_simulations');
    state.blue_num_simulations = parseInt(e.target.value);
  });
  document.getElementById('blue-exploration-constant').addEventListener('input', (e) => {
    state.blue_exploration_constant = parseFloat(e.target.value);
  });


  document.getElementById('red-num-simulations').addEventListener('input', (e) => {
    markSettingAsModified('red', 'num_simulations');
    state.red_num_simulations = parseInt(e.target.value);
  });
  document.getElementById('red-exploration-constant').addEventListener('input', (e) => {
    state.red_exploration_constant = parseFloat(e.target.value);
  });

  // Gumbel controls
  document.getElementById('blue-enable-gumbel').addEventListener('change', (e) => {
    markSettingAsModified('blue', 'gumbel_enabled');
    onGumbelToggle('blue', e.target.checked);
  });
  document.getElementById('blue-gumbel-max-sims').addEventListener('input', (e) => {
    state.blue_gumbel_max_sims = parseInt(e.target.value);
  });
  document.getElementById('red-enable-gumbel').addEventListener('change', (e) => {
    markSettingAsModified('red', 'gumbel_enabled');
    onGumbelToggle('red', e.target.checked);
  });
  document.getElementById('red-gumbel-max-sims').addEventListener('input', (e) => {
    state.red_gumbel_max_sims = parseInt(e.target.value);
  });

  // Move method selection controls
  const blueMoveMethod = document.getElementById('blue-move-method');
  const redMoveMethod = document.getElementById('red-move-method');
  if (blueMoveMethod) blueMoveMethod.addEventListener('change', (e) => {
    state.blue_move_method = e.target.value;
    onMoveMethodChange('blue', e.target.value);
  });
  if (redMoveMethod) redMoveMethod.addEventListener('change', (e) => {
    state.red_move_method = e.target.value;
    onMoveMethodChange('red', e.target.value);
  });

  // Fixed tree search controls
  const blueSearchWidths = document.getElementById('blue-search-widths');
  const redSearchWidths = document.getElementById('red-search-widths');
  if (blueSearchWidths) blueSearchWidths.addEventListener('input', (e) => {
    state.blue_search_widths_str = e.target.value;
    const validation = validateSearchWidths(e.target.value);
    if (validation.valid) {
      state.blue_search_widths = validation.widths;
      e.target.style.borderColor = '';
      e.target.title = '';
    } else {
      e.target.style.borderColor = 'red';
      e.target.title = validation.error;
    }
  });
  if (redSearchWidths) redSearchWidths.addEventListener('input', (e) => {
    state.red_search_widths_str = e.target.value;
    const validation = validateSearchWidths(e.target.value);
    if (validation.valid) {
      state.red_search_widths = validation.widths;
      e.target.style.borderColor = '';
      e.target.title = '';
    } else {
      e.target.style.borderColor = 'red';
      e.target.title = validation.error;
    }
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
      document.getElementById('auto-step-checkbox').checked = true;
    }
    
    state.trmph = '#13,';
    // Use blue's settings for reset
    const result = await fetchState(
      state.trmph,
      state.blue_model_id,
      getPlayerTemperature('blue', state.blue_move_method)
    );
    state.board = result.board;
    state.player = result.player;
    state.legal_moves = result.legal_moves;
    state.winner = result.winner;
    state.last_move = null;
    state.last_move_player = null; // Reset last_move_player
    updateUI();
    void refreshMoveHeatmap();
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

  const heatmapEnabled = document.getElementById('heatmap-enabled');
  if (heatmapEnabled) {
    heatmapEnabled.addEventListener('change', (e) => {
      state.heatmap_enabled = e.target.checked;
      if (!state.heatmap_enabled) {
        heatmapRequestToken += 1;
        state.heatmap_loading = false;
        clearHeatmapData();
        updateUI();
      } else {
        void refreshMoveHeatmap();
      }
    });
  }

  const heatmapOpacity = document.getElementById('heatmap-opacity');
  if (heatmapOpacity) {
    heatmapOpacity.addEventListener('input', (e) => {
      state.heatmap_opacity = parseFloat(e.target.value);
      updateUI();
    });
  }

  const heatmapScope = document.getElementById('heatmap-scope');
  if (heatmapScope) {
    heatmapScope.addEventListener('change', (e) => {
      state.heatmap_scope = e.target.value === 'all_legal' ? 'all_legal' : 'policy_top_k';
      updateUI();
      if (state.heatmap_enabled) {
        void refreshMoveHeatmap();
      }
    });
  }

  const heatmapTopK = document.getElementById('heatmap-top-k');
  if (heatmapTopK) {
    heatmapTopK.addEventListener('input', (e) => {
      const parsed = parseInt(e.target.value, 10);
      if (Number.isFinite(parsed)) {
        state.heatmap_top_k = Math.min(GAME_CONSTANTS.BOARD_SIZE * GAME_CONSTANTS.BOARD_SIZE, Math.max(1, parsed));
      }
      updateUI();
    });
    heatmapTopK.addEventListener('change', () => {
      if (state.heatmap_enabled) {
        void refreshMoveHeatmap();
      }
    });
  }

  const heatmapRefresh = document.getElementById('heatmap-refresh');
  if (heatmapRefresh) {
    heatmapRefresh.addEventListener('click', () => {
      if (state.heatmap_enabled) {
        void refreshMoveHeatmap();
      }
    });
  }

  // Dark mode toggle
  document.getElementById('dark-mode-toggle').addEventListener('click', toggleDarkMode);

  document.getElementById('undo-btn').addEventListener('click', () => {
    if (state.move_history.length > 0) {
      // Save current state to redo history before undoing
      const currentState = {
        trmph: state.trmph,
        board: JSON.parse(JSON.stringify(state.board)),
        player: state.player,
        legal_moves: [...state.legal_moves],
        winner: state.winner,
        last_move: state.last_move ? [...state.last_move] : null,
        last_move_player: state.last_move_player
      };
      state.redo_history.push(currentState);
      
      // Restore previous state
      const previousState = state.move_history.pop();
      Object.assign(state, previousState);
      updateUI();
      void refreshMoveHeatmap();
    }
  });

  document.getElementById('redo-btn').addEventListener('click', () => {
    if (state.redo_history.length > 0) {
      // Save current state to undo history before redoing
      const currentState = {
        trmph: state.trmph,
        board: JSON.parse(JSON.stringify(state.board)),
        player: state.player,
        legal_moves: [...state.legal_moves],
        winner: state.winner,
        last_move: state.last_move ? [...state.last_move] : null,
        last_move_player: state.last_move_player
      };
      state.move_history.push(currentState);
      
      // Restore next state from redo history
      const nextState = state.redo_history.pop();
      Object.assign(state, nextState);
      updateUI();
      void refreshMoveHeatmap();
    }
  });

});

// --- Debug Output Functions ---
function displayAlgorithmInfo(debugInfo) {
  if (!debugInfo || state.verbose_level === 0) return;
  
  const debugContent = document.getElementById('debug-content');
  if (!debugContent) return;
  
  let output = '';
  
  // Algorithm identification (always show at top)
  if (debugInfo.algorithm_info) {
    const algo = debugInfo.algorithm_info;
    output += '=== ALGORITHM USED ===\n';
    output += `Algorithm: ${algo.algorithm}\n`;
    
    if (algo.early_termination) {
      output += `Early Termination: YES (${algo.early_termination_reason})\n`;
      
      // Add specific information for terminal move detection
      if (algo.early_termination_reason === 'terminal_move' && algo.early_termination_details) {
        const details = algo.early_termination_details;
        if (details.move) {
          output += `Terminal Move: ${details.move[0]},${details.move[1]} (${String.fromCharCode(97 + details.move[1])}${details.move[0] + 1})\n`;
        }
        output += `Win Probability: ${(details.win_probability * 100).toFixed(1)}%\n`;
      }
    } else {
      output += `Early Termination: NO\n`;
    }
    
    // Display algorithm-specific parameters
    if (algo.parameters) {
      output += `Parameters: `;
      const paramStrings = [];
      for (const [key, value] of Object.entries(algo.parameters)) {
        if (Array.isArray(value)) {
          paramStrings.push(`${key}=[${value.join(',')}]`);
        } else {
          paramStrings.push(`${key}=${value}`);
        }
      }
      output += paramStrings.join(', ');
      output += '\n';
    }
    output += '\n';
  }
  
  return output;
}

function shouldShowMCTSStats(mctsDebugInfo) {
  // Determine if MCTS-specific statistics should be displayed.
  if (!mctsDebugInfo || !mctsDebugInfo.algorithm_info) return false;
  
  const algorithmUsed = mctsDebugInfo.algorithm_info.algorithm;
  const earlyTerminated = mctsDebugInfo.algorithm_info.early_termination;
  
  // Don't show MCTS stats for early termination cases
  return algorithmUsed === "MCTS" && !earlyTerminated;
}

function displayMCTSDebugInfo(mctsDebugInfo) {
  if (!mctsDebugInfo || state.verbose_level === 0) return;
  
  const debugContent = document.getElementById('debug-content');
  if (!debugContent) return;
  
  let output = '';
  
  // Display algorithm information first
  output += displayAlgorithmInfo(mctsDebugInfo);
  
  // Determine once whether to show MCTS-specific stats
  const showMCTSStats = shouldShowMCTSStats(mctsDebugInfo);
  
  // MCTS Search Statistics (condensed) - only show if MCTS was used
  if (mctsDebugInfo.search_stats && showMCTSStats) {
    output += '=== MCTS SEARCH STATISTICS ===\n';
    output += `Simulations: ${mctsDebugInfo.search_stats.num_simulations} | `;
    output += `Time: ${mctsDebugInfo.search_stats.search_time.toFixed(3)}s | `;
    output += `Inferences: ${mctsDebugInfo.search_stats.total_inferences} | `;
    output += `Exploration: ${mctsDebugInfo.search_stats.exploration_constant} | `;
    output += `Temperature: ${mctsDebugInfo.search_stats.temperature}\n\n`;
  }
  
  // Move Selection
  if (mctsDebugInfo.move_selection) {
    output += '=== MOVE SELECTION ===\n';
    output += `Selected: ${mctsDebugInfo.move_selection.selected_move} (${mctsDebugInfo.move_selection.selected_move_coords[0]}, ${mctsDebugInfo.move_selection.selected_move_coords[1]})\n\n`;
  }
  
  // Tree Statistics (condensed) - only show if MCTS was used
  if (mctsDebugInfo.tree_statistics && showMCTSStats) {
    output += '=== TREE STATISTICS ===\n';
    output += `Nodes: ${mctsDebugInfo.tree_statistics.total_nodes} | `;
    output += `Max Depth: ${mctsDebugInfo.tree_statistics.max_depth} | `;
    output += `Total Visits: ${mctsDebugInfo.tree_statistics.total_visits}\n\n`;
  }
  
  // Move Probabilities
  if (mctsDebugInfo.move_probabilities) {
    output += '=== MOVE PROBABILITIES ===\n';
    
    // MCTS visit counts (top moves only) - only show if MCTS was used
    if (mctsDebugInfo.move_probabilities.mcts_visits && showMCTSStats) {
      output += 'MCTS Visit Counts:\n';
      const sortedVisits = Object.entries(mctsDebugInfo.move_probabilities.mcts_visits)
        .sort(([,a], [,b]) => b - a);
      
      // Show only moves with visits > 0, limit to top 10
      const nonZeroVisits = sortedVisits.filter(([, visits]) => visits > 0).slice(0, 10);
      
      if (nonZeroVisits.length > 0) {
        nonZeroVisits.forEach(([move, visits]) => {
          const prob = visits / mctsDebugInfo.tree_statistics.total_visits * 100;
          output += `  ${move}: ${visits} visits (${prob.toFixed(1)}%)\n`;
        });
        
        // Add summary for remaining moves
        const remainingVisits = sortedVisits.filter(([, visits]) => visits > 0).slice(10);
        if (remainingVisits.length > 0) {
          const totalRemaining = remainingVisits.reduce((sum, [, visits]) => sum + visits, 0);
          const remainingProb = totalRemaining / mctsDebugInfo.tree_statistics.total_visits * 100;
          output += `  ... and ${remainingVisits.length} more moves (${remainingProb.toFixed(1)}%)\n`;
        }
      } else {
        output += '  (no moves were visited)\n';
      }
      output += '\n';
    }
    
    // Direct policy probabilities (top moves only)
    if (mctsDebugInfo.move_probabilities.direct_policy) {
      output += 'Direct Policy Probabilities (Top 10):\n';
      const sortedDirect = Object.entries(mctsDebugInfo.move_probabilities.direct_policy)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 10);
      sortedDirect.forEach(([move, prob]) => {
        const probPercent = (prob * 100).toFixed(2);
        output += `  ${move}: ${probPercent}%\n`;
      });
      output += '\n';
    }
  }
  
  // Comparison (top differences only) - only show if MCTS was used
  if (mctsDebugInfo.comparison && mctsDebugInfo.comparison.mcts_vs_direct && showMCTSStats) {
    output += '=== MCTS vs DIRECT POLICY COMPARISON ===\n';
    const sortedComparison = Object.entries(mctsDebugInfo.comparison.mcts_vs_direct)
      .sort(([,a], [,b]) => Math.abs(b.difference) - Math.abs(a.difference))
      .slice(0, 10); // Show top 10 biggest differences
    sortedComparison.forEach(([move, data]) => {
      const mctsPercent = (data.mcts_probability * 100).toFixed(1);
      const directPercent = (data.direct_probability * 100).toFixed(1);
      const diffPercent = (data.difference * 100).toFixed(1);
      const diffSign = data.difference >= 0 ? '+' : '';
      output += `  ${move}: MCTS ${mctsPercent}% vs Direct ${directPercent}% (${diffSign}${diffPercent}%)\n`;
    });
    output += '\n';
  }
  
  // Win Rate Analysis
  if (mctsDebugInfo.win_rate_analysis) {
    output += '=== WIN RATE ANALYSIS ===\n';
    const winRate = mctsDebugInfo.win_rate_analysis;
    output += `Root Value: ${winRate.root_value.toFixed(4)}\n`;
    output += `Best Child Value: ${winRate.best_child_value.toFixed(4)}\n`;
    output += `Win Probability: ${(winRate.win_probability * 100).toFixed(2)}%\n\n`;
  }
  
  // Gumbel Root Selection (if used)
  if (mctsDebugInfo.gumbel_analysis && mctsDebugInfo.gumbel_analysis.gumbel_used) {
    const g = mctsDebugInfo.gumbel_analysis;
    output += '=== GUMBEL ROOT SELECTION ===\n';
    if (g.gumbel_selection_note) output += `${g.gumbel_selection_note}\n`;
    if (g.gumbel_final_rank_top_move) output += `Top Final-Rank Move: ${g.gumbel_final_rank_top_move}\n`;
    if (g.gumbel_v_pi_01 !== null && g.gumbel_v_pi_01 !== undefined) output += `v_pi (0-1): ${Number(g.gumbel_v_pi_01).toFixed(4)}\n`;
    if (Array.isArray(g.gumbel_final_rank_top5) && g.gumbel_final_rank_top5.length > 0) {
      output += 'Final-Rank Top 5 Candidate Rows (log_prior + c_scale*(q - v_pi)):\n';
      g.gumbel_final_rank_top5.forEach((row) => {
        const score = (row.score !== null && row.score !== undefined) ? Number(row.score).toFixed(4) : 'N/A';
        const logPrior = (row.log_prior !== null && row.log_prior !== undefined) ? Number(row.log_prior).toFixed(4) : 'N/A';
        const adv = (row.adv_01 !== null && row.adv_01 !== undefined) ? Number(row.adv_01).toFixed(4) : 'N/A';
        const q01 = (row.q_01 !== null && row.q_01 !== undefined) ? Number(row.q_01).toFixed(4) : 'N/A';
        const qSource = row.q_source ? `, qsrc=${row.q_source}` : '';
        output += `  ${row.move}: score=${score} (log_prior=${logPrior}, q01=${q01}, adv=${adv}, visits=${row.visits}${qSource})\n`;
      });
    }
    output += '\n';
  }
  
  // Summary
  if (mctsDebugInfo.summary) {
    output += '=== SUMMARY ===\n';
    const summary = mctsDebugInfo.summary;
    // Selected move can differ from argmax due to temperature sampling / Gumbel.
    output += `Selected Move: ${summary.selected_move || (mctsDebugInfo.move_selection ? mctsDebugInfo.move_selection.selected_move : 'N/A')}\n`;
    output += `Top MCTS Move (Raw Visits): ${summary.top_mcts_move_raw_visits || summary.top_mcts_move || 'N/A'}\n`;
    if (summary.top_mcts_move_temperature_scaled) {
      output += `Top MCTS Move (Temperature Scaled): ${summary.top_mcts_move_temperature_scaled}\n`;
    }
    output += `Top Direct Move: ${summary.top_direct_move || 'N/A'}\n`;
    if (summary.moves_explored !== null && summary.moves_explored !== undefined) {
      output += `Moves Explored: ${summary.moves_explored}/${summary.total_legal_moves}\n`;
    } else {
      output += `Moves Explored: N/A (Algorithm Termination)\n`;
    }
    output += `Search Efficiency: ${summary.search_efficiency.toFixed(2)} inferences/simulation\n\n`;
  }
  
  // Move Sequence Analysis
  if (mctsDebugInfo.move_sequence_analysis) {
    output += '=== MOVE SEQUENCE ANALYSIS ===\n';
    const seqAnalysis = mctsDebugInfo.move_sequence_analysis;
    
    if (seqAnalysis.principal_variation && seqAnalysis.principal_variation.length > 0) {
      output += `Principal Variation (${seqAnalysis.pv_length} moves):\n`;
      seqAnalysis.principal_variation.forEach((move, index) => {
        output += `  ${index + 1}. ${move}\n`;
      });
      output += '\n';
    }
    
    if (seqAnalysis.alternative_lines && seqAnalysis.alternative_lines.length > 0) {
      output += 'Alternative Lines:\n';
      seqAnalysis.alternative_lines.forEach(alt => {
        const probPercent = (alt.probability * 100).toFixed(1);
        output += `  Depth ${alt.depth + 1}: ${alt.move} (${alt.visits} visits, ${probPercent}%, value: ${alt.value.toFixed(4)})\n`;
      });
      output += '\n';
    }
  }
  
  debugContent.textContent = output;
}

function displayFixedTreeDebugInfo(fixedTreeDebugInfo) {
  if (!fixedTreeDebugInfo || state.verbose_level === 0) return;
  
  const debugContent = document.getElementById('debug-content');
  if (!debugContent) return;
  
  let output = '';
  
  // Display algorithm information first
  output += displayAlgorithmInfo(fixedTreeDebugInfo);
  
  // Fixed Tree Search Statistics
  if (fixedTreeDebugInfo.search_stats) {
    output += '=== FIXED TREE SEARCH STATISTICS ===\n';
    output += `Search Widths: [${fixedTreeDebugInfo.search_stats.search_widths.join(', ')}] | `;
    output += `Time: ${fixedTreeDebugInfo.search_stats.search_time.toFixed(3)}s | `;
    output += `Temperature: ${fixedTreeDebugInfo.search_stats.temperature}\n\n`;
  }
  
  // Move Selection
  if (fixedTreeDebugInfo.move_selection) {
    output += '=== MOVE SELECTION ===\n';
    output += `Selected: ${fixedTreeDebugInfo.move_selection.selected_move} (${fixedTreeDebugInfo.move_selection.selected_move_coords[0]}, ${fixedTreeDebugInfo.move_selection.selected_move_coords[1]})\n\n`;
  }
  
  // Tree Statistics
  if (fixedTreeDebugInfo.tree_statistics) {
    output += '=== TREE STATISTICS ===\n';
    output += `Total Positions: ${fixedTreeDebugInfo.tree_statistics.total_positions} | `;
    output += `Tree Depth: ${fixedTreeDebugInfo.tree_statistics.tree_depth} | `;
    output += `Tree Width: ${fixedTreeDebugInfo.tree_statistics.tree_width} | `;
    output += `Policy Evaluations: ${fixedTreeDebugInfo.tree_statistics.policy_evaluations} | `;
    output += `Value Evaluations: ${fixedTreeDebugInfo.tree_statistics.value_evaluations}\n\n`;
  }
  
  // Move Probabilities
  if (fixedTreeDebugInfo.move_probabilities) {
    output += '=== MOVE PROBABILITIES ===\n';
    
    // Direct policy probabilities (top moves only)
    if (fixedTreeDebugInfo.move_probabilities.direct_policy) {
      output += 'Direct Policy Probabilities:\n';
      const sortedPolicy = Object.entries(fixedTreeDebugInfo.move_probabilities.direct_policy)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 10);
      sortedPolicy.forEach(([move, prob]) => {
        const probPercent = (prob * 100).toFixed(2);
        output += `  ${move}: ${probPercent}%\n`;
      });
      output += '\n';
    }
  }
  
  // Win Rate Analysis
  if (fixedTreeDebugInfo.win_rate_analysis) {
    output += '=== WIN RATE ANALYSIS ===\n';
    output += `Root Value: ${fixedTreeDebugInfo.win_rate_analysis.root_value.toFixed(4)} | `;
    output += `Win Probability: ${(fixedTreeDebugInfo.win_rate_analysis.win_probability * 100).toFixed(2)}%\n\n`;
  }
  
  // Summary
  if (fixedTreeDebugInfo.summary) {
    output += '=== SUMMARY ===\n';
    output += `Algorithm: ${fixedTreeDebugInfo.summary.algorithm_summary}\n`;
    output += `Total Legal Moves: ${fixedTreeDebugInfo.summary.total_legal_moves}\n`;
    output += `Moves Explored: ${fixedTreeDebugInfo.summary.moves_explored}\n`;
    output += `Search Efficiency: ${(fixedTreeDebugInfo.summary.search_efficiency * 100).toFixed(1)}%\n`;
    if (fixedTreeDebugInfo.summary.top_direct_move) {
      output += `Top Direct Move: ${fixedTreeDebugInfo.summary.top_direct_move}\n`;
    }
    output += '\n';
  }
  
  // Profiling Summary
  if (fixedTreeDebugInfo.profiling_summary) {
    output += '=== PERFORMANCE ===\n';
    output += `Total Compute: ${fixedTreeDebugInfo.profiling_summary.total_compute_ms}ms | `;
    output += `Memory Usage: ${fixedTreeDebugInfo.profiling_summary.memory_usage_mb.toFixed(1)}MB\n`;
    output += `Tree Building: ${fixedTreeDebugInfo.profiling_summary.tree_building_time_ms}ms | `;
    output += `Leaf Evaluation: ${fixedTreeDebugInfo.profiling_summary.leaf_evaluation_time_ms}ms | `;
    output += `Backup: ${fixedTreeDebugInfo.profiling_summary.backup_time_ms}ms\n`;
    output += `Policy NN: ${fixedTreeDebugInfo.profiling_summary.policy_nn_time_ms}ms | `;
    output += `Value NN: ${fixedTreeDebugInfo.profiling_summary.value_nn_time_ms}ms\n\n`;
  }
  
  debugContent.textContent = output;
}

function displayDebugInfo(debugInfo) {
  if (!debugInfo || state.verbose_level === 0) {
    return;
  }
  
  const debugContent = document.getElementById('debug-content');
  if (!debugContent) return;
  
  let output = '';
  
  // Display algorithm information first
  output += displayAlgorithmInfo(debugInfo);
  
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
    output += `Value Signed: ${debugInfo.basic.value_signed.toFixed(4)}\n`;
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

  // Display detailed exploration if available
  displayDetailedExploration(debugInfo);
}

function displayPolicyInfo(policyInfo) {
  if (!policyInfo || state.verbose_level === 0) return;
  
  const debugContent = document.getElementById('debug-content');
  if (!debugContent) return;
  
  let output = '';
  
  // Policy move information
  output += '=== POLICY MOVE ===\n';
  output += `Selected Move: ${policyInfo.selected_move}\n`;
  output += `Probability: ${(policyInfo.selected_probability * 100).toFixed(2)}%\n`;
  output += `Temperature: ${policyInfo.temperature}\n\n`;
  
  // Top moves
  if (policyInfo.top_moves && policyInfo.top_moves.length > 0) {
    output += 'Top Policy Moves:\n';
    policyInfo.top_moves.forEach(([move, prob], index) => {
      const probPercent = (prob * 100).toFixed(2);
      output += `  ${index + 1}. ${move}: ${probPercent}%\n`;
    });
    output += '\n';
  }
  
  debugContent.textContent = output;
}

function formatFinite(value, digits = 4, fallback = 'n/a') {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return num.toFixed(digits);
}

function moveLabelFromRow(row) {
  if (!row || typeof row !== 'object') return '?';
  if (row.move) return String(row.move);
  if (row.tensor_action !== undefined && row.tensor_action !== null) return `#${row.tensor_action}`;
  return '?';
}

function formatTopMRows(rows, maxRows = 16) {
  if (!Array.isArray(rows) || rows.length === 0) return '  (none)\n';
  const limited = rows.slice(0, maxRows);
  let output = '';
  output += '  move   prior    logP      g      g+logP\n';
  output += '  -----  -------  -------  -------  -------\n';
  limited.forEach((row) => {
    const move = moveLabelFromRow(row).padEnd(5);
    const prior = formatFinite(row.prior, 4).padStart(7);
    const logP = formatFinite(row.log_prior, 4).padStart(7);
    const g = formatFinite(row.gumbel, 4).padStart(7);
    const score = formatFinite(row.top_m_score, 4).padStart(7);
    output += `  ${move}  ${prior}  ${logP}  ${g}  ${score}\n`;
  });
  if (rows.length > limited.length) {
    output += `  ... ${rows.length - limited.length} more\n`;
  }
  return output;
}

function formatGumbelScoreRows(rows, options = {}) {
  const maxRows = options.maxRows || 16;
  const showGumbel = options.showGumbel !== false;

  if (!Array.isArray(rows) || rows.length === 0) return '  (none)\n';
  const limited = rows.slice(0, maxRows);

  let output = '';
  if (showGumbel) {
    output += '  move   N    qsrc  prior    logP     q01      adv     c*adv      g      score\n';
    output += '  -----  ---  ----  -------  -------  -------  -------  -------  -------  -------\n';
  } else {
    output += '  move   N    qsrc  prior    logP     q01      adv     c*adv    score\n';
    output += '  -----  ---  ----  -------  -------  -------  -------  -------  -------\n';
  }

  limited.forEach((row) => {
    const move = moveLabelFromRow(row).padEnd(5);
    const visits = String(Number.isFinite(Number(row.visits)) ? Number(row.visits) : 0).padStart(3);
    const qsrc = (row.q_source === 'v_pi_completion' ? 'vpi' : 'tree').padEnd(4);
    const prior = formatFinite(row.prior, 4).padStart(7);
    const logP = formatFinite(row.log_prior, 4).padStart(7);
    const q01 = formatFinite(row.q_01, 4).padStart(7);
    const adv = formatFinite(row.adv_01, 4).padStart(7);
    const cAdv = formatFinite(row.value_term, 4).padStart(7);
    const score = formatFinite(row.score, 4).padStart(7);

    if (showGumbel) {
      const gumbel = formatFinite(row.gumbel_term, 4).padStart(7);
      output += `  ${move}  ${visits}  ${qsrc}  ${prior}  ${logP}  ${q01}  ${adv}  ${cAdv}  ${gumbel}  ${score}\n`;
    } else {
      output += `  ${move}  ${visits}  ${qsrc}  ${prior}  ${logP}  ${q01}  ${adv}  ${cAdv}  ${score}\n`;
    }
  });

  if (rows.length > limited.length) {
    output += `  ... ${rows.length - limited.length} more\n`;
  }

  return output;
}

function formatProbAsPercent(prob, digits = 2) {
  const num = Number(prob);
  if (!Number.isFinite(num)) return 'n/a';
  return `${(num * 100).toFixed(digits)}%`;
}

function formatValueSummaryInline(summary) {
  if (!summary || typeof summary !== 'object') return 'n/a';
  const rootWin = formatProbAsPercent(summary.root_win_prob, 2);
  const rootSigned = formatFinite(summary.root_ref_signed, 4);
  const redSigned = formatFinite(summary.red_ref_signed, 4);
  const ptmWin = formatProbAsPercent(summary.ptm_win_prob, 2);
  const toPlay = summary.to_play || 'n/a';
  const cacheFlag = summary.from_cache ? 'cache' : 'fresh';
  return `root_win=${rootWin}, root_signed=${rootSigned}, red_signed=${redSigned}, ptm_win=${ptmWin}, to_play=${toPlay}, src=${cacheFlag}`;
}

function formatActionDiveReplyRows(rows, maxRows = 6) {
  if (!Array.isArray(rows) || rows.length === 0) return '  (none)\n';

  const limited = rows.slice(0, maxRows);
  let output = '';
  output += '  move   N    prior    q_src   q_root(tree)  root_win(tree)  value_head(root_win)\n';
  output += '  -----  ---  -------  ------  ------------  --------------  --------------------\n';

  limited.forEach((row) => {
    const move = String(row.move || '?').padEnd(5);
    const visits = String(Number.isFinite(Number(row.visits)) ? Number(row.visits) : 0).padStart(3);
    const prior = formatFinite(row.prior, 4).padStart(7);
    const qSource = String(row.q_source || 'n/a').padEnd(6);
    const qRootTree = formatFinite(row.q_root_ref_signed, 4).padStart(12);
    const rootWinTree = formatProbAsPercent(row.q_root_win_prob, 2).padStart(14);
    const valueHeadRootWin = formatProbAsPercent(row.value_after_reply?.root_win_prob, 2).padStart(20);
    output += `  ${move}  ${visits}  ${prior}  ${qSource}  ${qRootTree}  ${rootWinTree}  ${valueHeadRootWin}\n`;
  });

  if (rows.length > limited.length) {
    output += `  ... ${rows.length - limited.length} more\n`;
  }
  return output;
}

function renderGumbelDetailedTrace(detailedExploration, trace) {
  const gumbelEvents = trace.filter((step) => typeof step?.type === 'string' && step.type.startsWith('gumbel_'));
  if (gumbelEvents.length === 0) return null;

  let output = '';
  output += '=== DETAILED GUMBEL ROOT TRACE ===\n';
  output += `Simulation threshold: ${detailedExploration.simulation_threshold}\n`;
  output += `Total simulations: ${detailedExploration.total_simulations}\n\n`;

  const setup = gumbelEvents.find((step) => step.type === 'gumbel_root_setup');
  if (setup) {
    output += 'Scoring formulas used:\n';
    output += `  Round ranking: ${setup.score_formula_round}\n`;
    output += `  Final ranking: ${setup.score_formula_final}\n`;
    output += `  v_pi=${formatFinite(setup.v_pi_01, 4)} | candidates=${setup.candidate_count}/${setup.legal_action_count} | rounds=${setup.rounds_R} | c_scale=${formatFinite(setup.c_scale, 3)}\n`;
    output += '  qsrc=tree means visited child Q; qsrc=vpi means unvisited and q01 is completed to v_pi.\n\n';
  }

  gumbelEvents.forEach((step) => {
    switch (step.type) {
      case 'gumbel_root_setup':
        break;

      case 'gumbel_top_m_selection':
        output += '=== TOP-M CANDIDATE DRAW (g + log_prior) ===\n';
        output += `Selected candidates: ${step.candidate_count}\n`;
        output += formatTopMRows(step.selected_rows, 20);
        if (Array.isArray(step.excluded_rows) && step.excluded_rows.length > 0) {
          output += 'Near-miss non-candidates:\n';
          output += formatTopMRows(step.excluded_rows, 5);
        }
        output += '\n';
        break;

      case 'gumbel_round_start':
        output += `=== ROUND ${step.round_index}/${step.rounds_total} BEFORE FORCED SIMS ===\n`;
        output += `Arms: ${step.arms} | per_arm: ${step.per_arm} | sims used: ${step.sims_used_before} | sims left: ${step.sims_left_before}\n`;
        output += formatGumbelScoreRows(step.candidate_rows, { maxRows: 24, showGumbel: true });
        output += '\n';
        break;

      case 'gumbel_round_end':
        output += `=== ROUND ${step.round_index}/${step.rounds_total} AFTER FORCED SIMS ===\n`;
        output += `Sims used: ${step.sims_used_after} | sims left: ${step.sims_left_after}\n`;
        if (step.reason) output += `Reason: ${step.reason}\n`;
        output += formatGumbelScoreRows(step.candidate_rows, { maxRows: 24, showGumbel: true });
        if (Array.isArray(step.kept_moves) && step.kept_moves.length > 0) {
          output += `Kept (${step.keep_count}): ${step.kept_moves.join(', ')}\n`;
        }
        if (Array.isArray(step.dropped_moves) && step.dropped_moves.length > 0) {
          output += `Dropped: ${step.dropped_moves.join(', ')}\n`;
        }
        output += '\n';
        break;

      case 'gumbel_final_selection':
        output += '=== FINAL DETERMINISTIC RANKING (NO GUMBEL TERM) ===\n';
        output += formatGumbelScoreRows(step.final_rank_rows, { maxRows: 24, showGumbel: false });
        if (step.selected_move) {
          output += `Selected move: ${step.selected_move}\n`;
        }
        if (Array.isArray(step.final_rank_rows) && step.final_rank_rows.length >= 2) {
          const best = Number(step.final_rank_rows[0].score);
          const second = Number(step.final_rank_rows[1].score);
          if (Number.isFinite(best) && Number.isFinite(second)) {
            output += `Top-2 margin: ${(best - second).toFixed(4)}\n`;
          }
        }
        output += '\n';
        break;

      case 'gumbel_action_dive':
        output += `=== ACTION DEEP DIVE: ${step.move || `#${step.tensor_action}`} ===\n`;
        output += `Root child stats: visits=${step.root_child_visits}, q_ptm=${formatFinite(step.root_child_q_ptm_signed, 4)}, q01=${formatFinite(step.root_child_q_01, 4)}\n`;
        output += `Value after root move: ${formatValueSummaryInline(step.value_after_root)}\n`;
        output += `Replies explored: visited=${step.reply_count_visited}/${step.reply_count_total}\n`;
        if (step.note) output += `Note: ${step.note}\n`;

        output += 'Top opponent replies by visits:\n';
        output += formatActionDiveReplyRows(step.top_replies_by_visits, 6);

        output += 'Top opponent replies by policy prior:\n';
        output += formatActionDiveReplyRows(step.top_replies_by_policy, 6);
        output += '\n';
        break;

      default:
        break;
    }
  });

  return output;
}

function renderLegacyDetailedTrace(detailedExploration, trace) {
  let output = '';
  output += '=== DETAILED MCTS EXPLORATION TRACE ===\n';
  output += `Simulation threshold: ${detailedExploration.simulation_threshold}\n`;
  output += `Total simulations: ${detailedExploration.total_simulations}\n\n`;

  if (!Array.isArray(trace) || trace.length === 0) {
    output += 'No exploration trace available.\n';
    return output;
  }

  trace.forEach((step) => {
    try {
      switch (step.type) {
        case 'descent_start':
          output += `DESCENT #${step.sim} | root visits: ${step.root_visits} | gumbel_forced: ${step.gumbel_forced}\n`;
          if (step.pv_hint) output += `PV: ${step.pv_hint.join(' -> ')}\n`;
          break;
        case 'forced_root_action':
          output += `Forced root action: ${step.tensor_action} (legal: ${step.legal_at_root})\n`;
          break;
        case 'select_action':
          output += `Select: depth=${step.depth} score=${formatFinite(step.score, 4)} (Q=${formatFinite(step.q, 4)}, U=${formatFinite(step.u, 4)}, P=${formatFinite(step.p, 4)}, N=${step.n})\n`;
          break;
        case 'node_realized':
          output += `Created child ${step.move} @depth ${step.depth}\n`;
          break;
        case 'leaf_selected':
          output += `Leaf @depth ${step.depth} (${step.leaf_reason}) T=${step.T} U=${step.U}\n`;
          break;
        case 'batch_flush':
          output += `Batch flush: ${step.reason} (T=${step.T}, U=${step.U}, target=${step.distinct_target})\n`;
          break;
        case 'nn_eval_start':
          output += `NN start: batch=${step.batch_size}, to_eval=${step.to_eval}, distinct=${step.distinct}\n`;
          break;
        case 'nn_eval_done':
          output += `NN done: batch=${step.effective_batch_size}, time=${formatFinite(step.time_ms, 1)}ms\n`;
          break;
        case 'expand_node':
          output += `Expanded node @depth ${step.depth} (children=${step.children_count})\n`;
          break;
        case 'backprop_update':
          output += `Backprop: root_Q ${formatFinite(step.root_q_before, 4)} -> ${formatFinite(step.root_q_after, 4)}\n`;
          break;
        default:
          output += `Unknown step type: ${step.type}\n`;
      }
      output += '\n';
    } catch (error) {
      output += `Error processing step: ${error.message}\n\n`;
    }
  });

  return output;
}

function displayDetailedExploration(debugInfo) {
  const explorationDiv = document.getElementById('detailed-exploration');
  const explorationContent = document.getElementById('exploration-content');
  
  if (!explorationDiv || !explorationContent) {
    console.warn('Could not find detailed exploration DOM elements');
    return;
  }
  
  // Always show the section, but explain why it might be empty
  explorationDiv.style.display = 'block';
  
  // Check if we have detailed exploration data
  // Look for tree_data either directly or nested in mcts_debug_info
  const mctsData = debugInfo.mcts_debug_info;
  const treeData = debugInfo.tree_data || (mctsData && mctsData.tree_data);
  
  const detailedExploration = treeData && treeData.detailed_exploration;
  
  if (!detailedExploration || !detailedExploration.enabled) {
    let output = '=== DETAILED MCTS EXPLORATION TRACE ===\n';
    output += '❌ Detailed exploration is not available.\n\n';
    
    if (!debugInfo) {
      output += 'Reason: No debug info provided\n';
    } else if (!mctsData) {
      output += 'Reason: No MCTS debug info found\n';
    } else if (!treeData) {
      output += 'Reason: No tree data found\n';
    } else if (!detailedExploration) {
      output += 'Reason: No detailed exploration data in tree data\n';
      if (treeData) {
        output += `Tree data keys: ${Object.keys(treeData).join(', ')}\n`;
      }
    } else if (!detailedExploration.enabled) {
      output += `Reason: Detailed exploration is disabled\n`;
      output += `Simulation threshold: ${detailedExploration.simulation_threshold || '≤47'}\n`;
      output += `Current simulations: ${mctsData?.search_stats?.num_simulations || 'unknown'}\n`;
    }
    
    explorationContent.textContent = output;
    return;
  }
  
  const trace = Array.isArray(detailedExploration.trace) ? detailedExploration.trace : [];
  const gumbelOutput = renderGumbelDetailedTrace(detailedExploration, trace);
  if (gumbelOutput) {
    explorationContent.textContent = gumbelOutput;
    return;
  }

  explorationContent.textContent = renderLegacyDetailedTrace(detailedExploration, trace);
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
  
  // Clear redo history when new moves are made
  state.redo_history = [];
  
  // Keep only last 5,000 moves in history
  if (state.move_history.length > 5000) {
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
    if (state.heatmap_enabled) {
      void refreshMoveHeatmap();
    }
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

// --- TRMPH Sequence Functions ---
async function applyTrmphSequence() {
  const trmphSequenceInput = document.getElementById('trmph-sequence-input');
  const statusElement = document.getElementById('trmph-sequence-status');
  const trmphSequence = trmphSequenceInput.value.trim();
  
  if (!trmphSequence) {
    showTrmphStatus('Please enter a TRMPH sequence', 'error');
    return;
  }
  
  try {
    showTrmphStatus('Applying TRMPH sequence...', 'info');
    
    const response = await fetch('/api/apply_trmph_sequence', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        trmph: state.trmph,
        trmph_sequence: trmphSequence,
        model_id: getCurrentPlayerSettings().model_id,
        temperature: getCurrentPlayerSettings().temperature,
        verbose: state.verbose_level
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to apply TRMPH sequence');
    }
    
    const result = await response.json();
    
    if (result.success !== undefined && !result.success) {
      throw new Error(result.error || 'Failed to apply TRMPH sequence');
    }
    
    // Update state with the new board state
    state.trmph = result.new_trmph;
    state.board = result.board;
    state.player = result.player;
    state.legal_moves = result.legal_moves;
    state.winner = result.winner;
    state.policy = result.policy;
    state.value = result.value_signed;
    state.win_prob = result.win_prob;
    
    // Update the TRMPH string display
    document.getElementById('trmph-string').value = state.trmph;
    
    // Update the UI
    updateUI();
    void refreshMoveHeatmap();
    
    // Show success message
    const movesApplied = result.moves_applied || 0;
    const gameStatus = result.game_over ? ' (Game Over)' : '';
    showTrmphStatus(`Successfully applied ${movesApplied} moves${gameStatus}`, 'success');
    
  } catch (error) {
    console.error('Error applying TRMPH sequence:', error);
    showTrmphStatus(`Error: ${error.message}`, 'error');
  }
}

function clearTrmphSequence() {
  document.getElementById('trmph-sequence-input').value = '';
  document.getElementById('trmph-sequence-status').innerHTML = '';
  document.getElementById('trmph-sequence-status').className = 'status-message';
}

function showTrmphStatus(message, type) {
  const statusElement = document.getElementById('trmph-sequence-status');
  statusElement.textContent = message;
  statusElement.className = `status-message ${type}`;
}

// Save game functionality
async function saveGame() {
  try {
    // Get current MCTS parameters
    const mctsParams = {
      blue: {
        num_simulations: state.blue_num_simulations,
        exploration_constant: state.blue_exploration_constant,
        temperature: state.blue_temperature
      },
      red: {
        num_simulations: state.red_num_simulations,
        exploration_constant: state.red_exploration_constant,
        temperature: state.red_temperature
      }
    };
    
    // Determine which model to use (use blue's model for now)
    const modelId = state.blue_model_id;
    
    const response = await fetch('/api/save_game', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        trmph: state.trmph,
        winner: state.winner,
        model_id: modelId,
        mcts_params: mctsParams
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      alert(`Game saved successfully!\n\nWinner: ${state.winner}\nSaved to: ${result.trmph_file}`);
    } else if (result.needs_winner_input) {
      // Show winner selection modal
      showWinnerModal();
    } else {
      alert(`Error saving game: ${result.error}`);
    }
  } catch (error) {
    console.error('Error saving game:', error);
    alert(`Error saving game: ${error.message}`);
  }
}

function showWinnerModal() {
  document.getElementById('winner-modal').style.display = 'block';
}

function closeWinnerModal() {
  document.getElementById('winner-modal').style.display = 'none';
}

async function selectWinner(winner) {
  closeWinnerModal();
  
  try {
    // Get current MCTS parameters
    const mctsParams = {
      blue: {
        num_simulations: state.blue_num_simulations,
        exploration_constant: state.blue_exploration_constant,
        temperature: state.blue_temperature
      },
      red: {
        num_simulations: state.red_num_simulations,
        exploration_constant: state.red_exploration_constant,
        temperature: state.red_temperature
      }
    };
    
    // Determine which model to use (use blue's model for now)
    const modelId = state.blue_model_id;
    
    const response = await fetch('/api/save_game', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        trmph: state.trmph,
        winner: winner,
        model_id: modelId,
        mcts_params: mctsParams
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      alert(`Game saved successfully!\n\nWinner: ${winner}\nSaved to: ${result.trmph_file}`);
    } else {
      alert(`Error saving game: ${result.error}`);
    }
  } catch (error) {
    console.error('Error saving game:', error);
    alert(`Error saving game: ${error.message}`);
  }
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
  
  // TRMPH sequence functionality
  document.getElementById('apply-trmph-sequence').addEventListener('click', applyTrmphSequence);
  document.getElementById('clear-trmph-sequence').addEventListener('click', clearTrmphSequence);
  
  // Save game functionality
  document.getElementById('save-game-btn').addEventListener('click', saveGame);
  
  // Winner selection modal
  document.getElementById('winner-blue-btn').addEventListener('click', () => selectWinner('blue'));
  document.getElementById('winner-red-btn').addEventListener('click', () => selectWinner('red'));
  document.getElementById('winner-cancel-btn').addEventListener('click', closeWinnerModal);
  
  // Close winner modal when clicking outside
  document.getElementById('winner-modal').addEventListener('click', (e) => {
    if (e.target.id === 'winner-modal') {
      closeWinnerModal();
    }
  });
  
  // Close winner modal with X button
  document.querySelector('#winner-modal .close').addEventListener('click', closeWinnerModal);
}); 
