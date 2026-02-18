// =============================================================================
// Hex AI Public Web App - Simplified Mobile-Optimized Version
// =============================================================================

class HexGame {
    constructor() {
        this.boardSize = null;  // Must be set from backend - fail fast if not
        this.displayBoardSize = null; // Top-left KxK region exposed to the user
        this.displayBoardSizeOptions = [];
        this.currentTRMPH = "";
        this.gameHistory = [];
        this.redoHistory = []; // Track undone moves for redo functionality
        this.moveCount = 0;
        this.currentElo = null;  // Must be set from backend - fail fast if not
        this.blueComputer = false;
        this.redComputer = true;
        this.isLoading = false;
        this.isInitialLoad = true; // Track if this is the initial page load
        this.darkMode = false; // Dark mode state
        this.heatmapEnabled = false;
        this.heatmapLoading = false;
        this.heatmapError = null;
        this.heatmapScores = {};
        this.heatmapPolicyProbs = {};
        this.heatmapMinScore = null;
        this.heatmapMaxScore = null;
        this.heatmapOpacity = 0.62;
        this.heatmapScope = 'policy_top_k';
        this.heatmapTopK = 12;
        this.heatmapScoreType = 'policy_value';
        this.heatmapPolicyTemperature = 1.0;
        this.heatmapRequestToken = 0;
        this.pieRuleEnabled = true;
        this.pieRuleCanSwap = false;
        this.pieRuleAction = 'none';
        this.pieRuleArmed = true;

        // Track previous board state for efficient updates
        this.previousBoard = null;
        this.hexElements = new Map(); // Cache hex elements by position

        // Difficulty levels will be loaded from backend in loadGameConstants()
        this.difficultyLevels = null;

        // Purple hexes configuration using TRMPH coordinates
        this.PURPLE_HEXES = ['b10', 'b11', 'b12', 'b3', 'b4', 'b5', 'b6', 'd10',
            'b7', 'b8', 'b9', 'c10', 'c11', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9',
            'd11', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'e10', 'e11', 'e3', 'e4', 'e5',
            'e6', 'e7', 'e8', 'e9', 'f10', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'g10', 'g4',
            'g5', 'g6', 'g7', 'g8', 'g9', 'h10', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'i10',
            'i11', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'j10', 'j11', 'j3', 'j4', 'j5', 'j6',
            'j7', 'j8', 'j9', 'k10', 'k11', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'l2', 'l3',
            'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'l11'];

        this.virtualBoardPrefillMoves = {
            13: '',
            12: 'a13m1b13m2c13m3d13m4e13m5f13m6g13m7h13m8i13m9j13m10k13m11l13m12',
            11: 'a12l1b12l2c12l3d12l4e12l5f12l6g12l7h12l8i12l9j12l10k12l11k13m11',
            10: 'a11k1b11k2c11k3d11k4e11k5f11k6g11k7h11k8i11k9j11k10j12l10j13m10',
            9: 'a10j1b10j2c10j3d10j4e10j5f10j6g10j7h10j8i10j9i11k9i12l9i13m9',
            8: 'a9i1b9i2c9i3d9i4e9i5f9i6g9i7h9i8h10j8h11k8h12l8h13m8',
            7: 'a8h1b8h2c8h3d8h4e8h5f8h6g8h7g9i7g10j7g11k7g12l7g13m7',
            6: 'a7g1b7g2c7g3d7g4e7g5f7g6f8h6f9i6f10j6f11k6f12l6f13m6',
            5: 'a6f1b6f2c6f3d6f4e6f5e7g5e8h5e9i5e10j5e11k5e12l5e13m5',
            4: 'a5e1b5e2c5e3d5e4d6f4d7g4d8h4d9i4d10j4d11k4d12l4d13m4',
            3: 'a4d1b4d2c4d3c5e3c6f3c7g3c8h3c9i3c10j3c11k3c12l3c13m3',
            2: 'a3c1b3c2b4d2b5e2b6f2b7g2b8h2b9i2b10j2b11k2b12l2b13m2'
        };

        this.initializeElements();
        this.defaultInstructionText = this.instructionText ? this.instructionText.textContent : '';
        this.setupEventListeners();
        this.loadGameConstants();
        this.initializeDarkMode();
        this.updateHeatmapControls();
        this.updatePieRuleUi();
    }

    // =============================================================================
    // INITIALIZATION
    // =============================================================================

    initializeElements() {
        this.statusLine = document.getElementById('status-line');
        this.boardContainer = document.getElementById('board-container');
        this.instructionText = document.getElementById('instruction-text');
        this.resetBtn = document.getElementById('reset-btn');
        this.undoBtn = document.getElementById('undo-btn');
        this.redoBtn = document.getElementById('redo-btn');
        this.computerMoveBtn = document.getElementById('computer-move-btn');
        this.difficultyPreset = document.getElementById('difficulty-preset');
        this.boardSizeSelect = document.getElementById('board-size-select');
        this.eloSlider = document.getElementById('elo-slider');
        this.eloDisplay = document.getElementById('elo-display');
        this.blueComputerCheck = document.getElementById('blue-computer');
        this.redComputerCheck = document.getElementById('red-computer');
        this.trmphDisplay = document.getElementById('trmph-display');
        this.trmphInput = document.getElementById('trmph-input');
        this.copyTrmphBtn = document.getElementById('copy-trmph');
        this.applyTrmphBtn = document.getElementById('apply-trmph');
        this.trmphError = document.getElementById('trmph-error');
        this.darkModeToggle = document.getElementById('dark-mode-toggle');
        this.heatmapEnabledCheck = document.getElementById('heatmap-enabled');
        this.heatmapOpacityInput = document.getElementById('heatmap-opacity');
        this.heatmapOpacityValue = document.getElementById('heatmap-opacity-value');
        this.heatmapStatus = document.getElementById('heatmap-status');
        this.heatmapScopeSelect = document.getElementById('heatmap-scope');
        this.heatmapTopKInput = document.getElementById('heatmap-top-k');
        this.heatmapRefreshBtn = document.getElementById('heatmap-refresh');
        this.pieRuleEnabledCheck = document.getElementById('pie-rule-enabled');
        this.pieRuleStatus = document.getElementById('pie-rule-status');
        this.pieRuleBadge = document.getElementById('pie-rule-badge');
        this.pieRuleLabel = document.getElementById('pie-rule-label');
        this.pieRulePlayer = document.getElementById('pie-rule-player');
        this.pieRulePlayerColor = document.getElementById('pie-rule-player-color');
        this.swapFlash = document.getElementById('swap-flash');
    }

    initializeDifficultyDropdown() {
        if (!this.difficultyLevels) {
            throw new Error('Difficulty levels not loaded from backend - this should not happen');
        }

        // Build all options first, then replace the dropdown content
        const optionsHtml = this.difficultyLevels.map(level =>
            `<option value="${level.elo}">${level.label} (ELO ${level.elo})</option>`
        ).join('');

        // Replace all options at once to avoid empty state
        this.difficultyPreset.innerHTML = optionsHtml;

        // Set the dropdown to show the appropriate level for current ELO
        this.updateDifficultyPreset();
    }

    initializeBoardSizeDropdown() {
        if (!this.boardSizeSelect) {
            return;
        }
        if (!Array.isArray(this.displayBoardSizeOptions) || this.displayBoardSizeOptions.length === 0) {
            throw new Error('Display board size options not loaded from backend');
        }

        const optionsHtml = this.displayBoardSizeOptions
            .map((size) => `<option value="${size}">${size}x${size}</option>`)
            .join('');
        this.boardSizeSelect.innerHTML = optionsHtml;
        this.boardSizeSelect.value = String(this.validateBoardSize());
    }

    setupEventListeners() {
        this.resetBtn.addEventListener('click', () => this.resetGame());
        this.undoBtn.addEventListener('click', () => this.undoMove());
        this.redoBtn.addEventListener('click', () => this.redoMove());
        this.computerMoveBtn.addEventListener('click', () => this.makeComputerMove());
        this.copyTrmphBtn.addEventListener('click', () => this.copyTrmph());
        this.applyTrmphBtn.addEventListener('click', () => this.applyTrmphSequence());

        this.difficultyPreset.addEventListener('change', (e) => {
            this.currentElo = parseInt(e.target.value);
            this.eloSlider.value = this.currentElo;
            this.eloDisplay.textContent = this.currentElo;
        });

        if (this.boardSizeSelect) {
            this.boardSizeSelect.addEventListener('change', async (e) => {
                const selected = parseInt(e.target.value, 10);
                if (!Number.isFinite(selected) || selected === this.displayBoardSize) {
                    return;
                }
                this.displayBoardSize = selected;
                localStorage.setItem('hex_ai_display_board_size', String(selected));
                this.configureHeatmapTopKBounds();
                this.clearHeatmapData();
                this.updateHeatmapControls();
                this.updateInstructionTextForBoardMode();
                await this.resetGame();
            });
        }

        this.eloSlider.addEventListener('input', (e) => {
            this.currentElo = parseInt(e.target.value);
            this.eloDisplay.textContent = this.currentElo;
            this.updateDifficultyPreset();
        });

        this.blueComputerCheck.addEventListener('change', (e) => {
            this.blueComputer = e.target.checked;
            this.updatePieRuleUi();
        });

        this.redComputerCheck.addEventListener('change', (e) => {
            this.redComputer = e.target.checked;
            this.updatePieRuleUi();
        });

        if (this.pieRuleEnabledCheck) {
            this.pieRuleEnabledCheck.addEventListener('change', async (e) => {
                this.pieRuleEnabled = e.target.checked;
                localStorage.setItem('hex_ai_pie_rule_enabled', String(this.pieRuleEnabled));
                this.updatePieRuleUi();
                await this.loadGameStateWithoutAutoMove('pie_rule_toggle');
            });
        }

        this.darkModeToggle.addEventListener('click', () => this.toggleDarkMode());

        if (this.heatmapEnabledCheck) {
            this.heatmapEnabledCheck.addEventListener('change', async (e) => {
                this.heatmapEnabled = e.target.checked;
                if (!this.heatmapEnabled) {
                    this.heatmapRequestToken += 1;
                    this.heatmapLoading = false;
                    this.clearHeatmapData();
                    this.updateHeatmapControls();
                    this.redrawCurrentBoard();
                } else {
                    await this.refreshHeatmap(true);
                }
            });
        }

        if (this.heatmapOpacityInput) {
            this.heatmapOpacityInput.addEventListener('input', (e) => {
                this.heatmapOpacity = parseFloat(e.target.value);
                this.updateHeatmapControls();
                if (this.heatmapEnabled) {
                    this.redrawCurrentBoard();
                }
            });
        }

        if (this.heatmapScopeSelect) {
            this.heatmapScopeSelect.addEventListener('change', async (e) => {
                this.heatmapScope = e.target.value === 'all_legal' ? 'all_legal' : 'policy_top_k';
                this.updateHeatmapControls();
                if (this.heatmapEnabled) {
                    await this.refreshHeatmap(true);
                }
            });
        }

        if (this.heatmapTopKInput) {
            this.heatmapTopKInput.addEventListener('input', (e) => {
                const parsed = parseInt(e.target.value, 10);
                if (Number.isFinite(parsed)) {
                    const boardSize = this.validateBoardSize();
                    const maxMoves = boardSize * boardSize;
                    this.heatmapTopK = Math.max(1, Math.min(maxMoves, parsed));
                }
                this.updateHeatmapControls();
            });
            this.heatmapTopKInput.addEventListener('change', async () => {
                if (this.heatmapEnabled) {
                    await this.refreshHeatmap(true);
                }
            });
        }

        if (this.heatmapRefreshBtn) {
            this.heatmapRefreshBtn.addEventListener('click', async () => {
                if (this.heatmapEnabled) {
                    await this.refreshHeatmap(true);
                }
            });
        }
    }

    // =============================================================================
    // DARK MODE FUNCTIONS
    // =============================================================================

    toggleDarkMode() {
        this.darkMode = !this.darkMode;

        // Update the data-theme attribute on the document
        if (this.darkMode) {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }

        // Update the toggle button text and icon
        if (this.darkMode) {
            this.darkModeToggle.textContent = '☀️ Light';
            this.darkModeToggle.title = 'Switch to light mode';
        } else {
            this.darkModeToggle.textContent = '🌙 Dark';
            this.darkModeToggle.title = 'Switch to dark mode';
        }

        // Redraw the board with new colors
        if (this.svg) {
            this.drawHexBoard(this.previousBoard);
        }

        // Save preference to localStorage
        localStorage.setItem('hex_ai_dark_mode', this.darkMode.toString());
    }

    initializeDarkMode() {
        // Check localStorage for saved preference
        const savedDarkMode = localStorage.getItem('hex_ai_dark_mode');
        if (savedDarkMode !== null) {
            this.darkMode = savedDarkMode === 'true';
        } else {
            // Default to light mode instead of following system preference
            this.darkMode = false;
        }

        // Apply the theme
        if (this.darkMode) {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }

        // Update the toggle button
        if (this.darkMode) {
            this.darkModeToggle.textContent = '☀️ Light';
            this.darkModeToggle.title = 'Switch to light mode';
        } else {
            this.darkModeToggle.textContent = '🌙 Dark';
            this.darkModeToggle.title = 'Switch to dark mode';
        }
    }

    // =============================================================================
    // COLOR PALETTES FOR DARK MODE
    // =============================================================================

    getColors() {
        return this.darkMode ? this.DARK_COLORS : this.LIGHT_COLORS;
    }

    get LIGHT_COLORS() {
        return {
            WHITE: '#fff',
            LIGHT_GRAY: '#f8f8fa',
            MEDIUM_GRAY: '#bbb',
            DARK_GRAY: '#222',

            // Board colors
            EMPTY_HEX_GRAY: '#f0f0f0',      // ⭐ LIGHT GRAY for empty hexagons
            GRID_WHITE: '#f8f8fa',          // ⭐ LIGHT GRAY for grid lines between hexagons
            BOARD_BACKGROUND: '#f8f8fa',     // ⭐ LIGHT GRAY for board background

            // Blue palette - using original bright colors
            LIGHT_BLUE: '#e7fcfc',
            MEDIUM_BLUE: '#bbeeee',         // ⭐ LIGHT CYAN - used for grid lines
            DARK_BLUE: '#0099ff',           // ⭐ ORIGINAL BRIGHT BLUE - used for blue pieces
            VERY_DARK_BLUE: '#0099ff',      // ⭐ VIVID BLUE - used for edges and winning pieces
            DARKER_BLUE: '#0066cc',         // ⭐ DARK BLUE - used for last moves

            // Red palette - using original bright colors
            LIGHT_RED: '#fff4ea',
            MEDIUM_RED: '#ffe1c8',
            DARK_RED: '#ff4444',            // ⭐ ORIGINAL BRIGHT RED - used for red pieces
            VERY_DARK_RED: '#ff4444',       // ⭐ ORIGINAL BRIGHT RED - used for edges and winning pieces
            DARKER_RED: '#cc3300',          // ⭐ DARK RED - used for last moves
        };
    }

    get DARK_COLORS() {
        return {
            WHITE: '#2d2d2d',
            LIGHT_GRAY: '#1a1a1a',
            MEDIUM_GRAY: '#666',
            DARK_GRAY: '#e0e0e0',

            // Board colors
            EMPTY_HEX_GRAY: '#3a3a3a',      // ⭐ DARK GRAY for empty hexagons
            GRID_WHITE: '#4a4a4a',          // ⭐ DARK GRAY for grid lines between hexagons
            BOARD_BACKGROUND: '#232323',     // ⭐ LIGHTER GRAY - halfway between #1a1a1a and #2d2d2d

            // Blue palette - using bright blue for dark theme
            LIGHT_BLUE: '#1a3a4a',
            MEDIUM_BLUE: '#2a5a6a',         // ⭐ DARKER CYAN for grid lines
            DARK_BLUE: '#0099ff',           // ⭐ SAME BRIGHT BLUE as light theme
            VERY_DARK_BLUE: '#0099ff',      // ⭐ SAME BRIGHT BLUE as light theme
            DARKER_BLUE: '#0066cc',         // ⭐ DARKER BLUE for last moves

            // Red palette - using bright red for dark theme
            LIGHT_RED: '#4a2a1a',
            MEDIUM_RED: '#6a3a2a',
            DARK_RED: '#ff4444',            // ⭐ SAME BRIGHT RED as light theme
            VERY_DARK_RED: '#ff4444',       // ⭐ SAME BRIGHT RED as light theme
            DARKER_RED: '#cc3300',          // ⭐ DARKER RED for last moves
        };
    }

    updateDifficultyPreset() {
        if (!this.difficultyLevels) {
            throw new Error('Difficulty levels not loaded from backend - this should not happen');
        }

        // Find the highest difficulty level that the user hasn't exceeded
        let selectedLevel = this.difficultyLevels[0]; // Default to first level
        for (const level of this.difficultyLevels) {
            if (this.currentElo >= level.elo) {
                selectedLevel = level;
            } else {
                break;
            }
        }

        // Update the dropdown to show the selected level
        this.difficultyPreset.value = selectedLevel.elo;
    }

    clearHeatmapData() {
        this.heatmapScores = {};
        this.heatmapPolicyProbs = {};
        this.heatmapMinScore = null;
        this.heatmapMaxScore = null;
        this.heatmapError = null;
    }

    getHeatmapScoreForMove(row, col) {
        if (!this.heatmapEnabled) {
            return null;
        }
        const trmph = this.rowColToTRMPH(row, col);
        const value = this.heatmapScores[trmph];
        return Number.isFinite(value) ? value : null;
    }

    getHeatmapFillColor(score) {
        const fallback = this.getColors().EMPTY_HEX_GRAY;
        if (!Number.isFinite(score)) {
            return fallback;
        }
        if (window.HexHeatmap && typeof window.HexHeatmap.scoreToColor === 'function') {
            return window.HexHeatmap.scoreToColor(score, {
                alpha: this.heatmapOpacity,
                darkMode: this.darkMode,
                baseColor: fallback,
                fallback
            });
        }
        return fallback;
    }

    getHeatmapBandClass(score) {
        if (!Number.isFinite(score)) {
            return null;
        }
        if (window.HexHeatmap && typeof window.HexHeatmap.classifyScore === 'function') {
            return window.HexHeatmap.classifyScore(score, { neutralBand: 0.035 });
        }
        if (score < 0.465) {
            return 'below';
        }
        if (score > 0.535) {
            return 'above';
        }
        return 'even';
    }

    getTooltipTextForHex(row, col) {
        const trmph = this.rowColToTRMPH(row, col);
        const score = this.getHeatmapScoreForMove(row, col);
        if (!Number.isFinite(score)) {
            return trmph;
        }
        const percent = window.HexHeatmap && typeof window.HexHeatmap.formatPercent === 'function'
            ? window.HexHeatmap.formatPercent(score)
            : `${(score * 100).toFixed(1)}%`;
        const band = this.getHeatmapBandClass(score);
        const bandLabel = band === 'above'
            ? '>50%'
            : band === 'below'
                ? '<50%'
                : '~50%';
        return `${trmph} (${percent} win, ${bandLabel})`;
    }

    applyHeatmapClasses(hexElement, cellValue, row, col) {
        hexElement.classList.remove('heatmap-scored', 'heatmap-above', 'heatmap-below', 'heatmap-even');
        if (cellValue !== this.pieceValues.EMPTY) {
            return;
        }
        const score = this.getHeatmapScoreForMove(row, col);
        if (!Number.isFinite(score)) {
            return;
        }
        hexElement.classList.add('heatmap-scored');
        const bandClass = this.getHeatmapBandClass(score);
        if (bandClass) {
            hexElement.classList.add(`heatmap-${bandClass}`);
        }
    }

    updateHeatmapControls() {
        if (this.heatmapEnabledCheck) {
            this.heatmapEnabledCheck.checked = this.heatmapEnabled;
        }
        if (this.heatmapOpacityInput) {
            this.heatmapOpacityInput.value = this.heatmapOpacity.toFixed(2);
        }
        if (this.heatmapOpacityValue) {
            this.heatmapOpacityValue.textContent = `${Math.round(this.heatmapOpacity * 100)}%`;
        }
        if (this.heatmapScopeSelect) {
            this.heatmapScopeSelect.value = this.heatmapScope;
        }
        if (this.heatmapTopKInput) {
            this.heatmapTopKInput.value = this.heatmapTopK;
            this.heatmapTopKInput.disabled = this.heatmapScope === 'all_legal';
        }
        if (this.heatmapRefreshBtn) {
            this.heatmapRefreshBtn.disabled = !this.heatmapEnabled || this.heatmapLoading;
            this.heatmapRefreshBtn.textContent = this.heatmapLoading ? 'Analyzing...' : 'Analyze';
        }
        if (this.heatmapStatus) {
            if (!this.heatmapEnabled) {
                this.heatmapStatus.textContent = 'Off';
            } else if (this.heatmapLoading) {
                this.heatmapStatus.textContent = 'Loading...';
            } else if (this.heatmapError) {
                this.heatmapStatus.textContent = `Error: ${this.heatmapError}`;
            } else if (Number.isFinite(this.heatmapMinScore) && Number.isFinite(this.heatmapMaxScore)) {
                const selected = Object.keys(this.heatmapScores).length;
                const scopeLabel = this.heatmapScope === 'all_legal'
                    ? 'all legal'
                    : `top ${this.heatmapTopK}`;
                if (window.HexHeatmap && typeof window.HexHeatmap.formatPercent === 'function') {
                    this.heatmapStatus.textContent = `${scopeLabel}: ${selected} moves, ${window.HexHeatmap.formatPercent(this.heatmapMinScore)} to ${window.HexHeatmap.formatPercent(this.heatmapMaxScore)}`;
                } else {
                    this.heatmapStatus.textContent = `${scopeLabel}: ${selected} moves, ${(this.heatmapMinScore * 100).toFixed(1)}% to ${(this.heatmapMaxScore * 100).toFixed(1)}%`;
                }
            } else {
                this.heatmapStatus.textContent = 'No legal moves';
            }
        }
    }

    updatePieRuleUi() {
        if (this.pieRuleEnabledCheck) {
            this.pieRuleEnabledCheck.checked = this.pieRuleEnabled;
        }
        if (this.pieRuleStatus) {
            if (!this.pieRuleEnabled) {
                this.pieRuleStatus.textContent = 'Disabled';
            } else if (this.pieRuleCanSwap) {
                this.pieRuleStatus.textContent = 'Swap window';
            } else {
                this.pieRuleStatus.textContent = 'Enabled';
            }
        }
        const shouldShowOverlay = this.shouldShowPieRuleReminder();
        if (this.pieRuleBadge) {
            this.pieRuleBadge.classList.toggle('pie-rule-hidden', !shouldShowOverlay);
            this.pieRuleBadge.classList.toggle('pie-rule-visible', shouldShowOverlay);
        }
        if (this.pieRulePlayer) {
            this.pieRulePlayer.classList.remove('pie-rule-hidden');
            this.pieRulePlayer.classList.add('pie-rule-visible');
        }
        if (this.pieRuleLabel) {
            this.pieRuleLabel.classList.toggle('pie-rule-hidden', !shouldShowOverlay);
            this.pieRuleLabel.classList.toggle('pie-rule-visible', shouldShowOverlay);
            const mainText = this.pieRuleCanSwap ? 'Pie rule: swap window' : 'Pie rule enabled';
            this.pieRuleLabel.innerHTML = `<span class="pie-rule-link-main">${mainText}</span><span class="pie-rule-link-sub">(Rules)</span>`;
        }

        if (this.pieRulePlayerColor) {
            const display = this.getHumanPlayerDisplayState();
            this.pieRulePlayerColor.textContent = display.label;
            this.pieRulePlayerColor.classList.remove(
                'pie-rule-player-red',
                'pie-rule-player-blue',
                'pie-rule-player-both',
                'pie-rule-player-none'
            );
            this.pieRulePlayerColor.classList.add(display.className);
        }
    }

    shouldShowPieRuleReminder() {
        if (!this.pieRuleEnabled) {
            return false;
        }
        let moveCount = 0;
        if (this.currentTRMPH && this.currentTRMPH.trim().length > 0) {
            try {
                moveCount = this.parseTrmphMoves(this.currentTRMPH).length;
            } catch (_error) {
                moveCount = 0;
            }
        }
        return moveCount === 0 && this.pieRuleArmed;
    }

    getHumanPlayerDisplayState() {
        if (!this.blueComputer && !this.redComputer) {
            return { label: 'Both', className: 'pie-rule-player-both' };
        }
        if (this.blueComputer && this.redComputer) {
            return { label: 'None', className: 'pie-rule-player-none' };
        }

        if (!this.blueComputer && this.redComputer) {
            return { label: 'Blue', className: 'pie-rule-player-blue' };
        }
        if (this.blueComputer && !this.redComputer) {
            return { label: 'Red', className: 'pie-rule-player-red' };
        }

        let moveCount = 0;
        if (this.currentTRMPH && this.currentTRMPH.trim().length > 0) {
            try {
                moveCount = this.parseTrmphMoves(this.currentTRMPH).length;
            } catch (_error) {
                moveCount = 0;
            }
        }
        return moveCount % 2 === 0
            ? { label: 'Blue', className: 'pie-rule-player-blue' }
            : { label: 'Red', className: 'pie-rule-player-red' };
    }

    applyPieRuleStateFromResponse(data, announceSwap = false) {
        if (!data || typeof data !== 'object') {
            return;
        }

        this.pieRuleCanSwap = Boolean(data.pie_rule_can_swap);
        this.pieRuleAction = typeof data.pie_rule_action === 'string' ? data.pie_rule_action : 'none';

        if (this.pieRuleAction === 'swapped' || this.pieRuleAction === 'declined') {
            this.pieRuleArmed = false;
        } else if (this.pieRuleCanSwap) {
            this.pieRuleArmed = true;
        } else if (
            (typeof data.new_trmph === 'string' && data.new_trmph.length === 0) ||
            (typeof data.trmph === 'string' && data.trmph.length === 0)
        ) {
            this.pieRuleArmed = true;
        }

        if (this.pieRuleAction === 'swapped' && data.pie_rule_swap_computer_colors) {
            this.swapComputerColorAssignments();
            this.showSwapFlash();
            if (announceSwap) {
                this.showSuccess('Pie rule: computer swapped colors. Your turn.');
            }
        }

        this.updatePieRuleUi();
    }

    getPieRuleRequestEnabled() {
        return this.pieRuleEnabled && this.pieRuleArmed;
    }

    showSwapFlash() {
        if (!this.swapFlash) {
            return;
        }
        this.swapFlash.classList.remove('swap-flash-active');
        void this.swapFlash.offsetWidth;
        this.swapFlash.classList.add('swap-flash-active');
    }

    swapComputerColorAssignments() {
        const oldBlueComputer = this.blueComputer;
        this.blueComputer = this.redComputer;
        this.redComputer = oldBlueComputer;

        if (this.blueComputerCheck) {
            this.blueComputerCheck.checked = this.blueComputer;
        }
        if (this.redComputerCheck) {
            this.redComputerCheck.checked = this.redComputer;
        }
    }

    configureHeatmapTopKBounds() {
        const boardSize = this.validateBoardSize();
        const maxMoves = boardSize * boardSize;
        this.heatmapTopK = Math.max(1, Math.min(this.heatmapTopK, maxMoves));
        if (this.heatmapTopKInput) {
            this.heatmapTopKInput.max = String(maxMoves);
            this.heatmapTopKInput.value = String(this.heatmapTopK);
        }
    }

    updateInstructionTextForBoardMode() {
        if (!this.instructionText) {
            return;
        }
        if (this.validateBoardSize() === this.boardSize) {
            this.instructionText.textContent = this.defaultInstructionText;
        } else {
            const size = this.validateBoardSize();
            this.instructionText.textContent = `Changed board size to ${size}`;
        }
    }

    async fetchMoveHeatmap() {
        const response = await fetch('/api/move_heatmap', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                trmph: this.currentTRMPH,
                elo_rating: this.validateEloRating(),
                display_board_size: this.validateBoardSize(),
                score_type: this.heatmapScoreType,
                selection_mode: this.heatmapScope,
                top_k: this.heatmapTopK,
                policy_temperature: this.heatmapPolicyTemperature
            })
        });

        if (!response.ok) {
            let message = `API error (${response.status})`;
            try {
                const errorData = await response.json();
                if (errorData && errorData.error) {
                    message = errorData.error;
                }
            } catch (_ignore) {
                // Keep default message when body is not JSON.
            }
            throw new Error(message);
        }

        return await response.json();
    }

    buildInlineHeatmapRequestPayload() {
        return {
            heatmap_enabled: this.heatmapEnabled,
            heatmap_selection_mode: this.heatmapScope,
            heatmap_top_k: this.heatmapTopK,
            heatmap_policy_temperature: this.heatmapPolicyTemperature
        };
    }

    applyHeatmapPayload(payload) {
        this.heatmapScores = payload && payload.scores ? payload.scores : {};
        this.heatmapPolicyProbs = payload && payload.policy_probs ? payload.policy_probs : {};
        this.heatmapMinScore = payload && Number.isFinite(payload.min_score) ? payload.min_score : null;
        this.heatmapMaxScore = payload && Number.isFinite(payload.max_score) ? payload.max_score : null;
        this.heatmapError = null;
    }

    redrawCurrentBoard() {
        if (!this.previousBoard) {
            return;
        }
        const boardCopy = this.previousBoard.map(row => [...row]);
        this.clearBoard();
        this.drawHexBoard(boardCopy);
        this.previousBoard = boardCopy.map(row => [...row]);
    }

    async refreshHeatmap(redraw = true) {
        if (!this.heatmapEnabled) {
            this.heatmapLoading = false;
            this.clearHeatmapData();
            this.updateHeatmapControls();
            if (redraw) {
                this.redrawCurrentBoard();
            }
            return;
        }

        const requestToken = ++this.heatmapRequestToken;
        this.heatmapLoading = true;
        this.heatmapError = null;
        this.updateHeatmapControls();

        try {
            const data = await this.fetchMoveHeatmap();
            if (requestToken !== this.heatmapRequestToken) {
                return;
            }
            this.applyHeatmapPayload(data);
        } catch (error) {
            if (requestToken !== this.heatmapRequestToken) {
                return;
            }
            this.clearHeatmapData();
            this.heatmapError = error.message || 'Failed to load heatmap';
        } finally {
            if (requestToken === this.heatmapRequestToken) {
                this.heatmapLoading = false;
                this.updateHeatmapControls();
                if (redraw) {
                    this.redrawCurrentBoard();
                }
            }
        }
    }

    async renderBoardWithHeatmap(board, inlineHeatmap = null) {
        if (this.heatmapEnabled && inlineHeatmap) {
            this.heatmapLoading = false;
            this.applyHeatmapPayload(inlineHeatmap);
            this.updateHeatmapControls();
        } else {
            await this.refreshHeatmap(false);
        }
        await this.renderBoard(board);
    }

    // =============================================================================
    // VALIDATION HELPERS
    // =============================================================================

    validateEloRating() {
        if (this.currentElo === null) {
            throw new Error('ELO rating not initialized from backend - this indicates a configuration error');
        }
        return this.currentElo;
    }

    validateBoardSize() {
        if (this.displayBoardSize === null) {
            throw new Error('Display board size not initialized from backend - this indicates a configuration error');
        }
        return this.displayBoardSize;
    }

    async loadGameConstants() {
        try {
            const response = await fetch('/api/constants');
            const data = await response.json();
            this.boardSize = data.BOARD_SIZE;
            this.pieceValues = data.PIECE_VALUES;
            this.playerValues = data.PLAYER_VALUES;
            this.winnerValues = data.WINNER_VALUES;
            this.displayBoardSizeOptions = Array.isArray(data.DISPLAY_BOARD_SIZE_OPTIONS) && data.DISPLAY_BOARD_SIZE_OPTIONS.length > 0
                ? data.DISPLAY_BOARD_SIZE_OPTIONS.map(v => parseInt(v, 10)).filter(Number.isFinite)
                : [this.boardSize];

            const backendDefaultDisplaySize = Number.isFinite(parseInt(data.DEFAULT_DISPLAY_BOARD_SIZE, 10))
                ? parseInt(data.DEFAULT_DISPLAY_BOARD_SIZE, 10)
                : this.boardSize;
            const savedDisplaySize = parseInt(localStorage.getItem('hex_ai_display_board_size'), 10);
            const validSizes = new Set(this.displayBoardSizeOptions);
            this.displayBoardSize = validSizes.has(savedDisplaySize)
                ? savedDisplaySize
                : backendDefaultDisplaySize;
            if (!validSizes.has(this.displayBoardSize)) {
                this.displayBoardSize = this.boardSize;
            }

            // Load difficulty levels from backend
            this.difficultyLevels = data.DIFFICULTY_LEVELS;

            // Load ELO configuration from backend
            this.minElo = data.ELO_CONFIG.MIN_ELO;
            this.maxElo = data.ELO_CONFIG.MAX_ELO;
            this.currentElo = data.ELO_CONFIG.DEFAULT_ELO;
            this.configureHeatmapTopKBounds();

            const backendPieDefault = typeof data.DEFAULT_PIE_RULE_ENABLED === 'boolean'
                ? data.DEFAULT_PIE_RULE_ENABLED
                : true;
            const savedPieRule = localStorage.getItem('hex_ai_pie_rule_enabled');
            if (savedPieRule === 'true' || savedPieRule === 'false') {
                this.pieRuleEnabled = savedPieRule === 'true';
            } else {
                this.pieRuleEnabled = backendPieDefault;
            }
            this.updatePieRuleUi();

            // Update slider and display with backend values
            this.eloSlider.min = this.minElo;
            this.eloSlider.max = this.maxElo;
            this.eloSlider.value = this.currentElo;
            this.eloDisplay.textContent = this.currentElo;

            // Initialize difficulty dropdown now that we have the data
            this.initializeDifficultyDropdown();
            this.initializeBoardSizeDropdown();
            this.updateInstructionTextForBoardMode();

            this.initializeBoard();
            // Initial load - allow computer auto-move
            await this.loadGameState(true, 'initial_load');
            this.isInitialLoad = false; // Mark initial load as complete
        } catch (error) {
            console.error('Failed to load game constants:', error);
            this.handleNetworkError(error, 'load game');
        }
    }

    initializeBoard() {
        // Create SVG board
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 600 600');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.style.cursor = 'pointer';

        // Add click event listener (let browser handle native touch gestures)
        svg.addEventListener('click', (e) => this.handleBoardClick(e));


        this.boardContainer.appendChild(svg);
        this.svg = svg;
    }

    async resetGame() {
        this.setLoading(true);
        try {
            this.currentTRMPH = "";
            this.gameHistory = [];
            this.moveCount = 0;
            this.pieRuleArmed = true;
            this.isInitialLoad = false; // Mark that this is no longer initial load
            this.previousBoard = null; // Clear board cache
            this.hexElements.clear(); // Clear hex cache
            this.updateTrmphDisplay();
            await this.loadGameState(false, 'reset'); // Don't auto-move after reset

            // Show instruction text for the new empty board
            this.showInstructionText();
        } catch (error) {
            console.error('Failed to reset game:', error);
            this.showError('Failed to reset game.');
        } finally {
            this.setLoading(false);
            this.updateButtonStates();
        }
    }

    async undoMove() {
        if (this.gameHistory.length === 0) return;

        this.setLoading(true);
        try {
            // Save current TRMPH string (not object) to redo history
            this.redoHistory.push(this.currentTRMPH);

            // Restore previous state
            this.gameHistory.pop();
            this.currentTRMPH = this.gameHistory.length > 0 ?
                this.gameHistory[this.gameHistory.length - 1] : "";

            // Recalculate moveCount from TRMPH string
            this.moveCount = this.currentTRMPH ?
                this.parseTrmphMoves(this.currentTRMPH).length : 0;
            if (this.moveCount === 0) {
                this.pieRuleArmed = true;
            }

            // Clear cache for undo to ensure clean state
            this.previousBoard = null;
            this.hexElements.clear();

            await this.loadGameStateWithoutAutoMove('undo');

            // Show instruction text if we're back to an empty board
            if (this.isBoardEmpty(this.previousBoard)) {
                this.showInstructionText();
            }
        } catch (error) {
            console.error('Failed to undo move:', error);
            this.showError('Failed to undo move.');
        } finally {
            this.setLoading(false);
            this.updateButtonStates();
        }
    }

    async makeComputerMove() {
        if (this.isLoading) return;

        // Hide instruction text since computer is making a move
        this.hideInstructionText();

        this.setLoading(true);
        try {
            const response = await fetch('/api/mcts_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trmph: this.currentTRMPH,
                    elo_rating: this.validateEloRating(),
                    display_board_size: this.validateBoardSize(),
                    pie_rule_enabled: this.getPieRuleRequestEnabled(),
                    ...this.buildInlineHeatmapRequestPayload()
                })
            });

            const data = await response.json();

            if (data.success) {
                this.applyPieRuleStateFromResponse(data, true);

                // Log the MCTS configuration that was actually used
                if (data.mcts_config) {
                    console.log('=== COMPUTER MOVE CONFIGURATION ===');
                    console.log('ELO Rating:', this.validateEloRating());
                    console.log('Algorithm:', data.mcts_config.algorithm || 'mcts');
                    console.log('Model:', data.mcts_config.model);
                    console.log('Simulations:', data.mcts_config.num_simulations);
                    // console.log('Exploration constant (c_puct):', data.mcts_config.exploration_constant);
                    // console.log('Temperature:', data.mcts_config.temperature, '->', data.mcts_config.temperature_end);
                    // console.log('Gumbel enabled:', data.mcts_config.enable_gumbel);
                    // console.log('Gumbel max sims:', data.mcts_config.gumbel_max_sims);
                    console.log('=====================================');
                }

                this.currentTRMPH = data.new_trmph;
                const moveWasPlayed = Boolean(data.move_made);
                if (moveWasPlayed) {
                    this.gameHistory.push(this.currentTRMPH);
                    this.moveCount++;

                    // Clear redo history when new moves are made
                    this.redoHistory = [];
                }
                this.legalMoves = data.legal_moves || [];

                await this.renderBoardWithHeatmap(data.board, data.move_heatmap || null);
                this.updateTrmphDisplay();
                this.updateButtonStates();

                // Auto-move for computer players
                if (data.winner) {
                    this.showGameOver(data.winner);
                } else if (data.pie_rule_action === 'swapped') {
                    // Hand control back to the user after swap.
                } else if (this.shouldMakeComputerMove(data.player)) {
                    this.scheduleAutoMove();
                }
            } else {
                console.log('Computer move failed, data.success:', data.success, 'data.error:', data.error);
                this.showError(data.error || 'Computer move failed');
            }
        } catch (error) {
            console.error('Computer move error:', error);
            this.handleNetworkError(error, 'get computer move');
        } finally {
            this.setLoading(false);
        }
    }

    shouldMakeComputerMove(currentPlayer) {
        return (currentPlayer === 'blue' && this.blueComputer) ||
            (currentPlayer === 'red' && this.redComputer);
    }

    scheduleAutoMove() {
        setTimeout(() => this.makeComputerMove(), 500);
    }

    async loadGameState(autoMove = true, stateReason = 'manual_refresh') {
        try {
            console.log(`Loading game state with TRMPH: '${this.currentTRMPH}', ELO: ${this.validateEloRating()}`);

            const response = await fetch('/api/state', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trmph: this.currentTRMPH,
                    elo_rating: this.validateEloRating(),
                    display_board_size: this.validateBoardSize(),
                    pie_rule_enabled: this.getPieRuleRequestEnabled(),
                    state_reason: stateReason
                })
            });

            if (!response.ok) {
                if (response.status === 429) {
                    const errorData = await response.json();
                    const waitTime = errorData.retry_after || 1;
                    this.showError(`Too many requests. Please wait ${waitTime.toFixed(1)} seconds.`);
                    return;
                }
                const errorText = await response.text();
                console.error(`API error: ${response.status} - ${errorText}`);
                this.showError(`Server error: ${response.status}`);
                return;
            }

            const data = await response.json();
            console.log('API response data:', data);

            if (data.error) {
                console.error('API returned error:', data.error);
                this.showError(data.error);
                return;
            }

            this.applyPieRuleStateFromResponse(data, false);

            // Store legal moves for move validation
            this.legalMoves = data.legal_moves || [];

            await this.renderBoardWithHeatmap(data.board);
            this.updateTrmphDisplay();

            // Handle game over and auto-move logic
            if (data.winner) {
                this.showGameOver(data.winner);
            } else if (autoMove && this.shouldMakeComputerMove(data.player) && this.isInitialLoad) {
                // Only auto-move on initial page load, not after reset
                this.scheduleAutoMove();
            }

            // Update button states (needed for undo/redo operations)
            if (!autoMove) {
                this.updateButtonStates();
            }
        } catch (error) {
            console.error('Failed to load game state:', error);
            this.handleNetworkError(error, 'load game state');
        }
    }

    async loadGameStateWithoutAutoMove(stateReason = 'manual_refresh') {
        return this.loadGameState(false, stateReason);
    }

    // =============================================================================
    // BOARD RENDERING
    // =============================================================================

    async renderBoard(board) {
        if (!this.svg) return;

        if (this.heatmapEnabled) {
            this.clearBoard();
            this.drawHexBoard(board);
            this.previousBoard = board.map(row => [...row]);
            return;
        }

        // If this is the first render or we don't have cached elements, do full render
        if (this.previousBoard === null || this.hexElements.size === 0) {
            this.clearBoard();
            this.drawHexBoard(board);
            this.previousBoard = board.map(row => [...row]); // Deep copy
            return;
        }

        // Otherwise, do efficient incremental update
        // Use requestAnimationFrame to ensure smooth rendering
        requestAnimationFrame(() => {
            this.updateBoardIncremental(board);
        });
    }

    clearBoard() {
        // Clear existing board
        this.svg.innerHTML = '';
        this.hexElements.clear();
    }

    updateBoardIncremental(newBoard) {
        // Only update hexes that have changed
        const boardSize = this.validateBoardSize();
        for (let row = 0; row < boardSize; row++) {
            for (let col = 0; col < boardSize; col++) {
                const oldValue = this.previousBoard[row][col];
                const newValue = newBoard[row][col];

                if (oldValue !== newValue) {
                    this.updateHex(row, col, newValue);
                }
            }
        }

        // Update legal moves highlighting
        this.updateLegalMovesHighlighting();

        // Update previous board state
        this.previousBoard = newBoard.map(row => [...row]); // Deep copy
    }

    updateHex(row, col, newValue) {
        const key = `${row},${col}`;
        let hexElement = this.hexElements.get(key);

        if (!hexElement) {
            // Create new hex if it doesn't exist (shouldn't happen in normal flow)
            const { x, y } = this.hexCenter(row, col, 18);
            hexElement = this.makeHex(x, y, 18, this.getHexColor(newValue, row, col), false);
            hexElement.setAttribute('data-row', row);
            hexElement.setAttribute('data-col', col);
            this.applyHeatmapClasses(hexElement, newValue, row, col);
            this.svg.appendChild(hexElement);
            this.hexElements.set(key, hexElement);
        } else {
            // Update existing hex with immediate color change (no transition during moves)
            const newColor = this.getHexColor(newValue, row, col);
            hexElement.style.transition = 'none'; // Disable transition for instant update
            hexElement.setAttribute('fill', newColor);
            this.applyHeatmapClasses(hexElement, newValue, row, col);

            // Re-enable transitions after a brief delay for hover effects
            setTimeout(() => {
                if (hexElement) {
                    hexElement.style.transition = '';
                }
            }, 50);
        }
    }

    getHexColor(cellValue, row = null, col = null) {
        const colors = this.getColors();
        if (cellValue === this.pieceValues.BLUE) return colors.DARK_BLUE;
        if (cellValue === this.pieceValues.RED) return colors.DARK_RED;
        if (row !== null && col !== null) {
            const score = this.getHeatmapScoreForMove(row, col);
            if (Number.isFinite(score)) {
                return this.getHeatmapFillColor(score);
            }
        }
        return colors.EMPTY_HEX_GRAY; // Empty hex color
    }

    shouldShadeHex(row, col) {
        if (this.validateBoardSize() !== this.boardSize) {
            return false;
        }
        // Convert row/col to TRMPH format and check if it's in the purple hex list
        const trmph = this.rowColToTRMPH(row, col);
        return this.PURPLE_HEXES.includes(trmph);
    }

    isBoardEmpty(board) {
        // Check if board is completely empty (first move)
        const boardSize = this.validateBoardSize();
        for (let row = 0; row < boardSize; row++) {
            for (let col = 0; col < boardSize; col++) {
                if (board[row][col] !== this.pieceValues.EMPTY) {
                    return false;
                }
            }
        }
        return true;
    }

    clearAllPurpleShading() {
        // Remove purple shading from all hexes
        const boardSize = this.validateBoardSize();
        for (let row = 0; row < boardSize; row++) {
            for (let col = 0; col < boardSize; col++) {
                const key = `${row},${col}`;
                const hexElement = this.hexElements.get(key);
                if (hexElement) {
                    hexElement.classList.remove('purple-shaded');
                }
            }
        }
    }

    hideInstructionText() {
        // Hide the instruction text about purple hexes
        if (this.instructionText) {
            this.instructionText.style.display = 'none';
        }
    }

    showInstructionText() {
        // Show the instruction text about purple hexes
        if (this.instructionText) {
            this.instructionText.style.display = 'block';
        }
    }

    updateLegalMovesHighlighting() {
        // Only update clickability for empty hexes to avoid unnecessary DOM operations
        const boardSize = this.validateBoardSize();
        for (let row = 0; row < boardSize; row++) {
            for (let col = 0; col < boardSize; col++) {
                const key = `${row},${col}`;
                const hexElement = this.hexElements.get(key);
                if (hexElement) {
                    const isLegal = this.isLegalMove(row, col);
                    const isEmpty = this.previousBoard[row][col] === this.pieceValues.EMPTY;

                    if (isLegal && isEmpty && !this.isLoading) {
                        // Only update if not already clickable
                        if (!hexElement.classList.contains('clickable')) {
                            hexElement.style.cursor = 'pointer';
                            hexElement.classList.add('clickable');
                            // Ensure event listener is attached
                            this.attachHexClickListener(hexElement, row, col);
                        }
                    } else {
                        // Only update if currently clickable
                        if (hexElement.classList.contains('clickable')) {
                            hexElement.style.cursor = 'default';
                            hexElement.classList.remove('clickable');
                        }
                    }
                }
            }
        }
    }

    attachHexClickListener(hexElement, row, col) {
        // Check if listener is already attached to avoid unnecessary DOM manipulation
        if (hexElement.hasAttribute('data-listener-attached')) {
            return;
        }

        // Add new listener
        hexElement.addEventListener('click', (e) => {
            this.onCellClick(e);
        });

        // Mark as having listener attached
        hexElement.setAttribute('data-listener-attached', 'true');
    }

    drawHexBoard(board) {
        if (
            window.HexHeatmap &&
            window.HexHeatmap.tooltip &&
            typeof window.HexHeatmap.tooltip.hide === 'function'
        ) {
            window.HexHeatmap.tooltip.hide();
        }

        // Constants for hexagonal board - adapted from working dev version
        const HEX_RADIUS = 18; // Slightly smaller for mobile
        const BOARD_SIZE = this.validateBoardSize();

        // Math for flat-topped hex grid, blue at top/bottom
        const w = HEX_RADIUS * Math.sqrt(3);
        const h = HEX_RADIUS * 1.5;

        // Calculate SVG dimensions with balanced padding
        const boardWidth = w * (BOARD_SIZE - 1 + 0.5) + 2 * HEX_RADIUS;
        const boardHeight = h * (BOARD_SIZE - 1) + 2 * HEX_RADIUS;
        const edgeBorderWidth = 17; // Account for thick edge borders
        const padding = HEX_RADIUS * 0.5; // Balanced padding on all sides

        // Diamond shape compensation - account for hex board's diagonal offset
        const diagonalOffset = w * (BOARD_SIZE - 1) * 0.45;

        const svgWidth = boardWidth + 2 * padding + edgeBorderWidth + diagonalOffset;
        const svgHeight = boardHeight + 2 * padding + edgeBorderWidth;

        // Set SVG dimensions
        this.svg.setAttribute('width', svgWidth);
        this.svg.setAttribute('height', svgHeight);
        this.svg.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);
        this.svg.style.background = this.getColors().BOARD_BACKGROUND;

        // Draw edge indicators (blue: top/bottom, red: left/right)
        this.drawEdgeIndicators(svgWidth, svgHeight, HEX_RADIUS, BOARD_SIZE);

        // Check if board is empty (first move)
        const isEmpty = this.isBoardEmpty(board);

        // Draw hexagons and cache them
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                const { x, y } = this.hexCenter(row, col, HEX_RADIUS);
                const cell = board[row]?.[col] || this.pieceValues.EMPTY;

                // Determine fill color
                const fill = this.getHexColor(cell, row, col);

                const isLegal = this.isLegalMove(row, col);
                const shouldShade = !this.heatmapEnabled && isEmpty && this.shouldShadeHex(row, col);
                const hex = this.makeHex(x, y, HEX_RADIUS, fill, isLegal, shouldShade);
                hex.setAttribute('data-row', row);
                hex.setAttribute('data-col', col);
                this.applyHeatmapClasses(hex, cell, row, col);

                if (isLegal && !this.isLoading) {
                    hex.classList.add('clickable');
                    // Use proper event handling like the dev version
                    const self = this;
                    hex.addEventListener('click', function (e) {
                        self.onCellClick(e);
                    });
                    hex.setAttribute('data-listener-attached', 'true');
                }

                this.svg.appendChild(hex);

                // Cache the hex element
                const key = `${row},${col}`;
                this.hexElements.set(key, hex);
            }
        }
    }

    drawEdgeIndicators(svgWidth, svgHeight, HEX_RADIUS, BOARD_SIZE) {
        const colors = this.getColors();

        // Blue edges (top and bottom) - adapted from working dev version
        const w = HEX_RADIUS * Math.sqrt(3);

        // Top edge - across the topmost hexes
        const topLine = this.makeEdgeLine(
            this.hexCenter(0, 0, HEX_RADIUS).x, this.hexCenter(0, 0, HEX_RADIUS).y - HEX_RADIUS,
            this.hexCenter(0, BOARD_SIZE - 1, HEX_RADIUS).x, this.hexCenter(0, BOARD_SIZE - 1, HEX_RADIUS).y - HEX_RADIUS,
            colors.VERY_DARK_BLUE,
            18
        );
        this.svg.appendChild(topLine);

        // Bottom edge - across the bottommost hexes
        const bottomLine = this.makeEdgeLine(
            this.hexCenter(BOARD_SIZE - 1, 0, HEX_RADIUS).x, this.hexCenter(BOARD_SIZE - 1, 0, HEX_RADIUS).y + HEX_RADIUS,
            this.hexCenter(BOARD_SIZE - 1, BOARD_SIZE - 1, HEX_RADIUS).x, this.hexCenter(BOARD_SIZE - 1, BOARD_SIZE - 1, HEX_RADIUS).y + HEX_RADIUS,
            colors.VERY_DARK_BLUE,
            18
        );
        this.svg.appendChild(bottomLine);

        // Red edges (left and right) - using proper edge midpoints
        const tl = this.hexVertices(this.hexCenter(0, 0, HEX_RADIUS).x, this.hexCenter(0, 0, HEX_RADIUS).y, HEX_RADIUS);
        const bl = this.hexVertices(this.hexCenter(BOARD_SIZE - 1, 0, HEX_RADIUS).x, this.hexCenter(BOARD_SIZE - 1, 0, HEX_RADIUS).y, HEX_RADIUS);
        const tr = this.hexVertices(this.hexCenter(0, BOARD_SIZE - 1, HEX_RADIUS).x, this.hexCenter(0, BOARD_SIZE - 1, HEX_RADIUS).y, HEX_RADIUS);
        const br = this.hexVertices(this.hexCenter(BOARD_SIZE - 1, BOARD_SIZE - 1, HEX_RADIUS).x, this.hexCenter(BOARD_SIZE - 1, BOARD_SIZE - 1, HEX_RADIUS).y, HEX_RADIUS);

        // Left edge - between vertices 2 and 3
        const leftTopMid = this.edgeMidpoint(tl[2], tl[3]);
        const leftBotMid = this.edgeMidpoint(bl[2], bl[3]);
        const leftLine = this.makeEdgeLine(leftTopMid.x, leftTopMid.y, leftBotMid.x, leftBotMid.y, colors.VERY_DARK_RED, 22);
        this.svg.appendChild(leftLine);

        // Right edge - between vertices 0 and 5
        const rightTopMid = this.edgeMidpoint(tr[0], tr[5]);
        const rightBotMid = this.edgeMidpoint(br[0], br[5]);
        const rightLine = this.makeEdgeLine(rightTopMid.x, rightTopMid.y, rightBotMid.x, rightBotMid.y, colors.VERY_DARK_RED, 22);
        this.svg.appendChild(rightLine);
    }

    hexCenter(row, col, HEX_RADIUS) {
        const padding = HEX_RADIUS * 0.5;
        const extraTopPadding = HEX_RADIUS * 0.5;
        const extraLeftPadding = HEX_RADIUS * 0.5;
        const x = HEX_RADIUS * Math.sqrt(3) * (col + row / 2) + HEX_RADIUS + padding + extraLeftPadding;
        const y = HEX_RADIUS * 1.5 * row + HEX_RADIUS + padding + extraTopPadding;
        return { x, y };
    }

    makeHex(cx, cy, r, fill, highlight, shouldShade = false) {
        const points = [];
        for (let i = 0; i < 6; i++) {
            const angle = Math.PI / 3 * i + Math.PI / 6;
            points.push([
                cx + r * Math.cos(angle),
                cy + r * Math.sin(angle)
            ]);
        }
        const hex = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        hex.setAttribute('points', points.map(p => p.join(',')).join(' '));
        hex.setAttribute('fill', fill);
        hex.setAttribute('stroke', '#ddd');
        hex.setAttribute('stroke-width', '1');

        // Apply purple shading if needed
        if (shouldShade) {
            hex.classList.add('purple-shaded');
        }

        if (highlight) {
            hex.style.cursor = 'pointer';
        }

        hex.addEventListener('mouseenter', (event) => {
            const row = parseInt(hex.getAttribute('data-row'), 10);
            const col = parseInt(hex.getAttribute('data-col'), 10);
            if (Number.isNaN(row) || Number.isNaN(col)) {
                return;
            }
            const text = this.getTooltipTextForHex(row, col);
            if (
                window.HexHeatmap &&
                window.HexHeatmap.tooltip &&
                typeof window.HexHeatmap.tooltip.show === 'function'
            ) {
                window.HexHeatmap.tooltip.show(event, text, { darkMode: this.darkMode });
            }
        });

        hex.addEventListener('mousemove', (event) => {
            if (
                window.HexHeatmap &&
                window.HexHeatmap.tooltip &&
                typeof window.HexHeatmap.tooltip.move === 'function'
            ) {
                window.HexHeatmap.tooltip.move(event);
            }
        });

        hex.addEventListener('mouseleave', () => {
            if (
                window.HexHeatmap &&
                window.HexHeatmap.tooltip &&
                typeof window.HexHeatmap.tooltip.hide === 'function'
            ) {
                window.HexHeatmap.tooltip.hide();
            }
        });
        return hex;
    }

    // Helper methods for edge indicators
    makeEdgeLine(x1, y1, x2, y2, color, width) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', color);
        line.setAttribute('stroke-width', width);
        return line;
    }

    edgeMidpoint(vA, vB) {
        return { x: (vA.x + vB.x) / 2, y: (vA.y + vB.y) / 2 };
    }

    hexVertices(cx, cy, r) {
        const vertices = [];
        for (let i = 0; i < 6; i++) {
            const angle = Math.PI / 3 * i + Math.PI / 6;
            vertices.push({
                x: cx + r * Math.cos(angle),
                y: cy + r * Math.sin(angle)
            });
        }
        return vertices;
    }

    // Proper click handler adapted from dev version
    // =============================================================================
    // USER INTERACTION
    // =============================================================================

    async onCellClick(e) {
        if (this.isLoading) {
            console.log('Loading, ignoring click');
            return;
        }

        const row = parseInt(e.target.getAttribute('data-row'));
        const col = parseInt(e.target.getAttribute('data-col'));

        // Clear purple shading immediately when a move is about to be made
        // This ensures the move color shows properly
        this.clearAllPurpleShading();

        // Hide instruction text since purple hexes are no longer relevant
        this.hideInstructionText();

        // Allow user clicks regardless of computer settings
        // Users can always make moves when they click

        this.setLoading(true);
        try {
            const move = this.rowColToTRMPH(row, col);
            const response = await fetch('/api/apply_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trmph: this.currentTRMPH,
                    move: move,
                    elo_rating: this.validateEloRating(),
                    display_board_size: this.validateBoardSize(),
                    pie_rule_enabled: this.getPieRuleRequestEnabled()
                })
            });

            const data = await response.json();

            if (data.error) {
                this.showError(data.error);
                return;
            }

            this.applyPieRuleStateFromResponse(data, false);

            this.currentTRMPH = data.new_trmph;
            this.gameHistory.push(this.currentTRMPH);
            this.moveCount++;
            this.legalMoves = data.legal_moves || [];

            // Clear redo history when new moves are made
            this.redoHistory = [];
            await this.renderBoardWithHeatmap(data.board);
            this.updateTrmphDisplay();
            this.updateButtonStates();

            // Auto-move for computer players
            if (data.winner) {
                this.showGameOver(data.winner);
            } else if (this.shouldMakeComputerMove(data.player)) {
                this.scheduleAutoMove();
            }
        } catch (error) {
            console.error('Move error:', error);
            this.showError('Move failed.');
        } finally {
            this.setLoading(false);
        }
    }

    isLegalMove(row, col) {
        // Check if this move is legal using the legal moves from the API
        if (!this.legalMoves) {
            console.log('No legal moves available');
            return false;
        }
        const move = this.rowColToTRMPH(row, col);
        const isLegal = this.legalMoves.includes(move);
        return isLegal;
    }


    handleBoardClick(e) {
        // Handle board-level clicks if needed
        // If we clicked on a hex, handle it
        if (e.target.tagName === 'polygon') {
            const row = parseInt(e.target.getAttribute('data-row'));
            const col = parseInt(e.target.getAttribute('data-col'));

            if (!isNaN(row) && !isNaN(col)) {
                // Create a synthetic event for the hex click
                const syntheticEvent = {
                    target: e.target,
                    preventDefault: () => e.preventDefault(),
                    stopPropagation: () => e.stopPropagation()
                };
                this.onCellClick(syntheticEvent);
            }
        }
    }


    getCurrentPlayer() {
        // Simple heuristic: even moves = blue, odd moves = red
        return this.moveCount % 2 === 0 ? 'blue' : 'red';
    }

    rowColToTRMPH(row, col) {
        // Convert row, col to TRMPH format
        const boardSize = this.validateBoardSize();
        const letters = 'abcdefghijklmnopqrstuvwxyz'.substring(0, boardSize);
        return letters[col] + (row + 1);
    }


    updateTrmphDisplay() {
        this.trmphDisplay.value = this.currentTRMPH;
        this.updatePieRuleUi();
    }


    showGameOver(winner) {
        this.computerMoveBtn.disabled = true;
        this.showSuccess(`${winner.toUpperCase()} wins!`);
    }

    setLoading(loading) {
        this.isLoading = loading;
        this.computerMoveBtn.disabled = loading;
        this.resetBtn.disabled = loading;
        this.undoBtn.disabled = loading;
        this.redoBtn.disabled = loading;

        // Don't apply loading class to body to avoid flickering
        // The loading state is handled by button states only
    }

    updateButtonStates() {
        // Update undo button state
        this.undoBtn.disabled = this.isLoading || this.gameHistory.length === 0;

        // Update redo button state
        this.redoBtn.disabled = this.isLoading || this.redoHistory.length === 0;
    }

    showError(message) {
        console.error('Game Error:', message);
        this.statusLine.textContent = `Error: ${message}`;
        this.statusLine.style.color = '#dc3545';
        this.statusLine.style.fontWeight = 'bold';
        setTimeout(() => {
            this.statusLine.style.color = '';
            this.statusLine.style.fontWeight = '';
        }, 5000); // Show errors longer
    }

    handleNetworkError(error, context) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            this.showError(`Network error: Unable to ${context}. Please check your connection.`);
        } else if (error.name === 'SyntaxError') {
            this.showError('Server error: Invalid response from server. Please try again.');
        } else {
            this.showError(`${context} failed. Please try again.`);
        }
    }

    showSuccess(message) {
        console.log('Game Success:', message);
        this.statusLine.textContent = message;
        this.statusLine.style.color = '#28a745';
        this.statusLine.style.fontWeight = 'bold';
        setTimeout(() => {
            this.statusLine.style.color = '';
            this.statusLine.style.fontWeight = '';
        }, 3000);
    }


    async redoMove() {
        if (this.redoHistory.length === 0) return;

        this.setLoading(true);
        try {
            // Save current TRMPH string (not object) to game history
            this.gameHistory.push(this.currentTRMPH);

            // Restore next state from redo history (already a string)
            const nextTrmph = this.redoHistory.pop();
            this.currentTRMPH = nextTrmph;

            // Recalculate moveCount from TRMPH string
            this.moveCount = this.currentTRMPH ?
                this.parseTrmphMoves(this.currentTRMPH).length : 0;

            // Clear cache for redo to ensure clean state
            this.previousBoard = null;
            this.hexElements.clear();

            await this.loadGameStateWithoutAutoMove('redo');

            // Show instruction text if we're back to an empty board
            if (this.isBoardEmpty(this.previousBoard)) {
                this.showInstructionText();
            }
        } catch (error) {
            console.error('Failed to redo move:', error);
            this.showError('Failed to redo move.');
        } finally {
            this.setLoading(false);
            this.updateButtonStates();
        }
    }


    copyTrmph() {
        this.trmphDisplay.select();
        this.trmphDisplay.setSelectionRange(0, 99999); // For mobile devices
        document.execCommand('copy');
        this.showSuccess('TRMPH sequence copied to clipboard');
    }


    // =============================================================================
    // VALIDATION AND UTILITIES
    // =============================================================================

    parseTrmphMoves(trmphString) {
        // Parse TRMPH string into individual moves (matches Python split_trmph_moves + strip_trmph_preamble)
        const boardSize = this.validateBoardSize();
        const letters = 'abcdefghijklmnopqrstuvwxyz'.substring(0, boardSize);

        // Strip preamble first (like Python strip_trmph_preamble)
        let bareMoves = trmphString;
        const preambleMatch = trmphString.match(/^#(\d+),/);
        if (preambleMatch) {
            bareMoves = trmphString.substring(preambleMatch[0].length);
        }

        // Split into moves (like Python split_trmph_moves)
        const moves = [];
        let i = 0;
        while (i < bareMoves.length) {
            if (!letters.includes(bareMoves[i])) {
                throw new Error(`Expected letter at position ${i} in ${bareMoves}`);
            }
            let j = i + 1;
            while (j < bareMoves.length && /\d/.test(bareMoves[j])) {
                j++;
            }
            moves.push(bareMoves.substring(i, j));
            i = j;
        }
        return moves;
    }

    cleanInput(input) {
        // Remove move numbers (e.g. "1.", "12.")
        let cleaned = input.replace(/\b\d+\./g, '');

        // Remove "swap" keyword (case insensitive)
        cleaned = cleaned.replace(/swap/gi, '');

        // Remove all non-alphanumeric characters
        cleaned = cleaned.replace(/[^a-zA-Z0-9]/g, '');

        return cleaned;
    }

    stripVirtualPrefillIfPresent(cleanedMoves) {
        const displaySize = this.validateBoardSize();
        const prefill = this.virtualBoardPrefillMoves[displaySize] || '';
        if (prefill && cleanedMoves.startsWith(prefill)) {
            return cleanedMoves.substring(prefill.length);
        }
        return cleanedMoves;
    }

    validateTrmphInput(input) {
        // Enhanced TRMPH validation with detailed error messages
        if (!input || typeof input !== 'string') {
            return { valid: false, error: 'Input must be a non-empty string' };
        }

        const trimmed = input.trim();
        if (!trimmed) {
            return { valid: false, error: 'Input cannot be empty or just whitespace' };
        }

        // Clean the input for validation purposes (remove numbers, swap, etc.)
        // We validate the *intent* (the moves), not the exact formatting
        const cleaned = this.cleanInput(trimmed);
        const cleanedForDisplayBoard = this.stripVirtualPrefillIfPresent(cleaned);

        if (!cleanedForDisplayBoard) {
            return { valid: true, error: null };
        }

        // Check move count by parsing the TRMPH string properly
        // This handles moves of varying length (a1, b13, m12, etc.)
        try {
            const moves = this.parseTrmphMoves(cleanedForDisplayBoard);
            const boardSize = this.validateBoardSize();
            const maxMoves = boardSize * boardSize; // Complete game moves
            if (moves.length > maxMoves) {
                return { valid: false, error: `[Frontend: Count] Too many moves (maximum ${maxMoves} moves for a complete game)` };
            }
        } catch (error) {
            return { valid: false, error: `[Frontend: Parse] Invalid TRMPH format: ${error.message}` };
        }

        // Validate TRMPH format dynamically based on board size
        const boardSize = this.validateBoardSize();
        const lastLetter = String.fromCharCode(96 + boardSize); // 'a' + boardSize - 1
        const lastNumber = boardSize;

        let numberPattern;
        if (boardSize <= 9) {
            numberPattern = `[1-${lastNumber}]`;
        } else {
            numberPattern = `(1[0-${lastNumber % 10}]|[1-9])`;
        }

        // Regex check on the CLEANED input
        // Note: We use case-insensitive flag 'i' here to allow "A1" to pass validation
        // The backend will normalize it to lowercase
        const trmphRegex = new RegExp(`^([a-${lastLetter}]${numberPattern})+$`, 'i');
        if (!trmphRegex.test(cleanedForDisplayBoard)) {
            return {
                valid: false,
                error: `[Frontend: Regex] Invalid format. Only letters a-${lastLetter} followed by numbers 1-${lastNumber} are allowed (e.g., a1b2c3)`
            };
        }

        return { valid: true, error: null };
    }

    async applyTrmphSequence() {
        const input = this.trmphInput.value.trim();

        if (!input) {
            this.showTrmphError('Please enter a TRMPH sequence');
            return;
        }

        const validation = this.validateTrmphInput(input);
        if (!validation.valid) {
            this.showTrmphError(validation.error);
            return;
        }

        // Clear purple shading immediately when applying a sequence
        // This ensures the new pieces show properly without purple overlay
        this.clearAllPurpleShading();

        // Hide instruction text since purple hexes are no longer relevant
        this.hideInstructionText();

        this.setLoading(true);
        try {
            const response = await fetch('/api/apply_trmph_sequence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trmph: this.currentTRMPH,
                    trmph_sequence: input,
                    elo_rating: this.validateEloRating(),
                    display_board_size: this.validateBoardSize(),
                    pie_rule_enabled: this.getPieRuleRequestEnabled()
                })
            });

            const data = await response.json();

            if (data.error) {
                this.showTrmphError(data.error);
                return;
            }

            this.applyPieRuleStateFromResponse(data, false);

            this.currentTRMPH = data.new_trmph;
            this.gameHistory.push(this.currentTRMPH);
            this.moveCount += data.moves_applied || 0;
            this.legalMoves = data.legal_moves || [];

            // Clear redo history when new moves are made
            this.redoHistory = [];
            this.updateTrmphDisplay();
            await this.renderBoardWithHeatmap(data.board);
            this.updateButtonStates();

            // Clear the input
            this.trmphInput.value = '';
            this.hideTrmphError();

            if (data.winner) {
                this.showGameOver(data.winner);
            } else if (this.shouldMakeComputerMove(data.player)) {
                this.scheduleAutoMove();
            }
        } catch (error) {
            console.error('TRMPH sequence error:', error);
            this.showTrmphError('Failed to apply TRMPH sequence');
        } finally {
            this.setLoading(false);
        }
    }

    showTrmphError(message) {
        this.trmphError.textContent = message;
        this.trmphError.style.display = 'block';
    }

    hideTrmphError() {
        this.trmphError.style.display = 'none';
    }
}

const COOKIE_CONSENT_COOKIE = 'sf25_cookie_consent';
const COOKIE_CONSENT_ACCEPTED = 'accepted';
const COOKIE_CONSENT_DECLINED = 'declined';
const COOKIE_CONSENT_DAYS = 180;

function getCookieValue(name) {
    const cookieParts = document.cookie ? document.cookie.split('; ') : [];
    for (const part of cookieParts) {
        const [key, ...valueParts] = part.split('=');
        if (decodeURIComponent(key) === name) {
            return decodeURIComponent(valueParts.join('='));
        }
    }
    return null;
}

function setCookieValue(name, value, days) {
    const maxAge = Math.max(1, Math.floor(days * 24 * 60 * 60));
    const secure = window.location.protocol === 'https:' ? '; Secure' : '';
    document.cookie =
        `${encodeURIComponent(name)}=${encodeURIComponent(value)}; Max-Age=${maxAge}; Path=/; SameSite=Lax${secure}`;
}

function hideCookieConsentBanner(banner) {
    if (!banner) {
        return;
    }
    banner.hidden = true;
}

function initCookieConsentBanner() {
    const banner = document.getElementById('cookie-consent-banner');
    const acceptBtn = document.getElementById('cookie-consent-accept');
    const declineBtn = document.getElementById('cookie-consent-decline');
    if (!banner || !acceptBtn || !declineBtn) {
        return;
    }

    const existingChoice = getCookieValue(COOKIE_CONSENT_COOKIE);
    if (existingChoice === COOKIE_CONSENT_ACCEPTED || existingChoice === COOKIE_CONSENT_DECLINED) {
        hideCookieConsentBanner(banner);
        return;
    }

    banner.hidden = false;

    acceptBtn.addEventListener('click', () => {
        setCookieValue(COOKIE_CONSENT_COOKIE, COOKIE_CONSENT_ACCEPTED, COOKIE_CONSENT_DAYS);
        hideCookieConsentBanner(banner);
    });

    declineBtn.addEventListener('click', () => {
        setCookieValue(COOKIE_CONSENT_COOKIE, COOKIE_CONSENT_DECLINED, COOKIE_CONSENT_DAYS);
        hideCookieConsentBanner(banner);
    });
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initCookieConsentBanner();
    new HexGame();
});
