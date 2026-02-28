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
        this.maxHistoryEntries = 5000;
        this.autoMoveTimeoutId = null;
        this.moveCount = 0;
        this.currentElo = null;  // Must be set from backend - fail fast if not
        this.blueComputer = false;
        this.redComputer = true;
        this.isLoading = false;
        this.isInitialLoad = true; // Track if this is the initial page load
        this.darkMode = false; // Dark mode state
        this.colorScheme = 'wood';
        this.pieceStyle = 'disc';
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
        this.sessionStateStorageKey = 'hex_ai_session_state_v1';
        this.sessionStateVersion = 1;
        this.historyUtils = window.HexHistoryUtils;
        if (!this.historyUtils) {
            throw new Error('HexHistoryUtils is required but was not loaded');
        }

        // Track previous board state for efficient updates
        this.previousBoard = null;
        this.hexElements = new Map(); // Cache hex elements by position
        this.pieceElements = new Map(); // Cache disc-style piece elements by position
        this.themeColors = null;

        // Difficulty levels will be loaded from backend in loadGameConstants()
        this.difficultyLevels = null;
        this.trmphBoardShareBaseUrl = null;
        this.virtualBoardPrefillMoves = {};
        this.openingGuideWeakThreshold = 0.40;
        this.openingGuideStrongThreshold = 0.55;
        this.openingGuideStatusByMove = new Map();
        this.openingGuideMarkerElements = new Map();
        this.openingGuideEnabled = false;
        this.openingGuideStorageKey = 'hex_ai_opening_guide_enabled';

        this.initializeElements();
        this.defaultInstructionText = this.instructionText ? this.instructionText.textContent : '';
        this.setupEventListeners();
        this.initializeDarkMode();
        this.initializeColorScheme();
        this.initializePieceStyle();
        this.initializeOpeningGuidePreference();
        this.refreshThemeColors();
        this.loadGameConstants();
        this.updateHeatmapControls();
        this.updatePieRuleUi();
        this.installDevtoolsThemeApi();
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
        this.bluePlayerLabel = document.getElementById('blue-player-label');
        this.redPlayerLabel = document.getElementById('red-player-label');
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
        this.heatmapLegend = document.getElementById('heatmap-legend');
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
            localStorage.setItem('hex_ai_preferred_elo', String(this.currentElo));
            this.persistSessionState();
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

        this.eloSlider.addEventListener('change', () => {
            localStorage.setItem('hex_ai_preferred_elo', String(this.currentElo));
            this.persistSessionState();
        });

        this.blueComputerCheck.addEventListener('change', (e) => {
            this.blueComputer = e.target.checked;
            this.updatePieRuleUi();
            this.persistSessionState();
        });

        this.redComputerCheck.addEventListener('change', (e) => {
            this.redComputer = e.target.checked;
            this.updatePieRuleUi();
            this.persistSessionState();
        });

        if (this.pieRuleEnabledCheck) {
            this.pieRuleEnabledCheck.addEventListener('change', async (e) => {
                this.pieRuleEnabled = e.target.checked;
                localStorage.setItem('hex_ai_pie_rule_enabled', String(this.pieRuleEnabled));
                this.updatePieRuleUi();
                this.persistSessionState();
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

        window.addEventListener('pagehide', () => {
            this.persistSessionState();
        });
    }

    // =============================================================================
    // DARK MODE FUNCTIONS
    // =============================================================================

    toggleDarkMode() {
        this.darkMode = !this.darkMode;
        this.applyDarkModeAttribute();
        this.refreshThemeColors();

        // Update the toggle button text and icon
        if (this.darkMode) {
            this.darkModeToggle.textContent = '☀️ Light';
            this.darkModeToggle.title = 'Switch to light mode';
        } else {
            this.darkModeToggle.textContent = '🌙 Dark';
            this.darkModeToggle.title = 'Switch to dark mode';
        }

        // Redraw the board with new colors
        if (this.svg && Array.isArray(this.previousBoard)) {
            this.clearBoard();
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
        this.applyDarkModeAttribute();
        this.refreshThemeColors();

        // Update the toggle button
        if (this.darkMode) {
            this.darkModeToggle.textContent = '☀️ Light';
            this.darkModeToggle.title = 'Switch to light mode';
        } else {
            this.darkModeToggle.textContent = '🌙 Dark';
            this.darkModeToggle.title = 'Switch to dark mode';
        }
    }

    applyDarkModeAttribute() {
        if (this.darkMode) {
            document.documentElement.setAttribute('data-theme', 'dark');
            return;
        }
        document.documentElement.removeAttribute('data-theme');
    }

    normalizeColorScheme(rawScheme) {
        if (rawScheme === 'classic' || rawScheme === 'classic_blue_first' || rawScheme === 'wood') {
            return rawScheme;
        }
        // Migrate legacy placeholder value to the new default.
        if (rawScheme === 'default') {
            return 'wood';
        }
        return 'wood';
    }

    initializeColorScheme() {
        const storedScheme = localStorage.getItem('hex_ai_color_scheme');
        this.colorScheme = this.normalizeColorScheme(storedScheme);
        document.documentElement.setAttribute('data-color-scheme', this.colorScheme);
        this.refreshThemeColors();
        this.updatePlayerTerminologyUi();
        this.updateHeatmapLegendText();

        if (storedScheme !== this.colorScheme) {
            localStorage.setItem('hex_ai_color_scheme', this.colorScheme);
        }
    }

    normalizePieceStyle(rawStyle) {
        return rawStyle === 'hex_fill' ? 'hex_fill' : 'disc';
    }

    initializePieceStyle() {
        const storedPieceStyle = localStorage.getItem('hex_ai_piece_style');
        this.pieceStyle = this.normalizePieceStyle(storedPieceStyle);
        document.documentElement.setAttribute('data-piece-style', this.pieceStyle);

        if (storedPieceStyle !== this.pieceStyle) {
            localStorage.setItem('hex_ai_piece_style', this.pieceStyle);
        }
    }

    initializeOpeningGuidePreference() {
        const storedValue = localStorage.getItem(this.openingGuideStorageKey);
        if (storedValue === 'true' || storedValue === 'false') {
            this.openingGuideEnabled = storedValue === 'true';
            return;
        }
        this.openingGuideEnabled = true;
        localStorage.setItem(this.openingGuideStorageKey, 'false');
    }

    isDiscPieceStyle() {
        return this.normalizePieceStyle(this.pieceStyle) === 'disc';
    }

    getDisplayColorKeyForInternalSide(side) {
        const normalizedScheme = this.normalizeColorScheme(this.colorScheme);
        if (normalizedScheme === 'classic') {
            if (side === 'blue') {
                return 'red';
            }
            if (side === 'red') {
                return 'blue';
            }
        }
        return side === 'red' ? 'red' : 'blue';
    }

    getPieRulePlayerClassName(side) {
        const displayColor = this.getDisplayColorKeyForInternalSide(side);
        return displayColor === 'red' ? 'pie-rule-player-red' : 'pie-rule-player-blue';
    }

    getPlayerDisplayNames() {
        if (this.normalizeColorScheme(this.colorScheme) === 'wood') {
            return { blue: 'Black', red: 'White' };
        }
        const blueName = this.getDisplayColorKeyForInternalSide('blue') === 'red' ? 'Red' : 'Blue';
        const redName = this.getDisplayColorKeyForInternalSide('red') === 'red' ? 'Red' : 'Blue';
        return { blue: blueName, red: redName };
    }

    getPlayerDisplayName(side) {
        const names = this.getPlayerDisplayNames();
        return side === 'blue' ? names.blue : names.red;
    }

    updatePlayerTerminologyUi() {
        const names = this.getPlayerDisplayNames();
        if (this.bluePlayerLabel) {
            this.bluePlayerLabel.textContent = names.blue;
        }
        if (this.redPlayerLabel) {
            this.redPlayerLabel.textContent = names.red;
        }
    }

    updateHeatmapLegendText() {
        if (!this.heatmapLegend) {
            return;
        }
        if (this.normalizeColorScheme(this.colorScheme) === 'wood') {
            this.heatmapLegend.textContent = 'Dashed red = below 50%, solid green = above 50%, dotted slate = near-even';
            return;
        }
        this.heatmapLegend.textContent = 'Dashed amber = below 50%, solid green = above 50%, dotted slate = near-even';
    }

    // =============================================================================
    // COLOR TOKENS
    // =============================================================================

    refreshThemeColors() {
        const rootStyle = getComputedStyle(document.documentElement);
        const readToken = (name, fallback = '') => {
            const raw = rootStyle.getPropertyValue(name);
            const trimmed = typeof raw === 'string' ? raw.trim() : '';
            return trimmed || fallback;
        };
        const blueDisplayColor = this.getDisplayColorKeyForInternalSide('blue');
        const redDisplayColor = this.getDisplayColorKeyForInternalSide('red');
        const readPieceToken = (displayColor, variant, fallback = '') => {
            return readToken(`--board-piece-${displayColor}-${variant}`, fallback);
        };
        const readEdgeToken = (displayColor, fallback = '') => {
            return readToken(`--board-edge-${displayColor}`, fallback);
        };
        const fillFallbackForDisplayColor = (displayColor) => {
            return displayColor === 'red' ? '#ff4444' : '#0099ff';
        };
        const strokeFallbackForDisplayColor = (displayColor) => {
            return displayColor === 'red' ? '#952500' : '#0064af';
        };
        this.themeColors = {
            EMPTY_HEX_GRAY: readToken('--board-cell-empty', '#f0f0f0'),
            BOARD_BACKGROUND: readToken('--board-bg', '#f8f8fa'),
            DARK_BLUE: readPieceToken(blueDisplayColor, 'fill', fillFallbackForDisplayColor(blueDisplayColor)),
            DARK_RED: readPieceToken(redDisplayColor, 'fill', fillFallbackForDisplayColor(redDisplayColor)),
            VERY_DARK_BLUE: readEdgeToken(blueDisplayColor, fillFallbackForDisplayColor(blueDisplayColor)),
            VERY_DARK_RED: readEdgeToken(redDisplayColor, fillFallbackForDisplayColor(redDisplayColor)),
            DISC_BLUE_STROKE: readPieceToken(blueDisplayColor, 'stroke', strokeFallbackForDisplayColor(blueDisplayColor)),
            DISC_RED_STROKE: readPieceToken(redDisplayColor, 'stroke', strokeFallbackForDisplayColor(redDisplayColor)),
            HEX_STROKE: readToken('--board-cell-stroke', '#555555'),
            STATUS_ERROR: readToken('--status-error-text', '#dc3545'),
            STATUS_SUCCESS: readToken('--status-success-text', '#28a745'),
        };
        return this.themeColors;
    }

    getColors() {
        return this.themeColors || this.refreshThemeColors();
    }

    installDevtoolsThemeApi() {
        if (typeof window === 'undefined') {
            return;
        }
        const self = this;
        let liveTimerId = null;
        // DevTools convenience hooks for rapid theme iteration.
        window.hexGame = this;
        const themeApi = {
            get(name) {
                const value = getComputedStyle(document.documentElement).getPropertyValue(name);
                return typeof value === 'string' ? value.trim() : '';
            },
            set(name, value, redraw = true) {
                document.documentElement.style.setProperty(name, value);
                self.refreshThemeColors();
                if (redraw) {
                    self.redrawCurrentBoard();
                }
                return this.get(name);
            },
            reset(name, redraw = true) {
                document.documentElement.style.removeProperty(name);
                self.refreshThemeColors();
                if (redraw) {
                    self.redrawCurrentBoard();
                }
                return this.get(name);
            },
            redraw() {
                self.refreshThemeColors();
                self.redrawCurrentBoard();
            },
            live(intervalMs = 90) {
                const numeric = Number(intervalMs);
                const ms = Number.isFinite(numeric) ? Math.max(16, Math.floor(numeric)) : 90;
                this.stopLive();
                liveTimerId = window.setInterval(() => {
                    self.refreshThemeColors();
                    self.redrawCurrentBoard();
                }, ms);
                return liveTimerId;
            },
            stopLive() {
                if (liveTimerId === null) {
                    return false;
                }
                window.clearInterval(liveTimerId);
                liveTimerId = null;
                return true;
            },
            isLive() {
                return liveTimerId !== null;
            },
            dump(prefix = '--') {
                const styles = getComputedStyle(document.documentElement);
                const tokens = {};
                for (let i = 0; i < styles.length; i++) {
                    const name = styles[i];
                    if (typeof name === 'string' && name.startsWith(prefix)) {
                        tokens[name] = styles.getPropertyValue(name).trim();
                    }
                }
                return tokens;
            }
        };
        const pieTokenNames = [
            '--pie-badge-bg',
            '--pie-badge-border',
            '--pie-player-bg',
            '--pie-player-border',
            '--pie-player-title-color',
            '--pie-player-red',
            '--pie-player-red-shadow',
            '--pie-player-blue',
            '--pie-player-both',
            '--pie-player-none',
            '--pie-label-bg',
            '--pie-label-border',
            '--pie-label-shadow',
            '--pie-link-sub',
        ];
        themeApi.pie = {
            tokens: [...pieTokenNames],
            dump() {
                const values = {};
                for (const name of pieTokenNames) {
                    values[name] = themeApi.get(name);
                }
                return values;
            },
            set(name, value, redraw = false) {
                if (!pieTokenNames.includes(name)) {
                    throw new Error(`Unknown pie token: ${name}`);
                }
                return themeApi.set(name, value, redraw);
            },
            reset(name = null, redraw = false) {
                if (name === null) {
                    for (const tokenName of pieTokenNames) {
                        themeApi.reset(tokenName, false);
                    }
                    if (redraw) {
                        themeApi.redraw();
                    }
                    return this.dump();
                }
                if (!pieTokenNames.includes(name)) {
                    throw new Error(`Unknown pie token: ${name}`);
                }
                themeApi.reset(name, redraw);
                return this.dump();
            },
            apply(overrides, redraw = false) {
                if (!overrides || typeof overrides !== 'object') {
                    throw new Error('pie.apply(overrides) expects an object map of token values');
                }
                for (const [name, value] of Object.entries(overrides)) {
                    this.set(name, value, false);
                }
                if (redraw) {
                    themeApi.redraw();
                }
                return this.dump();
            }
        };
        window.hexTheme = themeApi;
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
                colorScheme: this.normalizeColorScheme(this.colorScheme),
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
        const names = this.getPlayerDisplayNames();
        if (!this.blueComputer && !this.redComputer) {
            return { label: 'Both', className: 'pie-rule-player-both' };
        }
        if (this.blueComputer && this.redComputer) {
            return { label: 'None', className: 'pie-rule-player-none' };
        }

        if (!this.blueComputer && this.redComputer) {
            return { label: names.blue, className: this.getPieRulePlayerClassName('blue') };
        }
        if (this.blueComputer && !this.redComputer) {
            return { label: names.red, className: this.getPieRulePlayerClassName('red') };
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
            ? { label: names.blue, className: this.getPieRulePlayerClassName('blue') }
            : { label: names.red, className: this.getPieRulePlayerClassName('red') };
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

        this.syncPlayerToggleInputs();
    }

    syncPlayerToggleInputs() {
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

    parseOpeningGuideMove(move, boardSize) {
        const normalized = typeof move === 'string' ? move.trim().toLowerCase() : '';
        const match = normalized.match(/^([a-z])([1-9][0-9]*)$/);
        if (!match) {
            throw new Error(`Opening win-rate move key has invalid format: ${move}`);
        }
        const col = match[1].charCodeAt(0) - 97;
        const row = parseInt(match[2], 10) - 1;
        if (!Number.isFinite(row) || row < 0 || row >= boardSize || col < 0 || col >= boardSize) {
            throw new Error(`Opening win-rate move key is out of board range: ${move}`);
        }
        return { row, col };
    }

    openingGuideMoveFromRowCol(row, col, boardSize) {
        if (boardSize > 26) {
            throw new Error(`Unsupported board size for opening guide coordinates: ${boardSize}`);
        }
        if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) {
            throw new Error(`Opening guide row/col out of range: row=${row}, col=${col}, boardSize=${boardSize}`);
        }
        return `${String.fromCharCode(97 + col)}${row + 1}`;
    }

    getSymmetricOpeningGuideMove(move, boardSize) {
        const { row, col } = this.parseOpeningGuideMove(move, boardSize);
        const mirroredRow = boardSize - 1 - row;
        const mirroredCol = boardSize - 1 - col;
        return this.openingGuideMoveFromRowCol(mirroredRow, mirroredCol, boardSize);
    }

    configureOpeningGuideStatuses(openingWinRates) {
        if (!openingWinRates || typeof openingWinRates !== 'object' || Array.isArray(openingWinRates)) {
            throw new Error('PIE_RULE_VALUE_BALANCED_OPENING_WIN_RATES_13X13 is missing from backend constants');
        }
        if (!Number.isFinite(this.boardSize) || this.boardSize < 2) {
            throw new Error('BOARD_SIZE is not initialized before opening-guide setup');
        }
        const boardSize = this.boardSize;
        const rawWinRatesByMove = new Map();
        for (const [moveRaw, winRateRaw] of Object.entries(openingWinRates)) {
            const move = typeof moveRaw === 'string' ? moveRaw.trim().toLowerCase() : '';
            if (!move) {
                throw new Error('Opening win-rate map includes an invalid move key');
            }
            this.parseOpeningGuideMove(move, boardSize);
            const winRate = Number(winRateRaw);
            if (!Number.isFinite(winRate) || winRate < 0 || winRate > 1) {
                throw new Error(`Opening win rate for ${move} is invalid: ${winRateRaw}`);
            }
            rawWinRatesByMove.set(move, winRate);
        }

        const symmetrizedWinRatesByMove = new Map();
        for (const [move, winRate] of rawWinRatesByMove.entries()) {
            const symmetricMove = this.getSymmetricOpeningGuideMove(move, boardSize);
            const symmetricWinRate = rawWinRatesByMove.get(symmetricMove);
            if (!Number.isFinite(symmetricWinRate)) {
                throw new Error(
                    `Opening win-rate map is missing symmetric counterpart: ${move} <-> ${symmetricMove}`
                );
            }
            symmetrizedWinRatesByMove.set(move, (winRate + symmetricWinRate) / 2);
        }

        const statusesByMove = new Map();
        for (const [move, winRate] of symmetrizedWinRatesByMove.entries()) {
            if (winRate < this.openingGuideWeakThreshold) {
                statusesByMove.set(move, 'weak');
            } else if (winRate > this.openingGuideStrongThreshold) {
                statusesByMove.set(move, 'strong');
            } else {
                statusesByMove.set(move, 'balanced');
            }
        }
        this.openingGuideStatusByMove = statusesByMove;
    }

    updateInstructionTextForBoardMode() {
        if (!this.instructionText) {
            return;
        }
        if (this.validateBoardSize() !== this.boardSize) {
            const size = this.validateBoardSize();
            this.instructionText.textContent = `Changed board size to ${size}`;
            this.instructionText.style.display = 'block';
            return;
        }
        this.instructionText.textContent = this.defaultInstructionText;
        this.instructionText.style.display = this.openingGuideEnabled ? 'block' : 'none';
    }

    getSessionStorage() {
        try {
            return window.sessionStorage;
        } catch (_error) {
            return null;
        }
    }

    clearPersistedSessionState() {
        const storage = this.getSessionStorage();
        if (!storage) {
            return;
        }
        try {
            storage.removeItem(this.sessionStateStorageKey);
        } catch (_error) {
            // Ignore browser storage errors; game state remains in memory.
        }
    }

    sanitizeSessionStateArray(rawValue) {
        if (!Array.isArray(rawValue)) {
            return [];
        }
        return rawValue
            .filter((entry) => typeof entry === 'string')
            .map((entry) => entry.trim())
            .filter((entry) => entry.length > 0);
    }

    dedupeConsecutiveStates(states) {
        return this.historyUtils.dedupeConsecutive(states, { getKey: (entry) => entry });
    }

    normalizeHistoryStacks() {
        const keyOptions = { getKey: (entry) => entry };
        this.currentTRMPH = typeof this.currentTRMPH === 'string' ? this.currentTRMPH.trim() : '';
        this.gameHistory = this.historyUtils.dedupeConsecutive(this.sanitizeSessionStateArray(this.gameHistory), keyOptions);
        this.redoHistory = this.historyUtils.dedupeConsecutive(this.sanitizeSessionStateArray(this.redoHistory), keyOptions);

        if (this.currentTRMPH) {
            const hasCurrent = this.historyUtils.truncateAfterLastMatch(this.gameHistory, this.currentTRMPH, keyOptions);
            if (!hasCurrent) {
                this.historyUtils.pushDistinct(this.gameHistory, this.currentTRMPH, {
                    ...keyOptions,
                    maxEntries: this.maxHistoryEntries,
                });
            }
        } else {
            this.gameHistory = [];
        }

        if (this.gameHistory.length > this.maxHistoryEntries) {
            this.gameHistory = this.gameHistory.slice(-this.maxHistoryEntries);
        }
    }

    recomputeMoveCountFromCurrentTrmph() {
        if (!this.currentTRMPH) {
            this.moveCount = 0;
            this.pieRuleArmed = true;
            return;
        }
        try {
            this.moveCount = this.parseTrmphMoves(this.currentTRMPH).length;
        } catch (_error) {
            this.moveCount = Number.isFinite(this.moveCount) ? this.moveCount : 0;
        }
        if (this.moveCount === 0) {
            this.pieRuleArmed = true;
        }
    }

    recordReachedState(newTrmph, clearRedo = true) {
        const keyOptions = { getKey: (entry) => entry };
        this.currentTRMPH = typeof newTrmph === 'string' ? newTrmph.trim() : '';
        if (this.currentTRMPH) {
            this.historyUtils.pushDistinct(this.gameHistory, this.currentTRMPH, {
                ...keyOptions,
                maxEntries: this.maxHistoryEntries,
            });
        } else {
            this.gameHistory = [];
        }
        if (clearRedo) {
            this.redoHistory = [];
        }
        this.normalizeHistoryStacks();
        this.recomputeMoveCountFromCurrentTrmph();
    }

    undoHistoryStep() {
        const keyOptions = { getKey: (entry) => entry };
        this.normalizeHistoryStacks();
        if (this.gameHistory.length === 0) {
            return false;
        }
        const currentState = this.gameHistory.pop();
        if (currentState) {
            this.historyUtils.pushDistinct(this.redoHistory, currentState, keyOptions);
        }
        this.currentTRMPH = this.gameHistory.length > 0 ? this.gameHistory[this.gameHistory.length - 1] : '';
        this.normalizeHistoryStacks();
        this.recomputeMoveCountFromCurrentTrmph();
        return true;
    }

    redoHistoryStep() {
        const keyOptions = { getKey: (entry) => entry };
        this.normalizeHistoryStacks();
        if (this.redoHistory.length === 0) {
            return false;
        }
        const nextState = this.historyUtils.popLastDistinct(this.redoHistory, this.currentTRMPH, keyOptions);
        if (nextState === null) {
            return false;
        }
        this.currentTRMPH = nextState;
        this.historyUtils.pushDistinct(this.gameHistory, nextState, {
            ...keyOptions,
            maxEntries: this.maxHistoryEntries,
        });
        this.normalizeHistoryStacks();
        this.recomputeMoveCountFromCurrentTrmph();
        return true;
    }

    clearCachedBoardRenderingState() {
        this.previousBoard = null;
        this.hexElements.clear();
        this.pieceElements.clear();
        this.openingGuideMarkerElements.clear();
    }

    cancelScheduledAutoMove() {
        if (this.autoMoveTimeoutId !== null) {
            clearTimeout(this.autoMoveTimeoutId);
            this.autoMoveTimeoutId = null;
        }
    }

    restoreSessionState(validBoardSizes, minElo, maxElo) {
        const storage = this.getSessionStorage();
        if (!storage) {
            return false;
        }
        try {
            const rawSnapshot = storage.getItem(this.sessionStateStorageKey);
            if (!rawSnapshot) {
                return false;
            }
            const snapshot = JSON.parse(rawSnapshot);
            if (!snapshot || typeof snapshot !== 'object' || snapshot.version !== this.sessionStateVersion) {
                this.clearPersistedSessionState();
                return false;
            }

            const restoredTrmph = typeof snapshot.currentTRMPH === 'string' ? snapshot.currentTRMPH.trim() : '';
            const restoredBoardSize = parseInt(snapshot.displayBoardSize, 10);
            const restoredElo = parseInt(snapshot.currentElo, 10);

            if (Number.isFinite(restoredBoardSize) && validBoardSizes.has(restoredBoardSize)) {
                this.displayBoardSize = restoredBoardSize;
            }
            if (Number.isFinite(restoredElo) && restoredElo >= minElo && restoredElo <= maxElo) {
                this.currentElo = restoredElo;
            }
            if (typeof snapshot.blueComputer === 'boolean') {
                this.blueComputer = snapshot.blueComputer;
            }
            if (typeof snapshot.redComputer === 'boolean') {
                this.redComputer = snapshot.redComputer;
            }
            if (typeof snapshot.pieRuleEnabled === 'boolean') {
                this.pieRuleEnabled = snapshot.pieRuleEnabled;
            }
            if (typeof snapshot.pieRuleArmed === 'boolean') {
                this.pieRuleArmed = snapshot.pieRuleArmed;
            }

            this.currentTRMPH = restoredTrmph;

            let restoredHistory = this.sanitizeSessionStateArray(snapshot.gameHistory);
            if (!restoredTrmph) {
                restoredHistory = [];
            } else if (restoredHistory.length === 0 || restoredHistory[restoredHistory.length - 1] !== restoredTrmph) {
                restoredHistory.push(restoredTrmph);
            }
            this.gameHistory = restoredHistory;
            this.redoHistory = this.sanitizeSessionStateArray(snapshot.redoHistory);
            this.normalizeHistoryStacks();
            this.recomputeMoveCountFromCurrentTrmph();

            return true;
        } catch (error) {
            console.warn('Ignoring invalid session snapshot:', error);
            this.clearPersistedSessionState();
            return false;
        }
    }

    persistSessionState() {
        const storage = this.getSessionStorage();
        if (!storage) {
            return;
        }

        this.normalizeHistoryStacks();
        this.recomputeMoveCountFromCurrentTrmph();

        const snapshot = {
            version: this.sessionStateVersion,
            updated_at_ms: Date.now(),
            currentTRMPH: this.currentTRMPH,
            gameHistory: [...this.gameHistory],
            redoHistory: [...this.redoHistory],
            moveCount: this.moveCount,
            displayBoardSize: this.displayBoardSize,
            currentElo: this.currentElo,
            blueComputer: this.blueComputer,
            redComputer: this.redComputer,
            pieRuleEnabled: this.pieRuleEnabled,
            pieRuleArmed: this.pieRuleArmed,
        };

        try {
            storage.setItem(this.sessionStateStorageKey, JSON.stringify(snapshot));
        } catch (error) {
            console.warn('Failed to persist session snapshot:', error);
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
            if (typeof data.TRMPH_BOARD_SHARE_BASE_URL !== 'string' || !data.TRMPH_BOARD_SHARE_BASE_URL.trim()) {
                throw new Error('TRMPH_BOARD_SHARE_BASE_URL is missing from backend constants');
            }
            this.trmphBoardShareBaseUrl = data.TRMPH_BOARD_SHARE_BASE_URL.trim();

            if (!data.VIRTUAL_BOARD_PREFILL_MOVES || typeof data.VIRTUAL_BOARD_PREFILL_MOVES !== 'object') {
                throw new Error('VIRTUAL_BOARD_PREFILL_MOVES is missing from backend constants');
            }
            this.virtualBoardPrefillMoves = Object.fromEntries(
                Object.entries(data.VIRTUAL_BOARD_PREFILL_MOVES).map(([size, moves]) => {
                    const parsedSize = parseInt(size, 10);
                    if (!Number.isFinite(parsedSize)) {
                        throw new Error(`Invalid display board size in VIRTUAL_BOARD_PREFILL_MOVES: ${size}`);
                    }
                    return [parsedSize, typeof moves === 'string' ? moves : ''];
                })
            );
            this.configureOpeningGuideStatuses(data.PIE_RULE_VALUE_BALANCED_OPENING_WIN_RATES_13X13);
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
            const backendDefaultElo = data.ELO_CONFIG.DEFAULT_ELO;
            const savedElo = parseInt(localStorage.getItem('hex_ai_preferred_elo'), 10);
            this.currentElo = Number.isFinite(savedElo) && savedElo >= this.minElo && savedElo <= this.maxElo
                ? savedElo
                : backendDefaultElo;
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
            const restoredSession = this.restoreSessionState(validSizes, this.minElo, this.maxElo);
            this.syncPlayerToggleInputs();
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
            const initialStateReason = restoredSession ? 'session_restore' : 'initial_load';
            const shouldAutoMove = !restoredSession;
            await this.loadGameState(shouldAutoMove, initialStateReason);
            this.isInitialLoad = false; // Mark initial load as complete
            this.persistSessionState();
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
        this.cancelScheduledAutoMove();
        this.setLoading(true);
        try {
            this.currentTRMPH = "";
            this.gameHistory = [];
            this.redoHistory = [];
            this.moveCount = 0;
            this.pieRuleArmed = true;
            this.isInitialLoad = false; // Mark that this is no longer initial load
            this.clearCachedBoardRenderingState();
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
            this.persistSessionState();
        }
    }

    async undoMove() {
        if (this.isLoading || this.gameHistory.length === 0) return;

        this.cancelScheduledAutoMove();
        this.setLoading(true);
        try {
            if (!this.undoHistoryStep()) {
                return;
            }

            // Clear cache for undo to ensure clean state
            this.clearCachedBoardRenderingState();

            await this.loadGameStateWithoutAutoMove('undo');

            // Show instruction text if we're back to an empty board
            if (this.isBoardEmpty(this.previousBoard)) {
                this.showInstructionText();
            }
            this.persistSessionState();
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
        this.autoMoveTimeoutId = null;

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

                const moveWasPlayed = Boolean(data.move_made);
                if (moveWasPlayed) {
                    this.recordReachedState(data.new_trmph, true);
                } else {
                    this.currentTRMPH = typeof data.new_trmph === 'string' ? data.new_trmph.trim() : this.currentTRMPH;
                    this.normalizeHistoryStacks();
                    this.recomputeMoveCountFromCurrentTrmph();
                }
                this.legalMoves = data.legal_moves || [];

                await this.renderBoardWithHeatmap(data.board, data.move_heatmap || null);
                this.updateTrmphDisplay();
                this.updateButtonStates();
                this.persistSessionState();

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
        this.cancelScheduledAutoMove();
        this.autoMoveTimeoutId = setTimeout(() => {
            this.autoMoveTimeoutId = null;
            void this.makeComputerMove();
        }, 500);
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
            this.persistSessionState();

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
        this.pieceElements.clear();
        this.openingGuideMarkerElements.clear();
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
        this.syncOpeningGuideMarkers(newBoard, 18);

        // Update previous board state
        this.previousBoard = newBoard.map(row => [...row]); // Deep copy
    }

    updateHex(row, col, newValue) {
        this.refreshThemeColors();
        const key = `${row},${col}`;
        let hexElement = this.hexElements.get(key);
        const { x, y } = this.hexCenter(row, col, 18);

        if (!hexElement) {
            // Create new hex if it doesn't exist (shouldn't happen in normal flow)
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

        this.updatePieceDisc(row, col, newValue, x, y, 18);
    }

    getHexColor(cellValue, row = null, col = null) {
        const colors = this.getColors();
        if (cellValue === this.pieceValues.BLUE) {
            return this.isDiscPieceStyle() ? colors.EMPTY_HEX_GRAY : colors.DARK_BLUE;
        }
        if (cellValue === this.pieceValues.RED) {
            return this.isDiscPieceStyle() ? colors.EMPTY_HEX_GRAY : colors.DARK_RED;
        }
        if (row !== null && col !== null) {
            const score = this.getHeatmapScoreForMove(row, col);
            if (Number.isFinite(score)) {
                return this.getHeatmapFillColor(score);
            }
        }
        return colors.EMPTY_HEX_GRAY; // Empty hex color
    }

    getDiscStyle(cellValue) {
        const colors = this.getColors();

        if (cellValue === this.pieceValues.BLUE) {
            return {
                fill: colors.DARK_BLUE,
                stroke: colors.DISC_BLUE_STROKE,
            };
        }
        if (cellValue === this.pieceValues.RED) {
            return {
                fill: colors.DARK_RED,
                stroke: colors.DISC_RED_STROKE,
            };
        }
        return null;
    }

    makePieceDisc(cx, cy, r, style) {
        const disc = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        disc.setAttribute('cx', String(cx));
        disc.setAttribute('cy', String(cy));
        disc.setAttribute('r', String(r * 0.62));
        disc.setAttribute('fill', style.fill);
        disc.setAttribute('stroke', style.stroke);
        disc.setAttribute('stroke-width', '1.6');
        disc.style.pointerEvents = 'none';
        return disc;
    }

    updatePieceDisc(row, col, cellValue, cx, cy, hexRadius) {
        const key = `${row},${col}`;
        const existingDisc = this.pieceElements.get(key);

        if (!this.isDiscPieceStyle() || cellValue === this.pieceValues.EMPTY) {
            if (existingDisc) {
                existingDisc.remove();
                this.pieceElements.delete(key);
            }
            return;
        }

        const discStyle = this.getDiscStyle(cellValue);
        if (!discStyle) {
            return;
        }

        if (!existingDisc) {
            const created = this.makePieceDisc(cx, cy, hexRadius, discStyle);
            this.svg.appendChild(created);
            this.pieceElements.set(key, created);
            return;
        }

        existingDisc.setAttribute('cx', String(cx));
        existingDisc.setAttribute('cy', String(cy));
        existingDisc.setAttribute('r', String(hexRadius * 0.62));
        existingDisc.setAttribute('fill', discStyle.fill);
        existingDisc.setAttribute('stroke', discStyle.stroke);
    }

    getOpeningGuideStatusForHex(row, col) {
        if (this.validateBoardSize() !== this.boardSize) {
            return null;
        }
        const trmph = this.rowColToTRMPH(row, col).toLowerCase();
        return this.openingGuideStatusByMove.get(trmph) || null;
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

    clearAllOpeningGuideMarkers() {
        for (const marker of this.openingGuideMarkerElements.values()) {
            marker.remove();
        }
        this.openingGuideMarkerElements.clear();
    }

    makeOpeningGuideMarker(cx, cy, hexRadius, status) {
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        marker.setAttribute('cx', String(cx));
        marker.setAttribute('cy', String(cy));
        marker.setAttribute('r', String(Math.max(4.2, hexRadius * 0.45)));
        marker.classList.add('opening-guide-marker', `opening-guide-marker-${status}`);
        marker.style.pointerEvents = 'none';
        return marker;
    }

    syncOpeningGuideMarkers(board, hexRadius) {
        const shouldShow = this.openingGuideEnabled &&
            !this.heatmapEnabled &&
            this.validateBoardSize() === this.boardSize &&
            this.isBoardEmpty(board);
        if (!shouldShow) {
            this.clearAllOpeningGuideMarkers();
            return;
        }

        this.clearAllOpeningGuideMarkers();
        const boardSize = this.validateBoardSize();
        for (let row = 0; row < boardSize; row++) {
            for (let col = 0; col < boardSize; col++) {
                const status = this.getOpeningGuideStatusForHex(row, col);
                if (!status) {
                    continue;
                }
                const { x, y } = this.hexCenter(row, col, hexRadius);
                const marker = this.makeOpeningGuideMarker(x, y, hexRadius, status);
                this.svg.appendChild(marker);
                this.openingGuideMarkerElements.set(`${row},${col}`, marker);
            }
        }
    }

    hideInstructionText() {
        // Hide the instruction text about opening guide markers.
        if (this.instructionText) {
            this.instructionText.style.display = 'none';
        }
    }

    showInstructionText() {
        // Show the instruction text about opening guide markers.
        if (this.instructionText) {
            if (this.validateBoardSize() === this.boardSize && !this.openingGuideEnabled) {
                this.instructionText.style.display = 'none';
                return;
            }
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
        this.refreshThemeColors();

        // Constants for hexagonal board - adapted from working dev version
        const HEX_RADIUS = 18; // Slightly smaller for mobile
        const BOARD_SIZE = this.validateBoardSize();

        // Math for flat-topped hex grid. Internal blue side is top/bottom.
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

        // Draw edge indicators (internal blue: top/bottom, internal red: left/right).
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
                const hex = this.makeHex(x, y, HEX_RADIUS, fill, isLegal);
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

                this.updatePieceDisc(row, col, cell, x, y, HEX_RADIUS);
            }
        }
        if (isEmpty) {
            this.syncOpeningGuideMarkers(board, HEX_RADIUS);
        } else {
            this.clearAllOpeningGuideMarkers();
        }
    }

    drawEdgeIndicators(svgWidth, svgHeight, HEX_RADIUS, BOARD_SIZE) {
        const colors = this.getColors();

        // Internal blue edges (top and bottom). Display color depends on selected scheme.
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

        // Internal red edges (left and right) using proper edge midpoints.
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

    makeHex(cx, cy, r, fill, highlight) {
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
        hex.setAttribute('stroke', this.getColors().HEX_STROKE);
        hex.setAttribute('stroke-width', '1');

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

        this.cancelScheduledAutoMove();
        const row = parseInt(e.target.getAttribute('data-row'));
        const col = parseInt(e.target.getAttribute('data-col'));

        // Clear opening-guide markers immediately when a move is about to be made.
        this.clearAllOpeningGuideMarkers();

        // Hide instruction text after the opening move.
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

            this.recordReachedState(data.new_trmph, true);
            this.legalMoves = data.legal_moves || [];
            await this.renderBoardWithHeatmap(data.board);
            this.updateTrmphDisplay();
            this.updateButtonStates();
            this.persistSessionState();

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
        // Internal turn order: even moves = blue, odd moves = red.
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
        if (winner === 'blue' || winner === 'red') {
            this.showSuccess(`${this.getPlayerDisplayName(winner)} wins!`);
            return;
        }
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
        this.refreshThemeColors();
        this.statusLine.textContent = `Error: ${message}`;
        this.statusLine.style.color = this.getColors().STATUS_ERROR;
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
        this.refreshThemeColors();
        this.statusLine.textContent = message;
        this.statusLine.style.color = this.getColors().STATUS_SUCCESS;
        this.statusLine.style.fontWeight = 'bold';
        setTimeout(() => {
            this.statusLine.style.color = '';
            this.statusLine.style.fontWeight = '';
        }, 3000);
    }


    async redoMove() {
        if (this.isLoading || this.redoHistory.length === 0) return;

        this.cancelScheduledAutoMove();
        this.setLoading(true);
        try {
            if (!this.redoHistoryStep()) {
                return;
            }

            // Clear cache for redo to ensure clean state
            this.clearCachedBoardRenderingState();

            await this.loadGameStateWithoutAutoMove('redo');

            // Show instruction text if we're back to an empty board
            if (this.isBoardEmpty(this.previousBoard)) {
                this.showInstructionText();
            }
            this.persistSessionState();
        } catch (error) {
            console.error('Failed to redo move:', error);
            this.showError('Failed to redo move.');
        } finally {
            this.setLoading(false);
            this.updateButtonStates();
        }
    }


    getCurrentBareTrmphMoves() {
        const current = (this.currentTRMPH || '').trim();
        const preambleMatch = current.match(/^#(\d+),/);
        const bareMoves = preambleMatch ? current.substring(preambleMatch[0].length) : current;
        return this.stripVirtualPrefillIfPresent(bareMoves);
    }

    buildTrmphBoardUrl() {
        if (!Number.isFinite(this.boardSize)) {
            throw new Error('BOARD_SIZE not initialized from backend');
        }
        if (typeof this.trmphBoardShareBaseUrl !== 'string' || !this.trmphBoardShareBaseUrl) {
            throw new Error('TRMPH board share base URL not initialized from backend');
        }

        const displaySize = this.validateBoardSize();
        const prefill = this.virtualBoardPrefillMoves[displaySize];
        if (typeof prefill !== 'string') {
            throw new Error(`Missing virtual-board prefill for display board size ${displaySize}`);
        }

        const bareMoves = this.getCurrentBareTrmphMoves();
        return `${this.trmphBoardShareBaseUrl}#${this.boardSize},${prefill}${bareMoves}`;
    }

    async writeTextToClipboard(text) {
        if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
            try {
                await navigator.clipboard.writeText(text);
                return;
            } catch (_clipboardError) {
                // Fall back to execCommand below when clipboard API is unavailable/blocked.
            }
        }

        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.setAttribute('readonly', '');
        textArea.style.position = 'fixed';
        textArea.style.opacity = '0';
        textArea.style.pointerEvents = 'none';
        document.body.appendChild(textArea);
        textArea.select();
        textArea.setSelectionRange(0, text.length);

        const copied = document.execCommand('copy');
        document.body.removeChild(textArea);
        if (!copied) {
            throw new Error('document.execCommand(copy) failed');
        }
    }

    async copyTrmph() {
        try {
            const boardUrl = this.buildTrmphBoardUrl();
            await this.writeTextToClipboard(boardUrl);
            this.showSuccess('TRMPH board link copied to clipboard');
        } catch (error) {
            console.error('Failed to copy TRMPH board link:', error);
            this.showError('Failed to copy TRMPH board link');
        }
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

        // Clear opening-guide markers immediately when applying a sequence.
        this.clearAllOpeningGuideMarkers();

        // Hide instruction text after moves are already present.
        this.hideInstructionText();

        this.cancelScheduledAutoMove();
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

            this.recordReachedState(data.new_trmph, true);
            this.legalMoves = data.legal_moves || [];
            this.updateTrmphDisplay();
            await this.renderBoardWithHeatmap(data.board);
            this.updateButtonStates();
            this.persistSessionState();

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
