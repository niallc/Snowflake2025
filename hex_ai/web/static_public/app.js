// =============================================================================
// Hex AI Public Web App - Simplified Mobile-Optimized Version
// =============================================================================

class HexGame {
    constructor() {
        this.boardSize = 13;
        this.currentTRMPH = "";
        this.gameHistory = [];
        this.redoHistory = []; // Track undone moves for redo functionality
        this.moveCount = 0;
        this.currentElo = 1000;
        this.blueComputer = true;
        this.redComputer = false;
        this.isLoading = false;
        this.isInitialLoad = true; // Track if this is the initial page load
        
        this.initializeElements();
        this.setupEventListeners();
        this.loadGameConstants();
    }
    
    // =============================================================================
    // INITIALIZATION
    // =============================================================================
    
    initializeElements() {
        this.statusLine = document.getElementById('status-line');
        this.boardContainer = document.getElementById('board-container');
        this.resetBtn = document.getElementById('reset-btn');
        this.undoBtn = document.getElementById('undo-btn');
        this.redoBtn = document.getElementById('redo-btn');
        this.computerMoveBtn = document.getElementById('computer-move-btn');
        this.difficultyPreset = document.getElementById('difficulty-preset');
        this.eloSlider = document.getElementById('elo-slider');
        this.eloDisplay = document.getElementById('elo-display');
        this.blueComputerCheck = document.getElementById('blue-computer');
        this.redComputerCheck = document.getElementById('red-computer');
        this.trmphDisplay = document.getElementById('trmph-display');
        this.trmphInput = document.getElementById('trmph-input');
        this.copyTrmphBtn = document.getElementById('copy-trmph');
        this.applyTrmphBtn = document.getElementById('apply-trmph');
        this.trmphError = document.getElementById('trmph-error');
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
        
        this.eloSlider.addEventListener('input', (e) => {
            this.currentElo = parseInt(e.target.value);
            this.eloDisplay.textContent = this.currentElo;
            this.updateDifficultyPreset();
        });
        
        this.blueComputerCheck.addEventListener('change', (e) => {
            this.blueComputer = e.target.checked;
        });
        
        this.redComputerCheck.addEventListener('change', (e) => {
            this.redComputer = e.target.checked;
        });
    }
    
    updateDifficultyPreset() {
        // Find closest preset to current ELO
        const presets = [500, 1000, 1500, 1800, 2100, 2150, 2250, 2350];
        const closest = presets.reduce((prev, curr) => 
            Math.abs(curr - this.currentElo) < Math.abs(prev - this.currentElo) ? curr : prev
        );
        this.difficultyPreset.value = closest;
    }
    
    // =============================================================================
    // GAME STATE MANAGEMENT
    // =============================================================================
    
    async loadGameConstants() {
        try {
            const response = await fetch('/api/constants');
            const data = await response.json();
            this.boardSize = data.BOARD_SIZE;
            this.pieceValues = data.PIECE_VALUES;
            this.playerValues = data.PLAYER_VALUES;
            this.winnerValues = data.WINNER_VALUES;
            
            this.initializeBoard();
            // Initial load - allow computer auto-move
            await this.loadGameState(true);
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
        
        // Add a general click handler to debug
        svg.addEventListener('click', (e) => {
            console.log('SVG click event:', e.target);
        });
        
        this.boardContainer.appendChild(svg);
        this.svg = svg;
    }
    
    async resetGame() {
        this.setLoading(true);
        try {
            this.currentTRMPH = "";
            this.gameHistory = [];
            this.moveCount = 0;
            this.isInitialLoad = false; // Mark that this is no longer initial load
            this.updateTrmphDisplay();
            await this.loadGameState(false); // Don't auto-move after reset
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
            // Save current state to redo history before undoing
            const currentState = {
                trmph: this.currentTRMPH,
                moveCount: this.moveCount
            };
            this.redoHistory.push(currentState);
            
            // Restore previous state
            this.gameHistory.pop();
            this.currentTRMPH = this.gameHistory.length > 0 ? 
                this.gameHistory[this.gameHistory.length - 1] : "";
            this.moveCount = Math.max(0, this.moveCount - 1);
            await this.loadGameStateWithoutAutoMove();
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
        
        this.setLoading(true);
        try {
            const response = await fetch('/api/mcts_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trmph: this.currentTRMPH,
                    elo_rating: this.currentElo
                })
            });
            
            console.log('Computer move response status:', response.status);
            console.log('Computer move response ok:', response.ok);
            
            const data = await response.json();
            console.log('Computer move API response:', data);
            console.log('Computer move data.success:', data.success);
            console.log('Computer move data.error:', data.error);
            
            if (data.success) {
                this.currentTRMPH = data.new_trmph;
                this.gameHistory.push(this.currentTRMPH);
                this.moveCount++;
                
                // Clear redo history when new moves are made
                this.redoHistory = [];
                await this.renderBoard(data.board);
                this.updateTrmphDisplay();
                this.updateButtonStates();
                
                // Auto-move for computer players
                if (data.winner) {
                    this.showGameOver(data.winner);
                } else if (this.shouldMakeComputerMove(data.player)) {
                    console.log(`Auto-triggering computer move for player: ${data.player}`);
                    this.scheduleAutoMove();
                } else {
                    console.log(`No auto-move needed. Player: ${data.player}, shouldMakeComputerMove: ${this.shouldMakeComputerMove(data.player)}`);
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
    
    async loadGameState(autoMove = true) {
        try {
            console.log(`Loading game state with TRMPH: '${this.currentTRMPH}', ELO: ${this.currentElo}`);
            
            const response = await fetch('/api/state', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trmph: this.currentTRMPH,
                    elo_rating: this.currentElo
                })
            });
            
            console.log(`API response status: ${response.status}`);
            
            if (!response.ok) {
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
            
            // Store legal moves for move validation
            this.legalMoves = data.legal_moves || [];
            console.log('Legal moves received:', this.legalMoves);
            
            await this.renderBoard(data.board);
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
    
    async loadGameStateWithoutAutoMove() {
        return this.loadGameState(false);
    }
    
    // =============================================================================
    // BOARD RENDERING
    // =============================================================================
    
    async renderBoard(board) {
        if (!this.svg) return;
        
        console.log('Rendering board with legal moves:', this.legalMoves);
        
        // Clear existing board
        this.svg.innerHTML = '';
        
        // Use the proper hexagonal board rendering from the dev version
        this.drawHexBoard(board);
    }
    
    drawHexBoard(board) {
        // Constants for hexagonal board - adapted from working dev version
        const HEX_RADIUS = 18; // Slightly smaller for mobile
        const BOARD_SIZE = this.boardSize;
        
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
        this.svg.style.background = '#f8f8fa';
        
        // Draw edge indicators (blue: top/bottom, red: left/right)
        this.drawEdgeIndicators(svgWidth, svgHeight, HEX_RADIUS, BOARD_SIZE);
        
        // Draw hexagons
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                const { x, y } = this.hexCenter(row, col, HEX_RADIUS);
                const cell = board[row]?.[col] || this.pieceValues.EMPTY;
                
                // Determine fill color
                let fill = '#f0f0f0'; // Empty hex color
                if (cell === this.pieceValues.BLUE) fill = '#0099ff';
                if (cell === this.pieceValues.RED) fill = '#ff4444';
                
                const isLegal = this.isLegalMove(row, col);
                console.log(`Hex at row=${row}, col=${col}: isLegal=${isLegal}, isLoading=${this.isLoading}`);
                const hex = this.makeHex(x, y, HEX_RADIUS, fill, isLegal);
                hex.setAttribute('data-row', row);
                hex.setAttribute('data-col', col);
                
                if (isLegal && !this.isLoading) {
                    hex.classList.add('clickable');
                    // Use proper event handling like the dev version
                    const self = this;
                    hex.addEventListener('click', function(e) {
                        console.log('Hex click event triggered', e.target);
                        self.onCellClick(e);
                    });
                }
                
                this.svg.appendChild(hex);
            }
        }
    }
    
    drawEdgeIndicators(svgWidth, svgHeight, HEX_RADIUS, BOARD_SIZE) {
        // Blue edges (top and bottom) - adapted from working dev version
        const w = HEX_RADIUS * Math.sqrt(3);
        
        // Top edge - across the topmost hexes
        const topLine = this.makeEdgeLine(
            this.hexCenter(0, 0, HEX_RADIUS).x, this.hexCenter(0, 0, HEX_RADIUS).y - HEX_RADIUS,
            this.hexCenter(0, BOARD_SIZE - 1, HEX_RADIUS).x, this.hexCenter(0, BOARD_SIZE - 1, HEX_RADIUS).y - HEX_RADIUS,
            '#0099ff',
            18
        );
        this.svg.appendChild(topLine);
        
        // Bottom edge - across the bottommost hexes
        const bottomLine = this.makeEdgeLine(
            this.hexCenter(BOARD_SIZE - 1, 0, HEX_RADIUS).x, this.hexCenter(BOARD_SIZE - 1, 0, HEX_RADIUS).y + HEX_RADIUS,
            this.hexCenter(BOARD_SIZE - 1, BOARD_SIZE - 1, HEX_RADIUS).x, this.hexCenter(BOARD_SIZE - 1, BOARD_SIZE - 1, HEX_RADIUS).y + HEX_RADIUS,
            '#0099ff',
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
        const leftLine = this.makeEdgeLine(leftTopMid.x, leftTopMid.y, leftBotMid.x, leftBotMid.y, '#ff4444', 22);
        this.svg.appendChild(leftLine);
        
        // Right edge - between vertices 0 and 5
        const rightTopMid = this.edgeMidpoint(tr[0], tr[5]);
        const rightBotMid = this.edgeMidpoint(br[0], br[5]);
        const rightLine = this.makeEdgeLine(rightTopMid.x, rightTopMid.y, rightBotMid.x, rightBotMid.y, '#ff4444', 22);
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
        hex.setAttribute('stroke', '#ddd');
        hex.setAttribute('stroke-width', '1');
        if (highlight) {
            hex.style.cursor = 'pointer';
            console.log('Made hex clickable at', cx, cy);
        }
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
        console.log('onCellClick called', e.target);
        if (this.isLoading) {
            console.log('Loading, ignoring click');
            return;
        }
        
        const row = parseInt(e.target.getAttribute('data-row'));
        const col = parseInt(e.target.getAttribute('data-col'));
        console.log(`Clicked on row=${row}, col=${col}`);
        
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
                    elo_rating: this.currentElo
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                this.showError(data.error);
                return;
            }
            
            this.currentTRMPH = data.new_trmph;
            this.gameHistory.push(this.currentTRMPH);
            this.moveCount++;
            
            // Clear redo history when new moves are made
            this.redoHistory = [];
            await this.renderBoard(data.board);
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
        console.log(`Checking move ${move} (row=${row}, col=${col}): legal=${isLegal}, available moves:`, this.legalMoves);
        return isLegal;
    }
    
    
    handleBoardClick(e) {
        // Handle board-level clicks if needed
        console.log('Board click event:', e.target);
        
        // If we clicked on a hex, handle it
        if (e.target.tagName === 'polygon') {
            const row = parseInt(e.target.getAttribute('data-row'));
            const col = parseInt(e.target.getAttribute('data-col'));
            console.log(`Board click on hex row=${row}, col=${col}`);
            
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
        const letters = 'abcdefghijklm';
        return letters[col] + (row + 1);
    }
    
    
    updateTrmphDisplay() {
        this.trmphDisplay.value = this.currentTRMPH;
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
        
        if (loading) {
            document.body.classList.add('loading');
        } else {
            document.body.classList.remove('loading');
        }
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
            // Save current state to undo history before redoing
            const currentState = {
                trmph: this.currentTRMPH,
                moveCount: this.moveCount
            };
            this.gameHistory.push(currentState);
            
            // Restore next state from redo history
            const nextState = this.redoHistory.pop();
            this.currentTRMPH = nextState.trmph;
            this.moveCount = nextState.moveCount;
            await this.loadGameStateWithoutAutoMove();
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
        // Parse TRMPH string into individual moves (similar to split_trmph_moves in Python)
        const moves = [];
        let i = 0;
        const letters = 'abcdefghijklm';
        
        while (i < trmphString.length) {
            if (!letters.includes(trmphString[i])) {
                throw new Error(`Expected letter at position ${i} in ${trmphString}`);
            }
            let j = i + 1;
            while (j < trmphString.length && /\d/.test(trmphString[j])) {
                j++;
            }
            moves.push(trmphString.substring(i, j));
            i = j;
        }
        return moves;
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
        
        // Check move count by parsing the TRMPH string properly
        // This handles moves of varying length (a1, b13, m12, etc.)
        try {
            const moves = this.parseTrmphMoves(trimmed);
            const maxMoves = 13 * 13; // 169 moves for a complete game
            if (moves.length > maxMoves) {
                return { valid: false, error: `Too many moves (maximum ${maxMoves} moves for a complete game)` };
            }
        } catch (error) {
            return { valid: false, error: `Invalid TRMPH format: ${error.message}` };
        }
        
        // Validate TRMPH format: ([a-m]([1-9]|1[0-3]))+
        const trmphRegex = /^([a-m]([1-9]|1[0-3]))+$/;
        if (!trmphRegex.test(trimmed)) {
            return { 
                valid: false, 
                error: 'Invalid format. Only letters a-m followed by numbers 1-13 are allowed (e.g., a1b2c3)' 
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
        
        this.setLoading(true);
        try {
            const response = await fetch('/api/apply_trmph_sequence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trmph: this.currentTRMPH,
                    trmph_sequence: input,
                    elo_rating: this.currentElo
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                this.showTrmphError(data.error);
                return;
            }
            
            this.currentTRMPH = data.new_trmph;
            this.gameHistory.push(this.currentTRMPH);
            this.moveCount += data.moves_applied || 0;
            
            // Clear redo history when new moves are made
            this.redoHistory = [];
            this.updateTrmphDisplay();
            await this.renderBoard(data.board);
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

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new HexGame();
});
