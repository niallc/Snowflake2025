class HexReviewPage {
    constructor() {
        this.form = document.getElementById('review-form');
        this.trmphInput = document.getElementById('review-trmph-input');
        this.boardSizeSelect = document.getElementById('review-board-size');
        this.eloInput = document.getElementById('review-elo');
        this.status = document.getElementById('review-status');
        this.root = document.getElementById('review-root');

        this.defaultBoardSize = 13;
        this.boardSizeOptions = [13];
        this.defaultElo = 600;
        this.minElo = 1;
        this.maxElo = 2350;

        this.installEventListeners();
        window.HexGameReviewUi.applyStoredThemePreferences();
        this.loadConstants();
    }

    installEventListeners() {
        this.form.addEventListener('submit', async (event) => {
            event.preventDefault();
            await this.runReview();
        });
    }

    async loadConstants() {
        try {
            const response = await fetch('/api/constants');
            const data = await response.json();
            this.boardSizeOptions = Array.isArray(data.DISPLAY_BOARD_SIZE_OPTIONS) && data.DISPLAY_BOARD_SIZE_OPTIONS.length > 0
                ? data.DISPLAY_BOARD_SIZE_OPTIONS.map((value) => parseInt(value, 10)).filter(Number.isFinite)
                : [13];
            this.defaultBoardSize = Number.isFinite(parseInt(data.DEFAULT_DISPLAY_BOARD_SIZE, 10))
                ? parseInt(data.DEFAULT_DISPLAY_BOARD_SIZE, 10)
                : 13;
            this.minElo = data.ELO_CONFIG && Number.isFinite(data.ELO_CONFIG.MIN_ELO) ? data.ELO_CONFIG.MIN_ELO : 1;
            this.maxElo = data.ELO_CONFIG && Number.isFinite(data.ELO_CONFIG.MAX_ELO) ? data.ELO_CONFIG.MAX_ELO : 2350;
            this.defaultElo = data.ELO_CONFIG && Number.isFinite(data.ELO_CONFIG.DEFAULT_ELO) ? data.ELO_CONFIG.DEFAULT_ELO : 600;
            this.populateControls();
            this.applyQueryDefaults();
        } catch (error) {
            this.setStatus(`Failed to load review defaults: ${error.message}`, true);
        }
    }

    populateControls() {
        this.boardSizeSelect.innerHTML = this.boardSizeOptions
            .map((size) => `<option value="${size}">${size}x${size}</option>`)
            .join('');

        this.eloInput.min = String(this.minElo);
        this.eloInput.max = String(this.maxElo);

        const storedBoardSize = parseInt(localStorage.getItem('hex_ai_display_board_size'), 10);
        const storedElo = parseInt(localStorage.getItem('hex_ai_preferred_elo'), 10);

        this.boardSizeSelect.value = String(
            this.boardSizeOptions.includes(storedBoardSize) ? storedBoardSize : this.defaultBoardSize
        );
        this.eloInput.value = String(
            Number.isFinite(storedElo) && storedElo >= this.minElo && storedElo <= this.maxElo
                ? storedElo
                : this.defaultElo
        );
    }

    applyQueryDefaults() {
        const params = new URLSearchParams(window.location.search);
        const trmph = params.get('trmph');
        const displayBoardSize = parseInt(params.get('display_board_size'), 10);
        const eloRating = parseInt(params.get('elo_rating'), 10);

        if (typeof trmph === 'string' && trmph.trim()) {
            this.trmphInput.value = trmph.trim();
        }
        if (Number.isFinite(displayBoardSize) && this.boardSizeOptions.includes(displayBoardSize)) {
            this.boardSizeSelect.value = String(displayBoardSize);
        }
        if (Number.isFinite(eloRating) && eloRating >= this.minElo && eloRating <= this.maxElo) {
            this.eloInput.value = String(eloRating);
        }

        if (this.trmphInput.value.trim()) {
            this.runReview();
        }
    }

    setStatus(message, isError = false) {
        this.status.textContent = message || '';
        this.status.classList.toggle('error', Boolean(isError));
    }

    currentRequestPayload() {
        return {
            trmph: this.trmphInput.value.trim(),
            display_board_size: parseInt(this.boardSizeSelect.value, 10),
            elo_rating: parseInt(this.eloInput.value, 10),
        };
    }

    async runReview() {
        const payload = this.currentRequestPayload();
        if (!payload.trmph) {
            this.setStatus('Enter a move sequence to review.', true);
            return;
        }

        this.setStatus('Analyzing game...');
        window.HexGameReviewUi.renderLoading(this.root, 'Analyzing game...');

        try {
            const response = await fetch('/api/game_review', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Review request failed');
            }

            const query = new URLSearchParams({
                trmph: payload.trmph,
                display_board_size: String(payload.display_board_size),
                elo_rating: String(payload.elo_rating),
            });
            window.history.replaceState({}, '', `/review?${query.toString()}`);

            window.HexGameReviewUi.mount(this.root, data.review);
            this.setStatus(`Review ready: ${data.review.summary.total_moves} moves analyzed.`);
        } catch (error) {
            this.setStatus(error.message || 'Review request failed', true);
            window.HexGameReviewUi.renderError(this.root, error.message || 'Review request failed');
        }
    }
}

window.addEventListener('DOMContentLoaded', () => {
    new HexReviewPage();
});
