class UserSettingsPage {
    constructor() {
        this.form = document.getElementById('user-settings-form');
        this.status = document.getElementById('settings-status');
        this.boardSizeSelect = document.getElementById('preferred-board-size');
        this.eloSlider = document.getElementById('preferred-elo');
        this.eloDisplay = document.getElementById('preferred-elo-display');
        this.colorSchemeSelect = document.getElementById('color-scheme');
        this.pieceStyleSelect = document.getElementById('piece-style');
        this.resetDefaultsBtn = document.getElementById('reset-defaults-btn');

        this.defaultSettings = {
            preferred_board_size: 13,
            preferred_elo: 600,
            color_scheme: 'wood',
            piece_style: 'disc',
        };
        this.colorSchemeOptions = [
            { value: 'wood', label: 'Soft Wood (Black/White Pieces)' },
            { value: 'classic', label: 'Red / Blue (Red Goes First)' },
            { value: 'classic_blue_first', label: 'Blue / Red (Blue Goes First)' },
        ];
        this.pieceStyleOptions = [
            { value: 'disc', label: 'Disc Pieces in Hexagons' },
            { value: 'hex_fill', label: 'Fill Hexagons' },
        ];
        this.storageKeys = {
            preferredBoardSize: 'hex_ai_display_board_size',
            preferredElo: 'hex_ai_preferred_elo',
            colorScheme: 'hex_ai_color_scheme',
            pieceStyle: 'hex_ai_piece_style',
            darkMode: 'hex_ai_dark_mode',
        };
    }

    init() {
        if (!this.form || !this.boardSizeSelect || !this.eloSlider || !this.colorSchemeSelect || !this.pieceStyleSelect) {
            console.error('Settings page is missing required elements');
            return;
        }

        this.applySavedDarkModePreference();

        this.form.addEventListener('submit', async (event) => {
            event.preventDefault();
            await this.saveSettings(this.getFormSettings());
        });

        this.resetDefaultsBtn.addEventListener('click', async () => {
            this.applySettingsToForm(this.defaultSettings);
            await this.saveSettings(this.defaultSettings);
        });

        this.eloSlider.addEventListener('input', () => {
            this.updateEloDisplay();
        });

        window.addEventListener('storage', (event) => {
            if (event.key === this.storageKeys.darkMode) {
                this.applySavedDarkModePreference();
            }
        });

        this.loadSettingsFromConfig();
    }

    setStatus(message, type = 'info') {
        if (!this.status) {
            return;
        }
        this.status.textContent = message;
        this.status.classList.remove('status-error', 'status-success');
        if (type === 'error') {
            this.status.classList.add('status-error');
        } else if (type === 'success') {
            this.status.classList.add('status-success');
        }
    }

    updateEloDisplay() {
        this.eloDisplay.textContent = this.eloSlider.value;
    }

    populateBoardSizeOptions(options) {
        this.boardSizeSelect.innerHTML = '';
        for (const size of options) {
            const option = document.createElement('option');
            option.value = String(size);
            option.textContent = `${size}x${size}`;
            this.boardSizeSelect.appendChild(option);
        }
    }

    populateColorSchemeOptions(options) {
        this.colorSchemeSelect.innerHTML = '';
        for (const optionData of options) {
            const option = document.createElement('option');
            option.value = optionData.value;
            option.textContent = optionData.label;
            this.colorSchemeSelect.appendChild(option);
        }
    }

    populatePieceStyleOptions(options) {
        this.pieceStyleSelect.innerHTML = '';
        for (const optionData of options) {
            const option = document.createElement('option');
            option.value = optionData.value;
            option.textContent = optionData.label;
            this.pieceStyleSelect.appendChild(option);
        }
    }

    applySettingsToForm(settings) {
        this.boardSizeSelect.value = String(settings.preferred_board_size);
        this.eloSlider.value = String(settings.preferred_elo);
        this.colorSchemeSelect.value = this.normalizeColorScheme(settings.color_scheme);
        this.pieceStyleSelect.value = this.normalizePieceStyle(settings.piece_style);
        this.updateEloDisplay();
    }

    getFormSettings() {
        return {
            preferred_board_size: parseInt(this.boardSizeSelect.value, 10),
            preferred_elo: parseInt(this.eloSlider.value, 10),
            color_scheme: this.normalizeColorScheme(this.colorSchemeSelect.value),
            piece_style: this.normalizePieceStyle(this.pieceStyleSelect.value),
        };
    }

    normalizeColorScheme(rawValue) {
        if (rawValue === 'default') {
            return 'wood';
        }
        const allowedSchemes = new Set(this.colorSchemeOptions.map((option) => option.value));
        return allowedSchemes.has(rawValue) ? rawValue : this.defaultSettings.color_scheme;
    }

    normalizePieceStyle(rawValue) {
        const allowedStyles = new Set(this.pieceStyleOptions.map((option) => option.value));
        return allowedStyles.has(rawValue) ? rawValue : this.defaultSettings.piece_style;
    }

    applySavedDarkModePreference() {
        const savedDarkMode = localStorage.getItem(this.storageKeys.darkMode);
        if (savedDarkMode === 'true') {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
    }

    getValidatedStoredSettings(boardOptions, minElo, maxElo) {
        const validBoardSizes = new Set(boardOptions);
        const storedBoardSize = parseInt(localStorage.getItem(this.storageKeys.preferredBoardSize), 10);
        const storedElo = parseInt(localStorage.getItem(this.storageKeys.preferredElo), 10);
        const storedColorScheme = this.normalizeColorScheme(localStorage.getItem(this.storageKeys.colorScheme));
        const storedPieceStyle = this.normalizePieceStyle(localStorage.getItem(this.storageKeys.pieceStyle));
        const allowedSchemes = new Set(this.colorSchemeOptions.map((option) => option.value));
        const allowedStyles = new Set(this.pieceStyleOptions.map((option) => option.value));

        return {
            preferred_board_size: validBoardSizes.has(storedBoardSize)
                ? storedBoardSize
                : this.defaultSettings.preferred_board_size,
            preferred_elo: Number.isFinite(storedElo) && storedElo >= minElo && storedElo <= maxElo
                ? storedElo
                : this.defaultSettings.preferred_elo,
            color_scheme: allowedSchemes.has(storedColorScheme)
                ? storedColorScheme
                : this.defaultSettings.color_scheme,
            piece_style: allowedStyles.has(storedPieceStyle)
                ? storedPieceStyle
                : this.defaultSettings.piece_style,
        };
    }

    async loadSettingsFromConfig() {
        this.setStatus('Loading settings...');
        try {
            const response = await fetch('/api/constants', { method: 'GET' });
            if (!response.ok) {
                throw new Error(`Failed to load settings (${response.status})`);
            }

            const payload = await response.json();
            const boardOptions = Array.isArray(payload.DISPLAY_BOARD_SIZE_OPTIONS)
                ? payload.DISPLAY_BOARD_SIZE_OPTIONS
                : [13];
            const eloConfig = payload.ELO_CONFIG || {};
            const minElo = Number.isFinite(parseInt(eloConfig.MIN_ELO, 10))
                ? parseInt(eloConfig.MIN_ELO, 10)
                : 1;
            const maxElo = Number.isFinite(parseInt(eloConfig.MAX_ELO, 10))
                ? parseInt(eloConfig.MAX_ELO, 10)
                : 2350;
            const defaultBoardSize = Number.isFinite(parseInt(payload.DEFAULT_DISPLAY_BOARD_SIZE, 10))
                ? parseInt(payload.DEFAULT_DISPLAY_BOARD_SIZE, 10)
                : boardOptions[0];
            const defaultElo = Number.isFinite(parseInt(eloConfig.DEFAULT_ELO, 10))
                ? parseInt(eloConfig.DEFAULT_ELO, 10)
                : 600;

            this.defaultSettings = {
                preferred_board_size: defaultBoardSize,
                preferred_elo: defaultElo,
                color_scheme: 'wood',
                piece_style: 'disc',
            };
            const settings = this.getValidatedStoredSettings(boardOptions, minElo, maxElo);

            this.populateBoardSizeOptions(boardOptions);
            this.populateColorSchemeOptions(this.colorSchemeOptions);
            this.populatePieceStyleOptions(this.pieceStyleOptions);
            this.eloSlider.min = String(minElo);
            this.eloSlider.max = String(maxElo);
            this.applySettingsToForm(settings);

            this.setStatus('Settings loaded.');
        } catch (error) {
            console.error(error);
            // Fall back to local-only defaults if constants endpoint is unavailable.
            const fallbackBoardOptions = [13];
            this.populateBoardSizeOptions(fallbackBoardOptions);
            this.populateColorSchemeOptions(this.colorSchemeOptions);
            this.populatePieceStyleOptions(this.pieceStyleOptions);
            this.eloSlider.min = '1';
            this.eloSlider.max = '2350';
            this.applySettingsToForm(this.defaultSettings);
            this.setStatus('Could not load server defaults. Using local fallback values.', 'error');
        }
    }

    async saveSettings(settings) {
        this.setStatus('Saving settings...');
        try {
            const normalizedSettings = {
                preferred_board_size: settings.preferred_board_size,
                preferred_elo: settings.preferred_elo,
                color_scheme: this.normalizeColorScheme(settings.color_scheme),
                piece_style: this.normalizePieceStyle(settings.piece_style),
            };

            localStorage.setItem(this.storageKeys.preferredBoardSize, String(normalizedSettings.preferred_board_size));
            localStorage.setItem(this.storageKeys.preferredElo, String(normalizedSettings.preferred_elo));
            localStorage.setItem(this.storageKeys.colorScheme, String(normalizedSettings.color_scheme));
            localStorage.setItem(this.storageKeys.pieceStyle, String(normalizedSettings.piece_style));
            this.applySettingsToForm(normalizedSettings);
            this.setStatus('Settings saved.', 'success');
        } catch (error) {
            console.error(error);
            this.setStatus(`Save failed: ${error.message}`, 'error');
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const page = new UserSettingsPage();
    page.init();
});
