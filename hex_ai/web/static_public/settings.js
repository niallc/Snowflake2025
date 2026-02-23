class UserSettingsPage {
    constructor() {
        this.form = document.getElementById('user-settings-form');
        this.status = document.getElementById('settings-status');
        this.boardSizeSelect = document.getElementById('preferred-board-size');
        this.eloSlider = document.getElementById('preferred-elo');
        this.eloDisplay = document.getElementById('preferred-elo-display');
        this.colorSchemeSelect = document.getElementById('color-scheme');
        this.resetDefaultsBtn = document.getElementById('reset-defaults-btn');

        this.defaultSettings = {
            preferred_board_size: 13,
            preferred_elo: 600,
            color_scheme: 'default',
        };
        this.colorSchemeOptions = [{ value: 'default', label: 'Default (options coming soon)' }];
        this.storageKeys = {
            preferredBoardSize: 'hex_ai_display_board_size',
            preferredElo: 'hex_ai_preferred_elo',
            colorScheme: 'hex_ai_color_scheme',
        };
    }

    init() {
        if (!this.form || !this.boardSizeSelect || !this.eloSlider || !this.colorSchemeSelect) {
            console.error('Settings page is missing required elements');
            return;
        }

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

    applySettingsToForm(settings) {
        this.boardSizeSelect.value = String(settings.preferred_board_size);
        this.eloSlider.value = String(settings.preferred_elo);
        this.colorSchemeSelect.value = settings.color_scheme;
        this.updateEloDisplay();
    }

    getFormSettings() {
        return {
            preferred_board_size: parseInt(this.boardSizeSelect.value, 10),
            preferred_elo: parseInt(this.eloSlider.value, 10),
            color_scheme: this.colorSchemeSelect.value,
        };
    }

    getValidatedStoredSettings(boardOptions, minElo, maxElo) {
        const validBoardSizes = new Set(boardOptions);
        const storedBoardSize = parseInt(localStorage.getItem(this.storageKeys.preferredBoardSize), 10);
        const storedElo = parseInt(localStorage.getItem(this.storageKeys.preferredElo), 10);
        const storedColorScheme = localStorage.getItem(this.storageKeys.colorScheme);
        const allowedSchemes = new Set(this.colorSchemeOptions.map((option) => option.value));

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
                color_scheme: 'default',
            };
            const settings = this.getValidatedStoredSettings(boardOptions, minElo, maxElo);

            this.populateBoardSizeOptions(boardOptions);
            this.populateColorSchemeOptions(this.colorSchemeOptions);
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
            this.eloSlider.min = '1';
            this.eloSlider.max = '2350';
            this.applySettingsToForm(this.defaultSettings);
            this.setStatus('Could not load server defaults. Using local fallback values.', 'error');
        }
    }

    async saveSettings(settings) {
        this.setStatus('Saving settings...');
        try {
            localStorage.setItem(this.storageKeys.preferredBoardSize, String(settings.preferred_board_size));
            localStorage.setItem(this.storageKeys.preferredElo, String(settings.preferred_elo));
            localStorage.setItem(this.storageKeys.colorScheme, String(settings.color_scheme));
            this.applySettingsToForm(settings);
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
