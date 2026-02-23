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

        this.loadSettings();
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

    async loadSettings() {
        this.setStatus('Loading settings...');
        try {
            const response = await fetch('/api/user_settings', { method: 'GET' });
            if (!response.ok) {
                throw new Error(`Failed to load settings (${response.status})`);
            }

            const payload = await response.json();
            const boardOptions = Array.isArray(payload.display_board_size_options)
                ? payload.display_board_size_options
                : [13];
            const colorOptions = Array.isArray(payload.color_scheme_options)
                ? payload.color_scheme_options
                : [{ value: 'default', label: 'Default' }];
            const eloConfig = payload.elo_config || {};
            const minElo = Number.isFinite(parseInt(eloConfig.min_elo, 10))
                ? parseInt(eloConfig.min_elo, 10)
                : 1;
            const maxElo = Number.isFinite(parseInt(eloConfig.max_elo, 10))
                ? parseInt(eloConfig.max_elo, 10)
                : 2350;

            this.defaultSettings = payload.default_settings || this.defaultSettings;
            const settings = payload.settings || this.defaultSettings;

            this.populateBoardSizeOptions(boardOptions);
            this.populateColorSchemeOptions(colorOptions);
            this.eloSlider.min = String(minElo);
            this.eloSlider.max = String(maxElo);
            this.applySettingsToForm(settings);

            this.setStatus('Settings loaded.');
        } catch (error) {
            console.error(error);
            this.setStatus('Could not load settings. Please refresh and try again.', 'error');
        }
    }

    async saveSettings(settings) {
        this.setStatus('Saving settings...');
        try {
            const response = await fetch('/api/user_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                const message = payload && payload.error ? payload.error : `Save failed (${response.status})`;
                throw new Error(message);
            }
            this.applySettingsToForm(payload.settings || settings);
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
