(function (global) {
  'use strict';

  function normalizeColorScheme(rawScheme) {
    if (rawScheme === 'classic' || rawScheme === 'classic_blue_first' || rawScheme === 'wood') {
      return rawScheme;
    }
    if (rawScheme === 'default') {
      return 'wood';
    }
    return 'wood';
  }

  function readStoredThemePreferences() {
    if (typeof localStorage === 'undefined') {
      return { darkMode: false, colorScheme: 'wood' };
    }

    try {
      return {
        darkMode: localStorage.getItem('hex_ai_dark_mode') === 'true',
        colorScheme: normalizeColorScheme(localStorage.getItem('hex_ai_color_scheme')),
      };
    } catch (error) {
      return { darkMode: false, colorScheme: 'wood' };
    }
  }

  function applyStoredThemePreferences() {
    const preferences = readStoredThemePreferences();
    if (typeof document !== 'undefined' && document.documentElement) {
      if (preferences.darkMode) {
        document.documentElement.setAttribute('data-theme', 'dark');
      } else {
        document.documentElement.removeAttribute('data-theme');
      }
      document.documentElement.setAttribute('data-color-scheme', preferences.colorScheme);
    }
    return preferences;
  }

  function getDisplayColorKeyForInternalSide(side, colorScheme) {
    const normalizedScheme = normalizeColorScheme(colorScheme);
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

  function getPlayerDisplayNames(colorScheme) {
    const normalizedScheme = normalizeColorScheme(
      colorScheme || readStoredThemePreferences().colorScheme
    );
    if (normalizedScheme === 'wood') {
      return { blue: 'Black', red: 'White' };
    }
    const blueDisplayColor = getDisplayColorKeyForInternalSide('blue', normalizedScheme);
    const redDisplayColor = getDisplayColorKeyForInternalSide('red', normalizedScheme);
    return {
      blue: blueDisplayColor === 'red' ? 'Red' : 'Blue',
      red: redDisplayColor === 'red' ? 'Red' : 'Blue',
    };
  }

  global.HexThemePreferences = {
    normalizeColorScheme,
    readStoredThemePreferences,
    applyStoredThemePreferences,
    getDisplayColorKeyForInternalSide,
    getPlayerDisplayNames,
  };
})(window);
