(function (global) {
  'use strict';

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function hexToRgb(hexColor) {
    const normalized = hexColor.replace('#', '');
    if (normalized.length !== 6) {
      return { r: 128, g: 128, b: 128 };
    }
    return {
      r: parseInt(normalized.slice(0, 2), 16),
      g: parseInt(normalized.slice(2, 4), 16),
      b: parseInt(normalized.slice(4, 6), 16),
    };
  }

  function lerpChannel(a, b, t) {
    return Math.round(a + (b - a) * t);
  }

  function interpolateRgb(colorA, colorB, t) {
    const a = hexToRgb(colorA);
    const b = hexToRgb(colorB);
    return {
      r: lerpChannel(a.r, b.r, t),
      g: lerpChannel(a.g, b.g, t),
      b: lerpChannel(a.b, b.b, t),
    };
  }

  function scoreToColor(score, options = {}) {
    if (!Number.isFinite(score)) {
      return options.fallback || 'rgba(128,128,128,0.4)';
    }

    const palette = options.darkMode
      ? {
          low: '#ff5a4f',
          mid: '#5d5d5d',
          high: '#4cd27c',
        }
      : {
          low: '#e34b4b',
          mid: '#f3e8c8',
          high: '#35b56a',
        };

    const normalized = clamp(score, 0.0, 1.0);
    const alpha = clamp(
      Number.isFinite(options.alpha) ? options.alpha : 0.62,
      0.05,
      0.95
    );

    let rgb;
    if (normalized <= 0.5) {
      rgb = interpolateRgb(palette.low, palette.mid, normalized / 0.5);
    } else {
      rgb = interpolateRgb(palette.mid, palette.high, (normalized - 0.5) / 0.5);
    }

    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
  }

  function formatPercent(score) {
    if (!Number.isFinite(score)) {
      return 'n/a';
    }
    return `${(score * 100).toFixed(1)}%`;
  }

  global.HexHeatmap = {
    clamp,
    scoreToColor,
    formatPercent,
  };
})(window);
