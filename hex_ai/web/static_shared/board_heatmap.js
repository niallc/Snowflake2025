(function (global) {
  'use strict';

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function hexToRgb(hexColor) {
    const normalized = hexColor.replace('#', '');
    if (normalized.length === 3) {
      return {
        r: parseInt(normalized[0] + normalized[0], 16),
        g: parseInt(normalized[1] + normalized[1], 16),
        b: parseInt(normalized[2] + normalized[2], 16),
      };
    }
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

  function parseRgbString(color) {
    if (typeof color !== 'string') {
      return null;
    }
    const match = color.trim().match(
      /^rgba?\(\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)(?:\s*,\s*[+-]?\d+\.?\d*)?\s*\)$/i
    );
    if (!match) {
      return null;
    }
    return {
      r: clamp(Math.round(Number(match[1])), 0, 255),
      g: clamp(Math.round(Number(match[2])), 0, 255),
      b: clamp(Math.round(Number(match[3])), 0, 255),
    };
  }

  function colorToRgb(color, fallback) {
    if (typeof color !== 'string') {
      return fallback;
    }
    const trimmed = color.trim();
    if (trimmed.startsWith('#')) {
      return hexToRgb(trimmed);
    }
    return parseRgbString(trimmed) || fallback;
  }

  function scoreToColor(score, options = {}) {
    if (!Number.isFinite(score)) {
      return options.fallback || 'rgb(128, 128, 128)';
    }

    const palette = options.darkMode
      ? {
          low: '#d79a45',
          mid: '#77809a',
          high: '#57c98b',
        }
      : {
          low: '#c9832d',
          mid: '#9ea7c2',
          high: '#2f9f68',
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

    if (options.useTrueAlpha === true) {
      return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
    }

    // Blend with base cell color so the cell remains visually opaque.
    const base = colorToRgb(options.baseColor || options.fallback, { r: 128, g: 128, b: 128 });
    const blended = {
      r: lerpChannel(base.r, rgb.r, alpha),
      g: lerpChannel(base.g, rgb.g, alpha),
      b: lerpChannel(base.b, rgb.b, alpha),
    };
    return `rgb(${blended.r}, ${blended.g}, ${blended.b})`;
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
