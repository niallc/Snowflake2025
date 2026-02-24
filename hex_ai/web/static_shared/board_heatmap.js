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
    const fallback = { r: 128, g: 128, b: 128 };
    const a = colorToRgb(colorA, fallback);
    const b = colorToRgb(colorB, fallback);
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

  function readCssToken(name, fallback) {
    if (typeof document === 'undefined' || !document.documentElement) {
      return fallback;
    }
    const raw = getComputedStyle(document.documentElement).getPropertyValue(name);
    const trimmed = typeof raw === 'string' ? raw.trim() : '';
    return trimmed || fallback;
  }

  let tokenProbeElement = null;

  function ensureTokenProbe() {
    if (typeof document === 'undefined' || !document.documentElement) {
      return null;
    }
    if (!tokenProbeElement) {
      tokenProbeElement = document.createElement('div');
      tokenProbeElement.style.position = 'absolute';
      tokenProbeElement.style.visibility = 'hidden';
      tokenProbeElement.style.pointerEvents = 'none';
      tokenProbeElement.style.width = '0';
      tokenProbeElement.style.height = '0';
      tokenProbeElement.style.overflow = 'hidden';
      document.documentElement.appendChild(tokenProbeElement);
    }
    return tokenProbeElement;
  }

  function resolveCssColorToken(name, fallback, property = 'color') {
    const probe = ensureTokenProbe();
    if (!probe) {
      return fallback;
    }
    const cssVar = `var(${name})`;
    if (property === 'backgroundColor') {
      probe.style.backgroundColor = cssVar;
      const resolved = getComputedStyle(probe).backgroundColor;
      return (typeof resolved === 'string' && resolved.trim()) ? resolved.trim() : fallback;
    }
    if (property === 'borderColor') {
      probe.style.borderColor = cssVar;
      const resolved = getComputedStyle(probe).borderColor;
      return (typeof resolved === 'string' && resolved.trim()) ? resolved.trim() : fallback;
    }
    probe.style.color = cssVar;
    const resolved = getComputedStyle(probe).color;
    return (typeof resolved === 'string' && resolved.trim()) ? resolved.trim() : fallback;
  }

  function emphasizeScoreAroundMidpoint(score, options = {}) {
    const normalized = clamp(score, 0.0, 1.0);
    const contrastGamma = clamp(
      Number.isFinite(options.contrastGamma) ? options.contrastGamma : 0.78,
      0.4,
      1.0
    );
    const distanceFromMid = Math.abs(normalized - 0.5) * 2.0;
    const emphasizedDistance = Math.pow(distanceFromMid, contrastGamma) * 0.5;
    return normalized >= 0.5
      ? 0.5 + emphasizedDistance
      : 0.5 - emphasizedDistance;
  }

  function classifyScore(score, options = {}) {
    if (!Number.isFinite(score)) {
      return null;
    }
    const midpoint = Number.isFinite(options.midpoint) ? options.midpoint : 0.5;
    const neutralBand = clamp(
      Number.isFinite(options.neutralBand) ? options.neutralBand : 0.03,
      0.0,
      0.2
    );
    if (score < midpoint - neutralBand) {
      return 'below';
    }
    if (score > midpoint + neutralBand) {
      return 'above';
    }
    return 'even';
  }

  let tooltipElement = null;

  function ensureTooltipElement(darkMode) {
    if (tooltipElement) {
      return tooltipElement;
    }
    tooltipElement = document.createElement('div');
    tooltipElement.style.position = 'fixed';
    tooltipElement.style.padding = '4px 8px';
    tooltipElement.style.borderRadius = '4px';
    tooltipElement.style.fontSize = '12px';
    tooltipElement.style.fontFamily = 'monospace';
    tooltipElement.style.pointerEvents = 'none';
    tooltipElement.style.zIndex = '1000';
    tooltipElement.style.border = '1px solid transparent';
    tooltipElement.style.boxShadow = '0 2px 6px rgba(0, 0, 0, 0.25)';
    applyTooltipTheme(tooltipElement, darkMode);
    return tooltipElement;
  }

  function applyTooltipTheme(element, darkMode) {
    if (!element) {
      return;
    }
    const fallbackBackground = darkMode ? 'rgba(20, 20, 24, 0.94)' : 'rgba(18, 22, 28, 0.9)';
    const fallbackText = darkMode ? '#f0f3ff' : '#f6fbff';
    const fallbackBorder = darkMode ? '#5f6b89' : '#7b88a7';
    const fallbackShadow = 'rgba(0, 0, 0, 0.25)';
    element.style.background = resolveCssColorToken('--heatmap-tooltip-bg', fallbackBackground, 'backgroundColor');
    element.style.color = resolveCssColorToken('--heatmap-tooltip-text', fallbackText, 'color');
    element.style.borderColor = resolveCssColorToken('--heatmap-tooltip-border', fallbackBorder, 'borderColor');
    element.style.boxShadow = `0 2px 6px ${readCssToken('--heatmap-tooltip-shadow', fallbackShadow)}`;
  }

  function defaultPalette(options = {}) {
    const isWoodScheme = options.colorScheme === 'wood';
    if (isWoodScheme) {
      if (options.darkMode) {
        return {
          low: '#de6f72',
          mid: '#8f9dc0',
          high: '#5fc88e',
        };
      }
      return {
        low: '#bf4c53',
        mid: '#7f8db6',
        high: '#2f9965',
      };
    }
    if (options.darkMode) {
      return {
        low: '#d09138',
        mid: '#6b7292',
        high: '#46c985',
      };
    }
    return {
      low: '#b87418',
      mid: '#8a93b0',
      high: '#1f9f60',
    };
  }

  function resolvePalette(options = {}) {
    const fallback = defaultPalette(options);
    const provided = options.palette;
    if (
      provided &&
      typeof provided === 'object' &&
      typeof provided.low === 'string' &&
      typeof provided.mid === 'string' &&
      typeof provided.high === 'string'
    ) {
      return provided;
    }
    return {
      low: resolveCssColorToken('--heatmap-low', fallback.low, 'color'),
      mid: resolveCssColorToken('--heatmap-mid', fallback.mid, 'color'),
      high: resolveCssColorToken('--heatmap-high', fallback.high, 'color'),
    };
  }

  function positionTooltip(event, options = {}) {
    if (!tooltipElement || !event) {
      return;
    }
    const offsetX = Number.isFinite(options.offsetX) ? options.offsetX : 12;
    const offsetY = Number.isFinite(options.offsetY) ? options.offsetY : -24;
    let left = event.clientX + offsetX;
    let top = event.clientY + offsetY;
    const rect = tooltipElement.getBoundingClientRect();
    const maxLeft = Math.max(8, window.innerWidth - rect.width - 8);
    const maxTop = Math.max(8, window.innerHeight - rect.height - 8);
    left = clamp(left, 8, maxLeft);
    top = clamp(top, 8, maxTop);
    tooltipElement.style.left = `${left}px`;
    tooltipElement.style.top = `${top}px`;
  }

  function showTooltip(event, text, options = {}) {
    if (typeof document === 'undefined' || !event) {
      return;
    }
    const element = ensureTooltipElement(Boolean(options.darkMode));
    applyTooltipTheme(element, Boolean(options.darkMode));
    element.textContent = text;
    if (!element.parentElement) {
      document.body.appendChild(element);
    }
    positionTooltip(event, options);
  }

  function moveTooltip(event, options = {}) {
    if (!tooltipElement || !event) {
      return;
    }
    positionTooltip(event, options);
  }

  function hideTooltip() {
    if (!tooltipElement) {
      return;
    }
    tooltipElement.remove();
    tooltipElement = null;
  }

  function scoreToColor(score, options = {}) {
    if (!Number.isFinite(score)) {
      return options.fallback || 'rgb(128, 128, 128)';
    }

    const palette = resolvePalette(options);

    const normalized = emphasizeScoreAroundMidpoint(score, options);
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
    classifyScore,
    emphasizeScoreAroundMidpoint,
    scoreToColor,
    formatPercent,
    tooltip: {
      show: showTooltip,
      move: moveTooltip,
      hide: hideTooltip,
    },
  };
})(window);
