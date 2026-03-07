(function (global) {
  'use strict';

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function formatPercent(value, digits = 1) {
    if (!Number.isFinite(value)) {
      return 'n/a';
    }
    return `${(value * 100).toFixed(digits)}%`;
  }

  function formatPoints(value, digits = 1) {
    if (!Number.isFinite(value)) {
      return 'n/a';
    }
    return `${(value * 100).toFixed(digits)} pts`;
  }

  function createElement(tagName, className, textContent) {
    const element = document.createElement(tagName);
    if (className) {
      element.className = className;
    }
    if (typeof textContent === 'string') {
      element.textContent = textContent;
    }
    return element;
  }

  function applyStoredThemePreferences() {
    if (typeof document === 'undefined' || !document.documentElement || typeof localStorage === 'undefined') {
      return;
    }
    const darkMode = localStorage.getItem('hex_ai_dark_mode');
    const colorScheme = localStorage.getItem('hex_ai_color_scheme');
    if (darkMode === 'true') {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    if (colorScheme) {
      document.documentElement.setAttribute('data-color-scheme', colorScheme === 'default' ? 'wood' : colorScheme);
    } else {
      document.documentElement.setAttribute('data-color-scheme', 'wood');
    }
  }

  function buildSummaryCard(label, value) {
    const card = createElement('div', 'review-summary-card');
    card.appendChild(createElement('p', 'review-summary-label', label));
    card.appendChild(createElement('p', 'review-summary-value', value));
    return card;
  }

  function buildChartSvg(trajectory) {
    const width = 920;
    const height = 260;
    const padding = { top: 22, right: 20, bottom: 28, left: 20 };
    const innerWidth = width - padding.left - padding.right;
    const innerHeight = height - padding.top - padding.bottom;
    const maxIndex = Math.max(1, trajectory.length - 1);

    const pointFor = (item, index, field) => {
      const x = padding.left + (index / maxIndex) * innerWidth;
      const y = padding.top + (1 - Math.max(0, Math.min(1, Number(item[field])))) * innerHeight;
      return { x, y };
    };

    const actualPoints = trajectory.map((item, index) => pointFor(item, index, 'blue_win_probability'));
    const bestPoints = trajectory.map((item, index) => {
      const field = Number.isFinite(item.best_blue_win_probability) ? 'best_blue_win_probability' : 'blue_win_probability';
      return pointFor(item, index, field);
    });

    const pathFor = (points) => points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(' ');
    const actualPath = pathFor(actualPoints);
    const bestPath = pathFor(bestPoints);

    const midlineY = padding.top + innerHeight / 2;
    const labels = trajectory
      .map((item, index) => ({ item, index }))
      .filter(({ index }) => index === 0 || index === trajectory.length - 1 || index % 10 === 0)
      .map(({ item, index }) => {
        const point = pointFor(item, index, 'blue_win_probability');
        const label = item.move_number === 0 ? '0' : String(item.move_number);
        return `<text x="${point.x.toFixed(2)}" y="${(height - 8).toFixed(2)}" text-anchor="middle" font-size="11" fill="var(--text-tertiary)">${label}</text>`;
      })
      .join('');

    const dots = trajectory
      .slice(1)
      .map((item, index) => {
        const point = pointFor(item, index + 1, 'blue_win_probability');
        const radius = Number(item.loss) >= 0.08 ? 3.3 : 2.2;
        return `<circle cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="${radius}" fill="var(--review-chart-dot)" opacity="0.86"></circle>`;
      })
      .join('');

    const startLabel = formatPercent(trajectory[0].blue_win_probability, 0);
    const endLabel = formatPercent(trajectory[trajectory.length - 1].blue_win_probability, 0);

    return `
      <svg viewBox="0 0 ${width} ${height}" aria-label="Blue win probability trajectory">
        <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
        <line x1="${padding.left}" y1="${midlineY}" x2="${width - padding.right}" y2="${midlineY}" stroke="var(--review-chart-midline)" stroke-width="1.4" stroke-dasharray="5 5"></line>
        <path d="${bestPath}" fill="none" stroke="var(--review-chart-best)" stroke-width="2.6" stroke-dasharray="8 6" opacity="0.8"></path>
        <path d="${actualPath}" fill="none" stroke="var(--review-chart-actual)" stroke-width="3.2"></path>
        ${dots}
        ${labels}
        <text x="${padding.left}" y="${padding.top - 4}" font-size="12" fill="var(--text-secondary)">Start ${escapeHtml(startLabel)}</text>
        <text x="${width - padding.right}" y="${padding.top - 4}" text-anchor="end" font-size="12" fill="var(--text-secondary)">End ${escapeHtml(endLabel)}</text>
      </svg>
    `;
  }

  function severityClass(analysis) {
    if (!analysis || !analysis.mistake_severity) {
      return 'severity-none';
    }
    return `severity-${analysis.mistake_severity}`;
  }

  function renderBoard(container, analysis) {
    const renderer = new global.HexBoardRenderer.BoardRenderer(container);
    const markers = [
      {
        row: Number(analysis.move_played[0]),
        col: Number(analysis.move_played[1]),
        kind: 'played',
      },
    ];

    (analysis.suggestions || []).forEach((suggestion, index) => {
      markers.push({
        row: Number(suggestion.row),
        col: Number(suggestion.col),
        kind: 'suggestion',
        label: String(index + 1),
      });
    });

    renderer.render({
      board: analysis.board_before,
      display_board_size: analysis.board_before.length,
      markers,
    });
  }

  function buildBoardLegend() {
    const legend = createElement('div', 'review-board-legend');
    const played = createElement('span', 'review-board-legend-item');
    played.innerHTML = '<span class="review-board-swatch played"></span>Played move';
    legend.appendChild(played);

    const suggestion = createElement('span', 'review-board-legend-item');
    suggestion.innerHTML = '<span class="review-board-swatch suggestion"></span>Suggested alternatives';
    legend.appendChild(suggestion);
    return legend;
  }

  function buildSuggestionList(analysis) {
    const suggestions = Array.isArray(analysis.suggestions) ? analysis.suggestions : [];
    if (suggestions.length === 0) {
      return createElement('p', 'review-moment-copy', 'No higher-ranked alternatives were selected in the review candidate set.');
    }

    const list = createElement('div', 'review-suggestion-list');
    suggestions.forEach((suggestion, index) => {
      const item = createElement('div', 'review-suggestion-item');
      item.innerHTML = `
        <span class="review-suggestion-index">${index + 1}</span>
        <span class="review-suggestion-move">${escapeHtml(suggestion.move_trmph)}</span>
        <span class="review-suggestion-detail">${escapeHtml(formatPercent(suggestion.win_probability_for_player))} for player</span>
        <span class="review-suggestion-detail">policy rank ${escapeHtml(String(suggestion.policy_rank))}</span>
      `;
      list.appendChild(item);
    });
    return list;
  }

  function buildMomentCard(analysis) {
    const severity = severityClass(analysis);
    const card = createElement('article', `review-moment-card ${severity}`);

    const boardShell = createElement('div', 'review-board-shell');
    const board = createElement('div', 'review-board');
    boardShell.appendChild(board);
    boardShell.appendChild(buildBoardLegend());

    const body = createElement('div');
    const titleRow = createElement('div', 'review-moment-title-row');
    titleRow.appendChild(createElement('h3', 'review-moment-title', `Move ${analysis.move_number}: ${analysis.move_played_trmph}`));
    const badge = createElement('span', `review-severity-badge ${severity}`, analysis.is_losing_move ? 'Losing swing' : analysis.mistake_severity);
    titleRow.appendChild(badge);
    body.appendChild(titleRow);

    const statGrid = createElement('div', 'review-moment-grid');
    const stats = [
      ['Player', analysis.player.toUpperCase()],
      ['Played', analysis.move_played_trmph],
      ['Best', analysis.best_move_trmph],
      ['Loss', formatPoints(analysis.win_probability_loss)],
      ['Before', formatPercent(analysis.position_win_probability_for_player)],
      ['After', formatPercent(analysis.move_played_win_probability_for_player)],
    ];
    stats.forEach(([label, value]) => {
      const stat = createElement('div', 'review-stat');
      stat.appendChild(createElement('p', 'review-stat-label', label));
      stat.appendChild(createElement('p', 'review-stat-value', value));
      statGrid.appendChild(stat);
    });
    body.appendChild(statGrid);

    body.appendChild(createElement('p', 'review-moment-copy', analysis.mistake_reason));
    body.appendChild(buildSuggestionList(analysis));

    card.appendChild(boardShell);
    card.appendChild(body);
    renderBoard(board, analysis);
    return card;
  }

  function buildMoveTable(analyses) {
    const wrap = createElement('div', 'review-table-wrap');
    const table = createElement('table', 'review-table');
    table.innerHTML = `
      <thead>
        <tr>
          <th>Move</th>
          <th>Player</th>
          <th>Played</th>
          <th>Best</th>
          <th>Loss</th>
          <th>Policy Rank</th>
          <th>Phase</th>
        </tr>
      </thead>
      <tbody></tbody>
    `;
    const body = table.querySelector('tbody');

    analyses.forEach((analysis) => {
      const row = document.createElement('tr');
      const severity = severityClass(analysis);
      row.innerHTML = `
        <td>${escapeHtml(String(analysis.move_number))}</td>
        <td>${escapeHtml(analysis.player.toUpperCase())}</td>
        <td>${escapeHtml(analysis.move_played_trmph)}</td>
        <td>${escapeHtml(analysis.best_move_trmph)}</td>
        <td class="loss-cell ${severity}">${escapeHtml(formatPoints(analysis.win_probability_loss))}</td>
        <td>${escapeHtml(String(analysis.move_played_policy_rank))}</td>
        <td>${escapeHtml(analysis.game_phase)}</td>
      `;
      body.appendChild(row);
    });

    wrap.appendChild(table);
    return wrap;
  }

  function buildResults(payload) {
    const root = createElement('div', 'review-results');
    const summary = payload.summary || {};
    const analyses = Array.isArray(payload.move_analyses) ? payload.move_analyses : [];
    const analysisMetadata = payload.analysis_metadata || {};

    const hero = createElement('section', 'review-hero');
    hero.innerHTML = `
      <div class="review-hero-top">
        <div>
          <h2 class="review-hero-title">Visual Review</h2>
          <p class="review-page-subtitle">Played move plus policy-ranked candidate alternatives, scored by the current model's value head.</p>
        </div>
        <div class="review-hero-meta">
          <span class="review-chip">Model ${escapeHtml(String(analysisMetadata.model || 'unknown'))}</span>
          <span class="review-chip">Board ${escapeHtml(String(analysisMetadata.display_board_size || '?'))}x${escapeHtml(String(analysisMetadata.display_board_size || '?'))}</span>
          <span class="review-chip">Candidates top ${escapeHtml(String(analysisMetadata.candidate_policy_top_k || '?'))} + played</span>
        </div>
      </div>
    `;

    const summaryGrid = createElement('div', 'review-summary-grid');
    summaryGrid.appendChild(buildSummaryCard('Moves', String(summary.total_moves || 0)));
    summaryGrid.appendChild(buildSummaryCard('Mistakes', String(summary.total_mistakes || 0)));
    summaryGrid.appendChild(buildSummaryCard('Major', String(summary.major_mistakes || 0)));
    summaryGrid.appendChild(buildSummaryCard('Losing Swings', String(summary.losing_moves || 0)));
    summaryGrid.appendChild(buildSummaryCard('Blue Mistakes', String(summary.mistake_by_player ? summary.mistake_by_player.blue : 0)));
    summaryGrid.appendChild(buildSummaryCard('Red Mistakes', String(summary.mistake_by_player ? summary.mistake_by_player.red : 0)));
    hero.appendChild(summaryGrid);
    root.appendChild(hero);

    const chartPanel = createElement('section', 'review-panel');
    chartPanel.appendChild(createElement('h2', null, 'Win Probability'));
    const chartFrame = createElement('div', 'review-chart-frame');
    chartFrame.innerHTML = buildChartSvg(payload.win_probability_trajectory || []);
    chartPanel.appendChild(chartFrame);
    chartPanel.appendChild(
      createElement(
        'p',
        'review-chart-note',
        'Solid line shows the played game. Dashed line shows the best reviewed candidate for each move, still from Blue’s perspective.'
      )
    );
    root.appendChild(chartPanel);

    const criticalPanel = createElement('section', 'review-panel');
    criticalPanel.appendChild(createElement('h2', null, 'Critical Moments'));
    const momentList = createElement('div', 'review-moment-list');
    const criticalAnalyses = analyses
      .filter((analysis) => analysis.is_mistake || analysis.is_losing_move)
      .sort((left, right) => Number(right.win_probability_loss) - Number(left.win_probability_loss))
      .slice(0, 8);

    if (criticalAnalyses.length === 0) {
      criticalPanel.appendChild(
        createElement(
          'p',
          'review-moment-copy',
          'No critical moments crossed the current review thresholds.'
        )
      );
    } else {
      criticalAnalyses.forEach((analysis) => {
        momentList.appendChild(buildMomentCard(analysis));
      });
      criticalPanel.appendChild(momentList);
    }
    root.appendChild(criticalPanel);

    const tablePanel = createElement('section', 'review-panel');
    tablePanel.appendChild(createElement('h2', null, 'Move List'));
    tablePanel.appendChild(buildMoveTable(analyses));
    root.appendChild(tablePanel);

    return root;
  }

  function clearAndAppend(root, node) {
    root.className = '';
    root.innerHTML = '';
    root.appendChild(node);
  }

  function renderLoading(root, message) {
    const wrapper = createElement('div', 'review-loading');
    wrapper.appendChild(createElement('span', 'review-loading-spinner'));
    wrapper.appendChild(createElement('span', null, message || 'Analyzing game...'));
    clearAndAppend(root, wrapper);
  }

  function renderError(root, message) {
    const error = createElement('div', 'review-empty');
    error.textContent = message || 'Failed to load review.';
    clearAndAppend(root, error);
  }

  function mount(root, payload) {
    applyStoredThemePreferences();
    clearAndAppend(root, buildResults(payload));
  }

  function mountStandalone(root, payload) {
    const layout = createElement('div', 'review-page-layout');
    layout.appendChild(buildResults(payload));
    clearAndAppend(root, layout);
  }

  global.HexGameReviewUi = {
    applyStoredThemePreferences,
    mount,
    mountStandalone,
    renderError,
    renderLoading,
  };
})(window);
