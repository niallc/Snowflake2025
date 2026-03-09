(function (global) {
  'use strict';

  const PLAYED_MOVE_WARNING_THRESHOLD = 0.05;
  const REVIEWED_MOVE_DISPLAY_COUNT = 3;

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

  function reviewLossValue(analysis) {
    const reviewLoss = Number(analysis && analysis.review_score_loss);
    if (Number.isFinite(reviewLoss)) {
      return reviewLoss;
    }
    const winProbabilityLoss = Number(analysis && analysis.win_probability_loss);
    if (Number.isFinite(winProbabilityLoss)) {
      return winProbabilityLoss;
    }
    return 0;
  }

  function isSwapAwareOpening(analysis) {
    return Boolean(analysis && analysis.review_metric === 'swap_evenness');
  }

  function playedMarkerTone(analysis) {
    if (!analysis) {
      return 'mistake';
    }
    if (analysis.best_move_trmph === analysis.move_played_trmph) {
      return 'recommended';
    }
    return reviewLossValue(analysis) < PLAYED_MOVE_WARNING_THRESHOLD ? 'close' : 'mistake';
  }

  function reviewRankValue(candidate) {
    const rank = Number(candidate && candidate.review_rank);
    if (!Number.isFinite(rank) || rank < 1) {
      return Number.POSITIVE_INFINITY;
    }
    return rank;
  }

  function formatReviewedRank(rank) {
    const numericRank = Number(rank);
    if (!Number.isFinite(numericRank) || numericRank < 1) {
      return '#?';
    }
    return `#${String(numericRank)}`;
  }

  function markerLabelForReviewedRank(rank) {
    const numericRank = Number(rank);
    if (!Number.isFinite(numericRank) || numericRank < 1) {
      return '?';
    }
    return String(numericRank);
  }

  function reviewedMoveSort(left, right) {
    const rankDelta = reviewRankValue(left) - reviewRankValue(right);
    if (rankDelta !== 0) {
      return rankDelta;
    }
    if (Boolean(left && left.is_played_move) !== Boolean(right && right.is_played_move)) {
      return left && left.is_played_move ? -1 : 1;
    }
    return String((left && left.move_trmph) || '').localeCompare(String((right && right.move_trmph) || ''));
  }

  function buildDisplayedReviewedMoves(analysis) {
    if (!analysis) {
      return [];
    }

    const playedMove = {
      row: Number(analysis.move_played[0]),
      col: Number(analysis.move_played[1]),
      move_trmph: analysis.move_played_trmph,
      review_rank: Number(analysis.move_played_review_rank),
      is_played_move: true,
      win_probability_for_player: Number(analysis.move_played_win_probability_for_player),
      blue_win_probability: Number(analysis.blue_win_probability_after_played),
      policy_probability: Number(analysis.move_played_policy_probability),
      policy_rank: Number(analysis.move_played_policy_rank),
      review_score: Number(analysis.review_score_played),
      distance_to_even: analysis.move_played_distance_to_even,
    };

    const alternatives = (Array.isArray(analysis.suggestions) ? analysis.suggestions : [])
      .filter((candidate) => candidate && candidate.move_trmph && candidate.move_trmph !== analysis.move_played_trmph)
      .slice()
      .sort(reviewedMoveSort);

    if (!(reviewRankValue(playedMove) < Number.POSITIVE_INFINITY)) {
      return [playedMove].concat(alternatives.slice(0, Math.max(0, REVIEWED_MOVE_DISPLAY_COUNT - 1)));
    }

    if (reviewRankValue(playedMove) <= REVIEWED_MOVE_DISPLAY_COUNT) {
      return [playedMove]
        .concat(alternatives)
        .sort(reviewedMoveSort)
        .slice(0, REVIEWED_MOVE_DISPLAY_COUNT);
    }

    return alternatives
      .slice(0, Math.max(0, REVIEWED_MOVE_DISPLAY_COUNT - 1))
      .concat([playedMove])
      .sort(reviewedMoveSort);
  }

  function impactLabel(analysis) {
    return isSwapAwareOpening(analysis) ? 'Balance Gap' : 'Loss';
  }

  function formatImpact(analysis) {
    const value = reviewLossValue(analysis);
    if (!Number.isFinite(value)) {
      return 'n/a';
    }
    return isSwapAwareOpening(analysis)
      ? `${formatPoints(value)} from 50%`
      : formatPoints(value);
  }

  function formatDistanceToEven(value) {
    if (!Number.isFinite(value)) {
      return 'n/a';
    }
    return `${formatPoints(value)} from 50%`;
  }

  function applyStoredThemePreferences() {
    const themeApi = global.HexThemePreferences;
    if (themeApi && typeof themeApi.applyStoredThemePreferences === 'function') {
      return themeApi.applyStoredThemePreferences();
    }

    if (typeof document === 'undefined' || !document.documentElement || typeof localStorage === 'undefined') {
      return { darkMode: false, colorScheme: 'wood' };
    }
    const darkMode = localStorage.getItem('hex_ai_dark_mode') === 'true';
    const colorScheme = localStorage.getItem('hex_ai_color_scheme');
    if (darkMode) {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    document.documentElement.setAttribute('data-color-scheme', colorScheme === 'default' ? 'wood' : (colorScheme || 'wood'));
    return { darkMode, colorScheme: colorScheme === 'default' ? 'wood' : (colorScheme || 'wood') };
  }

  function getPlayerDisplayNames() {
    const themeApi = global.HexThemePreferences;
    if (themeApi && typeof themeApi.getPlayerDisplayNames === 'function') {
      return themeApi.getPlayerDisplayNames();
    }
    return { blue: 'Blue', red: 'Red' };
  }

  function playerDisplayName(player) {
    const names = getPlayerDisplayNames();
    return player === 'red' ? names.red : names.blue;
  }

  function formatPhase(value) {
    if (!value) {
      return 'Unknown';
    }
    return `${String(value).charAt(0).toUpperCase()}${String(value).slice(1)}`;
  }

  function momentBadgeText(analysis) {
    if (analysis && analysis.is_losing_move) {
      return 'Losing swing';
    }
    if (isSwapAwareOpening(analysis)) {
      return analysis && analysis.is_mistake ? 'Opening balance' : 'Opening';
    }
    if (analysis && analysis.is_mistake && analysis.mistake_severity) {
      return analysis.mistake_severity;
    }
    return 'Steady';
  }

  function severityClass(analysis) {
    if (!analysis || !analysis.mistake_severity) {
      return 'severity-none';
    }
    return `severity-${analysis.mistake_severity}`;
  }

  function isQuietAnalysis(analysis) {
    return !analysis.is_mistake && !analysis.is_losing_move;
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

  function buildSummaryCard(label, value, detail) {
    const card = createElement('div', 'review-summary-card');
    card.appendChild(createElement('p', 'review-summary-label', label));
    card.appendChild(createElement('p', 'review-summary-value', value));
    if (detail) {
      card.appendChild(createElement('p', 'review-summary-detail', detail));
    }
    return card;
  }

  function buildStatCard(label, value) {
    const stat = createElement('div', 'review-stat');
    stat.appendChild(createElement('p', 'review-stat-label', label));
    stat.appendChild(createElement('p', 'review-stat-value', value));
    return stat;
  }

  function buildBoardLegend() {
    const legendWrap = createElement('div', 'review-board-legend-wrap');
    const legend = createElement('div', 'review-board-legend');

    const playedRecommended = createElement('span', 'review-board-legend-item');
    playedRecommended.innerHTML = '<span class="review-board-swatch played-recommended"></span>Played and recommended';
    legend.appendChild(playedRecommended);

    const playedClose = createElement('span', 'review-board-legend-item');
    playedClose.innerHTML = '<span class="review-board-swatch played-close"></span>Played, under 5% loss';
    legend.appendChild(playedClose);

    const playedMistake = createElement('span', 'review-board-legend-item');
    playedMistake.innerHTML = '<span class="review-board-swatch played-mistake"></span>Played, 5%+ loss';
    legend.appendChild(playedMistake);

    const suggestion = createElement('span', 'review-board-legend-item');
    suggestion.innerHTML = '<span class="review-board-swatch suggestion"></span>Other reviewed move';
    legend.appendChild(suggestion);

    legendWrap.appendChild(legend);
    legendWrap.appendChild(
      createElement(
        'p',
        'review-board-legend-note',
        'The board shows three reviewed moves total. Numbers are reviewed ranks, and the played move is always included.'
      )
    );
    return legendWrap;
  }

  function renderBoard(container, analysis) {
    const renderer = new global.HexBoardRenderer.BoardRenderer(container);
    const markers = buildDisplayedReviewedMoves(analysis).map((candidate) => ({
      row: Number(candidate.row),
      col: Number(candidate.col),
      kind: candidate.is_played_move ? 'played' : 'suggestion',
      tone: candidate.is_played_move ? playedMarkerTone(analysis) : null,
      label: markerLabelForReviewedRank(candidate.review_rank),
    }));

    renderer.render({
      board: analysis.board_before,
      display_board_size: analysis.board_before.length,
      markers,
    });
  }

  function buildSuggestionDetails(suggestion, analysis) {
    if (isSwapAwareOpening(analysis)) {
      return [
        `${playerDisplayName('blue')} ${formatPercent(suggestion.blue_win_probability)}`,
        formatDistanceToEven(Number(suggestion.distance_to_even)),
        `policy rank ${String(suggestion.policy_rank)}`,
      ];
    }

    return [
      `${formatPercent(suggestion.win_probability_for_player)} for player`,
      `policy rank ${String(suggestion.policy_rank)}`,
      `${formatPercent(suggestion.policy_probability, 0)} policy`,
    ];
  }

  function buildSuggestionList(analysis) {
    const wrapper = createElement('section', 'review-alternatives');
    wrapper.appendChild(createElement('h3', 'review-section-heading', 'Top Reviewed Moves'));

    const intro = createElement(
      'p',
      'review-moment-copy',
      'The board shows three reviewed moves total. Numbers are reviewed ranks, and the played move is always included even when it ranked lower.'
    );
    wrapper.appendChild(intro);

    const reviewedMoves = buildDisplayedReviewedMoves(analysis);
    if (reviewedMoves.length === 0) {
      wrapper.appendChild(
        createElement(
          'p',
          'review-moment-copy',
          'No reviewed move ranking is available for this position.'
        )
      );
      return wrapper;
    }

    const list = createElement('div', 'review-suggestion-list');
    reviewedMoves.forEach((candidate) => {
      const detailHtml = buildSuggestionDetails(candidate, analysis)
        .map((detail) => `<span class="review-suggestion-detail">${escapeHtml(detail)}</span>`)
        .join('');
      const item = createElement('div', `review-suggestion-item${candidate.is_played_move ? ' is-played' : ''}`);
      const statusBadge = candidate.is_played_move
        ? '<span class="review-chip review-chip-soft">Played</span>'
        : '';
      item.innerHTML = `
        <span class="review-suggestion-index">${escapeHtml(formatReviewedRank(candidate.review_rank))}</span>
        <span class="review-suggestion-move">${escapeHtml(candidate.move_trmph)}</span>
        ${statusBadge}
        ${detailHtml}
      `;
      list.appendChild(item);
    });
    wrapper.appendChild(list);
    return wrapper;
  }

  function buildChartSvg(trajectory, selectedMoveNumber, visibleMoveNumbers) {
    if (!Array.isArray(trajectory) || trajectory.length === 0) {
      return '<div class="review-chart-empty">No trajectory data available.</div>';
    }

    const width = 920;
    const height = 260;
    const padding = { top: 22, right: 20, bottom: 28, left: 20 };
    const innerWidth = width - padding.left - padding.right;
    const innerHeight = height - padding.top - padding.bottom;
    const maxIndex = Math.max(1, trajectory.length - 1);
    const visibleSet = visibleMoveNumbers instanceof Set ? visibleMoveNumbers : null;

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
        const moveNumber = Number(item.move_number);
        const point = pointFor(item, index + 1, 'blue_win_probability');
        const isSelected = moveNumber === selectedMoveNumber;
        const isVisible = !visibleSet || visibleSet.has(moveNumber);
        const radius = isSelected ? 5.2 : (Number(item.loss) >= 0.08 ? 3.3 : 2.2);
        const fill = isSelected ? 'var(--review-chart-selected)' : 'var(--review-chart-dot)';
        const stroke = isSelected ? 'var(--review-chart-selected-stroke)' : 'transparent';
        const opacity = isVisible ? 0.9 : 0.22;
        return `
          <circle
            cx="${point.x.toFixed(2)}"
            cy="${point.y.toFixed(2)}"
            r="${radius}"
            fill="${fill}"
            stroke="${stroke}"
            stroke-width="${isSelected ? '2.1' : '0'}"
            opacity="${opacity}"
          ></circle>
        `;
      })
      .join('');

    const selectedTrajectoryItem = trajectory.find((item) => Number(item.move_number) === selectedMoveNumber);
    const selectedMarker = selectedTrajectoryItem
      ? (() => {
          const selectedIndex = trajectory.indexOf(selectedTrajectoryItem);
          const point = pointFor(selectedTrajectoryItem, selectedIndex, 'blue_win_probability');
          return `
            <line
              x1="${point.x.toFixed(2)}"
              y1="${padding.top}"
              x2="${point.x.toFixed(2)}"
              y2="${(height - padding.bottom).toFixed(2)}"
              stroke="var(--review-chart-selected-stroke)"
              stroke-width="1.4"
              stroke-dasharray="4 5"
              opacity="0.9"
            ></line>
            <text
              x="${point.x.toFixed(2)}"
              y="${(padding.top + 14).toFixed(2)}"
              text-anchor="middle"
              font-size="12"
              fill="var(--review-chart-selected-stroke)"
            >Move ${escapeHtml(String(selectedMoveNumber))}</text>
          `;
        })()
      : '';

    const startLabel = formatPercent(Number(trajectory[0].blue_win_probability), 0);
    const endLabel = formatPercent(Number(trajectory[trajectory.length - 1].blue_win_probability), 0);

    return `
      <svg viewBox="0 0 ${width} ${height}" aria-label="Blue win probability trajectory">
        <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
        <line x1="${padding.left}" y1="${midlineY}" x2="${width - padding.right}" y2="${midlineY}" stroke="var(--review-chart-midline)" stroke-width="1.4" stroke-dasharray="5 5"></line>
        ${selectedMarker}
        <path d="${bestPath}" fill="none" stroke="var(--review-chart-best)" stroke-width="2.6" stroke-dasharray="8 6" opacity="0.82"></path>
        <path d="${actualPath}" fill="none" stroke="var(--review-chart-actual)" stroke-width="3.2"></path>
        ${dots}
        ${labels}
        <text x="${padding.left}" y="${padding.top - 4}" font-size="12" fill="var(--text-secondary)">Start ${escapeHtml(startLabel)}</text>
        <text x="${width - padding.right}" y="${padding.top - 4}" text-anchor="end" font-size="12" fill="var(--text-secondary)">End ${escapeHtml(endLabel)}</text>
      </svg>
    `;
  }

  class ReviewExplorer {
    constructor(root, payload) {
      this.root = root;
      this.payload = payload || {};
      this.summary = this.payload.summary || {};
      this.analysisMetadata = this.payload.analysis_metadata || {};
      this.analyses = Array.isArray(this.payload.move_analyses) ? this.payload.move_analyses.slice() : [];
      this.analysisByMoveNumber = new Map(this.analyses.map((analysis) => [Number(analysis.move_number), analysis]));
      this.elements = {};
      this.moveButtons = new Map();

      const baseState = {
        showBlue: true,
        showRed: true,
        showMinor: false,
        showQuietMoves: false,
        activeTab: 'why',
        isPlaying: false,
        selectedMoveNumber: null,
        hoveredMoveNumber: null,
      };
      const initialVisible = this.getVisibleAnalyses(baseState);
      if (initialVisible.length === 0) {
        baseState.showQuietMoves = true;
      }
      this.defaultFilters = {
        showBlue: baseState.showBlue,
        showRed: baseState.showRed,
        showMinor: baseState.showMinor,
        showQuietMoves: baseState.showQuietMoves,
      };
      baseState.selectedMoveNumber = this.pickPreferredMoveNumber(this.getVisibleAnalyses(baseState));
      this.state = baseState;
      this.visibleAnalyses = [];
      this.playbackDelayMs = 2000;
      this.playTimer = null;
    }

    mount() {
      this.buildLayout();
      this.refreshFilteredUi();
    }

    getVisibleAnalyses(stateOverride) {
      const state = stateOverride || this.state;
      return this.analyses.filter((analysis) => {
        if (analysis.player === 'blue' && !state.showBlue) {
          return false;
        }
        if (analysis.player === 'red' && !state.showRed) {
          return false;
        }
        if (!state.showMinor && analysis.mistake_severity === 'minor' && !analysis.is_losing_move) {
          return false;
        }
        if (!state.showQuietMoves && isQuietAnalysis(analysis)) {
          return false;
        }
        return true;
      });
    }

    pickPreferredMoveNumber(visibleAnalyses) {
      if (!Array.isArray(visibleAnalyses) || visibleAnalyses.length === 0) {
        return this.analyses.length > 0 ? Number(this.analyses[0].move_number) : null;
      }
      const spotlightAnalyses = this.getSpotlightAnalyses(visibleAnalyses);
      if (spotlightAnalyses.length > 0) {
        return Number(spotlightAnalyses[0].move_number);
      }
      return Number(visibleAnalyses[0].move_number);
    }

    getSpotlightAnalyses(visibleAnalyses) {
      return visibleAnalyses
        .filter((analysis) => analysis.is_mistake || analysis.is_losing_move)
        .sort((left, right) => reviewLossValue(right) - reviewLossValue(left))
        .slice(0, 5);
    }

    getAnalysisByMoveNumber(moveNumber) {
      return this.analysisByMoveNumber.get(Number(moveNumber)) || null;
    }

    currentVisibleIndex() {
      if (this.visibleAnalyses.length === 0) {
        return -1;
      }
      const active = this.activeAnalysis();
      const moveNumber = active ? Number(active.move_number) : Number(this.state.selectedMoveNumber);
      const activeIndex = this.visibleAnalyses.findIndex((analysis) => Number(analysis.move_number) === moveNumber);
      if (activeIndex >= 0) {
        return activeIndex;
      }
      return 0;
    }

    moveToVisibleIndex(index, options) {
      const config = options || {};
      const target = this.visibleAnalyses[index];
      if (!target) {
        return false;
      }
      if (!config.keepPlaying) {
        this.stopPlayback({ refresh: false });
      }
      this.state.selectedMoveNumber = Number(target.move_number);
      this.state.hoveredMoveNumber = null;
      this.refreshActiveUi();
      return true;
    }

    jumpToBoundary(edge, options) {
      if (this.visibleAnalyses.length === 0) {
        return;
      }
      const targetIndex = edge === 'end' ? this.visibleAnalyses.length - 1 : 0;
      this.moveToVisibleIndex(targetIndex, options);
    }

    jumpRelative(offset, options) {
      if (this.visibleAnalyses.length === 0) {
        return;
      }
      const currentIndex = this.currentVisibleIndex();
      const nextIndex = Math.max(0, Math.min(this.visibleAnalyses.length - 1, currentIndex + offset));
      this.moveToVisibleIndex(nextIndex, options);
    }

    startPlayback() {
      if (this.state.isPlaying || this.visibleAnalyses.length === 0) {
        return;
      }
      if (this.currentVisibleIndex() >= this.visibleAnalyses.length - 1) {
        this.jumpToBoundary('start', { keepPlaying: true });
      }
      this.state.isPlaying = true;
      this.state.hoveredMoveNumber = null;
      this.refreshActiveUi();
      this.playTimer = global.setInterval(() => {
        if (!this.state.isPlaying) {
          return;
        }
        const currentIndex = this.currentVisibleIndex();
        if (currentIndex < 0 || currentIndex >= this.visibleAnalyses.length - 1) {
          this.stopPlayback();
          return;
        }
        this.moveToVisibleIndex(currentIndex + 1, { keepPlaying: true });
      }, this.playbackDelayMs);
    }

    stopPlayback(options) {
      const config = options || {};
      if (this.playTimer !== null) {
        global.clearInterval(this.playTimer);
        this.playTimer = null;
      }
      const wasPlaying = this.state.isPlaying;
      this.state.isPlaying = false;
      if (wasPlaying && config.refresh !== false) {
        this.refreshActiveUi();
      }
    }

    setActiveTab(tabName) {
      if (this.state.activeTab === tabName) {
        return;
      }
      this.state.activeTab = tabName;
      this.refreshActiveUi();
    }

    ensureSelectionIsVisible() {
      if (this.visibleAnalyses.length === 0) {
        this.state.selectedMoveNumber = null;
        this.state.hoveredMoveNumber = null;
        return;
      }

      const visibleNumbers = new Set(this.visibleAnalyses.map((analysis) => Number(analysis.move_number)));
      if (!visibleNumbers.has(Number(this.state.selectedMoveNumber))) {
        this.state.selectedMoveNumber = this.pickPreferredMoveNumber(this.visibleAnalyses);
      }
      if (!visibleNumbers.has(Number(this.state.hoveredMoveNumber))) {
        this.state.hoveredMoveNumber = null;
      }
    }

    activeAnalysis() {
      const hovered = this.getAnalysisByMoveNumber(this.state.hoveredMoveNumber);
      if (hovered && this.visibleAnalyses.some((analysis) => Number(analysis.move_number) === Number(hovered.move_number))) {
        return hovered;
      }
      const selected = this.getAnalysisByMoveNumber(this.state.selectedMoveNumber);
      if (selected && this.visibleAnalyses.some((analysis) => Number(analysis.move_number) === Number(selected.move_number))) {
        return selected;
      }
      return this.visibleAnalyses[0] || null;
    }

    buildLayout() {
      const results = createElement('div', 'review-results');

      const stage = createElement('section', 'review-stage-grid');

      const focusColumn = createElement('div', 'review-focus-column');
      this.elements.focusPanel = createElement('section', 'review-panel review-focus-panel');
      focusColumn.appendChild(this.elements.focusPanel);

      const railColumn = createElement('div', 'review-rail-column');

      const railPanel = createElement('section', 'review-panel review-rail-panel');
      const railHeader = createElement('div', 'review-rail-header');
      railHeader.appendChild(createElement('h2', 'review-panel-title', 'Mistakes And Moves'));
      this.elements.moveRailCount = createElement('p', 'review-rail-count');
      railHeader.appendChild(this.elements.moveRailCount);
      railPanel.appendChild(railHeader);
      this.elements.moveList = createElement('div', 'review-move-list');
      railPanel.appendChild(this.elements.moveList);

      railColumn.appendChild(railPanel);

      stage.appendChild(focusColumn);
      stage.appendChild(railColumn);
      results.appendChild(stage);

      this.elements.chartPanel = createElement('section', 'review-panel review-chart-panel');
      results.appendChild(this.elements.chartPanel);

      clearAndAppend(this.root, results);
    }

    buildFilterButton(label, key, extraClassName) {
      const button = createElement('button', `review-filter-toggle${extraClassName ? ` ${extraClassName}` : ''}`, label);
      button.type = 'button';
      button.addEventListener('click', () => {
        this.state[key] = !this.state[key];
        this.state.hoveredMoveNumber = null;
        this.refreshFilteredUi();
      });
      this.elements[`${key}Button`] = button;
      return button;
    }

    buildControlBar() {
      const panel = createElement('section', 'review-inline-filters');
      const top = createElement('div', 'review-control-top');
      top.appendChild(createElement('h3', 'review-section-heading', 'Show'));
      this.elements.controlStatus = createElement('p', 'review-control-status');
      top.appendChild(this.elements.controlStatus);
      panel.appendChild(top);

      const toggleRow = createElement('div', 'review-filter-row');
      toggleRow.appendChild(this.buildFilterButton(playerDisplayName('blue'), 'showBlue', 'player-blue'));
      toggleRow.appendChild(this.buildFilterButton(playerDisplayName('red'), 'showRed', 'player-red'));
      toggleRow.appendChild(this.buildFilterButton('Show Minor', 'showMinor'));
      toggleRow.appendChild(this.buildFilterButton('Include Quiet Moves', 'showQuietMoves'));
      panel.appendChild(toggleRow);

      const footer = createElement('div', 'review-control-footer');
      this.elements.controlHint = createElement(
        'p',
        'review-control-hint',
        'Hover a move on the right to preview it on the board. Click a move to pin it.'
      );
      footer.appendChild(this.elements.controlHint);

      const resetButton = createElement('button', 'review-filter-reset', 'Reset Filters');
      resetButton.type = 'button';
      resetButton.addEventListener('click', () => {
        this.state.showBlue = this.defaultFilters.showBlue;
        this.state.showRed = this.defaultFilters.showRed;
        this.state.showMinor = this.defaultFilters.showMinor;
        this.state.showQuietMoves = this.defaultFilters.showQuietMoves;
        this.state.hoveredMoveNumber = null;
        this.refreshFilteredUi();
      });
      footer.appendChild(resetButton);
      panel.appendChild(footer);

      return panel;
    }

    updateFilterButtons() {
      const buttonStates = [
        ['showBlueButton', this.state.showBlue],
        ['showRedButton', this.state.showRed],
        ['showMinorButton', this.state.showMinor],
        ['showQuietMovesButton', this.state.showQuietMoves],
      ];

      buttonStates.forEach(([elementKey, isActive]) => {
        const button = this.elements[elementKey];
        if (!button) {
          return;
        }
        button.classList.toggle('is-active', Boolean(isActive));
        button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      });
    }

    updateSummaryAndStatus() {
      const visibleMoves = this.visibleAnalyses.length;
      if (this.elements.controlStatus) {
        this.elements.controlStatus.textContent = `${visibleMoves} moves shown`;
      }
    }

    refreshFilteredUi() {
      this.visibleAnalyses = this.getVisibleAnalyses();
      this.ensureSelectionIsVisible();
      if (this.state.isPlaying && this.visibleAnalyses.length <= 1) {
        this.stopPlayback({ refresh: false });
      }
      const active = this.activeAnalysis();

      this.updateFilterButtons();
      this.updateSummaryAndStatus();
      this.renderMoveRail();
      this.renderFocusPanel(active);
      this.renderChart(active);
      this.syncActiveStates(active);
    }

    refreshActiveUi() {
      const active = this.activeAnalysis();
      this.renderFocusPanel(active);
      this.renderChart(active);
      this.syncActiveStates(active);
    }

    setHoveredMoveNumber(moveNumber) {
      if (this.state.isPlaying) {
        return;
      }
      if (this.state.hoveredMoveNumber === moveNumber) {
        return;
      }
      this.state.hoveredMoveNumber = moveNumber;
      this.refreshActiveUi();
    }

    clearHoveredMoveNumber(moveNumber) {
      if (moveNumber !== null && Number(this.state.hoveredMoveNumber) !== Number(moveNumber)) {
        return;
      }
      if (this.state.hoveredMoveNumber === null) {
        return;
      }
      this.state.hoveredMoveNumber = null;
      this.refreshActiveUi();
    }

    selectMoveNumber(moveNumber) {
      this.stopPlayback({ refresh: false });
      this.state.selectedMoveNumber = Number(moveNumber);
      this.state.hoveredMoveNumber = null;
      this.refreshActiveUi();
    }

    attachMoveInteractions(element, moveNumber) {
      element.addEventListener('mouseenter', () => {
        this.setHoveredMoveNumber(Number(moveNumber));
      });
      element.addEventListener('mouseleave', () => {
        this.clearHoveredMoveNumber(Number(moveNumber));
      });
      element.addEventListener('focus', () => {
        this.setHoveredMoveNumber(Number(moveNumber));
      });
      element.addEventListener('blur', () => {
        this.clearHoveredMoveNumber(Number(moveNumber));
      });
      element.addEventListener('click', () => {
        this.selectMoveNumber(Number(moveNumber));
      });
    }

    buildMoveCard(analysis) {
      const severity = severityClass(analysis);
      const button = createElement('button', `review-move-card ${severity}`);
      button.type = 'button';
      button.innerHTML = `
        <span class="review-move-card-top">
          <span class="review-move-card-title">#${escapeHtml(String(analysis.move_number))} ${escapeHtml(analysis.move_played_trmph)}</span>
          <span class="review-move-card-impact">${escapeHtml(formatImpact(analysis))}</span>
          <span class="review-severity-badge ${severity}">${escapeHtml(momentBadgeText(analysis))}</span>
        </span>
        <span class="review-move-card-subline">
          <span>${escapeHtml(playerDisplayName(analysis.player))}</span>
          <span>&middot;</span>
          <span>${escapeHtml(formatPhase(analysis.game_phase))}</span>
          <span>&middot;</span>
          <span>Best ${escapeHtml(analysis.best_move_trmph)}</span>
        </span>
        <span class="review-move-card-copy">${escapeHtml(analysis.mistake_reason || (isQuietAnalysis(analysis) ? 'No review threshold crossed for this move.' : ''))}</span>
      `;
      this.attachMoveInteractions(button, analysis.move_number);
      return button;
    }

    renderMoveRail() {
      this.moveButtons.clear();
      this.elements.moveList.innerHTML = '';
      const visibleCount = this.visibleAnalyses.length;
      const totalCount = this.analyses.length;
      this.elements.moveRailCount.textContent = visibleCount === totalCount
        ? `${totalCount} moves`
        : `${visibleCount} of ${totalCount} moves`;

      if (this.visibleAnalyses.length === 0) {
        const empty = createElement('div', 'review-empty review-filter-empty');
        empty.textContent = 'No moves match the current filters. Reset filters or re-enable a hidden category.';
        this.elements.moveList.appendChild(empty);
        return;
      }

      const fragment = document.createDocumentFragment();
      this.visibleAnalyses.forEach((analysis) => {
        const card = this.buildMoveCard(analysis);
        this.moveButtons.set(Number(analysis.move_number), card);
        fragment.appendChild(card);
      });
      this.elements.moveList.appendChild(fragment);
    }

    buildFocusStats(analysis) {
      const stats = [
        ['Player', playerDisplayName(analysis.player)],
        ['Phase', formatPhase(analysis.game_phase)],
        ['Played', analysis.move_played_trmph],
        ['Best', analysis.best_move_trmph],
        ['Reviewed Rank', formatReviewedRank(analysis.move_played_review_rank)],
        [impactLabel(analysis), formatImpact(analysis)],
        ['Review', analysis.review_metric_label || 'Win probability'],
        ['Policy Rank', String(analysis.move_played_policy_rank)],
        ['Candidates', `${String(analysis.candidate_move_count)} reviewed`],
        ['Legal Moves', String(analysis.legal_move_count)],
      ];

      if (isSwapAwareOpening(analysis)) {
        stats.push(['Played from 50%', formatDistanceToEven(Number(analysis.move_played_distance_to_even))]);
        stats.push(['Best from 50%', formatDistanceToEven(Number(analysis.best_move_distance_to_even))]);
      } else {
        stats.push(['Before', formatPercent(analysis.position_win_probability_for_player)]);
        stats.push(['After', formatPercent(analysis.move_played_win_probability_for_player)]);
      }
      return stats;
    }

    buildQuickFacts(analysis) {
      const facts = [
        ['Played', analysis.move_played_trmph],
        ['Best', analysis.best_move_trmph],
        ['Reviewed Rank', formatReviewedRank(analysis.move_played_review_rank)],
        [impactLabel(analysis), formatImpact(analysis)],
      ];

      if (isSwapAwareOpening(analysis)) {
        facts.push(['From 50%', formatDistanceToEven(Number(analysis.move_played_distance_to_even))]);
      } else {
        facts.push([
          'Swing',
          `${formatPercent(analysis.position_win_probability_for_player)} to ${formatPercent(analysis.move_played_win_probability_for_player)}`,
        ]);
      }
      return facts;
    }

    buildFocusSummary(analysis) {
      const section = createElement('section', 'review-focus-summary');
      const header = createElement('div', 'review-focus-header');
      const titleGroup = createElement('div', 'review-focus-title-group');
      titleGroup.appendChild(createElement('p', 'review-focus-kicker', `${playerDisplayName(analysis.player)} · ${formatPhase(analysis.game_phase)}`));
      titleGroup.appendChild(createElement('h2', 'review-focus-title', `Move ${analysis.move_number}: ${analysis.move_played_trmph}`));
      header.appendChild(titleGroup);

      const badges = createElement('div', 'review-focus-badges');
      const severity = severityClass(analysis);
      badges.appendChild(createElement('span', `review-severity-badge ${severity}`, momentBadgeText(analysis)));
      badges.appendChild(createElement('span', 'review-chip review-chip-soft', `Played rank ${formatReviewedRank(analysis.move_played_review_rank)}`));
      if (analysis.best_move_trmph === analysis.move_played_trmph) {
        badges.appendChild(createElement('span', 'review-chip review-chip-soft', 'Played matched best'));
      }
      header.appendChild(badges);

      section.appendChild(header);
      section.appendChild(
        createElement(
          'p',
          'review-focus-subtitle',
          'Hover a move to preview it. Click a move to keep it selected.'
        )
      );
      return section;
    }

    buildTransportButton(label, disabled, onClick, extraClassName) {
      const button = createElement('button', `review-transport-button${extraClassName ? ` ${extraClassName}` : ''}`, label);
      button.type = 'button';
      button.disabled = Boolean(disabled);
      button.addEventListener('click', onClick);
      return button;
    }

    buildTransportBar() {
      const transport = createElement('div', 'review-transport');
      const currentIndex = this.currentVisibleIndex();
      const visibleCount = this.visibleAnalyses.length;
      const atStart = currentIndex <= 0;
      const atEnd = currentIndex === -1 || currentIndex >= visibleCount - 1;

      const controls = createElement('div', 'review-transport-controls');
      controls.appendChild(this.buildTransportButton('Beginning', visibleCount === 0 || atStart, () => {
        this.jumpToBoundary('start');
      }));
      controls.appendChild(this.buildTransportButton('Prev', visibleCount === 0 || atStart, () => {
        this.jumpRelative(-1);
      }));
      controls.appendChild(this.buildTransportButton('Play', visibleCount <= 1 || this.state.isPlaying, () => {
        this.startPlayback();
      }, 'is-primary'));
      controls.appendChild(this.buildTransportButton('Stop', !this.state.isPlaying, () => {
        this.stopPlayback();
      }));
      controls.appendChild(this.buildTransportButton('Next', visibleCount === 0 || atEnd, () => {
        this.jumpRelative(1);
      }));
      controls.appendChild(this.buildTransportButton('End', visibleCount === 0 || atEnd, () => {
        this.jumpToBoundary('end');
      }));
      transport.appendChild(controls);

      const status = createElement('p', 'review-transport-status');
      if (visibleCount === 0) {
        status.textContent = 'No visible moves.';
      } else if (this.state.isPlaying) {
        status.textContent = `Playing visible moves every ${Math.round(this.playbackDelayMs / 1000)} seconds.`;
      } else if (currentIndex >= 0) {
        status.textContent = `Move ${currentIndex + 1} of ${visibleCount} in the current filtered set.`;
      } else {
        status.textContent = `${visibleCount} visible moves.`;
      }
      transport.appendChild(status);
      return transport;
    }

    buildFocusTabs() {
      const tabs = createElement('div', 'review-focus-tabs');
      [
        ['why', 'Why'],
        ['alternatives', 'Reviewed'],
        ['numbers', 'Numbers'],
      ].forEach(([key, label]) => {
        const button = createElement('button', `review-focus-tab${this.state.activeTab === key ? ' is-active' : ''}`, label);
        button.type = 'button';
        button.setAttribute('aria-pressed', this.state.activeTab === key ? 'true' : 'false');
        button.addEventListener('click', () => {
          this.setActiveTab(key);
        });
        tabs.appendChild(button);
      });
      return tabs;
    }

    buildFocusDetail(analysis) {
      const detail = createElement('div', 'review-focus-detail');
      const body = createElement('div', 'review-focus-detail-body');

      if (this.state.activeTab === 'alternatives') {
        body.appendChild(buildSuggestionList(analysis));
      } else if (this.state.activeTab === 'numbers') {
        const section = createElement('section', 'review-focus-detail-section');
        section.appendChild(createElement('h3', 'review-section-heading', 'Detailed Numbers'));
        const statGrid = createElement('div', 'review-focus-grid');
        this.buildFocusStats(analysis).forEach(([label, value]) => {
          statGrid.appendChild(buildStatCard(label, value));
        });
        section.appendChild(statGrid);
        body.appendChild(section);
      } else {
        const explanation = createElement('section', 'review-focus-detail-section');
        explanation.appendChild(createElement('h3', 'review-section-heading', 'What Changed'));
        explanation.appendChild(createElement('p', 'review-moment-copy', analysis.mistake_reason));
        if (isSwapAwareOpening(analysis)) {
          explanation.appendChild(
            createElement(
              'p',
              'review-focus-footnote',
              'Move 1 uses the swap-aware balance metric, so impact is measured by distance from an even 50% opening rather than raw first-player value.'
            )
          );
        }
        body.appendChild(explanation);
      }

      detail.appendChild(this.buildFocusTabs());
      detail.appendChild(body);
      return detail;
    }

    buildEmptyFocusState() {
      const fragment = document.createDocumentFragment();

      const boardShell = createElement('div', 'review-board-shell review-focus-board-shell');
      const emptyBoard = createElement('div', 'review-board review-focus-board review-board-empty');
      emptyBoard.appendChild(
        createElement(
          'p',
          'review-board-empty-message',
          'No move is available to preview under the current filters.'
        )
      );
      boardShell.appendChild(emptyBoard);
      fragment.appendChild(boardShell);

      fragment.appendChild(this.buildTransportBar());
      fragment.appendChild(this.buildControlBar());

      const emptyDetail = createElement('div', 'review-empty review-filter-empty');
      emptyDetail.textContent = 'Adjust the filters below to bring moves back into the preview.';
      fragment.appendChild(emptyDetail);

      return fragment;
    }

    renderFocusPanel(activeAnalysis) {
      this.elements.focusPanel.innerHTML = '';

      if (!activeAnalysis) {
        this.elements.focusPanel.appendChild(this.buildEmptyFocusState());
        this.updateFilterButtons();
        this.updateSummaryAndStatus();
        return;
      }

      const boardShell = createElement('div', 'review-board-shell review-focus-board-shell');
      const board = createElement('div', 'review-board review-focus-board');
      boardShell.appendChild(board);
      boardShell.appendChild(buildBoardLegend());
      this.elements.focusPanel.appendChild(boardShell);
      renderBoard(board, activeAnalysis);

      this.elements.focusPanel.appendChild(this.buildTransportBar());
      this.elements.focusPanel.appendChild(this.buildControlBar());
      this.updateFilterButtons();
      this.updateSummaryAndStatus();
      this.elements.focusPanel.appendChild(this.buildFocusSummary(activeAnalysis));

      const quickGrid = createElement('div', 'review-focus-quick-grid');
      this.buildQuickFacts(activeAnalysis).forEach(([label, value]) => {
        quickGrid.appendChild(buildStatCard(label, value));
      });
      this.elements.focusPanel.appendChild(quickGrid);

      this.elements.focusPanel.appendChild(this.buildFocusDetail(activeAnalysis));
    }

    renderChart(activeAnalysis) {
      this.elements.chartPanel.innerHTML = '';
      this.elements.chartPanel.appendChild(createElement('h2', 'review-panel-title', 'Game Arc'));

      const selectionText = activeAnalysis
        ? `Move ${activeAnalysis.move_number} is highlighted on the chart. Hidden moves stay on the line but are faded.`
        : 'All moves are shown on the trajectory.';
      this.elements.chartPanel.appendChild(createElement('p', 'review-chart-selection', selectionText));

      const frame = createElement('div', 'review-chart-frame');
      const visibleMoveNumbers = new Set(this.visibleAnalyses.map((analysis) => Number(analysis.move_number)));
      frame.innerHTML = buildChartSvg(
        this.payload.win_probability_trajectory || [],
        activeAnalysis ? Number(activeAnalysis.move_number) : null,
        visibleMoveNumbers
      );
      this.elements.chartPanel.appendChild(frame);
      this.elements.chartPanel.appendChild(
        createElement(
          'p',
          'review-chart-note',
          'Solid line shows the played game. Dashed line shows the strongest reviewed alternative for each move. Move 1 still uses the swap-aware opening measure.'
        )
      );
    }

    syncActiveStates(activeAnalysis) {
      const activeMoveNumber = activeAnalysis ? Number(activeAnalysis.move_number) : null;
      const selectedMoveNumber = Number(this.state.selectedMoveNumber);

      this.moveButtons.forEach((button, moveNumber) => {
        const isActive = Number(moveNumber) === activeMoveNumber;
        const isSelected = Number(moveNumber) === selectedMoveNumber;
        button.classList.toggle('is-active', isActive);
        button.classList.toggle('is-selected', isSelected);
        button.setAttribute('aria-pressed', isSelected ? 'true' : 'false');
      });
    }
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
    const explorer = new ReviewExplorer(root, payload);
    explorer.mount();
  }

  function mountStandalone(root, payload) {
    const layout = createElement('div', 'review-page-layout');
    const content = createElement('div');
    layout.appendChild(content);
    clearAndAppend(root, layout);
    applyStoredThemePreferences();
    const explorer = new ReviewExplorer(content, payload);
    explorer.mount();
  }

  global.HexGameReviewUi = {
    applyStoredThemePreferences,
    mount,
    mountStandalone,
    renderError,
    renderLoading,
  };
})(window);
