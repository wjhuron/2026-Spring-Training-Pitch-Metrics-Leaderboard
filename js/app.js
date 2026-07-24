(function () {
  let currentTab = 'pitcherStats';
  let currentSection = 'pitchers';  // 'home', 'pitchers', 'hitters'
  let viewMode = 'player'; // 'player' | 'team'
  let selectedPitchTypes = []; // empty = all; or array of selected types
  let allData = []; // current filtered + sorted data (full, before pagination)
  let columnRangeFilters = {}; // { colKey: { min: number|null, max: number|null } }

  const TAB_SECTION = {
    pitcherStats: 'pitchers', pitchMetrics: 'pitchers',
    pitcherBattedBall: 'pitchers', pitcherSwingDecisions: 'pitchers',
    hitterStats: 'hitters', hitterBattedBall: 'hitters',
    hitterSwingDecisions: 'hitters', hitterBatTracking: 'hitters',
    hitterPitch: 'hitters'
  };

  const TAB_DATA = {
    pitcherStats: 'pitcher', pitchMetrics: 'pitch',
    pitcherBattedBall: 'pitcher', pitcherSwingDecisions: 'pitcher',
    hitterStats: 'hitter', hitterBattedBall: 'hitter',
    hitterSwingDecisions: 'hitter', hitterBatTracking: 'hitter',
    hitterPitch: 'hitterPitch'
  };

  const TAB_ROUTE = {
    pitcherStats: 'pitchers/stats', pitchMetrics: 'pitchers/pitch-metrics',
    pitcherBattedBall: 'pitchers/batted-ball', pitcherSwingDecisions: 'pitchers/plate-discipline',
    hitterStats: 'hitters/stats', hitterBattedBall: 'hitters/batted-ball',
    hitterSwingDecisions: 'hitters/plate-discipline', hitterBatTracking: 'hitters/bat-tracking',
    hitterPitch: 'hitters/pitch-type'
  };

  const ROUTE_TAB = {};
  Object.keys(TAB_ROUTE).forEach(function (t) { ROUTE_TAB[TAB_ROUTE[t]] = t; });

  const OLD_TAB_MAP = {
    pitcher: 'pitcherStats', pitch: 'pitchMetrics',
    hitterStats: 'hitterStats', hitterBattedBall: 'hitterBattedBall',
    hitterSwingDecisions: 'hitterSwingDecisions', hitterPitch: 'hitterPitch'
  };

  const PITCH_TYPE_TABS = {
    pitchMetrics: true, hitterPitch: true,
    pitcherBattedBall: true, pitcherSwingDecisions: true,
    hitterBattedBall: true, hitterSwingDecisions: true,
    hitterBatTracking: true
  };

  // Per-tab default sort. Falls back to alphabetical by player name.
  const TAB_DEFAULT_SORT = {
    pitcherStats:          { key: 'kbbPct',     dir: 'desc' },
    pitchMetrics:          { key: 'xRv100',     dir: 'desc' },
    pitcherSwingDecisions: { key: 'swStrPct',   dir: 'desc' },
    pitcherBattedBall:     { key: 'xwOBAcon',   dir: 'asc'  },
    hitterStats:           { key: 'hitterPlus', dir: 'desc' },
    hitterSwingDecisions:  { key: 'ctPlus',     dir: 'desc' },
    hitterBattedBall:      { key: 'bbPlus',     dir: 'desc' },
    hitterBatTracking:     { key: 'batSpeed',   dir: 'desc' },
    hitterPitch:           { key: 'xRv100',     dir: 'desc' }
  };

  function defaultSortFor(tab) {
    const def = TAB_DEFAULT_SORT[tab];
    if (def) return { key: def.key, dir: def.dir };
    return { key: isHitterTab(tab) ? 'hitter' : 'pitcher', dir: 'asc' };
  }

  function isHitterTab(tab) {
    return TAB_SECTION[tab] === 'hitters';
  }

  function isPitcherTab(tab) {
    return TAB_SECTION[tab] === 'pitchers';
  }

  let teamSelect, throwsSelect, vsHandSelect, minCountInput, minSwingsInput, searchInput;
  let minIpInput, minTbfInput, minBipInput, minPitcherSwingsInput;
  let dateStartInput, dateEndInput;
  let sidePanel, panelOverlay, panelClose;

  // ?roc=1 URL parameter enables hidden ROC team filter
  const rocMode = (new URLSearchParams(window.location.search)).get('roc') === '1';
  window._rocMode = rocMode;

  function init() {
    DataStore.load().then(function () {
      return Aggregator.load(DataStore.active().microData);
    }).then(function () {
      if (!DataStore.metadata) {
        document.getElementById('no-results').textContent = 'Failed to load data.';
        document.getElementById('no-results').style.display = '';
        return;
      }
      setupDOM();
      setupFilters();
      setupTabs();
      setupToolbar();
      setupPagination();
      setupSidePanel();
      setupCompareModal();
      setupPercentileTooltips();
      setupKeyboardNav();
      setupColumnSettings();
      setupRangeFilters();

      const pitcherCount = document.getElementById('home-pitcher-count');
      const hitterCount = document.getElementById('home-hitter-count');
      const rocTeamsInit = (DataStore.metadata.rocTeams || []);
      // Homepage counts = distinct players with at least one MLB appearance.
      // ROC/AAA-only players are excluded; a player with both MLB and AAA
      // stints counts once, and a traded player's per-team + 2TM/3TM aggregate
      // rows collapse to a single mlbId (no double counting).
      function countDistinctMlbPlayers(rows, nameKey) {
        const seen = Object.create(null);
        for (let i = 0; i < rows.length; i++) {
          const r = rows[i];
          if (rocTeamsInit.indexOf(r.team) !== -1) continue;   // skip AAA rows
          const key = (r.mlbId != null) ? ('id:' + r.mlbId) : ('nm:' + (r[nameKey] || i));
          seen[key] = true;
        }
        return Object.keys(seen).length;
      }
      if (pitcherCount && DataStore.pitcherData) {
        pitcherCount.textContent = countDistinctMlbPlayers(DataStore.pitcherData, 'pitcher') + ' pitchers';
      }
      if (hitterCount && DataStore.hitterData) {
        hitterCount.textContent = countDistinctMlbPlayers(DataStore.hitterData, 'hitter') + ' hitters';
      }

      document.getElementById('player-back').addEventListener('click', function () {
        PlayerPage.close();
      });

      window.addEventListener('hashchange', function () {
        handleRoute();
      });

      handleRoute();


      var overlay = document.getElementById('loading-overlay');
      if (overlay) overlay.remove();
    }).catch(function (err) {
      console.error('Failed to load data:', err);
      document.getElementById('no-results').textContent = 'Error loading data. Please refresh.';
      document.getElementById('no-results').style.display = '';
      var overlay = document.getElementById('loading-overlay');
      if (overlay) overlay.remove();
    });
  }

  function handleRoute() {
    const hash = window.location.hash.replace(/^#/, '');

    if (hash.indexOf('player=') === 0) {
      const mlbId = hash.split('=')[1];
      if (mlbId && /^\d+$/.test(mlbId) && !PlayerPage.isOpen) {
        var preferTeam = teamSelect ? teamSelect.value : null;
        if (preferTeam === 'all') preferTeam = null;
        PlayerPage.open(mlbId, preferTeam);
      }
      return;
    }

    if (PlayerPage.isOpen) {
      PlayerPage.close();
    }

    if (!hash || hash === 'home') {
      showHome();
      return;
    }

    // Backward compat: old format like "tab=pitcher&team=NYY"
    if (hash.indexOf('tab=') !== -1) {
      const oldParams = Utils.readHash();
      const newTab = OLD_TAB_MAP[oldParams.tab] || oldParams.tab || 'pitcherStats';
      if (oldParams.team) teamSelect.value = oldParams.team;
      if (oldParams.throws) throwsSelect.value = oldParams.throws;
      if (oldParams.vsHand) vsHandSelect.value = oldParams.vsHand;
      if (oldParams.min) minCountInput.value = oldParams.min;
      if (oldParams.search) searchInput.value = oldParams.search;
      if (oldParams.sort) Leaderboard.currentSort.key = oldParams.sort;
      if (oldParams.dir) Leaderboard.currentSort.dir = oldParams.dir;
      if (oldParams.page) Leaderboard.currentPage = parseInt(oldParams.page, 10) || 1;
      navigateToTab(newTab);
      return;
    }

    const parts = hash.split('?');
    let routePart = parts[0];
    if (routePart === 'pitchers/swing-decisions') routePart = 'pitchers/plate-discipline';
    if (routePart === 'hitters/swing-decisions') routePart = 'hitters/plate-discipline';
    if (routePart === 'abs' || routePart.indexOf('abs/') === 0) {
      const av = routePart.split('/')[1] || 'leaders';
      showAbs(['leaders', 'matrix', 'film'].indexOf(av) >= 0 ? av : 'leaders', true);
      return;
    }
    const tab = ROUTE_TAB[routePart];
    if (tab) {
      if (parts[1]) {
        const qp = {};
        parts[1].split('&').forEach(function (p) {
          const kv = p.split('=');
          qp[kv[0]] = decodeURIComponent(kv[1] || '');
        });
        function _setIfValid(sel, val) {
          for (var oi = 0; oi < sel.options.length; oi++) {
            if (sel.options[oi].value === val) { sel.value = val; return; }
          }
        }
        viewMode = (qp.view === 'team' && Aggregator.loaded) ? 'team' : 'player';
        if (qp.team) _setIfValid(teamSelect, qp.team);
        if (qp.throws) _setIfValid(throwsSelect, qp.throws);
        if (qp.vsHand) _setIfValid(vsHandSelect, qp.vsHand);
        if (qp.min) _setIfValid(minCountInput, qp.min);
        if (qp.search) searchInput.value = qp.search.substring(0, 50);
        if (qp.dateStart && /^\d{4}-\d{2}-\d{2}$/.test(qp.dateStart)) dateStartInput.value = qp.dateStart;
        if (qp.dateEnd && /^\d{4}-\d{2}-\d{2}$/.test(qp.dateEnd)) dateEndInput.value = qp.dateEnd;
        var urlState = {};
        if (qp.sort) urlState.sortKey = qp.sort;
        if (qp.dir) urlState.sortDir = qp.dir;
        if (qp.page) urlState.page = parseInt(qp.page, 10) || 1;
        if (qp.pitch) urlState.pitchTypes = qp.pitch.split(',');
        if (qp.role) document.getElementById('role-filter').value = qp.role;
      }
      navigateToTab(tab, true, urlState);  // true = don't push hash (already there)
    } else {
      showHome();
    }
  }

  function showHome() {
    currentSection = 'home';
    document.getElementById('home-page').style.display = '';
    document.querySelector('.controls').style.display = 'none';
    document.querySelector('.toolbar').style.display = 'none';
    document.querySelector('.table-wrapper').style.display = 'none';
    document.getElementById('pagination').style.display = 'none';
    const banner = document.getElementById('tab-banner');
    if (banner) banner.style.display = 'none';
    const pctlLeg = document.getElementById('pctl-legend');
    if (pctlLeg) pctlLeg.style.display = 'none';

    document.querySelectorAll('.section-tab').forEach(function (t) { t.classList.remove('active'); });
    const homeBtn = document.querySelector('.section-tab[data-section="home"]');
    if (homeBtn) homeBtn.classList.add('active');

    document.getElementById('pitcher-subtabs').style.display = 'none';
    document.getElementById('hitter-subtabs').style.display = 'none';
    hideAbs();
  }

  function hideAbs() {
    const ap = document.getElementById('abs-page');
    if (ap) ap.style.display = 'none';
    const as = document.getElementById('abs-subtabs');
    if (as) as.style.display = 'none';
  }

  function showAbs(view, skipHash) {
    currentSection = 'abs';
    if (!skipHash) window.location.hash = 'abs/' + (view || 'leaders');
    // hide the pitch-data leaderboard UI and home
    document.getElementById('home-page').style.display = 'none';
    document.querySelector('.controls').style.display = 'none';
    document.querySelector('.toolbar').style.display = 'none';
    document.querySelector('.table-wrapper').style.display = 'none';
    document.getElementById('pagination').style.display = 'none';
    const banner = document.getElementById('tab-banner'); if (banner) banner.style.display = 'none';
    const pctlLeg = document.getElementById('pctl-legend'); if (pctlLeg) pctlLeg.style.display = 'none';
    document.getElementById('pitcher-subtabs').style.display = 'none';
    document.getElementById('hitter-subtabs').style.display = 'none';
    document.getElementById('abs-page').style.display = '';
    document.getElementById('abs-subtabs').style.display = '';
    document.querySelectorAll('.section-tab').forEach(function (t) { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
    const sb = document.querySelector('.section-tab[data-section="abs"]');
    if (sb) { sb.classList.add('active'); sb.setAttribute('aria-selected', 'true'); }
    document.querySelectorAll('.abs-tab').forEach(function (t) {
      const on = t.getAttribute('data-absview') === (view || 'leaders');
      t.classList.toggle('active', on); t.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    if (window.ABS) { ABS.setView(view || 'leaders'); ABS.render(); }
  }

  function showLeaderboard() {
    document.getElementById('home-page').style.display = 'none';
    hideAbs();
    document.querySelector('.controls').style.display = '';
    document.querySelector('.toolbar').style.display = '';
    document.querySelector('.table-wrapper').style.display = '';
    document.getElementById('pagination').style.display = '';
    const pctlLeg = document.getElementById('pctl-legend');
    if (pctlLeg) pctlLeg.style.display = '';
  }

  var _navigating = false;
  function navigateToTab(tab, skipHash, urlState) {
    if (_navigating) return;
    _navigating = true;
    try {
    currentTab = tab;
    currentSection = TAB_SECTION[tab];

    if (!skipHash) {
      window.location.hash = TAB_ROUTE[tab];
    }

    showLeaderboard();

    // Update section tabs + ARIA
    document.querySelectorAll('.section-tab').forEach(function (t) {
      t.classList.remove('active');
      t.setAttribute('aria-selected', 'false');
    });
    const sectionBtn = document.querySelector('.section-tab[data-section="' + currentSection + '"]');
    if (sectionBtn) { sectionBtn.classList.add('active'); sectionBtn.setAttribute('aria-selected', 'true'); }

    document.getElementById('pitcher-subtabs').style.display = currentSection === 'pitchers' ? '' : 'none';
    document.getElementById('hitter-subtabs').style.display = currentSection === 'hitters' ? '' : 'none';

    document.querySelectorAll('.nav-subtabs .tab').forEach(function (t) {
      t.classList.remove('active');
      t.setAttribute('aria-selected', 'false');
    });
    const activeTab = document.querySelector('.tab[data-tab="' + tab + '"]');
    if (activeTab) { activeTab.classList.add('active'); activeTab.setAttribute('aria-selected', 'true'); }

    Leaderboard.initHiddenColumns(currentTab);
    Leaderboard.currentSort = defaultSortFor(currentTab);
    Leaderboard.currentPage = 1;
    Leaderboard.keyboardFocusIndex = -1;

    document.getElementById('pitch-type-filter-group').style.display =
      PITCH_TYPE_TABS[currentTab] ? '' : 'none';

    document.getElementById('min-swings-filter-group').style.display =
      (isHitterTab(currentTab) && currentTab !== 'hitterPitch' && currentTab !== 'hitterBattedBall') ? '' : 'none';

    document.getElementById('min-ip-filter-group').style.display =
      currentTab === 'pitcherStats' ? '' : 'none';
    document.getElementById('min-tbf-filter-group').style.display =
      currentTab === 'pitcherStats' ? '' : 'none';
    document.getElementById('min-bip-filter-group').style.display =
      (currentTab === 'pitcherBattedBall' || currentTab === 'hitterBattedBall') ? '' : 'none';
    document.getElementById('min-pitcher-swings-filter-group').style.display =
      currentTab === 'pitcherSwingDecisions' ? '' : 'none';

    const isPitcherTab = currentTab === 'pitcherStats' || currentTab === 'pitchMetrics' ||
                       currentTab === 'pitcherBattedBall' || currentTab === 'pitcherSwingDecisions';
    document.getElementById('role-filter-group').style.display = isPitcherTab ? '' : 'none';

    // Hide Min Pitches on pitcherStats (uses Min IP / Min TBF instead)
    document.getElementById('min-count').parentElement.style.display =
      currentTab === 'pitcherStats' ? 'none' : '';

    selectedPitchTypes = [];
    if (currentTab === 'hitterPitch') {
      buildHitterPitchChips(false);
    } else if (PITCH_TYPE_TABS[currentTab] && isHitterTab(currentTab)) {
      buildHitterPitchChips(false);
    } else if (PITCH_TYPE_TABS[currentTab]) {
      buildPitchChipsWithAll();
    } else {
      buildPitchChips();
    }

    const throwsLabel = document.querySelector('#throws-filter-group label');
    if (throwsLabel) {
      throwsLabel.textContent = isHitterTab(currentTab) ? 'Bats' : 'Throws';
    }
    const minCountLabel = document.querySelector('[for="min-count"]');
    const isMinPA = isHitterTab(currentTab) && currentTab !== 'hitterPitch';
    if (minCountLabel) {
      minCountLabel.textContent = isMinPA ? 'Min PA' : 'Min Pitches';
    }
    const savedVal = minCountInput.value;
    const paOpts = [['Q','Qualified'],['1','1'],['10','10'],['25','25'],['50','50'],['75','75'],['100','100'],['150','150']];
    const pitchOpts = [['Q','Qualified'],['1','1'],['10','10'],['25','25'],['50','50'],['100','100'],['150','150'],['200','200']];
    const opts = isMinPA ? paOpts : pitchOpts;
    minCountInput.innerHTML = '';
    for (let oi = 0; oi < opts.length; oi++) {
      const o = document.createElement('option');
      o.value = opts[oi][0]; o.textContent = opts[oi][1];
      minCountInput.appendChild(o);
    }
    let found = false;
    for (let oi2 = 0; oi2 < minCountInput.options.length; oi2++) {
      if (minCountInput.options[oi2].value === savedVal) { found = true; break; }
    }
    minCountInput.value = found ? savedVal : 'Q';
    updateVsHandLabels();
    searchInput.placeholder = isHitterTab(currentTab) ? 'Hitter name...' : 'Pitcher name...';

    document.getElementById('compare-btn').style.display = isHitterTab(currentTab) ? 'none' : '';

    const banner = document.getElementById('tab-banner');
    if (banner) {
      banner.style.display = 'none';
    }

    columnRangeFilters = {};
    updateRangeFilterBadge();

    if (urlState) {
      if (urlState.sortKey) Leaderboard.currentSort.key = urlState.sortKey;
      if (urlState.sortDir) Leaderboard.currentSort.dir = urlState.sortDir;
      if (urlState.page) Leaderboard.currentPage = urlState.page;
      if (urlState.pitchTypes && urlState.pitchTypes.length) {
        selectedPitchTypes = urlState.pitchTypes;
        updateHitterPitchChipVisuals();
      }
    }

    applyViewModeUI();

    refresh();
    scrollTableToTop();
    } finally {
      _navigating = false;
    }
  }

  // Sync toggle buttons + filter-control visibility with the player/team view.
  // Re-applies the tab-specific rules for the controls that team view hides,
  // so flipping back to player view restores them without a tab change.
  function applyViewModeUI() {
    const team = viewMode === 'team';
    document.querySelectorAll('#view-mode-toggle .view-mode-btn').forEach(function (b) {
      const active = b.getAttribute('data-mode') === viewMode;
      b.classList.toggle('active', active);
      b.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    document.getElementById('compare-btn').style.display =
      (team || isHitterTab(currentTab)) ? 'none' : '';
    document.getElementById('min-count').parentElement.style.display =
      (team || currentTab === 'pitcherStats') ? 'none' : '';
    document.getElementById('min-swings-filter-group').style.display =
      (!team && isHitterTab(currentTab) && currentTab !== 'hitterPitch' && currentTab !== 'hitterBattedBall') ? '' : 'none';
    document.getElementById('min-ip-filter-group').style.display =
      (!team && currentTab === 'pitcherStats') ? '' : 'none';
    document.getElementById('min-tbf-filter-group').style.display =
      (!team && currentTab === 'pitcherStats') ? '' : 'none';
    document.getElementById('min-bip-filter-group').style.display =
      (!team && (currentTab === 'pitcherBattedBall' || currentTab === 'hitterBattedBall')) ? '' : 'none';
    document.getElementById('min-pitcher-swings-filter-group').style.display =
      (!team && currentTab === 'pitcherSwingDecisions') ? '' : 'none';
    searchInput.placeholder = team ? 'Team...' :
      (isHitterTab(currentTab) ? 'Hitter name...' : 'Pitcher name...');
  }

  function setupDOM() {
    teamSelect = document.getElementById('team-filter');
    throwsSelect = document.getElementById('throws-filter');
    vsHandSelect = document.getElementById('vs-hand-filter');
    minCountInput = document.getElementById('min-count');
    minSwingsInput = document.getElementById('min-swings');
    minIpInput = document.getElementById('min-ip');
    minTbfInput = document.getElementById('min-tbf');
    minBipInput = document.getElementById('min-bip');
    minPitcherSwingsInput = document.getElementById('min-pitcher-swings');
    searchInput = document.getElementById('search-input');
    dateStartInput = document.getElementById('date-start');
    dateEndInput = document.getElementById('date-end');
    sidePanel = document.getElementById('side-panel');
    panelOverlay = document.getElementById('panel-overlay');
    panelClose = document.getElementById('panel-close');

    const tableContainer = document.getElementById('table-container');
    if (tableContainer) {
      tableContainer.addEventListener('scroll', function () {
        if (tableContainer.scrollLeft > 0) {
          tableContainer.classList.add('scrolled-h');
        } else {
          tableContainer.classList.remove('scrolled-h');
        }
      });
    }
  }

  function rebuildTeamDropdown() {
    teamSelect.innerHTML = '<option value="all">All Teams</option>';
    const rocTeams = (DataStore.metadata.rocTeams || []);
    DataStore.metadata.teams.forEach(function(team) {
      if (team === 'WBC') return;
      // Hide ROC/AAA teams unless ?roc=1 URL parameter is present
      if (rocTeams.indexOf(team) !== -1 && !window._rocMode) return;
      const opt = document.createElement('option');
      opt.value = team;
      opt.textContent = team;
      teamSelect.appendChild(opt);
    });
  }

  function setupFilters() {
    rebuildTeamDropdown();

    buildPitchChips();

    const genDate = document.getElementById('generated-date');
    if (genDate) genDate.textContent = DataStore.metadata.generatedAt;
    const freshness = document.getElementById('data-freshness');
    if (freshness && DataStore.metadata.generatedAt) {
      freshness.textContent = '| Updated ' + DataStore.metadata.generatedAt;
    }

    // Set date range min/max from micro data dates
    if (Aggregator.loaded && Aggregator.data && Aggregator.data.lookups.dates.length > 0) {
      const dates = Aggregator.data.lookups.dates;
      dateStartInput.min = dates[0];
      dateStartInput.max = dates[dates.length - 1];
      dateEndInput.min = dates[0];
      dateEndInput.max = dates[dates.length - 1];
    }

    // Disable vs-hand and date filters if micro data not available
    if (!Aggregator.loaded) {
      vsHandSelect.disabled = true;
      vsHandSelect.title = 'Requires micro data (run process_data.py)';
      dateStartInput.disabled = true;
      dateEndInput.disabled = true;
      dateStartInput.title = 'Requires micro data';
      dateEndInput.title = 'Requires micro data';
      // Team aggregation also needs micro data
      const vmGroup = document.getElementById('view-mode-group');
      if (vmGroup) vmGroup.style.display = 'none';
      viewMode = 'player';
    }

    document.querySelectorAll('#view-mode-toggle .view-mode-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        const mode = btn.getAttribute('data-mode');
        if (mode === viewMode) return;
        viewMode = mode;
        Leaderboard.currentPage = 1;
        Leaderboard.keyboardFocusIndex = -1;
        applyViewModeUI();
        refresh();
      });
    });

    teamSelect.addEventListener('change', function () { Leaderboard.currentPage = 1; refresh(); });
    throwsSelect.addEventListener('change', function () { Leaderboard.currentPage = 1; refresh(); });
    vsHandSelect.addEventListener('change', function () { Leaderboard.currentPage = 1; refresh(); });
    minCountInput.addEventListener('change', function () { Leaderboard.currentPage = 1; refresh(); });
    minSwingsInput.addEventListener('input', function () { Leaderboard.currentPage = 1; refresh(); });
    minIpInput.addEventListener('change', function () { Leaderboard.currentPage = 1; refresh(); });
    minTbfInput.addEventListener('input', function () { Leaderboard.currentPage = 1; refresh(); });
    minBipInput.addEventListener('input', function () { Leaderboard.currentPage = 1; refresh(); });
    minPitcherSwingsInput.addEventListener('input', function () { Leaderboard.currentPage = 1; refresh(); });
    document.getElementById('role-filter').addEventListener('change', function () { Leaderboard.currentPage = 1; refresh(); });
    dateStartInput.addEventListener('change', function () { Leaderboard.currentPage = 1; refresh(); });
    dateEndInput.addEventListener('change', function () { Leaderboard.currentPage = 1; refresh(); });

    let searchTimer = null;
    searchInput.addEventListener('input', function () {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(function () { Leaderboard.currentPage = 1; refresh(); }, 200);
    });
  }

  // Standard pitch type ordering used across all tabs
  const PITCH_TYPE_ORDER = [
    'FF', 'SI', '|',
    'FC', 'SL', 'ST', 'CU', 'SV', '|',
    'CH', 'FS', 'KN'
  ];

  // Ordered chip list for hitterPitch tab (adds All + categories before the standard order)
  const HITTER_PITCH_CHIP_ORDER = [
    'All', '|',
    'Hard', 'Breaking', 'Offspeed', '|'
  ].concat(PITCH_TYPE_ORDER);

  // Category chip colors
  // Shared chip creation helper — avoids duplicating DOM logic across 3 builders
  function _createChip(item, onClick, isSelected) {
    const color = (item === 'All' ? '#888' : Utils.CATEGORY_COLORS[item]) || Utils.getPitchColor(item);
    const btn = document.createElement('button');
    btn.className = 'pitch-chip';
    btn.textContent = item;
    btn.setAttribute('data-pitch', item);
    btn.style.setProperty('--chip-bg', color);
    btn.style.borderColor = color;
    if (item === 'SI' || item === 'SV') btn.style.color = '';
    if (Utils.PITCH_TYPE_LABELS[item]) btn.title = Utils.pitchTypeLabel(item);

    if (isSelected) {
      btn.classList.add('selected');
      btn.style.backgroundColor = color;
      btn.style.borderColor = 'transparent';
    }

    btn.addEventListener('click', function () {
      onClick(item, btn);
      Leaderboard.currentPage = 1;
      refresh();
    });
    return btn;
  }

  function _appendDivider(container) {
    const divider = document.createElement('span');
    divider.className = 'chip-divider';
    container.appendChild(divider);
  }

  function buildPitchChips() {
    const container = document.getElementById('pitch-type-chips');
    container.innerHTML = '';
    const available = DataStore.metadata.pitchTypes;

    PITCH_TYPE_ORDER.forEach(function (item) {
      if (item === '|') { _appendDivider(container); return; }
      if (available.indexOf(item) === -1) return;
      container.appendChild(_createChip(item, togglePitchChip, false));
    });
  }

  function buildPitchChipsWithAll() {
    const container = document.getElementById('pitch-type-chips');
    container.innerHTML = '';
    const available = DataStore.metadata.pitchTypes;
    selectedPitchTypes = ['All'];

    container.appendChild(_createChip('All', function (item) { toggleHitterPitchChip(item); }, true));
    _appendDivider(container);

    // Category chips combine member pitch types into one row per pitcher/team
    ['Hard', 'Breaking', 'Offspeed'].forEach(function (item) {
      container.appendChild(_createChip(item, function (it) { toggleHitterPitchChip(it); }, false));
    });
    _appendDivider(container);

    PITCH_TYPE_ORDER.forEach(function (item) {
      if (item === '|') { _appendDivider(container); return; }
      if (available.indexOf(item) === -1) return;
      container.appendChild(_createChip(item, function (it) { toggleHitterPitchChip(it); }, false));
    });
  }

  function buildHitterPitchChips(preserveSelection) {
    const container = document.getElementById('pitch-type-chips');
    container.innerHTML = '';
    if (!preserveSelection) { selectedPitchTypes = ['All']; }

    HITTER_PITCH_CHIP_ORDER.forEach(function (item) {
      if (item === '|') { _appendDivider(container); return; }
      const isSelected = selectedPitchTypes.indexOf(item) !== -1;
      container.appendChild(_createChip(item, function (it) { toggleHitterPitchChip(it); }, isSelected));
    });
  }

  function toggleHitterPitchChip(pt) {
    const isAll = (pt === 'All');
    const idx = selectedPitchTypes.indexOf(pt);

    if (isAll) {
      if (idx === -1) {
        selectedPitchTypes = ['All'];
      }
    } else {
      const allIdx = selectedPitchTypes.indexOf('All');
      if (allIdx !== -1) {
        selectedPitchTypes.splice(allIdx, 1);
      }

      if (idx === -1) {
        selectedPitchTypes.push(pt);
      } else {
        const newIdx = selectedPitchTypes.indexOf(pt);
        if (newIdx !== -1) {
          selectedPitchTypes.splice(newIdx, 1);
        }
      }

      if (selectedPitchTypes.length === 0) {
        selectedPitchTypes = ['All'];
      }
    }

    updateHitterPitchChipVisuals();
  }

  function updateHitterPitchChipVisuals() {
    const container = document.getElementById('pitch-type-chips');
    const chips = container.querySelectorAll('.pitch-chip');
    chips.forEach(function (btn) {
      const pt = btn.getAttribute('data-pitch');
      const color = btn.style.getPropertyValue('--chip-bg');
      if (selectedPitchTypes.indexOf(pt) !== -1) {
        btn.classList.add('selected');
        btn.style.backgroundColor = color;
        btn.style.borderColor = 'transparent';
      } else {
        btn.classList.remove('selected');
        btn.style.backgroundColor = '';
        btn.style.borderColor = '';
      }
    });
  }

  function togglePitchChip(pt, btn) {
    const idx = selectedPitchTypes.indexOf(pt);
    if (idx === -1) {
      selectedPitchTypes.push(pt);
      btn.classList.add('selected');
      btn.style.backgroundColor = btn.style.getPropertyValue('--chip-bg');
      btn.style.borderColor = 'transparent';
    } else {
      selectedPitchTypes.splice(idx, 1);
      btn.classList.remove('selected');
      btn.style.backgroundColor = '';
      btn.style.borderColor = '';
    }
  }

  function setupTabs() {
    document.querySelectorAll('.section-tab').forEach(function (btn) {
      btn.addEventListener('click', function () {
        const section = btn.getAttribute('data-section');
        if (section === 'home') {
          window.location.hash = 'home';
        } else if (section === 'pitchers') {
          navigateToTab('pitcherStats');
        } else if (section === 'hitters') {
          navigateToTab('hitterStats');
        } else if (section === 'abs') {
          showAbs('leaders');
        }
      });
    });

    document.querySelectorAll('.nav-subtabs .tab').forEach(function (tab) {
      tab.addEventListener('click', function () {
        const av = tab.getAttribute('data-absview');
        if (av) { showAbs(av); return; }
        const tabKey = tab.getAttribute('data-tab');
        if (tabKey) navigateToTab(tabKey);
      });
    });
  }

  function updateVsHandLabels() {
    if (!vsHandSelect) return;
    const opts = vsHandSelect.options;
    const isHitter = isHitterTab(currentTab);
    for (let i = 0; i < opts.length; i++) {
      if (opts[i].value === 'R') opts[i].textContent = isHitter ? 'RHP' : 'RHH';
      if (opts[i].value === 'L') opts[i].textContent = isHitter ? 'LHP' : 'LHH';
    }
  }

  function _getMaxTeamGames() {
    const tg = Aggregator.loaded ? Aggregator.getTeamGamesPlayed() : {};
    let max = 0;
    for (const t in tg) { if (tg[t] > max) max = tg[t]; }
    return max;
  }

  function _resolveMinCount() {
    const val = minCountInput.value;
    if (val === 'Q') {
      // Hitter stats: use per-row _qualified flag (team-specific 3.1 PA/game)
      if (isHitterTab(currentTab) && currentTab !== 'hitterPitch') {
        return 'Q';
      }
      // pitchMetrics (Arsenal): 25 pitches per pitch type — checked BEFORE
      // the generic isPitcherTab branch because pitchMetrics IS a pitcher
      // tab and would otherwise fall through to return 1.
      if (currentTab === 'pitchMetrics') {
        return 25;
      }
      // Other pitcher tabs: IP-based qualification handles the real filter (via minIp='Q')
      if (isPitcherTab(currentTab)) {
        return 1;
      }
      // hitterPitch (vs Pitches): per-pitch-type minimum.
      return 10;
    }
    return parseInt(val) || 1;
  }

  function _resolveMinIp() {
    const val = minIpInput.value;
    if (val === 'Q') {
      // Use per-row _qualified flag (team-specific SP/RP thresholds)
      return 'Q';
    }
    return parseFloat(val) || 0;
  }

  // Team view: drop player-identity columns and promote Team to the primary
  // sticky column. Player-level minimum-sample filters don't apply.
  const TEAM_VIEW_DROP_KEYS = { _compare: true, pitcher: true, hitter: true, throws: true, stands: true };
  const _teamColumnsCache = {};
  function teamColumns(tab) {
    if (_teamColumnsCache[tab]) return _teamColumnsCache[tab];
    const out = [];
    COLUMNS[tab].forEach(function (c) {
      if (TEAM_VIEW_DROP_KEYS[c.key]) return;
      if (c.key === 'team') {
        const tc = Object.assign({}, c);
        tc.sticky = true;
        delete tc.stickyIdx;
        tc.align = 'left';
        tc.cls = 'col-pitcher';
        tc.noToggle = true;
        out.push(tc);
      } else {
        out.push(c);
      }
    });
    _teamColumnsCache[tab] = out;
    return out;
  }

  function activeColumns() {
    return viewMode === 'team' ? teamColumns(currentTab) : COLUMNS[currentTab];
  }

  function getFilters() {
    const f = _getFiltersBase();
    f.viewMode = viewMode;
    if (viewMode === 'team') {
      // Player-level sample-size gates don't apply to team totals
      f.minCount = 0;
      f.minSwings = 0;
      f.minIp = 0;
      f.minTbf = 0;
      f.minBip = 0;
      f.minPitcherSwings = 0;
    }
    return f;
  }

  function _getFiltersBase() {
    return {
      team: teamSelect.value,
      pitchTypes: (selectedPitchTypes.length === 0 || (selectedPitchTypes.length === 1 && selectedPitchTypes[0] === 'All')) ? ['all'] : selectedPitchTypes.slice(),
      throws: throwsSelect.value,
      vsHand: vsHandSelect.value,
      minCount: currentTab === 'pitcherStats' ? 0 : _resolveMinCount(),
      minSwings: parseInt(minSwingsInput.value) || 1,
      // Arsenal (pitchMetrics) qualification is per-pitch-type (>=25 pitches),
      // NOT pitcher-level IP qualification — having thrown 25 of a pitch type
      // is enough for that pitch row to qualify regardless of total IP.
      minIp: currentTab === 'pitcherStats' ? _resolveMinIp() : (isPitcherTab(currentTab) && currentTab !== 'pitchMetrics' && minCountInput.value === 'Q' ? 'Q' : 0),
      minTbf: currentTab === 'pitcherStats' ? (parseInt(minTbfInput.value) || 1) : 0,
      minBip: (currentTab === 'pitcherBattedBall' || currentTab === 'hitterBattedBall') ? (parseInt(minBipInput.value) || 1) : 0,
      minPitcherSwings: currentTab === 'pitcherSwingDecisions' ? (parseInt(minPitcherSwingsInput.value) || 1) : 0,
      search: searchInput.value.trim(),
      dateStart: dateStartInput.value || '',
      dateEnd: dateEndInput.value || '',
      role: document.getElementById('role-filter').value,
    };
  }

  // Columns hidden when viewing ROC (AAA) team data. Audited 2026-07-20
  // against actual ROC null-rates: xwOBA/xwOBAcon (3D-table fill), xRV
  // family (no-arm model), and xWRC+ are POPULATED for ROC now and were
  // unhidden; actual-RV (runValue/rv100 + the RVOE family), arm angle,
  // sprint speed, bat tracking, and per-pitch xBA/xSLG remain blank.
  const ROC_HIDDEN_PITCHER = ['armAngle', 'xIVB', 'ivbOE', 'xHB', 'hbOE', 'runValue', 'rv100', 'rvoe', 'xrvoe', 'rvoe100', 'xrvoe100', 'xBA', 'xSLG'];
  const ROC_HIDDEN_HITTER = ['batSpeed', 'swingLength', 'attackAngle', 'attackDirection', 'swingPathTilt', 'nCompSwings', 'blastPct', 'idealAAPct', 'squaredUpPct', 'xBA', 'xSLG', 'runValue', 'rv100', 'sprintSpeed'];
  const ALL_ROC_HIDDEN = ROC_HIDDEN_PITCHER.concat(ROC_HIDDEN_HITTER);

  function refresh() {
    const filters = getFilters();
    let dataTab = TAB_DATA[currentTab] || 'pitcher';

    const hasPitchTypeFilter = selectedPitchTypes.length > 0 && !(selectedPitchTypes.length === 1 && selectedPitchTypes[0] === 'All');

    if ((currentTab === 'pitcherSwingDecisions' || currentTab === 'pitcherBattedBall') && hasPitchTypeFilter) {
      dataTab = 'pitch';
    }
    if ((currentTab === 'hitterBattedBall' || currentTab === 'hitterSwingDecisions') && hasPitchTypeFilter) {
      dataTab = 'hitterPitch';
    }

    // ROC column visibility: hide columns that have no data for AAA teams
    const rocTeamsRefresh = (DataStore.metadata && DataStore.metadata.rocTeams) || [];
    const isROCTeam = rocTeamsRefresh.indexOf(filters.team) !== -1;

    ALL_ROC_HIDDEN.forEach(function (k) {
      if (Leaderboard._rocHidden && Leaderboard._rocHidden[k]) {
        delete Leaderboard.hiddenColumns[k];
      }
    });
    Leaderboard._rocHidden = {};

    if (isROCTeam) {
      const toHide = isPitcherTab(currentTab) ? ROC_HIDDEN_PITCHER : ROC_HIDDEN_HITTER;
      toHide.forEach(function (k) {
        Leaderboard.hiddenColumns[k] = true;
        Leaderboard._rocHidden[k] = true;
      });
    }

    // Hide bat tracking tab for ROC (no bat tracking data for AAA)
    const batTrackingTab = document.querySelector('.tab[data-tab="hitterBatTracking"]');
    if (batTrackingTab) {
      batTrackingTab.style.display = isROCTeam ? 'none' : '';
    }
    // If on bat tracking tab and ROC selected, switch to hitterStats
    if (isROCTeam && currentTab === 'hitterBatTracking') {
      navigateToTab('hitterStats');
      return;
    }

    let data = DataStore.getFilteredDataV2(dataTab, filters);
    const columns = activeColumns();

    data = applyRangeFilters(data, columns);

    if (!Leaderboard.currentSort.key) {
      Leaderboard.currentSort = defaultSortFor(currentTab);
    }

    const sortKey = Leaderboard.currentSort.key;
    let col = null;
    for (let i = 0; i < columns.length; i++) {
      if (columns[i].key === sortKey) { col = columns[i]; break; }
    }

    if (col && col.sortType) {
      const sk = col.sortKey || col.key;
      const dir = Leaderboard.currentSort.dir === 'asc' ? 1 : -1;
      data.sort(function (a, b) {
        const va = a[sk], vb = b[sk];
        if (va === null || va === undefined) { return vb === null || vb === undefined ? 0 : 1; }
        if (vb === null || vb === undefined) return -1;
        if (col.sortType === 'string') return dir * String(va).localeCompare(String(vb));
        return dir * (va - vb);
      });
    }

    allData = data;

    // Compute league-wide data (all teams) for league avg row
    // Keep contextual filters (vsHand, throws, role) but strip display thresholds
    const leagueFilters = {};
    for (const fk in filters) leagueFilters[fk] = filters[fk];
    leagueFilters.team = 'all';
    leagueFilters.minCount = 1;
    leagueFilters.minIp = 0;
    leagueFilters.minTbf = 0;
    leagueFilters.minPitcherSwings = 0;
    leagueFilters.minBip = 0;
    leagueFilters.minSwings = 0;
    delete leagueFilters.dateStart;
    delete leagueFilters.dateEnd;
    delete leagueFilters.search;
    let leagueData = DataStore.getFilteredDataV2(dataTab, leagueFilters);

    Leaderboard.render(data, columns, {
      teamFilter: filters.team,
      leagueData: leagueData,
      pitchTypes: filters.pitchTypes,
      vsHand: filters.vsHand,
      throws: filters.throws,
      role: filters.role,
      viewMode: viewMode,
    });
    saveURLState();
  }

  function setupToolbar() {
    (function () {
      const compactBtn = document.getElementById('compact-toggle');
      if (!compactBtn) return;
      if (localStorage.getItem('compactMode') === '1') {
        document.body.classList.add('compact');
        compactBtn.classList.add('active');
      }
      compactBtn.addEventListener('click', function () {
        document.body.classList.toggle('compact');
        const on = document.body.classList.contains('compact');
        compactBtn.classList.toggle('active', on);
        localStorage.setItem('compactMode', on ? '1' : '0');
      });
    })();

    document.getElementById('export-csv-btn').addEventListener('click', function () {
      const visCols = Leaderboard.getVisibleColumns(activeColumns(), allData).filter(function (c) {
        return c.key !== '_rank' && !c.isCompare;
      });
      const csv = Utils.toCSV(allData, visCols);
      Utils.downloadFile(csv, 'leaderboard_' + currentTab + '.csv');
      const btn = document.getElementById('export-csv-btn');
      const orig = btn.textContent;
      btn.textContent = 'Downloaded!';
      setTimeout(function () { btn.textContent = orig; }, 1500);
    });

    document.getElementById('copy-clipboard-btn').addEventListener('click', function () {
      const visCols = Leaderboard.getVisibleColumns(activeColumns(), allData).filter(function (c) {
        return c.key !== '_rank' && !c.isCompare;
      });
      const tsv = Utils.toTSV(allData, visCols);
      Utils.copyToClipboard(tsv);
      const btn = document.getElementById('copy-clipboard-btn');
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(function () { btn.textContent = orig; }, 1500);
    });
  }

  function setupPagination() {
    document.getElementById('page-prev').addEventListener('click', function () {
      if (Leaderboard.currentPage > 1) {
        Leaderboard.currentPage--;
        Leaderboard.keyboardFocusIndex = -1;
        refresh();
        scrollTableToTop();
      }
    });

    document.getElementById('page-next').addEventListener('click', function () {
      Leaderboard.currentPage++;
      Leaderboard.keyboardFocusIndex = -1;
      refresh();
      scrollTableToTop();
    });

    document.getElementById('page-size').addEventListener('change', function () {
      Leaderboard.pageSize = parseInt(this.value) || 0;
      Leaderboard.currentPage = 1;
      Leaderboard.keyboardFocusIndex = -1;
      refresh();
    });
  }

  function scrollTableToTop() {
    const container = document.getElementById('table-container');
    if (container) {
      container.scrollTop = 0;
      container.scrollLeft = 0;
    }
  }

  function setupSidePanel() {
    window.App = window.App || {};

    App.openSidePanel = function (name, team, hand, rowData) {
      document.getElementById('panel-pitcher-name').textContent = name;
      const info = [];
      if (team) info.push(team);
      if (isHitterTab(currentTab)) {
        if (hand) info.push(hand === 'R' ? 'RHH' : hand === 'L' ? 'LHH' : hand === 'S' ? 'Switch' : hand);
      } else {
        if (hand) info.push(hand === 'R' ? 'RHP' : 'LHP');
      }
      document.getElementById('panel-pitcher-info').textContent = info.join(' | ');

      sidePanel.classList.add('open');
      panelOverlay.classList.add('visible');

      const chartContainer = sidePanel.querySelector('.chart-container');

      if (isHitterTab(currentTab)) {
        if (chartContainer) chartContainer.style.display = 'none';
        ScatterChart.destroy();
        buildHitterPanelTable(name, team);
      } else {
        if (chartContainer) chartContainer.style.display = '';
        ScatterChart.render(name, team);
        buildPanelMetricsTable(name, team);
      }
    };

    App.closeSidePanel = function () {
      sidePanel.classList.remove('open');
      panelOverlay.classList.remove('visible');
      ScatterChart.destroy();
      const chartContainer = sidePanel.querySelector('.chart-container');
      if (chartContainer) chartContainer.style.display = '';
      const active = document.querySelectorAll('.active-row');
      for (let i = 0; i < active.length; i++) active[i].classList.remove('active-row');
    };

    panelClose.addEventListener('click', App.closeSidePanel);
    panelOverlay.addEventListener('click', App.closeSidePanel);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        // Close compare modal if open
        const cm = document.getElementById('compare-modal');
        if (cm && cm.style.display !== 'none') {
          cm.style.display = 'none';
          return;
        }
        App.closeSidePanel();
      }
    });

  }

  function buildPanelMetricsTable(pitcherName, team) {
    const container = document.getElementById('panel-metrics-table');
    container.innerHTML = '';

    const pitchData = DataStore.pitchData;
    if (!pitchData) return;

    const pitcherRows = pitchData.filter(function (r) { return r.pitcher === pitcherName && r.team === team; });
    if (pitcherRows.length === 0) return;

    pitcherRows.sort(function (a, b) { return (b.usagePct || 0) - (a.usagePct || 0); });

    const metricCols = [
      { key: 'pitchType', label: 'Pitch', format: function (v) { return v; } },
      { key: 'usagePct', label: 'Usage', format: Utils.formatPct },
      { key: 'velocity', label: 'Velo', format: Utils.formatDecimal(1) },
      { key: 'maxVelo', label: 'Max Velo', format: Utils.formatDecimal(1) },
      { key: 'spinRate', label: 'Spin', format: Utils.formatInt },
      { key: 'breakTilt', label: 'Tilt', format: function (v) { return v || '--'; } },
      { key: 'indVertBrk', label: 'IVB', format: Utils.formatDecimal(1) },
      { key: 'horzBrk', label: 'HB', format: Utils.formatDecimal(1) },
      { key: 'swStrPct', label: 'Whiff%', format: Utils.formatPct },
    ];

    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headerTr = document.createElement('tr');
    metricCols.forEach(function (mc) {
      const th = document.createElement('th');
      th.textContent = mc.label;
      if (mc.key === 'pitchType') th.style.textAlign = 'left';
      headerTr.appendChild(th);
    });
    thead.appendChild(headerTr);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    pitcherRows.forEach(function (row) {
      const tr = document.createElement('tr');
      metricCols.forEach(function (mc) {
        const td = document.createElement('td');
        if (mc.key === 'pitchType') {
          const badge = Utils.createPitchBadge(row[mc.key], false);
          td.appendChild(badge);
          td.style.textAlign = 'left';
        } else {
          td.textContent = mc.format(row[mc.key]);
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    container.appendChild(table);
  }

  function buildHitterPanelTable(hitterName, team) {
    const container = document.getElementById('panel-metrics-table');
    container.innerHTML = '';

    const details = window.HITTER_PITCH_DETAILS;
    const key = hitterName + '|' + (team || '');
    if (!details || !details[key]) return;

    const ptData = details[key];
    if (ptData.length === 0) return;

    const statCols = [
      { key: 'pitchType', label: 'Pitch', format: function (v) { return v; } },
      { key: 'count', label: '#', format: Utils.formatInt },
      { key: 'swingPct', label: 'Swing%', format: Utils.formatPct },
      { key: 'whiffPct', label: 'Whiff%', format: Utils.formatPct },
      { key: 'avgEVAll', label: 'Avg EV', format: Utils.formatDecimal(1) },
    ];

    // Group individual pitch rows by category
    const CATS = Aggregator.PITCH_CATEGORIES;
    const CAT_ORDER = ['Hard', 'Breaking', 'Offspeed'];
    const catGroups = {};
    for (let ci = 0; ci < CAT_ORDER.length; ci++) catGroups[CAT_ORDER[ci]] = [];

    ptData.forEach(function (row) {
      for (const cat in CATS) {
        if (CATS[cat].indexOf(row.pitchType) !== -1) {
          catGroups[cat].push(row);
          break;
        }
      }
    });

    // Compute weighted category summaries
    function computeCategorySummary(catRows, catName) {
      let totalCount = 0, sumSwings = 0, sumWhiffs = 0, sumEV = 0, nEV = 0;
      catRows.forEach(function (r) {
        const cnt = r.count || 0;
        totalCount += cnt;
        if (r.swingPct != null) sumSwings += (r.swingPct * cnt);
        if (r.whiffPct != null) {
          const swings = r.nSwings || (r.swingPct != null ? r.swingPct * cnt : 0);
          sumWhiffs += (r.whiffPct * swings);
        }
        if (r.avgEVAll != null && r.nBip) { sumEV += r.avgEVAll * r.nBip; nEV += r.nBip; }
      });
      return {
        pitchType: catName,
        count: totalCount,
        swingPct: totalCount > 0 ? sumSwings / totalCount : null,
        whiffPct: sumSwings > 0 ? sumWhiffs / sumSwings : null,
        avgEVAll: nEV > 0 ? sumEV / nEV : null
      };
    }

    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headerTr = document.createElement('tr');
    statCols.forEach(function (mc) {
      const th = document.createElement('th');
      th.textContent = mc.label;
      if (mc.key === 'pitchType') th.style.textAlign = 'left';
      headerTr.appendChild(th);
    });
    thead.appendChild(headerTr);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');

    // Render category rows with expand/collapse
    CAT_ORDER.forEach(function (catName) {
      const catRows = catGroups[catName];
      if (catRows.length === 0) return;
      catRows.sort(function (a, b) { return (b.count || 0) - (a.count || 0); });

      const summary = computeCategorySummary(catRows, catName);
      let expanded = false;
      const subRowEls = [];

      // Category row
      const tr = document.createElement('tr');
      tr.className = 'category-row';
      tr.style.cursor = 'pointer';
      statCols.forEach(function (mc) {
        const td = document.createElement('td');
        if (mc.key === 'pitchType') {
          const indicator = document.createElement('span');
          indicator.className = 'expand-indicator';
          indicator.textContent = '\u25B6';
          td.appendChild(indicator);
          const badge = Utils.createCategoryBadge(catName, false);
          td.appendChild(badge);
          td.style.textAlign = 'left';
        } else {
          td.textContent = mc.format(summary[mc.key]);
        }
        tr.appendChild(td);
      });

      tr.addEventListener('click', function () {
        expanded = !expanded;
        const ind = tr.querySelector('.expand-indicator');
        if (ind) ind.textContent = expanded ? '\u25BC' : '\u25B6';

        if (expanded && subRowEls.length === 0) {
          const nextSibling = tr.nextSibling;
          catRows.forEach(function (row) {
            const subTr = document.createElement('tr');
            subTr.className = 'sub-row';
            statCols.forEach(function (mc) {
              const td = document.createElement('td');
              if (mc.key === 'pitchType') {
                td.style.paddingLeft = '24px';
                const badge = Utils.createPitchBadge(row[mc.key], false);
                td.appendChild(badge);
                td.style.textAlign = 'left';
              } else {
                td.textContent = mc.format(row[mc.key]);
              }
              subTr.appendChild(td);
            });
            tbody.insertBefore(subTr, nextSibling);
            subRowEls.push(subTr);
          });
        } else {
          for (let s = 0; s < subRowEls.length; s++) {
            subRowEls[s].style.display = expanded ? '' : 'none';
          }
        }
      });

      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    container.appendChild(table);
  }

  function setupCompareModal() {
    window.App = window.App || {};

    App.updateCompareButton = function () {
      const list = Leaderboard.getCompareList();
      const btn = document.getElementById('compare-btn');
      btn.textContent = list.length > 0 ? 'Compare (' + list.length + ')' : 'Compare';
      btn.disabled = list.length < 2;
    };

    document.getElementById('compare-btn').addEventListener('click', function () {
      const list = Leaderboard.getCompareList();
      if (list.length < 2) return;
      const modal = document.getElementById('compare-modal');
      modal.style.display = '';
      ScatterChart.renderCompare(list);
    });

    document.getElementById('compare-close').addEventListener('click', function () {
      document.getElementById('compare-modal').style.display = 'none';
      ScatterChart.destroyCompare();
    });
  }

  function setupPercentileTooltips() {
    const tooltip = document.getElementById('pctl-tooltip');
    const tbody = document.getElementById('table-body');

    tbody.addEventListener('mouseover', function (e) {
      const td = e.target.closest('td[data-pctl]');
      if (!td) { tooltip.classList.remove('visible'); return; }
      const tr = td.closest('tr');
      if (tr && tr.classList.contains('league-avg-row')) { tooltip.classList.remove('visible'); return; }
      const pctl = td.getAttribute('data-pctl');
      const label = td.getAttribute('data-col-label');
      const colKey = td.getAttribute('data-col-key');

      let text = pctl + 'th percentile';
      const runs100 = td.getAttribute('data-runs100');
      if (runs100 !== null) {
        // toFixed keeps the '-' on negatives and adds no '+' on positives
        // (house number-format rule).
        // Pitcher+'s slope is predictive (next-season xRV/100 on this-season
        // Pitcher+), so it reads 'expected' where Pitching+ reads 'vs avg'.
        text += ' · ' + parseFloat(runs100).toFixed(2) + ' runs/100 '
              + (colKey === 'pitcherPlus' ? 'expected' : 'vs avg');
      }
      if (td.getAttribute('data-low-support')) {
        text += ' · low model support (unusual profile, score less certain)';
      }
      tooltip.textContent = text;
      tooltip.classList.add('visible');
    });

    tbody.addEventListener('mousemove', function (e) {
      if (tooltip.classList.contains('visible')) {
        tooltip.style.left = (e.clientX + 12) + 'px';
        tooltip.style.top = (e.clientY - 30) + 'px';
      }
    });

    tbody.addEventListener('mouseout', function (e) {
      if (!e.target.closest('td[data-pctl]')) {
        tooltip.classList.remove('visible');
      }
    });
  }

  function setupKeyboardNav() {
    document.addEventListener('keydown', function (e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

      const tbody = document.getElementById('table-body');
      const rows = tbody.querySelectorAll('tr.clickable-row');
      if (rows.length === 0) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        Leaderboard.keyboardFocusIndex = Math.min(Leaderboard.keyboardFocusIndex + 1, rows.length - 1);
        updateKeyboardFocus(rows);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        Leaderboard.keyboardFocusIndex = Math.max(Leaderboard.keyboardFocusIndex - 1, 0);
        updateKeyboardFocus(rows);
      } else if (e.key === 'Enter' && Leaderboard.keyboardFocusIndex >= 0) {
        e.preventDefault();
        rows[Leaderboard.keyboardFocusIndex].click();
      }
    });
  }

  function updateKeyboardFocus(rows) {
    for (let i = 0; i < rows.length; i++) {
      rows[i].classList.toggle('keyboard-focus', i === Leaderboard.keyboardFocusIndex);
    }
    if (Leaderboard.keyboardFocusIndex >= 0 && rows[Leaderboard.keyboardFocusIndex]) {
      rows[Leaderboard.keyboardFocusIndex].scrollIntoView({ block: 'nearest' });
    }
  }

  function setupRangeFilters() {
    const btn = document.getElementById('range-filter-btn');
    const panel = document.getElementById('range-filter-panel');

    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      panel.classList.toggle('open');
      if (panel.classList.contains('open')) {
        buildRangeFilterPanel();
      }
    });

    document.addEventListener('click', function (e) {
      if (!panel.contains(e.target) && e.target !== btn) {
        panel.classList.remove('open');
      }
    });
  }

  function isPercentageColumn(col) {
    return col.format === Utils.formatPct;
  }

  function buildRangeFilterPanel() {
    const panel = document.getElementById('range-filter-panel');
    panel.innerHTML = '';

    const columns = activeColumns();
    let currentGroup = '';

    columns.forEach(function (col) {
      if (col.sortType !== 'numeric') return;
      if (col.key === '_rank' || col.isCompare) return;

      if (col.group && col.group !== currentGroup) {
        currentGroup = col.group;
        const groupLabel = document.createElement('div');
        groupLabel.className = 'range-filter-group-label';
        groupLabel.textContent = currentGroup.charAt(0).toUpperCase() + currentGroup.slice(1);
        panel.appendChild(groupLabel);
      }

      const row = document.createElement('div');
      row.className = 'range-filter-row';

      const label = document.createElement('span');
      label.className = 'rf-label';
      label.textContent = col.label;
      row.appendChild(label);

      const existing = columnRangeFilters[col.key] || {};
      const isPct = isPercentageColumn(col);

      const minInput = document.createElement('input');
      minInput.type = 'number';
      minInput.placeholder = 'Min';
      minInput.step = 'any';
      if (existing.min !== null && existing.min !== undefined) {
        minInput.value = isPct ? (existing.min * 100) : existing.min;
      }

      const sep = document.createElement('span');
      sep.className = 'rf-sep';
      sep.textContent = '–';

      const maxInput = document.createElement('input');
      maxInput.type = 'number';
      maxInput.placeholder = 'Max';
      maxInput.step = 'any';
      if (existing.max !== null && existing.max !== undefined) {
        maxInput.value = isPct ? (existing.max * 100) : existing.max;
      }

      row.appendChild(minInput);
      row.appendChild(sep);
      row.appendChild(maxInput);
      panel.appendChild(row);

      let debounceTimer = null;
      function onInput() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(function () {
          let minVal = minInput.value !== '' ? parseFloat(minInput.value) : null;
          let maxVal = maxInput.value !== '' ? parseFloat(maxInput.value) : null;

          if (isPct) {
            if (minVal !== null) minVal = minVal / 100;
            if (maxVal !== null) maxVal = maxVal / 100;
          }

          if (minVal === null && maxVal === null) {
            delete columnRangeFilters[col.key];
          } else {
            columnRangeFilters[col.key] = { min: minVal, max: maxVal };
          }

          updateRangeFilterBadge();
          Leaderboard.currentPage = 1;
          refresh();
        }, 200);
      }

      minInput.addEventListener('input', onInput);
      maxInput.addEventListener('input', onInput);
    });

    const actions = document.createElement('div');
    actions.className = 'range-filter-actions';
    const clearBtn = document.createElement('button');
    clearBtn.className = 'rf-clear-btn';
    clearBtn.textContent = 'Clear All';
    clearBtn.addEventListener('click', function () {
      columnRangeFilters = {};
      updateRangeFilterBadge();
      buildRangeFilterPanel(); // rebuild to clear inputs
      Leaderboard.currentPage = 1;
      refresh();
    });
    actions.appendChild(clearBtn);
    panel.appendChild(actions);
  }

  function applyRangeFilters(data, columns) {
    const keys = Object.keys(columnRangeFilters);
    if (keys.length === 0) return data;

    return data.filter(function (row) {
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        const filter = columnRangeFilters[key];
        const val = row[key];

        if (val === null || val === undefined) return false;

        if (filter.min !== null && val < filter.min) return false;
        if (filter.max !== null && val > filter.max) return false;
      }
      return true;
    });
  }

  function updateRangeFilterBadge() {
    const badge = document.getElementById('range-filter-badge');
    const count = Object.keys(columnRangeFilters).length;
    if (count > 0) {
      badge.textContent = count;
      badge.style.display = '';
    } else {
      badge.style.display = 'none';
    }
  }

  function setupColumnSettings() {
    const btn = document.getElementById('col-settings-btn');
    const panel = document.getElementById('col-settings-panel');

    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      panel.classList.toggle('open');
      if (panel.classList.contains('open')) {
        buildColumnSettingsPanel();
      }
    });

    document.addEventListener('click', function (e) {
      if (!panel.contains(e.target) && e.target !== btn) {
        panel.classList.remove('open');
      }
    });
  }

  function buildColumnSettingsPanel() {
    const panel = document.getElementById('col-settings-panel');
    panel.innerHTML = '';

    const columns = activeColumns();
    let currentGroup = '';

    columns.forEach(function (col) {
      if (col.noToggle) return;

      if (col.group && col.group !== currentGroup) {
        currentGroup = col.group;
        const groupLabel = document.createElement('div');
        groupLabel.className = 'col-settings-group-label';
        groupLabel.textContent = currentGroup.charAt(0).toUpperCase() + currentGroup.slice(1);
        panel.appendChild(groupLabel);
      }

      const label = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = !Leaderboard.hiddenColumns[col.key];
      cb.addEventListener('change', function () {
        if (cb.checked) {
          delete Leaderboard.hiddenColumns[col.key];
        } else {
          Leaderboard.hiddenColumns[col.key] = true;
        }
        refresh();
      });
      label.appendChild(cb);
      label.appendChild(document.createTextNode(' ' + col.label));
      panel.appendChild(label);
    });
  }

  function saveURLState() {
    if (PlayerPage.isOpen) return;
    const route = TAB_ROUTE[currentTab] || 'pitchers/stats';
    const parts = [];
    if (viewMode === 'team') parts.push('view=team');
    if (teamSelect.value !== 'all') parts.push('team=' + teamSelect.value);
    if (throwsSelect.value !== 'all') parts.push('throws=' + throwsSelect.value);
    if (vsHandSelect.value !== 'all') parts.push('vsHand=' + vsHandSelect.value);
    if (minCountInput.value !== 'Q') parts.push('min=' + minCountInput.value);
    if (searchInput.value) parts.push('search=' + encodeURIComponent(searchInput.value));
    if (Leaderboard.currentSort.key) parts.push('sort=' + Leaderboard.currentSort.key);
    if (Leaderboard.currentSort.dir) parts.push('dir=' + Leaderboard.currentSort.dir);
    if (Leaderboard.currentPage > 1) parts.push('page=' + Leaderboard.currentPage);
    if (dateStartInput.value) parts.push('dateStart=' + dateStartInput.value);
    if (dateEndInput.value) parts.push('dateEnd=' + dateEndInput.value);
    if (selectedPitchTypes.length > 0) parts.push('pitch=' + selectedPitchTypes.join(','));
    var roleVal = document.getElementById('role-filter').value;
    if (roleVal && roleVal !== 'all') parts.push('role=' + roleVal);
    const hash = route + (parts.length > 0 ? '?' + parts.join('&') : '');
    if (window.location.hash !== '#' + hash) {
      history.replaceState(null, '', '#' + hash);
    }
  }


  init();
})();
