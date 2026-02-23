(function () {
  // ---- State ----
  var currentTab = 'pitch';
  var selectedPitchTypes = []; // empty = all; or array of selected types
  var allData = []; // current filtered + sorted data (full, before pagination)

  // ---- DOM refs ----
  var teamSelect, throwsSelect, minCountInput, searchInput;
  var sidePanel, panelOverlay, panelClose;

  // ---- Init ----
  function init() {
    DataStore.load().then(function () {
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
      setupDarkMode();
      applyURLState();
      refresh();
    });
  }

  function setupDOM() {
    teamSelect = document.getElementById('team-filter');
    throwsSelect = document.getElementById('throws-filter');
    minCountInput = document.getElementById('min-count');
    searchInput = document.getElementById('search-input');
    sidePanel = document.getElementById('side-panel');
    panelOverlay = document.getElementById('panel-overlay');
    panelClose = document.getElementById('panel-close');
  }

  // ---- Filters ----
  function setupFilters() {
    // Populate team dropdown
    DataStore.metadata.teams.forEach(function (team) {
      var opt = document.createElement('option');
      opt.value = team;
      opt.textContent = team;
      teamSelect.appendChild(opt);
    });

    // Populate pitch type chips
    buildPitchChips();

    // Set generated date
    var genDate = document.getElementById('generated-date');
    if (genDate) genDate.textContent = DataStore.metadata.generatedAt;

    // Filter listeners
    teamSelect.addEventListener('change', function () { Leaderboard.currentPage = 1; refresh(); });
    throwsSelect.addEventListener('change', function () { Leaderboard.currentPage = 1; refresh(); });
    minCountInput.addEventListener('input', function () { Leaderboard.currentPage = 1; refresh(); });

    var searchTimer = null;
    searchInput.addEventListener('input', function () {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(function () { Leaderboard.currentPage = 1; refresh(); }, 200);
    });
  }

  function buildPitchChips() {
    var container = document.getElementById('pitch-type-chips');
    container.innerHTML = '';

    DataStore.metadata.pitchTypes.forEach(function (pt) {
      var btn = document.createElement('button');
      btn.className = 'pitch-chip';
      btn.textContent = pt;
      btn.setAttribute('data-pitch', pt);
      var color = Utils.getPitchColor(pt);
      btn.style.setProperty('--chip-bg', color);
      btn.style.borderColor = color;
      // For light-colored pitches, adjust text
      if (pt === 'SI' || pt === 'SV') btn.style.color = '';
      btn.title = Utils.pitchTypeLabel(pt);

      btn.addEventListener('click', function () {
        togglePitchChip(pt, btn);
        Leaderboard.currentPage = 1;
        refresh();
      });

      container.appendChild(btn);
    });
  }

  function togglePitchChip(pt, btn) {
    var idx = selectedPitchTypes.indexOf(pt);
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

  // ---- Tabs ----
  function setupTabs() {
    document.querySelectorAll('.tab').forEach(function (tab) {
      tab.addEventListener('click', function () {
        document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
        tab.classList.add('active');
        currentTab = tab.getAttribute('data-tab');
        Leaderboard.currentSort = { key: 'count', dir: 'desc' };
        Leaderboard.currentPage = 1;
        Leaderboard.keyboardFocusIndex = -1;
        // Show/hide pitch type filter
        document.getElementById('pitch-type-filter-group').style.display =
          currentTab === 'pitch' ? '' : 'none';
        refresh();
      });
    });
  }

  // ---- Core refresh ----
  function getFilters() {
    return {
      team: teamSelect.value,
      pitchTypes: selectedPitchTypes.length > 0 ? selectedPitchTypes : 'all',
      throws: throwsSelect.value,
      minCount: parseInt(minCountInput.value) || 1,
      search: searchInput.value.trim(),
    };
  }

  function refresh() {
    var filters = getFilters();
    var data = DataStore.getFilteredData(currentTab, filters);
    var columns = COLUMNS[currentTab];

    // Apply sort
    if (!Leaderboard.currentSort.key) {
      Leaderboard.currentSort = { key: 'count', dir: 'desc' };
    }

    var sortKey = Leaderboard.currentSort.key;
    var col = null;
    for (var i = 0; i < columns.length; i++) {
      if (columns[i].key === sortKey) { col = columns[i]; break; }
    }

    if (col && col.sortType) {
      var sk = col.sortKey || col.key;
      var dir = Leaderboard.currentSort.dir === 'asc' ? 1 : -1;
      data.sort(function (a, b) {
        var va = a[sk], vb = b[sk];
        if (va === null || va === undefined) { return vb === null || vb === undefined ? 0 : 1; }
        if (vb === null || vb === undefined) return -1;
        if (col.sortType === 'string') return dir * String(va).localeCompare(String(vb));
        return dir * (va - vb);
      });
    }

    allData = data;
    Leaderboard.render(data, columns);
    saveURLState();
  }

  // ---- Toolbar ----
  function setupToolbar() {
    // League average toggle
    var avgBtn = document.getElementById('league-avg-toggle');
    avgBtn.addEventListener('click', function () {
      Leaderboard.showLeagueAvg = !Leaderboard.showLeagueAvg;
      avgBtn.classList.toggle('active', Leaderboard.showLeagueAvg);
      refresh();
    });

    // Export CSV
    document.getElementById('export-csv-btn').addEventListener('click', function () {
      var visCols = Leaderboard.getVisibleColumns(COLUMNS[currentTab]).filter(function (c) {
        return c.key !== '_rank' && !c.isCompare;
      });
      var csv = Utils.toCSV(allData, visCols);
      Utils.downloadFile(csv, 'leaderboard_' + currentTab + '.csv');
    });

    // Copy to clipboard
    document.getElementById('copy-clipboard-btn').addEventListener('click', function () {
      var visCols = Leaderboard.getVisibleColumns(COLUMNS[currentTab]).filter(function (c) {
        return c.key !== '_rank' && !c.isCompare;
      });
      var tsv = Utils.toTSV(allData, visCols);
      Utils.copyToClipboard(tsv);
      // Visual feedback
      var btn = document.getElementById('copy-clipboard-btn');
      var orig = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(function () { btn.textContent = orig; }, 1500);
    });
  }

  // ---- Pagination ----
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
    var container = document.getElementById('table-container');
    if (container) container.scrollTop = 0;
  }

  // ---- Side Panel ----
  function setupSidePanel() {
    window.App = window.App || {};

    App.openSidePanel = function (pitcherName, team, throws) {
      document.getElementById('panel-pitcher-name').textContent = pitcherName;
      var info = [];
      if (team) info.push(team);
      if (throws) info.push(throws === 'R' ? 'RHP' : 'LHP');
      document.getElementById('panel-pitcher-info').textContent = info.join(' | ');

      sidePanel.classList.add('open');
      panelOverlay.classList.add('visible');

      // Reset chart view toggle
      ScatterChart.currentView = 'movement';
      document.querySelectorAll('.chart-view-btn').forEach(function (b) {
        b.classList.toggle('active', b.getAttribute('data-view') === 'movement');
      });

      ScatterChart.render(pitcherName);
      buildPanelMetricsTable(pitcherName);
    };

    App.closeSidePanel = function () {
      sidePanel.classList.remove('open');
      panelOverlay.classList.remove('visible');
      ScatterChart.destroy();
      var active = document.querySelectorAll('.active-row');
      for (var i = 0; i < active.length; i++) active[i].classList.remove('active-row');
    };

    panelClose.addEventListener('click', App.closeSidePanel);
    panelOverlay.addEventListener('click', App.closeSidePanel);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        // Close compare modal if open
        var cm = document.getElementById('compare-modal');
        if (cm && cm.style.display !== 'none') {
          cm.style.display = 'none';
          return;
        }
        App.closeSidePanel();
      }
    });

    // Chart view toggle
    document.querySelectorAll('.chart-view-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.chart-view-btn').forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');
        var view = btn.getAttribute('data-view');
        if (ScatterChart.currentPitcher) {
          ScatterChart.render(ScatterChart.currentPitcher, view);
        }
      });
    });
  }

  function buildPanelMetricsTable(pitcherName) {
    var container = document.getElementById('panel-metrics-table');
    container.innerHTML = '';

    // Get pitch data for this pitcher
    var pitchData = DataStore.pitchData;
    if (!pitchData) return;

    var pitcherRows = pitchData.filter(function (r) { return r.pitcher === pitcherName; });
    if (pitcherRows.length === 0) return;

    // Sort by usage descending
    pitcherRows.sort(function (a, b) { return (b.usagePct || 0) - (a.usagePct || 0); });

    var metricCols = [
      { key: 'pitchType', label: 'Pitch', format: function (v) { return v; } },
      { key: 'usagePct', label: 'Usage', format: Utils.formatPct },
      { key: 'velocity', label: 'Velo', format: Utils.formatDecimal(1) },
      { key: 'spinRate', label: 'Spin', format: Utils.formatInt },
      { key: 'indVertBrk', label: 'IVB', format: Utils.formatDecimal(1) },
      { key: 'horzBrk', label: 'HB', format: Utils.formatDecimal(1) },
      { key: 'extension', label: 'Ext', format: Utils.formatDecimal(1) },
    ];

    var table = document.createElement('table');
    var thead = document.createElement('thead');
    var headerTr = document.createElement('tr');
    metricCols.forEach(function (mc) {
      var th = document.createElement('th');
      th.textContent = mc.label;
      if (mc.key === 'pitchType') th.style.textAlign = 'left';
      headerTr.appendChild(th);
    });
    thead.appendChild(headerTr);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    pitcherRows.forEach(function (row) {
      var tr = document.createElement('tr');
      metricCols.forEach(function (mc) {
        var td = document.createElement('td');
        if (mc.key === 'pitchType') {
          var badge = document.createElement('span');
          badge.className = 'pitch-badge';
          badge.textContent = row[mc.key];
          badge.style.backgroundColor = Utils.getPitchColor(row[mc.key]);
          if (row[mc.key] === 'SI' || row[mc.key] === 'SV') badge.style.color = '#1a1a2e';
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

  // ---- Compare Modal ----
  function setupCompareModal() {
    window.App = window.App || {};

    App.updateCompareButton = function () {
      var list = Leaderboard.getCompareList();
      var btn = document.getElementById('compare-btn');
      btn.textContent = 'Compare (' + list.length + ')';
      btn.disabled = list.length < 2;
    };

    document.getElementById('compare-btn').addEventListener('click', function () {
      var list = Leaderboard.getCompareList();
      if (list.length < 2) return;
      var modal = document.getElementById('compare-modal');
      modal.style.display = '';
      ScatterChart.renderCompare(list);
    });

    document.getElementById('compare-close').addEventListener('click', function () {
      document.getElementById('compare-modal').style.display = 'none';
      ScatterChart.destroyCompare();
    });
  }

  // ---- Percentile Tooltips ----
  function setupPercentileTooltips() {
    var tooltip = document.getElementById('pctl-tooltip');
    var tbody = document.getElementById('table-body');

    tbody.addEventListener('mouseover', function (e) {
      var td = e.target.closest('td[data-pctl]');
      if (!td) { tooltip.classList.remove('visible'); return; }
      var pctl = td.getAttribute('data-pctl');
      var label = td.getAttribute('data-col-label');
      var colKey = td.getAttribute('data-col-key');

      // Build tooltip text
      var text = pctl + 'th percentile';

      // Add league average if available
      var meta = DataStore.metadata;
      if (meta && meta.leagueAverages) {
        // Try to find which pitch type this row belongs to
        var tr = td.closest('tr');
        if (tr && tr._pitcherName) {
          // For simplicity, just show the percentile
        }
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

  // ---- Keyboard Navigation ----
  function setupKeyboardNav() {
    document.addEventListener('keydown', function (e) {
      // Only handle when not focused on an input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

      var tbody = document.getElementById('table-body');
      var rows = tbody.querySelectorAll('tr.clickable-row');
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
    for (var i = 0; i < rows.length; i++) {
      rows[i].classList.toggle('keyboard-focus', i === Leaderboard.keyboardFocusIndex);
    }
    if (Leaderboard.keyboardFocusIndex >= 0 && rows[Leaderboard.keyboardFocusIndex]) {
      rows[Leaderboard.keyboardFocusIndex].scrollIntoView({ block: 'nearest' });
    }
  }

  // ---- Column Settings ----
  function setupColumnSettings() {
    var btn = document.getElementById('col-settings-btn');
    var panel = document.getElementById('col-settings-panel');

    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      panel.classList.toggle('open');
      if (panel.classList.contains('open')) {
        buildColumnSettingsPanel();
      }
    });

    // Close on outside click
    document.addEventListener('click', function (e) {
      if (!panel.contains(e.target) && e.target !== btn) {
        panel.classList.remove('open');
      }
    });
  }

  function buildColumnSettingsPanel() {
    var panel = document.getElementById('col-settings-panel');
    panel.innerHTML = '';

    var columns = COLUMNS[currentTab];
    var currentGroup = '';

    columns.forEach(function (col) {
      if (col.noToggle) return;

      // Group header
      if (col.group && col.group !== currentGroup) {
        currentGroup = col.group;
        var groupLabel = document.createElement('div');
        groupLabel.className = 'col-settings-group-label';
        groupLabel.textContent = currentGroup.charAt(0).toUpperCase() + currentGroup.slice(1);
        panel.appendChild(groupLabel);
      }

      var label = document.createElement('label');
      var cb = document.createElement('input');
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

  // ---- Dark Mode ----
  function setupDarkMode() {
    var toggle = document.getElementById('dark-mode-toggle');
    var sunIcon = toggle.querySelector('.icon-sun');
    var moonIcon = toggle.querySelector('.icon-moon');

    // Check saved preference or system preference
    var saved = localStorage.getItem('darkMode');
    if (saved === 'true' || (saved === null && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      enableDark();
    }

    toggle.addEventListener('click', function () {
      if (document.body.classList.contains('dark')) {
        disableDark();
      } else {
        enableDark();
      }
      // Re-render to update percentile colors
      refresh();
    });

    function enableDark() {
      document.body.classList.add('dark');
      sunIcon.style.display = 'none';
      moonIcon.style.display = '';
      localStorage.setItem('darkMode', 'true');
    }

    function disableDark() {
      document.body.classList.remove('dark');
      sunIcon.style.display = '';
      moonIcon.style.display = 'none';
      localStorage.setItem('darkMode', 'false');
    }
  }

  // ---- URL State ----
  function saveURLState() {
    var params = {
      tab: currentTab,
      team: teamSelect.value,
      throws: throwsSelect.value,
      min: minCountInput.value,
      search: searchInput.value,
      sort: Leaderboard.currentSort.key || '',
      dir: Leaderboard.currentSort.dir || '',
      page: Leaderboard.currentPage.toString(),
    };
    if (selectedPitchTypes.length > 0) {
      params.pitch = selectedPitchTypes.join(',');
    }
    Utils.writeHash(params);
  }

  function applyURLState() {
    var params = Utils.readHash();
    if (!params || Object.keys(params).length === 0) return;

    if (params.tab) {
      currentTab = params.tab;
      document.querySelectorAll('.tab').forEach(function (t) {
        t.classList.toggle('active', t.getAttribute('data-tab') === currentTab);
      });
      document.getElementById('pitch-type-filter-group').style.display =
        currentTab === 'pitch' ? '' : 'none';
    }
    if (params.team) teamSelect.value = params.team;
    if (params.throws) throwsSelect.value = params.throws;
    if (params.min) minCountInput.value = params.min;
    if (params.search) searchInput.value = params.search;
    if (params.sort) Leaderboard.currentSort.key = params.sort;
    if (params.dir) Leaderboard.currentSort.dir = params.dir;
    if (params.page) Leaderboard.currentPage = parseInt(params.page) || 1;

    // Restore pitch type chips
    if (params.pitch) {
      var types = params.pitch.split(',');
      var chips = document.querySelectorAll('.pitch-chip');
      types.forEach(function (pt) {
        chips.forEach(function (chip) {
          if (chip.getAttribute('data-pitch') === pt) {
            togglePitchChip(pt, chip);
          }
        });
      });
    }
  }

  // Start the app
  init();
})();
