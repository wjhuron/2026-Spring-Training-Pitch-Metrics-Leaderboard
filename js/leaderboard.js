var COLUMNS = {
  pitch: [
    { key: '_rank',       label: '#',        format: function(v){ return v; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, group: 'info', width: '36px' },
    { key: '_compare',    label: '',         format: function(){ return ''; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, isCompare: true, group: 'info', width: '32px' },
    { key: 'pitcher',     label: 'Pitcher',  format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', sticky: true, cls: 'col-pitcher', noPercentile: true, noToggle: true, group: 'info' },
    { key: 'team',        label: 'Team',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', noPercentile: true, group: 'info', isTeam: true },
    { key: 'throws',      label: 'Throws',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', noPercentile: true, group: 'info' },
    { key: 'pitchType',   label: 'Pitch',    format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', noPercentile: true, group: 'info', isPitchType: true },
    { key: 'count',       label: 'Count',    format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'usagePct',    label: 'Usage%',   format: Utils.formatPct, sortType: 'numeric', noPercentile: true, group: 'info' },
    // Metrics
    { key: 'velocity',    label: 'Velo',     format: Utils.formatDecimal(1), sortType: 'numeric', sectionStart: true, group: 'metrics' },
    { key: 'spinRate',    label: 'Spin',     format: Utils.formatInt, sortType: 'numeric', group: 'metrics' },
    { key: 'breakTilt',   label: 'Tilt',     format: Utils.formatTilt, sortType: 'numeric', sortKey: 'breakTiltMinutes', noPercentile: true, group: 'metrics' },
    { key: 'indVertBrk',  label: 'IVB',      format: Utils.formatDecimal(1), sortType: 'numeric', group: 'metrics' },
    { key: 'horzBrk',     label: 'HB',       format: Utils.formatDecimal(1), sortType: 'numeric', group: 'metrics' },
    { key: 'relPosZ',     label: 'RelZ',     format: Utils.formatDecimal(1), sortType: 'numeric', group: 'metrics', defaultHidden: true },
    { key: 'relPosX',     label: 'RelX',     format: Utils.formatDecimal(1), sortType: 'numeric', group: 'metrics', defaultHidden: true },
    { key: 'extension',   label: 'Ext',      format: Utils.formatDecimal(1), sortType: 'numeric', group: 'metrics' },
    { key: 'vaa',         label: 'VAA',      format: Utils.formatDecimal(2), sortType: 'numeric', group: 'metrics', defaultHidden: true },
    { key: 'haa',         label: 'HAA',      format: Utils.formatDecimal(2), sortType: 'numeric', group: 'metrics', defaultHidden: true },
    { key: 'vra',         label: 'VRA',      format: Utils.formatDecimal(2), sortType: 'numeric', group: 'metrics', defaultHidden: true },
    { key: 'hra',         label: 'HRA',      format: Utils.formatDecimal(2), sortType: 'numeric', group: 'metrics', defaultHidden: true },
    // Stats
    { key: 'izPct',       label: 'IZ%',      format: Utils.formatPct, sortType: 'numeric', sectionStart: true, group: 'stats' },
    { key: 'swStrPct',    label: 'SwStr%',   format: Utils.formatPct, sortType: 'numeric', group: 'stats' },
    { key: 'cswPct',      label: 'CSW%',     format: Utils.formatPct, sortType: 'numeric', group: 'stats' },
    { key: 'chasePct',    label: 'Chase%',   format: Utils.formatPct, sortType: 'numeric', group: 'stats' },
    { key: 'gbPct',       label: 'GB%',      format: Utils.formatPct, sortType: 'numeric', group: 'stats' },
  ],
  pitcher: [
    { key: '_rank',       label: '#',        format: function(v){ return v; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, group: 'info', width: '36px' },
    { key: '_compare',    label: '',         format: function(){ return ''; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, isCompare: true, group: 'info', width: '32px' },
    { key: 'pitcher',     label: 'Pitcher',  format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', sticky: true, cls: 'col-pitcher', noPercentile: true, noToggle: true, group: 'info' },
    { key: 'team',        label: 'Team',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', noPercentile: true, group: 'info', isTeam: true },
    { key: 'throws',      label: 'Throws',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', noPercentile: true, group: 'info' },
    { key: 'count',       label: 'Count',    format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'izPct',       label: 'IZ%',      format: Utils.formatPct, sortType: 'numeric', sectionStart: true, group: 'stats' },
    { key: 'swStrPct',    label: 'SwStr%',   format: Utils.formatPct, sortType: 'numeric', group: 'stats' },
    { key: 'cswPct',      label: 'CSW%',     format: Utils.formatPct, sortType: 'numeric', group: 'stats' },
    { key: 'chasePct',    label: 'Chase%',   format: Utils.formatPct, sortType: 'numeric', group: 'stats' },
    { key: 'gbPct',       label: 'GB%',      format: Utils.formatPct, sortType: 'numeric', group: 'stats' },
  ],
  hitter: [
    { key: '_rank',       label: '#',        format: function(v){ return v; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, group: 'info', width: '36px' },
    { key: 'hitter',      label: 'Hitter',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', sticky: true, cls: 'col-pitcher', noPercentile: true, noToggle: true, group: 'info' },
    { key: 'team',        label: 'Team',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', noPercentile: true, group: 'info', isTeam: true },
    { key: 'stands',      label: 'Stands',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', noPercentile: true, group: 'info' },
    { key: 'count',       label: 'Pitches',  format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'nSwings',    label: 'Swings',   format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    // Discipline
    { key: 'swingPct',    label: 'Swing%',   format: Utils.formatPct, sortType: 'numeric', sectionStart: true, group: 'discipline' },
    { key: 'izSwingPct',  label: 'IZSw%',    format: Utils.formatPct, sortType: 'numeric', group: 'discipline' },
    { key: 'chasePct',    label: 'Chase%',   format: Utils.formatPct, sortType: 'numeric', group: 'discipline' },
    { key: 'izSwChase',   label: 'IZSw-Ch',  format: Utils.formatPct, sortType: 'numeric', group: 'discipline' },
    { key: 'whiffPct',    label: 'Whiff%',   format: Utils.formatPct, sortType: 'numeric', group: 'discipline' },
    // Quality
    { key: 'medEV',       label: 'Med EV',   format: Utils.formatDecimal(1), sortType: 'numeric', sectionStart: true, group: 'quality' },
    { key: 'maxEV',       label: 'Max EV',   format: Utils.formatDecimal(1), sortType: 'numeric', group: 'quality' },
    { key: 'barrelPct',   label: 'Barrel%',  format: Utils.formatPct, sortType: 'numeric', group: 'quality' },
    { key: 'xBA',         label: 'xBA',      format: Utils.formatDecimal(3), sortType: 'numeric', group: 'quality' },
    { key: 'xSLG',        label: 'xSLG',     format: Utils.formatDecimal(3), sortType: 'numeric', group: 'quality' },
    // Batted Ball
    { key: 'gbPct',       label: 'GB%',      format: Utils.formatPct, sortType: 'numeric', sectionStart: true, group: 'batted_ball' },
    { key: 'ldPct',       label: 'LD%',      format: Utils.formatPct, sortType: 'numeric', group: 'batted_ball' },
    { key: 'fbPct',       label: 'FB%',      format: Utils.formatPct, sortType: 'numeric', group: 'batted_ball' },
    { key: 'medLA',       label: 'Med LA',   format: Utils.formatDecimal(1), sortType: 'numeric', group: 'batted_ball' },
  ],
};

var Leaderboard = {
  currentSort: { key: null, dir: 'desc' },
  hiddenColumns: {},  // key -> true if hidden
  showLeagueAvg: false,
  currentPage: 1,
  pageSize: 50,
  lastRenderedData: null,
  lastRenderedColumns: null,
  selectedForCompare: {},  // pitcher name -> true
  keyboardFocusIndex: -1,

  initHiddenColumns: function () {
    var allCols = COLUMNS.pitch.concat(COLUMNS.pitcher).concat(COLUMNS.hitter);
    for (var i = 0; i < allCols.length; i++) {
      if (allCols[i].defaultHidden) {
        this.hiddenColumns[allCols[i].key] = true;
      }
    }
  },

  getVisibleColumns: function (columns) {
    var self = this;
    return columns.filter(function (col) {
      return !self.hiddenColumns[col.key];
    });
  },

  sortData: function (data, columnKey, columns) {
    var col = null;
    for (var i = 0; i < columns.length; i++) {
      if (columns[i].key === columnKey) { col = columns[i]; break; }
    }
    if (!col || col.sortType === null) return data;

    var sortKey = col.sortKey || col.key;

    if (this.currentSort.key === columnKey) {
      this.currentSort.dir = this.currentSort.dir === 'desc' ? 'asc' : 'desc';
    } else {
      this.currentSort.key = columnKey;
      this.currentSort.dir = col.sortType === 'string' ? 'asc' : 'desc';
    }

    var dir = this.currentSort.dir === 'asc' ? 1 : -1;

    data.sort(function (a, b) {
      var va = a[sortKey];
      var vb = b[sortKey];
      if (va === null || va === undefined) {
        if (vb === null || vb === undefined) return 0;
        return 1;
      }
      if (vb === null || vb === undefined) return -1;
      if (col.sortType === 'string') return dir * String(va).localeCompare(String(vb));
      return dir * (va - vb);
    });

    return data;
  },

  computeLeagueAvgRow: function (data, columns) {
    var avg = {};
    var numericKeys = [];
    for (var i = 0; i < columns.length; i++) {
      if (columns[i].sortType === 'numeric' && !columns[i].noPercentile && columns[i].key !== '_rank') {
        numericKeys.push(columns[i].key);
      }
    }
    numericKeys.forEach(function (key) {
      var sum = 0, count = 0;
      for (var j = 0; j < data.length; j++) {
        var v = data[j][key];
        if (v !== null && v !== undefined) { sum += v; count++; }
      }
      avg[key] = count > 0 ? sum / count : null;
    });
    avg.pitcher = 'League Avg';
    avg._isLeagueAvg = true;
    avg._rank = '';
    return avg;
  },

  render: function (data, columns) {
    var self = this;
    var visCols = this.getVisibleColumns(columns);
    var headerRow = document.getElementById('table-header');
    var tbody = document.getElementById('table-body');
    var noResults = document.getElementById('no-results');
    var isDark = document.body.classList.contains('dark');

    this.lastRenderedData = data;
    this.lastRenderedColumns = columns;

    // Pagination
    var totalRows = data.length;
    var pageSize = this.pageSize;
    var totalPages = pageSize > 0 ? Math.max(1, Math.ceil(totalRows / pageSize)) : 1;
    if (this.currentPage > totalPages) this.currentPage = totalPages;
    var startIdx = pageSize > 0 ? (this.currentPage - 1) * pageSize : 0;
    var endIdx = pageSize > 0 ? Math.min(startIdx + pageSize, totalRows) : totalRows;
    var pageData = data.slice(startIdx, endIdx);

    // Update pagination UI
    var pageInfo = document.getElementById('page-info');
    var pagePrev = document.getElementById('page-prev');
    var pageNext = document.getElementById('page-next');
    if (pageInfo) pageInfo.textContent = 'Page ' + this.currentPage + ' of ' + totalPages;
    if (pagePrev) pagePrev.disabled = this.currentPage <= 1;
    if (pageNext) pageNext.disabled = this.currentPage >= totalPages;

    // Build header
    headerRow.innerHTML = '';
    visCols.forEach(function (col) {
      var th = document.createElement('th');
      if (col.isCompare) {
        th.classList.add('col-compare');
        th.style.width = col.width || 'auto';
      } else {
        th.textContent = col.label;
      }
      th.setAttribute('data-key', col.key);
      if (col.align) th.classList.add('align-' + col.align);
      if (col.sticky) th.classList.add('sticky-col');
      if (col.sectionStart) th.classList.add('section-start');
      if (col.width) th.style.width = col.width;
      if (Utils.TOOLTIPS[col.label]) th.title = Utils.TOOLTIPS[col.label];

      if (self.currentSort.key === col.key) {
        th.classList.add('sorted', self.currentSort.dir);
      }

      if (col.sortType !== null) {
        th.addEventListener('click', function () {
          self.sortData(data, col.key, columns);
          self.currentPage = 1;
          self.render(data, columns);
        });
      }

      headerRow.appendChild(th);
    });

    // Build body
    tbody.innerHTML = '';

    if (data.length === 0) {
      noResults.style.display = '';
      document.getElementById('row-count').textContent = '0';
      document.getElementById('pagination').style.display = 'none';
      return;
    }
    noResults.style.display = 'none';
    document.getElementById('pagination').style.display = '';

    var fragment = document.createDocumentFragment();

    // League average row
    if (this.showLeagueAvg && pageData.length > 0) {
      var avgRow = this.computeLeagueAvgRow(data, visCols);
      var avgTr = this._createRow(avgRow, visCols, -1, isDark, true);
      avgTr.classList.add('league-avg-row');
      fragment.appendChild(avgTr);
    }

    // Data rows
    for (var ri = 0; ri < pageData.length; ri++) {
      var row = pageData[ri];
      var globalRank = startIdx + ri + 1;
      var tr = this._createRow(row, visCols, globalRank, isDark, false);
      tr.classList.add('clickable-row');
      tr._pitcherName = row.pitcher || row.hitter;
      tr._rowIndex = ri;

      // Click handler
      (function (r, idx) {
        tr.addEventListener('click', function (e) {
          // Don't trigger on compare checkbox clicks
          if (e.target.type === 'checkbox') return;
          // Remove active from all rows
          var prev = tbody.querySelectorAll('.active-row');
          for (var k = 0; k < prev.length; k++) prev[k].classList.remove('active-row');
          // Highlight all rows for this person
          var personName = r.pitcher || r.hitter;
          var allRows = tbody.querySelectorAll('tr');
          allRows.forEach(function (row) {
            if (row._pitcherName === personName) row.classList.add('active-row');
          });
          self.keyboardFocusIndex = idx;
          if (typeof App !== 'undefined' && App.openSidePanel) {
            App.openSidePanel(personName, r.team, r.throws || r.stands, r);
          }
        });
      })(row, ri);

      if (this.keyboardFocusIndex === ri) {
        tr.classList.add('keyboard-focus');
      }

      fragment.appendChild(tr);
    }

    tbody.appendChild(fragment);
    document.getElementById('row-count').textContent = totalRows;
  },

  _createRow: function (row, visCols, rank, isDark, isAvgRow) {
    var self = this;
    var tr = document.createElement('tr');

    visCols.forEach(function (col) {
      var td = document.createElement('td');

      // Special: rank column
      if (col.key === '_rank') {
        td.textContent = isAvgRow ? '' : rank;
        td.classList.add('col-rank');
        if (col.align) td.classList.add('align-' + col.align);
        tr.appendChild(td);
        return;
      }

      // Special: compare checkbox column
      if (col.isCompare) {
        td.classList.add('col-compare');
        if (!isAvgRow) {
          var cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.checked = !!self.selectedForCompare[row.pitcher];
          cb.addEventListener('change', function () {
            if (cb.checked) {
              self.selectedForCompare[row.pitcher] = true;
            } else {
              delete self.selectedForCompare[row.pitcher];
            }
            if (typeof App !== 'undefined' && App.updateCompareButton) {
              App.updateCompareButton();
            }
          });
          td.appendChild(cb);
        }
        tr.appendChild(td);
        return;
      }

      // Special: pitch type badge
      if (col.isPitchType && row[col.key] && !isAvgRow) {
        var badge = document.createElement('span');
        badge.className = 'pitch-badge';
        badge.textContent = row[col.key];
        var pitchColor = Utils.getPitchColor(row[col.key]);
        badge.style.backgroundColor = pitchColor;
        // Ensure readability for light colors
        if (row[col.key] === 'SI' || row[col.key] === 'SV') {
          badge.style.color = '#1a1a2e';
        }
        td.appendChild(badge);
        if (col.align) td.classList.add('align-' + col.align);
        tr.appendChild(td);
        return;
      }

      // Regular cell
      var val = row[col.key];
      td.textContent = col.format(val);
      if (col.align) td.classList.add('align-' + col.align);
      if (col.sticky) td.classList.add('sticky-col');
      if (col.cls) td.classList.add(col.cls);
      if (col.sectionStart) td.classList.add('section-start');
      if (val === null || val === undefined) td.classList.add('col-null');

      // Percentile coloring
      if (!col.noPercentile && !isAvgRow) {
        var pctlKey = col.key + '_pctl';
        var pctl = row[pctlKey];
        if (pctl !== null && pctl !== undefined) {
          if (isDark) {
            td.style.backgroundColor = Utils.percentileColorDark(pctl);
            td.style.color = Utils.percentileTextColorDark(pctl);
          } else {
            td.style.backgroundColor = Utils.percentileColor(pctl);
            td.style.color = Utils.percentileTextColor(pctl);
          }
          // Store percentile for tooltip
          td.setAttribute('data-pctl', pctl);
          td.setAttribute('data-col-key', col.key);
          td.setAttribute('data-col-label', col.label);
        }
      }

      tr.appendChild(td);
    });

    return tr;
  },

  getCompareList: function () {
    return Object.keys(this.selectedForCompare);
  },

  clearCompare: function () {
    this.selectedForCompare = {};
  },
};

// Initialize hidden columns on load
Leaderboard.initHiddenColumns();
