// Cache-bust the data fetch with the same ?v= build tag the pipeline
// stamps onto this script's own <script> tag in index.html (via
// bump_asset_version). Captured at parse time — document.currentScript
// is only valid while the script is executing synchronously.
var DATA_VERSION = (function () {
  try {
    var src = (document.currentScript && document.currentScript.src) || '';
    var m = src.match(/[?&]v=([\w-]+)/);
    return m ? m[1] : '';
  } catch (e) {
    return '';
  }
})();

const DataStore = {
  rs: {},

  active: function () {
    return this.rs;
  },

  /**
   * Fetch the gzipped payload and inflate it in the browser.
   *
   * The data used to be a ~100 MB `window.RS_DATA = {...}` inline script
   * (over GitHub's 100 MB file wall, slow to download). It now ships as
   * data/data_embedded.json.gz (~13-16 MB) and is inflated here via the
   * native DecompressionStream. Returns a Promise — app.js already
   * chains .then()/.catch() on this, so the async path needs no caller
   * changes; a fetch/inflate failure routes to app.js's existing
   * "Error loading data. Please refresh." handler.
   */
  // Two-stage load (2026-07-29). data_core.json.gz carries the leaderboard
  // tables + metadata (~5 MB gz) and renders the first table in well under a
  // second; data_heavy.json.gz (microData + pitchDetails + swing locations,
  // ~33 MB gz) prefetches in the background immediately after and powers
  // client-side filters and player pages. Consumers that need heavy data
  // gate on DataStore.heavyReady / DataStore.whenHeavy(cb) — with the
  // prefetch starting at once, those gates only bite in the first seconds.
  heavyReady: false,
  _heavyCallbacks: [],

  whenHeavy: function (cb) {
    if (this.heavyReady) { cb(); return; }
    this._heavyCallbacks.push(cb);
  },

  _fetchGz: function (name) {
    var url = 'data/' + name + (DATA_VERSION ? ('?v=' + DATA_VERSION) : '');
    return fetch(url).then(function (resp) {
      if (!resp.ok) throw new Error('Data fetch failed: HTTP ' + resp.status + ' (' + name + ')');
      if (!resp.body) throw new Error('Data fetch returned no body stream (' + name + ')');
      var inflated = resp.body.pipeThrough(new DecompressionStream('gzip'));
      return new Response(inflated).text();
    }).then(function (text) { return JSON.parse(text); });
  },

  load: function () {
    var self = this;

    if (typeof DecompressionStream === 'undefined') {
      return Promise.reject(new Error(
        'This browser does not support gzip DecompressionStream. ' +
        'Please use a current version of Chrome, Firefox, Safari, or Edge.'));
    }

    return this._fetchGz('data_core.json.gz').then(function (rd) {
      self.rs = {
        pitcherData: rd.pitcherData || [],
        pitchData: rd.pitchData || [],
        hitterData: rd.hitterData || [],
        hitterPitchData: rd.hitterPitchData || [],
        metadata: rd.metadata || {},
        microData: null,
        pitchDetails: {},
        hitterPitchDetails: {},
        hitterSwingLocations: {},
      };
      self.updateGlobals();
      // background prefetch — never blocks the first render
      self._loadHeavy();
    });
  },

  _loadHeavy: function () {
    var self = this;
    this._fetchGz('data_heavy.json.gz').then(function (rd) {
      self.rs.microData = rd.microData || null;
      self.rs.pitchDetails = rd.pitchDetails || {};
      self.rs.hitterPitchDetails = rd.hitterPitchDetails || {};
      self.rs.hitterSwingLocations = rd.hitterSwingLocations || {};
      self.updateGlobals();
      self.heavyReady = true;
      var cbs = self._heavyCallbacks; self._heavyCallbacks = [];
      cbs.forEach(function (cb) { try { cb(); } catch (e) { console.error(e); } });
    }).catch(function (e) {
      // Filters/player pages stay in their degraded state; leaderboards work.
      console.error('Heavy data chunk failed to load:', e);
    });
  },

  updateGlobals: function () {
    const d = this.rs;
    window.PITCHER_DATA = d.pitcherData;
    window.PITCH_DATA = d.pitchData;
    window.HITTER_DATA = d.hitterData;
    window.HITTER_PITCH_LB = d.hitterPitchData;
    window.METADATA = d.metadata;
    window.MICRO_DATA = d.microData;
    window.PITCH_DETAILS = d.pitchDetails;
    window.HITTER_PITCH_DETAILS = d.hitterPitchDetails;
    window.HITTER_SWING_LOCATIONS = d.hitterSwingLocations;

    this.metadata = d.metadata;
    this.pitcherData = d.pitcherData;
    this.pitchData = d.pitchData;
    this.hitterData = d.hitterData;
    this.hitterPitchData = d.hitterPitchData;
  },

  /**
   * Smart filter: uses Aggregator when date/hand filters are active,
   * otherwise falls back to pre-aggregated data.
   * @param {'pitcher'|'pitch'|'hitter'|'hitterPitch'} tab - Data source tab.
   * @param {FilterState} filters - Current filter state.
   * @returns {(PitcherRow|PitchRow|HitterRow)[]} Filtered row array.
   */
  getFilteredDataV2: function (tab, filters) {
    if (filters.viewMode === 'team') {
      // Team aggregation requires micro data; the toggle is hidden when the
      // Aggregator failed to load, so this is just a safety net.
      return Aggregator.loaded ? Aggregator.aggregate(tab, filters) : [];
    }
    if (Aggregator.needsReaggregation(filters)) {
      return Aggregator.aggregate(tab, filters);
    }
    return this.getFilteredData(tab, filters);
  },

  /**
   * Filter pre-aggregated data based on current filters.
   * @param {'pitcher'|'pitch'|'hitter'|'hitterPitch'} tab - Data source tab.
   * @param {FilterState} filters - Current filter state (pitchTypes is always an array).
   * @returns {(PitcherRow|PitchRow|HitterRow)[]} Filtered row array.
   */
  getFilteredData: function (tab, filters) {
    const d = this.rs;
    let source;
    if (tab === 'pitch') source = d.pitchData;
    else if (tab === 'pitcher') source = d.pitcherData;
    else if (tab === 'hitter') source = d.hitterData;
    else if (tab === 'hitterPitch') source = d.hitterPitchData;
    if (!source) return [];

    const isHitter = (tab === 'hitter' || tab === 'hitterPitch');
    const hasPitchType = (tab === 'pitch' || tab === 'hitterPitch');
    const selectedPitchTypes = filters.pitchTypes; // always array

    const rocTeamsArr = (this.metadata && this.metadata.rocTeams) || [];
    var rocTeamSet = {};
    for (var ri = 0; ri < rocTeamsArr.length; ri++) rocTeamSet[rocTeamsArr[ri]] = true;
    // Team games for per-team qualifying thresholds
    var _teamGames = (filters.minIp === 'Q' || filters.minCount === 'Q')
      ? (Aggregator.loaded ? Aggregator.getTeamGamesPlayed() : {}) : {};

    // Multi-team support: scan once to build player→combined-row map and cumulative team games.
    // When "All Teams" is selected, per-team rows of multi-team players are hidden; the 2TM/3TM
    // row stands in. Qualification for multi-team players uses combined IP/PA and summed team games.
    var combinedByPlayer = {};
    var isCombinedRe = /^\d+TM$/;
    // Key on mlbId when present so two distinct players sharing a name (e.g. two
    // "Max Muncy") don't collide; fall back to name only when no id exists.
    var playerKey = function (r) {
      return (r.mlbId != null && r.mlbId !== '')
        ? ('id:' + r.mlbId)
        : ('nm:' + (r.pitcher || r.hitter || ''));
    };
    for (var di2 = 0; di2 < source.length; di2++) {
      var drow = source[di2];
      if (isCombinedRe.test(drow.team)) {
        if (drow.pitcher || drow.hitter) combinedByPlayer[playerKey(drow)] = drow;
      }
    }
    // For multi-team players, the qualifier denominator is max(team games) across
    // their MLB teams — approximates tenure span. Summing would double-count and
    // inflate the threshold past what any traded player could realistically meet.
    var cumTeamGames = {};
    if (filters.minIp === 'Q' || filters.minCount === 'Q') {
      for (var di3 = 0; di3 < source.length; di3++) {
        var drow2 = source[di3];
        var pk2 = playerKey(drow2);
        if ((drow2.pitcher || drow2.hitter) && combinedByPlayer[pk2] && !isCombinedRe.test(drow2.team)) {
          var tgv = _teamGames[drow2.team] || 0;
          if (tgv > (cumTeamGames[pk2] || 0)) cumTeamGames[pk2] = tgv;
        }
      }
    }
    return source.filter(function (row) {
      // Hide ROC players unless user explicitly selected their team
      if (rocTeamSet[row.team] && filters.team !== row.team) return false;
      // Multi-team: "All Teams" view shows only the combined row for multi-team players.
      // Specific-team view shows only per-team rows (combined row hidden).
      var pkey = playerKey(row);
      var isCombinedRow = isCombinedRe.test(row.team);
      if (filters.team === 'all') {
        if (combinedByPlayer[pkey] && !isCombinedRow) return false;
      } else {
        if (isCombinedRow) return false;
      }
      if (filters.team !== 'all' && row.team !== filters.team) return false;

      // Throws filter applies to pitchers; stands filter applies to hitters (same dropdown)
      if (filters.throws !== 'all') {
        if (isHitter) {
          if (row.stands !== filters.throws) return false;
        } else {
          if (row.throws !== filters.throws) return false;
        }
      }

      // SP/RP role filter (pitcher tabs only)
      if (filters.role && filters.role !== 'all' && !isHitter) {
        let g = row.g, gs = row.gs;
        // For pitch-level rows without G/GS, look up from pitcher role cache
        if (g == null && row.pitcher) {
          if (!DataStore._roleCache) {
            DataStore._roleCache = {};
            const pData = DataStore.rs.pitcherData || [];
            for (let pi = 0; pi < pData.length; pi++) {
              const rk = pData[pi].pitcher + '|' + pData[pi].team;
              DataStore._roleCache[rk] = { g: pData[pi].g, gs: pData[pi].gs };
            }
          }
          const cached = DataStore._roleCache[row.pitcher + '|' + row.team];
          if (cached) { g = cached.g; gs = cached.gs; }
        }
        g = g || 0; gs = gs || 0;
        const isStarter = g > 0 && (gs / g) > QUAL.SP_GS_RATIO;
        if (filters.role === 'SP' && !isStarter) return false;
        if (filters.role === 'RP' && isStarter) return false;
      }

      if (hasPitchType && selectedPitchTypes.indexOf('all') === -1) {
        if (selectedPitchTypes.indexOf(row.pitchType) === -1) return false;
      }
      // For multi-team players, qualification uses the combined row's stats and
      // the cumulative team games across their MLB teams.
      var mtRow = combinedByPlayer[pkey];
      var _tg = (mtRow && !isCombinedRow) ? (cumTeamGames[pkey] || 0) : (_teamGames[row.team] || 0);
      var _qPa = (mtRow && !isCombinedRow) ? (mtRow.pa || 0) : (row.pa || 0);
      var _qIp = (mtRow && !isCombinedRow) ? mtRow.ip : row.ip;
      var _qG = (mtRow && !isCombinedRow) ? mtRow.g : row.g;
      var _qGs = (mtRow && !isCombinedRow) ? mtRow.gs : row.gs;
      // ROC-aware qualification (3.1 PA×TG MLB / 2.7 ROC for hitters;
      // 1.0/0.5 MLB & 0.8/0.4 ROC IP×TG for pitchers).
      var _isROC = (typeof Aggregator !== 'undefined') &&
                   Aggregator._isROCTeam && Aggregator._isROCTeam(row.team);
      // Min count: use PA for hitters, pitch count for pitchers and hitterPitch
      if (tab === 'hitter') {
        if (filters.minCount === 'Q') {
          if (_qPa < _tg * Utils.hitterPaPerGame(_isROC)) return false;
        } else if ((row.pa || 0) < filters.minCount) return false;
      } else {
        if (row.count < filters.minCount) return false;
      }
      if (tab === 'hitter' && filters.minSwings && row.nSwings < filters.minSwings) return false;
      if (tab === 'pitcher' && filters.minTbf && (row.pa || 0) < filters.minTbf) return false;
      if (tab === 'pitcher' && filters.minIp) {
        if (filters.minIp === 'Q') {
          var ipFloat = Utils.parseIP(_qIp);
          var isStarter = Utils.isStarter(_qG, _qGs);
          var ipThresh = _tg * Utils.pitcherIpPerGame(isStarter, _isROC);
          if (ipFloat < ipThresh) return false;
        } else if ((row.ip || 0) < filters.minIp) return false;
      }
      if ((tab === 'pitcher' || tab === 'hitter') && filters.minBip && row.nBip != null && row.nBip < filters.minBip) return false;
      if (tab === 'pitcher' && filters.minPitcherSwings && row.nSwings != null && row.nSwings < filters.minPitcherSwings) return false;
      if (filters.search) {
        const name = (row.pitcher || row.hitter || '').toLowerCase();
        if (name.indexOf(filters.search.toLowerCase()) === -1) return false;
      }
      return true;
    });
  },
};
