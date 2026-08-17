/**
 * Client-side aggregator for micro-aggregate data.
 * Enables filtering by opponent hand and date range by
 * summing pre-computed counts and recomputing final stats + percentiles.
 */

// ---- Qualification & classification constants ----
// These thresholds appear across aggregator, data, and leaderboard modules.
// Centralizing them here prevents silent drift when one file is updated but not another.
var QUAL = {
  // Hitter qualification: PA >= teamGames × per-game multiplier.
  PA_PER_GAME:       3.1,   // MLB hitter
  PA_PER_GAME_MILB:  2.7,   // ROC / MiLB hitter
  // Pitcher qualification: IP >= teamGames × per-game multiplier.
  // Reliever multiplier (0.30) matches FanGraphs' relief-pitching
  // leaderboard exactly: empirically, IP >= 0.30 × teamGames reproduces
  // FG's qualified-reliever set with zero mismatches across all 30 teams.
  // ROC reliever (0.24) keeps the same 0.80 ROC-to-MLB scaling as
  // starters. Keep in sync with pipeline/utils.py QUAL_* constants.
  SP_IP_PER_GAME:      1.0,  // MLB starter
  RP_IP_PER_GAME:      0.30, // MLB reliever (FanGraphs-matched)
  SP_IP_PER_GAME_MILB: 0.8,  // ROC / MiLB starter
  RP_IP_PER_GAME_MILB: 0.24, // ROC / MiLB reliever
  SP_GS_RATIO:       0.5,   // Starter if GS/G > 0.5
  HARD_HIT_MPH:      95,    // Exit velo >= 95 mph = hard hit
  MIN_BIP_PCTL:      25,    // Minimum BIP for batted-ball percentile coloring
  MIN_BAT_TRACKING:  10,    // Minimum competitive swings for bat tracking stats (kept low: bat-tracking samples are naturally scarce)
  MIN_SPRINT_RUNS:   10,    // Minimum competitive runs for sprint speed (kept low: sprint runs are naturally scarce)
  MIN_PITCH_PCTL:    25,    // Minimum pitches for outcome-metric percentiles (pitch-type level)
  MIN_HITTER_PT:     25,    // Minimum pitches seen for hitter pitch-type percentile coloring
  // Loc+ is the one pitch-type outcome metric that cannot ride the flat 25.
  // Displayed Loc+ is an UNSHRUNK mean (pipeline_locplus.py zeroes N_PRIOR_PT
  // for the coherent-canon ledger property), so reliability at n pitches is
  // n/(n+k) where k is that pitch type's measured split-half r=0.5 crossing.
  // At 25 pitches an FF cell is only 25/(25+71) = 0.26 reliable and a cutter
  // 0.18; gating at k itself means "color only when >= half the variance is
  // signal". Keep in sync with pipeline_locplus.py STABILIZE_N_PT.
  MIN_PITCH_LOCPLUS:         { FF: 73, SI: 81, FC: 74, SL: 67, CU: 83, CH: 79 },
  // Pitch types with NO measured constant (today only EP: 40 cells, largest
  // 105 pitches, so k can't be measured). Policy, not a measurement: don't
  // color what we can't validate. Colors zero cells today either way.
  // Note KN/SC are NOT unmeasured — they map to the CH group below.
  MIN_PITCH_LOCPLUS_DEFAULT: Infinity,
  MIN_SACQ:          20,    // Minimum zone count for SACQ wOBA lookup
  MIN_ELLIPSE_PTS:   6,     // Minimum points for scatter ellipse computation
};

// Pitch type -> Loc+ group. Mirrors pipeline_locplus.py GROUP exactly; the
// stabilization constants above are measured per GROUP, not per raw type.
QUAL.LOCPLUS_GROUP = {
  FF: 'FF', FA: 'FF',
  SI: 'SI',
  FC: 'FC', CF: 'FC',
  SL: 'SL', ST: 'SL', SW: 'SL', SV: 'SL',
  CU: 'CU', KC: 'CU', CS: 'CU',
  CH: 'CH', FS: 'CH', KN: 'CH', SC: 'CH'
};
// Category rows (PITCH_CATEGORIES) pool several types — take the stiffest
// member gate. Mirrors pipeline_locplus.py STABILIZE_N_CATEGORY.
QUAL.LOCPLUS_CATEGORY = { Hard: 81, Breaking: 83, Offspeed: 79 };

// Minimum pitches for a pitch-type (or category) Loc+ cell to clear r >= 0.5.
QUAL.locPlusMinPitches = function (pitchType) {
  if (QUAL.LOCPLUS_CATEGORY[pitchType] != null) return QUAL.LOCPLUS_CATEGORY[pitchType];
  const g = QUAL.LOCPLUS_GROUP[pitchType];
  const k = g ? QUAL.MIN_PITCH_LOCPLUS[g] : null;
  return k != null ? k : QUAL.MIN_PITCH_LOCPLUS_DEFAULT;
};

const Aggregator = {
  data: null,
  loaded: false,
  _colIdx: {},

  _matInvGeneral: function (M) {
    const n = M.length;
    const aug = [];
    for (let i = 0; i < n; i++) {
      aug[i] = [];
      for (let j = 0; j < n; j++) aug[i][j] = M[i][j];
      for (let j2 = 0; j2 < n; j2++) aug[i][n + j2] = (i === j2) ? 1.0 : 0.0;
    }
    for (let col = 0; col < n; col++) {
      let maxRow = col;
      for (let r = col + 1; r < n; r++) {
        if (Math.abs(aug[r][col]) > Math.abs(aug[maxRow][col])) maxRow = r;
      }
      const tmp = aug[col]; aug[col] = aug[maxRow]; aug[maxRow] = tmp;
      if (Math.abs(aug[col][col]) < 1e-15) return null;
      for (let r2 = col + 1; r2 < n; r2++) {
        const f = aug[r2][col] / aug[col][col];
        for (let c = 0; c < 2 * n; c++) aug[r2][c] -= f * aug[col][c];
      }
    }
    for (let col2 = n - 1; col2 >= 0; col2--) {
      const piv = aug[col2][col2];
      for (let c2 = 0; c2 < 2 * n; c2++) aug[col2][c2] /= piv;
      for (let r3 = 0; r3 < col2; r3++) {
        const f2 = aug[r3][col2];
        for (let c3 = 0; c3 < 2 * n; c3++) aug[r3][c3] -= f2 * aug[col2][c3];
      }
    }
    const inv = [];
    for (let ii = 0; ii < n; ii++) inv[ii] = aug[ii].slice(n);
    return inv;
  },

  _mvnConditional: function (modelParams, relValues) {
    const mu = modelParams.mu;
    const cov = modelParams.cov;
    const nAcc = 2; // IVB, HB
    const nRel = mu.length - nAcc;
    if (relValues.length !== nRel) return null;
    const sigmaRel = [];
    for (let i = 0; i < nRel; i++) {
      sigmaRel[i] = [];
      for (let j = 0; j < nRel; j++) sigmaRel[i][j] = cov[nAcc + i][nAcc + j];
    }
    const sigmaRel_inv = this._matInvGeneral(sigmaRel);
    if (!sigmaRel_inv) return null;
    const rDiff = [];
    for (let k = 0; k < nRel; k++) rDiff[k] = relValues[k] - mu[nAcc + k];
    const sriRdiff = [];
    for (let ii = 0; ii < nRel; ii++) {
      let s = 0;
      for (let jj = 0; jj < nRel; jj++) s += sigmaRel_inv[ii][jj] * rDiff[jj];
      sriRdiff[ii] = s;
    }
    const muBar = [];
    for (let a = 0; a < nAcc; a++) {
      let adj = 0;
      for (let b = 0; b < nRel; b++) adj += cov[a][nAcc + b] * sriRdiff[b];
      muBar[a] = mu[a] + adj;
    }
    return muBar;
  },

  load: function (microData) {
    this._roleCache = null;  // Clear role cache on reload
    if (microData) {
      this.data = microData;
      this._buildIndexes();
      this.loaded = true;
      return Promise.resolve();
    }
    if (window.MICRO_DATA) {
      this.data = window.MICRO_DATA;
      this._buildIndexes();
      this.loaded = true;
      return Promise.resolve();
    }
    const self = this;
    // data/micro_data_rs.json is untracked (local pipeline artifact only) —
    // in production microData always arrives via data_heavy.json.gz, so
    // this fetch 404s on Pages and lands in the graceful catch below.
    return fetch('data/micro_data_rs.json')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        self.data = d;
        self._buildIndexes();
        self.loaded = true;
      })
      .catch(function (e) {
        console.warn('Micro data not available (advanced filters disabled):', e);
        self.loaded = false;
      });
  },

  _buildIndexes: function () {
    const d = this.data;
    const tables = ['pitcherCols', 'pitcherBipCols', 'pitchCols', 'hitterCols', 'hitterBipCols', 'hitterPitchCols', 'hitterPitchBipCols', 'veloTrendCols'];
    for (let t = 0; t < tables.length; t++) {
      const key = tables[t];
      this._colIdx[key] = {};
      for (let i = 0; i < d[key].length; i++) {
        this._colIdx[key][d[key][i]] = i;
      }
    }
  },

  // Always reaggregate when loaded -- qualifying thresholds affect percentile pools
  needsReaggregation: function (filters) {
    if (!this.loaded) return false;
    return true;
  },

  _HP_X: 125.42,
  _HP_Y: 198.27,
  // Negative-LA split at -10 — must match process_data.py / HitterCards.py
  // LA_BINS exactly (same edges AND order) or client-recomputed xwOBAsp
  // desyncs from the server sacqZones table.
  _LA_BINS: [[-999,-10],[-10,0],[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,50],[50,999]],

  computeSprayAngle: function (hcX, hcY) {
    if (hcX == null || hcY == null) return null;
    const dx = hcX - this._HP_X;
    const dy = this._HP_Y - hcY;
    if (dy <= 0) return null;
    return Math.atan2(dx, dy) * (180 / Math.PI);
  },

  sprayDirection: function (angle, bats) {
    if (angle == null || !bats) return null;
    if (bats === 'R') {
      if (angle < -30) return 'pull';
      if (angle < -15) return 'pull_side';
      if (angle < 0) return 'center_pull';
      if (angle < 15) return 'center_oppo';
      if (angle < 30) return 'oppo_side';
      return 'oppo';
    } else {
      if (angle > 30) return 'pull';
      if (angle > 15) return 'pull_side';
      if (angle > 0) return 'center_pull';
      if (angle > -15) return 'center_oppo';
      if (angle > -30) return 'oppo_side';
      return 'oppo';
    }
  },

  getLABinIdx: function (la) {
    const bins = this._LA_BINS;
    for (let i = 0; i < bins.length; i++) {
      if (la >= bins[i][0] && la < bins[i][1]) return i;
    }
    return null;
  },

  // Build hand-specific + pooled zone maps from METADATA.sacqZones, plus the
  // LA-only marginals (METADATA.sacqLaZones) used for the sprayVal residual.
  // Returns { hand, pooled, laHand, laPooled } (key→zone maps).
  buildSacqZoneMaps: function () {
    var zones = (window.METADATA && window.METADATA.sacqZones) || [];
    var hand = {}, pooled = {};
    for (var i = 0; i < zones.length; i++) {
      var sz = zones[i];
      if (sz.bats) {
        hand[sz.spray + '|' + sz.laBin + '|' + sz.bats] = sz;
      } else {
        pooled[sz.spray + '|' + sz.laBin] = sz;
      }
    }
    var laZones = (window.METADATA && window.METADATA.sacqLaZones) || [];
    var laHand = {}, laPooled = {};
    for (var j = 0; j < laZones.length; j++) {
      var lz = laZones[j];
      if (lz.bats) {
        laHand[lz.laBin + '|' + lz.bats] = lz;
      } else {
        laPooled[String(lz.laBin)] = lz;
      }
    }
    return { hand: hand, pooled: pooled, laHand: laHand, laPooled: laPooled };
  },

  // LA-only league wOBAcon: hand-specific first, pooled fallback. Mirrors
  // Python la_lookup (same MIN_SACQ convention).
  sacqLaLookup: function (maps, laBin, bats) {
    var info = maps.laHand[laBin + '|' + bats];
    if (info && info.count >= QUAL.MIN_SACQ && info.wobacon != null) return info.wobacon;
    info = maps.laPooled[String(laBin)];
    if (info && info.count >= QUAL.MIN_SACQ && info.wobacon != null) return info.wobacon;
    return null;
  },

  // Look up zone wOBAcon: try hand-specific first, fall back to pooled.
  // Field is "wobacon" going forward; "woba" alias kept for transition (older JSON).
  sacqLookup: function (maps, dir, laBin, bats) {
    var info = maps.hand[dir + '|' + laBin + '|' + bats];
    var v;
    if (info && info.count >= QUAL.MIN_SACQ) {
      v = info.wobacon != null ? info.wobacon : info.woba;
      if (v != null) return v;
    }
    info = maps.pooled[dir + '|' + laBin];
    if (info && info.count >= QUAL.MIN_SACQ) {
      v = info.wobacon != null ? info.wobacon : info.woba;
      if (v != null) return v;
    }
    return null;
  },

  /**
   * Main entry: aggregate micro data for the given tab and filters.
   * @param {'pitcher'|'pitch'|'hitter'|'hitterPitch'} tab - Data source tab.
   * @param {FilterState} filters - Current filter state.
   * @returns {(PitcherRow|PitchRow|HitterRow)[]} Aggregated rows with percentiles.
   */
  aggregate: function (tab, filters) {
    if (tab === 'pitcher') return this._aggregatePitcher(filters);
    if (tab === 'pitch') return this._aggregatePitch(filters);
    if (tab === 'hitter') return this._aggregateHitter(filters);
    if (tab === 'hitterPitch') return this._aggregateHitterPitch(filters);
    return [];
  },

  _getMlbIdMap: function (type) {
    const data = type === 'pitcher' ? (window.PITCHER_DATA || []) : (window.HITTER_DATA || []);
    const nameKey = type === 'pitcher' ? 'pitcher' : 'hitter';
    const map = {};
    for (let i = 0; i < data.length; i++) {
      if (data[i].mlbId) {
        map[data[i][nameKey] + '|' + data[i].team] = data[i].mlbId;
      }
    }
    return map;
  },

  _getValidDateSet: function (filters) {
    const dates = this.data.lookups.dates;
    const minDate = filters.dateStart || '';
    const maxDate = filters.dateEnd || '\uffff';
    const valid = {};
    for (let i = 0; i < dates.length; i++) {
      if (dates[i] >= minDate && dates[i] <= maxDate) {
        valid[i] = true;
      }
    }
    return valid;
  },

  // Helper: check if a team is a ROC/AAA team (excluded from MLB percentile pool)
  _rocTeamSet: null,
  _isROCTeam: function (team) {
    if (!this._rocTeamSet) {
      const rocTeams = (window.METADATA && window.METADATA.rocTeams) ||
                     (DataStore && DataStore.metadata && DataStore.metadata.rocTeams) || [];
      this._rocTeamSet = {};
      for (let i = 0; i < rocTeams.length; i++) this._rocTeamSet[rocTeams[i]] = true;
    }
    return !!this._rocTeamSet[team];
  },

  // Helper: combined multi-team label — "2TM", "3TM", etc.
  _isCombinedTeam: function (team) {
    return typeof team === 'string' && /^\d+TM$/.test(team);
  },

  // Key for grouping a player's per-team + combined (2TM/3TM) rows. Uses mlbId
  // when present so two distinct players sharing a name (e.g. two "Max Muncy")
  // don't collide; falls back to name only when no id exists. Behavior-preserving
  // for normal players since a player's rows all share one mlbId.
  _combinedKey: function (row) {
    return (row.mlbId != null && row.mlbId !== '')
      ? ('id:' + row.mlbId)
      : ('nm:' + (row.pitcher || row.hitter || ''));
  },

  _bisectLeft: function (arr, val) {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] < val) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  },

  _bisectRight: function (arr, val) {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] <= val) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  },

  // Compute percentile for a value against a sorted MLB array using binary search
  _percentileFromSorted: function (sortedMlb, val, isPool) {
    const n = sortedMlb.length;
    const below = this._bisectLeft(sortedMlb, val);
    const aboveOrEqual = this._bisectRight(sortedMlb, val);
    const equal = aboveOrEqual - below;
    let pctl;
    if (isPool) {
      // Pool member: use (below + 0.5*(equal-1)) / (n-1) formula
      pctl = (below + 0.5 * (equal - 1)) / Math.max(1, n - 1) * 100;
    } else {
      // Non-pool (ROC/sub-minimum): use (below + 0.5*equal) / n formula
      pctl = (below + 0.5 * equal) / n * 100;
    }
    return Math.max(0, Math.min(100, Math.round(pctl)));
  },

  // ---- Unified percentile computation (O(n log n) via sort + binary search) ----
  // ROC-aware: excludes ROC teams from the percentile pool, then interpolates ROC players
  // qualifyFn: function(row) returning true if row qualifies for the pool (replaces both minCount and _qualified modes)
  _computePercentiles: function (rows, metricKey, minCount, countKey, useAbs) {
    minCount = minCount || 0;
    countKey = countKey || 'count';
    this._computePercentilesUnified(rows, metricKey, function (row) {
      return minCount === 0 || (row[countKey] || 0) >= minCount;
    }, useAbs, minCount > 0);
  },

  // Percentile computation using _qualified flag for pool membership.
  // The qualified pool defines the distribution. Non-qualified rows are
  // interpolated against that distribution (interpolateSubMinimum=true) so
  // they carry an informational percentile rank for tooltip display while
  // the rendering layer separately suppresses cell coloring for them.
  // Rows with _inPool === false are kept qualified but interpolated against
  // the pool rather than being members of it (used for per-team rows of
  // multi-team players whose 2TM row is the pool representative).
  _computePercentilesQualified: function (rows, metricKey) {
    this._computePercentilesUnified(rows, metricKey, function (row) {
      return !!row._qualified;
    }, false, true);
  },

  // Shared implementation: O(n log n) percentile computation
  _computePercentilesUnified: function (rows, metricKey, qualifyFn, useAbs, interpolateSubMinimum) {
    const pctlKey = metricKey + '_pctl';
    const self = this;

    const mlbValid = [];
    const rocValid = [];  // interpolated (ROC + per-team rows of multi-team players)
    for (let i = 0; i < rows.length; i++) {
      const rawVal = rows[i][metricKey];
      if (rawVal !== null && rawVal !== undefined && rawVal === rawVal && qualifyFn(rows[i])) {
        const entry = { idx: i, val: useAbs ? Math.abs(rawVal) : rawVal };
        if (self._isROCTeam(rows[i].team) || rows[i]._inPool === false) {
          rocValid.push(entry);
        } else {
          mlbValid.push(entry);
        }
      }
    }

    if (mlbValid.length < 2) {
      for (let j = 0; j < rows.length; j++) {
        const rv = rows[j][metricKey];
        rows[j][pctlKey] = (rv !== null && rv !== undefined && qualifyFn(rows[j])) ? 50 : null;
      }
      return;
    }

    // Sort MLB values once: O(n log n) instead of O(n^2)
    const sortedMlb = mlbValid.map(function (v) { return v.val; });
    sortedMlb.sort(function (a, b) { return a - b; });

    var processed = {};

    // Compute percentiles for MLB rows using binary search: O(n log n)
    for (let k = 0; k < mlbValid.length; k++) {
      rows[mlbValid[k].idx][pctlKey] = self._percentileFromSorted(sortedMlb, mlbValid[k].val, true);
      processed[mlbValid[k].idx] = true;
    }

    // Interpolate ROC rows into MLB distribution: O(r log n)
    for (let r = 0; r < rocValid.length; r++) {
      rows[rocValid[r].idx][pctlKey] = self._percentileFromSorted(sortedMlb, rocValid[r].val, false);
      processed[rocValid[r].idx] = true;
    }

    // Interpolate sub-minimum rows: O(s log n)
    if (interpolateSubMinimum) {
      for (let s = 0; s < rows.length; s++) {
        if (processed[s]) continue;
        const sVal = rows[s][metricKey];
        if (sVal === null || sVal === undefined) {
          rows[s][pctlKey] = null;
          processed[s] = true;
          continue;
        }
        const sv = useAbs ? Math.abs(sVal) : sVal;
        rows[s][pctlKey] = self._percentileFromSorted(sortedMlb, sv, false);
        processed[s] = true;
      }
    }

    // Fill nulls for unprocessed rows (overwrite any stale pre-aggregated pctls)
    for (let j2 = 0; j2 < rows.length; j2++) {
      if (!processed[j2]) {
        rows[j2][pctlKey] = null;
      }
    }
  },


  // Build (and cache) pitcher SP/RP role map from pre-aggregated PITCHER_DATA.
  _ensureRoleCache: function () {
    if (this._roleCache) return this._roleCache;
    this._roleCache = {};
    const pd = window.PITCHER_DATA || [];
    for (let i = 0; i < pd.length; i++) {
      const rKey = pd[i].pitcher + '|' + pd[i].team;
      const g = pd[i].g || 0, gs = pd[i].gs || 0;
      this._roleCache[rKey] = g > 0 && (gs / g) > QUAL.SP_GS_RATIO ? 'SP' : 'RP';
    }
    return this._roleCache;
  },

  // Team mode: pitcher-attribute filters (throws / SP-RP role) must be applied
  // to each micro row BEFORE rolling up into the team total, since the team row
  // has no single hand or role of its own.
  _teamModePitcherRowOk: function (filters, pitcherIdx, teamIdx, throws) {
    if (filters.throws !== 'all' && throws !== filters.throws) return false;
    if (filters.role && filters.role !== 'all') {
      const lk = this.data.lookups;
      const roleKey = lk.pitchers[pitcherIdx] + '|' + lk.teams[teamIdx];
      if ((this._ensureRoleCache()[roleKey] || 'RP') !== filters.role) return false;
    }
    return true;
  },

  /**
   * Group pitcherMicro rows by (pitcherIdx, teamIdx), applying date and hand filters.
   * Returns { groups, bipByPitcher } where groups maps "pitcherIdx|teamIdx" to
   * { pitcherIdx, teamIdx, throws, counts[] } and bipByPitcher maps pitcherIdx
   * to arrays of BIP records.
   *
   * Team mode (filters.viewMode === 'team'): groups by teamIdx alone, skipping
   * combined 2TM/3TM micro rows (their counts duplicate the per-team rows) and
   * applying the throws/role filters per micro row before accumulation.
   */
  _groupPitcherMicro: function (filters) {
    const d = this.data;
    const ci = this._colIdx.pitcherCols;
    const micro = d.pitcherMicro;
    const validDates = this._getValidDateSet(filters);
    const vsHand = filters.vsHand || 'all';
    const teamMode = filters.viewMode === 'team';
    const teams = d.lookups.teams;

    // Team mode: pitcherBip rows carry no throws column — map it from micro.
    const throwsByPT = {};
    if (teamMode) {
      for (let ti = 0; ti < micro.length; ti++) {
        throwsByPT[micro[ti][ci.pitcherIdx] + '|' + micro[ti][ci.teamIdx]] = micro[ti][ci.throws];
      }
    }

    const groups = {};
    for (let i = 0; i < micro.length; i++) {
      const row = micro[i];
      if (!validDates[row[ci.dateIdx]]) continue;
      if (vsHand !== 'all' && row[ci.batterHand] !== vsHand) continue;
      if (teamMode) {
        if (this._isCombinedTeam(teams[row[ci.teamIdx]])) continue;
        if (!this._teamModePitcherRowOk(filters, row[ci.pitcherIdx], row[ci.teamIdx], row[ci.throws])) continue;
      }

      const gk = teamMode ? String(row[ci.teamIdx]) : (row[ci.pitcherIdx] + '|' + row[ci.teamIdx]);
      if (!groups[gk]) {
        groups[gk] = {
          pitcherIdx: teamMode ? null : row[ci.pitcherIdx],
          teamIdx: row[ci.teamIdx],
          throws: teamMode ? null : row[ci.throws],
          counts: new Array(37)
        };
        for (let z = 0; z < 37; z++) groups[gk].counts[z] = 0;
      }
      const c = groups[gk].counts;
      // 31 base counts + 6 grade-atom sums/counts (fields 31-36; absent in
      // embeds built before 2026-07-18, so guard the row length)
      const nf = Math.min(37, row.length - ci.n);
      for (let f = 0; f < nf; f++) {
        c[f] += row[ci.n + f];
      }
    }

    const pbci = this._colIdx.pitcherBipCols;
    const pitcherBipData = d.pitcherBip || [];
    const bipByPitcher = {};
    for (let bi = 0; bi < pitcherBipData.length; bi++) {
      const brow = pitcherBipData[bi];
      if (!validDates[brow[pbci.dateIdx]]) continue;
      if (vsHand !== 'all' && brow[pbci.batterHand] !== vsHand) continue;
      let pKey;
      if (teamMode) {
        if (this._isCombinedTeam(teams[brow[pbci.teamIdx]])) continue;
        const ptKey = brow[pbci.pitcherIdx] + '|' + brow[pbci.teamIdx];
        if (!this._teamModePitcherRowOk(filters, brow[pbci.pitcherIdx], brow[pbci.teamIdx], throwsByPT[ptKey])) continue;
        pKey = String(brow[pbci.teamIdx]);
      } else {
        pKey = brow[pbci.pitcherIdx] + '|' + brow[pbci.teamIdx];
      }
      if (!bipByPitcher[pKey]) bipByPitcher[pKey] = [];
      bipByPitcher[pKey].push(brow);
    }

    return { groups: groups, bipByPitcher: bipByPitcher };
  },

  /**
   * Compute batted ball stats (avgEV, maxEV, hardHit%, barrel%, LD/FB/PU%)
   * from the filtered BIP records for a single pitcher.
   * Returns an object with: avgEVAgainst, maxEVAgainst, hardHitPct, barrelPctAgainst,
   *   ldPct, fbPct, puPct.
   */
  _computePitcherBipStats: function (bipRecs, sacqMaps) {
    const pbci = this._colIdx.pitcherBipCols;

    function isBarrel(ev, la) {
      if (ev == null || la == null) return false;
      return la >= 8 && la <= 50 && ev >= 98 && ev * 1.5 - la >= 117 && ev + la >= 124;
    }

    // Prefer the official barrel flag (launch_speed_angle==6) shipped in the micro,
    // matching the server/Sheets; fall back to the EV/LA heuristic only for older
    // data lacking the column.
    const hasBarrelCol = pbci.barrel != null;
    const evs = []; let n_hard = 0, n_barrel = 0;
    let n_ld = 0, n_fb_bb = 0, n_pu_bb = 0; const n_bip_total = bipRecs.length;
    for (let bri = 0; bri < bipRecs.length; bri++) {
      const bev = bipRecs[bri][pbci.exitVelo];
      const bla = bipRecs[bri][pbci.launchAngle];
      const bbt = bipRecs[bri][pbci.bbType]; // 0=gb, 1=ld, 2=fb, 3=pu
      if (bev !== null) { evs.push(bev); if (bev >= QUAL.HARD_HIT_MPH) n_hard++; }
      if (hasBarrelCol ? bipRecs[bri][pbci.barrel] === 1 : isBarrel(bev, bla)) n_barrel++;
      if (bbt === 1) n_ld++;
      if (bbt === 2) n_fb_bb++;
      if (bbt === 3) n_pu_bb++;
    }

    // xwOBAsp: average league zone wOBA for BIP against this pitcher (hand-specific with pooled fallback)
    var xwOBAsp = null;
    if (bipRecs.length > 0) {
      var xsp_sum = 0, xsp_count = 0;
      for (var sri = 0; sri < bipRecs.length; sri++) {
        var sla = bipRecs[sri][pbci.launchAngle];
        var shcX = bipRecs[sri][pbci.hcX];
        var shcY = bipRecs[sri][pbci.hcY];
        var sBats = bipRecs[sri][pbci.bats];
        if (sla == null || shcX == null || shcY == null || !sBats) continue;
        var sAngle = Aggregator.computeSprayAngle(shcX, shcY);
        var sDir = Aggregator.sprayDirection(sAngle, sBats);
        if (!sDir) continue;
        var sLaBin = Aggregator.getLABinIdx(sla);
        if (sLaBin == null) continue;
        var zWoba = Aggregator.sacqLookup(sacqMaps, sDir, sLaBin, sBats);
        if (zWoba != null) {
          xsp_sum += zWoba;
          xsp_count++;
        }
      }
      xwOBAsp = xsp_count > 0 ? xsp_sum / xsp_count : null;
    }

    return {
      avgEVAgainst: evs.length > 0 ? Math.round(evs.reduce(function(a,b){return a+b;},0) / evs.length * 10) / 10 : null,
      maxEVAgainst: evs.length > 0 ? Math.round(Math.max.apply(null, evs) * 10) / 10 : null,
      hardHitPct: evs.length > 0 ? n_hard / evs.length : null,
      barrelPctAgainst: evs.length > 0 ? n_barrel / evs.length : null,
      ldPct: n_bip_total > 0 ? n_ld / n_bip_total : null,
      fbPct: n_bip_total > 0 ? n_fb_bb / n_bip_total : null,
      puPct: n_bip_total > 0 ? n_pu_bb / n_bip_total : null,
      xwOBAsp: xwOBAsp
    };
  },

  /**
   * Convert a single group entry (counts + BIP stats) into a final row object
   * with rate stats (kPct, bbPct, babip, etc.).
   */
  _buildPitcherRow: function (g, lookups, mlbIdMap, bipStats) {
    const c = g.counts;
    const n = c[0], iz = c[1], sw = c[2], wh = c[3], csw = c[4];
    const ooz = c[5], oozSw = c[6], bip = c[7], gb = c[8];
    const pa = c[9], h = c[10], hr = c[11], k = c[12], bb = c[13];
    const hbp = c[14], sf = c[15], sh = c[16], ci_val = c[17];
    const izSw = c[18], izWh = c[19];
    const firstPitches = c[20], firstPitchStrikes = c[21], fb_cnt = c[22], nHrBip = c[23], ldHr = c[24], pu_cnt = c[25], nStrikes = c[26], ibb = c[27];
    const oneOneTotal = c[28], oneOneWins = c[29], earlyActionPAs = c[30];
    const ab = pa - bb - hbp - sf - sh - ci_val;

    const strikePct = n > 0 ? nStrikes / n : null;
    const kPct = pa > 0 ? k / pa : null;
    // uBB/PA: exclude intentional walks (c[27]) to match the server (n_bb_all - n_ibb).
    const bbPct = pa > 0 ? (bb - ibb) / pa : null;
    const kbbPct = (kPct !== null && bbPct !== null) ? Math.round((kPct - bbPct) * 10000) / 10000 : null;
    const babip_denom = ab - k - hr + sf;
    const babip = babip_denom > 0 ? Math.round((h - hr) / babip_denom * 1000) / 1000 : null;
    // AVG/OBP against. Micro carries H but not 2B/3B, so there is no SLG
    // against — the platoon views lean on xSLG for the slugging axis.
    // OBP keeps IBB in the numerator (unlike bbPct, which strips it).
    const avgAgainst = ab > 0 ? h / ab : null;
    const obp_denom = ab + bb + hbp + sf;
    const obpAgainst = obp_denom > 0 ? (h + bb + hbp) / obp_denom : null;
    const fpsPct = firstPitches > 0 ? firstPitchStrikes / firstPitches : null;
    const oneOneWinPct = oneOneTotal > 0 ? oneOneWins / oneOneTotal : null;
    const earlyActionPct = pa > 0 ? earlyActionPAs / pa : null;
    // Strict-conditional HR/FB (2026-08-14): fly-ball HRs / outfield flies.
    // FB-HRs = all BIP HRs minus line-drive HRs (GB inside-the-park HRs are
    // a couple per league-season — negligible). Mirrors pipeline_compute.
    const hrFbPct = fb_cnt > 0 ? Math.max(nHrBip - ldHr, 0) / fb_cnt : null;

    // Grade atoms (fields 31-36): filtered overall Stuff+/Loc+/Pitching+ as
    // plain averages of the per-pitch integers (same digits as the Sheets
    // grade columns / window cards). Zero counts on pre-atom embeds -> null.
    const sumStuff = c[31] || 0, nStuff = c[32] || 0;
    const sumLoc = c[33] || 0, nLoc = c[34] || 0;
    const sumPitching = c[35] || 0, nPitching = c[36] || 0;

    const pitcherName = g.pitcherIdx != null ? lookups.pitchers[g.pitcherIdx] : null;
    const teamName = lookups.teams[g.teamIdx];
    return {
      pitcher: pitcherName,
      team: teamName,
      mlbId: pitcherName != null ? (mlbIdMap[pitcherName + '|' + teamName] || null) : null,
      throws: g.throws,
      count: n,
      pa: pa,
      nSwings: sw,
      nBip: bip,
      strikePct: strikePct,
      izPct: n > 0 ? iz / n : null,
      swStrRate: n > 0 ? wh / n : null,
      swStrPct: sw > 0 ? wh / sw : null,
      cswPct: n > 0 ? csw / n : null,
      izWhiffPct: izSw > 0 ? izWh / izSw : null,
      chasePct: ooz > 0 ? oozSw / ooz : null,
      gbPct: bip > 0 ? gb / bip : null,
      kPct: kPct,
      bbPct: bbPct,
      kbbPct: kbbPct,
      babip: babip,
      avgAgainst: avgAgainst,
      obpAgainst: obpAgainst,
      fpsPct: fpsPct,
      oneOneWinPct: oneOneWinPct,
      earlyActionPct: earlyActionPct,
      hrFbPct: hrFbPct,
      avgEVAgainst: bipStats.avgEVAgainst,
      maxEVAgainst: bipStats.maxEVAgainst,
      hardHitPct: bipStats.hardHitPct,
      barrelPctAgainst: bipStats.barrelPctAgainst,
      ldPct: bipStats.ldPct,
      fbPct: bipStats.fbPct,
      puPct: bipStats.puPct,
      xwOBAsp: bipStats.xwOBAsp,
      stuffScore: nStuff > 0 ? Number((sumStuff / nStuff).toFixed(1)) : null,
      locPlus: nLoc > 0 ? Number((sumLoc / nLoc).toFixed(1)) : null,
      locPlusN: nLoc > 0 ? nLoc : null,
      pitchingScore: nPitching > 0 ? Number((sumPitching / nPitching).toFixed(1)) : null,
    };
  },

  _aggregatePitcher: function (filters) {
    const d = this.data;
    const lookups = d.lookups;
    const vsHand = filters.vsHand || 'all';
    const teamMode = filters.viewMode === 'team';

    const grouped = this._groupPitcherMicro(filters);
    const groups = grouped.groups;
    const bipByPitcher = grouped.bipByPitcher;

    const mlbIdMap = this._getMlbIdMap('pitcher');

    let STAT_KEYS = ['strikePct', 'izPct', 'cswPct', 'izWhiffPct', 'swStrPct', 'chasePct', 'gbPct', 'puPct', 'kPct', 'bbPct', 'kbbPct', 'babip', 'fpsPct', 'oneOneWinPct', 'earlyActionPct', 'hrFbPct',
                     'avgAgainst', 'obpAgainst',
                     'avgEVAgainst', 'maxEVAgainst', 'hardHitPct', 'barrelPctAgainst', 'xwOBAsp',
                     'stuffScore', 'locPlus', 'pitchingScore'];
    let INVERT = { bbPct: true, babip: true, hrFbPct: true, avgAgainst: true, obpAgainst: true, avgEVAgainst: true, maxEVAgainst: true, hardHitPct: true, barrelPctAgainst: true, xwOBAsp: true };
    if (teamMode) {
      // Boxscore-merged stats have no pre-aggregated _pctl at team level —
      // rank them across the team pool here.
      STAT_KEYS = STAT_KEYS.concat(['era', 'fip', 'xFIP', 'siera', 'hr9',
        'wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon', 'twoStrikeWhiffPct',
        'runValue', 'rv100', 'xRunValue', 'xRv100', 'extension']);
      INVERT = Object.assign({}, INVERT, { era: true, fip: true, xFIP: true, siera: true, hr9: true,
        wOBA: true, xBA: true, xSLG: true, xwOBA: true, xwOBAcon: true });
    }
    let rows = [];
    const sacqMaps = this.buildSacqZoneMaps();

    for (let gk2 in groups) {
      const g = groups[gk2];
      const bipKey = teamMode ? String(g.teamIdx) : (g.pitcherIdx + '|' + g.teamIdx);
      const bipRecs = bipByPitcher[bipKey] || [];
      const bipStats = this._computePitcherBipStats(bipRecs, sacqMaps);
      const obj = this._buildPitcherRow(g, lookups, mlbIdMap, bipStats);

      if (teamMode) {
        // Throws/role applied per micro row pre-rollup; player-level minimum
        // sample filters are intentionally ignored for team totals.
        obj._isTeamRow = true;
        rows.push(obj);
        continue;
      }

      // Apply baseball-context filters (comparison group — affects percentiles)
      if (filters.throws !== 'all' && obj.throws !== filters.throws) continue;
      if (obj.count < (filters.minCount || 1)) continue;
      if (filters.minTbf && (obj.pa || 0) < filters.minTbf) continue;
      if (filters.minBip && (obj.nBip || 0) < filters.minBip) continue;
      if (filters.minPitcherSwings && (obj.nSwings || 0) < filters.minPitcherSwings) continue;

      rows.push(obj);
    }

    if (teamMode) {
      return this._finishPitcherTeamRows(rows, filters, STAT_KEYS, INVERT);
    }

    // Merge boxscore stats (G, GS, IP, W, L, SV, HLD, TBF, ERA, HR/9, runValue)
    // from pre-aggregated PITCHER_DATA — these aren't in micro-data
    const boxFields = ['g', 'gs', 'ip', 'w', 'l', 'sv', 'hld', 'tbf', 'era', 'hr9', 'runValue', 'rv100', 'xRunValue', 'xRv100',
                     'era_pctl', 'hr9_pctl', 'runValue_pctl', 'rv100_pctl', 'xRunValue_pctl', 'xRv100_pctl', 'fip', 'fip_pctl', 'xFIP', 'xFIP_pctl', 'siera', 'siera_pctl',
                     'wOBA', 'wOBA_pctl', 'xBA', 'xBA_pctl', 'xSLG', 'xSLG_pctl', 'xwOBA', 'xwOBA_pctl',
                     'xwOBAcon', 'xwOBAcon_pctl',
                     // xwOBAsp deliberately NOT merged: _computePitcherBipStats
                     // recomputes it from filtered BIP records (like avgEVAgainst),
                     // and merging the season value here clobbered that recompute,
                     // making the stat silently ignore date/hand filters.
                     'twoStrikeWhiffPct', 'twoStrikeWhiffPct_pctl',
                     // Stuff+/Loc+/Pitching+ VALUES come from micro grade
                     // atoms (exact filtered plain averages, computed in
                     // _buildPitcherRow) — only season-level metadata is
                     // merged here; a conditional fallback below covers
                     // pre-atom embeds where the atoms are absent.
                     'locPlusRaw', 'stuffScore_lowSupport', 'pitchingRuns100',
                     'xrvoe100', 'xrvoe100_pctl', 'rvoe100', 'rvoe100_pctl',
                     'rvoe', 'rvoe_pctl', 'xrvoe', 'xrvoe_pctl',
                     // Pitcher+ is season-level like SIERA/FIP/xRV100: its
                     // heaviest component (xRv100, weight .23) is itself
                     // boxscore-merged and doesn't refilter, so recomputing
                     // the composite from half-filtered parts would be less
                     // honest than carrying the season value through.
                     'pitcherPlus', 'pitcherPlus_pctl', 'pitcherRuns100',
                     // hdERA/hpERA are season-level by definition (shrunk
                     // inputs, 30+ IP z-pools): filtered views carry the
                     // season value, same contract as Pitcher+.
                     'hdERA', 'hdERA_pctl', 'hpERA', 'hpERA_pctl',
                     'hdERAPlus', 'hdERAPlus_pctl',
                     'hpERAPlus', 'hpERAPlus_pctl',
                     // Command+ is season-level like Pitcher+: targets are
                     // fit on the full season, so filtered views preserve
                     // rather than recompute it.
                     'commandPlus', 'commandPlus_pctl', 'commandPlusRaw', 'commandPlusN',
                     'pitcherPlusProj', 'pitcherPlusProj_pctl',
                     // Arm angle / extension are season-level physical traits
                     // (all-pitch averages from PITCHER_DATA); like Command+
                     // they carry through filtered views rather than recompute.
                     'armAngle', 'extension', 'extension_pctl'];
    const preAgg = window.PITCHER_DATA || [];
    const preAggMap = {};
    for (let bi = 0; bi < preAgg.length; bi++) {
      preAggMap[preAgg[bi].pitcher + '|' + preAgg[bi].team] = preAgg[bi];
    }
    // Fields that have per-hand splits (stored as field_vsL / field_vsR in PITCHER_DATA)
    const handSplitFields = ['twoStrikeWhiffPct', 'fpsPct',
      'strikePct', 'izPct', 'swStrPct', 'cswPct', 'izWhiffPct', 'chasePct',
      'kPct', 'bbPct', 'kbbPct', 'babip', 'gbPct',
      'avgEVAgainst', 'maxEVAgainst', 'hardHitPct', 'barrelPctAgainst',
      'gbPct_bb', 'ldPct', 'fbPct', 'puPct', 'hrFbPct',
      'wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon'];
    const handSuffix = vsHand !== 'all' ? '_vs' + vsHand : '';
    for (let mi = 0; mi < rows.length; mi++) {
      const key2 = rows[mi].pitcher + '|' + rows[mi].team;
      const pre = preAggMap[key2];
      if (pre) {
        for (let fi = 0; fi < boxFields.length; fi++) {
          const bf = boxFields[fi];
          // If filtering by hand and this field has per-hand split, use it
          if (handSuffix && handSplitFields.indexOf(bf) >= 0) {
            const handKey = bf + handSuffix;
            if (pre[handKey] !== undefined) {
              rows[mi][bf] = pre[handKey];
              continue;
            }
          }
          if (pre[bf] !== undefined) rows[mi][bf] = pre[bf];
        }
        // Grade fallback for pre-atom embeds: no atoms in micro data means
        // _buildPitcherRow left these null — surface the season values.
        if (rows[mi].stuffScore == null && pre.stuffScore !== undefined) rows[mi].stuffScore = pre.stuffScore;
        if (rows[mi].locPlus == null && pre.locPlus !== undefined) {
          rows[mi].locPlus = pre.locPlus;
          if (pre.locPlusN !== undefined) rows[mi].locPlusN = pre.locPlusN;
        }
        if (rows[mi].pitchingScore == null && pre.pitchingScore !== undefined) rows[mi].pitchingScore = pre.pitchingScore;
      }
    }

    // Apply role filter AFTER boxscore merge so G/GS are available
    if (filters.role && filters.role !== 'all') {
      rows = rows.filter(function (r) {
        const pg = r.g || 0, pgs = r.gs || 0;
        const isSP = pg > 0 && (pgs / pg) > QUAL.SP_GS_RATIO;
        if (filters.role === 'SP') return isSP;
        if (filters.role === 'RP') return !isSP;
        return true;
      });
    }
    // Compute percentiles with IP-based qualifying
    // Starter (GS/G > 0.5): 1.0 IP/team game. Reliever: 0.1 IP (⅓ inning)/team game.
    const teamGames = this.getTeamGamesPlayed();

    // Multi-team handling: players with a 2TM/3TM row. Per-team rows inherit the
    // combined row's qualification (cumulative IP across teams) and its team-games
    // denominator (sum of the per-team team games). Percentile pool uses combined
    // rows only — per-team rows of multi-team players interpolate against it.
    const combinedByPitcher = {};
    for (let ci2 = 0; ci2 < rows.length; ci2++) {
      if (Aggregator._isCombinedTeam(rows[ci2].team)) {
        combinedByPitcher[Aggregator._combinedKey(rows[ci2])] = rows[ci2];
      }
    }
    // Multi-team qualifier denominator: max(team games) across the player's MLB
    // teams — approximates tenure span. Summing over-inflates the threshold.
    const cumTeamGames = {};
    for (let cg = 0; cg < rows.length; cg++) {
      const cr = rows[cg];
      const crKey = Aggregator._combinedKey(cr);
      if (combinedByPitcher[crKey] && !Aggregator._isCombinedTeam(cr.team)) {
        const tgv = teamGames[cr.team] || 0;
        if (tgv > (cumTeamGames[crKey] || 0)) cumTeamGames[crKey] = tgv;
      }
    }

    // Mark each row as qualified or not
    for (let qi = 0; qi < rows.length; qi++) {
      const r = rows[qi];
      const rKey = Aggregator._combinedKey(r);
      const mtRow = combinedByPitcher[rKey];
      const tg = mtRow ? (cumTeamGames[rKey] || 0) : (teamGames[r.team] || 0);
      const ipFloat = Utils.parseIP(mtRow ? mtRow.ip : r.ip);
      const isStarter = Utils.isStarter(mtRow ? mtRow.g : r.g, mtRow ? mtRow.gs : r.gs);
      const _isROC = Aggregator._isROCTeam(r.team);
      r._qualified = ipFloat >= tg * Utils.pitcherIpPerGame(isStarter, _isROC);
      // Per-team row of a multi-team player: interpolate against the combined-row pool
      r._inPool = mtRow && r !== mtRow ? false : true;
    }
    // Set bipQual flag BEFORE percentiles so BIP stats can use it
    for (let bqi = 0; bqi < rows.length; bqi++) {
      rows[bqi].bipQual = (rows[bqi].nBip || 0) >= QUAL.MIN_BIP_PCTL;
    }
    // Pool: ALL MLB pitchers (no qualifier). Matches the pipeline's
    // post-2026-05-29 design where displayed league avg and percentile pool
    // are the same population, so "below league avg → above 50th pctl"
    // reads correctly. Qualification (and bipQual for BIP-dependent stats)
    // is a render-only gate — non-qualified rows still get a percentile
    // rank stored for tooltip + sort, but the leaderboard suppresses cell
    // coloring on them.
    for (let si = 0; si < STAT_KEYS.length; si++) {
      this._computePercentiles(rows, STAT_KEYS[si]);
    }
    // Invert where lower is better
    for (let ri = 0; ri < rows.length; ri++) {
      for (let inv in INVERT) {
        const pk = inv + '_pctl';
        if (rows[ri][pk] !== null && rows[ri][pk] !== undefined) {
          rows[ri][pk] = 100 - rows[ri][pk];
        }
      }
    }

    // Note: previously this block nulled all _pctl values on unqualified
    // pitchers so their cells would not display percentile colors. That role
    // now belongs to the render layer (leaderboard.js showColor check) — the
    // pre-aggregated _pctl values are computed against the qualified-pitcher
    // pool in process_data.py, so unqualified pitchers carry an informational
    // rank that the tooltip surfaces while the cell stays uncolored. Keeping
    // the _pctl values intact mirrors the hitter-side behavior.

    // Apply min IP filter AFTER percentiles (don't change comparison group)
    if (filters.minIp) {
      if (filters.minIp === 'Q') {
        rows = rows.filter(function (r) { return r._qualified; });
      } else {
        rows = rows.filter(function (r) { return Utils.parseIP(r.ip) >= filters.minIp; });
      }
    }

    // Apply view-narrowing filters AFTER percentiles (don't change comparison group).
    // "All Teams": hide ROC; hide per-team rows of multi-team players (their 2TM row stands in).
    // Specific team: hide the combined 2TM/3TM rows and show only the matching per-team row.
    const self = this;
    if (filters.team !== 'all') {
      rows = rows.filter(function (r) { return r.team === filters.team; });
    } else {
      rows = rows.filter(function (r) {
        // includeROC: player-page platoon aggregations only — ROC pitcher
        // pages split by hand (2026-08-06). Percentiles already interpolated
        // vs the MLB pool above; leaderboards never set the flag.
        if (self._isROCTeam(r.team)) return !!filters.includeROC;
        // keepStints: player-page platoon aggregations keep per-team rows of
        // multi-team players so stint-view pages can look up their split row
        // by (name, team). Their percentiles interpolate vs the combined-row
        // pool above (_inPool === false), so the comparison group is unchanged.
        if (combinedByPitcher[Aggregator._combinedKey(r)] && !self._isCombinedTeam(r.team)) return !!filters.keepStints;
        return true;
      });
    }
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      rows = rows.filter(function (r) { return r.pitcher.toLowerCase().indexOf(searchLower) !== -1; });
    }

    return rows;
  },

  // ---- Team view helpers ----

  // Shared view-narrowing for team rows: specific team → that row only;
  // All Teams → hide ROC (unless the user is searching, so "ROC" is findable);
  // search matches the team code.
  _narrowTeamRows: function (rows, filters) {
    const self = this;
    if (filters.team !== 'all') {
      return rows.filter(function (r) { return r.team === filters.team; });
    }
    if (filters.search) {
      const q = filters.search.toLowerCase();
      return rows.filter(function (r) { return r.team.toLowerCase().indexOf(q) !== -1; });
    }
    return rows.filter(function (r) { return !self._isROCTeam(r.team); });
  },

  // Mark team rows pool-eligible. ROC team rows are interpolated against the
  // 30-team MLB pool by _computePercentilesUnified (via _isROCTeam) and the
  // render layer suppresses their coloring.
  _flagTeamRows: function (rows) {
    for (let i = 0; i < rows.length; i++) {
      rows[i]._isTeamRow = true;
      rows[i]._qualified = true;
      rows[i]._inPool = rows[i]._inPool !== false;
      rows[i].bipQual = true;
    }
  },

  _formatIPThirds: function (thirds) {
    return Math.floor(thirds / 3) + '.' + (thirds % 3);
  },

  /**
   * Aggregate season boxscore/pre-computed pitcher stats to team level from
   * PITCHER_DATA. True totals where the components are recoverable (ERA, HR/9
   * from ER/HR × IP; RV sums at full precision); weighted means elsewhere
   * (FIP/xFIP/SIERA by IP, expected stats by TBF, Loc+ by its sample size).
   * Respects throws/role filters so team rows match the micro-side rollup.
   */
  _teamPitcherBoxscore: function (filters) {
    const roleCache = this._ensureRoleCache();
    const pd = window.PITCHER_DATA || [];
    const acc = {};
    const IP_W = ['fip', 'xFIP', 'siera'];
    const PA_W = ['wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon'];

    function wadd(a, key, val, w) {
      if (val == null || !(w > 0)) return;
      a.sums[key] = (a.sums[key] || 0) + val * w;
      a.wts[key] = (a.wts[key] || 0) + w;
    }

    for (let i = 0; i < pd.length; i++) {
      const p = pd[i];
      if (this._isCombinedTeam(p.team)) continue;
      if (filters.throws !== 'all' && p.throws !== filters.throws) continue;
      if (filters.role && filters.role !== 'all' &&
          (roleCache[p.pitcher + '|' + p.team] || 'RP') !== filters.role) continue;

      let a = acc[p.team];
      if (!a) {
        a = acc[p.team] = { ipThirds: 0, er: null, hrA: null, w: 0, l: 0, sv: 0, hld: 0, gs: 0,
                            tbf: 0, count: 0, runValue: null, xRunValue: null, sums: {}, wts: {} };
      }
      const ipF = Utils.parseIP(p.ip);
      a.ipThirds += Math.round(ipF * 3);
      if (p.era != null && ipF > 0) a.er = (a.er || 0) + p.era * ipF / 9;
      if (p.hr9 != null && ipF > 0) a.hrA = (a.hrA || 0) + p.hr9 * ipF / 9;
      a.w += p.w || 0; a.l += p.l || 0; a.sv += p.sv || 0; a.hld += p.hld || 0;
      a.gs += p.gs || 0; a.tbf += p.tbf || 0; a.count += p.count || 0;
      // RV sums stay full precision; rounding happens only at display
      if (p.runValue != null) a.runValue = (a.runValue || 0) + p.runValue;
      if (p.xRunValue != null) a.xRunValue = (a.xRunValue || 0) + p.xRunValue;
      for (let wi = 0; wi < IP_W.length; wi++) wadd(a, IP_W[wi], p[IP_W[wi]], ipF);
      for (let pi = 0; pi < PA_W.length; pi++) wadd(a, PA_W[pi], p[PA_W[pi]], p.pa || p.tbf);
      wadd(a, 'twoStrikeWhiffPct', p.twoStrikeWhiffPct, p.nSwings);
      wadd(a, 'locPlus', p.locPlus, p.locPlusN);
      wadd(a, 'armAngle', p.armAngle, p.count);
      wadd(a, 'extension', p.extension, p.count);
    }

    const teamGames = this.getTeamGamesPlayed();
    const out = {};
    for (const team in acc) {
      const a = acc[team];
      const ipF = a.ipThirds / 3;
      const o = {
        ip: this._formatIPThirds(a.ipThirds),
        g: teamGames[team] || null,
        gs: a.gs, w: a.w, l: a.l, sv: a.sv, hld: a.hld, tbf: a.tbf,
        era: (a.er != null && ipF > 0) ? a.er * 9 / ipF : null,
        hr9: (a.hrA != null && ipF > 0) ? a.hrA * 9 / ipF : null,
        runValue: a.runValue,
        xRunValue: a.xRunValue,
        rv100: (a.runValue != null && a.count > 0) ? a.runValue / a.count * 100 : null,
        xRv100: (a.xRunValue != null && a.count > 0) ? a.xRunValue / a.count * 100 : null,
        locPlusN: a.wts.locPlus || 0,
      };
      const W_KEYS = ['fip', 'xFIP', 'siera', 'wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon', 'twoStrikeWhiffPct', 'locPlus', 'armAngle', 'extension'];
      for (let ki = 0; ki < W_KEYS.length; ki++) {
        const k = W_KEYS[ki];
        o[k] = (a.wts[k] > 0) ? a.sums[k] / a.wts[k] : null;
      }
      out[team] = o;
    }
    return out;
  },

  // Team mode tail of _aggregatePitcher: merge team boxscore stats, compute
  // percentiles over the team pool, invert lower-is-better, narrow the view.
  _finishPitcherTeamRows: function (rows, filters, STAT_KEYS, INVERT) {
    const box = this._teamPitcherBoxscore(filters);
    for (let i = 0; i < rows.length; i++) {
      const b = box[rows[i].team];
      if (b) {
        for (const k in b) rows[i][k] = b[k];
      }
    }
    this._flagTeamRows(rows);
    for (let si = 0; si < STAT_KEYS.length; si++) {
      this._computePercentiles(rows, STAT_KEYS[si]);
    }
    for (let ri = 0; ri < rows.length; ri++) {
      for (const inv in INVERT) {
        const pk = inv + '_pctl';
        if (rows[ri][pk] !== null && rows[ri][pk] !== undefined) {
          rows[ri][pk] = 100 - rows[ri][pk];
        }
      }
    }
    return this._narrowTeamRows(rows, filters);
  },

  _PITCH_METRIC_MAP: [
    { key: 'velocity', sum: 'sumVelo', cnt: 'nVelo', round: 1 },
    { key: 'spinRate', sum: 'sumSpin', cnt: 'nSpin', round: 0 },
    { key: 'indVertBrk', sum: 'sumIVB', cnt: 'nIVB', round: 1 },
    { key: 'horzBrk', sum: 'sumHB', cnt: 'nHB', round: 1 },
    { key: 'relPosZ', sum: 'sumRelZ', cnt: 'nRelZ', round: 1 },
    { key: 'relPosX', sum: 'sumRelX', cnt: 'nRelX', round: 1 },
    { key: 'extension', sum: 'sumExt', cnt: 'nExt', round: 1 },
    { key: 'armAngle', sum: 'sumArmAngle', cnt: 'nArmAngle', round: 1 },
    { key: 'vaa', sum: 'sumVAA', cnt: 'nVAA', round: 2 },
    { key: 'haa', sum: 'sumHAA', cnt: 'nHAA', round: 2 },
    { key: '_plateZ', sum: 'sumPlateZ', cnt: 'nPlateZ', round: 2 },
    { key: '_plateX', sum: 'sumPlateX', cnt: 'nPlateX', round: 2 },
  ],

  _newPitchOvr: function () {
    return { nVaaS: 0, nVaaW: 0, nHaaS: 0, nHaaW: 0, xIvbS: 0, xIvbW: 0,
             ivbOeS: 0, ivbOeW: 0, xHbS: 0, xHbW: 0, hbOeS: 0, hbOeW: 0 };
  },

  /**
   * Model-derived values for a per-(pitcher, pitchType) micro group: nVAA/nHAA
   * (location-normalized approach angles via per-type regressions) and
   * xIVB/xHB + IVBOE/HBOE (MVN conditional movement model, hand-specific).
   * All values signed from the pitcher's POV; callers decide whether to
   * magnitude/align them when mixing hands.
   */
  _pitchGroupModel: function (g, lookups) {
    const ms = g.metricSums;
    const mean = function (s, c) { return (ms[c] || 0) > 0 ? ms[s] / ms[c] : null; };
    const ptName = lookups.pitchTypes[g.pitchTypeIdx];
    const out = { w: g.counts[0] || 0, nVAA: null, nHAA: null, xIVB: null, xHB: null, ivbOE: null, hbOE: null };

    const vaaRegs = DataStore.metadata && DataStore.metadata.vaaRegressions;
    const vaaReg = vaaRegs && vaaRegs[ptName];
    const vaaMean = mean('sumVAA', 'nVAA');
    const plateZMean = mean('sumPlateZ', 'nPlateZ');
    if (vaaMean != null && plateZMean != null && vaaReg && vaaReg.leagueAvgPlateZ != null) {
      out.nVAA = vaaMean - vaaReg.slope * (plateZMean - vaaReg.leagueAvgPlateZ);
    }

    const haaRegs = DataStore.metadata && DataStore.metadata.haaRegressions;
    const haaReg = haaRegs && haaRegs[ptName];
    const haaMean = mean('sumHAA', 'nHAA');
    const plateXMean = mean('sumPlateX', 'nPlateX');
    // leagueAvgPlateX is hand-specific ({R:…, L:…}); slope is fit mirrored +
    // within-pitcher server-side. Hand-mixed groups (team mode, throws=null)
    // get null, same as the hand-specific MVN model below.
    const haaLgPX = (haaReg && haaReg.leagueAvgPlateX) ? haaReg.leagueAvgPlateX[g.throws] : null;
    if (haaMean != null && plateXMean != null && haaReg && haaLgPX != null) {
      out.nHAA = haaMean - haaReg.slope * (plateXMean - haaLgPX);
    }

    const mvnModels = DataStore.metadata && DataStore.metadata.mvnModels;
    const ptModel = mvnModels && mvnModels[ptName + '_' + g.throws];
    const armMean = mean('sumArmAngle', 'nArmAngle');
    const extMean = mean('sumExt', 'nExt');
    const veloMean = mean('sumVelo', 'nVelo');
    const relZMean = mean('sumRelZ', 'nRelZ');
    const relXMean = mean('sumRelX', 'nRelX');
    const ivbMean = mean('sumIVB', 'nIVB');
    const hbMean = mean('sumHB', 'nHB');
    let xIVB_i = null, xHB_i = null;
    if (ptModel && ptModel.mlb && armMean != null && extMean != null && veloMean != null) {
      const muBar = this._mvnConditional(ptModel.mlb, [armMean, extMean, veloMean]);
      if (muBar) { xIVB_i = muBar[0]; xHB_i = muBar[1]; }
    }
    if (xIVB_i === null && ptModel && ptModel.roc && relZMean != null && relXMean != null && extMean != null && veloMean != null) {
      const muBar2 = this._mvnConditional(ptModel.roc, [relZMean, relXMean, extMean, veloMean]);
      if (muBar2) { xIVB_i = muBar2[0]; xHB_i = muBar2[1]; }
    }
    if (xIVB_i != null) {
      out.xIVB = xIVB_i;
      if (ivbMean != null) out.ivbOE = ivbMean - xIVB_i;
    }
    if (xHB_i != null) {
      out.xHB = xHB_i;
      if (hbMean != null) out.hbOE = hbMean - xHB_i;
    }
    return out;
  },

  /**
   * Team mode: roll per-(pitcher, team, pitchType) groups up to (team, pitchType).
   * Counts and most metric sums add directly. Hand-mirrored metrics (HB, RelX,
   * HAA) accumulate |per-pitcher mean| × n so LHP/RHP signs don't cancel — the
   * team value reads as an average magnitude. Model-derived values (xIVB/xHB,
   * IVBOE/HBOE, nVAA/nHAA) are computed per pitcher (they depend on the
   * pitcher's hand and release) and combined as pitch-count-weighted means;
   * HBOE is sign-aligned to each pitch's natural break direction first.
   */
  _rollupPitchTeamGroups: function (groups, lookups) {
    const MM = this._PITCH_METRIC_MAP;
    const MIRRORED = { sumHB: true, sumRelX: true, sumHAA: true };
    const teamGroups = {};

    for (const gk in groups) {
      const g = groups[gk];
      const ms = g.metricSums;
      const tk = g.teamIdx + '|' + g.pitchTypeIdx;
      let tg = teamGroups[tk];
      if (!tg) {
        tg = teamGroups[tk] = {
          pitcherIdx: null, teamIdx: g.teamIdx, throws: null, pitchTypeIdx: g.pitchTypeIdx,
          counts: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          metricSums: {},
          _ovr: this._newPitchOvr()
        };
        for (const mk in ms) tg.metricSums[mk] = 0;
      }
      for (let f = 0; f < 22; f++) tg.counts[f] += g.counts[f];
      for (let mi = 0; mi < MM.length; mi++) {
        const m = MM[mi];
        const cnt = ms[m.cnt] || 0;
        if (!cnt) continue;
        tg.metricSums[m.sum] += MIRRORED[m.sum] ? Math.abs(ms[m.sum] / cnt) * cnt : ms[m.sum];
        tg.metricSums[m.cnt] += cnt;
      }
      tg.metricSums.sumTiltSin += ms.sumTiltSin || 0;
      tg.metricSums.sumTiltCos += ms.sumTiltCos || 0;
      tg.metricSums.nTilt += ms.nTilt || 0;
      // Grade atoms: additive integer sums — team value stays a plain
      // average over every pitch the team threw (weighted true total).
      tg.metricSums.sumStuff += ms.sumStuff || 0;
      tg.metricSums.nStuff += ms.nStuff || 0;
      tg.metricSums.sumLoc += ms.sumLoc || 0;
      tg.metricSums.nLoc += ms.nLoc || 0;
      tg.metricSums.sumPitching += ms.sumPitching || 0;
      tg.metricSums.nPitching += ms.nPitching || 0;

      // Per-pitcher model values, weighted into the team accumulator.
      // Magnitude for hand-mirrored values (nHAA, xHB); HBOE aligned to the
      // pitch's natural break direction; nVAA/xIVB/IVBOE are not mirrored.
      const mv = this._pitchGroupModel(g, lookups);
      const w = mv.w;
      if (!(w > 0)) continue;
      const o = tg._ovr;
      if (mv.nVAA != null) { o.nVaaS += mv.nVAA * w; o.nVaaW += w; }
      if (mv.nHAA != null) { o.nHaaS += Math.abs(mv.nHAA) * w; o.nHaaW += w; }
      if (mv.xIVB != null) { o.xIvbS += mv.xIVB * w; o.xIvbW += w; }
      if (mv.ivbOE != null) { o.ivbOeS += mv.ivbOE * w; o.ivbOeW += w; }
      if (mv.xHB != null) { o.xHbS += Math.abs(mv.xHB) * w; o.xHbW += w; }
      if (mv.hbOE != null && mv.xHB != null) {
        const sgn = mv.xHB < 0 ? -1 : 1;
        o.hbOeS += mv.hbOE * sgn * w; o.hbOeW += w;
      }
    }
    return teamGroups;
  },

  /**
   * Add combined category groups (Hard / Breaking / Offspeed) built from the
   * base per-type groups. Player mode combines within a pitcher (one hand, so
   * raw sums are safe); team mode combines already-rolled-up team groups whose
   * metric sums are magnitude-corrected and whose _ovr accumulators are
   * additive. Category rows carry catName instead of a pitchTypeIdx.
   */
  _addCategoryPitchGroups: function (base, lookups, selectedPitchTypes, teamMode) {
    const CATS = this.PITCH_CATEGORIES;
    const cats = selectedPitchTypes.filter(function (s) { return !!CATS[s]; });
    if (cats.length === 0) return base;

    const out = {};
    for (const bk in base) out[bk] = base[bk];

    for (let ci = 0; ci < cats.length; ci++) {
      const cat = cats[ci];
      const typeSet = {};
      for (let ti = 0; ti < CATS[cat].length; ti++) typeSet[CATS[cat][ti]] = true;

      for (const gk in base) {
        const g = base[gk];
        if (g.catName) continue;
        if (!typeSet[lookups.pitchTypes[g.pitchTypeIdx]]) continue;

        const ck = (teamMode ? String(g.teamIdx) : (g.pitcherIdx + '|' + g.teamIdx)) + '|cat|' + cat;
        let cg = out[ck];
        if (!cg) {
          cg = out[ck] = {
            pitcherIdx: g.pitcherIdx, teamIdx: g.teamIdx, throws: g.throws,
            pitchTypeIdx: null, catName: cat,
            counts: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            metricSums: {},
            _ovr: this._newPitchOvr()
          };
          for (const mk in g.metricSums) cg.metricSums[mk] = 0;
        }
        for (let f = 0; f < 22; f++) cg.counts[f] += g.counts[f];
        for (const mk2 in g.metricSums) cg.metricSums[mk2] += g.metricSums[mk2] || 0;

        if (teamMode) {
          // Team base groups carry additive _ovr accumulators
          const so = g._ovr, co = cg._ovr;
          for (const ok in so) co[ok] += so[ok];
        } else {
          // Per-pitcher per-type model values, signed (single hand)
          const mv = this._pitchGroupModel(g, lookups);
          const w = mv.w;
          if (w > 0) {
            const o = cg._ovr;
            if (mv.nVAA != null) { o.nVaaS += mv.nVAA * w; o.nVaaW += w; }
            if (mv.nHAA != null) { o.nHaaS += mv.nHAA * w; o.nHaaW += w; }
            if (mv.xIVB != null) { o.xIvbS += mv.xIVB * w; o.xIvbW += w; }
            if (mv.ivbOE != null) { o.ivbOeS += mv.ivbOE * w; o.ivbOeW += w; }
            if (mv.xHB != null) { o.xHbS += mv.xHB * w; o.xHbW += w; }
            if (mv.hbOE != null) { o.hbOeS += mv.hbOE * w; o.hbOeW += w; }
          }
        }
      }
    }
    return out;
  },

  /**
   * Team mode: aggregate the pre-computed pitch-type stats (run values, expected
   * stats, batted-ball, Loc+, Stuff+) from PITCH_DATA to (team, pitchType) and
   * merge into the team rows. RV totals are true sums kept at full precision;
   * rates are weighted by their natural denominators (PA for expected stats,
   * BIP for batted-ball, pitches for Strike%/Stuff+, sample N for Loc+).
   */
  // Categories (Hard/Breaking/Offspeed) containing the given pitch type that
  // are present in the user's selection.
  _selectedCatsForType: function (selectedPitchTypes, pitchType) {
    const CATS = this.PITCH_CATEGORIES;
    const out = [];
    for (let i = 0; i < selectedPitchTypes.length; i++) {
      const sel = selectedPitchTypes[i];
      if (CATS[sel] && CATS[sel].indexOf(pitchType) !== -1) out.push(sel);
    }
    return out;
  },

  // Accumulate one PITCH_DATA row into a (team|type) or (entity|category)
  // pre-agg bucket. RV sums stay full precision; rates weight by their
  // natural denominators.
  _accumPitchPreAgg: function (acc, k, p, handSfx, PITCH_BB_KEYS) {
    const X_KEYS = ['wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon', 'xwOBAsp'];
    function wadd(a, key, val, w) {
      if (val == null || !(w > 0)) return;
      a.sums[key] = (a.sums[key] || 0) + val * w;
      a.wts[key] = (a.wts[key] || 0) + w;
    }
    let a = acc[k];
    if (!a) a = acc[k] = { runValue: null, xRunValue: null, count: 0, maxVelo: null, sums: {}, wts: {} };
    a.count += p.count || 0;
    if (p.runValue != null) a.runValue = (a.runValue || 0) + p.runValue;
    if (p.xRunValue != null) a.xRunValue = (a.xRunValue || 0) + p.xRunValue;
    if (p.maxVelo != null && (a.maxVelo == null || p.maxVelo > a.maxVelo)) a.maxVelo = p.maxVelo;
    wadd(a, 'strikePct', p.strikePct, p.count);
    wadd(a, 'twoStrikeWhiffPct', p.twoStrikeWhiffPct, p.nSwings);
    wadd(a, 'stuffScore', p.stuffScore, p.count);
    wadd(a, 'locPlus', p.locPlus, p.locPlusN);
    wadd(a, 'pitchingScore', p.pitchingScore, p.count);
    for (let xi = 0; xi < X_KEYS.length; xi++) {
      const xk = X_KEYS[xi];
      const v = (handSfx && p[xk + handSfx] !== undefined) ? p[xk + handSfx] : p[xk];
      wadd(a, xk, v, p.pa);
    }
    for (let bi = 0; bi < PITCH_BB_KEYS.length; bi++) {
      const bk = PITCH_BB_KEYS[bi];
      const bv = (handSfx && p[bk + handSfx] !== undefined) ? p[bk + handSfx] : p[bk];
      wadd(a, bk, bv, p.nBip);
    }
  },

  _assignPitchPreAgg: function (r, a) {
    r.runValue = a.runValue;
    r.xRunValue = a.xRunValue;
    r.rv100 = (a.runValue != null && a.count > 0) ? a.runValue / a.count * 100 : null;
    r.xRv100 = (a.xRunValue != null && a.count > 0) ? a.xRunValue / a.count * 100 : null;
    r.maxVelo = a.maxVelo;
    // Grade metrics: keep the micro-atom plain averages (exact under any
    // filter) — only fall back to season pre-agg when no atoms existed.
    const GRADE_KEYS = { stuffScore: true, locPlus: true, pitchingScore: true };
    for (const sk in a.sums) {
      if (GRADE_KEYS[sk] && r[sk] != null) continue;
      if (a.wts[sk] > 0) r[sk] = a.sums[sk] / a.wts[sk];
    }
  },

  _mergeTeamPitchPreAgg: function (rows, filters, PITCH_BB_KEYS, vsHand) {
    const roleCache = this._ensureRoleCache();
    const pd = window.PITCH_DATA || [];
    const handSfx = (vsHand === 'L') ? '_vsL' : (vsHand === 'R') ? '_vsR' : '';
    const selected = filters.pitchTypes || ['all'];
    const acc = {};

    for (let i = 0; i < pd.length; i++) {
      const p = pd[i];
      if (this._isCombinedTeam(p.team)) continue;
      if (filters.throws !== 'all' && p.throws !== filters.throws) continue;
      if (filters.role && filters.role !== 'all' &&
          (roleCache[p.pitcher + '|' + p.team] || 'RP') !== filters.role) continue;

      this._accumPitchPreAgg(acc, p.team + '|' + p.pitchType, p, handSfx, PITCH_BB_KEYS);
      // Selected category groups aggregate the same rows under the category key
      const cats = this._selectedCatsForType(selected, p.pitchType);
      for (let ci = 0; ci < cats.length; ci++) {
        this._accumPitchPreAgg(acc, p.team + '|' + cats[ci], p, handSfx, PITCH_BB_KEYS);
      }
    }

    for (let ri = 0; ri < rows.length; ri++) {
      const a = acc[rows[ri].team + '|' + rows[ri].pitchType];
      if (a) this._assignPitchPreAgg(rows[ri], a);
    }
  },

  /**
   * Player mode: category rows (Hard/Breaking/Offspeed) have no pre-agg
   * PITCH_DATA row — aggregate the member pitch types per pitcher and merge.
   * Individual-type rows keep their season pre-agg values (and _pctl) from the
   * regular merge.
   */
  _mergePlayerCatPitchPreAgg: function (rows, filters, PITCH_BB_KEYS, vsHand) {
    const CATS = this.PITCH_CATEGORIES;
    const pd = window.PITCH_DATA || [];
    const handSfx = (vsHand === 'L') ? '_vsL' : (vsHand === 'R') ? '_vsR' : '';
    const selected = filters.pitchTypes || ['all'];
    const acc = {};

    for (let i = 0; i < pd.length; i++) {
      const p = pd[i];
      const cats = this._selectedCatsForType(selected, p.pitchType);
      for (let ci = 0; ci < cats.length; ci++) {
        this._accumPitchPreAgg(acc, p.pitcher + '|' + p.team + '|' + cats[ci], p, handSfx, PITCH_BB_KEYS);
      }
    }

    for (let ri = 0; ri < rows.length; ri++) {
      const r = rows[ri];
      if (!CATS[r.pitchType]) continue;
      const a = acc[r.pitcher + '|' + r.team + '|' + r.pitchType];
      if (a) this._assignPitchPreAgg(r, a);
    }
  },

  _aggregatePitch: function (filters) {
    const self = this;
    const d = this.data;
    const ci = this._colIdx.pitchCols;
    const micro = d.pitchMicro;
    const lookups = d.lookups;
    const validDates = this._getValidDateSet(filters);
    const vsHand = filters.vsHand || 'all';
    const mlbIdMap = this._getMlbIdMap('pitcher');
    const teamMode = filters.viewMode === 'team';

    // Build role cache upfront from PITCHER_DATA (SP vs RP based on GS/G ratio)
    this._ensureRoleCache();

    const METRIC_MAP = Aggregator._PITCH_METRIC_MAP;
    const METRIC_KEYS_LIST = METRIC_MAP.map(function (m) { return m.key; }).filter(function (k) { return k !== '_plateZ' && k !== '_plateX'; });
    const NO_PCTL_METRICS = { relPosZ: true, relPosX: true, armAngle: true };
    const METRIC_PCTL_KEYS = METRIC_KEYS_LIST.filter(function (k) { return !NO_PCTL_METRICS[k]; });
    const PITCH_STAT_KEYS = ['izPct', 'swStrPct', 'cswPct', 'izWhiffPct', 'chasePct', 'gbPct', 'fpsPct'];
    const PITCH_BB_KEYS = ['avgEVAgainst', 'maxEVAgainst', 'hardHitPct', 'barrelPctAgainst', 'hrFbPct', 'ldPct', 'fbPct', 'puPct'];
    // ldPct/fbPct inverted to match the server (pipeline_compute.PITCH_BB_INVERT):
    // higher line-drive / fly-ball rate against = worse for the pitcher (red).
    const PITCH_BB_INVERT = { avgEVAgainst: true, maxEVAgainst: true, hardHitPct: true, barrelPctAgainst: true, hrFbPct: true, ldPct: true, fbPct: true };
    const PITCH_EXPECTED_KEYS = ['wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon', 'xwOBAsp'];
    const PITCH_EXPECTED_INVERT = { wOBA: true, xBA: true, xSLG: true, xwOBA: true, xwOBAcon: true, xwOBAsp: true };
    let PITCH_PCTL_KEYS = METRIC_PCTL_KEYS.concat(['nVAA', 'nHAA', 'ivbOE', 'hbOE', 'stuffScore', 'pitchingScore', 'locPlus']).concat(PITCH_STAT_KEYS).concat(PITCH_BB_KEYS).concat(PITCH_EXPECTED_KEYS);
    if (teamMode) {
      // Stats merged from pre-agg data carry no team-level _pctl — rank them here
      PITCH_PCTL_KEYS = PITCH_PCTL_KEYS.concat(['runValue', 'rv100', 'xRunValue', 'xRv100', 'strikePct', 'twoStrikeWhiffPct']);
    }

    const groups = {};
    const pitcherTotals = {};
    const teamTotals = {};

    for (let i = 0; i < micro.length; i++) {
      const row = micro[i];
      if (!validDates[row[ci.dateIdx]]) continue;
      if (vsHand !== 'all' && row[ci.batterHand] !== vsHand) continue;
      if (teamMode) {
        if (this._isCombinedTeam(lookups.teams[row[ci.teamIdx]])) continue;
        if (!this._teamModePitcherRowOk(filters, row[ci.pitcherIdx], row[ci.teamIdx], row[ci.throws])) continue;
        teamTotals[row[ci.teamIdx]] = (teamTotals[row[ci.teamIdx]] || 0) + row[ci.n];
      }

      const pitcherKey = row[ci.pitcherIdx] + '|' + row[ci.teamIdx];
      const gk = pitcherKey + '|' + row[ci.pitchTypeIdx];

      pitcherTotals[pitcherKey] = (pitcherTotals[pitcherKey] || 0) + row[ci.n];

      if (!groups[gk]) {
        groups[gk] = {
          pitcherIdx: row[ci.pitcherIdx],
          teamIdx: row[ci.teamIdx],
          throws: row[ci.throws],
          pitchTypeIdx: row[ci.pitchTypeIdx],
          counts: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          metricSums: {}
        };
        METRIC_MAP.forEach(function (m) {
          groups[gk].metricSums[m.sum] = 0;
          groups[gk].metricSums[m.cnt] = 0;
        });
        groups[gk].metricSums.sumTiltSin = 0;
        groups[gk].metricSums.sumTiltCos = 0;
        groups[gk].metricSums.nTilt = 0;
        // Grade-atom sums: the same integers as the Sheets grade columns,
        // so filtered Stuff+/Loc+/Pitching+ = plain average = AVERAGEIF.
        groups[gk].metricSums.sumStuff = 0;
        groups[gk].metricSums.nStuff = 0;
        groups[gk].metricSums.sumLoc = 0;
        groups[gk].metricSums.nLoc = 0;
        groups[gk].metricSums.sumPitching = 0;
        groups[gk].metricSums.nPitching = 0;
      }

      const g = groups[gk];
      for (let f = 0; f < 22; f++) {
        g.counts[f] += row[ci.n + f];
      }
      METRIC_MAP.forEach(function (m) {
        g.metricSums[m.sum] += row[ci[m.sum]];
        g.metricSums[m.cnt] += row[ci[m.cnt]];
      });
      g.metricSums.sumTiltSin += row[ci.sumTiltSin];
      g.metricSums.sumTiltCos += row[ci.sumTiltCos];
      g.metricSums.nTilt += row[ci.nTilt];
      // Grade atoms (absent in embeds built before 2026-07-18 — guard)
      if (ci.sumStuff !== undefined) {
        g.metricSums.sumStuff += row[ci.sumStuff] || 0;
        g.metricSums.nStuff += row[ci.nStuff] || 0;
        g.metricSums.sumLoc += row[ci.sumLoc] || 0;
        g.metricSums.nLoc += row[ci.nLoc] || 0;
        g.metricSums.sumPitching += row[ci.sumPitching] || 0;
        g.metricSums.nPitching += row[ci.nPitching] || 0;
      }
    }

    let rows = [];
    const selectedPT = filters.pitchTypes || ['all'];
    let buildGroups = teamMode ? this._rollupPitchTeamGroups(groups, lookups) : groups;
    buildGroups = this._addCategoryPitchGroups(buildGroups, lookups, selectedPT, teamMode);
    const hasCats = selectedPT.some(function (s) { return !!Aggregator.PITCH_CATEGORIES[s]; });
    for (let gk2 in buildGroups) {
      const g = buildGroups[gk2];
      const c = g.counts;
      const ms = g.metricSums;
      const n = c[0], iz = c[1], sw = c[2], wh = c[3], csw = c[4];
      const ooz = c[5], oozSw = c[6], bip = c[7], gb = c[8];
      const pa = c[9], h = c[10], hr = c[11], k = c[12], bb = c[13];
      const hbp = c[14], sf = c[15], sh = c[16], ci_val = c[17];
      const izSw = c[18], izWh = c[19];
      const firstPitches = c[20], firstPitchStrikes = c[21];
      const ab = pa - bb - hbp - sf - sh - ci_val;
      const pitcherKey = g.pitcherIdx + '|' + g.teamIdx;
      // Usage% denominator: team mode uses ALL pitches thrown by the team
      const pitcherTotal = teamMode ? (teamTotals[g.teamIdx] || 0) : (pitcherTotals[pitcherKey] || 0);

      const kPct = pa > 0 ? k / pa : null;
      const bbPct = pa > 0 ? bb / pa : null;
      const kbbPct = (kPct !== null && bbPct !== null) ? Math.round((kPct - bbPct) * 10000) / 10000 : null;
      const babip_denom = ab - k - hr + sf;
      const babip_val = babip_denom > 0 ? Math.round((h - hr) / babip_denom * 1000) / 1000 : null;
      const fpsPct_val = firstPitches > 0 ? firstPitchStrikes / firstPitches : null;

      const pitcherName2 = g.pitcherIdx != null ? lookups.pitchers[g.pitcherIdx] : null;
      const teamName2 = lookups.teams[g.teamIdx];
      const obj = {
        pitcher: pitcherName2,
        team: teamName2,
        mlbId: pitcherName2 != null ? (mlbIdMap[pitcherName2 + '|' + teamName2] || null) : null,
        throws: g.throws,
        pitchType: g.catName || lookups.pitchTypes[g.pitchTypeIdx],
        count: n,
        nSwings: sw,
        nBip: bip,
        usagePct: pitcherTotal > 0 ? Math.round(n / pitcherTotal * 10000) / 10000 : null,
        pa: pa,
        izPct: n > 0 ? iz / n : null,
        swStrRate: n > 0 ? wh / n : null,
        swStrPct: sw > 0 ? wh / sw : null,
        cswPct: n > 0 ? csw / n : null,
        izWhiffPct: izSw > 0 ? izWh / izSw : null,
        chasePct: ooz > 0 ? oozSw / ooz : null,
        gbPct: bip > 0 ? gb / bip : null,
        kPct: kPct,
        bbPct: bbPct,
        kbbPct: kbbPct,
        babip: babip_val,
        fpsPct: fpsPct_val,
      };

      METRIC_MAP.forEach(function (m) {
        const cnt = ms[m.cnt];
        if (cnt > 0) {
          obj[m.key] = Number((ms[m.sum] / cnt).toFixed(m.round));
        } else {
          obj[m.key] = null;
        }
      });

      // Normalized VAA (location-independent, per pitch type):
      // nVAA = VAA - slope * (pitcher_avgPlateZ - league_avgPlateZ)
      // Per-pitch-type slopes capture different VAA~PlateZ relationships by pitch type
      const vaaRegs = DataStore.metadata && DataStore.metadata.vaaRegressions;
      const vaaReg = vaaRegs && vaaRegs[obj.pitchType];
      if (obj.vaa !== null && obj._plateZ !== null && vaaReg && vaaReg.leagueAvgPlateZ != null) {
        obj.nVAA = Number((obj.vaa - vaaReg.slope * (obj._plateZ - vaaReg.leagueAvgPlateZ)).toFixed(2));
      } else {
        obj.nVAA = null;
      }
      // Normalized HAA (location-independent, per pitch type):
      // nHAA = HAA - slope * (pitcher_avgPlateX - league_avgPlateX[throws])
      // Slope is fit mirrored + within-pitcher server-side (geometric ~1.05
      // deg/ft for every pitch type); league PlateX mean is hand-specific.
      const haaRegs = DataStore.metadata && DataStore.metadata.haaRegressions;
      const haaReg = haaRegs && haaRegs[obj.pitchType];
      const haaLgPX = (haaReg && haaReg.leagueAvgPlateX) ? haaReg.leagueAvgPlateX[obj.throws] : null;
      if (obj.haa !== null && obj._plateX !== null && haaReg && haaLgPX != null) {
        obj.nHAA = Number((obj.haa - haaReg.slope * (obj._plateX - haaLgPX)).toFixed(2));
      } else {
        obj.nHAA = null;
      }
      delete obj._plateZ;  // internal, not displayed
      delete obj._plateX;  // internal, not displayed

      // xIVB/xHB + IVBOE/HBOE from MVN conditional model (per pitch type + handedness)
      const mvnModels = DataStore.metadata && DataStore.metadata.mvnModels;
      const mvnKey = obj.pitchType + '_' + obj.throws;
      const ptModel = mvnModels && mvnModels[mvnKey];
      let xIVB_val = null, xHB_val = null;
      // Try MLB model first: condition on [ArmAngle, Extension, Velocity]
      if (ptModel && ptModel.mlb && obj.armAngle !== null && obj.extension !== null && obj.velocity !== null) {
        const muBar = self._mvnConditional(ptModel.mlb, [obj.armAngle, obj.extension, obj.velocity]);
        if (muBar) { xIVB_val = muBar[0]; xHB_val = muBar[1]; }
      }
      // Fallback to ROC model if no MLB model or missing ArmAngle: [RelPosZ, RelPosX, Extension, Velocity]
      if (xIVB_val === null && ptModel && ptModel.roc && obj.relPosZ !== null && obj.relPosX !== null && obj.extension !== null && obj.velocity !== null) {
        const muBar2 = self._mvnConditional(ptModel.roc, [obj.relPosZ, obj.relPosX, obj.extension, obj.velocity]);
        if (muBar2) { xIVB_val = muBar2[0]; xHB_val = muBar2[1]; }
      }
      if (xIVB_val !== null) {
        obj.xIVB = Number(xIVB_val.toFixed(1));
        obj.ivbOE = obj.indVertBrk !== null ? Number((obj.indVertBrk - xIVB_val).toFixed(1)) : null;
      } else {
        obj.xIVB = null;
        obj.ivbOE = null;
      }
      if (xHB_val !== null) {
        obj.xHB = Number(xHB_val.toFixed(1));
        obj.hbOE = obj.horzBrk !== null ? Number((obj.horzBrk - xHB_val).toFixed(1)) : null;
      } else {
        obj.xHB = null;
        obj.hbOE = null;
      }

      // Grade atoms: filtered Stuff+/Loc+/Pitching+ as plain averages of the
      // per-pitch integers (identical to the Sheets columns / window cards).
      obj.stuffScore = ms.nStuff > 0 ? Number((ms.sumStuff / ms.nStuff).toFixed(1)) : null;
      obj.locPlus = ms.nLoc > 0 ? Number((ms.sumLoc / ms.nLoc).toFixed(1)) : null;
      obj.locPlusN = ms.nLoc > 0 ? ms.nLoc : null;
      obj.pitchingScore = ms.nPitching > 0 ? Number((ms.sumPitching / ms.nPitching).toFixed(1)) : null;

      // Break Tilt (circular mean)
      if (ms.nTilt > 0) {
        const sinAvg = ms.sumTiltSin / ms.nTilt;
        const cosAvg = ms.sumTiltCos / ms.nTilt;
        let avgAngle = Math.atan2(sinAvg, cosAvg);
        if (avgAngle < 0) avgAngle += 2 * Math.PI;
        const avgMinutes = Math.round(avgAngle / (2 * Math.PI) * 720);
        let thh = Math.floor(avgMinutes / 60);
        const tmm = avgMinutes % 60;
        if (thh === 0) thh = 12;
        obj.breakTilt = thh + ':' + (tmm < 10 ? '0' : '') + tmm;
        obj.breakTiltMinutes = avgMinutes;
      } else {
        obj.breakTilt = null;
        obj.breakTiltMinutes = null;
      }

      if (g._ovr) {
        // Model-derived values for team and category rows: replace the
        // generic group-sum versions with the per-(pitcher, type) weighted
        // means accumulated during rollup/combination (the per-type
        // regressions and hand-specific MVN models don't apply to mixed
        // groups directly).
        const o = g._ovr;
        obj.nVAA = o.nVaaW > 0 ? Number((o.nVaaS / o.nVaaW).toFixed(2)) : null;
        obj.nHAA = o.nHaaW > 0 ? Number((o.nHaaS / o.nHaaW).toFixed(2)) : null;
        obj.xIVB = o.xIvbW > 0 ? Number((o.xIvbS / o.xIvbW).toFixed(1)) : null;
        obj.ivbOE = o.ivbOeW > 0 ? Number((o.ivbOeS / o.ivbOeW).toFixed(1)) : null;
        obj.xHB = o.xHbW > 0 ? Number((o.xHbS / o.xHbW).toFixed(1)) : null;
        obj.hbOE = o.hbOeW > 0 ? Number((o.hbOeS / o.hbOeW).toFixed(1)) : null;
      }

      if (teamMode) {
        obj._isTeamRow = true;
        // Team rows mix LHP and RHP of a pitch type; their signed tilt sin/cos
        // sum toward cancellation, so the circular mean is a meaningless clock
        // value. Suppress it (unlike category rows, which are one pitcher/one hand).
        obj.breakTilt = null;
        obj.breakTiltMinutes = null;
        if (filters.pitchTypes && filters.pitchTypes.indexOf('all') === -1 && filters.pitchTypes.indexOf(obj.pitchType) === -1) continue;
        // Same default floor as player mode ('Qualified' = 25 pitches of the
        // type): a team row built on a handful of pitches is noise.
        if (obj.count < QUAL.MIN_PITCH_PCTL) continue;
        rows.push(obj);
        continue;
      }

      // Apply baseball-context filters (comparison group — affects percentiles)
      if (filters.throws !== 'all' && obj.throws !== filters.throws) continue;
      if (filters.role && filters.role !== 'all') {
        // Pitch rows don't have G/GS — look up from pre-built role cache
        const pitcherKey2 = obj.pitcher + '|' + obj.team;
        if ((this._roleCache[pitcherKey2] || 'RP') !== filters.role) continue;
      }
      if (filters.pitchTypes && filters.pitchTypes.indexOf('all') === -1 && filters.pitchTypes.indexOf(obj.pitchType) === -1) continue;
      if (obj.count < (filters.minCount || 1)) continue;
      if (filters.minPitcherSwings && (obj.nSwings || 0) < filters.minPitcherSwings) continue;
      if (filters.minBip && (obj.nBip || 0) < filters.minBip) continue;

      rows.push(obj);
    }

    if (teamMode) {
      this._mergeTeamPitchPreAgg(rows, filters, PITCH_BB_KEYS, vsHand);
      this._flagTeamRows(rows);
    } else {
    const pitchPreAgg = window.PITCH_DATA || [];
    const pitchPreMap = {};
    for (let ppi = 0; ppi < pitchPreAgg.length; ppi++) {
      const ppk = pitchPreAgg[ppi].pitcher + '|' + pitchPreAgg[ppi].team + '|' + pitchPreAgg[ppi].pitchType;
      pitchPreMap[ppk] = pitchPreAgg[ppi];
    }
    for (let pmi = 0; pmi < rows.length; pmi++) {
      const pmk = rows[pmi].pitcher + '|' + rows[pmi].team + '|' + rows[pmi].pitchType;
      const ppre = pitchPreMap[pmk];
      if (ppre) {
        // Run value (always merge — not hand-dependent at pitch level)
        if (ppre.runValue !== undefined) rows[pmi].runValue = ppre.runValue;
        if (ppre.runValue_pctl !== undefined) rows[pmi].runValue_pctl = ppre.runValue_pctl;
        if (ppre.rv100 !== undefined) rows[pmi].rv100 = ppre.rv100;
        if (ppre.rv100_pctl !== undefined) rows[pmi].rv100_pctl = ppre.rv100_pctl;
        if (ppre.xRunValue !== undefined) rows[pmi].xRunValue = ppre.xRunValue;
        if (ppre.xRunValue_pctl !== undefined) rows[pmi].xRunValue_pctl = ppre.xRunValue_pctl;
        if (ppre.xRv100 !== undefined) rows[pmi].xRv100 = ppre.xRv100;
        if (ppre.xRv100_pctl !== undefined) rows[pmi].xRv100_pctl = ppre.xRv100_pctl;
        // Plate discipline fields not in micro counters
        if (ppre.strikePct !== undefined) rows[pmi].strikePct = ppre.strikePct;
        if (ppre.strikePct_pctl !== undefined) rows[pmi].strikePct_pctl = ppre.strikePct_pctl;
        if (ppre.twoStrikeWhiffPct !== undefined) rows[pmi].twoStrikeWhiffPct = ppre.twoStrikeWhiffPct;
        if (ppre.twoStrikeWhiffPct_pctl !== undefined) rows[pmi].twoStrikeWhiffPct_pctl = ppre.twoStrikeWhiffPct_pctl;
        // Batted ball and expected stats: use per-hand values when hand filter active
        const handSfx = (vsHand === 'L') ? '_vsL' : (vsHand === 'R') ? '_vsR' : '';
        // Batted ball fields
        for (let bbfi = 0; bbfi < PITCH_BB_KEYS.length; bbfi++) {
          const bbf = PITCH_BB_KEYS[bbfi];
          const bbSrc = (handSfx && ppre[bbf + handSfx] !== undefined) ? bbf + handSfx : bbf;
          if (ppre[bbSrc] !== undefined) rows[pmi][bbf] = ppre[bbSrc];
        }
        // Expected stats
        const xKeys = ['wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon', 'xwOBAsp'];
        for (let xi = 0; xi < xKeys.length; xi++) {
          const xk = xKeys[xi];
          const xSrc = (handSfx && ppre[xk + handSfx] !== undefined) ? xk + handSfx : xk;
          if (ppre[xSrc] !== undefined) rows[pmi][xk] = ppre[xSrc];
          // Percentiles are only meaningful for overall (not per-hand)
          if (!handSfx && ppre[xk + '_pctl'] !== undefined) rows[pmi][xk + '_pctl'] = ppre[xk + '_pctl'];
        }
        // Max velo
        if (ppre.maxVelo !== undefined) rows[pmi].maxVelo = ppre.maxVelo;
        // Release Tilt — pre-aggregated only (no micro-data for circular mean
        // under filters). Filtered views fall back to the season value.
        if (ppre.releaseTilt !== undefined) rows[pmi].releaseTilt = ppre.releaseTilt;
        if (ppre.releaseTiltMinutes !== undefined) rows[pmi].releaseTiltMinutes = ppre.releaseTiltMinutes;
        // Stuff+/Loc+/Pitching+: computed from micro grade atoms when present
        // (filtered plain averages, matching sheets/cards); merge the season
        // value only as a fallback for rows without atoms (pre-atom embeds).
        if (rows[pmi].stuffScore == null && ppre.stuffScore !== undefined) rows[pmi].stuffScore = ppre.stuffScore;
        if (ppre.stuffScore_lowSupport !== undefined) rows[pmi].stuffScore_lowSupport = ppre.stuffScore_lowSupport;
        if (rows[pmi].locPlus == null && ppre.locPlus !== undefined) {
          rows[pmi].locPlus = ppre.locPlus;
          if (ppre.locPlusN !== undefined) rows[pmi].locPlusN = ppre.locPlusN;
        }
        if (rows[pmi].pitchingScore == null && ppre.pitchingScore !== undefined) rows[pmi].pitchingScore = ppre.pitchingScore;
        if (ppre.pitchingRuns100 !== undefined) rows[pmi].pitchingRuns100 = ppre.pitchingRuns100;
        // RVOE/xRVOE family (pre-computed residuals vs the stuff+loc
        // expectation — needs the OOF fold models + Loc surfaces,
        // unavailable client-side)
        const oeKeys = ['xrvoe100', 'xrvoe100_pctl', 'rvoe100', 'rvoe100_pctl',
                        'rvoe', 'rvoe_pctl', 'xrvoe', 'xrvoe_pctl'];
        for (let oi = 0; oi < oeKeys.length; oi++) {
          if (ppre[oeKeys[oi]] !== undefined) rows[pmi][oeKeys[oi]] = ppre[oeKeys[oi]];
        }
      }
    }

    // Category rows aggregate their member types' pre-agg stats per pitcher
    if (hasCats) {
      this._mergePlayerCatPitchPreAgg(rows, filters, PITCH_BB_KEYS, vsHand);
    }

    // Multi-team _inPool: per-team rows of players with a 2TM/3TM row interpolate
    // against the combined-row pool rather than participating as pool members.
    const combinedByPitchRowEarly = {};
    for (let cpi = 0; cpi < rows.length; cpi++) {
      if (Aggregator._isCombinedTeam(rows[cpi].team)) {
        combinedByPitchRowEarly[Aggregator._combinedKey(rows[cpi])] = true;
      }
    }
    for (let pi2 = 0; pi2 < rows.length; pi2++) {
      const rp = rows[pi2];
      rp._inPool = !(combinedByPitchRowEarly[Aggregator._combinedKey(rp)] && !Aggregator._isCombinedTeam(rp.team));
    }
    } // end player-mode merge

    // Percentiles per pitch type
    const ptGroups = {};
    rows.forEach(function (r) {
      if (!ptGroups[r.pitchType]) ptGroups[r.pitchType] = [];
      ptGroups[r.pitchType].push(r);
    });

    const MIN_PITCH_TYPE_PCTL = QUAL.MIN_PITCH_PCTL;  // minimum pitches for outcome metrics
    const ABS_PCTL_KEYS = { horzBrk: true, haa: true, nHAA: true, hbOE: true };  // use |value| for RHP/LHP fairness
    // Shape metrics: physical measurements, no minimum needed
    const SHAPE_METRICS = { velocity: true, spinRate: true, indVertBrk: true, horzBrk: true, vaa: true, haa: true, nVAA: true, nHAA: true, ivbOE: true, hbOE: true, stuffScore: true };
    // Batted-ball rate stats qualify on BIP count (>=25 BIP of that pitch type),
    // NOT pitch count — matching the server's PITCH_BB_QUAL_KEYS. Using pitch
    // count would let a type with many pitches but few BIP (e.g. 40 CU / 5 BIP)
    // join the pool on 5 noisy batted balls and diverge from the leaderboard.
    // maxEVAgainst is intentionally excluded (server keeps it on pitch count).
    const PITCH_BB_QUAL_KEYS = { avgEVAgainst: true, hardHitPct: true, barrelPctAgainst: true, hrFbPct: true, ldPct: true, fbPct: true, puPct: true, gbPct: true };
    PITCH_PCTL_KEYS.forEach(function (key) {
      const isBBQual = !!PITCH_BB_QUAL_KEYS[key];
      const minPctl = SHAPE_METRICS[key] ? 0 : (isBBQual ? QUAL.MIN_BIP_PCTL : MIN_PITCH_TYPE_PCTL);
      const countKey = isBBQual ? 'nBip' : 'count';
      for (let pt in ptGroups) {
        // Loc+ gates on its own measured stabilization constant per pitch type
        // (see QUAL.MIN_PITCH_LOCPLUS) — it's displayed unshrunk, so the flat
        // 25 would color cells that are only ~0.26 reliable.
        const ptMin = (key === 'locPlus') ? QUAL.locPlusMinPitches(pt) : minPctl;
        self._computePercentiles(ptGroups[pt], key, ptMin, countKey, ABS_PCTL_KEYS[key] || false);
      }
    });

    // Player-mode category rows merge their RV/discipline stats client-side
    // (no pre-agg _pctl exists for a category) — rank them within the
    // category's own group. Team mode already ranks these keys for all groups.
    if (!teamMode && hasCats) {
      const CAT_MERGED_KEYS = ['runValue', 'rv100', 'xRunValue', 'xRv100', 'strikePct', 'twoStrikeWhiffPct', 'locPlus'];
      for (let ptc in ptGroups) {
        if (!Aggregator.PITCH_CATEGORIES[ptc]) continue;
        CAT_MERGED_KEYS.forEach(function (key) {
          const catMin = (key === 'locPlus') ? QUAL.locPlusMinPitches(ptc) : MIN_PITCH_TYPE_PCTL;
          self._computePercentiles(ptGroups[ptc], key, catMin, 'count', false);
        });
      }
    }

    // --- Pitch-type-specific percentile inversions ---
    // Category groups follow their dominant convention: Hard reads like a
    // fastball (FF), Offspeed like CH/FS; Breaking mixes IVB conventions so
    // its IVB percentile is suppressed.

    // IVB: FF/FC = higher is better (default). SI/CU/CH/FS = lower is better (invert).
    // SL/ST/SV = IVB not meaningful, suppress percentile.
    const IVB_INVERT = { SI: true, CU: true, CH: true, FS: true, Offspeed: true };
    const IVB_SUPPRESS = { SL: true, ST: true, SV: true, Breaking: true };
    for (let ptIVB in ptGroups) {
      if (IVB_INVERT[ptIVB]) {
        ptGroups[ptIVB].forEach(function (r) {
          if (r.indVertBrk_pctl !== null && r.indVertBrk_pctl !== undefined) {
            r.indVertBrk_pctl = 100 - r.indVertBrk_pctl;
          }
        });
      } else if (IVB_SUPPRESS[ptIVB]) {
        ptGroups[ptIVB].forEach(function (r) {
          r.indVertBrk_pctl = null;
        });
      }
    }

    // Spin: higher is better for all EXCEPT CH/FS where lower spin = better (invert)
    const SPIN_INVERT = { CH: true, FS: true, Offspeed: true };
    for (let ptSpin in ptGroups) {
      if (SPIN_INVERT[ptSpin]) {
        ptGroups[ptSpin].forEach(function (r) {
          if (r.spinRate_pctl !== null && r.spinRate_pctl !== undefined) {
            r.spinRate_pctl = 100 - r.spinRate_pctl;
          }
        });
      }
    }

    // VAA/nVAA: FF/FC = closer to 0 is better (default higher = higher pctl, no inversion)
    // All others: further from 0 = better (invert)
    const VAA_NO_INVERT = { FF: true, FC: true, Hard: true };
    for (let ptV in ptGroups) {
      if (!VAA_NO_INVERT[ptV]) {
        ptGroups[ptV].forEach(function (r) {
          if (r.vaa_pctl !== null && r.vaa_pctl !== undefined) {
            r.vaa_pctl = 100 - r.vaa_pctl;
          }
          if (r.nVAA_pctl !== null && r.nVAA_pctl !== undefined) {
            r.nVAA_pctl = 100 - r.nVAA_pctl;
          }
        });
      }
    }

    // HAA/nHAA: uses absolute values, so further from 0 = higher pctl by default
    // FF/FC: closer to 0 = better, so invert for fastballs
    for (let ptH in ptGroups) {
      if (VAA_NO_INVERT[ptH]) {
        ptGroups[ptH].forEach(function (r) {
          if (r.haa_pctl !== null && r.haa_pctl !== undefined) {
            r.haa_pctl = 100 - r.haa_pctl;
          }
          if (r.nHAA_pctl !== null && r.nHAA_pctl !== undefined) {
            r.nHAA_pctl = 100 - r.nHAA_pctl;
          }
        });
      }
    }

    // Invert batted ball + expected stat percentiles where lower is better for pitchers
    for (let ptBB in ptGroups) {
      ptGroups[ptBB].forEach(function (r) {
        for (let bbInv in PITCH_BB_INVERT) {
          const bbPk = bbInv + '_pctl';
          if (r[bbPk] !== null && r[bbPk] !== undefined) {
            r[bbPk] = 100 - r[bbPk];
          }
        }
        for (let xInv in PITCH_EXPECTED_INVERT) {
          const xPk = xInv + '_pctl';
          if (r[xPk] !== null && r[xPk] !== undefined) {
            r[xPk] = 100 - r[xPk];
          }
        }
      });
    }

    // Team mode: every team row is pool-qualified; player-level Q filters and
    // multi-team handling don't apply.
    if (teamMode) {
      return this._narrowTeamRows(rows, filters);
    }

    // Multi-team lookups (any row with team like "2TM"/"3TM" stands in for its player)
    const combinedByPitchRow = {};
    for (let cti = 0; cti < rows.length; cti++) {
      if (Aggregator._isCombinedTeam(rows[cti].team)) {
        combinedByPitchRow[Aggregator._combinedKey(rows[cti])] = true;
      }
    }

    // Apply IP-based qualification filter for pitch-type data (Qualified mode).
    // For multi-team players, qualification uses the combined 2TM/3TM row's IP and
    // the sum of team games across their MLB teams (cumulative per-player view).
    if (filters.minIp === 'Q') {
      const teamGames = this.getTeamGamesPlayed();
      const ipLookup = {};
      const preAggIP = window.PITCHER_DATA || [];
      for (var ipi = 0; ipi < preAggIP.length; ipi++) {
        ipLookup[preAggIP[ipi].pitcher + '|' + preAggIP[ipi].team] = preAggIP[ipi];
      }
      // Precompute cumulative team games per multi-team pitcher
      const cumTeamGamesPitch = {};
      for (var ckey in ipLookup) {
        const ent = ipLookup[ckey];
        const entKey = Aggregator._combinedKey(ent);
        if (combinedByPitchRow[entKey] && !Aggregator._isCombinedTeam(ent.team)) {
          const tgv = teamGames[ent.team] || 0;
          if (tgv > (cumTeamGamesPitch[entKey] || 0)) cumTeamGamesPitch[entKey] = tgv;
        }
      }
      rows = rows.filter(function (r) {
        const rKey = Aggregator._combinedKey(r);
        // For multi-team players use their combined row's IP + cumulative team games
        if (combinedByPitchRow[rKey]) {
          const numTeams = Object.keys(ipLookup).filter(function (k) {
            return Aggregator._combinedKey(ipLookup[k]) === rKey && Aggregator._isCombinedTeam(ipLookup[k].team);
          });
          const combinedKey = numTeams[0];
          const cp = combinedKey ? ipLookup[combinedKey] : null;
          if (!cp) return false;
          const tg = cumTeamGamesPitch[rKey] || 0;
          const ipFloat = Utils.parseIP(cp.ip);
          const isStarter = Utils.isStarter(cp.g, cp.gs);
          const isROC = Aggregator._isROCTeam(r.team);
          return ipFloat >= tg * Utils.pitcherIpPerGame(isStarter, isROC);
        }
        var p = ipLookup[r.pitcher + '|' + r.team];
        if (!p) return false;
        var tg = teamGames[r.team] || 0;
        var ipFloat = Utils.parseIP(p.ip);
        var isStarter = Utils.isStarter(p.g, p.gs);
        var isROC = Aggregator._isROCTeam(r.team);
        return ipFloat >= tg * Utils.pitcherIpPerGame(isStarter, isROC);
      });
    }

    // Apply view-narrowing filters AFTER percentiles (don't change comparison group)
    // "All Teams": hide ROC; hide per-team rows of multi-team players (2TM/3TM row stands in).
    // Specific team: hide combined rows; show only per-team rows for that team.
    if (filters.team !== 'all') {
      rows = rows.filter(function (r) { return r.team === filters.team; });
    } else {
      rows = rows.filter(function (r) {
        // includeROC/keepStints: see the pitcher narrowing — player-page platoon only.
        if (self._isROCTeam(r.team)) return !!filters.includeROC;
        if (combinedByPitchRow[Aggregator._combinedKey(r)] && !Aggregator._isCombinedTeam(r.team)) return !!filters.keepStints;
        return true;
      });
    }
    if (filters.search) {
      const searchLower2 = filters.search.toLowerCase();
      rows = rows.filter(function (r) { return r.pitcher.toLowerCase().indexOf(searchLower2) !== -1; });
    }

    return rows;
  },

  // Team mode: hitter bats (stands) filter resolved per hitter — switch
  // hitters are excluded from R/L views, matching player-mode semantics.
  // FanGraphs Guts per-event linear weights, shipped in metadata by
  // process_data.py. Absent on embeds built before 2026-08-04, in which case
  // every wOBA consumer here falls back to the boxscore-merged season value.
  _wobaWeights: function () {
    if (this._wobaW) return this._wobaW;
    const w = (typeof DataStore !== 'undefined' && DataStore.metadata &&
               DataStore.metadata.wobaWeights) || null;
    const ok = w && ['BB', 'HBP', '1B', '2B', '3B', 'HR'].every(function (k) {
      return typeof w[k] === 'number';
    });
    // Only a successful lookup is cached — metadata may not have landed yet
    // on the first call, and caching that miss would disable wOBA all session.
    if (ok) this._wobaW = w;
    return ok ? w : null;
  },

  _buildHitterStandsMap: function (micro, ci) {
    const sets = {};
    for (let i = 0; i < micro.length; i++) {
      const hk = micro[i][ci.hitterIdx] + '|' + micro[i][ci.teamIdx];
      if (!sets[hk]) sets[hk] = {};
      sets[hk][micro[i][ci.bats]] = true;
    }
    const map = {};
    for (const hk in sets) {
      const keys = Object.keys(sets[hk]);
      map[hk] = keys.length > 1 ? 'S' : (keys[0] || null);
    }
    return map;
  },

  /**
   * Team mode: aggregate season boxscore/pre-computed hitter stats from
   * HITTER_DATA to team level. Counting stats are true sums (RV at full
   * precision); rates are weighted by their natural denominators (PA for
   * wOBA/wRC+/Hitter+, competitive swings for bat tracking, competitive runs
   * for sprint speed, scored-pitch N for SD+/CT+). Respects the bats filter.
   */
  _teamHitterBoxscore: function (filters) {
    const hd = window.HITTER_DATA || [];
    const acc = {};

    function wadd(a, key, val, w) {
      if (val == null || !(w > 0)) return;
      a.sums[key] = (a.sums[key] || 0) + val * w;
      a.wts[key] = (a.wts[key] || 0) + w;
    }

    for (let i = 0; i < hd.length; i++) {
      const h = hd[i];
      if (this._isCombinedTeam(h.team)) continue;
      if (filters.throws !== 'all' && h.stands !== filters.throws) continue;

      let a = acc[h.team];
      if (!a) {
        a = acc[h.team] = { tb: 0, sb: 0, cs: 0, wRC: null, runValue: null,
                            nCompSwings: 0, nCompRuns: 0, pa: 0, sums: {}, wts: {} };
      }
      a.tb += h.tb || 0; a.sb += h.sb || 0; a.cs += h.cs || 0;
      a.pa += h.pa || 0;
      a.nCompSwings += h.nCompSwings || 0;
      a.nCompRuns += h.nCompRuns || 0;
      if (h.wRC != null) a.wRC = (a.wRC || 0) + h.wRC;
      // RV sums stay full precision; rounding happens only at display
      if (h.runValue != null) a.runValue = (a.runValue || 0) + h.runValue;
      wadd(a, 'wOBA', h.wOBA, h.pa);
      wadd(a, 'wRCplus', h.wRCplus, h.pa);
      wadd(a, 'xWRCplus', h.xWRCplus, h.pa);
      wadd(a, 'hitterPlus', h.hitterPlus, h.pa);
      wadd(a, 'sdPlus', h.sdPlus, h.sdPlusN);
      wadd(a, 'ctPlus', h.ctPlus, h.ctPlusN);
      const compW = h.nCompSwings;
      wadd(a, 'batSpeed', h.batSpeed, compW);
      wadd(a, 'swingLength', h.swingLength, compW);
      wadd(a, 'attackAngle', h.attackAngle, compW);
      wadd(a, 'attackDirection', h.attackDirection, compW);
      wadd(a, 'swingPathTilt', h.swingPathTilt, compW);
      wadd(a, 'blastPct', h.blastPct, compW);
      wadd(a, 'squaredUpPct', h.squaredUpPct, compW);
      wadd(a, 'idealAAPct', h.idealAAPct, compW);
      wadd(a, 'sprintSpeed', h.sprintSpeed, h.nCompRuns);
    }

    const teamGames = this.getTeamGamesPlayed();
    const out = {};
    for (const team in acc) {
      const a = acc[team];
      const o = {
        g: teamGames[team] || null,
        tb: a.tb, sb: a.sb, cs: a.cs,
        sbPct: (a.sb + a.cs) > 0 ? a.sb / (a.sb + a.cs) * 100 : null,
        wRC: a.wRC,
        runValue: a.runValue,
        nCompSwings: a.nCompSwings,
        nCompRuns: a.nCompRuns,
        paAll: a.pa,
        sprintQual: true,
        sdPlusN: a.wts.sdPlus || 0,
        ctPlusN: a.wts.ctPlus || 0,
      };
      for (const k in a.sums) {
        if (a.wts[k] > 0) o[k] = a.sums[k] / a.wts[k];
      }
      out[team] = o;
    }
    return out;
  },

  // Team mode tail of _aggregateHitter: merge team boxscore stats, compute
  // percentiles over the team pool, invert lower-is-better, narrow the view.
  _finishHitterTeamRows: function (rows, filters, STAT_KEYS, INVERT) {
    const box = this._teamHitterBoxscore(filters);
    for (let i = 0; i < rows.length; i++) {
      const b = box[rows[i].team];
      if (b) {
        for (const k in b) rows[i][k] = b[k];
      }
    }
    this._flagTeamRows(rows);
    const self = this;
    STAT_KEYS.forEach(function (key) {
      self._computePercentiles(rows, key);
    });
    for (let ri = 0; ri < rows.length; ri++) {
      for (const inv in INVERT) {
        const pk = inv + '_pctl';
        if (rows[ri][pk] !== null && rows[ri][pk] !== undefined) {
          rows[ri][pk] = 100 - rows[ri][pk];
        }
      }
    }
    return this._narrowTeamRows(rows, filters);
  },

  _aggregateHitter: function (filters) {
    const d = this.data;
    const ci = this._colIdx.hitterCols;
    const bci = this._colIdx.hitterBipCols;
    const micro = d.hitterMicro;
    const hitterMlbIdMap = this._getMlbIdMap('hitter');
    const bipData = d.hitterBip;
    const lookups = d.lookups;
    const validDates = this._getValidDateSet(filters);
    const vsHand = filters.vsHand || 'all';
    const teamMode = filters.viewMode === 'team';
    const standsMap = teamMode ? this._buildHitterStandsMap(micro, ci) : null;
    const self2 = this;
    const teamRowOk = function (hitterIdx, teamIdx) {
      if (self2._isCombinedTeam(lookups.teams[teamIdx])) return false;
      if (filters.throws !== 'all' && standsMap[hitterIdx + '|' + teamIdx] !== filters.throws) return false;
      return true;
    };

    // Group by (hitterIdx, teamIdx) — or by teamIdx alone in team mode
    const groups = {};
    for (let i = 0; i < micro.length; i++) {
      const row = micro[i];
      if (!validDates[row[ci.dateIdx]]) continue;
      if (vsHand !== 'all' && row[ci.pitcherHand] !== vsHand) continue;
      if (teamMode && !teamRowOk(row[ci.hitterIdx], row[ci.teamIdx])) continue;

      const gk = teamMode ? String(row[ci.teamIdx]) : (row[ci.hitterIdx] + '|' + row[ci.teamIdx]);
      if (!groups[gk]) {
        groups[gk] = {
          hitterIdx: teamMode ? null : row[ci.hitterIdx],
          teamIdx: row[ci.teamIdx],
          batsSet: {},
          counts: new Array(51)  // hitter-micro data cols (buntAB at 49, ibb at 50)
        };
        for (let z = 0; z < 51; z++) groups[gk].counts[z] = 0;
      }

      const g = groups[gk];
      g.batsSet[row[ci.bats]] = true;

      // Pre-ibb embeds are 50 wide; row[55] is undefined there and must not
      // poison the accumulator with NaN.
      for (let f = 0; f < 51; f++) {
        g.counts[f] += row[5 + f] || 0;
      }
    }

    // Filter BIP records for medians
    const bipByHitter = {};
    for (let bi = 0; bi < bipData.length; bi++) {
      const brow = bipData[bi];
      if (!validDates[brow[bci.dateIdx]]) continue;
      if (vsHand !== 'all' && brow[bci.pitcherHand] !== vsHand) continue;
      if (teamMode && !teamRowOk(brow[bci.hitterIdx], brow[bci.teamIdx])) continue;

      const hKey = teamMode ? String(brow[bci.teamIdx]) : (brow[bci.hitterIdx] + '|' + brow[bci.teamIdx]);
      if (!bipByHitter[hKey]) bipByHitter[hKey] = [];
      bipByHitter[hKey].push(brow);
    }

    function median(arr) {
      if (arr.length === 0) return null;
      arr.sort(function (a, b) { return a - b; });
      const mid = Math.floor(arr.length / 2);
      return arr.length % 2 === 1 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;
    }

    const HITTER_STAT_KEYS = [
      'avg', 'obp', 'slg', 'ops', 'iso', 'wOBA', 'babip', 'kPct', 'bbPct', 'bbToK',
      'xBA', 'xSLG', 'xwOBA', 'xwOBAcon', 'xwOBAsp', 'sprayVal', 'bbPlus',
      'avgEVAll', 'ev50', 'maxEV', 'hardHitPct', 'barrelPct',
      'gbPct', 'ldPct', 'fbPct', 'puPct', 'hrFbPct',
      'pullPct', 'airPullPct',
      'swingPct', 'izSwingPct', 'chasePct', 'izSwChase', 'contactPct', 'izContactPct', 'whiffPct', 'sdPlus', 'ctPlus',
      'batSpeed', 'swingLength', 'blastPct', 'squaredUpPct', 'idealAAPct',
      'twoStrikeWhiffPct', 'firstPitchSwingPct',
      'avgFbDist', 'avgHrDist',
      'sprintSpeed', 'runValue',
      'wRCplus', 'xWRCplus', 'hitterPlus',
      'hr', 'sb',
    ];
    const HITTER_INVERT = {
      chasePct: true, whiffPct: true, gbPct: true, kPct: true, puPct: true, twoStrikeWhiffPct: true, firstPitchSwingPct: true,
      oppoPct: true   // mirrors pipeline_compute.HITTER_INVERT_PCTL (2026-08-12:
                      // swingPct dropped, a swing rate has no honest good direction)
    };

    let rows = [];
    const sacqMaps = Aggregator.buildSacqZoneMaps();
    for (let gk2 in groups) {
      const g = groups[gk2];
      const c = g.counts;
      const batsKeys = Object.keys(g.batsSet);
      const stands = batsKeys.length > 1 ? 'S' : (batsKeys[0] || null);

      const n_total = c[0], pa = c[1], h = c[2], db = c[3], tp = c[4], hr = c[5];
      const bb = c[6], hbp = c[7], sf = c[8], sh = c[9], ci_v = c[10], k = c[11];
      const swings = c[12], whiffs = c[13];
      const izPitches = c[14], oozPitches = c[15];
      const izSwings = c[16], oozSwings = c[17], contact = c[18];
      const izSwNonBunt = c[19], izContact = c[20];
      const bip = c[21], gb_c = c[22], ld = c[23], fb = c[24], pu = c[25];
      const barrels = c[26], nSpray = c[27], pull = c[28], center = c[29], oppo = c[30], airPull = c[31];
      const hardHit = c[32], nHrBip = c[33], ldHr = c[34];
      const twoStrikeSwings = c[35], twoStrikeWhiffs = c[36];
      const firstPitchAppearances = c[37], firstPitchSwings = c[38];
      const xBA_sum = c[39], xBA_count = c[40], xSLG_sum = c[41], xSLG_count = c[42];
      const xwOBA_sum = c[43], xwOBA_count = c[44], xwOBAcon_sum = c[45], xwOBAcon_count = c[46];
      const swingsNonBunt = c[47], contactNonBunt = c[48], buntAB = c[49] || 0;
      // Intentional walks. Absent on pre-2026-08-04 embeds (hitter micro was
      // 50 counts wide) — hasIbbCol distinguishes "no IBBs" from "column not
      // shipped", since only the former is safe to build a uBB-based wOBA from.
      const ibb = c[50] || 0;
      const hasIbbCol = ci.ibb !== undefined;

      const ab = pa - bb - hbp - sf - sh - ci_v;
      const nonbuntAB = ab - buntAB;  // xBA/xSLG denominator (Savant excludes bunts)
      const singles = h - db - tp - hr;
      const tb_val = singles + 2 * db + 3 * tp + 4 * hr;
      const xbh = db + tp + hr;

      const batting_avg = ab > 0 ? Math.round(h / ab * 1000) / 1000 : null;
      const obp_denom = ab + bb + hbp + sf;
      const obp_val = obp_denom > 0 ? Math.round((h + bb + hbp) / obp_denom * 1000) / 1000 : null;
      const slg_val = ab > 0 ? Math.round(tb_val / ab * 1000) / 1000 : null;
      const ops_val = (obp_val !== null && slg_val !== null) ? Math.round((obp_val + slg_val) * 1000) / 1000 : null;
      const iso_val = (slg_val !== null && batting_avg !== null) ? Math.round((slg_val - batting_avg) * 1000) / 1000 : null;
      const babip_denom = ab - k - hr + sf;
      const babip_val = babip_denom > 0 ? Math.round((h - hr) / babip_denom * 1000) / 1000 : null;

      // wOBA from the micro counters + FanGraphs Guts linear weights, so it
      // survives a handedness or date filter instead of falling back to the
      // frozen season value merged from the boxscore. Same formula and same
      // weights process_data.py uses on the boxscore counts, including uBB
      // (IBB carries no wOBA weight), so a filtered wOBA sits on the identical
      // scale as the unfiltered one. Null unless BOTH the weights and the ibb
      // column are present — a BB-based approximation would silently read a
      // few points high against the season number beside it.
      const wobaW = Aggregator._wobaWeights();
      let woba_val = null;
      if (wobaW && hasIbbCol) {
        const ubb = bb - ibb;
        const woba_denom = ab + ubb + sf + hbp;
        if (woba_denom > 0) {
          woba_val = Math.round((wobaW.BB * ubb + wobaW.HBP * hbp +
                                 wobaW['1B'] * singles + wobaW['2B'] * db +
                                 wobaW['3B'] * tp + wobaW.HR * hr) / woba_denom * 1000) / 1000;
        }
      }

      const kPct = pa > 0 ? k / pa : null;
      const bbPct = pa > 0 ? bb / pa : null;
      // BB/K — full precision; display rounds to 2 decimals
      const bbToK = k > 0 ? bb / k : null;
      const izSwingPct = izPitches > 0 ? izSwings / izPitches : null;
      const chasePct_val = oozPitches > 0 ? oozSwings / oozPitches : null;
      const izSwChase = (izSwingPct !== null && chasePct_val !== null)
        ? Math.round((izSwingPct - chasePct_val) * 10000) / 10000 : null;
      const contactPct = swingsNonBunt > 0 ? contactNonBunt / swingsNonBunt : null;
      const izContactPct = izSwNonBunt > 0 ? izContact / izSwNonBunt : null;
      // strict-conditional HR/FB — mirrors pipeline_compute
      const hrFbPct_val = fb > 0 ? Math.max(nHrBip - ldHr, 0) / fb : null;

      // BIP medians
      const bipRecords = bipByHitter[teamMode ? String(g.teamIdx) : (g.hitterIdx + '|' + g.teamIdx)] || [];
      const evsAll = [], allLA = [];
      for (let bri = 0; bri < bipRecords.length; bri++) {
        const bev = bipRecords[bri][bci.exitVelo];
        const bla = bipRecords[bri][bci.launchAngle];
        if (bev !== null) evsAll.push(bev);
        if (bla !== null) allLA.push(bla);
      }

      const avgEVAll = evsAll.length > 0 ? Math.round(evsAll.reduce(function(a,b){return a+b;},0) / evsAll.length * 10) / 10 : null;
      const maxEV = evsAll.length > 0 ? Math.round(Math.max.apply(null, evsAll) * 10) / 10 : null;
      const medLA = allLA.length > 0 ? Math.round(median(allLA.slice()) * 10) / 10 : null;

      // EV50: average of top 50% hardest-hit balls across ALL BIP (no LA filter)
      // Matches Savant's "Best Speed" / EV50 — weak contact is noise
      let ev50 = null;
      if (evsAll.length > 0) {
        const sorted = evsAll.slice().sort(function (a, b) { return b - a; });
        const topHalf = sorted.slice(0, Math.max(1, Math.floor(sorted.length / 2)));
        ev50 = Math.round(topHalf.reduce(function (s, v) { return s + v; }, 0) / topHalf.length * 10) / 10;
      }

      // hardHitPct and barrelPct: use EV-valid count as denominator (not total BIP)
      const hardHitPct = evsAll.length > 0 ? hardHit / evsAll.length : null;

      // xwOBAsp + sprayVal — computed from BIP records using the hand-specific
      // zone tables with pooled fallback. sprayVal is the LA-residualized
      // placement skill: zone wOBAcon minus LA-only league wOBAcon per BIP.
      let xwOBAsp_val = null;
      let sprayVal_val = null;
      if (bipRecords.length > 0) {
        let xwOBAsp_sum = 0, xwOBAsp_count = 0;
        let spray_sum = 0, spray_count = 0;
        for (let sri = 0; sri < bipRecords.length; sri++) {
          const sla = bipRecords[sri][bci.launchAngle];
          const shcX = bipRecords[sri][bci.hcX];
          const shcY = bipRecords[sri][bci.hcY];
          if (sla == null || shcX == null || shcY == null) continue;
          // Use the per-BIP bat side (not the aggregated stands) so a switch
          // hitter's righty-side and lefty-side BIP each use the correct spray
          // orientation + hand-specific zone, matching Python compute_xwobasp.
          // Team rows already required this; non-team switch hitters ('S') were
          // collapsing to the L-branch + pooled zones.
          const sBats = bipRecords[sri][bci.batSide] || stands;
          const sAngle = Aggregator.computeSprayAngle(shcX, shcY);
          const sDir = Aggregator.sprayDirection(sAngle, sBats);
          if (!sDir) continue;
          const sLaBin = Aggregator.getLABinIdx(sla);
          if (sLaBin == null) continue;
          const zWoba = Aggregator.sacqLookup(sacqMaps, sDir, sLaBin, sBats);
          if (zWoba != null) {
            xwOBAsp_sum += zWoba;
            xwOBAsp_count++;
            const laWoba = Aggregator.sacqLaLookup(sacqMaps, sLaBin, sBats);
            if (laWoba != null) {
              spray_sum += zWoba - laWoba;
              spray_count++;
            }
          }
        }
        xwOBAsp_val = xwOBAsp_count > 0 ? xwOBAsp_sum / xwOBAsp_count : null;
        sprayVal_val = spray_count > 0 ? spray_sum / spray_count : null;
      }

      const hitterName = g.hitterIdx != null ? lookups.hitters[g.hitterIdx] : null;
      const hitterTeam = lookups.teams[g.teamIdx];
      const obj = {
        hitter: hitterName,
        team: hitterTeam,
        mlbId: hitterName != null ? (hitterMlbIdMap[hitterName + '|' + hitterTeam] || null) : null,
        stands: teamMode ? null : stands,
        count: n_total,
        pa: pa,
        ab: ab,
        nSwings: swings,
        nBip: bip,
        avg: batting_avg,
        obp: obp_val,
        slg: slg_val,
        ops: ops_val,
        doubles: db,
        triples: tp,
        hr: hr,
        xbh: xbh,
        kPct: kPct,
        bbPct: bbPct,
        bbToK: bbToK,
        iso: iso_val,
        babip: babip_val,
        avgEVAll: avgEVAll,
        ev50: ev50,
        maxEV: maxEV,
        medLA: medLA,
        hardHitPct: hardHitPct,
        barrelPct: evsAll.length > 0 ? barrels / evsAll.length : null,
        xwOBAsp: xwOBAsp_val,
        sprayVal: sprayVal_val,
        gbPct: bip > 0 ? gb_c / bip : null,
        ldPct: bip > 0 ? ld / bip : null,
        fbPct: bip > 0 ? fb / bip : null,
        puPct: bip > 0 ? pu / bip : null,
        hrFbPct: hrFbPct_val,
        pullPct: nSpray > 0 ? pull / nSpray : null,
        middlePct: nSpray > 0 ? center / nSpray : null,
        oppoPct: nSpray > 0 ? oppo / nSpray : null,
        airPullPct: bip > 0 ? airPull / bip : null,
        swingPct: n_total > 0 ? swings / n_total : null,
        izSwingPct: izSwingPct,
        chasePct: chasePct_val,
        izSwChase: izSwChase,
        contactPct: contactPct,
        izContactPct: izContactPct,
        whiffPct: swingsNonBunt > 0 ? whiffs / swingsNonBunt : null,
        twoStrikeWhiffPct: twoStrikeSwings > 0 ? twoStrikeWhiffs / twoStrikeSwings : null,
        firstPitchSwingPct: firstPitchAppearances > 0 ? firstPitchSwings / firstPitchAppearances : null,
        xBA: nonbuntAB > 0 && xBA_count > 0 ? xBA_sum / nonbuntAB : null,
        xSLG: nonbuntAB > 0 && xSLG_count > 0 ? xSLG_sum / nonbuntAB : null,
        xwOBA: xwOBA_count > 0 ? xwOBA_sum / xwOBA_count : null,
        xwOBAcon: xwOBAcon_count > 0 ? xwOBAcon_sum / xwOBAcon_count : null,
        wOBA: woba_val,
      };

      // xWRC+ from the filtered xwOBA. Pure function of xwOBA and the Guts
      // constants — deliberately NO park factor (xwOBA is already mapped to
      // league-average-park outcomes; process_data.py drops the PF term here
      // for the same reason), so this reproduces the server formula exactly.
      // wRC+ is NOT recomputed: process_data.py overwrites it with the
      // canonical FanGraphs value, and FanGraphs publishes no per-hand split,
      // so a pipeline-formula wRC+ would sit on a visibly different scale than
      // the season number next to it.
      const guts = (typeof DataStore !== 'undefined' && DataStore.metadata &&
                    DataStore.metadata.gutsConstants) || null;
      if (guts && guts.wOBAScale > 0 && guts.lgRPA > 0 && obj.xwOBA != null) {
        var _xw = (((obj.xwOBA - guts.lgWOBA) / guts.wOBAScale) + guts.lgRPA)
                  / guts.lgRPA * 100;
        // run-truth cap (2026-08-15): player rows ship rescaled from the
        // pipeline; apply the same published factor here so team-mode
        // formula values stay on the shipped scale.
        var _xs = (DataStore.metadata.plusWrcScale || {}).xWRCplus;
        if (_xs && _xs.factor) _xw = 100 + (_xw - 100) * _xs.factor + (_xs.shift || 0);
        obj.xWRCplus = Math.round(_xw);
      }

      // BB+ (2026-07-13 definition): weighted xwOBAcon+ / sprayPlus indexed
      // so 100 = league avg, then Bayesian-regressed toward 100 by sample
      // size. The shipped weights are pure-con (con 1.0 / sp 0.0 — the
      // multi-season derivation retired the spray term), but the formula
      // stays general. Weights AND shrinkage n0 come from metadata
      // (bbPlusWeights / bbPlusShrinkN0 / bbPlusMinBip — single source of
      // truth, set in process_data.py) so this recompute can never drift
      // from the server definition again.
      const hLgAvgs = (DataStore && DataStore.metadata && DataStore.metadata.hitterLeagueAverages) || {};
      const lgXC = hLgAvgs.xwOBAcon;
      // Reliability handling: shrink toward 100 with n0 pseudo-BIPs
      // (measured r=.50 crossing, 2021-2026 study), display-floored at
      // bbPlusMinBip (below it a score is >2/3 prior). Matters most here
      // because date/hand filters shrink the sample — a filtered 40-BIP
      // slice now shows a heavily regressed value instead of a blank.
      // (Hitter+ is pass-through season value, not recomputed client-side,
      // so it's gated server-side instead.)
      const bbW = (DataStore && DataStore.metadata && DataStore.metadata.bbPlusWeights) || null;
      const bbN0 = (DataStore && DataStore.metadata && DataStore.metadata.bbPlusShrinkN0) || 60;
      const bbMinBip = (DataStore && DataStore.metadata && DataStore.metadata.bbPlusMinBip) || 30;
      const nBipBB = obj.nBip || 0;
      const spOk = bbW && (bbW.sp === 0 || obj.sprayVal != null);
      if (obj.xwOBAcon != null && spOk && lgXC && bbW && nBipBB >= bbMinBip) {
        const conPlus = 100 * obj.xwOBAcon / lgXC;
        const spPlus = obj.sprayVal != null ? 100 * (lgXC + obj.sprayVal) / lgXC : 100;
        const rawBB = bbW.con * conPlus + bbW.sp * spPlus;
        // Shrink BEFORE the re-anchor, mirroring the server order.
        const shrunkBB = (nBipBB * rawBB + bbN0 * 100) / (nBipBB + bbN0);
        // Mirror the server's re-anchor (all-MLB PA-weighted mean = 100).
        // sd/ct/hitterPlus are pass-through; bbPlus is the only "+"
        // recomputed client-side, so it must apply the same factor.
        const bbReAnchor = (DataStore && DataStore.metadata &&
                            DataStore.metadata.plusReanchor &&
                            DataStore.metadata.plusReanchor.bbPlus) || 1;
        let bbVal = shrunkBB * bbReAnchor;
        // Then the wRC+-spread match, same order as the server: rescale around
        // 100, then the additive shift. Without this a filtered BB+ would sit
        // on the old ratio scale while SD+/CT+ (pass-through, already matched)
        // sat on the new one.
        const bbWrc = (DataStore && DataStore.metadata &&
                       DataStore.metadata.plusWrcScale &&
                       DataStore.metadata.plusWrcScale.bbPlus) || null;
        if (bbWrc && bbWrc.factor) {
          bbVal = 100 + (bbVal - 100) * bbWrc.factor + (bbWrc.shift || 0);
        }
        obj.bbPlus = Math.round(bbVal * 10) / 10;
      } else {
        obj.bbPlus = null;
      }

      // Hitter+ is precomputed on the server (needs SD+ and CT+ which require
      // per-pitch weight-table lookups not available client-side). Keep the
      // season-long hitterPlus value from the boxscore merge — it flows through
      // the hBoxAlways fields alongside sdPlus/ctPlus/wRCplus.

      // Compute avgFbDist and avgHrDist from BIP records
      if (bipRecords.length > 0 && bci.distance !== undefined) {
        const fbDists = [], hrDists = [];
        for (let dri = 0; dri < bipRecords.length; dri++) {
          const dr = bipRecords[dri];
          const dist = dr[bci.distance];
          if (dist == null) continue;
          if (dr[bci.bbType] === 2) fbDists.push(dist); // fly_ball = 2
          if (dr[bci.event] === 4) hrDists.push(dist);   // HR = 4
        }
        obj.avgFbDist = fbDists.length > 0 ? Math.round(fbDists.reduce(function(a,b){return a+b;},0) / fbDists.length) : null;
        obj.avgHrDist = hrDists.length > 0 ? Math.round(hrDists.reduce(function(a,b){return a+b;},0) / hrDists.length) : null;
      }

      if (teamMode) {
        // Bats filter applied per hitter pre-rollup; player-level minimum
        // sample filters are intentionally ignored for team totals.
        obj._isTeamRow = true;
        rows.push(obj);
        continue;
      }

      // Apply baseball-context filters (comparison group — affects percentiles)
      if (filters.throws !== 'all' && obj.stands !== filters.throws) continue;
      if (filters.minCount !== 'Q' && (obj.pa || 0) < (filters.minCount || 1)) continue;
      if (filters.minSwings && obj.nSwings < filters.minSwings) continue;
      if (filters.minBip && (obj.nBip || 0) < filters.minBip) continue;

      rows.push(obj);
    }

    if (teamMode) {
      return this._finishHitterTeamRows(rows, filters, HITTER_STAT_KEYS, HITTER_INVERT);
    }

    // Merge boxscore stats from pre-aggregated HITTER_DATA
    // When contextual filters (vsHand, date range) are active, skip rate stats
    // that micro data already computed correctly for the filtered subset.
    const hasHitterContextFilter = (vsHand !== 'all') ||
                                    (filters.dateStart || filters.dateEnd);
    // Stats that only boxscore/external data can provide (always merge)
    // wOBA/wRC/wRCplus/xWRCplus need FanGraphs weights + park factors, not computable from micro
    const hBoxAlways = ['g', 'tb', 'sb', 'cs', 'sbPct', 'runValue',
                        'batSpeed', 'swingLength', 'attackAngle', 'attackDirection', 'swingPathTilt', 'nCompSwings', 'blastPct', 'squaredUpPct', 'idealAAPct',
                        'sprintSpeed', 'nCompRuns', 'sprintQual',
                        'wOBA', 'wRC', 'wRCplus', 'xWRCplus',
                        // SD+, CT+, and Hitter+ are precomputed against the
                        // full season (need the 360-cell / 60-cell RV weight
                        // tables and hitter-standardization SDs that aren't
                        // available client-side), so always surface the
                        // season values even under filters.
                        'sdPlus', 'sdPlusN', 'sdPlusRaw',
                        'ctPlus', 'ctPlusN', 'ctPlusRaw',
                        'hitterPlus'];
    // Rate stats that micro data computes (skip when filtered)
    const hBoxRateStats = ['avg', 'obp', 'slg', 'ops', 'iso', 'babip', 'kPct', 'bbPct', 'bbToK',
                           'doubles', 'triples', 'hr', 'xbh'];
    const hBoxFields = hasHitterContextFilter ? hBoxAlways : hBoxAlways.concat(hBoxRateStats);
    const hPreAgg = window.HITTER_DATA || [];
    const hPreAggMap = {};
    for (let hbi = 0; hbi < hPreAgg.length; hbi++) {
      hPreAggMap[hPreAgg[hbi].hitter + '|' + hPreAgg[hbi].team] = hPreAgg[hbi];
    }
    // wOBA and xWRC+ sit in hBoxAlways because the client historically could
    // not compute them. It can now — micro counters + the Guts weights and
    // constants shipped in metadata — so under a filter the micro value wins.
    //
    // Note the merge is skipped under a filter even when the micro value came
    // back null (an embed built before the weights/ibb column shipped). That
    // leaves the cell empty for one pipeline run, which is the honest answer:
    // pasting the SEASON wOBA into a row labelled "vs LHP" is not a missing
    // number, it's a wrong one.
    const hBoxYieldToMicro = { wOBA: true, xWRCplus: true };
    for (let hmi = 0; hmi < rows.length; hmi++) {
      const hKey = rows[hmi].hitter + '|' + rows[hmi].team;
      const hPre = hPreAggMap[hKey];
      if (hPre) {
        for (let hfi = 0; hfi < hBoxFields.length; hfi++) {
          const hbf = hBoxFields[hfi];
          if (hBoxYieldToMicro[hbf] && hasHitterContextFilter) continue;
          if (hPre[hbf] !== undefined) rows[hmi][hbf] = hPre[hbf];
        }
        // Always attach overall-season PA for qualification gates (platoon-split safe).
        // row.pa stays as the filtered value used for display/computation; row.paAll
        // is what qualification logic (Q-filter, percentile coloring) reads.
        if (hPre.pa !== undefined) rows[hmi].paAll = hPre.pa;
        // Override PA/AB with boxscore values only when unfiltered
        if (!hasHitterContextFilter) {
          if (hPre.pa !== undefined) rows[hmi].pa = hPre.pa;
          if (hPre.ab !== undefined) rows[hmi].ab = hPre.ab;
        }
      }
    }

    // Multi-team handling: mark per-team rows of hitters who have a 2TM row
    // as non-pool, so percentile pool uses the combined row as the representative.
    const combinedByHitter = {};
    for (let hci = 0; hci < rows.length; hci++) {
      if (Aggregator._isCombinedTeam(rows[hci].team)) {
        combinedByHitter[Aggregator._combinedKey(rows[hci])] = rows[hci];
      }
    }
    for (let hpi = 0; hpi < rows.length; hpi++) {
      const hr = rows[hpi];
      hr._inPool = !(combinedByHitter[Aggregator._combinedKey(hr)] && !Aggregator._isCombinedTeam(hr.team));
    }

    // Mark each hitter row qualified for percentile-pool inclusion.
    // Mirrors the display-filter logic at the bottom of this function: a row
    // qualifies when overall-season PA >= 3.1 × team games played. Multi-team
    // players are evaluated on their combined 2TM/3TM row's PA against the
    // largest single-team games count among the teams they played for.
    const hitterTg = this.getTeamGamesPlayed();
    const _combinedRowByHitter = {};
    for (let cbi = 0; cbi < rows.length; cbi++) {
      if (Aggregator._isCombinedTeam(rows[cbi].team)) {
        _combinedRowByHitter[Aggregator._combinedKey(rows[cbi])] = rows[cbi];
      }
    }
    const _cumTg = {};
    for (let cti = 0; cti < rows.length; cti++) {
      const cr = rows[cti];
      const crKey = Aggregator._combinedKey(cr);
      if (_combinedRowByHitter[crKey] && !Aggregator._isCombinedTeam(cr.team)) {
        const tgv = hitterTg[cr.team] || 0;
        if (tgv > (_cumTg[crKey] || 0)) _cumTg[crKey] = tgv;
      }
    }
    for (let qi = 0; qi < rows.length; qi++) {
      const r = rows[qi];
      const rKey = Aggregator._combinedKey(r);
      const mt = _combinedRowByHitter[rKey];
      let _tg, _pa;
      if (mt) {
        _tg = _cumTg[rKey] || 0;
        _pa = (mt.paAll != null ? mt.paAll : mt.pa) || 0;
      } else {
        _tg = hitterTg[r.team] || 0;
        _pa = (r.paAll != null ? r.paAll : r.pa) || 0;
      }
      r._qualified = _tg > 0 && _pa >= _tg * Utils.hitterPaPerGame(Aggregator._isROCTeam(r.team));
    }

    // Compute percentiles. Pool: ALL MLB hitters (no qualifier) — matches
    // the displayed-league-avg pool so "below league avg → above 50th
    // pctl" reads correctly. Qualification is enforced as a render-only
    // gate (non-qualified rows store a rank for tooltip but get no
    // coloring). Counting stats (hr, sb) were already using the full
    // pool — unchanged.
    const self = this;
    HITTER_STAT_KEYS.forEach(function (key) {
      self._computePercentiles(rows, key);
    });

    // Set bipQual flag for each hitter
    for (let bqi = 0; bqi < rows.length; bqi++) {
      rows[bqi].bipQual = (rows[bqi].nBip || 0) >= QUAL.MIN_BIP_PCTL;
    }

    // Invert where lower is better
    for (let ri2 = 0; ri2 < rows.length; ri2++) {
      for (let inv2 in HITTER_INVERT) {
        const pk2 = inv2 + '_pctl';
        if (rows[ri2][pk2] !== null && rows[ri2][pk2] !== undefined) {
          rows[ri2][pk2] = 100 - rows[ri2][pk2];
        }
      }
    }

    // Apply min PA qualified filter AFTER percentiles.
    // Qualification always uses overall-season PA (paAll), not the filtered row.pa,
    // so that platoon/date filters don't drop overall-qualified hitters from the board.
    // Multi-team hitters qualify on cumulative PA against the sum of team games.
    if (filters.minCount === 'Q') {
      const tgHitter = this.getTeamGamesPlayed();
      const cumTgH = {};
      for (let ht = 0; ht < rows.length; ht++) {
        const hrr = rows[ht];
        const hrrKey = Aggregator._combinedKey(hrr);
        if (combinedByHitter[hrrKey] && !Aggregator._isCombinedTeam(hrr.team)) {
          const tgv = tgHitter[hrr.team] || 0;
          if (tgv > (cumTgH[hrrKey] || 0)) cumTgH[hrrKey] = tgv;
        }
      }
      rows = rows.filter(function (r) {
        const rKey = Aggregator._combinedKey(r);
        const mt = combinedByHitter[rKey];
        const isROC = Aggregator._isROCTeam(r.team);
        if (mt) {
          const tg = cumTgH[rKey] || 0;
          const pa = (mt.paAll != null ? mt.paAll : mt.pa) || 0;
          return pa >= tg * Utils.hitterPaPerGame(isROC);
        }
        var tg = tgHitter[r.team] || 0;
        var pa = (r.paAll != null ? r.paAll : r.pa) || 0;
        return pa >= tg * Utils.hitterPaPerGame(isROC);
      });
    }

    // Apply view-narrowing filters AFTER percentiles (don't change comparison group)
    const self3 = this;
    if (filters.team !== 'all') {
      rows = rows.filter(function (r) { return r.team === filters.team; });
    } else {
      rows = rows.filter(function (r) {
        // includeROC: player-page platoon aggregations keep ROC rows so ROC
        // hitter pages can split by hand (their percentiles are already
        // interpolated vs the MLB pool above). Leaderboard views never set
        // it, so the All-Teams board still hides ROC.
        if (self3._isROCTeam(r.team)) return !!filters.includeROC;
        // keepStints: see the pitcher narrowing — player-page platoon only.
        if (combinedByHitter[Aggregator._combinedKey(r)] && !Aggregator._isCombinedTeam(r.team)) return !!filters.keepStints;
        return true;
      });
    }
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      rows = rows.filter(function (r) { return r.hitter.toLowerCase().indexOf(searchLower) !== -1; });
    }

    return rows;
  },

  // Category definitions for hitter pitch type grouping
  PITCH_CATEGORIES: {
    'Hard': ['FF', 'SI'],
    'Breaking': ['FC', 'SL', 'ST', 'CU', 'SV'],
    'Offspeed': ['CH', 'FS', 'KN']
  },

  _aggregateHitterPitch: function (filters) {
    const d = this.data;
    const ci = this._colIdx.hitterPitchCols;
    const bci = this._colIdx.hitterPitchBipCols;
    const micro = d.hitterPitchMicro;
    const hpMlbIdMap = this._getMlbIdMap('hitter');
    const bipData = d.hitterPitchBip;
    const lookups = d.lookups;
    const validDates = this._getValidDateSet(filters);
    const vsHand = filters.vsHand || 'all';
    const teamMode = filters.viewMode === 'team';

    if (!micro || !ci) return [];

    // Team mode: bats filter resolved per hitter (switch hitters excluded from
    // R/L views, matching player mode); combined 2TM/3TM rows skipped.
    const standsMapHP = teamMode ? this._buildHitterStandsMap(micro, ci) : null;
    const selfHP = this;
    const teamRowOkHP = function (hitterIdx, teamIdx) {
      if (selfHP._isCombinedTeam(lookups.teams[teamIdx])) return false;
      if (filters.throws !== 'all' && standsMapHP[hitterIdx + '|' + teamIdx] !== filters.throws) return false;
      return true;
    };

    const selectedPitchTypes = filters.pitchTypes; // always array (e.g. ['all'], ['FF','SL'])
    const CATS = this.PITCH_CATEGORIES;

    // Build reverse lookup: pitch type name -> set of pitchTypeIdx values
    const ptNameToIdx = {};
    for (let pi = 0; pi < lookups.pitchTypes.length; pi++) {
      ptNameToIdx[lookups.pitchTypes[pi]] = pi;
    }

    // Build category -> set of pitchTypeIdx
    const catIdxSets = {};
    for (let catName in CATS) {
      catIdxSets[catName] = {};
      for (let ci2 = 0; ci2 < CATS[catName].length; ci2++) {
        const idx = ptNameToIdx[CATS[catName][ci2]];
        if (idx !== undefined) catIdxSets[catName][idx] = true;
      }
    }

    // Determine which output groups we need
    // Each selected chip becomes an output group with its own grouping logic
    const outputGroups = []; // { name, type: 'all'|'category'|'individual', idxSet }
    if (selectedPitchTypes.length === 1 && selectedPitchTypes[0] === 'all') {
      // Default: show all individual pitch types
      outputGroups.push({ name: 'all_individual', type: 'all_individual' });
    } else {
      for (let si = 0; si < selectedPitchTypes.length; si++) {
        const sel = selectedPitchTypes[si];
        if (sel === 'All') {
          outputGroups.push({ name: 'All', type: 'all' });
        } else if (CATS[sel]) {
          outputGroups.push({ name: sel, type: 'category', idxSet: catIdxSets[sel] });
        } else {
          outputGroups.push({ name: sel, type: 'individual', idx: ptNameToIdx[sel] });
        }
      }
    }

    // First pass: accumulate per-hitter-team-pitchType micro rows
    // and track hitter totals
    const perPT = {}; // hitterIdx|teamIdx|pitchTypeIdx -> { counts, batsSet }
    const hitterTotals = {};

    for (let i = 0; i < micro.length; i++) {
      const row = micro[i];
      if (!validDates[row[ci.dateIdx]]) continue;
      if (vsHand !== 'all' && row[ci.pitcherHand] !== vsHand) continue;
      if (teamMode && !teamRowOkHP(row[ci.hitterIdx], row[ci.teamIdx])) continue;

      const hIdx = teamMode ? null : row[ci.hitterIdx];
      const hk = hIdx + '|' + row[ci.teamIdx];
      if (!hitterTotals[hk]) hitterTotals[hk] = { total: 0, batsSet: {} };
      hitterTotals[hk].total += row[6];
      hitterTotals[hk].batsSet[row[ci.bats]] = true;

      const gk = hIdx + '|' + row[ci.teamIdx] + '|' + row[ci.pitchTypeIdx];
      if (!perPT[gk]) {
        perPT[gk] = {
          hitterIdx: hIdx,
          teamIdx: row[ci.teamIdx],
          pitchTypeIdx: row[ci.pitchTypeIdx],
          counts: new Array(49)
        };
        for (let z = 0; z < 49; z++) perPT[gk].counts[z] = 0;
      }
      for (let f = 0; f < 49; f++) {
        perPT[gk].counts[f] += row[6 + f];
      }
    }

    // BIP records by hitter+pitchTypeIdx
    const bipByKey = {};
    if (bipData && bci) {
      for (let bi = 0; bi < bipData.length; bi++) {
        const brow = bipData[bi];
        if (!validDates[brow[bci.dateIdx]]) continue;
        if (vsHand !== 'all' && brow[bci.pitcherHand] !== vsHand) continue;
        if (teamMode && !teamRowOkHP(brow[bci.hitterIdx], brow[bci.teamIdx])) continue;

        const bipKey = (teamMode ? null : brow[bci.hitterIdx]) + '|' + brow[bci.teamIdx] + '|' + brow[bci.pitchTypeIdx];
        if (!bipByKey[bipKey]) bipByKey[bipKey] = [];
        bipByKey[bipKey].push(brow);
      }
    }

    function median(arr) {
      if (arr.length === 0) return null;
      arr.sort(function (a, b) { return a - b; });
      const mid = Math.floor(arr.length / 2);
      return arr.length % 2 === 1 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;
    }

    // Second pass: build output rows per output group
    // For 'all' and 'category', we combine multiple perPT entries per hitter
    const groups = {}; // outputGroupName|hitterIdx|teamIdx -> combined counts + bipRecords

    for (let gk2 in perPT) {
      const entry = perPT[gk2];
      const ptIdx = entry.pitchTypeIdx;
      const ptName = lookups.pitchTypes[ptIdx];

      for (let oi = 0; oi < outputGroups.length; oi++) {
        const og = outputGroups[oi];
        let match = false;

        if (og.type === 'all') {
          match = true;
        } else if (og.type === 'category') {
          match = !!og.idxSet[ptIdx];
        } else if (og.type === 'individual') {
          match = (ptIdx === og.idx);
        } else if (og.type === 'all_individual') {
          // For 'all_individual', each pitch type becomes its own output row
          // handled differently below
          match = false;
        }

        if (match) {
          const outKey = og.name + '|' + entry.hitterIdx + '|' + entry.teamIdx;
          if (!groups[outKey]) {
            groups[outKey] = {
              hitterIdx: entry.hitterIdx,
              teamIdx: entry.teamIdx,
              outputName: og.name,
              counts: new Array(49),
              bipPtIdxs: []
            };
            for (let z2 = 0; z2 < 49; z2++) groups[outKey].counts[z2] = 0;
          }
          const gg = groups[outKey];
          for (let f2 = 0; f2 < 49; f2++) {
            gg.counts[f2] += entry.counts[f2];
          }
          if (gg.bipPtIdxs.indexOf(ptIdx) === -1) gg.bipPtIdxs.push(ptIdx);
        }
      }

      // Handle all_individual: each perPT entry is its own output row
      if (outputGroups.length === 1 && outputGroups[0].type === 'all_individual') {
        const outKey2 = ptName + '|' + entry.hitterIdx + '|' + entry.teamIdx;
        if (!groups[outKey2]) {
          groups[outKey2] = {
            hitterIdx: entry.hitterIdx,
            teamIdx: entry.teamIdx,
            outputName: ptName,
            counts: new Array(49),
            bipPtIdxs: [ptIdx]
          };
          for (let z3 = 0; z3 < 49; z3++) groups[outKey2].counts[z3] = 0;
        }
        for (let f3 = 0; f3 < 49; f3++) {
          groups[outKey2].counts[f3] += entry.counts[f3];
        }
      }
    }

    const HITTER_PITCH_PCTL_KEYS = [
      'avg', 'slg', 'iso', 'wOBA',
      'xBA', 'xSLG', 'xwOBA', 'xwOBAcon', 'xwOBAsp',
      'ev50', 'maxEV', 'hardHitPct', 'barrelPct', 'babip',
      'gbPct', 'ldPct', 'fbPct', 'puPct', 'hrFbPct',
      'pullPct', 'oppoPct', 'airPullPct',
      'swingPct', 'izSwingPct', 'chasePct', 'izSwChase', 'firstPitchSwingPct',
      'contactPct', 'izContactPct', 'whiffPct', 'twoStrikeWhiffPct',
      'runValue', 'rv100', 'xRunValue', 'xRv100',
    ];
    // babip/puPct/oppoPct added 2026-08-06 — mirrors process_data
    // HITTER_PITCH_PCTL_KEYS / HITTER_PITCH_INVERT_PCTL (batted-ball coloring:
    // BABIP/LD/FB higher-is-better, PU/Oppo lower-is-better).
    const HITTER_PITCH_INVERT = {
      chasePct: true, whiffPct: true, gbPct: true, twoStrikeWhiffPct: true,
      puPct: true, oppoPct: true   // swingPct dropped 2026-08-12, matches server
    };

    let rows = [];

    for (let gk3 in groups) {
      const gg2 = groups[gk3];
      const c = gg2.counts;
      const hk2 = gg2.hitterIdx + '|' + gg2.teamIdx;
      const ht = hitterTotals[hk2];
      const hTotal = ht ? ht.total : 1;
      const batsKeys = ht ? Object.keys(ht.batsSet) : [];
      const stands = batsKeys.length > 1 ? 'S' : (batsKeys[0] || null);

      const n_total = c[0], pa = c[1], h = c[2], db = c[3], tp = c[4], hr = c[5];
      const bb = c[6], hbp = c[7], sf = c[8], sh = c[9], ci_v = c[10], k = c[11];
      const swings = c[12], whiffs = c[13];
      const izPitches = c[14], oozPitches = c[15];
      const izSwings = c[16], oozSwings = c[17], contact = c[18];
      const izSwNonBunt = c[19], izContact = c[20];
      const bip = c[21], gb_c = c[22], ld = c[23], fb = c[24], pu = c[25];
      const barrels = c[26], nSpray = c[27], pull = c[28], center = c[29], oppo = c[30], airPull = c[31];
      const hardHit = c[32], nHrBip = c[33], ldHr = c[34];
      const twoStrikeSwings = c[35], twoStrikeWhiffs = c[36];
      const firstPitchAppearances = c[37], firstPitchSwings = c[38];
      const xBA_sum = c[39], xBA_count = c[40], xSLG_sum = c[41], xSLG_count = c[42];
      const xwOBA_sum = c[43], xwOBA_count = c[44], xwOBAcon_sum = c[45], xwOBAcon_count = c[46];
      const swingsNonBunt = c[47], contactNonBunt = c[48], buntAB = c[49] || 0;

      const ab = pa - bb - hbp - sf - sh - ci_v;
      const nonbuntAB = ab - buntAB;  // xBA/xSLG denominator (Savant excludes bunts)
      const singles = h - db - tp - hr;
      const tb_val = singles + 2 * db + 3 * tp + 4 * hr;

      const batting_avg = ab > 0 ? Math.round(h / ab * 1000) / 1000 : null;
      const slg_val = ab > 0 ? Math.round(tb_val / ab * 1000) / 1000 : null;
      const iso_val = (slg_val !== null && batting_avg !== null) ? Math.round((slg_val - batting_avg) * 1000) / 1000 : null;
      const babip_denom3 = ab - k - hr + sf;
      const babip_val3 = babip_denom3 > 0 ? Math.round((h - hr) / babip_denom3 * 1000) / 1000 : null;

      const izSwingPct = izPitches > 0 ? izSwings / izPitches : null;
      const chasePct_val = oozPitches > 0 ? oozSwings / oozPitches : null;
      const contactPct = swingsNonBunt > 0 ? contactNonBunt / swingsNonBunt : null;
      const izContactPct = izSwNonBunt > 0 ? izContact / izSwNonBunt : null;
      // strict-conditional HR/FB — mirrors pipeline_compute
      const hrFbPct_val = fb > 0 ? Math.max(nHrBip - ldHr, 0) / fb : null;

      // BIP medians — combine BIP records from all pitch types in this group
      const evsAll2 = [], allLA = [];
      for (let bpi = 0; bpi < gg2.bipPtIdxs.length; bpi++) {
        const bpKey = gg2.hitterIdx + '|' + gg2.teamIdx + '|' + gg2.bipPtIdxs[bpi];
        const bipRecords = bipByKey[bpKey] || [];
        for (let bri = 0; bri < bipRecords.length; bri++) {
          const bev = bipRecords[bri][bci.exitVelo];
          const bla = bipRecords[bri][bci.launchAngle];
          if (bev !== null) evsAll2.push(bev);
          if (bla !== null) allLA.push(bla);
        }
      }

      const avgEVAll2 = evsAll2.length > 0 ? Math.round(evsAll2.reduce(function(a,b){return a+b;},0) / evsAll2.length * 10) / 10 : null;
      const maxEV = evsAll2.length > 0 ? Math.round(Math.max.apply(null, evsAll2) * 10) / 10 : null;
      const medLA = allLA.length > 0 ? Math.round(median(allLA.slice()) * 10) / 10 : null;

      // EV50: average of top 50% hardest-hit balls across ALL BIP (no LA filter)
      let ev50 = null;
      if (evsAll2.length > 0) {
        const sorted = evsAll2.slice().sort(function (a, b) { return b - a; });
        const topHalf = sorted.slice(0, Math.max(1, Math.floor(sorted.length / 2)));
        ev50 = Math.round(topHalf.reduce(function (s, v) { return s + v; }, 0) / topHalf.length * 10) / 10;
      }

      // hardHitPct and barrelPct: use EV-valid count as denominator (not total BIP)
      const hardHitPct = evsAll2.length > 0 ? hardHit / evsAll2.length : null;

      const hpName = gg2.hitterIdx != null ? lookups.hitters[gg2.hitterIdx] : null;
      const hpTeam = lookups.teams[gg2.teamIdx];
      const obj = {
        hitter: hpName,
        team: hpTeam,
        mlbId: hpName != null ? (hpMlbIdMap[hpName + '|' + hpTeam] || null) : null,
        stands: teamMode ? null : stands,
        pitchType: gg2.outputName,
        count: n_total,
        seenPct: Math.round(n_total / hTotal * 10000) / 10000,
        pa: pa,
        nSwings: swings,
        nBip: bip,
        avg: batting_avg,
        slg: slg_val,
        iso: iso_val,
        babip: babip_val3,
        avgEVAll: avgEVAll2,
        ev50: ev50,
        maxEV: maxEV,
        medLA: medLA,
        hardHitPct: hardHitPct,
        barrelPct: evsAll2.length > 0 ? barrels / evsAll2.length : null,
        gbPct: bip > 0 ? gb_c / bip : null,
        ldPct: bip > 0 ? ld / bip : null,
        fbPct: bip > 0 ? fb / bip : null,
        hrFbPct: hrFbPct_val,
        puPct: bip > 0 ? pu / bip : null,
        pullPct: nSpray > 0 ? pull / nSpray : null,
        middlePct: nSpray > 0 ? center / nSpray : null,
        oppoPct: nSpray > 0 ? oppo / nSpray : null,
        airPullPct: bip > 0 ? airPull / bip : null,
        swingPct: n_total > 0 ? swings / n_total : null,
        izSwingPct: izSwingPct,
        chasePct: chasePct_val,
        contactPct: contactPct,
        izContactPct: izContactPct,
        whiffPct: swingsNonBunt > 0 ? whiffs / swingsNonBunt : null,
        xBA: nonbuntAB > 0 && xBA_count > 0 ? xBA_sum / nonbuntAB : null,
        xSLG: nonbuntAB > 0 && xSLG_count > 0 ? xSLG_sum / nonbuntAB : null,
        xwOBA: xwOBA_count > 0 ? xwOBA_sum / xwOBA_count : null,
        xwOBAcon: xwOBAcon_count > 0 ? xwOBAcon_sum / xwOBAcon_count : null,
        twoStrikeWhiffPct: twoStrikeSwings > 0 ? twoStrikeWhiffs / twoStrikeSwings : null,
        firstPitchSwingPct: firstPitchAppearances > 0 ? firstPitchSwings / firstPitchAppearances : null,
        izSwChase: (izSwingPct !== null && chasePct_val !== null) ? Math.round((izSwingPct - chasePct_val) * 10000) / 10000 : null,
      };

      if (teamMode) {
        // Bats filter applied per hitter pre-rollup; player-level minimum
        // sample filters are intentionally ignored for team totals, except a
        // pitch-count floor matching player mode's qualified default — a team
        // row built on a handful of pitches of a rare type is noise.
        obj._isTeamRow = true;
        if (obj.count < QUAL.MIN_HITTER_PT) continue;
        rows.push(obj);
        continue;
      }

      // Apply baseball-context filters (comparison group — affects percentiles)
      if (filters.throws !== 'all' && obj.stands !== filters.throws) continue;
      // Min sample is PA, not pitches (2026-08-06, per Wally): the board's
      // outcome stats are PA-denominated, so a pitch-count floor was the
      // wrong unit. Only applied above 1 so callers that pass no minimum
      // (the player-page platoon aggregation) keep pa=0 rows, whose
      // swing/whiff columns are still real data.
      if ((filters.minCount || 1) > 1 && (obj.pa || 0) < filters.minCount) continue;
      if (filters.minSwings && (obj.nSwings || 0) < filters.minSwings) continue;
      if (filters.minBip && (obj.nBip || 0) < filters.minBip) continue;

      rows.push(obj);
    }

    if (teamMode) {
      this._mergeTeamHitterPitchPreAgg(rows, filters);
      this._flagTeamRows(rows);
    } else {
    // Merge expected stats from pre-aggregated HITTER_PITCH_LB
    const hpPreAgg = window.HITTER_PITCH_LB || [];
    const hpPreMap = {};
    for (let hpi = 0; hpi < hpPreAgg.length; hpi++) {
      const hpk = hpPreAgg[hpi].hitter + '|' + hpPreAgg[hpi].team + '|' + hpPreAgg[hpi].pitchType;
      hpPreMap[hpk] = hpPreAgg[hpi];
    }
    const hpXKeys = ['wOBA', 'runValue', 'rv100', 'xRunValue', 'xRv100', 'avgFbDist', 'avgHrDist', 'xwOBAsp'];
    for (let hpmi = 0; hpmi < rows.length; hpmi++) {
      const hpmk = rows[hpmi].hitter + '|' + rows[hpmi].team + '|' + rows[hpmi].pitchType;
      const hpPre = hpPreMap[hpmk];
      if (hpPre) {
        for (let hpxi = 0; hpxi < hpXKeys.length; hpxi++) {
          const hpxk = hpXKeys[hpxi];
          if (hpPre[hpxk] !== undefined) rows[hpmi][hpxk] = hpPre[hpxk];
          if (hpPre[hpxk + '_pctl'] !== undefined) rows[hpmi][hpxk + '_pctl'] = hpPre[hpxk + '_pctl'];
        }
      }
      // Compute rv100 client-side if not in pre-agg data. Kept at full precision;
      // display rounding (1 decimal) happens at render time via Utils.formatDecimal.
      if (rows[hpmi].runValue != null && rows[hpmi].rv100 == null && rows[hpmi].count > 0) {
        rows[hpmi].rv100 = rows[hpmi].runValue / rows[hpmi].count * 100;
      }
    }
    } // end player-mode merge

    // Multi-team pool handling: per-team rows of hitters with a 2TM row are
    // marked non-pool so the combined row is the pool representative.
    const combinedByHitterPT = {};
    if (!teamMode) {
    for (let hci = 0; hci < rows.length; hci++) {
      if (Aggregator._isCombinedTeam(rows[hci].team)) {
        combinedByHitterPT[Aggregator._combinedKey(rows[hci])] = true;
      }
    }
    for (let hpi2 = 0; hpi2 < rows.length; hpi2++) {
      const rp = rows[hpi2];
      rp._inPool = !(combinedByHitterPT[Aggregator._combinedKey(rp)] && !Aggregator._isCombinedTeam(rp.team));
    }
    }

    // Compute percentiles per pitch type (output group name)
    const ptGroups = {};
    for (let ri = 0; ri < rows.length; ri++) {
      const pt = rows[ri].pitchType;
      if (!ptGroups[pt]) ptGroups[pt] = [];
      ptGroups[pt].push(rows[ri]);
    }

    const self = this;
    for (let ptKey in ptGroups) {
      const ptRows = ptGroups[ptKey];
      HITTER_PITCH_PCTL_KEYS.forEach(function (key) {
        // Pool: rows with at least MIN_HITTER_PT pitches of this type seen.
        // Sub-minimum rows are interpolated automatically (minCount > 0 path).
        self._computePercentiles(ptRows, key, QUAL.MIN_HITTER_PT);
      });
    }

    // Invert where lower is better
    for (let ri2 = 0; ri2 < rows.length; ri2++) {
      for (let inv in HITTER_PITCH_INVERT) {
        const pk = inv + '_pctl';
        if (rows[ri2][pk] !== null && rows[ri2][pk] !== undefined) {
          rows[ri2][pk] = 100 - rows[ri2][pk];
        }
      }
    }

    if (teamMode) {
      return this._narrowTeamRows(rows, filters);
    }

    // Apply view-narrowing filters AFTER percentiles (don't change comparison group).
    // All Teams: drop ROC + per-team rows of multi-team hitters. Specific team:
    // drop combined rows and keep only the selected team's per-team rows.
    const self4 = this;
    if (filters.team !== 'all') {
      rows = rows.filter(function (r) { return r.team === filters.team; });
    } else {
      rows = rows.filter(function (r) {
        // includeROC/keepStints: see the hitter narrowing above — player-page
        // platoon aggregations only.
        if (self4._isROCTeam(r.team)) return !!filters.includeROC;
        if (combinedByHitterPT[Aggregator._combinedKey(r)] && !Aggregator._isCombinedTeam(r.team)) return !!filters.keepStints;
        return true;
      });
    }
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      rows = rows.filter(function (r) { return r.hitter.toLowerCase().indexOf(searchLower) !== -1; });
    }

    return rows;
  },

  /**
   * Team mode: aggregate pre-computed hitter-vs-pitch-type stats from
   * HITTER_PITCH_LB to (team, pitchType). RV totals are true sums at full
   * precision; wOBA weights by PA, distance/spray-adjusted stats by BIP.
   * Category/All output groups have no pre-agg rows (same as player mode) —
   * only individual pitch types merge.
   */
  _mergeTeamHitterPitchPreAgg: function (rows, filters) {
    const hp = window.HITTER_PITCH_LB || [];
    // Hitter stands per (hitter, team) for the bats filter
    const standsByHitter = {};
    const hd = window.HITTER_DATA || [];
    for (let hi = 0; hi < hd.length; hi++) {
      standsByHitter[hd[hi].hitter + '|' + hd[hi].team] = hd[hi].stands;
    }
    const acc = {};

    function wadd(a, key, val, w) {
      if (val == null || !(w > 0)) return;
      a.sums[key] = (a.sums[key] || 0) + val * w;
      a.wts[key] = (a.wts[key] || 0) + w;
    }

    for (let i = 0; i < hp.length; i++) {
      const p = hp[i];
      if (this._isCombinedTeam(p.team)) continue;
      if (filters.throws !== 'all' &&
          standsByHitter[p.hitter + '|' + p.team] !== filters.throws) continue;

      const k = p.team + '|' + p.pitchType;
      let a = acc[k];
      if (!a) a = acc[k] = { runValue: null, xRunValue: null, count: 0, sums: {}, wts: {} };
      a.count += p.count || 0;
      // RV sums stay full precision; rounding happens only at display
      if (p.runValue != null) a.runValue = (a.runValue || 0) + p.runValue;
      if (p.xRunValue != null) a.xRunValue = (a.xRunValue || 0) + p.xRunValue;
      wadd(a, 'wOBA', p.wOBA, p.pa);
      wadd(a, 'xwOBAsp', p.xwOBAsp, p.nBip);
      wadd(a, 'avgFbDist', p.avgFbDist, p.nBip);
      wadd(a, 'avgHrDist', p.avgHrDist, p.nBip);
    }

    for (let ri = 0; ri < rows.length; ri++) {
      const r = rows[ri];
      const a = acc[r.team + '|' + r.pitchType];
      if (!a) continue;
      r.runValue = a.runValue;
      r.xRunValue = a.xRunValue;
      r.rv100 = (a.runValue != null && a.count > 0) ? a.runValue / a.count * 100 : null;
      r.xRv100 = (a.xRunValue != null && a.count > 0) ? a.xRunValue / a.count * 100 : null;
      for (const sk in a.sums) {
        if (a.wts[sk] > 0) r[sk] = a.sums[sk] / a.wts[sk];
      }
    }
  },

  //  Team games played (distinct game dates per team)
  /**
   * Get velocity trend data for a pitcher, grouped by pitch type.
   * Returns { pitchType: [{ date: 'YYYY-MM-DD', avgVelo: N }, ...], ... }
   */
  getVeloTrend: function(pitcherName, teamName) {
    const d = this.data;
    if (!d || !d.veloTrend) return {};
    const lookups = d.lookups;
    const ci = this._colIdx.veloTrendCols;
    const piIdx = lookups.pitchers.indexOf(pitcherName);
    if (piIdx < 0) return {};
    const tiIdx = teamName ? lookups.teams.indexOf(teamName) : -1;

    const result = {};
    const rows = d.veloTrend;
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      if (r[ci.pitcherIdx] !== piIdx) continue;
      if (tiIdx >= 0 && r[ci.teamIdx] !== tiIdx) continue;
      const pt = lookups.pitchTypes[r[ci.pitchTypeIdx]];
      const date = lookups.dates[r[ci.dateIdx]];
      const sumV = r[ci.sumVelo];
      const nV = r[ci.nVelo];
      if (nV <= 0) continue;
      if (!result[pt]) result[pt] = [];
      result[pt].push({ date: date, avgVelo: Math.round(sumV / nV * 10) / 10 });
    }
    // Sort each pitch type's entries by date
    for (let pt2 in result) {
      result[pt2].sort(function(a, b) { return a.date < b.date ? -1 : 1; });
    }
    return result;
  },

  /**
   * Per-game Stuff+/Loc+/Pitching+ for one pitcher — plain averages of the
   * per-pitch grade atoms in pitchMicro, summed across pitch types and
   * batter hands per date. The same integers as the Sheets grade columns,
   * so each row matches that outing's window card / sheet AVERAGEIF.
   * Returns [{date, n, stuffScore, locPlus, pitchingScore}], newest first;
   * [] when the embed predates grade atoms.
   */
  getPitcherGameGrades: function (pitcher, team) {
    const d = this.data;
    if (!d) return [];
    const ci = this._colIdx.pitchCols;
    if (ci.sumStuff === undefined) return [];
    const lookups = d.lookups;
    const pIdx = lookups.pitchers.indexOf(pitcher);
    const tIdx = lookups.teams.indexOf(team);
    if (pIdx < 0 || tIdx < 0) return [];
    const micro = d.pitchMicro;
    const byDate = {};
    for (let i = 0; i < micro.length; i++) {
      const row = micro[i];
      if (row[ci.pitcherIdx] !== pIdx || row[ci.teamIdx] !== tIdx) continue;
      const di = row[ci.dateIdx];
      let a = byDate[di];
      if (!a) a = byDate[di] = { n: 0, sS: 0, nS: 0, sL: 0, nL: 0, sP: 0, nP: 0 };
      a.n += row[ci.n];
      a.sS += row[ci.sumStuff] || 0; a.nS += row[ci.nStuff] || 0;
      a.sL += row[ci.sumLoc] || 0;  a.nL += row[ci.nLoc] || 0;
      a.sP += row[ci.sumPitching] || 0; a.nP += row[ci.nPitching] || 0;
    }
    const out = [];
    for (const di2 in byDate) {
      const a2 = byDate[di2];
      out.push({
        date: lookups.dates[di2],
        n: a2.n,
        stuffScore: a2.nS > 0 ? Number((a2.sS / a2.nS).toFixed(1)) : null,
        locPlus: a2.nL > 0 ? Number((a2.sL / a2.nL).toFixed(1)) : null,
        pitchingScore: a2.nP > 0 ? Number((a2.sP / a2.nP).toFixed(1)) : null,
      });
    }
    out.sort(function (x, y) { return x.date < y.date ? 1 : -1; });
    return out;
  },

  getTeamGamesPlayed: function(dateStart, dateEnd) {
    const d = this.data;
    if (!d) return {};
    const dates = d.lookups.dates;
    const teams = d.lookups.teams;
    const ci = this._colIdx.pitcherCols;
    const micro = d.pitcherMicro;

    // Build valid date set based on optional range
    const validDates = {};
    for (let di = 0; di < dates.length; di++) {
      const dt = dates[di];
      if (dateStart && dt < dateStart) continue;
      if (dateEnd && dt > dateEnd) continue;
      validDates[di] = true;
    }

    const teamDates = {};  // teamIdx -> { dateIdx: true }
    for (let i = 0; i < micro.length; i++) {
      const row = micro[i];
      const ti = row[ci.teamIdx];
      const dIdx = row[ci.dateIdx];
      if (!validDates[dIdx]) continue;
      if (!teamDates[ti]) teamDates[ti] = {};
      teamDates[ti][dIdx] = true;
    }

    const result = {};
    for (let ti2 in teamDates) {
      result[teams[ti2]] = Object.keys(teamDates[ti2]).length;
    }
    return result;
  },
};
