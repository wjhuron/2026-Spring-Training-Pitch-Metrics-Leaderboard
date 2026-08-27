// ============================================================================
//  JSDoc Type Definitions -- shared data shapes used across the codebase
// ============================================================================

/**
 * A column definition used in leaderboard tables and player-page tables.
 * @typedef {Object} ColumnDef
 * @property {string}   key          - Data-row property name (e.g. 'velocity', 'kPct').
 * @property {string}   label        - Display header text.
 * @property {function} [format]     - Formatter: (value) => string. Omit for pitch-type badge cols.
 * @property {string}   [sortType]   - 'numeric' | 'string' | null (unsortable).
 * @property {string}   [sortKey]    - Alternate row property to sort on (defaults to key).
 * @property {boolean}  [sectionStart] - If true, renders a left-border divider before this column.
 * @property {boolean}  [noPercentile] - If true, skip percentile coloring for this column.
 * @property {string}   [desc]       - Tooltip description for header hover.
 * @property {string}   [group]      - Column group label (e.g. 'metrics', 'ev', 'discipline').
 * @property {boolean}  [rocHide]    - If true, hide this column for ROC/AAA players.
 * @property {boolean}  [noPctl]     - Player-page: skip percentile circle for this stat.
 */

/**
 * A row from the pitcher leaderboard (PITCHER_DATA / Aggregator output).
 * @typedef {Object} PitcherRow
 * @property {string}  pitcher     - Pitcher full name.
 * @property {string}  team        - Team abbreviation (e.g. 'NYY').
 * @property {number}  [mlbId]     - MLB player ID (for linking to player page).
 * @property {string}  throws      - Throwing hand: 'R' or 'L'.
 * @property {number}  count       - Total pitches thrown.
 * @property {number}  [pa]        - Plate appearances (total batters faced).
 * @property {number}  [nSwings]   - Swing count.
 * @property {number}  [nBip]      - Batted balls in play.
 * @property {number}  [g]         - Games.
 * @property {number}  [gs]        - Games started.
 * @property {string}  [ip]        - Innings pitched (baseball notation, e.g. '6.2').
 * @property {number}  [kPct]      - Strikeout rate (0-1).
 * @property {number}  [bbPct]     - Walk rate (0-1).
 * @property {number}  [era]       - Earned run average.
 * @property {number}  [fip]       - Fielding-independent pitching.
 * @property {number}  [xwOBA]     - Expected wOBA against.
 * @property {boolean} [_qualified] - True if IP-qualified for percentile pool.
 * @property {boolean} [bipQual]   - True if BIP >= MIN_BIP_PCTL for batted-ball percentiles.
 */

/**
 * A row from the pitch-type leaderboard (PITCH_DATA / Aggregator output).
 * @typedef {Object} PitchRow
 * @property {string}  pitcher     - Pitcher full name.
 * @property {string}  team        - Team abbreviation.
 * @property {string}  throws      - Throwing hand: 'R' or 'L'.
 * @property {string}  pitchType   - Pitch type code (e.g. 'FF', 'SL', 'CH').
 * @property {number}  count       - Number of pitches of this type.
 * @property {number}  [velocity]  - Average velocity (mph).
 * @property {number}  [spinRate]  - Average spin rate (rpm).
 * @property {number}  [indVertBrk]- Induced vertical break (inches).
 * @property {number}  [horzBrk]   - Horizontal break (inches).
 * @property {string}  [breakTilt] - Break tilt (clock notation, e.g. '1:30').
 * @property {number}  [nVAA]      - Normalized vertical approach angle (degrees).
 * @property {number}  [swStrPct]  - Whiff rate (0-1).
 * @property {number}  [cswPct]    - Called strike + whiff rate (0-1).
 */

/**
 * A row from the hitter leaderboard (HITTER_DATA / Aggregator output).
 * @typedef {Object} HitterRow
 * @property {string}  hitter      - Hitter full name.
 * @property {string}  team        - Team abbreviation.
 * @property {number}  [mlbId]     - MLB player ID.
 * @property {string}  stands      - Batting side: 'R', 'L', or 'S'.
 * @property {number}  [pa]        - Plate appearances.
 * @property {number}  [nSwings]   - Swing count.
 * @property {number}  [nBip]      - Batted balls in play.
 * @property {number}  [avg]       - Batting average.
 * @property {number}  [obp]       - On-base percentage.
 * @property {number}  [slg]       - Slugging percentage.
 * @property {number}  [xwOBA]     - Expected wOBA.
 * @property {number}  [avgEVAll]  - Average exit velocity (all BIP, mph).
 * @property {number}  [barrelPct] - Barrel rate (0-1).
 * @property {number}  [batSpeed]  - Average bat speed (mph).
 * @property {boolean} [_qualified]- True if PA-qualified for percentile pool.
 */

/**
 * Filter state object passed between app.js, data.js, and aggregator.js.
 * @typedef {Object} FilterState
 * @property {string}   team         - Team abbreviation or 'all'.
 * @property {string}   throws       - 'R', 'L', or 'all' (also used as stands filter for hitters).
 * @property {string}   [vsHand]     - Opponent hand filter: 'R', 'L', or 'all'.
 * @property {string}   [role]       - 'SP', 'RP', or 'all'.
 * @property {string[]} pitchTypes   - Array of pitch type codes, or ['all'].
 * @property {number|string} minCount - Minimum pitch/PA count, or 'Q' for qualified.
 * @property {number}   [minSwings]  - Minimum swing count (hitter tabs).
 * @property {number}   [minBip]     - Minimum BIP count.
 * @property {number|string} [minIp] - Minimum IP, or 'Q' for qualified (pitcher tabs).
 * @property {number}   [minTbf]     - Minimum total batters faced.
 * @property {string}   [dateStart]  - YYYY-MM-DD start date filter.
 * @property {string}   [dateEnd]    - YYYY-MM-DD end date filter.
 * @property {string}   [search]     - Name search substring.
 */

const Utils = {
  ordinal: function (n) {
    const s = ['th', 'st', 'nd', 'rd'];
    const v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  },

  formatPct: function (value, _sign) {
    // Convention: bare positives, "-" for negatives only. The sign argument is
    // accepted for back-compat but ignored — negatives include "-" naturally.
    if (value === null || value === undefined) return '--';
    return (value * 100).toFixed(1) + '%';
  },

  formatDecimal: function (places) {
    return function (value) {
      if (value === null || value === undefined) return '--';
      var s = Number(value).toFixed(places);
      // Baseball convention: 3-decimal rate stats display as ".XXX" (no leading 0).
      // 1- and 2-decimal stats (ERA, FIP, velo, etc.) keep the leading number.
      if (places === 3) {
        if (s.charAt(0) === '0' && s.charAt(1) === '.') return s.substring(1);
        if (s.charAt(0) === '-' && s.charAt(1) === '0' && s.charAt(2) === '.') return '-' + s.substring(2);
      }
      return s;
    };
  },

  formatSignedDecimal: function (places) {
    return function (value) {
      if (value === null || value === undefined) return '--';
      return (value > 0 ? '+' : '') + Number(value).toFixed(places);
    };
  },

  formatInt: function (value) {
    if (value === null || value === undefined) return '--';
    return Math.round(value).toString();
  },

  formatTilt: function (value) {
    if (!value) return '--';
    return value;
  },

  // Convert decimal feet to feet'inches format (e.g., 5.86 → "5'10", -1.8 → "-1'10")
  formatFeetInches: function (value) {
    if (value === null || value === undefined) return '--';
    const sign = value < 0 ? '-' : '';
    const abs = Math.abs(value);
    let ft = Math.floor(abs);
    let inches = Math.round((abs - ft) * 12);
    if (inches === 12) { ft++; inches = 0; }
    return sign + ft + "'" + inches + '"';
  },

  PITCH_TYPE_LABELS: {
    FF: 'Four-Seam', SI: 'Sinker', FC: 'Cutter', CH: 'Changeup',
    CU: 'Curveball', SL: 'Slider', ST: 'Sweeper', FS: 'Splitter',
    SV: 'Slurve', KN: 'Knuckleball', SC: 'Screwball', CS: 'Slow Curve',
  },

  pitchTypeLabel: function (code) {
    return this.PITCH_TYPE_LABELS[code] || code;
  },

  // Pitch type colors — single source of truth (used by scatter.js, leaderboard, player page)
  PITCH_COLORS: {
    FF: '#0072B2', SI: '#E0A81E', FC: '#8B5A2B', SL: '#D55E00',
    ST: '#56B4E9', SV: '#882255', CU: '#332288', CH: '#009E73',
    FS: '#CC79A7', KN: '#AAAAAA', SC: '#999999', CS: '#666666',
  },

  // Border colors for scatter chart markers (derived from PITCH_COLORS)
  PITCH_BORDER_COLORS: {
    FF: '#3366CC', SI: '#A87B12', FC: '#CC8400', SL: '#BBBBBB',
    ST: '#CC1076', SV: '#28A428', CU: '#B32626', CH: '#A352BE',
    FS: '#33B3A6', KN: '#888888', SC: '#777777', CS: '#4D4D4D',
  },

  getPitchColor: function (pt) {
    return this.PITCH_COLORS[pt] || '#999';
  },

  getPitchBorderColor: function (pt) {
    return this.PITCH_BORDER_COLORS[pt] || '#777';
  },

  // Canonical pitch type sort order — used everywhere pitch types are sorted
  PITCH_ORDER: ['FF','SI','FC','SL','ST','CU','SV','CH','FS','KN','SC','CS'],

  // Sort comparator for pitch type codes (for use with Array.sort)
  pitchTypeSortCompare: function (a, b) {
    var ai = this.PITCH_ORDER.indexOf(a); if (ai === -1) ai = 999;
    var bi = this.PITCH_ORDER.indexOf(b); if (bi === -1) bi = 999;
    return ai - bi;
  },

  // Sort an array of pitch type strings in canonical order
  sortPitchTypes: function (types) {
    var self = this;
    return types.slice().sort(function (a, b) { return self.pitchTypeSortCompare(a, b); });
  },

  // Category colors for grouped pitch type display (Hard/Breaking/Offspeed)
  CATEGORY_COLORS: {
    'Hard': '#d62728',
    'Breaking': '#2ca02c',
    'Offspeed': '#ff7f0e',
  },

  // Parse baseball IP notation (e.g. "5.2" = 5 and 2/3 innings) to a float
  parseIP: function (ipStr) {
    if (ipStr == null) return 0;
    var parts = String(ipStr).split('.');
    return parseInt(parts[0], 10) + (parts[1] ? parseInt(parts[1], 10) / 3 : 0);
  },

  // Determine if a pitcher is a starter based on games / games started
  isStarter: function (g, gs) {
    g = g || 0;
    gs = gs || 0;
    return g > 0 && (gs / g) > QUAL.SP_GS_RATIO;
  },

  // ── Canonical qualification multipliers ──────────────────────────────
  // Mirror of pipeline_utils.hitter_pa_per_game / pitcher_ip_per_game.
  // Callers resolve `isROC` themselves (Aggregator._isROCTeam(row.team))
  // and pass it in, so these helpers stay dependency-free.
  //   MLB hitter 3.1 PA×TG / ROC 2.7
  //   MLB SP 1.0 IP×TG, RP 0.5 / ROC SP 0.8, RP 0.4
  hitterPaPerGame: function (isROC) {
    return isROC ? QUAL.PA_PER_GAME_MILB : QUAL.PA_PER_GAME;
  },
  pitcherIpPerGame: function (isStarter, isROC) {
    if (isROC) return isStarter ? QUAL.SP_IP_PER_GAME_MILB : QUAL.RP_IP_PER_GAME_MILB;
    return isStarter ? QUAL.SP_IP_PER_GAME : QUAL.RP_IP_PER_GAME;
  },

  // ── Canonical qualification ──────────────────────────────────────────
  // ONE home for "is this row qualified?". Before 2026-08-19 the Aggregator,
  // DataStore.filterRows (the pre-microData first paint) and the player page
  // each answered it separately, and they disagreed on 16 MLB players.

  // "2TM"/"3TM"/…/"10TM". A multi-team LABEL, not a franchise: it has no
  // schedule, so it must never supply a team-games denominator.
  isCombinedTeam: function (team) {
    return typeof team === 'string' && /^\d+TM$/.test(team);
  },

  // Group a player's per-team and combined rows. Key on mlbId when present so
  // two distinct players sharing a name (two "Max Muncy") do not collide.
  playerKey: function (row) {
    return (row.mlbId != null && row.mlbId !== '')
      ? ('id:' + row.mlbId)
      : ('nm:' + (row.pitcher || row.hitter || ''));
  },

  /**
   * Build the lookups a qualification test needs. Call once per row set, then
   * pass the result to isQualified/resolveQual for each row.
   *
   * @param rows        every row of the set being tested: each MLB stint, each
   *                    ROC stint, and the combined 2TM/3TM row. Unfiltered.
   * @param teamGames   team code -> distinct game dates.
   * @param isROCFn     function(team) -> true for a ROC/AAA team.
   * @param rosterRows  optional. Where the most-recent-team answer is read
   *                    from. Defaults to `rows`. The Aggregator re-aggregates
   *                    under the active filters and does not carry
   *                    lastGameDate, so it passes the shipped leaderboard rows
   *                    (window.PITCHER_DATA / window.HITTER_DATA) here. Which
   *                    club a player is on does not depend on a date filter.
   */
  buildQualContext: function (rows, teamGames, isROCFn, rosterRows) {
    var self = this;
    var ctx = {
      teamGames: teamGames || {},
      isROC: isROCFn || function () { return false; },
      combined: {},
      denomTeam: {},
      datelessPlayers: [],
    };
    var i, k;
    // The combined row supplies the IP/PA/G/GS a multi-team player is judged
    // on, so it must come from the set being tested.
    for (i = 0; i < rows.length; i++) {
      if (self.isCombinedTeam(rows[i].team)) ctx.combined[self.playerKey(rows[i])] = rows[i];
    }

    // A multi-team player is measured against the team he is on now, i.e. the
    // one he most recently played for (Wally, 2026-08-19). The combined label
    // cannot supply this: "2TM" spans 144 distinct dates because it pools every
    // traded player in the league, "3TM" spans only 109, and "4TM" has no entry
    // at all, which zeroed the threshold and put a 133-PA hitter on the
    // qualified board.
    //
    // Only MLB stints count. A ROC appearance belongs to another league and
    // cannot set an MLB denominator, so a player sent down still measures
    // against his last MLB club.
    var roster = rosterRows || rows;
    var rosterCombined = {};
    for (i = 0; i < roster.length; i++) {
      if (self.isCombinedTeam(roster[i].team)) rosterCombined[self.playerKey(roster[i])] = true;
    }
    var stints = {};
    for (i = 0; i < roster.length; i++) {
      var r = roster[i], t = r.team;
      if (self.isCombinedTeam(t) || ctx.isROC(t)) continue;
      k = self.playerKey(r);
      if (!rosterCombined[k]) continue;
      (stints[k] || (stints[k] = [])).push(r);
    }
    for (k in stints) {
      var list = stints[k], pick = null, dated = true, j;
      for (j = 0; j < list.length; j++) { if (!list[j].lastGameDate) { dated = false; break; } }
      if (dated) {
        for (j = 0; j < list.length; j++) {
          // A tie is impossible in practice: one player plays for one club on
          // one date. Break it on team code so the answer never depends on row
          // order.
          if (!pick || list[j].lastGameDate > pick.lastGameDate ||
              (list[j].lastGameDate === pick.lastGameDate && list[j].team < pick.team)) {
            pick = list[j];
          }
        }
      } else {
        // Announced degrade: an embed built before the stint rows carried
        // lastGameDate. Fall back to the longest schedule among his clubs,
        // which is the pre-2026-08-19 behaviour.
        ctx.datelessPlayers.push(k);
        for (j = 0; j < list.length; j++) {
          var gj = ctx.teamGames[list[j].team] || 0;
          var gp = pick ? (ctx.teamGames[pick.team] || 0) : -1;
          if (!pick || gj > gp || (gj === gp && list[j].team < pick.team)) pick = list[j];
        }
      }
      if (pick) ctx.denomTeam[k] = pick.team;
    }
    if (ctx.datelessPlayers.length && !this._qualDateWarned) {
      this._qualDateWarned = true;
      console.warn('Qualification: ' + ctx.datelessPlayers.length + ' multi-team player(s) have a ' +
                   'stint row with no lastGameDate. Falling back to the longest schedule among ' +
                   'their clubs. Rebuild the embed to restore the current-team rule.');
    }
    return ctx;
  },

  /**
   * Resolve every input the qualification test uses, so callers that need the
   * combined row or the threshold do not recompute them.
   * @param isPitcher true for IP x team games, false for PA x team games.
   */
  resolveQual: function (row, ctx, isPitcher) {
    var isROCRow = ctx.isROC(row.team);
    // A ROC/AAA stint is its own league and never inherits the combined row.
    // That row is built from MLB stints only, so borrowing its innings put
    // Jake Bird's 1.1 ROC innings on the qualified ROC board. (2026-08-19)
    var mt = isROCRow ? null : (ctx.combined[this.playerKey(row)] || null);
    var src = mt || row;
    var tg;
    if (mt) {
      var dt = ctx.denomTeam[this.playerKey(row)];
      tg = (dt && ctx.teamGames[dt]) || 0;
    } else {
      tg = ctx.teamGames[row.team] || 0;
    }
    var amount = isPitcher
      ? this.parseIP(src.ip)
      : ((src.paAll != null ? src.paAll : src.pa) || 0);
    var perGame = isPitcher
      ? this.pitcherIpPerGame(this.isStarter(src.g, src.gs), isROCRow)
      : this.hitterPaPerGame(isROCRow);
    return {
      combinedRow: mt,
      denomTeam: mt ? ctx.denomTeam[this.playerKey(row)] : row.team,
      teamGames: tg,
      amount: amount,
      threshold: tg * perGame,
      // No team games means no rate, and "unknown" is not "qualified". This
      // guard is why a 4TM row can no longer clear a threshold of zero.
      qualified: tg > 0 && amount >= tg * perGame,
    };
  },

  isQualified: function (row, ctx, isPitcher) {
    return this.resolveQual(row, ctx, isPitcher).qualified;
  },

  // Create a pitch badge <span> element (small or regular size)
  createPitchBadge: function (pitchType, small) {
    var badge = document.createElement('span');
    badge.className = small ? 'pitch-badge-sm' : 'pitch-badge';
    var color = this.getPitchColor(pitchType);
    badge.style.backgroundColor = color;
    badge.style.color = this.badgeTextColor(color);
    badge.textContent = pitchType;
    return badge;
  },

  // Create a category badge (Hard/Breaking/Offspeed) <span> element
  createCategoryBadge: function (category, small) {
    var badge = document.createElement('span');
    badge.className = small ? 'pitch-badge-sm' : 'pitch-badge';
    var color = this.CATEGORY_COLORS[category] || '#888';
    badge.style.backgroundColor = color;
    badge.style.color = this.badgeTextColor(color);
    badge.textContent = category;
    return badge;
  },

  badgeTextColor: function (hexColor) {
    const hex = hexColor.replace('#', '');
    let r = parseInt(hex.substring(0, 2), 16) / 255;
    let g = parseInt(hex.substring(2, 4), 16) / 255;
    let b = parseInt(hex.substring(4, 6), 16) / 255;
    r = r <= 0.03928 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4);
    g = g <= 0.03928 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4);
    b = b <= 0.03928 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4);
    const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return lum > 0.25 ? 'black' : 'white';
  },

  // Percentile color — print scale: paper neutral at 50, slate below, brick above.
  // Blend from mid-paper toward the target with intensity (|p-50|/50)^1.3 * 0.72,
  // so mid percentiles stay close to the page and extremes read as inked tints.
  percentileColor: function (pctl) {
    if (pctl === null || pctl === undefined) return null;
    const base = [236, 227, 209];                      // mid-paper (between row and zebra)
    const target = pctl >= 50 ? [176, 64, 47]          // brick (good)
                              : [86, 120, 155];        // slate (bad)
    const t = Math.pow(Math.abs(pctl - 50) / 50, 1.3) * 0.72;
    const r = Math.round(base[0] + (target[0] - base[0]) * t);
    const g = Math.round(base[1] + (target[1] - base[1]) * t);
    const b = Math.round(base[2] + (target[2] - base[2]) * t);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  },

  // Percentile color for BUBBLES (player-page circles + bars). Unlike the cell
  // scale above, this keeps a VISIBLE warm-greige floor at the 50th percentile
  // so a mid bubble still reads as a filled disc on cream instead of vanishing
  // into the paper. Endpoints kept light enough for ink text. Cells (dense
  // tables) deliberately keep the near-paper midpoint via percentileColor.
  percentileBubbleColor: function (pctl) {
    if (pctl === null || pctl === undefined) return 'rgb(203, 184, 156)';
    const neutral = [203, 184, 156];                   // warm greige (visible on cream)
    const target = pctl >= 50 ? [168, 54, 40]          // brick (good), ink-safe
                              : [86, 118, 152];         // slate (bad), ink-safe
    const t = Math.pow(Math.abs(pctl - 50) / 50, 0.72);
    const r = Math.round(neutral[0] + (target[0] - neutral[0]) * t);
    const g = Math.round(neutral[1] + (target[1] - neutral[1]) * t);
    const b = Math.round(neutral[2] + (target[2] - neutral[2]) * t);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  },

  // Text color for percentile backgrounds — always ink on the print scale
  percentileTextColor: function (pctl) {
    if (pctl === null || pctl === undefined) return null;
    return '#1a1612';
  },

  // Deep, readable TEXT color carrying the percentile hue — for value text that
  // sits ON the paper (not a filled cell/bubble). The light bubble/cell tints
  // are background colors and vanish as text on cream, so this blends warm
  // ink-gray toward deep terracotta (good) / deep slate (bad) instead.
  percentileTextInk: function (pctl) {
    if (pctl === null || pctl === undefined) return '#6a5f55';
    const neutral = [0x6a, 0x5f, 0x55];                // warm ink-gray
    const target = pctl >= 50 ? [0x9f, 0x30, 0x26]     // deep terracotta (good)
                              : [0x2f, 0x55, 0x73];     // deep slate (bad)
    const t = Math.pow(Math.abs(pctl - 50) / 50, 1.1);
    const r = Math.round(neutral[0] + (target[0] - neutral[0]) * t);
    const g = Math.round(neutral[1] + (target[1] - neutral[1]) * t);
    const b = Math.round(neutral[2] + (target[2] - neutral[2]) * t);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  },


  TOOLTIPS: {
    '#': 'Rank',
    'Pitcher': 'Pitcher name',
    'Team': 'Team abbreviation',
    'Throws': 'Throwing hand (R/L)',
    'Pitch': 'Pitch type',
    'Count': 'Number of pitches',
    'Usage%': 'Usage rate (% of pitcher\'s total pitches)',
    'Velo': 'Average Velocity (mph)',
    'Spin': 'Average Spin Rate (rpm)',
    'Tilt': 'Observed Tilt — direction of total break as a clock face (12:00 = pure ride straight up, 6:00 = straight drop). Measured from actual movement, unlike RTilt (Release Tilt), which is the spin axis at release',
    'IVB': 'Induced Vertical Break (inches) — vertical movement created by spin, measured against a spin-free pitch. High IVB = ride/carry; negative = true drop',
    'HB': 'Horizontal Break (inches, pitcher POV): positive = arm side for a RHP, glove side for a LHP',
    'RelZ': 'Vertical Release Position (feet)',
    'RelX': 'Horizontal Release Position (feet)',
    'Ext': 'Extension (feet)',
    'VAA': 'Vertical Approach Angle — how steeply the pitch crosses the plate (degrees). Always negative; closer to 0 = flatter',
    'nVAA': 'VAA with the location effect removed (high pitches are naturally flatter). Positive = flatter than its height predicts — the sneaky-ride trait that plays at the top of the zone',
    'HAA': 'Horizontal Approach Angle (degrees)',
    'Z-%': 'In-Zone Rate',
    'CSW%': 'Called Strike + Whiff Rate',
    'Chase%': 'Out-of-Zone Swing Rate',
    'GB%': 'Ground Ball Rate',
    // Hitter tooltips
    'Hitter': 'Hitter name',
    'Bats': 'Batting side (R/L/S)',
    'Pitches': 'Total pitches',
    'Swings': 'Total swings',
    'K%': 'Strikeout Rate (strikeouts / plate appearances)',
    'BB%': 'Walk Rate (walks / plate appearances)',
    'Swing%': 'Swing Rate (swings / total pitches)',
    'Z-Swing%': 'In-Zone Swing Rate',
    'Z-Sw% - Chase%': 'Z-Swing% minus Chase% — swings at strikes minus swings at balls. Higher = attacking the zone while laying off junk',
    'Whiff%': 'Whiff Rate (misses / total swings)',
    'Avg EV': 'Average Exit Velocity (mph, all BIP)',
    'Max EV': 'Max Exit Velocity (mph)',
    'Barrel%': 'Barrel Rate — barrels are the EV/LA combos that historically return at least a .500 BA and 1.500 SLG (starts at 98 mph EV)',
    'LD%': 'Line Drive Rate',
    'FB%': 'Fly Ball Rate',
    'PU%': 'Popup Rate',
    'Med LA': 'Median Launch Angle (degrees)',
    'Pull%': 'Pull Rate (batted balls to pull side)',
    'Middle%': 'Center rate (BIP up the middle)',
    'Oppo%': 'Oppo Rate (batted balls to opposite field)',
    'AirPull%': 'Air Pull Rate (pulled LD + FB / total BIP; popups excluded from the numerator)',
  },

  readHash: function () {
    const hash = window.location.hash.replace(/^#/, '');
    if (!hash) return {};
    const params = {};
    hash.split('&').forEach(function (part) {
      const kv = part.split('=');
      if (kv.length === 2) params[decodeURIComponent(kv[0])] = decodeURIComponent(kv[1]);
    });
    return params;
  },

  toCSV: function (data, columns) {
    const lines = [];
    // Header
    lines.push(columns.map(function (c) { return '"' + c.label + '"'; }).join(','));
    // Rows
    data.forEach(function (row) {
      const cells = columns.map(function (c) {
        const v = row[c.key];
        if (v === null || v === undefined) return '';
        const formatted = c.format(v);
        return '"' + String(formatted).replace(/"/g, '""') + '"';
      });
      lines.push(cells.join(','));
    });
    return lines.join('\n');
  },

  // Add a horizontal scroll fade indicator to a scrollable container
  addScrollFade: function (container) {
    container.style.position = 'relative';
    var fadeDiv = document.createElement('div');
    fadeDiv.style.cssText = 'position:absolute;right:0;top:0;bottom:0;width:24px;background:linear-gradient(to right, transparent, var(--bg-card, #1a1d21));pointer-events:none;z-index:1;opacity:0;transition:opacity 0.2s;';
    container.appendChild(fadeDiv);
    container.addEventListener('scroll', function() {
      var maxScroll = container.scrollWidth - container.clientWidth;
      fadeDiv.style.opacity = (container.scrollLeft >= maxScroll - 2) ? '0' : '1';
    });
    setTimeout(function() {
      if (container.scrollWidth > container.clientWidth) fadeDiv.style.opacity = '1';
    }, 100);
  },

  downloadFile: function (content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType || 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
  },

  copyToClipboard: function (text) {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).catch(function () {
        Utils._fallbackCopy(text);
      });
    } else {
      Utils._fallbackCopy(text);
    }
  },

  _fallbackCopy: function (text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  },

  toTSV: function (data, columns) {
    const lines = [];
    lines.push(columns.map(function (c) { return c.label; }).join('\t'));
    data.forEach(function (row) {
      const cells = columns.map(function (c) {
        const v = row[c.key];
        if (v === null || v === undefined) return '';
        return String(c.format(v));
      });
      lines.push(cells.join('\t'));
    });
    return lines.join('\n');
  },
};
