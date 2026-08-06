/* Player Page — full-page pitcher/hitter profile view */
var PlayerPage = {
  chart: null,
  isOpen: false,
  _playerType: null, // 'pitcher' or 'hitter'

  PITCHING_STATS: [
    // Stuff foundation: velocity (dynamic — expands to FF and/or SI velo when present).
    { key: '_veloPlaceholder',  label: '',                  format: function() { return ''; }, _dynamic: true },
    // The grades trio (2026-07-20: bubbles for the flagship composites —
    // Loc+ had one since v2; Stuff+/Pitching+ were missing since launch).
    { key: 'stuffScore',        label: 'Stuff+',           format: function(v) { return v != null ? Math.round(v) : '—'; } },
    { key: 'locPlus',           label: 'Loc+',             format: function(v) { return v != null ? Math.round(v) : '—'; } },
    { key: 'pitchingScore',     label: 'Pitching+',        format: function(v) { return v != null ? Math.round(v) : '—'; } },
    // Pitcher+ (2026-07-24): the all-encompassing composite. Sits with the
    // grades but is pitcher-level, not a per-pitch atom mean — see
    // pipeline_pitcherplus.py.
    { key: 'pitcherPlus',       label: 'Pitcher+',         format: function(v) { return v != null ? Math.round(v) : '—'; } },
    // Pitcher+ Proj removed 2026-07-30 (already hidden on leaderboards).
    // Rate stats and their composite estimator.
    { key: 'kPct',              label: 'K%',               format: function(v) { return Utils.formatPct(v); } },
    { key: 'bbPct',             label: 'BB%',              format: function(v) { return Utils.formatPct(v); } },
    { key: 'kbbPct',            label: 'K-BB%',            format: function(v) { return Utils.formatPct(v, true); } },
    { key: 'siera',             label: 'SIERA',            format: function(v) { return v != null ? v.toFixed(2) : '—'; } },
    // Bat-missing process. (Zone% and Z-Whiff% live in the tables below —
    // battery 2026-07-20: Zone% pred ~0, Z-Whiff% r=.83 with Whiff%.)
    { key: 'swStrPct',          label: 'Whiff%',           format: function(v) { return Utils.formatPct(v); } },
    { key: 'chasePct',          label: 'Chase%',           format: function(v) { return Utils.formatPct(v); } },
    // Contact allowed. (xwOBAcon stays in the batted-ball table — pitcher-side
    // split-half reliability .26; a percentile bubble overstates it.)
    { key: 'xwOBA',             label: 'xwOBA',            format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'hardHitPct',        label: 'Hard-Hit%',        format: function(v) { return Utils.formatPct(v); } },
    { key: 'barrelPctAgainst',  label: 'Barrel%',          format: function(v) { return Utils.formatPct(v); } },
    { key: 'gbPct',             label: 'GB%',              format: function(v) { return Utils.formatPct(v); } },
  ],

  HITTING_STATS: [
    // Quality summary — headlines first
    { key: 'hitterPlus',   label: 'Hitter+',     format: function(v) { return v != null ? Math.round(v) : '—'; } },
    { key: 'xWRCplus',     label: 'xWRC+',       format: function(v) { return v != null ? Math.round(v) : '—'; } },
    // Hitter+ components (weights: BB+ 65%, CT+ 28%, SD+ 7%)
    { key: 'bbPlus',       label: 'BB+',         format: function(v) { return v != null ? Math.round(v) : '—'; } },
    { key: 'ctPlus',       label: 'CT+',         format: function(v) { return v != null ? Math.round(v) : '—'; } },
    { key: 'sdPlus',       label: 'SD+',         format: function(v) { return v != null ? Math.round(v) : '—'; } },
    // Expected outcomes (wOBA family)
    { key: 'xwOBA',        label: 'xwOBA',       format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'xwOBAcon',     label: 'xwOBAcon',    format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'xwOBAsp',      label: 'xwOBAsp',     format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    // Contact quality (EV). EV50 over Avg EV (battery 2026-07-20: rel .93
    // vs .87, same predictive power — mirrors the leaderboard call).
    { key: 'ev50',         label: 'EV50',        format: function(v) { return v != null ? v.toFixed(1) + ' mph' : '—'; } },
    { key: 'maxEV',        label: 'Max EV',      format: function(v) { return v != null ? v.toFixed(1) + ' mph' : '—'; } },
    // Contact quality (rate)
    { key: 'hardHitPct',   label: 'Hard-Hit%',   format: function(v) { return Utils.formatPct(v); } },
    { key: 'barrelPct',    label: 'Barrel%',     format: function(v) { return Utils.formatPct(v); } },
    // Discipline — PA-level rates
    { key: 'bbPct',        label: 'BB%',         format: function(v) { return Utils.formatPct(v); } },
    { key: 'kPct',         label: 'K%',          format: function(v) { return Utils.formatPct(v); } },
    // Discipline — per-pitch process. (Whiff% bubble dropped 2026-07-20:
    // contact axis covered by Z-Contact% + K%; it stays in the tables.)
    { key: 'chasePct',     label: 'Chase%',      format: function(v) { return Utils.formatPct(v); } },
    { key: 'izContactPct', label: 'Z-Contact%', format: function(v) { return Utils.formatPct(v); } },
    // Bat tracking
    { key: 'batSpeed',     label: 'Bat Speed',   format: function(v) { return v != null ? v.toFixed(1) + ' mph' : '—'; }, rocHide: true, batSpeedQual: true },
    { key: 'squaredUpPct', label: 'Squared-Up%', format: function(v) { return Utils.formatPct(v); }, rocHide: true, batSpeedQual: true },
    { key: 'blastPct',     label: 'Blast%',      format: function(v) { return Utils.formatPct(v); }, rocHide: true, batSpeedQual: true },
    // Spray
    { key: 'airPullPct',   label: 'Air Pull%',   format: function(v) { return Utils.formatPct(v); } },
    // Speed
    { key: 'sprintSpeed',  label: 'Sprint Speed', format: function(v) { return v != null ? v.toFixed(1) + ' ft/s' : '—'; }, rocHide: true, sprintQual: true },
  ],

  HITTER_STATS_COLS: [
    { key: 'g', label: 'G', format: function(v) { return v != null ? v : '—'; }, noPctl: true },
    { key: 'pa', label: 'PA', format: function(v) { return v != null ? v : '—'; }, noPctl: true },
    { key: 'ab', label: 'AB', format: function(v) { return v != null ? v : '—'; }, noPctl: true },
    { key: 'hr', label: 'HR', format: function(v) { return v != null ? v : '—'; } },
    { key: 'avg', label: 'AVG', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'obp', label: 'OBP', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'slg', label: 'SLG', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'ops', label: 'OPS', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'iso', label: 'ISO', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'babip', label: 'BABIP', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'wOBA', label: 'wOBA', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'wRCplus', label: 'wRC+', format: function(v) { return v != null ? v : '—'; } },
    { key: 'xWRCplus', label: 'xWRC+', format: function(v) { return v != null ? v : '—'; } },
    { key: 'hitterPlus', label: 'Hitter+', format: function(v) { return v != null ? Math.round(v) : '—'; } },
    { key: 'bbPct', label: 'BB%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'kPct', label: 'K%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'bbToK', label: 'BB/K', format: function(v) { return v != null ? v.toFixed(2) : '—'; } },
  ],

  // Shown in place of HITTER_STATS_COLS whenever a handedness is selected.
  // G is meaningless per hand. wOBA and xWRC+ now split — the Guts linear
  // weights ship in metadata and the hitter micro carries IBB, so the
  // aggregator rebuilds both under the filter on the same scale as the season
  // value. wRC+ and Hitter+ stay out: wRC+ is overwritten with the canonical
  // FanGraphs number (which has no per-hand split, so a pipeline-formula
  // version would read on a different scale), and Hitter+ needs the SD+/CT+
  // weight tables, which are server-side only.
  HITTER_PLATOON_STATS_COLS: [
    { key: 'pa', label: 'PA', format: function(v) { return v != null ? v : '—'; }, noPctl: true, noDiff: true },
    { key: 'ab', label: 'AB', format: function(v) { return v != null ? v : '—'; }, noPctl: true, noDiff: true },
    // HR stays uncolored in platoon mode: ranking a counting stat inside a
    // split reads as skill when it's mostly the PA gap between the two sides.
    { key: 'hr', label: 'HR', format: function(v) { return v != null ? v : '—'; }, noPctl: true, noDiff: true },
    { key: 'avg', label: 'AVG', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'obp', label: 'OBP', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'slg', label: 'SLG', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'ops', label: 'OPS', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'iso', label: 'ISO', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'babip', label: 'BABIP', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'wOBA', label: 'wOBA', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'xwOBA', label: 'xwOBA', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'xWRCplus', label: 'xWRC+', format: function(v) { return v != null ? v : '—'; } },
    { key: 'xBA', label: 'xBA', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'xSLG', label: 'xSLG', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'xwOBAcon', label: 'xwOBAcon', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'hardHitPct', label: 'Hard-Hit%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'barrelPct', label: 'Barrel%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'bbPct', label: 'BB%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'kPct', label: 'K%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'bbToK', label: 'BB/K', format: function(v) { return v != null ? v.toFixed(2) : '—'; } },
  ],

  HITTER_BATTED_BALL_COLS: [
    { key: 'pitchType', label: 'Pitch', noPctl: true },
    { key: 'count', label: 'Pitches', format: function(v) { return v != null ? v : '—'; }, noPctl: true },
    { key: 'nBip', label: 'BIP', format: function(v) { return v != null ? v : '—'; }, noPctl: true },
    // Outcome
    { key: 'babip', label: 'BABIP', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'xwOBAcon', label: 'xwOBAcon', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    // Quality
    { key: 'avgEVAll', label: 'Avg EV', format: function(v) { return v != null ? v.toFixed(1) : '—'; } },
    { key: 'ev50', label: 'EV50', format: function(v) { return v != null ? v.toFixed(1) : '—'; } },
    { key: 'maxEV', label: 'Max EV', format: function(v) { return v != null ? v.toFixed(1) : '—'; } },
    { key: 'hardHitPct', label: 'Hard-Hit%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'barrelPct', label: 'Barrel%', format: function(v) { return Utils.formatPct(v); } },
    // Composition
    // LD/FB/PU coloring enabled 2026-08-06 (per Wally): LD%/FB% color
    // higher-is-better, PU% lower-is-better; directions live in the
    // server/aggregator invert lists, this table just stopped suppressing them.
    { key: 'gbPct', label: 'GB%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'ldPct', label: 'LD%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'fbPct', label: 'FB%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'puPct', label: 'PU%', format: function(v) { return Utils.formatPct(v); } },
    // Spray
    { key: 'pullPct', label: 'Pull%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'middlePct', label: 'Mid%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'oppoPct', label: 'Oppo%', format: function(v) { return Utils.formatPct(v); } },
  ],

  HITTER_PLATE_DISCIPLINE_COLS: [
    { key: 'pitchType', label: 'Pitch', noPctl: true },
    { key: 'count', label: 'Pitches', format: function(v) { return v != null ? v : '—'; }, noPctl: true },
    { key: 'nSwings', label: 'Swings', format: function(v) { return v != null ? v : '—'; }, noPctl: true },
    // Swing decisions
    { key: 'swingPct', label: 'Swing%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'izSwingPct', label: 'Z-Swing%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'chasePct', label: 'Chase%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'izSwChase', label: 'Z-Sw-Ch%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'firstPitchSwingPct', label: 'FPSw%', format: function(v) { return Utils.formatPct(v); } },
    // Miss rates
    { key: 'whiffPct', label: 'Whiff%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'twoStrikeWhiffPct', label: '2K Whiff%', format: function(v) { return Utils.formatPct(v); } },
    // Contact rates
    { key: 'contactPct', label: 'Contact%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'izContactPct', label: 'Z-Contact%', format: function(v) { return Utils.formatPct(v); } },
  ],

  HITTER_BAT_TRACKING_COLS: [
    { key: 'batSpeed', label: 'Bat Speed', format: function(v) { return v != null ? v.toFixed(1) + ' mph' : '—'; } },
    { key: 'swingLength', label: 'Swing Length', format: function(v) { return v != null ? v.toFixed(1) + ' ft' : '—'; } },
    // Direction-ambiguous angles (no inherent better/worse) — show value, no color.
    { key: 'attackAngle', label: 'Attack Angle', format: function(v) { return v != null ? v.toFixed(1) + '°' : '—'; }, noPctl: true },
    { key: 'attackDirection', label: 'Attack Dir', format: function(v) { return v != null ? v.toFixed(1) + '°' : '—'; }, noPctl: true },
    { key: 'swingPathTilt', label: 'Swing Path Tilt', format: function(v) { return v != null ? v.toFixed(1) + '°' : '—'; }, noPctl: true },
    // Quality-of-contact rates (higher is better; percentile-colored).
    { key: 'squaredUpPct', label: 'Squared-Up%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'blastPct', label: 'Blast%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'idealAAPct', label: 'IdealAtkAngle%', format: function(v) { return Utils.formatPct(v); } },
  ],

  PITCH_TABLE_COLS: [
    { key: 'pitchType', label: 'Pitch' },
    { key: 'count', label: 'Count', format: function(v) { return v != null ? v : '—'; } },
    { key: 'usagePct', label: 'Usage', format: function(v) { return Utils.formatPct(v); } },
    { key: 'velocity', label: 'MPH',   format: function(v) { return v != null ? v.toFixed(1) : '—'; } },
    { key: 'spinRate', label: 'Spin',   format: function(v) { return v != null ? Math.round(v) : '—'; } },
    { key: 'breakTilt', label: 'OTilt', format: function(v) { return v || '—'; } },
    { key: 'indVertBrk', label: 'IVB',  format: function(v) { return v != null ? v.toFixed(1) + '"' : '—'; } },
    { key: 'horzBrk',    label: 'HB',   format: function(v) { return v != null ? v.toFixed(1) + '"' : '—'; } },
    { key: 'extension', label: 'Ext',   format: function(v) { return v != null ? Utils.formatFeetInches(v) : '—'; } },
    { key: 'armAngle',  label: 'Arm\u00B0', format: function(v) { return v != null ? v.toFixed(1) + '\u00B0' : '—'; } },
  ],

  EXPANDED_PITCH_COLS: [
    { key: 'pitchType', label: 'Pitch' },
    { key: 'count', label: 'Count', format: function(v) { return v != null ? v : '—'; } },
    { key: 'usagePct', label: 'Usage', format: function(v) { return Utils.formatPct(v); } },
    { key: 'velocity', label: 'Velo', format: function(v) { return v != null ? v.toFixed(1) + ' mph' : '—'; } },
    { key: 'maxVelo', label: 'Max Velo', format: function(v) { return v != null ? v.toFixed(1) + ' mph' : '—'; } },
    { key: 'spinRate', label: 'Spin Rate', format: function(v) { return v != null ? Math.round(v) + ' rpm' : '—'; } },
    { key: 'releaseTilt', label: 'Release Tilt', format: function(v) { return v || '—'; } },
    { key: 'breakTilt', label: 'Observed Tilt', format: function(v) { return v || '—'; } },
    { key: 'indVertBrk', label: 'IndVertBrk', format: function(v) { return v != null ? v.toFixed(1) + '"' : '—'; } },
    { key: 'horzBrk', label: 'HorzBrk', format: function(v) { return v != null ? v.toFixed(1) + '"' : '—'; } },
    { key: 'relPosZ', label: 'RelHeight', format: function(v) { return v != null ? Utils.formatFeetInches(v) : '—'; } },
    { key: 'relPosX', label: 'RelSide', format: function(v) { return v != null ? Utils.formatFeetInches(v) : '—'; } },
    { key: 'extension', label: 'Extension', format: function(v) { return v != null ? Utils.formatFeetInches(v) : '—'; } },
    { key: 'armAngle', label: 'Arm Angle', format: function(v) { return v != null ? v.toFixed(1) + '\u00B0' : '—'; }, rocHide: true },
    { key: 'nVAA', label: 'nVAA', format: function(v) { return v != null ? v.toFixed(2) + '\u00B0' : '—'; } },
    { key: 'nHAA', label: 'nHAA', format: function(v) { return v != null ? v.toFixed(2) + '\u00B0' : '—'; } },
    { key: 'stuffScore', label: 'Stuff+', format: function(v) { return v != null ? Math.round(v) : '—'; } },
    { key: 'locPlus', label: 'Loc+', format: function(v) { return v != null ? Math.round(v) : '—'; } },
    { key: 'pitchingScore', label: 'Pitching+', format: function(v) { return v != null ? Math.round(v) : '—'; } },
  ],

  STATS_COLS: [
    { key: 'g', label: 'G', format: function(v) { return v != null ? v : '—'; } },
    { key: 'gs', label: 'GS', format: function(v) { return v != null ? v : '—'; } },
    { key: 'ip', label: 'IP', format: function(v) { return v != null ? v : '—'; } },
    { key: 'w', label: 'W', format: function(v) { return v != null ? v : '—'; } },
    { key: 'l', label: 'L', format: function(v) { return v != null ? v : '—'; } },
    { key: 'sv', label: 'SV', format: function(v) { return v != null ? v : '—'; } },
    { key: 'hld', label: 'HLD', format: function(v) { return v != null ? v : '—'; } },
    { key: 'kPct', label: 'K%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'bbPct', label: 'BB%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'kbbPct', label: 'K-BB%', format: function(v) { return Utils.formatPct(v, true); } },
    { key: 'era', label: 'ERA', format: function(v) { return v != null ? v.toFixed(2) : '—'; } },
    { key: 'fip', label: 'FIP', format: function(v) { return v != null ? v.toFixed(2) : '—'; } },
    { key: 'xFIP', label: 'xFIP', format: function(v) { return v != null ? v.toFixed(2) : '—'; } },
    { key: 'siera', label: 'SIERA', format: function(v) { return v != null ? v.toFixed(2) : '—'; } },
  ],

  // Shown in place of STATS_COLS whenever a handedness is selected. The
  // season line's counting stats (G/GS/IP/W/L/SV/HLD) and IP-denominated
  // rates (ERA/FIP/xFIP/SIERA) have no per-hand split in the data — leaving
  // them on screen made the toggle look inert. Every column here does split:
  // TBF/AVG/OBP/BABIP/K%/BB%/Whiff% come from the micro counters, the
  // expected stats from PITCHER_DATA's _vsL/_vsR fields.
  PLATOON_STATS_COLS: [
    { key: 'pa', label: 'TBF', format: function(v) { return v != null ? v : '—'; }, noPctl: true, noDiff: true },
    { key: 'avgAgainst', label: 'AVG', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'obpAgainst', label: 'OBP', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'wOBA', label: 'wOBA', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'xwOBA', label: 'xwOBA', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'xBA', label: 'xBA', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'xSLG', label: 'xSLG', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'babip', label: 'BABIP', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; }, dec3: true },
    { key: 'kPct', label: 'K%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'bbPct', label: 'BB%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'kbbPct', label: 'K-BB%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'swStrPct', label: 'Whiff%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'hardHitPct', label: 'Hard-Hit%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'barrelPctAgainst', label: 'Barrel%', format: function(v) { return Utils.formatPct(v); } },
  ],

  BATTED_BALL_COLS: [
    { key: 'pitchType', label: 'Pitch' },
    { key: 'count', label: 'Count', format: function(v) { return v != null ? v : '—'; } },
    { key: 'nBip', label: 'BIP', format: function(v) { return v != null ? v : '—'; } },
    { key: 'babip', label: 'BABIP', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'xwOBAcon', label: 'xwOBAcon', format: function(v) { return v != null ? v.toFixed(3).replace(/^0/, '') : '—'; } },
    { key: 'avgEVAgainst', label: 'Avg EV', format: function(v) { return v != null ? v.toFixed(1) : '—'; } },
    { key: 'hardHitPct', label: 'Hard-Hit%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'barrelPctAgainst', label: 'Barrel%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'gbPct', label: 'GB%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'ldPct', label: 'LD%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'fbPct', label: 'FB%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'puPct', label: 'PU%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'hrFbPct', label: 'HR/FB', format: function(v) { return Utils.formatPct(v); } },
  ],

  PLATE_DISCIPLINE_COLS: [
    { key: 'pitchType', label: 'Pitch' },
    { key: 'count', label: 'Count', format: function(v) { return v != null ? v : '—'; } },
    { key: 'nSwings', label: 'Swings', format: function(v) { return v != null ? v : '—'; } },
    { key: 'strikePct', label: 'Strike%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'izPct', label: 'Zone%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'cswPct', label: 'CSW%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'swStrPct', label: 'Whiff%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'izWhiffPct', label: 'Z-Whiff%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'twoStrikeWhiffPct', label: '2K Whiff%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'chasePct', label: 'Chase%', format: function(v) { return Utils.formatPct(v); } },
    { key: 'fpsPct', label: 'FPS%', format: function(v) { return Utils.formatPct(v); } },
  ],

  _getFilteredPlayerData: function (data, isPitcher) {
    var nameKey = isPitcher ? 'pitcher' : 'hitter';
    var source = isPitcher ? window.PITCHER_DATA : window.HITTER_DATA;
    if (!source) return data;

    var name = data[nameKey];
    var team = data.team;
    for (var i = 0; i < source.length; i++) {
      if (source[i][nameKey] === name && source[i].team === team) return source[i];
    }
    return null;
  },

  _getFilteredPitchRows: function (data) {
    return this._getPitchRows(data.pitcher, data.team);
  },

  open: function (mlbId, team) {
    // Embed split: player pages need the heavy chunk (pitch details,
    // swing locations). If the background prefetch hasn't landed yet
    // (only possible in the first seconds after load), queue the open.
    if (!DataStore.heavyReady) {
      var _args = [mlbId, team];
      DataStore.whenHeavy(function () { PlayerPage.open.apply(PlayerPage, _args); });
      return;
    }
    var pitcherData = this._findPitcherByMlbId(mlbId, team);
    var hitterData = !pitcherData ? this._findHitterByMlbId(mlbId, team) : null;

    if (!pitcherData && !hitterData) {
      console.warn('Player not found for mlbId:', mlbId);
      return;
    }

    // Pitch details are sharded per pitcher, so pull this one's shard before
    // rendering — everything downstream reads window.PITCH_DETAILS
    // synchronously. Resolves immediately once the shard is cached, so
    // re-opening the same pitcher doesn't re-fetch.
    if (pitcherData) {
      var key = pitcherData.pitcher + '|' + pitcherData.team;
      if (!(window.PITCH_DETAILS || {})[key] && DataStore.hasPitchDetails(key)) {
        var self = this;
        DataStore.ensurePitchDetails(key).then(function () {
          self._openResolved(pitcherData, null);
        });
        return;
      }
    }
    this._openResolved(pitcherData, hitterData);
  },

  _openResolved: function (pitcherData, hitterData) {
    var curHash = window.location.hash.replace(/^#/, '');
    if (curHash.indexOf('player=') === -1) {
      this._lastRoute = curHash;
    }

    this.isOpen = true;
    this._playerType = pitcherData ? 'pitcher' : 'hitter';

    document.getElementById('player-page').style.display = 'block';
    var pctlLeg = document.getElementById('pctl-legend');
    if (pctlLeg) pctlLeg.style.display = 'none';

    if (pitcherData) {
      this._renderPitcherPage(pitcherData);
    } else {
      this._renderHitterPage(hitterData);
    }

    document.getElementById('player-page').scrollTop = 0;

    this._bindClickOutside();
  },

  _renderPitcherPage: function (data) {
    this._showPitcherLayout();
    this._renderIdentity(data);
    this._heatMapHand = 'R';
    this._countHand = 'R';
    this._platoonHand = 'all';
    this._gameDate = null; // null = all games
    this._currentData = data;
    this._renderGameLog(data);
    this._renderPitcherContent(data);
    this._bindHandToggles();
    this._bindPlatoonToggle('pitcher');
    this._bindGameLog();
  },

  _hasDataForGameType: function (data) {
    var isPitcher = !!(data.pitcher);
    if (isPitcher) {
      var details = window.PITCH_DETAILS || {};
      var key = data.pitcher + '|' + data.team;
      var pitches = details[key];
      return pitches && pitches.length > 0;
    } else {
      var hitterData = window.HITTER_DATA || [];
      for (var i = 0; i < hitterData.length; i++) {
        if (hitterData[i].hitter === data.hitter && hitterData[i].team === data.team) return true;
      }
      return false;
    }
  },

  _renderPitcherContent: function (data) {
    this._platoonHand = 'all';
    this._resetPlatoonToggleUI('pitcher-platoon-toggle');
    document.getElementById('player-percentiles').innerHTML = '';
    var sections = ['player-pitch-usage-table', 'player-stats-table', 'player-platoon-split-table',
                    'player-expanded-pitch-table',
                    'player-batted-ball-table', 'player-plate-discipline-table',
                    'player-heat-maps', 'player-count-table', 'player-game-grades-table'];
    for (var i = 0; i < sections.length; i++) {
      var el = document.getElementById(sections[i]);
      if (el) el.innerHTML = '';
    }
    this.destroyChart();
    this._renderUsage(data);

    if (!this._hasDataForGameType(data)) {
      var container = document.getElementById('player-percentiles');
      container.innerHTML = '<p style="color:var(--text-secondary);padding:20px;text-align:center;">No data available for this period.</p>';
      return;
    }
    var filteredData = this._getFilteredPlayerData(data, true) || data;
    var filteredPitchRows = this._getFilteredPitchRows(data);
    this._filteredData = filteredData;
    this._filteredPitchRows = filteredPitchRows;

    var rocTeams = (DataStore.metadata && DataStore.metadata.rocTeams) || [];
    var isROCPlayer = rocTeams.indexOf(data.team) !== -1;
    this._isROC = isROCPlayer;
    // ROC pitcher pages get the platoon toggles too (2026-08-06, per Wally,
    // matching the ROC hitter enable): pitcherMicro/pitchMicro/pitcherBip all
    // carry ROC rows split by batter hand, and the platoon aggregation keeps
    // them via filters.includeROC.
    this._setPlatoonTogglesVisible('pitcher', true);

    if (!isROCPlayer) {
      this._renderPitchRunValues(filteredData);
    }

    var pitchingStats = this.PITCHING_STATS;
    if (isROCPlayer) {
      pitchingStats = pitchingStats.filter(function (s) { return !s.rocHide; });
    }
    this._renderPercentiles(filteredData, pitchingStats, true);
    this._renderMovementChart(data); // uses PITCH_DETAILS, already filtered
    this._renderPitchTable(data); // uses PITCH_DETAILS, already filtered
    this._renderStatsTable(filteredData);
    this._renderPlatoonSplitDeferred('pitcher', data);
    this._renderExpandedPitchTable(data); // will use _filteredPitchRows
    this._renderPlateDisciplineTable(data); // will use _filteredPitchRows
    this._renderBattedBallTable(data); // will use _filteredPitchRows
    this._renderHeatMaps(data);
    this._renderCountTable(data);
    this._renderGameGrades(data);
    this._addAllDownloadButtons(data);
  },

  // Per-game Stuff+/Loc+/Pitching+ — plain averages of the per-pitch grade
  // atoms (micro data), so each row matches that outing's card and the
  // Sheets AVERAGEIF for the date. "Which start had his best stuff."
  _renderGameGrades: function (data) {
    var section = document.getElementById('player-game-grades-section');
    var container = document.getElementById('player-game-grades-table');
    if (!section || !container) return;
    container.innerHTML = '';

    var games = (typeof Aggregator !== 'undefined' && Aggregator.loaded)
      ? Aggregator.getPitcherGameGrades(data.pitcher, data.team) : [];
    games = games.filter(function (g) { return g.stuffScore != null || g.locPlus != null; });
    if (games.length === 0) { section.style.display = 'none'; return; }
    section.style.display = '';

    var table = document.createElement('table');
    table.className = 'count-table';
    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    var headers = ['Date', 'Pitches', 'Stuff+', 'Loc+', 'Pitching+'];
    for (var hi = 0; hi < headers.length; hi++) {
      var th = document.createElement('th');
      th.textContent = headers[hi];
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    for (var i = 0; i < games.length; i++) {
      var g = games[i];
      var tr = document.createElement('tr');
      var parts = g.date.split('-');
      var cells = [
        parseInt(parts[1]) + '/' + parseInt(parts[2]) + '/' + parts[0],
        g.n,
        g.stuffScore != null ? Math.round(g.stuffScore) : '—',
        g.locPlus != null ? Math.round(g.locPlus) : '—',
        g.pitchingScore != null ? Math.round(g.pitchingScore) : '—',
      ];
      for (var ci = 0; ci < cells.length; ci++) {
        var td = document.createElement('td');
        td.textContent = cells[ci];
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    container.appendChild(table);
  },

  // Get PITCH_DETAILS for this pitcher (already game-type-specific), optionally filtered by _gameDate and _platoonDetailHand
  _getFilteredDetails: function (data) {
    var details = window.PITCH_DETAILS || {};
    var key = data.pitcher + '|' + data.team;
    var pitches = details[key];
    if (!pitches || pitches.length === 0) return [];

    // Filter by specific game date if set
    if (this._gameDate) {
      var gd = this._gameDate;
      pitches = pitches.filter(function (p) { return p.gd === gd; });
    }
    // Filter by batter hand when platoon toggle is active
    if (this._platoonDetailHand) {
      var bh = this._platoonDetailHand;
      pitches = pitches.filter(function (p) { return p.bh === bh; });
    }
    return pitches;
  },

  // Get unique game dates for this pitcher from PITCH_DETAILS (already game-type-specific)
  _getGameDates: function (data) {
    var details = window.PITCH_DETAILS || {};
    var key = data.pitcher + '|' + data.team;
    var pitches = details[key];
    if (!pitches) return [];

    var dateSet = {};
    for (var i = 0; i < pitches.length; i++) {
      var gd = pitches[i].gd;
      if (gd) dateSet[gd] = true;
    }
    return Object.keys(dateSet).sort();
  },

  _renderGameLog: function (data) {
    var container = document.getElementById('player-game-log');
    container.innerHTML = '';
    var dates = this._getGameDates(data);
    if (dates.length === 0) { container.style.display = 'none'; return; }

    container.style.display = '';
    var label = document.createElement('span');
    label.className = 'game-log-label';
    label.textContent = 'Game:';
    container.appendChild(label);

    var select = document.createElement('select');
    select.id = 'player-game-select';
    select.className = 'game-log-select';

    var allOpt = document.createElement('option');
    allOpt.value = '';
    allOpt.textContent = 'All Games';
    select.appendChild(allOpt);

    for (var i = 0; i < dates.length; i++) {
      var opt = document.createElement('option');
      opt.value = dates[i];
      var parts = dates[i].split('-');
      opt.textContent = parseInt(parts[1]) + '/' + parseInt(parts[2]) + '/' + parts[0];
      select.appendChild(opt);
    }

    if (this._gameDate) select.value = this._gameDate;
    container.appendChild(select);
  },

  _bindGameLog: function () {
    var self = this;
    this._gameLogHandler = function () {
      var select = document.getElementById('player-game-select');
      if (!select) return;
      var date = select.value || null;
      if (date === (self._gameDate || '')) return;
      self._gameDate = date || null;
      if (self._currentData) self._renderPitcherContent(self._currentData);
    };
    var select = document.getElementById('player-game-select');
    if (select) select.addEventListener('change', this._gameLogHandler);
  },

  _unbindGameLog: function () {
    if (this._gameLogHandler) {
      var select = document.getElementById('player-game-select');
      if (select) select.removeEventListener('change', this._gameLogHandler);
      this._gameLogHandler = null;
    }
  },

  _renderHitterPage: function (data) {
    this._showHitterLayout();
    this._renderHitterIdentity(data);
    this._platoonHand = 'all';
    this._currentData = data;
    this._sprayMode = 'all';
    this._sprayBatSide = (data.stands === 'S') ? 'L' : null; // null = not switch hitter
    this._renderHitterContent(data);
    this._bindSprayToggle();
    this._bindSprayBatSideToggle(data);
    this._bindPlatoonToggle('hitter');
  },

  _renderHitterContent: function (data) {
    this._platoonHand = 'all';
    this._resetPlatoonToggleUI('hitter-platoon-toggle');
    document.getElementById('player-percentiles').innerHTML = '';
    var sections = ['player-hitter-stats-table', 'player-hitter-platoon-split-table',
                    'player-hitter-batted-ball-table',
                    'player-hitter-plate-discipline-table', 'player-hitter-bat-tracking-table'];
    for (var i = 0; i < sections.length; i++) {
      var el = document.getElementById(sections[i]);
      if (el) el.innerHTML = '';
    }

    if (!this._hasDataForGameType(data)) {
      var container = document.getElementById('player-percentiles');
      container.innerHTML = '<p style="color:var(--text-secondary);padding:20px;text-align:center;">No data available for this period.</p>';
      return;
    }
    var rocTeams = (DataStore.metadata && DataStore.metadata.rocTeams) || [];
    var isROCPlayer = rocTeams.indexOf(data.team) !== -1;
    this._isROC = isROCPlayer;
    // Hitter platoon toggles show for ROC too (2026-08-06, per Wally): the
    // micro tables carry ROC hitter rows split by pitcher hand (they didn't
    // when the toggle was first hidden in 12fa8b9), and the aggregation path
    // keeps them via filters.includeROC. Pitcher-page toggles stay MLB-only.
    this._setPlatoonTogglesVisible('hitter', true);

    if (!isROCPlayer) {
      this._renderHitterRunValue(data);
    }

    var hittingStats = this.HITTING_STATS;
    if (isROCPlayer) {
      hittingStats = hittingStats.filter(function (s) { return !s.rocHide; });
    }
    this._renderPercentiles(data, hittingStats, true);
    this._renderSprayChart(data);
    this._renderLASprayChart(data);
    this._renderHitterSmallStats(data);
    this._renderHitterSwingHeatMaps(data);
    this._renderHitterStatsFullTable(data, isROCPlayer);
    this._renderPlatoonSplitDeferred('hitter', data);
    this._renderHitterPlateDisciplineTable(data);
    this._renderHitterBattedBallTable(data, isROCPlayer);
    if (!isROCPlayer) {
      this._renderHitterBatTrackingTable(data);
    }
    this._addAllDownloadButtons(data);
  },

  _showPitcherLayout: function () {
    var usageSection = document.querySelector('.player-usage-section');
    var movementCol = document.querySelector('.player-col-movement');
    if (usageSection) usageSection.style.display = '';
    if (movementCol) movementCol.style.display = '';
    var title = document.querySelector('.player-col-movement .section-title');
    if (title) title.textContent = 'Movement Profile';
    var pitchTable = document.getElementById('player-pitch-usage-table');
    if (pitchTable) pitchTable.style.display = '';
    var sprayToggle = document.getElementById('spray-toggle-inline');
    if (sprayToggle) sprayToggle.style.display = 'none';
    var sprayLegend = document.getElementById('spray-legend-inline');
    if (sprayLegend) sprayLegend.style.display = 'none';
    var hitterSections = ['player-spray-section', 'player-la-spray-section', 'player-swing-heatmaps-section',
      'player-hitter-stats-section', 'player-hitter-platoon-split-section',
      'player-hitter-batted-ball-section',
      'player-hitter-plate-discipline-section', 'player-hitter-bat-tracking-section'];
    for (var i = 0; i < hitterSections.length; i++) {
      var el = document.getElementById(hitterSections[i]);
      if (el) el.style.display = 'none';
    }
    var pitcherSections = ['player-stats-section', 'player-platoon-split-section',
      'player-expanded-pitch-section',
      'player-batted-ball-section', 'player-plate-discipline-section',
      'player-location-section', 'player-count-section'];
    for (var j = 0; j < pitcherSections.length; j++) {
      var el2 = document.getElementById(pitcherSections[j]);
      if (el2) el2.style.display = '';
    }
  },

  _showHitterLayout: function () {
    var usageSection = document.querySelector('.player-usage-section');
    if (usageSection) usageSection.style.display = 'none';
    var aaLabel = document.getElementById('player-arm-angle');
    if (aaLabel) aaLabel.style.display = 'none';
    var movementCol = document.querySelector('.player-col-movement');
    if (movementCol) movementCol.style.display = '';
    var title = document.querySelector('.player-col-movement .section-title');
    if (title) title.textContent = 'Spray Chart';
    var pitchTable = document.getElementById('player-pitch-usage-table');
    if (pitchTable) pitchTable.style.display = 'none';
    var sprayToggle = document.getElementById('spray-toggle-inline');
    if (sprayToggle) sprayToggle.style.display = '';
    var sprayLegend = document.getElementById('spray-legend-inline');
    if (sprayLegend) sprayLegend.style.display = '';
    var pitcherSections = ['player-stats-section', 'player-platoon-split-section',
      'player-expanded-pitch-section',
      'player-batted-ball-section', 'player-plate-discipline-section',
      'player-location-section', 'player-count-section', 'player-game-grades-section'];
    for (var i = 0; i < pitcherSections.length; i++) {
      var el = document.getElementById(pitcherSections[i]);
      if (el) el.style.display = 'none';
    }
    var gameLog = document.getElementById('player-game-log');
    if (gameLog) gameLog.style.display = 'none';
    var hitterSections = ['player-la-spray-section', 'player-swing-heatmaps-section',
      'player-hitter-stats-section', 'player-hitter-platoon-split-section',
      'player-hitter-batted-ball-section',
      'player-hitter-plate-discipline-section', 'player-hitter-bat-tracking-section'];
    for (var j = 0; j < hitterSections.length; j++) {
      var el2 = document.getElementById(hitterSections[j]);
      if (el2) el2.style.display = '';
    }
    var spraySec = document.getElementById('player-spray-section');
    if (spraySec) spraySec.style.display = 'none';
  },


  close: function () {
    this.isOpen = false;
    this._heatMapHand = 'R';
    this._countHand = 'R';
    this._gameDate = null;
    this._currentData = null;
    this.destroyChart();
    this._unbindClickOutside();
    this._unbindHandToggles();
    this._unbindPlatoonToggle();
    this._unbindGameLog();

    var sections = ['player-expanded-pitch-section', 'player-location-section', 'player-count-section',
      'player-game-grades-section',
      'player-spray-section', 'player-la-spray-section', 'player-swing-heatmaps-section',
      'player-hitter-stats-section', 'player-hitter-batted-ball-section',
      'player-hitter-plate-discipline-section', 'player-hitter-bat-tracking-section'];
    for (var i = 0; i < sections.length; i++) {
      var el = document.getElementById(sections[i]);
      if (el) el.style.display = 'none';
    }
    this._unbindSprayToggle();

    document.getElementById('player-page').style.display = 'none';
    var pctlLeg = document.getElementById('pctl-legend');
    if (pctlLeg) pctlLeg.style.display = '';
    if (this._lastRoute) {
      history.replaceState(null, '', '#' + this._lastRoute);
    }
  },

  _bindClickOutside: function () {
    var self = this;
    this._clickOutsideHandler = function (e) {
      var inner = document.querySelector('.player-page-inner');
      if (inner && !inner.contains(e.target)) {
        self.close();
      }
    };
    document.getElementById('player-page').addEventListener('click', this._clickOutsideHandler);
  },

  _unbindClickOutside: function () {
    if (this._clickOutsideHandler) {
      document.getElementById('player-page').removeEventListener('click', this._clickOutsideHandler);
      this._clickOutsideHandler = null;
    }
  },

  destroyChart: function () {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
  },


  _findPitcherByMlbId: function (mlbId, preferTeam) {
    return this._findPlayerByMlbId(window.PITCHER_DATA || [], mlbId, preferTeam);
  },

  // MLB priority: multi-team aggregate (2TM/3TM) > preferTeam match > any MLB
  // match. Multi-team players open on their combined row so the page shows the
  // full MLB season, not just games from whichever team filter was active.
  // ROC rows live outside that aggregate (2TM/3TM combines MLB stints only),
  // so a click from a ROC team must land on the ROC row, and an MLB-context
  // click must never fall back onto one.
  _findPlayerByMlbId: function (rows, mlbId, preferTeam) {
    mlbId = parseInt(mlbId, 10);
    var rocTeams = (DataStore.metadata && DataStore.metadata.rocTeams) || [];
    var preferROC = preferTeam && rocTeams.indexOf(preferTeam) !== -1;
    var multiTeam = null, preferred = null, fallback = null, rocFallback = null;
    var MULTI_RE = /^\d+TM$/;
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].mlbId !== mlbId) continue;
      var t = rows[i].team;
      if (MULTI_RE.test(t)) multiTeam = rows[i];
      else if (preferTeam && t === preferTeam) preferred = rows[i];
      else if (rocTeams.indexOf(t) !== -1) { if (!rocFallback) rocFallback = rows[i]; }
      else if (!fallback) fallback = rows[i];
    }
    if (preferROC) return preferred || rocFallback || multiTeam || fallback;
    return multiTeam || preferred || fallback || rocFallback;
  },

  _getPitchRows: function (pitcherName, team) {
    var pitchData = window.PITCH_DATA || [];
    var rows = [];
    for (var i = 0; i < pitchData.length; i++) {
      if (pitchData[i].pitcher === pitcherName && pitchData[i].team === team) {
        rows.push(pitchData[i]);
      }
    }
    rows.sort(function (a, b) { return (b.usagePct || 0) - (a.usagePct || 0); });
    return rows;
  },


  // Shared headshot + name rendering for both pitcher and hitter identity
  _FALLBACK_HEADSHOT: 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="160" height="160" viewBox="0 0 160 160"><rect fill="%23333" width="160" height="160"/><text fill="%23888" font-size="40" x="50%" y="55%" text-anchor="middle" dominant-baseline="middle">?</text></svg>'),

  _renderHeadshotAndName: function (mlbId, name) {
    var img = document.getElementById('player-headshot');
    if (mlbId) {
      img.src = 'https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/' + mlbId + '/headshot/67/current';
      img.alt = name;
    } else {
      img.src = '';
      img.alt = '';
    }
    var fallback = this._FALLBACK_HEADSHOT;
    img.onerror = function () { this.src = fallback; };

    var nameParts = name.split(', ');
    var displayName = nameParts.length === 2 ? nameParts[1] + ' ' + nameParts[0] : name;
    document.getElementById('player-name').textContent = displayName;
  },

  _renderIdentity: function (data) {
    this._renderHeadshotAndName(data.mlbId, data.pitcher);

    var pos = (data.throws === 'L' ? 'LHP' : 'RHP');
    // Append SP / RP role suffix when game-appearance data is available so a
    // closer's profile reads differently from a starter's at a glance.
    var role = '';
    if (data.g && data.g > 0) {
      role = ' | ' + (Utils.isStarter(data.g, data.gs) ? 'SP' : 'RP');
    }
    var posEl = document.getElementById('player-position');
    var ageEl = document.getElementById('player-age');
    posEl.textContent = pos + ' | ' + (data.team || '') + role;
    ageEl.textContent = '';

    if (data.mlbId) {
      fetch('https://statsapi.mlb.com/api/v1/people/' + data.mlbId)
        .then(function (res) { return res.json(); })
        .then(function (json) {
          var person = json.people && json.people[0];
          if (person && person.currentAge != null) {
            posEl.textContent = pos + ' | ' + (data.team || '') + ' | Age: ' + person.currentAge + role;
          }
        })
        .catch(function () { /* silently ignore */ });
    }
  },

  _renderHitterIdentity: function (data) {
    this._renderHeadshotAndName(data.mlbId, data.hitter);

    var batHand = data.stands === 'S' ? 'S' : (data.stands === 'L' ? 'L' : 'R');
    // Prepend primary MLB position (most games this season) to the bio line.
    // Sourced from MLB Stats API in the pipeline; null when no MLB games yet.
    var posPrefix = data.position ? data.position + ' | ' : '';
    var posEl = document.getElementById('player-position');
    var ageEl = document.getElementById('player-age');
    // Compact bio format so the line fits on one row of the player card.
    // Initial render (sync): position + bats + team. Throws hand and age
    // come in from the async MLB API call below.
    posEl.textContent = posPrefix + batHand + ' | ' + (data.team || '');
    ageEl.textContent = '';

    if (data.mlbId) {
      fetch('https://statsapi.mlb.com/api/v1/people/' + data.mlbId)
        .then(function (res) { return res.json(); })
        .then(function (json) {
          var person = json.people && json.people[0];
          if (person) {
            var throwHand = person.pitchHand && person.pitchHand.code ? person.pitchHand.code : '';
            var btLabel = throwHand ? batHand + '/' + throwHand : batHand;
            var agePart = person.currentAge != null ? ' | Age: ' + person.currentAge : '';
            posEl.textContent = posPrefix + btLabel + ' | ' + (data.team || '') + agePart;
          }
        })
        .catch(function () { /* silently ignore */ });
    }
  },


  _renderUsage: function (data) {
    var pitches = this._getFilteredDetails(data);

    var usageByHand = { L: {}, R: {} };
    var totalByHand = { L: 0, R: 0 };

    for (var i = 0; i < pitches.length; i++) {
      var p = pitches[i];
      var bh = p.bh;
      var pt = p.pt;
      if (!bh || !pt) continue;
      if (!usageByHand[bh]) usageByHand[bh] = {};
      if (!usageByHand[bh][pt]) usageByHand[bh][pt] = 0;
      usageByHand[bh][pt]++;
      if (!totalByHand[bh]) totalByHand[bh] = 0;
      totalByHand[bh]++;
    }

    this._renderUsageBars('player-usage-lhh', usageByHand.L || {}, totalByHand.L || 0);
    this._renderUsageBars('player-usage-rhh', usageByHand.R || {}, totalByHand.R || 0);
  },

  _renderUsageBars: function (containerId, usageMap, total) {
    var container = document.getElementById(containerId);
    container.innerHTML = '';

    if (total === 0) {
      container.innerHTML = '<p class="no-data">No data</p>';
      return;
    }

    var entries = [];
    for (var pt in usageMap) {
      entries.push({ pt: pt, count: usageMap[pt], pct: usageMap[pt] / total });
    }
    entries.sort(function (a, b) { return b.pct - a.pct; });

    for (var i = 0; i < entries.length; i++) {
      var e = entries[i];
      var row = document.createElement('div');
      row.className = 'usage-bar-row';

      var label = document.createElement('span');
      label.className = 'usage-label';
      var badge = Utils.createPitchBadge(e.pt, true);
      label.appendChild(badge);

      var barWrap = document.createElement('div');
      barWrap.className = 'usage-bar-track';
      var bar = document.createElement('div');
      bar.className = 'usage-bar-fill';
      bar.style.width = Math.round(e.pct * 100) + '%';
      bar.style.backgroundColor = Utils.getPitchColor(e.pt);
      barWrap.appendChild(bar);

      var pct = document.createElement('span');
      pct.className = 'usage-pct';
      // Sub-1% usage reads as "< 1%" rather than rounding down to "0%".
      pct.textContent = (e.pct > 0 && e.pct * 100 < 1) ? '< 1%' : Math.round(e.pct * 100) + '%';

      row.appendChild(label);
      row.appendChild(barWrap);
      row.appendChild(pct);
      container.appendChild(row);
    }
  },


  // Team games used for the qualification threshold. Multi-team (2TM/3TM) rows
  // have no teamGames entry for the combined label, so fall back to the
  // cumulative (max across the player's stint teams) games — matching the
  // aggregator's cumTeamGames so the player page agrees with the leaderboard.
  // Returning 0 here (rather than an undefined→0 that zeroes the threshold) lets
  // the callers' `tg > 0` guard correctly treat a genuinely non-qualifying
  // multi-team player as unqualified.
  _qualifyingTeamGames: function (data, isPitcher) {
    var teamGames = Aggregator.loaded ? Aggregator.getTeamGamesPlayed() : {};
    if (!(Aggregator.loaded && Aggregator._isCombinedTeam(data.team))) {
      return teamGames[data.team] || 0;
    }
    var rows = (isPitcher ? window.PITCHER_DATA : window.HITTER_DATA) || [];
    var maxTg = 0;
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      if (r.mlbId === data.mlbId && !Aggregator._isCombinedTeam(r.team)) {
        var tgv = teamGames[r.team] || 0;
        if (tgv > maxTg) maxTg = tgv;
      }
    }
    return maxTg;
  },


  _renderPercentiles: function (data, statsDef, append) {
    var container = document.getElementById('player-percentiles');
    if (!append) container.innerHTML = '';

    var isPitcher = !!(data.pitcher);

    // Determine if player qualifies based on team games played
    // Micro data is already game-type-specific, so no date range needed
    var tg = this._qualifyingTeamGames(data, isPitcher);
    var isQualified;
    var _isROC = Aggregator.loaded && Aggregator._isROCTeam(data.team);
    if (isPitcher) {
      var ipFloat = Utils.parseIP(data.ip);
      var isStarter = Utils.isStarter(data.g, data.gs);
      var ipThreshold = tg * Utils.pitcherIpPerGame(isStarter, _isROC);
      isQualified = tg > 0 && ipFloat >= ipThreshold;
    } else {
      isQualified = tg > 0 && (data.pa || 0) >= tg * Utils.hitterPaPerGame(_isROC);
    }
    var alwaysColorKeys = isPitcher ? { ffVelo: true, siVelo: true } : { maxEV: true };

    var dynamicVeloStats = [];
    if (isPitcher) {
      var pitchRows = this._filteredPitchRows || this._getPitchRows(data.pitcher, data.team);
      var veloFormat = function(v) { return v != null ? v.toFixed(1) + ' mph' : '—'; };
      for (var vi = 0; vi < pitchRows.length; vi++) {
        var pr = pitchRows[vi];
        if (pr.pitchType === 'FF') {
          dynamicVeloStats.push({ key: 'ffVelo', label: 'Fastball Velo', format: veloFormat,
            _val: pr.velocity, _pctl: pr.velocity_pctl });
        }
        if (pr.pitchType === 'SI') {
          dynamicVeloStats.push({ key: 'siVelo', label: 'Sinker Velo', format: veloFormat,
            _val: pr.velocity, _pctl: pr.velocity_pctl });
        }
      }
      dynamicVeloStats.sort(function(a, b) { return a.key === 'ffVelo' ? -1 : 1; });
    }

    // BIP-dependent stats that show gray when bipQual is false
    var HITTER_BIP_STATS = {
      avgEVAll: true, ev50: true, maxEV: true,
      hardHitPct: true, barrelPct: true,
      xBA: true, xSLG: true, xwOBA: true, xwOBAcon: true, xwOBAsp: true,
      babip: true, hrFbPct: true, airPullPct: true
    };
    var PITCHER_BIP_STATS = {
      barrelPctAgainst: true, gbPct: true, avgEVAgainst: true, hardHitPct: true,
      xBA: true, xSLG: true, xwOBA: true, xwOBAcon: true, xwOBAsp: true
    };

    var effectiveStats = [];
    for (var si2 = 0; si2 < statsDef.length; si2++) {
      if (statsDef[si2]._dynamic) {
        for (var dv = 0; dv < dynamicVeloStats.length; dv++) {
          effectiveStats.push(dynamicVeloStats[dv]);
        }
      } else {
        effectiveStats.push(statsDef[si2]);
      }
    }

    for (var i = 0; i < effectiveStats.length; i++) {
      var stat = effectiveStats[i];

      // Section header row — small uppercase divider between groups of stats
      if (stat._section) {
        var hdr = document.createElement('div');
        hdr.className = 'pctl-section-header';
        hdr.textContent = stat._section;
        container.appendChild(hdr);
        continue;
      }

      var val = stat._val !== undefined ? stat._val : data[stat.key];
      if (stat.key === 'hitterPlus' && val == null && data.hitterPlus != null) {
        val = data.hitterPlus;
      }
      var pctl = stat._pctl !== undefined ? stat._pctl : data[stat.key + '_pctl'];
      // BIP qualification: <20 BIP → show gray outline
      var bipStats = isPitcher ? PITCHER_BIP_STATS : HITTER_BIP_STATS;
      var bipUnqual = bipStats[stat.key] && data.bipQual === false;
      // Sprint speed: qualified with ≥10 competitive runs
      var sprintUnqual = stat.sprintQual && (data.nCompRuns || 0) < 10;
      // Bat speed: qualified with ≥10 competitive swings
      var batSpeedUnqual = stat.batSpeedQual && (data.nCompSwings || 0) < 10;
      var showColor = (isQualified || alwaysColorKeys[stat.key]) && !bipUnqual && !sprintUnqual && !batSpeedUnqual;

      var row = document.createElement('div');
      row.className = 'pctl-row';

      var labelEl = document.createElement('span');
      labelEl.className = 'pctl-label';
      labelEl.textContent = stat.label;

      var valEl = document.createElement('span');
      valEl.className = 'pctl-value';
      valEl.textContent = stat.format(val);

      var circleWrap = document.createElement('div');
      circleWrap.className = 'pctl-circle-wrap';

      if (pctl != null) {
        var circle = document.createElement('div');
        circle.className = 'pctl-circle';
        if (showColor) {
          var bgColor = Utils.percentileBubbleColor(pctl);
          var textColor = Utils.percentileTextColor(pctl);
          circle.style.backgroundColor = bgColor;
          circle.style.color = textColor;
          var qualPool = isPitcher ? 'qualified pitchers' : 'qualified hitters';
          circle.title = Utils.ordinal(Math.round(pctl)) + ' percentile among ' + qualPool;
        } else {
          // Unqualified: outline ring instead of filled circle
          circle.style.backgroundColor = 'transparent';
          circle.style.border = '2px solid #b3a68c';
          circle.style.color = '#8a7f75';
          circle.title = 'Below minimum qualification threshold';
        }
        circle.textContent = Math.round(pctl);
        circleWrap.appendChild(circle);
      }

      var barTrack = document.createElement('div');
      barTrack.className = 'pctl-bar-track';
      var barFill = document.createElement('div');
      barFill.className = 'pctl-bar-fill';
      if (pctl != null) {
        barFill.style.width = Math.round(pctl) + '%';
        if (showColor) {
          var barColor = Utils.percentileBubbleColor(pctl);
          barFill.style.background = barColor;
        } else {
          // Unqualified: gray bar with white diagonal hatching
          var barBg = '#d8ccb4';
          var stripColor = 'rgba(240,232,216,0.85)';
          barFill.style.background = barBg + ' repeating-linear-gradient(135deg, ' + stripColor + ', ' + stripColor + ' 2px, transparent 2px, transparent 6px)';
        }
      }
      barTrack.appendChild(barFill);

      row.appendChild(labelEl);
      row.appendChild(barTrack);
      row.appendChild(circleWrap);
      row.appendChild(valEl);
      container.appendChild(row);
    }
  },


  _renderPitchRunValues: function (data) {
    var container = document.getElementById('player-percentiles');

    // Get this pitcher's pitch rows (use filtered if available)
    var pitchRows = this._filteredPitchRows || this._getPitchRows(data.pitcher, data.team);
    if (pitchRows.length === 0) return;

    var sectionLabel = document.createElement('div');
    sectionLabel.className = 'pctl-section-label';
    sectionLabel.style.cssText = 'font-size: 12px; font-weight: 700; text-transform: uppercase; color: var(--text-muted, #888); margin-bottom: 6px; letter-spacing: 0.5px;';
    sectionLabel.textContent = 'xPitch Run Value / 100';
    container.appendChild(sectionLabel);

    // Compute total pitches for overall RV qualifying
    var totalPitches = 0;
    for (var tp = 0; tp < pitchRows.length; tp++) {
      totalPitches += (pitchRows[tp].count || 0);
    }

    // Helper to build a single percentile row
    // pitchCount: number of pitches for this row (null = use totalPitches for overall)
    function buildPctlRow(labelContent, displayVal, pctl, pitchCount) {
      var row = document.createElement('div');
      row.className = 'pctl-row';

      var labelEl = document.createElement('span');
      labelEl.className = 'pctl-label';
      if (typeof labelContent === 'string') {
        labelEl.textContent = labelContent;
      } else {
        labelEl.appendChild(labelContent);
      }

      var valEl = document.createElement('span');
      valEl.className = 'pctl-value';
      valEl.textContent = (displayVal != null) ? displayVal.toFixed(1) : '—';

      // Determine if this row qualifies for coloring.
      // Per-pitch: QUAL.MIN_PITCH_PCTL (25 — matches the per-pitch percentile
      // pool minimum used by the leaderboards and the pipeline). Overall: 100
      // pitches across the whole arsenal.
      var rvQualified;
      if (pitchCount === null) {
        rvQualified = totalPitches >= 100; // overall
      } else {
        rvQualified = (pitchCount || 0) >= QUAL.MIN_PITCH_PCTL; // per-pitch
      }

      var circleWrap = document.createElement('div');
      circleWrap.className = 'pctl-circle-wrap';
      if (pctl != null) {
        var circle = document.createElement('div');
        circle.className = 'pctl-circle';
        if (rvQualified) {
          var bgColor = Utils.percentileBubbleColor(pctl);
          var textColor = Utils.percentileTextColor(pctl);
          circle.style.backgroundColor = bgColor;
          circle.style.color = textColor;
        } else {
          circle.style.backgroundColor = 'transparent';
          circle.style.border = '2px solid #b3a68c';
          circle.style.color = '#8a7f75';
          circle.title = 'Below minimum qualification threshold';
        }
        circle.textContent = Math.round(pctl);
        circleWrap.appendChild(circle);
      }

      var barTrack = document.createElement('div');
      barTrack.className = 'pctl-bar-track';
      var barFill = document.createElement('div');
      barFill.className = 'pctl-bar-fill';
      if (pctl != null) {
        barFill.style.width = Math.round(pctl) + '%';
        if (rvQualified) {
          var barColor = Utils.percentileBubbleColor(pctl);
          barFill.style.background = barColor;
        } else {
          var barBg = '#d8ccb4';
          var stripColor = 'rgba(240,232,216,0.85)';
          barFill.style.background = barBg + ' repeating-linear-gradient(135deg, ' + stripColor + ', ' + stripColor + ' 2px, transparent 2px, transparent 6px)';
        }
      }
      barTrack.appendChild(barFill);

      row.appendChild(labelEl);
      row.appendChild(barTrack);
      row.appendChild(circleWrap);
      row.appendChild(valEl);
      return row;
    }

    // Overall row (xRV/100 — positive = good for pitcher)
    var overallRV100 = data.xRv100;
    container.appendChild(buildPctlRow('Overall', overallRV100, data.xRv100_pctl, null));

    // Per-pitch-type rows (fixed order)
    var sortedPitchRows = pitchRows.slice().sort(function(a, b) {
      return Utils.pitchTypeSortCompare(a.pitchType, b.pitchType);
    });
    for (var i = 0; i < sortedPitchRows.length; i++) {
      var pitch = sortedPitchRows[i];
      var displayVal = pitch.xRv100;
      var pctl = pitch.xRv100_pctl;

      var badge = Utils.createPitchBadge(pitch.pitchType, true);

      container.appendChild(buildPctlRow(badge, displayVal, pctl, pitch.count));
    }

    // Divider before the stat percentiles that follow
    var divider = document.createElement('div');
    divider.style.cssText = 'border-top: 1px solid var(--border, #ddd); margin: 12px 0 8px 0;';
    container.appendChild(divider);
  },


  _renderHitterRunValue: function (data) {
    // "Overall" row at the top of the hitter percentile panel — mirrors the
    // pitcher panel's xPitch Run Value / 100 row in style. Uses the standard
    // pctl-row layout (label + bar + circle + value). Value is xRV per 100
    // PITCHES (unified 2026-07-03 with the per-pitch-type rows below it and
    // the Savant/FG convention; was per 100 PA); percentile comes from the
    // pool computed in process_data.py.
    var container = document.getElementById('player-percentiles');

    var sectionLabel = document.createElement('div');
    sectionLabel.className = 'pctl-section-label';
    sectionLabel.style.cssText = 'font-size: 12px; font-weight: 700; text-transform: uppercase; color: var(--text-muted, #888); margin-bottom: 6px; letter-spacing: 0.5px;';
    sectionLabel.textContent = 'xRun Value / 100 Pitches';
    container.appendChild(sectionLabel);

    // Qualification (PA × team games, ROC-aware: 3.1 MLB / 2.7 ROC) —
    // same gate the generic _renderPercentiles uses for hitter rate stats.
    var tg = this._qualifyingTeamGames(data, false);
    var pa = data.pa || 0;
    var _isROC = Aggregator.loaded && Aggregator._isROCTeam(data.team);
    var isQualified = tg > 0 && pa >= tg * Utils.hitterPaPerGame(_isROC);

    var xRv100 = data.xRv100;
    var pctl = data.xRv100_pctl;

    var row = document.createElement('div');
    row.className = 'pctl-row';

    var labelEl = document.createElement('span');
    labelEl.className = 'pctl-label';
    labelEl.textContent = 'Overall';

    var valEl = document.createElement('span');
    valEl.className = 'pctl-value';
    valEl.textContent = xRv100 != null ? xRv100.toFixed(1) : '—';

    var circleWrap = document.createElement('div');
    circleWrap.className = 'pctl-circle-wrap';
    if (pctl != null) {
      var circle = document.createElement('div');
      circle.className = 'pctl-circle';
      if (isQualified) {
        var bgColor = Utils.percentileBubbleColor(pctl);
        var textColor = Utils.percentileTextColor(pctl);
        circle.style.backgroundColor = bgColor;
        circle.style.color = textColor;
        circle.title = Utils.ordinal(Math.round(pctl)) + ' percentile among qualified hitters';
      } else {
        circle.style.backgroundColor = 'transparent';
        circle.style.border = '2px solid #b3a68c';
        circle.style.color = '#8a7f75';
        circle.title = 'Below minimum qualification threshold';
      }
      circle.textContent = Math.round(pctl);
      circleWrap.appendChild(circle);
    }

    var barTrack = document.createElement('div');
    barTrack.className = 'pctl-bar-track';
    var barFill = document.createElement('div');
    barFill.className = 'pctl-bar-fill';
    if (pctl != null) {
      barFill.style.width = Math.round(pctl) + '%';
      if (isQualified) {
        var barColor = Utils.percentileBubbleColor(pctl);
        barFill.style.background = barColor;
      } else {
        var barBg = '#d8ccb4';
        var stripColor = 'rgba(240,232,216,0.85)';
        barFill.style.background = barBg + ' repeating-linear-gradient(135deg, ' + stripColor + ', ' + stripColor + ' 2px, transparent 2px, transparent 6px)';
      }
    }
    barTrack.appendChild(barFill);

    row.appendChild(labelEl);
    row.appendChild(barTrack);
    row.appendChild(circleWrap);
    row.appendChild(valEl);
    container.appendChild(row);

    var divider = document.createElement('div');
    divider.style.cssText = 'border-top: 1px solid var(--border, #ddd); margin: 12px 0 8px 0;';
    container.appendChild(divider);
  },


  _renderMovementChart: function (data) {
    this.destroyChart();

    var pitcherName = data.pitcher;
    var team = data.team;

    // Build movement data from filtered pitch details
    var filteredPitches = this._getFilteredDetails(data);
    if (filteredPitches.length === 0) return;

    // Compute and display average arm angle (hide for ROC — no arm angle data)
    var aaLabel = document.getElementById('player-arm-angle');
    if (aaLabel) {
      if (this._isROC) {
        aaLabel.style.display = 'none';
      } else {
        var aaVals = [];
        for (var ai = 0; ai < filteredPitches.length; ai++) {
          if (filteredPitches[ai].aa != null) aaVals.push(filteredPitches[ai].aa);
        }
        if (aaVals.length > 0) {
          var avgAA = (aaVals.reduce(function(a,b){return a+b;},0) / aaVals.length).toFixed(1);
          aaLabel.textContent = 'Arm Angle = ' + avgAA + '\u00B0';
          aaLabel.style.display = '';
        } else {
          aaLabel.style.display = 'none';
        }
      }
    }
    var groups = {};
    for (var fi = 0; fi < filteredPitches.length; fi++) {
      var fp = filteredPitches[fi];
      if (fp.ivb == null || fp.hb == null) continue;
      if (!groups[fp.pt]) groups[fp.pt] = [];
      groups[fp.pt].push({ x: fp.hb, y: fp.ivb });
    }
    if (Object.keys(groups).length === 0) return;

    var datasets = [];
    var ellipseMeta = [];
    var pitchTypes = Utils.sortPitchTypes(Object.keys(groups));
    // Hairline edge matching the panel bg — de-blobs dense clusters without a contrasting ring
    var dotEdge = 'rgba(240,232,216,0.9)';

    for (var j = 0; j < pitchTypes.length; j++) {
      var pt = pitchTypes[j];
      var pts = groups[pt];
      var color = ScatterChart.getColor(pt);
      var label = pt + ' - ' + Utils.pitchTypeLabel(pt);

      datasets.push({
        label: label,
        data: pts,
        backgroundColor: color.bg,
        borderColor: dotEdge,
        borderWidth: 1,
        pointRadius: 5,
        pointHoverRadius: 7,
      });

      // Actual-spread ellipse (covariance, dashed/unfilled, pitch-colored) — matches the cards
      var ellipse = ScatterChart.computeEllipse(pts);
      if (ellipse) {
        ellipseMeta.push({ color: color.bg, ellipse: ellipse });
      }
    }

    var canvas = document.getElementById('player-pitch-chart');
    var ctx = canvas.getContext('2d');

    var gridColor = 'rgba(58,48,38,0.12)';
    var tickColor = '#6a5f55';
    var crossColor = 'rgba(58,48,38,0.22)';

    this.chart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            title: { display: true, text: 'Horizontal Break (in)', color: tickColor },
            min: -25, max: 25,
            grid: { color: gridColor },
            ticks: { color: tickColor, stepSize: 5 },
          },
          y: {
            title: { display: true, text: 'Induced Vertical Break (in)', color: tickColor },
            min: -25, max: 25,
            grid: { color: gridColor },
            ticks: { color: tickColor, stepSize: 5 },
          },
        },
        plugins: {
          legend: { display: true, position: 'bottom', labels: { color: tickColor, usePointStyle: true, padding: 10, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ': IVB ' + ctx.parsed.y.toFixed(1) + ', HB ' + ctx.parsed.x.toFixed(1);
              },
            },
          },
        },
      },
      plugins: [{
        id: 'crosshairs-ellipses-player',
        beforeDatasetsDraw: function (chart) {
          var ctx = chart.ctx;
          var xAxis = chart.scales.x;
          var yAxis = chart.scales.y;

          // Actual-spread ellipses (dashed, unfilled, pitch-colored) — matches the cards
          for (var ei = 0; ei < ellipseMeta.length; ei++) {
            var em = ellipseMeta[ei];
            if (!em.ellipse) continue;
            var ecx = xAxis.getPixelForValue(em.ellipse.cx);
            var ecy = yAxis.getPixelForValue(em.ellipse.cy);
            var erx = Math.abs(xAxis.getPixelForValue(em.ellipse.rx) - xAxis.getPixelForValue(0));
            var ery = Math.abs(yAxis.getPixelForValue(em.ellipse.ry) - yAxis.getPixelForValue(0));
            ctx.save();
            ctx.strokeStyle = em.color;
            ctx.lineWidth = 1.3;
            ctx.globalAlpha = 0.7;
            ctx.setLineDash([6, 4]);
            ctx.translate(ecx, ecy);
            ctx.rotate(-em.ellipse.angle);
            ctx.beginPath();
            ctx.ellipse(0, 0, erx, ery, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
          }

          // Crosshairs
          ctx.save();
          ctx.strokeStyle = crossColor;
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]);
          var cx = xAxis.getPixelForValue(0);
          var cy = yAxis.getPixelForValue(0);
          ctx.beginPath();
          ctx.moveTo(cx, yAxis.top);
          ctx.lineTo(cx, yAxis.bottom);
          ctx.moveTo(xAxis.left, cy);
          ctx.lineTo(xAxis.right, cy);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.restore();
        },
      }],
    });

  },


  // Aggregate PITCH_DETAILS into per-pitch-type rows (same format as PITCH_DATA)
  _aggregateDetailsToRows: function (pitches) {
    var byType = {};
    var total = pitches.length;
    for (var i = 0; i < pitches.length; i++) {
      var p = pitches[i];
      if (!p.pt) continue;
      if (!byType[p.pt]) byType[p.pt] = { count: 0, velos: [], spins: [], ivbs: [], hbs: [], tiltSins: [], tiltCoss: [], armAngles: [], exts: [] };
      var g = byType[p.pt];
      g.count++;
      if (p.v != null) g.velos.push(p.v);
      if (p.sp != null) g.spins.push(p.sp);
      if (p.ivb != null) g.ivbs.push(p.ivb);
      if (p.hb != null) g.hbs.push(p.hb);
      if (p.aa != null) g.armAngles.push(p.aa);
      if (p.ext != null) g.exts.push(p.ext);
      // Tilt: convert H:MM to angle for circular mean
      if (p.tl) {
        var parts = p.tl.split(':');
        if (parts.length === 2) {
          var h = parseInt(parts[0], 10);
          var m = parseInt(parts[1], 10);
          if (h === 12) h = 0;
          var totalMin = h * 60 + m;
          var angle = totalMin * 2 * Math.PI / 720;
          g.tiltSins.push(Math.sin(angle));
          g.tiltCoss.push(Math.cos(angle));
        }
      }
    }
    var rows = [];
    for (var pt in byType) {
      var g = byType[pt];
      var avg = function(arr) { return arr.length > 0 ? arr.reduce(function(a,b){return a+b;},0) / arr.length : null; };
      // Circular mean for tilt
      var tiltStr = null;
      if (g.tiltSins.length > 0) {
        var avgSin = avg(g.tiltSins);
        var avgCos = avg(g.tiltCoss);
        var avgAngle = Math.atan2(avgSin, avgCos);
        var avgMin = ((avgAngle * 720 / (2 * Math.PI)) % 720 + 720) % 720;
        var tH = Math.floor(avgMin / 60);
        var tM = Math.round(avgMin % 60);
        if (tM === 60) { tH++; tM = 0; }
        if (tH === 0) tH = 12;
        tiltStr = tH + ':' + (tM < 10 ? '0' : '') + tM;
      }
      rows.push({
        pitchType: pt,
        count: g.count,
        usagePct: total > 0 ? g.count / total : null,
        velocity: avg(g.velos),
        spinRate: avg(g.spins),
        breakTilt: tiltStr,
        indVertBrk: avg(g.ivbs),
        horzBrk: avg(g.hbs),
        armAngle: avg(g.armAngles),
        extension: avg(g.exts),
      });
    }
    rows.sort(function(a, b) { return (b.usagePct || 0) - (a.usagePct || 0); });
    return rows;
  },

  _renderPitchTable: function (data) {
    var container = document.getElementById('player-pitch-usage-table');
    container.innerHTML = '';

    // Always compute from filtered details (respects game type + game date)
    var filtered = this._getFilteredDetails(data);
    var pitchRows = filtered.length > 0 ? this._aggregateDetailsToRows(filtered) : [];
    if (pitchRows.length === 0) return;

    var cols = this.PITCH_TABLE_COLS;
    if (this._isROC) {
      cols = cols.filter(function (c) { return !c.rocHide; });
    }

    var table = document.createElement('table');
    table.className = 'player-pitch-stats-table';

    // Header
    var thead = document.createElement('thead');
    var headerRow = document.createElement('tr');
    for (var i = 0; i < cols.length; i++) {
      var th = document.createElement('th');
      th.textContent = cols[i].label;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    var tbody = document.createElement('tbody');
    for (var r = 0; r < pitchRows.length; r++) {
      var row = pitchRows[r];
      var tr = document.createElement('tr');
      for (var c = 0; c < cols.length; c++) {
        var col = cols[c];
        var td = document.createElement('td');
        if (col.key === 'pitchType') {
          var badge = Utils.createPitchBadge(row.pitchType, true);
          td.appendChild(badge);
        } else {
          var val = row[col.key];
          td.textContent = col.format ? col.format(val) : (val != null ? val : '—');
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    container.appendChild(table);
  },


  /**
   * Create a small SVG sparkline from an array of {date, avgVelo} points.
   * Returns an SVG element (inline, ~120×28px).
   */
  _createVeloSparkline: function (points, color) {
    if (!points || points.length < 2) return null;
    var W = 160, H = 44, PAD_TOP = 4, PAD_BOT = 14, PAD_L = 4, PAD_R = 4;
    var plotW = W - PAD_L - PAD_R;
    var plotH = H - PAD_TOP - PAD_BOT;
    var velos = points.map(function(p) { return p.avgVelo; });
    var avgV = velos.reduce(function(a, b) { return a + b; }, 0) / velos.length;
    var dataMin = Math.min.apply(null, velos);
    var dataMax = Math.max.apply(null, velos);
    // Use at least ±3 mph from mean for consistent scale across pitch types
    var minV = Math.min(dataMin, avgV - 3);
    var maxV = Math.max(dataMax, avgV + 3);
    var range = maxV - minV || 1;

    var ns = 'http://www.w3.org/2000/svg';

    // Wrapper div for sparkline + tooltip
    var wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:inline-block;position:relative;vertical-align:middle;';

    var svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('width', W);
    svg.setAttribute('height', H);
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.style.display = 'block';
    svg.style.cursor = 'crosshair';

    // Season average reference line (dashed)
    var avgY = PAD_TOP + (1 - (avgV - minV) / range) * plotH;
    var avgLine = document.createElementNS(ns, 'line');
    avgLine.setAttribute('x1', PAD_L);
    avgLine.setAttribute('y1', avgY.toFixed(1));
    avgLine.setAttribute('x2', PAD_L + plotW);
    avgLine.setAttribute('y2', avgY.toFixed(1));
    avgLine.setAttribute('stroke', '#8a7f75');
    avgLine.setAttribute('stroke-width', '0.75');
    avgLine.setAttribute('stroke-dasharray', '3,2');
    svg.appendChild(avgLine);

    // Build path and compute coordinates
    var coords = [];
    var pathParts = [];
    for (var i = 0; i < points.length; i++) {
      var x = PAD_L + (i / (points.length - 1)) * plotW;
      var y = PAD_TOP + (1 - (points[i].avgVelo - minV) / range) * plotH;
      coords.push({ x: x, y: y });
      pathParts.push((i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1));
    }

    var path = document.createElementNS(ns, 'path');
    path.setAttribute('d', pathParts.join(' '));
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', color || '#888');
    path.setAttribute('stroke-width', '1.5');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(path);

    // End dot
    var lastCoord = coords[coords.length - 1];
    var dot = document.createElementNS(ns, 'circle');
    dot.setAttribute('cx', lastCoord.x.toFixed(1));
    dot.setAttribute('cy', lastCoord.y.toFixed(1));
    dot.setAttribute('r', '2.5');
    dot.setAttribute('fill', color || '#888');
    svg.appendChild(dot);

    // Date axis labels (first and last date, formatted as M/D)
    var formatDate = function(d) {
      var parts = d.split('-');
      return parseInt(parts[1]) + '/' + parseInt(parts[2]);
    };
    var startLabel = document.createElementNS(ns, 'text');
    startLabel.setAttribute('x', PAD_L);
    startLabel.setAttribute('y', H - 1);
    startLabel.setAttribute('fill', '#6a5f55');
    startLabel.setAttribute('font-size', '9');
    startLabel.setAttribute('font-family', 'Barlow, sans-serif');
    startLabel.textContent = formatDate(points[0].date);
    svg.appendChild(startLabel);

    var endLabel = document.createElementNS(ns, 'text');
    endLabel.setAttribute('x', PAD_L + plotW);
    endLabel.setAttribute('y', H - 1);
    endLabel.setAttribute('text-anchor', 'end');
    endLabel.setAttribute('fill', '#6a5f55');
    endLabel.setAttribute('font-size', '9');
    endLabel.setAttribute('font-family', 'Barlow, sans-serif');
    endLabel.textContent = formatDate(points[points.length - 1].date);
    svg.appendChild(endLabel);

    // Invisible hover rects for each data point (tooltip targets)
    var tooltip = document.createElement('div');
    tooltip.style.cssText = 'position:absolute;display:none;background:#2a241d;border:1px solid #6a5f55;border-radius:4px;padding:3px 7px;font-size:11px;color:#f0e8d8;white-space:nowrap;pointer-events:none;z-index:10;font-family:IBM Plex Sans,sans-serif;font-variant-numeric:tabular-nums;';
    wrapper.appendChild(tooltip);

    var segW = plotW / (points.length - 1 || 1);
    for (var j = 0; j < points.length; j++) {
      (function(idx) {
        var hoverRect = document.createElementNS(ns, 'rect');
        var rx = coords[idx].x - segW / 2;
        hoverRect.setAttribute('x', Math.max(0, rx).toFixed(1));
        hoverRect.setAttribute('y', '0');
        hoverRect.setAttribute('width', segW.toFixed(1));
        hoverRect.setAttribute('height', H);
        hoverRect.setAttribute('fill', 'transparent');
        hoverRect.style.cursor = 'crosshair';

        // Visible hover dot
        var hDot = document.createElementNS(ns, 'circle');
        hDot.setAttribute('cx', coords[idx].x.toFixed(1));
        hDot.setAttribute('cy', coords[idx].y.toFixed(1));
        hDot.setAttribute('r', '3');
        hDot.setAttribute('fill', color || '#888');
        hDot.setAttribute('stroke', '#f0e8d8');
        hDot.setAttribute('stroke-width', '1');
        hDot.style.display = 'none';
        svg.appendChild(hDot);

        hoverRect.addEventListener('mouseenter', function() {
          hDot.style.display = '';
          tooltip.style.display = 'block';
          tooltip.textContent = formatDate(points[idx].date) + '  ' + points[idx].avgVelo.toFixed(1) + ' mph';
          // Position tooltip above the point
          var tipX = coords[idx].x - 30;
          if (tipX < 0) tipX = 0;
          if (tipX > W - 80) tipX = W - 80;
          tooltip.style.left = tipX + 'px';
          tooltip.style.top = (coords[idx].y - 22) + 'px';
        });
        hoverRect.addEventListener('mouseleave', function() {
          hDot.style.display = 'none';
          tooltip.style.display = 'none';
        });
        svg.appendChild(hoverRect);
      })(j);
    }

    // Overall tooltip on the SVG (range + avg)
    svg.setAttribute('title', 'Range: ' + minV.toFixed(1) + '–' + maxV.toFixed(1) + ' mph | Avg: ' + avgV.toFixed(1) + ' mph');

    wrapper.appendChild(svg);
    return wrapper;
  },

  _renderExpandedPitchTable: function (data) {
    var section = document.getElementById('player-expanded-pitch-section');
    var container = document.getElementById('player-expanded-pitch-table');
    container.innerHTML = '';

    var pitchRows = this._filteredPitchRows || this._getPitchRows(data.pitcher, data.team);
    if (pitchRows.length === 0) { if (section) section.style.display = 'none'; return; }

    section.style.display = '';

    // Get velocity trend data for sparklines
    var veloTrend = Aggregator.loaded ? Aggregator.getVeloTrend(data.pitcher, data.team) : {};

    var cols = this.EXPANDED_PITCH_COLS;
    if (this._isROC) {
      cols = cols.filter(function (c) { return !c.rocHide; });
    }

    var table = document.createElement('table');
    table.className = 'player-pitch-stats-table expanded-pitch-table';

    // Header
    var thead = document.createElement('thead');
    var headerRow = document.createElement('tr');
    for (var i = 0; i < cols.length; i++) {
      var th = document.createElement('th');
      th.textContent = cols[i].label;
      headerRow.appendChild(th);
    }
    // Add Velo Trend column header
    var thVt = document.createElement('th');
    thVt.textContent = 'Velo Trend';
    thVt.style.minWidth = '130px';
    headerRow.appendChild(thVt);
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    var tbody = document.createElement('tbody');
    for (var r = 0; r < pitchRows.length; r++) {
      var row = pitchRows[r];
      var tr = document.createElement('tr');
      for (var c = 0; c < cols.length; c++) {
        var col = cols[c];
        var td = document.createElement('td');
        if (col.key === 'pitchType') {
          var badge = Utils.createPitchBadge(row.pitchType, true);
          td.appendChild(badge);
        } else {
          var val = row[col.key];
          td.textContent = col.format ? col.format(val) : (val != null ? val : '—');
          // Pitching+ run-value companion on hover (this table has no custom
          // tooltip layer like the leaderboard, so use a native title). toFixed
          // keeps '-' on negatives, adds no '+' on positives (house rule).
          if (col.key === 'pitchingScore' && row.pitchingRuns100 != null) {
            td.title = parseFloat(row.pitchingRuns100).toFixed(2) + ' runs/100 vs avg';
          }
          // Pitcher+ companion reads 'expected', not 'vs avg': its slope is
          // predictive (next-season xRV/100 on this-season Pitcher+), not
          // the same-season fit pitchingRuns100 uses.
          if (col.key === 'pitcherPlus' && row.pitcherRuns100 != null) {
            td.title = parseFloat(row.pitcherRuns100).toFixed(2) + ' runs/100 expected';
          }
        }
        tr.appendChild(td);
      }
      // Add sparkline cell
      var tdVt = document.createElement('td');
      tdVt.style.textAlign = 'center';
      var trendData = veloTrend[row.pitchType];
      var sparkColor = Utils.getPitchColor(row.pitchType);
      var sparkline = this._createVeloSparkline(trendData, sparkColor);
      if (sparkline) {
        tdVt.appendChild(sparkline);
      } else {
        tdVt.textContent = '—';
      }
      tr.appendChild(tdVt);
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    container.appendChild(table);
  },


  _renderStatsTable: function (data) {
    var section = document.getElementById('player-stats-section');
    var container = document.getElementById('player-stats-table');
    container.innerHTML = '';

    if (!data) { if (section) section.style.display = 'none'; return; }
    section.style.display = '';

    // Platoon mode swaps in a column set where every cell actually splits.
    var hand = this._platoonHand || 'all';
    var cols = hand === 'all' ? this.STATS_COLS : this.PLATOON_STATS_COLS;

    var table = document.createElement('table');
    table.className = 'player-pitch-stats-table expanded-pitch-table';

    var thead = document.createElement('thead');
    var headerRow = document.createElement('tr');
    for (var i = 0; i < cols.length; i++) {
      var th = document.createElement('th');
      th.textContent = cols[i].label;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    var tr = document.createElement('tr');
    for (var c = 0; c < cols.length; c++) {
      var col = cols[c];
      var td = document.createElement('td');
      var val = data[col.key];
      td.textContent = col.format ? col.format(val) : (val != null ? val : '—');
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
    table.appendChild(tbody);
    container.appendChild(table);
  },


  // Pool-wide platoon aggregations are expensive (~350 ms) and identical for
  // every pitcher, so they're computed once per session and reused. Filters
  // match _refreshPlatoonStats exactly so both share the cache.
  PLATOON_PT_REQUEST: ['Hard', 'Breaking', 'Offspeed', 'FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'SV', 'CH', 'FS', 'KN'],

  _platoonFilters: function (hand) {
    // includeROC keeps ROC hitters in the returned rows (their percentiles
    // interpolate vs the MLB pool inside the aggregator), so ROC hitter pages
    // can platoon-split. Pitcher/pitch narrowings ignore the flag, so pitcher
    // pages are unchanged. Cache stays shared across MLB and ROC pages.
    return { vsHand: hand, team: 'all', throws: 'all', search: '', role: 'all',
             pitchTypes: this.PLATOON_PT_REQUEST, includeROC: true };
  },

  _platoonAggregate: function (tab, hand) {
    if (!Aggregator.loaded) return null;
    // Drop the cache if the aggregator was reloaded under us.
    if (this._platoonAggData !== Aggregator.data) {
      this._platoonAggCache = null;
      this._platoonAggData = Aggregator.data;
    }
    if (!this._platoonAggCache) this._platoonAggCache = {};
    var key = tab + '|' + hand;
    if (!this._platoonAggCache[key]) {
      this._platoonAggCache[key] = Aggregator.aggregate(tab, this._platoonFilters(hand));
    }
    return this._platoonAggCache[key];
  },

  // On the pitcher side wOBA/xwOBA/xBA/xSLG are boxscore-merged: the aggregator
  // swaps in the _vsL/_vsR VALUE but carries the season _pctl alongside it, so
  // coloring a split cell with that rank would paint the wrong number. Rank
  // them within the same filtered pool instead, under a panel-local key so
  // nothing else on the page changes. Pool is every player in the aggregation
  // (house rule: all MLB defines the distribution, qualification gates color).
  // Hitters pass an empty list — every column in their platoon set is computed
  // from micro, so the aggregator's own _pctl already matches the value shown.
  _ensurePlatoonPctls: function (rows, keys) {
    if (!rows || !keys || !keys.length || rows._platoonPctlsDone) return;
    for (var ki = 0; ki < keys.length; ki++) {
      var k = keys[ki];
      var pk = k + '_pctl';
      var saved = [];
      for (var s = 0; s < rows.length; s++) saved.push(rows[s][pk]);
      Aggregator._computePercentiles(rows, k);
      for (var r = 0; r < rows.length; r++) {
        var v = rows[r][pk];
        // Lower is better against a pitcher, so invert like the aggregator's
        // own INVERT pass does for the stats it ranks itself.
        rows[r][k + '_platoonPctl'] = (v == null) ? null : 100 - v;
        rows[r][pk] = saved[r];
      }
    }
    rows._platoonPctlsDone = true;
  },

  _findPlatoonRow: function (tab, hand, nameKey, name, team) {
    var rows = this._platoonAggregate(tab, hand);
    if (!rows) return null;
    for (var i = 0; i < rows.length; i++) {
      if (rows[i][nameKey] === name && rows[i].team === team) return rows[i];
    }
    return null;
  },


  // One config per player type; the panel markup and render path are shared.
  PLATOON_SPLIT_CFG: {
    pitcher: {
      sectionId: 'player-platoon-split-section',
      tableId:   'player-platoon-split-table',
      tab: 'pitcher', nameKey: 'pitcher', colsKey: 'PLATOON_STATS_COLS',
      rerank: ['wOBA', 'xwOBA', 'xBA', 'xSLG'],
      labelL: 'vs LHH', labelR: 'vs RHH',
    },
    hitter: {
      sectionId: 'player-hitter-platoon-split-section',
      tableId:   'player-hitter-platoon-split-table',
      tab: 'hitter', nameKey: 'hitter', colsKey: 'HITTER_PLATOON_STATS_COLS',
      rerank: [],
      labelL: 'vs LHP', labelR: 'vs RHP',
    },
  },

  // The two pool-wide aggregations behind the split panel cost ~700 ms the
  // first time (cached for the rest of the session), so the panel fills in
  // after the page paints rather than blocking it.
  _renderPlatoonSplitDeferred: function (type, data) {
    var cfg = this.PLATOON_SPLIT_CFG[type];
    var section = document.getElementById(cfg.sectionId);
    var container = document.getElementById(cfg.tableId);
    if (!section || !container) return;
    if (!Aggregator.loaded) { section.style.display = 'none'; return; }

    // Cache warm: render synchronously so the panel doesn't flash a skeleton.
    if (this._platoonAggData === Aggregator.data && this._platoonAggCache &&
        this._platoonAggCache[cfg.tab + '|L'] && this._platoonAggCache[cfg.tab + '|R']) {
      this._renderPlatoonSplit(type, data);
      return;
    }

    section.style.display = '';
    container.innerHTML = '<p class="platoon-split-loading">Computing platoon splits…</p>';
    var self = this;
    var token = (this._platoonSplitToken = (this._platoonSplitToken || 0) + 1);
    setTimeout(function () {
      // A different player was opened while this was queued — drop it.
      if (token !== self._platoonSplitToken) return;
      var cur = self._currentData;
      if (!cur || cur[cfg.nameKey] !== data[cfg.nameKey] || cur.team !== data.team) return;
      self._renderPlatoonSplit(type, data);
    }, 50);
  },


  // Side-by-side vs LHH/RHH (pitchers) or vs LHP/RHP (hitters) with the L-R
  // difference underneath, so the platoon gap reads at a glance instead of
  // requiring the toggle to be flipped back and forth. The row matching the
  // active toggle is highlighted.
  _renderPlatoonSplit: function (type, data) {
    var cfg = this.PLATOON_SPLIT_CFG[type];
    var section = document.getElementById(cfg.sectionId);
    var container = document.getElementById(cfg.tableId);
    if (!section || !container) return;
    container.innerHTML = '';

    if (!Aggregator.loaded) { section.style.display = 'none'; return; }

    var cols = this[cfg.colsKey];
    var name = data[cfg.nameKey];
    this._ensurePlatoonPctls(this._platoonAggregate(cfg.tab, 'L'), cfg.rerank);
    this._ensurePlatoonPctls(this._platoonAggregate(cfg.tab, 'R'), cfg.rerank);
    var vsL = this._findPlatoonRow(cfg.tab, 'L', cfg.nameKey, name, data.team);
    var vsR = this._findPlatoonRow(cfg.tab, 'R', cfg.nameKey, name, data.team);
    if (!vsL && !vsR) { section.style.display = 'none'; return; }
    section.style.display = '';

    var table = document.createElement('table');
    table.className = 'player-pitch-stats-table expanded-pitch-table platoon-split-table';

    var thead = document.createElement('thead');
    var headerRow = document.createElement('tr');
    var thLabel = document.createElement('th');
    thLabel.textContent = '';
    headerRow.appendChild(thLabel);
    for (var i = 0; i < cols.length; i++) {
      var th = document.createElement('th');
      th.textContent = cols[i].label;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    var self = this;
    function valueRow(label, row, hand) {
      var tr = document.createElement('tr');
      if (self._platoonHand === hand) tr.className = 'platoon-split-active';
      var tdLabel = document.createElement('td');
      tdLabel.className = 'platoon-split-label';
      tdLabel.textContent = label;
      tr.appendChild(tdLabel);
      for (var c = 0; c < cols.length; c++) {
        var col = cols[c];
        var td = document.createElement('td');
        var val = row ? row[col.key] : null;
        td.textContent = col.format ? col.format(val) : (val != null ? val : '—');
        // Percentile coloring uses the same-handedness pool, which is the
        // honest comparison group for a split. Qualification gates color on
        // the SEASON total (IP for pitchers, paAll for hitters), not on the
        // split's own sample — the convention the leaderboard already uses.
        var pctl = null;
        if (!col.noPctl && row) {
          pctl = row[col.key + '_platoonPctl'];
          if (pctl === undefined) pctl = row[col.key + '_pctl'];
        }
        if (pctl != null && val != null && row._qualified) {
          td.style.backgroundColor = Utils.percentileColor(pctl);
          td.style.color = Utils.percentileTextColor(pctl);
        }
        tr.appendChild(td);
      }
      return tr;
    }

    var tbody = document.createElement('tbody');
    // vs R on top (2026-08-06, per Wally) — the majority split reads first.
    tbody.appendChild(valueRow(cfg.labelR, vsR, 'R'));
    tbody.appendChild(valueRow(cfg.labelL, vsL, 'L'));

    // Difference row. House format rule: no leading "+" on positives, so the
    // header spells out the direction rather than relying on signs.
    var diffRow = document.createElement('tr');
    diffRow.className = 'platoon-split-diff';
    var tdDiff = document.createElement('td');
    tdDiff.className = 'platoon-split-label';
    tdDiff.textContent = 'L − R';
    diffRow.appendChild(tdDiff);
    for (var d = 0; d < cols.length; d++) {
      var dcol = cols[d];
      var td2 = document.createElement('td');
      var lv = vsL ? vsL[dcol.key] : null;
      var rv = vsR ? vsR[dcol.key] : null;
      if (lv == null || rv == null || dcol.noDiff) {
        // Counting stats (TBF/PA/AB/HR) differ by playing time, not by skill —
        // blank them rather than invite the gap to be read as a split.
        td2.textContent = '—';
      } else if (dcol.dec3) {
        // col.format's /^0/ strip misses negatives, so sign and magnitude
        // are formatted separately to keep ".022" / "-.022" symmetric.
        var diff = lv - rv;
        td2.textContent = (diff < 0 ? '-' : '') + Math.abs(diff).toFixed(3).replace(/^0/, '');
      } else {
        td2.textContent = dcol.format ? dcol.format(lv - rv) : String(lv - rv);
      }
      diffRow.appendChild(td2);
    }
    tbody.appendChild(diffRow);
    table.appendChild(tbody);
    container.appendChild(table);
  },


  _renderBattedBallTable: function (data) {
    var section = document.getElementById('player-batted-ball-section');
    var container = document.getElementById('player-batted-ball-table');
    container.innerHTML = '';

    var pitchRows = this._filteredPitchRows || this._getPitchRows(data.pitcher, data.team);
    if (pitchRows.length === 0) { if (section) section.style.display = 'none'; return; }
    section.style.display = '';

    // Compute total row from filtered pitcher-level data (include _pctl keys)
    var totalRow = { pitchType: 'Total' };
    var fd = this._filteredData || data;
    for (var k = 0; k < this.BATTED_BALL_COLS.length; k++) {
      var key = this.BATTED_BALL_COLS[k].key;
      if (key !== 'pitchType') {
        totalRow[key] = fd[key];
        if (fd[key + '_pctl'] != null) totalRow[key + '_pctl'] = fd[key + '_pctl'];
      }
    }

    this._renderPerPitchTable(container, this.BATTED_BALL_COLS, pitchRows, totalRow);
  },


  _renderPlateDisciplineTable: function (data) {
    var section = document.getElementById('player-plate-discipline-section');
    var container = document.getElementById('player-plate-discipline-table');
    container.innerHTML = '';

    var pitchRows = this._filteredPitchRows || this._getPitchRows(data.pitcher, data.team);
    if (pitchRows.length === 0) { if (section) section.style.display = 'none'; return; }
    section.style.display = '';

    // Compute total row from filtered pitcher-level data (include _pctl keys)
    var totalRow = { pitchType: 'Total' };
    var fd = this._filteredData || data;
    for (var k = 0; k < this.PLATE_DISCIPLINE_COLS.length; k++) {
      var key = this.PLATE_DISCIPLINE_COLS[k].key;
      if (key !== 'pitchType') {
        totalRow[key] = fd[key];
        if (fd[key + '_pctl'] != null) totalRow[key + '_pctl'] = fd[key + '_pctl'];
      }
    }

    this._renderPerPitchTable(container, this.PLATE_DISCIPLINE_COLS, pitchRows, totalRow);
  },


  _renderPerPitchTable: function (container, cols, pitchRows, totalRow) {
    var table = document.createElement('table');
    table.className = 'player-pitch-stats-table expanded-pitch-table';

    // Header
    var thead = document.createElement('thead');
    var headerRow = document.createElement('tr');
    for (var i = 0; i < cols.length; i++) {
      var th = document.createElement('th');
      th.textContent = cols[i].label;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body — pitch type rows
    var tbody = document.createElement('tbody');
    for (var r = 0; r < pitchRows.length; r++) {
      var row = pitchRows[r];
      var rowInPool = (row.count || 0) >= QUAL.MIN_HITTER_PT;
      var tr = document.createElement('tr');
      for (var c = 0; c < cols.length; c++) {
        var col = cols[c];
        var td = document.createElement('td');
        if (col.key === 'pitchType') {
          var badge = Utils.createPitchBadge(row.pitchType, true);
          td.appendChild(badge);
        } else {
          var val = row[col.key];
          td.textContent = col.format ? col.format(val) : (val != null ? val : '—');
          // Apply percentile coloring if a _pctl value exists for this key.
          // Sub-minimum rows keep the tooltip rank but skip background color.
          var pctl = col.noPctl ? null : row[col.key + '_pctl'];
          if (pctl != null && val != null) {
            if (rowInPool) {
              var bgColor = Utils.percentileColor(pctl);
              var txtColor = Utils.percentileTextColor(pctl);
              td.style.backgroundColor = bgColor;
              td.style.color = txtColor;
            }
            td.title = Math.round(pctl) + 'th percentile';
          }
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }

    // Total row
    if (totalRow) {
      var totalTr = document.createElement('tr');
      totalTr.style.fontWeight = '700';
      totalTr.style.borderTop = '2px solid #9f3026';
      for (var c2 = 0; c2 < cols.length; c2++) {
        var col2 = cols[c2];
        var td2 = document.createElement('td');
        if (col2.key === 'pitchType') {
          td2.textContent = 'Total';
        } else {
          var val2 = totalRow[col2.key];
          td2.textContent = col2.format ? col2.format(val2) : (val2 != null ? val2 : '—');
          // Apply percentile coloring to total row (league-wide percentiles)
          var pctl2 = col2.noPctl ? null : totalRow[col2.key + '_pctl'];
          if (pctl2 != null && val2 != null) {
            var bgColor2 = Utils.percentileColor(pctl2);
            var txtColor2 = Utils.percentileTextColor(pctl2);
            td2.style.backgroundColor = bgColor2;
            td2.style.color = txtColor2;
            td2.title = Math.round(pctl2) + 'th percentile';
          }
        }
        totalTr.appendChild(td2);
      }
      tbody.appendChild(totalTr);
    }

    table.appendChild(tbody);
    container.appendChild(table);

    Utils.addScrollFade(container);
  },


  _renderHeatMaps: function(data) {
    var section = document.getElementById('player-location-section');
    var grid = document.getElementById('player-heatmap-grid');
    grid.innerHTML = '';

    var pitches = this._getFilteredDetails(data);
    if (!pitches || pitches.length === 0) { section.style.display = 'none'; return; }

    section.style.display = '';
    this._renderLocCommandMap(data);
    var hand = this._heatMapHand || 'R';

    // Compute average strike zone across ALL pitches (not per-type)
    var szTopSum = 0, szBotSum = 0, szCount = 0;
    for (var i = 0; i < pitches.length; i++) {
      if (pitches[i].szt != null && pitches[i].szb != null) {
        szTopSum += pitches[i].szt;
        szBotSum += pitches[i].szb;
        szCount++;
      }
    }
    var avgSzTop = szCount > 0 ? szTopSum / szCount : 3.5;
    var avgSzBot = szCount > 0 ? szBotSum / szCount : 1.5;

    // Group by pitch type, filter by hand
    var byType = {};
    for (var i = 0; i < pitches.length; i++) {
      var p = pitches[i];
      if (p.bh !== hand) continue;
      if (p.px == null || p.pz == null) continue;
      if (!byType[p.pt]) byType[p.pt] = [];
      byType[p.pt].push(p);
    }

    // Sort pitch types by fixed order
    var types = Utils.sortPitchTypes(Object.keys(byType));

    // Total pitches against the selected hand — denominator for usage %.
    var totalForHand = 0;
    for (var totKey in byType) totalForHand += byType[totKey].length;

    // Render each pitch type
    for (var t = 0; t < types.length; t++) {
      var pt = types[t];
      var cell = document.createElement('div');
      cell.className = 'heatmap-cell';

      var label = document.createElement('div');
      label.className = 'heatmap-label';
      var badge = Utils.createPitchBadge(pt, true);
      label.appendChild(badge);
      // Sample-size + usage meta: "[ST] · 112 (57.1%)" — count first as the
      // trustworthiness cue, usage % in parens for context.
      var meta = document.createElement('span');
      meta.className = 'heatmap-label-meta';
      var ptCount = byType[pt].length;
      var ptPct = totalForHand > 0 ? (ptCount / totalForHand * 100) : 0;
      meta.textContent = '· ' + ptCount + ' (' + ptPct.toFixed(1) + '%)';
      label.appendChild(meta);
      cell.appendChild(label);

      var canvas = document.createElement('canvas');
      canvas.width = 250;
      canvas.height = 300;
      cell.appendChild(canvas);

      grid.appendChild(cell);
      this._renderSingleHeatMap(canvas, byType[pt], avgSzTop, avgSzBot, hand);
    }

    // Legend removed — blue-to-red heat scale is intuitive
  },

  // Loc+ command map: the pitcher's own pitch locations colored by the league
  // EXPECTED hitter-perspective run value of each spot (blue = good location
  // for the pitcher, red = costly), opacity by how often he throws there.
  // Season-level (not refiltered); reads the per-pitcher heatmap grid shipped
  // by pipeline_locplus.py.
  _renderLocCommandMap: function(data) {
    var wrap = document.getElementById('player-command-map');
    if (!wrap) return;
    var hm = data && data.locPlusHeatmap;
    if (!hm || !hm.length) { wrap.style.display = 'none'; wrap.innerHTML = ''; return; }
    wrap.style.display = '';
    wrap.innerHTML = '';

    var label = document.createElement('div');
    label.className = 'heatmap-label';
    var title = document.createElement('span');
    title.textContent = 'Loc+ Command Map (season, all hitters)';
    label.appendChild(title);
    if (data.locPlus != null) {
      var meta = document.createElement('span');
      meta.className = 'heatmap-label-meta';
      // No leading + on positive runs/100 (house number-format rule).
      var rr = (data.locRuns100 != null) ? (' · ' + data.locRuns100.toFixed(2) + ' runs/100') : '';
      meta.textContent = '· ' + Math.round(data.locPlus) + rr;
      label.appendChild(meta);
    }
    wrap.appendChild(label);

    var canvas = document.createElement('canvas');
    canvas.width = 250;
    canvas.height = 300;
    wrap.appendChild(canvas);

    var note = document.createElement('div');
    note.className = 'command-map-note';
    // Season-only by construction: pipeline_locplus.py ships one grid per
    // pitcher, not one per batter hand, so the toggle above cannot move it.
    note.textContent = 'blue = good location · red = costly · opacity = usage · does not follow the vs LHH/RHH toggle';
    wrap.appendChild(note);

    this._renderLocValueCanvas(canvas, hm);
  },

  _renderLocValueCanvas: function(canvas, hm) {
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;

    // Grid mirrors pipeline_locplus.py: x in feet, z is zone-normalized.
    var X_MIN = -1.5, BIN_X = 2 / 12, Z_MIN = -0.6, BIN_Z = 0.10;
    // Display bounds (a little padding beyond the zone).
    var xMin = -1.7, xMax = 1.7, znMin = -0.5, znMax = 1.6;
    function cx(px) { return ((px - xMin) / (xMax - xMin)) * W; }
    function cy(zn) { return ((znMax - zn) / (znMax - znMin)) * H; } // flip: top = high z

    ctx.fillStyle = '#e8dfcb';
    ctx.fillRect(0, 0, W, H);

    var nMax = 1;
    for (var k = 0; k < hm.length; k++) { if (hm[k][3] > nMax) nMax = hm[k][3]; }
    // Diverging domain on hitter-perspective xRV: low = good (blue), high = bad (red).
    var LO = -0.06, HI = 0.06;
    for (var k = 0; k < hm.length; k++) {
      var i = hm[k][0], j = hm[k][1], val = hm[k][2], n = hm[k][3];
      var pxc = X_MIN + (i + 0.5) * BIN_X;
      var znc = Z_MIN + (j + 0.5) * BIN_Z;
      var x0 = cx(pxc - BIN_X / 2), x1 = cx(pxc + BIN_X / 2);
      var y0 = cy(znc + BIN_Z / 2), y1 = cy(znc - BIN_Z / 2);
      var t = (val - LO) / (HI - LO);
      if (t < 0) t = 0; else if (t > 1) t = 1;
      ctx.globalAlpha = 0.2 + 0.8 * Math.sqrt(n / nMax);
      ctx.fillStyle = this._heatColor(t);
      ctx.fillRect(x0, y0, (x1 - x0) + 0.5, (y1 - y0) + 0.5);
    }
    ctx.globalAlpha = 1;

    // Strike zone (zone-normalized 0..1; plate half-width ~0.83 ft)
    var zl = cx(-0.83), zr = cx(0.83), zt = cy(1), zb = cy(0);
    ctx.strokeStyle = 'rgba(58,48,38,0.65)';
    ctx.lineWidth = 2;
    ctx.strokeRect(zl, zt, zr - zl, zb - zt);
    ctx.strokeStyle = 'rgba(58,48,38,0.28)';
    ctx.lineWidth = 1;
    var tw = (zr - zl) / 3, th = (zb - zt) / 3;
    for (var ti = 1; ti < 3; ti++) {
      ctx.beginPath(); ctx.moveTo(zl + ti * tw, zt); ctx.lineTo(zl + ti * tw, zb); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(zl, zt + ti * th); ctx.lineTo(zr, zt + ti * th); ctx.stroke();
    }

    var py = cy(-0.4), pcx = cx(0);
    ctx.fillStyle = 'rgba(58,48,38,0.18)';
    ctx.beginPath();
    ctx.moveTo(pcx - 8, py); ctx.lineTo(pcx + 8, py); ctx.lineTo(pcx + 5, py + 5);
    ctx.lineTo(pcx, py + 8); ctx.lineTo(pcx - 5, py + 5); ctx.closePath(); ctx.fill();
  },

  _renderSingleHeatMap: function(canvas, pitches, szTop, szBot, hand) {
    var ctx = canvas.getContext('2d');
    var W = canvas.width;
    var H = canvas.height;

    // Plot bounds in feet
    var xMin = -2.0, xMax = 2.0;
    var zMin = 0.5, zMax = 4.5;

    // Grid resolution
    var gridX = 50, gridZ = 60;
    var cellW = W / gridX;
    var cellH = H / gridZ;
    var bw = 0.25; // bandwidth in feet
    var bw2 = 2 * bw * bw;

    // Compute KDE density grid
    var density = new Array(gridX * gridZ);
    var maxDensity = 0;
    for (var gx = 0; gx < gridX; gx++) {
      for (var gz = 0; gz < gridZ; gz++) {
        var cx = xMin + (gx + 0.5) * (xMax - xMin) / gridX;
        var cz = zMax - (gz + 0.5) * (zMax - zMin) / gridZ; // flip z axis (top = high z)
        var d = 0;
        for (var p = 0; p < pitches.length; p++) {
          var dx = cx - pitches[p].px;
          var dz = cz - pitches[p].pz;
          d += Math.exp(-(dx * dx + dz * dz) / bw2);
        }
        var idx = gx + gz * gridX;
        density[idx] = d;
        if (d > maxDensity) maxDensity = d;
      }
    }

    // Clear canvas
    ctx.fillStyle = '#e8dfcb';
    ctx.fillRect(0, 0, W, H);

    // Draw density
    if (maxDensity > 0) {
      for (var gx = 0; gx < gridX; gx++) {
        for (var gz = 0; gz < gridZ; gz++) {
          var norm = density[gx + gz * gridX] / maxDensity;
          if (norm < 0.05) continue;
          ctx.fillStyle = this._heatColor(norm);
          ctx.fillRect(gx * cellW, gz * cellH, cellW + 0.5, cellH + 0.5);
        }
      }
    }

    // Draw strike zone rectangle
    var zoneLeft = ((-0.83 - xMin) / (xMax - xMin)) * W;
    var zoneRight = ((0.83 - xMin) / (xMax - xMin)) * W;
    var zoneTop = ((zMax - szTop) / (zMax - zMin)) * H;
    var zoneBottom = ((zMax - szBot) / (zMax - zMin)) * H;

    ctx.strokeStyle = 'rgba(58,48,38,0.65)';
    ctx.lineWidth = 2;
    ctx.strokeRect(zoneLeft, zoneTop, zoneRight - zoneLeft, zoneBottom - zoneTop);

    // Inner zone thirds — quick "did this pitch hit the up-and-in / down-the-
    // middle / etc." quadrant read without counting pixels. Subtle alpha so
    // the gridlines don't compete with the heat data.
    ctx.strokeStyle = 'rgba(58,48,38,0.28)';
    ctx.lineWidth = 1;
    var thirdW = (zoneRight - zoneLeft) / 3;
    var thirdH = (zoneBottom - zoneTop) / 3;
    for (var ti = 1; ti < 3; ti++) {
      ctx.beginPath();
      ctx.moveTo(zoneLeft + ti * thirdW, zoneTop);
      ctx.lineTo(zoneLeft + ti * thirdW, zoneBottom);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(zoneLeft, zoneTop + ti * thirdH);
      ctx.lineTo(zoneRight, zoneTop + ti * thirdH);
      ctx.stroke();
    }

    // Draw home plate at bottom
    var plateY = ((zMax - 0.5) / (zMax - zMin)) * H; // approximate bottom
    var plateCX = W / 2;
    ctx.fillStyle = 'rgba(58,48,38,0.18)';
    ctx.beginPath();
    ctx.moveTo(plateCX - 8, plateY);
    ctx.lineTo(plateCX + 8, plateY);
    ctx.lineTo(plateCX + 5, plateY + 5);
    ctx.lineTo(plateCX, plateY + 8);
    ctx.lineTo(plateCX - 5, plateY + 5);
    ctx.closePath();
    ctx.fill();

  },


  _heatColor: function(t) {
    // Print scale: slate (cold) → paper (mid) → brick (hot). RGB blend from
    // the paper midpoint so heat maps sit on the cream canvases seamlessly
    // (matches the leaderboard percentile scale in Utils.percentileColor).
    var base = [240, 232, 216];                       // paper
    var target = t >= 0.5 ? [176, 64, 47]             // brick (hot)
                          : [66, 100, 138];           // deep slate (cold)
    var p = Math.pow(Math.abs(t - 0.5) / 0.5, 0.9);
    var r = Math.round(base[0] + (target[0] - base[0]) * p);
    var g = Math.round(base[1] + (target[1] - base[1]) * p);
    var b = Math.round(base[2] + (target[2] - base[2]) * p);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  },


  // ────────────────────────────────────────────────────────────────────
  // Hitter swing heat maps — three modes (Swings / Whiffs / Damage) in a
  // single row, platoon-toggleable. Reads per-pitch swing records from
  // window.HITTER_SWING_LOCATIONS (populated by the pipeline).
  // ────────────────────────────────────────────────────────────────────
  _renderHitterSwingHeatMaps: function (data) {
    var section = document.getElementById('player-swing-heatmaps-section');
    var grid = document.getElementById('player-swing-heatmaps-grid');
    if (!section || !grid) return;

    var locs = window.HITTER_SWING_LOCATIONS || {};
    var key = data.hitter + '|' + data.team;
    var hd = locs[key];
    if (!hd || !hd.records || hd.records.length === 0) {
      section.style.display = 'none';
      return;
    }
    section.style.display = '';
    grid.innerHTML = '';

    var hand = this._platoonHand || 'all';
    var szTop = hd.szTop != null ? hd.szTop : 3.5;
    var szBot = hd.szBot != null ? hd.szBot : 1.5;

    // Apply platoon filter
    var filteredSwings = [];
    for (var i = 0; i < hd.records.length; i++) {
      var r = hd.records[i];
      if (hand !== 'all' && r[4] !== hand) continue;
      filteredSwings.push(r);
    }

    // Total pitches faced for this platoon (denominator for Swings %)
    var totalPitches = hand === 'L' ? hd.nL : (hand === 'R' ? hd.nR : hd.nAll);
    var totalSwings = filteredSwings.length;

    // Damage: every BIP with a non-null xwOBA (zeros are legitimate weak
    // contact and DO contribute to the league xwOBAcon — must not filter).
    var damagePts = [];
    var dmgXwSum = 0;
    var whiffPts = [];
    for (var j = 0; j < filteredSwings.length; j++) {
      var rec = filteredSwings[j];
      if (rec[2] === 2) whiffPts.push(rec);
      if (rec[2] === 3 && rec[3] != null) {
        damagePts.push(rec);
        dmgXwSum += rec[3];
      }
    }
    var dmgAvgXwoba = damagePts.length > 0 ? dmgXwSum / damagePts.length : null;
    var dmgFmt = dmgAvgXwoba != null
      ? Number(dmgAvgXwoba).toFixed(3).replace(/^0/, '')
      : '—';

    // Mode definitions — badge color, points, weighting, subtitle.
    // Subtitles call out their denominator explicitly so % values aren't
    // ambiguous (swings = % of pitches; whiffs = % of swings; damage = the
    // xwOBAcon stat itself, BIP count in parens).
    var modes = [
      {
        label: 'SWINGS',
        badgeBg: '#56789b', badgeFg: '#f0e8d8',
        points: filteredSwings,
        weighted: false,
        subtitle: totalPitches > 0
          ? totalSwings + ' (' + (totalSwings / totalPitches * 100).toFixed(1) + '% of pitches)'
          : String(totalSwings)
      },
      {
        label: 'WHIFFS',
        badgeBg: '#d8ccb4', badgeFg: '#1a1612',
        points: whiffPts,
        weighted: false,
        subtitle: totalSwings > 0
          ? whiffPts.length + ' (' + (whiffPts.length / totalSwings * 100).toFixed(1) + '% of swings)'
          : String(whiffPts.length)
      },
      {
        label: 'DAMAGE',
        badgeBg: '#9f3026', badgeFg: '#f0e8d8',
        points: damagePts,
        weighted: true,
        subtitle: 'xwOBAcon: ' + dmgFmt + ' (' + damagePts.length + ' BIP)'
      }
    ];

    for (var mi = 0; mi < modes.length; mi++) {
      var m = modes[mi];
      var cell = document.createElement('div');
      cell.className = 'swing-heatmap-cell';
      // Header (badge + subtitle)
      var labelDiv = document.createElement('div');
      labelDiv.className = 'swing-heatmap-label';
      var badge = document.createElement('span');
      badge.className = 'swing-heatmap-badge';
      badge.style.backgroundColor = m.badgeBg;
      badge.style.color = m.badgeFg;
      badge.textContent = m.label;
      labelDiv.appendChild(badge);
      var meta = document.createElement('span');
      meta.className = 'swing-heatmap-meta';
      meta.textContent = '· ' + m.subtitle;
      labelDiv.appendChild(meta);
      cell.appendChild(labelDiv);
      // Canvas
      var canvas = document.createElement('canvas');
      canvas.width = 320;
      canvas.height = 320;
      cell.appendChild(canvas);
      grid.appendChild(cell);
      this._renderSwingHeatCanvas(canvas, m.points, m.weighted, szTop, szBot);
    }
  },

  // KDE-based density rendering on canvas. Same color palette as the pitcher
  // heat maps, but with alpha-blending so low-density areas fade to the panel
  // background (organic "bleed" look matching the pitcher aesthetic).
  _renderSwingHeatCanvas: function (canvas, points, weighted, szTop, szBot) {
    var ctx = canvas.getContext('2d');
    var W = canvas.width;
    var H = canvas.height;

    // Plot bounds — match pitcher heat maps
    var xMin = -2.0, xMax = 2.0;
    var zMin = 0.5, zMax = 4.5;

    var gridX = 60, gridZ = 60;
    var cellW = W / gridX;
    var cellH = H / gridZ;
    var bw = 0.32; // bandwidth in feet
    var bw2 = 2 * bw * bw;

    // Clear — card panel cream (print redesign)
    ctx.fillStyle = '#e8dfcb';
    ctx.fillRect(0, 0, W, H);

    if (points && points.length >= 5) {
      var density = new Array(gridX * gridZ);
      var maxDensity = 0;
      for (var gx = 0; gx < gridX; gx++) {
        for (var gz = 0; gz < gridZ; gz++) {
          var cx = xMin + (gx + 0.5) * (xMax - xMin) / gridX;
          var cz = zMax - (gz + 0.5) * (zMax - zMin) / gridZ; // flip Z (top = high Z)
          var d = 0;
          for (var p = 0; p < points.length; p++) {
            var dx = cx - points[p][0];
            var dz = cz - points[p][1];
            var w = weighted ? points[p][3] : 1;
            d += w * Math.exp(-(dx * dx + dz * dz) / bw2);
          }
          var idx = gx + gz * gridX;
          density[idx] = d;
          if (d > maxDensity) maxDensity = d;
        }
      }
      if (maxDensity > 0) {
        for (var gx2 = 0; gx2 < gridX; gx2++) {
          for (var gz2 = 0; gz2 < gridZ; gz2++) {
            var norm = density[gx2 + gz2 * gridX] / maxDensity;
            if (norm < 0.04) continue;
            // Alpha curve: density^0.45 — fades low values to panel bg, keeps
            // hot spots saturated. Matches the matplotlib bleed in the mockup.
            var alpha = Math.pow(norm, 0.45);
            ctx.globalAlpha = alpha;
            ctx.fillStyle = this._heatColor(norm);
            ctx.fillRect(gx2 * cellW, gz2 * cellH, cellW + 0.5, cellH + 0.5);
          }
        }
        ctx.globalAlpha = 1.0;
      }
    } else {
      ctx.fillStyle = '#6a5f55';
      ctx.font = '12px IBM Plex Sans, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('n = ' + (points ? points.length : 0) + ' (insufficient)', W / 2, H / 2);
    }

    // Strike zone (white outline + subtle thirds)
    var zoneLeft = ((-0.708 - xMin) / (xMax - xMin)) * W;
    var zoneRight = ((0.708 - xMin) / (xMax - xMin)) * W;
    var zoneTop = ((zMax - szTop) / (zMax - zMin)) * H;
    var zoneBottom = ((zMax - szBot) / (zMax - zMin)) * H;
    ctx.strokeStyle = 'rgba(58,48,38,0.7)';
    ctx.lineWidth = 1.6;
    ctx.strokeRect(zoneLeft, zoneTop, zoneRight - zoneLeft, zoneBottom - zoneTop);
    ctx.strokeStyle = 'rgba(58,48,38,0.28)';
    ctx.lineWidth = 1;
    var thirdW = (zoneRight - zoneLeft) / 3;
    var thirdH = (zoneBottom - zoneTop) / 3;
    for (var ti = 1; ti < 3; ti++) {
      ctx.beginPath();
      ctx.moveTo(zoneLeft + ti * thirdW, zoneTop);
      ctx.lineTo(zoneLeft + ti * thirdW, zoneBottom);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(zoneLeft, zoneTop + ti * thirdH);
      ctx.lineTo(zoneRight, zoneTop + ti * thirdH);
      ctx.stroke();
    }
  },


  _renderCountTable: function(data) {
    var section = document.getElementById('player-count-section');
    var container = document.getElementById('player-count-table');
    container.innerHTML = '';

    var pitches = this._getFilteredDetails(data);
    if (pitches.length === 0) { section.style.display = 'none'; return; }

    section.style.display = '';
    var hand = this._countHand || 'R';

    var COUNT_GROUPS = {
      'First Pitch': ['0-0'],
      'Ahead': ['0-1', '0-2', '1-2'],
      'Behind': ['1-0', '2-0', '3-0', '2-1', '3-1'],
      'Even': ['1-1', '2-2'],
      'Two-Strike': ['0-2', '1-2', '2-2', '3-2']
    };
    var groupNames = ['First Pitch', 'Ahead', 'Behind', 'Even', 'Two-Strike'];

    // Build lookup: count string -> array of group names it belongs to
    var countToGroups = {};
    for (var g = 0; g < groupNames.length; g++) {
      var gn = groupNames[g];
      var counts = COUNT_GROUPS[gn];
      for (var c = 0; c < counts.length; c++) {
        if (!countToGroups[counts[c]]) countToGroups[counts[c]] = [];
        countToGroups[counts[c]].push(gn);
      }
    }

    // Filter by hand and group by pitch type and count group
    var pitchTypes = {};
    var groupTotals = {};
    for (var g = 0; g < groupNames.length; g++) groupTotals[groupNames[g]] = 0;

    for (var i = 0; i < pitches.length; i++) {
      var p = pitches[i];
      if (p.bh !== hand) continue;
      if (!p.cnt) continue;
      var groups = countToGroups[p.cnt];
      if (!groups) continue;

      if (!pitchTypes[p.pt]) {
        pitchTypes[p.pt] = { total: 0 };
        for (var g = 0; g < groupNames.length; g++) pitchTypes[p.pt][groupNames[g]] = 0;
      }
      pitchTypes[p.pt].total++;
      for (var g = 0; g < groups.length; g++) {
        pitchTypes[p.pt][groups[g]]++;
        groupTotals[groups[g]]++;
      }
    }

    // Sort pitch types by total count descending
    var types = Object.keys(pitchTypes);
    types.sort(function(a, b) { return pitchTypes[b].total - pitchTypes[a].total; });

    if (types.length === 0) { section.style.display = 'none'; return; }

    // Compute overall total pitches for usage% column
    var overallTotal = 0;
    for (var t = 0; t < types.length; t++) overallTotal += pitchTypes[types[t]].total;

    // Build HTML table
    var table = document.createElement('table');
    table.className = 'count-table';

    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    var th0 = document.createElement('th');
    th0.textContent = 'Pitch';
    headRow.appendChild(th0);
    var thUsage = document.createElement('th');
    thUsage.textContent = 'Usage%';
    headRow.appendChild(thUsage);
    // Each bucket header gets its sample size appended: "Two-Strike (n=85)" so
    // small-sample buckets are visible at a glance.
    for (var g = 0; g < groupNames.length; g++) {
      var th = document.createElement('th');
      th.textContent = groupNames[g] + ' (n=' + groupTotals[groupNames[g]] + ')';
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    // Row-level coloring gate: a pitch type with fewer than 25 total throws is
    // below the per-pitch sample-size threshold used elsewhere on the site, so
    // its bucket-level deviation values are too noisy to color reliably.
    var MIN_PITCH_TOTAL_FOR_COLOR = 25;
    var tbody = document.createElement('tbody');
    for (var t = 0; t < types.length; t++) {
      var pt = types[t];
      var tr = document.createElement('tr');

      // First column: badge + total-count meta (e.g. "[ST] · 112"). The count
      // is the pitch type's overall total against the selected hand — the
      // sample anchor for the row's deviation pattern.
      var tdLabel = document.createElement('td');
      tdLabel.style.textAlign = 'center';
      var badge = Utils.createPitchBadge(pt, true);
      tdLabel.appendChild(badge);
      var meta = document.createElement('span');
      meta.className = 'count-table-row-meta';
      meta.textContent = ' · ' + pitchTypes[pt].total;
      tdLabel.appendChild(meta);
      tr.appendChild(tdLabel);

      // Baseline usage for this pitch type
      var baselinePct = overallTotal > 0 ? (pitchTypes[pt].total / overallTotal * 100) : 0;
      var rowQualifiesForColor = pitchTypes[pt].total >= MIN_PITCH_TOTAL_FOR_COLOR;

      // Overall usage% column (no coloring — this is the reference)
      var tdUsage = document.createElement('td');
      if (overallTotal > 0) {
        tdUsage.textContent = baselinePct.toFixed(1) + '%';
      } else {
        tdUsage.textContent = '—';
      }
      tdUsage.style.fontWeight = '600';
      tr.appendChild(tdUsage);

      for (var g = 0; g < groupNames.length; g++) {
        var gn = groupNames[g];
        var td = document.createElement('td');
        var total = groupTotals[gn];
        if (total > 0) {
          var pct = (pitchTypes[pt][gn] / total * 100);
          td.textContent = pct.toFixed(1) + '%';
          // Color based on deviation from pitcher's own baseline usage —
          // suppressed for low-total pitch types whose deviation values are
          // too small-sample to read.
          if (baselinePct > 0 && rowQualifiesForColor) {
            var ratio = pct / baselinePct;
            // Map ratio to 0-100 scale: 0x=0, 1x=50, 2x=100
            var devPctl = Math.max(0, Math.min(100, 50 + (ratio - 1) * 50));
            var bgColor = Utils.percentileColor(devPctl);
            var txtColor = Utils.percentileTextColor(devPctl);
            td.style.backgroundColor = bgColor;
            td.style.color = txtColor;
          }
        } else {
          td.textContent = '—';
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    container.appendChild(table);
  },


  _bindHandToggles: function() {
    var self = this;
    // Heat map toggle
    this._heatToggleHandler = function(e) {
      var btn = e.target.closest('.hand-toggle-btn');
      if (!btn) return;
      var hand = btn.getAttribute('data-hand');
      if (hand === self._heatMapHand) return;
      self._heatMapHand = hand;
      var btns = document.querySelectorAll('#hand-toggle .hand-toggle-btn');
      for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
      btn.classList.add('active');
      if (self._currentData) self._renderHeatMaps(self._currentData);
    };
    var heatToggle = document.getElementById('hand-toggle');
    if (heatToggle) heatToggle.addEventListener('click', this._heatToggleHandler);

    // Count table toggle
    this._countToggleHandler = function(e) {
      var btn = e.target.closest('.hand-toggle-btn');
      if (!btn) return;
      var hand = btn.getAttribute('data-hand');
      if (hand === self._countHand) return;
      self._countHand = hand;
      var btns = document.querySelectorAll('#count-hand-toggle .hand-toggle-btn');
      for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
      btn.classList.add('active');
      if (self._currentData) self._renderCountTable(self._currentData);
    };
    var countToggle = document.getElementById('count-hand-toggle');
    if (countToggle) countToggle.addEventListener('click', this._countToggleHandler);
  },

  _unbindHandToggles: function() {
    if (this._heatToggleHandler) {
      var el = document.getElementById('hand-toggle');
      if (el) el.removeEventListener('click', this._heatToggleHandler);
      this._heatToggleHandler = null;
    }
    if (this._countToggleHandler) {
      var el = document.getElementById('count-hand-toggle');
      if (el) el.removeEventListener('click', this._countToggleHandler);
      this._countToggleHandler = null;
    }
  },


  _bindPlatoonToggle: function(type) {
    var self = this;
    var mainToggleId = type === 'pitcher' ? 'pitcher-platoon-toggle' : 'hitter-platoon-toggle';
    // All synced toggle IDs for pitchers
    var syncedIds = type === 'pitcher'
      ? ['pitcher-platoon-toggle', 'pitcher-platedisc-toggle', 'pitcher-battedball-toggle']
      // hitter-battracking-toggle is deliberately absent — bat tracking is
      // season-only in the embed, so that section carries no hand toggle.
      : ['hitter-platoon-toggle', 'hitter-platedisc-toggle', 'hitter-battedball-toggle',
         'hitter-swingheat-toggle'];

    this._platoonSyncedIds = syncedIds;

    this._platoonToggleHandler = function(e) {
      var btn = e.target.closest('.hand-toggle-btn');
      if (!btn) return;
      var hand = btn.getAttribute('data-hand');
      if (hand === self._platoonHand) return;
      self._platoonHand = hand;
      // Sync all toggles
      for (var si = 0; si < syncedIds.length; si++) {
        var btns = document.querySelectorAll('#' + syncedIds[si] + ' .hand-toggle-btn');
        for (var i = 0; i < btns.length; i++) {
          btns[i].classList.toggle('active', btns[i].getAttribute('data-hand') === hand);
        }
      }
      self._refreshPlatoonStats(type);
    };

    for (var ti = 0; ti < syncedIds.length; ti++) {
      var toggle = document.getElementById(syncedIds[ti]);
      if (toggle) toggle.addEventListener('click', this._platoonToggleHandler);
    }
  },

  // ROC/MiLB players have no rows in the micro tables the platoon aggregations
  // read, so every micro-driven hand toggle is a dead control for them — it
  // could only ever render "No data vs LHH". Hide the family instead.
  // The heat-map and usage-by-count toggles read PITCH_DETAILS, which DOES
  // carry ROC, so those stay put.
  _setPlatoonTogglesVisible: function(type, visible) {
    var ids = type === 'pitcher'
      ? ['pitcher-platoon-toggle', 'pitcher-platedisc-toggle', 'pitcher-battedball-toggle']
      : ['hitter-platoon-toggle', 'hitter-platedisc-toggle', 'hitter-battedball-toggle',
         'hitter-swingheat-toggle'];
    for (var i = 0; i < ids.length; i++) {
      var el = document.getElementById(ids[i]);
      if (el) el.style.display = visible ? '' : 'none';
    }
  },

  _resetPlatoonToggleUI: function(toggleId) {
    // Reset all synced toggles
    var syncedIds = this._platoonSyncedIds || [toggleId];
    for (var si = 0; si < syncedIds.length; si++) {
      var btns = document.querySelectorAll('#' + syncedIds[si] + ' .hand-toggle-btn');
      for (var i = 0; i < btns.length; i++) {
        btns[i].classList.toggle('active', btns[i].getAttribute('data-hand') === 'all');
      }
    }
  },

  _unbindPlatoonToggle: function() {
    if (this._platoonToggleHandler) {
      var syncedIds = this._platoonSyncedIds || ['pitcher-platoon-toggle', 'hitter-platoon-toggle'];
      for (var si = 0; si < syncedIds.length; si++) {
        var el = document.getElementById(syncedIds[si]);
        if (el) el.removeEventListener('click', this._platoonToggleHandler);
      }
      this._platoonToggleHandler = null;
      this._platoonSyncedIds = null;
    }
  },

  _refreshPlatoonStats: function(type) {
    var data = this._currentData;
    if (!data) return;

    var hand = this._platoonHand;
    var rocTeams = (DataStore.metadata && DataStore.metadata.rocTeams) || [];
    var isROC = rocTeams.indexOf(data.team) !== -1;

    if (hand === 'all') {
      // Use pre-computed data (no aggregation needed)
      if (type === 'pitcher') {
        var filteredData = this._getFilteredPlayerData(data, true) || data;
        var filteredPitchRows = this._getFilteredPitchRows(data);
        this._filteredData = filteredData;
        this._filteredPitchRows = filteredPitchRows;
        this._renderStatsTable(filteredData);
        this._renderPlatoonSplit('pitcher', data); // clears the active-row highlight
        this._renderExpandedPitchTable(data);
        this._renderBattedBallTable(data);
        this._renderPlateDisciplineTable(data);
        this._renderHeatMaps(data);
        this._renderCountTable(data);
      } else {
        this._renderHitterStatsFullTable(data, isROC);
        this._renderPlatoonSplit('hitter', data); // clears the active-row highlight
        this._renderHitterSwingHeatMaps(data);
        this._renderHitterBattedBallTable(data, isROC);
        this._renderHitterPlateDisciplineTable(data);
        if (!isROC) this._renderHitterBatTrackingTable(data);
      }
      return;
    }

    // Platoon views are driven entirely by the micro aggregator. Without it
    // the toggle can do nothing — say so instead of silently no-opping, and
    // roll the state back so the next click isn't swallowed by the
    // hand === _platoonHand early return in the click handler.
    if (!Aggregator.loaded) {
      var statsId = type === 'pitcher' ? 'player-stats-table' : 'player-hitter-stats-table';
      var statsEl = document.getElementById(statsId);
      if (statsEl) {
        statsEl.innerHTML = '<p style="color:var(--text-secondary);padding:12px;text-align:center;font-size:13px;">' +
          'Splits are still loading — try again in a moment.</p>';
      }
      this._platoonHand = 'all';
      this._resetPlatoonToggleUI(type === 'pitcher' ? 'pitcher-platoon-toggle' : 'hitter-platoon-toggle');
      return;
    }

    // Request both category rows (Hard/Breaking/Offspeed) and individual pitch
    // types in one aggregation pass so the player-page tables can mirror the
    // default-view's grouped/expandable layout in platoon mode too.
    var noDataMsg = '<p style="color:var(--text-secondary);padding:12px;text-align:center;font-size:13px;">No data vs ' +
      (type === 'pitcher' ? (hand === 'L' ? 'LHH' : 'RHH') : (hand === 'L' ? 'LHP' : 'RHP')) + '</p>';

    if (type === 'pitcher') {
      // Re-aggregate pitcher-level stats
      var rows = this._platoonAggregate('pitcher', hand);
      var found = null;
      for (var i = 0; i < rows.length; i++) {
        if (rows[i].pitcher === data.pitcher && rows[i].team === data.team) {
          found = rows[i];
          break;
        }
      }
      if (!found) {
        var container = document.getElementById('player-stats-table');
        if (container) container.innerHTML = noDataMsg;
        return;
      }
      this._filteredData = found;
      this._renderStatsTable(found);
      this._renderPlatoonSplit('pitcher', data); // moves the active-row highlight

      // Re-aggregate pitch-type rows with hand filter
      var pitchRows = this._platoonAggregate('pitch', hand);
      var myPitchRows = [];
      for (var i = 0; i < pitchRows.length; i++) {
        if (pitchRows[i].pitcher === data.pitcher && pitchRows[i].team === data.team) {
          myPitchRows.push(pitchRows[i]);
        }
      }
      myPitchRows.sort(function(a, b) { return (b.usagePct || 0) - (a.usagePct || 0); });
      this._filteredPitchRows = myPitchRows;
      this._renderExpandedPitchTable(data);
      this._renderBattedBallTable(data);
      this._renderPlateDisciplineTable(data);

      // Heat maps, zone profiles, count table use PITCH_DETAILS — filter by batter hand
      this._platoonDetailHand = hand;
      this._renderHeatMaps(data);
      this._renderCountTable(data);
      this._platoonDetailHand = null;

    } else {
      // Re-aggregate hitter-level stats
      var rows = this._platoonAggregate('hitter', hand);
      var found = null;
      for (var i = 0; i < rows.length; i++) {
        if (rows[i].hitter === data.hitter && rows[i].team === data.team) {
          found = rows[i];
          break;
        }
      }
      if (!found) {
        var container = document.getElementById('player-hitter-stats-table');
        if (container) container.innerHTML = noDataMsg;
        return;
      }
      this._renderHitterStatsFullTable(found, isROC);
      this._renderPlatoonSplit('hitter', data); // moves the active-row highlight

      // Re-aggregate hitter pitch-type rows with hand filter, then split into
      // category rows (Hard/Breaking/Offspeed) for the main table view and a
      // sub-rows map for category expansion. Pitch types the hitter hasn't
      // seen vs this handedness are dropped (count == 0).
      var hpRows = this._platoonAggregate('hitterPitch', hand);
      var CATS = Aggregator.PITCH_CATEGORIES;
      var ptToCat = {};
      for (var cn in CATS) {
        for (var ci = 0; ci < CATS[cn].length; ci++) ptToCat[CATS[cn][ci]] = cn;
      }
      var CAT_ORDER = ['Hard', 'Breaking', 'Offspeed'];
      var myCategoryRows = [];
      var subRowsByCat = { Hard: [], Breaking: [], Offspeed: [] };
      for (var i = 0; i < hpRows.length; i++) {
        var r = hpRows[i];
        if (r.hitter !== data.hitter || r.team !== data.team) continue;
        if (r.count == null || r.count === 0) continue;
        if (CAT_ORDER.indexOf(r.pitchType) !== -1) {
          myCategoryRows.push(r);
        } else if (ptToCat[r.pitchType]) {
          subRowsByCat[ptToCat[r.pitchType]].push(r);
        }
      }
      myCategoryRows.sort(function (a, b) {
        return CAT_ORDER.indexOf(a.pitchType) - CAT_ORDER.indexOf(b.pitchType);
      });
      for (var ck = 0; ck < CAT_ORDER.length; ck++) {
        subRowsByCat[CAT_ORDER[ck]].sort(function (a, b) { return (b.count || 0) - (a.count || 0); });
      }
      this._filteredHitterCategoryRows = myCategoryRows;
      this._filteredHitterPitchSubRows = subRowsByCat;
      this._renderHitterBattedBallTable(found, isROC);
      this._renderHitterPlateDisciplineTable(found);
      if (!isROC) this._renderHitterBatTrackingTable(found);
      this._filteredHitterCategoryRows = null;
      this._filteredHitterPitchSubRows = null;
      // Swing heat maps don't go through the aggregator — they read raw
      // pitch-level records from window.HITTER_SWING_LOCATIONS and apply
      // the platoon filter inline based on this._platoonHand.
      this._renderHitterSwingHeatMaps(data);
    }
  },


  _findHitterByMlbId: function (mlbId, preferTeam) {
    return this._findPlayerByMlbId(window.HITTER_DATA || [], mlbId, preferTeam);
  },


  _renderSprayChart: function (data) {
    var canvas = document.getElementById('player-pitch-chart');
    if (!canvas) return;
    // Set canvas size for spray chart — wider than tall to fit foul lines
    canvas.width = 500;
    canvas.height = 420;
    var ctx = canvas.getContext('2d');
    var W = canvas.width;
    var H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    // Home plate in Statcast coords
    var HP_X = 125.42;
    var HP_Y = 198.27;

    // Canvas mapping: HP at bottom center, field fills canvas with padding
    var canvasHPX = W / 2;
    var canvasHPY = H - 15;
    // Max radius so foul lines don't clip: W/2 / cos(45°) with padding
    var maxRadius = (W / 2 - 15) / 0.707;
    var scale = 1.8; // pixels per statcast unit

    function toCanvas(hcX, hcY) {
      var dx = hcX - HP_X;
      var dy = HP_Y - hcY; // Statcast Y increases downward, field Y increases upward
      return [canvasHPX + dx * scale, canvasHPY - dy * scale];
    }

    // Clear to transparent
    ctx.clearRect(0, 0, W, H);

    // Draw outfield grass (only inside foul lines — no background fill)
    ctx.fillStyle = '#c8e6c8';
    ctx.beginPath();
    ctx.moveTo(canvasHPX, canvasHPY);
    // Foul lines angle: LF line at ~135 deg from right, RF line at ~45 deg
    var foulAngleLeft = -Math.PI * 3 / 4;
    var foulAngleRight = -Math.PI / 4;
    ctx.arc(canvasHPX, canvasHPY, maxRadius, foulAngleLeft, foulAngleRight);
    ctx.closePath();
    ctx.fill();

    // Scale: feet to canvas pixels (CF fence at 401ft = maxRadius)
    var fenceScale = maxRadius / 401;

    // Draw infield dirt — 94.5ft radius circle centered on pitcher's mound (60.5ft from home)
    // This gives a dirt boundary 155ft from home toward CF
    var moundDist = 60.5 * fenceScale; // mound distance from home in pixels
    var moundX = canvasHPX;
    var moundY = canvasHPY - moundDist; // mound is straight up from home
    var dirtRadius = 94.5 * fenceScale;
    ctx.save();
    // Clip to foul line wedge so dirt doesn't extend outside fair territory
    ctx.beginPath();
    ctx.moveTo(canvasHPX, canvasHPY);
    ctx.arc(canvasHPX, canvasHPY, maxRadius, foulAngleLeft, foulAngleRight);
    ctx.closePath();
    ctx.clip();
    // Draw dirt circle centered on mound
    ctx.fillStyle = '#d4b896';
    ctx.beginPath();
    ctx.arc(moundX, moundY, dirtRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Draw distance reference arcs (300, 350, 400 ft)
    var distArcs = [300, 350, 400];
    for (var ai = 0; ai < distArcs.length; ai++) {
      var arcR = distArcs[ai] * fenceScale;
      ctx.strokeStyle = 'rgba(0,0,0,0.15)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(canvasHPX, canvasHPY, arcR, foulAngleLeft, foulAngleRight);
      ctx.stroke();
      ctx.setLineDash([]);
      // Label at top of arc
      var labelX = canvasHPX + arcR * Math.cos(-Math.PI / 2);
      var labelY = canvasHPY + arcR * Math.sin(-Math.PI / 2) - 3;
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.font = '10px Barlow, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(distArcs[ai] + 'ft', labelX, labelY);
    }

    // Draw foul lines
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(canvasHPX, canvasHPY);
    ctx.lineTo(canvasHPX + maxRadius * 1.02 * Math.cos(foulAngleLeft), canvasHPY + maxRadius * 1.02 * Math.sin(foulAngleLeft));
    ctx.moveTo(canvasHPX, canvasHPY);
    ctx.lineTo(canvasHPX + maxRadius * 1.02 * Math.cos(foulAngleRight), canvasHPY + maxRadius * 1.02 * Math.sin(foulAngleRight));
    ctx.stroke();

    // Distance markers (300ft, 350ft, 400ft arcs)
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 0.5;
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.font = '9px Barlow';
    ctx.textAlign = 'center';
    var distMarks = [300, 350, 400];
    for (var di = 0; di < distMarks.length; di++) {
      var distR = distMarks[di] * fenceScale;
      ctx.beginPath();
      ctx.arc(canvasHPX, canvasHPY, distR, foulAngleLeft, foulAngleRight, false);
      ctx.stroke();
      ctx.fillText(distMarks[di] + "'", canvasHPX, canvasHPY - distR - 3);
    }

    // Draw infield diamond — 90ft between bases
    var baseDist = 90 * fenceScale;
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(canvasHPX, canvasHPY); // home
    ctx.lineTo(canvasHPX + baseDist * 0.707, canvasHPY - baseDist * 0.707); // 1B
    ctx.lineTo(canvasHPX, canvasHPY - baseDist * 1.414); // 2B
    ctx.lineTo(canvasHPX - baseDist * 0.707, canvasHPY - baseDist * 0.707); // 3B
    ctx.closePath();
    ctx.stroke();

    // Draw home plate
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.moveTo(canvasHPX - 5, canvasHPY);
    ctx.lineTo(canvasHPX + 5, canvasHPY);
    ctx.lineTo(canvasHPX + 4, canvasHPY + 4);
    ctx.lineTo(canvasHPX, canvasHPY + 6);
    ctx.lineTo(canvasHPX - 4, canvasHPY + 4);
    ctx.closePath();
    ctx.fill();

    // Get BIP data
    var microData = window.MICRO_DATA;
    if (!microData || !microData.hitterBip) {
      this._renderSprayLegend('all');
      return;
    }

    var lookups = microData.lookups || {};
    var hitterIdx = (lookups.hitters || []).indexOf(data.hitter);
    if (hitterIdx < 0) {
      this._renderSprayLegend('all');
      return;
    }
    var teamIdx = (lookups.teams || []).indexOf(data.team);

    var bipCols = microData.hitterBipCols;
    var hiIdx = bipCols.indexOf('hitterIdx');
    var tiIdx = bipCols.indexOf('teamIdx');
    var hcXIdx = bipCols.indexOf('hcX');
    var hcYIdx = bipCols.indexOf('hcY');
    var bbTypeIdx = bipCols.indexOf('bbType');
    var eventIdx = bipCols.indexOf('event');
    var evIdx = bipCols.indexOf('exitVelo');
    var distIdx = bipCols.indexOf('distance');
    var batSideIdx = bipCols.indexOf('batSide');

    var bips = microData.hitterBip;
    var activeSide = this._sprayBatSide; // 'L', 'R', 'both', or null
    var filteredBips = [];
    for (var bi = 0; bi < bips.length; bi++) {
      if (bips[bi][hiIdx] !== hitterIdx) continue;
      if (teamIdx >= 0 && tiIdx >= 0 && bips[bi][tiIdx] !== teamIdx) continue;
      if (activeSide && activeSide !== 'both' && batSideIdx >= 0 && bips[bi][batSideIdx] !== activeSide) continue;
      filteredBips.push(bips[bi]);
    }

    var mode = this._sprayMode || 'all';
    var bbTypeColors = { 0: '#4e79a7', 1: '#59a14f', 2: '#f28e2b', 3: '#e15759' };
    var hitEventColors = { 1: '#ff8c00', 2: '#7b68ee', 3: '#20b2aa', 4: '#dc143c' };

    // Sample-size counts for toggle button labels — same denominator the
    // user's currently-applied bat-side filter produced (filteredBips).
    var allCount = filteredBips.length;
    var hitsCount = 0;
    var hardCount = 0;
    for (var ci = 0; ci < filteredBips.length; ci++) {
      var bipC = filteredBips[ci];
      var evtC = bipC[eventIdx];
      var evC = bipC[evIdx];
      if (evtC >= 1 && evtC <= 4) hitsCount++;
      if (evC != null && evC >= QUAL.HARD_HIT_MPH) hardCount++;
    }
    var sprayBtns = document.querySelectorAll('#spray-toggle-inline .spray-toggle-btn, #spray-toggle .spray-toggle-btn');
    for (var bi2 = 0; bi2 < sprayBtns.length; bi2++) {
      var btn = sprayBtns[bi2];
      var bm = btn.getAttribute('data-mode');
      if (bm === 'all') btn.textContent = 'All BIP (' + allCount + ')';
      else if (bm === 'hits') btn.textContent = 'Hits (' + hitsCount + ')';
      else if (bm === 'hard') btn.textContent = 'Hard Hit (' + hardCount + ')';
    }

    // Plot BIP dots
    for (var di = 0; di < filteredBips.length; di++) {
      var bip = filteredBips[di];
      var hcX = bip[hcXIdx];
      var hcY = bip[hcYIdx];
      if (hcX == null || hcY == null) continue;

      var bbType = bip[bbTypeIdx];
      var evtCode = bip[eventIdx];
      var ev = bip[evIdx];

      var color = null;
      if (mode === 'all') {
        color = bbTypeColors[bbType] || '#888';
      } else if (mode === 'hits') {
        if (evtCode >= 1 && evtCode <= 4) {
          color = hitEventColors[evtCode] || '#888';
        } else {
          continue;
        }
      } else if (mode === 'hard') {
        if (ev != null && ev >= QUAL.HARD_HIT_MPH) {
          color = bbTypeColors[bbType] || '#888';
        } else {
          continue;
        }
      }

      // Use angle from HC_X/HC_Y but distance from Distance field when available
      var pos;
      var dist = distIdx >= 0 ? bip[distIdx] : null;
      if (dist != null && dist > 0) {
        // Compute angle from HC_X/HC_Y relative to home plate
        var dx = hcX - HP_X;
        var dy = HP_Y - hcY; // invert: Statcast Y increases downward
        var angle = Math.atan2(dy, dx);
        // Convert distance (feet) to canvas radius
        var r = dist * fenceScale;
        pos = [canvasHPX + r * Math.cos(angle), canvasHPY - r * Math.sin(angle)];
      } else {
        pos = toCanvas(hcX, hcY);
      }
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.75;
      ctx.beginPath();
      ctx.arc(pos[0], pos[1], 4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1.0;

    this._renderSprayLegend(mode);
  },

  _renderSprayLegend: function (mode) {
    var container = document.getElementById('spray-legend-inline') || document.getElementById('player-spray-legend');
    if (!container) return;
    container.innerHTML = '';

    var items = [];
    if (mode === 'all' || mode === 'hard') {
      items = [
        { color: '#4e79a7', label: 'GB' },
        { color: '#59a14f', label: 'LD' },
        { color: '#f28e2b', label: 'FB' },
        { color: '#e15759', label: 'PU' },
      ];
    } else if (mode === 'hits') {
      items = [
        { color: '#ff8c00', label: 'Single' },
        { color: '#7b68ee', label: 'Double' },
        { color: '#20b2aa', label: 'Triple' },
        { color: '#dc143c', label: 'HR' },
      ];
    }

    for (var i = 0; i < items.length; i++) {
      var item = document.createElement('span');
      item.className = 'spray-legend-item';
      var dot = document.createElement('span');
      dot.className = 'spray-legend-dot';
      dot.style.backgroundColor = items[i].color;
      item.appendChild(dot);
      item.appendChild(document.createTextNode(items[i].label));
      container.appendChild(item);
    }
  },

  _bindSprayToggle: function () {
    var self = this;
    this._sprayToggleHandler = function (e) {
      var btn = e.target.closest('.spray-toggle-btn');
      if (!btn) return;
      var mode = btn.getAttribute('data-mode');
      if (mode === self._sprayMode) return;
      self._sprayMode = mode;
      var btns = document.querySelectorAll('.spray-toggle-btn');
      for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
      btn.classList.add('active');
      if (self._currentData) self._renderSprayChart(self._currentData);
    };
    var toggle = document.getElementById('spray-toggle-inline') || document.getElementById('spray-toggle');
    if (toggle) toggle.addEventListener('click', this._sprayToggleHandler);
  },

  _bindSprayBatSideToggle: function (data) {
    var self = this;
    var isSwitch = data.stands === 'S';
    // Show/hide bat side toggles
    var sprayBSToggle = document.getElementById('spray-bat-side-toggle');
    var laBSToggle = document.getElementById('la-spray-bat-side-toggle');
    if (sprayBSToggle) sprayBSToggle.style.display = isSwitch ? '' : 'none';
    if (laBSToggle) laBSToggle.style.display = isSwitch ? '' : 'none';
    if (!isSwitch) return;

    // Reset active buttons to match default (LHH)
    var allBSBtns = document.querySelectorAll('#spray-bat-side-toggle .spray-toggle-btn, #la-spray-bat-side-toggle .spray-toggle-btn');
    for (var i = 0; i < allBSBtns.length; i++) {
      allBSBtns[i].classList.toggle('active', allBSBtns[i].getAttribute('data-side') === 'L');
    }

    // Per-side BIP counts (switch hitters only) — append to button labels
    var microData = window.MICRO_DATA;
    if (microData && microData.hitterBip && microData.hitterBipCols) {
      var bipCols = microData.hitterBipCols;
      var hiIdx = bipCols.indexOf('hitterIdx');
      var tiIdx = bipCols.indexOf('teamIdx');
      var batSideIdx = bipCols.indexOf('batSide');
      var lookups = microData.lookups;
      var playerIdx = -1;
      for (var pi = 0; pi < lookups.hitters.length; pi++) {
        if (lookups.hitters[pi] === data.hitter) { playerIdx = pi; break; }
      }
      var teamIdx = (lookups.teams || []).indexOf(data.team);
      var lCount = 0, rCount = 0;
      if (playerIdx !== -1 && batSideIdx >= 0) {
        var bipData = microData.hitterBip;
        for (var bi = 0; bi < bipData.length; bi++) {
          var row = bipData[bi];
          if (row[hiIdx] !== playerIdx) continue;
          if (teamIdx >= 0 && tiIdx >= 0 && row[tiIdx] !== teamIdx) continue;
          var side = row[batSideIdx];
          if (side === 'L') lCount++;
          else if (side === 'R') rCount++;
        }
      }
      var bothCount = lCount + rCount;
      for (var ci = 0; ci < allBSBtns.length; ci++) {
        var b = allBSBtns[ci];
        var ds = b.getAttribute('data-side');
        if (ds === 'L') b.textContent = 'as LHH (' + lCount + ')';
        else if (ds === 'R') b.textContent = 'as RHH (' + rCount + ')';
        else if (ds === 'both') b.textContent = 'Both (' + bothCount + ')';
      }
    }

    this._sprayBatSideHandler = function (e) {
      var btn = e.target.closest('.spray-toggle-btn');
      if (!btn) return;
      var side = btn.getAttribute('data-side');
      if (!side || side === self._sprayBatSide) return;
      self._sprayBatSide = side === 'both' ? 'both' : side;
      // Sync active state on both toggles
      var allBtns = document.querySelectorAll('#spray-bat-side-toggle .spray-toggle-btn, #la-spray-bat-side-toggle .spray-toggle-btn');
      for (var i = 0; i < allBtns.length; i++) {
        allBtns[i].classList.toggle('active', allBtns[i].getAttribute('data-side') === side);
      }
      if (self._currentData) {
        self._renderSprayChart(self._currentData);
        self._renderLASprayChart(self._currentData);
      }
    };
    if (sprayBSToggle) sprayBSToggle.addEventListener('click', this._sprayBatSideHandler);
    if (laBSToggle) laBSToggle.addEventListener('click', this._sprayBatSideHandler);
  },

  _unbindSprayToggle: function () {
    if (this._sprayToggleHandler) {
      var el = document.getElementById('spray-toggle-inline') || document.getElementById('spray-toggle');
      if (el) el.removeEventListener('click', this._sprayToggleHandler);
      this._sprayToggleHandler = null;
    }
    if (this._laSprayToggleHandler) {
      var el2 = document.getElementById('la-spray-toggle');
      if (el2) el2.removeEventListener('click', this._laSprayToggleHandler);
      this._laSprayToggleHandler = null;
    }
    if (this._sprayBatSideHandler) {
      var el3 = document.getElementById('spray-bat-side-toggle');
      if (el3) el3.removeEventListener('click', this._sprayBatSideHandler);
      var el4 = document.getElementById('la-spray-bat-side-toggle');
      if (el4) el4.removeEventListener('click', this._sprayBatSideHandler);
      this._sprayBatSideHandler = null;
    }
    if (this._laSprayChart) {
      this._laSprayChart.destroy();
      this._laSprayChart = null;
    }
  },


  _laSprayChart: null,
  _laSprayMode: 'damage',

  _renderLASprayChart: function (data) {
    var canvas = document.getElementById('player-la-spray-chart');
    if (!canvas) return;

    var microData = window.MICRO_DATA;
    if (!microData || !microData.hitterBip || !microData.hitterBipCols) return;

    var activeSide = this._sprayBatSide; // 'L', 'R', 'both', or null (non-switch)
    // For pull/oppo labels: use the active side filter (or data.stands for non-switch)
    var bats = (activeSide && activeSide !== 'both') ? activeSide : (data.stands || 'R');
    // Switch hitters ('S') in the pooled "Both" view have no single pull/oppo
    // orientation. Normalize to 'R' so the rendered zones, hitterZoneCounts, and
    // xwOBAsp all use ONE convention. Otherwise sprayDirection('S') falls to
    // L-logic while sprayBounds/labels (gated on bats==='L') stay R-oriented,
    // mirror-flipping the zone counts and xwOBAsp for every switch hitter.
    if (bats === 'S') bats = 'R';
    var bipCols = microData.hitterBipCols;
    var hiIdx = bipCols.indexOf('hitterIdx');
    var tiIdx = bipCols.indexOf('teamIdx');
    var laIdx = bipCols.indexOf('launchAngle');
    var hcXIdx = bipCols.indexOf('hcX');
    var hcYIdx = bipCols.indexOf('hcY');
    var evIdx = bipCols.indexOf('exitVelo');
    var bbTypeIdx = bipCols.indexOf('bbType');
    var eventIdx = bipCols.indexOf('event');
    var batSideIdx = bipCols.indexOf('batSide');
    var barrelIdx = bipCols.indexOf('barrel');  // official barrel flag (1/0); -1 on old data

    // Find hitter index and team index
    var lookups = microData.lookups;
    var playerIdx = -1;
    for (var i = 0; i < lookups.hitters.length; i++) {
      if (lookups.hitters[i] === data.hitter) { playerIdx = i; break; }
    }
    if (playerIdx === -1) return;
    var teamIdx = (lookups.teams || []).indexOf(data.team);

    // Collect BIP data points
    var points = [];
    var totalBip = 0;
    var bipData = microData.hitterBip;
    for (var bi = 0; bi < bipData.length; bi++) {
      var row = bipData[bi];
      if (row[hiIdx] !== playerIdx) continue;
      if (teamIdx >= 0 && tiIdx >= 0 && row[tiIdx] !== teamIdx) continue;
      if (activeSide && activeSide !== 'both' && batSideIdx >= 0 && row[batSideIdx] !== activeSide) continue;
      totalBip++;
      var la = row[laIdx];
      var hcX = row[hcXIdx];
      var hcY = row[hcYIdx];
      if (la == null || hcX == null || hcY == null) continue;
      var sprayAngle = Aggregator.computeSprayAngle(hcX, hcY);
      if (sprayAngle == null) continue;
      // Clamp LA to chart bounds so extreme values appear at edges
      var clampedLA = Math.max(-20, Math.min(60, la));
      points.push({
        x: sprayAngle,
        y: clampedLA,
        realLA: la,
        ev: row[evIdx],
        bbType: row[bbTypeIdx],
        event: row[eventIdx],
        bs: (batSideIdx >= 0 ? row[batSideIdx] : null),  // per-BIP bat side for xwOBAsp
        barrel: (barrelIdx >= 0 ? row[barrelIdx] : null),  // official barrel flag (1/0) or null
        clamped: la !== clampedLA,
      });
    }

    // Get SACQ zones for overlay — prefer hand-specific for this batter, pooled as fallback
    var _allSacqZones = (window.METADATA && window.METADATA.sacqZones) || [];
    var sacqZones = [];
    // Build best zone per (spray, laBin): hand-specific if available, else pooled
    var _handZones = {}, _pooledZones = {};
    for (var _zi = 0; _zi < _allSacqZones.length; _zi++) {
      var _z = _allSacqZones[_zi];
      var _zk = _z.spray + '|' + _z.laBin;
      if (_z.bats === bats) _handZones[_zk] = _z;
      else if (!_z.bats) _pooledZones[_zk] = _z;
    }
    Object.keys(_pooledZones).forEach(function (k) {
      sacqZones.push(_handZones[k] || _pooledZones[k]);
    });

    var self = this;
    var mode = this._laSprayMode;

    // Color functions
    var OUTCOME_COLORS = { 0: '#888', 1: '#ff8c00', 2: '#7b68ee', 3: '#20b2aa', 4: '#dc143c', 5: '#888' };

    // EV gradient — bright blue → neutral gray → bright red. Continuous DATA
    // colormap (like the wOBAcon legend) — kept vivid on purpose; not UI chrome.
    function evColor(ev) {
      if (ev == null) return 'rgba(150,150,150,0.6)';
      var t = Math.max(0, Math.min(1, (ev - 70) / 45)); // 70–115 range
      var r, g, b;
      if (t < 0.5) {
        var s = t / 0.5;
        r = Math.round(0 + s * 140);
        g = Math.round(100 + s * 40);
        b = Math.round(255 - s * 115);
      } else {
        var s2 = (t - 0.5) / 0.5;
        r = Math.round(140 + s2 * 115);
        g = Math.round(140 - s2 * 120);
        b = Math.round(140 - s2 * 120);
      }
      return 'rgba(' + r + ',' + g + ',' + b + ',0.85)';
    }

    // Barrel id — prefer the official flag from the pipeline (launch_speed_angle
    // ==6, matches the card). Falls back to the Statcast is_barrel heuristic only
    // for old data that predates the barrel column.
    function isBarrel(ev, la) {
      if (ev == null || la == null) return false;
      return (la >= 8 && la <= 50 && ev >= 98 && ev * 1.5 - la >= 117 && ev + la >= 124);
    }
    function pointIsBarrel(pt) {
      if (pt.barrel != null) return pt.barrel === 1;
      return isBarrel(pt.ev, pt.realLA != null ? pt.realLA : pt.y);
    }
    function damageTier(pt) {
      if (pointIsBarrel(pt)) return 2;                   // barrel
      if (pt.ev != null && pt.ev >= 95) return 1;        // hard-hit 95+ (non-barrel)
      return 0;                                          // other BIP
    }
    var DAMAGE_COLORS = { 2: 'rgba(168,38,30,0.96)', 1: 'rgba(224,135,30,0.93)', 0: 'rgba(110,102,89,0.42)' };

    function getPointColor(pt) {
      if (mode === 'damage') return DAMAGE_COLORS[damageTier(pt)];
      if (mode === 'ev') return evColor(pt.ev);
      return OUTCOME_COLORS[pt.event] || '#666'; // outcome
    }

    // EV-based dot sizing: 7px (weak) to 12px (barreled). Sized for the
    // 700px-wide chart — a 24px max-diameter dot fits inside a 5° LA bin
    // (~36px tall at this chart height) without overflowing zones.
    function evRadius(ev) {
      if (ev == null) return 7;
      if (ev < 80) return 7;
      if (ev < 90) return 8;
      if (ev < 95) return 9;
      if (ev < 100) return 10;
      if (ev < 105) return 11;
      return 12;
    }

    // Build datasets
    // Damage view: draw weak contact first, barrels last (on top). Downstream
    // sums/medians are order-independent, so sorting here is safe.
    if (mode === 'damage') {
      points.sort(function (a, b) { return damageTier(a) - damageTier(b); });
    }
    var pointColors = points.map(function (p) { return getPointColor(p); });
    var pointRadii = points.map(function (p) { return evRadius(p.ev); });
    var pointHoverRadii = pointRadii.map(function (r) { return r + 2; });
    var datasets = [{
      data: points,
      backgroundColor: pointColors,
      borderColor: (mode === 'damage' ? 'rgba(240,232,216,0.9)' : 'rgba(0,0,0,0.5)'),
      borderWidth: (mode === 'damage' ? 1.5 : 1),
      pointRadius: pointRadii,
      pointHoverRadius: pointHoverRadii,
    }];

    // Zone overlay plugin — MLB wOBAcon heatmap gradient (empirical wOBA on contact
    // by spray + LA bin, restricted to MLB BIPs). Falls back to legacy `woba` field
    // for older JSON where this hadn't been renamed yet.
    var SACQ_MIN_BIP = 20;
    var sprayBounds = {
      pull: [-50, -30], pull_side: [-30, -15], center_pull: [-15, 0],
      center_oppo: [0, 15], oppo_side: [15, 30], oppo: [30, 50]
    };
    if (bats === 'L') {
      sprayBounds = {
        pull: [30, 50], pull_side: [15, 30], center_pull: [0, 15],
        center_oppo: [-15, 0], oppo_side: [-30, -15], oppo: [-50, -30]
      };
    }
    var zonePlugin = {
      id: 'sacqZones',
      beforeDatasetsDraw: function (chart) {
        var ctx2 = chart.ctx;
        var xScale = chart.scales.x;
        var yScale = chart.scales.y;
        // Color scale: blue (cold, low wOBA) → yellow (mid) → red (hot, high wOBA)
        // Range: 0.0 → 0.5 → 1.0+
        function wobaColorRGB(woba) {
          var t = Math.max(0, Math.min(1, woba / 1.0));
          var r, g, b;
          if (t < 0.35) {
            var s = t / 0.35;
            r = Math.round(8 + s * 30);
            g = Math.round(48 + s * 100);
            b = Math.round(107 + s * 40);
          } else if (t < 0.55) {
            var s = (t - 0.35) / 0.2;
            r = Math.round(38 + s * 200);
            g = Math.round(148 + s * 80);
            b = Math.round(147 - s * 100);
          } else {
            var s = (t - 0.55) / 0.45;
            r = Math.round(238 - s * 23);
            g = Math.round(228 - s * 180);
            b = Math.round(47 - s * 8);
          }
          return [r, g, b];
        }
        for (var zi = 0; zi < sacqZones.length; zi++) {
          var zone = sacqZones[zi];
          var zoneVal = zone.wobacon != null ? zone.wobacon : zone.woba;
          if (zoneVal == null) continue;
          // Damage view: only warm (>= yellow-anchor) zones, softened as backdrop.
          if (mode === 'damage' && zoneVal < 0.55) continue;
          var bounds = sprayBounds[zone.spray];
          if (!bounds) continue;
          var laMin = zone.laMin != null ? zone.laMin : -20;
          var laMax = zone.laMax != null ? zone.laMax : 65;
          var x1 = xScale.getPixelForValue(bounds[0]);
          var x2 = xScale.getPixelForValue(bounds[1]);
          var y1 = yScale.getPixelForValue(laMax);
          var y2 = yScale.getPixelForValue(laMin);
          var rgb = wobaColorRGB(zoneVal);
          var lowSample = zone.count < SACQ_MIN_BIP;
          // Improvement 1: stronger opacity (0.25), muted for low-sample zones (0.10)
          var alpha = lowSample ? 0.10 : (mode === 'damage' ? 0.34 : 0.25);
          ctx2.fillStyle = 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + alpha + ')';
          ctx2.fillRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
          // Improvement 4: diagonal hatch pattern for low-sample zones
          if (lowSample) {
            ctx2.save();
            ctx2.beginPath();
            ctx2.rect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
            ctx2.clip();
            ctx2.strokeStyle = 'rgba(255,255,255,0.08)';
            ctx2.lineWidth = 0.5;
            var rx = Math.min(x1, x2), ry = Math.min(y1, y2);
            var rw = Math.abs(x2 - x1), rh = Math.abs(y2 - y1);
            var step = 6;
            for (var hi = -rh; hi < rw; hi += step) {
              ctx2.beginPath();
              ctx2.moveTo(rx + hi, ry);
              ctx2.lineTo(rx + hi + rh, ry + rh);
              ctx2.stroke();
            }
            ctx2.restore();
          }
          // Improvement 3: stronger grid lines (0.15)
          ctx2.strokeStyle = 'rgba(255,255,255,0.15)';
          ctx2.lineWidth = 0.5;
          ctx2.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
        }

        // Damage view: quality-contact concentration ellipse (barrels + hard-hit
        // 95+), ~68% coverage (1.51 sigma), outline only. Sampled in data space
        // and mapped to pixels so the non-uniform x/y scales stay correct.
        if (mode === 'damage') {
          var qxs = [], qys = [];
          for (var qi = 0; qi < points.length; qi++) {
            var qp = points[qi];
            if (pointIsBarrel(qp) || (qp.ev != null && qp.ev >= 95)) {
              qxs.push(qp.x); qys.push(qp.y);
            }
          }
          if (qxs.length >= 6) {
            var qn = qxs.length, qmx = 0, qmy = 0;
            for (var qk = 0; qk < qn; qk++) { qmx += qxs[qk]; qmy += qys[qk]; }
            qmx /= qn; qmy /= qn;
            var caa = 0, cdd = 0, cbb = 0;
            for (var qk2 = 0; qk2 < qn; qk2++) {
              var dxq = qxs[qk2] - qmx, dyq = qys[qk2] - qmy;
              caa += dxq * dxq; cdd += dyq * dyq; cbb += dxq * dyq;
            }
            caa /= (qn - 1); cdd /= (qn - 1); cbb /= (qn - 1);
            var qtr = (caa + cdd) / 2, qdet = caa * cdd - cbb * cbb;
            var qdisc = Math.sqrt(Math.max(0, qtr * qtr - qdet));
            var l1 = qtr + qdisc, l2 = qtr - qdisc;
            if (l1 > 0 && l2 > 0) {
              var qth = Math.atan2(l1 - caa, cbb);
              var NS = 1.51;
              var axA = NS * Math.sqrt(l1), axB = NS * Math.sqrt(l2);
              ctx2.save();
              ctx2.beginPath();
              for (var eq = 0; eq <= 64; eq++) {
                var ea = (eq / 64) * Math.PI * 2;
                var exq = axA * Math.cos(ea), eyq = axB * Math.sin(ea);
                var pxD = qmx + exq * Math.cos(qth) - eyq * Math.sin(qth);
                var pyD = qmy + exq * Math.sin(qth) + eyq * Math.cos(qth);
                var PX = xScale.getPixelForValue(pxD), PY = yScale.getPixelForValue(pyD);
                if (eq === 0) ctx2.moveTo(PX, PY); else ctx2.lineTo(PX, PY);
              }
              ctx2.closePath();
              ctx2.setLineDash([6, 4]);
              ctx2.lineWidth = 2;
              ctx2.strokeStyle = '#9a3b1e';
              ctx2.stroke();
              ctx2.restore();
            }
          }
        }
      }
    };

    // Determine pull/oppo labels based on handedness
    var leftLabel = bats === 'L' ? 'Oppo' : 'Pull';
    var rightLabel = bats === 'L' ? 'Pull' : 'Oppo';

    // --- Compute hitter's per-zone BIP counts for zone hover ---
    var hitterZoneCounts = {};
    for (var hzi = 0; hzi < points.length; hzi++) {
      var hzDir = Aggregator.sprayDirection(points[hzi].x, bats);
      if (!hzDir) continue;
      var hzLaBin = Aggregator.getLABinIdx(points[hzi].realLA != null ? points[hzi].realLA : points[hzi].y);
      if (hzLaBin == null) continue;
      var hzKey = hzDir + '|' + hzLaBin;
      hitterZoneCounts[hzKey] = (hitterZoneCounts[hzKey] || 0) + 1;
    }

    // --- Compute xwOBAsp for annotation (hand-specific with pooled fallback) ---
    // Use each BIP's actual bat side (not the R-normalized `bats`, which drives
    // only the visual axis/zone orientation) so a switch hitter's per-side BIP
    // hit the correct spray direction + hand zone, matching Python compute_xwobasp.
    var _sacqMaps = Aggregator.buildSacqZoneMaps();
    var xwOBAsp_sum = 0, xwOBAsp_count = 0;
    for (var xsi = 0; xsi < points.length; xsi++) {
      var xsBats = points[xsi].bs || bats;
      var xsDir = Aggregator.sprayDirection(points[xsi].x, xsBats);
      if (!xsDir) continue;
      var xsLaBin = Aggregator.getLABinIdx(points[xsi].realLA != null ? points[xsi].realLA : points[xsi].y);
      if (xsLaBin == null) continue;
      var xsWoba = Aggregator.sacqLookup(_sacqMaps, xsDir, xsLaBin, xsBats);
      if (xsWoba != null) {
        xwOBAsp_sum += xsWoba;
        xwOBAsp_count++;
      }
    }
    var xwOBAsp_val = xwOBAsp_count > 0 ? (xwOBAsp_sum / xwOBAsp_count) : null;

    // --- Typical-placement marker (median spray + median LA across BIPs).
    // Median (not mean) so a single 60° popup or extreme chopper doesn't pull
    // the marker away from where the hitter typically places the ball.
    // Surface label stays "Avg Placement" — most readers don't distinguish. ---
    var med_spray = null, med_la = null;
    if (points.length > 0) {
      var _sortedSprays = points.map(function (p) { return p.x; }).sort(function (a, b) { return a - b; });
      var _sortedLAs = points.map(function (p) { return p.realLA != null ? p.realLA : p.y; }).sort(function (a, b) { return a - b; });
      var _midIdx = Math.floor(points.length / 2);
      med_spray = (points.length % 2 === 0) ? (_sortedSprays[_midIdx - 1] + _sortedSprays[_midIdx]) / 2 : _sortedSprays[_midIdx];
      med_la = (points.length % 2 === 0) ? (_sortedLAs[_midIdx - 1] + _sortedLAs[_midIdx]) / 2 : _sortedLAs[_midIdx];
    }
    // Resolve colors once. Percentile color drives the xwOBAsp value and the
    // avg-placement value text (quality story). The marker dot uses the brand
    // terracotta accent so it never collides with the EV-gradient (blue→gray→red)
    // dots in the BIP scatter — it reads as a system annotation, not data.
    var pctlColor = Utils.percentileTextInk(data.xwOBAsp_pctl);
    var markerColor = '#9f3026';

    // --- xwOBAsp annotation (HTML element, outside chart) ---
    var xwobaspNote = document.getElementById('la-spray-xwobasp-note');
    if (xwOBAsp_val != null) {
      xwobaspNote.textContent = '';
      // Line 1 — xwOBAsp value (percentile-colored)
      var line1 = document.createElement('div');
      var labelSpan = document.createElement('span');
      labelSpan.style.cssText = 'font-size:18px;font-weight:600;color:#6a5f55;';
      labelSpan.textContent = 'xwOBAsp: ';
      line1.appendChild(labelSpan);
      var valSpan = document.createElement('span');
      valSpan.style.cssText = 'font-size:18px;font-weight:700;cursor:help;' + (pctlColor ? 'color:' + pctlColor + ';' : 'color:#ccc;');
      valSpan.textContent = Utils.formatDecimal(3)(xwOBAsp_val);
      if (data.xwOBAsp_pctl != null) {
        valSpan.title = Math.round(data.xwOBAsp_pctl) + 'th percentile';
      }
      line1.appendChild(valSpan);
      var countSpan = document.createElement('span');
      countSpan.style.cssText = 'font-size:12px;color:#8a7f75;margin-left:4px;';
      countSpan.textContent = totalBip > xwOBAsp_count
        ? '(' + xwOBAsp_count + ' of ' + totalBip + ' qualifying BIP)'
        : '(' + xwOBAsp_count + ' qualifying BIP)';
      line1.appendChild(countSpan);
      xwobaspNote.appendChild(line1);

      // Line 2 — Avg Placement (color-matched to xwOBAsp percentile, links
      // visually to the marker dot on the chart)
      if (med_spray != null && med_la != null) {
        var line2 = document.createElement('div');
        line2.style.cssText = 'margin-top:8px;';
        var apLabel = document.createElement('span');
        apLabel.style.cssText = 'font-size:14px;font-weight:600;color:#6a5f55;';
        apLabel.textContent = 'Avg Placement: ';
        line2.appendChild(apLabel);
        var sprayDir = (bats === 'L')
          ? (med_spray > 0 ? 'Pull' : 'Oppo')
          : (med_spray < 0 ? 'Pull' : 'Oppo');
        var sprayValSpan = document.createElement('span');
        sprayValSpan.style.cssText = 'font-size:14px;font-weight:700;' + (pctlColor ? 'color:' + pctlColor + ';' : 'color:#ccc;');
        sprayValSpan.textContent = Math.abs(med_spray).toFixed(1) + '° ' + sprayDir;
        line2.appendChild(sprayValSpan);
        var sepSpan = document.createElement('span');
        sepSpan.style.cssText = 'font-size:13px;color:#c5b89f;margin:0 6px;';
        sepSpan.textContent = '|';
        line2.appendChild(sepSpan);
        var laValSpan = document.createElement('span');
        laValSpan.style.cssText = 'font-size:14px;font-weight:700;' + (pctlColor ? 'color:' + pctlColor + ';' : 'color:#ccc;');
        laValSpan.textContent = med_la.toFixed(1) + '° LA';
        line2.appendChild(laValSpan);
        xwobaspNote.appendChild(line2);
      }
      xwobaspNote.style.display = '';
    } else {
      xwobaspNote.style.display = 'none';
    }

    // --- Zone hover tooltip ---
    var zoneTooltipEl = document.getElementById('la-spray-zone-tooltip');
    if (!zoneTooltipEl) {
      zoneTooltipEl = document.createElement('div');
      zoneTooltipEl.id = 'la-spray-zone-tooltip';
      zoneTooltipEl.style.cssText = 'position:absolute;display:none;pointer-events:none;' +
        'background:rgba(20,20,25,0.92);color:#ccc;font-family:Barlow,sans-serif;' +
        'font-size:12px;padding:8px 10px;border-radius:4px;z-index:100;' +
        'border:1px solid rgba(255,255,255,0.15);line-height:1.5;white-space:nowrap;';
      canvas.parentElement.style.position = 'relative';
      canvas.parentElement.appendChild(zoneTooltipEl);
    }
    var zoneHoverSprayBounds = sprayBounds; // capture for closure
    var zoneHoverSacqZones = sacqZones;
    var zoneHoverHitterCounts = hitterZoneCounts;
    // Marker hover: capture mean position + bats so the hover handler can
    // detect proximity to the cyan center-of-mass dot and show its tooltip.
    var hoverMedSpray = med_spray;
    var hoverMedLA = med_la;
    var hoverBats = bats;
    // LA bin label helper
    var LA_BIN_LABELS = ['< -10°', '-10–0°', '0–5°', '5–10°', '10–15°', '15–20°', '20–25°',
                         '25–30°', '30–35°', '35–40°', '40–50°', '50°+'];
    var SPRAY_LABELS = { pull: 'Pull', pull_side: 'Pull-Side', center_pull: 'Center-Pull',
                         center_oppo: 'Center-Oppo', oppo_side: 'Oppo-Side', oppo: 'Oppo' };

    // Remove previous handler if any
    if (this._laSprayZoneHoverHandler) {
      canvas.removeEventListener('mousemove', this._laSprayZoneHoverHandler);
      canvas.removeEventListener('mouseleave', this._laSprayZoneLeaveHandler);
    }
    this._laSprayZoneHoverHandler = function (e) {
      var chart = self._laSprayChart;
      if (!chart) { zoneTooltipEl.style.display = 'none'; return; }
      // If Chart.js is showing a point tooltip, hide zone tooltip
      if (chart.tooltip && chart.tooltip.opacity > 0) {
        zoneTooltipEl.style.display = 'none';
        return;
      }
      var rect = canvas.getBoundingClientRect();
      var mx = e.clientX - rect.left;
      var my = e.clientY - rect.top;
      var xScale = chart.scales.x;
      var yScale = chart.scales.y;
      var dataX = xScale.getValueForPixel(mx);
      var dataY = yScale.getValueForPixel(my);
      // Check if inside chart area
      if (mx < chart.chartArea.left || mx > chart.chartArea.right ||
          my < chart.chartArea.top || my > chart.chartArea.bottom) {
        zoneTooltipEl.style.display = 'none';
        return;
      }
      // Marker proximity check — if hovering near the avg-placement dot,
      // show the marker tooltip and short-circuit the zone lookup.
      if (hoverMedSpray != null && hoverMedLA != null) {
        var clampedMedLA = Math.max(-20, Math.min(60, hoverMedLA));
        var markerPx = chart.scales.x.getPixelForValue(hoverMedSpray);
        var markerPy = chart.scales.y.getPixelForValue(clampedMedLA);
        var ddx = mx - markerPx, ddy = my - markerPy;
        if (ddx * ddx + ddy * ddy <= 144) { // within 12px of marker center
          var msDir = (hoverBats === 'L')
            ? (hoverMedSpray > 0 ? 'Pull' : 'Oppo')
            : (hoverMedSpray < 0 ? 'Pull' : 'Oppo');
          zoneTooltipEl.innerHTML = '<strong>Avg Placement</strong><br>' +
            Math.abs(hoverMedSpray).toFixed(1) + '° ' + msDir + ' &middot; ' +
            hoverMedLA.toFixed(1) + '° LA';
          zoneTooltipEl.style.display = 'block';
          var ttWm = zoneTooltipEl.offsetWidth;
          var ttHm = zoneTooltipEl.offsetHeight;
          var txm = mx + 12, tym = my - ttHm - 8;
          if (txm + ttWm > rect.width) txm = mx - ttWm - 12;
          if (tym < 0) tym = my + 16;
          zoneTooltipEl.style.left = txm + 'px';
          zoneTooltipEl.style.top = tym + 'px';
          return;
        }
      }
      // Find which zone the cursor is in
      var foundZone = null;
      for (var fzi = 0; fzi < zoneHoverSacqZones.length; fzi++) {
        var fz = zoneHoverSacqZones[fzi];
        var fb = zoneHoverSprayBounds[fz.spray];
        if (!fb) continue;
        var fxMin = Math.min(fb[0], fb[1]);
        var fxMax = Math.max(fb[0], fb[1]);
        var fyMin = fz.laMin != null ? fz.laMin : -20;
        var fyMax = fz.laMax != null ? fz.laMax : 60;
        if (dataX >= fxMin && dataX <= fxMax && dataY >= fyMin && dataY <= fyMax) {
          foundZone = fz;
          break;
        }
      }
      if (!foundZone) { zoneTooltipEl.style.display = 'none'; return; }
      var zKey = foundZone.spray + '|' + foundZone.laBin;
      var hCount = zoneHoverHitterCounts[zKey] || 0;
      var zoneWobacon = foundZone.wobacon != null ? foundZone.wobacon : foundZone.woba;
      var sprayLabel = SPRAY_LABELS[foundZone.spray] || foundZone.spray;
      var laLabel = LA_BIN_LABELS[foundZone.laBin] || '';
      var html = '<strong>' + sprayLabel + ', LA ' + laLabel + '</strong><br>';
      html += 'MLB wOBAcon: ' + (zoneWobacon != null ? zoneWobacon.toFixed(3).replace(/^0/, '') : '—');
      html += ' · xwOBAcon: ' + (foundZone.xwobacon != null ? foundZone.xwobacon.toFixed(3).replace(/^0/, '') : '—') + '<br>';
      html += 'MLB BIP: ' + foundZone.count;
      html += ' · Hitter BIP: ' + hCount;
      zoneTooltipEl.innerHTML = html;
      zoneTooltipEl.style.display = 'block';
      // Position near cursor but inside container
      var ttW = zoneTooltipEl.offsetWidth;
      var ttH = zoneTooltipEl.offsetHeight;
      var tx = mx + 12;
      var ty = my - ttH - 8;
      if (tx + ttW > rect.width) tx = mx - ttW - 12;
      if (ty < 0) ty = my + 16;
      zoneTooltipEl.style.left = tx + 'px';
      zoneTooltipEl.style.top = ty + 'px';
    };
    this._laSprayZoneLeaveHandler = function () {
      zoneTooltipEl.style.display = 'none';
    };
    canvas.addEventListener('mousemove', this._laSprayZoneHoverHandler);
    canvas.addEventListener('mouseleave', this._laSprayZoneLeaveHandler);

    // Destroy previous chart
    if (this._laSprayChart) {
      this._laSprayChart.destroy();
      this._laSprayChart = null;
    }

    // Typical-placement marker plugin — single dot at (med_spray, med_la),
    // colored by xwOBAsp percentile so the marker, the xwOBAsp value, and
    // the avg-placement values all share one visual link.
    var markerPlugin = null;
    if (med_spray != null && med_la != null) {
      var _mSpray = med_spray;
      var _mLA = Math.max(-20, Math.min(60, med_la));
      var _mColor = markerColor;
      markerPlugin = {
        id: 'centerOfMassMarker',
        afterDatasetsDraw: function (chart) {
          var ctx = chart.ctx;
          var px = chart.scales.x.getPixelForValue(_mSpray);
          var py = chart.scales.y.getPixelForValue(_mLA);
          // White outer ring for contrast against any heatmap zone color
          ctx.beginPath();
          ctx.arc(px, py, 11, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(255,255,255,0.95)';
          ctx.fill();
          // Inner dot — percentile color, black border to anchor against light bg
          ctx.beginPath();
          ctx.arc(px, py, 8, 0, Math.PI * 2);
          ctx.fillStyle = _mColor;
          ctx.fill();
          ctx.lineWidth = 2;
          ctx.strokeStyle = '#000';
          ctx.stroke();
        }
      };
    }

    this._laSprayChart = new Chart(canvas, {
      type: 'scatter',
      data: { datasets: datasets },
      plugins: markerPlugin ? [zonePlugin, markerPlugin] : [zonePlugin],
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1.2,
        animation: false,
        clip: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var pt = ctx.raw;
                var evStr = pt.ev != null ? pt.ev.toFixed(1) + ' mph' : '—';
                var laStr = (pt.clamped ? pt.realLA.toFixed(1) : pt.y.toFixed(1)) + '°';
                var sprayStr = pt.x.toFixed(1) + '°';
                var evtNames = ['Out', 'Single', 'Double', 'Triple', 'HR', 'Error/FC'];
                var result = evtNames[pt.event] || 'Out';
                return result + ' | EV: ' + evStr + ' | LA: ' + laStr + ' | Spray: ' + sprayStr;
              }
            }
          }
        },
        scales: {
          x: {
            type: 'linear',
            min: -50,
            max: 50,
            title: {
              display: true,
              text: '← ' + leftLabel + '          Spray Angle          ' + rightLabel + ' →',
              color: '#ccc',
              font: { family: 'Barlow', size: 12 }
            },
            // Tick at every actual spray bin boundary so the gridlines line up
            // with the heatmap zones (no fake dividers at ±45°).
            afterBuildTicks: function (axis) {
              axis.ticks = [-50, -30, -15, 0, 15, 30, 50].map(function (v) { return { value: v }; });
            },
            ticks: {
              autoSkip: false,
              color: '#ccc',
              font: { family: 'Barlow', size: 11 },
              callback: function (value) {
                return value + '°';
              }
            },
            grid: { color: 'rgba(255,255,255,0.2)' }
          },
          y: {
            min: -20,
            max: 60,
            title: {
              display: true,
              text: 'Launch Angle',
              color: '#ccc',
              font: { family: 'Barlow', size: 12 }
            },
            ticks: {
              stepSize: 10,
              color: '#ccc',
              font: { family: 'Barlow', size: 11 },
              callback: function (value) {
                return value + '°';
              }
            },
            grid: { color: 'rgba(255,255,255,0.2)' }
          }
        }
      }
    });

    // Render legend
    var legendEl = document.getElementById('player-la-spray-legend');
    if (legendEl) {
      var legendItems = [];
      if (mode === 'damage') {
        legendItems = [
          { color: '#a8261e', label: 'Barrel' },
          { color: '#e0871e', label: 'Hard-Hit 95+' },
          { color: '#6e6557', label: 'Other BIP' },
        ];
      } else if (mode === 'outcome') {
        legendItems = [
          { color: '#888', label: 'Out/E/FC' },
          { color: '#ff8c00', label: '1B' },
          { color: '#7b68ee', label: '2B' },
          { color: '#20b2aa', label: '3B' },
          { color: '#dc143c', label: 'HR' },
        ];
      } else if (mode === 'ev') {
        legendItems = [
          { color: 'rgb(0,100,255)', label: '70 mph' },
          { color: 'rgb(140,140,140)', label: '95 mph' },
          { color: 'rgb(255,20,20)', label: '115 mph' },
        ];
      }
      var html = '';
      for (var li = 0; li < legendItems.length; li++) {
        html += '<span class="spray-legend-item"><span class="spray-legend-dot" style="background:' +
          legendItems[li].color + '"></span>' + legendItems[li].label + '</span>';
      }
      // EV → size reference
      html += '<span class="spray-legend-item" style="margin-left:12px;">' +
        '<span class="spray-legend-dot" style="background:#888;width:10px;height:10px;border-radius:50%;"></span>' +
        '<span style="font-size:11px;color:var(--text-muted,#888);">Size = EV</span></span>';
      // Zone gradient legend bar
      html += '<div class="la-spray-gradient-legend">' +
        '<span class="la-spray-gradient-label">MLB wOBAcon:</span>' +
        '<span class="la-spray-gradient-low">.000</span>' +
        '<span class="la-spray-gradient-bar"></span>' +
        '<span class="la-spray-gradient-high">1.000+</span>' +
        '</div>';
      legendEl.innerHTML = html;
    }

    // Sync active class to current _laSprayMode (defends against stale HTML state)
    var modeBtns = document.querySelectorAll('#la-spray-toggle .spray-toggle-btn');
    for (var mbi = 0; mbi < modeBtns.length; mbi++) {
      modeBtns[mbi].classList.toggle('active', modeBtns[mbi].getAttribute('data-mode') === self._laSprayMode);
    }

    // Bind toggle buttons
    this._bindLASprayToggle(data);
  },

  _bindLASprayToggle: function (data) {
    if (this._laSprayToggleHandler) return; // already bound
    var self = this;
    this._laSprayToggleHandler = function (e) {
      var btn = e.target.closest('.spray-toggle-btn');
      if (!btn) return;
      var mode = btn.getAttribute('data-mode');
      if (mode === self._laSprayMode) return;
      self._laSprayMode = mode;
      // Update active state only within la-spray-toggle
      var btns = document.querySelectorAll('#la-spray-toggle .spray-toggle-btn');
      for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
      btn.classList.add('active');
      if (self._currentData) self._renderLASprayChart(self._currentData);
    };
    var toggle = document.getElementById('la-spray-toggle');
    if (toggle) toggle.addEventListener('click', this._laSprayToggleHandler);
  },


  _renderHitterSmallStats: function (data) {
    // Render into the movement col's table area (reuse player-pitch-usage-table)
    var container = document.getElementById('player-pitch-usage-table');
    container.innerHTML = '';
    // Small stats not shown for hitters since we have full table; leave empty
  },


  _renderHitterStatsFullTable: function (data, isROC) {
    var section = document.getElementById('player-hitter-stats-section');
    var container = document.getElementById('player-hitter-stats-table');
    if (!container) return;
    container.innerHTML = '';

    if (!data) { if (section) section.style.display = 'none'; return; }
    section.style.display = '';

    // Platoon mode swaps in a column set where every cell actually splits.
    var hand = this._platoonHand || 'all';
    var cols = hand === 'all' ? this.HITTER_STATS_COLS : this.HITTER_PLATOON_STATS_COLS;
    if (isROC) {
      cols = cols.filter(function (c) { return !c.rocHide; });
    }

    var table = document.createElement('table');
    table.className = 'player-pitch-stats-table expanded-pitch-table';

    var thead = document.createElement('thead');
    var headerRow = document.createElement('tr');
    for (var i = 0; i < cols.length; i++) {
      var th = document.createElement('th');
      th.textContent = cols[i].label;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    var tr = document.createElement('tr');
    for (var c = 0; c < cols.length; c++) {
      var col = cols[c];
      var td = document.createElement('td');
      var val = data[col.key];
      td.textContent = col.format ? col.format(val) : (val != null ? val : '—');
      if (!col.noPctl) {
        var pctl = data[col.key + '_pctl'];
        if (pctl != null) {
          var bg2 = Utils.percentileColor(pctl);
          if (bg2) { td.style.backgroundColor = bg2; td.style.color = Utils.percentileTextColor(pctl); }
        }
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
    table.appendChild(tbody);
    container.appendChild(table);
  },


  _getHitterPitchRows: function (hitterName, team) {
    var hpData = window.HITTER_PITCH_LB || [];
    var rows = [];
    for (var i = 0; i < hpData.length; i++) {
      var r = hpData[i];
      if (r.hitter === hitterName && r.team === team &&
          r.pitchType !== 'All' && r.pitchType !== 'Hard' &&
          r.pitchType !== 'Breaking' && r.pitchType !== 'Offspeed' &&
          r.count != null && r.count > 0) {
        rows.push(r);
      }
    }
    rows.sort(function (a, b) { return (b.count || 0) - (a.count || 0); });
    return rows;
  },

  _getHitterCategoryRows: function (hitterName, team) {
    var hpData = window.HITTER_PITCH_LB || [];
    var CATEGORY_ORDER = ['Hard', 'Breaking', 'Offspeed'];
    var rows = [];
    for (var i = 0; i < hpData.length; i++) {
      var r = hpData[i];
      if (r.hitter === hitterName && r.team === team &&
          CATEGORY_ORDER.indexOf(r.pitchType) !== -1 &&
          r.count != null && r.count > 0) {
        rows.push(r);
      }
    }
    rows.sort(function (a, b) {
      return CATEGORY_ORDER.indexOf(a.pitchType) - CATEGORY_ORDER.indexOf(b.pitchType);
    });
    return rows;
  },

  _getHitterPitchRowsForCategory: function (hitterName, team, category) {
    var CATS = Aggregator.PITCH_CATEGORIES;
    var pitchTypes = CATS[category] || [];
    var hpData = window.HITTER_PITCH_LB || [];
    var rows = [];
    for (var i = 0; i < hpData.length; i++) {
      var r = hpData[i];
      if (r.hitter === hitterName && r.team === team &&
          pitchTypes.indexOf(r.pitchType) !== -1 &&
          r.count != null && r.count > 0) {
        rows.push(r);
      }
    }
    rows.sort(function (a, b) { return (b.count || 0) - (a.count || 0); });
    return rows;
  },

  _renderGroupedPitchTable: function (container, cols, categoryRows, totalRow, hitterName, team, subRowsByCategory) {
    var self = this;
    var table = document.createElement('table');
    table.className = 'player-pitch-stats-table expanded-pitch-table';

    // Header
    var thead = document.createElement('thead');
    var headerRow = document.createElement('tr');
    for (var i = 0; i < cols.length; i++) {
      var th = document.createElement('th');
      th.textContent = cols[i].label;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Helper: apply percentile coloring to a stat cell. Sub-minimum rows
    // (count < QUAL.MIN_HITTER_PT) keep the tooltip rank but skip the
    // background color — the percentile is informational, not load-bearing,
    // since the sample is too small for stable estimation.
    function applyPctlColor(td, col, row, val) {
      var pctl = col.noPctl ? null : row[col.key + '_pctl'];
      if (pctl != null && val != null) {
        var inPool = (row.count || 0) >= QUAL.MIN_HITTER_PT;
        if (inPool) {
          var bg = Utils.percentileColor(pctl);
          var fg = Utils.percentileTextColor(pctl);
          td.style.backgroundColor = bg;
          td.style.color = fg;
        }
        td.title = Math.round(pctl) + 'th percentile';
      }
    }

    // Body
    var tbody = document.createElement('tbody');

    for (var r = 0; r < categoryRows.length; r++) {
      (function (catRow) {
        var tr = document.createElement('tr');
        tr.className = 'category-row';
        tr.style.cursor = 'pointer';
        var expanded = false;
        var subRowEls = [];

        for (var c = 0; c < cols.length; c++) {
          var col = cols[c];
          var td = document.createElement('td');
          if (col.key === 'pitchType') {
            td.style.textAlign = 'left';
            td.style.paddingLeft = '8px';
            var indicator = document.createElement('span');
            indicator.className = 'expand-indicator';
            indicator.textContent = '\u25B6';
            td.appendChild(indicator);
            var badge = Utils.createCategoryBadge(catRow.pitchType, true);
            td.appendChild(badge);
          } else {
            var val = catRow[col.key];
            td.textContent = col.format ? col.format(val) : (val != null ? val : '—');
            applyPctlColor(td, col, catRow, val);
          }
          tr.appendChild(td);
        }

        tr.addEventListener('click', function () {
          expanded = !expanded;
          var indicator = tr.querySelector('.expand-indicator');
          if (indicator) indicator.textContent = expanded ? '\u25BC' : '\u25B6';

          if (expanded && subRowEls.length === 0) {
            // First expand: create sub-rows. Use the pre-computed platoon-
            // filtered subrow map when provided; otherwise fall back to the
            // default-view source (window.HITTER_PITCH_LB).
            var pitchRows = subRowsByCategory
              ? (subRowsByCategory[catRow.pitchType] || [])
              : self._getHitterPitchRowsForCategory(hitterName, team, catRow.pitchType);
            var nextSibling = tr.nextSibling;
            for (var p = 0; p < pitchRows.length; p++) {
              var subTr = document.createElement('tr');
              subTr.className = 'sub-row';
              for (var sc = 0; sc < cols.length; sc++) {
                var subCol = cols[sc];
                var subTd = document.createElement('td');
                if (subCol.key === 'pitchType') {
                  subTd.style.paddingLeft = '28px';
                  var subBadge = Utils.createPitchBadge(pitchRows[p].pitchType, true);
                  subTd.appendChild(subBadge);
                } else {
                  var subVal = pitchRows[p][subCol.key];
                  subTd.textContent = subCol.format ? subCol.format(subVal) : (subVal != null ? subVal : '—');
                  applyPctlColor(subTd, subCol, pitchRows[p], subVal);
                }
                subTr.appendChild(subTd);
              }
              tbody.insertBefore(subTr, nextSibling);
              subRowEls.push(subTr);
            }
          } else {
            // Toggle visibility
            for (var s = 0; s < subRowEls.length; s++) {
              subRowEls[s].style.display = expanded ? '' : 'none';
            }
          }
        });

        tbody.appendChild(tr);
      })(categoryRows[r]);
    }

    // Total row
    if (totalRow) {
      var totalTr = document.createElement('tr');
      totalTr.style.fontWeight = '700';
      totalTr.style.borderTop = '2px solid #9f3026';
      for (var c2 = 0; c2 < cols.length; c2++) {
        var col2 = cols[c2];
        var td2 = document.createElement('td');
        if (col2.key === 'pitchType') {
          td2.textContent = 'Total';
        } else {
          var val2 = totalRow[col2.key];
          td2.textContent = col2.format ? col2.format(val2) : (val2 != null ? val2 : '—');
          var pctl2 = col2.noPctl ? null : totalRow[col2.key + '_pctl'];
          if (pctl2 != null && val2 != null) {
            var bgColor2 = Utils.percentileColor(pctl2);
            var txtColor2 = Utils.percentileTextColor(pctl2);
            td2.style.backgroundColor = bgColor2;
            td2.style.color = txtColor2;
            td2.title = Math.round(pctl2) + 'th percentile';
          }
        }
        totalTr.appendChild(td2);
      }
      tbody.appendChild(totalTr);
    }

    table.appendChild(tbody);
    container.appendChild(table);

    Utils.addScrollFade(container);
  },


  _renderHitterBattedBallTable: function (data, isROC) {
    var section = document.getElementById('player-hitter-batted-ball-section');
    var container = document.getElementById('player-hitter-batted-ball-table');
    if (!container) return;
    container.innerHTML = '';

    // Filter out columns unavailable for ROC players
    var ROC_HIDE_BB = { xBA: true, xSLG: true, xwOBA: true, xwOBAcon: true, xwOBAsp: true };
    var cols = this.HITTER_BATTED_BALL_COLS;
    if (isROC) {
      cols = cols.filter(function (c) { return !ROC_HIDE_BB[c.key]; });
    }

    var totalRow = { pitchType: 'Total' };
    for (var k = 0; k < cols.length; k++) {
      var key = cols[k].key;
      if (key !== 'pitchType') {
        totalRow[key] = data[key];
        if (data[key + '_pctl'] != null) totalRow[key + '_pctl'] = data[key + '_pctl'];
      }
    }

    // Platoon mode: use the platoon-aggregated category rows + subrow map.
    // Default mode: read directly from the embedded HITTER_PITCH_LB.
    if (this._filteredHitterCategoryRows) {
      var catRows = this._filteredHitterCategoryRows;
      if (catRows.length === 0) { if (section) section.style.display = 'none'; return; }
      section.style.display = '';
      this._renderGroupedPitchTable(container, cols, catRows, totalRow,
        data.hitter, data.team, this._filteredHitterPitchSubRows);
    } else {
      var categoryRows = this._getHitterCategoryRows(data.hitter, data.team);
      if (categoryRows.length > 0) {
        section.style.display = '';
        this._renderGroupedPitchTable(container, cols, categoryRows, totalRow, data.hitter, data.team);
      } else {
        var pitchRows = this._getHitterPitchRows(data.hitter, data.team);
        if (pitchRows.length === 0) { if (section) section.style.display = 'none'; return; }
        section.style.display = '';
        this._renderPerPitchTable(container, cols, pitchRows, totalRow);
      }
    }
  },


  _renderHitterPlateDisciplineTable: function (data) {
    var section = document.getElementById('player-hitter-plate-discipline-section');
    var container = document.getElementById('player-hitter-plate-discipline-table');
    if (!container) return;
    container.innerHTML = '';

    var totalRow = { pitchType: 'Total' };
    for (var k = 0; k < this.HITTER_PLATE_DISCIPLINE_COLS.length; k++) {
      var key = this.HITTER_PLATE_DISCIPLINE_COLS[k].key;
      if (key !== 'pitchType') {
        totalRow[key] = data[key];
        if (data[key + '_pctl'] != null) totalRow[key + '_pctl'] = data[key + '_pctl'];
      }
    }

    // Platoon mode: use the platoon-aggregated category rows + subrow map.
    // Default mode: read directly from the embedded HITTER_PITCH_LB.
    if (this._filteredHitterCategoryRows) {
      var catRows = this._filteredHitterCategoryRows;
      if (catRows.length === 0) { if (section) section.style.display = 'none'; return; }
      section.style.display = '';
      this._renderGroupedPitchTable(container, this.HITTER_PLATE_DISCIPLINE_COLS,
        catRows, totalRow, data.hitter, data.team, this._filteredHitterPitchSubRows);
    } else {
      var categoryRows = this._getHitterCategoryRows(data.hitter, data.team);
      if (categoryRows.length > 0) {
        section.style.display = '';
        this._renderGroupedPitchTable(container, this.HITTER_PLATE_DISCIPLINE_COLS, categoryRows, totalRow, data.hitter, data.team);
      } else {
        var pitchRows = this._getHitterPitchRows(data.hitter, data.team);
        if (pitchRows.length === 0) { if (section) section.style.display = 'none'; return; }
        section.style.display = '';
        this._renderPerPitchTable(container, this.HITTER_PLATE_DISCIPLINE_COLS, pitchRows, totalRow);
      }
    }
  },


  _renderHitterBatTrackingTable: function (data) {
    var section = document.getElementById('player-hitter-bat-tracking-section');
    var container = document.getElementById('player-hitter-bat-tracking-table');
    if (!container) return;
    container.innerHTML = '';

    section.style.display = '';

    var table = document.createElement('table');
    table.className = 'player-pitch-stats-table expanded-pitch-table';

    var thead = document.createElement('thead');
    var headerRow = document.createElement('tr');
    for (var i = 0; i < this.HITTER_BAT_TRACKING_COLS.length; i++) {
      var th = document.createElement('th');
      th.textContent = this.HITTER_BAT_TRACKING_COLS[i].label;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    var tr = document.createElement('tr');
    for (var c = 0; c < this.HITTER_BAT_TRACKING_COLS.length; c++) {
      var col = this.HITTER_BAT_TRACKING_COLS[c];
      var td = document.createElement('td');
      var val = data[col.key];
      td.textContent = col.format ? col.format(val) : (val != null ? val : '—');
      if (!col.noPctl) {
        var pctl = data[col.key + '_pctl'];
        if (pctl != null) {
          var bg2 = Utils.percentileColor(pctl);
          if (bg2) { td.style.backgroundColor = bg2; td.style.color = Utils.percentileTextColor(pctl); }
        }
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
    table.appendChild(tbody);
    container.appendChild(table);
  },


  checkURL: function () {
    var hash = window.location.hash.replace(/^#/, '');
    var match = hash.match(/player=(\d+)/);
    if (match) {
      var teamEl = document.getElementById('team-filter');
      var preferTeam = teamEl ? teamEl.value : null;
      if (preferTeam === 'all') preferTeam = null;
      this.open(match[1], preferTeam);
      return true;
    }
    return false;
  },

  // ── Section downloads (PNG via html2canvas) ──────────────────────────────
  // Each player-page section gets a small download icon in its header. Click
  // captures the section element via html2canvas at 2× and triggers a PNG
  // download. The button itself is excluded from the capture so the export
  // matches what the user sees minus the affordance.

  _DOWNLOAD_ICON_SVG:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"' +
    ' stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>' +
    '<polyline points="7 10 12 15 17 10"/>' +
    '<line x1="12" y1="15" x2="12" y2="3"/></svg>',

  _slugify: function (s) {
    return String(s || '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'section';
  },

  _addDownloadButton: function (sectionId, sectionSlug) {
    var section = document.getElementById(sectionId);
    if (!section) return;
    if (section.querySelector(':scope .section-download-btn')) return; // already added

    var slot;
    if (sectionId === 'player-la-spray-section') {
      // LA × Spray: header is a 3-col grid; toggle group on the right is the
      // natural spot so the button doesn't break the grid alignment.
      slot = section.querySelector('.la-spray-toggle-group');
    } else {
      slot = section.querySelector(':scope > .location-section-header, :scope > .spray-section-header');
      if (!slot) {
        // Bare-h3 section (e.g. ARSENAL). Promote the h3 into a flex header.
        var bareH3 = section.querySelector(':scope > .section-title');
        if (bareH3) {
          slot = document.createElement('div');
          slot.className = 'location-section-header';
          section.insertBefore(slot, bareH3);
          slot.appendChild(bareH3);
        }
      }
    }
    if (!slot) return;

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'section-download-btn';
    btn.title = 'Download as PNG';
    btn.setAttribute('aria-label', 'Download section as PNG');
    btn.innerHTML = this._DOWNLOAD_ICON_SVG;
    btn.dataset.sectionSlug = sectionSlug;

    var self = this;
    btn.addEventListener('click', function () {
      self._downloadSectionAsPng(section, btn, sectionSlug);
    });
    slot.appendChild(btn);
  },

  _downloadSectionAsPng: function (sectionEl, btn, sectionSlug) {
    if (!window.html2canvas) {
      console.warn('html2canvas not loaded yet');
      return;
    }
    var data = this._currentData || {};
    var playerName = data.hitter || data.pitcher || 'player';
    var team = data.team || '';
    var filename = this._slugify(playerName) + (team ? '-' + this._slugify(team) : '') +
                   '-' + sectionSlug + '.png';

    // Resolve background from the page so the captured image isn't transparent.
    var bgColor = getComputedStyle(document.body).backgroundColor || '#1a1d23';

    btn.disabled = true;
    window.html2canvas(sectionEl, {
      backgroundColor: bgColor,
      scale: 2,
      logging: false,
      useCORS: true,
      ignoreElements: function (el) {
        // Don't render the download button itself or any tooltip overlay.
        return el.classList && (
          el.classList.contains('section-download-btn') ||
          el.id === 'la-spray-zone-tooltip'
        );
      }
    }).then(function (canvas) {
      var link = document.createElement('a');
      link.download = filename;
      link.href = canvas.toDataURL('image/png');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }).catch(function (err) {
      console.error('Section download failed', err);
    }).then(function () {
      btn.disabled = false;
    });
  },

  _addAllDownloadButtons: function (data) {
    if (!data) return;
    var isPitcher = !!data.pitcher;
    var sections = isPitcher ? [
      ['player-stats-section',             'stats'],
      ['player-platoon-split-section',     'platoon-split'],
      ['player-expanded-pitch-section',    'arsenal'],
      ['player-plate-discipline-section',  'plate-discipline'],
      ['player-batted-ball-section',       'batted-ball'],
      ['player-location-section',          'pitch-locations'],
      ['player-count-section',             'usage-by-count'],
      ['player-game-grades-section',       'game-grades'],
    ] : [
      ['player-spray-section',                       'spray-chart'],
      ['player-la-spray-section',                    'la-spray'],
      ['player-hitter-stats-section',                'stats'],
      ['player-hitter-platoon-split-section',        'platoon-split'],
      ['player-hitter-plate-discipline-section',     'plate-discipline'],
      ['player-hitter-batted-ball-section',          'batted-ball'],
      ['player-hitter-bat-tracking-section',         'bat-tracking'],
    ];
    for (var i = 0; i < sections.length; i++) {
      var sec = document.getElementById(sections[i][0]);
      if (!sec || sec.style.display === 'none') continue;
      this._addDownloadButton(sections[i][0], sections[i][1]);
    }
    // Hitter pages also get a one-click "everything" export (2026-08-06, per
    // Wally): stats + platoon split + plate discipline + batted ball + bat
    // tracking stacked into a single PNG. Lives in the STATS header, which
    // _addDownloadButton just promoted to a flex header above.
    if (!isPitcher) this._addCombinedDownloadButton();
  },

  // The five table sections, in page order. Bat tracking is appended only for
  // MLB hitters — ROC has no bat-tracking data, so the export excludes it
  // rather than shipping an empty box.
  _combinedDownloadSections: function () {
    var ids = ['player-hitter-stats-section',
               'player-hitter-platoon-split-section',
               'player-hitter-plate-discipline-section',
               'player-hitter-batted-ball-section'];
    if (!this._isROC) ids.push('player-hitter-bat-tracking-section');
    var parts = [];
    for (var i = 0; i < ids.length; i++) {
      var el = document.getElementById(ids[i]);
      if (el && el.style.display !== 'none' && el.offsetHeight > 0) parts.push(el);
    }
    return parts;
  },

  _addCombinedDownloadButton: function () {
    var section = document.getElementById('player-hitter-stats-section');
    if (!section) return;
    if (section.querySelector(':scope .section-download-all-btn')) return;
    var slot = section.querySelector(':scope > .location-section-header, :scope > .spray-section-header');
    if (!slot) return;

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'section-download-btn section-download-all-btn';
    btn.title = 'Download all tables as one PNG';
    btn.setAttribute('aria-label', 'Download all hitter tables as one PNG');
    btn.innerHTML = this._DOWNLOAD_ICON_SVG +
      '<span style="font-size:11px;font-weight:600;margin-left:3px;">All</span>';

    var self = this;
    btn.addEventListener('click', function () { self._downloadCombinedPng(btn); });
    slot.appendChild(btn);
  },

  _downloadCombinedPng: function (btn) {
    if (!window.html2canvas) {
      console.warn('html2canvas not loaded yet');
      return;
    }
    var parts = this._combinedDownloadSections();
    if (!parts.length) return;

    var data = this._currentData || {};
    var playerName = data.hitter || 'player';
    var filename = this._slugify(playerName) +
                   (data.team ? '-' + this._slugify(data.team) : '') +
                   '-all-tables.png';
    var bgColor = getComputedStyle(document.body).backgroundColor || '#1a1d23';

    // Clone into an off-screen stack and capture THAT: the live sections are
    // not contiguous siblings, and these five are pure tables (no <canvas>),
    // which is exactly the case cloneNode + html2canvas handles faithfully.
    var maxW = 0;
    for (var i = 0; i < parts.length; i++) maxW = Math.max(maxW, parts[i].offsetWidth);
    var wrap = document.createElement('div');
    wrap.style.cssText = 'position:absolute;left:-10000px;top:0;padding:12px;' +
                         'background:' + bgColor + ';width:' + (maxW + 24) + 'px;';
    for (var j = 0; j < parts.length; j++) {
      var clone = parts[j].cloneNode(true);
      clone.style.display = '';
      clone.style.marginBottom = (j < parts.length - 1) ? '18px' : '0';
      wrap.appendChild(clone);
    }
    document.body.appendChild(wrap);

    btn.disabled = true;
    window.html2canvas(wrap, {
      backgroundColor: bgColor,
      scale: 2,
      logging: false,
      useCORS: true,
      ignoreElements: function (el) {
        return el.classList && (
          el.classList.contains('section-download-btn') ||
          el.id === 'la-spray-zone-tooltip'
        );
      }
    }).then(function (canvas) {
      var link = document.createElement('a');
      link.download = filename;
      link.href = canvas.toDataURL('image/png');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }).catch(function (err) {
      console.error('Combined download failed', err);
    }).then(function () {
      btn.disabled = false;
      if (wrap.parentNode) wrap.parentNode.removeChild(wrap);
    });
  },
};
