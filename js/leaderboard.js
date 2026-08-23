const COLUMNS = {
  pitchMetrics: [
    { key: '_rank',       label: '#',        format: function(v){ return v; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, group: 'info', width: '36px' },
    { key: '_compare',    label: '',         format: function(){ return ''; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, isCompare: true, group: 'info', width: '32px' },
    { key: 'pitcher',     label: 'Pitcher',  format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', sticky: true, cls: 'col-pitcher', noPercentile: true, noToggle: true, group: 'info' },
    { key: 'team',        label: 'Team',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isTeam: true, sticky: true, stickyIdx: 1 },
    { key: 'throws',      label: 'Throws',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info' },
    { key: 'pitchType',   label: 'Pitch',    format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isPitchType: true },
    { key: 'count',       label: 'Pitches',  format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'usagePct',    label: 'Usage%',   format: Utils.formatPct, sortType: 'numeric', noPercentile: true, desc: 'Pct of pitcher\'s total pitches', group: 'info' },
    { key: 'velocity',       label: 'Velo',     format: Utils.formatDecimal(1), sortType: 'numeric', sectionStart: true, desc: 'Average velocity (mph)', group: 'metrics' },
    { key: 'spinRate',    label: 'Spin',     format: Utils.formatInt, sortType: 'numeric', desc: 'Average spin rate (rpm)', group: 'metrics' },
    { key: 'indVertBrk',  label: 'IVB',      format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Induced vertical break (inches) — vertical movement created by spin, measured against a spin-free pitch. High IVB = ride/carry; negative = true drop', group: 'metrics' },
    { key: 'horzBrk',     label: 'HB',       format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Horizontal break (inches, pitcher POV): positive = arm side for a RHP, glove side for a LHP. A RHP sinker is ~+16; a RHP sweeper is strongly negative', group: 'metrics' },
    { key: 'breakTilt',   label: 'OTilt',    format: Utils.formatTilt, sortType: 'numeric', sortKey: 'breakTiltMinutes', noPercentile: true, desc: 'Observed Tilt — direction of total break as a clock face (12:00 = pure ride straight up, 6:00 = straight drop; RHP fastballs sit near 1:00, LHP near 11:00). Measured from actual movement, unlike RTilt (Release Tilt), which is the spin axis at release', group: 'metrics' },
    { key: 'relPosZ', rocHide: true, label: 'RelZ',     format: Utils.formatFeetInches, sortType: 'numeric', noPercentile: true, desc: 'Vertical release point (feet)', group: 'metrics' },
    { key: 'relPosX', rocHide: true, label: 'RelX',     format: Utils.formatFeetInches, sortType: 'numeric', noPercentile: true, desc: 'Horizontal release point (feet, pitcher POV)', group: 'metrics' },
    { key: 'extension',   label: 'Ext',      format: Utils.formatFeetInches, sortType: 'numeric', desc: 'Extension toward home plate at release (feet)', group: 'metrics' },
    { key: 'armAngle',                   label: 'Arm Angle', format: Utils.formatDecimal(1), sortType: 'numeric', noPercentile: true, showAvg: true, desc: 'Arm angle at release (degrees above horizontal): 0 = sidearm, 90 = straight over the top. League median is ~38', group: 'metrics' },
    { key: 'nVAA',        label: 'nVAA',     format: Utils.formatDecimal(2), sortType: 'numeric', desc: 'VAA with the location effect removed (high pitches are naturally flatter). Positive = flatter than its height predicts — the sneaky-ride trait that plays at the top of the zone', group: 'metrics' },
    { key: 'nHAA',        label: 'nHAA',     format: Utils.formatDecimal(2), sortType: 'numeric', noPercentile: true, desc: 'HAA with the location effect removed. Measures whether the pitch approaches from a wider or straighter angle than its plate location predicts', group: 'metrics' },
    { key: 'stuffScore',  label: 'Stuff+',   format: Utils.formatInt, sortType: 'numeric', desc: 'Stuff+ for this pitch type — pitch quality from physical characteristics only (velocity, movement, release, arm angle), independent of location or outcome. Standardized within the pitch-type group. 100 = group avg, +10 = 1 SD better. Standard-deviation ruler, NOT a percent: a point is not 1% better. Hitter + metrics are percents; the pitcher ones are not, because Stuff+/Loc+/Pitching+ sit on the shared public convention so a value here means what it means on other sites.', group: 'outcomes' },
    { key: 'locPlus',     label: 'Loc+',     format: Utils.formatInt, sortType: 'numeric', desc: 'Location+ for this pitch type — xRV-weighted location quality standardized within the pitch-type group (FF, SI, FC, SL, CB, CH). 100 = group avg, +10 = 1 SD better. Standard-deviation ruler, NOT a percent: a point is not 1% better. Hitter + metrics are percents; the pitcher ones are not, because Stuff+/Loc+/Pitching+ sit on the shared public convention so a value here means what it means on other sites. ROC pitchers scored against the MLB baseline.', group: 'outcomes' },
    { key: 'pitchingScore', label: 'Pitching+', format: Utils.formatInt, sortType: 'numeric', desc: 'Pitching+ for this pitch type — overall pitch quality: exactly 0.72 x Stuff+ + 0.28 x Loc+ (weight validated on next-season run prevention, re-swept 2026-08). Auditable from the two component columns. Standard-deviation ruler, NOT a percent: a point is not 1% better. Hitter + metrics are percents; the pitcher ones are not, because Stuff+/Loc+/Pitching+ sit on the shared public convention so a value here means what it means on other sites.', group: 'outcomes' },
    { key: 'xRv100',      label: 'xRV/100',  format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Expected run value per 100 pitches (positive = better for pitcher)', group: 'outcomes' },
    { key: 'swStrPct',    label: 'Whiff%',   format: Utils.formatPct, sortType: 'numeric', desc: 'Whiff rate on swings (whiffs / swings) for this pitch type', group: 'outcomes' },
    { key: 'chasePct',    label: 'Chase%',   format: Utils.formatPct, sortType: 'numeric', desc: 'Out-of-zone swing rate for this pitch type', group: 'outcomes' },
    { key: 'maxVelo',       label: 'Max Velo', format: Utils.formatDecimal(1), sortType: 'numeric', noPercentile: true, desc: 'Maximum velocity (mph)', group: 'metrics' },
    { key: 'vaa',         label: 'VAA',      format: Utils.formatDecimal(2), sortType: 'numeric', desc: 'Vertical approach angle — how steeply the pitch crosses the plate (degrees). Always negative; closer to 0 = flatter', group: 'metrics' },
    { key: 'haa',         label: 'HAA',      format: Utils.formatDecimal(2), sortType: 'numeric', desc: 'Horizontal approach angle at the plate (degrees)', group: 'metrics' },
    { key: 'cswPct',      label: 'CSW%',     format: Utils.formatPct, sortType: 'numeric', desc: 'Called strikes + whiffs / total pitches for this pitch type', group: 'outcomes' },
    { key: 'barrelPctAgainst', label: 'Barrel%', format: Utils.formatPct, sortType: 'numeric', desc: 'Barrel rate against for this pitch type — barrels are the EV/LA combos that historically return at least a .500 BA and 1.500 SLG (starts at 98 mph EV). Denominator = BIP with valid EV', group: 'outcomes' },
    { key: 'xwOBA',       label: 'xwOBA',    format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected wOBA (Statcast model, based on EV + LA)', group: 'outcomes' },
    { key: 'xBA',         label: 'xBA',      format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected batting average (Statcast model, based on EV + LA)', group: 'outcomes' },
    { key: 'xSLG',        label: 'xSLG',     format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected slugging (Statcast model, based on EV + LA)', group: 'outcomes' },
    { key: 'rv100',                      label: 'RV/100',   format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Run value per 100 pitches (positive = better for pitcher)', group: 'outcomes' },
    { key: 'runValue',                   label: 'RV',       format: Utils.formatDecimal(1), sortType: 'numeric', sectionStart: true, desc: 'Pitch-level run value — runs saved vs league avg (positive = better for pitcher)', group: 'outcomes' },
    { key: 'xRunValue',   label: 'xRV',      format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Expected pitch-level run value — uses Statcast expected outcomes on BIP (positive = better for pitcher)', group: 'outcomes' },
    { key: 'rvoe',        label: 'RVOE',     format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Runs above expectation — total actual runs (luck included) better (+) or worse (-) than this pitch\'s Stuff+ and location predict. Raw accounting, unregressed. RVOE minus xRVOE = contact luck. Min 150 pitches.', group: 'outcomes' },
    { key: 'xrvoe',       label: 'xRVOE',    format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Expected runs above expectation — total luck-neutral runs (xwOBA on contact) better (+) or worse (-) than this pitch\'s Stuff+ and location predict. Raw accounting, unregressed. Min 150 pitches.', group: 'outcomes' },
    { key: 'rvoe100',     label: 'RVOE/100', format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Runs above expectation per 100 pitches, actual results (luck included). Raw rate, unregressed — compare with xRVOE/100 (regressed, luck-neutral) to see how much is fortune. Min 150 pitches.', group: 'outcomes' },
    { key: 'xrvoe100',    label: 'xRVOE/100', format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Outperformance vs expectation, runs per 100 pitches: how much better (+) or worse (-) this pitch performs than its Stuff+ and location predict. A durable trait (year-to-year r=.41) capturing deception, seam effects, tunneling — everything the models can\'t see. Regressed toward 0; min 150 pitches.', group: 'outcomes' },
  ],
  pitcherStats: [
    { key: '_rank',       label: '#',        format: function(v){ return v; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, group: 'info', width: '36px' },
    { key: '_compare',    label: '',         format: function(){ return ''; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, isCompare: true, group: 'info', width: '32px' },
    { key: 'pitcher',     label: 'Pitcher',  format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', sticky: true, cls: 'col-pitcher', noPercentile: true, noToggle: true, group: 'info' },
    { key: 'team',        label: 'Team',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isTeam: true, sticky: true, stickyIdx: 1 },
    { key: 'throws',      label: 'Throws',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info' },
    { key: 'g',           label: 'G',        format: Utils.formatInt, sortType: 'numeric', noPercentile: true, sectionStart: true, group: 'counting' },
    { key: 'gs',          label: 'GS',       format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'counting' },
    { key: 'ip',          label: 'IP',       format: function(v){ return v != null ? v : '—'; }, sortType: 'numeric', noPercentile: true, group: 'counting' },
    { key: 'era',         label: 'ERA',      format: Utils.formatDecimal(2), sortType: 'numeric', sectionStart: true, desc: 'Earned run average', group: 'advanced' },
    { key: 'siera',       label: 'SIERA',    format: Utils.formatDecimal(2), sortType: 'numeric', desc: 'ERA estimator built only from strikeouts, walks, and ground balls — the outcomes a pitcher controls most. Better than ERA or FIP at predicting future performance', group: 'advanced' },
    { key: 'hdERA',       label: 'hdERA',    format: Utils.formatDecimal(2), sortType: 'numeric', desc: 'Huronalytics descriptive ERA — what this season deserves with defense, sequencing, ballpark, and batted-ball luck stripped: xwOBA against, shrunk for sample size and mapped to the ERA scale. Describes the season better than FIP once luck channels are excluded (the gap between hdERA and actual ERA IS the luck). Season-long metric: it does not react to leaderboard filters.', group: 'advanced' },
    { key: 'hpERA',       label: 'hpERA',    format: Utils.formatDecimal(2), sortType: 'numeric', desc: 'Huronalytics predictive ERA — projected ERA going forward: the Pitcher+ component set (Stuff+, Loc+, K%, zone whiffs, xRV, ground balls) plus role and park, weighted by out-of-sample fit against future ERA, 2021-2026. Beats SIERA at predicting both rest-of-season and next-season ERA in every tested season. Every pitcher gets a value (small samples shrink toward league average); qualification gates only the coloring, like SIERA. Season-long metric: it does not react to leaderboard filters.', group: 'advanced' },
    { key: 'hdERAPlus',   label: 'hdERA+',   format: Utils.formatInt, sortType: 'numeric', desc: 'hdERA on the 100 scale (like wRC+): 100 = league average, higher = better, each point = 1% of league-average run prevention deserved. Same information as hdERA, reversed scale.', group: 'advanced' },
    { key: 'hpERAPlus',   label: 'hpERA+',   format: Utils.formatInt, sortType: 'numeric', desc: 'hpERA on the 100 scale (like wRC+): 100 = league average, higher = better, each point = 1% of league-average run prevention projected going forward. Same information as hpERA, reversed scale.', group: 'advanced' },
    { key: 'stuffScore',  label: 'Stuff+',   format: Utils.formatInt, sortType: 'numeric', sectionStart: true, desc: 'Stuff+ — overall pitch quality from physical characteristics only (velocity, movement, release, arm angle), usage-weighted across the arsenal and independent of location or outcome. 100 = league avg. The per-pitch-type Stuff+ columns are standardized to +10 = 1 SD; this overall number is their usage-weighted mean, and averaging an arsenal compresses the spread, so 1 SD is about 8 points here. Standard-deviation ruler, NOT a percent: a point is not 1% better. Hitter + metrics are percents; the pitcher ones are not, because Stuff+/Loc+/Pitching+ sit on the shared public convention so a value here means what it means on other sites.', group: 'run_value' },
    { key: 'locPlus',     label: 'Loc+',     format: Utils.formatInt, sortType: 'numeric', desc: 'Location+ — per-pitch location quality scored against an xRV-weighted (zone × count × pitch-type × handedness) model. Command independent of stuff or contact luck. 100 = league avg. The per-pitch-type Loc+ columns are standardized to +10 = 1 SD; this overall number is both averaged across the arsenal and shrunk toward 100 by its reliability prior, so 1 SD is about 6 points here. Standard-deviation ruler, NOT a percent: a point is not 1% better. Hitter + metrics are percents; the pitcher ones are not, because Stuff+/Loc+/Pitching+ sit on the shared public convention so a value here means what it means on other sites.', group: 'run_value' },
    { key: 'commandPlus', label: 'Command+', format: Utils.formatInt, sortType: 'numeric', desc: "Command+ — execution repeatability: average miss distance from the pitcher's inferred targets (fit per pitch type, batter hand, and count situation, pooled coarser where he throws a pitch too rarely to read the situation). 100 = league avg and one point is one percent: a 115 misses by 15% less distance than the league. Unlike the other pitcher + columns this one IS a percent, because miss distance has a real zero (perfect command = 0). It is not run-denominated on purpose: measured 2021-2026, command's correlation with runs prevented ran +0.02 to +0.23, so a run-scaled version would be a dead column. Forecasts future walk rate beyond current BB%; it does NOT predict run prevention beyond Loc+ (command's run impact already lives in Loc+), and higher velocity trades against it. ROC pitchers scored against their own targets — no MLB translation.", group: 'run_value' },
    { key: 'pitchingScore', label: 'Pitching+', format: Utils.formatInt, sortType: 'numeric', desc: 'Pitching+ — overall arsenal quality: exactly 0.72 x Stuff+ + 0.28 x Loc+ (weight validated on next-season run prevention, re-swept 2026-08). Auditable from the two component columns. It inherits their compression, so 1 SD is about 6 points. Standard-deviation ruler, NOT a percent: a point is not 1% better. Hitter + metrics are percents; the pitcher ones are not, because Stuff+/Loc+/Pitching+ sit on the shared public convention so a value here means what it means on other sites.', group: 'run_value' },
    { key: 'pitcherPlus', label: 'Pitcher+', format: Utils.formatInt, sortType: 'numeric', desc: 'Pitcher+ — the all-encompassing pitcher grade: stuff, command, whiffs, and results blended into one number. 100 = league avg, +10 = 1 SD better. Standard-deviation ruler, NOT a percent: a point is not 1% better. Hitter + metrics are percents; the pitcher ones are not, because Stuff+/Loc+/Pitching+ sit on the shared public convention so a value here means what it means on other sites. Measured 2021-2025 on next-season run prevention (same-season is circular, xRV/100 is a component): one point predicts about 1.2 points, so this scale understates it by roughly a fifth.', group: 'run_value' },
    { key: 'pitcherPlusProj', label: 'Pitcher+ Proj', format: Utils.formatInt, sortType: 'numeric', desc: 'Projected NEXT-SEASON Pitcher+: 70% this season + 30% last season, re-standardized to 100 = league avg. Pitchers without a prior season keep their current Pitcher+ (the standard Marcel/Steamer pattern). Blending two years lifts out-of-fold prediction of next-season xRV/100 from .61 to .63; a third year adds nothing. Not age-adjusted.', group: 'run_value' },
    { key: 'kPct',        label: 'K%',       format: Utils.formatPct, sortType: 'numeric', sectionStart: true, desc: 'Strikeout rate (K / TBF)', group: 'stats' },
    { key: 'bbPct',       label: 'BB%',      format: Utils.formatPct, sortType: 'numeric', desc: 'Walk rate (BB / TBF)', group: 'stats' },
    { key: 'kbbPct',      label: 'K-BB%',    format: Utils.formatPct, sortType: 'numeric', desc: 'K% minus BB%', group: 'stats' },
    { key: 'rv100',                      label: 'RV/100',   format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Run value per 100 pitches, roughly one start\'s workload (positive = better for pitcher)', group: 'run_value' },
    { key: 'xRv100',      label: 'xRV/100',  format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Expected run value per 100 pitches, roughly one start\'s workload (positive = better for pitcher)', group: 'run_value' },
    { key: 'xrvoe100',    label: 'xRVOE/100', format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Outperformance vs expectation, runs per 100 pitches: how much better (+) or worse (-) results are than Stuff+ and location predict, across the arsenal. A durable trait (year-to-year r=.41): deception, seam effects, tunneling. Regressed toward 0; min 300 pitches.', group: 'run_value' },
    { key: 'w',           label: 'W',        format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'counting' },
    { key: 'l',           label: 'L',        format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'counting' },
    { key: 'sv',          label: 'SV',       format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'counting' },
    { key: 'hld',         label: 'HLD',      format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'counting' },
    { key: 'tbf',         label: 'TBF',      format: Utils.formatInt, sortType: 'numeric', noPercentile: true, desc: 'Total batters faced', group: 'counting' },
    { key: 'fip',         label: 'FIP',      format: Utils.formatDecimal(2), sortType: 'numeric', desc: 'Fielding Independent Pitching — what ERA should be based only on K, BB, HBP, and HR, removing defense and sequencing. On the ERA scale', group: 'advanced' },
    { key: 'xFIP',        label: 'xFIP',     format: Utils.formatDecimal(2), sortType: 'numeric', desc: 'FIP with home runs replaced by a league-average HR/FB rate — strips home run luck. On the ERA scale', group: 'advanced' },
    { key: 'runValue',                   label: 'RV',       format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Run value — runs saved vs league average (positive = better for pitcher)', group: 'run_value' },
    { key: 'xRunValue',   label: 'xRV',      format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Expected run value — uses Statcast expected outcomes on BIP (positive = better for pitcher)', group: 'run_value' },
    { key: 'rvoe',        label: 'RVOE',     format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Runs above expectation — total actual runs (luck included) better (+) or worse (-) than Stuff+ and location predict, across the arsenal. Raw accounting, unregressed. RVOE minus xRVOE = contact luck. Min 300 pitches.', group: 'run_value' },
    { key: 'xrvoe',       label: 'xRVOE',    format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Expected runs above expectation — total luck-neutral runs (xwOBA on contact) better (+) or worse (-) than Stuff+ and location predict, across the arsenal. Raw accounting, unregressed. Min 300 pitches.', group: 'run_value' },
    { key: 'rvoe100',     label: 'RVOE/100', format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Runs above expectation per 100 pitches, actual results (luck included). Raw rate, unregressed — compare with xRVOE/100 (regressed, luck-neutral) to see how much is fortune. Min 300 pitches.', group: 'run_value' },
    { key: 'armAngle',    label: 'Arm Angle', format: Utils.formatDecimal(1), sortType: 'numeric', sectionStart: true, noPercentile: true, showAvg: true, desc: 'Average arm angle at release across all pitches (degrees above horizontal): 0 = sidearm, 90 = straight over the top. League median is ~38', group: 'metrics' },
    { key: 'extension',   label: 'Ext',      format: Utils.formatFeetInches, sortType: 'numeric', desc: 'Average extension toward home plate at release across all pitches (feet)', group: 'metrics' },
  ],
  pitcherBattedBall: [
    { key: '_rank',       label: '#',        format: function(v){ return v; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, group: 'info', width: '36px' },
    { key: '_compare',    label: '',         format: function(){ return ''; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, isCompare: true, group: 'info', width: '32px' },
    { key: 'pitcher',     label: 'Pitcher',  format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', sticky: true, cls: 'col-pitcher', noPercentile: true, noToggle: true, group: 'info' },
    { key: 'team',        label: 'Team',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isTeam: true, sticky: true, stickyIdx: 1 },
    { key: 'throws',      label: 'Throws',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info' },
    { key: 'count',       label: 'Pitches',  format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'nBip',        label: 'BIP',      format: Utils.formatInt, sortType: 'numeric', noPercentile: true, desc: 'Balls in play — every non-bunt batted ball, home runs included', group: 'info' },
    { key: 'xwOBA',       label: 'xwOBA',    format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected wOBA against (Statcast, EV + LA)', group: 'batted_ball' },
    { key: 'xwOBAcon',   label: 'xwOBAcon', format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected wOBA on contact — avg xwOBA on BIP only', group: 'batted_ball' },
    { key: 'avgEVAgainst', label: 'Avg EV',  format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Average exit velocity against (mph)', group: 'batted_ball' },
    { key: 'hardHitPct',  label: 'Hard-Hit%', format: Utils.formatPct, sortType: 'numeric', desc: 'Pct of trackable BIP with EV ≥ 95 mph (denominator = BIP with valid EV)', group: 'batted_ball' },
    { key: 'barrelPctAgainst', label: 'Barrel%', format: Utils.formatPct, sortType: 'numeric', desc: 'Barrel rate against — barrels are the EV/LA combos that historically return at least a .500 BA and 1.500 SLG (starts at 98 mph EV). Denominator = BIP with valid EV', group: 'batted_ball' },
    { key: 'gbPct',       label: 'GB%',      format: Utils.formatPct, sortType: 'numeric', desc: 'Ground ball rate', group: 'batted_ball' },
    { key: 'puPct',       label: 'PU%',      format: Utils.formatPct, sortType: 'numeric', desc: 'Popup rate — the other free out (higher is better for pitchers; split-half reliability .44)', group: 'batted_ball' },
    { key: 'babip',       label: 'BABIP',    format: Utils.formatDecimal(3), sortType: 'numeric', sectionStart: true, noPercentile: true, desc: 'Batting avg on balls in play against — luck barometer, deliberately uncolored (split-half reliability .17)', group: 'batted_ball' },
    { key: 'hrFbPct',     label: 'HR/FB',    format: Utils.formatPct, sortType: 'numeric', desc: 'Fly-ball HRs per outfield fly — % of fly balls that left the yard (popups and line-drive HR excluded; FG counts popups so reads lower)', group: 'batted_ball' },
    { key: 'xwOBAsp',    label: 'xwOBASp',  format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected wOBA against from spray direction + launch angle — judges where balls were hit rather than how hard. Complements xwOBA, which ignores direction entirely', group: 'batted_ball' },
    { key: 'xBA',         label: 'xBA',      format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected BA against (Statcast, EV + LA)', group: 'batted_ball' },
    { key: 'xSLG',        label: 'xSLG',     format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected SLG against (Statcast, EV + LA)', group: 'batted_ball' },
    { key: 'maxEVAgainst', label: 'Max EV',  format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Max exit velocity against (mph)', group: 'batted_ball' },
  ],
  pitcherSwingDecisions: [
    { key: '_rank',       label: '#',        format: function(v){ return v; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, group: 'info', width: '36px' },
    { key: '_compare',    label: '',         format: function(){ return ''; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, isCompare: true, group: 'info', width: '32px' },
    { key: 'pitcher',     label: 'Pitcher',  format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', sticky: true, cls: 'col-pitcher', noPercentile: true, noToggle: true, group: 'info' },
    { key: 'team',        label: 'Team',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isTeam: true, sticky: true, stickyIdx: 1 },
    { key: 'throws',      label: 'Throws',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info' },
    { key: 'count',       label: 'Pitches',  format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'nSwings',     label: 'Swings',   format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'locPlus',     label: 'Loc+',     format: Utils.formatInt, sortType: 'numeric', desc: 'Location+: per-pitch location quality scored against an xRV-weighted (zone × count × pitch-type × handedness) table. Captures command independent of stuff or contact luck. 100 = league avg; 1 SD is about 6 points at the season level (the per-pitch-type columns are the ones standardized to +10 = 1 SD, before arsenal averaging and the reliability prior shrink the overall). ROC pitchers scored against the MLB baseline. Standard-deviation ruler, NOT a percent: a point is not 1% better. Hitter + metrics are percents; the pitcher ones are not, because Stuff+/Loc+/Pitching+ sit on the shared public convention so a value here means what it means on other sites.', group: 'stats' },
    { key: 'strikePct',   label: 'Strike%',  format: Utils.formatPct, sortType: 'numeric', sectionStart: true, desc: 'Total strike rate (called + swinging + foul)', group: 'stats' },
    { key: 'izPct',       label: 'Zone%',    format: Utils.formatPct, sortType: 'numeric', desc: 'Pct of pitches in the strike zone', group: 'stats' },
    { key: 'cswPct',      label: 'CSW%',     format: Utils.formatPct, sortType: 'numeric', desc: 'Called strikes + whiffs / total pitches', group: 'stats' },
    { key: 'swStrPct',    label: 'Whiff%',   format: Utils.formatPct, sortType: 'numeric', desc: 'Swinging strikes / total swings', group: 'stats' },
    { key: 'izWhiffPct',  label: 'Z-Whiff%', format: Utils.formatPct, sortType: 'numeric', desc: 'Whiff rate on in-zone swings', group: 'stats' },
    { key: 'chasePct',    label: 'Chase%',   format: Utils.formatPct, sortType: 'numeric', desc: 'Out-of-zone swing rate', group: 'stats' },
    { key: 'twoStrikeWhiffPct', label: '2K Whiff%', format: Utils.formatPct, sortType: 'numeric', desc: 'Whiff rate on 2-strike swings — putaway ability', group: 'stats' },
    { key: 'fpsPct',      label: 'FPS%',     format: Utils.formatPct, sortType: 'numeric', desc: 'First-pitch strike rate', group: 'stats' },
    { key: 'oneOneWinPct', label: '1-1 Win%', format: Utils.formatPct, sortType: 'numeric', desc: 'Strike rate on 1-1 counts (strikes + BIP / total 1-1 pitches)', group: 'stats' },
    { key: 'earlyActionPct', label: 'EarlyAction%', format: Utils.formatPct, sortType: 'numeric', desc: 'PAs ending in 3 or fewer pitches / total PAs', group: 'stats' },
  ],
  hitterStats: [
    { key: '_rank',       label: '#',        format: function(v){ return v; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, group: 'info', width: '36px' },
    { key: 'hitter',      label: 'Hitter',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', sticky: true, cls: 'col-pitcher', noPercentile: true, noToggle: true, group: 'info' },
    { key: 'team',        label: 'Team',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isTeam: true, sticky: true, stickyIdx: 1 },
    { key: 'stands',      label: 'Bats',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info' },
    { key: 'pa',          label: 'PA',       format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'avg',         label: 'AVG',      format: Utils.formatDecimal(3), sortType: 'numeric', sectionStart: true, desc: 'Batting average', group: 'stats' },
    { key: 'obp',         label: 'OBP',      format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'On-base percentage', group: 'stats' },
    { key: 'slg',         label: 'SLG',      format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Slugging percentage', group: 'stats' },
    { key: 'wOBA',        label: 'wOBA',     format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Weighted on-base average — plate outcomes weighted by run value', group: 'stats' },
    { key: 'xwOBA',       label: 'xwOBA',    format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected wOBA (Statcast, EV + LA)', group: 'expected' },
    { key: 'wRCplus',     label: 'wRC+',     format: Utils.formatInt, sortType: 'numeric', desc: 'Total offensive production per PA, park-adjusted. 100 = league average and every point is 1% better or worse: 130 = 30% better than average', group: 'expected' },
    { key: 'xWRCplus',    label: 'xWRC+',    format: Utils.formatInt, sortType: 'numeric', desc: 'wRC+ computed from expected wOBA instead of results — judges contact quality rather than outcomes. Printed at 0.85 x wRC+\'s spread, its measured agreement with wRC+ (2021-2026): an estimate never prints wider than the thing it estimates, so values read directly as expected production.', group: 'expected' },
    { key: 'hitterPlus',  label: 'Hitter+',  format: Utils.formatInt, sortType: 'numeric', desc: 'Hitter composite: weighted blend of BB+ (contact quality), SD+ (swing decisions), CT+ (contact rate). 100 = league avg and one point is one percent, exactly like wRC+: 115 means 15% better at producing runs. The spread is deflated to r x wRC+\'s SD, with r measured live each run, which is what makes that slope exactly 1 (printing it at wRC+\'s FULL spread would drop the slope to r and make 115 worth only ~12%). A gap vs wRC+ reads as process ahead of/behind results.', group: 'expected' },
    { key: 'kPct',        label: 'K%',       format: Utils.formatPct, sortType: 'numeric', desc: 'Strikeout rate (K / PA)', group: 'stats' },
    { key: 'bbPct',       label: 'BB%',      format: Utils.formatPct, sortType: 'numeric', desc: 'Walk rate (uBB / PA, excludes IBB)', group: 'stats' },
    { key: 'hr',          label: 'HR',       format: Utils.formatInt, sortType: 'numeric', group: 'counting' },
    { key: 'sb',          label: 'SB',       format: Utils.formatInt, sortType: 'numeric', sectionStart: true, group: 'baserunning' },
    { key: 'sprintSpeed', rocHide: true, label: 'Sprint Speed', format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Sprint speed on max-effort runs (ft/s): 27 = league average, 30 = elite', group: 'baserunning' },
    { key: 'g',           label: 'G',        format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'ab',          label: 'AB',       format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'ops',         label: 'OPS',      format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'OBP + SLG', group: 'stats' },
    { key: 'iso',         label: 'ISO',      format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Isolated power (SLG − AVG)', group: 'stats' },
    { key: 'babip',       label: 'BABIP',    format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Batting average on balls in play. Sustained highs can be real (speed, line drives), but big swings from career norms usually regress', group: 'stats' },
    { key: 'bbToK',       label: 'BB/K',     format: Utils.formatDecimal(2), sortType: 'numeric', desc: 'Walks per strikeout (uBB / K). Higher = more discipline.', group: 'stats' },
    { key: 'xwOBAcon',    label: 'xwOBAcon', format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected wOBA on contact — avg xwOBA on BIP only', group: 'expected' },
    { key: 'xBA',         label: 'xBA',      format: Utils.formatDecimal(3), sortType: 'numeric', sectionStart: true, desc: 'Expected BA (Statcast, EV + LA)', group: 'expected' },
    { key: 'xSLG',        label: 'xSLG',     format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected SLG (Statcast, EV + LA)', group: 'expected' },
    { key: 'doubles',     label: '2B',       format: Utils.formatInt, sortType: 'numeric', sectionStart: true, noPercentile: true, group: 'counting' },
    { key: 'triples',     label: '3B',       format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'counting' },
    { key: 'cs',          label: 'CS',       format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'baserunning' },
    { key: 'sbPct',       label: 'SB%',      format: function(v){ return v != null ? v.toFixed(1) + '%' : '—'; }, sortType: 'numeric', noPercentile: true, desc: 'Stolen base success rate', group: 'baserunning' },
  ],
  hitterBattedBall: [
    { key: '_rank',       label: '#',        format: function(v){ return v; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, group: 'info', width: '36px' },
    { key: 'hitter',      label: 'Hitter',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', sticky: true, cls: 'col-pitcher', noPercentile: true, noToggle: true, group: 'info' },
    { key: 'team',        label: 'Team',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isTeam: true, sticky: true, stickyIdx: 1 },
    { key: 'stands',      label: 'Bats',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info' },
    { key: 'pa',          label: 'PA',       format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'count',       label: 'Pitches',  format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'nBip',        label: 'BIP',      format: Utils.formatInt, sortType: 'numeric', noPercentile: true, desc: 'Balls in play — every non-bunt batted ball, home runs included', group: 'info' },
    { key: 'pitchType',   label: 'Pitch',    format: function(v){ return v || ''; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isPitchType: true },
    { key: 'ev50',        label: 'EV50',     format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Avg EV of top 50% hardest-hit BIP (mph)', group: 'ev' },
    { key: 'maxEV',       label: 'Max EV',   format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Maximum exit velocity (mph)', group: 'ev' },
    { key: 'hardHitPct',  label: 'Hard-Hit%', format: Utils.formatPct, sortType: 'numeric', desc: 'Pct of trackable BIP with EV ≥ 95 mph (denominator = BIP with valid EV)', group: 'quality' },
    { key: 'barrelPct',   label: 'Barrel%',  format: Utils.formatPct, sortType: 'numeric', desc: 'Barrel rate — barrels are the EV/LA combos that historically return at least a .500 BA and 1.500 SLG (starts at 98 mph EV). Denominator = BIP with valid EV', group: 'quality' },
    { key: 'xwOBAcon',   label: 'xwOBAcon', format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected wOBA on contact — avg xwOBA on BIP only', group: 'expected' },
    { key: 'xwOBAsp',    label: 'xwOBAsp',  format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected wOBA from spray direction + launch angle — judges where the ball was hit rather than how hard. Complements xwOBA, which ignores direction entirely', group: 'expected' },
    { key: 'bbPlus',     label: 'BB+',      format: Utils.formatInt, sortType: 'numeric', desc: 'Batted-ball contact quality: mean xwOBAcon (60%) blended with 95th-percentile exit velocity (40%), each regressed to league by sample before blending. The EV term is first converted into xwOBAcon units, so the scale is unchanged: 100 = league avg and one point is one percent, meaning 115 is 15% better contact quality than league. EV95 is in there because a hard-hit ceiling is a far steadier read of contact skill than an average of individual batted balls. Known limitation: the EV95 half is blind to launch angle, so BB+ reads about 1.5 points high for hitters who hit the ball hard but on the ground, and about 1.5 points low for steep fly-ball hitters. Measured 2021-2026. Check GB% and median LA alongside it for those profiles.', group: 'expected' },
    { key: 'gbPct',       label: 'GB%',      format: Utils.formatPct, sortType: 'numeric', sectionStart: true, desc: 'Ground ball rate', group: 'composition' },
    { key: 'fbPct',       label: 'FB%',      format: Utils.formatPct, sortType: 'numeric', desc: 'Fly ball rate', group: 'composition' },
    { key: 'hrFbPct',     label: 'HR/FB',    format: Utils.formatPct, sortType: 'numeric', desc: 'Home runs per fly ball', group: 'composition' },
    { key: 'pullPct',     label: 'Pull%',    format: Utils.formatPct, sortType: 'numeric', sectionStart: true, desc: 'Pull rate (BIP to pull side)', group: 'spray' },
    { key: 'airPullPct',  label: 'AirPull%', format: Utils.formatPct, sortType: 'numeric', desc: 'Air-pull rate — pulled (LD + FB + PU) / total air balls (LD + FB + PU)', group: 'spray' },
    { key: 'nSwings',     label: 'Swings',   format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'avgEVAll',    label: 'Avg EV',       format: Utils.formatDecimal(1), sortType: 'numeric', sectionStart: true, desc: 'Average exit velocity on all BIP (mph)', group: 'ev' },
    { key: 'medLA',       label: 'Med LA',   format: Utils.formatDecimal(1), sortType: 'numeric', noPercentile: true, sectionStart: true, desc: 'Median launch angle (degrees)', group: 'quality' },
    { key: 'ldPct',       label: 'LD%',      format: Utils.formatPct, sortType: 'numeric', desc: 'Line drive rate', group: 'composition' },
    { key: 'puPct',       label: 'PU%',      format: Utils.formatPct, sortType: 'numeric', desc: 'Popup rate (lower is better — popups almost always out)', group: 'composition' },
    { key: 'sprayVal',   label: 'SprayVal', format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Placement skill in wOBA points — value added by WHERE balls go (mostly pulled air balls) beyond how high they are hit. The trait behind consistent xwOBA-beaters', group: 'expected' },
    { key: 'xwOBA',       label: 'xwOBA',    format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected wOBA (Statcast, EV + LA)', group: 'expected' },
    { key: 'xBA',         label: 'xBA',      format: Utils.formatDecimal(3), sortType: 'numeric', sectionStart: true, desc: 'Expected BA (Statcast, EV + LA)', group: 'expected' },
    { key: 'xSLG',        label: 'xSLG',     format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected SLG (Statcast, EV + LA)', group: 'expected' },
    { key: 'middlePct',   label: 'Middle%',  format: Utils.formatPct, sortType: 'numeric', noPercentile: true, showAvg: true, desc: 'Center rate (BIP up the middle)', group: 'spray' },
    { key: 'oppoPct',     label: 'Oppo%',    format: Utils.formatPct, sortType: 'numeric', noPercentile: true, showAvg: true, desc: 'Oppo rate (BIP to opposite field)', group: 'spray' },
    { key: 'avgFbDist',   label: 'Avg FB Dist', format: Utils.formatInt, sortType: 'numeric', noPercentile: true, showAvg: true, desc: 'Average fly ball distance (feet)', sectionStart: true, group: 'distance' },
    { key: 'avgHrDist',   label: 'Avg HR Dist', format: Utils.formatInt, sortType: 'numeric', noPercentile: true, showAvg: true, desc: 'Average home run distance (feet)', group: 'distance' },
  ],
  hitterSwingDecisions: [
    { key: '_rank',       label: '#',        format: function(v){ return v; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, group: 'info', width: '36px' },
    { key: 'hitter',      label: 'Hitter',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', sticky: true, cls: 'col-pitcher', noPercentile: true, noToggle: true, group: 'info' },
    { key: 'team',        label: 'Team',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isTeam: true, sticky: true, stickyIdx: 1 },
    { key: 'stands',      label: 'Bats',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info' },
    { key: 'pa',          label: 'PA',       format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'count',       label: 'Pitches',  format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'nSwings',     label: 'Swings',   format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'pitchType',   label: 'Pitch',    format: function(v){ return v || ''; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isPitchType: true },
    { key: 'sdPlus',      label: 'SD+',      format: Utils.formatInt, sortType: 'numeric', desc: 'Swing-Decisions+ — did he swing at the right pitches? Every swing/take graded by the run value of that choice for its location and count, umpire-independent via the true zone. 100 = league avg and one point is one percent: 115 means his average swing decision was worth 15% more runs than the league\'s. Measured 2021-2026: one SD of swing decisions carries about 6 wRC+ points, so this is a real skill with a modest run payoff — a big SD+ edge is worth less in runs than the same edge in BB+.', group: 'discipline' },
    { key: 'ctPlus',      label: 'CT+',      format: Utils.formatInt, sortType: 'numeric', desc: 'Contact+: leverage-weighted contact rate on swings. Frequency only — contact quality lives in BB+. 100 = league avg and one point is one percent: 115 means he made 15% more leverage-weighted contact than the league would on the same swings. Measured 2021-2026: frequency alone carries roughly ZERO wRC+ (quantity trades against quality), so a high CT+ is not by itself a production claim; its payoff arrives via the Hitter+ weights.', group: 'discipline' },
    { key: 'swingPct',    label: 'Swing%',   format: Utils.formatPct, sortType: 'numeric', sectionStart: true, desc: 'Overall swing rate (swings / pitches seen). Coloured higher = more aggressive, NOT better: swinging more only helps if it is aimed well. Read it beside Chase% and Z-Sw% - Chase%', group: 'discipline' },
    { key: 'izSwingPct',  label: 'Z-Swing%',    format: Utils.formatPct, sortType: 'numeric', desc: 'In-zone swing rate', group: 'discipline' },
    { key: 'chasePct',    label: 'Chase%',   format: Utils.formatPct, sortType: 'numeric', desc: 'Out-of-zone swing rate', group: 'discipline' },
    { key: 'izSwChase',   label: 'Z-Sw% - Chase%',  format: Utils.formatPct, sortType: 'numeric', desc: 'Z-Swing% minus Chase% — swings at strikes minus swings at balls. Higher = attacking the zone while laying off junk', group: 'discipline' },
    { key: 'contactPct',  label: 'Contact%', format: Utils.formatPct, sortType: 'numeric', desc: 'Contact rate excluding bunts (contact / swings)', group: 'discipline' },
    { key: 'izContactPct', label: 'Z-Contact%',   format: Utils.formatPct, sortType: 'numeric', desc: 'In-zone contact rate', group: 'discipline' },
    { key: 'firstPitchSwingPct', label: 'FPSw%',  format: Utils.formatPct, sortType: 'numeric', desc: 'First-pitch swing rate (% of PAs swinging on 0-0)', group: 'discipline' },
    { key: 'whiffPct',    label: 'Whiff%',   format: Utils.formatPct, sortType: 'numeric', desc: 'Whiff rate (misses / swings)', group: 'discipline' },
    { key: 'twoStrikeWhiffPct', label: '2K Whiff%', format: Utils.formatPct, sortType: 'numeric', desc: 'Whiff rate on 2-strike swings — lower means better at battling with two strikes', group: 'discipline' },
  ],
  hitterBatTracking: [
    { key: '_rank',       label: '#',        format: function(v){ return v; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, group: 'info', width: '36px' },
    { key: 'hitter',      label: 'Hitter',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', sticky: true, cls: 'col-pitcher', noPercentile: true, noToggle: true, group: 'info' },
    { key: 'team',        label: 'Team',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isTeam: true, sticky: true, stickyIdx: 1 },
    { key: 'stands',      label: 'Bats',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info' },
    { key: 'nCompSwings', rocHide: true, label: 'Comp. Swings', format: Utils.formatInt, sortType: 'numeric', noPercentile: true, desc: 'Competitive swings tracked by Statcast — excludes checked swings and other non-competitive swings. The denominator for all bat-tracking stats', group: 'info' },
    { key: 'batSpeed', rocHide: true,    label: 'Bat Speed', format: Utils.formatDecimal(1), sortType: 'numeric', sectionStart: true, desc: 'Avg bat speed on competitive swings (mph): 71 = league average, 75+ = Statcast\'s fast-swing bar', group: 'bat_tracking' },
    { key: 'swingLength', rocHide: true, label: 'Swing Length', format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Avg swing length — total bat-head distance from start to contact (feet)', group: 'bat_tracking' },
    { key: 'attackAngle', rocHide: true, label: 'Attack Angle', format: Utils.formatDecimal(1), sortType: 'numeric', noPercentile: true, showAvg: true, desc: 'Avg attack angle — bat direction at contact (degrees, positive = upward)', group: 'bat_tracking' },
    { key: 'squaredUpPct', rocHide: true, label: 'Squared-Up%', format: Utils.formatPct, sortType: 'numeric', desc: 'Pct of competitive swings converting at least 80% of the max possible exit velo given bat and pitch speed — pure contact efficiency, independent of swing speed', group: 'bat_tracking' },
    { key: 'blastPct', rocHide: true, label: 'Blast%', format: Utils.formatPct, sortType: 'numeric', desc: 'Squared-up contact on a fast swing (bat speed ≥75 mph) — the best single bat-tracking outcome', group: 'bat_tracking' },
    { key: 'idealAAPct', rocHide: true, label: 'IdealAtkAngle%', format: Utils.formatPct, sortType: 'numeric', desc: 'Pct of competitive swings with attack angle in the 5–20° ideal range', group: 'bat_tracking' },
    { key: 'attackDirection', rocHide: true, label: 'Attack Dir', format: Utils.formatDecimal(1), sortType: 'numeric', noPercentile: true, showAvg: true, desc: 'Avg attack direction at contact (degrees, positive = pull side)', group: 'bat_tracking' },
    { key: 'swingPathTilt', rocHide: true, label: 'Path Tilt', format: Utils.formatDecimal(1), sortType: 'numeric', noPercentile: true, showAvg: true, desc: 'Avg swing path tilt — bat path angle over 40ms before contact (degrees)', group: 'bat_tracking' },
  ],
  hitterPitch: [
    { key: '_rank',       label: '#',        format: function(v){ return v; }, sortType: null, align: 'center', noPercentile: true, noToggle: true, group: 'info', width: '36px' },
    { key: 'hitter',      label: 'Hitter',   format: function(v){ return v || '--'; }, sortType: 'string', align: 'left', sticky: true, cls: 'col-pitcher', noPercentile: true, noToggle: true, group: 'info' },
    { key: 'team',        label: 'Team',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isTeam: true, sticky: true, stickyIdx: 1 },
    { key: 'stands',      label: 'Bats',     format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info' },
    { key: 'pitchType',   label: 'Pitch',    format: function(v){ return v || '--'; }, sortType: 'string', align: 'center', noPercentile: true, group: 'info', isPitchType: true },
    { key: 'seenPct',     label: '% Seen',   format: Utils.formatPct, sortType: 'numeric', noPercentile: true, desc: 'Pct of pitches seen of this type', group: 'info' },
    { key: 'count',       label: 'Pitches',  format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'nSwings',     label: 'Swings',   format: Utils.formatInt, sortType: 'numeric', noPercentile: true, group: 'info' },
    { key: 'nBip',        label: 'BIP',      format: Utils.formatInt, sortType: 'numeric', noPercentile: true, desc: 'Balls in play — every non-bunt batted ball, home runs included', group: 'info' },
    { key: 'rv100',                      label: 'RV/100',   format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Run value per 100 pitches of this type (positive = better for hitter)', group: 'info' },
    { key: 'xRv100',     label: 'xRV/100',  format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Expected run value per 100 pitches of this type (positive = better for hitter)', group: 'info' },
    { key: 'wOBA',        label: 'wOBA',     format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Weighted on-base average vs this pitch type', group: 'stats' },
    { key: 'xwOBA',       label: 'xwOBA',    format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected wOBA vs this pitch type (Statcast, EV + LA)', group: 'stats' },
    { key: 'whiffPct',    label: 'Whiff%',   format: Utils.formatPct, sortType: 'numeric', desc: 'Whiff rate (misses / swings)', group: 'discipline' },
    { key: 'chasePct',    label: 'Chase%',   format: Utils.formatPct, sortType: 'numeric', desc: 'Out-of-zone swing rate', group: 'discipline' },
    { key: 'hardHitPct',  label: 'Hard-Hit%', format: Utils.formatPct, sortType: 'numeric', desc: 'Pct of BIP with EV ≥ 95 mph vs this pitch type (denominator = BIP with valid EV)', group: 'batted_ball' },
    { key: 'barrelPct',   label: 'Barrel%',  format: Utils.formatPct, sortType: 'numeric', desc: 'Barrel rate — barrels are the EV/LA combos that historically return at least a .500 BA and 1.500 SLG (starts at 98 mph EV). Denominator = BIP with valid EV', group: 'batted_ball' },
    { key: 'gbPct',       label: 'GB%',      format: Utils.formatPct, sortType: 'numeric', sectionStart: true, desc: 'Ground ball rate', group: 'batted_ball' },
    { key: 'runValue',                   label: 'PitchRV',  format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Pitch-level run value vs this pitch type (positive = better for hitter)', group: 'info' },
    { key: 'xRunValue',  label: 'xPitchRV', format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Expected pitch-level run value vs this pitch type (positive = better for hitter)', group: 'info' },
    { key: 'avg',         label: 'AVG',      format: Utils.formatDecimal(3), sortType: 'numeric', sectionStart: true, desc: 'Batting average vs this pitch type', group: 'stats' },
    { key: 'slg',         label: 'SLG',      format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Slugging vs this pitch type', group: 'stats' },
    { key: 'iso',         label: 'ISO',      format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Isolated power vs this pitch type (SLG − AVG)', group: 'stats' },
    { key: 'xBA',         label: 'xBA',      format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected BA vs this pitch type (Statcast, EV + LA)', group: 'stats' },
    { key: 'xSLG',        label: 'xSLG',     format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected SLG vs this pitch type (Statcast, EV + LA)', group: 'stats' },
    { key: 'xwOBAcon',    label: 'xwOBAcon', format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected wOBA on contact vs this pitch type — avg xwOBA on BIP only', group: 'stats' },
    { key: 'xwOBAsp',     label: 'xwOBAsp',  format: Utils.formatDecimal(3), sortType: 'numeric', desc: 'Expected wOBA vs this pitch type from spray direction + launch angle — judges where the ball was hit rather than how hard', group: 'stats' },
    { key: 'ev50',        label: 'EV50',     format: Utils.formatDecimal(1), sortType: 'numeric', sectionStart: true, desc: 'Avg EV of top 50% hardest-hit BIP (mph)', group: 'ev' },
    { key: 'maxEV',       label: 'Max EV',   format: Utils.formatDecimal(1), sortType: 'numeric', desc: 'Maximum exit velocity (mph)', group: 'ev' },
    { key: 'medLA',       label: 'Med LA',   format: Utils.formatDecimal(1), sortType: 'numeric', noPercentile: true, sectionStart: true, desc: 'Median launch angle (degrees)', group: 'batted_ball' },
    { key: 'ldPct',       label: 'LD%',      format: Utils.formatPct, sortType: 'numeric', noPercentile: true, desc: 'Line drive rate', group: 'batted_ball' },
    { key: 'fbPct',       label: 'FB%',      format: Utils.formatPct, sortType: 'numeric', noPercentile: true, desc: 'Fly ball rate', group: 'batted_ball' },
    { key: 'hrFbPct',     label: 'HR/FB',    format: Utils.formatPct, sortType: 'numeric', desc: 'Home runs per fly ball vs this pitch type', group: 'batted_ball' },
    { key: 'pullPct',     label: 'Pull%',    format: Utils.formatPct, sortType: 'numeric', noPercentile: true, sectionStart: true, desc: 'Pull rate', group: 'spray' },
    { key: 'oppoPct',     label: 'Oppo%',    format: Utils.formatPct, sortType: 'numeric', noPercentile: true, desc: 'Opposite field rate', group: 'spray' },
    { key: 'swingPct',    label: 'Swing%',   format: Utils.formatPct, sortType: 'numeric', sectionStart: true, desc: 'Swing rate vs this pitch type. Coloured higher = more aggressive, not better', group: 'discipline' },
    { key: 'izSwingPct',  label: 'Z-Swing%',    format: Utils.formatPct, sortType: 'numeric', desc: 'In-zone swing rate', group: 'discipline' },
    { key: 'contactPct',  label: 'Contact%', format: Utils.formatPct, sortType: 'numeric', desc: 'Contact rate excluding bunts (contact / swings)', group: 'discipline' },
    { key: 'izContactPct', label: 'Z-Contact%',   format: Utils.formatPct, sortType: 'numeric', desc: 'In-zone contact rate', group: 'discipline' },
  ],
};

const Leaderboard = {
  currentSort: { key: null, dir: 'desc' },
  hiddenColumns: {},  // key -> true if hidden
  showLeagueAvg: true,
  currentPage: 1,
  pageSize: 50,
  selectedForCompare: {},  // pitcher name -> true
  keyboardFocusIndex: -1,

  _tabDefaultHidden: {},  // tracks which keys were hidden by tab defaults

  _TAB_HIDDEN_DEFAULTS: {
    pitchMetrics:          ['maxVelo', 'vaa', 'haa', 'cswPct', 'barrelPctAgainst', 'xwOBA', 'xBA', 'xSLG', 'rv100', 'runValue', 'xRunValue', 'rvoe', 'xrvoe', 'rvoe100', 'xrvoe100'],
    pitcherStats:          ['w', 'l', 'sv', 'hld', 'tbf', 'siera', 'fip', 'xFIP', 'pitchingScore', 'hdERAPlus', 'hpERAPlus', 'runValue', 'xRunValue', 'rvoe', 'xrvoe', 'rvoe100'],
    pitcherSwingDecisions: ['twoStrikeWhiffPct', 'fpsPct', 'oneOneWinPct', 'earlyActionPct'],
    pitcherBattedBall:     ['hrFbPct', 'xwOBAsp', 'xBA', 'xSLG', 'maxEVAgainst'],
    hitterStats:           ['g', 'ab', 'ops', 'iso', 'babip', 'bbToK', 'xwOBAcon', 'xBA', 'xSLG', 'doubles', 'triples', 'cs', 'sbPct'],
    hitterBattedBall:      ['nSwings', 'avgEVAll', 'medLA', 'ldPct', 'puPct', 'sprayVal', 'xwOBA', 'xBA', 'xSLG', 'middlePct', 'oppoPct', 'avgFbDist', 'avgHrDist'],
    hitterSwingDecisions:  ['firstPitchSwingPct', 'whiffPct', 'twoStrikeWhiffPct'],
    hitterBatTracking:     ['attackDirection', 'swingPathTilt'],
    hitterPitch:           ['runValue', 'xRunValue', 'avg', 'slg', 'iso', 'xBA', 'xSLG', 'xwOBAcon', 'xwOBAsp', 'ev50', 'maxEV', 'medLA', 'ldPct', 'fbPct', 'hrFbPct', 'pullPct', 'oppoPct', 'swingPct', 'izSwingPct', 'contactPct', 'izContactPct'],
  },

  initHiddenColumns: function (tab) {
    // Clear previous tab-default hiding
    var self = this;
    Object.keys(this._tabDefaultHidden).forEach(function (k) {
      if (self._tabDefaultHidden[k]) {
        delete self.hiddenColumns[k];
      }
    });
    this._tabDefaultHidden = {};

    // Always hide these regardless of tab
    this.hiddenColumns['vaa'] = true;
    this.hiddenColumns['haa'] = true;
    // Pitcher+ Proj (2026-07-28, per Wally): hidden across every leaderboard.
    // Not deleted — the column stays in the picker and on the player page, so
    // the projection is still reachable; it just isn't a default leaderboard
    // column. Deliberately NOT registered in _tabDefaultHidden, which is
    // cleared on tab switch — this one persists like vaa/haa.
    this.hiddenColumns['pitcherPlusProj'] = true;

    // Per-tab defaults
    var defaults = this._TAB_HIDDEN_DEFAULTS[tab];
    if (defaults) {
      for (var i = 0; i < defaults.length; i++) {
        this.hiddenColumns[defaults[i]] = true;
        this._tabDefaultHidden[defaults[i]] = true;
      }
    }
  },

  getVisibleColumns: function (columns, data) {
    const self = this;
    // ROC-only view: hide columns flagged rocHide. Triggers when every visible
    // row has team === 'ROC' (e.g. user filtered the team dropdown to ROC).
    // For mixed views (All Teams), the column stays so non-ROC rows still
    // display their values.
    let allROC = false;
    if (data && data.length > 0) {
      allROC = data.every(function (r) { return r && r.team === 'ROC'; });
    }
    return columns.filter(function (col) {
      if (self.hiddenColumns[col.key]) return false;
      if (allROC && col.rocHide) return false;
      return true;
    });
  },

  /**
   * Sort data in-place by the given column, toggling asc/desc on repeated clicks.
   * @param {(PitcherRow|PitchRow|HitterRow)[]} data - Row array to sort.
   * @param {string} columnKey - The ColumnDef.key to sort by.
   * @param {ColumnDef[]} columns - Full column definition list.
   * @returns {(PitcherRow|PitchRow|HitterRow)[]} The sorted data (same reference).
   */
  sortData: function (data, columnKey, columns) {
    let col = null;
    for (let i = 0; i < columns.length; i++) {
      if (columns[i].key === columnKey) { col = columns[i]; break; }
    }
    if (!col || col.sortType === null) return data;

    const sortKey = col.sortKey || col.key;

    // For numeric columns that carry a percentile, infer whether the stat is
    // "lower is better" (SIERA, ERA, BB% for hitters, etc.) from the sample
    // covariance between raw value and `_pctl`. The pipeline always encodes
    // percentile so higher = better, so a negative covariance means low raw
    // values pair with high pctls. We use this to set the default first-click
    // sort direction so the BEST players land at the top regardless of which
    // way the underlying scale runs. Same column clicked again toggles, as
    // before. Non-percentile columns keep the existing high-to-low default.
    // Cheap: two passes over the already-in-memory data (~ms on 1500 rows),
    // and adapts automatically if a stat's direction ever flips per context
    // (e.g. K% is good for pitchers, bad for hitters — same key, different
    // table, correct direction inferred from the data on hand).
    let lowerIsBetter = false;
    if (col.sortType === 'numeric' && !col.noPercentile) {
      const pctlKey = col.key + '_pctl';
      let sumR = 0, sumP = 0, n = 0;
      for (let r = 0; r < data.length; r++) {
        const rv = data[r][sortKey];
        const pv = data[r][pctlKey];
        if (rv != null && pv != null) { sumR += rv; sumP += pv; n++; }
      }
      if (n >= 5) {
        const meanR = sumR / n;
        const meanP = sumP / n;
        let cov = 0;
        for (let r = 0; r < data.length; r++) {
          const rv = data[r][sortKey];
          const pv = data[r][pctlKey];
          if (rv != null && pv != null) cov += (rv - meanR) * (pv - meanP);
        }
        lowerIsBetter = cov < 0;
      }
    }

    if (this.currentSort.key === columnKey) {
      this.currentSort.dir = this.currentSort.dir === 'desc' ? 'asc' : 'desc';
    } else {
      this.currentSort.key = columnKey;
      if (col.sortType === 'string') {
        this.currentSort.dir = 'asc';
      } else {
        this.currentSort.dir = lowerIsBetter ? 'asc' : 'desc';
      }
    }

    const dir = this.currentSort.dir === 'asc' ? 1 : -1;

    data.sort(function (a, b) {
      const va = a[sortKey];
      const vb = b[sortKey];
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

  computeLeagueAvgRow: function (data, columns, opts) {
    const avg = {};
    const meta = DataStore.metadata || {};

    // When contextual filters (hand, role) are active, skip precomputed static
    // averages and compute dynamically from the filtered league data
    const hasContextualFilter = (opts.vsHand && opts.vsHand !== 'all') ||
                                 (opts.throws && opts.throws !== 'all') ||
                                 (opts.role && opts.role !== 'all');

    const isPitcher = data.length > 0 && data[0].pitcher;
    const overallAvgs = isPitcher ? (meta.pitcherLeagueAverages || {}) : (meta.hitterLeagueAverages || {});
    const pitchTypeAvgs = meta.leagueAverages || {};
    const pitchTypes = (opts && opts.pitchTypes) || ['all'];
    let precomputed;
    var isAllPitches = pitchTypes.indexOf('all') !== -1;
    if (!isAllPitches && pitchTypes.length === 1 && pitchTypeAvgs[pitchTypes[0]]) {
      precomputed = pitchTypeAvgs[pitchTypes[0]];
    } else if (!isAllPitches && pitchTypes.length > 1) {
      precomputed = {};
    } else {
      precomputed = overallAvgs;
    }

    // Stats that should recalculate when contextual filters are active.
    // Everything else (pitch metrics, plate discipline) keeps precomputed values.
    // Note (era/fip/xFIP/siera): the precomputed (no-filter) values come from
    // the pipeline as the QUALIFIED-POOL MEDIAN, which exactly aligns with the
    // percentile pool. The dynamic-filter branch below falls back to an
    // IP-weighted MEAN of the FILTERED subset, since (a) percentiles don't
    // re-rank against the filtered pool either way, and (b) computing a
    // median over an arbitrary filtered slice has no canonical reference.
    // Accepted small visual jump when toggling a filter on/off.
    var DYNAMIC_STATS = { runValue:1, rv100:1, xRunValue:1, xRv100:1, xBA:1, xSLG:1, wOBA:1, xwOBA:1, xwOBAcon:1, xwOBAsp:1, bbPlus:1,
                          era:1, fip:1, xFIP:1, siera:1, hdERA:1, hpERA:1,
                          avg:1, obp:1, slg:1, ops:1, iso:1 };

    // Keys where average should use absolute values (RHP/LHP have opposite signs)
    const ABS_AVG_KEYS = { horzBrk: true, haa: true, relPosX: true };
    const numericKeys = [];
    for (let i = 0; i < columns.length; i++) {
      const col = columns[i];
      if (col.sortType !== 'numeric' || col.key === '_rank') continue;
      // Include columns that have a percentile OR columns explicitly flagged
      // with showAvg: true (e.g., noPercentile columns like Arm Angle, Attack
      // Angle/Dir, Path Tilt, spray %s, distance — we still want to display a
      // league average row value for these).
      if (col.noPercentile && !col.showAvg) continue;
      numericKeys.push(col.key);
    }

    // IP parser for ERA/FIP-style stats (ip stored as string like "6.1")
    var _parseIP = Utils.parseIP;

    // Weight mapping — matches process_data.py precomputed average methodology
    var IP_WEIGHTED = { era:1, fip:1, xFIP:1, siera:1, hdERA:1, hpERA:1 };
    var BIP_WEIGHTED = { avgEVAgainst:1, maxEVAgainst:1, hardHitPct:1, barrelPctAgainst:1,
                          gbPct:1, ldPct:1, fbPct:1, hrFbPct:1, xwOBAsp:1, bbPlus:1,
                          avgEV:1, avgEVAll:1, ev50:1, maxEV:1, barrelPct:1, pullPct:1, airPullPct:1 };
    var PA_WEIGHTED = { wOBA:1, xBA:1, xSLG:1, xwOBA:1, xwOBAcon:1,
                         kPct:1, bbPct:1, kbbPct:1, babip:1,
                         avg:1, obp:1, slg:1, ops:1, iso:1 };

    numericKeys.forEach(function (key) {
      if (DYNAMIC_STATS[key]) {
        // Dynamic stats: use precomputed when no contextual filter, else compute from filtered data
        if (!hasContextualFilter && precomputed[key] !== undefined && precomputed[key] !== null) {
          avg[key] = precomputed[key];
          return;
        }
        var useAbs = ABS_AVG_KEYS[key] || false;
        var sumW = 0, totalW = 0;
        for (var j = 0; j < data.length; j++) {
          var v = data[j][key];
          if (v === null || v === undefined) continue;
          var w;
          if (IP_WEIGHTED[key]) {
            w = _parseIP(data[j].ip);
          } else if (BIP_WEIGHTED[key]) {
            w = data[j].nBip || 0;
          } else if (PA_WEIGHTED[key]) {
            w = data[j].pa || 0;
          } else {
            w = data[j].count || 0;
          }
          if (w > 0) {
            sumW += (useAbs ? Math.abs(v) : v) * w;
            totalW += w;
          }
        }
        avg[key] = totalW > 0 ? sumW / totalW : null;
      } else {
        // Non-dynamic stats (velo, spin, IVB, etc.): only use precomputed values.
        // Shows "--" when no precomputed value exists (e.g. all pitch types view).
        if (precomputed[key] !== undefined && precomputed[key] !== null) {
          avg[key] = precomputed[key];
        }
      }
    });
    avg.pitcher = 'League Avg';
    avg.hitter = 'League Avg';
    avg._isLeagueAvg = true;
    avg._rank = '';
    // wRC+, xWRC+, Stuff+, and Pitching+ are by definition 100 for league average
    avg.wRCplus = 100;
    avg.xWRCplus = 100;
    avg.stuffScore = 100;
    avg.pitchingScore = 100;
    avg.pitcherPlus = 100;
    avg.pitcherPlusProj = 100;
    avg.hdERAPlus = 100;
    avg.hpERAPlus = 100;
    // hdERA/hpERA league averages ride the same IP-weighted DYNAMIC_STATS
    // path as ERA/FIP/xFIP/SIERA — one weighting convention across the
    // whole ERA-scale family, so cross-column reads (deserved vs allowed)
    // compare like with like.
    return avg;
  },

  /**
   * Render the leaderboard table for the current page of data.
   * @param {(PitcherRow|PitchRow|HitterRow)[]} data - Full sorted dataset.
   * @param {ColumnDef[]} columns - Column definitions for the active tab.
   * @param {Object} [opts] - Render options (pitchTypes, vsHand, throws, role, tab).
   */
  render: function (data, columns, opts) {
    opts = opts || {};
    const self = this;
    const visCols = this.getVisibleColumns(columns, data);
    let headerRow = document.getElementById('table-header');
    const tbody = document.getElementById('table-body');
    const pinnedBody = document.getElementById('table-pinned-body');
    const noResults = document.getElementById('no-results');

    // ROC-filtered view where every stat column is rocHide'd (bat tracking:
    // no AAA measurement exists) — show an honest empty state instead of a
    // names-only table.
    const statCols = visCols.filter(function (c) { return c.group !== 'info' && !c.noToggle; });
    if (data.length > 0 && statCols.length === 0) {
      if (noResults) {
        noResults.style.display = '';
        noResults.textContent = 'Not tracked for AAA — bat tracking data exists only for MLB.';
      }
      if (tbody) tbody.innerHTML = '';
      if (pinnedBody) pinnedBody.innerHTML = '';
      if (headerRow) headerRow.innerHTML = '';
      return;
    }
    if (noResults && data.length > 0) noResults.style.display = 'none';

    this._lastRenderOpts = opts;

    const totalRows = data.length;
    const pageSize = this.pageSize;
    const totalPages = pageSize > 0 ? Math.max(1, Math.ceil(totalRows / pageSize)) : 1;
    if (this.currentPage > totalPages) this.currentPage = totalPages;
    const startIdx = pageSize > 0 ? (this.currentPage - 1) * pageSize : 0;
    const endIdx = pageSize > 0 ? Math.min(startIdx + pageSize, totalRows) : totalRows;
    const pageData = data.slice(startIdx, endIdx);

    const pageInfo = document.getElementById('page-info');
    const pagePrev = document.getElementById('page-prev');
    const pageNext = document.getElementById('page-next');
    if (pageInfo) pageInfo.textContent = 'Page ' + this.currentPage + ' of ' + totalPages;
    if (pagePrev) pagePrev.disabled = this.currentPage <= 1;
    if (pageNext) pageNext.disabled = this.currentPage >= totalPages;

    let thead = document.querySelector('#leaderboard-table thead');
    thead.innerHTML = '';

    const groupRow = document.createElement('tr');
    groupRow.id = 'table-group-header';
    const groupLabels = { info: '', rates: 'Rates', stats: 'Stats', metrics: 'Metrics', counting: 'Counting', advanced: 'Advanced', ev: 'Exit Velo', batted_ball: 'Batted Ball', spray: 'Spray', discipline: 'Discipline', bat_tracking: 'Bat Tracking', outcomes: 'Outcomes', expected: 'Expected', run_value: 'Run Value', quality: 'Quality', composition: 'Composition', supplemental: 'Supplemental', distance: 'Distance', baserunning: 'Baserunning' };
    let prevGroup = null;
    const groupSpans = [];
    visCols.forEach(function (col) {
      const g = col.group || 'info';
      if (g === prevGroup) {
        groupSpans[groupSpans.length - 1].span++;
      } else {
        groupSpans.push({ group: g, span: 1, sticky: col.sticky });
        prevGroup = g;
      }
    });
    const hasGroups = groupSpans.some(function (gs) { return groupLabels[gs.group]; });
    if (hasGroups) {
      let colIdx = 0;
      groupSpans.forEach(function (gs) {
        const th = document.createElement('th');
        th.setAttribute('colspan', gs.span);
        th.textContent = groupLabels[gs.group] || '';
        th.classList.add('group-header-cell');
        if (gs.sticky) { th.classList.add('sticky-col'); th.classList.add('sticky-col-last'); }
        if (gs.group !== 'info' && groupLabels[gs.group]) th.classList.add('group-header-labeled');
        // Check if first col in this span is sectionStart
        if (visCols[colIdx] && visCols[colIdx].sectionStart) th.classList.add('section-start');
        colIdx += gs.span;
        groupRow.appendChild(th);
      });
      thead.appendChild(groupRow);
    }

    headerRow = document.createElement('tr');
    headerRow.id = 'table-header';
    visCols.forEach(function (col) {
      const th = document.createElement('th');
      if (col.isCompare) {
        th.classList.add('col-compare');
        th.style.width = col.width || 'auto';
      } else {
        const labelSpan = document.createElement('span');
        labelSpan.textContent = col.label;
        th.appendChild(labelSpan);
        if (col.sortType !== null) {
          const sortSpan = document.createElement('span');
          sortSpan.className = 'sort-indicator';
          sortSpan.style.display = 'inline-block';
          sortSpan.style.width = '12px';
          sortSpan.style.textAlign = 'center';
          sortSpan.style.fontSize = '9px';
          sortSpan.style.marginLeft = '2px';
          sortSpan.style.color = 'var(--accent)';
          if (self.currentSort.key === col.key) {
            sortSpan.textContent = self.currentSort.dir === 'asc' ? '\u25B2' : '\u25BC';
          }
          th.appendChild(sortSpan);
        }
      }
      th.setAttribute('data-key', col.key);
      if (col.align) th.classList.add('align-' + col.align);
      if (col.sticky) th.classList.add('sticky-col');
      if (col.stickyIdx === 1) th.classList.add('sticky-col-last');
      if (col.sectionStart) th.classList.add('section-start');
      if (col.width) th.style.width = col.width;
      if (col.desc) th.title = col.desc;
      else if (Utils.TOOLTIPS[col.label]) th.title = Utils.TOOLTIPS[col.label];

      if (self.currentSort.key === col.key) {
        th.classList.add('sorted', self.currentSort.dir);
        th.setAttribute('aria-sort', self.currentSort.dir === 'asc' ? 'ascending' : 'descending');
      }

      if (col.sortType !== null) {
        th.addEventListener('click', function () {
          self.sortData(data, col.key, columns);
          self.currentPage = 1;
          self.render(data, columns, self._lastRenderOpts);
        });
      }

      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);

    if (hasGroups) {
      const groupRowHeight = groupRow.offsetHeight || 25;
      for (let hi = 0; hi < headerRow.cells.length; hi++) {
        headerRow.cells[hi].style.top = groupRowHeight + 'px';
      }
    }

    this._stickyLeftOffsets = {};
    let firstStickyTh = null;
    for (let si = 0; si < visCols.length; si++) {
      if (visCols[si].sticky && !visCols[si].stickyIdx) {
        firstStickyTh = headerRow.cells[si];
        break;
      }
    }
    if (firstStickyTh) {
      const firstStickyWidth = firstStickyTh.offsetWidth;
      for (let si2 = 0; si2 < visCols.length; si2++) {
        if (visCols[si2].stickyIdx === 1) {
          headerRow.cells[si2].style.left = firstStickyWidth + 'px';
          this._stickyLeftOffsets[visCols[si2].key] = firstStickyWidth;
          if (hasGroups) {
          }
        }
      }
    }

    tbody.innerHTML = '';
    if (pinnedBody) pinnedBody.innerHTML = '';

    if (data.length === 0) {
      noResults.style.display = '';
      document.getElementById('row-count').textContent = '0';
      document.getElementById('pagination').style.display = 'none';
      return;
    }
    noResults.style.display = 'none';
    document.getElementById('pagination').style.display = '';

    // Pinned average rows
    if (pinnedBody && this.showLeagueAvg && data.length > 0) {
      const thead2 = document.querySelector('#leaderboard-table thead');
      const thHeight = thead2 ? thead2.offsetHeight : 36;

      // League Average: computed from all-teams data (ignores team filter)
      const leagueAvgData = opts.leagueData || data;
      const leagueAvgRow = this.computeLeagueAvgRow(leagueAvgData, visCols, opts);
      leagueAvgRow.pitcher = 'League Avg';
      leagueAvgRow.hitter = 'League Avg';
      if (opts.viewMode === 'team') leagueAvgRow.team = 'League Avg';
      const leagueTr = this._createRow(leagueAvgRow, visCols, -1, true);
      leagueTr.classList.add('league-avg-row');
      pinnedBody.appendChild(leagueTr);

      // Make league avg row cells sticky
      for (let ci = 0; ci < leagueTr.cells.length; ci++) {
        leagueTr.cells[ci].style.position = 'sticky';
        leagueTr.cells[ci].style.top = thHeight + 'px';
        leagueTr.cells[ci].style.zIndex = leagueTr.cells[ci].classList.contains('sticky-col') ? '5' : '3';
      }

    }

    const fragment = document.createDocumentFragment();

    for (let ri = 0; ri < pageData.length; ri++) {
      const row = pageData[ri];
      const globalRank = startIdx + ri + 1;
      const tr = this._createRow(row, visCols, globalRank, false);
      tr.classList.add('clickable-row');
      tr._playerName = row.pitcher || row.hitter;
      tr._rowData = row;
      tr._rowIndex = ri;

      if (this.keyboardFocusIndex === ri) {
        tr.classList.add('keyboard-focus');
      }

      fragment.appendChild(tr);
    }

    tbody.appendChild(fragment);

    // Warm the pitch-detail shards for the rows now on screen. Details are
    // fetched per pitcher rather than shipped in the payload, so without this
    // the first click on any player stalls on the network; ~50 rows is about
    // 500 KB and it runs at idle/low priority. Hover below covers rows the
    // user reaches by search or paging before this finishes.
    if (typeof DataStore !== 'undefined' && DataStore.prefetchPitchDetails) {
      var _pfKeys = [];
      for (var pi = 0; pi < pageData.length; pi++) {
        var _r = pageData[pi];
        if (_r.pitcher && !_r._isTeamRow) _pfKeys.push(_r.pitcher + '|' + _r.team);
      }
      DataStore.prefetchPitchDetails(_pfKeys);
    }

    if (tbody._delegatedHover) {
      tbody.removeEventListener('mouseover', tbody._delegatedHover);
    }
    tbody._delegatedHover = function (e) {
      var tr = e.target.closest && e.target.closest('tr.clickable-row');
      if (!tr || !tr._rowData || tr._rowData._isTeamRow) return;
      if (!tr._rowData.pitcher) return;
      DataStore.prefetchPitchDetails([tr._rowData.pitcher + '|' + tr._rowData.team]);
    };
    tbody.addEventListener('mouseover', tbody._delegatedHover);

    if (tbody._delegatedClick) {
      tbody.removeEventListener('click', tbody._delegatedClick);
    }
    tbody._delegatedClick = function (e) {
        if (e.target.type === 'checkbox') return;
        const tr = e.target.closest('tr.clickable-row');
        if (!tr || !tr._rowData) return;
        if (tr._rowData._isTeamRow) return; // no player side panel for team rows
        const prev = tbody.querySelectorAll('.active-row');
        for (let k = 0; k < prev.length; k++) prev[k].classList.remove('active-row');
        const personName = tr._playerName;
        const allRows = tbody.querySelectorAll('tr');
        allRows.forEach(function (row) {
          if (row._playerName === personName) row.classList.add('active-row');
        });
        self.keyboardFocusIndex = tr._rowIndex;
        const r = tr._rowData;
        if (typeof App !== 'undefined' && App.openSidePanel) {
          App.openSidePanel(personName, r.team, r.throws || r.stands, r);
        }
    };
    tbody.addEventListener('click', tbody._delegatedClick);
    document.getElementById('row-count').textContent = totalRows;
  },

  _createRow: function (row, visCols, rank, isAvgRow) {
    const self = this;
    const tr = document.createElement('tr');

    visCols.forEach(function (col) {
      const td = document.createElement('td');

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
          const cb = document.createElement('input');
          cb.type = 'checkbox';
          const compareKey = (row.pitcher || '') + '|' + (row.team || '');
          cb.checked = !!self.selectedForCompare[compareKey];
          cb.addEventListener('change', function () {
            if (cb.checked) {
              self.selectedForCompare[compareKey] = true;
            } else {
              delete self.selectedForCompare[compareKey];
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
        const badge = Utils.createPitchBadge(row[col.key], false);
        td.appendChild(badge);
        if (col.align) td.classList.add('align-' + col.align);
        tr.appendChild(td);
        return;
      }

      // Player name as clickable link to player page (pitcher or hitter)
      if ((col.key === 'pitcher' || col.key === 'hitter') && !isAvgRow && row.mlbId) {
        const nameLink = document.createElement('a');
        // Copyable link reproduces this row's view: per-team rows carry a
        // pteam pin; combined 2TM/3TM rows resolve to the full-season page.
        nameLink.href = '#player=' + row.mlbId +
          (row.team && !/^\d+TM$/.test(row.team) ? '&pteam=' + row.team : '');
        nameLink.className = 'pitcher-name-link';
        nameLink.textContent = col.format(row[col.key]);
        nameLink.addEventListener('click', function (e) {
          e.preventDefault();
          e.stopPropagation();
          PlayerPage.open(row.mlbId, row.team);
        });
        td.appendChild(nameLink);
        if (col.align) td.classList.add('align-' + col.align);
        if (col.sticky) {
          td.classList.add('sticky-col');
          if (col.stickyIdx === 1) { td.classList.add('sticky-col-last'); }
          if (col.stickyIdx === 1 && self._stickyLeftOffsets[col.key]) {
            td.style.left = self._stickyLeftOffsets[col.key] + 'px';
          }
        }
        if (col.cls) td.classList.add(col.cls);
        tr.appendChild(td);
        return;
      }

      // Regular cell
      const val = row[col.key];
      td.textContent = col.format(val);
      if (col.align) td.classList.add('align-' + col.align);
      if (col.sticky) {
        td.classList.add('sticky-col');
        if (col.stickyIdx === 1) { td.classList.add('sticky-col-last'); }
        if (col.stickyIdx === 1 && self._stickyLeftOffsets[col.key]) {
          td.style.left = self._stickyLeftOffsets[col.key] + 'px';
        }
      }
      if (col.cls) td.classList.add(col.cls);
      if (col.sectionStart) td.classList.add('section-start');
      if (val === null || val === undefined) td.classList.add('col-null');

      // Stuff+ low-model-support marker: this (pitcher, pitch type) sits far
      // from the model's training data (worst ~1.5% of units) — the score is
      // an extrapolation. Surfaced in the hover tooltip (app.js reads the
      // data attribute).
      if (col.key === 'stuffScore' && !isAvgRow && row.stuffScore_lowSupport &&
          val !== null && val !== undefined) {
        td.setAttribute('data-low-support', '1');
      }

      // Pitching+ run-value companion: surfaced in the hover tooltip (app.js
      // reads the data attribute) so the composite index also reads in runs.
      if (col.key === 'pitchingScore' && !isAvgRow &&
          row.pitchingRuns100 !== null && row.pitchingRuns100 !== undefined) {
        td.setAttribute('data-runs100', row.pitchingRuns100);
      }
      // Same for Pitcher+ — app.js words this one 'expected' rather than
      // 'vs avg' (its slope is predictive, not same-season).
      if (col.key === 'pitcherPlus' && !isAvgRow &&
          row.pitcherRuns100 !== null && row.pitcherRuns100 !== undefined) {
        td.setAttribute('data-runs100', row.pitcherRuns100);
      }

      // Percentile coloring (only for qualified players, with exceptions)
      if (!col.noPercentile && !isAvgRow) {
        const pctlKey = col.key + '_pctl';
        const pctl = row[pctlKey];
        if (pctl !== null && pctl !== undefined) {
          // Determine qualifying status
          const isPitcherRow = !!row.pitcher;
          const isHitterRow = !!row.hitter;
          const isPitcherPitchType = isPitcherRow && row.pitchType != null;
          const isHitterPitchType = isHitterRow && row.pitchType != null;
          const teamGames = Aggregator.loaded ? Aggregator.getTeamGamesPlayed() : {};
          const tg = teamGames[row.team] || 0;
          let showColor;
          // Pitch shape metrics always show color on pitch-type views (no min count)
          const PITCH_SHAPE_ALWAYS_COLOR = {
            velocity: true, spinRate: true, indVertBrk: true, horzBrk: true,
            extension: true,
            vaa: true, haa: true, nVAA: true
          };
          // Hitter BIP-gated stats: none by design.
          // Batted-ball tab stats all use PA qualification (3.1 PA/team game);
          // Max Exit Velo (maxEV) bypasses qualification via the always-color override below.
          const HITTER_BIP_STATS = {};
          // Hitter stats that use PA qualifier (3.1 PA/team game)
          const HITTER_PA_STATS = {
            babip: true, hrFbPct: true,
            xBA: true, xSLG: true, xwOBA: true, xwOBAcon: true,
            avgEVAll: true, ev50: true, medLA: true,
            hardHitPct: true, barrelPct: true,
            xwOBAsp: true, sprayVal: true, airPullPct: true, bbPlus: true, hitterPlus: true,
            gbPct: true, ldPct: true, fbPct: true,
            pullPct: true, oppoPct: true
          };
          // Hitter stats that retain the ≥10 competitive swings gate (bat speed + swing length only).
          // Blast% and IdealAtkAngle% now use PA qual like everything else.
          const HITTER_BAT_TRACKING = { batSpeed: true, swingLength: true };
          // Hitter stats that always color regardless of qualification.
          // maxEV: single-BIP value, sample-size invariant.
          // hr, sb: counting stats; sample-size moot.
          // sprintSpeed: running skill; no PA threshold makes sense.
          const HITTER_ALWAYS_COLOR = { maxEV: true, hr: true, sb: true, sprintSpeed: true };

          if (row._isTeamRow) {
            // Team rows: every MLB team is pool-qualified. ROC keeps its
            // interpolated rank for the tooltip but renders uncolored.
            showColor = !(Aggregator.loaded && Aggregator._isROCTeam(row.team));
          } else if (isPitcherPitchType) {
            // Pitcher pitch-type data: shape metrics always qualify; outcome metrics need minimum pitches.
            // BIP-denominated stats gate on BIP count instead — at the pitch-type
            // level a 25-pitch row often has <10 BIP behind xwOBAsp/EV stats
            // (52% of colored xwOBAsp cells had <20 BIP before this gate).
            const PITCH_BIP_GATED = { xwOBAsp: true, avgEVAgainst: true, hardHitPct: true,
                                      barrelPctAgainst: true, hrFbPct: true, ldPct: true,
                                      fbPct: true, puPct: true, gbPct: true, babip: true };
            if (PITCH_BIP_GATED[col.key]) {
              showColor = (row.nBip || 0) >= QUAL.MIN_BIP_PCTL;
            } else if (col.key === 'locPlus') {
              // Loc+ is displayed unshrunk, so it needs its pitch type's own
              // measured r=0.5 crossing instead of the flat 25 (an FF cell at
              // 25 pitches is only 0.26 reliable). Rank still shows in the
              // tooltip; this suppresses the color only.
              showColor = (row.count || 0) >= QUAL.locPlusMinPitches(row.pitchType);
            } else {
              showColor = PITCH_SHAPE_ALWAYS_COLOR[col.key] || (row.count || 0) >= QUAL.MIN_PITCH_PCTL;
            }
          } else if (isPitcherRow) {
            // Pitcher overall: IP-based qualification (ROC-aware:
            // MLB SP 1.0 / RP 0.5, ROC SP 0.8 / RP 0.4 × team games).
            const ipFloat = Utils.parseIP(row.ip);
            const isStarter = Utils.isStarter(row.g, row.gs);
            const isROC = Aggregator.loaded && Aggregator._isROCTeam(row.team);
            const ipThresh = tg * Utils.pitcherIpPerGame(isStarter, isROC);
            showColor = ipFloat >= ipThresh;
            // Pitcher always-color: extension
            if (!showColor) showColor = col.key === 'extension';
          } else if (isHitterPitchType) {
            // Hitter pitch-type: minimum pitches of that type seen
            showColor = (row.count || 0) >= QUAL.MIN_HITTER_PT;
          } else {
            // Hitter overall: per-stat qualification gates.
            // Always use overall-season PA (paAll) so platoon/date splits honour
            // overall qualification rather than requiring a split-specific minimum.
            // ROC-aware: MLB 3.1 PA×TG, ROC 2.7.
            const overallPa = (row.paAll != null ? row.paAll : row.pa) || 0;
            const isROC = Aggregator.loaded && Aggregator._isROCTeam(row.team);
            const paQual = overallPa >= tg * Utils.hitterPaPerGame(isROC);
            if (HITTER_ALWAYS_COLOR[col.key]) {
              showColor = true;
            } else if (HITTER_BIP_STATS[col.key]) {
              showColor = (row.nBip || 0) >= QUAL.MIN_BIP_PCTL;
            } else if (HITTER_BAT_TRACKING[col.key]) {
              showColor = (row.nCompSwings || 0) >= QUAL.MIN_BAT_TRACKING;
            } else if (HITTER_PA_STATS[col.key]) {
              showColor = paQual;
            } else {
              showColor = paQual;
            }
          }

          if (showColor) {
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

// Initialize hidden columns on load (default tab is pitcherStats)
Leaderboard.initHiddenColumns('pitcherStats');
