"""hwar.py — position-player hWAR: batting, baserunning, fielding, positional and replacement runs.

    hWAR = (hBatRuns + hBsrRuns + hFldRuns + hPosRuns + hReplRuns) / RPW
    RPW  = 4 r / (2 r)^WAR_PYTH_EXP with r = lgRA9, the season constant pitcher hWAR uses

BASERUNNING (hBsrRuns) = Savant Baserunning Run Value (extra-base advancement + steal
    attempts, tracking-based, fetch_baserunning_runs; a runner under Savant's minimum scores
    0 and is counted) + double-play runs: expected DP cost at the league rate over his
    opportunities minus his actual DP cost, where an opportunity is a PA with a runner on
    first and fewer than two outs and the cost of a GIDP is what the SECOND out and the
    erased runner take beyond an ordinary out in that base-out state, from the published
    24-state run-expectancy table (RE24, Tango 2010-2015; a labeled convention, the state mix
    is measured). Savant's own delta_run_exp cannot price it: on 2021-2025 it reads a GIDP at
    -.33 against -.30 for a force out, one out's worth, so it is not used here.
FIELDING (hFldRuns) = Savant Fielding Run Value total (range, arm, double play, framing,
    blocking, throwing; fetch_fielding_runs with min=1, 635 fielders in 2026), 0 when unlisted.
POSITIONAL (hPosRuns) = innings by position (MLB API, fetch_fielding_innings; a DH game counts
    nine innings) x HWAR_POS_ADJ / 1458, the fWAR values per 162 games. A convention.
REPLACEMENT (hReplRuns) = per PA, pinned so the club-row total of hWAR is exactly
    HWAR_REPL_SHARE x 1000 x (league games / 2430): the fWAR 57/43 split the pitcher side's 0.12
    wins per 9 is built on. About 19 runs per 600 PA in 2026 (fWAR: about 20).
Season-level components (fielding, baserunning, positional) are split across a traded
hitter's stint rows by PA share; his combined 2TM row carries the whole. ROC rows: None.
hWAR_se = batting sampling noise only: shrink x WAR_XW_PA_SD x sqrt(PA) / wOBAscale / RPW,
    on the linear-weights scale the batting runs use. Fielding and baserunning noise is not
    in it, so it is a floor (Savant publishes no error on FRV or BRV).

2026-09-05: batting runs first, the rest the same day.

    hBatRuns = (rate_adj - lgXW) x PA / wOBAscale        deserved batting runs above average

    rate      the hitter's PA-level xwOBA on the hitter basis, the same quantity the row's
              xwOBA column and xwRC+ stand on: Savant EV x LA per ball in play plus the
              pulled-air term (XWOBA_PULLAIR_C), K 0, BB and HBP at their weights; IBB, SH,
              CI and bunt ABs out (compute_expected_stats). Read at full precision from the
              private _xwOBAraw / _xwOBAn fields, never from the rounded column.
    shrink    (rate x n + HWAR_N0_BAT x lgXW) / (n + HWAR_N0_BAT), n = the rate's denominator
    rate_adj  shrink - HWAR_PARK_PASS_BAT x (exposure - 1) x lgRPA x wOBAscale
              exposure = (PF/100 + 1)/2 of his club on the Savant runs factor
              (data/park_factors.json, the pitcher side's file, eraplus._load_park); a
              combined 2TM.. row takes its MOST RECENT club (current_team_by_player, the
              same rule as pitcher hWAR); ROC rows get None
    lgXW      PA-weighted mean of the adjusted rate over MLB club rows (the shift above the raw
              league mean is reported), so batting runs above average sum to zero
    PA        official plate appearances, IBB included: an intentional walk is valued at the
              hitter's own rate (the fWAR convention) while the rate itself excludes it
    lgRPA, wOBAscale: FanGraphs Guts (metadata gutsConstants)

Why this rate (scripts/research/hitter/hwar_hitter_rate_validation.py, 2021-2026): Savant
xwOBA beats actual wOBA on every objective (split-half .54 vs .29, next season at 150 PA
+.088 5/5, rest of season +.099 6/6, the 50-batter disagreement set .51 vs .39). The
pulled-air basis loses .003 r of prediction to plain xwOBA and wins .007 r of same-season
description (5/6, hwar_hitter_rate_desc.py); a value metric is descriptive, and one
deserved-hitting number on the site beats two.
HWAR_N0_BAT: the shrink where the LOSO slope of actual wOBA on the park-adjusted rate is
1.0 (hwar_hitter_rate_desc.py: 77 PA, bracketed 75-100), so runs on the linear-weights
scale need no fitted slope. Next-season prediction prefers 300-500, but that is a
forecast, not a value. The hitter selection gradient is mild (.89/.82/.82 by PA tercile),
so the same-season criterion is usable here; it is not on the pitcher side.
HWAR_PARK_PASS_BAT: the share of the published runs factor that reaches xwOBA, measured
WITHIN batter, home minus road (hwar_park_pass_within.py: .35, LOSO .33-.38; actual wOBA
reads .99 in the same design). The across-batter exposure design is confounded by club
quality and read -.18; do not re-measure it that way.
"""
import math
from pipeline.utils import current_team_by_player, player_key, is_combined_team
from pipeline.eraplus import WAR_PYTH_EXP, WAR_XW_PA_SD

HWAR_N0_BAT = 77            # PA; LOSO calibration slope of actual wOBA on the park-adjusted rate = 1.0
HWAR_PARK_PASS_BAT = 0.35   # share of the published runs factor that reaches xwOBA, within batter


def hitter_park_map(rows, park, aaa_teams):
    """{id(row) -> runs park factor (100 = neutral)}: a club row takes its club, a combined
    row its most recent MLB club, a ROC row nothing. A club that fails to resolve is
    announced, because neutral is a real park factor and a silent one would look measured."""
    cur = current_team_by_player(rows, 'hitter', set(aaa_teams))
    out, unresolved = {}, []
    for r in rows:
        team = r.get('team')
        if team in aaa_teams:
            continue
        if is_combined_team(team):
            team = cur.get(player_key(r, 'hitter'))
        pf = park.get(team) if team is not None else None
        if pf is None:
            unresolved.append(f"{r.get('hitter')} ({r.get('team')})")
            continue
        out[id(r)] = pf
    if unresolved:
        print(f'  hwar WARNING: {len(unresolved)} hitter rows have no park factor and score '
              f'a NEUTRAL park: {", ".join(sorted(unresolved)[:6])}' + (' ...' if len(unresolved) > 6 else ''))
    return out


def apply_batting_runs(rows, park, lg_rpa, woba_scale, aaa_teams=('ROC', 'AAA')):
    """Writes hBatRuns on every row (None for ROC/AAA or no rate). Returns the constants
    bundle for metadata, or None when the inputs are missing (announced)."""
    aaa = set(aaa_teams)
    mlb = [r for r in rows if r.get('team') not in aaa and r.get('_xwOBAraw') is not None and (r.get('_xwOBAn') or 0) > 0]
    if not mlb or not lg_rpa or not woba_scale:
        print(f'  hwar WARNING: batting runs skipped (MLB rows with a rate {len(mlb)}, lgRPA {lg_rpa}, '
              f'wOBA scale {woba_scale}); hBatRuns left None')
        for r in rows:
            r['hBatRuns'] = None
        return None
    den = sum(r['_xwOBAn'] for r in mlb)
    lg_xw = sum(r['_xwOBAraw'] * r['_xwOBAn'] for r in mlb) / den
    pmap = hitter_park_map(rows, park, aaa)
    adj = {}
    for r in rows:
        n = r.get('_xwOBAn') or 0
        if r.get('team') in aaa or r.get('_xwOBAraw') is None or n <= 0 or not r.get('pa'):
            continue
        sh = (r['_xwOBAraw'] * n + HWAR_N0_BAT * lg_xw) / (n + HWAR_N0_BAT)
        exposure = (pmap.get(id(r), 100.0) / 100.0 + 1.0) / 2.0
        adj[id(r)] = sh - HWAR_PARK_PASS_BAT * (exposure - 1.0) * lg_rpa * woba_scale
    # Recenter on the PA-weighted mean of the adjusted rate over MLB CLUB rows (a combined
    # 2TM.. row repeats its stints), so batting runs above average sum to zero, as pitcher
    # hWAR's shift does. Without it the league summed to +377 runs (2026-09-05): the shrink
    # pulls a low-PA hitter toward the mean hardest, and low-PA hitters are below average.
    _num = _den = 0.0
    for r in rows:
        if id(r) in adj and not is_combined_team(r.get('team')):
            _num += adj[id(r)] * r['pa']; _den += r['pa']
    lg_adj = _num / _den if _den > 0 else lg_xw
    n_set = 0
    for r in rows:
        if id(r) not in adj:
            r['hBatRuns'] = None
            continue
        r['hBatRuns'] = round((adj[id(r)] - lg_adj) * r['pa'] / woba_scale, 2)
        n_set += 1
    club_sum = sum(r['hBatRuns'] for r in rows
                   if r.get('hBatRuns') is not None and not is_combined_team(r.get('team')))
    const = {'lgXW': round(lg_xw, 4), 'shift': round(lg_adj - lg_xw, 4), 'n0': HWAR_N0_BAT,
             'parkPass': HWAR_PARK_PASS_BAT, 'lgRPA': lg_rpa, 'wobaScale': woba_scale,
             'nRows': n_set, 'clubSum': round(club_sum, 1)}
    print(f'  hWAR batting runs: {n_set} rows, league xwOBA {lg_xw:.4f}, shift {lg_adj - lg_xw:+.4f}, '
          f'N0 {HWAR_N0_BAT} PA, park pass {HWAR_PARK_PASS_BAT}, club-row sum {club_sum:+.1f}')
    return const


# ── double plays: the cost of the second out, by base-out state ──
# RE24, runs expected from the state to the end of the inning (Tango, 2010-2015 published
# table; the 2026 run environment, 4.55 RA9, sits inside that era's range). A convention:
# the sheets carry no inning runs, so the table cannot be rebuilt in house yet.
RE24 = {'---': (0.481, 0.254, 0.098), '1--': (0.859, 0.509, 0.224), '-2-': (1.100, 0.664, 0.319),
        '--3': (1.352, 0.950, 0.353), '12-': (1.437, 0.884, 0.429), '1-3': (1.784, 1.130, 0.478),
        '-23': (1.964, 1.376, 0.580), '123': (2.292, 1.541, 0.752)}
# typical double-play result with a runner on first and no outs: who is left, runs scored
_DP_AFTER = {'1--': ('---', 0), '12-': ('--3', 0), '1-3': ('---', 1), '123': ('--3', 1)}


def runners_state(runners):
    """sheet Runners token ('0', '1', '1+2', '1+2+3' ...) -> RE24 key, or None."""
    if runners is None:
        return None
    t = set(str(runners).split('+')) - {'0', ''}
    return ('1' if '1' in t else '-') + ('2' if '2' in t else '-') + ('3' if '3' in t else '-')


def gdp_extra_cost(runners, outs):
    """Runs a GIDP costs the batting team beyond an ordinary out (runners hold) in this state;
    None when the state is not a double-play situation (no runner on first, or two outs)."""
    st = runners_state(runners)
    try:
        o = int(outs)
    except (TypeError, ValueError):
        return None
    if st is None or st[0] != '1' or o > 1:
        return None
    re_out = RE24[st][o + 1] if o + 1 <= 2 else 0.0
    if o == 0:
        after, runs = _DP_AFTER[st]; re_dp = RE24[after][2] + runs
    else:
        re_dp = 0.0
    return re_out - re_dp


# ── positional and replacement conventions (fWAR) ──
HWAR_POS_ADJ = {'C': 12.5, 'SS': 7.5, '2B': 2.5, '3B': 2.5, 'CF': 2.5, 'LF': -7.5, 'RF': -7.5, '1B': -12.5, 'DH': -17.5}
HWAR_POS_INNINGS = 1458.0     # a full season of innings at a position (162 x 9)
HWAR_REPL_SHARE = 0.57        # share of the 1000-WAR pool that is position players (fWAR: 570)


def apply_hitter_war(rows, fielding, innings, baserunning, lg_ra9, woba_scale, team_games, aaa_teams=('ROC', 'AAA')):
    """Writes hBsrRuns, hFldRuns, hPosRuns, hReplRuns, hWAR, hWAR_se on every row that has
    hBatRuns (None elsewhere). Returns the constants bundle, or None (announced)."""
    aaa = set(aaa_teams)
    live = [r for r in rows if r.get('hBatRuns') is not None]
    if not live or not lg_ra9 or not woba_scale or not team_games:
        print(f'  hwar WARNING: WAR assembly skipped (rows with batting runs {len(live)}, lgRA9 {lg_ra9}, '
              f'scale {woba_scale}, team games {bool(team_games)}); hWAR left None')
        for r in rows:
            for k in ('hBsrRuns', 'hFldRuns', 'hPosRuns', 'hReplRuns', 'hWAR', 'hWAR_se'):
                r[k] = None
        return None
    rpw = 4.0 * lg_ra9 / (2.0 * lg_ra9) ** WAR_PYTH_EXP
    club = [r for r in live if not is_combined_team(r.get('team'))]
    lg_games = sum(team_games.values()) / 2.0
    lg_pa = sum(r['pa'] for r in club)
    # double plays: league rate and mean cost
    opp = sum(r.get('gdpOpp') or 0 for r in club); gdp = sum(r.get('gdp') or 0 for r in club)
    cost = sum(r.get('gdpCost') or 0.0 for r in club)
    lg_gdp_rate = gdp / opp if opp else 0.0; lg_gdp_cost = cost / gdp if gdp else 0.0
    # season-level components split across a traded hitter's stints by PA share
    pa_by_id = {}
    for r in club:
        mid = r.get('mlbId')
        if mid:
            pa_by_id[mid] = pa_by_id.get(mid, 0) + r['pa']
    n_fld = n_bsr = n_pos = 0
    for r in rows:
        if r.get('hBatRuns') is None:
            for k in ('hBsrRuns', 'hFldRuns', 'hPosRuns', 'hReplRuns', 'hWAR', 'hWAR_se'):
                r[k] = None
            continue
        mid = str(r.get('mlbId') or '')
        share = 1.0 if is_combined_team(r.get('team')) else (r['pa'] / pa_by_id[r['mlbId']] if r.get('mlbId') and pa_by_id.get(r['mlbId']) else 1.0)
        fr = fielding.get(mid); fld = (fr.get('total') or 0.0) if fr else 0.0; n_fld += 1 if fr else 0
        br = baserunning.get(mid); bsr_sav = br['tot'] if br else 0.0; n_bsr += 1 if br else 0
        wgdp = lg_gdp_rate * (r.get('gdpOpp') or 0) * lg_gdp_cost - (r.get('gdpCost') or 0.0)
        inn = innings.get(mid) or {}
        pos = sum((inn.get(p, 0.0) if p != 'DH' else 9.0 * inn.get('DH_games', 0)) * adj / HWAR_POS_INNINGS
                  for p, adj in HWAR_POS_ADJ.items()); n_pos += 1 if inn else 0
        r['hBsrRuns'] = round(bsr_sav * share + wgdp, 2)
        r['hFldRuns'] = round(fld * share, 2)
        r['hPosRuns'] = round(pos * share, 2)
    # replacement runs per PA, pinned so the club-row total is exactly the share of the pool:
    # the fWAR positional values sum to about -17.5 runs per team-season (the DH), so a fixed
    # per-PA replacement would leave the league at 92% of its share (452 of 493 on 2026-09-05)
    target = HWAR_REPL_SHARE * 1000.0 * rpw * (lg_games / 2430.0)
    above = sum(r['hBatRuns'] + r['hBsrRuns'] + r['hFldRuns'] + r['hPosRuns'] for r in club)
    repl_per_pa = (target - above) / lg_pa
    for r in rows:
        if r.get('hBatRuns') is None:
            continue
        r['hReplRuns'] = round(repl_per_pa * r['pa'], 2)
        r['hWAR'] = round((r['hBatRuns'] + r['hBsrRuns'] + r['hFldRuns'] + r['hPosRuns'] + r['hReplRuns']) / rpw, 2)
        n = r.get('_xwOBAn') or 0
        r['hWAR_se'] = round((n / (n + HWAR_N0_BAT)) * WAR_XW_PA_SD * math.sqrt(r['pa']) / woba_scale / rpw, 2) if n > 0 else None
    tot = sum(r['hWAR'] for r in club)
    const = {'rpw': round(rpw, 4), 'replShare': HWAR_REPL_SHARE, 'replPerPa': round(repl_per_pa, 5),
             'lgGames': lg_games, 'lgPa': lg_pa, 'gdpRate': round(lg_gdp_rate, 4), 'gdpCost': round(lg_gdp_cost, 3),
             'posAdj': HWAR_POS_ADJ, 'posInnings': HWAR_POS_INNINGS, 'nFld': n_fld, 'nBsr': n_bsr, 'nPos': n_pos,
             'sumWar': round(tot, 1), 'sumFld': round(sum(r['hFldRuns'] for r in club), 1),
             'sumBsr': round(sum(r['hBsrRuns'] for r in club), 1), 'sumPos': round(sum(r['hPosRuns'] for r in club), 1)}
    print(f'  hWAR (hitters): {len(live)} rows, RPW {rpw:.2f}, replacement {repl_per_pa * 600:.1f} runs per 600 PA, '
          f'GIDP rate {lg_gdp_rate:.3f} at {lg_gdp_cost:.3f} runs each, fielding listed {n_fld}, baserunning listed {n_bsr}, '
          f'innings listed {n_pos}; club-row sums WAR {tot:.1f} fld {const["sumFld"]:+.1f} bsr {const["sumBsr"]:+.1f} pos {const["sumPos"]:+.1f}')
    return const
