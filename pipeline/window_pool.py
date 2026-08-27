#!/usr/bin/env python3
"""window_pool.py — score ONE hitter over ANY date range, ranked against the
SEASON pool.

THE RULE (Wally, 2026-08-18, after I got this wrong twice):

    The VALUES come from the date range. The PERCENTILES come from the
    season, for everybody. There is no sample-size gate.

So a three-week window that produces an elite rate reads as an elite
percentile, because that is the point: "for these three weeks he was the best
hitter in baseball." Comparing a pre-break window to a post-break window then
works, because both are ranked against the same fixed season ruler.

WHAT I BUILT FIRST AND THREW AWAY. A window-specific all-MLB pool: every
hitter recomputed over the same range, percentiles ranked inside it, cell
tables rebuilt over the window. That is a different and worse question. It
made every window its own ruler, so two windows were not comparable to each
other, and it needed a qualification gate that returned nothing on short
ranges. It also mirrored ~150 lines of process_data sequencing and grew four
bugs in an afternoon. None of that is needed.

WHAT THIS DOES INSTEAD:
  * window values: computed from the window's pitches by the pipeline's own
    functions (compute_hitter_stats, compute_expected_stats).
  * BB+ / SD+ / CT+ / Hitter+: the hitter's WINDOW pitches scored against the
    SEASON league anchors and cell tables, so the number means "at season
    league rates, this is what he did over the window".
  * percentiles: the window value's rank inside the SHIPPED season
    leaderboard's distribution for that stat, honouring HITTER_INVERT_PCTL.
    No minimum PA, no qualification, no pool rebuild.
"""

import math

from pipeline.compute import (
    compute_hitter_stats, compute_expected_stats,
    HITTER_STAT_KEYS, HITTER_INVERT_PCTL,
)
from pipeline.utils import (
    NON_PA_EVENTS, HIT_EVENTS, K_EVENTS, BB_EVENTS,
    HBP_EVENTS, SF_EVENTS, SH_EVENTS, CI_EVENTS,
)

EXPECTED_KEYS = ['wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon']
AAA_TEAMS = ('ROC', 'AAA')

HITTER_PLUS_W_BB = 0.52
HITTER_PLUS_W_SD = 0.17
HITTER_PLUS_W_CT = 0.31


def _is_combined(t):
    return isinstance(t, str) and t.endswith('TM') and t[:-2].isdigit()


def build_window_hitter_row(pitches, metadata, identity=None):
    """One hitter's raw window row: everything computable from the pitches
    alone. The + family and percentiles are layered on separately."""
    row = dict(identity or {})
    _d = [p.get('Game Date') for p in pitches if p.get('Game Date')]
    row['count'] = len(pitches)
    row['lastGameDate'] = max(_d) if _d else None
    row.update(compute_hitter_stats(pitches))
    # xwoba_key='xwOBA_hb' (2026-08-27 audit): the window xwOBA was the one
    # hitter surface still on the raw basis, so it ranked a raw value inside
    # the hb-basis season pool and skewed window xWRC+ with it.
    row.update(compute_expected_stats(
        pitches, woba_weights=metadata.get('wobaWeights'),
        xwoba_key='xwOBA_hb'))

    pa_p = [p for p in pitches
            if p.get('Event') and p['Event'] not in NON_PA_EVENTS]
    n_pa = len(pa_p)
    n_h = sum(1 for p in pa_p if p['Event'] in HIT_EVENTS)
    n_2b = sum(1 for p in pa_p if p['Event'] == 'Double')
    n_3b = sum(1 for p in pa_p if p['Event'] == 'Triple')
    n_hr = sum(1 for p in pa_p if p['Event'] == 'Home Run')
    n_k = sum(1 for p in pa_p if p['Event'] in K_EVENTS)
    n_bb = sum(1 for p in pa_p if p['Event'] in BB_EVENTS)
    n_hbp = sum(1 for p in pa_p if p['Event'] in HBP_EVENTS)
    n_sf = sum(1 for p in pa_p if p['Event'] in SF_EVENTS)
    n_sh = sum(1 for p in pa_p if p['Event'] in SH_EVENTS)
    n_ci = sum(1 for p in pa_p if p['Event'] in CI_EVENTS)
    n_ab = n_pa - n_bb - n_hbp - n_sf - n_sh - n_ci
    n_tb = n_h + n_2b + 2 * n_3b + 3 * n_hr
    obp_d = n_ab + n_bb + n_hbp + n_sf
    row['tb'] = n_tb
    row['avg'] = round(n_h / n_ab, 3) if n_ab > 0 else None
    row['obp'] = round((n_h + n_bb + n_hbp) / obp_d, 3) if obp_d > 0 else None
    row['slg'] = round(n_tb / n_ab, 3) if n_ab > 0 else None
    row['ops'] = (round(row['obp'] + row['slg'], 3)
                  if row['obp'] is not None and row['slg'] is not None else None)
    row['iso'] = (round(row['slg'] - row['avg'], 3)
                  if row['slg'] is not None and row['avg'] is not None else None)
    row['kPct'] = round(n_k / n_pa, 4) if n_pa > 0 else None
    row['bbPct'] = round(n_bb / n_pa, 4) if n_pa > 0 else None
    row['bbToK'] = (n_bb / n_k) if n_k > 0 else None
    return row


def _season_pool_rows(season_rows):
    """The shipped percentile pool: MLB rows only, with a traded player's
    per-team stint rows dropped in favour of his combined row. Mirrors
    compute_percentile_ranks_with_aaa (pipeline/compute.py), which works from
    the pipeline's _isROC / _isCombined flags; the leaderboard JSON has
    neither, so the same split is read off `team`."""
    from pipeline.utils import is_combined_team
    def _key(r):
        return r.get('mlbId') or r.get('hitter')
    combined = {_key(r) for r in season_rows if is_combined_team(r.get('team'))}
    out = []
    for r in season_rows:
        t = r.get('team')
        if t in AAA_TEAMS:
            continue
        if not is_combined_team(t) and _key(r) in combined:
            continue
        out.append(r)
    return out


def add_season_percentiles(row, season_rows, stats=None):
    """Rank each of `row`'s values inside the SEASON distribution.

    No qualification and no minimum sample: a window value is ranked for what
    it is. Inverted stats (K%, Chase%, Whiff%, GB%...) are flipped, matching
    process_data's separate inversion pass.
    """
    if stats is None:
        stats = HITTER_STAT_KEYS + EXPECTED_KEYS
    season_rows = _season_pool_rows(season_rows)
    for stat in stats:
        val = row.get(stat)
        pk = stat + '_pctl'
        if val is None:
            row[pk] = None
            continue
        pool = [r[stat] for r in season_rows if r.get(stat) is not None]
        if len(pool) < 10:
            row[pk] = None
            continue
        below = sum(1 for x in pool if x < val)
        equal = sum(1 for x in pool if x == val)
        p = (below + 0.5 * equal) / len(pool) * 100.0
        if stat in HITTER_INVERT_PCTL:
            p = 100.0 - p
        row[pk] = max(0, min(100, round(p)))
    return row


_FG_RANGE_CACHE = {}


def _fg_range_line(start_date, end_date):
    """The official hitting line for a date range, memoised per range.

    Degrades to {} rather than raising: FanGraphs Cloudflare-blocks some IPs,
    and a card should still render with our computed wRC+ if the fetch fails.
    The degrade announces itself.
    """
    key = (start_date, end_date)
    if key in _FG_RANGE_CACHE:
        return _FG_RANGE_CACHE[key]
    try:
        from pipeline.fg_overrides import fetch_mlb_hitters_range
        out = fetch_mlb_hitters_range(start_date, end_date)
    except Exception as e:
        print(f"    WARNING: FanGraphs range fetch failed "
              f"({type(e).__name__}: {e}) — using our computed wRC+")
        out = {}
    _FG_RANGE_CACHE[key] = out
    return out


def _season_mlb_id(season_rows, hitter_key):
    for r in season_rows:
        if (r.get('hitter'), r.get('team')) == hitter_key and r.get('mlbId'):
            return r['mlbId']
    for r in season_rows:            # any team row for the same name
        if r.get('hitter') == hitter_key[0] and r.get('mlbId'):
            return r['mlbId']
    return None


def score_window_against_season(hitter_key, window_pitches, all_pitches,
                                season_rows, metadata, verbose=True,
                                date_range=None, identity_mlb_id=None):
    """The whole job: one hitter, one date range, season-ranked.

    hitter_key is (hitter, team). Returns the row, ready for the card.
    """
    row = build_window_hitter_row(
        window_pitches, metadata,
        {'hitter': hitter_key[0], 'team': hitter_key[1],
         '_isROC': hitter_key[1] in AAA_TEAMS})

    G = metadata.get('gutsConstants') or {}

    # ── BB+ : the window's pitches through the FULL shipped chain, against
    # SEASON anchors (2026-08-21 — this block was two definitions stale:
    # pure-xwOBAcon with a dead metadata key falling to n0=60, feeding a
    # standardization computed from post-chain season values).
    #
    # Anchors come from metadata hitterLeagueAverages — the same values the
    # js/aggregator.js mirror reads. The removed _season_anchor helper inverted
    # the pre-2026-08-19 pure-ratio recipe and recovered a wrong league.
    # Chain, in server order (see the BB+ block in process_data and its JS
    # mirror): ingredients -> per-ingredient shrink toward 100 -> weight
    # blend -> bat-tracking prior (metadata bbPlusBtPrior; absent on a
    # pre-prior artifact -> stage skipped, which IS the pre-prior
    # definition) -> slope-match -> re-anchor -> wRC scale.
    #
    # DELIBERATE divergence from the site: no bbPlusMinBip display floor.
    # The window rule at the top of this file (no sample-size gate) wins —
    # a short window's BB+ is now mostly prior, which is the honest
    # estimate, not a blank. A full-season window can therefore show BB+
    # where the season row is None (sub-30-BIP hitters); that is this rule
    # working, not a bug.
    _lg = metadata.get('hitterLeagueAverages') or {}
    lg_con = _lg.get('xwOBAcon')
    lg_ev95 = _lg.get('ev95')
    _bw = metadata.get('bbPlusWeights') or {}
    _w_con = _bw.get('con', 0.60)
    _w_ev = _bw.get('ev', 0.40)
    _n0_con = metadata.get('bbPlusShrinkN0Con')
    _n0_con = 130 if _n0_con is None else _n0_con
    _n0_ev = metadata.get('bbPlusShrinkN0Ev') or 0
    _bb_beta = metadata.get('bbPlusBeta') or 4.205
    _bb_slope = metadata.get('bbPlusSlopeMatch') or 1.2352
    xc, nb = row.get('xwOBAcon'), row.get('nBip') or 0
    _ev95 = row.get('ev95')
    _ev_ok = (_w_ev == 0) or (_ev95 is not None and lg_ev95)
    if xc is not None and lg_con and _ev_ok and nb > 0:
        _con_plus = 100.0 * xc / lg_con
        _ev_c = (100.0 + (100.0 * _ev95 / lg_ev95 - 100.0) * _bb_beta
                 if _w_ev else 100.0)
        _con_adj = (nb * _con_plus + _n0_con * 100.0) / (nb + _n0_con)
        _ev_adj = ((nb * _ev_c + _n0_ev * 100.0) / (nb + _n0_ev)
                   if _n0_ev else _ev_c)
        _raw = _w_con * _con_adj + _w_ev * _ev_adj
        _btp = metadata.get('bbPlusBtPrior') or {}
        _anch = _btp.get('anchors') or {}
        _bs, _sq = row.get('batSpeed'), row.get('squaredUpPct')
        _nsw = row.get('nCompSwings') or 0
        if (_anch and _bs is not None and _sq is not None and _nsw > 0
                and (_anch.get('bsSd') or 0) > 0
                and (_anch.get('squpSd') or 0) > 0):
            _prior = (_anch['raw']
                      + _btp['betaBs'] * (_bs - _anch['bsMean'])
                      / _anch['bsSd']
                      + _btp['betaSqup'] * (_sq - _anch['squpMean'])
                      / _anch['squpSd'])
            _prior_eff = ((_nsw * _prior + _btp['s0'] * 100.0)
                          / (_nsw + _btp['s0']))
            _raw = (nb * _raw + _btp['k'] * _prior_eff) / (nb + _btp['k'])
        _v = 100.0 + (_raw - 100.0) * _bb_slope
        _v *= ((metadata.get('plusReanchor') or {}).get('bbPlus') or 1.0)
        _wrc = (metadata.get('plusWrcScale') or {}).get('bbPlus') or {}
        if _wrc.get('factor'):
            _v = (100.0 + (_v - 100.0) * _wrc['factor']
                  + (_wrc.get('shift') or 0.0))
        # 6 dp, the PLUS_STORE_DP convention — this value feeds the Hitter+
        # standardization below at computation precision; display rounds
        # at the card layer.
        row['bbPlus'] = round(_v, 6)
    else:
        row['bbPlus'] = None

    # ── SD+ / CT+ : the hitter's WINDOW swings scored against the SEASON cell
    # tables. all_pitches builds the league table, so the tables and the
    # regression anchor are season-scoped; only this hitter's pitches are the
    # window's. Everyone else is passed at full season so the league mean is
    # the season's, not a one-hitter artifact.
    from collections import defaultdict
    from pipeline.utils import ALL_TEAMS
    from pipeline.sdplus import compute_sd_plus
    from pipeline.contact import compute_ct_plus
    by = defaultdict(list)
    # Position-player pitches leave every skill-metric input (2026-08-27
    # audit), mirroring the season path: the league cell tables, every
    # hitter's decision/contact pitches, and the window hitter's own.
    _ep = {(p.get('Pitcher'), p.get('PTeam'))
           for p in all_pitches if p.get('Pitch Type') == 'EP'}
    def _no_ep(plist):
        return ([q for q in plist
                 if (q.get('Pitcher'), q.get('PTeam')) not in _ep]
                if _ep else plist)
    all_pitches = _no_ep(all_pitches)
    for p in all_pitches:
        if p.get('_roc_pitcher_pitch'):
            continue
        b, bt = p.get('Batter'), p.get('BTeam')
        if b and bt and bt in ALL_TEAMS:
            by[(b, bt)].append(p)
    by[hitter_key] = _no_ep(window_pitches)  # this hitter only, window-scoped
    if verbose:
        print(f"    scoring SD+/CT+ against season tables "
              f"({len(all_pitches)} league pitches)...")
    sd_res, _ = compute_sd_plus(all_pitches, dict(by),
                                lg_woba=G.get('lgWOBA'),
                                woba_scale=G.get('wOBAScale'))
    # CT+ bat prior for the window hitter: the WINDOW's bat speed and
    # tracked-swing count against the SERVER's season anchors (metadata
    # ctPlusBtPrior — same season-anchor convention as the BB+ chain
    # above). Absent metadata (pre-prior artifact) or no bat tracking on
    # the window = the prior-free definition, exactly.
    _ctp = metadata.get('ctPlusBtPrior') or {}
    _cta = _ctp.get('anchors') or {}
    _ct_bt_z = None
    _w_bs = row.get('batSpeed')
    _w_nsw = row.get('nCompSwings') or 0
    if (_ctp and _cta.get('bsSd') and _w_bs is not None and _w_nsw > 0):
        _ct_bt_z = {hitter_key: ((_w_bs - _cta['bsMean']) / _cta['bsSd'],
                                 _w_nsw)}
    ct_res, _ = compute_ct_plus(all_pitches, dict(by),
                                lg_woba=G.get('lgWOBA'),
                                woba_scale=G.get('wOBAScale'),
                                bt_z=_ct_bt_z,
                                bt_beta=_ctp.get('betaBs') or 0.0,
                                bt_k=(_ctp.get('k') or 0) if _ct_bt_z else 0,
                                bt_s0=_ctp.get('s0') or 0)
    s, c = sd_res.get(hitter_key), ct_res.get(hitter_key)
    row['sdPlus'] = s['sdPlus'] if s else None
    row['sdPlusRaw'] = round(s['raw_sd_adj'], 5) if s else None
    row['sdPlusN'] = s['n_decisions'] if s else 0
    row['ctPlus'] = c['ctPlus'] if c else None
    row['ctPlusRaw'] = round(c['raw_ct_adj'], 5) if c else None
    row['ctPlusN'] = c['n_swings'] if c else 0
    # Post-chain scaling (2026-08-27 audit): the season path multiplies by
    # plusReanchor and applies the plusWrcScale factor/shift AFTER the raw
    # computation; the window skipped both, so window SD+/CT+ read ~1.2
    # points high and fed pre-chain values into the post-chain Hitter+
    # anchors below. Same two stages the bbPlus block above applies.
    for _pk in ('sdPlus', 'ctPlus'):
        if row.get(_pk) is None:
            continue
        _v = row[_pk] * ((metadata.get('plusReanchor') or {}).get(_pk) or 1.0)
        _wrc = (metadata.get('plusWrcScale') or {}).get(_pk) or {}
        if _wrc.get('factor'):
            _v = (100.0 + (_v - 100.0) * _wrc['factor']
                  + (_wrc.get('shift') or 0.0))
        row[_pk] = round(_v, 6)

    # ── Hitter+ : composite on the SEASON standardization, so a window number
    # sits on the same ruler as the season card and as every other window.
    std = metadata.get('hitterPlusStandardization') or {}
    wsm = std.get('wrcScaleMatch') or {}
    ok = all(std.get(k, {}).get('sd') for k in ('bbPlus', 'sdPlus', 'ctPlus'))
    if ok and all(row.get(k) is not None for k in ('bbPlus', 'sdPlus', 'ctPlus')):
        z = (HITTER_PLUS_W_BB * (row['bbPlus'] - std['bbPlus']['mean']) / std['bbPlus']['sd']
             + HITTER_PLUS_W_SD * (row['sdPlus'] - std['sdPlus']['mean']) / std['sdPlus']['sd']
             + HITTER_PLUS_W_CT * (row['ctPlus'] - std['ctPlus']['mean']) / std['ctPlus']['sd'])
        v = 100.0 + (std.get('scale') or 40.0) * z
        shift = (metadata.get('plusReanchor') or {}).get('hitterPlusShift') or 0.0
        v += shift
        if wsm.get('factor'):
            v = 100.0 + (v - 100.0) * wsm['factor']
        row['hitterPlus'] = round(v, 1)
    else:
        row['hitterPlus'] = None

    # ── wRC+ : FanGraphs' own value FOR THIS DATE RANGE. FG serves custom
    # ranges (month=1000 + startdate/enddate), so there is no reason to
    # substitute our formula, which reads a couple of points different.
    # xWRC+ stays ours - FG does not publish it.
    pf = _load_park_factors().get(hitter_key[1], 1.0)
    lgw, sc, rpa = G.get('lgWOBA'), G.get('wOBAScale'), G.get('lgRPA')
    w = row.get('wOBA')
    if lgw and sc and rpa and rpa > 0 and w is not None:
        row['wRCplus'] = round(((w - lgw) / sc + rpa + (rpa - pf * rpa)) / rpa * 100)
        xw = row.get('xwOBA')
        row['xWRCplus'] = (round((((xw - lgw) / sc) + rpa) / rpa * 100)
                           if xw is not None else None)
        # Run-truth cap (2026-08-27 audit): the season path prints xWRC+ at
        # the published factor/shift (plusWrcScale.xWRCplus); the window
        # skipped it, printing ~27% too wide and then ranking that uncapped
        # value inside the capped season pool.
        _xf = (metadata.get('plusWrcScale') or {}).get('xWRCplus') or {}
        if row.get('xWRCplus') is not None and _xf.get('factor'):
            row['xWRCplus'] = round(100.0 + (row['xWRCplus'] - 100.0)
                                    * _xf['factor'] + (_xf.get('shift') or 0.0))
    else:
        row['wRCplus'] = row['xWRCplus'] = None

    # ── OFFICIAL LINE for the range. Exactly the role the boxscore merge
    # plays for a season row, and for the same reason: a no-pitch intentional
    # walk leaves no pitch, so a pitch-derived PA runs short and drags BB%,
    # K% and OBP with it. An IBB is a PA and, for a hitter, a walk - which is
    # what FanGraphs' PA/BB/BB% already carry.
    #
    # Only the official ledger is overridden. Everything Statcast-derived
    # (xwOBA, EV, barrels, swing rates, bat tracking) and the whole + family
    # stay pitch-derived, same split as the season card.
    mid = (identity_mlb_id if identity_mlb_id is not None
           else _season_mlb_id(season_rows, hitter_key))
    if mid is not None and date_range:
        hit = _fg_range_line(date_range[0], date_range[1]).get(str(int(mid)))
        if hit and hit.get('pa'):
            _b = (row.get('pa'), row.get('bbPct'), row.get('kPct'), row.get('obp'))
            for src, nd in (('pa', 0), ('ab', 0), ('hr', 0), ('doubles', 0),
                            ('triples', 0), ('avg', 3), ('obp', 3), ('slg', 3),
                            ('ops', 3), ('wOBA', 3), ('babip', 3),
                            ('bbPct', 4), ('kPct', 4)):
                v = hit.get(src)
                if v is not None:
                    row[src] = int(v) if nd == 0 else round(v, nd)
            if row.get('slg') is not None and row.get('avg') is not None:
                row['iso'] = round(row['slg'] - row['avg'], 3)
            if hit.get('bb') and hit.get('so'):
                row['bbToK'] = hit['bb'] / hit['so']
            if hit.get('wRCplus') is not None:
                row['wRCplus'] = hit['wRCplus']
            if verbose:
                print(f"    official line for the range: PA {_b[0]}→{row['pa']}"
                      f" (IBB {hit.get('ibb')}), BB% {_b[1]}→{row['bbPct']}, "
                      f"K% {_b[2]}→{row['kPct']}, OBP {_b[3]}→{row['obp']}, "
                      f"wRC+ {row.get('wRCplus')}")
        elif verbose:
            print(f"    no official row for this range — keeping the "
                  f"pitch-derived line, which runs short by any no-pitch IBBs")

    add_season_percentiles(row, season_rows)
    return row


def _load_park_factors():
    """Park factors for wRC+, from data/fg_manual.json (the same fallback the
    pipeline uses; the live FanGraphs fetch is Cloudflare-blocked locally)."""
    import json as _json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'fg_manual.json')
    try:
        with open(path) as f:
            return (_json.load(f) or {}).get('parkFactors') or {}
    except (OSError, ValueError):
        return {}
