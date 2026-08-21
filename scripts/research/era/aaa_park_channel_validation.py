"""aaa_park_channel_validation.py — should hpERA's park channel activate
for Triple-A rows, and in which DIRECTION?

THE DESIGN ISSUE THIS TESTS. For an MLB arm the park channel is forward:
a hitter park raises his projected ERA because he keeps the park (fitted
weight W_PH['park']=0.168). A ROC arm does not keep Rochester — his hpERA
is a translation to MLB — so for him the park's correct role is
RETRODICTIVE: stats earned in a hitter park overstate his run allowance,
and the correction runs the OPPOSITE direction from the fitted channel.
Naively activating the channel could apply the wrong sign.

PRE-REGISTERED PROTOCOL (2026-08-21, before any result was seen):
  Universe   AAA pitcher-seasons 2023-2026 from data/_aaa_battery.json,
             >= FLOOR outs proxy (pa >= 150). Club = modal home club over
             the pitcher's games (data/_aaa_gamepk_home.json); a pitcher is
             at his own park in ~half his games, every other park in ~4%,
             so the mode is his club.
  Park       BA 2025 factors (data/_milb_park_factors_2025.json). Runs PF
             -> xw deflator via the runs~wOBA^2 rule at half home share:
             woba_mult = 1 + (runs_pf - 100)/400. The formula is checked
             against the 9 clubs whose actual BA all-wOBA PF was
             transcribed, and the check prints with the results.
  Baseline   six-channel AAA hpERA (xw, k, izwh, gb, loc, xrv — the same
             construction as aaa_level_correction.py: eraplus shrinkage
             constants, in-season z pools, W_PH weights, park and gs at
             zero).
  Target     NEXT-season ROAD xwOBA-against (>= 200 road PA-ends), park-
             free by construction. Road = game whose home club is not the
             pitcher's modal club. Year-pairs 23->24, 24->25, 25->26 (the
             last is the only pair with in-period factors; staleness of
             the others is reported, not hidden).
  Tests      T1  partial slope of park (runs mult) on the target given
             the baseline: ~0 = noise, negative = retrodictive direction.
         T2  baseline + NAIVE channel (+W_PH['park'] * z(mult)): expected
             to hurt or tie.
         T3  baseline with the xrv channel park-CORRECTED before shrinkage
             (xrv100 - (runs_mult - 1) * league runs per 100 pitches, the
             run-currency channel): ships only if it beats the baseline r
             in a majority of year-pairs.
  Decision   T3 wins majority -> activate as input correction. Otherwise
             the ROC park stays neutral and the result is recorded.

  AMENDMENT (result-blind, before any correlation printed): the original
  T3 targeted an xw channel that hpERA does not have (W_PH carries no
  'xw' — that channel is hdERA's). The retrodictive correction moved to
  xrv, the only run-currency channel. Baseline is the six-channel set
  stuff/loc/k/izwh/gb/xrv with production signs (all channels negated
  into ERA direction where higher skill = fewer runs); AAA stuff is the
  instrument-corrected _aaa_stuff_geomfix value (2023-2025, which covers
  every channel year needed). Road-PA floor 200 -> 120, also result-
  blind, for pool size.

Usage: python3 scripts/research/era/aaa_park_channel_validation.py
"""
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from pipeline.eraplus import (N0_XW, N0_K, N0_IZWH, N0_GB, N0_XRV,  # noqa: E402
                              IZSW_PER_PITCH, W_PH)

D = lambda n: json.load(open(os.path.join(ROOT, 'data', n)))
YEARS = (2023, 2024, 2025, 2026)
PA_FLOOR = 150
ROAD_PA_FLOOR = 120   # lowered from 200 result-blind (pool size only)

# Schedule-API club name -> BA table name (AAA only).
BA_NAME = {
    'Buffalo Bisons': ('international_league', 'Buffalo'),
    'Charlotte Knights': ('international_league', 'Charlotte'),
    'Columbus Clippers': ('international_league', 'Columbus'),
    'Durham Bulls': ('international_league', 'Durham'),
    'Gwinnett Stripers': ('international_league', 'Gwinnett'),
    'Indianapolis Indians': ('international_league', 'Indianapolis'),
    'Iowa Cubs': ('international_league', 'Iowa'),
    'Jacksonville Jumbo Shrimp': ('international_league', 'Jacksonville'),
    'Lehigh Valley IronPigs': ('international_league', 'Lehigh Valley'),
    'Louisville Bats': ('international_league', 'Louisville'),
    'Memphis Redbirds': ('international_league', 'Memphis'),
    'Nashville Sounds': ('international_league', 'Nashville'),
    'Norfolk Tides': ('international_league', 'Norfolk'),
    'Omaha Storm Chasers': ('international_league', 'Omaha'),
    'Rochester Red Wings': ('international_league', 'Rochester'),
    'Scranton/Wilkes-Barre RailRiders': ('international_league', 'Scranton/WB'),
    'St. Paul Saints': ('international_league', 'St. Paul'),
    'Syracuse Mets': ('international_league', 'Syracuse'),
    'Toledo Mud Hens': ('international_league', 'Toledo'),
    'Worcester Red Sox': ('international_league', 'Worcester'),
    'Albuquerque Isotopes': ('pacific_coast_league', 'Albuquerque'),
    'El Paso Chihuahuas': ('pacific_coast_league', 'El Paso'),
    'Las Vegas Aviators': ('pacific_coast_league', 'Las Vegas'),
    'Oklahoma City Comets': ('pacific_coast_league', 'Okla. City'),
    'Oklahoma City Baseball Club': ('pacific_coast_league', 'Okla. City'),
    'Oklahoma City Dodgers': ('pacific_coast_league', 'Okla. City'),
    'Reno Aces': ('pacific_coast_league', 'Reno'),
    'Round Rock Express': ('pacific_coast_league', 'Round Rock'),
    'Sacramento River Cats': ('pacific_coast_league', 'Sacramento'),
    'Salt Lake Bees': ('pacific_coast_league', 'Salt Lake'),
    'Sugar Land Space Cowboys': ('pacific_coast_league', 'Sugar Land'),
    'Tacoma Rainiers': ('pacific_coast_league', 'Tacoma'),
}

K_EV = {'strikeout', 'strikeout_double_play'}
BB_EV = {'walk', 'intent_walk'}
NON_PA_EV = {
    'caught_stealing_2b', 'caught_stealing_3b', 'caught_stealing_home',
    'pickoff_1b', 'pickoff_2b', 'pickoff_3b', 'pickoff_caught_stealing_2b',
    'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home',
    'stolen_base_2b', 'stolen_base_3b', 'stolen_base_home',
    'wild_pitch', 'passed_ball', 'game_advisory', 'ejection', 'other_advance'}
# per-season wOBA weights for the BB/HBP legs of the PA-level target
W_BB, W_HBP = 0.693, 0.723


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return None, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def zs(v):
    v = np.asarray(v, float)
    mu, sd = np.nanmean(v), np.nanstd(v)
    return (v - mu) / sd if sd > 0 else v * 0.0


def main():
    homes = D('_aaa_gamepk_home.json')
    parks = D('_milb_park_factors_2025.json')
    bat = D('_aaa_battery.json')

    # formula check: predicted woba PF vs the 9 transcribed actuals
    print('── woba_mult formula check (runs~wOBA^2, half home share) ──')
    errs = []
    for lg in ('international_league', 'pacific_coast_league'):
        for t, row in parks[lg].items():
            actual = (parks['lr_woba_splits_aaa']['rows']
                      .get(t, {}).get('all_pf'))
            if actual is None:
                continue
            pred = 100 + (row['r_g']['pf'] - 100) / 2.0
            errs.append(abs(pred - actual))
            print(f'  {t:<12} runs {row["r_g"]["pf"]:>3} -> pred wOBA PF '
                  f'{pred:5.1f} vs actual {actual}')
    print(f'  mean abs err {np.mean(errs):.1f} PF points on {len(errs)} clubs')

    def park_mult(club_key):
        lg, name = club_key
        pf = parks[lg][name]['r_g']['pf']
        return 1.0 + (pf - 100) / 400.0     # xw deflator (half share, wOBA)

    def park_runs_mult(club_key):
        lg, name = club_key
        return parks[lg][name]['r_g']['mult']

    # per pitcher-season: modal club + road target from the caches
    club_of = {}          # (year, pid) -> club key
    road_xw = {}          # (year, pid) -> (road paxw, n road PA)
    for y in YEARS:
        p = os.path.join(ROOT, 'data', f'_aaa_statcast{y}_cache.pkl')
        if not os.path.exists(p):
            continue
        df = pickle.load(open(p, 'rb'))
        df = df.reset_index(drop=True)
        gk = df['game_pk'].astype('Int64').astype(str)
        df['_home'] = gk.map(lambda k: (homes.get(k) or {}).get('home'))
        ev = df['events'].where(df['events'].astype(str).str.len() > 0)
        xw = pd.to_numeric(df['estimated_woba_using_speedangle'],
                           errors='coerce')
        is_k = ev.isin(K_EV).fillna(False)
        is_bb = ev.isin(BB_EV).fillna(False)
        is_hbp = (ev == 'hit_by_pitch').fillna(False)
        is_bip = (df['description'] == 'hit_into_play') & xw.notna()
        pa_val = np.where(is_bip, xw,
                          np.where(is_k, 0.0,
                                   np.where(is_bb, W_BB,
                                            np.where(is_hbp, W_HBP, np.nan))))
        df['_paval'] = pa_val
        for pid, g in df.groupby('pitcher'):
            hc = Counter(g['_home'].dropna())
            if not hc:
                continue
            club_name = hc.most_common(1)[0][0]
            key = BA_NAME.get(club_name)
            if key is None:
                continue      # non-AAA venue mode (rehab road games etc.)
            club_of[(y, int(pid))] = key
            road = g[g['_home'] != club_name]
            vals = road['_paval'].dropna()
            if len(vals) >= ROAD_PA_FLOOR:
                road_xw[(y, int(pid))] = (float(vals.mean()), len(vals))
        print(f'  {y}: clubs mapped {sum(1 for k in club_of if k[0]==y)}, '
              f'road targets {sum(1 for k in road_xw if k[0]==y)}')

    # six-channel baseline per season (production signs, in-season z pools)
    stuff_aaa = D('_aaa_stuff_geomfix.json')['AAA']
    CH = {}
    for y in YEARS:
        rows = bat.get(str(y), {})
        st_y = stuff_aaa.get(str(y), {})
        recs = []
        for pid, r in rows.items():
            b = r.get('battery') or {}
            if (b.get('pa') or 0) < PA_FLOOR:
                continue
            key = (y, int(pid))
            club = club_of.get(key)
            if club is None:
                continue
            loc = (r.get('loc') or {}).get('v')
            xrv = (r.get('xrv') or {}).get('v')
            stf = (st_y.get(pid) or {}).get('v')
            izwh = (1 - b['zcon_pct']) if b.get('zcon_pct') is not None else None
            if None in (stf, b.get('k_pct'), izwh, b.get('gb_pct'), loc, xrv):
                continue
            recs.append({
                'pid': int(pid), 'club': club,
                'mult': park_runs_mult(club), 'league': club[0],
                'n_pa': b['pa'], 'n_p': b['pitches'], 'n_bip': b['bip'],
                'k': b['k_pct'], 'izwh': izwh, 'gb': b['gb_pct'],
                'loc': loc, 'xrv100': xrv * 100.0, 'stuff': stf,
            })
        if len(recs) < 50:
            continue
        t = pd.DataFrame(recs)
        lg_k = float((t['k'] * t['n_pa']).sum() / t['n_pa'].sum())
        lg_gb = float((t['gb'] * t['n_bip']).sum() / t['n_bip'].sum())
        lg_izwh = float(t['izwh'].mean())
        # league absolute run environment per 100 pitches, for the xrv
        # park correction: BA home R/G by league over pitches per team-game
        r100 = {}
        for lg_key in ('international_league', 'pacific_coast_league'):
            rg = np.mean([v['r_g']['home'] for v in parks[lg_key].values()])
            in_lg = t[t['league'] == lg_key]
            ppg = (in_lg['n_p'].sum() / max(1, in_lg['n_pa'].sum())) * 38.6
            # pitches per team-game ~= pitches-per-PA * ~38.6 PA/team-game
            r100[lg_key] = rg / (2 * ppg) * 100 if ppg else 6.5
        t['k_s'] = (t['k'] * t['n_pa'] + N0_K * lg_k) / (t['n_pa'] + N0_K)
        t['_izsw'] = t['n_p'] * IZSW_PER_PITCH
        t['izwh_s'] = (t['izwh'] * t['_izsw'] + N0_IZWH * lg_izwh) \
            / (t['_izsw'] + N0_IZWH)
        t['gb_s'] = (t['gb'] * t['n_bip'] + N0_GB * lg_gb) \
            / (t['n_bip'] + N0_GB)
        t['xrv_s'] = (t['xrv100'] * t['n_p']) / (t['n_p'] + N0_XRV)
        _base = t['league'].map(r100)
        xrv_corr = t['xrv100'] - (t['mult'] - 1.0) * _base * 2.0 * 0.5
        # (mult is already the HALF-share multiplier; the park's full home
        #  effect on his season line is (mult-1) of the absolute rate)
        t['xrv_cs'] = (xrv_corr * t['n_p']) / (t['n_p'] + N0_XRV)

        def hp(xrv_col):
            return (W_PH['stuff'] * zs(-t['stuff'])
                    + W_PH['loc'] * zs(-t['loc'])
                    + W_PH['k'] * zs(-t['k_s'])
                    + W_PH['izwh'] * zs(-t['izwh_s'])
                    + W_PH['gb'] * zs(-t['gb_s'])
                    + W_PH['xrv'] * zs(t[xrv_col]))

        t['hp6'] = hp('xrv_s')
        t['hp6_naive'] = t['hp6'] + W_PH['park'] * zs(t['mult'])
        t['hp6_corr'] = hp('xrv_cs')
        CH[y] = t.set_index('pid')

    # ── tests over year-pairs ──
    print('\n── year-pair prediction: next-season ROAD xwOBA-against ──')
    wins = {'naive': 0, 'corr': 0}
    n_pairs = 0
    for y0, y1 in ((2023, 2024), (2024, 2025), (2025, 2026)):
        if y0 not in CH:
            continue
        t = CH[y0]
        tgt = {pid: v[0] for (yy, pid), v in road_xw.items() if yy == y1}
        m = t.index.isin(tgt)
        if m.sum() < 30:
            print(f'  {y0}->{y1}: pool {int(m.sum())} too thin — skipped')
            continue
        sub = t[m]
        yv = np.array([tgt[p] for p in sub.index])
        r0, n = pearson(sub['hp6'], yv)
        r1, _ = pearson(sub['hp6_naive'], yv)
        r2, _ = pearson(sub['hp6_corr'], yv)
        # T1: park partial given baseline
        res_y = yv - np.poly1d(np.polyfit(sub['hp6'], yv, 1))(sub['hp6'])
        res_p = sub['mult'] - np.poly1d(
            np.polyfit(sub['hp6'], sub['mult'], 1))(sub['hp6'])
        rp, _ = pearson(res_p, res_y)
        stale = '' if y0 == 2025 else '  [2025 factors: stale for this pair]'
        print(f'  {y0}->{y1} n={n}: baseline r={r0:+.3f}  naive r={r1:+.3f}  '
              f'xw-corrected r={r2:+.3f}  park-partial r={rp:+.3f}{stale}')
        n_pairs += 1
        wins['naive'] += (r1 > r0)
        wins['corr'] += (r2 > r0)

    # ── IL-only pass (result-AWARE amendment, so labeled): the pooled
    # park-partial came back POSITIVE, but park mult is confounded with
    # league (PCL parks run hot, and a PCL arm's "park-free" road target is
    # still PCL-denominated: altitude + league context). The production
    # consumer is Rochester, an IL club, so the decision test is IL-only
    # with z-pools rebuilt within league. ──
    print('\n── IL-only, in-league z (decision test) ──')
    il_wins = {'naive': 0, 'corr': 0}
    il_pairs = 0
    for y0, y1 in ((2023, 2024), (2024, 2025), (2025, 2026)):
        if y0 not in CH:
            continue
        t = CH[y0]
        sub = t[t['league'] == 'international_league'].copy()
        def hp_il(xrv_col):
            return (W_PH['stuff'] * zs(-sub['stuff'])
                    + W_PH['loc'] * zs(-sub['loc'])
                    + W_PH['k'] * zs(-sub['k_s'])
                    + W_PH['izwh'] * zs(-sub['izwh_s'])
                    + W_PH['gb'] * zs(-sub['gb_s'])
                    + W_PH['xrv'] * zs(sub[xrv_col]))
        sub['hp6'] = hp_il('xrv_s')
        sub['hp6_naive'] = sub['hp6'] + W_PH['park'] * zs(sub['mult'])
        sub['hp6_corr'] = hp_il('xrv_cs')
        tgt = {pid: v[0] for (yy, pid), v in road_xw.items() if yy == y1}
        m = sub.index.isin(tgt)
        if m.sum() < 30:
            print(f'  {y0}->{y1}: IL pool {int(m.sum())} too thin — skipped')
            continue
        ss = sub[m]
        yv = np.array([tgt[p] for p in ss.index])
        r0, n = pearson(ss['hp6'], yv)
        r1, _ = pearson(ss['hp6_naive'], yv)
        r2, _ = pearson(ss['hp6_corr'], yv)
        res_y = yv - np.poly1d(np.polyfit(ss['hp6'], yv, 1))(ss['hp6'])
        res_p = ss['mult'] - np.poly1d(
            np.polyfit(ss['hp6'], ss['mult'], 1))(ss['hp6'])
        rp, _ = pearson(res_p, res_y)
        stale = '' if y0 == 2025 else '  [stale factors]'
        print(f'  {y0}->{y1} n={n}: baseline r={r0:+.3f}  naive r={r1:+.3f}  '
              f'corr r={r2:+.3f}  park-partial r={rp:+.3f}{stale}')
        il_pairs += 1
        il_wins['naive'] += (r1 > r0)
        il_wins['corr'] += (r2 > r0)
    print(f'  IL-only: naive beats baseline {il_wins["naive"]}/{il_pairs}, '
          f'corr beats baseline {il_wins["corr"]}/{il_pairs}')

    print(f'\n  naive channel beats baseline in {wins["naive"]}/{n_pairs} '
          f'pairs; xw-correction beats baseline in {wins["corr"]}/{n_pairs}')
    out = {'wins': wins, 'n_pairs': n_pairs,
           'protocol': 'see module docstring'}
    with open(os.path.join(ROOT, 'data',
                           '_aaa_park_validation.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('  saved data/_aaa_park_validation.json')


if __name__ == '__main__':
    main()
