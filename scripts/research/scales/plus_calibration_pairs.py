#!/usr/bin/env python3
"""plus_calibration_pairs.py — is a displayed point a true point? (2026-08-23)

Year-pair calibration for SD+, Loc+, Command+, per the tuning rule: decided
on independent samples (season pairs), not on within-season reliability,
which the failure log flags as gameable.

SD+ (the strict percent-contract test):
  displayed_Y  = the module's sdPlus, tables and league built self-contained
                 on season Y (pre-reanchor; the process_data reanchor is a
                 small multiplicative recenter and cannot change a slope).
  realized_Y1  = 100 * raw_sd(Y+1, mix-neutral, UNSHRUNK) / lg_raw(Y+1),
                 the channel outcome in the same percent units the contract
                 claims.
  contract     = OLS slope of realized on displayed = 1. Slope < 1 means the
                 displayed spread overstates true skill by that factor.

Loc+ and Command+ (forward-reading factors, NOT contract tests — Loc+ sits
on the borrowed SD ruler, Command+ is a self-ratio on miss distance):
  slope of the SAME construction next season on this season's displayed
  value. "A +10 today reads as +10*slope next season."

Seasons 2021-2025 from the public Statcast caches (adapted to pipeline
shape), 2026 from the sheet cache; the 2025->2026 hitter join goes through
the leaderboard mlbId map.

Usage: python3 scripts/research/scales/plus_calibration_pairs.py
"""
import gc
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'locplus'))
import pipeline.sdplus as sd                             # noqa: E402
import pipeline.locplus as lp                            # noqa: E402
import pipeline.commandplus as cp                        # noqa: E402
from pipeline.utils import compute_in_zone               # noqa: E402
from locplus_constants_multiseason import DMAP, _f, _s   # noqa: E402

DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(DATA, '_plus_calibration_pairs.json')
CACHE = {2021: '_statcast2021_cache.pkl', 2022: '_statcast2022_cache.pkl',
         2023: '_statcast2023_cache.pkl', 2024: '_statcast2024_cache.pkl',
         2025: '_statcast2025_full_cache.pkl'}
GUTS = {2021: (0.314, 1.209), 2022: (0.310, 1.259), 2023: (0.318, 1.204),
        2024: (0.310, 1.242), 2025: (0.3131, 1.2317)}
SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
PAIRS = [(y, y + 1) for y in (2021, 2022, 2023, 2024, 2025)]
MIN_PITCH = 300           # pitcher floors per season (Loc+/Command+)
EVMAP = {'intent_walk': 'Intent Walk'}


def adapt2(path):
    """Statcast cache -> pipeline dicts with Batter id, Event, InZone."""
    df = pickle.load(open(os.path.join(DATA, path), 'rb'))
    cols = ['pitch_type', 'plate_x', 'plate_z', 'sz_top', 'sz_bot', 'balls',
            'strikes', 'description', 'delta_run_exp',
            'estimated_woba_using_speedangle', 'stand', 'p_throws',
            'player_name', 'game_date', 'batter', 'events']
    out = []
    for r in df[cols].itertuples(index=False):
        d = DMAP.get(r.description)
        if d is None or r.pitch_type is None:
            continue
        try:
            b, s = int(r.balls), int(r.strikes)
        except (TypeError, ValueError):
            continue
        if not (0 <= b <= 3 and 0 <= s <= 2):
            continue
        try:
            re = float(r.delta_run_exp)
            re = None if re != re else re
        except (TypeError, ValueError):
            re = None
        p = {'Pitch Type': _s(r.pitch_type), 'Bats': _s(r.stand),
             'Throws': _s(r.p_throws),
             'PlateX': _f(r.plate_x), 'PlateZ': _f(r.plate_z),
             'SzTop': _f(r.sz_top), 'SzBot': _f(r.sz_bot),
             'Count': f'{b}-{s}', 'Description': d,
             'RunExp': (None if re is None else -re),
             'xwOBA': _f(r.estimated_woba_using_speedangle),
             'Pitcher': _s(r.player_name), 'Game Date': str(r.game_date)[:10],
             'Batter': int(r.batter) if r.batter == r.batter else None,
             'Event': EVMAP.get(_s(r.events)), 'BBType': None,
             '_source': 'MLB'}
        p['InZone'] = compute_in_zone(p)
        out.append(p)
    del df
    gc.collect()
    return out


def load_2026():
    allp = pickle.load(open(os.path.join(DATA, 'all_pitches_rs_cache.pkl'), 'rb'))
    mlb = [p for p in allp if p.get('_source') == 'MLB']
    del allp
    # hitter name -> mlbId so 2025 (id-keyed) joins 2026
    rows = json.load(open(os.path.join(DATA, 'hitter_leaderboard_rs.json')))
    id_of = {r['hitter']: int(r['mlbId']) for r in rows
             if r.get('mlbId') and r.get('team') not in ('ROC', 'AAA')}
    for p in mlb:
        p['Batter'] = id_of.get(p.get('Batter'))
    gc.collect()
    return mlb


def ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        return float('nan'), float('nan'), 0
    sl, _ = np.polyfit(x[m], y[m], 1)
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    return float(sl), r, int(m.sum())


def sd_season(pitches, guts):
    """(displayed sdPlus by batter id, realized-percent by batter id)."""
    by_h = defaultdict(list)
    for p in pitches:
        if p.get('Batter') is not None:
            by_h[(p['Batter'], '')].append(p)
    norm, _tbl = sd.compute_sd_plus(pitches, by_h, *guts)
    disp = {k[0]: v['sdPlus'] for k, v in norm.items()}
    raws = {k[0]: (v['raw_sd'], v['n_decisions']) for k, v in norm.items()}
    lg = np.mean([r for r, n in raws.values()])
    real = {b: 100.0 * r / lg for b, (r, n) in raws.items()} if abs(lg) > 1e-9 else {}
    return disp, real


def loc_season(pitches, guts):
    base = [p for p in pitches if lp.is_eligible_baseline(p)]
    S = lp.build_surfaces(base, *guts)
    acc = defaultdict(lambda: [0.0, 0])
    for p in base:
        v = lp.score_pitch(p, S)
        if v is not None:
            a = acc[p.get('Pitcher')]
            a[0] += v
            a[1] += 1
    raw = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_PITCH}
    mu = float(np.mean(list(raw.values())))
    sg = float(np.std(list(raw.values())))
    return {k: 100.0 - 10.0 * (v - mu) / sg for k, v in raw.items()}


def cmd_season(pitches):
    by_p = defaultdict(list)
    for p in pitches:
        if cp.is_eligible(p):
            by_p[p.get('Pitcher')].append(p)
    res = cp.score_misses(by_p)
    pool = [v['raw_miss'] for v in res.values() if v['n_pitches'] >= MIN_PITCH]
    lg = float(np.mean(pool))
    return {k: 200.0 - 100.0 * v['raw_miss'] / lg for k, v in res.items()
            if v['n_pitches'] >= MIN_PITCH}


def main():
    md = json.load(open(os.path.join(DATA, 'metadata_rs.json')))
    G26 = md.get('gutsConstants') or {}
    guts26 = (G26.get('lgWOBA', 0.3169), G26.get('wOBAScale', 1.2393))
    sd_disp, sd_real, locv, cmdv = {}, {}, {}, {}
    for y in SEASONS:
        print(f'season {y} ...', flush=True)
        pitches = load_2026() if y == 2026 else adapt2(CACHE[y])
        guts = guts26 if y == 2026 else GUTS[y]
        sd_disp[y], sd_real[y] = sd_season(pitches, guts)
        locv[y] = loc_season(pitches, guts)
        cmdv[y] = cmd_season(pitches)
        print(f'  sd+ {len(sd_disp[y])}  loc+ {len(locv[y])}  cmd+ {len(cmdv[y])}',
              flush=True)
        del pitches
        gc.collect()

    out = {}
    print('\n=== SD+ percent-contract: slope of realized(Y+1) on displayed(Y) ===')
    for Y, Y1 in PAIRS:
        ks = [k for k in sd_disp[Y] if k in sd_real[Y1]]
        sl, r, n = ols([sd_disp[Y][k] for k in ks], [sd_real[Y1][k] for k in ks])
        out.setdefault('sdPlus', {})[f'{Y}-{Y1}'] = dict(slope=sl, r=r, n=n)
        print(f'  {Y}->{Y1}: slope {sl:.3f}  r {r:.3f}  n {n}')
    for name, series in (('locPlus', locv), ('commandPlus', cmdv)):
        print(f'\n=== {name}: slope of next-season value on this season ===')
        for Y, Y1 in PAIRS:
            ks = [k for k in series[Y] if k in series[Y1]]
            sl, r, n = ols([series[Y][k] for k in ks], [series[Y1][k] for k in ks])
            out.setdefault(name, {})[f'{Y}-{Y1}'] = dict(slope=sl, r=r, n=n)
            print(f'  {Y}->{Y1}: slope {sl:.3f}  r {r:.3f}  n {n}')
    for name in out:
        sls = [v['slope'] for v in out[name].values() if np.isfinite(v['slope'])]
        print(f'{name}: mean slope {np.mean(sls):.3f}  range '
              f'{min(sls):.3f}-{max(sls):.3f}')
    json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
