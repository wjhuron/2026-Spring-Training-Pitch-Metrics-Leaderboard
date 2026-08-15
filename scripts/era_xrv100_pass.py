"""era_xrv100_pass.py — per-pitcher-season xRV/100 (luck-neutral per-pitch
run value), 2021-2026, full + h1 scopes. The Pitcher+ composite's heaviest
component (w = 0.23), needed to test the Pitcher+ component set against
future ERA in the phERA frame.

Uses pipeline_sdplus.make_rv_xrv with the same LG/SCALE constants the Loc+
multiseason harness uses, over the same adapters (statcast caches for
2021-2025, sheet pickle for 2026 — both produce pipeline-shaped dicts with
Description / Count / xwOBA, which is all make_rv_xrv needs).

Output: data/_era_xrv100.json {season: {pid: {full, h1, n_full, n_h1}}}
"""
import gc
import json
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from pipeline_sdplus import make_rv_xrv
import locplus_constants_multiseason as base
from era_battery_build import build_2026_name_map, _f

OUT = os.path.join(ROOT, 'data', '_era_xrv100.json')
TARGETS = json.load(open(os.path.join(ROOT, 'data', '_era_targets.json')))
SEASONS = [(2021, 'data/_statcast2021_cache.pkl'),
           (2022, 'data/_statcast2022_cache.pkl'),
           (2023, 'data/_statcast2023_cache.pkl'),
           (2024, 'data/_statcast2024_cache.pkl'),
           (2025, 'data/_statcast2025_full_cache.pkl')]
MIN_N = 50


def name_map(season):
    from pipeline_utils import _fullname_to_lastfirst
    m, ambig = {}, set()
    for pid, rec in TARGETS[str(season)]['pitchers'].items():
        full = (rec['name'] or '').strip()
        variants = {_fullname_to_lastfirst(full).lower()}
        parts = full.split()
        if len(parts) >= 3:
            variants.add((' '.join(parts[-2:]) + ', '
                          + ' '.join(parts[:-2])).lower())
        for lf in variants:
            if lf in m and m[lf] != int(pid):
                ambig.add(lf)
            m[lf] = int(pid)
    for lf in ambig:
        del m[lf]
    return m


def aggregate(pitches, rv_fn, asg, keyer):
    accf = defaultdict(lambda: [0.0, 0])
    acch = defaultdict(lambda: [0.0, 0])
    for p in pitches:
        v = rv_fn(p)
        if v is None:
            continue
        k = keyer(p)
        if k is None:
            continue
        accf[k][0] += v
        accf[k][1] += 1
        d = p.get('Game Date')
        if d and str(d)[:10] <= asg:
            acch[k][0] += v
            acch[k][1] += 1
    out = {}
    for k, (s, n) in accf.items():
        if n < MIN_N:
            continue
        rec = {'full': 100.0 * s / n, 'n_full': n}
        sh, nh = acch.get(k, (0.0, 0))
        if nh >= MIN_N:
            rec['h1'] = 100.0 * sh / nh
            rec['n_h1'] = nh
        out[k] = rec
    return out


def main():
    rv_fn = make_rv_xrv(base.LG, base.SCALE)
    result = {}
    for season, path in SEASONS:
        print(f'{season}: adapting', flush=True)
        pitches = base.adapt(os.path.join(ROOT, path))
        nm = name_map(season)
        agg = aggregate(pitches, rv_fn, TARGETS[str(season)]['asg'],
                        lambda p: nm.get((p.get('Pitcher') or '').lower()))
        result[str(season)] = {str(pid): rec for pid, rec in agg.items()}
        print(f'  {len(agg)} pitchers', flush=True)
        del pitches
        gc.collect()

    print('2026: sheet pickle', flush=True)
    raw = pickle.load(open(os.path.join(ROOT, 'data',
                                        'all_pitches_rs_cache.pkl'), 'rb'))
    nm26 = build_2026_name_map()
    mlb = []
    for p in raw:
        if p.get('_source') != 'MLB':
            continue
        pid = nm26.get(p.get('Pitcher'))
        if pid is None:
            continue
        mlb.append({'Pitcher': pid, 'Game Date': p.get('Game Date'),
                    'Description': p.get('Description'),
                    'Count': p.get('Count'), 'xwOBA': _f(p.get('xwOBA')),
                    'Pitch Type': p.get('Pitch Type'),
                    'Bats': p.get('Bats'), 'Throws': p.get('Throws')})
    del raw
    gc.collect()
    agg = aggregate(mlb, rv_fn, TARGETS['2026']['asg'],
                    lambda p: p['Pitcher'])
    result['2026'] = {str(pid): rec for pid, rec in agg.items()}
    print(f'  {len(agg)} pitchers', flush=True)
    with open(OUT, 'w') as f:
        json.dump(result, f)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
