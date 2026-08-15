"""era_parity_halves.py — alternating-game-date split-half components per
pitcher-season, 2021-2026. Input for the shrinkage measurement
(era_shrinkage_sweep.py): each half is an independent same-season sample
of the same pitcher, so predicting half B from shrunk half A measures the
shrinkage constant at the rendered unit.

Halves alternate over the LEAGUE's sorted game dates (the convention the
Command+/Loc+ batteries use), so both halves span the whole season and
schedule effects cancel.

Per (season, pitcher, half): xwOBA numerator/denominator (same PA
construction as the battery: IBB out, K in denominator, BIP at xwOBAcon
with actual-weight fallback), K, BB, PA, BIP, xwOBAcon sum/n, and for
2026 only the per-pitch Stuff+/Loc+/Pitching+ sums/counts (the sheet has
per-pitch values; past seasons' internal scores were not persisted
per-pitch, so their stabilization is measured on 2026 and flagged).

Output: data/_era_parity_halves.json
  {season: {pid: {'a'|'b': {xw_num, xw_den, k, bb, pa, bip,
                            xwc_sum, xwc_n, st_sum, st_n, lo_sum, lo_n,
                            pp_sum, pp_n}}}}
"""
import gc
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from pipeline_utils import (HIT_EVENTS, K_EVENTS, BB_EVENTS, HBP_EVENTS,
                            SF_EVENTS, SH_EVENTS, CI_EVENTS, NON_PA_EVENTS,
                            BUNT_BB_TYPES)
from era_battery_build import adapt_statcast, adapt_sheet, W_BB, W_HBP

OUT = os.path.join(ROOT, 'data', '_era_parity_halves.json')

SEASONS = [(2021, 'data/_statcast2021_cache.pkl'),
           (2022, 'data/_statcast2022_cache.pkl'),
           (2023, 'data/_statcast2023_cache.pkl'),
           (2024, 'data/_statcast2024_cache.pkl'),
           (2025, 'data/_statcast2025_full_cache.pkl')]


def tally_halves(pitches):
    dates = sorted({p['Game Date'] for p in pitches if p['Game Date']})
    par = {d: ('a' if i % 2 == 0 else 'b') for i, d in enumerate(dates)}
    from pipeline_utils import compute_in_zone, SWING_DESCRIPTIONS
    from pipeline_sdplus import make_rv_xrv
    rv_fn = make_rv_xrv(0.3169, 1.2393)   # frozen, matches pipeline_eraplus
    acc = defaultdict(lambda: defaultdict(float))
    for p in pitches:
        d = p['Game Date']
        if not d:
            continue
        half = par[d]
        c = acc[(p['Pitcher'], half)]
        # hpERA channel components (added 2026-08-15 for the sub-30-IP
        # shrinkage measurement): in-zone whiffs, GB, per-pitch xRV
        _desc = p.get('Description')
        if compute_in_zone(p) == 'Yes' and _desc in SWING_DESCRIPTIONS:
            c['izsw'] += 1
            if _desc == 'Swinging Strike':
                c['izwh'] += 1
        _bbt = p.get('BBType')
        if _bbt and _bbt not in BUNT_BB_TYPES:
            if _bbt == 'ground_ball':
                c['gb'] += 1
        _rv = rv_fn(p)
        if _rv is not None:
            c['xrv_sum'] += _rv
            c['xrv_n'] += 1
        for src, key in (('StuffPlus', 'st'), ('LocPlus', 'lo'),
                         ('PitchingPlus', 'pp')):
            v = p.get(src)
            if v is not None:
                c[key + '_sum'] += v
                c[key + '_n'] += 1
        bbt = p.get('BBType')
        if bbt and bbt not in BUNT_BB_TYPES:
            c['bip'] += 1
            xw = p.get('xwOBA')
            if xw is not None:
                c['xwc_sum'] += xw
                c['xwc_n'] += 1
        ev = p.get('Event')
        if not ev or ev in NON_PA_EVENTS:
            continue
        c['pa'] += 1
        if ev in K_EVENTS:
            c['k'] += 1
        elif ev in BB_EVENTS and ev != 'Intent Walk':
            c['bb'] += 1
        # PA-level xwOBA (mirrors the battery)
        if ev == 'Intent Walk' or ev in SH_EVENTS or ev in CI_EVENTS:
            continue
        if ev in BB_EVENTS:
            c['xw_num'] += W_BB
            c['xw_den'] += 1
        elif ev in HBP_EVENTS:
            c['xw_num'] += W_HBP
            c['xw_den'] += 1
        elif ev in K_EVENTS:
            c['xw_den'] += 1
        else:
            xw = p.get('xwOBA')
            c['xw_den'] += 1
            c['xw_num'] += xw if xw is not None else 0.0
    out = defaultdict(dict)
    for (pid, half), c in acc.items():
        out[str(pid)][half] = {k: round(v, 4) for k, v in c.items()}
    return dict(out)


def main():
    result = {}
    for season, path in SEASONS:
        print(f'{season}: {path}', flush=True)
        pitches = adapt_statcast(os.path.join(ROOT, path))
        result[str(season)] = tally_halves(pitches)
        print(f'  {len(result[str(season)])} pitchers', flush=True)
        del pitches
        gc.collect()
    print('2026: sheet pickle', flush=True)
    pitches = adapt_sheet(os.path.join(ROOT, 'data',
                                       'all_pitches_rs_cache.pkl'))
    result['2026'] = tally_halves(pitches)
    print(f'  {len(result["2026"])} pitchers', flush=True)
    del pitches
    gc.collect()
    with open(OUT, 'w') as f:
        json.dump(result, f)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
