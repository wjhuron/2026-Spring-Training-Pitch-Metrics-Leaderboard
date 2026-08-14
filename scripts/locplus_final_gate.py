"""locplus_final_gate.py — pre-ship gate: the EXACT candidate shipping
configs, composed, against shipped. Everything before this validated the
count surfaces at 4.5-inch kernels; the count grids inherit the group
kernels, so the composed package must be validated as one object.

Variants:
  shipped     current production
  count45     wh_count + xw_clevel at shipped 4.5 kernels
  count_full  count package + FF/CU x-flat (200), SI 9
  count_mod   count package + FF/SI/CU at 9

Objectives: the standard three, 2021-2025 replicates.

Usage: python3 scripts/locplus_final_gate.py
"""
import gc
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline_locplus as lp
import locplus_constants_multiseason as base
import locplus_countwhiff_multiseason as W
import locplus_countphys_extension as X
import locplus_countcompose_multiseason as C
from pipeline_sdplus import make_rv_xrv

BW = {
    'shipped': {},
    'count45': {},
    'count_full': {'FF': (200.0, 0.22), 'SI': (9.0, 0.22), 'CU': (200.0, 0.22)},
    'count_mod': {'FF': (9.0, 0.22), 'SI': (9.0, 0.22), 'CU': (9.0, 0.22)},
}
USE_COUNT = {'shipped': False, 'count45': True, 'count_full': True,
             'count_mod': True}
NAMES = ('shipped', 'count45', 'count_full', 'count_mod')


def eval_season(pitches, rv_fn):
    b = [p for p in pitches if lp.is_eligible_baseline(p)]
    dates = sorted({p['Game Date'] for p in b if p['Game Date']})
    q = len(dates) // 4
    cuts = [dates[q], dates[2 * q], dates[3 * q]]
    seg = defaultdict(list)
    for p in b:
        d = p['Game Date']
        seg['A1' if d < cuts[0] else 'A2' if d < cuts[1]
            else 'B1' if d < cuts[2] else 'B2'].append(p)
    par = {d: i % 2 for i, d in enumerate(dates)}
    g0 = [p for p in b if par.get(p['Game Date']) == 0]
    g1 = [p for p in b if par.get(p['Game Date']) == 1]

    prep = {}
    for half, f, s in (('A', 'A1', 'A2'), ('B', 'B1', 'B2')):
        actual = {}
        for k, ps in base.by_pitcher(seg[s]).items():
            v = [x for x in (rv_fn(p) for p in ps) if x is not None]
            if len(v) >= base.MIN_ACTUAL:
                actual[k] = sum(v) / len(v)
        velo = {}
        for k, ps in base.by_pitcher(seg[f]).items():
            v = [x for x in (lp.safe_float(p['Velocity']) for p in ps
                             if p['Pitch Type'] == 'FF') if x is not None]
            if len(v) >= base.MIN_VELO:
                velo[k] = sum(v) / len(v)
        prep[half] = {'first': seg[f], 'byp_f': base.by_pitcher(seg[f]),
                      'actual': actual, 'velo': velo}

    res = {}
    for name in NAMES:
        lp.PHYS_BW_PT = dict(BW[name])
        try:
            S = {h: lp.build_surfaces(prep[h]['first'], base.LG, base.SCALE)
                 for h in ('A', 'B')}
            Sr = {0: lp.build_surfaces(g0, base.LG, base.SCALE),
                  1: lp.build_surfaces(g1, base.LG, base.SCALE)}
            if USE_COUNT[name]:
                WH = {h: W.build_wh_count(prep[h]['first']) for h in ('A', 'B')}
                WHr = {0: W.build_wh_count(g0), 1: W.build_wh_count(g1)}
                XWc = {h: X.build_xw_clevel(prep[h]['first'], base.LG, base.SCALE)
                       for h in ('A', 'B')}
                XWr = {0: X.build_xw_clevel(g0, base.LG, base.SCALE),
                       1: X.build_xw_clevel(g1, base.LG, base.SCALE)}
            else:
                WH = WHr = XWc = XWr = None
            raws, parts, rlvs = [], [], []
            for half in ('A', 'B'):
                c = prep[half]
                loc = C.score_map_c(c['byp_f'], S[half], base.MIN_SCORE,
                                    WH[half] if WH else None,
                                    XWc[half] if XWc else None)
                kk = [k for k in loc if k in c['actual'] and k in c['velo']]
                if len(kk) < 30:
                    continue
                r_ly = base.pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
                r_ls = base.pearson([loc[k] for k in kk], [c['velo'][k] for k in kk])
                r_sy = base.pearson([c['velo'][k] for k in kk], [c['actual'][k] for k in kk])
                raws.append(r_ly)
                parts.append(base.partial(r_ly, r_ls, r_sy))
                rlvs.append(r_ls)
            a0 = C.score_map_c(base.by_pitcher(g0), Sr[0], base.MIN_REL,
                               WHr[0] if WHr else None, XWr[0] if XWr else None)
            a1 = C.score_map_c(base.by_pitcher(g1), Sr[1], base.MIN_REL,
                               WHr[1] if WHr else None, XWr[1] if XWr else None)
            ks = [k for k in a0 if k in a1]
            rel = base.pearson([a0[k] for k in ks], [a1[k] for k in ks])
            res[name] = {'partial': sum(parts) / len(parts),
                         'raw': sum(raws) / len(raws),
                         'rlv': sum(rlvs) / len(rlvs), 'rel': rel}
        finally:
            lp.PHYS_BW_PT = {}
    return res


def main():
    rv_fn = make_rv_xrv(base.LG, base.SCALE)
    seasons = [(2021, 'data/_statcast2021_cache.pkl'),
               (2022, 'data/_statcast2022_cache.pkl'),
               (2023, 'data/_statcast2023_cache.pkl'),
               (2024, 'data/_statcast2024_cache.pkl'),
               (2025, 'data/_statcast2025_full_cache.pkl')]
    table = {}
    for yr, path in seasons:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            continue
        print(f"adapting {yr}...", file=sys.stderr)
        pitches = base.adapt(p)
        table[yr] = eval_season(pitches, rv_fn)
        print(f"{yr}: " + '  '.join(
            f"{n}: p={table[yr][n]['partial']:.3f} rel={table[yr][n]['rel']:.3f}"
            for n in NAMES), flush=True)
        del pitches
        gc.collect()

    print("\nFINAL GATE (mean across seasons; delta vs shipped)")
    s = {m: sum(table[y]['shipped'][m] for y in table) / len(table)
         for m in ('partial', 'rel', 'rlv')}
    for n in NAMES:
        pm = sum(table[y][n]['partial'] for y in table) / len(table)
        rm = sum(table[y][n]['rel'] for y in table) / len(table)
        lm = sum(table[y][n]['rlv'] for y in table) / len(table)
        print(f"  {n:>10s}: partial {pm:.4f} ({pm-s['partial']:+.4f})  "
              f"rel {rm:.4f} ({rm-s['rel']:+.4f})  leak {lm:+.3f}")
    print("\nPer-season partial wins vs shipped:")
    for n in NAMES:
        if n == 'shipped':
            continue
        wins = sum(1 for y in table
                   if table[y][n]['partial'] > table[y]['shipped']['partial'])
        print(f"  {n}: {wins}/{len(table)}")


if __name__ == '__main__':
    main()
