"""locplus_countwhiff_ksweep.py — sweep K_WH_COUNT for the count-specific
whiff surface (won 5/5 on partial at the borrowed K=20; a constant is not
settled until the sweep brackets an interior optimum). K in {5, 10, 20,
40, 80}, same harness and objectives as locplus_countwhiff_multiseason.py.

Usage: python3 scripts/locplus_countwhiff_ksweep.py
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
from pipeline_sdplus import make_rv_xrv

KS = (5, 10, 20, 40, 80)


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

    S_first = {h: lp.build_surfaces(prep[h]['first'], base.LG, base.SCALE)
               for h in ('A', 'B')}
    S_rel = {0: lp.build_surfaces(g0, base.LG, base.SCALE),
             1: lp.build_surfaces(g1, base.LG, base.SCALE)}

    res = {}
    for K in KS:
        W.K_WH_COUNT = K
        Wf = {h: W.build_wh_count(prep[h]['first']) for h in ('A', 'B')}
        Wr = {0: W.build_wh_count(g0), 1: W.build_wh_count(g1)}
        raws, parts = [], []
        for half in ('A', 'B'):
            c = prep[half]
            loc = W.score_map_whc(c['byp_f'], S_first[half], Wf[half],
                                  base.MIN_SCORE)
            kk = [k for k in loc if k in c['actual'] and k in c['velo']]
            if len(kk) < 30:
                continue
            r_ly = base.pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
            r_ls = base.pearson([loc[k] for k in kk], [c['velo'][k] for k in kk])
            r_sy = base.pearson([c['velo'][k] for k in kk], [c['actual'][k] for k in kk])
            raws.append(r_ly)
            parts.append(base.partial(r_ly, r_ls, r_sy))
        a0 = W.score_map_whc(base.by_pitcher(g0), S_rel[0], Wr[0], base.MIN_REL)
        a1 = W.score_map_whc(base.by_pitcher(g1), S_rel[1], Wr[1], base.MIN_REL)
        ks = [k for k in a0 if k in a1]
        rel = base.pearson([a0[k] for k in ks], [a1[k] for k in ks])
        res[K] = {'partial': sum(parts) / len(parts),
                  'raw': sum(raws) / len(raws), 'rel': rel}
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
            f"K={K}: p={table[yr][K]['partial']:.3f} rel={table[yr][K]['rel']:.3f}"
            for K in KS), flush=True)
        del pitches
        gc.collect()

    print("\nSWEEP SUMMARY (mean across seasons)")
    for K in KS:
        ps = [table[y][K]['partial'] for y in table]
        rs = [table[y][K]['rel'] for y in table]
        print(f"  K={K:3d}: partial {sum(ps)/len(ps):.4f}  rel {sum(rs)/len(rs):.4f}")
    print("\nPer-season argmax on partial:")
    for y in sorted(table):
        bk = max(KS, key=lambda K: table[y][K]['partial'])
        print(f"  {y}: K={bk}")
    print("\nInterior optimum required; an edge win at K=5 or K=80 means")
    print("extending the grid.")


if __name__ == '__main__':
    main()
