"""locplus_count_placebo.py — negative control for the count-surface wins.

wh_count and xw_clevel carry more parameters than shipped. If the harness
rewards flexibility itself, a FAKE count structure would also win. Here the
count labels are randomly permuted among eligible pitches (within the
baseline slice, seed-fixed) when BUILDING the count-specific whiff grids
and the contact offsets; scoring still uses each pitch's real count. The
permutation destroys real count signal while keeping the extra degrees of
freedom.

Pass criteria: placebo must NOT beat shipped, and the real variants must
beat the placebo, across the replicate seasons. Otherwise the earlier wins
are harness bias.

Usage: python3 scripts/research/locplus/locplus_count_placebo.py
"""
import gc
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pipeline.locplus as lp
import locplus_constants_multiseason as base
import locplus_countwhiff_multiseason as W
import locplus_countphys_extension as X
import locplus_countcompose_multiseason as C
from pipeline.sdplus import make_rv_xrv

SEED = 17


class shuffled_counts:
    """Temporarily permute 'Count' among the given pitches (seed-fixed);
    restore on exit. Single-threaded use only."""

    def __init__(self, pitches, seed=SEED):
        self.pitches = pitches
        self.seed = seed

    def __enter__(self):
        self.orig = [p.get('Count') for p in self.pitches]
        sh = self.orig[:]
        random.Random(self.seed).shuffle(sh)
        for p, c in zip(self.pitches, sh):
            p['Count'] = c

    def __exit__(self, *a):
        for p, c in zip(self.pitches, self.orig):
            p['Count'] = c


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

    def builders(pitch_slice, placebo):
        if placebo:
            with shuffled_counts(pitch_slice):
                wh = W.build_wh_count(pitch_slice)
                xw = X.build_xw_clevel(pitch_slice, base.LG, base.SCALE)
        else:
            wh = W.build_wh_count(pitch_slice)
            xw = X.build_xw_clevel(pitch_slice, base.LG, base.SCALE)
        return wh, xw

    variants = {'shipped': None, 'real_both': False, 'placebo_both': True}
    res = {}
    for name, placebo in variants.items():
        if placebo is None:
            WHf = XWf = {h: None for h in ('A', 'B')}
            WHr = XWr = {0: None, 1: None}
        else:
            WHf, XWf = {}, {}
            for h in ('A', 'B'):
                WHf[h], XWf[h] = builders(prep[h]['first'], placebo)
            WHr, XWr = {}, {}
            for i, g in ((0, g0), (1, g1)):
                WHr[i], XWr[i] = builders(g, placebo)
        raws, parts = [], []
        for half in ('A', 'B'):
            c = prep[half]
            loc = C.score_map_c(c['byp_f'], S_first[half], base.MIN_SCORE,
                                WHf[half], XWf[half])
            kk = [k for k in loc if k in c['actual'] and k in c['velo']]
            if len(kk) < 30:
                continue
            r_ly = base.pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
            r_ls = base.pearson([loc[k] for k in kk], [c['velo'][k] for k in kk])
            r_sy = base.pearson([c['velo'][k] for k in kk], [c['actual'][k] for k in kk])
            raws.append(r_ly)
            parts.append(base.partial(r_ly, r_ls, r_sy))
        a0 = C.score_map_c(base.by_pitcher(g0), S_rel[0], base.MIN_REL,
                           WHr[0], XWr[0])
        a1 = C.score_map_c(base.by_pitcher(g1), S_rel[1], base.MIN_REL,
                           WHr[1], XWr[1])
        ks = [k for k in a0 if k in a1]
        rel = base.pearson([a0[k] for k in ks], [a1[k] for k in ks])
        res[name] = {'partial': sum(parts) / len(parts), 'rel': rel}
    return res


def main():
    rv_fn = make_rv_xrv(base.LG, base.SCALE)
    seasons = [(2021, 'data/_statcast2021_cache.pkl'),
               (2022, 'data/_statcast2022_cache.pkl'),
               (2023, 'data/_statcast2023_cache.pkl'),
               (2024, 'data/_statcast2024_cache.pkl'),
               (2025, 'data/_statcast2025_full_cache.pkl')]
    names = ('shipped', 'real_both', 'placebo_both')
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
            for n in names), flush=True)
        del pitches
        gc.collect()

    print("\nPLACEBO VERDICT (partial deltas vs shipped, per season)")
    for n in ('real_both', 'placebo_both'):
        cells = []
        wins = 0
        for yr in sorted(table):
            d = table[yr][n]['partial'] - table[yr]['shipped']['partial']
            cells.append(f"{yr}:{d:+.3f}")
            wins += 1 if d > 0 else 0
        print(f"  {n}: wins {wins}/{len(cells)}  " + '  '.join(cells))
    print("\nPASS = placebo does NOT beat shipped while real does. A winning")
    print("placebo means the count-surface gains are harness bias.")


if __name__ == '__main__':
    main()
