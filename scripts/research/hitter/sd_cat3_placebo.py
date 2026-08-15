"""sd_cat3_placebo.py — placebo control for SD+'s cat3 cells.

The CT+ cat3 placebo showed fake category structure BEATS real cat3 on
split-half reliability (flexibility artifact) while real cat3 wins on
prediction. SD+'s cat3 shipped 2026-07-02 primarily on reliability, and
the phase-2 battery's predictive leg slightly favored REMOVAL (+0.003).
Same three-way design: no-cat vs real cat3 vs permuted-label cat3
(permuted for table build, real labels for scoring).

If placebo >= real on rel AND prediction favors removal, the honest read
is that SD+'s cat dimension is an artifact and should come out.

Usage: python3 scripts/research/hitter/sd_cat3_placebo.py
"""
import random
import sys
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import statcast_hitter_adapter as A
import pipeline.sdplus as sd
import hitter_phase2_multiseason as H

SEEDS = (0, 1, 2)
HALF_MIN_DEC, FULL_MIN_DEC = 125, 200
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
PLACEBO_SEED = 17
KEYS = ('sd_nocat', 'sd_cat3', 'sd_cat3_placebo')


def build_table(elig, rv_fn, catfn):
    orig = sd.cat_of
    try:
        sd.cat_of = catfn
        raw = sd.build_weight_table(elig, rv_fn)
        zm = sd.zone_level_means(elig, rv_fn)
        return sd.shrink_table(raw, zm)
    finally:
        sd.cat_of = orig


def season_components(elig, lg, sc, min_dec):
    """SD raws under the three variants (shipped config otherwise:
    un-anchored rv, mix-neutral aggregation)."""
    res = {}
    with H.patched('_z16', True):
        rv_fn = sd.make_rv_xrv(lg, sc)
        real_cat = sd.cat_of

        cats = [real_cat(p) for p in elig]
        random.Random(PLACEBO_SEED).shuffle(cats)
        for p, c in zip(elig, cats):
            p['_catshuf'] = c

        tables = {
            'sd_nocat': build_table(elig, rv_fn, lambda p: 'FB'),
            'sd_cat3': build_table(elig, rv_fn, real_cat),
            'sd_cat3_placebo': build_table(elig, rv_fn,
                                           lambda p: p['_catshuf']),
        }
        worst = max(abs(tables['sd_cat3_placebo'][k][0]
                        - tables['sd_cat3'][k][0])
                    for k in tables['sd_cat3'])
        assert worst > 1e-6, 'placebo tables identical — patch inert'

        zc = defaultdict(int)
        for p in elig:
            zc[sd.classify_zone(p)] += 1
        tot = sum(zc.values())
        lgw = {z: n / tot for z, n in zc.items()}
        by_h = defaultdict(list)
        for p in elig:
            h = p.get('Batter')
            if h:
                by_h[h].append(p)

        # scoring: real cats for nocat's lookups must be 'FB' (its cells)
        for name, tab in tables.items():
            orig = sd.cat_of
            try:
                sd.cat_of = (lambda p: 'FB') if name == 'sd_nocat' else real_cat
                res[name] = H.sd_score(by_h, tab, lgw, min_dec)
            finally:
                sd.cat_of = orig
    return res


def main():
    agg = defaultdict(list)
    print("SPLIT-HALF RELIABILITY", flush=True)
    for year in SEASONS:
        P = H.load_season(year)
        elig = H.precompute(P)
        lg, sc = H.guts(year)
        dates = sorted({p.get('Game Date') for p in elig if p.get('Game Date')})
        for seed in SEEDS:
            rnd = random.Random(seed)
            sh = dates[:]
            rnd.shuffle(sh)
            ha = set(sh[:len(sh) // 2])
            Ea = [p for p in elig if p.get('Game Date') in ha]
            Eb = [p for p in elig if p.get('Game Date') and p.get('Game Date') not in ha]
            ra = season_components(Ea, lg, sc, HALF_MIN_DEC)
            rb = season_components(Eb, lg, sc, HALF_MIN_DEC)
            row = []
            for k in KEYS:
                common = [h for h in ra[k] if h in rb[k]]
                r = H.pearson([ra[k][h] for h in common], [rb[k][h] for h in common])
                if r is not None:
                    agg[k].append((year, r))
                row.append(f"{k}={r:.3f}" if r is not None else f"{k}=NA")
            print(f"  {year} s{seed}: " + '  '.join(row), flush=True)
        del P, elig
        import gc
        gc.collect()

    print("\nMEAN split-half r:")
    for k in KEYS:
        rs = [r for _, r in agg[k]]
        by = defaultdict(list)
        for y, r in agg[k]:
            by[y].append(r)
        print(f"  {k}: {sum(rs)/len(rs):.4f}  "
              + '  '.join(f"{y}:{sum(v)/len(v):.3f}" for y, v in sorted(by.items())))

    print("\nPREDICTIVE (4 pairs, vs next-season wOBA >=200 events)", flush=True)
    pagg = defaultdict(list)
    for yn, yn1 in PAIRS:
        P = H.load_season(yn)
        elig = H.precompute(P)
        lg, sc = H.guts(yn)
        comp = season_components(elig, lg, sc, FULL_MIN_DEC)
        y_map = A.target_y(yn1)
        row = []
        for k in KEYS:
            xs, ys = [], []
            for h, v in comp[k].items():
                yv = y_map.get(h)
                if yv and yv[1] >= 200:
                    xs.append(v)
                    ys.append(yv[0] / yv[1])
            r = H.pearson(xs, ys)
            if r is not None:
                pagg[k].append(r)
            row.append(f"{k}={r:+.3f}" if r is not None else f"{k}=NA")
        print(f"  {yn}->{yn1}: " + '  '.join(row), flush=True)
        del P, elig

    print("\nMEAN predictive r:")
    for k in KEYS:
        if pagg[k]:
            print(f"  {k}: {sum(pagg[k])/len(pagg[k]):+.4f}")

    print("\nREAD: if placebo >= real on rel while real does not beat both")
    print("nocat and placebo on prediction, SD+'s cat dimension is artifact.")


if __name__ == '__main__':
    main()
