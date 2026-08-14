"""diet_reliability_check.py — split-half reliability of the diet_swrv
candidate atom (mean league swing-cell RV of pitches faced), 2021-2026,
3 seeds, same protocol as the phase-2 batteries.

Usage: python3 scripts/diet_reliability_check.py
"""
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline_sdplus as sd
import hitter_phase2_multiseason as H

SEEDS = (0, 1, 2)
HALF_MIN = 125
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]


def diet_map(elig, lg, sc, min_n):
    offsets = sd.build_bip_count_offsets(elig, lg, sc)
    rv_fn = sd.make_rv_xrv(lg, sc, offsets)
    raw = sd.build_weight_table(elig, rv_fn)
    zm = sd.zone_level_means(elig, rv_fn)
    table = sd.shrink_table(raw, zm)
    by_h = defaultdict(list)
    for p in elig:
        h = p.get('Batter')
        if h:
            by_h[h].append(p)
    out = {}
    for h, ps in by_h.items():
        if len(ps) < min_n:
            continue
        v = [table[(sd.classify_zone(p), sd.get_count(p), sd.cat_of(p),
                    'swing')][0] for p in ps]
        out[h] = sum(v) / len(v)
    return out


def main():
    agg = defaultdict(list)
    for year in SEASONS:
        P = H.load_season(year)
        elig = H.precompute(P)
        lg, sc = H.guts(year)
        dates = sorted({p.get('Game Date') for p in elig if p.get('Game Date')})
        with H.patched('_z16', True):
            for seed in SEEDS:
                rnd = random.Random(seed)
                sh = dates[:]
                rnd.shuffle(sh)
                ha = set(sh[:len(sh) // 2])
                Ea = [p for p in elig if p.get('Game Date') in ha]
                Eb = [p for p in elig
                      if p.get('Game Date') and p.get('Game Date') not in ha]
                a = diet_map(Ea, lg, sc, HALF_MIN)
                b = diet_map(Eb, lg, sc, HALF_MIN)
                common = [h for h in a if h in b]
                r = H.pearson([a[h] for h in common], [b[h] for h in common])
                agg[year].append(r)
                print(f"  {year} seed{seed}: diet_swrv r={r:.3f} (n={len(common)})",
                      flush=True)
        del P, elig
        import gc
        gc.collect()
    print("\nper-season means: " + '  '.join(
        f"{y}:{sum(v)/len(v):.3f}" for y, v in sorted(agg.items())))
    allr = [r for v in agg.values() for r in v if r is not None]
    print(f"overall mean split-half r: {sum(allr)/len(allr):.4f}")


if __name__ == '__main__':
    main()
