"""n0_remeasure_sd_revert.py — n0 for the 2026-08-15 SD+ definition
(un-anchored AND category-collapsed). The 165 shipped this morning was
measured with cat3 in; the revert changes table noise, so re-measure.
Protocol identical to n0_remeasure_2026_08.py (SD leg only).

Usage: python3 scripts/research/hitter/n0_remeasure_sd_revert.py
"""
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import pipeline.sdplus as sd
from handsplit_sdct_test import load_season, guts, pearson

SEASONS = [2024, 2025, 2026]
SEEDS = range(3)


def main():
    sd_rows = defaultdict(list)
    for year in SEASONS:
        P = load_season(year)
        lg, sc = guts(year)
        elig = [p for p in P if p.get('_source', 'MLB') == 'MLB' and sd.is_eligible(p)]
        rv_fn = sd.make_rv_xrv(lg, sc)
        table = sd.shrink_table(sd.build_weight_table(elig, rv_fn),
                                sd.zone_level_means(elig, rv_fn))
        zc = defaultdict(int)
        for p in elig:
            zc[sd.classify_zone(p)] += 1
        tot = sum(zc.values())
        lgw = {z: n / tot for z, n in zc.items()}
        dv_by_h = defaultdict(list)
        for p in elig:
            h = p.get('Batter')
            if h:
                dv_by_h[h].append((sd.classify_zone(p), sd.compute_dv(p, table)))

        def sd_raw(sample):
            zdv = defaultdict(list)
            for z, dv in sample:
                zdv[z].append(dv)
            zm = {z: sum(v) / len(v) for z, v in zdv.items()}
            w = sum(lgw.get(z, 0.0) for z in zm)
            if w <= 0:
                return None
            return sum(m * lgw.get(z, 0.0) for z, m in zm.items()) / w

        for N in (60, 90, 125, 190, 250, 375):
            elig_h = {h: v for h, v in dv_by_h.items() if len(v) >= 2 * N}
            if len(elig_h) < 30:
                continue
            for seed in SEEDS:
                rnd = random.Random(1000 * seed + N)
                xs, ys = [], []
                for h, v in elig_h.items():
                    s = v[:]
                    rnd.shuffle(s)
                    a, b = sd_raw(s[:N]), sd_raw(s[N:2 * N])
                    if a is not None and b is not None:
                        xs.append(a)
                        ys.append(b)
                rr = pearson(xs, ys)
                if rr is not None:
                    sd_rows[N].append(rr)
        print(f"{year} done", flush=True)
        del P

    print("\nSD+ un-anchored, category-collapsed:  N/half : mean r : implied n0")
    n0s = []
    for N, rs in sorted(sd_rows.items()):
        m = sum(rs) / len(rs)
        n0 = N * (1 - m) / m if 0 < m < 1 else None
        print(f"  {N:5d}  {m:6.3f}  {n0:8.0f}" if n0 else f"  {N:5d}  {m:6.3f}  --")
        if n0 and 90 <= N <= 375:
            n0s.append(n0)
    if n0s:
        print(f"  => consensus n0 (N in [90,375]): {sum(n0s)/len(n0s):.0f}")


if __name__ == '__main__':
    main()
