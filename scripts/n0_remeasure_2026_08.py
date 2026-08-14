"""n0_remeasure_2026_08.py — re-measure SD+/CT+ stabilization constants for
the 2026-08-15 definitions (SD+ un-anchored BIP branch; CT+ cat3 cells).

The shipped HITTER_PRIOR_N values (SD+ 200, CT+ 65) were measured
2026-07-13 on the anchored / zone-x-count tables. Definition changes move
table noise, so n0 must be re-measured at the shipped config (tune at the
sample size and definition you run at). Protocol identical to
scripts/n0_remeasure_ship.py: per-hitter subsample split-half at fixed N
per half, implied n0 = N(1-r)/r, full-data league tables, seasons
2024-2026, 3 seeds.

Usage: python3 scripts/n0_remeasure_2026_08.py
"""
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import pipeline_sdplus as sd
import pipeline_contact as ct
from handsplit_sdct_test import load_season, guts, pearson

SEASONS = [2024, 2025, 2026]
SEEDS = range(3)


def implied(rows):
    out = {}
    for N, rs in sorted(rows.items()):
        m = sum(rs) / len(rs)
        out[N] = (m, N * (1 - m) / m if 0 < m < 1 else None)
    return out


def report(name, rows, lo, hi):
    print(f"\n{name}:  N/half : mean r : implied n0")
    n0s = []
    for N, (m, n0) in implied(rows).items():
        print(f"  {N:5d}  {m:6.3f}  {n0:8.0f}" if n0 else f"  {N:5d}  {m:6.3f}  --")
        if n0 and lo <= N <= hi:
            n0s.append(n0)
    if n0s:
        print(f"  => consensus n0 (N in [{lo},{hi}]): {sum(n0s)/len(n0s):.0f}")


def main():
    sd_rows = defaultdict(list)
    ct_rows = defaultdict(list)

    for year in SEASONS:
        P = load_season(year)
        lg, sc = guts(year)
        elig = [p for p in P if p.get('_source', 'MLB') == 'MLB' and sd.is_eligible(p)]
        swings = [p for p in elig if ct.is_ct_eligible(p)]

        # SD+: NEW definition — un-anchored rv (matches compute_sd_plus)
        rv_sd = sd.make_rv_xrv(lg, sc)
        table = sd.shrink_table(sd.build_weight_table(elig, rv_sd),
                                sd.zone_level_means(elig, rv_sd))

        # CT+: anchored rv (unchanged) + cat3 tables (new)
        offsets = ct.build_bip_count_offsets(swings, lg, sc)
        rv_ct = ct.make_rv_xrv(lg, sc, offsets)
        tctb = ct.shrink_contact_cells(ct.build_contact_cell_weights(swings, rv_ct),
                                       ct.zone_level_contact_means(swings, rv_ct))

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
        sw_by_h = defaultdict(list)
        for p in swings:
            h = p.get('Batter')
            if not h:
                continue
            cell = tctb[(sd.classify_zone(p), sd.get_count(p), sd.cat_of(p))]
            lev = cell['rv_contact'] - cell['rv_whiff']
            if lev <= 0:
                continue
            con = 1 if ct.classify_contact_outcome(p) == 'contact' else 0
            sw_by_h[h].append((lev * con, lev * (1.0 - cell['p_whiff'])))

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

        for N in (25, 40, 60, 85, 125, 180):
            elig_h = {h: v for h, v in sw_by_h.items() if len(v) >= 2 * N}
            if len(elig_h) < 30:
                continue
            for seed in SEEDS:
                rnd = random.Random(1000 * seed + N)
                xs, ys = [], []
                for h, v in elig_h.items():
                    s = v[:]
                    rnd.shuffle(s)
                    ea = sum(e for _, e in s[:N])
                    eb = sum(e for _, e in s[N:2 * N])
                    if ea > 0 and eb > 0:
                        xs.append(sum(a for a, _ in s[:N]) / ea)
                        ys.append(sum(a for a, _ in s[N:2 * N]) / eb)
                rr = pearson(xs, ys)
                if rr is not None:
                    ct_rows[N].append(rr)
        print(f"{year} done", flush=True)
        del P

    report("SD+ un-anchored (decisions)", sd_rows, 90, 375)
    report("CT+ cat3 (swings)", ct_rows, 40, 180)


if __name__ == '__main__':
    main()
