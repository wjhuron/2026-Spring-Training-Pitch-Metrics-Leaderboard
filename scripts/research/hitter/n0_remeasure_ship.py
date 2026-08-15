"""n0_remeasure_ship.py — re-measure stabilization constants for the SHIPPED
definitions (2026-07-13 adoptions):

  1. BB+ n0 for the PURE-CON definition (BB+ = xwOBAcon+ only; the measured
     63 was for the retired 85/15 con/spray composite).
  2. SD+ / CT+ n0 with k=200 cell tables (the measured 250 / 85 were for
     k=50; heavier cell smoothing lowers table noise, so n0 drops).

Method matches the prior studies: per-hitter subsample split-half at fixed
N per half, r across hitters, implied n0 = N(1-r)/r under rel(n)=n/(n+n0).
Full-data league tables (they're constants), seasons 2024, 2025, 2026,
3 seeds each.

Usage: python3 scripts/research/hitter/n0_remeasure_ship.py
"""
import os, sys, random
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import statcast_hitter_adapter as A
import pipeline.sdplus as sd
import pipeline.contact as ct
from pipeline.utils import safe_float
from handsplit_sdct_test import load_season, guts, pearson
from bbplus_n0_cliff_test import bip_records

SEASONS = [2024, 2025, 2026]
SEEDS = range(3)
K_NEW = 200


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
    bb_rows = defaultdict(list)
    sd_rows = defaultdict(list)
    ct_rows = defaultdict(list)

    for year in SEASONS:
        P = load_season(year)
        lg, sc = guts(year)

        # ── BB+ pure-con: per-hitter (xw) records ──
        recs, lg_xc = bip_records(P)
        for N in (20, 30, 40, 60, 80, 120):
            elig = {h: r for h, r in recs.items() if len(r) >= 2 * N}
            if len(elig) < 30:
                continue
            for seed in SEEDS:
                rnd = random.Random(1000 * seed + N)
                xs, ys = [], []
                for h, r in elig.items():
                    s = r[:]
                    rnd.shuffle(s)
                    xs.append(sum(v[0] for v in s[:N]) / N / lg_xc * 100.0)
                    ys.append(sum(v[0] for v in s[N:2 * N]) / N / lg_xc * 100.0)
                rr = pearson(xs, ys)
                if rr is not None:
                    bb_rows[N].append(rr)

        # ── SD+ / CT+ with k=200 tables ──
        elig = [p for p in P if p.get('_source', 'MLB') == 'MLB' and sd.is_eligible(p)]
        offsets = sd.build_bip_count_offsets(elig, lg, sc)
        rv_fn = sd.make_rv_xrv(lg, sc, offsets)
        swings = [p for p in elig if ct.is_ct_eligible(p)]

        table = sd.shrink_table(sd.build_weight_table(elig, rv_fn),
                                sd.zone_level_means(elig, rv_fn), k=K_NEW)
        tctb = ct.shrink_contact_cells(ct.build_contact_cell_weights(swings, rv_fn),
                                       ct.zone_level_contact_means(swings, rv_fn),
                                       k=K_NEW)
        zc = defaultdict(int)
        for p in elig:
            zc[sd.classify_zone(p)] += 1
        tot = sum(zc.values())
        lgw = {z: n / tot for z, n in zc.items()}

        # per-hitter dv streams (zone, dv) and swing streams (lev, con, exp)
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
            cell = tctb[(sd.classify_zone(p), sd.get_count(p))]
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

    report("BB+ pure-con (BIPs)", bb_rows, 30, 120)
    report(f"SD+ k={K_NEW} (decisions)", sd_rows, 90, 375)
    report(f"CT+ k={K_NEW} (swings)", ct_rows, 40, 180)


if __name__ == '__main__':
    main()
