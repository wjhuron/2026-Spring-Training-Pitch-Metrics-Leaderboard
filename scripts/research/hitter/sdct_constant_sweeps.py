"""sdct_constant_sweeps.py — the never-measured small constants in SD+/CT+.

Part A — cell shrinkage k sweep {10,25,50,100,200} (production: 50 in both
pipelines): split-half reliability of raw SD+ / CT+, seasons 2023-2026,
2 random game-date partitions each. Raw cell tables are built once per half;
each k re-shrinks and re-scores.

Part B — CT+ DEFAULT fallback usage: how often does shrink_contact_cells hit
the hardcoded {p_whiff .25, rv_contact 0, rv_whiff -.05} zone default, and
how often does a cell have 0 swings (full zone-prior fallback), per season
and per half-season?

Part C — count-anchor offset stability: build_bip_count_offsets per season
2021-2026 side by side; which counts fail min_n=50 (silent 0.0 fallback);
cross-season spread per count. If offsets are stable across seasons, a
pooled multi-season table is a better sparse-count fallback than 0.0.

Usage: python3 scripts/research/hitter/sdct_constant_sweeps.py
"""
import os, sys, math, pickle, random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import statcast_hitter_adapter as A
import pipeline.sdplus as sd
import pipeline.contact as ct
from handsplit_sdct_test import (load_season, guts, pearson,
                                 sd_score_baseline, ct_score,
                                 HALF_MIN_DEC, HALF_MIN_SW)

KS = [10, 25, 50, 100, 200]
SWEEP_SEASONS = [2023, 2024, 2025, 2026]
ALL_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
SEEDS = (0, 1)


def half_components_by_k(P, lg, sc):
    """Build raw tables once, then per-k shrink + score. Returns
    {k: (sd_scores, ct_scores)}."""
    elig = [p for p in P if p.get('_source', 'MLB') == 'MLB' and sd.is_eligible(p)]
    offsets = sd.build_bip_count_offsets(elig, lg, sc)
    rv_fn = sd.make_rv_xrv(lg, sc, offsets)
    swings = [p for p in elig if ct.is_ct_eligible(p)]

    raw_sd_table = sd.build_weight_table(elig, rv_fn)
    zm_sd = sd.zone_level_means(elig, rv_fn)
    raw_ct_table = ct.build_contact_cell_weights(swings, rv_fn)
    zm_ct = ct.zone_level_contact_means(swings, rv_fn)

    zc = defaultdict(int)
    for p in elig:
        zc[sd.classify_zone(p)] += 1
    tot = sum(zc.values())
    lgw = {z: n / tot for z, n in zc.items()}

    by_hitter = defaultdict(list)
    for p in elig:
        h = p.get('Batter')
        if h:
            by_hitter[h].append(p)

    out = {}
    for k in KS:
        table = sd.shrink_table(raw_sd_table, zm_sd, k=k)
        tctb = ct.shrink_contact_cells(raw_ct_table, zm_ct, k=k)
        out[k] = (sd_score_baseline(by_hitter, table, lgw, HALF_MIN_DEC),
                  ct_score(by_hitter, lambda p: tctb, HALF_MIN_SW))
    return out


def main():
    # ── Part A ──
    print(f"PART A — cell-k sweep, split-half r (seasons {SWEEP_SEASONS})", flush=True)
    agg = defaultdict(lambda: defaultdict(list))
    for year in SWEEP_SEASONS:
        P = load_season(year)
        lg, sc = guts(year)
        dates = sorted({p.get('Game Date') for p in P if p.get('Game Date')})
        for seed in SEEDS:
            rnd = random.Random(seed)
            sh = dates[:]
            rnd.shuffle(sh)
            ha = set(sh[:len(sh) // 2])
            Pa = [p for p in P if p.get('Game Date') in ha]
            Pb = [p for p in P if p.get('Game Date') and p.get('Game Date') not in ha]
            ca = half_components_by_k(Pa, lg, sc)
            cb = half_components_by_k(Pb, lg, sc)
            for k in KS:
                for mi, name in ((0, 'sd'), (1, 'ct')):
                    common = [h for h in ca[k][mi] if h in cb[k][mi]]
                    r = pearson([ca[k][mi][h] for h in common],
                                [cb[k][mi][h] for h in common])
                    if r is not None:
                        agg[name][k].append(r)
            print(f"  {year} seed{seed} done", flush=True)
        del P
    print(f"\n  {'k':>5s} {'SD+ r':>8s} {'CT+ r':>8s}")
    for k in KS:
        print(f"  {k:5d} {sum(agg['sd'][k])/len(agg['sd'][k]):8.4f} "
              f"{sum(agg['ct'][k])/len(agg['ct'][k]):8.4f}")

    # ── Part B ──
    print("\nPART B — CT+ DEFAULT / sparse-cell usage", flush=True)
    for year in ALL_SEASONS:
        P = load_season(year)
        lg, sc = guts(year)
        elig = [p for p in P if p.get('_source', 'MLB') == 'MLB' and sd.is_eligible(p)]
        offsets = sd.build_bip_count_offsets(elig, lg, sc)
        rv_fn = sd.make_rv_xrv(lg, sc, offsets)
        swings = [p for p in elig if ct.is_ct_eligible(p)]
        for label, sub in (('full', swings), ('half', swings[:len(swings) // 2])):
            raw = ct.build_contact_cell_weights(sub, rv_fn)
            zm = ct.zone_level_contact_means(sub, rv_fn)
            missing_zone = [z for z in ct.ZONES if z not in zm]
            empty_cells = sum(1 for z in ct.ZONES for c in ct.COUNTS
                              if (z, c) not in raw or raw[(z, c)]['n_swings'] == 0)
            thin_cells = sum(1 for z in ct.ZONES for c in ct.COUNTS
                             if (z, c) in raw and 0 < raw[(z, c)]['n_swings'] < 20)
            print(f"  {year} {label}: DEFAULT zones {len(missing_zone)} "
                  f"{missing_zone or ''}, empty cells {empty_cells}/60, "
                  f"cells n<20 {thin_cells}", flush=True)
        del P

    # ── Part C ──
    print("\nPART C — count-anchor offset stability across seasons", flush=True)
    tables = {}
    for year in ALL_SEASONS:
        P = load_season(year)
        lg, sc = guts(year)
        elig = [p for p in P if p.get('_source', 'MLB') == 'MLB' and sd.is_eligible(p)]
        tables[year] = sd.build_bip_count_offsets(elig, lg, sc)
        del P
    counts = sorted({c for t in tables.values() for c in t})
    hdr = '  count  ' + ' '.join(f"{y:>7d}" for y in ALL_SEASONS) + '   spread'
    print(hdr)
    for c in sorted([(b, s) for b in range(4) for s in range(3)]):
        vals = []
        row = f"  {c[0]}-{c[1]}    "
        for y in ALL_SEASONS:
            v = tables[y].get(c)
            row += (f"{v:+7.3f} " if v is not None else "   MISS ")
            if v is not None:
                vals.append(v)
        spread = (max(vals) - min(vals)) if len(vals) >= 2 else float('nan')
        print(row + f"  {spread:.3f}")


if __name__ == '__main__':
    main()
