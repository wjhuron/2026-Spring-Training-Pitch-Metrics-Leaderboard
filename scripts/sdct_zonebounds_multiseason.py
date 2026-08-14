"""sdct_zonebounds_multiseason.py — sweep the SD+/CT+ zone-boundary constants
(taken from the Savant attack-zone diagram in 2026-04 and never swept).

Boundaries (pipeline_sdplus.py):
    HEART_X 6.7/12   SHADOW_X 13.3/12   CHASE_X 20/12
    SHADOW_VERT_FRAC 1/6   CHASE_VERT_FRAC 0.5
(HEART_VERT_FRAC 1/6 already validated by hitter_phase2_multiseason.py.)

One-at-a-time variants around each (10 + BASE), everything else at the
shipped config (count anchor ON, cat3, mix-neutral, k=200). Scored on SD+
raw; the CT+ effect is checked only for any winning boundary afterwards.

Metrics: split-half rel (2021-2026 x 3 seeds), predictive (4 pairs vs
next-season wOBA). Adopt only if a variant beats BASE on rel in most
seasons without losing prediction; a monotone edge means extend the grid.

Usage: python3 scripts/sdct_zonebounds_multiseason.py
"""
import json
import math
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statcast_hitter_adapter as A
import pipeline_sdplus as sd
import hitter_phase2_multiseason as H

SEEDS = (0, 1, 2)
HALF_MIN_DEC, FULL_MIN_DEC = 125, 200
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
OUT_JSON = os.path.join(ROOT, 'data', '_sdct_zonebounds_results.json')

BASECFG = {'HEART_X': 6.7 / 12, 'SHADOW_X': 13.3 / 12, 'CHASE_X': 20.0 / 12,
           'SHADOW_VERT_FRAC': 1.0 / 6.0, 'CHASE_VERT_FRAC': 0.5}
VARIANTS = [('BASE', {})] + [
    (f'{k}={v}', {k: val}) for k, pair in (
        ('HEART_X', (5.7, 7.7)),
        ('SHADOW_X', (12.3, 14.3)),
        ('CHASE_X', (18.0, 22.0)),
    ) for v, val in ((f'{pair[0]}', pair[0] / 12), (f'{pair[1]}', pair[1] / 12))
] + [
    ('SHADOW_VF=1/8', {'SHADOW_VERT_FRAC': 0.125}),
    ('SHADOW_VF=1/4', {'SHADOW_VERT_FRAC': 0.25}),
    ('CHASE_VF=0.4', {'CHASE_VERT_FRAC': 0.4}),
    ('CHASE_VF=0.6', {'CHASE_VERT_FRAC': 0.6}),
]
VNAMES = [v[0] for v in VARIANTS]


def stash_zones(P):
    """Per variant, stash the zone under its boundary config; return the
    SD+-eligible subset (boundary moves never create/remove None zones)."""
    orig = {k: getattr(sd, k) for k in BASECFG}
    try:
        for vi, (name, over) in enumerate(VARIANTS):
            for k, v in BASECFG.items():
                setattr(sd, k, over.get(k, v))
            key = f'_zv{vi}'
            for p in P:
                p[key] = H._orig_classify_zone(p)
    finally:
        for k, v in orig.items():
            setattr(sd, k, v)
    for p in P:
        p['_cnt'] = H._orig_get_count(p)
    return [p for p in P if sd.is_eligible(p)]


def season_components(elig, lg, sc, min_dec):
    res = {}
    offsets_cache = {}
    for vi, (name, _) in enumerate(VARIANTS):
        with H.patched(f'_zv{vi}', True):
            # offsets are zone-independent (count-level); compute once
            if 'o' not in offsets_cache:
                offsets_cache['o'] = sd.build_bip_count_offsets(elig, lg, sc)
            rv_fn = sd.make_rv_xrv(lg, sc, offsets_cache['o'])
            raw = sd.build_weight_table(elig, rv_fn)
            zm = sd.zone_level_means(elig, rv_fn)
            table = sd.shrink_table(raw, zm)
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
            res[name] = H.sd_score(by_hitter, table, lgw, min_dec)
    return res


def main():
    results = {'split_half': {}, 'predictive': {}}
    agg = defaultdict(list)
    print(f"SPLIT-HALF RELIABILITY (floor {HALF_MIN_DEC}/half)", flush=True)
    for year in SEASONS:
        P = H.load_season(year)
        elig = stash_zones(P)
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
            row = {}
            for k in VNAMES:
                common = [h for h in ra[k] if h in rb[k]]
                r = H.pearson([ra[k][h] for h in common], [rb[k][h] for h in common])
                row[k] = [r, len(common)]
                if r is not None:
                    agg[k].append((year, seed, r))
            results['split_half'][f'{year}_s{seed}'] = row
            print(f"  {year} seed{seed}: " + '  '.join(
                f"{k}={row[k][0]:.3f}" if row[k][0] is not None else f"{k}=NA"
                for k in VNAMES), flush=True)
        del P, elig
        import gc
        gc.collect()

    print("\nPREDICTIVE (floor 200, vs next-season wOBA >=200 events)", flush=True)
    pagg = defaultdict(list)
    for yn, yn1 in PAIRS:
        P = H.load_season(yn)
        elig = stash_zones(P)
        lg, sc = H.guts(yn)
        comp = season_components(elig, lg, sc, FULL_MIN_DEC)
        y_map = A.target_y(yn1)
        row = {}
        for k in VNAMES:
            xs, ys = [], []
            for h, v in comp[k].items():
                yv = y_map.get(h)
                if yv and yv[1] >= 200:
                    xs.append(v)
                    ys.append(yv[0] / yv[1])
            r = H.pearson(xs, ys)
            row[k] = [r, len(xs)]
            if r is not None:
                pagg[k].append((yn, r))
        results['predictive'][f'{yn}_{yn1}'] = row
        print(f"  {yn}->{yn1}: " + '  '.join(
            f"{k}={row[k][0]:+.3f}" if row[k][0] is not None else f"{k}=NA"
            for k in VNAMES), flush=True)
        del P, elig

    print("\nVERDICTS (variant minus BASE)")
    base_by = defaultdict(list)
    for y, s, r in agg['BASE']:
        base_by[y].append(r)
    pb = [r for _, r in pagg['BASE']]
    for k in VNAMES:
        if k == 'BASE':
            continue
        var_by = defaultdict(list)
        for y, s, r in agg[k]:
            var_by[y].append(r)
        wins, cells = 0, []
        for y in sorted(base_by):
            d = sum(var_by[y]) / len(var_by[y]) - sum(base_by[y]) / len(base_by[y])
            cells.append(f"{y}:{d:+.3f}")
            wins += 1 if d > 0 else 0
        pv = [r for _, r in pagg[k]]
        pd_ = (sum(pv) / len(pv) - sum(pb) / len(pb)) if pb and pv else None
        print(f"  {k}: rel wins {wins}/{len(base_by)}  " + '  '.join(cells)
              + (f"   pred {pd_:+.4f}" if pd_ is not None else ""))

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == '__main__':
    main()
