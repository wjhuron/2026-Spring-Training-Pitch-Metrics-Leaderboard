"""commandplus_bb_followup.py — the walk-channel test + velo-confound isolation.

The battery left one open verdict. Command+ (mean miss) is spectacularly
reliable (0.795) and persistent (0.793) — but its partial predictive value
for next-season xRV GIVEN Loc+ came out NEGATIVE in all four year-pairs
(-0.15 to -0.05), and even its RAW correlation with future xRV is ~zero.

Two hypotheses before concluding "descriptive tool grade only":

  H1 (walk channel): command's natural outcome is WALKS, not xRV — run
     prevention is stuff-dominated, but execution tightness should show up
     in BB%. Test: r(miss_N, BB%_N) same-season and r(miss_N, BB%_{N+1})
     next-season, plus incremental over Loc+.

  H2 (velo confound): miss correlates +0.27-0.31 with velo in every season,
     and velo predicts run prevention. The negative partial may be velocity
     hiding inside the miss term. Test: partial r(miss_N -> xRV_{N+1})
     controlling BOTH loc_N and velo_N (two-stage residual partial).

BB% here = walks / PA-ending pitches, reconstructed from pitch data: a walk
is a Ball on a 3-ball count; a PA end is any In Play / strikeout-strike /
walk / HBP pitch. Good enough for correlation work.

Usage: python3 scripts/commandplus_bb_followup.py
"""
import os, sys, math, pickle, gc
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pipeline.locplus as lp
from pipeline.sdplus import make_rv_xrv
from locplus_constants_multiseason import adapt
from commandplus_v1 import score_pitches_multi, aggregate

LG, SCALE = 0.3169, 1.2393
MIN_FULL = 300
CACHE = {2021: 'data/_statcast2021_cache.pkl', 2022: 'data/_statcast2022_cache.pkl',
         2023: 'data/_statcast2023_cache.pkl', 2024: 'data/_statcast2024_cache.pkl',
         2025: 'data/_statcast2025_full_cache.pkl'}
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def residualize(y, xs_list):
    """OLS residual of y on the given predictors (with intercept)."""
    X = np.column_stack([np.ones(len(y))] + xs_list)
    yv = np.asarray(y)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return yv - X @ beta


def season_pitches(year):
    if year == 2026:
        D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
        ep = {(p.get('Pitcher'), p.get('PTeam')) for p in D if p.get('Pitch Type') == 'EP'}
        return [p for p in D if p.get('_source', 'MLB') == 'MLB'
                and (p.get('Pitcher'), p.get('PTeam')) not in ep]
    return adapt(os.path.join(ROOT, CACHE[year]))


K_DESC = {'Swinging Strike', 'Called Strike'}


def process(year):
    pitches = season_pitches(year)
    multi = score_pitches_multi(pitches, [0.0])
    miss = {k: v[0] for k, v in aggregate(multi[0.0][0], MIN_FULL).items()}
    del multi

    # BB% reconstruction + velo + loc + xrv
    bb = defaultdict(lambda: [0, 0])            # walks, pa-ends
    velo = defaultdict(list)
    for p in pitches:
        key = (p.get('Pitcher'), p.get('Throws'))
        d = p.get('Description'); c = p.get('Count') or ''
        if d == 'Ball' and c.startswith('3-'):
            bb[key][0] += 1; bb[key][1] += 1
        elif d == 'In Play' or d == 'Hit By Pitch':
            bb[key][1] += 1
        elif d in K_DESC and c.endswith('-2'):
            bb[key][1] += 1
        if p.get('Pitch Type') == 'FF':
            v = lp.safe_float(p.get('Velocity'))
            if v is not None:
                velo[key].append(v)
    bbp = {k: w / n for k, (w, n) in bb.items() if n >= 100}
    velo = {k: sum(v) / len(v) for k, v in velo.items() if len(v) >= 50}

    base = [p for p in pitches if lp.is_eligible_baseline(p)]
    S = lp.build_surfaces(base, LG, SCALE)
    acc = defaultdict(list)
    for p in base:
        v = lp.score_pitch(p, S)
        if v is not None:
            acc[(p.get('Pitcher'), p.get('Throws'))].append(v)
    loc = {k: sum(v) / len(v) for k, v in acc.items() if len(v) >= MIN_FULL}

    rv_fn = make_rv_xrv(LG, SCALE)
    acc = defaultdict(list)
    for p in pitches:
        v = rv_fn(p)
        if v is not None:
            acc[(p.get('Pitcher'), p.get('Throws'))].append(v)
    xrv = {k: sum(v) / len(v) for k, v in acc.items() if len(v) >= MIN_FULL}
    print(f"  {year} done ({len(miss)} pitchers)", flush=True)
    del pitches, base, S
    gc.collect()
    return {'miss': miss, 'bb': bbp, 'velo': velo, 'loc': loc, 'xrv': xrv}


def main():
    data = {y: process(y) for y in SEASONS}

    print()
    print("H1 — the walk channel")
    print(f"{'year':>6s} {'r(miss,BB%) same':>17s}")
    for y in SEASONS:
        d = data[y]
        kk = [k for k in d['miss'] if k in d['bb']]
        print(f"{y:>6d} {pearson([d['miss'][k] for k in kk], [d['bb'][k] for k in kk]):>+17.3f}")
    pairs = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
    print(f"{'pair':>12s} {'r(miss->BB%)':>13s} {'r(BB%->BB%)':>12s} "
          f"{'partial miss|BB%':>17s} {'partial miss|loc':>17s}")
    for a, b in pairs:
        da, db = data[a], data[b]
        kk = [k for k in da['miss'] if k in da['bb'] and k in db['bb'] and k in da['loc']]
        m = [da['miss'][k] for k in kk]; b0 = [da['bb'][k] for k in kk]
        b1 = [db['bb'][k] for k in kk]; l0 = [da['loc'][k] for k in kk]
        r_mb = pearson(m, b1); r_bb = pearson(b0, b1)
        # incremental: miss residualized on same-season BB% / on loc
        rm_b = residualize(m, [b0]); rb1_b = residualize(b1, [b0])
        rm_l = residualize(m, [l0]); rb1_l = residualize(b1, [l0])
        print(f"{a}->{b:>5d} {r_mb:>+13.3f} {r_bb:>+12.3f} "
              f"{pearson(rm_b.tolist(), rb1_b.tolist()):>+17.3f} "
              f"{pearson(rm_l.tolist(), rb1_l.tolist()):>+17.3f}")

    print()
    print("H2 — velo-confound isolation: partial r(miss_N -> xRV_{N+1})")
    print(f"{'pair':>12s} {'| loc':>8s} {'| loc+velo':>11s}")
    for a, b in pairs:
        da, db = data[a], data[b]
        kk = [k for k in da['miss'] if k in da['loc'] and k in da['velo']
              and k in db['xrv']]
        m = [da['miss'][k] for k in kk]; l0 = [da['loc'][k] for k in kk]
        v0 = [da['velo'][k] for k in kk]; y1 = [db['xrv'][k] for k in kk]
        rm_l = residualize(m, [l0]); ry_l = residualize(y1, [l0])
        rm_lv = residualize(m, [l0, v0]); ry_lv = residualize(y1, [l0, v0])
        print(f"{a}->{b:>5d} {pearson(rm_l.tolist(), ry_l.tolist()):>+8.3f} "
              f"{pearson(rm_lv.tolist(), ry_lv.tolist()):>+11.3f}  (n={len(kk)})")

    print()
    print("READ: H1 positive same-season + incremental next-season BB% -> the")
    print("metric has a real outcome channel (walks) and ships as more than a")
    print("tool grade. H2 partial moving toward 0 with velo controlled -> the")
    print("negative xRV partial was the velocity tradeoff, not 'wildness good'.")


if __name__ == '__main__':
    main()
