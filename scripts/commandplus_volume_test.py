"""Command+ volume-bias test.

QUESTION: does raw_miss depend on how many pitches a pitcher threw, through
MIN_CELL=20 and the sample-dependent K_CAPS (30/60)?  If yes, the leaderboard
is comparing starters and relievers on an uneven scale.

TWO OPPOSING MECHANISMS, both real:
  (a) fewer pitches -> lower K cap -> fewer targets -> LARGER miss, and more
      cells fall under MIN_CELL and drop out entirely
  (b) fewer pitches -> targets are fit on the same points they score
      (in-sample optimism) -> SMALLER miss
Only measurement says which wins, and by how much.

DESIGN: take high-volume pitchers, randomly downsample their own pitches to
reliever-scale N, rescore with the production scorer, compare to their
full-sample value.  Random subsampling of one pitcher's own pitches is the
right null: it reproduces the thin-cell structure of a low-volume pitcher
while holding true command fixed.

Also reports the observed raw_miss-vs-volume relationship in the live pool
(confounded by quality, so it is a descriptive check, not the causal one).

Usage: python3 scripts/commandplus_volume_test.py
"""
import json
import math
import os
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_commandplus import (MIN_CELL, MIN_POOL, CMD_SCALE_K,
                                  count_group, fit_targets, is_eligible)
from pipeline_utils import safe_float

CACHE = 'data/all_pitches_rs_cache.pkl'
LEADERBOARD = 'data/pitcher_leaderboard_rs.json'

N_GRID = [400, 600, 900, 1300]
SEEDS = 5
MIN_FULL = 1400          # need headroom above the largest downsample


# ── scoring (mirrors pipeline_commandplus.score_misses for one pitcher) ──
def score_one(pts_by_cell):
    total, n_tot, n_cells = 0.0, 0, 0
    for pts in pts_by_cell.values():
        if len(pts) < MIN_CELL:
            continue
        targets = fit_targets(pts)
        n_cells += 1
        for x, z in pts:
            total += min(math.hypot(x - tx, z - tz) for tx, tz in targets)
            n_tot += 1
    if n_tot == 0:
        return None, 0, 0
    return total / n_tot, n_tot, n_cells


def cells_from(pitches):
    cells = defaultdict(list)
    for p in pitches:
        cells[(p[0], p[1], p[2])].append((p[3], p[4]))
    return cells


def run_pitcher(job):
    """job = (name, [(pt, bats, cgroup, x_in, z_in), ...])"""
    name, pitches = job
    out = {'name': name}
    full_miss, full_n, full_cells = score_one(cells_from(pitches))
    if full_miss is None:
        return None
    out['full'] = (full_miss, full_n, full_cells)
    out['sub'] = {}
    for N in N_GRID:
        if N >= len(pitches):
            continue
        vals = []
        for s in range(SEEDS):
            rng = random.Random(hash((name, N, s)) & 0xFFFFFFFF)
            sample = rng.sample(pitches, N)
            m, nt, nc = score_one(cells_from(sample))
            if m is not None:
                vals.append((m, nt, nc))
        if vals:
            out['sub'][N] = (
                sum(v[0] for v in vals) / len(vals),      # mean miss
                sum(v[1] for v in vals) / len(vals),      # mean scored pitches
                sum(v[2] for v in vals) / len(vals),      # mean cells used
            )
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float('nan')
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float('nan')
    return sxy / math.sqrt(sxx * syy)


def main():
    rows = pd.read_pickle(CACHE)

    by_p = defaultdict(list)
    for r in rows:
        if not is_eligible(r):
            continue
        by_p[r.get('Pitcher')].append((
            r['Pitch Type'], r['Bats'], count_group(r['Count']),
            safe_float(r['PlateX']) * 12.0, safe_float(r['PlateZ']) * 12.0,
        ))

    jobs = [(k, v) for k, v in by_p.items() if len(v) >= MIN_FULL]
    print(f'{len(by_p)} pitchers in cache; {len(jobs)} with >= {MIN_FULL} '
          f'eligible pitches (downsample cohort)\n')

    # ── pool sigma, so deltas can be quoted in Command+ points ──
    lb = json.load(open(LEADERBOARD))
    pool = [r['commandPlusRaw'] for r in lb
            if r.get('commandPlusRaw') and (r.get('commandPlusN') or 0) >= MIN_POOL
            and r.get('Team') != 'ROC']
    mu = sum(pool) / len(pool)
    sigma = math.sqrt(sum((x - mu) ** 2 for x in pool) / len(pool))
    pts_per_inch = CMD_SCALE_K / sigma
    print(f'MLB pool (n={len(pool)}): mean miss {mu:.3f}"  sigma {sigma:.3f}"'
          f'   ->  1 inch = {pts_per_inch:.1f} Command+ points\n')

    with Pool() as pool_exec:
        results = [r for r in pool_exec.map(run_pitcher, jobs) if r]

    # ── headline: bias vs full sample at each downsampled N ──
    print('=' * 78)
    print('DOWNSAMPLE TEST  (same pitchers, same true command, fewer pitches)')
    print('=' * 78)
    print(f'{"N":>6} {"pitchers":>9} {"d miss":>9} {"d Cmd+":>8} '
          f'{"cells full":>11} {"cells sub":>10} {"% scored":>9}')
    for N in N_GRID:
        deltas, cf, cs, frac = [], [], [], []
        for r in results:
            if N not in r['sub']:
                continue
            deltas.append(r['sub'][N][0] - r['full'][0])
            cf.append(r['full'][2])
            cs.append(r['sub'][N][2])
            frac.append(r['sub'][N][1] / N)
        if not deltas:
            continue
        d = sum(deltas) / len(deltas)
        sd = math.sqrt(sum((x - d) ** 2 for x in deltas) / len(deltas))
        se = sd / math.sqrt(len(deltas))
        print(f'{N:>6} {len(deltas):>9} {d:>+8.3f}" {d * -pts_per_inch:>+8.2f} '
              f'{sum(cf) / len(cf):>11.1f} {sum(cs) / len(cs):>10.1f} '
              f'{100 * sum(frac) / len(frac):>8.1f}%')
        print(f'{"":>6} {"":>9} {"+/-" + f"{1.96 * se:.3f}":>9} '
              f'{"+/-" + f"{1.96 * se * pts_per_inch:.2f}":>8}   (95% CI on the mean)')

    # per-pitcher spread at the tightest N, to see if the bias is uniform
    N = N_GRID[0]
    ds = sorted(r['sub'][N][0] - r['full'][0] for r in results if N in r['sub'])
    if ds:
        print(f'\nper-pitcher delta at N={N}: min {ds[0]:+.2f}"  '
              f'p25 {ds[len(ds) // 4]:+.2f}"  med {ds[len(ds) // 2]:+.2f}"  '
              f'p75 {ds[3 * len(ds) // 4]:+.2f}"  max {ds[-1]:+.2f}"')
        print(f'  ({sum(1 for d in ds if d > 0)}/{len(ds)} pitchers got WORSE '
              f'(larger miss) on the thinner sample)')

    # ── descriptive: observed volume relationship in the live pool ──
    print('\n' + '=' * 78)
    print('OBSERVED POOL  (confounded by quality; descriptive only)')
    print('=' * 78)
    live = [(r['commandPlusN'], r['commandPlusRaw'], r.get('Position') or r.get('Pos'))
            for r in lb if r.get('commandPlusRaw')
            and (r.get('commandPlusN') or 0) >= MIN_POOL and r.get('Team') != 'ROC']
    xs = [a for a, b, c in live]
    ys = [b for a, b, c in live]
    print(f'r(n_pitches, raw_miss) = {pearson(xs, ys):+.3f}  (n={len(live)})')
    buckets = [(300, 600), (600, 1000), (1000, 1600), (1600, 10 ** 9)]
    for lo, hi in buckets:
        v = [b for a, b, c in live if lo <= a < hi]
        if v:
            print(f'  {lo:>5}-{hi if hi < 10 ** 9 else "inf":>5} pitches  '
                  f'n={len(v):>4}  mean miss {sum(v) / len(v):.3f}"')


if __name__ == '__main__':
    main()
