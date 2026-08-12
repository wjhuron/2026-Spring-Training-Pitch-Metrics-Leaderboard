"""Command+ volume test, follow-ups to scripts/commandplus_volume_test.py.

(A) Is the per-pitcher volume delta real heterogeneity, or just seed noise?
    Decompose the N=400 delta variance into within-pitcher (seed) and
    between-pitcher (true) components.

(B) How much of a low-volume pitcher's arsenal actually gets scored in
    PRODUCTION?  MIN_CELL=20 silently drops thin cells, so a reliever may be
    graded on his fastball only.  Reports live coverage and which pitch types
    go missing.
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

from pipeline_commandplus import (MIN_CELL, MIN_POOL, count_group,
                                  fit_targets, is_eligible)
from pipeline_utils import safe_float

CACHE = 'data/all_pitches_rs_cache.pkl'
LEADERBOARD = 'data/pitcher_leaderboard_rs.json'
N_SUB = 400
SEEDS = 8
MIN_FULL = 1400


def score_cells(cells):
    total, n_tot = 0.0, 0
    for pts in cells.values():
        if len(pts) < MIN_CELL:
            continue
        t = fit_targets(pts)
        for x, z in pts:
            total += min(math.hypot(x - tx, z - tz) for tx, tz in t)
            n_tot += 1
    return (total / n_tot, n_tot) if n_tot else (None, 0)


def cells_from(pitches):
    c = defaultdict(list)
    for p in pitches:
        c[(p[0], p[1], p[2])].append((p[3], p[4]))
    return c


def job(arg):
    name, pitches = arg
    full, _ = score_cells(cells_from(pitches))
    if full is None:
        return None
    per_seed = []
    for s in range(SEEDS):
        rng = random.Random(hash((name, N_SUB, s)) & 0xFFFFFFFF)
        m, _n = score_cells(cells_from(rng.sample(pitches, N_SUB)))
        if m is not None:
            per_seed.append(m - full)
    return (name, full, per_seed) if len(per_seed) == SEEDS else None


def main():
    rows = pd.read_pickle(CACHE)
    by_p = defaultdict(list)
    raw_by_p = defaultdict(list)
    for r in rows:
        if not is_eligible(r):
            continue
        by_p[r['Pitcher']].append((
            r['Pitch Type'], r['Bats'], count_group(r['Count']),
            safe_float(r['PlateX']) * 12.0, safe_float(r['PlateZ']) * 12.0))
        raw_by_p[r['Pitcher']].append(r['Pitch Type'])

    # ── (A) variance decomposition ──
    jobs = [(k, v) for k, v in by_p.items() if len(v) >= MIN_FULL]
    with Pool() as p:
        res = [r for r in p.map(job, jobs) if r]

    within_ss, within_df = 0.0, 0
    means = []
    for _name, _full, ds in res:
        m = sum(ds) / len(ds)
        means.append(m)
        within_ss += sum((d - m) ** 2 for d in ds)
        within_df += len(ds) - 1
    var_within = within_ss / within_df                    # per-single-seed
    gm = sum(means) / len(means)
    var_of_means = sum((m - gm) ** 2 for m in means) / (len(means) - 1)
    var_between = max(0.0, var_of_means - var_within / SEEDS)

    print('=' * 74)
    print(f'(A) VARIANCE OF THE N={N_SUB} DELTA  ({len(res)} pitchers x {SEEDS} seeds)')
    print('=' * 74)
    print(f'  grand mean delta        {gm:+.3f}"')
    print(f'  within-pitcher SD (seed noise)      {math.sqrt(var_within):.3f}"')
    print(f'  between-pitcher SD (true, denoised) {math.sqrt(var_between):.3f}"')
    tot = var_between + var_within
    print(f'  -> {100 * var_between / tot:.0f}% of the delta spread is real '
          f'pitcher-to-pitcher heterogeneity')
    ms = sorted(means)
    print(f'  denoised per-pitcher delta: p05 {ms[int(.05*len(ms))]:+.3f}"  '
          f'med {ms[len(ms)//2]:+.3f}"  p95 {ms[int(.95*len(ms))]:+.3f}"')

    # ── (B) live production coverage ──
    lb = json.load(open(LEADERBOARD))
    print('\n' + '=' * 74)
    print('(B) LIVE COVERAGE: share of eligible pitches actually scored')
    print('=' * 74)
    buckets = [(MIN_POOL, 600), (600, 1000), (1000, 1600), (1600, 10 ** 9)]
    lost_pt = defaultdict(int)
    seen_pt = defaultdict(int)
    for lo, hi in buckets:
        cov, n = [], 0
        for r in lb:
            if r.get('team') == 'ROC' or not r.get('commandPlusRaw'):
                continue
            nn = r.get('commandPlusN') or 0
            if not (lo <= nn < hi):
                continue
            elig = len(by_p.get(r.get('pitcher'), ()))
            if elig:
                cov.append(nn / elig)
                n += 1
        if cov:
            print(f'  {lo:>5}-{hi if hi < 10**9 else "inf":>5} scored pitches   '
                  f'n={n:>4}   median coverage {100 * sorted(cov)[len(cov)//2]:.1f}%   '
                  f'min {100 * min(cov):.1f}%')

    # which pitch types get dropped, for the thinnest arms
    thin = [r for r in lb if r.get('team') != 'ROC' and r.get('commandPlusRaw')
            and MIN_POOL <= (r.get('commandPlusN') or 0) < 600]
    for r in thin:
        pl = by_p.get(r.get('pitcher'))
        if not pl:
            continue
        cells = cells_from(pl)
        for ck, pts in cells.items():
            seen_pt[ck[0]] += len(pts)
            if len(pts) < MIN_CELL:
                lost_pt[ck[0]] += len(pts)
    print(f'\n  pitch types dropped by MIN_CELL={MIN_CELL} among the {len(thin)} '
          f'thinnest-sample pitchers:')
    for pt in sorted(seen_pt, key=lambda k: -lost_pt[k])[:8]:
        if seen_pt[pt]:
            print(f'    {pt:<4} {lost_pt[pt]:>6} of {seen_pt[pt]:>6} pitches dropped '
                  f'({100 * lost_pt[pt] / seen_pt[pt]:>5.1f}%)')


if __name__ == '__main__':
    main()
