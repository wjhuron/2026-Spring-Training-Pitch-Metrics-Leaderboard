"""locplus_stabilize_celllevel.py — measure the Loc+ stabilization constant k
at the unit the LEADERBOARD RENDERS, for use as the pitch-type coloring gate.

Supersedes locplus_nprior_multiseed.py for gate purposes. That script measured
at the (Pitcher, Throws) level within a pitch GROUP, which is the right unit
for a Bayesian prior on a pitcher's group-level location skill. It is the wrong
unit for a display gate: the Arsenal row is one pitcher's ONE PITCH TYPE, so a
CH row must not be validated by a constant fitted to a pooled CH+FS+KN cell.

THREE DESIGN CHOICES, each load-bearing:

  1. CELLS ARE PER PITCH TYPE, not per group. Matches the rendered row. Pooling
     unlike pitches (a changeup with a splitter) inflates between-cell variance
     and so inflates apparent reliability — measured at 104 pooled vs 72 at the
     type level for the CH group, the most heterogeneous of the six.

  2. SURFACES ARE BUILT ONCE ON THE FULL SEASON, not per half. Independent
     per-half surfaces charge each cell for surface ESTIMATION noise, but that
     noise is a property of the model and is common across cells — it does not
     make one pitcher's rank less trustworthy than another's. Charging it
     inflates k badly (FF reads 103 instead of 81 under per-half surfaces).

  3. SPLITS ARE RANDOM WITHIN THE CELL, not chronological. The displayed number
     estimates the pitcher's SEASON AGGREGATE, so a chronological split would
     charge the estimate for genuine in-season drift that the season aggregate
     is supposed to contain. (A chronological split that takes the FIRST n of
     each half is worse still: both sides then share an early-season window and
     the shared conditions inflate the correlation.)

With two INDEPENDENT n-pitch estimates A and B of the same cell,
corr(A, B) == rho(n) directly — no Spearman-Brown correction, because both
sides are n-pitch estimates. Inverting the true-score model rho = n/(n+k)
gives k = n(1 - rho)/rho at each n; report the median across the n-grid, over
10 seeds.

The resulting k IS the coloring gate: gating at n = k means rho = 0.5, i.e.
half the variance in a colored cell is signal. See js/aggregator.js
QUAL.MIN_PITCH_LOCPLUS and pipeline_locplus.STABILIZE_N_PT (keep in sync).

Usage: python3 scripts/research/locplus/locplus_stabilize_celllevel.py
"""
import os, sys, math, pickle, statistics, random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline.locplus as lp

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393

N_GRID = [25, 40, 50, 60, 71, 85, 100, 117, 135, 150, 175, 200]
SEEDS = range(10)
MIN_CELLS = 40          # don't fit k from a thin cell pool
GROUPS = ['FF', 'SI', 'FC', 'SL', 'CU', 'CH']


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


def main():
    print("loading cache...", file=sys.stderr)
    D = pickle.load(open(PKL, 'rb'))
    base = [p for p in D if lp.is_eligible_baseline(p)]

    print("building full-season surfaces (once)...", file=sys.stderr)
    S = lp.build_surfaces(base, LG, SCALE)

    cells = defaultdict(list)
    for p in base:
        v = lp.score_pitch(p, S)
        if v is not None:
            cells[(p.get('Pitcher'), p.get('Throws'), p.get('Pitch Type'))].append(v)
    print(f"cells (pitcher x throws x pitch type): {len(cells)}", file=sys.stderr)

    print()
    print(f"{'grp':>5s} {'shipped':>8s} {'k_cell':>7s} {'min':>5s} {'max':>5s} "
          f"{'cells@71':>9s}   (10 seeds, median of per-seed medians)")
    print('-' * 72)

    out = {}
    for G in GROUPS:
        per_seed, ncell = [], 0
        for s in SEEDS:
            random.seed(s)
            ks = []
            for n in N_GRID:
                A, B = [], []
                for key, v in cells.items():
                    if lp.GROUP.get(key[2]) != G or len(v) < 2 * n:
                        continue
                    smp = random.sample(v, 2 * n)
                    A.append(sum(smp[:n]) / n)
                    B.append(sum(smp[n:]) / n)
                if len(A) < MIN_CELLS:
                    continue
                if n == 71 and s == 0:
                    ncell = len(A)
                r = pearson(A, B)
                if r and r > 0:
                    ks.append(n * (1 - r) / r)
            if ks:
                per_seed.append(statistics.median(ks))
        if not per_seed:
            print(f"{G:>5s}    insufficient data")
            continue
        k = statistics.median(per_seed)
        out[G] = k
        print(f"{G:>5s} {lp.STABILIZE_N_PT.get(G, 0):>8d} {k:>7.0f} "
              f"{min(per_seed):>5.0f} {max(per_seed):>5.0f} {ncell:>9d}")

    print()
    print("Gate at n = k  =>  rho = 0.5 (half the variance in a colored cell is signal).")
    print("Categories take the stiffest member gate:")
    cat = {'Hard': ['FF', 'SI'], 'Breaking': ['FC', 'SL', 'CU'], 'Offspeed': ['CH']}
    for name, members in cat.items():
        vals = [out[m] for m in members if m in out]
        if vals:
            print(f"  {name:>9s}: {max(vals):.0f}")


if __name__ == '__main__':
    main()
