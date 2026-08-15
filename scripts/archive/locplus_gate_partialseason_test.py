"""locplus_gate_partialseason_test.py — break the gate-constant confound NOW.

The shipped Loc+ gates (FF 81 / SI 96 / FC 122 / SL 70 / CU 93 / CH 72) were
measured on partial-season 2026. Historical medians run ~22% lower, but that
gap had two inseparable explanations:

  (a) partial season -> noisier surfaces -> k genuinely inflated in 2026
  (b) historical caches use raw Statcast pitch_type while 2026 is retagged
      -> different cell composition -> historical k deflated

The season-end re-measure resolves both at once, but (a) alone is testable
TODAY: measure k inside 2025 twice — once on the season through the same
calendar date 2026 currently reaches (a partial-season replica), once on the
full season — same tags, same cells, same pipeline. The partial-vs-full gap
IS the partial-season effect, isolated.

Reading:
  k(partial) >> k(full)  ->  (a) is real; the 2026 gates are inflated and will
                             loosen at season end, as predicted.
  k(partial) ~= k(full)  ->  (a) is negligible; the 2026-vs-history gap is
                             tags or genuine season variation, and the shipped
                             gates should NOT be expected to drop much.

Direction is not obvious a priori: fewer pitches make surfaces noisier, but a
surface error at a location hits both halves of a random split identically for
pitchers who live there (shared error INFLATES reliability, deflating k),
while error interacting with within-pitcher location spread adds unshared
variance (inflating k). Hence measurement, not argument.

Same design as scripts/research/locplus/locplus_stabilize_celllevel.py: cells per pitch type,
surfaces built once per slice, random within-cell splits, 10 seeds.

Usage: python3 scripts/locplus_gate_partialseason_test.py
"""
import os, sys, math, pickle, statistics, random, gc
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import pipeline.locplus as lp
from locplus_constants_multiseason import adapt

LG, SCALE = 0.3169, 1.2393
N_GRID = [25, 40, 50, 60, 71, 85, 100, 117, 135, 150, 175, 200]
SEEDS = range(10)
MIN_CELLS = 40
GROUPS = ['FF', 'SI', 'FC', 'SL', 'CU', 'CH']
# 2026's cache currently reaches 7/24; replicate that calendar coverage in 2025.
PARTIAL_CUTOFF = '2025-07-24'
CACHE = 'data/_statcast2025_full_cache.pkl'
SHIPPED = {'FF': 81, 'SI': 96, 'FC': 122, 'SL': 70, 'CU': 93, 'CH': 72}


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


def measure(base, label):
    S = lp.build_surfaces(base, LG, SCALE)
    cells = defaultdict(list)
    for p in base:
        v = lp.score_pitch(p, S)
        if v is not None:
            cells[(p['Pitcher'], p['Throws'], p['Pitch Type'])].append(v)
    out = {}
    for G in GROUPS:
        pool = [v for key, v in cells.items() if lp.GROUP.get(key[2]) == G]
        per_seed = []
        for s in SEEDS:
            random.seed(s)
            ks = []
            for n in N_GRID:
                A, B = [], []
                for v in pool:
                    if len(v) < 2 * n:
                        continue
                    smp = random.sample(v, 2 * n)
                    A.append(sum(smp[:n]) / n); B.append(sum(smp[n:]) / n)
                if len(A) < MIN_CELLS:
                    continue
                r = pearson(A, B)
                if r and r > 0:
                    ks.append(n * (1 - r) / r)
            if ks:
                per_seed.append(statistics.median(ks))
        out[G] = ((statistics.median(per_seed), min(per_seed), max(per_seed))
                  if per_seed else None)
    print(f"  measured {label}: {len(base)} pitches, {len(cells)} cells",
          file=sys.stderr)
    return out


def main():
    print("adapting 2025...", file=sys.stderr)
    pitches = adapt(os.path.join(ROOT, CACHE))
    base = [p for p in pitches if lp.is_eligible_baseline(p)]
    del pitches
    gc.collect()
    dates = sorted({p['Game Date'] for p in base})
    partial = [p for p in base if p['Game Date'] <= PARTIAL_CUTOFF]
    pd_ = sorted({p['Game Date'] for p in partial})
    print(f"full: {dates[0]}..{dates[-1]} ({len(dates)} dates)  "
          f"partial: ..{pd_[-1]} ({len(pd_)} dates, "
          f"{100*len(partial)/len(base):.0f}% of pitches)", file=sys.stderr)

    part = measure(partial, 'partial')
    full = measure(base, 'full')

    print()
    print(f"{'grp':>5s} {'k_partial':>16s} {'k_full':>16s} {'ratio':>6s} "
          f"{'2026 shipped':>13s}")
    print('-' * 62)
    ratios = []
    for G in GROUPS:
        p_, f_ = part.get(G), full.get(G)
        if not p_ or not f_:
            print(f"{G:>5s}   insufficient data")
            continue
        r = p_[0] / f_[0]
        ratios.append(r)
        print(f"{G:>5s} {p_[0]:>7.0f} [{p_[1]:.0f}-{p_[2]:.0f}]"
              f" {f_[0]:>8.0f} [{f_[1]:.0f}-{f_[2]:.0f}] {r:>6.2f} {SHIPPED[G]:>13d}")
    if ratios:
        print()
        print(f"median partial/full ratio: {statistics.median(ratios):.2f}")
        print()
        print("Ratio >> 1: the partial-season effect is real — expect the 2026")
        print("gates to loosen by roughly this factor at season end.")
        print("Ratio ~= 1: partial-season noise is NOT inflating k; the shipped")
        print("gates reflect the retagged data and should hold as-is.")


if __name__ == '__main__':
    main()
