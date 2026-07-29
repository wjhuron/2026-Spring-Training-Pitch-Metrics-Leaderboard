"""locplus_gates_multiseason.py — do the shipped Loc+ gate constants transfer?

The pitch-type coloring gates shipped in edd8074 (FF 81 / SI 96 / FC 122 /
SL 70 / CU 93 / CH 72) were measured on the 2026 season-to-date cache — the
same PARTIAL season that produced a bandwidth optimum which then lost 0/5
across 2021-2025. Different mechanism, so not automatically wrong: k is the
split-half r=0.5 crossing measured at FIXED n, which is far less exposed to
total season length than a bandwidth optimum is. But mid-season surfaces are
noisier, that noise lands in every pitcher's score, and it inflates k. Worth
measuring rather than assuming.

Same design as scripts/locplus_stabilize_celllevel.py, applied per season:
cells are per PITCH TYPE (the rendered unit), surfaces built ONCE on that
season, random within-cell splits, 10 seeds, k = median of n(1-rho)/rho across
the n-grid. Seasons are never pooled.

Reads the raw Statcast caches through the adapter in
locplus_constants_multiseason.py (which handles the delta_run_exp sign flip
and pandas NA sanitizing).

Usage: python3 scripts/locplus_gates_multiseason.py
"""
import os, sys, math, statistics, random, gc
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import pipeline_locplus as lp
from locplus_constants_multiseason import adapt

LG, SCALE = 0.3169, 1.2393
N_GRID = [25, 40, 50, 60, 71, 85, 100, 117, 135, 150, 175, 200]
SEEDS = range(10)
MIN_CELLS = 40
GROUPS = ['FF', 'SI', 'FC', 'SL', 'CU', 'CH']

SEASONS = [(2021, 'data/_statcast2021_cache.pkl'),
           (2022, 'data/_statcast2022_cache.pkl'),
           (2023, 'data/_statcast2023_cache.pkl'),
           (2024, 'data/_statcast2024_cache.pkl'),
           (2025, 'data/_statcast2025_full_cache.pkl')]


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


def measure(pitches):
    """k per group for one season, surfaces built in-season."""
    base = [p for p in pitches if lp.is_eligible_baseline(p)]
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
        out[G] = statistics.median(per_seed) if per_seed else None
    return out


def main():
    shipped = dict(lp.STABILIZE_N_PT)
    table = {}
    for yr, path in SEASONS:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            print(f"{yr}: cache missing, skipped", file=sys.stderr)
            continue
        print(f"adapting {yr}...", file=sys.stderr)
        pitches = adapt(p)
        print(f"  {yr}: {len(pitches)} pitches, measuring...", file=sys.stderr)
        table[yr] = measure(pitches)
        print(f"  {yr}: " + " ".join(
            f"{G}={table[yr][G]:.0f}" if table[yr][G] else f"{G}=-"
            for G in GROUPS), flush=True)
        del pitches
        gc.collect()

    print()
    print(f"{'group':>6s} {'shipped':>8s} " + "".join(f"{yr:>8d}" for yr in sorted(table))
          + f"{'median':>9s} {'ratio':>7s}")
    print('-' * (15 + 8 * len(table) + 16))
    for G in GROUPS:
        vals = [table[yr][G] for yr in sorted(table) if table[yr].get(G)]
        med = statistics.median(vals) if vals else None
        cells = "".join(f"{table[yr][G]:>8.0f}" if table[yr].get(G) else f"{'-':>8s}"
                        for yr in sorted(table))
        ratio = med / shipped[G] if med and shipped.get(G) else None
        print(f"{G:>6s} {shipped[G]:>8d} {cells}"
              + (f"{med:>9.0f} {ratio:>7.2f}" if med else f"{'-':>9s} {'-':>7s}"))
    print()
    print("ratio = historical median / shipped (measured on partial-season 2026).")
    print("Ratios near 1.0 mean the gates transfer. Ratios well below 1.0 mean the")
    print("2026 values are inflated by mid-season surface noise and the gates are")
    print("stricter than they need to be.")


if __name__ == '__main__':
    main()
