"""commandplus_gate_measure.py — stabilization constant for Command+ display.

Measured at the RENDERED unit (pitcher-season aggregate) on the PRODUCTION
scorer (pipeline_commandplus), per the stabilization lesson: random
within-pitcher splits (the displayed number estimates a season aggregate),
targets fit per half independently (that is what production does on partial
data), rel(n) = corr of two independent n-pitch estimates, k = n(1-r)/r.

The practical question: is k below the pitch counts implied by the standard
IP qualification gate (SP ~1500+, RP ~450+ pitches)? If yes, the ordinary
render gate already covers Command+ and no special gate is needed.

Usage: python3 scripts/research/commandplus/commandplus_gate_measure.py
"""
import os, sys, math, pickle, random, statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import pipeline.commandplus as cmd

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
N_GRID = [100, 150, 200, 250, 300, 400]
SEEDS = range(5)
MIN_POOL = 40


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
    D = pickle.load(open(PKL, 'rb'))
    ep = {(p.get('Pitcher'), p.get('PTeam')) for p in D if p.get('Pitch Type') == 'EP'}
    byp = defaultdict(list)
    for p in D:
        if (p.get('_source', 'MLB') == 'MLB'
                and (p.get('Pitcher'), p.get('PTeam')) not in ep
                and cmd.is_eligible(p)):
            byp[(p.get('Pitcher'), p.get('Throws'))].append(p)
    print(f"{len(byp)} pitchers with eligible pitches", file=sys.stderr)

    per_seed = []
    for s in SEEDS:
        random.seed(s)
        ks = []
        for n in N_GRID:
            pool = {k: v for k, v in byp.items() if len(v) >= 2 * n}
            if len(pool) < MIN_POOL:
                continue
            g0, g1 = {}, {}
            for k, v in pool.items():
                smp = random.sample(v, 2 * n)
                g0[k] = smp[:n]; g1[k] = smp[n:]
            a = {k: r['raw_miss'] for k, r in cmd.score_misses(g0).items()}
            b = {k: r['raw_miss'] for k, r in cmd.score_misses(g1).items()}
            keys = [k for k in a if k in b]
            r = pearson([a[k] for k in keys], [b[k] for k in keys])
            if r and 0 < r < 1:
                ks.append(n * (1 - r) / r)
                print(f"  seed {s} n={n}: rel {r:.3f} -> k {n*(1-r)/r:.0f} "
                      f"({len(keys)} pitchers)", flush=True)
        if ks:
            per_seed.append(statistics.median(ks))

    k = statistics.median(per_seed)
    print()
    print(f"k per seed: {[round(x) for x in per_seed]}")
    print(f"MEASURED STABILIZE_N (median): {k:.0f}")
    print(f"IP-qualification pitch counts: SP ~1500+, RP ~450+ — "
          f"{'ordinary qual gate COVERS Command+' if k < 450 else 'special gate NEEDED'}")


if __name__ == '__main__':
    main()
