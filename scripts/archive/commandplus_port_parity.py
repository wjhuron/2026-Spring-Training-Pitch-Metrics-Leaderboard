"""commandplus_port_parity.py — acceptance test for the pure-Python port.

The port (pipeline_commandplus) must reproduce the VALIDATED research engine
(scripts/commandplus_v1, sklearn GMM) on real data before it can be wired
into the pipeline. The two differ by construction in one place: the port's
EM uses deterministic farthest-point seeding, sklearn uses random restarts —
so cell-level target placement can differ where the likelihood surface is
multi-modal. What must match is what ships:

  1. per-pitcher mean miss: correlation ~1 and near-zero mean absolute gap
  2. the pitcher RANKING (display is rank-driven): Spearman-style agreement
  3. distribution anchors (pool mean, sigma) within noise
  4. runtime sane for a pipeline step

Sample: full 2026 MLB. Verdict rule pre-registered: r >= 0.995 and mean
|gap| <= 0.15in -> port ACCEPTED; else investigate cells with the largest
disagreement before any wiring.

Usage: python3 scripts/commandplus_port_parity.py
"""
import os, sys, math, pickle, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import pipeline_commandplus as port
from commandplus_v1 import score_pitches_multi, aggregate

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
MIN_N = 300


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    return pearson(ranks(xs), ranks(ys))


def main():
    print("loading 2026 MLB...", file=sys.stderr)
    D = pickle.load(open(PKL, 'rb'))
    ep = {(p.get('Pitcher'), p.get('PTeam')) for p in D if p.get('Pitch Type') == 'EP'}
    mlb = [p for p in D if p.get('_source', 'MLB') == 'MLB'
           and (p.get('Pitcher'), p.get('PTeam')) not in ep]

    # research engine (sklearn), sep=0 = the validated no-guard config
    t0 = time.time()
    multi = score_pitches_multi(mlb, [0.0])
    ref = {k: v[0] for k, v in aggregate(multi[0.0][0], MIN_N).items()}
    t_ref = time.time() - t0
    print(f"research engine: {len(ref)} pitchers ({t_ref:.0f}s)", file=sys.stderr)

    # port (pure python), same grouping key
    byp = defaultdict(list)
    for p in mlb:
        byp[(p.get('Pitcher'), p.get('Throws'))].append(p)
    t0 = time.time()
    res = port.score_misses(byp)
    got = {k: v['raw_miss'] for k, v in res.items() if v['n_pitches'] >= MIN_N}
    t_port = time.time() - t0
    print(f"port: {len(got)} pitchers ({t_port:.0f}s)", file=sys.stderr)

    keys = [k for k in ref if k in got]
    a = [ref[k] for k in keys]; b = [got[k] for k in keys]
    gaps = [abs(x - y) for x, y in zip(a, b)]
    r = pearson(a, b)
    rho = spearman(a, b)
    print()
    print(f"pitchers compared: {len(keys)}")
    print(f"pearson r        : {r:.5f}")
    print(f"spearman rho     : {rho:.5f}")
    print(f"mean |gap|       : {sum(gaps)/len(gaps):.3f} in")
    print(f"max  |gap|       : {max(gaps):.3f} in "
          f"({keys[gaps.index(max(gaps))][0]})")
    print(f"pool mean        : ref {sum(a)/len(a):.3f}  port {sum(b)/len(b):.3f}")
    sda = math.sqrt(sum((x - sum(a)/len(a))**2 for x in a)/len(a))
    sdb = math.sqrt(sum((x - sum(b)/len(b))**2 for x in b)/len(b))
    print(f"pool sd          : ref {sda:.3f}  port {sdb:.3f}")
    print(f"runtime          : research {t_ref:.0f}s, port {t_port:.0f}s")
    print()
    ok = r >= 0.995 and sum(gaps)/len(gaps) <= 0.15
    print("VERDICT:", "ACCEPTED — port reproduces the validated engine"
          if ok else "NOT ACCEPTED — investigate the largest-gap cells")


if __name__ == '__main__':
    main()
