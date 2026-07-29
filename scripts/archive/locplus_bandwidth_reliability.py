"""locplus_bandwidth_reliability.py — the objective the bandwidth sweep was blind to.

The holdout harness scores predictive validity and stuff-independence. Neither
punishes a NOISY surface directly, and narrowing the horizontal bandwidth makes
surfaces noisier. So "prediction and velo leak keep improving as x narrows" is
not sufficient evidence that narrow x is right — the objective set has to be
able to see the failure mode that narrowing causes.

Split-half reliability is that objective. Note the asymmetry:

  - In the WIDE direction reliability is GAMEABLE and must not be maximized.
    Smoothing toward a constant gives every pitcher the same flat surface,
    which is perfectly reliable and carries no information. This is why the
    earlier per-group pass, which ranked by reliability, picked the widest
    bandwidth on its grid for four of six groups and had to be discarded.

  - In the NARROW direction it is a genuine noise penalty. There is no way to
    fake reliability by under-smoothing, so a falling rel_r as x shrinks is
    real information loss, not an artifact.

Read it as a FLOOR, not a target: find where reliability starts to fall off a
cliff, and stay on the safe side of it, while prediction and stuff-leak choose
among the survivors.

Measured independently inside each chronological half (odd/even game dates
within that half, surfaces rebuilt per parity group), so the same held-out
structure as locplus_bandwidth_holdout.py applies.

Usage: python3 scripts/locplus_bandwidth_reliability.py
"""
import os, sys, math, pickle
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline_locplus as lp

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393
MIN_REL = 100      # pitches per parity group within a half

CANDIDATES = [4.5, 3.75, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
Z_FIXED = 0.22


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


def by_pitcher(pitches):
    d = defaultdict(list)
    for p in pitches:
        d[(p.get('Pitcher'), p.get('Throws'))].append(p)
    return d


def main():
    print("loading cache...", file=sys.stderr)
    D = pickle.load(open(PKL, 'rb'))
    base = [p for p in D if lp.is_eligible_baseline(p)]
    dates = sorted({p.get('Game Date') for p in base if p.get('Game Date')})
    mid = dates[len(dates) // 2]

    # halves, then odd/even parity WITHIN each half
    halves = {'A': [p for p in base if p.get('Game Date') < mid],
              'B': [p for p in base if p.get('Game Date') >= mid]}
    ctx = {}
    for h, ps in halves.items():
        hd = sorted({p.get('Game Date') for p in ps})
        par = {d: i % 2 for i, d in enumerate(hd)}
        g0 = [p for p in ps if par.get(p.get('Game Date')) == 0]
        g1 = [p for p in ps if par.get(p.get('Game Date')) == 1]
        ctx[h] = (g0, g1, by_pitcher(g0), by_pitcher(g1))
        print(f"half {h}: {len(ps)} pitches, parity {len(g0)}/{len(g1)}", file=sys.stderr)

    print()
    print(f"split-half reliability at z={Z_FIXED}, measured inside each half")
    print(f"{'x_in':>6s} {'rel_A':>7s} {'nA':>5s} {'rel_B':>7s} {'nB':>5s} {'mean':>7s}")
    print('-' * 42)
    prev = None
    for x in CANDIDATES:
        lp.PHYS_X_IN = x; lp.PHYS_Z_FRAC = Z_FIXED
        lp._KX = lp._k1d(x / 2.0); lp._KZ = lp._k1d(Z_FIXED / lp.BIN_Z)
        rels, ns = [], []
        for h in ('A', 'B'):
            g0, g1, b0, b1 = ctx[h]
            S0 = lp.build_surfaces(g0, LG, SCALE)
            S1 = lp.build_surfaces(g1, LG, SCALE)
            def sc(byp, S):
                out = {}
                for k, ps in byp.items():
                    v = [s for s in (lp.score_pitch(p, S) for p in ps) if s is not None]
                    if len(v) >= MIN_REL:
                        out[k] = sum(v) / len(v)
                return out
            a, b = sc(b0, S0), sc(b1, S1)
            keys = [k for k in a if k in b]
            rels.append(pearson([a[k] for k in keys], [b[k] for k in keys]))
            ns.append(len(keys))
        m = sum(rels) / len(rels)
        delta = f"{m - prev:+.3f}" if prev is not None else "   -"
        mark = '  <- shipped' if abs(x - 4.5) < 1e-9 else ''
        print(f"{x:>6.2f} {rels[0]:>7.3f} {ns[0]:>5d} {rels[1]:>7.3f} {ns[1]:>5d} "
              f"{m:>7.3f}  step {delta}{mark}", flush=True)
        prev = m

    print()
    print("Read as a FLOOR, not a target: reliability rises monotonically with")
    print("bandwidth and is maximized by the degenerate flat surface, so the")
    print("question is only where it starts falling off a cliff on the narrow side.")


if __name__ == '__main__':
    main()
