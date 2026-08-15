"""locplus_bandwidth_holdout.py — does the bandwidth optimum survive out of sample?

The 2-D sweep picked x=2.5in / z=0.28 over the shipped 4.5in / 0.22z on the
FULL season. That is the sample the choice was made on, so it cannot also be
the evidence the choice generalizes — the count-demeaning result looked good
in-sample too, and failed out of sample.

DESIGN. Split the season chronologically into A (early) and B (late), then
evaluate every candidate independently inside each:

    predictive  A:  surfaces + score from A1, actual xRV from A2
    predictive  B:  surfaces + score from B1, actual xRV from B2
    stuff-leak  A:  surfaces + score on all of A, vs A whiff% / A FF velo
    stuff-leak  B:  surfaces + score on all of B, vs B whiff% / B FF velo

A and B share no games, so B is a genuine holdout for anything chosen on A.
The question is not "what is the argmax" — it is whether the RANKING of
shipped vs proposed is the same in both halves. A bandwidth that is better
only in the half it was picked from is a fit to that half's noise.

Thresholds are lower than the full-season harness (each half has ~60 dates,
each quarter ~30) or the pitcher pool empties out; pool sizes are printed so
the noise level is visible rather than assumed.

Usage: python3 scripts/locplus_bandwidth_holdout.py
"""
import os, sys, math, pickle
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline.locplus as lp
from pipeline.sdplus import make_rv_xrv

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393

MIN_SCORE = 150     # pitches to score a pitcher within a quarter
MIN_ACTUAL = 150    # pitches to measure actual xRV within a quarter
MIN_LEAK = 200      # pitches to enter the stuff-leak pool within a half

CANDIDATES = [
    ('shipped',  4.5, 0.22),
    ('proposed', 2.5, 0.28),
    ('narrow-x', 2.0, 0.28),
    ('mid-x',    3.0, 0.28),
    ('x-only',   2.5, 0.22),
    ('wide-z',   2.5, 0.34),
]


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


def set_bw(x, z):
    lp.PHYS_X_IN = x; lp.PHYS_Z_FRAC = z
    lp._KX = lp._k1d(x / 2.0); lp._KZ = lp._k1d(z / lp.BIN_Z)


def by_pitcher(pitches):
    d = defaultdict(list)
    for p in pitches:
        d[(p.get('Pitcher'), p.get('Throws'))].append(p)
    return d


def scores(byp, S, min_n):
    out = {}
    for k, ps in byp.items():
        v = [s for s in (lp.score_pitch(p, S) for p in ps) if s is not None]
        if len(v) >= min_n:
            out[k] = sum(v) / len(v)
    return out


def main():
    print("loading cache...", file=sys.stderr)
    D = pickle.load(open(PKL, 'rb'))
    base = [p for p in D if lp.is_eligible_baseline(p)]
    dates = sorted({p.get('Game Date') for p in base if p.get('Game Date')})
    print(f"{len(base)} pitches, {dates[0]} .. {dates[-1]} ({len(dates)} dates)",
          file=sys.stderr)

    # chronological quarters -> halves A = Q1+Q2, B = Q3+Q4
    q = len(dates) // 4
    cuts = [dates[q], dates[2 * q], dates[3 * q]]
    seg = {'A1': [], 'A2': [], 'B1': [], 'B2': []}
    for p in base:
        d = p.get('Game Date')
        if d is None:
            continue
        if d < cuts[0]:
            seg['A1'].append(p)
        elif d < cuts[1]:
            seg['A2'].append(p)
        elif d < cuts[2]:
            seg['B1'].append(p)
        else:
            seg['B2'].append(p)
    A = seg['A1'] + seg['A2']
    B = seg['B1'] + seg['B2']
    print(f"A = {dates[0]}..{cuts[1]} ({len(A)} pitches), "
          f"B = {cuts[1]}..{dates[-1]} ({len(B)} pitches)", file=sys.stderr)

    rv_fn = make_rv_xrv(LG, SCALE)
    ctx = {}
    for half, first, second, whole in (('A', 'A1', 'A2', A), ('B', 'B1', 'B2', B)):
        byp_first = by_pitcher(seg[first])
        byp_whole = by_pitcher(whole)
        actual = {}
        for k, ps in by_pitcher(seg[second]).items():
            v = [x for x in (rv_fn(p) for p in ps) if x is not None]
            if len(v) >= MIN_ACTUAL:
                actual[k] = sum(v) / len(v)
        whiff, ffv = {}, {}
        for k, ps in byp_whole.items():
            sw = [p for p in ps if p.get('Description') in lp.SWING_DESC]
            wh = [p for p in sw if p.get('Description') == 'Swinging Strike']
            if len(sw) >= 80:
                whiff[k] = len(wh) / len(sw)
            v = [f for f in (lp.safe_float(p.get('Velocity')) for p in ps
                             if p.get('Pitch Type') == 'FF') if f is not None]
            if len(v) >= 40:
                ffv[k] = sum(v) / len(v)
        ctx[half] = {'first': seg[first], 'byp_first': byp_first, 'actual': actual,
                     'whole': whole, 'byp_whole': byp_whole,
                     'whiff': whiff, 'ffv': ffv}

    print()
    print(f"{'candidate':>10s} {'x':>5s} {'z':>5s} | "
          f"{'predA':>6s} {'leakA_w':>8s} {'leakA_v':>8s} | "
          f"{'predB':>6s} {'leakB_w':>8s} {'leakB_v':>8s}")
    print('-' * 78)
    out = {}
    for name, x, z in CANDIDATES:
        set_bw(x, z)
        row = []
        ns = {}
        for half in ('A', 'B'):
            c = ctx[half]
            S1 = lp.build_surfaces(c['first'], LG, SCALE)
            sc = scores(c['byp_first'], S1, MIN_SCORE)
            kp = [k for k in sc if k in c['actual']]
            pred = pearson([sc[k] for k in kp], [c['actual'][k] for k in kp])
            Sw = lp.build_surfaces(c['whole'], LG, SCALE)
            full = scores(c['byp_whole'], Sw, MIN_LEAK)
            kw = [k for k in full if k in c['whiff']]
            kv = [k for k in full if k in c['ffv']]
            rw = pearson([full[k] for k in kw], [c['whiff'][k] for k in kw])
            rv_ = pearson([full[k] for k in kv], [c['ffv'][k] for k in kv])
            row += [pred, abs(rw) if rw else None, abs(rv_) if rv_ else None]
            ns[half] = (len(kp), len(kw))
        out[name] = row
        f = lambda v: f"{v:.3f}" if v is not None else "  n/a"
        print(f"{name:>10s} {x:>5.2f} {z:>5.2f} | {f(row[0]):>6s} {f(row[1]):>8s} "
              f"{f(row[2]):>8s} | {f(row[3]):>6s} {f(row[4]):>8s} {f(row[5]):>8s}",
              flush=True)
        print(f"   pools: A pred n={ns['A'][0]}, leak n={ns['A'][1]}; "
              f"B pred n={ns['B'][0]}, leak n={ns['B'][1]}", file=sys.stderr)

    # Verdict compares every candidate against 'shipped'; tolerant of an edited
    # CANDIDATES list so the grid can be re-pointed without breaking the report.
    if 'shipped' not in out or len(out) < 2:
        return
    s = out['shipped']
    p_name = next(n for n in out if n != 'shipped')
    p = out[p_name]
    print()
    print(f"VERDICT — {p_name} vs shipped, per half:")
    for i, (half, lbl) in enumerate([('A', 'predictive'), ('A', 'whiff leak'),
                                     ('A', 'velo leak'), ('B', 'predictive'),
                                     ('B', 'whiff leak'), ('B', 'velo leak')]):
        if s[i] is None or p[i] is None:
            continue
        better = (p[i] > s[i]) if 'predictive' in lbl else (p[i] < s[i])
        print(f"  half {half} {lbl:>11s}: {s[i]:.3f} -> {p[i]:.3f}  "
              f"{'BETTER' if better else 'worse'}")
    print()
    print("A bandwidth that wins in BOTH halves generalizes. One that wins only")
    print("in A is a fit to A's noise.")


if __name__ == '__main__':
    main()
