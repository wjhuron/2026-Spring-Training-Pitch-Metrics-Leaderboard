"""locplus_joint_regularizer_search.py — joint search over Loc+'s TWO regularizers.

Kernel smoothing and the per-surface shrinkage pseudo-counts are two dials on
the same thing: how much a sparse cell borrows strength from elsewhere. The
kernel borrows from NEIGHBORING CELLS; the shrinkage prior borrows from the
GROUP MEAN. Every sweep before this one moved the kernel while holding the
shrinkage constants at values that were tuned when the kernel was doing 4.5in
of horizontal work — so "reliability falls as x narrows" may not be an
unavoidable cost of narrowing at all. It may just be the model running
under-regularized by exactly the amount the kernel used to contribute.

If so, more shrinkage should buy that reliability back while prediction and
stuff-independence stay at their plateau. That is the hypothesis.

GRID: x (horizontal kernel bandwidth) x m (multiplier on ALL shrinkage
pseudo-counts: K_WHIFF, K_FOUL, K_XWCON, K_SWING_COLL, K_SWING_COUNT, K_CS).
The vertical bandwidth stays at 0.22, which earlier holdout runs bracketed
from both sides.

OBJECTIVES, and how to read them (all measured inside each chronological half,
so half B is a genuine holdout for anything chosen on half A):
  pred    predictive validity, first-quarter score vs next-quarter actual xRV.
          PRIMARY. Cannot be gamed by either over- or under-regularizing.
  velo    |r| vs FF velocity. Loc+ is a location metric; correlation with
          velocity is contamination. Cannot be gamed by smoothing to a constant.
  whiff   |r| vs whiff%. Same idea, but this one crosses ZERO as regularization
          changes, so |r| has a spurious minimum at the crossing and the
          crossing MOVES between samples. Treat as weak evidence only.
  rel     split-half reliability. NOT an accuracy measure — precision. Gameable
          in the over-regularized direction (a flat surface is perfectly
          reliable and carries no information), honest in the under-regularized
          direction. Read as a floor, and as the thing more shrinkage should
          recover.

Usage: python3 scripts/locplus_joint_regularizer_search.py
"""
import os, sys, math, pickle, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline_locplus as lp
from pipeline_sdplus import make_rv_xrv

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393

X_GRID = [4.5, 2.5, 1.5, 1.0]
M_GRID = [1, 2, 4, 8]
Z_FIXED = 0.22

MIN_SCORE, MIN_ACTUAL, MIN_LEAK, MIN_REL = 150, 150, 200, 100

BASE_K = dict(K_WHIFF=lp.K_WHIFF, K_FOUL=lp.K_FOUL, K_XWCON=lp.K_XWCON,
              K_SWING_COLL=lp.K_SWING_COLL, K_SWING_COUNT=lp.K_SWING_COUNT,
              K_CS=lp.K_CS)


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


def apply_config(x, m):
    lp.PHYS_X_IN = x; lp.PHYS_Z_FRAC = Z_FIXED
    lp._KX = lp._k1d(x / 2.0); lp._KZ = lp._k1d(Z_FIXED / lp.BIN_Z)
    for k, v in BASE_K.items():
        setattr(lp, k, v * m)


def score_map(byp, S, min_n):
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
    print(f"{len(base)} pitches, {dates[0]}..{dates[-1]}", file=sys.stderr)

    q = len(dates) // 4
    cuts = [dates[q], dates[2 * q], dates[3 * q]]
    seg = defaultdict(list)
    for p in base:
        d = p.get('Game Date')
        if d is None:
            continue
        seg['A1' if d < cuts[0] else 'A2' if d < cuts[1]
            else 'B1' if d < cuts[2] else 'B2'].append(p)

    rv_fn = make_rv_xrv(LG, SCALE)
    ctx = {}
    for half, f, s in (('A', 'A1', 'A2'), ('B', 'B1', 'B2')):
        whole = seg[f] + seg[s]
        hd = sorted({p.get('Game Date') for p in whole})
        par = {d: i % 2 for i, d in enumerate(hd)}
        g0 = [p for p in whole if par.get(p.get('Game Date')) == 0]
        g1 = [p for p in whole if par.get(p.get('Game Date')) == 1]
        actual = {}
        for k, ps in by_pitcher(seg[s]).items():
            v = [x for x in (rv_fn(p) for p in ps) if x is not None]
            if len(v) >= MIN_ACTUAL:
                actual[k] = sum(v) / len(v)
        whiff, ffv = {}, {}
        for k, ps in by_pitcher(whole).items():
            sw = [p for p in ps if p.get('Description') in lp.SWING_DESC]
            wh = [p for p in sw if p.get('Description') == 'Swinging Strike']
            if len(sw) >= 80:
                whiff[k] = len(wh) / len(sw)
            v = [f2 for f2 in (lp.safe_float(p.get('Velocity')) for p in ps
                               if p.get('Pitch Type') == 'FF') if f2 is not None]
            if len(v) >= 40:
                ffv[k] = sum(v) / len(v)
        ctx[half] = {'first': seg[f], 'byp_first': by_pitcher(seg[f]),
                     'actual': actual, 'whole': whole,
                     'byp_whole': by_pitcher(whole), 'whiff': whiff, 'ffv': ffv,
                     'g0': g0, 'g1': g1, 'b0': by_pitcher(g0), 'b1': by_pitcher(g1)}

    print()
    print(f"{'x':>5s} {'m':>3s} | {'predA':>6s} {'predB':>6s} {'pred':>6s} | "
          f"{'veloA':>6s} {'veloB':>6s} {'velo':>6s} | {'whfA':>5s} {'whfB':>5s} | "
          f"{'relA':>5s} {'relB':>5s} {'rel':>5s}")
    print('-' * 92)
    results = {}
    for x in X_GRID:
        for m in M_GRID:
            t0 = time.time()
            apply_config(x, m)
            o = {}
            for half in ('A', 'B'):
                c = ctx[half]
                S1 = lp.build_surfaces(c['first'], LG, SCALE)
                sc = score_map(c['byp_first'], S1, MIN_SCORE)
                kp = [k for k in sc if k in c['actual']]
                o['pred' + half] = pearson([sc[k] for k in kp],
                                           [c['actual'][k] for k in kp])
                Sw = lp.build_surfaces(c['whole'], LG, SCALE)
                full = score_map(c['byp_whole'], Sw, MIN_LEAK)
                kv = [k for k in full if k in c['ffv']]
                kw = [k for k in full if k in c['whiff']]
                o['velo' + half] = abs(pearson([full[k] for k in kv],
                                               [c['ffv'][k] for k in kv]))
                o['whf' + half] = abs(pearson([full[k] for k in kw],
                                              [c['whiff'][k] for k in kw]))
                S0 = lp.build_surfaces(c['g0'], LG, SCALE)
                S1b = lp.build_surfaces(c['g1'], LG, SCALE)
                a0, a1 = score_map(c['b0'], S0, MIN_REL), score_map(c['b1'], S1b, MIN_REL)
                kk = [k for k in a0 if k in a1]
                o['rel' + half] = pearson([a0[k] for k in kk], [a1[k] for k in kk])
            o['pred'] = (o['predA'] + o['predB']) / 2
            o['velo'] = (o['veloA'] + o['veloB']) / 2
            o['rel'] = (o['relA'] + o['relB']) / 2
            results[(x, m)] = o
            mark = '  <- shipped' if (x == 4.5 and m == 1) else ''
            print(f"{x:>5.2f} {m:>3d} | {o['predA']:>6.3f} {o['predB']:>6.3f} "
                  f"{o['pred']:>6.3f} | {o['veloA']:>6.3f} {o['veloB']:>6.3f} "
                  f"{o['velo']:>6.3f} | {o['whfA']:>5.3f} {o['whfB']:>5.3f} | "
                  f"{o['relA']:>5.3f} {o['relB']:>5.3f} {o['rel']:>5.3f}{mark}",
                  flush=True)
            print(f"   ({time.time()-t0:.0f}s)", file=sys.stderr)

    ship = results[(4.5, 1)]
    print()
    print(f"shipped (x=4.5, m=1): pred {ship['pred']:.3f}, velo {ship['velo']:.3f}, "
          f"rel {ship['rel']:.3f}")
    print()
    # Accuracy first (pred, then velo), with reliability as a tie-breaker among
    # configs whose accuracy is within noise of the best.
    best_pred = max(results.values(), key=lambda o: o['pred'])['pred']
    near = {k: o for k, o in results.items() if o['pred'] >= best_pred - 0.003}
    print(f"configs within 0.003 of best pred ({best_pred:.3f}), "
          f"ranked by reliability:")
    for k, o in sorted(near.items(), key=lambda kv: -kv[1]['rel'])[:8]:
        print(f"  x={k[0]:.2f} m={k[1]}: pred {o['pred']:.3f} velo {o['velo']:.3f} "
              f"rel {o['rel']:.3f}  (vs shipped rel {ship['rel']:.3f})")
    print()
    print("HYPOTHESIS CHECK — does more shrinkage recover the reliability that")
    print("narrowing the kernel costs, without giving back prediction?")
    for x in X_GRID:
        row = " ".join(f"m={m}: rel {results[(x, m)]['rel']:.3f}/pred "
                       f"{results[(x, m)]['pred']:.3f}" for m in M_GRID)
        print(f"  x={x:.2f}  {row}")


if __name__ == '__main__':
    main()
