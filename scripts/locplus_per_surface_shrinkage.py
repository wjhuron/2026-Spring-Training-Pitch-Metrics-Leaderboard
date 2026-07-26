"""locplus_per_surface_shrinkage.py — do the six Loc+ shrinkage constants want
different amounts of shrinkage?

locplus_joint_regularizer_search.py found (x=1.5, all K x8) by moving ONE
scalar multiplier across all six pseudo-counts at once. That is a 2-D search
through a 7-D space. There is specific reason to expect the per-surface optima
to differ: x8 puts K_XWCON at 1600, which flattens the contact surface to
nearly its group mean. That may be correct — location-driven contact
suppression is mostly luck (THT), which is why xwK was already heavy at 200 —
but it means one constant is doing something qualitatively different from the
other five, and a shared multiplier cannot express that.

METHOD — coordinate descent, then a combination check.
  PASS 1: from the uniform x8 baseline, sweep each constant's own factor over
          {2,4,8,16,32} while the other five stay at x8. Whichever factor wins
          per surface is that surface's conditional optimum.
  PASS 2: combine the six winners and compare against uniform x8 and against
          shipped. This step is not a formality — coordinate descent from a
          single point misses joint optima when constants interact, and the
          combination UNDERPERFORMING its parts is exactly how that shows up.
          If it does, report the interaction rather than shipping the combo.

Objectives and how to read them are identical to the joint search: prediction
is primary and ungameable; velo leak is an accuracy criterion; whiff leak
crosses zero and is weak evidence; reliability is precision, gameable in the
over-regularized direction, and read as a floor. All measured inside each
chronological half so half B holds out anything chosen on half A.

Usage: python3 scripts/locplus_per_surface_shrinkage.py
"""
import os, sys, math, pickle, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline_locplus as lp
from pipeline_sdplus import make_rv_xrv

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393

X_FIXED, Z_FIXED = 1.5, 0.22
BASE_M = 8                                  # the uniform winner to descend from
FACTORS = [2, 4, 8, 16, 32]
KEYS = ['K_WHIFF', 'K_FOUL', 'K_XWCON', 'K_SWING_COLL', 'K_SWING_COUNT', 'K_CS']
ORIG = {k: getattr(lp, k) for k in KEYS}

MIN_SCORE, MIN_ACTUAL, MIN_LEAK, MIN_REL = 150, 150, 200, 100


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


def apply(factors, x=X_FIXED):
    lp.PHYS_X_IN = x; lp.PHYS_Z_FRAC = Z_FIXED
    lp._KX = lp._k1d(x / 2.0); lp._KZ = lp._k1d(Z_FIXED / lp.BIN_Z)
    for k in KEYS:
        setattr(lp, k, ORIG[k] * factors[k])


def score_map(byp, S, min_n):
    out = {}
    for k, ps in byp.items():
        v = [s for s in (lp.score_pitch(p, S) for p in ps) if s is not None]
        if len(v) >= min_n:
            out[k] = sum(v) / len(v)
    return out


def build_ctx():
    D = pickle.load(open(PKL, 'rb'))
    base = [p for p in D if lp.is_eligible_baseline(p)]
    dates = sorted({p.get('Game Date') for p in base if p.get('Game Date')})
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
    return ctx


def evaluate(ctx):
    o = {}
    for half in ('A', 'B'):
        c = ctx[half]
        S1 = lp.build_surfaces(c['first'], LG, SCALE)
        sc = score_map(c['byp_first'], S1, MIN_SCORE)
        kp = [k for k in sc if k in c['actual']]
        o['pred' + half] = pearson([sc[k] for k in kp], [c['actual'][k] for k in kp])
        Sw = lp.build_surfaces(c['whole'], LG, SCALE)
        full = score_map(c['byp_whole'], Sw, MIN_LEAK)
        kv = [k for k in full if k in c['ffv']]
        o['velo' + half] = abs(pearson([full[k] for k in kv], [c['ffv'][k] for k in kv]))
        S0 = lp.build_surfaces(c['g0'], LG, SCALE)
        S1b = lp.build_surfaces(c['g1'], LG, SCALE)
        a0, a1 = score_map(c['b0'], S0, MIN_REL), score_map(c['b1'], S1b, MIN_REL)
        kk = [k for k in a0 if k in a1]
        o['rel' + half] = pearson([a0[k] for k in kk], [a1[k] for k in kk])
    o['pred'] = (o['predA'] + o['predB']) / 2
    o['velo'] = (o['veloA'] + o['veloB']) / 2
    o['rel'] = (o['relA'] + o['relB']) / 2
    return o


def show(label, o, flag=''):
    print(f"{label:>26s} | {o['predA']:>6.3f} {o['predB']:>6.3f} {o['pred']:>6.3f} "
          f"| {o['velo']:>6.3f} | {o['rel']:>6.3f}{flag}", flush=True)


def main():
    print("loading cache...", file=sys.stderr)
    ctx = build_ctx()
    hdr = (f"{'config':>26s} | {'predA':>6s} {'predB':>6s} {'pred':>6s} "
           f"| {'velo':>6s} | {'rel':>6s}")

    uniform = {k: BASE_M for k in KEYS}
    print()
    print("PASS 1 — coordinate descent, one constant at a time from uniform x8")
    print(hdr)
    print('-' * len(hdr))
    apply(uniform)
    baseline = evaluate(ctx)
    show(f'uniform x{BASE_M}', baseline, '  <- descent origin')

    winners = {}
    for k in KEYS:
        best_f, best_o = BASE_M, baseline
        for f in FACTORS:
            if f == BASE_M:
                continue
            fac = dict(uniform); fac[k] = f
            apply(fac)
            o = evaluate(ctx)
            show(f'{k} x{f}', o)
            if o['pred'] > best_o['pred']:
                best_f, best_o = f, o
        winners[k] = best_f
        print(f"   -> {k}: best factor x{best_f} "
              f"(pred {best_o['pred']:.3f} vs {baseline['pred']:.3f})")

    print()
    print("PASS 2 — combination check")
    print(f"winners: " + ", ".join(f"{k}=x{v}" for k, v in winners.items()))
    print(hdr)
    print('-' * len(hdr))
    apply(uniform); show(f'uniform x{BASE_M}', evaluate(ctx))
    apply(winners); combo = evaluate(ctx); show('per-surface combo', combo)
    # shipped reference
    lp.PHYS_X_IN = 4.5; lp.PHYS_Z_FRAC = 0.22
    lp._KX = lp._k1d(4.5 / 2.0); lp._KZ = lp._k1d(0.22 / lp.BIN_Z)
    for k in KEYS:
        setattr(lp, k, ORIG[k])
    show('shipped (x=4.5, all x1)', evaluate(ctx))

    print()
    if winners == uniform:
        print("No surface preferred a different factor: the uniform multiplier")
        print("already expresses the optimum on this grid.")
    else:
        print("If 'per-surface combo' does not beat 'uniform x8', the constants")
        print("INTERACT and the one-at-a-time winners do not compose — report that")
        print("rather than shipping the combo.")


if __name__ == '__main__':
    main()
