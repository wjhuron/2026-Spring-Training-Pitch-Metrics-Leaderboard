"""locplus_shrinkage_extend.py — push K_WHIFF and K_SWING_COLL past the grid edge.

The per-surface coordinate descent put both of these at x32, the largest factor
on its grid, so that result was a grid boundary rather than an optimum. This
extends both to x256 to bracket them.

PRIMARY OBJECTIVE IS partial r(loc, future xRV | Stuff+), not raw prediction.
Raw prediction has a demonstrated failure mode: stuff predicts future runs, so
a location metric that absorbs stuff scores better on it while getting worse at
its actual job. The partial correlation holds the pitcher's Stuff+ fixed and
asks whether location still predicts — which is the question the metric exists
to answer. (The earlier suspicion that the per-surface combo was absorbing
stuff was tested and REFUTED: its partial gain was larger than its raw gain,
and its correlation with Stuff+ fell to ~0.009 from the shipped -0.118.)

Raw FF-velo correlation is reported but is NOT a penalty (Wally, 2026-07-25):
fastballs belong up and breaking/offspeed belong down, so a hard thrower
locating his fastball correctly SHOULD read as location value. That correlation
is descriptive of real pitch-type location structure, not contamination. The
contamination question is answered by Stuff+ correlation and the partial.

Expect collapse at the far end: K_WHIFF x256 = 2048 against ~40 pitches/cell
flattens the whiff surface to its group mean entirely. If nothing degrades by
x256 the harness is suspect, not the model.

Usage: python3 scripts/locplus_shrinkage_extend.py
"""
import os, sys, math, pickle, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline_locplus as lp
from pipeline_sdplus import make_rv_xrv

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393
MIN_SCORE, MIN_ACTUAL, MIN_REL = 150, 150, 100
X_FIXED, Z_FIXED = 1.5, 0.22

KEYS = ['K_WHIFF', 'K_FOUL', 'K_XWCON', 'K_SWING_COLL', 'K_SWING_COUNT', 'K_CS']
ORIG = {k: getattr(lp, k) for k in KEYS}
FIXED = {'K_FOUL': 2, 'K_XWCON': 8, 'K_SWING_COUNT': 2, 'K_CS': 2}

WHIFF_GRID = [32, 64, 128, 256]
SWING_GRID = [32, 64, 128, 256]


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


def partial(r_ly, r_ls, r_sy):
    d = (1 - r_ls ** 2) * (1 - r_sy ** 2)
    return (r_ly - r_ls * r_sy) / math.sqrt(d) if d > 0 else None


def by_pitcher(pitches):
    d = defaultdict(list)
    for p in pitches:
        d[(p.get('Pitcher'), p.get('Throws'))].append(p)
    return d


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
        actual = {}
        for k, ps in by_pitcher(seg[s]).items():
            v = [x for x in (rv_fn(p) for p in ps) if x is not None]
            if len(v) >= MIN_ACTUAL:
                actual[k] = sum(v) / len(v)
        stuff, velo = {}, {}
        for k, ps in by_pitcher(seg[f]).items():
            sp = [x for x in (lp.safe_float(p.get('Stuff+')) for p in ps) if x is not None]
            if len(sp) >= 100:
                stuff[k] = sum(sp) / len(sp)
            v = [x for x in (lp.safe_float(p.get('Velocity')) for p in ps
                             if p.get('Pitch Type') == 'FF') if x is not None]
            if len(v) >= 40:
                velo[k] = sum(v) / len(v)
        ctx[half] = {'first': seg[f], 'byp_f': by_pitcher(seg[f]), 'actual': actual,
                     'stuff': stuff, 'velo': velo,
                     'g0': [p for p in whole if par.get(p.get('Game Date')) == 0],
                     'g1': [p for p in whole if par.get(p.get('Game Date')) == 1]}
        ctx[half]['b0'] = by_pitcher(ctx[half]['g0'])
        ctx[half]['b1'] = by_pitcher(ctx[half]['g1'])

    def evaluate():
        o = {}
        for half in ('A', 'B'):
            c = ctx[half]
            S = lp.build_surfaces(c['first'], LG, SCALE)
            loc = score_map(c['byp_f'], S, MIN_SCORE)
            kk = [k for k in loc if k in c['actual'] and k in c['stuff']]
            r_ly = pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
            r_ls = pearson([loc[k] for k in kk], [c['stuff'][k] for k in kk])
            r_sy = pearson([c['stuff'][k] for k in kk], [c['actual'][k] for k in kk])
            o['p' + half] = partial(r_ly, r_ls, r_sy)
            o['raw' + half] = r_ly
            o['rls' + half] = r_ls
            kv = [k for k in loc if k in c['velo']]
            o['rlv' + half] = pearson([loc[k] for k in kv], [c['velo'][k] for k in kv])
            S0 = lp.build_surfaces(c['g0'], LG, SCALE)
            S1 = lp.build_surfaces(c['g1'], LG, SCALE)
            a0, a1 = score_map(c['b0'], S0, MIN_REL), score_map(c['b1'], S1, MIN_REL)
            ks = [k for k in a0 if k in a1]
            o['rel' + half] = pearson([a0[k] for k in ks], [a1[k] for k in ks])
        for m in ('p', 'raw', 'rls', 'rlv', 'rel'):
            o[m] = (o[m + 'A'] + o[m + 'B']) / 2
        return o

    print()
    print(f"{'K_WHIFF':>8s} {'K_SWCOLL':>9s} | {'PARTIAL|stuff':>14s} | {'raw':>6s} "
          f"| {'r(Stuff+)':>10s} {'r(velo)':>8s} | {'rel':>6s}")
    print('-' * 74)
    results = {}
    for w in WHIFF_GRID:
        for sc in SWING_GRID:
            t0 = time.time()
            lp.PHYS_X_IN = X_FIXED; lp.PHYS_Z_FRAC = Z_FIXED
            lp._KX = lp._k1d(X_FIXED / 2.0); lp._KZ = lp._k1d(Z_FIXED / lp.BIN_Z)
            facs = dict(FIXED); facs['K_WHIFF'] = w; facs['K_SWING_COLL'] = sc
            for k in KEYS:
                setattr(lp, k, ORIG[k] * facs[k])
            o = evaluate()
            results[(w, sc)] = o
            mark = '  <- descent winner' if (w == 32 and sc == 32) else ''
            print(f"{w:>8d} {sc:>9d} | {o['p']:>14.3f} | {o['raw']:>6.3f} "
                  f"| {o['rls']:>+10.3f} {o['rlv']:>+8.3f} | {o['rel']:>6.3f}{mark}",
                  flush=True)
            print(f"   ({time.time()-t0:.0f}s)", file=sys.stderr)

    best = max(results.items(), key=lambda kv: kv[1]['p'])
    print()
    print(f"best PARTIAL|stuff: K_WHIFF x{best[0][0]}, K_SWING_COLL x{best[0][1]} "
          f"-> {best[1]['p']:.3f}")
    edge = best[0][0] == WHIFF_GRID[-1] or best[0][1] == SWING_GRID[-1]
    print("STILL AT THE GRID EDGE — extend again before treating as an optimum."
          if edge else "Interior optimum bracketed on both axes.")
    print()
    print("partial|stuff by K_WHIFF (rows) x K_SWING_COLL (cols):")
    print(f"{'':>8s}" + "".join(f"{s:>9d}" for s in SWING_GRID))
    for w in WHIFF_GRID:
        print(f"{w:>8d}" + "".join(f"{results[(w, s)]['p']:>9.3f}" for s in SWING_GRID))


if __name__ == '__main__':
    main()
