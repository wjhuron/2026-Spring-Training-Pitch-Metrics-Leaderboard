#!/usr/bin/env python3
"""pitchingplus_nxt_sweep.py — the Pitching+ blend weight on the shipped
scales, NEXT-SEASON objective (2026-08-23).

Port of pitchingplus_loso_full.py to the gate-v2 protocol. That script's
objective was within-season (first-half Pitching+ vs second-half xRV); the
shipped 0.80 was kept there only because 2026 alone preferred it (2021-25
argmin .72, flat .66-.78). This asks the question Pitching+ exists for:
which w best predicts the pitcher's NEXT season.

Per pair (Y, Y+1), Y in 2021..2025:
  Stuff+   v14 config fit on every season except Y and Y+1 (gate-v2 frames,
           prepare()), raw prediction on Y, per-type atoms exactly as
           _standardize/_atom_grade, pitcher = mean atom (>= MIN_PITCH).
  Loc+     pipeline.locplus surfaces built on season Y, scored on Y, pitcher
           mean raw -> 100 - 10*z over the season pool (n_prior = 0), as
           pitchingplus_loso_full did.
  target   pitcher mean of -target_xrv over Y+1 (>= MIN_ACTUAL pitches), the
           luck-neutral currency nxt_r uses; actual RV reported alongside.
  curve    r(w*S + (1-w)*L, target) for w in 0..1 step .01, per pair;
           cross-pair optimum = argmax of the mean Fisher z; flat region =
           within 1% of the best; paired bootstrap SE of r(w=argmin) -
           r(w=0.80) over pitchers, per pair.

Usage: python3 scripts/research/stuff/pitchingplus_nxt_sweep.py
"""
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'locplus'))
import stuff_plus.train_stuff as T                       # noqa: E402
import stuff_gate_v2 as G                                # noqa: E402
import pipeline.locplus as lp                            # noqa: E402
from locplus_constants_multiseason import adapt          # noqa: E402
from pitchingplus_loso_full import stuff_plus_from_raw, CACHE, LG, SCALE  # noqa: E402

MIN_PITCH, MIN_ACTUAL = 300, 300
W_GRID = [round(0.01 * i, 2) for i in range(101)]
OUT = os.path.join(ROOT, 'data', '_pplus_nxt_sweep.json')
from pipeline.utils import PITCHING_W_STUFF as SHIPPED_W  # noqa: E402


def fisher(r):
    r = max(-0.999999, min(0.999999, r))
    return 0.5 * math.log((1 + r) / (1 - r))


def locplus_season(y):
    pitches = adapt(os.path.join(ROOT, CACHE[y]))
    base = [p for p in pitches if lp.is_eligible_baseline(p)]
    S = lp.build_surfaces(base, LG, SCALE)
    acc = defaultdict(lambda: [0.0, 0])
    for p in base:
        v = lp.score_pitch(p, S)
        if v is not None:
            a = acc[p['Pitcher']]
            a[0] += v
            a[1] += 1
    raw = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_PITCH}
    vals = np.array(list(raw.values()))
    mu, sg = vals.mean(), vals.std()
    del pitches, base, S
    gc.collect()
    return {k: 100.0 - 10.0 * (v - mu) / sg for k, v in raw.items()}


def stuffplus_season(Y, Y1, frames):
    train_years = [y for y in G.SEASONS if y not in (Y, Y1)]
    slopes = G.fit_vaa_slopes([frames[y] for y in train_years])
    P = {y: G.prepare(frames[y], slopes) for y in train_years + [Y]}
    tr = [G.apply_variant(P[y], {}) for y in train_years]
    feats = tr[0][1]
    Xtr = pd.concat([G.design(d, feats) for d, _ in tr], ignore_index=True)
    ytr = np.concatenate([d['target_xrv'].values for d, _ in tr])
    del tr
    m = xgb.XGBRegressor(**T._params_for(Xtr))
    m.fit(Xtr, ytr)
    dY, _ = G.apply_variant(P[Y], {})
    dY = dY.assign(stuff_raw=-m.predict(G.design(dY, feats)))
    del Xtr, m, P
    gc.collect()
    return stuff_plus_from_raw(dY)


def actual_next(d):
    g = d.assign(neg=-d['target_xrv'], rv=-d['rv_raw']).groupby('pitcher').agg(
        t=('neg', 'mean'), rv=('rv', 'mean'), n=('neg', 'size'))
    g = g[g['n'] >= MIN_ACTUAL]
    return g['t'].to_dict(), g['rv'].to_dict()


def main():
    frames = {y: pd.read_pickle(G.season_path(y)) for y in G.SEASONS}
    G.set_arm_side_sign(list(frames.values()))
    print(f'config: feats {T.BASE_FEATS}, depth {T.TUNED["max_depth"]}, '
          f'shipped w {SHIPPED_W}')
    curves, curves_rv, tables = {}, {}, {}
    for Y, Y1 in G.PAIRS:
        t0 = time.time()
        L = locplus_season(Y)
        S = stuffplus_season(Y, Y1, frames)
        A, R = actual_next(frames[Y1])
        keys = sorted(k for k in S if k in L and k in A)
        s = np.array([S[k] for k in keys]); l = np.array([L[k] for k in keys])
        a = np.array([A[k] for k in keys]); r = np.array([R[k] for k in keys])
        tables[Y] = dict(keys=keys, s=s, l=l, a=a, r=r)
        curves[Y] = {w: G.pear(w * s + (1 - w) * l, a) for w in W_GRID}
        curves_rv[Y] = {w: G.pear(w * s + (1 - w) * l, r) for w in W_GRID}
        best = max(curves[Y], key=curves[Y].get)
        print(f'{Y}->{Y1}  n {len(keys)}  SD(S+) {s.std():.2f}  SD(L+) {l.std():.2f}  '
              f'r@0.5 {curves[Y][0.5]:.3f}  r@0.8 {curves[Y][0.8]:.3f}  '
              f'r@1.0 {curves[Y][1.0]:.3f}  argmax {best:.2f} '
              f'({curves[Y][best]:.3f})  [{time.time()-t0:.0f}s]', flush=True)

    agg = {w: np.mean([fisher(curves[Y][w]) for Y in curves]) for w in W_GRID}
    agg_rv = {w: np.mean([fisher(curves_rv[Y][w]) for Y in curves_rv]) for w in W_GRID}
    wbest = max(agg, key=agg.get)
    tol = abs(agg[wbest]) * 0.01
    flat = [w for w in W_GRID if agg[w] >= agg[wbest] - tol]
    wbest_rv = max(agg_rv, key=agg_rv.get)
    flat_rv = [w for w in W_GRID if agg_rv[w] >= agg_rv[wbest_rv] - abs(agg_rv[wbest_rv]) * 0.01]
    print(f'\nCROSS-PAIR OPTIMUM, next-season luck-neutral target: argmax w = {wbest:.2f} '
          f'(mean z {agg[wbest]:.4f}), flat within 1%: [{min(flat):.2f}, {max(flat):.2f}]'
          f'  shipped {SHIPPED_W:.2f} mean z {agg[SHIPPED_W]:.4f}')
    print(f'CROSS-PAIR OPTIMUM, next-season actual RV:            argmax w = {wbest_rv:.2f} '
          f'(mean z {agg_rv[wbest_rv]:.4f}), flat within 1%: [{min(flat_rv):.2f}, {max(flat_rv):.2f}]')
    print(f'\n{"w":>5s} {"mean z":>8s} ' + ''.join(f'{Y:>8d}' for Y in sorted(curves)))
    for w in (0.5, 0.6, 0.66, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.85, 0.9, 0.95, 1.0):
        mark = '  <- argmax' if abs(w - wbest) < 1e-9 else ('  <- shipped' if abs(w - SHIPPED_W) < 1e-9 else '')
        print(f'{w:>5.2f} {agg[w]:>8.4f} ' + ''.join(f'{curves[Y][w]:>8.3f}' for Y in sorted(curves)) + mark)

    # paired bootstrap: r(argmax) - r(shipped) per pair
    print('\nPaired bootstrap over pitchers, r(w=argmax) - r(w=shipped):')
    rng = np.random.default_rng(0)
    for Y in sorted(tables):
        tb = tables[Y]
        n = len(tb['keys'])
        idx = rng.integers(0, n, size=(2000, n))
        def rr(x, y):
            x = x - x.mean(1, keepdims=True); y = y - y.mean(1, keepdims=True)
            return (x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))
        pa = wbest * tb['s'] + (1 - wbest) * tb['l']
        ps = SHIPPED_W * tb['s'] + (1 - SHIPPED_W) * tb['l']
        d = rr(pa[idx], tb['a'][idx]) - rr(ps[idx], tb['a'][idx])
        print(f'  {Y}: d {curves[Y][wbest]-curves[Y][SHIPPED_W]:+.4f}  se {d.std():.4f}')
    json.dump({'curves': {str(Y): {str(w): v for w, v in c.items()} for Y, c in curves.items()},
               'curves_rv': {str(Y): {str(w): v for w, v in c.items()} for Y, c in curves_rv.items()},
               'argmax': wbest, 'flat': [min(flat), max(flat)], 'argmax_rv': wbest_rv,
               'flat_rv': [min(flat_rv), max(flat_rv)], 'shipped': SHIPPED_W},
              open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
