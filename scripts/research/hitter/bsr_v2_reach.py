"""bsr_v2_reach.py — the BSR v2 open question: is the up branch
contaminated by reach swings?

bsr_v2_build.py left one question open: the league-mean up slope is
NEGATIVE (~-1.3 mph/ft), and the suspicion is that extreme-length reach
or lunge swings drag it down, hiding the true near-range lengthening
return. This battery fits a THREE-REGIME model per hitter-season:

    bs = a + b_dn * min(d, 0) + b_near * clip(d, 0, CAP)
           + b_reach * max(d - CAP, 0),        d = sl - mean(sl)

and sweeps CAP over {0.4, 0.6, 0.8, 1.0} ft, against the v2 two-regime
fit as baseline. If a hitter has < 20 swings beyond the cap the reach
term is dropped and b_near comes from the swings at or below the cap.

OBJECTIVES (fixed before the run):
  S  year-to-year r of b_near (raw), common-hitter pool across all
     caps + baseline.
  V  the v2 prospective interaction test with b_near in place of the
     up branch (d_bs ~ dsl_p x near_c + dsl_n x dn_c + mains).
  Descriptives: league mean of b_near and b_reach per cap, share of
  above-mean swings beyond the cap, yy r of b_reach where estimable.
  DECISION: the three-regime form earns its place only if some cap
  beats the v2 baseline on S in BOTH pairs or on V t_near in both
  pairs; a flat curve means the reach story is not load-bearing and
  v2 stands as shipped.

Gate: the v2 convention (bs >= max(50, top-half median - 14)).
Qualification: >= 100 gated swings, >= 40 below the hinge, >= 40 in
(0, CAP] (so pools shrink as CAP tightens; hence the common pool).

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/bsr_v2_reach.py
Output: data/_bsr_v2_reach.json + printed tables.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bsr_screen import swing_frame, pearson  # noqa: E402
from bsr_v2_build import gate, hinge_fit  # noqa: E402

SEASONS = (2024, 2025, 2026)
PAIRS = ((2024, 2025), (2025, 2026))
CAPS = (0.4, 0.6, 0.8, 1.0)
OFFSET = 14
MIN_SW = 100
MIN_SIDE = 40
MIN_REACH = 20


def three_regime(sl, bs, cap, min_sw=MIN_SW, min_side=MIN_SIDE,
                 min_reach=MIN_REACH):
    """Returns dict with dn / near / reach slopes (+SEs) or None."""
    ok = np.isfinite(sl) & np.isfinite(bs)
    sl, bs = sl[ok], bs[ok]
    if len(sl) < min_sw:
        return None
    m = float(sl.mean())
    d = sl - m
    n_dn = int((d < 0).sum())
    n_near = int(((d > 0) & (d <= cap)).sum())
    n_reach = int((d > cap).sum())
    if n_dn < min_side or n_near < min_side:
        return None
    cols = [np.ones(len(d)), np.minimum(d, 0.0), np.clip(d, 0.0, cap)]
    with_reach = n_reach >= min_reach
    if with_reach:
        cols.append(np.maximum(d - cap, 0.0))
    else:
        # drop reach swings entirely so b_near is a pure near-range slope
        keep = d <= cap
        d, bs = d[keep], bs[keep]
        cols = [np.ones(len(d)), np.minimum(d, 0.0), np.clip(d, 0.0, cap)]
    X = np.column_stack(cols)
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    beta = XtX_inv @ (X.T @ bs)
    resid = bs - X @ beta
    dof = len(d) - X.shape[1]
    if dof < 10:
        return None
    s2 = float((resid ** 2).sum()) / dof
    se = np.sqrt(s2 * np.diag(XtX_inv))
    out = {'dn': float(beta[1]), 'near': float(beta[2]),
           'se_dn': float(se[1]), 'se_near': float(se[2]),
           'reach': float(beta[3]) if with_reach else None,
           'se_reach': float(se[3]) if with_reach else None,
           'n_dn': n_dn, 'n_near': n_near, 'n_reach': n_reach,
           'm': m, 'mean_bs': float(bs.mean())}
    return out


def season_build_reach(frame, cap):
    out = {}
    for hid, g in frame.groupby('hid'):
        gg = gate(g, OFFSET)
        fit = three_regime(gg['sl'].to_numpy(float),
                           gg['bs'].to_numpy(float), cap)
        if fit:
            out[hid] = fit
    return out


def season_build_v2(frame):
    out = {}
    for hid, g in frame.groupby('hid'):
        gg = gate(g, OFFSET)
        fit = hinge_fit(gg['sl'].to_numpy(float), gg['bs'].to_numpy(float))
        if fit:
            fit['near'] = fit['up']       # unify the key for comparison
            out[hid] = fit
    return out


def yy_r(b0, b1, key, pool):
    common = [h for h in b0 if h in b1 and h in pool
              and b0[h].get(key) is not None and b1[h].get(key) is not None]
    return pearson([b0[h][key] for h in common],
                   [b1[h][key] for h in common])


def validity(b0, b1):
    """d_bs ~ dsl_p x near + dsl_n x dn (+ mains). Raw slopes."""
    rows = []
    for h, r0 in b0.items():
        r1 = b1.get(h)
        if not r1:
            continue
        rows.append((r1['m'] - r0['m'], r1['mean_bs'] - r0['mean_bs'],
                     r0['near'], r0['dn']))
    if len(rows) < 60:
        return None
    d_sl, d_bs, near, dn = (np.array(c) for c in zip(*rows))
    near_c, dn_c = near - near.mean(), dn - dn.mean()
    dsl_p, dsl_n = np.maximum(d_sl, 0), np.minimum(d_sl, 0)
    X = np.column_stack([np.ones(len(d_sl)), dsl_p, dsl_n, near_c, dn_c,
                         dsl_p * near_c, dsl_n * dn_c])
    beta, *_ = np.linalg.lstsq(X, d_bs, rcond=None)
    resid = d_bs - X @ beta
    s2 = float((resid ** 2).sum()) / (len(d_sl) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    return {'b_near': float(beta[5]),
            't_near': float(beta[5] / cov[5, 5] ** 0.5), 'n': len(d_sl)}


def main():
    frames = {y: swing_frame(y) for y in SEASONS}
    builds = {'v2': {y: season_build_v2(frames[y]) for y in SEASONS}}
    for cap in CAPS:
        builds[f'cap{cap}'] = {y: season_build_reach(frames[y], cap)
                               for y in SEASONS}

    print('── descriptives (2025) ──')
    print('  variant  hitters  mean near  mean reach  reach-swing share  '
          'reach est.')
    desc = {}
    for name, B in builds.items():
        b = B[2025]
        nears = [v['near'] for v in b.values()]
        reaches = [v['reach'] for v in b.values() if v.get('reach') is not None]
        n_abv = sum(v.get('n_near', 0) + v.get('n_reach', 0)
                    for v in b.values())
        n_rch = sum(v.get('n_reach', 0) for v in b.values())
        share = n_rch / n_abv if n_abv else float('nan')
        desc[name] = {'hitters': len(b), 'mean_near': float(np.mean(nears)),
                      'mean_reach': (float(np.mean(reaches))
                                     if reaches else None),
                      'reach_share': share, 'n_reach_est': len(reaches)}
        rtxt = (f'{np.mean(reaches):+7.2f}' if reaches else '     --')
        print(f'  {name:7s}  {len(b):5d}   {np.mean(nears):+7.2f}   '
              f'{rtxt}        {share:5.1%}          {len(reaches):4d}')

    # common pool per pair: qualified under every variant in both years
    print('\n── S: year-to-year r of b_near (raw, common pool) ──')
    results = {'desc': desc, 'yy': {}, 'validity': {}}
    pools = {}
    for y0, y1 in PAIRS:
        pools[(y0, y1)] = set.intersection(*[
            set(B[y0]) & set(B[y1]) for B in builds.values()])
    print('  variant   24->25         25->26')
    for name, B in builds.items():
        cells = [yy_r(B[y0], B[y1], 'near', pools[(y0, y1)])
                 for y0, y1 in PAIRS]
        results['yy'][name] = cells
        print(f'  {name:8s}' + ''.join(f'  {r:+.3f} ({n:3d})'
                                       for r, n in cells))
    print('  (reach branch, where estimable:)')
    for name, B in builds.items():
        if name == 'v2':
            continue
        cells = [yy_r(B[y0], B[y1], 'reach', pools[(y0, y1)])
                 for y0, y1 in PAIRS]
        results['yy'][name + '_reach'] = cells
        print(f'  {name:8s}' + ''.join(
            f'  {r:+.3f} ({n:3d})' if r is not None else '     --      '
            for r, n in cells))

    print('\n── V: prospective t_near per pair (full per-variant pool) ──')
    print('  variant   24->25 b (t)        25->26 b (t)')
    for name, B in builds.items():
        cells = [validity(B[y0], B[y1]) for y0, y1 in PAIRS]
        results['validity'][name] = cells
        print(f'  {name:8s}' + ''.join(
            f'  {c["b_near"]:+.3f} ({c["t_near"]:+.2f}, n{c["n"]})'
            if c else '       --        ' for c in cells))

    out = os.path.join(ROOT, 'data', '_bsr_v2_reach.json')
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=1, default=float)
    os.replace(tmp, out)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
