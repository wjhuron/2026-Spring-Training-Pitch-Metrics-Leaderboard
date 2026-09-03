"""Ridge penalty for the expected-movement fit, and the pool floor under it.

xmove_min_n_sweep.py found the OLS basis unstable at small pools on groups
whose release tilt barely varies (CH_L: 1" excess RMSE at n = 1000). The cause
is collinearity: with tilt nearly constant, spin x sin(tilt) is a multiple of
spin. A ridge penalty on the standardised columns (intercept unpenalised)
trades a little bias for that variance.

Sweep: lambda in a log grid x n in a grid, three groups, held-out RMSE excess
over the unpenalised full-pool fit, median of 20 draws, after the production
MAD screen. Two questions, answered separately:
  1. which lambda minimises the excess at small n (the early-season regime)
  2. whether that lambda costs anything at n >= 5000 (the production regime)
Then the floor: the smallest n whose excess is under 0.10" on the worst group
under the chosen lambda. The 0.10" bar is a convention (display rounding).

Usage: python3 scripts/research/xmove/xmove_ridge_sweep.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.xmove import _design, _mad_keep  # noqa: E402
from xmove_agnostic_flight import build_cache  # noqa: E402

LAMBDAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
NS = [150, 300, 500, 1000, 2000, 5000]
DRAWS = 20
GROUPS = [('FF', 'R'), ('SL', 'R'), ('CH', 'L'), ('FS', 'L')]


def ridge_fit(X, y, lam):
    """Standardised ridge, intercept unpenalised. X has an intercept column 0."""
    Z = X[:, 1:]
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Zs = (Z - mu) / sd
    ym = y.mean()
    n, k = Zs.shape
    b = np.linalg.solve(Zs.T @ Zs + lam * n * np.eye(k), Zs.T @ (y - ym))
    braw = b / sd
    b0 = ym - braw @ mu
    return np.concatenate([[b0], braw])


def main():
    d = build_cache(2025)
    d = d.dropna(subset=['xIndVrtBrk', 'xHorzBrk', 'Velocity', 'Extension', 'ArmAngle',
                         'SpinAxis', 'Spin Rate'])
    rng = np.random.default_rng(0)
    worst = {}
    for pt, thr in GROUPS:
        g = d[(d['Pitch Type'] == pt) & (d['Throws'] == thr)]
        s = 1.0 if thr == 'R' else -1.0
        keep = _mad_keep(np.column_stack([g['xIndVrtBrk'].values, g['xHorzBrk'].values * s,
                                          g['ArmAngle'].values, g['Extension'].values,
                                          g['Velocity'].values, g['Spin Rate'].values]))
        g = g[keep]
        theta = np.radians(((g['SpinAxis'].values - 180.0) % 360.0) * s)
        X = _design({'aa': g['ArmAngle'].values, 'ext': g['Extension'].values,
                     'velo': g['Velocity'].values, 'spin': g['Spin Rate'].values}, theta)
        yi, yh = g['xIndVrtBrk'].values, g['xHorzBrk'].values * s
        idx = rng.permutation(len(g))
        hold = min(40000, len(g) // 3)
        te, pool = idx[:hold], idx[hold:]
        bi = np.linalg.lstsq(X[pool], yi[pool], rcond=None)[0]
        bh = np.linalg.lstsq(X[pool], yh[pool], rcond=None)[0]
        base = (np.sqrt(np.mean((yi[te] - X[te] @ bi) ** 2)),
                np.sqrt(np.mean((yh[te] - X[te] @ bh) ** 2)))
        print(f'\n{pt}_{thr}  pool {len(pool):,}  OLS full-pool RMSE {base[0]:.2f} / {base[1]:.2f}'
              f'   cells = median excess (IVB+HB)/2 over 20 draws, inches')
        print(f'{"n":>6}' + ''.join(f'{l:>9.0e}' for l in LAMBDAS))
        draws = {n: [rng.choice(pool, min(n, len(pool)), replace=False) for _ in range(DRAWS)] for n in NS}
        for n in NS:
            line = f'{n:>6}'
            for lam in LAMBDAS:
                ex = []
                for tr in draws[n]:
                    ci = ridge_fit(X[tr], yi[tr], lam)
                    ch = ridge_fit(X[tr], yh[tr], lam)
                    ei = np.sqrt(np.mean((yi[te] - X[te] @ ci) ** 2)) - base[0]
                    eh = np.sqrt(np.mean((yh[te] - X[te] @ ch) ** 2)) - base[1]
                    ex.append((ei + eh) / 2)
                m = float(np.median(ex))
                worst[(n, lam)] = max(worst.get((n, lam), -9), m)
                line += f'{m:>9.3f}'
            print(line)
    print('\nWORST group per cell:')
    print(f'{"n":>6}' + ''.join(f'{l:>9.0e}' for l in LAMBDAS))
    for n in NS:
        print(f'{n:>6}' + ''.join(f'{worst[(n, lam)]:>9.3f}' for lam in LAMBDAS))


if __name__ == '__main__':
    main()
