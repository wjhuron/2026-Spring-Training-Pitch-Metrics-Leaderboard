"""Group floor for the expected-movement fit (pipeline.xmove.XMOVE_MIN_N).

The shipped MVN used a 150-pitch floor for 3 regressors. The basis has 13
terms, so the floor is re-measured: for a large (type, hand) group in 2025,
draw n pitches, fit, score a fixed 40k held-out set, and record the RMSE
excess over the fit on the whole remaining pool, after the production MAD
screen. Median of 20 draws per n.
Groups: FF_R (tightest), SL_R (widest tilt scatter), CH_L (a small-hand group).

The bar is a convention and is labelled one: the smallest n whose median
excess is under 0.10", the site's display rounding.

Usage: python3 scripts/research/xmove/xmove_min_n_sweep.py [--groups CH_L ...]
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.xmove import _design, _mad_keep  # noqa: E402
from xmove_agnostic_flight import build_cache  # noqa: E402

GRID = [100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000]
DRAWS = 20
HOLD = 40000


def main():
    d = build_cache(2025)
    d = d.dropna(subset=['xIndVrtBrk', 'xHorzBrk', 'Velocity', 'Extension', 'ArmAngle',
                         'SpinAxis', 'Spin Rate'])
    rng = np.random.default_rng(0)
    print(f'{"group":<6}{"n":>6}{"excess IVB":>12}{"excess HB":>11}   (median of {DRAWS} draws, inches)')
    groups = [('FF', 'R'), ('SL', 'R'), ('CH', 'L')]
    if '--groups' in sys.argv:
        groups = [tuple(a.split('_')) for a in sys.argv[sys.argv.index('--groups') + 1:]]
    for pt, thr in groups:
        g = d[(d['Pitch Type'] == pt) & (d['Throws'] == thr)]
        s = 1.0 if thr == 'R' else -1.0
        # the production fit screens the pool at MAD_THRESH first; do the same
        keep = _mad_keep(np.column_stack([g['xIndVrtBrk'].values, g['xHorzBrk'].values * s,
                                          g['ArmAngle'].values, g['Extension'].values,
                                          g['Velocity'].values, g['Spin Rate'].values]))
        g = g[keep]
        theta = np.radians(((g['SpinAxis'].values - 180.0) % 360.0) * s)
        cols = {'aa': g['ArmAngle'].values, 'ext': g['Extension'].values,
                'velo': g['Velocity'].values, 'spin': g['Spin Rate'].values}
        X = _design(cols, theta)
        yi, yh = g['xIndVrtBrk'].values, g['xHorzBrk'].values * s
        idx = rng.permutation(len(g))
        hold = min(HOLD, len(g) // 3)
        te, pool = idx[:hold], idx[hold:]
        full_i = np.linalg.lstsq(X[pool], yi[pool], rcond=None)[0]
        full_h = np.linalg.lstsq(X[pool], yh[pool], rcond=None)[0]
        base_i = np.sqrt(np.mean((yi[te] - X[te] @ full_i) ** 2))
        base_h = np.sqrt(np.mean((yh[te] - X[te] @ full_h) ** 2))
        print(f'{pt}_{thr:<3}{len(pool):>6}{0:>12.3f}{0:>11.3f}   (pool fit, RMSE {base_i:.2f} / {base_h:.2f})')
        for n in GRID:
            if n > len(pool):
                break
            ex_i, ex_h = [], []
            for _ in range(DRAWS):
                tr = rng.choice(pool, n, replace=False)
                bi = np.linalg.lstsq(X[tr], yi[tr], rcond=None)[0]
                bh = np.linalg.lstsq(X[tr], yh[tr], rcond=None)[0]
                ex_i.append(np.sqrt(np.mean((yi[te] - X[te] @ bi) ** 2)) - base_i)
                ex_h.append(np.sqrt(np.mean((yh[te] - X[te] @ bh) ** 2)) - base_h)
            print(f'{"":<6}{n:>6}{np.median(ex_i):>12.3f}{np.median(ex_h):>11.3f}')


if __name__ == '__main__':
    main()
