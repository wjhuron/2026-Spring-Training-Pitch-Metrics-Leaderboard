"""Can the site evaluate the spin + axis basis at GROUP MEANS?

js/aggregator.js scores the MVN model at the per-group means of arm angle,
extension and velocity. That is exact for a linear model. The proposed basis
has sin/cos of release tilt and spin x sin/cos, and the mean of a sine is not
the sine of the mean, so scoring at (mean spin, circular-mean tilt) is an
approximation. This measures its size at the rendered unit against the exact
mean of per-pitch predictions, per pitch type, so the plumbing decision (new
per-group sums vs evaluate-at-means) rests on a number.

Usage: python3 scripts/research/xmove/xmove_basis_at_means.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from xmove_agnostic_flight import build_cache, PTS  # noqa: E402
from xmove_pertype_ladder import to_arrays, B1, design, MIN_N  # noqa: E402


def main():
    A = to_arrays(build_cache(2025))
    xi, xh = np.full(len(A['ivb']), np.nan), np.full(len(A['ivb']), np.nan)
    coefs = {}
    for gi in range(A['gid'].max() + 1):
        idx = np.where(A['gid'] == gi)[0]
        if len(idx) < MIN_N:
            continue
        X = design(A, B1, idx)
        bi = np.linalg.lstsq(X, A['ivb'][idx], rcond=None)[0]
        bh = np.linalg.lstsq(X, A['hb_s'][idx], rcond=None)[0]
        xi[idx], xh[idx] = X @ bi, X @ bh
        coefs[gi] = (bi, bh)
    ok = np.isfinite(xi)
    d = pd.DataFrame({k: A[k][ok] for k in ['pitcher', 'pt', 'thr', 'gid', 'aa', 'ext',
                                             'velo', 'spin', 'ct', 'st']})
    d['xi'], d['xh'] = xi[ok], xh[ok]
    u = d.groupby(['pitcher', 'pt', 'thr', 'gid']).agg(
        n=('xi', 'size'), xi=('xi', 'mean'), xh=('xh', 'mean'), aa=('aa', 'mean'),
        ext=('ext', 'mean'), velo=('velo', 'mean'), spin=('spin', 'mean'),
        ct=('ct', 'mean'), st=('st', 'mean')).reset_index()
    u = u[u.n >= 50].reset_index(drop=True)
    # evaluate the basis at the means: circular-mean tilt from mean sin/cos
    th = np.arctan2(u['st'].values, u['ct'].values)
    B = dict(aa=u['aa'].values, ext=u['ext'].values, velo=u['velo'].values, spin=u['spin'].values)
    for k in (1, 2):
        hs, hc = np.sin(k * th), np.cos(k * th)
        B[f'h{k}s'], B[f'h{k}c'] = hs, hc
        B[f'sp{k}s'], B[f'sp{k}c'] = u['spin'].values / 1000.0 * hs, u['spin'].values / 1000.0 * hc
    ai, ah = np.full(len(u), np.nan), np.full(len(u), np.nan)
    for gi, (bi, bh) in coefs.items():
        m = (u['gid'] == gi).values
        if m.any():
            X = design(B, B1, np.where(m)[0])
            ai[m], ah[m] = X @ bi, X @ bh
    u['di'], u['dh'] = ai - u['xi'], ah - u['xh']
    print('2025, pitcher x type x hand units with 50+ pitches. '
          'Error of basis-at-means against mean of per-pitch predictions, inches:')
    print(f'{"pt":<4}{"units":>6}{"|dIVB| med":>11}{"p95":>7}{"max":>7}{"|dHB| med":>11}{"p95":>7}{"max":>7}')
    for pt in PTS:
        b = u[u.pt == pt]
        if len(b) == 0:
            continue
        ai_, ah_ = b['di'].abs(), b['dh'].abs()
        print(f'{pt:<4}{len(b):>6}{ai_.median():>11.2f}{ai_.quantile(.95):>7.2f}{ai_.max():>7.2f}'
              f'{ah_.median():>11.2f}{ah_.quantile(.95):>7.2f}{ah_.max():>7.2f}')


if __name__ == '__main__':
    main()
