#!/usr/bin/env python3
"""How much basis does a PITCH-TYPE-AGNOSTIC expected-movement model need?

xmove_agnostic.py showed the agnostic concept delivers what it should (FF and
SI thrown from one axis get one expectation, so their separation survives into
the residual as seam-shifted wake) but that a single axis harmonic fits so
badly that within-pitch-type R^2 goes negative: worse than that type's own mean.

The diagnosis is flexibility, not framing. With pitch type in the grouping key
the model fits eight local surfaces; drop it and one surface must cover the
whole axis and spin space. Review Finding 5 already found two axis harmonics
plus a spin x axis tensor optimal even WITH per-type grouping, so the agnostic
form should need at least that much.

OBJECTIVE: mean WITHIN-pitch-type R^2, and the worst type. Pooled R^2 is
explicitly not used: pooled across types most of the variance is between types,
so a model that cannot describe a single pitch type still scores 0.7. That is
the wrong-unit trap this whole file exists to avoid.

Guard: DIST, |corr(OE, raw movement)| within type at the rendered unit. A form
can buy within-type R^2 by drifting the residual back toward the raw column.

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_agnostic_basis.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_compare import load_np, corr  # noqa: E402
from xmove_agnostic import fit_cv, r2, unit_frame, PTS  # noqa: E402


def add_harmonics(A, max_h=3):
    """sin/cos of k*theta for k = 1..max_h, plus the spin-rate tensor with each.

    theta is the measured release spin axis, hand-signed, so k=1 is the Magnus
    direction and higher k let the surface bend differently on the arm side than
    the glove side, which is where seam effects live.
    """
    th = np.arctan2(A['st'], A['ct'])
    for k in range(1, max_h + 1):
        A[f'h{k}s'], A[f'h{k}c'] = np.sin(k * th), np.cos(k * th)
        A[f'sv{k}s'] = A['spin_v'] * A[f'h{k}s']
        A[f'sv{k}c'] = A['spin_v'] * A[f'h{k}c']
        A[f'vl{k}s'] = A['velo'] * A[f'h{k}s']
        A[f'vl{k}c'] = A['velo'] * A[f'h{k}c']
    return A


BASE = ['aa', 'ext', 'velo', 'spin']


def form(nh, tensor=True, velo_ax=False):
    f = list(BASE)
    for k in range(1, nh + 1):
        f += [f'h{k}s', f'h{k}c']
        if tensor:
            f += [f'sv{k}s', f'sv{k}c']
        if velo_ax:
            f += [f'vl{k}s', f'vl{k}c']
    return f


CONFIGS = [
    ('H1 tensor',            form(1, True,  False)),
    ('H2 tensor',            form(2, True,  False)),
    ('H3 tensor',            form(3, True,  False)),
    ('H2 tensor + velo x ax', form(2, True,  True)),
    ('H3 tensor + velo x ax', form(3, True,  True)),
]


def main():
    A = add_harmonics(load_np())
    gid = pd.factorize(pd.Series(A['thr']) + '_' +
                       pd.Series(A['season']).astype(str))[0]
    print(f'{len(A["ivb"]):,} pitches, pooled by hand x season\n', file=sys.stderr)

    print(f'{"form":<24}{"k":>4}{"meanR2 i":>10}{"wrstT i":>9}{"meanR2 h":>10}'
          f'{"wrstT h":>9}{"DIST i":>8}{"DIST h":>8}')
    print('-' * 82)
    results = {}
    for name, feats in CONFIGS:
        xi, xh = fit_cv(A, feats, gid)
        u = unit_frame(A, xi, xh)
        m0 = np.isfinite(xi)
        r_i, r_h, d_i, d_h = [], [], [], []
        for pt in PTS:
            m = m0 & (A['pt'] == pt)
            if m.sum() < 5000:
                continue
            r_i.append(r2(A['ivb'], xi, m))
            r_h.append(r2(A['hb_s'], xh, m))
            b = u[u.pt == pt]
            if len(b) >= 100:
                d_i.append(abs(corr(b.ivb_oe.values, b.ivb.values)))
                d_h.append(abs(corr(b.hb_oe.values, b.hb.values)))
        results[name] = (np.mean(r_i), min(r_i), np.mean(r_h), min(r_h),
                         np.mean(d_i), np.mean(d_h), xi, xh, feats)
        print(f'{name:<24}{len(feats):>4}{np.mean(r_i):>10.3f}{min(r_i):>9.3f}'
              f'{np.mean(r_h):>10.3f}{min(r_h):>9.3f}'
              f'{np.mean(d_i):>8.3f}{np.mean(d_h):>8.3f}')

    best = max(results, key=lambda k: results[k][0] + results[k][2])
    print(f'\nBest by mean within-type R^2 (IVB + HB): {best}')
    xi, xh, feats = results[best][6], results[best][7], results[best][8]

    print(f'\nPer pitch type, {best}:')
    print(f'{"pt":<5}{"R2 IVB":>9}{"R2 HB":>9}{"DIST i":>9}{"DIST h":>9}')
    u = unit_frame(A, xi, xh)
    m0 = np.isfinite(xi)
    for pt in PTS:
        m = m0 & (A['pt'] == pt)
        if m.sum() < 5000:
            continue
        b = u[u.pt == pt]
        print(f'{pt:<5}{r2(A["ivb"], xi, m):>9.3f}{r2(A["hb_s"], xh, m):>9.3f}'
              f'{abs(corr(b.ivb_oe.values, b.ivb.values)):>9.3f}'
              f'{abs(corr(b.hb_oe.values, b.hb.values)):>9.3f}')

    ff = u[u.pt == 'FF'].set_index(['pitcher', 'thr', 'season'])
    si = u[u.pt == 'SI'].set_index(['pitcher', 'thr', 'season'])
    j = ff.join(si, lsuffix='_ff', rsuffix='_si', how='inner')
    print(f'\nFF/SI test, {best}  (n = {len(j)})')
    print(f'  observed IVB {np.median(j.ivb_ff - j.ivb_si):>6.2f}"  '
          f'HB {np.median(j.hb_ff - j.hb_si):>6.2f}"')
    print(f'  EXPECTED IVB {np.median(j.xivb_ff - j.xivb_si):>6.2f}"  '
          f'HB {np.median(j.xhb_ff - j.xhb_si):>6.2f}"')
    print(f'  residual IVB {np.median(j.ivb_oe_ff - j.ivb_oe_si):>6.2f}"  '
          f'HB {np.median(j.hb_oe_ff - j.hb_oe_si):>6.2f}"')
    print(f'\nFEATS = {feats}')


if __name__ == '__main__':
    main()
