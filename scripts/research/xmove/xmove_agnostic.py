#!/usr/bin/env python3
"""Pitch-type-AGNOSTIC expected movement: the pure-physics variant.

The question the metric should answer, per Wally: "if a pitcher throws from
this arm angle, with this spin axis, at this velocity and this spin rate, the
ball should move xIVB / xHB." Nothing in that sentence mentions a pitch type,
so pitch type should not be in the grouping key. The review
(docs/expected_movement_review.md, Recommended design section 1) flagged this as
a one-line change and a genuinely different metric, not a better version of the
same one. This measures it.

Why it may work at all: pitch types differ mainly BY release axis and spin, and
both are regressors. A curveball's axis already points down and glove-side, so
the model can expect a curveball's break without being told it is a curveball.

What dropping pitch type buys, if the fit holds up:
  * FF and SI thrown from the same axis get the SAME expectation, so the ~9"
    of separation reads entirely as seam-shifted wake (review Finding 3)
    instead of being absorbed into a per-type intercept.
  * Two review Limitations disappear. "Pitch type is partly movement-derived"
    stops applying, and a retag can no longer move a pitch's own baseline,
    because there is no per-type baseline left to move.

What it may cost: per-type fit. Reported per type so the cost is visible rather
than hidden in a pooled average.

Criteria are DESCRIPTIVE, per Wally: R^2 (how well the physics is described)
and DIST (is the residual distinct from the raw column it sits beside).
Run value is deliberately not a criterion here.

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_agnostic.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_compare import load_np, _design, MIN_N, corr, UNIT_MIN  # noqa: E402

# Arm angle, extension, velocity, spin rate, release axis, and the spin x axis
# interaction. Every one is upstream of movement; none is derived from break.
FEATS = ['aa', 'ext', 'velo', 'spin', 'ax_sin', 'ax_cos', 'sv_sin', 'sv_cos']
PTS = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']


def fit_cv(A, feats, gid):
    """Cross-fit by game parity within each group. Every pitch is scored by a
    model that never saw its game."""
    n = len(A['ivb'])
    x_ivb, x_hb = np.full(n, np.nan), np.full(n, np.nan)
    order = np.argsort(gid, kind='stable')
    bounds = np.searchsorted(gid[order], np.arange(gid.max() + 2))
    for gi in range(gid.max() + 1):
        idx = order[bounds[gi]:bounds[gi + 1]]
        if len(idx) == 0:
            continue
        par = A['game'][idx] % 2
        for p in (0, 1):
            tr, te = idx[par == p], idx[par == 1 - p]
            if len(tr) < MIN_N or len(te) == 0:
                continue
            Xt, Xs = _design(A, feats, tr), _design(A, feats, te)
            bi = np.linalg.lstsq(Xt, A['ivb'][tr], rcond=None)[0]
            bh = np.linalg.lstsq(Xt, A['hb_s'][tr], rcond=None)[0]
            x_ivb[te], x_hb[te] = Xs @ bi, Xs @ bh
    return x_ivb, x_hb


def r2(y, yhat, mask):
    yy, hh = y[mask], yhat[mask]
    return 1 - ((yy - hh) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


def unit_frame(A, xi, xh):
    ok = np.isfinite(xi) & np.isfinite(xh)
    d = pd.DataFrame({
        'pitcher': A['pitcher'][ok], 'thr': A['thr'][ok], 'pt': A['pt'][ok],
        'season': A['season'][ok],
        'ivb': A['ivb'][ok], 'hb': A['hb_s'][ok],
        'ivb_oe': A['ivb'][ok] - xi[ok], 'hb_oe': A['hb_s'][ok] - xh[ok],
        'xivb': xi[ok], 'xhb': xh[ok],
        'cross': A['cross'][ok],
    })
    u = d.groupby(['pitcher', 'thr', 'pt', 'season']).agg(
        n=('ivb', 'size'), **{c: (c, 'mean') for c in
                              ['ivb', 'hb', 'ivb_oe', 'hb_oe', 'xivb', 'xhb', 'cross']}
    ).reset_index()
    return u[u.n >= UNIT_MIN]


def main():
    A = load_np()
    print(f'{len(A["ivb"]):,} pitches\n', file=sys.stderr)

    gid_type = A['gid']                     # pitch type x hand x season
    gid_pool = pd.factorize(pd.Series(A['thr']) + '_' +
                            pd.Series(A['season']).astype(str))[0]   # hand x season

    runs = {}
    for name, g in (('per-type', gid_type), ('AGNOSTIC', gid_pool)):
        xi, xh = fit_cv(A, FEATS, g)
        runs[name] = (xi, xh, unit_frame(A, xi, xh))
        m = np.isfinite(xi)
        print(f'  {name:<10} fitted, coverage {m.mean():.3f}', file=sys.stderr)

    print('=' * 74)
    print('FIT — pooled R^2, and per pitch type (cross-fit by game parity)')
    print('=' * 74)
    print(f'{"":<6}{"per-type IVB":>14}{"agnostic IVB":>14}{"per-type HB":>14}{"agnostic HB":>14}')
    mi = np.isfinite(runs['per-type'][0]) & np.isfinite(runs['AGNOSTIC'][0])
    print(f'{"ALL":<6}'
          f'{r2(A["ivb"], runs["per-type"][0], mi):>14.3f}'
          f'{r2(A["ivb"], runs["AGNOSTIC"][0], mi):>14.3f}'
          f'{r2(A["hb_s"], runs["per-type"][1], mi):>14.3f}'
          f'{r2(A["hb_s"], runs["AGNOSTIC"][1], mi):>14.3f}')
    for pt in PTS:
        m = mi & (A['pt'] == pt)
        if m.sum() < 5000:
            continue
        print(f'{pt:<6}'
              f'{r2(A["ivb"], runs["per-type"][0], m):>14.3f}'
              f'{r2(A["ivb"], runs["AGNOSTIC"][0], m):>14.3f}'
              f'{r2(A["hb_s"], runs["per-type"][1], m):>14.3f}'
              f'{r2(A["hb_s"], runs["AGNOSTIC"][1], m):>14.3f}')

    print()
    print('=' * 74)
    print('DISTINCTNESS — |corr(OE, raw movement)| within pitch type, LOWER better')
    print('(the residual sits next to the raw column on the card; ~1 = a copy)')
    print('=' * 74)
    print(f'{"":<6}{"n":>6}{"per-type i":>12}{"agnostic i":>12}{"per-type h":>12}{"agnostic h":>12}')
    ut, ua = runs['per-type'][2], runs['AGNOSTIC'][2]
    for pt in PTS:
        a, b = ut[ut.pt == pt], ua[ua.pt == pt]
        if len(a) < 100 or len(b) < 100:
            continue
        print(f'{pt:<6}{len(b):>6}'
              f'{abs(corr(a.ivb_oe.values, a.ivb.values)):>12.3f}'
              f'{abs(corr(b.ivb_oe.values, b.ivb.values)):>12.3f}'
              f'{abs(corr(a.hb_oe.values, a.hb.values)):>12.3f}'
              f'{abs(corr(b.hb_oe.values, b.hb.values)):>12.3f}')

    print()
    print('=' * 74)
    print('THE FF/SI TEST — pitchers throwing both, >= 50 each, same season')
    print('Under the agnostic model a shared release axis must give a SHARED')
    print('expectation, so the observed gap should survive into the residual.')
    print('=' * 74)
    for name in ('per-type', 'AGNOSTIC'):
        u = runs[name][2]
        ff = u[u.pt == 'FF'].set_index(['pitcher', 'thr', 'season'])
        si = u[u.pt == 'SI'].set_index(['pitcher', 'thr', 'season'])
        j = ff.join(si, lsuffix='_ff', rsuffix='_si', how='inner')
        if not len(j):
            continue
        print(f'\n  {name}  (n = {len(j)} pitcher-seasons)')
        print(f'    observed  IVB gap {np.median(j.ivb_ff - j.ivb_si):>6.2f}"   '
              f'HB gap {np.median(j.hb_ff - j.hb_si):>6.2f}"')
        print(f'    EXPECTED  IVB gap {np.median(j.xivb_ff - j.xivb_si):>6.2f}"   '
              f'HB gap {np.median(j.xhb_ff - j.xhb_si):>6.2f}"')
        print(f'    residual  IVB gap {np.median(j.ivb_oe_ff - j.ivb_oe_si):>6.2f}"   '
              f'HB gap {np.median(j.hb_oe_ff - j.hb_oe_si):>6.2f}"')


if __name__ == '__main__':
    main()
