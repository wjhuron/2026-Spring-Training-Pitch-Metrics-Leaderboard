"""Does the movement residual carry signal the raw movement columns do not?

Two checks at the rendered unit (pitcher x hand x pitch type x season):

  VALUE   regress run value/100 on velo + IVB + HB + spin, then add the OE
          pair, and read the incremental R^2. If OE adds nothing over columns
          already on the leaderboard, it is decoration.
  DIST    per-pitch-type |corr(OE, raw movement)| for the shipped form vs the
          candidate, so the gates can be set per pitch type instead of flat.

Run after xmove_compare.py -- shares its cross-fit harness.
"""
import os, sys, math, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_compare import load_np, run_linear, FORMS, UNIT_MIN, corr


def rv_lookup(seasons=(2021, 2022, 2023, 2024, 2025)):
    rows = []
    for y in seasons:
        with open(f'/Users/wallyhuron/Huronalytics/data/_pitches{y}_training.pkl', 'rb') as f:
            d = pickle.load(f)
        for r in d:
            rv = r.get('RunExp')
            if rv is None:
                continue
            rows.append((y, r.get('Pitcher'), r.get('Throws'), r.get('Pitch Type'), rv))
    rv = pd.DataFrame(rows, columns=['season', 'pitcher', 'thr', 'pt', 'rv'])
    rv['rv'] = pd.to_numeric(rv.rv, errors='coerce')
    return (rv.dropna(subset=['rv']).groupby(['season', 'pitcher', 'thr', 'pt'])
            .agg(rv100=('rv', lambda s: 100 * s.mean()), n_rv=('rv', 'size')).reset_index())


def units(A, xi, xh):
    ok = np.isfinite(xi) & np.isfinite(xh)
    d = pd.DataFrame({
        'season': A['season'][ok], 'pitcher': A['pitcher'][ok], 'thr': A['thr'][ok],
        'pt': A['pt'][ok], 'ivb': A['ivb'][ok], 'hb': A['hb_s'][ok],
        'velo': A['velo'][ok], 'spin': A['spin'][ok],
        'ivb_oe': A['ivb'][ok] - xi[ok], 'hb_oe': A['hb_s'][ok] - xh[ok],
        'along_oe': (A['ivb'][ok] - xi[ok]) * A['ct'][ok] + (A['hb_s'][ok] - xh[ok]) * A['st'][ok],
        'cross_oe': -(A['ivb'][ok] - xi[ok]) * A['st'][ok] + (A['hb_s'][ok] - xh[ok]) * A['ct'][ok],
        'cross': A['cross'][ok],
    })
    u = d.groupby(['season', 'pitcher', 'thr', 'pt']).agg(
        n=('ivb', 'size'), **{c: (c, 'mean') for c in
        ['ivb', 'hb', 'velo', 'spin', 'ivb_oe', 'hb_oe', 'along_oe', 'cross_oe', 'cross']}
    ).reset_index()
    return u[u.n >= UNIT_MIN]


def r2_of(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()


if __name__ == '__main__':
    A = load_np()
    xi_s, xh_s = run_linear(A, FORMS['S1 shipped aa,ext,v'])
    xi_n, xh_n = run_linear(A, FORMS['S3b +spin x axis'])
    us, un = units(A, xi_s, xh_s), units(A, xi_n, xh_n)
    rv = rv_lookup()
    us = us.merge(rv, on=['season', 'pitcher', 'thr', 'pt'])
    un = un.merge(rv, on=['season', 'pitcher', 'thr', 'pt'])

    print(f'\nDISTINCTNESS  |corr(OE, raw movement)| per pitch type -- LOWER is better')
    print(f"{'pt':>4} {'units':>6} {'ship ivb':>9} {'new ivb':>8} {'ship hb':>8} {'new hb':>7}")
    for pt in ['FF', 'SI', 'FC', 'SL', 'ST', 'SV', 'CU', 'CH', 'FS']:
        a, b = us[us.pt == pt], un[un.pt == pt]
        if len(a) < 100:
            continue
        print(f'{pt:>4} {len(a):>6} {abs(corr(a.ivb_oe.values, a.ivb.values)):>9.3f} '
              f'{abs(corr(b.ivb_oe.values, b.ivb.values)):>8.3f} '
              f'{abs(corr(a.hb_oe.values, a.hb.values)):>8.3f} '
              f'{abs(corr(b.hb_oe.values, b.hb.values)):>7.3f}')

    print(f'\nRUN VALUE  incremental R^2 over velo+IVB+HB+spin (pitcher-season-pitchtype)')
    print(f"{'pt':>4} {'units':>6} {'base':>7} {'+ship OE':>9} {'+new OE':>8} "
          f"{'+along/cross':>13} {'+SSW inches':>12}")
    for pt in ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']:
        a, b = us[us.pt == pt], un[un.pt == pt]
        if len(b) < 150:
            continue
        y = b.rv100.values
        base = np.column_stack([b.velo, b.ivb, b.hb, b.spin])
        r0 = r2_of(y, base)
        ya = a.rv100.values
        basea = np.column_stack([a.velo, a.ivb, a.hb, a.spin])
        r_ship = r2_of(ya, np.column_stack([basea, a.ivb_oe, a.hb_oe])) - r2_of(ya, basea)
        r_new = r2_of(y, np.column_stack([base, b.ivb_oe, b.hb_oe])) - r0
        r_pol = r2_of(y, np.column_stack([base, b.along_oe, b.cross_oe])) - r0
        r_ssw = r2_of(y, np.column_stack([base, b.cross])) - r0
        print(f'{pt:>4} {len(b):>6} {r0:>7.3f} {r_ship:>9.4f} {r_new:>8.4f} '
              f'{r_pol:>13.4f} {r_ssw:>12.4f}')
