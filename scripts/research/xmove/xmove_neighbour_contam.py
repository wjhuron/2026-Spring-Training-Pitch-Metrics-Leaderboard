#!/usr/bin/env python3
"""How much of the DISPLAYED deviation is neighbour contamination?

The class-mean offset is already removed by percentile centring. What is not
removed is the spread: a curveball's leftover depends on how sweeper-dense its
corner of release space is, which is not a property of the pitcher.

Measured at the RENDERED unit (pitcher x pitch type x season, >=50 pitches),
because that is what the plate and the percentile actually show:

  share = R^2 of neighbour composition against the class-centred leftover.
          It is the fraction of what a viewer reads as "this pitcher's
          deviation" that is really "this pitch lives near other pitches".

Guard: SEAM (total cross-axis) is model-free, so it should show a far smaller
share. If it does, that is the argument for displaying SEAM over the residual.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np, _design
from xmove_agnostic_basis import add_harmonics, form

A = add_harmonics(load_np())
FEATS = form(3, True, True)
tilt = np.degrees(np.arctan2(A['st'], A['ct'])) % 360.0
tb = pd.cut(tilt, np.arange(0, 361, 15)).codes
vb = pd.cut(A['velo'], np.arange(70, 103, 3)).codes
cell = pd.Series(tb).astype(str) + '_' + pd.Series(vb).astype(str)

rows = []
for hand in ('R', 'L'):
    H = np.where(A['thr'] == hand)[0]
    X = _design(A, FEATS, H)
    ri = A['ivb'][H] - X @ np.linalg.lstsq(X, A['ivb'][H], rcond=None)[0]
    rh = A['hb_s'][H] - X @ np.linalg.lstsq(X, A['hb_s'][H], rcond=None)[0]
    d = pd.DataFrame({
        'pt': A['pt'][H], 'unit': A['unit'][H], 'cell': cell.values[H],
        'r_cross': -ri * A['st'][H] + rh * A['ct'][H],
        'seam': A['cross'][H]})
    # neighbour composition of each pitch's cell: share of the cell that is
    # some OTHER pitch type than this pitch's own
    tot = d.groupby('cell').size()
    for pt in d.pt.unique():
        own = d[d.pt == pt].groupby('cell').size()
        d.loc[d.pt == pt, 'foreign'] = (
            1 - (own / tot).reindex(d.loc[d.pt == pt, 'cell']).values)
    u = d.groupby(['pt', 'unit']).agg(
        n=('r_cross', 'size'), r_cross=('r_cross', 'mean'),
        seam=('seam', 'mean'), foreign=('foreign', 'mean')).reset_index()
    u = u[u.n >= 50]
    for pt, g in u.groupby('pt'):
        if len(g) < 150:
            continue
        for col in ('r_cross', 'seam'):
            y = g[col] - g[col].mean()          # class-centred, as displayed
            x = g.foreign - g.foreign.mean()
            b = (x @ y) / (x @ x)
            rows.append(dict(hand=hand, pt=pt, n=len(g), col=col,
                             sd=y.std(), share=1 - ((y - b * x).var() / y.var())))
t = pd.DataFrame(rows).pivot_table(index=['hand', 'pt'], columns='col',
                                   values=['sd', 'share'])
t.columns = [f'{b}_{a}' for a, b in t.columns]
print('=== share of class-centred spread explained by neighbour composition ===')
print('    (pitcher x type x season, >=50 pitches; sd in inches)')
print(t[['r_cross_sd', 'r_cross_share', 'seam_sd', 'seam_share']]
      .sort_values('r_cross_share', ascending=False).round(3).to_string())
