#!/usr/bin/env python3
"""Why does the pooled (option 1) model leave a 2.6" curveball residual when
the curveball's model-free cross-axis break is ~0.5"?

SEAM = cross-axis = the component of break PERPENDICULAR to the release spin
axis. Pure Magnus puts zero there, so it is the seam signature. But a model
that mis-predicts the MAGNITUDE of break along the axis leaves a residual that
cross-axis cannot see. Decompose the residual the same way and find out which
kind of error the curveball is.

  along : residual in the direction the spin says the ball should break
          (magnitude error - the model got the amount wrong)
  cross : residual perpendicular to it
          (direction error - what SEAM measures)
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np, _design
from xmove_agnostic_basis import add_harmonics, form

A = add_harmonics(load_np())
FEATS = form(3, True, True)


def resid(feats, idx):
    X = _design(A, FEATS if feats is None else feats, idx)
    ri = A['ivb'][idx] - X @ np.linalg.lstsq(X, A['ivb'][idx], rcond=None)[0]
    rh = A['hb_s'][idx] - X @ np.linalg.lstsq(X, A['hb_s'][idx], rcond=None)[0]
    ct, st = A['ct'][idx], A['st'][idx]
    return ri, rh, ri * ct + rh * st, -ri * st + rh * ct


for hand in ('R', 'L'):
    idx = np.where(A['thr'] == hand)[0]
    ri, rh, ra, rc = resid(None, idx)
    df = pd.DataFrame({'pt': A['pt'][idx], 'velo': A['velo'][idx],
                       'cross_tot': A['cross'][idx], 'along_tot': A['along'][idx],
                       'r_along': ra, 'r_cross': rc})
    g = df.groupby('pt').agg(n=('velo', 'size'), velo=('velo', 'mean'),
                             tot_along=('along_tot', 'mean'), tot_cross=('cross_tot', 'mean'),
                             r_along=('r_along', 'mean'), r_cross=('r_cross', 'mean')
                             ).sort_values('velo')
    print(f'\n=== {hand}HP  residual decomposed in the release-axis frame (in) ===')
    print('  tot_* = total break; r_* = what the pooled model leaves behind')
    print(g.round(2).to_string())

# velocity profile of the along-axis error, ALL pitch types, to see whether the
# curveball is special or just the slowest thing in the pool.
idx = np.where(A['thr'] == 'R')[0]
ri, rh, ra, rc = resid(None, idx)
df = pd.DataFrame({'pt': A['pt'][idx], 'velo': A['velo'][idx], 'r_along': ra})
df['vbin'] = pd.cut(df.velo, [0, 74, 78, 81, 84, 87, 90, 93, 96, 200])
print('\n=== RHP  along-axis residual by velocity, all pitch types ===')
print(df.groupby('vbin', observed=True).agg(
    n=('velo', 'size'), r_along=('r_along', 'mean')).round(2).to_string())
print('\n=== RHP  along-axis residual by velocity, WITHIN pitch type ===')
pv = df.pivot_table(index='vbin', columns='pt', values='r_along',
                    aggfunc='mean', observed=True)
cn = df.pivot_table(index='vbin', columns='pt', values='r_along',
                    aggfunc='size', observed=True)
print(pv.where(cn >= 3000).round(2).to_string())
