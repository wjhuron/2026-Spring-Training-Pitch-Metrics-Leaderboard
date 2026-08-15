#!/usr/bin/env python3
"""Is the curveball's cross-axis offset sweeper contamination, and is it
reducible with more basis flexibility?

H1 CONFUSION. CU and ST overlap in release space (velo, spin, axis, arm angle)
   but differ hugely in cross-axis break (0.5" vs 7.6"). If one pooled surface
   must cover both, it averages them: CU gets predicted seam it does not have,
   ST gets less than it does. Test = overlap of the release-input marginals,
   and whether the CU offset tracks how sweeper-dense its axis neighbourhood is.

H2 FLEXIBILITY. If the offset is basis starvation, more harmonics kill it. If
   it survives an over-parameterised basis, the two pitches are genuinely
   unidentifiable from release inputs and the offset is a floor, not a bug.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np, _design
from xmove_agnostic_basis import add_harmonics, form

A = add_harmonics(load_np(), max_h=8)
R = np.where(A['thr'] == 'R')[0]
tilt = np.degrees(np.arctan2(A['st'], A['ct'])) % 360.0

# ---- H1: release-space overlap -------------------------------------------
print('=== RHP release-input marginals (mean / 10th / 90th pctl) ===')
rows = []
for pt in ('CU', 'SV', 'ST', 'SL'):
    m = R[A['pt'][R] == pt]
    rows.append(dict(pt=pt, n=len(m),
                     velo=A['velo'][m].mean(), v10=np.percentile(A['velo'][m], 10),
                     v90=np.percentile(A['velo'][m], 90),
                     spin=A['spin'][m].mean(), tilt=tilt[m].mean(),
                     t10=np.percentile(tilt[m], 10), t90=np.percentile(tilt[m], 90),
                     aa=A['aa'][m].mean(), cross=A['cross'][m].mean()))
print(pd.DataFrame(rows).set_index('pt').round(1).to_string())

# how much of each CU's local neighbourhood (axis x velo x spin) is sweeper?
cu = R[A['pt'][R] == 'CU']
sw = R[np.isin(A['pt'][R], ['ST', 'SV'])]
tb = pd.cut(tilt, np.arange(0, 361, 15))
vb = pd.cut(A['velo'], np.arange(70, 96, 3))
key = pd.Series(tb.codes).astype(str) + '_' + pd.Series(vb.codes).astype(str)
kcu, ksw = key.values[cu], key.values[sw]
dens = pd.Series(ksw).value_counts()
tot = pd.Series(key.values[R]).value_counts()
frac = (dens / tot).reindex(pd.unique(kcu)).fillna(0.0)
Xr = _design(A, form(3, True, True), R)
bi = np.linalg.lstsq(Xr, A['ivb'][R], rcond=None)[0]
bh = np.linalg.lstsq(Xr, A['hb_s'][R], rcond=None)[0]
Xc = _design(A, form(3, True, True), cu)
ri = A['ivb'][cu] - Xc @ bi
rh = A['hb_s'][cu] - Xc @ bh
rc = -ri * A['st'][cu] + rh * A['ct'][cu]
d = pd.DataFrame({'swfrac': frac.reindex(kcu).values, 'r_cross': rc})
d['bin'] = pd.qcut(d.swfrac, 5, duplicates='drop')
print('\n=== RHP curveballs, binned by how sweeper-dense their axis x velo cell is ===')
print(d.groupby('bin', observed=True).agg(
    n=('r_cross', 'size'), swfrac=('swfrac', 'mean'),
    r_cross=('r_cross', 'mean')).round(3).to_string())

# ---- H2: does flexibility kill it? ---------------------------------------
print('\n=== RHP  per-class mean cross-axis leftover vs basis richness ===')
out = {}
for nm, f in [('H3 (shipped)', form(3, True, True)), ('H5', form(5, True, True)),
              ('H8', form(8, True, True)),
              ('H8 + v2,s2', form(8, True, True) + ['v2', 's2', 'vs'])]:
    A['v2'], A['s2'], A['vs'] = A['velo'] ** 2, A['spin'] ** 2, A['velo'] * A['spin']
    X = _design(A, f, R)
    ri = A['ivb'][R] - X @ np.linalg.lstsq(X, A['ivb'][R], rcond=None)[0]
    rh = A['hb_s'][R] - X @ np.linalg.lstsq(X, A['hb_s'][R], rcond=None)[0]
    rc = -ri * A['st'][R] + rh * A['ct'][R]
    out[f'{nm} ({len(f)}p)'] = pd.Series(rc).groupby(A['pt'][R]).mean()
print(pd.DataFrame(out).round(2).to_string())
