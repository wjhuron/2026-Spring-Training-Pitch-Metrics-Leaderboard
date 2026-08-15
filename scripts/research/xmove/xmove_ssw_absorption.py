#!/usr/bin/env python3
"""Does the per-class fit (option 2) absorb class-typical seam deflection?

If it does, the league-mean IVB-OE / HB-OE per pitch type should be a large,
ordered quantity under the pooled fit (option 1) and ~0 by construction under
the per-class fit. Column SEAM is the model-free cross-axis break for the same
class, so option 1's residual should track it.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np, _design
from xmove_agnostic_basis import add_harmonics, form

FEATS = form(3, True, True)
A = add_harmonics(load_np())

for hand in ('R', 'L'):
    idx = np.where(A['thr'] == hand)[0]
    Xt = _design(A, FEATS, idx)
    p_i = A['ivb'][idx] - Xt @ np.linalg.lstsq(Xt, A['ivb'][idx], rcond=None)[0]
    p_h = A['hb_s'][idx] - Xt @ np.linalg.lstsq(Xt, A['hb_s'][idx], rcond=None)[0]
    c_i, c_h = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    for pt in pd.unique(A['pt'][idx]):
        s = np.where(A['pt'][idx] == pt)[0]
        if len(s) < 2000:
            continue
        Xs = _design(A, FEATS, idx[s])
        c_i[s] = A['ivb'][idx[s]] - Xs @ np.linalg.lstsq(Xs, A['ivb'][idx[s]], rcond=None)[0]
        c_h[s] = A['hb_s'][idx[s]] - Xs @ np.linalg.lstsq(Xs, A['hb_s'][idx[s]], rcond=None)[0]
    df = pd.DataFrame({'pt': A['pt'][idx], 'seam': A['cross'][idx] if 'cross' in A else np.nan,
                       'o1i': p_i, 'o1h': p_h, 'o2i': c_i, 'o2h': c_h})
    g = df.dropna(subset=['o2i']).groupby('pt').agg(
        n=('o1i', 'size'), seam=('seam', 'mean'),
        o1_ivb=('o1i', 'mean'), o1_hb=('o1h', 'mean'),
        o2_ivb=('o2i', 'mean'), o2_hb=('o2h', 'mean')).sort_values('seam')
    g['o1_mag'] = np.hypot(g.o1_ivb, g.o1_hb)
    g['o2_mag'] = np.hypot(g.o2_ivb, g.o2_hb)
    print(f'\n=== {hand}HP  league-mean residual by pitch type (inches) ===')
    print(g.round(2).to_string())
