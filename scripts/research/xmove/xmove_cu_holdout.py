#!/usr/bin/env python3
"""Direct test: is the curveball's cross-axis offset CAUSED by sweepers in the
training pool, or merely correlated with sweeper density?

Refit the pooled model with a class removed from TRAINING only, then score the
held-out class with it. If dropping ST/SV collapses the curveball offset, the
contamination is causal. Control: dropping SL (which sits at a similar axis but
has ~0 cross-axis break) should do almost nothing. If SL moves it as much as ST
does, the story is wrong and something else drives it.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np, _design
from xmove_agnostic_basis import add_harmonics, form

A = add_harmonics(load_np())
FEATS = form(3, True, True)


def offset(hand, drop, score):
    H = np.where(A['thr'] == hand)[0]
    tr = H[~np.isin(A['pt'][H], drop)]
    X = _design(A, FEATS, tr)
    bi = np.linalg.lstsq(X, A['ivb'][tr], rcond=None)[0]
    bh = np.linalg.lstsq(X, A['hb_s'][tr], rcond=None)[0]
    s = H[A['pt'][H] == score]
    Xs = _design(A, FEATS, s)
    ri, rh = A['ivb'][s] - Xs @ bi, A['hb_s'][s] - Xs @ bh
    return (-ri * A['st'][s] + rh * A['ct'][s]).mean()


print('=== RHP curveball mean cross-axis leftover, by what was in TRAINING ===')
for lab, drop in [('full pool (shipped)', []), ('drop ST+SV (sweepers)', ['ST', 'SV']),
                  ('drop SL (control)', ['SL']), ('drop FF+SI (control)', ['FF', 'SI']),
                  ('drop ST+SV+SL', ['ST', 'SV', 'SL'])]:
    print(f'  {lab:24s} {offset("R", drop, "CU"):+6.2f}"')

print('\n=== reverse: RHP sweeper leftover, by what was in TRAINING ===')
for lab, drop in [('full pool (shipped)', []), ('drop CU', ['CU']),
                  ('drop SL (control)', ['SL'])]:
    print(f'  {lab:24s} {offset("R", drop, "ST"):+6.2f}"')

print('\n=== LHP replicate (curveball) ===')
for lab, drop in [('full pool (shipped)', []), ('drop ST+SV (sweepers)', ['ST', 'SV']),
                  ('drop SL (control)', ['SL'])]:
    print(f'  {lab:24s} {offset("L", drop, "CU"):+6.2f}"')
