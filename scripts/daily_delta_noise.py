#!/usr/bin/env python3
"""How big does a one-start vs-season delta have to be to mean anything?

The daily card prints "92.7 (-0.9)". Question: is -0.9 a real change or the
noise of averaging 34 pitches? Measured, not asserted.

Two quantities, both per (pitcher, game, pitch type) on 2021-2025 MLB:

  SE   = within-start SD / sqrt(n). The measurement error on "his velocity
         today". This is the floor: a delta below it is indistinguishable
         from having thrown a different 34 pitches.
  SD_d = SD of (start mean - his season mean) across all his starts. This is
         noise PLUS real day-to-day variation, so SD_d > SE always, and the
         gap between them is the real signal.

A fixed threshold cannot be right, because SE scales with 1/sqrt(n) and a
curveball start is 8 pitches while a fastball start is 34. So the useful
output is the multiplier, not the inches.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np

A = load_np()
d = pd.DataFrame({'pitcher': A['pitcher'], 'game': A['game'], 'season': A['season'],
                  'pt': A['pt'], 'velo': A['velo'], 'spin': A['spin'],
                  'ivb': A['ivb'], 'hb': A['hb_s']})
METRICS = ['velo', 'spin', 'ivb', 'hb']
g = d.groupby(['pitcher', 'season', 'pt', 'game'])
st = g.agg(n=('velo', 'size'), **{f'm_{m}': (m, 'mean') for m in METRICS},
           **{f's_{m}': (m, 'std') for m in METRICS}).reset_index()
st = st[st.n >= 6]
szn = d.groupby(['pitcher', 'season', 'pt']).agg(
    szn_n=('velo', 'size'), **{f'z_{m}': (m, 'mean') for m in METRICS}).reset_index()
st = st.merge(szn, on=['pitcher', 'season', 'pt'])
st = st[st.szn_n >= 50]
for m in METRICS:
    st[f'se_{m}'] = st[f's_{m}'] / np.sqrt(st.n)
    st[f'd_{m}'] = st[f'm_{m}'] - st[f'z_{m}']

print(f'{len(st):,} pitcher-game-pitchtype starts (>=6 in start, >=50 in season)\n')
print('=== within-start SD, and the SE it implies at typical n ===')
rows = []
for pt in ('FF', 'SI', 'SL', 'ST', 'CH', 'CU', 'FC'):
    q = st[st.pt == pt]
    if len(q) < 300:
        continue
    r = {'pt': pt, 'starts': len(q), 'med_n': q.n.median()}
    for m in METRICS:
        r[f'SD_{m}'] = q[f's_{m}'].median()
        r[f'SE_{m}'] = q[f'se_{m}'].median()
        r[f'SDd_{m}'] = q[f'd_{m}'].std()
    rows.append(r)
t = pd.DataFrame(rows).set_index('pt')
for m in METRICS:
    print(f'\n--- {m} ---   SD = pitch-to-pitch within a start;'
          f'  SE = SD/sqrt(n) at that start size;  SD_delta = spread of start-vs-season')
    print(t[['starts', 'med_n', f'SD_{m}', f'SE_{m}', f'SDd_{m}']].round(2).to_string())

print('\n=== what fraction of printed deltas would survive a k*SE filter ===')
for k in (1.0, 1.5, 2.0):
    line = {}
    for m in METRICS:
        line[m] = (st[f'd_{m}'].abs() > k * st[f'se_{m}']).mean()
    print(f'  k={k}: ' + '  '.join(f'{m} {v*100:.0f}%' for m, v in line.items()))

print('\n=== signal share: 1 - (SE^2 / SD_delta^2) ===')
print('    the fraction of delta variance that is NOT measurement noise')
for m in METRICS:
    se2 = (st[f'se_{m}'] ** 2).mean()
    sd2 = st[f'd_{m}'].var()
    print(f'  {m:5s} {1 - se2/sd2:.2f}')
