#!/usr/bin/env python3
"""Measure the DAY-TO-DAY scale for the daily card's Usage / Avg Velo shading.

Follow-up to daily_delta_noise.py (2026-08-13, per Wally: the SE-based
shading saturated on ordinary starts -- Gausman +0.3 mph on 44 FF rendered
full red). The card should color "is this start unusual FOR HIM?", so the
denominator must be the spread of start-vs-season deltas, not the sampling
SE of the start mean.

Model per (pitcher, season, pitch type, game):

  velo:   Var(obs delta) = SD_day^2 + s_within^2 / n
          -> SD_day^2 measured per pitch type as Var(d) - mean(s^2/n),
             i.e. the sampling-free day-to-day component.

  usage:  Var(obs delta) = c * u(1-u) + u(1-u) / tc
          -> c measured as (Var(d_u) - mean(u(1-u)/tc)) / mean(u(1-u)).
             The strategic (game-plan) component is assumed proportional to
             u(1-u), checked per pitch type below.

The card then shades z = delta / sqrt(day component + sampling component at
this start's own n), full color at 2 (the family's existing convention:
full ink = a start in his own ~5% tails).

Both constants are reported per season (2021-2025) as the independent-
replicate stability check before freezing.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np

A = load_np()
d = pd.DataFrame({'pitcher': A['pitcher'], 'game': A['game'], 'season': A['season'],
                  'pt': A['pt'], 'velo': A['velo']})

# Per-start per-type rows
g = d.groupby(['pitcher', 'season', 'game', 'pt'])
st = g.agg(n=('velo', 'size'), m=('velo', 'mean'), s=('velo', 'std')).reset_index()
# Game totals (all types) and season aggregates
tot = d.groupby(['pitcher', 'season', 'game']).size().rename('tc').reset_index()
szn = d.groupby(['pitcher', 'season', 'pt']).agg(
    szn_n=('velo', 'size'), z=('velo', 'mean')).reset_index()
szn_tot = d.groupby(['pitcher', 'season']).size().rename('szn_tc').reset_index()
st = st.merge(tot).merge(szn).merge(szn_tot)
st['u_base'] = st.szn_n / st.szn_tc
st['u_obs'] = st.n / st.tc

# Gates: real starts and established season baselines.
st = st[(st.tc >= 30) & (st.szn_tc >= 500)]

PTS = ('FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS')

print('=== VELO: SD_day per pitch type (sampling-free day-to-day SD, mph) ===')
print('start gate n>=6, season gate >=50 of that type')
v = st[(st.n >= 6) & (st.szn_n >= 50) & st.s.notna()].copy()
v['d'] = v.m - v.z
v['samp'] = v.s ** 2 / v.n
rows = []
for pt in PTS:
    q = v[v.pt == pt]
    if len(q) < 300:
        continue
    r = {'pt': pt, 'starts': len(q)}
    for yr, qq in [('ALL', q)] + [(y, q[q.season == y]) for y in range(2021, 2026)]:
        if len(qq) < 200:
            r[str(yr)] = np.nan; continue
        var_day = qq.d.var() - qq.samp.mean()
        r[str(yr)] = np.sqrt(max(var_day, 0.0))
    rows.append(r)
print(pd.DataFrame(rows).set_index('pt').round(3).to_string())

print('\n=== USAGE: overdispersion c per pitch type (day var = c * u(1-u)) ===')
print('all types with season share >= 3%; every start row (no n gate)')
u = st[st.u_base >= 0.03].copy()
u['d'] = u.u_obs - u.u_base
u['binom'] = u.u_base * (1 - u.u_base) / u.tc
rows = []
for pt in PTS:
    q = u[u.pt == pt]
    if len(q) < 300:
        continue
    r = {'pt': pt, 'starts': len(q), 'med_u': q.u_base.median()}
    for yr, qq in [('ALL', q)] + [(y, q[q.season == y]) for y in range(2021, 2026)]:
        if len(qq) < 200:
            r[str(yr)] = np.nan; continue
        c = (qq.d.var() - qq.binom.mean()) / (qq.u_base * (1 - qq.u_base)).mean()
        r[str(yr)] = c
    rows.append(r)
t = pd.DataFrame(rows).set_index('pt')
print(t.round(4).to_string())

q = u
c_all = (q.d.var() - q.binom.mean()) / (q.u_base * (1 - q.u_base)).mean()
print(f'\npooled c (all types): {c_all:.4f}')
print('implied usage SD at u=40%, tc=93: '
      f'{np.sqrt(0.4*0.6*(c_all + 1/93))*100:.1f} pts '
      f'(binomial alone: {np.sqrt(0.4*0.6/93)*100:.1f} pts)')

print('\n=== sanity: velo z distribution under the new scale ===')
for pt in ('FF', 'SL', 'CH'):
    q = v[v.pt == pt]
    sd_day2 = max(q.d.var() - q.samp.mean(), 0.0)
    z = q.d / np.sqrt(sd_day2 + q.samp)
    print(f'  {pt}: share |z|>=2 = {(z.abs() >= 2).mean()*100:.1f}%  '
          f'(target ~5% if day effects were normal)')
