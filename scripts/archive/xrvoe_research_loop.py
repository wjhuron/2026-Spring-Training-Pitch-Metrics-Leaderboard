"""xrvoe_research_loop.py — mine persistent xRVOE units for missing physics.

Takes the production xRVOE construction (bundle OOF stuff preds + Loc ExpRV,
per-group stacking), keeps units with >=300 pitches whose residual holds the
SAME SIGN in both date-halves (persistent, not lucky), and fingerprints the
top over/under-performers: each unit's physics vs its pitch type's league
mean. If the top group shares a trait the model doesn't see, that trait is a
v12 feature candidate.

Usage: python3 scripts/xrvoe_research_loop.py
"""
import os, sys, json, pickle, math
import numpy as np
import pandas as pd
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_utils import break_tilt_to_minutes

src = open(os.path.join(ROOT, 'stuff_plus_v11', 'train_stuff_v11.py')).read()
T = {'__name__': '_stuff_mod',
     '__file__': os.path.join(ROOT, 'stuff_plus_v11', 'train_stuff_v11.py')}
exec(compile(src.split('def main()')[0], 'train_stuff_v11.py', 'exec'), T)

D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
ep = {(p.get('Pitcher'), p.get('PTeam')) for p in D if p.get('Pitch Type') == 'EP'}
mlb = [p for p in D if p.get('_source', 'MLB') == 'MLB'
       and (p.get('Pitcher'), p.get('PTeam')) not in ep]

d = T['build_df'](mlb)
d = d[d['target_xrv'].notna()].reset_index(drop=True)
B = pickle.load(open(os.path.join(ROOT, 'stuff_plus_v11', 'stuff_models_v11.pkl'), 'rb'))
X = T['design'](d).reindex(columns=B['features'], fill_value=0)
fold_of = {p: k for k, ps in enumerate(B['fold_pitchers']) for p in ps}
pf = np.array([fold_of.get(p, 0) for p in d['pitcher'].values])
pred = np.full(len(d), np.nan)
for k, m in enumerate(B['fold_models']):
    msk = pf == k
    if msk.any():
        pred[msk] = m.predict(X[msk])
d['stuff_hit'] = pred

import pipeline_locplus as L
g26 = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json')))['gutsConstants']
baseline = [p for p in mlb if L.is_eligible_baseline(p)]
S = L.build_surfaces(baseline, g26['lgWOBA'], g26['wOBAScale'])
exprv = {p['PitchID']: v for p in baseline
         if p.get('PitchID') and (v := L.score_pitch(p, S)) is not None}
d['loc_exprv'] = d['pid'].map(exprv)
d = d[d['loc_exprv'].notna()].reset_index(drop=True)
d['grp'] = d['pitch_type'].map(L.group_of_code)

beta_cache = {}
d['expect'] = np.nan
A_all = np.column_stack([np.ones(len(d)), d['stuff_hit'], d['loc_exprv']])
beta_g, *_ = np.linalg.lstsq(A_all, d['target_xrv'].values, rcond=None)
for grp, sub in d.groupby('grp'):
    A = np.column_stack([np.ones(len(sub)), sub['stuff_hit'], sub['loc_exprv']])
    beta = (np.linalg.lstsq(A, sub['target_xrv'].values, rcond=None)[0]
            if len(sub) >= 5000 else beta_g)
    d.loc[sub.index, 'expect'] = A @ beta
d['resid'] = d['target_xrv'] - d['expect']

# axis_dev per pitch from cache tilts (hand-signed)
tilt = {}
for p in mlb:
    pid = p.get('PitchID')
    if not pid:
        continue
    o = break_tilt_to_minutes(p.get('OTilt'))
    r = break_tilt_to_minutes(p.get('RTilt'))
    if o is None or r is None:
        continue
    dv = (o - r) % 720
    if dv >= 360:
        dv -= 720
    s = 1.0 if p.get('Throws') == 'R' else -1.0
    tilt[pid] = s * dv * 0.5
d['axis_dev'] = d['pid'].map(tilt)

order = {dt: i for i, dt in enumerate(sorted(d['date'].dropna().unique()))}
d['half'] = d['date'].map(order).fillna(0).astype(int) % 2

PHYS = ['velocity', 'ivb', 'hb', 'extension', 'arm_angle', 'spin_rate',
        'vaa', 'axis_dev', 'velo_diff']
g = d.groupby(['pitcher', 'pitch_type']).agg(
    resid=('resid', 'mean'), n=('resid', 'size'),
    r0=('resid', lambda x: x[d.loc[x.index, 'half'] == 0].mean()),
    r1=('resid', lambda x: x[d.loc[x.index, 'half'] == 1].mean()),
    **{f: (f, 'mean') for f in PHYS})
g = g[(g['n'] >= 300) & (np.sign(g['r0']) == np.sign(g['r1']))]
lg = d.groupby('pitch_type')[PHYS].mean()

def fingerprint(row, pt):
    out = []
    for f in PHYS:
        v, m = row[f], lg.loc[pt, f]
        if pd.isna(v) or pd.isna(m):
            continue
        out.append(f'{f} {v - m:+.1f}')
    return ', '.join(out)

g = g.sort_values('resid')   # most negative resid = pitcher-beats (hitter persp.)
print(f'persistent units (n>=300, same-sign halves): {len(g)}\n')
print('TOP 15 OVERPERFORMERS (beat stuff+loc expectation, both halves):')
for (pitcher, pt), row in g.head(15).iterrows():
    print(f'  {pitcher:24s} {pt:3s} xRVOE {-row.resid*100:+.2f}/100 (n={int(row.n)})')
    print(f'     vs {pt} league: {fingerprint(row, pt)}')
print('\nTOP 10 UNDERPERFORMERS:')
for (pitcher, pt), row in g.tail(10).iloc[::-1].iterrows():
    print(f'  {pitcher:24s} {pt:3s} xRVOE {-row.resid*100:+.2f}/100 (n={int(row.n)})')
    print(f'     vs {pt} league: {fingerprint(row, pt)}')

# group physics deltas: top-30 vs bottom-30 (velo excluded from headline —
# the known lean) — what ELSE do overperformers share?
top, bot = g.head(30), g.tail(30)
print('\nSHARED-PHYSICS SUMMARY (top-30 overperformers minus bottom-30, '
      'each vs their pitch-type league mean):')
for f in PHYS:
    tvals = [(row[f] - lg.loc[pt, f]) for (pi, pt), row in top.iterrows()
             if not pd.isna(row[f]) and not pd.isna(lg.loc[pt, f])]
    bvals = [(row[f] - lg.loc[pt, f]) for (pi, pt), row in bot.iterrows()
             if not pd.isna(row[f]) and not pd.isna(lg.loc[pt, f])]
    if len(tvals) > 10 and len(bvals) > 10:
        print(f'  {f:12s} over {np.mean(tvals):+7.2f}   under {np.mean(bvals):+7.2f}   '
              f'gap {np.mean(tvals) - np.mean(bvals):+7.2f}')
