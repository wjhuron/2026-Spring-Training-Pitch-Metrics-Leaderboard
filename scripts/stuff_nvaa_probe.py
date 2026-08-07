#!/usr/bin/env python3
"""Would nVAA (location-adjusted VAA) beat raw VAA in Stuff+ v11?

Raw per-pitch VAA is location-confounded (higher pitch = flatter approach),
so in a model with no location features it doubles as a pitch-height proxy —
contamination for a skill-isolation model, and double-counting against Loc+.
nVAA removes the height component with the per-type slope, exactly like the
site's nVAA.

Test: pre-transform every pitch's VAA to its location-adjusted version
(VAA - slope_pt * (PlateZ - league mean PlateZ_pt), slopes fit on MLB), then
rerun the paired pitcher-grouped OOF harness. vaa and vaa_diff both flow
through build_df, so the swap is complete and automatic.

Metrics per variant: unit-level descriptive / reliability / predictive
(same as scripts/stuff_velodiff_probe.py) PLUS Stuff-vs-Loc independence:
corr(unit OOF stuff rawmean, unit locPlusRaw from the pitch leaderboard).
Lower is cleaner separation; by the house scope rule, prediction gained
through location leakage does not count for a skill-isolation model.

Usage: python3 scripts/stuff_nvaa_probe.py
"""
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus_v11'))
from train_stuff_v11 import (build_df, design, BASE_FEATS, _params_for,
                             sf, SUPPORTED)  # noqa: E402

PICKLE = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
PITCH_LB = os.path.join(ROOT, 'data', 'pitch_leaderboard_rs.json')

FAMILY = {'FF': 'fastball', 'SI': 'fastball', 'FC': 'cutter',
          'SL': 'breaking', 'ST': 'breaking', 'SV': 'breaking',
          'CU': 'breaking', 'KC': 'breaking',
          'CH': 'offspeed', 'FS': 'offspeed'}


def fit_slopes(mlb):
    """Per-type OLS slope of VAA on PlateZ + league mean PlateZ, MLB only."""
    acc = defaultdict(list)
    for p in mlb:
        pt = p.get('Pitch Type')
        v, z = sf(p.get('VAA')), sf(p.get('PlateZ'))
        if pt in SUPPORTED and v is not None and z is not None:
            acc[pt].append((z, v))
    out = {}
    for pt, rows in acc.items():
        if len(rows) < 2000:
            continue
        z = np.array([r[0] for r in rows])
        v = np.array([r[1] for r in rows])
        slope = float(np.cov(z, v)[0, 1] / np.var(z))
        out[pt] = (slope, float(z.mean()))
        print(f'  {pt}: slope {slope:.3f} deg/ft (n={len(rows)}), '
              f'lg PlateZ {z.mean():.2f}')
    return out


def transform_nvaa(pitches, slopes):
    """Copy of pitch dicts with VAA replaced by its location-adjusted value."""
    out = []
    for p in pitches:
        pt = p.get('Pitch Type')
        v, z = sf(p.get('VAA')), sf(p.get('PlateZ'))
        if pt in slopes and v is not None and z is not None:
            slope, zbar = slopes[pt]
            q = dict(p)
            q['VAA'] = v - slope * (z - zbar)
            out.append(q)
        else:
            out.append(p)
    return out


def unit_metrics(train, raw_col, loc_map):
    out = {}
    d = train.dropna(subset=[raw_col, 'target_xrv'])
    rows = []
    for key, sub in d.groupby(['pitcher', 'team', 'pitch_type']):
        if len(sub) < 50:
            continue
        a, b = sub.iloc[0::2], sub.iloc[1::2]
        rows.append({'fam': FAMILY.get(key[2], 'other'),
                     'raw': sub[raw_col].mean(), 'tgt': -sub['target_xrv'].mean(),
                     'raw_a': a[raw_col].mean(), 'raw_b': b[raw_col].mean(),
                     'tgt_a': -a['target_xrv'].mean(), 'tgt_b': -b['target_xrv'].mean(),
                     'loc': loc_map.get(key)})
    u = pd.DataFrame(rows)
    for scope, m in (('ALL', np.ones(len(u), bool)),
                     ('fastball', (u['fam'] == 'fastball').values),
                     ('breaking', (u['fam'] == 'breaking').values)):
        s = u[m]
        desc = float(np.corrcoef(s['raw'], s['tgt'])[0, 1])
        rel = float(np.corrcoef(s['raw_a'], s['raw_b'])[0, 1])
        pred = float(np.mean([np.corrcoef(s['raw_a'], s['tgt_b'])[0, 1],
                              np.corrcoef(s['raw_b'], s['tgt_a'])[0, 1]]))
        sl = s.dropna(subset=['loc'])
        indep = float(np.corrcoef(sl['raw'], sl['loc'])[0, 1]) if len(sl) > 50 else float('nan')
        out[scope] = (len(s), desc, rel, pred, indep)
    return out


def run_variant(label, pitches, loc_map):
    from sklearn.model_selection import GroupKFold
    df = build_df(pitches)
    df = df[df['arm_angle'].notna()].reset_index(drop=True)
    train = df.dropna(subset=['target_xrv']).reset_index(drop=True)
    y = train['target_xrv'].values
    X = design(train).reindex(columns=BASE_FEATS + ['platoon_same'], fill_value=0)
    oof = np.full(len(train), np.nan)
    gkf = GroupKFold(n_splits=6)
    for k, (tr, te) in enumerate(gkf.split(X, y, train['pitcher'].values)):
        m = xgb.XGBRegressor(**_params_for(X))
        m.fit(X.iloc[tr], y[tr])
        oof[te] = -m.predict(X.iloc[te])
        print(f'  {label} fold {k + 1}/6 done')
    train['raw_v'] = oof
    for scope, (n, desc, rel, pred, indep) in unit_metrics(train, 'raw_v', loc_map).items():
        print(f'  {label:<9} {scope:<9} units {n:>4} | desc {desc:.4f} | rel {rel:.4f} '
              f'| pred {pred:.4f} | corr w/ Loc+ {indep:.4f}')


def main():
    print('Loading ...')
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)
    mlb = [p for p in allp if p.get('_source') == 'MLB']
    with open(PITCH_LB) as f:
        lb = json.load(f)
    loc_map = {(r['pitcher'], r['team'], r['pitchType']): r.get('locPlusRaw')
               for r in lb if r.get('locPlusRaw') is not None}
    # locPlusRaw is expected RV where LOWER = better; flip so positive corr =
    # stuff and location skill aligned.
    loc_map = {k: -v for k, v in loc_map.items()}

    print('Fitting per-type VAA ~ PlateZ slopes ...')
    slopes = fit_slopes(mlb)
    print('\n=== BASELINE (raw VAA) ===')
    run_variant('BASELINE', mlb, loc_map)
    print('\n=== NVAA (location-adjusted VAA) ===')
    run_variant('NVAA', transform_nvaa(mlb, slopes), loc_map)


if __name__ == '__main__':
    main()
