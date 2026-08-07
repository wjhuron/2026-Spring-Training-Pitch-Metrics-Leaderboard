#!/usr/bin/env python3
"""Does velo_diff (gap off the fastball) carry REAL signal for breaking balls?

Wally's challenge: a curveball should not be dinged for being thrown hard;
separation logic belongs to changeups/splitters that mimic fastballs. Note
the trainer's monotone constraint already forces raw velocity to help, so any
"harder same-pitch = worse" behavior can only enter through velo_diff.

Three stages:
  1. COUNTERFACTUAL (shipped bundle): bump each pitch +1 mph two ways —
     (a) velocity and velo_diff move together (the real within-pitcher
     counterfactual: same arsenal, harder breaking ball), and
     (b) velocity only (what the pitch would gain if the gap feature didn't
     exist). Reported in Stuff+ points per pitch type. If (a) < 0 for
     breaking balls, the model literally punishes throwing them harder.
  2. LEARNED DEPENDENCE (shipped bundle): mean SHAP of velo_diff binned by
     velo_diff, per family — what shape the model actually learned.
  3. ABLATION (--ablate, retrains): pitcher-grouped OOF, identical folds and
     params, BASE_FEATS vs velo_diff MASKED (NaN) on every non-CH/FS pitch.
     Unit-level (pitcher, team, type, n>=50) metrics: descriptive corr with
     same-season unit xRV, split-half reliability, and split-half predictive
     corr (raw half A vs actual target half B). If masking costs nothing on
     breaking balls, the feature is a free-rider there.

Usage:
    python3 scripts/stuff_velodiff_probe.py            # stages 1-2 (fast)
    python3 scripts/stuff_velodiff_probe.py --ablate   # stage 3 (long)
"""
import argparse
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
from train_stuff_v11 import (build_df, design, K_SCALE, BASE_FEATS,
                             _params_for)  # noqa: E402

PICKLE = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
BUNDLE = os.path.join(ROOT, 'stuff_plus_v11', 'stuff_models_v11.pkl')

FAMILY = {'FF': 'fastball', 'SI': 'fastball', 'FC': 'cutter',
          'SL': 'breaking', 'ST': 'breaking', 'SV': 'breaking',
          'CU': 'breaking', 'KC': 'breaking',
          'CH': 'offspeed', 'FS': 'offspeed'}
OFFSPEED = {'CH', 'FS'}


def load():
    with open(BUNDLE, 'rb') as f:
        B = pickle.load(f)
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)
    mlb = [p for p in allp if p.get('_source') == 'MLB']
    df = build_df(mlb)
    df = df[df['arm_angle'].notna()].reset_index(drop=True)
    return B, df


def stage12(B, df):
    feats = B['features']
    booster = B['model'].get_booster()
    league = B['league']

    print('=== STAGE 1: +1 mph counterfactual (Stuff+ points, mean per type) ===')
    X = design(df).reindex(columns=feats, fill_value=0)
    base_pred = booster.predict(xgb.DMatrix(X))
    Xa = X.copy()
    Xa['velocity'] += 1.0
    Xa['velo_diff'] += 1.0
    pa = booster.predict(xgb.DMatrix(Xa))
    Xb = X.copy()
    Xb['velocity'] += 1.0
    pb = booster.predict(xgb.DMatrix(Xb))
    d_joint = -(pa - base_pred)     # raw-space change (higher = better)
    d_velo = -(pb - base_pred)
    print(f'{"type":<5}{"n":>8}{"joint(+velo,+gap)":>20}{"velocity-only":>16}{"gap term":>12}')
    for pt, sub in df.groupby('pitch_type'):
        sc = league.get(pt)
        if not sc or not sc.get('sd') or len(sub) < 2000:
            continue
        k = K_SCALE / sc['sd']
        j = float(np.mean(d_joint[sub.index.values]) * k)
        v = float(np.mean(d_velo[sub.index.values]) * k)
        print(f'{pt:<5}{len(sub):>8}{j:>20.2f}{v:>16.2f}{j - v:>12.2f}')

    print('\n=== STAGE 2: learned velo_diff dependence (mean SHAP in Stuff+ pts) ===')
    rng = np.random.RandomState(7)
    idx = rng.choice(len(df), size=min(120000, len(df)), replace=False)
    sub = df.iloc[idx]
    Xs = design(sub).reindex(columns=feats, fill_value=0)
    contrib = booster.predict(xgb.DMatrix(Xs), pred_contribs=True)
    j_vd = feats.index('velo_diff')
    shap_vd = -contrib[:, j_vd]
    fam = sub['pitch_type'].map(FAMILY).values
    vd = sub['velo_diff'].values
    bins = [(-18, -14), (-14, -11), (-11, -8), (-8, -5), (-5, -2), (-2, 1)]
    hdr = 'family     ' + ''.join(f'{f"[{a},{b})":>12}' for a, b in bins)
    print(hdr + '   (velo_diff bin, mph)')
    for f in ('breaking', 'offspeed', 'cutter', 'fastball'):
        m = fam == f
        cells = []
        for a, b in bins:
            mm = m & (vd >= a) & (vd < b)
            if mm.sum() >= 300:
                sd_med = np.median([B['league'][pt]['sd'] for pt in
                                    sub.loc[mm, 'pitch_type'].unique()
                                    if pt in B['league']])
                cells.append(f'{np.mean(shap_vd[mm]) * K_SCALE / sd_med:>12.2f}')
            else:
                cells.append(f'{"–":>12}')
        print(f'{f:<11}' + ''.join(cells))


def unit_metrics(df, raw_col):
    """Unit-level descriptive / reliability / predictive metrics."""
    out = {}
    df = df.dropna(subset=[raw_col, 'target_xrv'])
    units = df.groupby(['pitcher', 'team', 'pitch_type'])
    rows = []
    for key, sub in units:
        if len(sub) < 50:
            continue
        half_a, half_b = sub.iloc[0::2], sub.iloc[1::2]
        rows.append({'pt': key[2], 'fam': FAMILY.get(key[2], 'other'),
                     'raw': sub[raw_col].mean(), 'tgt': -sub['target_xrv'].mean(),
                     'raw_a': half_a[raw_col].mean(), 'raw_b': half_b[raw_col].mean(),
                     'tgt_a': -half_a['target_xrv'].mean(),
                     'tgt_b': -half_b['target_xrv'].mean()})
    u = pd.DataFrame(rows)
    for scope, mask in (('ALL', np.ones(len(u), bool)),
                        ('breaking', (u['fam'] == 'breaking').values),
                        ('offspeed', (u['fam'] == 'offspeed').values)):
        s = u[mask]
        desc = float(np.corrcoef(s['raw'], s['tgt'])[0, 1])
        rel = float(np.corrcoef(s['raw_a'], s['raw_b'])[0, 1])
        pred = float(np.mean([np.corrcoef(s['raw_a'], s['tgt_b'])[0, 1],
                              np.corrcoef(s['raw_b'], s['tgt_a'])[0, 1]]))
        out[scope] = (len(s), desc, rel, pred)
    return out


def stage3(df):
    print('=== STAGE 3: ablation — velo_diff masked for non-offspeed ===')
    from sklearn.model_selection import GroupKFold
    train = df.dropna(subset=['target_xrv']).reset_index(drop=True)
    y = train['target_xrv'].values
    groups = train['pitcher'].values
    X_base = design(train).reindex(columns=BASE_FEATS + ['platoon_same'],
                                   fill_value=0)
    X_mask = X_base.copy()
    non_off = ~train['pitch_type'].isin(OFFSPEED).values
    X_mask.loc[non_off, 'velo_diff'] = np.nan

    for label, Xv in (('BASELINE', X_base), ('MASKED', X_mask)):
        oof = np.full(len(train), np.nan)
        gkf = GroupKFold(n_splits=6)
        for k, (tr, te) in enumerate(gkf.split(Xv, y, groups)):
            m = xgb.XGBRegressor(**_params_for(Xv))
            m.fit(Xv.iloc[tr], y[tr])
            oof[te] = -m.predict(Xv.iloc[te])
            print(f'  {label} fold {k + 1}/6 done')
        train[f'raw_{label}'] = oof
        mets = unit_metrics(train, f'raw_{label}')
        for scope, (n, desc, rel, pred) in mets.items():
            print(f'  {label:<9} {scope:<9} units {n:>4} | desc {desc:.4f} '
                  f'| rel {rel:.4f} | pred {pred:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablate', action='store_true')
    args = ap.parse_args()
    print('Loading ...')
    B, df = load()
    if args.ablate:
        stage3(df)
    else:
        stage12(B, df)


if __name__ == '__main__':
    main()
