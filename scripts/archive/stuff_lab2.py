"""stuff_lab2.py — refined Stuff+ sweep: feature parsimony + SSW isolation +
regularization, on the winning pooled single-xRV structure.

Adds cleaner predictive targets (late whiff/swing and late xwOBAcon) and fixes
the reporting sign (all metrics: higher = better). Everything is out-of-fold
(GroupKFold by pitcher).
"""
import os, math, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

DF = '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/2c999aee-7a23-428c-9672-8140b8b4d58d/scratchpad/stuff_df.pkl'
N_FOLDS = 4
MIN_HALF, MIN_PERIOD, MIN_BIP_LATE = 40, 50, 15

VELO = ['velocity']
CORE = ['velocity', 'ivb', 'hb', 'velo_diff', 'ivb_diff', 'hb_diff']
LEAN = CORE + ['spin_rate', 'extension', 'arm_angle', 'rel_z']
LEAN_SSW = LEAN + ['ivb_oe', 'hb_oe']
LEAN_VAA = LEAN + ['vaa', 'vaa_diff']
FULL = LEAN + ['perceived_velo', 'vaa', 'haa', 'rel_x', 'vaa_diff', 'total_mov', 'mov_angle', 'spin_per_mph']
FULL_SSW = FULL + ['ivb_oe', 'hb_oe']

REG = dict(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.5, n_jobs=-1, tree_method='hist')
REG_HEAVY = dict(REG, max_depth=3, min_child_weight=30, reg_lambda=6.0, colsample_bytree=0.6)

def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5: return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0: return None
    return float(np.corrcoef(xs, ys)[0, 1])

def oof_pooled(df, feats, params):
    dum = pd.get_dummies(df['pitch_type'], prefix='pt')
    X = pd.concat([df[feats].reset_index(drop=True), dum.reset_index(drop=True),
                   df[['platoon_same']].reset_index(drop=True)], axis=1)
    y = df['target_xrv'].values; groups = df['pitcher'].values
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        m = xgb.XGBRegressor(**params); m.fit(X.iloc[tr], y[tr]); oof[te] = m.predict(X.iloc[te])
    return -oof   # stuff_raw, higher = better

def evaluate(df, stuff):
    d = df.copy(); d['stuff'] = stuff
    a0, a1 = [], []
    e_st, l_rv, l_wh, l_xwc = {}, {}, {}, {}
    f_st, f_rv = [], []
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= MIN_HALF and len(h1) >= MIN_HALF:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
        early, late = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(early) >= MIN_PERIOD and len(late) >= MIN_PERIOD:
            e_st[key] = early.stuff.mean()
            rv = late['run_value'].dropna(); l_rv[key] = rv.mean()*100 if len(rv) else np.nan
            sw = late['is_swing'].sum(); l_wh[key] = late['is_whiff'].sum()/sw if sw >= 20 else np.nan
            xb = late.loc[late['xwoba'].notna(), 'xwoba']; l_xwc[key] = xb.mean() if len(xb) >= MIN_BIP_LATE else np.nan
        if len(grp) >= MIN_PERIOD:
            f_st.append(grp.stuff.mean()); rv = grp['run_value'].dropna()
            f_rv.append(rv.mean()*100 if len(rv) else np.nan)
    rel = pearson(a0, a1)
    keys = list(e_st)
    pr_rv = pearson([e_st[k] for k in keys], [l_rv[k] for k in keys])
    pr_wh = pearson([e_st[k] for k in keys], [l_wh[k] for k in keys])
    pr_xwc = pearson([e_st[k] for k in keys], [l_xwc[k] for k in keys])
    de_rv = pearson(f_st, f_rv)
    return (rel, pr_rv, pr_wh, (-pr_xwc if pr_xwc is not None else None), de_rv, len(a0), len(keys))

def main():
    df = pd.read_pickle(DF)
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    print(f"loaded {len(df)} pitches")
    print(f"\n{'variant':24s} {'#f':>3s} {'reliab':>7s} {'pr_RV':>6s} {'pr_whf':>7s} {'pr_xwc':>7s} {'descr':>6s}")
    print('-'*68)
    sweep = [
        ('velocity only', VELO, REG),
        ('core6', CORE, REG),
        ('lean10', LEAN, REG),
        ('lean+SSW', LEAN_SSW, REG),
        ('lean+vaa', LEAN_VAA, REG),
        ('full18', FULL, REG),
        ('full+SSW', FULL_SSW, REG),
        ('full heavy-reg', FULL, REG_HEAVY),
    ]
    for name, feats, params in sweep:
        stuff = oof_pooled(df, feats, params)
        rel, prv, pwh, pxwc, descr, nr, npd = evaluate(df, stuff)
        print(f"{name:24s} {len(feats):3d} {rel:7.3f} {prv:6.3f} {pwh:7.3f} "
              f"{pxwc if pxwc is not None else float('nan'):7.3f} {descr:6.3f}", flush=True)
    print('-'*68)
    print(f"n_rel~{nr}  n_pred~{npd}   all higher=better (pr_xwc=stuff vs late xwOBAcon, sign-flipped)")

if __name__ == '__main__':
    main()
