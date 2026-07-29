"""stuff_lab3.py — round 3: sharpen the run-prediction read.

Primary predictive target switches to LATE luck-neutral xRV (late mean
target_xrv, which uses xwOBA for BIP), much less noisy than raw RunExp.
Adds a persistence baseline (early actual xRV -> late xRV = the ceiling for
predicting future run prevention) and a velocity-only baseline. Tight feature
sweep around lean / lean+vaa, all on pooled single-xRV XGB, out-of-fold.
"""
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

DF = '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/2c999aee-7a23-428c-9672-8140b8b4d58d/scratchpad/stuff_df.pkl'
N_FOLDS = 4
MIN_HALF, MIN_PERIOD = 40, 50

CORE = ['velocity', 'ivb', 'hb', 'velo_diff', 'ivb_diff', 'hb_diff']
LEAN = CORE + ['spin_rate', 'extension', 'arm_angle', 'rel_z']
LEAN_VAA = LEAN + ['vaa', 'vaa_diff']
LEAN_VAA_HAA = LEAN_VAA + ['haa', 'rel_x']
LEAN_VAA_SSW = LEAN_VAA + ['ivb_oe', 'hb_oe']
CORE_VAA = CORE + ['vaa', 'vaa_diff']
LEAN_VAA_MOV = LEAN_VAA + ['perceived_velo', 'total_mov']
REG = dict(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.5, n_jobs=-1, tree_method='hist')

def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5: return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0: return None
    return float(np.corrcoef(xs, ys)[0, 1])

def oof_pooled(df, feats):
    dum = pd.get_dummies(df['pitch_type'], prefix='pt')
    X = pd.concat([df[feats].reset_index(drop=True), dum.reset_index(drop=True),
                   df[['platoon_same']].reset_index(drop=True)], axis=1)
    y = df['target_xrv'].values; groups = df['pitcher'].values
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        m = xgb.XGBRegressor(**REG); m.fit(X.iloc[tr], y[tr]); oof[te] = m.predict(X.iloc[te])
    return -oof

def build_late_targets(df):
    late = {}
    for key, grp in df.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= MIN_PERIOD and len(l) >= MIN_PERIOD:
            xv = l['target_xrv'].dropna()
            sw = l['is_swing'].sum()
            late[key] = {
                'late_xrv': xv.mean() if len(xv) else np.nan,
                'late_whf': l['is_whiff'].sum()/sw if sw >= 20 else np.nan,
                'early_xrv': e['target_xrv'].dropna().mean(),
            }
    return late

def evaluate(df, stuff, late):
    d = df.copy(); d['stuff'] = stuff
    a0, a1 = [], []
    est = {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= MIN_HALF and len(h1) >= MIN_HALF:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
        if key in late:
            est[key] = grp[grp.period == 'early'].stuff.mean()
    rel = pearson(a0, a1)
    ks = list(est)
    pr_xrv = pearson([est[k] for k in ks], [late[k]['late_xrv'] for k in ks])
    pr_whf = pearson([est[k] for k in ks], [late[k]['late_whf'] for k in ks])
    return rel, (-pr_xrv if pr_xrv is not None else None), pr_whf, len(a0), len(ks)

def main():
    df = pd.read_pickle(DF)
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    late = build_late_targets(df)
    print(f"loaded {len(df)} pitches; {len(late)} (pitcher,type) with both periods")
    # persistence ceiling: early actual xRV -> late xRV
    ks = list(late)
    persist = pearson([late[k]['early_xrv'] for k in ks], [late[k]['late_xrv'] for k in ks])
    print(f"\nBASELINE persistence (early actual xRV -> late xRV): pred_xRV = {-persist:.3f}")
    print(f"\n{'feature set':20s} {'#f':>3s} {'reliab':>7s} {'pred_xRV':>8s} {'pred_whf':>8s}")
    print('-'*52)
    sweep = [
        ('velocity only', ['velocity']),
        ('core6', CORE),
        ('core+vaa', CORE_VAA),
        ('lean10', LEAN),
        ('lean+vaa', LEAN_VAA),
        ('lean+vaa+haa', LEAN_VAA_HAA),
        ('lean+vaa+ssw', LEAN_VAA_SSW),
        ('lean+vaa+mov', LEAN_VAA_MOV),
    ]
    for name, feats in sweep:
        stuff = oof_pooled(df, feats)
        rel, pxrv, pwhf, nr, npd = evaluate(df, stuff, late)
        print(f"{name:20s} {len(feats):3d} {rel:7.3f} {pxrv:8.3f} {pwhf:8.3f}", flush=True)
    print('-'*52)
    print(f"n_rel~{nr}  n_pred~{npd}   higher=better. pred_xRV target is luck-neutral.")

if __name__ == '__main__':
    main()
