"""stuff_lab5.py — round 5: hyperparameter tuning of the lean+vaa winner.

Pooled single-xRV XGB on the lean+vaa feature set. Grid over tree depth and
min_child_weight (the main bias/variance knobs), then tree count, judged by
out-of-sample pred_xRV (luck-neutral) and split-half reliability. Round where
flat; don't chase sub-noise gains.
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
LEAN_VAA = CORE + ['spin_rate', 'extension', 'arm_angle', 'rel_z', 'vaa', 'vaa_diff']

def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5: return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0: return None
    return float(np.corrcoef(xs, ys)[0, 1])

def oof(df, X, params):
    y = df['target_xrv'].values; groups = df['pitcher'].values
    out = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        m = xgb.XGBRegressor(**params); m.fit(X.iloc[tr], y[tr]); out[te] = m.predict(X.iloc[te])
    return -out

def evaluate(df, stuff, late):
    d = df.copy(); d['stuff'] = stuff
    a0, a1 = [], []; est = {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= MIN_HALF and len(h1) >= MIN_HALF:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
        if key in late: est[key] = grp[grp.period == 'early'].stuff.mean()
    rel = pearson(a0, a1); ks = list(est)
    pr = pearson([est[k] for k in ks], [late[k] for k in ks])
    return rel, (-pr if pr is not None else None)

def main():
    df = pd.read_pickle(DF)
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    late = {}
    for key, grp in df.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= MIN_PERIOD and len(l) >= MIN_PERIOD:
            late[key] = l['target_xrv'].dropna().mean()
    dum = pd.get_dummies(df['pitch_type'], prefix='pt')
    X = pd.concat([df[LEAN_VAA].reset_index(drop=True), dum.reset_index(drop=True),
                   df[['platoon_same']].reset_index(drop=True)], axis=1)
    print(f"loaded {len(df)} pitches; {len(late)} pred units")
    print(f"\n{'depth n  mcw  lr  lambda':30s} {'reliab':>7s} {'pred_xRV':>8s}")
    print('-'*50)
    base = dict(subsample=0.8, colsample_bytree=0.8, n_jobs=-1, tree_method='hist')
    grid = []
    for depth in (3, 4, 5, 6):
        for mcw in (5, 20):
            grid.append(dict(base, max_depth=depth, min_child_weight=mcw,
                             n_estimators=500, learning_rate=0.04, reg_lambda=1.5))
    # tree-count / lr at depth 4
    for n, lr in [(300, 0.05), (800, 0.025), (1200, 0.02)]:
        grid.append(dict(base, max_depth=4, min_child_weight=10,
                         n_estimators=n, learning_rate=lr, reg_lambda=1.5))
    for p in grid:
        rel, pr = evaluate(df, oof(df, X, p), late)
        tag = f"d{p['max_depth']} n{p['n_estimators']} mcw{p['min_child_weight']} lr{p['learning_rate']} L{p['reg_lambda']}"
        print(f"{tag:30s} {rel:7.3f} {pr:8.3f}", flush=True)
    print('-'*50)
    print("higher=better; round where flat.")

if __name__ == '__main__':
    main()
