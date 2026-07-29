"""stuff_lab6.py — final 'leave no stone unturned' round.

On the tuned lean+vaa winner, test the last genuinely-untried ideas:
  - axis_dev: spin-axis deviation (OTilt vs RTilt), a distinct SSW proxy
  - monotonic velocity constraint (robustness: higher velo never lowers stuff)
  - explicit platoon split (separate vs-same / vs-opposite models)
Tuned params: depth 4, 800 trees, lr 0.025, mcw 10, lambda 1.5.
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
TUNED = dict(max_depth=4, n_estimators=800, learning_rate=0.025, min_child_weight=10,
             reg_lambda=1.5, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, tree_method='hist')

def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5: return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0: return None
    return float(np.corrcoef(xs, ys)[0, 1])

def make_X(df, feats):
    dum = pd.get_dummies(df['pitch_type'], prefix='pt')
    return pd.concat([df[feats].reset_index(drop=True), dum.reset_index(drop=True),
                      df[['platoon_same']].reset_index(drop=True)], axis=1)

def oof(df, feats, params, mono_feat=None, platoon_split=False):
    y = df['target_xrv'].values; groups = df['pitcher'].values
    out = np.full(len(df), np.nan)
    if platoon_split:
        for pv in (0, 1):
            idx = np.where(df['platoon_same'].values == pv)[0]
            sub = df.iloc[idx]
            X = make_X(sub, feats); ys = sub['target_xrv'].values; gs = sub['pitcher'].values
            for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, ys, gs):
                m = xgb.XGBRegressor(**params); m.fit(X.iloc[tr], ys[tr]); out[idx[te]] = m.predict(X.iloc[te])
        return -out
    X = make_X(df, feats)
    p = dict(params)
    if mono_feat:
        cons = tuple(-1 if c == mono_feat else 0 for c in X.columns)
        p['monotone_constraints'] = cons
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        m = xgb.XGBRegressor(**p); m.fit(X.iloc[tr], y[tr]); out[te] = m.predict(X.iloc[te])
    return -out

def evaluate(df, stuff, late):
    d = df.copy(); d['stuff'] = stuff
    a0, a1 = [], []; est = {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= MIN_HALF and len(h1) >= MIN_HALF:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
        if key in late: est[key] = grp[grp.period == 'early'].stuff.mean()
    ks = list(est)
    return pearson(a0, a1), -pearson([est[k] for k in ks], [late[k] for k in ks])

def main():
    df = pd.read_pickle(DF)
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    late = {}
    for key, grp in df.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= MIN_PERIOD and len(l) >= MIN_PERIOD:
            late[key] = l['target_xrv'].dropna().mean()
    print(f"loaded {len(df)} pitches; {len(late)} pred units")
    print(f"\n{'config':32s} {'reliab':>7s} {'pred_xRV':>8s}")
    print('-'*50)
    runs = [
        ('lean+vaa (tuned baseline)', dict(feats=LEAN_VAA, params=TUNED)),
        ('  + axis_dev', dict(feats=LEAN_VAA + ['axis_dev'], params=TUNED)),
        ('  + monotonic velocity', dict(feats=LEAN_VAA, params=TUNED, mono_feat='velocity')),
        ('  platoon-split models', dict(feats=LEAN_VAA, params=TUNED, platoon_split=True)),
        ('  + axis_dev + mono velo', dict(feats=LEAN_VAA + ['axis_dev'], params=TUNED, mono_feat='velocity')),
    ]
    for name, kw in runs:
        rel, pr = evaluate(df, oof(df, **kw), late)
        print(f"{name:32s} {rel:7.3f} {pr:8.3f}", flush=True)
    print('-'*50)
    print("higher=better. baseline to beat: ~0.230 pred / 0.867 reliab")

if __name__ == '__main__':
    main()
