"""stuff_lab4.py — round 4: is VAA's gain real stuff, or location leakage?

Trains with plate location as a feature and scores at a NEUTRAL location, which
strips the location component out of VAA (and everything else). If lean+vaa's
run-prediction edge survives neutralization, approach angle is genuine stuff;
if it collapses toward lean10, VAA was smuggling location in.

Also reports a location-leak diagnostic: correlation of each model's stuff with
the pitcher's in-zone rate (a pure-stuff metric should not track where he throws).
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
LOC = ['plate_x', 'plate_z_norm']
REG = dict(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.5, n_jobs=-1, tree_method='hist')

def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5: return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0: return None
    return float(np.corrcoef(xs, ys)[0, 1])

def oof(df, feats, loc_feats=None, neutralize=False):
    tf = feats + (loc_feats or [])
    dum = pd.get_dummies(df['pitch_type'], prefix='pt')
    X = pd.concat([df[tf].reset_index(drop=True), dum.reset_index(drop=True),
                   df[['platoon_same']].reset_index(drop=True)], axis=1)
    y = df['target_xrv'].values; groups = df['pitcher'].values
    neutral = {f: df[f].mean() for f in (loc_feats or [])}
    out = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        m = xgb.XGBRegressor(**REG); m.fit(X.iloc[tr], y[tr])
        Xte = X.iloc[te].copy()
        if neutralize and loc_feats:
            for f in loc_feats: Xte[f] = neutral[f]
        out[te] = m.predict(Xte)
    return -out

def evaluate(df, stuff, late, zone):
    d = df.copy(); d['stuff'] = stuff
    a0, a1 = [], []; est = {}; fs = {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= MIN_HALF and len(h1) >= MIN_HALF:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
        if key in late: est[key] = grp[grp.period == 'early'].stuff.mean()
        if len(grp) >= MIN_PERIOD: fs[key] = grp.stuff.mean()
    rel = pearson(a0, a1)
    ks = list(est)
    pr = pearson([est[k] for k in ks], [late[k] for k in ks])
    lk = [k for k in fs if k in zone]
    leak = pearson([fs[k] for k in lk], [zone[k] for k in lk])
    return rel, (-pr if pr is not None else None), leak, len(a0), len(ks)

def main():
    df = pd.read_pickle(DF)
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    df['in_zone'] = (((df['plate_z_norm'] >= 0) & (df['plate_z_norm'] <= 1) &
                      (df['plate_x'].abs() <= 0.83)).astype(int))
    late, zone = {}, {}
    for key, grp in df.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= MIN_PERIOD and len(l) >= MIN_PERIOD:
            late[key] = l['target_xrv'].dropna().mean()
        if len(grp) >= MIN_PERIOD:
            zone[key] = grp['in_zone'].mean()
    print(f"loaded {len(df)} pitches; {len(late)} pred units")
    print(f"\n{'config':30s} {'reliab':>7s} {'pred_xRV':>8s} {'leak(zone)':>10s}")
    print('-'*60)
    configs = [
        ('lean10 (no loc)', LEAN, None, False),
        ('lean+vaa (no loc)', LEAN_VAA, None, False),
        ('lean+vaa loc-NEUTRAL', LEAN_VAA, LOC, True),
        ('lean10 loc-NEUTRAL', LEAN, LOC, True),
        ('lean+vaa loc-IN (Pitching+)', LEAN_VAA, LOC, False),
    ]
    for name, feats, lf, neut in configs:
        s = oof(df, feats, lf, neut)
        rel, pr, leak, nr, npd = evaluate(df, s, late, zone)
        print(f"{name:30s} {rel:7.3f} {pr:8.3f} {leak:10.3f}", flush=True)
    print('-'*60)
    print(f"n_rel~{nr} n_pred~{npd}. pred_xRV/reliab higher=better; |leak| near 0 = location-independent.")

if __name__ == '__main__':
    main()
