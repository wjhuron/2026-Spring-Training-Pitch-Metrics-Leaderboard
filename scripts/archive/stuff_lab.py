"""stuff_lab.py — Stuff+ model experimentation harness.

Loads the cached dataset (scripts/stuff_data.py) and trains candidate models
out-of-fold (GroupKFold by pitcher, no leakage), then evaluates each on three
objectives at the (pitcher x pitch-type) level:

  reliability : split-half (alternating game dates) Pearson of mean stuff
  predict_rv  : early-period stuff -> late-period run value allowed (rv100);
                reported sign-flipped so higher = better (more stuff => fewer runs)
  predict_whf : early-period stuff -> late-period whiff/swing rate
  descr_rv    : in-sample (OOF) stuff vs rv100  (descriptive, sign-flipped)

Variants sweep: target (single xRV vs learned-weight components), structure
(pooled single model vs per-pitch-type), feature set (lean / full / +SSW).
"""
import os, sys, math, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

DF = '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/2c999aee-7a23-428c-9672-8140b8b4d58d/scratchpad/stuff_df.pkl'
N_FOLDS = 4
MIN_HALF = 40
MIN_PERIOD = 50
MIN_PT_TRAIN = 500

LEAN = ['velocity', 'ivb', 'hb', 'spin_rate', 'extension', 'arm_angle', 'rel_z',
        'velo_diff', 'ivb_diff', 'hb_diff']
FULL = LEAN + ['perceived_velo', 'vaa', 'haa', 'rel_x', 'vaa_diff',
               'total_mov', 'mov_angle', 'spin_per_mph']
SSW = FULL + ['ivb_oe', 'hb_oe']

XGB_REG = dict(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8,
               colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.5,
               n_jobs=-1, tree_method='hist')
XGB_CLF = dict(XGB_REG, objective='binary:logistic', eval_metric='logloss')

def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5: return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0: return None
    return float(np.corrcoef(xs, ys)[0, 1])

def design(df, feats, structure):
    X = df[feats].copy()
    if structure == 'pooled':
        dum = pd.get_dummies(df['pitch_type'], prefix='pt')
        X = pd.concat([X.reset_index(drop=True), dum.reset_index(drop=True),
                       df[['platoon_same']].reset_index(drop=True)], axis=1)
    return X

def oof_regression(df, feats, structure, target='target_xrv'):
    oof = np.full(len(df), np.nan)
    groups = df['pitcher'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    if structure == 'pooled':
        X = design(df, feats, 'pooled'); y = df[target].values
        for tr, te in gkf.split(X, y, groups):
            m = xgb.XGBRegressor(**XGB_REG)
            m.fit(X.iloc[tr], y[tr]); oof[te] = m.predict(X.iloc[te])
    else:
        for pt in df['pitch_type'].unique():
            idx = np.where((df['pitch_type'] == pt).values)[0]
            sub = df.iloc[idx]
            if len(sub) < MIN_PT_TRAIN:
                oof[idx] = sub[target].mean(); continue
            Xs = sub[feats].reset_index(drop=True); ys = sub[target].values
            gs = sub['pitcher'].values
            for tr, te in gkf.split(Xs, ys, gs):
                m = xgb.XGBRegressor(**XGB_REG)
                m.fit(Xs.iloc[tr], ys[tr]); oof[idx[te]] = m.predict(Xs.iloc[te])
    return oof   # predicted xRV (lower = better stuff)

def oof_components(df, feats, structure):
    """Whiff prob, GB prob, contact (xwOBA) regressions -> OOF, then learn the
    linear combo that best predicts target_xrv. Returns predicted xRV-equivalent
    (lower = better) so it slots into the same evaluation."""
    n = len(df); groups = df['pitcher'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    whf = np.full(n, np.nan); gb = np.full(n, np.nan); con = np.full(n, np.nan)
    swing = df['is_swing'].values == 1
    bip = df['is_bip'].values == 1
    def run(mask, ycol, clf, out):
        idx = np.where(mask)[0]
        if len(idx) < MIN_PT_TRAIN: return
        sub = df.iloc[idx]; X = design(sub, feats, structure).reset_index(drop=True)
        y = sub[ycol].values; gs = sub['pitcher'].values
        for tr, te in gkf.split(X, y, gs):
            M = (xgb.XGBClassifier(**XGB_CLF) if clf else xgb.XGBRegressor(**XGB_REG))
            M.fit(X.iloc[tr], y[tr])
            pred = M.predict_proba(X.iloc[te])[:, 1] if clf else M.predict(X.iloc[te])
            out[idx[te]] = pred
    run(swing, 'is_whiff', True, whf)            # whiff | swing
    run(bip, 'is_gb', True, gb)                  # gb | bip
    run(bip & df['xwoba'].notna().values, 'xwoba', False, con)  # contact xwoba | bip
    # learn weights: regress target_xrv on the three component predictions
    comp = pd.DataFrame({'whf': whf, 'gb': gb, 'con': con, 'y': df['target_xrv'].values})
    comp = comp.dropna(subset=['y'])
    feat_cols = ['whf', 'gb', 'con']
    Xc = comp[feat_cols].fillna(comp[feat_cols].mean())
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(Xc, comp['y'])
    full = pd.DataFrame({'whf': whf, 'gb': gb, 'con': con})
    full = full.fillna(full.mean())
    return lr.predict(full[feat_cols]), dict(zip(feat_cols, lr.coef_))

def standardize_by_type(df, stuff_raw):
    """stuff_raw higher = better. z-score within pitch type -> 100 + 10z."""
    s = pd.Series(stuff_raw, index=df.index)
    out = np.full(len(df), 100.0)
    for pt, idx in df.groupby('pitch_type').groups.items():
        v = s.loc[idx]; mu, sd = v.mean(), v.std()
        if sd > 0: out[df.index.get_indexer(idx)] = 100 + 10 * (v - mu) / sd
    return out

def evaluate(df, stuff_raw):
    d = df.copy(); d['stuff'] = stuff_raw   # higher = better stuff
    g = d.groupby(['pitcher', 'throws', 'pitch_type'])
    a0, a1 = [], []
    e_st, l_rv, l_wh = {}, {}, {}
    f_st, f_rv, f_wh = [], [], []
    for key, grp in g:
        h0 = grp[grp.half == 0]; h1 = grp[grp.half == 1]
        if len(h0) >= MIN_HALF and len(h1) >= MIN_HALF:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
        early = grp[grp.period == 'early']; late = grp[grp.period == 'late']
        if len(early) >= MIN_PERIOD and len(late) >= MIN_PERIOD:
            e_st[key] = early.stuff.mean()
            lv = late['run_value'].dropna(); l_rv[key] = lv.mean()*100 if len(lv) else np.nan
            sw = late['is_swing'].sum(); l_wh[key] = late['is_whiff'].sum()/sw if sw >= 20 else np.nan
        if len(grp) >= MIN_PERIOD:
            f_st.append(grp.stuff.mean())
            rv = grp['run_value'].dropna(); f_rv.append(rv.mean()*100 if len(rv) else np.nan)
            sw = grp['is_swing'].sum(); f_wh.append(grp['is_whiff'].sum()/sw if sw >= 20 else np.nan)
    rel = pearson(a0, a1)
    keys = [k for k in e_st if k in l_rv]
    pr_rv = pearson([e_st[k] for k in keys], [l_rv[k] for k in keys])
    pr_wh = pearson([e_st[k] for k in keys], [l_wh[k] for k in keys])
    de_rv = pearson(f_st, f_rv)
    return {'rel': rel, 'pred_rv': -pr_rv if pr_rv is not None else None,
            'pred_wh': pr_wh, 'descr_rv': -de_rv if de_rv is not None else None,
            'n_rel': len(a0), 'n_pred': len(keys)}

def main():
    df = pd.read_pickle(DF)
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    print(f"loaded {len(df)} pitches, {df['pitcher'].nunique()} pitchers")
    print(f"\n{'variant':34s} {'rel':>6s} {'pred_rv':>8s} {'pred_wh':>8s} {'descr_rv':>9s}")
    print('-'*70)
    variants = [
        ('xrv pooled LEAN', 'xrv', 'pooled', LEAN),
        ('xrv pooled FULL', 'xrv', 'pooled', FULL),
        ('xrv pooled +SSW', 'xrv', 'pooled', SSW),
        ('xrv pertype +SSW', 'xrv', 'pertype', SSW),
        ('components pooled +SSW', 'comp', 'pooled', SSW),
    ]
    for name, target, structure, feats in variants:
        if target == 'xrv':
            oof = oof_regression(df, feats, structure)
            stuff_raw = -oof
        else:
            pred, weights = oof_components(df, feats, structure)
            stuff_raw = -pred
        r = evaluate(df, stuff_raw)
        extra = ''
        if target == 'comp':
            extra = '  w=' + ','.join(f'{k}:{v:+.2f}' for k, v in weights.items())
        print(f"{name:34s} {r['rel']:6.3f} {r['pred_rv']:8.3f} {r['pred_wh']:8.3f} "
              f"{r['descr_rv']:9.3f}{extra}", flush=True)
    print('-'*70)
    print(f"n_rel~{r['n_rel']}  n_pred~{r['n_pred']}")
    print("rel/pred/descr all higher = better; pred_rv/descr_rv sign-flipped (more stuff -> fewer runs)")

if __name__ == '__main__':
    main()
