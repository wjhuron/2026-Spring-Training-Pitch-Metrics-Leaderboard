"""phase2b_stuff_experiments.py — Phase 2b Stuff+ v11 experiments.

On the shipped v11 feature set (BASE_FEATS), test with lab6's harness:
  1. + rel_x ALONE (previous lab only tested it bundled with HAA; every
     public reference model carries release side)
  2. + monotone velocity constraint (robustness; adopt if flat-or-better)
  3. adjusted (density-corrected) movement in place of raw — ivb/hb and the
     fastball-reference diffs recomputed from xIndVrtBrk/xHorzBrk deltas
  4. winning combinations

Metrics: split-half reliability (>=40 pitches/half per pitcher×type) and
pred_xRV (OOF pitcher-grouped early-period stuff -> late-period mean xRV,
>=50 pitches per period). Matches scripts/stuff_lab6.py conventions.

Usage: STUFF_DF_OUT=... python3 scripts/phase2b_stuff_experiments.py
"""
import os, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF = os.environ.get('STUFF_DF_OUT', os.path.join(ROOT, 'data', '_stuff_df_scratch.pkl'))
N_FOLDS = 8
MIN_HALF, MIN_PERIOD = 40, 50
BASE = ['velocity', 'ivb', 'hb', 'velo_diff', 'ivb_diff', 'hb_diff',
        'spin_rate', 'extension', 'arm_angle', 'rel_z', 'vaa', 'vaa_diff']
TUNED = dict(max_depth=4, n_estimators=800, learning_rate=0.025, min_child_weight=10,
             reg_lambda=1.5, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, tree_method='hist')

FB_TYPES = {'FF', 'SI', 'FC'}


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def make_X(df, feats):
    dum = pd.get_dummies(df['pitch_type'], prefix='pt')
    return pd.concat([df[feats].reset_index(drop=True), dum.reset_index(drop=True),
                      df[['platoon_same']].reset_index(drop=True)], axis=1)


def oof(df, feats, mono_feat=None):
    y = df['target_xrv'].values
    groups = df['pitcher'].values
    X = make_X(df, feats)
    p = dict(TUNED)
    if mono_feat:
        p['monotone_constraints'] = tuple(-1 if c == mono_feat else 0 for c in X.columns)
    out = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        m = xgb.XGBRegressor(**p)
        m.fit(X.iloc[tr], y[tr])
        out[te] = m.predict(X.iloc[te])
    return -out


def evaluate(df, stuff, late):
    d = df.copy()
    d['stuff'] = stuff
    a0, a1 = [], []
    est = {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= MIN_HALF and len(h1) >= MIN_HALF:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
        if key in late:
            est[key] = grp[grp.period == 'early'].stuff.mean()
    ks = list(est)
    return pearson(a0, a1), -pearson([est[k] for k in ks], [late[k] for k in ks])


def add_adjusted_movement(df):
    """ivb_adj/hb_adj from the stored weather deltas (delta = raw − adjusted),
    with fastball-reference diffs recomputed on the adjusted values."""
    d = df.copy()
    d['ivb_adj'] = d['ivb'] - d['ivb_wx_delta'].fillna(0.0)
    d['hb_adj'] = d['hb'] - d['hb_wx_delta'].fillna(0.0)
    fb = d[d['pitch_type'].isin(FB_TYPES)]
    counts = fb.groupby(['pitcher', 'throws', 'pitch_type']).size().reset_index(name='n')
    prim = counts.sort_values('n', ascending=False).drop_duplicates(['pitcher', 'throws'])
    prim = prim.rename(columns={'pitch_type': '_prim_pt'})[['pitcher', 'throws', '_prim_pt']]
    fbref = fb.merge(prim, left_on=['pitcher', 'throws', 'pitch_type'],
                     right_on=['pitcher', 'throws', '_prim_pt'])
    ref = fbref.groupby(['pitcher', 'throws']).agg(
        _ref_v=('velocity', 'mean'), _ref_iv=('ivb_adj', 'mean'),
        _ref_hb=('hb_adj', 'mean'), _ref_vaa=('vaa', 'mean')).reset_index()
    d = d.merge(ref, on=['pitcher', 'throws'], how='left')
    d['ivb_diff_adj'] = d['ivb_adj'] - d['_ref_iv']
    d['hb_diff_adj'] = d['hb_adj'] - d['_ref_hb']
    return d


def main():
    df = pd.read_pickle(DF)
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    late = {}
    for key, grp in df.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= MIN_PERIOD and len(l) >= MIN_PERIOD:
            late[key] = l['target_xrv'].dropna().mean()
    print(f"loaded {len(df)} pitches; {len(late)} pred units")

    dadj = add_adjusted_movement(df)
    ADJ = ['velocity', 'ivb_adj', 'hb_adj', 'velo_diff', 'ivb_diff_adj', 'hb_diff_adj',
           'spin_rate', 'extension', 'arm_angle', 'rel_z', 'vaa', 'vaa_diff']

    runs = [
        ('v11 baseline', df, BASE, None),
        ('  + rel_x', df, BASE + ['rel_x'], None),
        ('  + mono velocity', df, BASE, 'velocity'),
        ('  adjusted movement', dadj, ADJ, None),
        ('  + rel_x + mono velo', df, BASE + ['rel_x'], 'velocity'),
    ]
    print(f"\n{'config':28s} {'reliab':>7s} {'pred_xRV':>8s}")
    print('-' * 48)
    for name, d, feats, mono in runs:
        rel, pr = evaluate(d, oof(d, feats, mono), late)
        print(f"{name:28s} {rel:7.3f} {pr:8.3f}", flush=True)
    print('-' * 48)


if __name__ == '__main__':
    main()
