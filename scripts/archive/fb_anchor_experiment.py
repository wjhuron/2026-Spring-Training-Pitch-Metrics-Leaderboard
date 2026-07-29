"""fb_anchor_experiment.py — validate the fastball-anchor rule change.

Compares the primary-fastball reference for the differential features:
  OLD: most-thrown fastball-family pitch (FF/SI/FC) — cutter can be the anchor
  NEW: most-thrown TRUE fastball (FF/SI); cutter only when neither exists

Same season-blocked scheme as production (2025 in every fold's training set,
2026 pitcher-grouped 8-fold OOF, agnostic features, monotone velocity, 2025
tags harmonized to 2026 labels, 2025 targets on 2025 Guts constants).

Metrics on 2026: split-half reliability, pred_xRV (early->late), descriptive.

Usage: python3 scripts/fb_anchor_experiment.py
"""
import os, sys, math, pickle, warnings
from collections import defaultdict
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus_v11.train_stuff_v11 as T

PKL26 = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
PKL25 = os.path.join(ROOT, 'data', '_pitches2025_training.pkl')
LG25, SCALE25 = T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    return float(np.corrcoef(xs, ys)[0, 1]) if xs.std() and ys.std() else None


def build(p26, p25, prefer):
    df26 = T.build_df(p26, prefer_true_fastball=prefer)
    _lg, _sc = T.LG_WOBA, T.WOBA_SCALE
    T.LG_WOBA, T.WOBA_SCALE = LG25, SCALE25
    df25 = T.build_df(p25, prefer_true_fastball=prefer)
    T.LG_WOBA, T.WOBA_SCALE = _lg, _sc
    df26 = df26[df26.target_xrv.notna()].reset_index(drop=True)
    df25 = df25[df25.target_xrv.notna()].reset_index(drop=True)
    order = {d: i for i, d in enumerate(sorted(df26['date'].dropna().unique()))}
    df26['half'] = df26['date'].map(order).fillna(0).astype(int) % 2
    df26['period'] = np.where(df26['date'] < '2026-05-01', 'early', 'late')
    return df26, df25


def evaluate(df26, df25):
    X26 = T.design(df26); y26 = df26.target_xrv.values; g = df26.pitcher.values
    X25 = T.design(df25).reindex(columns=X26.columns, fill_value=0); y25 = df25.target_xrv.values
    p = T._params_for(X26)
    oof = np.full(len(df26), np.nan)
    for tr, te in GroupKFold(n_splits=8).split(X26, y26, g):
        Xtr = pd.concat([X26.iloc[tr], X25], ignore_index=True)
        ytr = np.concatenate([y26[tr], y25])
        m = xgb.XGBRegressor(**p); m.fit(Xtr, ytr); oof[te] = m.predict(X26.iloc[te])
    d = df26.copy(); d['stuff'] = -oof
    late, a0, a1, est = {}, [], [], {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= 50 and len(l) >= 50:
            late[key] = l.target_xrv.dropna().mean()
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= 40 and len(h1) >= 40:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
        if key in late:
            est[key] = grp[grp.period == 'early'].stuff.mean()
    ks = list(est)
    dx = [g.stuff.mean() for _, g in d.groupby(['pitcher', 'throws', 'pitch_type']) if len(g) >= 100]
    dy = [g.target_xrv.mean() for _, g in d.groupby(['pitcher', 'throws', 'pitch_type']) if len(g) >= 100]
    return (pearson(a0, a1), -pearson([est[k] for k in ks], [late[k] for k in ks]),
            -pearson(dx, dy), len(a0), len(ks))


def main():
    p26 = [p for p in pickle.load(open(PKL26, 'rb')) if p.get('_source') == 'MLB']
    p25 = pickle.load(open(PKL25, 'rb'))
    # harmonize 2025 tags to 2026 labels (same as production)
    T._harmonize_tags(p25, p26)
    print(f"2026={len(p26)}  2025={len(p25)}\n")
    print(f"{'anchor rule':34s} {'reliab':>7s} {'pred_xRV':>8s} {'desc':>7s}")
    res = {}
    for name, prefer in [('OLD (most-thrown FF/SI/FC)', False),
                         ('NEW (true FB; cutter fallback)', True)]:
        df26, df25 = build(p26, p25, prefer)
        rel, pred, desc, nrel, npred = evaluate(df26, df25)
        res[name] = pred
        print(f"{name:34s} {rel:7.3f} {pred:8.3f} {desc:7.3f}  (n_rel={nrel}, n_pred={npred})")
    delta = res['NEW (true FB; cutter fallback)'] - res['OLD (most-thrown FF/SI/FC)']
    print(f"\npred_xRV delta (NEW - OLD): {delta:+.4f}")
    print("Decision rule: adopt NEW unless it drops pred by >0.010 (Wally's preference wins ties).")
    print("ADOPT" if delta > -0.010 else "HOLD — materially worse")


if __name__ == '__main__':
    main()
