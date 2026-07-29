"""agnostic_stuff_experiment.py — pitch-type AGNOSTIC Stuff+ experiment.

Question: does removing the pitch-type one-hot dummies from the v11 design
(judging every pitch purely on its physical characteristics, FanGraphs-style)
cost anything? Pitch type would remain for display grouping and per-type
standardization ONLY. platoon_same stays; the fastball-reference differentials
stay (FB_TYPES selection is independent of the model's feature design).

Both configs run the EXACT production season-blocked scheme
(stuff_plus_v11/train_stuff_v11.py):
  - 2025 training pickle tag-harmonized to 2026 labels (T._harmonize_tags)
  - 2025 joins EVERY fold's training set; 2025 targets use T.PRIOR_LG_WOBA /
    T.PRIOR_WOBA_SCALE
  - 2026 scored pitcher-grouped 8-fold OOF
  - TUNED params + monotone velocity constraint (T._params_for)

Configs:
  (a) with_dummies  : BASE_FEATS + pt_* one-hots + platoon_same  (current prod)
  (b) agnostic      : BASE_FEATS + platoon_same                  (no dummies)

Metrics on 2026 OOF stuff (unit = pitcher, throws, pitch_type):
  - split-half reliability: odd/even calendar dates, >=40 pitches per half
  - pred_xRV: early (<2026-05-01) OOF stuff -> late-period mean target,
    >=50 pitches in each period
  - descriptive: same-period OOF stuff vs mean target, >=100 pitches

Usage: python3 scripts/agnostic_stuff_experiment.py
"""
import os, sys, pickle, warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus_v11.train_stuff_v11 as T

PKL26 = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
PKL25 = os.path.join(ROOT, 'data', '_pitches2025_training.pkl')


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def design_agnostic(df, feats=None):
    """v11 design minus the pitch-type one-hot dummies. platoon_same stays."""
    feats = feats or T.BASE_FEATS
    return pd.concat([df[feats].reset_index(drop=True),
                      df[['platoon_same']].reset_index(drop=True)], axis=1)


def main():
    D26 = pickle.load(open(PKL26, 'rb'))
    p26 = [p for p in D26 if p.get('_source', 'MLB') == 'MLB']
    p25 = pickle.load(open(PKL25, 'rb'))

    # production tag harmonization (mutates p25 in place)
    T._harmonize_tags(p25, p26)

    # season dfs with season-correct target constants (production convention)
    df26 = T.build_df(p26)
    T.LG_WOBA, T.WOBA_SCALE = T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE
    df25 = T.build_df(p25)
    T.LG_WOBA, T.WOBA_SCALE = 0.3169, 1.2393
    df26 = df26[df26['target_xrv'].notna()].reset_index(drop=True)
    df25 = df25[df25['target_xrv'].notna()].reset_index(drop=True)

    date_order = {dt: i for i, dt in enumerate(sorted(df26['date'].dropna().unique()))}
    df26['half'] = df26['date'].map(date_order).fillna(0).astype(int) % 2
    df26['period'] = np.where(df26['date'] < '2026-05-01', 'early', 'late')
    print(f'training rows: 2025={len(df25)}, 2026={len(df26)}')

    y26, y25 = df26['target_xrv'].values, df25['target_xrv'].values
    g26 = df26['pitcher'].values

    configs = {
        'with_dummies (current)': (T.design(df26), T.design(df25)),
        'agnostic (no pt dummies)': (design_agnostic(df26), design_agnostic(df25)),
    }

    d = df26.copy()
    for name, (X26, X25) in configs.items():
        X25 = X25.reindex(columns=X26.columns, fill_value=0)
        params = T._params_for(X26)
        oof = np.full(len(df26), np.nan)
        for tr, te in GroupKFold(n_splits=8).split(X26, y26, g26):
            Xtr = pd.concat([X26.iloc[tr], X25], ignore_index=True)
            ytr = np.concatenate([y26[tr], y25])
            m = xgb.XGBRegressor(**params)
            m.fit(Xtr, ytr)
            oof[te] = m.predict(X26.iloc[te])
        d[name] = -oof
        print(f'  OOF done: {name} ({X26.shape[1]} features)')

    # late-period targets (config-independent)
    late = {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= 50 and len(l) >= 50:
            late[key] = l['target_xrv'].dropna().mean()

    print(f"\n{'config':26s} {'reliab':>7s} {'pred_xRV':>8s} {'desc':>7s} "
          f"{'n_rel':>6s} {'n_pred':>6s}")
    results = {}
    for name in configs:
        a0, a1, est, desc_x, desc_y = [], [], {}, [], []
        for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
            h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
            if len(h0) >= 40 and len(h1) >= 40:
                a0.append(h0[name].mean()); a1.append(h1[name].mean())
            if key in late:
                est[key] = grp[grp.period == 'early'][name].mean()
            if len(grp) >= 100:
                desc_x.append(grp[name].mean()); desc_y.append(grp['target_xrv'].mean())
        ks = list(est)
        rel = pearson(a0, a1)
        pr = -pearson([est[k] for k in ks], [late[k] for k in ks])
        dsc = -pearson(desc_x, desc_y)
        results[name] = (rel, pr, dsc)
        print(f'{name:26s} {rel:7.3f} {pr:8.3f} {dsc:7.3f} {len(a0):6d} {len(ks):6d}')

    (rel_a, pr_a, _), (rel_b, pr_b, _) = results.values()
    print(f'\ndeltas (agnostic - current): reliab {rel_b - rel_a:+.4f}, '
          f'pred_xRV {pr_b - pr_a:+.4f}')

    # Tyler Rogers spot check (standardize vs qualified SI pool per config)
    print('\nTyler Rogers (SI) Stuff+ by config (vs >=50-pitch SI pool):')
    for name in configs:
        pool = d[d.pitch_type == 'SI'].groupby(['pitcher', 'throws'])[name].agg(['mean', 'size'])
        pool = pool[pool['size'] >= 50]
        mu, sd_ = pool['mean'].mean(), pool['mean'].std()
        rog = d[(d.pitcher.str.contains('Rogers, Tyler')) & (d.pitch_type == 'SI')][name].mean()
        print(f'  {name:26s} {100 + 10 * (rog - mu) / sd_:.1f}')


if __name__ == '__main__':
    main()
