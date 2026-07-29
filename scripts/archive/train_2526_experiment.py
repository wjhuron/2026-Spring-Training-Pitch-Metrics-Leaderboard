"""train_2526_experiment.py — evaluate multi-season Stuff+ training.

Parts:
  1. Cross-year TAG CONSISTENCY AUDIT: for each pitcher in both seasons,
     match each 2025 tag's (velo, ivb, hb) centroid to his nearest 2026 tag
     centroid; report label flips (e.g. SL'25 -> FC'26). Flipped tags are
     harmonized to the 2026 label in the training copy (sheets untouched).
  2. SEASON-BLOCKED TRAINING: folds are pitcher-grouped over 2026 only;
     every fold's training set additionally contains ALL of 2025. 2026
     scores stay strictly out-of-fold (no same-season luck leakage), while
     cross-season identity signal is available (the Tyler Rogers fix).
  3. Evaluation vs the current 2026-only config on the standard harness
     (2026 split-half reliability, early->late pred), plus the first true
     YoY test: train on 2025 ONLY -> predict 2026.
  4. Tyler Rogers spot check.

Usage: python3 scripts/train_2526_experiment.py
"""
import os, sys, math, pickle, warnings
from collections import defaultdict

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
LG25, SCALE25 = 0.3125, 1.242   # 2025 FG Guts (fallback if live fetch fails)
try:
    from pipeline_fetch import fetch_guts_constants
    _w, _f, extra = fetch_guts_constants(2025)
    LG25, SCALE25 = extra['lgWOBA'], extra['wOBAScale']
    print(f'2025 Guts: lgWOBA={LG25}, scale={SCALE25}')
except Exception as e:
    print(f'Guts fetch failed ({e}); using fallback {LG25}/{SCALE25}')


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def tag_audit(p25, p26):
    """Return {(pitcher, tag25): tag26} for flips, and print a report."""
    def _sf(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def centroids(pitches):
        acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
        for p in pitches:
            v = _sf(p.get('Velocity'))
            iv = _sf(p.get('xIndVrtBrk'))
            hb = _sf(p.get('xHorzBrk'))
            thr = p.get('Throws')
            if None in (v, iv, hb) or thr not in ('L', 'R'):
                continue
            s = 1.0 if thr == 'R' else -1.0
            a = acc[(p.get('Pitcher'), p.get('Pitch Type'))]
            a[0] += v; a[1] += iv; a[2] += hb * s; a[3] += 1
        return {k: (a[0]/a[3], a[1]/a[3], a[2]/a[3], a[3]) for k, a in acc.items() if a[3] >= 30}

    c25, c26 = centroids(p25), centroids(p26)
    by_p26 = defaultdict(dict)
    for (pit, tag), c in c26.items():
        by_p26[pit][tag] = c
    flips = {}
    n_shared = n_stable = 0
    for (pit, tag25), c in c25.items():
        tags26 = by_p26.get(pit)
        if not tags26:
            continue
        n_shared += 1
        best_tag, best_d = None, 1e9
        for tag26, c2 in tags26.items():
            d = math.sqrt(((c[0]-c2[0])/1.5)**2 + ((c[1]-c2[1])/2.5)**2 + ((c[2]-c2[2])/2.5)**2)
            if d < best_d:
                best_d, best_tag = d, tag26
        if best_tag == tag25 or best_d > 1.5:
            n_stable += 1
            continue
        flips[(pit, tag25)] = best_tag
    print(f'\nTAG AUDIT: {n_shared} (pitcher, 2025-tag) units shared with 2026; '
          f'{len(flips)} label flips detected')
    for (pit, t25), t26 in sorted(flips.items())[:15]:
        print(f'  {pit:26s} {t25} (2025) -> {t26} (2026)')
    if len(flips) > 15:
        print(f'  ... and {len(flips) - 15} more')
    return flips


def main():
    D26 = pickle.load(open(PKL26, 'rb'))
    p26 = [p for p in D26 if p.get('_source', 'MLB') == 'MLB']
    p25 = pickle.load(open(PKL25, 'rb'))

    flips = tag_audit(p25, p26)
    n_re = 0
    for p in p25:
        k = (p.get('Pitcher'), p.get('Pitch Type'))
        if k in flips:
            p['Pitch Type'] = flips[k]
            n_re += 1
    print(f'harmonized {n_re} 2025 pitches to 2026 labels')

    # build season dfs with season-correct target constants
    df26 = T.build_df(p26)
    T.LG_WOBA, T.WOBA_SCALE = LG25, SCALE25
    df25 = T.build_df(p25)
    T.LG_WOBA, T.WOBA_SCALE = 0.3169, 1.2393
    df26['season'] = 2026
    df25['season'] = 2025
    df26 = df26[df26['target_xrv'].notna()].reset_index(drop=True)
    df25 = df25[df25['target_xrv'].notna()].reset_index(drop=True)
    # evaluation tags for 2026 (build_df doesn't add these)
    date_order = {dt: i for i, dt in enumerate(sorted(df26['date'].dropna().unique()))}
    df26['half'] = df26['date'].map(date_order).fillna(0).astype(int) % 2
    df26['period'] = np.where(df26['date'] < '2026-05-01', 'early', 'late')
    print(f'\ntraining rows: 2025={len(df25)}, 2026={len(df26)}')

    X26 = T.design(df26)
    X25 = T.design(df25).reindex(columns=X26.columns, fill_value=0)
    y26, y25 = df26['target_xrv'].values, df25['target_xrv'].values
    g26 = df26['pitcher'].values
    params = T._params_for(X26)

    # current config: 2026-only pitcher-grouped OOF
    oof_cur = np.full(len(df26), np.nan)
    for tr, te in GroupKFold(n_splits=8).split(X26, y26, g26):
        m = xgb.XGBRegressor(**params); m.fit(X26.iloc[tr], y26[tr])
        oof_cur[te] = m.predict(X26.iloc[te])

    # season-blocked: 2025 in every fold's training set
    oof_sb = np.full(len(df26), np.nan)
    for tr, te in GroupKFold(n_splits=8).split(X26, y26, g26):
        Xtr = pd.concat([X26.iloc[tr], X25], ignore_index=True)
        ytr = np.concatenate([y26[tr], y25])
        m = xgb.XGBRegressor(**params); m.fit(Xtr, ytr)
        oof_sb[te] = m.predict(X26.iloc[te])

    # YoY: 2025-only model predicts 2026
    m25 = xgb.XGBRegressor(**params); m25.fit(X25, y25)
    pred_yoy = m25.predict(X26)

    # evaluation on 2026
    d = df26.copy()
    d['cur'], d['sb'], d['yoy'] = -oof_cur, -oof_sb, -pred_yoy
    late, results = {}, {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= 50 and len(l) >= 50:
            late[key] = l['target_xrv'].dropna().mean()
    print(f"\n{'config':22s} {'reliab':>7s} {'pred_xRV':>8s} {'desc':>7s}")
    for col in ('cur', 'sb', 'yoy'):
        a0, a1, est, desc_x, desc_y = [], [], {}, [], []
        for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
            h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
            if len(h0) >= 40 and len(h1) >= 40:
                a0.append(h0[col].mean()); a1.append(h1[col].mean())
            if key in late:
                est[key] = grp[grp.period == 'early'][col].mean()
            if len(grp) >= 100:
                desc_x.append(grp[col].mean()); desc_y.append(grp['target_xrv'].mean())
        ks = list(est)
        rel = pearson(a0, a1)
        pr = -pearson([est[k] for k in ks], [late[k] for k in ks])
        dsc = -pearson(desc_x, desc_y)
        name = {'cur': '2026-only (current)', 'sb': 'season-blocked +2025',
                'yoy': '2025-only -> 2026'}[col]
        print(f'{name:22s} {rel:7.3f} {pr:8.3f} {dsc:7.3f}')

    # Rogers spot check (standardize vs qualified pool per config)
    print('\nTyler Rogers (SI) mean raw score by config, standardized vs pool:')
    for col in ('cur', 'sb', 'yoy'):
        pool = d[d.pitch_type == 'SI'].groupby(['pitcher', 'throws'])[col].agg(['mean', 'size'])
        pool = pool[pool['size'] >= 50]
        mu, sd_ = pool['mean'].mean(), pool['mean'].std()
        rog = d[(d.pitcher.str.contains('Rogers, Tyler')) & (d.pitch_type == 'SI')][col].mean()
        print(f'  {col}: stuff+ = {100 + 10 * (rog - mu) / sd_:.1f}')


if __name__ == '__main__':
    main()
