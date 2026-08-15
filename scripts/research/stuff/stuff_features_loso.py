#!/usr/bin/env python3
"""stuff_features_loso.py — do the 2026-validated feature configs survive
seasons they were never tuned on? (The multi-season defensibility gate.)

PROTOCOL (matches scripts/archive/stuff_hp_loso_check.py): for each held-out
season Y in 2021-2025, fit on the OTHER four seasons (per-season Guts,
production params + monotone velocity), score Y. Per (config, Y):

  pitch_r  r(-prediction, -target_xrv) over held-out pitches
  fut_r    pitcher-level: mean prediction over first-half pitches (>=200)
           vs actual mean target over second half (>=200)
  unit_r   same but (pitcher, pitch_type) units (>=100 each half)
  rel      split-half reliability of the score within Y (>=40/half units)

CONFIGS (winners of the 2026 batteries):
  SHIPPED        BASE_FEATS
  CROSS_VD       axis_dev/axis_dev_abs -> cross/cross_abs + velo_diff masked
                 to NaN on FF/SI
  NVAA_CROSS_VD  CROSS_VD + nVAA transform. Slopes are fit on the four
                 TRAINING seasons only (never the held-out one).

Adoption rule: a candidate must beat SHIPPED on fut_r in most held-out
seasons (the replicates are the verdict, not the 2026 harness).

Usage: python3 scripts/research/stuff/stuff_features_loso.py
"""
import gc
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import stuff_plus.train_stuff as T
import scripts.stuff_feature_battery_2026_08 as BAT

SEASONS = (2021, 2022, 2023, 2024, 2025)
PKL = {y: (T.HIST_PKL.format(year=y) if y != 2025 else T.PRIOR_PKL)
       for y in SEASONS}
BASE = list(T.BASE_FEATS)
CROSS_FEATS = [f for f in BASE if f not in ('axis_dev', 'axis_dev_abs')] \
    + ['cross', 'cross_abs']
MIN_HALF_P = 200     # pitcher-level fut_r floor per half
MIN_HALF_U = 100     # unit-level floor per half


def pear(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float('nan')
    return float(np.corrcoef(a[m], b[m])[0, 1])


def build_year(year, slopes=None):
    """Season df (KEEP columns); slopes -> nVAA dict transform first."""
    pk = pickle.load(open(PKL[year], 'rb'))
    if slopes is not None:
        pk = BAT.transform_nvaa(pk, slopes)
    d = BAT.build_season(pk, BAT.GUTS[year])
    del pk
    gc.collect()
    return d


def mask_ffsi(d):
    d = d.copy()
    d.loc[d['pitch_type'].isin({'FF', 'SI'}), 'velo_diff'] = np.nan
    return d


def season_metrics(dY):
    """dY carries 'stuff' (pitcher-positive) and target_xrv."""
    pitch_r = pear(dY['stuff'], -dY['target_xrv'])
    order = {d: i for i, d in
             enumerate(sorted(dY['date'].dropna().unique()))}
    half = dY['date'].map(order).fillna(0).astype(int) % 2
    fx, fy, ux, uy, r0, r1 = [], [], [], [], [], []
    for key, g in dY.groupby('pitcher'):
        a, b = g[half.loc[g.index] == 0], g[half.loc[g.index] == 1]
        if len(a) >= MIN_HALF_P and len(b) >= MIN_HALF_P:
            fx.append(a['stuff'].mean())
            fy.append(-b['target_xrv'].mean())
    for key, g in dY.groupby(['pitcher', 'pitch_type']):
        a, b = g[half.loc[g.index] == 0], g[half.loc[g.index] == 1]
        if len(a) >= MIN_HALF_U and len(b) >= MIN_HALF_U:
            ux.append(a['stuff'].mean())
            uy.append(-b['target_xrv'].mean())
            r0.append(a['stuff'].mean())
            r1.append(b['stuff'].mean())
    return dict(pitch_r=pitch_r, fut_r=pear(fx, fy), n_fut=len(fx),
                unit_r=pear(ux, uy), n_unit=len(ux), rel=pear(r0, r1))


def main():
    results = {}
    for Y in SEASONS:
        train_years = [y for y in SEASONS if y != Y]
        print(f'\n=== held-out {Y} (train {train_years}) ===', flush=True)
        base_train = [build_year(y) for y in train_years]
        base_Y = build_year(Y)
        # nVAA slopes from TRAINING seasons only
        slopes = BAT.fit_nvaa_slopes(base_train)

        variants = {}
        variants['SHIPPED'] = (base_train, base_Y, BASE)
        mt = [mask_ffsi(d) for d in base_train]
        mY = mask_ffsi(base_Y)
        variants['CROSS_VD'] = (mt, mY, CROSS_FEATS)

        for name in ('SHIPPED', 'CROSS_VD', 'NVAA_CROSS_VD'):
            if name == 'NVAA_CROSS_VD':
                nt = [mask_ffsi(build_year(y, slopes)) for y in train_years]
                nY = mask_ffsi(build_year(Y, slopes))
                tr_dfs, dY, feats = nt, nY, CROSS_FEATS
            else:
                tr_dfs, dY, feats = variants[name]
            t0 = time.time()
            Xtr = pd.concat([T.design(d, feats) for d in tr_dfs],
                            ignore_index=True)
            ytr = np.concatenate([d['target_xrv'].values for d in tr_dfs])
            XY = T.design(dY, feats).reindex(columns=Xtr.columns,
                                             fill_value=0)
            m = xgb.XGBRegressor(**T._params_for(Xtr))
            m.fit(Xtr, ytr)
            dY = dY.assign(stuff=-m.predict(XY))
            met = season_metrics(dY)
            results.setdefault(name, {})[Y] = met
            print(f'  {name:<14} pitch_r {met["pitch_r"]:.4f}  '
                  f'fut_r {met["fut_r"]:.4f} (n={met["n_fut"]})  '
                  f'unit_r {met["unit_r"]:.4f} (n={met["n_unit"]})  '
                  f'rel {met["rel"]:.4f}  [{time.time()-t0:.0f}s]',
                  flush=True)
            del Xtr, XY, m
            if name == 'NVAA_CROSS_VD':
                del nt, nY
            gc.collect()
        del base_train, base_Y, mt, mY, variants
        gc.collect()

    print('\n=== LOSO SUMMARY (delta vs SHIPPED per season) ===')
    for name in ('CROSS_VD', 'NVAA_CROSS_VD'):
        wins_f = wins_u = 0
        for Y in SEASONS:
            df_ = results[name][Y]['fut_r'] - results['SHIPPED'][Y]['fut_r']
            du_ = results[name][Y]['unit_r'] - results['SHIPPED'][Y]['unit_r']
            wins_f += df_ > 0
            wins_u += du_ > 0
            print(f'  {name:<14} {Y}  d_fut {df_:+.4f}  d_unit {du_:+.4f}  '
                  f'd_pitch {results[name][Y]["pitch_r"] - results["SHIPPED"][Y]["pitch_r"]:+.4f}  '
                  f'd_rel {results[name][Y]["rel"] - results["SHIPPED"][Y]["rel"]:+.4f}')
        print(f'  {name:<14} WINS: fut {wins_f}/5, unit {wins_u}/5')
    with open(os.path.join(ROOT, 'data', '_stuff_loso_results.json'),
              'w') as f:
        json.dump({k: {str(y): v for y, v in r.items()}
                   for k, r in results.items()}, f, indent=1)


if __name__ == '__main__':
    main()
