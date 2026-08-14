#!/usr/bin/env python3
"""stuff_pertype_loso_gate.py — LOSO replicate gate for per-type feature
config candidates (companion to stuff_pertype_atlas.py).

Runs SHIPPED (v12: nVAA + cross + FF/SI velo_diff mask) against candidate
variants across the five held-out seasons, same protocol as
scripts/stuff_features_loso.py. A candidate is adoptable only if it beats
SHIPPED on fut_r in most held-out seasons (and doesn't bleed unit_r/rel).

Variants are declared in a spec dict (or JSON via --spec):

  VARIANTS = {
    'VD_MASK_CU':   {'mask': {'velo_diff': ['FF', 'SI', 'CU']}},
    'DROP_SPIN_SL': {'mask': {'spin_rate': ['SL', 'ST']}},
    'ADD_KIN_CU':   {'add_typed': {'kin_eff': ['CU', 'KC']}},
    'ADD_RELZ':     {'add': ['rel_z']},
  }

  mask:      set feature to NaN on the listed pitch types (extends the
             production FF/SI velo_diff mask mechanism)
  add:       add a build_df column to the feature set globally
  add_typed: add a NEW column '<feat>__<TYPES>' = feature where pitch_type
             in list else NaN (type-gated inclusion)

Every variant keeps the production velo_diff FF/SI mask unless the variant
explicitly masks velo_diff itself. Writes data/_stuff_pertype_gate.json
incrementally per season.

Usage: python3 scripts/stuff_pertype_loso_gate.py [--spec spec.json]
                                                  [--seasons 2021,...]
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus.train_stuff as T
import scripts.stuff_feature_battery_2026_08 as BAT
from scripts.stuff_features_loso import (pear, season_metrics,
                                         MIN_HALF_P, MIN_HALF_U)

SEASONS = (2021, 2022, 2023, 2024, 2025)
PKL = {y: (T.HIST_PKL.format(year=y) if y != 2025 else T.PRIOR_PKL)
       for y in SEASONS}
OUT_PATH = os.path.join(ROOT, 'data', '_stuff_pertype_gate.json')
BASE = list(T.BASE_FEATS)

# Default spec — replaced by --spec once the atlas picks the candidates.
VARIANTS = {}

EXTRA_KEEP = ['rel_z', 'axis_dev', 'axis_dev_abs', 'spin_eff',
              'kin_eff', 'kin_dev', 'kin_cd', 'velo_diff_raw']


def build_year(year, slopes):
    pk = pickle.load(open(PKL[year], 'rb'))
    pk = BAT.transform_nvaa(pk, slopes)
    d = BAT.build_season(pk, BAT.GUTS[year])
    del pk
    gc.collect()
    for c in EXTRA_KEEP + ['haa_meas', 'plate_x']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    return d


def apply_variant(d, spec):
    """Return (df, feats) for a variant. Production FF/SI velo_diff mask is
    already in the df (build_df applies it); 'mask' on velo_diff REPLACES it
    (rebuilt from velo_diff_raw), other masks compose."""
    d = d.copy()
    feats = list(BASE)
    for f, types in (spec.get('mask') or {}).items():
        if f == 'velo_diff':
            d['velo_diff'] = d['velo_diff_raw']
        d.loc[d['pitch_type'].isin(set(types)), f] = np.nan
    for f in (spec.get('add') or []):
        if f not in feats:
            feats.append(f)
    for f, types in (spec.get('add_typed') or {}).items():
        col = f'{f}__{"".join(types)}'
        d[col] = np.where(d['pitch_type'].isin(set(types)), d[f], np.nan)
        feats.append(col)
    return d, feats


def design_of(d, feats):
    X = d[feats].reset_index(drop=True).copy()
    X['platoon_same'] = d['platoon_same'].reset_index(drop=True)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default=None)
    ap.add_argument('--seasons', default=None)
    args = ap.parse_args()
    variants = dict(VARIANTS)
    if args.spec:
        with open(args.spec) as f:
            variants.update(json.load(f))
    if not variants:
        sys.exit('no variants: pass --spec or fill VARIANTS')
    seasons = ([int(s) for s in args.seasons.split(',')]
               if args.seasons else list(SEASONS))

    BAT.KEEP = list(dict.fromkeys(BAT.KEEP + EXTRA_KEEP + ['haa_meas']))

    results = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)

    for Y in seasons:
        train_years = [y for y in SEASONS if y != Y]
        print(f'\n=== held-out {Y} (train {train_years}) ===', flush=True)
        raw_train = [pickle.load(open(PKL[y], 'rb')) for y in train_years]
        slopes = BAT.fit_nvaa_slopes(
            [BAT.build_season(pk, BAT.GUTS[y])
             for pk, y in zip(raw_train, train_years)])
        del raw_train
        gc.collect()
        tr_dfs = [build_year(y, slopes) for y in train_years]
        dY0 = build_year(Y, slopes)
        # per-pitch nHAA, residualized like production (slopes from training
        # seasons only) — raw HAA is horizontal-location leakage.
        from scripts.stuff_pertype_atlas import fit_haa_slopes, apply_haa_n
        haa_slopes = fit_haa_slopes(tr_dfs)
        tr_dfs = [apply_haa_n(d, haa_slopes) for d in tr_dfs]
        dY0 = apply_haa_n(dY0, haa_slopes)

        for name, spec in [('SHIPPED', {})] + list(variants.items()):
            if results.get(name, {}).get(str(Y)):
                print(f'  {name:<16} cached', flush=True)
                continue
            t0 = time.time()
            tr_v = [apply_variant(d, spec) for d in tr_dfs]
            dY, feats = apply_variant(dY0, spec)
            Xtr = pd.concat([design_of(d, feats) for d, _ in tr_v],
                            ignore_index=True)
            ytr = np.concatenate([d['target_xrv'].values for d, _ in tr_v])
            keep = np.isfinite(ytr)
            Xtr, ytr = Xtr[keep], ytr[keep]
            XY = design_of(dY, feats).reindex(columns=Xtr.columns,
                                              fill_value=0)
            m = xgb.XGBRegressor(**T._params_for(Xtr))
            m.fit(Xtr, ytr)
            dY = dY.assign(stuff=-m.predict(XY))
            met = season_metrics(dY[np.isfinite(dY['target_xrv'])])
            results.setdefault(name, {})[str(Y)] = met
            with open(OUT_PATH, 'w') as f:
                json.dump(results, f, indent=1)
            print(f'  {name:<16} pitch_r {met["pitch_r"]:.4f}  '
                  f'fut_r {met["fut_r"]:.4f} (n={met["n_fut"]})  '
                  f'unit_r {met["unit_r"]:.4f}  rel {met["rel"]:.4f}  '
                  f'[{time.time()-t0:.0f}s]', flush=True)
            del Xtr, XY, m, tr_v
            gc.collect()
        del tr_dfs, dY0
        gc.collect()

    print('\n=== GATE SUMMARY (delta vs SHIPPED per season) ===')
    for name in variants:
        if name not in results:
            continue
        wins_f = wins_u = 0
        for Y in seasons:
            a = results[name].get(str(Y))
            b = results['SHIPPED'].get(str(Y))
            if not a or not b:
                continue
            df_ = a['fut_r'] - b['fut_r']
            du_ = a['unit_r'] - b['unit_r']
            wins_f += df_ > 0
            wins_u += du_ > 0
            print(f'  {name:<16} {Y}  d_fut {df_:+.4f}  d_unit {du_:+.4f}  '
                  f'd_pitch {a["pitch_r"]-b["pitch_r"]:+.4f}  '
                  f'd_rel {a["rel"]-b["rel"]:+.4f}')
        print(f'  {name:<16} WINS: fut {wins_f}/{len(seasons)}, '
              f'unit {wins_u}/{len(seasons)}')


if __name__ == '__main__':
    main()
