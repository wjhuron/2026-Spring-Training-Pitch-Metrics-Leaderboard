#!/usr/bin/env python3
"""stuff_pertype_atlas.py — per-pitch-type earned importance + interaction
atlas for Stuff+, measured OUT OF SAMPLE across five held-out seasons.

MOTIVATION (Wally, 2026-08-13): which pitch characteristics matter for which
pitch type, which don't matter at all, and which PAIRS of characteristics the
model actually uses together (ivb x vaa, spin x velocity, arm_angle x nvaa,
...). Full scope: every physics column build_df emits, not just BASE_FEATS.

PROTOCOL (extends scripts/stuff_features_loso.py): for each held-out season
Y in 2021-2025, fit ONE extended-feature model (production params + monotone
velocity, nVAA slopes fit on the four training seasons only) on the other
four seasons, then measure on Y:

  1. PERMUTATION importance per (pitch_type, feature): permute the feature
     within Y's rows of that type, delta in pitch-level r on those rows.
     Earned, out-of-sample, per-type. 2 repeats, types with >= MIN_TYPE rows.
  2. SHAP shares per type (pred_contribs on <= SHAP_CAP rows/type): how the
     model distributes credit within that type.
  3. SHAP INTERACTION matrix per type (pred_interactions on <= INTER_CAP
     rows/type): mean |interaction| per feature pair — which pairs the model
     genuinely uses together (off-diagonal), vs main effects (diagonal).
  4. Feature-feature Pearson correlation per type (redundancy context).

The atlas model deliberately includes characteristics REJECTED for
production (rel_z, axis_dev, spin_eff, ax_sin/ax_cos, kinematics, nHAA) and
uses velo_diff UNMASKED (production masks FF/SI): the question here is
information content per type, not shipping config. Anything that looks like
a config improvement graduates to the paired battery + LOSO replicate gate
before it goes near production.

Writes data/_stuff_atlas_results.json incrementally (one entry per season,
so a crash keeps finished seasons). Runtime ~2-3h for all five seasons.

Usage: python3 scripts/stuff_pertype_atlas.py [--seasons 2021,2022,...]
                                              [--no-inter]
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

SEASONS = (2021, 2022, 2023, 2024, 2025)
PKL = {y: (T.HIST_PKL.format(year=y) if y != 2025 else T.PRIOR_PKL)
       for y in SEASONS}
OUT_PATH = os.path.join(ROOT, 'data', '_stuff_atlas_results.json')

# Every physics characteristic build_df emits. vaa/vaa_diff are nVAA (the
# production transform); velo_diff is the UNMASKED gap (velo_diff_raw) so
# FF/SI importance is measurable; haa_n is hand-mirrored measured HAA (the
# nHAA-style sign convention: positive = arm-side approach for either hand).
EXT_FEATS = ['velocity', 'ivb', 'hb', 'velo_diff', 'ivb_diff', 'hb_diff',
             'spin_rate', 'extension', 'arm_angle', 'vaa', 'vaa_diff',
             'rel_x', 'rel_z', 'cross', 'cross_abs', 'ax_sin', 'ax_cos',
             'axis_dev', 'axis_dev_abs', 'spin_eff',
             'kin_eff', 'kin_dev', 'kin_cd', 'haa_n']
# platoon_same rides along via T.design()
ALL_COLS = EXT_FEATS + ['platoon_same']

MIN_TYPE = 8000     # min rows of a type in the held-out season to measure
PERM_REPS = 2
PERM_CAP = 60000    # rows per type for the permutation pass
SHAP_CAP = 20000    # rows per type for pred_contribs
INTER_CAP = 2500    # rows per type for pred_interactions (O(F^2), heavy)
SEED = 20260813


def pear(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float('nan')
    return float(np.corrcoef(a[m], b[m])[0, 1])


def build_year(year, slopes=None):
    pk = pickle.load(open(PKL[year], 'rb'))
    if slopes is not None:
        pk = BAT.transform_nvaa(pk, slopes)
    d = BAT.build_season(pk, BAT.GUTS[year])
    del pk
    gc.collect()
    # KEEP extras arrive as object dtype (None-mixed) — coerce every numeric
    # the atlas touches before any arithmetic or DMatrix build.
    for c in ['haa_meas', 'plate_x', 'rel_z', 'axis_dev', 'axis_dev_abs',
              'spin_eff', 'kin_eff', 'kin_dev', 'kin_cd', 'velo_diff_raw'] + \
             [f for f in EXT_FEATS if f in d.columns]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    # atlas uses the unmasked gap
    d['velo_diff'] = d['velo_diff_raw']
    return d


def fit_haa_slopes(dfs, min_n=2000):
    """Per-type WITHIN-PITCHER slope of hand-mirrored HAA on hand-mirrored
    PlateX, per production's nHAA recipe (process_data). Raw HAA is a
    horizontal-location channel exactly the way raw VAA is a vertical one —
    feeding it unresidualized let the model use it as location leakage
    (v2 first attempt: perm deltas 10x anything else, predictions driven
    anti-correlated when permuted). Within-pitcher demeaning keeps pitcher
    identity out of the slope. Fit on TRAINING seasons only per fold."""
    pool = pd.concat([d[['pitch_type', 'pitcher', 'throws',
                         'haa_meas', 'plate_x']] for d in dfs],
                     ignore_index=True).dropna()
    s = np.where(pool['throws'] == 'R', 1.0, -1.0)
    pool = pool.assign(hm=pool['haa_meas'] * s, pm=pool['plate_x'] * s)
    out = {}
    for pt, sub in pool.groupby('pitch_type'):
        if len(sub) < min_n:
            continue
        g = sub.groupby('pitcher')
        hm_d = (sub['hm'] - g['hm'].transform('mean')).values
        pm_d = (sub['pm'] - g['pm'].transform('mean')).values
        var = float(np.var(pm_d))
        if var <= 0:
            continue
        out[pt] = (float(np.cov(pm_d, hm_d)[0, 1] / var),
                   float(sub['pm'].mean()))
    print('  nHAA slopes (deg/ft, within-pitcher): '
          + ', '.join(f'{pt}:{v[0]:.2f}' for pt, v in sorted(out.items())),
          flush=True)
    return out


def apply_haa_n(d, slopes):
    """haa_n = mirrored HAA residualized on mirrored PlateX (per-pitch nHAA)."""
    s = np.where(d['throws'] == 'R', 1.0, -1.0)
    hm = (d['haa_meas'] * s).values
    pm = (d['plate_x'] * s).values
    out = np.full(len(d), np.nan)
    for pt, (sl, pbar) in slopes.items():
        m = (d['pitch_type'] == pt).values
        out[m] = (hm - sl * (pm - pbar))[m]
    d['haa_n'] = out
    return d


def design_ext(d):
    X = d[EXT_FEATS].reset_index(drop=True).copy()
    X['platoon_same'] = d['platoon_same'].reset_index(drop=True)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', default=None,
                    help='comma list, default all five')
    ap.add_argument('--no-inter', action='store_true',
                    help='skip the SHAP interaction pass')
    args = ap.parse_args()
    seasons = ([int(s) for s in args.seasons.split(',')]
               if args.seasons else list(SEASONS))

    # KEEP must retain the atlas extras through build_season
    BAT.KEEP = list(dict.fromkeys(
        BAT.KEEP + ['rel_z', 'axis_dev', 'axis_dev_abs', 'spin_eff',
                    'kin_eff', 'kin_dev', 'kin_cd', 'velo_diff_raw']))

    results = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
        print(f'resuming: {sorted(results.keys())} already done', flush=True)

    rng = np.random.default_rng(SEED)
    for Y in seasons:
        if str(Y) in results:
            print(f'=== {Y} already in results, skipping ===', flush=True)
            continue
        t_season = time.time()
        train_years = [y for y in SEASONS if y != Y]
        print(f'\n=== held-out {Y} (train {train_years}) ===', flush=True)

        base_train = [build_year(y) for y in train_years]
        slopes = BAT.fit_nvaa_slopes(base_train)
        del base_train
        gc.collect()
        tr_dfs = [build_year(y, slopes) for y in train_years]
        dY = build_year(Y, slopes)
        # per-pitch nHAA: slopes from TRAINING seasons only, applied to all
        haa_slopes = fit_haa_slopes(tr_dfs)
        tr_dfs = [apply_haa_n(d, haa_slopes) for d in tr_dfs]
        dY = apply_haa_n(dY, haa_slopes)

        t0 = time.time()
        Xtr = pd.concat([design_ext(d) for d in tr_dfs], ignore_index=True)
        ytr = np.concatenate([d['target_xrv'].values for d in tr_dfs])
        keep = np.isfinite(ytr)
        Xtr, ytr = Xtr[keep], ytr[keep]
        del tr_dfs
        gc.collect()
        model = xgb.XGBRegressor(**T._params_for(Xtr))
        model.fit(Xtr, ytr)
        n_train = len(Xtr)
        del Xtr, ytr
        gc.collect()
        print(f'  fit: {n_train} rows [{time.time()-t0:.0f}s]', flush=True)

        dY = dY[np.isfinite(dY['target_xrv'])].reset_index(drop=True)
        XY = design_ext(dY)
        yY = -dY['target_xrv'].values
        booster = model.get_booster()

        season_out = {'n_train': n_train, 'types': {}}
        type_counts = dY['pitch_type'].value_counts()
        atlas_types = [t for t in T.SUPPORTED
                       if type_counts.get(t, 0) >= MIN_TYPE]
        print(f'  types: ' + ', '.join(
            f'{t}({type_counts[t]})' for t in atlas_types), flush=True)

        for pt in ['ALL'] + atlas_types:
            t0 = time.time()
            if pt == 'ALL':
                idx = np.arange(len(dY))
            else:
                idx = np.where((dY['pitch_type'] == pt).values)[0]
            if len(idx) > PERM_CAP:
                idx_perm = rng.choice(idx, PERM_CAP, replace=False)
            else:
                idx_perm = idx
            Xg = XY.iloc[idx_perm].reset_index(drop=True)
            yg = yY[idx_perm]
            base = -model.predict(Xg)
            r0 = pear(base, yg)

            # 1. permutation importance
            perm = {}
            for f in ALL_COLS:
                deltas = []
                Xp = Xg.copy()
                col = Xg[f].values
                for _ in range(PERM_REPS):
                    Xp[f] = rng.permutation(col)
                    deltas.append(r0 - pear(-model.predict(Xp), yg))
                perm[f] = round(float(np.mean(deltas)), 5)

            # 2. SHAP shares
            if len(idx) > SHAP_CAP:
                idx_shap = rng.choice(idx, SHAP_CAP, replace=False)
            else:
                idx_shap = idx
            Xs = XY.iloc[idx_shap].reset_index(drop=True)
            contrib = booster.predict(xgb.DMatrix(Xs), pred_contribs=True)
            mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)   # drop bias col
            total = mean_abs.sum()
            shap_share = {c: round(float(v / total), 4)
                          for c, v in zip(Xs.columns, mean_abs)}

            # 3. SHAP interactions (off-diagonal pairs)
            inter_pairs = None
            if not args.no_inter:
                if len(idx) > INTER_CAP:
                    idx_int = rng.choice(idx, INTER_CAP, replace=False)
                else:
                    idx_int = idx
                Xi = XY.iloc[idx_int].reset_index(drop=True)
                im = booster.predict(xgb.DMatrix(Xi), pred_interactions=True)
                im = np.abs(im[:, :-1, :-1]).mean(axis=0)     # drop bias
                cols = list(Xi.columns)
                pairs = []
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        pairs.append((cols[i], cols[j],
                                      float(im[i, j] + im[j, i])))
                pairs.sort(key=lambda p: -p[2])
                inter_pairs = [[a, b, round(v, 6)] for a, b, v in pairs[:25]]

            # 4. within-type feature correlation (top absolute pairs)
            corr_pairs = []
            Xc = Xg[EXT_FEATS]
            cm = Xc.corr().values
            for i in range(len(EXT_FEATS)):
                for j in range(i + 1, len(EXT_FEATS)):
                    if np.isfinite(cm[i, j]) and abs(cm[i, j]) >= 0.5:
                        corr_pairs.append([EXT_FEATS[i], EXT_FEATS[j],
                                           round(float(cm[i, j]), 3)])
            corr_pairs.sort(key=lambda p: -abs(p[2]))

            season_out['types'][pt] = {
                'n': int(len(idx)), 'r0': round(r0, 4), 'perm': perm,
                'shap_share': shap_share, 'inter_top': inter_pairs,
                'corr_pairs': corr_pairs[:20],
            }
            print(f'    {pt:<4} n={len(idx):>7}  r0={r0:+.4f}  '
                  f'[{time.time()-t0:.0f}s]', flush=True)

        results[str(Y)] = season_out
        with open(OUT_PATH, 'w') as f:
            json.dump(results, f)
        print(f'  season {Y} done [{(time.time()-t_season)/60:.1f} min] '
              f'-> {OUT_PATH}', flush=True)
        del dY, XY, model, booster
        gc.collect()

    print('\nAtlas complete.', flush=True)


if __name__ == '__main__':
    main()
