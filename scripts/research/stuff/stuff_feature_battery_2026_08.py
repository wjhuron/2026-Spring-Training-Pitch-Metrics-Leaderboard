#!/usr/bin/env python3
"""stuff_feature_battery_2026_08.py — paired feature battery at the FULL
production config (2021-25 priors in every fold, 8-fold pitcher-grouped OOF,
depth-7 TUNED). One battery, shared folds, so every comparison is paired.

Variants (approved 2026-08-09):
  BASE       shipped BASE_FEATS
  NHAA       + nhaa: location-adjusted horizontal approach angle. HAA is
             reconstructed geometrically for ALL seasons (priors carry no HAA
             field; using measured 2026 + recon priors would be a season seam,
             the axis_dev coverage lesson). Recon validated against measured
             2026 HAA and OLS-calibrated to it before use.
  CROSS      axis_dev/axis_dev_abs -> cross/cross_abs: the SSW proxy restated
             in release-axis-frame INCHES (linear in movement, gyro-safe)
             instead of circular degrees (sd 40.6 deg on SL).
  AXPOS      + ax_sin/ax_cos: the hand-mirrored release-axis direction itself
             (the model currently sees only the deviation from it).
  MONOEXT    BASE feats, monotone constraint on extension as well as velocity.
  VDMASK     BASE feats, velo_diff masked to NaN on every non-CH/FS pitch
             (Wally's challenge: separation logic belongs to offspeed only).
  NVAA       vaa/vaa_diff swapped to location-adjusted nVAA via the pitch-dict
             transform (VAA - slope_pt*(PlateZ - zbar_pt), slopes fit pooled
             2021-26), full build_df rebuild so vaa_diff stays internally
             consistent. Loc+ independence is the second axis here: by the
             house scope rule, prediction gained through location leakage
             does not count for a skill-isolation model.
  NVAA_NHAA  both approach angles location-adjusted.

Metrics per variant: split-half reliability (odd/even dates, >=40/half),
early->late predictive (<2026-05-01 vs after, >=50 each), descriptive
(units n>=100), Loc+ independence corr (units n>=50 vs locPlusRaw, flipped so
positive = stuff/location aligned; LOWER |r| = cleaner separation), plus
per-family desc/pred. Paired deltas vs BASE printed at the end.

Usage: python3 scripts/research/stuff/stuff_feature_battery_2026_08.py
"""
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import stuff_plus.train_stuff as T
import scripts.build_historical_training_set as H

GUTS = dict(H.GUTS)
GUTS[2025] = (T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE)
YEARS = (2021, 2022, 2023, 2024, 2025)

PICKLE = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
PITCH_LB = os.path.join(ROOT, 'data', 'pitch_leaderboard_rs.json')

FAMILY = {'FF': 'fastball', 'SI': 'fastball', 'FC': 'cutter',
          'SL': 'breaking', 'ST': 'breaking', 'SV': 'breaking',
          'CU': 'breaking', 'KC': 'breaking',
          'CH': 'offspeed', 'FS': 'offspeed'}
OFFSPEED = {'CH', 'FS'}

KEEP = list(dict.fromkeys(
    ['pitcher', 'team', 'throws', 'date', 'pitch_type', 'platoon_same',
     'target_xrv'] + T.BASE_FEATS + ['cross', 'cross_abs', 'ax_sin',
     'ax_cos', 'plate_x', 'plate_z', 'haa_meas', 'extension']))


def pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def build_season(pitch_dicts, guts, harmonize_against=None):
    if harmonize_against is not None:
        T._harmonize_tags(pitch_dicts, harmonize_against)
    _lg, _sc = T.LG_WOBA, T.WOBA_SCALE
    T.LG_WOBA, T.WOBA_SCALE = guts
    d = T.build_df(pitch_dicts)
    T.LG_WOBA, T.WOBA_SCALE = _lg, _sc
    d = d[d['target_xrv'].notna()].reset_index(drop=True)
    return d[[c for c in KEEP if c in d.columns]].copy()


def load_mlb_2026():
    allp = pickle.load(open(PICKLE, 'rb'))
    ep = {(p.get('Pitcher'), p.get('PTeam')) for p in allp
          if p.get('Pitch Type') == 'EP'}
    return [p for p in allp if p.get('_source') == 'MLB'
            and (p.get('Pitcher'), p.get('PTeam')) not in ep]


# ── HAA reconstruction (all seasons, one derivation — no season seam) ──
# Two measured physical terms: the straight-line geometry angle and the
# feed-convention break deflection,
#   HAA ~ b1 * deg((PlateX - RelPosX)/dy) + b2 * deg((-HB_raw/12)/dy)
# Coefficients are FIT on 2026 measured HAA and applied to every season.
# Validated 2026-08-09 on 200k pitches: b = [1.079, 1.001 on -HB], r=0.9998,
# per-type coefficients stable (geo 1.07-1.085 = drag factor; brk 0.99-1.02
# confirming breakHorizontal is sign-flipped vs the PlateX/RelPosX frame).
def add_haa_recon(df):
    s = np.where(df['throws'].values == 'R', 1.0, -1.0)
    rel_x_raw = df['rel_x'].values * s
    hb_raw = df['hb'].values * s
    dy = (60.5 - df['extension'].values) - 17.0 / 12.0
    df['_t_geo'] = np.degrees((df['plate_x'].values - rel_x_raw) / dy)
    df['_t_brk'] = np.degrees((hb_raw / 12.0) / dy)
    df['_s'] = s
    return df


def calibrate_haa(df26):
    m = (df26['haa_meas'].notna() & df26['_t_geo'].notna()
         & df26['_t_brk'].notna())
    A = np.column_stack([np.ones(int(m.sum())),
                         df26.loc[m, '_t_geo'].values,
                         df26.loc[m, '_t_brk'].values])
    yv = df26.loc[m, 'haa_meas'].values
    b, *_ = np.linalg.lstsq(A, yv, rcond=None)
    r = pear(A @ b, yv)
    print(f'HAA recon vs measured (2026, n={m.sum()}): r={r:.4f}, '
          f'b=[{b[0]:.3f}, geo {b[1]:.3f}, brk {b[2]:.3f}]')
    if r < 0.99:
        print('   *** WARNING: recon r < 0.99 — nhaa verdict is suspect ***')
    return b


def add_nhaa(dfs, calib):
    """haa_s = hand-normalized reconstructed HAA; nhaa = haa_s minus the
    per-type location component (slope on hand-normalized PlateX, pooled
    across all seasons)."""
    for d in dfs:
        d['haa_s'] = (calib[0] + calib[1] * d['_t_geo']
                      + calib[2] * d['_t_brk']) * d['_s']
        d['plate_x_s'] = d['plate_x'] * d['_s']
    pool = pd.concat([d[['pitch_type', 'haa_s', 'plate_x_s']] for d in dfs],
                     ignore_index=True).dropna()
    slopes = {}
    for pt, sub in pool.groupby('pitch_type'):
        if len(sub) >= 2000:
            slope = float(np.cov(sub.plate_x_s, sub.haa_s)[0, 1]
                          / np.var(sub.plate_x_s))
            slopes[pt] = (slope, float(sub.plate_x_s.mean()))
    print('nHAA slopes (deg/ft, hand-normalized): '
          + ', '.join(f'{pt}:{v[0]:.2f}' for pt, v in sorted(slopes.items())))
    for d in dfs:
        d['nhaa'] = np.nan
        for pt, (slope, xbar) in slopes.items():
            m = (d['pitch_type'] == pt).values
            d.loc[m, 'nhaa'] = (d.loc[m, 'haa_s']
                                - slope * (d.loc[m, 'plate_x_s'] - xbar))


# ── nVAA slopes + dict transform (from scripts/research/stuff/stuff_nvaa_probe.py) ──
def fit_nvaa_slopes(dfs):
    pool = pd.concat([d[['pitch_type', 'vaa', 'plate_z']] for d in dfs],
                     ignore_index=True).dropna()
    out = {}
    for pt, sub in pool.groupby('pitch_type'):
        if len(sub) >= 2000:
            slope = float(np.cov(sub.plate_z, sub.vaa)[0, 1]
                          / np.var(sub.plate_z))
            out[pt] = (slope, float(sub.plate_z.mean()))
    print('nVAA slopes (deg/ft): '
          + ', '.join(f'{pt}:{v[0]:.2f}' for pt, v in sorted(out.items())))
    return out


def transform_nvaa(pitches, slopes):
    out = []
    for p in pitches:
        pt = p.get('Pitch Type')
        v, z = T.sf(p.get('VAA')), T.sf(p.get('PlateZ'))
        if pt in slopes and v is not None and z is not None:
            slope, zbar = slopes[pt]
            q = dict(p)
            q['VAA'] = v - slope * (z - zbar)
            out.append(q)
        else:
            out.append(p)
    return out


def unit_metrics(d, loc_map):
    """(rel, pred, desc, indep, per-family dict). d must carry stuff, half,
    period."""
    a0, a1, est, late = [], [], {}, {}
    fam_est = defaultdict(dict)
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= 50 and len(l) >= 50:
            est[key] = e.stuff.mean(); late[key] = l.target_xrv.mean()
            fam_est[FAMILY.get(key[2], 'other')][key] = None
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= 40 and len(h1) >= 40:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
    ks = list(est)
    rel = pear(a0, a1)
    pred = -pear([est[k] for k in ks], [late[k] for k in ks])
    dx, dy_, fx = [], [], defaultdict(lambda: ([], []))
    for key, g in d.groupby(['pitcher', 'throws', 'pitch_type']):
        if len(g) >= 100:
            dx.append(g.stuff.mean()); dy_.append(g.target_xrv.mean())
            f = FAMILY.get(key[2], 'other')
            fx[f][0].append(g.stuff.mean()); fx[f][1].append(g.target_xrv.mean())
    desc = -pear(dx, dy_)
    fams = {}
    for f, (xs, ys) in fx.items():
        if len(xs) >= 40:
            ep = [(est[k], late[k]) for k in ks if FAMILY.get(k[2]) == f]
            fams[f] = {'desc': -pear(xs, ys),
                       'pred': (-pear([a for a, _ in ep], [b for _, b in ep])
                                if len(ep) >= 40 else float('nan'))}
    ix, iy = [], []
    for key, g in d.groupby(['pitcher', 'team', 'pitch_type']):
        if len(g) >= 50 and key in loc_map:
            ix.append(g.stuff.mean()); iy.append(loc_map[key])
    indep = pear(ix, iy) if len(ix) > 50 else float('nan')
    return rel, pred, desc, indep, fams, len(a0), len(ks)


def run_variant(name, df26, prior, feats, mono=('velocity',), mask_vd=False):
    t0 = time.time()
    d26 = df26.copy()
    dpr = prior
    if mask_vd:
        d26 = d26.copy()
        dpr = prior.copy()
        for dd in (d26, dpr):
            dd.loc[~dd['pitch_type'].isin(OFFSPEED), 'velo_diff'] = np.nan
    X = T.design(d26, feats)
    Xp = T.design(dpr, feats).reindex(columns=X.columns, fill_value=0)
    params = dict(T.TUNED)
    params['monotone_constraints'] = tuple(
        -1 if c in mono else 0 for c in X.columns)
    y = d26['target_xrv'].values
    yp = dpr['target_xrv'].values
    groups = d26['pitcher'].values
    oof = np.full(len(d26), np.nan)
    for k, (tr, te) in enumerate(
            GroupKFold(n_splits=8).split(X, y, groups)):
        m = xgb.XGBRegressor(**params)
        m.fit(pd.concat([X.iloc[tr], Xp], ignore_index=True),
              np.concatenate([y[tr], yp]))
        oof[te] = m.predict(X.iloc[te])
    d26 = d26.assign(stuff=-oof)
    return d26, time.time() - t0


def main():
    print('loading 2026 ...', flush=True)
    p26 = load_mlb_2026()
    df26 = build_season(p26, (T.LG_WOBA, T.WOBA_SCALE))

    print('loading priors ...', flush=True)
    priors = []
    for yr in YEARS:
        pk = pickle.load(open(
            os.path.join(ROOT, 'data', f'_pitches{yr}_training.pkl'), 'rb'))
        d = build_season(pk, GUTS[yr],
                         harmonize_against=p26 if yr == 2025 else None)
        d['_yr'] = yr
        priors.append(d)
        print(f'  {yr}: {len(d)} rows', flush=True)
    prior = pd.concat(priors, ignore_index=True)
    del priors

    # ── coverage audit: a candidate with a season-shaped hole is disqualified
    #    before it runs (the axis_dev lesson) ──
    print('\ncandidate coverage by season (%):')
    for col in ('cross', 'ax_sin', 'plate_x', 'plate_z', 'vaa'):
        row = [f'2026:{df26[col].notna().mean()*100:.1f}']
        for yr in YEARS:
            sub = prior[prior._yr == yr]
            row.append(f'{yr}:{sub[col].notna().mean()*100:.1f}')
        print(f'  {col:<8} ' + '  '.join(row))

    # HAA recon + calibration + nhaa (all seasons, one derivation)
    df26 = add_haa_recon(df26)
    prior = add_haa_recon(prior)
    calib = calibrate_haa(df26)
    add_nhaa([df26, prior], calib)
    print(f'  nhaa coverage: 2026 {df26.nhaa.notna().mean()*100:.1f}%, '
          f'prior {prior.nhaa.notna().mean()*100:.1f}%')

    # nVAA slopes from the pooled emitted columns, then family-B rebuild
    slopes = fit_nvaa_slopes([df26, prior])

    with open(PITCH_LB) as f:
        lb = json.load(f)
    loc_map = {(r['pitcher'], r['team'], r['pitchType']): -r['locPlusRaw']
               for r in lb if r.get('locPlusRaw') is not None}

    # date splits for metrics (fixed once, shared by every variant)
    order = {d: i for i, d in enumerate(sorted(df26['date'].dropna().unique()))}
    df26['half'] = df26['date'].map(order).fillna(0).astype(int) % 2
    df26['period'] = np.where(df26['date'] < '2026-05-01', 'early', 'late')

    BASE = list(T.BASE_FEATS)
    CROSS_FEATS = [f for f in BASE if f not in ('axis_dev', 'axis_dev_abs')] \
        + ['cross', 'cross_abs']

    results = {}

    def report(name, d26, dt):
        rel, pred, desc, indep, fams, n_rel, n_pred = unit_metrics(d26, loc_map)
        results[name] = dict(rel=rel, pred=pred, desc=desc, indep=indep,
                             fams=fams)
        fam_s = '  '.join(
            f'{f[:4]} d{v["desc"]:.3f}/p{v["pred"]:.3f}'
            for f, v in sorted(fams.items()))
        print(f'{name:<10} reliab {rel:.4f}  pred {pred:.4f}  desc {desc:.4f}  '
              f'locIndep {indep:+.4f}  (n_rel {n_rel}, n_pred {n_pred})  '
              f'[{dt:.0f}s]\n           {fam_s}', flush=True)

    print('\n=== FAMILY A (raw approach angles) ===', flush=True)
    for name, feats, kw in (
            ('BASE', BASE, {}),
            ('NHAA', BASE + ['nhaa'], {}),
            ('CROSS', CROSS_FEATS, {}),
            ('AXPOS', BASE + ['ax_sin', 'ax_cos'], {}),
            ('MONOEXT', BASE, {'mono': ('velocity', 'extension')}),
            ('VDMASK', BASE, {'mask_vd': True})):
        d26, dt = run_variant(name, df26, prior, feats, **kw)
        report(name, d26, dt)

    print('\n=== FAMILY B (nVAA rebuild) ===', flush=True)
    df26_b = build_season(transform_nvaa(p26, slopes),
                          (T.LG_WOBA, T.WOBA_SCALE))
    del p26
    priors_b = []
    for yr in YEARS:
        pk = pickle.load(open(
            os.path.join(ROOT, 'data', f'_pitches{yr}_training.pkl'), 'rb'))
        if yr == 2025:
            # same deterministic harmonization as pass 1 (needs raw 2026 tags,
            # which the transform does not touch — reload to be exact)
            p26h = load_mlb_2026()
            T._harmonize_tags(pk, p26h)
            del p26h
        d = build_season(transform_nvaa(pk, slopes), GUTS[yr])
        priors_b.append(d)
        print(f'  {yr}: {len(d)} rows (nVAA)', flush=True)
    prior_b = pd.concat(priors_b, ignore_index=True)
    del priors_b
    df26_b = add_haa_recon(df26_b)
    prior_b = add_haa_recon(prior_b)
    add_nhaa([df26_b, prior_b], calib)
    df26_b['half'] = df26_b['date'].map(order).fillna(0).astype(int) % 2
    df26_b['period'] = np.where(df26_b['date'] < '2026-05-01', 'early', 'late')
    if len(df26_b) != len(df26):
        print(f'  *** WARNING: family B row count {len(df26_b)} != family A '
              f'{len(df26)} — folds no longer perfectly paired ***')

    for name, feats in (('NVAA', BASE), ('NVAA_NHAA', BASE + ['nhaa'])):
        d26, dt = run_variant(name, df26_b, prior_b, feats)
        report(name, d26, dt)

    print('\n=== PAIRED DELTAS vs BASE ===')
    b = results['BASE']
    for name, r in results.items():
        if name == 'BASE':
            continue
        print(f'{name:<10} d_rel {r["rel"]-b["rel"]:+.4f}  '
              f'd_pred {r["pred"]-b["pred"]:+.4f}  '
              f'd_desc {r["desc"]-b["desc"]:+.4f}  '
              f'd_|locIndep| {abs(r["indep"])-abs(b["indep"]):+.4f}')
    with open(os.path.join(ROOT, 'data', '_battery_2026_08_results.json'),
              'w') as f:
        json.dump({k: {m: v for m, v in r.items() if m != 'fams'}
                   for k, r in results.items()}, f, indent=1)
    print('\nresults json -> data/_battery_2026_08_results.json')


if __name__ == '__main__':
    main()
