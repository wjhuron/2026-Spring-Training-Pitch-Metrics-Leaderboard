#!/usr/bin/env python3
"""stuff_gate_v2.py — Stuff+ replicate gate, version 2 (2026-08-23,
repaired 2026-09-02).

Why a v2. The v1 gate (stuff_pertype_loso_gate.py) had three problems:
  1. It applied the nVAA transform on the pitch dicts and then build_df
     applied the frozen NVAA_SLOPES again, so every variant (SHIPPED
     included) was scored on a DOUBLE-adjusted VAA once v12 shipped.
  2. Its adoption objective (fut_r) is a first-half -> second-half split
     within the held-out season. Non-stationary traits inflate it for every
     config, and it never asks the question Stuff+ exists for: does the
     grade carry to NEXT season?
  3. It had no standard error. v13 shipped on a +0.004 mean delta with a
     placebo spread of the same size.

Protocol. Seasons 2021-2026. A replicate is a PAIR (Y, Y+1), Y in
2021..2025. The model is fit on every season EXCEPT Y and Y+1 (so the
next-season target is never a training label), predicts Y, and is scored:

  pitch_r   r(pred, -target) over Y pitches
  fut_r     pitcher mean pred, first half of Y -> mean -target, second half
  unit_r    same at (pitcher, pitch_type)
  rel       split-half reliability of the pitcher-unit score within Y
  nxt_r     pitcher mean pred over Y (>=MIN_NXT pitches) -> mean -target
            over Y+1 (>=MIN_NXT pitches)            <- THE adoption objective
  nxt_unit_r  same at (pitcher, pitch_type) (>=MIN_NXT_U each season)
  nxt_rv_r  pitcher mean pred over Y -> mean actual RV (rv_raw, luck
            included) over Y+1. Secondary: the outcome the public models
            validate on.

2026-09-02 repairs (see the audit in memory/project_stuff_gate_v2_2026_08):
  * TWO grades are scored for every metric. "raw" = the pitcher's mean of
    the pooled xRV prediction (what the gate scored on 08-23). "rend" = the
    RENDERED unit: per-pitch atom int(round(100 + 10*(raw - mu_t)/sd_t))
    with (mu_t, sd_t) the mean and SD of (pitcher, pitch_type) unit means
    with >= ANCHOR_MIN pitches within season Y (train_stuff._standardize;
    ANCHOR_BORROW KN->CH/FS, SV->SL), pitcher grade = pitch-weighted mean of
    atoms. Metric keys carry a _rend suffix. Deviation from production: the
    unit is (pitcher, pitch_type), not (pitcher, team, pitch_type).
  * height is CLIPPED to T.HEIGHT_CLIP by default (v14.1); 'height_raw' is
    the unclipped candidate. Missing heights impute to T.HEIGHT_LEAGUE_MEAN
    as production does (the 08-23 gate left them NaN).
  * Seed variance: the summary estimates the per-pair seed SD of d_nxt
    from every reference config with >= 2 seeds paired with SHIPPED, and
    adds seed_sd^2 / m_seeds in quadrature to the bootstrap SE.
  * The 2025->2026 pair scores a PARTIAL 2026 target, and 2026 is a
    TRAINING season for the other four pairs. Every summary line flags it
    and the pooled result is reported with and without it.
  * Frames are float32 (8 GB machine); the design matrix is built without
    copying the season frame.
  * Spec keys 'frames' (season override, e.g. {"2025": "2025H"} for the
    harmonized 2025 frame built by `build-harm`), 'identity' (K pitcher-
    grouped folds: each fold's Y pitchers are removed from every training
    season before the fit that scores them) and 'subsample_rows' (random
    row fraction of the training pool, the sample-size control for
    identity), 'seeds' (per-variant seed list).

SHIPPED tracks T.BASE_FEATS / T.TUNED / T.HEIGHT_CLIP / T.VD_MASK_TYPES at
run time. After a config change, move data/_gate_v2/agg_SHIPPED_* aside
and delete the SHIPPED block of results.json, or the cached baseline is
the previous version.

Per-pitcher aggregates are cached per (variant, pair, seed), so the summary
computes a PAIRED pitcher-bootstrap SE on every delta vs SHIPPED without a
refit, and new variants never refit SHIPPED.

VAA is handled ONCE, post hoc from vaa_raw, with slopes fit on the training
seasons of each pair. Variant key 'vaa': 'raw' skips the adjustment.

Spec (JSON, --spec):
  {
    "DEPTH5":    {"params": {"max_depth": 5}},
    "NOCLIP":    {"replace": {"height": "height_raw"}},
    "ACCEL_ADD": {"add": ["acc_v", "acc_h", "acc_v_diff", "acc_h_diff"]},
    "VD_NOMASK": {"mask": {"velo_diff": []}},
    "NOKIN":     {"drop": ["kin_eff__ff"]},
    "HARMONIZE": {"frames": {"2025": "2025H"}},
    "IDENTITY":  {"identity": 5},
    "RAW_VAA":   {"vaa": "raw"},
    "ANCHOR_ANY": {"anchor": "any"}      # most-thrown of FF/SI/FC
  }
  keys: add, drop, replace, mask {feat: [types]}, add_typed {feat: [types]},
        params, vaa, anchor ('true' default | 'any'), frames, identity,
        subsample_rows, seeds

Usage:
  python3 scripts/research/stuff/stuff_gate_v2.py build
  python3 scripts/research/stuff/stuff_gate_v2.py build-harm
  python3 scripts/research/stuff/stuff_gate_v2.py run --spec spec.json [--pairs 2021,2024] [--seeds 0,1] [--names A,B]
  python3 scripts/research/stuff/stuff_gate_v2.py summary [--names A,B]
  python3 scripts/research/stuff/stuff_gate_v2.py needmore [--names A,B]
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import stuff_plus.train_stuff as T                      # noqa: E402

SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
PAIRS = [(y, y + 1) for y in (2021, 2022, 2023, 2024, 2025)]
PARTIAL_Y = 2025          # the (2025, 2026) pair targets a partial season
# FanGraphs Guts per season (scripts/builders/build_historical_training_set.py
# GUTS, copied because that module's imports are stale post-reorg)
GUTS = {2021: (0.314, 1.209), 2022: (0.310, 1.259),
        2023: (0.318, 1.204), 2024: (0.310, 1.242),
        2025: (T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE)}
CACHE = os.path.join(ROOT, 'data', '_gate_v2')
RESULTS = os.path.join(CACHE, 'results.json')
BASE = list(T.BASE_FEATS)
MIN_HALF_P, MIN_HALF_U = 200, 100
MIN_NXT, MIN_NXT_U = 300, 150
SLOPE_MIN = 2000
ANCHOR_MIN = T.QUAL_N          # 50: unit pitches to enter the (mu, sd) pool
K_SCALE = T.K_SCALE            # 10 points per anchor SD
BOOT_B = 2000
SEED_REFS = ('SHIPPED', 'DEPTH5')   # configs whose extra seeds estimate seed SD
NEEDMORE_Z = 1.5               # |mean d| < NEEDMORE_Z * combined SE -> 2 more seeds

KEEP = ['pitcher', 'team', 'throws', 'date', 'pitch_type', 'platoon_same',
        'target_xrv', 'rv_raw', 'velocity', 'ivb', 'hb', 'spin_rate',
        'extension', 'arm_angle', 'rel_x', 'rel_z', 'cross', 'cross_abs',
        'kin_eff', 'kin_eff__ff', 'vaa_raw', 'plate_x', 'plate_z', 'ax_sin',
        'ax_cos', 'spin_eff', 'axis_dev', 'axis_dev_abs', 'haa_meas',
        'kin_dev', 'kin_cd']
F64 = ('target_xrv', 'rv_raw')     # kept at full precision


def pear(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float('nan')
    return float(np.corrcoef(a[m], b[m])[0, 1])


# ── build: one cached frame per season, VAA raw ──────────────────────────
def season_path(y):
    return os.path.join(CACHE, f'season_{y}.pkl')


def load_2026():
    allp = pickle.load(open(os.path.join(ROOT, 'data',
                                         'all_pitches_rs_cache.pkl'), 'rb'))
    ep = {(p.get('Pitcher'), p.get('PTeam')) for p in allp
          if p.get('Pitch Type') == 'EP'}
    p26 = [p for p in allp if p.get('_source') == 'MLB'
           and (p.get('Pitcher'), p.get('PTeam')) not in ep]
    del allp
    n = T.apply_kin_sidecar(p26)
    if n < 0:
        sys.exit('kinematics sidecar missing; aborting (fail closed)')
    dates = sorted(p.get('Game Date') for p in p26 if p.get('Game Date'))
    print(f'  2026: {len(p26)} MLB pitches through {dates[-1]}, sidecar '
          f'filled {n}')
    return p26


def _frame_from_pitches(pk, guts, y):
    saved = T.NVAA_SLOPES
    T.NVAA_SLOPES = {}            # raw VAA in the cache; adjusted per pair
    lg, sc = T.LG_WOBA, T.WOBA_SCALE
    T.LG_WOBA, T.WOBA_SCALE = guts
    try:
        d = T.build_df(pk)
    finally:
        T.LG_WOBA, T.WOBA_SCALE = lg, sc
        T.NVAA_SLOPES = saved
    d = d[d['target_xrv'].notna()].reset_index(drop=True)
    d = d[[c for c in KEEP if c in d.columns]].copy()
    for c in d.columns:
        if c not in ('pitcher', 'team', 'throws', 'date', 'pitch_type'):
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d['season'] = y
    return d


def build():
    os.makedirs(CACHE, exist_ok=True)
    for y in SEASONS:
        if os.path.exists(season_path(y)):
            print(f'  {y}: cached')
            continue
        t0 = time.time()
        if y == 2026:
            pk = load_2026()
            guts = (T.LG_WOBA, T.WOBA_SCALE)
        else:
            path = T.HIST_PKL.format(year=y) if y != 2025 else T.PRIOR_PKL
            pk = pickle.load(open(path, 'rb'))
            guts = GUTS[y]
        d = _frame_from_pitches(pk, guts, y)
        del pk
        gc.collect()
        d.to_pickle(season_path(y))
        print(f'  {y}: {len(d)} rows, {time.time()-t0:.0f}s', flush=True)
        del d
        gc.collect()


def build_harm():
    """season_2025H.pkl: the 2025 pickle with T._harmonize_tags applied
    against the 2026 MLB pitches, exactly as a production retrain does."""
    out = season_path('2025H')
    if os.path.exists(out):
        print('  2025H: cached')
        return
    t0 = time.time()
    p26 = load_2026()
    p25 = pickle.load(open(T.PRIOR_PKL, 'rb'))
    tags_before = pd.Series([p.get('Pitch Type') for p in p25]).value_counts()
    T._harmonize_tags(p25, p26)
    del p26
    gc.collect()
    tags_after = pd.Series([p.get('Pitch Type') for p in p25]).value_counts()
    print('  2025 tag counts before -> after:')
    for t in sorted(set(tags_before.index) | set(tags_after.index)):
        b, a = int(tags_before.get(t, 0)), int(tags_after.get(t, 0))
        if b != a:
            print(f'    {t}: {b} -> {a} ({a - b:+d})')
    d = _frame_from_pitches(p25, GUTS[2025], 2025)
    del p25
    gc.collect()
    d.to_pickle(out)
    print(f'  2025H: {len(d)} rows, {time.time()-t0:.0f}s', flush=True)


def load_frame(key):
    d = pd.read_pickle(season_path(key))
    for c in d.columns:
        if d[c].dtype == np.float64 and c not in F64:
            d[c] = d[c].astype('float32')
    return d


# ── per-pair preparation: VAA, anchors, derived candidates ───────────────
def fit_vaa_slopes(dfs):
    pool = pd.concat([d[['pitch_type', 'vaa_raw', 'plate_z']] for d in dfs],
                     ignore_index=True).dropna()
    out = {}
    for pt, sub in pool.groupby('pitch_type'):
        if len(sub) >= SLOPE_MIN:
            z = sub.plate_z.values.astype(float)
            v = sub.vaa_raw.values.astype(float)
            slope = float(np.cov(z, v)[0, 1] / np.var(z))
            out[pt] = (slope, float(z.mean()))
    return out


def primary_type(d, anchor='true'):
    """Per (pitcher, throws): the reference fastball type, production rule."""
    fb = d[d['pitch_type'].isin(T.FB_TYPES)]
    cnt = fb.groupby(['pitcher', 'throws', 'pitch_type']).size()
    out = {}
    for (pit, thr), g in cnt.groupby(level=[0, 1]):
        by = {pt: n for (_, _, pt), n in g.items()}
        if anchor == 'true':
            if pit in T.FC_ANCHOR_PITCHERS and 'FC' in by:
                cand = {'FC': by['FC']}
            else:
                cand = {pt: n for pt, n in by.items() if pt in ('FF', 'SI')} or by
        else:
            cand = by
        out[(pit, thr)] = max(cand, key=cand.get)
    return out


def prepare(d, slopes, anchor='true'):
    """Adds vaa, vaa_diff, velo_diff(+raw), ivb_diff, hb_diff, and the
    derived candidate features. Reference means are per season frame, as
    production builds them."""
    d = d.copy()
    # VAA: adjusted once, from raw
    adj = np.zeros(len(d))
    if slopes:
        for pt, (sl, zbar) in slopes.items():
            m = (d['pitch_type'] == pt).values
            adj[m] = sl * (d.loc[m, 'plate_z'].values - zbar)
    adj = np.where(np.isfinite(adj), adj, 0.0)
    d['vaa'] = (d['vaa_raw'] - adj).astype('float32')
    # acceleration-equivalent movement (derived; priors carry no ax/az).
    # break = 0.5*a*t^2  ->  a ~ 2*break/t^2, t from release to plate at
    # release speed. Units: in/s^2 (scale irrelevant to a tree).
    dist = (60.5 - d['extension']) - 17.0 / 12.0
    tof = dist / (d['velocity'] * 1.46667)
    d['acc_v'] = 2.0 * d['ivb'] / tof ** 2
    d['acc_h'] = 2.0 * d['hb'] / tof ** 2
    # movement direction vs the arm plane (Pitch Profiler "difference from
    # arm angle"). Movement angle from horizontal in the hand-normalized
    # frame; sign of hb fixed so that arm-side run is positive (measured on
    # FF in this frame; see build log). arm_dev = mov_angle - arm_angle.
    mov = np.degrees(np.arctan2(d['ivb'], d['hb'] * ARM_SIDE_SIGN))
    d['arm_dev'] = mov - d['arm_angle']
    d['arm_dev_abs'] = d['arm_dev'].abs()
    # height (v14): stature in inches. Missing -> frozen league mean, as
    # production does (train_stuff.build_df); logged.
    d['height_raw'] = d['pitcher'].map(height_map()).astype('float32')
    miss = d['height_raw'].isna()
    if miss.any():
        print(f'    height impute: {int(miss.sum())} pitches from '
              f'{d.loc[miss, "pitcher"].nunique()} pitchers -> '
              f'{T.HEIGHT_LEAGUE_MEAN}')
        d['height_raw'] = d['height_raw'].fillna(T.HEIGHT_LEAGUE_MEAN)
    # v14.1: the shipped feature is the CLIPPED stature (the Schultz sweeper
    # lesson: above 80in the pool is ~3 pitchers). Pool percentiles
    # (pitch-weighted): p1=70, p99=80.
    d['height'] = d['height_raw'].clip(*T.HEIGHT_CLIP)
    d['height_w79'] = d['height_raw'].clip(70, 79)
    d['rel_z_rel'] = d['rel_z'] * 12.0 - d['height_raw']   # release height vs stature
    # release consistency: SD of release point within (pitcher, type) and
    # (pitcher) over the season frame
    g = d.groupby(['pitcher', 'throws', 'pitch_type'])
    d['rel_sd_x'] = g['rel_x'].transform('std')
    d['rel_sd_z'] = g['rel_z'].transform('std')
    ga = d.groupby(['pitcher', 'throws'])
    d['rel_sd_all'] = np.sqrt(ga['rel_x'].transform('var')
                              + ga['rel_z'].transform('var'))
    # fastball reference and differentials
    prim = primary_type(d, anchor)
    key = list(zip(d['pitcher'], d['throws']))
    d['_prim'] = [prim.get(k) for k in key]
    is_ref = (d['pitch_type'] == d['_prim']).values
    ref = d[is_ref].groupby(['pitcher', 'throws'])[
        ['velocity', 'ivb', 'hb', 'vaa', 'acc_v', 'acc_h']].mean()
    ref.columns = ['r_v', 'r_iv', 'r_hb', 'r_vaa', 'r_av', 'r_ah']
    d = d.join(ref, on=['pitcher', 'throws'])
    d['velo_diff_raw'] = d['velocity'] - d['r_v']
    d['velo_diff'] = d['velo_diff_raw'].where(
        ~d['pitch_type'].isin(T.VD_MASK_TYPES))
    d['ivb_diff'] = d['ivb'] - d['r_iv']
    d['hb_diff'] = d['hb'] - d['r_hb']
    d['vaa_diff'] = d['vaa'] - d['r_vaa']
    d['acc_v_diff'] = d['acc_v'] - d['r_av']
    d['acc_h_diff'] = d['acc_h'] - d['r_ah']
    d = d.drop(columns=['_prim', 'r_v', 'r_iv', 'r_hb', 'r_vaa',
                        'r_av', 'r_ah'])
    for c in d.columns:
        if d[c].dtype == np.float64 and c not in F64:
            d[c] = d[c].astype('float32')
    return d


ARM_SIDE_SIGN = 1.0
_HEIGHT = None


def height_map():
    """pitcher name -> height (inches), from data/_gate_v2/pitcher_height.json
    (rebuilt 2026-09-02 by gate_v2_height_map.py: Statcast ids 2021-25 +
    mlb_id_cache + pitcher_heights.json, heights from the shipped asset and
    the MLB Stats API). Ambiguous names (6) take the mean."""
    global _HEIGHT
    if _HEIGHT is None:
        j = json.load(open(os.path.join(CACHE, 'pitcher_height.json')))
        H = {int(k): v for k, v in j['height_in'].items()}
        _HEIGHT = {}
        for n, ids in j['name_to_ids'].items():
            hs = [H[i] for i in ids if i in H]
            if hs:
                _HEIGHT[n] = float(np.mean(hs))
    return _HEIGHT


def set_arm_side_sign(dfs):
    """hb is hand-normalized (hb_raw * s). Decide which sign is arm side
    from the FF mean, so arm_dev is defined the same on every season."""
    global ARM_SIDE_SIGN
    m = np.mean([d.loc[d['pitch_type'] == 'FF', 'hb'].mean() for d in dfs])
    ARM_SIDE_SIGN = 1.0 if m > 0 else -1.0
    print(f'  FF mean hb (hand-normalized) {m:+.2f} -> arm-side sign '
          f'{ARM_SIDE_SIGN:+.0f}')


def variant_feats(spec):
    feats = list(BASE)
    for old, new in (spec.get('replace') or {}).items():
        feats[feats.index(old)] = new
    for f in (spec.get('drop') or []):
        feats.remove(f)
    for f in (spec.get('add') or []):
        if f not in feats:
            feats.append(f)
    for f, types in (spec.get('add_typed') or {}).items():
        feats.append(f'{f}__{"".join(types)}')
    return feats


def design(d, spec, feats):
    """Design matrix for one prepared frame, without copying the frame."""
    cols = {}
    typed = {f'{f}__{"".join(types)}': (f, types)
             for f, types in (spec.get('add_typed') or {}).items()}
    mask = spec.get('mask') or {}
    for f in feats:
        if f in typed:
            src, types = typed[f]
            v = np.where(d['pitch_type'].isin(set(types)), d[src], np.nan)
        elif f == 'velo_diff' and 'velo_diff' in mask:
            v = d['velo_diff_raw'].values
        else:
            v = d[f].values
        cols[f] = np.asarray(v, dtype='float32')
    X = pd.DataFrame(cols)
    for f, types in mask.items():
        if f in X.columns:
            X.loc[d['pitch_type'].isin(set(types)).values, f] = np.nan
    X['platoon_same'] = d['platoon_same'].values
    return X


# ── rendered-unit grade ──────────────────────────────────────────────────
def anchors_for(dY):
    """(mu, sd) per pitch type from (pitcher, pitch_type) unit means with
    >= ANCHOR_MIN pitches in season Y, train_stuff._standardize rules."""
    a = dY.groupby(['pitcher', 'pitch_type'])['stuff'].agg(
        rawmean='mean', n='size').reset_index()
    out = {}
    for pt, sub in a.groupby('pitch_type'):
        q = sub[sub['n'] >= ANCHOR_MIN]
        base = q if len(q) >= 5 else sub
        if pt in T.ANCHOR_BORROW:
            donor = a[a['pitch_type'].isin(T.ANCHOR_BORROW[pt])]
            dq = donor[donor['n'] >= ANCHOR_MIN]
            base = dq if len(dq) >= 5 else donor
        out[pt] = (float(base['rawmean'].mean()), float(base['rawmean'].std()),
                   int(len(q)))
    return out


def atoms(dY, anc):
    mu = dY['pitch_type'].map({k: v[0] for k, v in anc.items()}).astype(float)
    sd = dY['pitch_type'].map({k: v[1] for k, v in anc.items()}).astype(float)
    z = (dY['stuff'].astype(float) - mu) / sd
    at = np.rint(100.0 + K_SCALE * z)
    return np.where(np.isfinite(at) & (sd > 0), at, np.nan)


# ── metrics ──────────────────────────────────────────────────────────────
def half_index(d):
    order = {x: i for i, x in enumerate(sorted(d['date'].dropna().unique()))}
    return (d['date'].map(order).fillna(0).astype(int) % 2).values


def aggregates(dY, dY1):
    """Per-pitcher and per-unit aggregates for Y and Y+1 (cached). 's' is
    the raw pooled-prediction mean, 's_r' the rendered-atom mean."""
    half = half_index(dY)
    anc = anchors_for(dY)
    dY = dY.assign(_h=half, _neg=-dY['target_xrv'], atom=atoms(dY, anc))
    dY1 = dY1.assign(_neg=-dY1['target_xrv'], _rv=-dY1['rv_raw'])  # pitcher-positive
    # within-Y halves
    gp = dY.groupby(['pitcher', '_h']).agg(s=('stuff', 'mean'),
                                            s_r=('atom', 'mean'),
                                            t=('_neg', 'mean'),
                                            n=('stuff', 'size')).unstack('_h')
    gu = dY.groupby(['pitcher', 'pitch_type', '_h']).agg(
        s=('stuff', 'mean'), s_r=('atom', 'mean'), t=('_neg', 'mean'),
        n=('stuff', 'size')).unstack('_h')
    # season Y -> Y+1
    py = dY.groupby('pitcher').agg(s=('stuff', 'mean'), s_r=('atom', 'mean'),
                                   n=('stuff', 'size'))
    py1 = dY1.groupby('pitcher').agg(t=('_neg', 'mean'), rv=('_rv', 'mean'),
                                     n=('_neg', 'size'))
    uy = dY.groupby(['pitcher', 'pitch_type']).agg(s=('stuff', 'mean'),
                                                   s_r=('atom', 'mean'),
                                                   n=('stuff', 'size'))
    uy1 = dY1.groupby(['pitcher', 'pitch_type']).agg(t=('_neg', 'mean'),
                                                     n=('_neg', 'size'))
    return dict(pitch=(dY['stuff'].values.astype('float32'),
                       dY['_neg'].values.astype('float32')),
                gp=gp, gu=gu, py=py, py1=py1, uy=uy, uy1=uy1, anchors=anc)


def metrics_from(agg):
    s, t = agg['pitch']
    out = dict(pitch_r=pear(s, t))
    gp, gu = agg['gp'], agg['gu']
    m = (gp[('n', 0)] >= MIN_HALF_P) & (gp[('n', 1)] >= MIN_HALF_P)
    out['fut_r'] = pear(gp.loc[m, ('s', 0)], gp.loc[m, ('t', 1)])
    out['fut_r_rend'] = pear(gp.loc[m, ('s_r', 0)], gp.loc[m, ('t', 1)])
    out['n_fut'] = int(m.sum())
    m = (gu[('n', 0)] >= MIN_HALF_U) & (gu[('n', 1)] >= MIN_HALF_U)
    out['unit_r'] = pear(gu.loc[m, ('s', 0)], gu.loc[m, ('t', 1)])
    out['rel'] = pear(gu.loc[m, ('s', 0)], gu.loc[m, ('s', 1)])
    j = nxt_table(agg)
    out['nxt_r'] = pear(j['s'], j['t'])
    out['nxt_r_rend'] = pear(j['s_r'], j['t'])
    out['nxt_rv_r'] = pear(j['s'], j['rv'])
    out['nxt_rv_r_rend'] = pear(j['s_r'], j['rv'])
    out['n_nxt'] = int(len(j))
    ju = agg['uy'].join(agg['uy1'], lsuffix='_y', rsuffix='_y1', how='inner')
    ju = ju[(ju['n_y'] >= MIN_NXT_U) & (ju['n_y1'] >= MIN_NXT_U)]
    out['nxt_unit_r'] = pear(ju['s'], ju['t'])
    out['nxt_unit_r_rend'] = pear(ju['s_r'], ju['t'])
    out['n_nxt_unit'] = int(len(ju))
    out['anchors'] = {k: {'mu': v[0], 'sd': v[1], 'nqual': v[2]}
                      for k, v in agg['anchors'].items()}
    return out


def nxt_table(agg):
    j = agg['py'].join(agg['py1'], lsuffix='_y', rsuffix='_y1', how='inner')
    j = j[(j['n_y'] >= MIN_NXT) & (j['n_y1'] >= MIN_NXT)]
    return j[['s', 's_r', 't', 'rv']]


# ── run ──────────────────────────────────────────────────────────────────
def agg_path(name, Y, seed):
    return os.path.join(CACHE, f'agg_{name}_{Y}_s{seed}.pkl')


def fit_predict(Xtr, ytr, XY, spec, seed):
    params = T._params_for(Xtr)
    params.update(spec.get('params') or {})
    params['monotone_constraints'] = tuple(
        -1 if c == T.MONO_FEAT else 0 for c in Xtr.columns)
    params['random_state'] = seed
    m = xgb.XGBRegressor(**params)
    m.fit(Xtr, ytr)
    out = m.predict(XY)
    del m
    return out


def run(spec_path, pairs, seeds, names_only=None):
    with open(spec_path) as f:
        variants = json.load(f)
    if names_only:
        variants = {k: v for k, v in variants.items() if k in names_only}
    results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    frames = {y: load_frame(y) for y in SEASONS}
    set_arm_side_sign(list(frames.values()))
    names = (['SHIPPED'] if 'SHIPPED' not in variants else []) + list(variants)
    for Y, Y1 in pairs:
        train_years = [y for y in SEASONS if y not in (Y, Y1)]
        todo = [(n, s) for n in names
                for s in (variants.get(n, {}).get('seeds') or seeds)
                if not os.path.exists(agg_path(n, Y, s))]
        if not todo:
            print(f'=== pair {Y}->{Y1}: all cached')
            continue
        print(f'\n=== pair {Y}->{Y1} (train {train_years})'
              + ('  [PARTIAL Y+1]' if Y == PARTIAL_Y else '') + ' ===',
              flush=True)
        # group by preparation key so one prepared set lives at a time
        def vkey(spec):
            return (spec.get('vaa', 'nvaa'), spec.get('anchor', 'true'),
                    tuple(sorted((spec.get('frames') or {}).items())))
        groups = {}
        for name, seed in todo:
            groups.setdefault(vkey(variants.get(name, {})), []).append((name, seed))
        for vk, items in groups.items():
            over = dict(vk[2])
            fr = {y: (load_frame(over[str(y)]) if str(y) in over else frames[y])
                  for y in train_years + [Y, Y1]}
            slopes = (fit_vaa_slopes([fr[y] for y in train_years])
                      if vk[0] == 'nvaa' else {})
            P = {y: prepare(fr[y], slopes, vk[1]) for y in train_years + [Y, Y1]}
            del fr
            gc.collect()
            for name, seed in items:
                spec = variants.get(name, {})
                t0 = time.time()
                feats = variant_feats(spec)
                Xtr = pd.concat([design(P[y], spec, feats) for y in train_years],
                                ignore_index=True)
                ytr = np.concatenate([P[y]['target_xrv'].values for y in train_years])
                ptr = np.concatenate([P[y]['pitcher'].values for y in train_years])
                dY = P[Y]
                XY = design(dY, spec, feats)
                rng = np.random.default_rng(1000 + seed)
                if spec.get('identity'):
                    # pitcher-grouped folds: a fold's Y pitchers leave every
                    # training season before the fit that scores them
                    K = int(spec['identity'])
                    pits = np.array(sorted(dY['pitcher'].unique()))
                    fold_of = dict(zip(pits, rng.permutation(len(pits)) % K))
                    fy = dY['pitcher'].map(fold_of).values
                    ftr = np.array([fold_of.get(p, -1) for p in ptr])
                    pred = np.full(len(dY), np.nan, dtype='float32')
                    for k in range(K):
                        keep = ftr != k
                        pred[fy == k] = fit_predict(Xtr[keep], ytr[keep],
                                                    XY[fy == k], spec, seed)
                        print(f'    fold {k}: train {int(keep.sum())} rows, '
                              f'{int((fy == k).sum())} scored '
                              f'[{time.time()-t0:.0f}s]', flush=True)
                    n_train = int(len(Xtr))
                elif spec.get('subsample_rows'):
                    keep = rng.random(len(Xtr)) < float(spec['subsample_rows'])
                    pred = fit_predict(Xtr[keep], ytr[keep], XY, spec, seed)
                    n_train = int(keep.sum())
                else:
                    pred = fit_predict(Xtr, ytr, XY, spec, seed)
                    n_train = int(len(Xtr))
                dY = dY.assign(stuff=-pred)
                agg = aggregates(dY, P[Y1])
                met = metrics_from(agg)
                met['n_train'] = n_train
                met['feats'] = feats
                met['partial_target'] = (Y == PARTIAL_Y)
                pd.to_pickle(agg, agg_path(name, Y, seed))
                results.setdefault(name, {}).setdefault(str(Y), {})[str(seed)] = met
                tmp = RESULTS + '.tmp'
                with open(tmp, 'w') as f:
                    json.dump(results, f, indent=1)
                os.replace(tmp, RESULTS)
                print(f'  {name:<12} s{seed} pitch {met["pitch_r"]:.4f} '
                      f'fut {met["fut_r"]:.4f} rel {met["rel"]:.4f} | '
                      f'nxt raw {met["nxt_r"]:.4f} rend {met["nxt_r_rend"]:.4f} '
                      f'(n={met["n_nxt"]}) nxt_u {met["nxt_unit_r"]:.4f}/'
                      f'{met["nxt_unit_r_rend"]:.4f} nxt_rv {met["nxt_rv_r"]:.4f}/'
                      f'{met["nxt_rv_r_rend"]:.4f}'
                      + ('  [PARTIAL Y+1]' if Y == PARTIAL_Y else '')
                      + f' [{time.time()-t0:.0f}s]', flush=True)
                del Xtr, XY, dY, agg, ytr, ptr, pred
                gc.collect()
            del P
            gc.collect()


# ── summary with paired pitcher bootstrap + seed variance ────────────────
def boot_delta(a, b, B=BOOT_B, seed=0):
    """a, b: nxt tables (index pitcher, cols s, s_r, t). Paired bootstrap
    over the common pitchers of r(a.s, t) - r(b.s, t), both grades on the
    same resamples."""
    j = a.join(b, lsuffix='_a', rsuffix='_b', how='inner')
    n = len(j)
    t = j['t_a'].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))

    def r(x, y):
        x = x - x.mean(1, keepdims=True)
        y = y - y.mean(1, keepdims=True)
        return (x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))
    out = {}
    for g, col in (('raw', 's'), ('rend', 's_r')):
        sa, sb = j[f'{col}_a'].values, j[f'{col}_b'].values
        d = r(sa[idx], t[idx]) - r(sb[idx], t[idx])
        out[g] = (float(pear(sa, t) - pear(sb, t)), float(d.std()))
    return out, n


def seed_sd(results):
    """Per-pair SD (ddof=1) of the seed-paired delta vs SHIPPED, from every
    SEED_REFS config with >= 2 common seeds. Pooled as the RMS across pairs
    and configs. Returns (pooled {grade: sd}, per-pair detail)."""
    var = {'raw': [], 'rend': []}
    detail = {}
    for name in SEED_REFS:
        if name not in results or name == 'SHIPPED':
            continue
        for Y in results[name]:
            if Y not in results.get('SHIPPED', {}):
                continue
            seeds = sorted(set(results[name][Y]) & set(results['SHIPPED'][Y]))
            if len(seeds) < 2:
                continue
            for g, k in (('raw', 'nxt_r'), ('rend', 'nxt_r_rend')):
                ds = [results[name][Y][s][k] - results['SHIPPED'][Y][s][k]
                      for s in seeds]
                v = float(np.var(ds, ddof=1))
                var[g].append(v)
                detail.setdefault(name, {}).setdefault(Y, {})[g] = {
                    'sd': float(np.sqrt(v)), 'seeds': seeds, 'd': ds}
    pooled = {g: (float(np.sqrt(np.mean(v))) if v else float('nan'))
              for g, v in var.items()}
    return pooled, detail


def pool(rows, sd_seed, key):
    """rows: per-pair dicts. Pooled mean, bootstrap-only SE, combined SE
    (bootstrap + seed_sd^2/m in quadrature), z, wins."""
    k = len(rows)
    if not k:
        return None
    d = np.array([r[f'd_{key}'] for r in rows])
    se_b = np.array([r[f'se_boot_{key}'] for r in rows])
    m = np.array([r['m_seeds'] for r in rows])
    se_c = np.sqrt(se_b ** 2 + (sd_seed ** 2 if np.isfinite(sd_seed) else 0.0) / m)
    mean = float(d.mean())
    sb = float(np.sqrt((se_b ** 2).sum()) / k)
    sc = float(np.sqrt((se_c ** 2).sum()) / k)
    return {'mean_d': mean, 'se_boot': sb, 'se_comb': sc,
            'z_boot': mean / sb if sb else float('nan'),
            'z_comb': mean / sc if sc else float('nan'),
            'wins': int((d > 0).sum()), 'k': k}


def summarize(names=None, quiet=False):
    results = json.load(open(RESULTS))
    results = {k: v for k, v in results.items() if not k.startswith('_')}
    variants = [n for n in results if n != 'SHIPPED']
    if names:
        variants = [n for n in variants if n in names]
    sd_pool, sd_detail = seed_sd(results)
    out = {'_seed_sd': {'pooled': sd_pool, 'detail': sd_detail,
                        'refs': list(SEED_REFS)}, 'variants': {}}
    P = print if not quiet else (lambda *a, **k: None)
    P('\nSHIPPED per pair (seed-mean):')
    for Y in sorted(results['SHIPPED']):
        ms = list(results['SHIPPED'][Y].values())
        P(f'  {Y}->{int(Y)+1}: ' + '  '.join(
            f'{k} {np.mean([m[k] for m in ms]):.4f}'
            for k in ('nxt_r', 'nxt_r_rend', 'nxt_unit_r', 'nxt_rv_r',
                      'nxt_rv_r_rend', 'fut_r', 'rel'))
          + f'  n_nxt {ms[0]["n_nxt"]}  seeds {len(ms)}'
          + ('  [PARTIAL 2026 target]' if int(Y) == PARTIAL_Y else ''))
    P(f'\nseed SD of d_nxt (per pair, pooled RMS over {list(SEED_REFS)}): '
      f'raw {sd_pool["raw"]:.4f}  rend {sd_pool["rend"]:.4f}')
    for name, byY in sd_detail.items():
        for Y, g in byY.items():
            P(f'    {name} {Y}: raw sd {g["raw"]["sd"]:.4f} d={["%+.4f" % x for x in g["raw"]["d"]]}'
              f'  rend sd {g["rend"]["sd"]:.4f} d={["%+.4f" % x for x in g["rend"]["d"]]}')
    for name in variants:
        P(f'\n{name}:')
        rows = []
        for Y in sorted(results[name]):
            if Y not in results['SHIPPED']:
                continue
            seeds = sorted(set(results[name][Y]) & set(results['SHIPPED'][Y]))
            if not seeds:
                continue
            dm = {k: float(np.mean([results[name][Y][s][k] - results['SHIPPED'][Y][s][k]
                                    for s in seeds]))
                  for k in ('nxt_r', 'nxt_r_rend', 'nxt_unit_r', 'nxt_unit_r_rend',
                            'nxt_rv_r', 'nxt_rv_r_rend', 'fut_r', 'unit_r',
                            'pitch_r', 'rel')}
            s0 = seeds[0]
            a = nxt_table(pd.read_pickle(agg_path(name, int(Y), int(s0))))
            b = nxt_table(pd.read_pickle(agg_path('SHIPPED', int(Y), int(s0))))
            bd, n = boot_delta(a, b)
            m = len(seeds)
            row = {'Y': int(Y), 'partial': int(Y) == PARTIAL_Y, 'n': n,
                   'm_seeds': m, 'seeds': seeds,
                   'd_raw': dm['nxt_r'], 'd_rend': dm['nxt_r_rend'],
                   'd_raw_s0': bd['raw'][0], 'd_rend_s0': bd['rend'][0],
                   'se_boot_raw': bd['raw'][1], 'se_boot_rend': bd['rend'][1],
                   'd_unit_raw': dm['nxt_unit_r'], 'd_unit_rend': dm['nxt_unit_r_rend'],
                   'd_rv_raw': dm['nxt_rv_r'], 'd_rv_rend': dm['nxt_rv_r_rend'],
                   'd_fut': dm['fut_r'], 'd_rel': dm['rel'], 'd_pitch': dm['pitch_r']}
            for g in ('raw', 'rend'):
                ssd = sd_pool[g] if np.isfinite(sd_pool[g]) else 0.0
                row[f'se_comb_{g}'] = float(np.sqrt(row[f'se_boot_{g}'] ** 2 + ssd ** 2 / m))
            rows.append(row)
            P(f'  {Y}->{int(Y)+1} d_nxt raw {row["d_raw"]:+.4f} (boot {row["se_boot_raw"]:.4f}, '
              f'comb {row["se_comb_raw"]:.4f})  rend {row["d_rend"]:+.4f} '
              f'(boot {row["se_boot_rend"]:.4f}, comb {row["se_comb_rend"]:.4f})  '
              f'n {n} seeds {m} | d_unit {row["d_unit_raw"]:+.4f}/{row["d_unit_rend"]:+.4f}  '
              f'd_rv {row["d_rv_raw"]:+.4f}/{row["d_rv_rend"]:+.4f}  d_fut {row["d_fut"]:+.4f}  '
              f'd_rel {row["d_rel"]:+.4f}'
              + ('  [PARTIAL 2026 target; 2026 trains the other pairs]'
                 if row['partial'] else ''))
        if not rows:
            continue
        rows4 = [r for r in rows if not r['partial']]
        block = {'pairs': rows}
        for tag, rs in (('all', rows), ('excl_2025_26', rows4)):
            block[tag] = {g: pool(rs, sd_pool[g], g) for g in ('raw', 'rend')}
        out['variants'][name] = block
        for tag, rs in (('ALL 5', rows), ('EXCL 2025-26', rows4)):
            pr, pe = block['all' if tag == 'ALL 5' else 'excl_2025_26']['raw'], \
                     block['all' if tag == 'ALL 5' else 'excl_2025_26']['rend']
            if not pr:
                continue
            P(f'  {tag:<12} raw  mean {pr["mean_d"]:+.4f}  se_boot {pr["se_boot"]:.4f} '
              f'se_comb {pr["se_comb"]:.4f}  z_boot {pr["z_boot"]:+.2f}  '
              f'z_comb {pr["z_comb"]:+.2f}  wins {pr["wins"]}/{pr["k"]}')
            P(f'  {"":<12} rend mean {pe["mean_d"]:+.4f}  se_boot {pe["se_boot"]:.4f} '
              f'se_comb {pe["se_comb"]:.4f}  z_boot {pe["z_boot"]:+.2f}  '
              f'z_comb {pe["z_comb"]:+.2f}  wins {pe["wins"]}/{pe["k"]}')
    full = json.load(open(RESULTS))
    full['_summary'] = out
    tmp = RESULTS + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(full, f, indent=1)
    os.replace(tmp, RESULTS)
    return out


def needmore(names=None):
    """Variants whose |mean d| < NEEDMORE_Z x combined SE on EITHER grade
    with fewer than 3 seeds (rule 4: they get two more seeds)."""
    out = summarize(names, quiet=True)
    need = []
    for name, b in out['variants'].items():
        m = min(r['m_seeds'] for r in b['pairs'])
        if m >= 3:
            continue
        flag = any(abs(b['all'][g]['mean_d']) < NEEDMORE_Z * b['all'][g]['se_comb']
                   for g in ('raw', 'rend'))
        if flag:
            need.append(name)
    print(','.join(need))
    return need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'build-harm', 'run', 'summary',
                                    'needmore'])
    ap.add_argument('--spec')
    ap.add_argument('--pairs', default=None, help='comma list of Y')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--names', default=None)
    a = ap.parse_args()
    names = a.names.split(',') if a.names else None
    if a.cmd == 'build':
        build()
    elif a.cmd == 'build-harm':
        build_harm()
    elif a.cmd == 'run':
        pairs = PAIRS
        if a.pairs:
            ys = {int(x) for x in a.pairs.split(',')}
            pairs = [p for p in PAIRS if p[0] in ys]
        seeds = [int(s) for s in a.seeds.split(',')]
        run(a.spec, pairs, seeds, names)
    elif a.cmd == 'summary':
        summarize(names)
    else:
        needmore(names)


if __name__ == '__main__':
    main()
