#!/usr/bin/env python3
"""stuff_gate_v2.py — Stuff+ replicate gate, version 2 (2026-08-23).

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

SHIPPED tracks T.BASE_FEATS / T.TUNED at run time. After a config change,
delete data/_gate_v2/agg_SHIPPED_* and the SHIPPED block of results.json,
or the cached baseline is the previous version.

Per-pitcher aggregates are cached per (variant, pair, seed), so the summary
computes a PAIRED pitcher-bootstrap SE on every delta vs SHIPPED without a
refit, and new variants never refit SHIPPED.

VAA is handled ONCE, post hoc from vaa_raw, with slopes fit on the training
seasons of each pair. Variant key 'vaa': 'raw' skips the adjustment.

Spec (JSON, --spec):
  {
    "ARMDEV":   {"add": ["arm_dev", "arm_dev_abs"]},
    "ACCEL":    {"replace": {"ivb": "acc_v", "hb": "acc_h",
                             "ivb_diff": "acc_v_diff", "hb_diff": "acc_h_diff"}},
    "RELSD":    {"add": ["rel_sd_x", "rel_sd_z"]},
    "RAW_VAA":  {"vaa": "raw"},
    "ADD_RELZ": {"add": ["rel_z"]},
    "DEPTH6":   {"params": {"max_depth": 6}},
    "ANCHOR_ANY": {"anchor": "any"}      # most-thrown of FF/SI/FC
  }
  keys: add, drop, replace, mask {feat: [types]}, add_typed {feat: [types]},
        params, vaa, anchor ('true' default | 'any')

Usage:
  python3 scripts/research/stuff/stuff_gate_v2.py build
  python3 scripts/research/stuff/stuff_gate_v2.py run --spec spec.json [--pairs 2021,2024] [--seeds 0,1]
  python3 scripts/research/stuff/stuff_gate_v2.py summary [--names A,B]
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

KEEP = ['pitcher', 'team', 'throws', 'date', 'pitch_type', 'platoon_same',
        'target_xrv', 'rv_raw', 'velocity', 'ivb', 'hb', 'spin_rate',
        'extension', 'arm_angle', 'rel_x', 'rel_z', 'cross', 'cross_abs',
        'kin_eff', 'kin_eff__ff', 'vaa_raw', 'plate_x', 'plate_z', 'ax_sin',
        'ax_cos', 'spin_eff', 'axis_dev', 'axis_dev_abs', 'haa_meas',
        'kin_dev', 'kin_cd']


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
    print(f'  2026: {len(p26)} MLB pitches, sidecar filled {n}')
    return p26


def build():
    os.makedirs(CACHE, exist_ok=True)
    saved = T.NVAA_SLOPES
    T.NVAA_SLOPES = {}            # raw VAA in the cache; adjusted per pair
    try:
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
            lg, sc = T.LG_WOBA, T.WOBA_SCALE
            T.LG_WOBA, T.WOBA_SCALE = guts
            d = T.build_df(pk)
            T.LG_WOBA, T.WOBA_SCALE = lg, sc
            del pk
            gc.collect()
            d = d[d['target_xrv'].notna()].reset_index(drop=True)
            d = d[[c for c in KEEP if c in d.columns]].copy()
            for c in d.columns:
                if c not in ('pitcher', 'team', 'throws', 'date', 'pitch_type'):
                    d[c] = pd.to_numeric(d[c], errors='coerce')
            d['season'] = y
            d.to_pickle(season_path(y))
            print(f'  {y}: {len(d)} rows, {time.time()-t0:.0f}s', flush=True)
            del d
            gc.collect()
    finally:
        T.NVAA_SLOPES = saved


# ── per-pair preparation: VAA, anchors, derived candidates ───────────────
def fit_vaa_slopes(dfs):
    pool = pd.concat([d[['pitch_type', 'vaa_raw', 'plate_z']] for d in dfs],
                     ignore_index=True).dropna()
    out = {}
    for pt, sub in pool.groupby('pitch_type'):
        if len(sub) >= SLOPE_MIN:
            slope = float(np.cov(sub.plate_z, sub.vaa_raw)[0, 1]
                          / np.var(sub.plate_z))
            out[pt] = (slope, float(sub.plate_z.mean()))
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
    d['vaa'] = d['vaa_raw'] - adj
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
    d['height'] = d['pitcher'].map(height_map())
    d['rel_z_rel'] = d['rel_z'] * 12.0 - d['height']    # release height vs stature
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
    return d.drop(columns=['_prim', 'r_v', 'r_iv', 'r_hb', 'r_vaa',
                           'r_av', 'r_ah'])


ARM_SIDE_SIGN = 1.0
_HEIGHT = None


def height_map():
    """pitcher name -> height (inches), from data/_gate_v2/pitcher_height.json
    (Statcast ids 2021-25 + mlb_id_cache, heights from the MLB Stats API,
    pulled 2026-08-23). Ambiguous names (6) take the mean."""
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


def apply_variant(d, spec):
    d = d.copy()
    feats = list(BASE)
    for old, new in (spec.get('replace') or {}).items():
        feats[feats.index(old)] = new
    for f in (spec.get('drop') or []):
        feats.remove(f)
    for f in (spec.get('add') or []):
        if f not in feats:
            feats.append(f)
    for f, types in (spec.get('mask') or {}).items():
        if f == 'velo_diff':
            d['velo_diff'] = d['velo_diff_raw']
        d.loc[d['pitch_type'].isin(set(types)), f] = np.nan
    for f, types in (spec.get('add_typed') or {}).items():
        col = f'{f}__{"".join(types)}'
        d[col] = np.where(d['pitch_type'].isin(set(types)), d[f], np.nan)
        feats.append(col)
    return d, feats


def design(d, feats):
    X = d[feats].reset_index(drop=True).astype('float32')
    X['platoon_same'] = d['platoon_same'].reset_index(drop=True).values
    return X


# ── metrics ──────────────────────────────────────────────────────────────
def half_index(d):
    order = {x: i for i, x in enumerate(sorted(d['date'].dropna().unique()))}
    return (d['date'].map(order).fillna(0).astype(int) % 2).values


def aggregates(dY, dY1):
    """Per-pitcher and per-unit aggregates for Y and Y+1 (cached)."""
    half = half_index(dY)
    dY = dY.assign(_h=half, _neg=-dY['target_xrv'])
    dY1 = dY1.assign(_neg=-dY1['target_xrv'], _rv=-dY1['rv_raw'])  # pitcher-positive
    # within-Y halves
    gp = dY.groupby(['pitcher', '_h']).agg(s=('stuff', 'mean'),
                                            t=('_neg', 'mean'),
                                            n=('stuff', 'size')).unstack('_h')
    gu = dY.groupby(['pitcher', 'pitch_type', '_h']).agg(
        s=('stuff', 'mean'), t=('_neg', 'mean'),
        n=('stuff', 'size')).unstack('_h')
    # season Y -> Y+1
    py = dY.groupby('pitcher').agg(s=('stuff', 'mean'), n=('stuff', 'size'))
    py1 = dY1.groupby('pitcher').agg(t=('_neg', 'mean'), rv=('_rv', 'mean'),
                                     n=('_neg', 'size'))
    uy = dY.groupby(['pitcher', 'pitch_type']).agg(s=('stuff', 'mean'),
                                                   n=('stuff', 'size'))
    uy1 = dY1.groupby(['pitcher', 'pitch_type']).agg(t=('_neg', 'mean'),
                                                     n=('_neg', 'size'))
    return dict(pitch=(dY['stuff'].values.astype('float32'),
                       dY['_neg'].values.astype('float32')),
                gp=gp, gu=gu, py=py, py1=py1, uy=uy, uy1=uy1)


def metrics_from(agg):
    s, t = agg['pitch']
    out = dict(pitch_r=pear(s, t))
    gp, gu = agg['gp'], agg['gu']
    m = (gp[('n', 0)] >= MIN_HALF_P) & (gp[('n', 1)] >= MIN_HALF_P)
    out['fut_r'] = pear(gp.loc[m, ('s', 0)], gp.loc[m, ('t', 1)])
    out['n_fut'] = int(m.sum())
    m = (gu[('n', 0)] >= MIN_HALF_U) & (gu[('n', 1)] >= MIN_HALF_U)
    out['unit_r'] = pear(gu.loc[m, ('s', 0)], gu.loc[m, ('t', 1)])
    out['rel'] = pear(gu.loc[m, ('s', 0)], gu.loc[m, ('s', 1)])
    j = agg['py'].join(agg['py1'], lsuffix='_y', rsuffix='_y1', how='inner')
    j = j[(j['n_y'] >= MIN_NXT) & (j['n_y1'] >= MIN_NXT)]
    out['nxt_r'] = pear(j['s'], j['t'])
    out['nxt_rv_r'] = pear(j['s'], j['rv'])
    out['n_nxt'] = int(len(j))
    ju = agg['uy'].join(agg['uy1'], lsuffix='_y', rsuffix='_y1', how='inner')
    ju = ju[(ju['n_y'] >= MIN_NXT_U) & (ju['n_y1'] >= MIN_NXT_U)]
    out['nxt_unit_r'] = pear(ju['s'], ju['t'])
    out['n_nxt_unit'] = int(len(ju))
    return out


def nxt_table(agg):
    j = agg['py'].join(agg['py1'], lsuffix='_y', rsuffix='_y1', how='inner')
    j = j[(j['n_y'] >= MIN_NXT) & (j['n_y1'] >= MIN_NXT)]
    return j[['s', 't', 'rv']]


# ── run ──────────────────────────────────────────────────────────────────
def agg_path(name, Y, seed):
    return os.path.join(CACHE, f'agg_{name}_{Y}_s{seed}.pkl')


def run(spec_path, pairs, seeds, names_only=None):
    with open(spec_path) as f:
        variants = json.load(f)
    if names_only:
        variants = {k: v for k, v in variants.items() if k in names_only}
    results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    frames = {y: pd.read_pickle(season_path(y)) for y in SEASONS}
    set_arm_side_sign(list(frames.values()))
    for Y, Y1 in pairs:
        train_years = [y for y in SEASONS if y not in (Y, Y1)]
        todo = [(n, s) for n in ['SHIPPED'] + list(variants)
                for s in seeds if not os.path.exists(agg_path(n, Y, s))]
        if not todo:
            print(f'=== pair {Y}->{Y1}: all cached')
            continue
        print(f'\n=== pair {Y}->{Y1} (train {train_years}) ===', flush=True)
        slopes = fit_vaa_slopes([frames[y] for y in train_years])
        prep = {}
        for name, seed in todo:
            spec = variants.get(name, {})
            vkey = (spec.get('vaa', 'nvaa'), spec.get('anchor', 'true'))
            if vkey not in prep:
                sl = slopes if vkey[0] == 'nvaa' else {}
                prep[vkey] = {y: prepare(frames[y], sl, vkey[1])
                              for y in train_years + [Y, Y1]}
            P = prep[vkey]
            t0 = time.time()
            tr = [apply_variant(P[y], spec) for y in train_years]
            feats = tr[0][1]
            Xtr = pd.concat([design(d, feats) for d, _ in tr],
                            ignore_index=True)
            ytr = np.concatenate([d['target_xrv'].values for d, _ in tr])
            del tr
            dY, _ = apply_variant(P[Y], spec)
            XY = design(dY, feats)
            params = T._params_for(Xtr)
            params.update(spec.get('params') or {})
            params['monotone_constraints'] = tuple(
                -1 if c == T.MONO_FEAT else 0 for c in Xtr.columns)
            params['random_state'] = seed
            m = xgb.XGBRegressor(**params)
            m.fit(Xtr, ytr)
            dY = dY.assign(stuff=-m.predict(XY))
            agg = aggregates(dY, P[Y1])
            met = metrics_from(agg)
            met['n_train'] = int(len(Xtr))
            met['feats'] = feats
            pd.to_pickle(agg, agg_path(name, Y, seed))
            results.setdefault(name, {}).setdefault(str(Y), {})[str(seed)] = met
            with open(RESULTS, 'w') as f:
                json.dump(results, f, indent=1)
            print(f'  {name:<14} s{seed} pitch {met["pitch_r"]:.4f} '
                  f'fut {met["fut_r"]:.4f} unit {met["unit_r"]:.4f} '
                  f'rel {met["rel"]:.4f} | nxt {met["nxt_r"]:.4f} '
                  f'(n={met["n_nxt"]}) nxt_u {met["nxt_unit_r"]:.4f} '
                  f'nxt_rv {met["nxt_rv_r"]:.4f} [{time.time()-t0:.0f}s]',
                  flush=True)
            del Xtr, XY, m, dY, agg
            gc.collect()
        del prep
        gc.collect()


# ── summary with paired pitcher bootstrap ────────────────────────────────
def boot_delta(a, b, B=2000, seed=0):
    """a, b: nxt tables (index pitcher, cols s,t). Paired bootstrap over the
    common pitchers of r(a.s, t) - r(b.s, t)."""
    j = a.join(b, lsuffix='_a', rsuffix='_b', how='inner')
    n = len(j)
    sa, sb, t = j['s_a'].values, j['s_b'].values, j['t_a'].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    def r(x, y):
        x = x - x.mean(1, keepdims=True); y = y - y.mean(1, keepdims=True)
        return (x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))
    d = r(sa[idx], t[idx]) - r(sb[idx], t[idx])
    return float(pear(sa, t) - pear(sb, t)), float(d.std()), n


def summary(names=None):
    results = json.load(open(RESULTS))
    variants = [n for n in results if n != 'SHIPPED']
    if names:
        variants = [n for n in variants if n in names]
    print('\nSHIPPED per pair (seed-mean):')
    for Y in results['SHIPPED']:
        ms = list(results['SHIPPED'][Y].values())
        print(f'  {Y}: ' + '  '.join(
            f'{k} {np.mean([m[k] for m in ms]):.4f}'
            for k in ('pitch_r', 'fut_r', 'unit_r', 'rel', 'nxt_r',
                      'nxt_unit_r', 'nxt_rv_r')) + f'  n_nxt {ms[0]["n_nxt"]}')
    for name in variants:
        print(f'\n{name}:')
        rows = []
        for Y in sorted(results[name]):
            if Y not in results['SHIPPED']:
                continue
            seeds = sorted(set(results[name][Y]) & set(results['SHIPPED'][Y]))
            if not seeds:
                continue
            dm = {k: np.mean([results[name][Y][s][k] - results['SHIPPED'][Y][s][k]
                              for s in seeds])
                  for k in ('pitch_r', 'fut_r', 'unit_r', 'rel', 'nxt_r',
                            'nxt_unit_r', 'nxt_rv_r')}
            # paired bootstrap SE on nxt_r, first common seed
            s0 = seeds[0]
            a = nxt_table(pd.read_pickle(agg_path(name, int(Y), int(s0))))
            b = nxt_table(pd.read_pickle(agg_path('SHIPPED', int(Y), int(s0))))
            dboot, se, n = boot_delta(a, b)
            rows.append((Y, dm, se, n, len(seeds)))
            print(f'  {Y} d_nxt {dm["nxt_r"]:+.4f} (se {se:.4f}, n {n}, '
                  f'seeds {len(seeds)})  d_nxt_u {dm["nxt_unit_r"]:+.4f}  '
                  f'd_nxt_rv {dm["nxt_rv_r"]:+.4f}  d_fut {dm["fut_r"]:+.4f}  '
                  f'd_unit {dm["unit_r"]:+.4f}  d_pitch {dm["pitch_r"]:+.4f}  '
                  f'd_rel {dm["rel"]:+.4f}')
        if rows:
            k = len(rows)
            mean_d = np.mean([r[1]['nxt_r'] for r in rows])
            se_pool = np.sqrt(np.sum([r[2] ** 2 for r in rows])) / k
            wins = sum(r[1]['nxt_r'] > 0 for r in rows)
            wins_u = sum(r[1]['nxt_unit_r'] > 0 for r in rows)
            wins_f = sum(r[1]['fut_r'] > 0 for r in rows)
            print(f'  MEAN d_nxt {mean_d:+.4f} +/- {se_pool:.4f}  '
                  f'(z {mean_d/se_pool if se_pool else float("nan"):+.2f})  '
                  f'WINS nxt {wins}/{k}  nxt_u {wins_u}/{k}  fut {wins_f}/{k}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'run', 'summary'])
    ap.add_argument('--spec')
    ap.add_argument('--pairs', default=None, help='comma list of Y')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--names', default=None)
    a = ap.parse_args()
    if a.cmd == 'build':
        build()
    elif a.cmd == 'run':
        pairs = PAIRS
        if a.pairs:
            ys = {int(x) for x in a.pairs.split(',')}
            pairs = [p for p in PAIRS if p[0] in ys]
        seeds = [int(s) for s in a.seeds.split(',')]
        names = a.names.split(',') if a.names else None
        run(a.spec, pairs, seeds, names)
    else:
        summary(a.names.split(',') if a.names else None)


if __name__ == '__main__':
    main()
