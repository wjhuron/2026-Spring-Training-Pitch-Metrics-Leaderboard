"""pitching_plus_stage2.py — Pitching+ stage 2 (2026-07-07).

Task 1 (--task weights): find the EXACT best Stuff+/Loc+ blend weight.
  Fine grid (0.50-0.95 step 0.01) at the pitcher level, with:
    - bootstrap over pitcher units (2000 resamples) -> CI on argmax w
    - multi-cut stability (early/late cut at 4/15, 5/1, 5/15, 6/1)
    - role-specific optima (SP/RP) with bootstrap CIs
    - the "within 1 SE of max" indistinguishability band

Task 2 (--task joint): is an FG-style JOINT model worth it?
  Train the production XGBoost (TUNED + mono velo, season-blocked 2025-in-
  every-fold, pitcher-grouped 8-fold OOF) on stuff features + location
  (+ count where available) and race it against the composite on IDENTICAL
  pitcher units. Variants:
    joint_loc        : BASE_FEATS + platoon + px_h, pz
    joint_loc_count  : + balls, strikes (2025 rows NaN until the full 2025
                       statcast re-pull lands; XGBoost handles missing)
    joint_loc_count_zn (--full-2025): + zone-normalized z, with 2025
                       count/sz backfilled from _statcast2025_full_cache.pkl
  Comparators evaluated on the same units: stuff-only OOF, composite at the
  task-1 weight, Loc+ alone.

Shared per-pitch scored frame is cached to $STUFF_RETUNE_CACHE/pp_scored*.pkl.

Usage:
  STUFF_RETUNE_CACHE=<dir> python3 scripts/pitching_plus_stage2.py --task weights
  STUFF_RETUNE_CACHE=<dir> python3 scripts/pitching_plus_stage2.py --task joint [--full-2025]
"""
import os, sys, json, pickle, time, argparse, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import stuff_plus_v11.train_stuff_v11 as T
import pipeline_locplus as L

PKL26 = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
PKL25 = os.path.join(ROOT, 'data', '_pitches2025_training.pkl')
SC25_FULL = os.path.join(ROOT, 'data', '_statcast2025_full_cache.pkl')
CACHE_DIR = os.environ.get('STUFF_RETUNE_CACHE', '/tmp')
SCORED26 = os.path.join(CACHE_DIR, 'pp_scored26.pkl')
FRAME25 = os.path.join(CACHE_DIR, 'pp_frame25.pkl')
MIN_PERIOD = 200          # pitcher-level pitches per period
BOOT_N = 2000
SEED = 20260707


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def build_mask(pitches):
    mask = np.zeros(len(pitches), bool)
    for i, p in enumerate(pitches):
        pt, thr, bats = p.get('Pitch Type'), p.get('Throws'), p.get('Bats')
        if pt not in T.SUPPORTED or thr not in ('L', 'R') or bats not in ('L', 'R'):
            continue
        v = T.sf(p.get('Velocity')); iv = T.sf(p.get('xIndVrtBrk'))
        hb = T.sf(p.get('xHorzBrk')); vaa = T.sf(p.get('VAA'))
        ext = T.sf(p.get('Extension')); rz = T.sf(p.get('RelPosZ'))
        rx = T.sf(p.get('RelPosX'))
        if None in (v, iv, hb, vaa, ext, rz, rx):
            continue
        mask[i] = True
    return mask


def _loc_extras(pitches, mask):
    """px (hand-normalized), pz, zn, balls, strikes arrays for masked pitches."""
    px_h, pz, zn, balls, strikes = [], [], [], [], []
    for p, m in zip(pitches, mask):
        if not m:
            continue
        s = 1.0 if p.get('Throws') == 'R' else -1.0
        x = T.sf(p.get('PlateX')); z = T.sf(p.get('PlateZ'))
        px_h.append(x * s if x is not None else np.nan)
        pz.append(z if z is not None else np.nan)
        z_n = L._znorm(p)
        zn.append(z_n if z_n is not None else np.nan)
        c = L.get_count(p)
        balls.append(c[0] if c else np.nan)
        strikes.append(c[1] if c else np.nan)
    return (np.array(px_h), np.array(pz), np.array(zn),
            np.array(balls, float), np.array(strikes, float))


def build_scored26():
    """2026 frame: features + loc_raw + stuff_raw OOF + location extras."""
    if os.path.exists(SCORED26):
        return pd.read_pickle(SCORED26)
    t0 = time.time()
    print('building scored 2026 frame ...', flush=True)
    D = pickle.load(open(PKL26, 'rb'))
    p26 = [p for p in D if p.get('_source', 'MLB') == 'MLB']
    del D
    guts = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json'))).get(
        'gutsConstants') or {}
    S = L.build_surfaces([p for p in p26 if L.is_eligible_baseline(p)],
                         guts.get('lgWOBA'), guts.get('wOBAScale'))
    locval = np.full(len(p26), np.nan)
    for i, p in enumerate(p26):
        if L._is_scorable(p):
            v = L.score_pitch(p, S)
            if v is not None:
                locval[i] = -v
    rv_actual = np.array([(-r if (r := T.sf(p.get('RunExp'))) is not None
                           else np.nan) for p in p26])
    df = T.build_df(p26)
    mask = build_mask(p26)
    assert mask.sum() == len(df)
    df['loc_raw'] = locval[mask]
    df['rv_actual'] = rv_actual[mask]
    px_h, pz, zn, balls, strikes = _loc_extras(p26, mask)
    df['px_h'], df['pz'], df['zn'] = px_h, pz, zn
    df['balls'], df['strikes'] = balls, strikes
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    date_order = {d: i for i, d in enumerate(sorted(df['date'].dropna().unique()))}
    df['half'] = df['date'].map(date_order).fillna(0).astype(int) % 2

    # production OOF stuff_raw (current 12-feature set, 2025 prior in-fold)
    df25 = pd.read_pickle(os.path.join(CACHE_DIR, 'stuff_retune_df25.pkl'))
    X26 = T.design(df); X25 = T.design(df25).reindex(columns=X26.columns,
                                                     fill_value=0)
    y26, y25 = df['target_xrv'].values, df25['target_xrv'].values
    params = T._params_for(X26)
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=8).split(X26, y26, df['pitcher'].values):
        m = xgb.XGBRegressor(**params)
        m.fit(pd.concat([X26.iloc[tr], X25], ignore_index=True),
              np.concatenate([y26[tr], y25]))
        oof[te] = m.predict(X26.iloc[te])
    df['stuff_raw'] = -oof
    df.to_pickle(SCORED26)
    print(f'  scored 2026 frame cached ({time.time()-t0:.0f}s)', flush=True)
    return df


def build_frame25(full_2025=False):
    """2025 frame with location extras (count/zn only when full cache joined)."""
    key = FRAME25.replace('.pkl', '_full.pkl') if full_2025 else FRAME25
    if os.path.exists(key):
        return pd.read_pickle(key)
    print('building 2025 frame ...', flush=True)
    D26 = pickle.load(open(PKL26, 'rb'))
    p26 = [p for p in D26 if p.get('_source', 'MLB') == 'MLB']
    del D26
    p25 = pickle.load(open(PKL25, 'rb'))
    T._harmonize_tags(p25, p26)
    del p26
    if full_2025:
        _join_count_sz(p25)
    _lg, _sc = T.LG_WOBA, T.WOBA_SCALE
    T.LG_WOBA, T.WOBA_SCALE = T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE
    df = T.build_df(p25)
    T.LG_WOBA, T.WOBA_SCALE = _lg, _sc
    mask = build_mask(p25)
    assert mask.sum() == len(df)
    px_h, pz, zn, balls, strikes = _loc_extras(p25, mask)
    df['px_h'], df['pz'], df['zn'] = px_h, pz, zn
    df['balls'], df['strikes'] = balls, strikes
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    df.to_pickle(key)
    print(f'  2025 frame cached (count coverage '
          f'{df.balls.notna().mean()*100:.0f}%)', flush=True)
    return df


def _join_count_sz(p25):
    """Fingerprint-join balls/strikes/sz_top/sz_bot from the full 2025 cache
    onto the training pitches (same method as build_2025_training_set.join)."""
    sc = pickle.load(open(SC25_FULL, 'rb'))
    need = ['game_date', 'player_name', 'release_speed', 'plate_x', 'plate_z',
            'balls', 'strikes', 'sz_top', 'sz_bot']
    sc = sc[[c for c in need if c in sc.columns]]
    pub = defaultdict(list)
    for row in sc.itertuples(index=False):
        pub[(row.game_date, row.player_name)].append(row)
    grouped = defaultdict(list)
    for p in p25:
        grouped[(p.get('Game Date'), p.get('Pitcher'))].append(p)
    matched = unmatched = 0
    for key, mine in grouped.items():
        cands = pub.get(key, [])
        used = [False] * len(cands)
        for p in mine:
            v, px, pz = (T.sf(p.get('Velocity')), T.sf(p.get('PlateX')),
                         T.sf(p.get('PlateZ')))
            if v is None:
                unmatched += 1
                continue
            best_i, best_d = None, 1e9
            for i, c in enumerate(cands):
                if used[i]:
                    continue
                cv = T.sf(c.release_speed)
                if cv is None or abs(cv - v) > 0.25:
                    continue
                d = abs(cv - v) * 2.0
                cx, cz = T.sf(c.plate_x), T.sf(c.plate_z)
                if px is not None and cx is not None:
                    d += abs(cx - px)
                if pz is not None and cz is not None:
                    d += abs(cz - pz)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is None or best_d > 0.5:
                unmatched += 1
                continue
            used[best_i] = True
            c = cands[best_i]
            matched += 1
            b, s = T.sf(c.balls), T.sf(c.strikes)
            p['Count'] = (f'{int(b)}-{int(s)}'
                          if b is not None and s is not None else None)
            p['SzTop'], p['SzBot'] = T.sf(c.sz_top), T.sf(c.sz_bot)
    print(f'  2025 count/sz join: {matched} matched, {unmatched} unmatched '
          f'({matched/max(matched+unmatched,1):.1%})', flush=True)


# ── task 1: exact weights ─────────────────────────────────────────────────
def unit_table(df, cut, min_period, role_filter=None):
    d = df[df['loc_raw'].notna()].copy()
    d['period'] = np.where(d['date'] < cut, 'early', 'late')
    if role_filter is not None:
        ppg = d.groupby(['pitcher', 'throws', 'date']).size().groupby(
            ['pitcher', 'throws']).mean()
        keep = ppg[(ppg >= 40) == role_filter].index
        d = d.set_index(['pitcher', 'throws']).loc[
            d.set_index(['pitcher', 'throws']).index.isin(keep)].reset_index()
    g = d.groupby(['pitcher', 'throws', 'period'])
    agg = g.agg(stuff=('stuff_raw', 'mean'), loc=('loc_raw', 'mean'),
                xrv=('target_xrv', 'mean'), n=('stuff_raw', 'size')).reset_index()
    w = agg.pivot_table(index=['pitcher', 'throws'], columns='period',
                        values=['stuff', 'loc', 'xrv', 'n'], aggfunc='first')
    ok = ((w[('n', 'early')].fillna(0) >= min_period) &
          (w[('n', 'late')].fillna(0) >= min_period))
    w = w[ok]
    zs = ((w[('stuff', 'early')] - w[('stuff', 'early')].mean())
          / w[('stuff', 'early')].std()).values
    zl = ((w[('loc', 'early')] - w[('loc', 'early')].mean())
          / w[('loc', 'early')].std()).values
    y = -w[('xrv', 'late')].values          # higher = better run prevention
    return zs, zl, y


def grid_curve(zs, zl, y, grid):
    return np.array([pearson(w * zs + (1 - w) * zl, y) for w in grid])


def task_weights(df):
    grid = np.round(np.arange(0.30, 0.981, 0.01), 2)
    rng = np.random.RandomState(SEED)
    print('\n== exact-weight analysis (pitcher level) ==', flush=True)
    rows = []
    for cut, min_p, label in [('2026-04-15', 150, 'cut 4/15'),
                              ('2026-05-01', 200, 'cut 5/01'),
                              ('2026-05-15', 200, 'cut 5/15'),
                              ('2026-06-01', 200, 'cut 6/01')]:
        zs, zl, y = unit_table(df, cut, min_p)
        curve = grid_curve(zs, zl, y, grid)
        best = grid[np.nanargmax(curve)]
        rows.append((label, len(y), best, np.nanmax(curve)))
        print(f'  {label}: n={len(y)}, argmax w={best:.2f} '
              f'(r={np.nanmax(curve):.3f})', flush=True)

    # main cut: bootstrap CI + 1-SE band
    zs, zl, y = unit_table(df, '2026-05-01', 200)
    curve = grid_curve(zs, zl, y, grid)
    n = len(y)
    boots = np.empty(BOOT_N)
    curves = np.empty((BOOT_N, len(grid)))
    for b in range(BOOT_N):
        idx = rng.randint(0, n, n)
        c = grid_curve(zs[idx], zl[idx], y[idx], grid)
        curves[b] = c
        boots[b] = grid[np.nanargmax(c)]
    lo, med, hi = np.percentile(boots, [2.5, 50, 97.5])
    best = grid[np.nanargmax(curve)]
    se_at_best = curves[:, np.nanargmax(curve)].std()
    band = grid[curve >= np.nanmax(curve) - se_at_best]
    print(f'\n  MAIN (cut 5/01, n={n}): argmax w = {best:.2f} '
          f'(r={np.nanmax(curve):.3f})')
    print(f'  bootstrap argmax: median {med:.2f}, 95% CI [{lo:.2f}, {hi:.2f}]')
    print(f'  within-1-SE band: w in [{band.min():.2f}, {band.max():.2f}]')
    for w in (0.60, 0.65, 0.70, 0.75, 0.80, best):
        i = np.where(grid == round(w, 2))[0]
        if len(i):
            print(f'    w={w:.2f}: pred r = {curve[i[0]]:.4f}')

    for role, lbl in ((True, 'starters'), (False, 'relievers')):
        zs_r, zl_r, y_r = unit_table(df, '2026-05-01', 200, role_filter=role)
        c_r = grid_curve(zs_r, zl_r, y_r, grid)
        b_r = np.empty(BOOT_N)
        nr = len(y_r)
        for b in range(BOOT_N):
            idx = rng.randint(0, nr, nr)
            b_r[b] = grid[np.nanargmax(grid_curve(zs_r[idx], zl_r[idx],
                                                  y_r[idx], grid))]
        lo_r, med_r, hi_r = np.percentile(b_r, [2.5, 50, 97.5])
        print(f'  {lbl}: n={nr}, argmax {grid[np.nanargmax(c_r)]:.2f}, '
              f'bootstrap median {med_r:.2f}, 95% CI [{lo_r:.2f}, {hi_r:.2f}]',
              flush=True)


# ── task 2: joint model race ──────────────────────────────────────────────
def oof_joint(df26, df25, extra_feats):
    feats = list(T.BASE_FEATS) + ['platoon_same'] + extra_feats
    X26 = df26[feats].reset_index(drop=True)
    X25 = df25[feats].reset_index(drop=True)
    y26, y25 = df26['target_xrv'].values, df25['target_xrv'].values
    params = dict(T.TUNED)
    params['monotone_constraints'] = tuple(
        -1 if c == T.MONO_FEAT else 0 for c in X26.columns)
    oof = np.full(len(df26), np.nan)
    for tr, te in GroupKFold(n_splits=8).split(X26, y26, df26['pitcher'].values):
        m = xgb.XGBRegressor(**params)
        m.fit(pd.concat([X26.iloc[tr], X25], ignore_index=True),
              np.concatenate([y26[tr], y25]))
        oof[te] = m.predict(X26.iloc[te])
    return -oof


def eval_scores(df, cols, w_star):
    """pred/reliab/desc for each score column + composite, identical units."""
    d = df[df['loc_raw'].notna()].copy()
    d['period'] = np.where(d['date'] < '2026-05-01', 'early', 'late')
    zsc = {}
    g = d.groupby(['pitcher', 'throws', 'period'])
    agg = g.agg(**{c: (c, 'mean') for c in cols},
                xrv=('target_xrv', 'mean'), n=('stuff_raw', 'size')).reset_index()
    wide = agg.pivot_table(index=['pitcher', 'throws'], columns='period',
                           values=cols + ['xrv', 'n'], aggfunc='first')
    ok = ((wide[('n', 'early')].fillna(0) >= MIN_PERIOD) &
          (wide[('n', 'late')].fillna(0) >= MIN_PERIOD))
    wide = wide[ok]
    y = -wide[('xrv', 'late')].values
    gh = d.groupby(['pitcher', 'throws', 'half'])
    ha = gh.agg(**{c: (c, 'mean') for c in cols},
                n=('stuff_raw', 'size')).reset_index()
    hw = ha.pivot_table(index=['pitcher', 'throws'], columns='half',
                        values=cols + ['n'], aggfunc='first')
    hok = ((hw[('n', 0)].fillna(0) >= MIN_PERIOD // 2) &
           (hw[('n', 1)].fillna(0) >= MIN_PERIOD // 2))
    hw = hw[hok]

    def z(v):
        v = np.asarray(v, float)
        return (v - np.nanmean(v)) / np.nanstd(v)

    print(f"\n  {'score':22s} {'pred':>7s} {'reliab':>7s}  (n_pred={len(y)}, "
          f'n_rel={len(hw)})')
    res = {}
    for c in cols:
        pr = pearson(z(wide[(c, 'early')].values), y)
        rel = pearson(hw[(c, 0)].values, hw[(c, 1)].values)
        res[c] = (pr, rel)
        print(f'  {c:22s} {pr:7.3f} {rel:7.3f}', flush=True)
    # composite on same units
    ce = (w_star * z(wide[('stuff_raw', 'early')].values)
          + (1 - w_star) * z(wide[('loc_raw', 'early')].values))
    pr = pearson(ce, y)
    c0 = (w_star * z(hw[('stuff_raw', 0)].values)
          + (1 - w_star) * z(hw[('loc_raw', 0)].values))
    c1 = (w_star * z(hw[('stuff_raw', 1)].values)
          + (1 - w_star) * z(hw[('loc_raw', 1)].values))
    rel = pearson(c0, c1)
    print(f'  {"composite w=%.2f" % w_star:22s} {pr:7.3f} {rel:7.3f}', flush=True)
    return res


def task_joint(df26, full_2025, w_star=0.70):
    df25 = build_frame25(full_2025=full_2025)
    print(f'\n== joint-model race (2025 rows: {len(df25)}, count coverage '
          f'{df25.balls.notna().mean()*100:.0f}%) ==', flush=True)
    t0 = time.time()
    df26['joint_loc'] = oof_joint(df26, df25, ['px_h', 'pz'])
    print(f'  joint_loc OOF done ({(time.time()-t0)/60:.1f} min)', flush=True)
    df26['joint_loc_count'] = oof_joint(df26, df25,
                                        ['px_h', 'pz', 'balls', 'strikes'])
    print(f'  joint_loc_count OOF done', flush=True)
    cols = ['stuff_raw', 'loc_raw', 'joint_loc', 'joint_loc_count']
    if full_2025:
        df26['joint_full'] = oof_joint(df26, df25,
                                       ['px_h', 'pz', 'zn', 'balls', 'strikes'])
        print(f'  joint_full OOF done', flush=True)
        cols.append('joint_full')
    eval_scores(df26, cols, w_star)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['weights', 'joint'])
    ap.add_argument('--full-2025', action='store_true')
    args = ap.parse_args()
    df26 = build_scored26()
    print(f'2026 scored frame: {len(df26)} rows', flush=True)
    if args.task == 'weights':
        task_weights(df26)
    else:
        task_joint(df26, args.full_2025)


if __name__ == '__main__':
    main()
