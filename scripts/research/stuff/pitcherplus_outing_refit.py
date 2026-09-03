#!/usr/bin/env python3
"""pitcherplus_outing_refit.py — redo the outing Pitching+ weight fit with
TRUE per-half stuff (2026-09-02).

The shipped PP_OUTING_W (cards/pitcher.py) came from
pitcherplus_outing_grade.py / _pick.py, where stuffRaw_o == stuffRaw_e ==
the full-game LOSO mean: stuff's half-reliability was 1.0 by construction,
its outing-grain k was never measured (the season 42 was used), the search
ran loc k 215 while the fit used 185, and the 1-SE rule's 2-term pick was
overridden by a hardcoded 4-term set. This script rebuilds the split panel
with per-pitch LOSO stuff (pitcherplus_stuff_loso_pitch.py) split by the
same odd/even PA index as pitcherplus_outing_tables.py, measures stuff's
outing-grain k with the same crossing as every other component, and refits.

Stages
  join   fingerprint-join the per-pitch LOSO stuff to the Savant cache rows
         (game_pk, pitch_type, velocity .1, plate_x .01, plate_z .01), take
         pid / pa_idx / pitch_ord from the cache, convert raw to integer
         atoms per season (pitcher x type units >= QUAL_N, ANCHOR_BORROW),
         aggregate per outing and per half. Rewrites
         data/_pplus_stuff_loso_pitch.pkl with the join columns and writes
         data/_pplus_outing_stuff_halves.pkl.
  fit    screen (half reliability, k), refit-4, exhaustive 1-SE search,
         shipped weights on the true-half panel, effective shares, stuff-k
         sweep, short-outing behaviour. Writes
         data/_pplus_outing_refit_2026_09.json and .md.

Usage:
  PYTHONHASHSEED=0 python3 scripts/research/stuff/pitcherplus_outing_refit.py join
  PYTHONHASHSEED=0 python3 scripts/research/stuff/pitcherplus_outing_refit.py fit
"""
import gc
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, 'data')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'misc'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import leaderboard_metric_battery as bat                     # noqa: E402
from stuff_plus.train_stuff import ANCHOR_BORROW, QUAL_N, K_SCALE  # noqa: E402
from pitcherplus_stuff_loso_pitch import (SEASONS, OUT_PKL as LOSO_PKL,  # noqa: E402
                                          fold_path)

TABLES = os.path.join(DATA, '_pplus_outing_tables.pkl')
STUFF_GAMES_OLD = os.path.join(DATA, '_pplus_stuff_loso_games.csv')
HALVES_PKL = os.path.join(DATA, '_pplus_outing_stuff_halves.pkl')
OUT_JSON = os.path.join(DATA, '_pplus_outing_refit_2026_09.json')
OUT_MD = os.path.join(DATA, '_pplus_outing_refit_2026_09.md')
SEARCH_CSV = os.path.join(DATA, '_pplus_outing_refit_search.csv')

MIN_N = 20            # outing floor for the analysis pool (as shipped)
MIN_HALF_N = 8        # per-half floor for the split panel (as shipped)
FEATS = ['stuffAtom', 'stuffRaw', 'locRaw', 'kPct', 'bbPct', 'kbbPct',
         'cswPct', 'whiffPct', 'izWhiffPct', 'chasePct', 'gbPct',
         'xrv100', 'rv100']
SHIPPED_W = {'stuffAtom': 0.205, 'locRaw': 0.169, 'cswPct': 0.252,
             'xrv100': 0.374}
SHIPPED_K = {'stuffAtom': 42.0, 'locRaw': 185.0, 'cswPct': 398.0,
             'xrv100': 1581.0}
FOUR = ['stuffAtom', 'locRaw', 'cswPct', 'xrv100']
LOC_K_FIXED = 185.0
N_BOOT = 300
SHORT_BINS = [4, 9, 14, 19, 29, 49, 79, 200]
SHORT_LABELS = ['5-9', '10-14', '15-19', '20-29', '30-49', '50-79', '80+']


# ══════════════════════════════════════════════════════════════════════════
# stage: join
# ══════════════════════════════════════════════════════════════════════════
def fingerprint(df, velo, px, pz):
    """Integer fingerprint columns. The 2025 pickle stores values rounded to
    (.1, .01, .01); the cache and the 2021-24 pickles are full precision, so
    rounding both sides the same way reproduces the pickle's stored value."""
    fp = pd.DataFrame(index=df.index)
    fp['game_pk'] = pd.to_numeric(df['game_pk'], errors='coerce').astype('Int64')
    fp['pitch_type'] = df['pitch_type'].astype(str)
    fp['v1'] = (pd.to_numeric(df[velo], errors='coerce') * 10).round().astype('Int64')
    fp['x2'] = (pd.to_numeric(df[px], errors='coerce') * 100).round().astype('Int64')
    fp['z2'] = (pd.to_numeric(df[pz], errors='coerce') * 100).round().astype('Int64')
    return fp


# pitch_type is NOT a key: the 2025 pickle is Wally's RETAGGED sheets, so
# its tags differ from Savant's on ~11% of pitches (measured 2026-09-02).
# Within one game, velocity .1 x plate .01 x .01 is unique to a handful of
# pairs per season, and those drop as duplicates on both sides.
FP_KEYS = ['game_pk', 'v1', 'x2', 'z2']


def norm_name(s):
    import unicodedata
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    s = s.lower().replace('.', '').replace(',', ' ')
    toks = [t for t in s.split() if t not in ('jr', 'sr', 'ii', 'iii', 'iv')]
    return ' '.join(toks)


def add_order(df):
    """pa_idx exactly as pitcherplus_outing_tables.add_pa_index, plus
    pitch_ord (chronological pitch index within pitcher x game_pk). The cache
    stores games in Savant's descending order, so chronological = reversed
    row order within game_pk."""
    df = df.copy()
    df['_row'] = np.arange(len(df))
    df = df.sort_values(['pitcher', 'game_pk', '_row'],
                        ascending=[True, True, False], kind='stable')
    grp = df.groupby(['pitcher', 'game_pk'], sort=False)
    df['pa_idx'] = grp['pa_end'].cumsum() - df['pa_end'].astype(int)
    df['pitch_ord'] = grp.cumcount()
    return df.sort_values('_row').drop(columns=['_row'])


def load_loso_season(year):
    """Per-pitch LOSO frame for one season: the assembled pickle if it has
    the season, else the fold file from the scratch dir."""
    if os.path.exists(LOSO_PKL):
        lo = pickle.load(open(LOSO_PKL, 'rb'))
        lo = lo[lo['season'] == year]
        if len(lo):
            return lo.reset_index(drop=True)
    fp = fold_path(year)
    if os.path.exists(fp):
        return pickle.load(open(fp, 'rb')).reset_index(drop=True)
    return None


def season_anchors(lo):
    """(mu, sd, nqual) per pitch type from pitcher x type units with
    >= QUAL_N pitches (train_stuff._standardize on (pitcher, pitch_type):
    the 2021-24 pickles carry no team). ANCHOR_BORROW as the trainer."""
    a = (lo.groupby(['pitcher', 'pitch_type'])['stuff_raw']
           .agg(rawmean='mean', n='size').reset_index())
    scale = {}
    for pt, sub in a.groupby('pitch_type'):
        q = sub[sub['n'] >= QUAL_N]
        base = q if len(q) >= 5 else sub
        if pt in ANCHOR_BORROW:
            donor = a[a['pitch_type'].isin(ANCHOR_BORROW[pt])]
            dq = donor[donor['n'] >= QUAL_N]
            base = dq if len(dq) >= 5 else donor
        mu, sd = float(base['rawmean'].mean()), float(base['rawmean'].std())
        scale[pt] = {'mu': mu, 'sd': sd, 'nqual': int(len(q)),
                     'nunits': int(len(sub))}
    return scale


def add_atoms(lo, scale):
    mu = lo['pitch_type'].map({k: v['mu'] for k, v in scale.items()})
    sd = lo['pitch_type'].map({k: v['sd'] for k, v in scale.items()})
    af = 100.0 + K_SCALE * (lo['stuff_raw'] - mu) / sd
    lo['atom_f'] = af
    lo['atom'] = np.rint(af).astype('float64')   # int(round(.)) per pitch
    return lo


def join_season(year, report):
    t0 = time.time()
    lo = load_loso_season(year)
    if lo is None:
        print(f'── {year}: no LOSO fold yet, skip', flush=True)
        return None, None
    scale = season_anchors(lo)
    lo = add_atoms(lo, scale)
    cache = bat.load_season(year)
    cache = cache[['pitcher', 'player_name', 'game_pk', 'game_date',
                   'pitch_type', 'release_speed', 'plate_x', 'plate_z',
                   'pa_end']].reset_index(drop=True)
    cache['pitcher'] = pd.to_numeric(cache['pitcher'], errors='coerce')
    cache = add_order(cache)
    cache['_crow'] = np.arange(len(cache))
    fc = fingerprint(cache, 'release_speed', 'plate_x', 'plate_z')
    fc['_crow'] = cache['_crow'].to_numpy()
    fl = fingerprint(lo, 'fp_velocity', 'fp_plate_x', 'fp_plate_z')
    fl['_lrow'] = np.arange(len(lo))
    ok_c = fc[FP_KEYS].notna().all(axis=1)
    ok_l = fl[FP_KEYS].notna().all(axis=1)
    fc, fl = fc[ok_c], fl[ok_l]
    dup_c = fc.duplicated(FP_KEYS, keep=False)
    dup_l = fl.duplicated(FP_KEYS, keep=False)
    m = fl[~dup_l].merge(fc[~dup_c], on=FP_KEYS, how='inner')
    lo['pid'] = np.nan
    lo['pa_idx'] = np.nan
    lo['pitch_ord'] = np.nan
    lo['cache_row'] = np.nan
    lr = m['_lrow'].to_numpy()
    cr = m['_crow'].to_numpy()
    lo.loc[lr, 'pid'] = cache['pitcher'].to_numpy()[cr]
    lo.loc[lr, 'pa_idx'] = cache['pa_idx'].to_numpy()[cr]
    lo.loc[lr, 'pitch_ord'] = cache['pitch_ord'].to_numpy()[cr]
    lo.loc[lr, 'cache_row'] = cr
    ln = lo.loc[lr, 'pitcher'].map(norm_name).to_numpy()
    cn = pd.Series(cache['player_name'].to_numpy()[cr]).map(norm_name).to_numpy()
    agree = ln == cn
    names_agree = float(agree.mean())
    types_agree = float(np.mean(
        lo.loc[lr, 'pitch_type'].to_numpy()
        == cache['pitch_type'].to_numpy()[cr]))
    # a name disagreement after normalization is a fingerprint collision
    # across pitchers: unjoin those rows
    if (~agree).any():
        bad = lr[~agree]
        lo.loc[bad, ['pid', 'pa_idx', 'pitch_ord', 'cache_row']] = np.nan
    n_match = int(agree.sum())
    rep = {'season': year, 'loso_rows': int(len(lo)),
           'loso_fp_ok': int(ok_l.sum()), 'loso_dup_keys': int(dup_l.sum()),
           'cache_rows': int(len(cache)), 'cache_dup_keys': int(dup_c.sum()),
           'matched': int(n_match),
           'match_rate_loso': n_match / len(lo),
           'match_rate_cache': n_match / len(cache),
           'name_agree': names_agree, 'type_agree': types_agree,
           'anchors': scale, 'seconds': time.time() - t0}
    print(f'── {year}: LOSO {len(lo)} rows, matched {n_match} '
          f'({n_match / len(lo):.4f} of LOSO, {n_match / len(cache):.4f} of '
          f'cache), dup keys L {int(dup_l.sum())} / C {int(dup_c.sum())}, '
          f'name agree {names_agree:.4f}, type agree {types_agree:.4f}, '
          f'{time.time() - t0:.0f}s', flush=True)
    report.append(rep)
    # per-outing, per-half stuff
    j = lo[lo['pid'].notna()].copy()
    j['pid'] = j['pid'].astype('int64')
    j['half'] = np.where(j['pa_idx'] % 2 == 1, 'o', 'e')

    def agg(g):
        return pd.Series({'stuffAtom': g['atom'].mean(),
                          'stuffAtomF': g['atom_f'].mean(),
                          'stuffRaw': g['stuff_raw'].mean(),
                          'stuffN': len(g)})
    full = j.groupby(['pid', 'game_pk']).apply(agg, include_groups=False)
    halves = (j.groupby(['pid', 'game_pk', 'half'])
                .apply(agg, include_groups=False).unstack('half'))
    halves.columns = [f'{a}_{b}' for a, b in halves.columns]
    out = full.join(halves, how='left').reset_index()
    for c in ('stuffN_o', 'stuffN_e'):
        out[c] = out[c].fillna(0)      # one-PA outings have no odd half
    out['season'] = year
    del cache, fc, fl, m
    gc.collect()
    return lo, out


def stage_join():
    report, los, outs = [], [], []
    for y in SEASONS:
        lo, out = join_season(y, report)
        if lo is not None:
            los.append(lo)
            outs.append(out)
    if not los:
        sys.exit('no LOSO folds found')
    lo_all = pd.concat(los, ignore_index=True)
    tmp = LOSO_PKL + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(lo_all, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, LOSO_PKL)
    print(f'rewrote {LOSO_PKL} ({len(lo_all)} rows, join columns added)')
    halves = pd.concat(outs, ignore_index=True)
    with open(HALVES_PKL, 'wb') as f:
        pickle.dump({'halves': halves, 'join_report': report}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f'saved {HALVES_PKL} ({len(halves)} outings)')


# ══════════════════════════════════════════════════════════════════════════
# stage: fit
# ══════════════════════════════════════════════════════════════════════════
def pear(a, b):
    a = pd.to_numeric(pd.Series(a), errors='coerce')
    b = pd.to_numeric(pd.Series(b), errors='coerce')
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def load_tables():
    t = pickle.load(open(TABLES, 'rb'))
    t['date'] = pd.to_datetime(t['date']).dt.strftime('%Y-%m-%d')
    h = pickle.load(open(HALVES_PKL, 'rb'))
    halves, join_report = h['halves'], h['join_report']
    t = t.merge(halves, on=['pid', 'season', 'game_pk'], how='left')
    seasons = sorted(halves['season'].unique().tolist())
    t = t[t['season'].isin(seasons)].reset_index(drop=True)
    # validation against the OLD full-game LOSO means (a different fit of
    # the same architecture: r should be ~1, the level may shift a little)
    sg = pd.read_csv(STUFF_GAMES_OLD)
    sg['date'] = pd.to_datetime(sg['date']).dt.strftime('%Y-%m-%d')
    sg = sg.rename(columns={'stuffRaw': 'stuffRaw_old', 'stuffN': 'stuffN_old'})
    t = t.merge(sg[['pid', 'season', 'date', 'stuffRaw_old', 'stuffN_old']],
                on=['pid', 'season', 'date'], how='left')
    r_old, n_old = pear(t['stuffRaw'], t['stuffRaw_old'])
    cov = float((t['stuffN'] / t['n']).mean())
    print(f'{len(t)} outings in {seasons}; stuff matched '
          f'{t["stuffAtom"].notna().mean():.4f}, scored-pitch coverage '
          f'{cov:.4f} of n; r(new full-outing stuffRaw, old games CSV) '
          f'{r_old:.4f} (n {n_old})')
    return t, join_report, {'r_vs_old_games_csv': r_old, 'n_vs_old': n_old,
                            'scored_coverage_of_n': cov,
                            'stuff_match_rate': float(t['stuffAtom'].notna().mean()),
                            'seasons': seasons}


def screen(t, feats=FEATS):
    pool = t[t['n'] >= MIN_N]
    hp = pool[(pool['n_o'] >= MIN_HALF_N) & (pool['n_e'] >= MIN_HALF_N)]
    half_n = float(np.nanmean(np.minimum(hp['n_o'], hp['n_e'])))
    rows = []
    for f in feats:
        rel, n_rel = pear(hp[f + '_o'], hp[f + '_e'])
        stab = half_n * (1 - rel) / rel if rel and rel > 0 else np.nan
        pa_ = np.concatenate([hp[f + '_o'], hp[f + '_e']])
        tb = np.concatenate([hp['xrv100_e'], hp['xrv100_o']])
        pred, n_pred = pear(pa_, tb)
        per_season = {}
        for s, g in hp.groupby('season'):
            r_s, _ = pear(g[f + '_o'], g[f + '_e'])
            hn_s = float(np.nanmean(np.minimum(g['n_o'], g['n_e'])))
            per_season[int(s)] = {'rel_r': r_s,
                                  'k': hn_s * (1 - r_s) / r_s if r_s and r_s > 0 else np.nan}
        rows.append({'feature': f, 'rel_r': rel, 'half_n': half_n,
                     'stabilize_n': stab, 'pred_half_r': pred, 'n': n_pred,
                     'per_season': per_season})
    return pd.DataFrame(rows)


def pool_params(t, feats=FEATS):
    params = {}
    pool = t[t['n'] >= MIN_N]
    for season, g in pool.groupby('season'):
        for f in feats:
            vals = g[f].dropna()
            if len(vals) < 100:
                continue
            sd = float(vals.std())
            if sd:
                params[(season, f)] = (float(vals.mean()), sd)
    return params


def apply_shrunk_z(df, feats, kmap, params, n_col, suffix_in=''):
    df = df.reset_index(drop=True)
    for f in feats:
        arr = np.zeros(len(df))
        col = pd.to_numeric(df[f + suffix_in], errors='coerce')
        for season, g in df.groupby('season'):
            p = params.get((season, f))
            if p is None:
                continue
            mu, sd = p
            idx = g.index[col[g.index].notna()]
            z = (col[idx] - mu) / sd
            shrink = g.loc[idx, n_col] / (g.loc[idx, n_col] + kmap[f])
            arr[idx] = (z * shrink).to_numpy()
        df[f + '_sz'] = arr
    return df


def build_split_panel(t, kmap, params, feats=FEATS, min_n=MIN_N,
                      min_half=MIN_HALF_N, keep_meta=False):
    hp = t[(t['n'] >= min_n) & (t['n_o'] >= min_half)
           & (t['n_e'] >= min_half)].copy()
    hp['_oid'] = np.arange(len(hp))
    frames = []
    for fit_suf, tgt_suf in (('_o', '_e'), ('_e', '_o')):
        d = hp.copy()
        for f in feats:
            d[f + '_fit'] = d[f + fit_suf]
        d['_n_half'] = d['n' + fit_suf]
        d['_target'] = d['xrv100' + tgt_suf]
        frames.append(d)
    panel = pd.concat(frames, ignore_index=True)
    panel = apply_shrunk_z(panel, feats, kmap, params, '_n_half',
                           suffix_in='_fit')
    X = panel[[f + '_sz' for f in feats]].to_numpy(float)
    y = panel['_target'].to_numpy(float)
    grp = panel['season'].to_numpy()
    oid = panel['_oid'].to_numpy()
    ok = np.isfinite(y)
    if keep_meta:
        return X[ok], y[ok], grp[ok], oid[ok], panel[ok]
    return X[ok], y[ok], grp[ok], oid[ok]


def oof_r(X, y, grp, cols):
    Xs = X[:, cols]
    pred = np.full(len(y), np.nan)
    folds = {}
    for g in np.unique(grp):
        tr, te = grp != g, grp == g
        A = np.column_stack([np.ones(tr.sum()), Xs[tr]])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p = np.column_stack([np.ones(te.sum()), Xs[te]]) @ beta
        pred[te] = p
        if te.sum() >= 100 and np.std(p) > 0:
            folds[int(g)] = float(np.corrcoef(p, y[te])[0, 1])
    return float(np.corrcoef(pred, y)[0, 1]), folds


def fixed_r(X, y, grp, cols, w):
    """r of a FIXED-weight composite (no fitting): pooled and per season."""
    comp = X[:, cols] @ np.asarray(w, float)
    folds = {int(g): float(np.corrcoef(comp[grp == g], y[grp == g])[0, 1])
             for g in np.unique(grp)}
    return float(np.corrcoef(comp, y)[0, 1]), folds


def full_fit_w(X, y, cols):
    A = np.column_stack([np.ones(len(y)), X[:, cols]])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta[1:] / np.abs(beta[1:]).sum(), beta


def boot_w(X, y, oid, cols, n_boot=N_BOOT, seed=0):
    """Cluster bootstrap by outing (both directions of an outing travel
    together) of the normalized full-fit weights."""
    rng = np.random.default_rng(seed)
    A = np.column_stack([np.ones(len(y)), X[:, cols]])
    n_out = oid.max() + 1
    ws = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, n_out, n_out), minlength=n_out)
        w = cnt[oid].astype(float)
        sw = np.sqrt(w)[:, None]
        beta, *_ = np.linalg.lstsq(A * sw, y * sw[:, 0], rcond=None)
        ws.append(beta[1:] / np.abs(beta[1:]).sum())
    ws = np.array(ws)
    return ws.mean(axis=0), ws.std(axis=0, ddof=1)


def se_of_folds(folds):
    v = np.array(list(folds.values()))
    return float(v.std(ddof=0) / np.sqrt(len(v))) if len(v) else np.nan


def search(X, y, grp, cand, feats):
    """Exhaustive subsets, sizes 1-5. Two 1-SE rules:
      onese        as coded in pitcherplus_outing_grade.py: se = UNPAIRED sd
                   of the best subset's fold r over sqrt(5). That sd is
                   dominated by between-season level differences (2021 r
                   .096 vs 2025 .059), so the threshold is loose.
      onese_paired se of the per-fold DIFFERENCE (subset - best), the
                   quantity the rule is meant to bound."""
    results = []
    for k in range(1, 6):
        for cc in itertools.combinations(cand, k):
            cols = [feats.index(c) for c in cc]
            r, folds = oof_r(X, y, grp, cols)
            results.append({'k': k, 'subset': '+'.join(cc), 'r': r,
                            'se': se_of_folds(folds),
                            'folds': np.array(list(folds.values()))})
    res = pd.DataFrame(results).sort_values('r', ascending=False,
                                            kind='stable').reset_index(drop=True)
    best = res.iloc[0]
    bf = best['folds']
    res['d_best'] = res['folds'].map(lambda a: float((a - bf).mean()))
    res['se_paired'] = res['folds'].map(
        lambda a: float((a - bf).std(ddof=1) / np.sqrt(len(a))))
    res['wins_vs_best'] = res['folds'].map(lambda a: int((a > bf).sum()))
    thresh = best['r'] - best['se']
    onese = res[res['r'] >= thresh].sort_values('k', kind='stable').iloc[0]
    onese_p = res[res['d_best'] >= -res['se_paired']].sort_values(
        'k', kind='stable').iloc[0]
    return res, best, onese, onese_p


def wins(folds_a, folds_b):
    ks = sorted(set(folds_a) & set(folds_b))
    return int(sum(folds_a[k] > folds_b[k] for k in ks)), len(ks)


def eff_shares(w, kmap, ns=(20, 50, 90)):
    out = {}
    for n in ns:
        e = {c: abs(wc) * n / (n + kmap[c]) for c, wc in w.items()}
        s = sum(e.values())
        out[n] = {c: v / s for c, v in e.items()}
    return out


def card_grade(t, w, kmap, params, min_pool=MIN_N):
    """The shipped card construction on the outing table: shrunk-z composite
    with (w, kmap), rescaled 100 +/- 10 against the season pool of outings
    with n >= min_pool (cards/pitcher.py _build_outing_pool + _outing_raw)."""
    feats = list(w)
    d = apply_shrunk_z(t, feats, kmap, params, 'n')
    comp = np.zeros(len(d))
    for c, wc in w.items():
        comp += wc * d[c + '_sz'].to_numpy()
    d['_comp'] = comp
    grade = np.full(len(d), np.nan)
    for s, g in d.groupby('season'):
        pool = g[g['n'] >= min_pool]['_comp']
        mu, sd = float(pool.mean()), float(pool.std())
        grade[g.index] = 100 + 10 * (g['_comp'] - mu) / sd
    d['grade'] = grade
    return d


def half_grade(t, w, kmap, params, min_half=1):
    """Composite on each half with the half's own n (the card construction
    applied to a half-outing), so a short outing's half-to-half signal can
    be measured the same way at every length."""
    feats = list(w)
    hp = t[(t['n_o'] >= min_half) & (t['n_e'] >= min_half)].copy()
    out = {}
    for suf in ('_o', '_e'):
        d = hp.copy()
        for f in feats:
            d[f + '_fit'] = d[f + suf]
        d['_nh'] = d['n' + suf]
        d = apply_shrunk_z(d, feats, kmap, params, '_nh', suffix_in='_fit')
        comp = np.zeros(len(d))
        for c, wc in w.items():
            comp += wc * d[c + '_sz'].to_numpy()
        out[suf] = comp
    hp['comp_o'] = out['_o']
    hp['comp_e'] = out['_e']
    return hp


def short_outing_panel(t, w, kmap, params, label):
    """Grade behaviour by outing length under one construction."""
    d = card_grade(t, w, kmap, params)
    d = d[d['n'] >= 5].copy()
    d['bin'] = pd.cut(d['n'], SHORT_BINS, labels=SHORT_LABELS)
    d = d.sort_values(['pid', 'season', 'date'])
    d['_next_xrv'] = d.groupby(['pid', 'season'])['xrv100'].shift(-1)
    d['_next_grade'] = d.groupby(['pid', 'season'])['grade'].shift(-1)
    hp = half_grade(t, w, kmap, params)
    hp['bin'] = pd.cut(hp['n'], SHORT_BINS, labels=SHORT_LABELS)
    rows = []
    for b in SHORT_LABELS:
        g = d[d['bin'] == b]
        h = hp[hp['bin'] == b]
        r_next, n_next = pear(g['grade'], g['_next_xrv'])
        r_next_g, _ = pear(g['grade'], g['_next_grade'])
        r_x_next, _ = pear(g['xrv100'], g['_next_xrv'])
        # half-to-half: composite on one half -> xRV/100 on the other, both
        # directions stacked; plus the composite's own half reliability
        a = np.concatenate([h['comp_o'], h['comp_e']])
        bb = np.concatenate([h['xrv100_e'], h['xrv100_o']])
        r_half, n_half = pear(a, bb)
        r_comp_rel, _ = pear(h['comp_o'], h['comp_e'])
        r_stuff_rel, _ = pear(h['stuffAtom_o'], h['stuffAtom_e'])
        r_xrv_rel, _ = pear(h['xrv100_o'], h['xrv100_e'])
        rows.append({
            'construction': label, 'bin': b, 'outings': int(len(g)),
            'grade_mean': float(g['grade'].mean()),
            'grade_sd': float(g['grade'].std()),
            'abs_dev_from_100': float((g['grade'] - 100).abs().mean()),
            'share_within_3': float(((g['grade'] - 100).abs() <= 3).mean()),
            'p5': float(g['grade'].quantile(0.05)),
            'p95': float(g['grade'].quantile(0.95)),
            'r_grade_next_xrv': r_next, 'n_next': n_next,
            'r_grade_next_grade': r_next_g,
            'r_xrv_next_xrv': r_x_next,
            'r_half_comp_to_other_xrv': r_half, 'n_half': n_half,
            'r_comp_half_rel': r_comp_rel,
            'r_stuff_half_rel': r_stuff_rel,
            'r_xrv_half_rel': r_xrv_rel})
    return pd.DataFrame(rows)


def evaluate_form(t, stuff_col, kmap, params, tag):
    """Everything the report needs for one stuff form (atom or raw)."""
    feats = [f for f in FEATS if f not in ('stuffAtom', 'stuffRaw')] + [stuff_col]
    X, y, grp, oid = build_split_panel(t, kmap, params, feats)
    four = [stuff_col, 'locRaw', 'cswPct', 'xrv100']
    col = {f: feats.index(f) for f in feats}
    res = {'tag': tag, 'stuff_col': stuff_col, 'panel_obs': int(len(y)),
           'kmap': {f: kmap[f] for f in feats}}
    # (1) stuff only
    r, folds = oof_r(X, y, grp, [col[stuff_col]])
    res['stuff_only'] = {'r': r, 'folds': folds}
    # (2) shipped weights, no refit: shipped k (stuff 42) and new stuff k
    w_ship = [SHIPPED_W[f if f != stuff_col else 'stuffAtom'] for f in four]
    Xs, ys, grps, _ = build_split_panel(
        t, {**kmap, stuff_col: SHIPPED_K['stuffAtom'],
            'locRaw': SHIPPED_K['locRaw'], 'cswPct': SHIPPED_K['cswPct'],
            'xrv100': SHIPPED_K['xrv100']}, params, feats)
    r, folds = fixed_r(Xs, ys, grps, [col[f] for f in four], w_ship)
    res['shipped_asis'] = {'r': r, 'folds': folds,
                           'w': dict(zip(four, w_ship)),
                           'k': {stuff_col: SHIPPED_K['stuffAtom'],
                                 **{f: SHIPPED_K[f] for f in four[1:]}}}
    r, folds = fixed_r(X, y, grp, [col[f] for f in four], w_ship)
    res['shipped_w_new_k'] = {'r': r, 'folds': folds}
    # (3) refit-4
    cols4 = [col[f] for f in four]
    r, folds = oof_r(X, y, grp, cols4)
    w4, beta4 = full_fit_w(X, y, cols4)
    bm, bse = boot_w(X, y, oid, cols4)
    res['refit4'] = {'r': r, 'folds': folds, 'w': dict(zip(four, map(float, w4))),
                     'w_boot_mean': dict(zip(four, map(float, bm))),
                     'w_boot_se': dict(zip(four, map(float, bse))),
                     'beta_raw': list(map(float, beta4))}
    # per-fold weights (are they stable across seasons?)
    pf = {}
    for g in np.unique(grp):
        wf, _ = full_fit_w(X[grp != g], y[grp != g], cols4)
        pf[int(g)] = dict(zip(four, map(float, wf)))
    res['refit4']['w_per_fold'] = pf
    # (4) exhaustive search + 1-SE rule as coded
    cand = [f for f in feats if f != 'rv100']
    sres, best, onese, onese_p = search(X, y, grp, cand, feats)
    pick = onese['subset'].split('+')
    pick_p = onese_p['subset'].split('+')
    r_p, folds_p = oof_r(X, y, grp, [col[f] for f in pick_p])
    wpp, _ = full_fit_w(X, y, [col[f] for f in pick_p])
    r, folds = oof_r(X, y, grp, [col[f] for f in pick])
    wp, _ = full_fit_w(X, y, [col[f] for f in pick])
    bmp, bsep = boot_w(X, y, oid, [col[f] for f in pick])
    res['search'] = {'best': {'subset': best['subset'], 'r': float(best['r']),
                              'se': float(best['se'])},
                     'onese': {'subset': onese['subset'], 'r': float(onese['r']),
                               'se': float(onese['se']), 'k': int(onese['k']),
                               'folds': folds,
                               'w': dict(zip(pick, map(float, wp))),
                               'w_boot_se': dict(zip(pick, map(float, bsep)))},
                     'onese_paired': {'subset': onese_p['subset'],
                                      'r': float(onese_p['r']),
                                      'd_best': float(onese_p['d_best']),
                                      'se_paired': float(onese_p['se_paired']),
                                      'k': int(onese_p['k']), 'folds': folds_p,
                                      'w': dict(zip(pick_p, map(float, wpp)))},
                     'best_per_size': [
                         {k2: (float(v) if isinstance(v, (float, np.floating)) else v)
                          for k2, v in sres[sres['k'] == kk].iloc[0].drop('folds').items()}
                         for kk in range(1, 6)],
                     'top12': sres.head(12).drop(columns=['folds']).to_dict('records'),
                     'four_rank': int((sres['subset'] == '+'.join(sorted(
                         four, key=feats.index))).to_numpy().argmax()) + 1
                     if (sres['subset'] == '+'.join(sorted(four, key=feats.index))).any() else None,
                     'four_r': float(sres.loc[sres['subset'] == '+'.join(
                         sorted(four, key=feats.index)), 'r'].iloc[0])
                     if (sres['subset'] == '+'.join(sorted(four, key=feats.index))).any() else None}
    # (5) wins
    W = {}
    for a, b in (('refit4', 'stuff_only'), ('refit4', 'shipped_asis'),
                 ('onese', 'stuff_only'), ('onese', 'shipped_asis'),
                 ('shipped_asis', 'stuff_only'), ('shipped_w_new_k', 'stuff_only'),
                 ('refit4', 'shipped_w_new_k')):
        fa = res['search']['onese']['folds'] if a == 'onese' else res[a]['folds']
        fb = res['search']['onese']['folds'] if b == 'onese' else res[b]['folds']
        W[f'{a}_vs_{b}'] = wins(fa, fb)
    res['wins'] = W
    # (6) effective shares
    res['shares_refit4'] = eff_shares(res['refit4']['w'], kmap)
    res['shares_shipped'] = eff_shares(
        {stuff_col: SHIPPED_W['stuffAtom'], 'locRaw': SHIPPED_W['locRaw'],
         'cswPct': SHIPPED_W['cswPct'], 'xrv100': SHIPPED_W['xrv100']},
        {stuff_col: SHIPPED_K['stuffAtom'], **{f: SHIPPED_K[f] for f in four[1:]}})
    # (7) stuff-k sweep on the refit-4 (is the objective flat in k?)
    sweep = []
    for kk in (2, 5, 10, 14, 20, 30, 42, 60, 100, 200, 400, 800):
        km = dict(kmap)
        km[stuff_col] = float(kk)
        Xk, yk, gk, _ = build_split_panel(t, km, params, feats)
        r, folds = oof_r(Xk, yk, gk, cols4)
        wk, _ = full_fit_w(Xk, yk, cols4)
        sweep.append({'k': kk, 'r': r, 'folds': folds,
                      'w': dict(zip(four, map(float, wk)))})
    res['stuff_k_sweep'] = sweep
    return res, sres


LEN_BINS = [19, 29, 49, 79, 200]
LEN_LABELS = ['20-29', '30-49', '50-79', '80+']
K_GRID = [2, 5, 10, 14, 20, 25, 30, 35, 42, 50, 60, 80, 100, 150, 300]


def _oof_pred(X, y, grp, cols, extra=None):
    Xs = X[:, cols] if extra is None else np.column_stack([X[:, cols], extra])
    pred = np.full(len(y), np.nan)
    for g in np.unique(grp):
        tr, te = grp != g, grp == g
        A = np.column_stack([np.ones(tr.sum()), Xs[tr]])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[te] = np.column_stack([np.ones(te.sum()), Xs[te]]) @ beta
    return pred


def diagnostics(t, kmap, params):
    """Why the objective prefers a stuff k near 42 when the measured
    reliability k is ~5: the stuff -> other-half xRV slope grows with outing
    length, and n/(n+k) with k ~ 42 reproduces that ratio. Atom form."""
    feats = [f for f in FEATS if f != 'stuffRaw']
    col = {f: feats.index(f) for f in feats}
    cols4 = [col[f] for f in FOUR]
    k_meas = kmap['stuffAtom']
    out = {'k_measured_atom': k_meas}
    # (a) fine k grid, paired against k = 42
    grid = {}
    for kk in K_GRID:
        km = dict(kmap)
        km['stuffAtom'] = float(kk)
        X, y, grp, oid = build_split_panel(t, km, params, feats)
        r, f = oof_r(X, y, grp, cols4)
        grid[kk] = (r, np.array(list(f.values())))
    ref = grid[42][1]
    out['k_grid'] = [{'k': kk, 'r': r, 'folds': f.tolist(),
                      'd_vs_42': float((f - ref).mean()),
                      'se_paired': float((f - ref).std(ddof=1) / np.sqrt(len(f))),
                      'wins_vs_42': int((f > ref).sum())}
                     for kk, (r, f) in grid.items()]
    # (b) mechanism: unshrunk stuff slope by outing length; OOF r by length
    km0 = dict(kmap)
    km0['stuffAtom'] = 1e-9
    X0, y0, g0, _, pan0 = build_split_panel(t, km0, params, feats, keep_meta=True)
    b0 = np.asarray(pd.cut(pan0['n'].to_numpy(), LEN_BINS, labels=LEN_LABELS))
    mech = []
    for lab in LEN_LABELS:
        m = b0 == lab
        x = X0[m, col['stuffAtom']]
        yy = y0[m]
        A = np.column_stack([np.ones(m.sum()), x])
        beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
        w, _ = full_fit_w(X0[m], y0[m], cols4)
        mech.append({'bin': lab, 'obs': int(m.sum()), 'slope_unshrunk': float(beta[1]),
                     'r': float(np.corrcoef(x, yy)[0, 1]),
                     'mean_z': float(x.mean()), 'sd_z': float(x.std()),
                     'mean_y': float(yy.mean()), 'sd_y': float(yy.std()),
                     'w4_unshrunk_stuff': dict(zip(FOUR, map(float, w)))})
    out['mechanism'] = mech
    by_len = []
    for kk in (k_meas, 42.0, 300.0):
        km = dict(kmap)
        km['stuffAtom'] = kk
        X, y, grp, oid, pan = build_split_panel(t, km, params, feats, keep_meta=True)
        nh = pan['_n_half'].to_numpy()
        n_out = pan['n'].to_numpy()
        p = _oof_pred(X, y, grp, cols4)
        p_n = _oof_pred(X, y, grp, cols4,
                        extra=np.column_stack([np.log(nh), np.log(n_out)]))
        comp = X[:, cols4] @ np.array([SHIPPED_W[f] for f in FOUR])
        b = np.asarray(pd.cut(n_out, LEN_BINS, labels=LEN_LABELS))
        for lab in LEN_LABELS + ['all']:
            m = np.ones(len(y), bool) if lab == 'all' else (b == lab)
            by_len.append({'stuff_k': round(kk, 1), 'bin': lab, 'obs': int(m.sum()),
                           'r_refit4': float(np.corrcoef(p[m], y[m])[0, 1]),
                           'r_refit4_plus_logn': float(np.corrcoef(p_n[m], y[m])[0, 1]),
                           'r_shippedW': float(np.corrcoef(comp[m], y[m])[0, 1]),
                           'r_stuff_sz': float(np.corrcoef(X[m, col['stuffAtom']], y[m])[0, 1])})
    out['oof_by_length'] = by_len
    # (c) refit-4 + bootstrap at k 20 / 30 / 42, deviation of the shipped
    #     vector in SE units, effective shares; paired 1-SE search at k 42
    ship = np.array([SHIPPED_W[f] for f in FOUR])
    refits = []
    for kk in (k_meas, 20.0, 30.0, 42.0):
        km = dict(kmap)
        km['stuffAtom'] = kk
        X, y, grp, oid = build_split_panel(t, km, params, feats)
        r, f = oof_r(X, y, grp, cols4)
        w, _ = full_fit_w(X, y, cols4)
        bm, bse = boot_w(X, y, oid, cols4)
        r_ship, f_ship = fixed_r(X, y, grp, cols4, ship)
        refits.append({'stuff_k': kk, 'r_oof': r, 'folds': f,
                       'w': dict(zip(FOUR, map(float, w))),
                       'w_boot_se': dict(zip(FOUR, map(float, bse))),
                       'shipped_dev_se': dict(zip(FOUR, map(float, (ship - w) / bse))),
                       'shares': eff_shares(dict(zip(FOUR, map(float, w))), km),
                       'r_shippedW_at_this_k': r_ship, 'folds_shippedW': f_ship,
                       'wins_refit_vs_shippedW': wins(f, f_ship)})
        if kk == 42.0:
            cand = [f2 for f2 in feats if f2 != 'rv100']
            sres, best, onese, onese_p = search(X, y, grp, cand, feats)
            out['search_k42'] = {
                'best': {'subset': best['subset'], 'r': float(best['r']),
                         'se': float(best['se'])},
                'onese': {'subset': onese['subset'], 'r': float(onese['r']),
                          'd_best': float(onese['d_best']),
                          'se_paired': float(onese['se_paired']),
                          'wins_vs_best': int(onese['wins_vs_best'])},
                'onese_paired': {'subset': onese_p['subset'], 'r': float(onese_p['r']),
                                 'd_best': float(onese_p['d_best']),
                                 'se_paired': float(onese_p['se_paired']),
                                 'wins_vs_best': int(onese_p['wins_vs_best'])},
                'best_per_size': [
                    {k2: (float(v) if isinstance(v, (float, np.floating)) else v)
                     for k2, v in sres[sres['k'] == kk2].iloc[0].drop('folds').items()}
                    for kk2 in range(1, 6)]}
    out['refits_by_k'] = refits
    return out


def stage_fit():
    t, join_report, val = load_tables()
    scr = screen(t)
    print('\n══ outing-grain screen (odd/even PA halves, TRUE stuff halves) ══')
    print(scr.drop(columns=['per_season']).round(4).to_string(index=False))
    kmeas = {r['feature']: float(r['stabilize_n']) for _, r in scr.iterrows()}
    kmap = dict(kmeas)
    kmap['locRaw'] = LOC_K_FIXED
    for f in FEATS:
        if not np.isfinite(kmap.get(f, np.nan)) or kmap[f] <= 0:
            kmap[f] = 200.0
    print('\nk (measured; loc fixed 185):',
          {f: round(k, 1) for f, k in kmap.items()})
    params = pool_params(t)

    results = {}
    searches = {}
    for stuff_col, tag in (('stuffAtom', 'atom'), ('stuffRaw', 'raw')):
        print(f'\n══ form: {tag} ({stuff_col}) ══', flush=True)
        res, sres = evaluate_form(t, stuff_col, kmap, params, tag)
        results[tag] = res
        searches[tag] = sres
        for name in ('stuff_only', 'shipped_asis', 'shipped_w_new_k', 'refit4'):
            print(f'  {name:16s} r {res[name]["r"]:.4f}  folds '
                  + ' '.join(f'{k}:{v:.4f}' for k, v in res[name]['folds'].items()))
        o = res['search']['onese']
        print(f'  1-SE pick {o["subset"]} r {o["r"]:.4f} folds '
              + ' '.join(f'{k}:{v:.4f}' for k, v in o['folds'].items()))
        print(f'  refit4 w {res["refit4"]["w"]}  se {res["refit4"]["w_boot_se"]}')
        print(f'  wins {res["wins"]}')
    searches['atom'].drop(columns=['folds']).to_csv(SEARCH_CSV, index=False)
    diag = diagnostics(t, kmap, params)

    # short outings under the shipped construction and under the refit
    w_ref = results['atom']['refit4']['w']
    short = pd.concat([
        short_outing_panel(t, {'stuffAtom': SHIPPED_W['stuffAtom'],
                               'locRaw': SHIPPED_W['locRaw'],
                               'cswPct': SHIPPED_W['cswPct'],
                               'xrv100': SHIPPED_W['xrv100']},
                           SHIPPED_K, params, 'shipped'),
        short_outing_panel(t, w_ref, kmap, params, 'refit4'),
    ], ignore_index=True)
    print('\n══ outing length behaviour ══')
    print(short.round(3).to_string(index=False))

    out = {
        'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'validation': val,
        'join_report': [{k: v for k, v in r.items() if k != 'anchors'}
                        for r in join_report],
        'anchors': {r['season']: r['anchors'] for r in join_report},
        'screen': [{**{k: v for k, v in r.items()}} for r in scr.to_dict('records')],
        'k_measured': kmeas, 'k_used': kmap,
        'shipped': {'w': SHIPPED_W, 'k': SHIPPED_K},
        'results': results,
        'short_outings': short.to_dict('records'),
        'diagnostics': diag,
        'settings': {'MIN_N': MIN_N, 'MIN_HALF_N': MIN_HALF_N,
                     'N_BOOT': N_BOOT, 'QUAL_N': QUAL_N,
                     'anchor_borrow': {k: list(v) for k, v in ANCHOR_BORROW.items()}},
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=1, default=_json_default)
    print(f'\nsaved {OUT_JSON}')
    write_md(out, scr, short)
    print(f'saved {OUT_MD}')


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.strftime('%Y-%m-%d')
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def _f(x, d=4):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def write_md(out, scr, short):
    L = []
    L.append('# Outing Pitching+ refit with true per-half stuff (2026-09-02)\n')
    v = out['validation']
    L.append(f'Seasons {v["seasons"]}; stuff matched on {_f(v["stuff_match_rate"])} of outings; '
             f'scored-pitch coverage {_f(v["scored_coverage_of_n"])} of n; '
             f'r(new full-outing stuffRaw, old games CSV) {_f(v["r_vs_old_games_csv"])} '
             f'(n {v["n_vs_old"]}).\n')
    L.append('## Join\n')
    L.append('| season | LOSO rows | matched | of LOSO | of cache | dup keys L/C | name agree | type agree |')
    L.append('|---|---|---|---|---|---|---|---|')
    for r in out['join_report']:
        L.append(f'| {r["season"]} | {r["loso_rows"]} | {r["matched"]} | '
                 f'{_f(r["match_rate_loso"])} | {_f(r["match_rate_cache"])} | '
                 f'{r["loso_dup_keys"]}/{r["cache_dup_keys"]} | {_f(r["name_agree"])} | '
                 f'{_f(r["type_agree"])} |')
    L.append('\n## Screen (outings >= 20 pitches, halves >= 8)\n')
    L.append('| feature | rel_r | half_n | k (stabilize_n) | pred_half_r | per-season k |')
    L.append('|---|---|---|---|---|---|')
    for r in scr.to_dict('records'):
        ps = ' / '.join(f'{s}:{_f(d["k"], 0)}' for s, d in r['per_season'].items())
        L.append(f'| {r["feature"]} | {_f(r["rel_r"])} | {_f(r["half_n"], 1)} | '
                 f'{_f(r["stabilize_n"], 1)} | {_f(r["pred_half_r"])} | {ps} |')
    L.append(f'\nk used: ' + ', '.join(f'{f} {k:.1f}' for f, k in out['k_used'].items()))
    L.append('\nShipped k: ' + ', '.join(f'{f} {k}' for f, k in out['shipped']['k'].items()) + '\n')
    for tag in ('atom', 'raw'):
        R = out['results'][tag]
        L.append(f'## Form: {tag} ({R["stuff_col"]}), panel {R["panel_obs"]} obs\n')
        L.append('| config | pooled r | ' + ' | '.join(str(s) for s in R['stuff_only']['folds']) + ' |')
        L.append('|---|---|' + '---|' * len(R['stuff_only']['folds']))
        rows = [('stuff only', R['stuff_only']),
                ('shipped W + shipped k', R['shipped_asis']),
                ('shipped W + new stuff k', R['shipped_w_new_k']),
                ('refit-4 (OOF)', R['refit4']),
                (f'1-SE pick {R["search"]["onese"]["subset"]} (OOF)', R['search']['onese'])]
        for name, d in rows:
            L.append(f'| {name} | {_f(d["r"])} | '
                     + ' | '.join(_f(x) for x in d['folds'].values()) + ' |')
        L.append('\nWins (season folds): ' + ', '.join(
            f'{k} {a}/{b}' for k, (a, b) in R['wins'].items()))
        L.append('\n| weights | ' + ' | '.join(R['refit4']['w']) + ' |')
        L.append('|---|' + '---|' * 4)
        L.append('| shipped | ' + ' | '.join(_f(x, 3) for x in R['shipped_asis']['w'].values()) + ' |')
        L.append('| refit-4 | ' + ' | '.join(_f(x, 3) for x in R['refit4']['w'].values()) + ' |')
        L.append('| refit-4 boot SE | ' + ' | '.join(_f(x, 3) for x in R['refit4']['w_boot_se'].values()) + ' |')
        for s, w in R['refit4']['w_per_fold'].items():
            L.append(f'| refit-4 fold {s} | ' + ' | '.join(_f(x, 3) for x in w.values()) + ' |')
        o = R['search']['onese']
        L.append(f'\n1-SE pick: {o["subset"]} (k={o["k"]}, r {_f(o["r"])}, se {_f(o["se"])}); '
                 f'best {R["search"]["best"]["subset"]} r {_f(R["search"]["best"]["r"])} '
                 f'se {_f(R["search"]["best"]["se"])}; the 4-term set ranks '
                 f'{R["search"]["four_rank"]} at r {_f(R["search"]["four_r"])}.')
        L.append('1-SE weights: ' + ', '.join(f'{k} {_f(v, 3)} (se {_f(o["w_boot_se"][k], 3)})'
                                             for k, v in o['w'].items()))
        op = R['search']['onese_paired']
        L.append(f'\nPAIRED 1-SE pick: {op["subset"]} (k={op["k"]}, r {_f(op["r"])}, deficit vs best '
                 f'{op["d_best"]:+.4f} +/- {_f(op["se_paired"])}); folds '
                 + ' '.join(_f(x) for x in op['folds'].values()) + '.')
        L.append('\nBest per size (deficit vs best, paired se, wins vs best):\n')
        L.append('| size | subset | r | deficit | paired se | wins |')
        L.append('|---|---|---|---|---|---|')
        for b in R['search']['best_per_size']:
            L.append(f'| {b["k"]} | {b["subset"]} | {_f(b["r"])} | {b["d_best"]:+.4f} | {_f(b["se_paired"])} | {b["wins_vs_best"]}/5 |')
        L.append('\nTop 12 subsets:\n')
        L.append('| k | subset | r | se |')
        L.append('|---|---|---|---|')
        for r in R['search']['top12']:
            L.append(f'| {r["k"]} | {r["subset"]} | {_f(r["r"])} | {_f(r["se"])} |')
        L.append('\nEffective shares (refit-4 / shipped):\n')
        L.append('| n | ' + ' | '.join(R['refit4']['w']) + ' |')
        L.append('|---|' + '---|' * 4)
        for n in (20, 50, 90):
            a = R['shares_refit4'][n]
            b = R['shares_shipped'][n]
            L.append(f'| {n} | ' + ' | '.join(f'{_f(a[c], 2)} / {_f(b[c if c in b else list(b)[0]], 2)}'
                                              for c in a) + ' |')
        L.append('\nStuff-k sweep (refit-4 OOF pooled r; weights refit at each k):\n')
        L.append('| k | r | ' + ' | '.join(R['refit4']['w']) + ' |')
        L.append('|---|---|' + '---|' * 4)
        for s in R['stuff_k_sweep']:
            L.append(f'| {s["k"]} | {_f(s["r"])} | ' + ' | '.join(_f(x, 3) for x in s['w'].values()) + ' |')
        L.append('')
    D = out['diagnostics']
    L.append('## Stuff k: fine grid on the refit-4 (atom), paired against k = 42\n')
    L.append(f'Measured outing-grain reliability k for the stuff atom: {_f(D["k_measured_atom"], 1)}.\n')
    L.append('| k | pooled OOF r | ' + ' | '.join(str(s) for s in out['results']['atom']['refit4']['folds']) + ' | d vs 42 | paired se | wins vs 42 |')
    L.append('|---|---|' + '---|' * 5 + '---|---|---|')
    for g in D['k_grid']:
        L.append(f'| {g["k"]} | {_f(g["r"])} | ' + ' | '.join(_f(x) for x in g['folds'])
                 + f' | {g["d_vs_42"]:+.5f} | {_f(g["se_paired"], 5)} | {g["wins_vs_42"]}/5 |')
    L.append('\n### Mechanism: unshrunk stuff z -> other-half xRV/100 by outing length\n')
    L.append('| outing n | obs | slope | r | mean z | sd z | mean y | sd y | 4-term w (stuff unshrunk) |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for m in D['mechanism']:
        L.append(f'| {m["bin"]} | {m["obs"]} | {_f(m["slope_unshrunk"])} | {_f(m["r"])} | '
                 f'{_f(m["mean_z"], 3)} | {_f(m["sd_z"], 3)} | {_f(m["mean_y"], 3)} | {_f(m["sd_y"], 3)} | '
                 + ' '.join(f'{k[:5]} {_f(v, 3)}' for k, v in m['w4_unshrunk_stuff'].items()) + ' |')
    L.append('\n| stuff k | outing n | obs | r refit-4 | r refit-4 + log n terms | r shipped W | r stuff sz alone |')
    L.append('|---|---|---|---|---|---|---|')
    for m in D['oof_by_length']:
        L.append(f'| {m["stuff_k"]} | {m["bin"]} | {m["obs"]} | {_f(m["r_refit4"])} | '
                 f'{_f(m["r_refit4_plus_logn"])} | {_f(m["r_shippedW"])} | {_f(m["r_stuff_sz"])} |')
    L.append('\n### Refit-4 by stuff k (atom): weights, cluster-bootstrap SE, shipped deviation\n')
    L.append('| stuff k | OOF r | r shipped W at this k | wins refit vs shipped | ' + ' | '.join(FOUR) + ' | shares n=20 | n=50 | n=90 |')
    L.append('|---|---|---|---|' + '---|' * 4 + '---|---|---|')
    for rf in D['refits_by_k']:
        L.append(f'| {_f(rf["stuff_k"], 1)} | {_f(rf["r_oof"])} | {_f(rf["r_shippedW_at_this_k"])} | '
                 f'{rf["wins_refit_vs_shippedW"][0]}/{rf["wins_refit_vs_shippedW"][1]} | '
                 + ' | '.join(f'{_f(rf["w"][f], 3)} (se {_f(rf["w_boot_se"][f], 3)}, ship {rf["shipped_dev_se"][f]:+.1f} se)' for f in FOUR)
                 + ' | ' + ' | '.join('/'.join(_f(rf['shares'][n][f], 2) for f in FOUR) for n in (20, 50, 90)) + ' |')
    S = D['search_k42']
    L.append(f'\n### Search at stuff k 42 (atom)\n')
    L.append(f'Best {S["best"]["subset"]} r {_f(S["best"]["r"])} (unpaired se {_f(S["best"]["se"])}). '
             f'1-SE as coded: {S["onese"]["subset"]} r {_f(S["onese"]["r"])}, deficit {S["onese"]["d_best"]:+.4f} '
             f'+/- {_f(S["onese"]["se_paired"])} paired, wins vs best {S["onese"]["wins_vs_best"]}/5. '
             f'1-SE paired: {S["onese_paired"]["subset"]} r {_f(S["onese_paired"]["r"])}, deficit '
             f'{S["onese_paired"]["d_best"]:+.4f} +/- {_f(S["onese_paired"]["se_paired"])}, wins vs best '
             f'{S["onese_paired"]["wins_vs_best"]}/5.\n')
    L.append('| size | best subset | r | deficit vs best | paired se | wins vs best |')
    L.append('|---|---|---|---|---|---|')
    for b in S['best_per_size']:
        L.append(f'| {b["k"]} | {b["subset"]} | {_f(b["r"])} | {b["d_best"]:+.4f} | {_f(b["se_paired"])} | {b["wins_vs_best"]}/5 |')
    L.append('')
    L.append('## Outing length behaviour (grade = card construction, season pool n >= 20)\n')
    cols = ['construction', 'bin', 'outings', 'grade_mean', 'grade_sd',
            'abs_dev_from_100', 'share_within_3', 'p5', 'p95',
            'r_grade_next_xrv', 'r_xrv_next_xrv', 'r_half_comp_to_other_xrv',
            'n_half', 'r_comp_half_rel', 'r_stuff_half_rel', 'r_xrv_half_rel']
    L.append('| ' + ' | '.join(cols) + ' |')
    L.append('|' + '---|' * len(cols))
    for r in short.to_dict('records'):
        L.append('| ' + ' | '.join(
            (str(r[c]) if c in ('construction', 'bin', 'outings', 'n_half')
             else _f(r[c], 3)) for c in cols) + ' |')
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(L) + '\n')


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'fit'
    if stage == 'join':
        stage_join()
    elif stage == 'fit':
        stage_fit()
    else:
        sys.exit(f'unknown stage {stage}')


if __name__ == '__main__':
    main()
