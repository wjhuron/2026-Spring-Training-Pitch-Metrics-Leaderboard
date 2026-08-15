#!/usr/bin/env python3
"""pitcherplus_search.py — Pitcher+ candidate-feature search, 2021-2025.

Phase 1 of the Pitcher+ project (2026-07-24): build pitcher-season-half
feature tables with a future-xRV/100 target, then screen every candidate on
two panels:

  panel S (split-half):  features on half A -> xRV/100 on half B (odd/even
                         game split within season, scored both directions)
  panel Y (year-pair):   features year N (full season) -> xRV/100 year N+1

Per-pitch xRV mirrors pipeline_compute.compute_xrv exactly:
  BIP with xwOBA:    (xwOBA - lgWOBA)/wOBAScale + per-count anchor offset
  BIP without xwOBA: league mean anchored BIP value for that count
  everything else:   RunExp (delta_pitcher_run_exp), pitcher perspective
Per-season FG Guts constants (2021-2024 from train_stuff.HIST_GUTS,
2025 from the v11 prior constants). Count offsets are rebuilt per season
from that season's league BIPs (min 50 per side, matching
build_bip_count_offsets); a count that misses the floor gets 0 and a
warning, not the 2026 fallback table.

Savant tags for 2021-2025 — tag-sensitive candidates (usage entropy,
per-type bests, fastball velo) carry the same caveat as the July battery.

Historical Loc+ (per-season surfaces) and LOSO Stuff+ merge in later via
data/_pplus_locplus_hist.csv and data/_pplus_stuff_loso.csv when present —
keyed on (pid, season, half).

Usage:
  python3 scripts/research/stuff/pitcherplus_search.py --build     (~10-20 min)
  python3 scripts/research/stuff/pitcherplus_search.py --screen
Outputs: data/_pplus_tables.pkl, data/_pplus_screen.csv
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'misc'))  # leaderboard_metric_battery moved in 2026-08 reorg

import leaderboard_metric_battery as bat  # noqa: E402

DATA = os.path.join(ROOT, 'data')
TABLES_PKL = os.path.join(DATA, '_pplus_tables.pkl')
SCREEN_CSV = os.path.join(DATA, '_pplus_screen.csv')
LOC_CSV = os.path.join(DATA, '_pplus_locplus_hist.csv')
STUFF_CSV = os.path.join(DATA, '_pplus_stuff_loso.csv')

SEASONS = [2021, 2022, 2023, 2024, 2025]

# FG Guts (lgWOBA, wOBAScale): 21-24 = train_stuff.HIST_GUTS,
# 2025 = PRIOR_LG_WOBA/PRIOR_WOBA_SCALE (0.3131, 1.2317).
GUTS = {
    2021: (0.314, 1.209), 2022: (0.310, 1.259),
    2023: (0.318, 1.204), 2024: (0.310, 1.242),
    2025: (0.3131, 1.2317),
}

ZONE_HALF_WIDTH = bat.ZONE_HALF_WIDTH
HEART_X = ZONE_HALF_WIDTH * 2.0 / 3.0     # inner 2/3 of the zone width
FB_TYPES = {'FF', 'SI', 'FA'}


# ── per-pitch xRV (mirrors compute_xrv) ─────────────────────────────────
def add_xrv(df, year):
    # cache index is non-unique (concatenated chunks) — label lookups like
    # xwv[g.index] silently pull duplicate rows without this reset
    df = df.reset_index(drop=True)
    lg, scale = GUTS[year]
    is_bip_desc = df['description'].fillna('') == 'hit_into_play'
    xw = df['estimated_woba_using_speedangle']
    xwv = (xw - lg) / scale                      # hitter-perspective wOBA value
    cnt = list(zip(df['balls'].astype('Int64'), df['strikes'].astype('Int64')))
    df['_cnt'] = cnt

    b = df[is_bip_desc]
    off, bip_mean = {}, {}
    for c, g in b.groupby('_cnt'):
        re_vals = -g['rv'].dropna()              # hitter perspective
        xw_vals = xwv[g.index].dropna()
        if pd.isna(c[0]) or pd.isna(c[1]):
            continue
        if len(re_vals) >= 50 and len(xw_vals) >= 50:
            off[c] = re_vals.mean() - xw_vals.mean()
        else:
            print(f'   [warn] {year} count {c}: BIP n {len(re_vals)}/'
                  f'{len(xw_vals)} < 50, offset 0')
    for c, g in b.groupby('_cnt'):
        v = (xwv[g.index] + off.get(c, 0.0)).dropna()
        if len(v) >= 50:
            bip_mean[c] = v.mean()

    off_col = pd.Series([off.get(c, 0.0) for c in cnt], index=df.index)
    mean_col = pd.Series([bip_mean.get(c, np.nan) for c in cnt],
                         index=df.index)
    hv = np.where(is_bip_desc & xw.notna(), xwv + off_col,
                  np.where(is_bip_desc, mean_col, -df['rv']))
    df['xrv_pitch'] = -pd.Series(hv, index=df.index)    # pitcher perspective
    df.drop(columns=['_cnt'], inplace=True)
    return df


def add_extra_flags(df):
    zn = (df['plate_z'] - df['sz_bot']) / (df['sz_top'] - df['sz_bot'])
    df['heart'] = (df['plate_x'].abs() <= HEART_X) & zn.between(1 / 6, 5 / 6)
    df['behind'] = df['balls'] > df['strikes']
    df['ahead'] = df['strikes'] > df['balls']
    df['putaway'] = df['s2'] & df['k']
    return df


# ── pitcher-half aggregation: battery rates + Pitcher+ extras ───────────
def _xrv100(g):
    v = g['xrv_pitch'].dropna()
    n = len(g)
    return (v.sum() / n * 100) if n and len(v) else np.nan


def pplus_aggs(g):
    out = bat.rate_aggs(g).to_dict()
    n = len(g)
    out['playerName'] = g['player_name'].iloc[0]
    out['xrv100'] = _xrv100(g)
    out['games'] = g['game_pk'].nunique()
    out['pitchesPerG'] = n / out['games'] if out['games'] else np.nan
    s2n = g['s2'].sum()
    out['putawayPct'] = g['putaway'].sum() / s2n if s2n else np.nan
    beh = g['behind'].sum()
    out['zoneBehindPct'] = ((g['iz'] & g['behind']).sum() / beh
                            if beh else np.nan)
    out['heartPct'] = g['heart'].sum() / n if n else np.nan
    out['avgVelo'] = g['release_speed'].mean()
    out['veloP90'] = g['release_speed'].quantile(0.9)
    out['extension'] = g['release_extension'].mean()
    fb = g[g['pitch_type'].isin(FB_TYPES)]
    out['fbVelo'] = fb['release_speed'].mean() if len(fb) >= 20 else np.nan
    out['fipCore'] = ((13 * g['hr'].sum()
                       + 3 * (g['bbw'].sum() + g['hbp'].sum())
                       - 2 * g['k'].sum()) / out['pa']
                      if out['pa'] else np.nan)

    # platoon (min 100 pitches per side)
    xl = _xrv100(g[g['stand'] == 'L']) if (g['stand'] == 'L').sum() >= 100 \
        else np.nan
    xr = _xrv100(g[g['stand'] == 'R']) if (g['stand'] == 'R').sum() >= 100 \
        else np.nan
    out['platoonGap'] = abs(xl - xr) if xl == xl and xr == xr else np.nan
    out['weakSideXrv100'] = min(xl, xr) if xl == xl and xr == xr else np.nan

    # arsenal shape (Savant tags — caveat)
    pt = g['pitch_type'].dropna()
    u = pt.value_counts(normalize=True) if len(pt) else pd.Series(dtype=float)
    out['usageEntropy'] = float(-(u * np.log(u)).sum()) if len(u) else np.nan
    out['nTypes5'] = int((u >= 0.05).sum()) if len(u) else np.nan
    best, worst = np.nan, np.nan
    rel_sx, rel_sz, loc_sc, wsum = [], [], [], []
    for t, gt in g.groupby('pitch_type'):
        nt = len(gt)
        if nt >= 30:
            rel_sx.append(gt['release_pos_x'].std() * nt)
            rel_sz.append(gt['release_pos_z'].std() * nt)
            loc_sc.append(math.sqrt(gt['plate_x'].var()
                                    + gt['plate_z'].var()) * nt)
            wsum.append(nt)
        if nt >= 50 and nt / n >= 0.10:
            x = _xrv100(gt)
            if x == x:
                best = x if not best == best else max(best, x)
                worst = x if not worst == worst else min(worst, x)
    out['bestPitchXrv100'], out['worstPitchXrv100'] = best, worst
    tot = sum(wsum)
    out['relStdX'] = sum(rel_sx) / tot if tot else np.nan
    out['relStdZ'] = sum(rel_sz) / tot if tot else np.nan
    out['locScatter'] = sum(loc_sc) / tot if tot else np.nan
    return pd.Series(out)


def build():
    rows = []
    for year in SEASONS:
        print(f'── {year}')
        df = bat.load_season(year)
        df = add_extra_flags(df)
        df = add_xrv(df, year)
        # league calibration check: summed xRV vs summed RV
        print(f'   league xRV {df["xrv_pitch"].sum():+.0f} '
              f'vs RV {df["rv"].sum():+.0f} '
              f'({len(df)} pitches)')
        dfx = bat.add_half(df, 'pitcher')
        for half_label, sub in (('full', dfx), ('A', dfx[dfx['half'] == 'A']),
                                ('B', dfx[dfx['half'] == 'B'])):
            r = sub.groupby('pitcher', group_keys=False).apply(pplus_aggs)
            r['season'] = year
            r['half'] = half_label
            r.index.name = 'pid'
            rows.append(r.reset_index())
        print(f'   pitchers {len(rows[-3])}')
    out = pd.concat(rows, ignore_index=True)
    with open(TABLES_PKL, 'wb') as f:
        pickle.dump(out, f)
    print(f'saved {TABLES_PKL} ({len(out)} rows)')


# ── screening ────────────────────────────────────────────────────────────
MIN_FULL, MIN_HALF = 800, 300

META = {'pid', 'playerName', 'season', 'half', 'n', 'pa', 'nbip', 'games'}


def merge_external(t):
    """Join historical Loc+ / LOSO Stuff+ when their CSVs exist."""
    for path, cols in ((LOC_CSV, ['locRaw']), (STUFF_CSV, ['stuffRaw'])):
        if os.path.exists(path):
            ext = pd.read_csv(path)
            t = t.merge(ext[['pid', 'season', 'half'] + cols],
                        on=['pid', 'season', 'half'], how='left')
            print(f'merged {os.path.basename(path)}')
    return t


def add_xrvoe(t):
    """xRVOE analog: xrv100 residual on (stuffRaw, locRaw), coefficients fit
    per season on full-season rows (league-level fit, minimal leakage),
    applied to every row. Mirrors the production idea: what the process
    models can't explain."""
    if 'stuffRaw' not in t.columns or 'locRaw' not in t.columns:
        return t
    t['xrvoe'] = np.nan
    for season, g in t.groupby('season'):
        fit = g[(g['half'] == 'full') & (g['n'] >= MIN_FULL)
                & g['stuffRaw'].notna() & g['locRaw'].notna()
                & g['xrv100'].notna()]
        if len(fit) < 100:
            continue
        X = np.column_stack([np.ones(len(fit)), fit['stuffRaw'],
                             fit['locRaw']])
        beta, *_ = np.linalg.lstsq(X, fit['xrv100'].values, rcond=None)
        m = ((t['season'] == season) & t['stuffRaw'].notna()
             & t['locRaw'].notna())
        t.loc[m, 'xrvoe'] = (t.loc[m, 'xrv100']
                             - (beta[0] + beta[1] * t.loc[m, 'stuffRaw']
                                + beta[2] * t.loc[m, 'locRaw']))
    return t


def screen():
    t = pickle.load(open(TABLES_PKL, 'rb'))
    t = merge_external(t)
    feats = [c for c in t.columns if c not in META]
    for c in feats:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    t = add_xrvoe(t)
    if 'xrvoe' in t.columns and 'xrvoe' not in feats:
        feats.append('xrvoe')
    full = t[(t['half'] == 'full') & (t['n'] >= MIN_FULL)]
    A = t[(t['half'] == 'A') & (t['n'] >= MIN_HALF)]
    B = t[(t['half'] == 'B') & (t['n'] >= MIN_HALF)]
    ab = A.merge(B, on=['pid', 'season'], suffixes=('_a', '_b'))
    pairs = full.merge(full.assign(season=full['season'] - 1),
                       on=['pid', 'season'], suffixes=('', '_n1'))
    rows = []
    for m in feats:
        sh, n_sh = bat.pear(ab[m + '_a'], ab[m + '_b'])
        half_n = float(np.nanmean(np.minimum(ab['n_a'], ab['n_b']))) \
            if len(ab) else np.nan
        stab = half_n * (1 - sh) / sh if sh and sh > 0 else np.nan
        # split-half predictive: both directions stacked
        pa_ = np.concatenate([pd.to_numeric(ab[m + '_a'], errors='coerce'),
                              pd.to_numeric(ab[m + '_b'], errors='coerce')])
        tb = np.concatenate([pd.to_numeric(ab['xrv100_b'], errors='coerce'),
                             pd.to_numeric(ab['xrv100_a'], errors='coerce')])
        pred_s, n_s = bat.pear(pa_, tb)
        pred_y, n_y = bat.pear(pairs[m], pairs['xrv100_n1'])
        pred_y2, _ = bat.pear(pairs[m], pairs['paxw_n1'])
        curr, _ = bat.pear(full[m], full['xrv100'])
        rows.append({'feature': m, 'reliability_r': sh, 'stabilize_n': stab,
                     'pred_split_r': pred_s, 'pred_yoy_r': pred_y,
                     'pred_yoy_paxw_r': pred_y2, 'curr_r': curr,
                     'n_split': n_s, 'n_pairs': n_y})
    res = pd.DataFrame(rows)
    res['abs_split'] = res['pred_split_r'].abs()
    res = res.sort_values('abs_split', ascending=False) \
             .drop(columns=['abs_split'])
    res.to_csv(SCREEN_CSV, index=False)
    pd.set_option('display.width', 220)
    print(f'\n════ Pitcher+ screening — target: future xRV/100 '
          f'(S: {len(ab)} split pairs, Y: {len(pairs)} year pairs) ════')
    print(res.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--screen', action='store_true')
    args = ap.parse_args()
    if args.build:
        build()
    if args.screen:
        screen()
