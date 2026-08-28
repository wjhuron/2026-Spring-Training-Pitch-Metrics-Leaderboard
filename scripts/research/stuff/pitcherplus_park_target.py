#!/usr/bin/env python3
"""pitcherplus_park_target.py — is park-adjusted future xRV/100 a better
Pitcher+ target? (Approved by Wally 2026-08-28.)

The shipped Pitcher+ predicts RAW future xRV/100, which carries each
pitcher's future park exposure — variance that is real but is not the
pitcher. This measures the season-level park pass-through into
pitcher-season xRV/100, builds the adjusted quantity, and answers:

  1. pass-through  WLS pitcher-season xrv100 ~ pfdev exposure (weight n),
                   per season + pooled; pfdev = pitch-weighted mean of
                   (PF/100 - 1) over the games the pitcher appeared in
  2. target race   frozen shipped composite -> raw vs adjusted NEXT-season
                   xRV/100 (panel Y). NOTE an r gain here can be purely
                   mechanical (the adjusted target is less noisy for every
                   predictor), so the race also runs the kwERA benchmark.
  3. input race    adjust the xrv100 COMPONENT the same way, refit vs
                   frozen, and check whether weights move beyond noise.

Usage: PYTHONHASHSEED=0 python3 scripts/research/stuff/pitcherplus_park_target.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, 'data')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'misc'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import leaderboard_metric_battery as bat  # noqa: E402
import pitcherplus_search as ps  # noqa: E402

SHIPPED = (
    ('stuffRaw',   0.20, 42.0),
    ('locRaw',     0.06, 215.0),
    ('kPct',       0.21, 398.0),
    ('izWhiffPct', 0.19, 421.0),
    ('xrv100',     0.23, 1046.0),
    ('gbPct',      0.12, 333.0),
)
MIN_FULL, Q_FULL = 800, 800
SEASONS = [2021, 2022, 2023, 2024, 2025]
EXPOSURE_PKL = os.path.join(DATA, '_pplus_park_exposure.pkl')


def build_exposure():
    """(pid, season) -> pitch-weighted pfdev over his games."""
    if os.path.exists(EXPOSURE_PKL):
        return pickle.load(open(EXPOSURE_PKL, 'rb'))
    pf = json.load(open(os.path.join(DATA, 'park_factors.json')))
    gh = json.load(open(os.path.join(DATA, '_mlb_gamepk_home.json')))
    rows = []
    for y in SEASONS:
        df = bat.load_season(y)[['pitcher', 'game_pk']]
        home = df['game_pk'].astype(int).astype(str).map(gh)
        dev = pd.Series([
            pf.get(str(y), {}).get(str(int(h)), 100.0) / 100.0 - 1.0
            if pd.notna(h) else np.nan for h in home], index=df.index)
        g = pd.DataFrame({'pid': df['pitcher'], 'pfdev': dev}) \
            .groupby('pid')['pfdev'].mean().reset_index()
        g['season'] = y
        rows.append(g)
        print(f'  {y}: exposure for {len(g)} pitchers', flush=True)
    out = pd.concat(rows, ignore_index=True)
    pickle.dump(out, open(EXPOSURE_PKL, 'wb'))
    return out


def wls_slope(x, y, w):
    m = np.isfinite(x) & np.isfinite(y)
    x, y, w = x[m], y[m], w[m]
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    return float(np.sum(w * (x - xm) * (y - ym))
                 / np.sum(w * (x - xm) ** 2)), len(x)


def main():
    exp = build_exposure()
    t = pickle.load(open(ps.TABLES_PKL, 'rb'))
    for path, cols in ((ps.LOC_CSV, ['locRaw']), (ps.STUFF_CSV, ['stuffRaw'])):
        ext = pd.read_csv(path)
        t = t.merge(ext[['pid', 'season', 'half'] + cols],
                    on=['pid', 'season', 'half'], how='left')
    feats = [f for f, _w, _k in SHIPPED]
    for c in feats + ['xrv100', 'kbbPct']:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    full = t[(t['half'] == 'full') & (t['n'] >= MIN_FULL)].copy()
    full = full.merge(exp, on=['pid', 'season'], how='left')
    print(f'{len(full)} pitcher-seasons, exposure matched '
          f'{full.pfdev.notna().mean():.4f}, pfdev spread '
          f'{full.pfdev.std():.4f} (min {full.pfdev.min():+.3f} '
          f'max {full.pfdev.max():+.3f})')

    print('\n══ season-level pass-through: xrv100 ~ pfdev (WLS) ══')
    slopes = []
    for season, g in full.groupby('season'):
        b, n = wls_slope(g['pfdev'].to_numpy(float),
                         g['xrv100'].to_numpy(float), g['n'].to_numpy(float))
        slopes.append(b)
        print(f'  {season}: b = {b:+.3f} (n {n})')
    b_pool, _ = wls_slope(full['pfdev'].to_numpy(float),
                          full['xrv100'].to_numpy(float),
                          full['n'].to_numpy(float))
    print(f'  pooled: b = {b_pool:+.3f}; season spread {np.std(slopes):.3f}')

    full['xrv100_adj'] = full['xrv100'] - b_pool * full['pfdev']

    # ── composites (shrunk z per season, frozen weights), raw and
    #    adjusted-input variants ──
    def composite(g_all, xcol):
        out = np.zeros(len(g_all))
        for season, g in g_all.groupby('season'):
            q = g[g['n'] >= Q_FULL]
            seas = np.zeros(len(g))
            for f, w, k in SHIPPED:
                col = xcol if f == 'xrv100' else f
                vals = q[col].dropna()
                mu = float(np.average(vals, weights=q.loc[vals.index, 'n']))
                sd = float(vals.std())
                z = ((g[col] - mu) / sd).fillna(0.0)
                seas += w * (z * (g['n'] / (g['n'] + k))).to_numpy()
            out[g_all.index.get_indexer(g.index)] = seas
        return out

    full['comp_raw'] = composite(full, 'xrv100')
    full['comp_adj'] = composite(full, 'xrv100_adj')
    # kbbPct benchmark z
    for season, g in full.groupby('season'):
        q = g[g['n'] >= Q_FULL]
        mu = float(np.average(q['kbbPct'].dropna(),
                              weights=q.loc[q['kbbPct'].notna(), 'n']))
        sd = float(q['kbbPct'].std())
        full.loc[g.index, 'kbb_z'] = ((g['kbbPct'] - mu) / sd).fillna(0.0)

    pairs = full.merge(
        full[['pid', 'season', 'xrv100', 'xrv100_adj']].assign(
            season=lambda d: d['season'] - 1),
        on=['pid', 'season'], suffixes=('', '_n1'))
    print(f'\npanel Y: {len(pairs)} year-pairs')

    print('\n══ target race (Pearson r, panel Y) ══')
    print(f'{"predictor":22s} {"-> raw next":>12s} {"-> adj next":>12s}')
    for pred, lab in (('comp_raw', 'frozen composite'),
                      ('comp_adj', 'composite, adj input'),
                      ('kbb_z', 'kwERA-core (bench)')):
        row = []
        for tgt in ('xrv100_n1', 'xrv100_adj_n1'):
            a = pairs[pred].to_numpy(float)
            b = pairs[tgt].to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b)
            row.append(np.corrcoef(a[m], b[m])[0, 1])
        print(f'{lab:22s} {row[0]:12.4f} {row[1]:12.4f}')

    # per-season fold wins: adj-input composite vs frozen, on the adj target
    print('\nper-season folds (target = adjusted next xRV/100):')
    wins = 0
    for season, g in pairs.groupby('season'):
        rr = np.corrcoef(g['comp_raw'], g['xrv100_adj_n1'])[0, 1]
        ra = np.corrcoef(g['comp_adj'], g['xrv100_adj_n1'])[0, 1]
        wins += ra > rr
        print(f'  {season}: raw-input {rr:.4f}  adj-input {ra:.4f}')
    print(f'adj-input wins {wins}/{pairs.season.nunique()} seasons')

    # refit weights on the adjusted target (full fit, normalized) vs frozen
    def refit(xcol, tgt):
        cols = []
        for f, _w, k in SHIPPED:
            col = xcol if f == 'xrv100' else f
            z = np.zeros(len(pairs))
            for season, g in pairs.groupby('season'):
                vals = g[col].dropna()
                mu, sd = float(vals.mean()), float(vals.std())
                zz = ((g[col] - mu) / sd).fillna(0.0)
                z[pairs.index.get_indexer(g.index)] = \
                    (zz * (g['n'] / (g['n'] + k))).to_numpy()
            cols.append(z)
        X = np.column_stack([np.ones(len(pairs))] + cols)
        y = pairs[tgt].to_numpy(float)
        m = np.isfinite(y)
        beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        w = beta[1:] / np.abs(beta[1:]).sum()
        return {f: round(float(v), 3) for (f, _w2, _k), v in zip(SHIPPED, w)}

    print('\nrefit weights (normalized):')
    print('  raw target, raw input: ', refit('xrv100', 'xrv100_n1'))
    print('  adj target, adj input: ', refit('xrv100_adj', 'xrv100_adj_n1'))


if __name__ == '__main__':
    main()
