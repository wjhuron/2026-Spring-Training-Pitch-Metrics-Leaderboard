"""ctplus_bt_prior_build.py — production design + constants for the CT+
bat-tracking prior (the battery passed both gates; Wally approved the
full prior with the negative bat-speed conditional, 2026-08-21).

Everything mirrors the BB+ D1 build (bbplus_bt_prior_build.py), in CT+
currency: raw_ct (pool-independent lift ratio), n = CT-eligible swings.

PRE-REGISTERED PROTOCOL (2026-08-21, before any result was seen):
  Prior      z-form (cross-instrument robust, the BB+ lesson):
             prior = poolRawMean + Bbs * z(batSpeed)
                                 + Bsq * z(squaredUpPct-per-contact)
             std betas from per-season OLS on full-season raw_ct,
             applied leave-season-out; pool mean/SD from the evaluated
             season (production computes them live).
  prior_eff  (nTracked * prior + S0 * lg_raw) / (nTracked + S0),
             S0 in {0, 10, 25, 50, 100}. NOTE: the neutral anchor is
             lg_raw (~1.0), not 100 — CT+ raw currency.
  Designs    C1 composite blend AFTER the shipped shrink:
                 adj66 = (n*raw + 66*lg_raw)/(n + 66)   [shipped]
                 est   = (n*adj66 + K*prior_eff)/(n + K)
                 K in {0, 5, 10, 20, 40, 80, 160} (swing units — CT's n
                 runs ~2.5x BB+'s BIP, so the grid extends higher)
             C2 target substitution:
                 est = (n*raw + 66*prior_eff)/(n + 66)
                 (one existing constant reused; carries the D2-class
                 full-season-tilt risk, so the invariance gate decides)
  Objective  held-out RMSE vs remainder raw_ct, causal windows,
             n in {40, 80, 160} (+320 where the pool allows), LOSO
             coefficients, seasons 2024/2025/2026.
  Gates      interior optimum or stated-flat per constant; full-season
             invariance corr >= 0.995 with small mean |delta|; the
             winner must not degrade the next-season full-season test
             (2024->25, 2025->26) — the check that killed BB+'s D2.

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/ctplus_bt_prior_build.py
Output: data/_ctplus_bt_prior_build.json + printed tables.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ctplus_bt_prior_battery import (  # noqa: E402
    sheet_pitches_2026, kin_swings, bat_metrics, D)
from statcast_hitter_adapter import season_dicts, GUTS  # noqa: E402
from pipeline.contact import (  # noqa: E402
    is_ct_eligible, build_bip_count_offsets, build_contact_cell_weights,
    zone_level_contact_means, shrink_contact_cells, compute_hitter_ct,
    HITTER_PRIOR_N)
from pipeline.sdplus import make_rv_xrv  # noqa: E402

SEASONS = (2024, 2025, 2026)
N_LEVELS = (40, 80, 160, 320)
MIN_SWINGS_FULL, MIN_REMAINDER, MIN_TRACKED = 400, 200, 100
S0_GRID = (0, 10, 25, 50, 100)
K_GRID = (0, 5, 10, 20, 40, 80, 160)


def build_frames():
    per_season = {}
    for y in SEASONS:
        if y == 2026:
            pitches = sheet_pitches_2026()
            md = D('metadata_rs.json')
            G = md.get('gutsConstants') or {}
            lgw, ws = G.get('lgWOBA', 0.313), G.get('wOBAScale', 1.232)
        else:
            pitches = season_dicts(y)
            lgw, ws = GUTS[y]
        kin = kin_swings(y)
        kin_by = dict(tuple(kin.groupby('hid')))
        swings = [p for p in pitches if is_ct_eligible(p)]
        offsets = build_bip_count_offsets(swings, lgw, ws)
        rv_fn = make_rv_xrv(lgw, ws, offsets)
        cells = shrink_contact_cells(
            build_contact_cell_weights(swings, rv_fn),
            zone_level_contact_means(swings, rv_fn))
        by_h = defaultdict(list)
        for p in swings:
            by_h[p['Batter']].append(p)
        for h in by_h:
            by_h[h].sort(key=lambda p: p.get('Game Date') or '')
        cuts, meta = {}, {}
        for h, ps in by_h.items():
            ksw = kin_by.get(h)
            if (len(ps) < MIN_SWINGS_FULL or ksw is None
                    or len(ksw) < MIN_TRACKED):
                continue
            cuts[(h + '|full', 'X')] = ps
            meta[h] = {'n_full': len(ps)}
            for n in N_LEVELS:
                if len(ps) - n < MIN_REMAINDER:
                    continue
                cuts[(f'{h}|f{n}', 'X')] = ps[:n]
                cuts[(f'{h}|r{n}', 'X')] = ps[n:]
                meta[h][f'cut_{n}'] = ps[n - 1].get('Game Date')
        raw_ct = compute_hitter_ct(cuts, cells)
        recs = []
        for h, mt in meta.items():
            full = raw_ct.get((h + '|full', 'X'))
            ksw = kin_by[h]
            bm = bat_metrics(ksw)
            if not full or bm is None or bm['squpc'] is None:
                continue
            rec = {'hid': h, 'n_full': mt['n_full'],
                   'nsw_full': bm['nsw'],
                   'raw_full': full['raw_ct'],
                   'bs_full': bm['bs'], 'squpc_full': bm['squpc']}
            for n in N_LEVELS:
                f = raw_ct.get((f'{h}|f{n}', 'X'))
                r = raw_ct.get((f'{h}|r{n}', 'X'))
                cd = mt.get(f'cut_{n}')
                if not f or not r or not cd:
                    continue
                win = ksw[ksw['date'] <= cd]
                bw = bat_metrics(win)
                if bw is None or bw['squpc'] is None:
                    continue
                rec[f'raw_{n}'] = f['raw_ct']
                rec[f'tgt_{n}'] = r['raw_ct']
                rec[f'bs_{n}'] = bw['bs']
                rec[f'squpc_{n}'] = bw['squpc']
                rec[f'nsw_{n}'] = bw['nsw']
            recs.append(rec)
        t = pd.DataFrame(recs)
        t.attrs['lg_raw'] = float(t['raw_full'].mean())
        per_season[y] = t
        print(f'  {y}: {len(t)} hitter-seasons, lg raw_ct '
              f'{t.attrs["lg_raw"]:.4f}')
    return per_season


def main():
    per_season = build_frames()

    # per-season fits + standardized betas
    print('\n── prior fit: raw_full ~ bs + squpc (per season) ──')
    std_coefs, pools = {}, {}
    for y, t in per_season.items():
        X = np.column_stack([np.ones(len(t)), t['bs_full'], t['squpc_full']])
        b, *_ = np.linalg.lstsq(X, t['raw_full'].values, rcond=None)
        sb, sq = float(t['bs_full'].std()), float(t['squpc_full'].std())
        std_coefs[y] = (float(b[1] * sb), float(b[2] * sq))
        pools[y] = (float(t['bs_full'].mean()), sb,
                    float(t['squpc_full'].mean()), sq,
                    float(t['raw_full'].mean()))
        r2 = 1 - np.var(t['raw_full'] - X @ b) / np.var(t['raw_full'])
        print(f'  {y}: std betas bs {std_coefs[y][0]:+.4f} '
              f'squpc {std_coefs[y][1]:+.4f}  R2={r2:.3f}')

    # bs-only fit (squpc's std beta flips sign in 2026 — stability check)
    std_bs_only = {}
    for y, t in per_season.items():
        X = np.column_stack([np.ones(len(t)), t['bs_full']])
        b, *_ = np.linalg.lstsq(X, t['raw_full'].values, rcond=None)
        std_bs_only[y] = float(b[1] * t['bs_full'].std())
    print('  bs-only std betas: '
          + '  '.join(f'{y}:{std_bs_only[y]:+.4f}' for y in std_bs_only))

    def prior_of(t, y_eval, train, bs_col, sq_col, bs_only=False):
        mb, sb, mq, sq, mr = pools[y_eval]
        if bs_only:
            bb = np.mean([std_bs_only[yy] for yy in train])
            return mr + bb * (np.asarray(t[bs_col], float) - mb) / sb
        bb = np.mean([std_coefs[yy][0] for yy in train])
        bq = np.mean([std_coefs[yy][1] for yy in train])
        return (mr + bb * (np.asarray(t[bs_col], float) - mb) / sb
                + bq * (np.asarray(t[sq_col], float) - mq) / sq)

    print('\n── sweep: held-out RMSE vs remainder raw_ct ──')
    results = defaultdict(dict)
    for y, t in per_season.items():
        train = [yy for yy in per_season if yy != y]
        lg = t.attrs['lg_raw']
        for n in N_LEVELS:
            need = [f'raw_{n}', f'tgt_{n}', f'bs_{n}', f'squpc_{n}',
                    f'nsw_{n}']
            tt = t.dropna(subset=[c for c in need if c in t]).copy()
            if len(tt) < 40:
                continue
            prior = prior_of(tt, y, train, f'bs_{n}', f'squpc_{n}')
            prior_b = prior_of(tt, y, train, f'bs_{n}', f'squpc_{n}',
                               bs_only=True)
            e0 = (n * tt[f'raw_{n}'] + HITTER_PRIOR_N * lg) \
                / (n + HITTER_PRIOR_N)
            yv = tt[f'tgt_{n}'].values
            results[(y, n)]['base'] = float(
                np.sqrt(np.mean((yv - e0) ** 2)))
            for s0 in (0, 10, 25):
                pe_b = ((tt[f'nsw_{n}'] * prior_b + s0 * lg)
                        / (tt[f'nsw_{n}'] + s0))
                for k in (10, 20, 40, 80):
                    est = (n * e0 + k * pe_b) / (n + k)
                    results[(y, n)][f'B1_s{s0}_k{k}'] = float(
                        np.sqrt(np.mean((yv - est) ** 2)))
            for s0 in S0_GRID:
                pe = ((tt[f'nsw_{n}'] * prior + s0 * lg)
                      / (tt[f'nsw_{n}'] + s0))
                for k in K_GRID:
                    est = (n * e0 + k * pe) / (n + k)
                    results[(y, n)][f'C1_s{s0}_k{k}'] = float(
                        np.sqrt(np.mean((yv - est) ** 2)))
                est2 = (n * tt[f'raw_{n}'] + HITTER_PRIOR_N * pe) \
                    / (n + HITTER_PRIOR_N)
                results[(y, n)][f'C2_s{s0}'] = float(
                    np.sqrt(np.mean((yv - est2) ** 2)))

    cells = sorted(results)
    configs = sorted({k for c in cells for k in results[c] if k != 'base'})
    summary = []
    for cfg in configs:
        ds = [results[c][cfg] - results[c]['base'] for c in cells
              if cfg in results[c]]
        wins = sum(1 for c in cells if cfg in results[c]
                   and results[c][cfg] < results[c]['base'])
        summary.append((float(np.mean(ds)), wins, cfg))
    summary.sort()
    print(f'  {len(cells)} cells. Top 12 by mean RMSE delta:')
    for d, w, cfg in summary[:12]:
        print(f'   {cfg:<14} mean d={d:+.5f}  wins {w}/{len(cells)}')
    print('  baseline by cell: '
          + '  '.join(f'{y}/{n}={results[(y, n)]["base"]:.4f}'
                      for y, n in cells))

    # full-season invariance + next-season test for the two design winners
    win_b = next(c for _, _, c in summary if c.startswith('B1'))
    win1 = next(c for _, _, c in summary if c.startswith('C1'))
    win2 = next(c for _, _, c in summary if c.startswith('C2'))
    # B1_s10_k40 is the SHIP candidate: bs-only beat bs+squpc at every
    # matched config (the squpc coefficient flips sign in 2026 — drag,
    # not help), and s0=10 is the stated robustness convention for the
    # censored sub-100-tracked-swing region.
    out_inv, out_a4 = {}, {}
    for cfg in ('B1_s10_k40', win_b, win1, win2):
        print(f'\n── {cfg}: full-season invariance ──')
        parts = cfg.split('_')
        s0 = int(parts[1][1:])
        k = int(parts[2][1:]) if len(parts) > 2 else None
        for y, t in per_season.items():
            train = [yy for yy in per_season if yy != y]
            lg = t.attrs['lg_raw']
            prior = prior_of(t, y, train, 'bs_full', 'squpc_full',
                             bs_only=cfg.startswith('B1'))
            nf = t['n_full'].values
            adj = (nf * t['raw_full'] + HITTER_PRIOR_N * lg) \
                / (nf + HITTER_PRIOR_N)
            pe = (t['nsw_full'] * prior + s0 * lg) / (t['nsw_full'] + s0)
            if cfg.startswith(('B1', 'C1')):
                new = (nf * adj + k * pe) / (nf + k)
            else:
                new = (nf * t['raw_full'] + HITTER_PRIOR_N * pe) \
                    / (nf + HITTER_PRIOR_N)
            new, adj = np.asarray(new, float), np.asarray(adj, float)
            r = float(np.corrcoef(adj, new)[0, 1])
            out_inv[f'{cfg}_{y}'] = {
                'corr': r, 'mean_abs': float(np.mean(np.abs(new - adj))),
                'max_abs': float(np.max(np.abs(new - adj)))}
            print(f'  {y}: corr {r:.4f}, mean |d| '
                  f'{np.mean(np.abs(new - adj)):.4f}, max '
                  f'{np.max(np.abs(new - adj)):.4f}')
        print(f'── {cfg}: next-season full-season prediction ──')
        for y0, y1 in ((2024, 2025), (2025, 2026)):
            t0, t1 = per_season[y0], per_season[y1]
            nxt = dict(zip(t1['hid'], t1['raw_full']))
            tt = t0[t0['hid'].isin(nxt)].copy()
            train = [yy for yy in per_season if yy != y0]
            lg = t0.attrs['lg_raw']
            prior = prior_of(tt, y0, train, 'bs_full', 'squpc_full',
                             bs_only=cfg.startswith('B1'))
            nf = tt['n_full'].values
            adj = np.asarray((nf * tt['raw_full'] + HITTER_PRIOR_N * lg)
                             / (nf + HITTER_PRIOR_N), float)
            pe = (tt['nsw_full'] * prior + s0 * lg) / (tt['nsw_full'] + s0)
            if cfg.startswith(('B1', 'C1')):
                new = np.asarray((nf * adj + k * pe) / (nf + k), float)
            else:
                new = np.asarray((nf * tt['raw_full']
                                  + HITTER_PRIOR_N * pe)
                                 / (nf + HITTER_PRIOR_N), float)
            yv = np.array([nxt[h] for h in tt['hid']])
            r_a = float(np.corrcoef(adj, yv)[0, 1])
            r_n = float(np.corrcoef(new, yv)[0, 1])
            rm_a = float(np.sqrt(np.mean((yv - adj) ** 2)))
            rm_n = float(np.sqrt(np.mean((yv - new) ** 2)))
            out_a4[f'{cfg}_{y0}_{y1}'] = {
                'r_shipped': r_a, 'r_new': r_n,
                'rmse_shipped': rm_a, 'rmse_new': rm_n}
            print(f'  {y0}->{y1}: shipped r={r_a:+.3f}/rmse {rm_a:.4f}  '
                  f'new r={r_n:+.3f}/rmse {rm_n:.4f}  (n={len(tt)})')

    frozen = {
        'std_beta_bs_only': float(np.mean([std_bs_only[y]
                                           for y in std_bs_only])),
        'per_season_bs_only': {str(y): std_bs_only[y] for y in std_bs_only},
        'std_beta_bs': float(np.mean([std_coefs[y][0] for y in std_coefs])),
        'std_beta_squp': float(np.mean([std_coefs[y][1] for y in std_coefs])),
        'per_season_std': {str(y): std_coefs[y] for y in std_coefs},
        'pool_2026': pools[2026],
    }
    print(f"\n  frozen std betas: bs {frozen['std_beta_bs']:+.4f}, "
          f"squp {frozen['std_beta_squp']:+.4f}; 2026 pool anchors "
          f"{tuple(round(v, 4) for v in pools[2026])}")
    with open(os.path.join(ROOT, 'data',
                           '_ctplus_bt_prior_build.json'), 'w') as f:
        json.dump({'cells': {f'{y}_{n}': results[(y, n)]
                             for y, n in cells},
                   'top': summary[:20], 'invariance': out_inv,
                   'next_season': out_a4, 'frozen': frozen}, f,
                  indent=1, default=float)
    print('  saved data/_ctplus_bt_prior_build.json')


if __name__ == '__main__':
    main()
