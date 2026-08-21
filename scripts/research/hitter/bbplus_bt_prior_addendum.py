"""bbplus_bt_prior_addendum.py — the four checks the build left open.

A1  blastPct as a third predictor (promised in the build docstring,
    never run): LOSO gain over bs+squp.
A2  Winner-config consistency: is D1_s10_k20's optimum stable per season,
    or driven by one?
A3  n0_con + the finalists under a FULL-RANGE objective: n levels
    extended to {20,40,80,160,320} so large-sample behavior is priced —
    the check the small-n-only objective could not perform.
A4  THE SKILL QUESTION, measured: at FULL season, which estimator best
    predicts NEXT season's unshrunk batted-ball quality (the operational
    definition of true skill)? shipped vs D1-full vs D2-full vs the pure
    bat prior, year-pairs 2024->2025 and 2025->2026. If D2's full-season
    tilt toward the swing is skill-signal, it wins here; if it is
    distortion, it loses.

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/bbplus_bt_prior_addendum.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bbplus_bt_prior_build import (  # noqa: E402
    load_events_cache, load_events_sheet, bat_metrics, raw_parts,
    shipped_raw, W_CON, W_EV, MIN_SWINGS_FULL)

SEASONS = (2024, 2025, 2026)
N_LEVELS_X = (20, 40, 80, 160, 320)
MIN_BIP, MIN_REMAINDER = 150, 100
BETA_FROZEN, BETA_BAND = 4.205, (3.0, 6.0)


def build_frames():
    per_season = {}
    for y in SEASONS:
        swings, bips = (load_events_sheet() if y == 2026
                        else load_events_cache(y))
        rows = []
        for hid, g in bips.groupby('hid'):
            if len(g) < MIN_BIP:
                continue
            sw = swings[swings['hid'] == hid]
            if len(sw) < MIN_SWINGS_FULL:
                continue
            rows.append((hid, g, sw))
        lg_con = float(np.nanmean(
            np.concatenate([g['xw'].values for _, g, _ in rows])))
        ev95s = np.array([np.percentile(g['ev'], 95) for _, g, _ in rows])
        wts = np.array([len(g) for _, g, _ in rows], float)
        lg_ev95 = float(np.average(ev95s, weights=wts))
        cons = np.array([100 * np.nanmean(g['xw']) / lg_con
                         for _, g, _ in rows])
        beta = float(np.std(cons) / np.std(100.0 * ev95s / lg_ev95))
        if not (BETA_BAND[0] <= beta <= BETA_BAND[1]):
            beta = BETA_FROZEN
        recs = []
        for hid, g, sw in rows:
            bm = bat_metrics(sw)
            if bm is None or bm['squp'] is None:
                continue
            con_f, evc_f = raw_parts(g['xw'].values, g['ev'].values,
                                     lg_con, lg_ev95, beta)
            rec = {'hid': hid, 'nbip_full': len(g), 'nsw_full': bm['nsw'],
                   'bs': bm['bs'], 'squp': bm['squp'], 'blast': bm['blast'],
                   'con_full': con_f, 'evc_full': evc_f,
                   'raw_full': W_CON * con_f + W_EV * evc_f}
            for n in N_LEVELS_X:
                if len(g) - n < MIN_REMAINDER:
                    continue
                first, rest = g.iloc[:n], g.iloc[n:]
                cutdate = first['date'].iloc[-1]
                w = sw[sw['date'] <= cutdate]
                bmw = bat_metrics(w)
                con_n, evc_n = raw_parts(first['xw'].values,
                                         first['ev'].values,
                                         lg_con, lg_ev95, beta)
                con_r, evc_r = raw_parts(rest['xw'].values, rest['ev'].values,
                                         lg_con, lg_ev95, beta)
                rec[f'con_{n}'], rec[f'evc_{n}'] = con_n, evc_n
                rec[f'tgt_{n}'] = W_CON * con_r + W_EV * evc_r
                rec[f'bs_{n}'] = bmw['bs'] if bmw else np.nan
                rec[f'squp_{n}'] = (bmw['squp'] if bmw
                                    and bmw['squp'] is not None else np.nan)
                rec[f'nsw_{n}'] = bmw['nsw'] if bmw else 0
            recs.append(rec)
        per_season[y] = pd.DataFrame(recs)
        print(f'  {y}: {len(recs)} rows, beta {beta:.2f}')
    return per_season


def std_prior_factory(per_season, feats):
    fits = {}
    for y, t in per_season.items():
        X = np.column_stack([np.ones(len(t))] + [t[f] for f in feats])
        b, *_ = np.linalg.lstsq(X, t['raw_full'].values, rcond=None)
        stds = [float(t[f].std()) for f in feats]
        fits[y] = ([float(b[i + 1] * stds[i]) for i in range(len(feats))],
                   {f: (float(t[f].mean()), float(t[f].std()))
                    for f in feats}, float(t['raw_full'].mean()))
    def prior(t, y_eval, train, cols):
        betas = np.mean([fits[yy][0] for yy in train], axis=0)
        pool, mr = fits[y_eval][1], fits[y_eval][2]
        v = np.full(len(t), mr)
        for i, f in enumerate(feats):
            m, s = pool[f]
            v = v + betas[i] * (np.asarray(t[cols[i]], float) - m) / s
        return v
    return prior, fits


def main():
    per_season = build_frames()

    # A1: blast as third predictor — LOSO R2 of the prior itself
    print('\n── A1: predictor set (LOSO R2 of the prior vs raw_full) ──')
    for feats in (['bs'], ['bs', 'squp'], ['bs', 'squp', 'blast']):
        prior, _ = std_prior_factory(per_season, feats)
        r2s = []
        for y, t in per_season.items():
            train = [yy for yy in SEASONS if yy != y]
            p = prior(t, y, train, feats)
            r2s.append(1 - np.var(t['raw_full'] - p) / np.var(t['raw_full']))
        print(f'  {"+".join(feats):<14} LOSO R2 ' +
              ' '.join(f'{y}:{r:.3f}' for y, r in zip(SEASONS, r2s)))

    prior, _ = std_prior_factory(per_season, ['bs', 'squp'])

    # A2 + A3: config performance per season and at extended n
    print('\n── A2/A3: full-range objective (n up to 320) ──')
    K_GRID = (0, 5, 10, 20, 40, 80)
    for y, t in per_season.items():
        train = [yy for yy in SEASONS if yy != y]
        line = []
        for k in K_GRID:
            ds = []
            for n in N_LEVELS_X:
                need = [f'con_{n}', f'evc_{n}', f'tgt_{n}', f'bs_{n}',
                        f'squp_{n}', f'nsw_{n}']
                tt = t.dropna(subset=[c for c in need if c in t]).copy()
                if len(tt) < 40:
                    continue
                pr = prior(tt, y, train, [f'bs_{n}', f'squp_{n}'])
                pe = ((tt[f'nsw_{n}'] * pr + 10 * 100.0)
                      / (tt[f'nsw_{n}'] + 10))
                e0 = shipped_raw(tt[f'con_{n}'], tt[f'evc_{n}'], n)
                est = (n * e0 + k * pe) / (n + k)
                yv = tt[f'tgt_{n}'].values
                base = float(np.sqrt(np.mean((yv - e0) ** 2)))
                ds.append(float(np.sqrt(np.mean((yv - est) ** 2))) - base)
            line.append(f'k{k}:{np.mean(ds):+.3f}')
        print(f'  D1 s10 {y}: ' + '  '.join(line))
    # D2 at extended n
    for y, t in per_season.items():
        train = [yy for yy in SEASONS if yy != y]
        line = []
        for n in N_LEVELS_X:
            need = [f'con_{n}', f'evc_{n}', f'tgt_{n}', f'bs_{n}',
                    f'squp_{n}', f'nsw_{n}']
            tt = t.dropna(subset=[c for c in need if c in t]).copy()
            if len(tt) < 40:
                continue
            pr = prior(tt, y, train, [f'bs_{n}', f'squp_{n}'])
            pe = ((tt[f'nsw_{n}'] * pr + 50 * 100.0) / (tt[f'nsw_{n}'] + 50))
            e0 = shipped_raw(tt[f'con_{n}'], tt[f'evc_{n}'], n)
            d2 = shipped_raw(tt[f'con_{n}'], tt[f'evc_{n}'], n,
                             tgt_con=pe, tgt_ev=pe, n0_ev=20)
            yv = tt[f'tgt_{n}'].values
            base = float(np.sqrt(np.mean((yv - e0) ** 2)))
            line.append(f'n{n}:{float(np.sqrt(np.mean((yv - d2) ** 2))) - base:+.3f}')
        print(f'  D2 s50e20 {y}: ' + '  '.join(line))

    # A4: full-season -> next-season skill test
    print('\n── A4: next-season prediction from FULL-season estimators ──')
    for y0, y1 in ((2024, 2025), (2025, 2026)):
        t0, t1 = per_season[y0], per_season[y1]
        nxt = dict(zip(t1['hid'], t1['raw_full']))
        tt = t0[t0['hid'].isin(nxt)].copy()
        if len(tt) < 40:
            continue
        train = [yy for yy in SEASONS if yy != y0]
        pr = prior(tt, y0, train, ['bs', 'squp'])
        nb = tt['nbip_full'].values
        shipped_f = np.asarray(shipped_raw(tt['con_full'], tt['evc_full'],
                                           nb), float)
        pe10 = (tt['nsw_full'] * pr + 10 * 100.0) / (tt['nsw_full'] + 10)
        d1 = (nb * shipped_f + 20 * np.asarray(pe10, float)) / (nb + 20)
        pe50 = (tt['nsw_full'] * pr + 50 * 100.0) / (tt['nsw_full'] + 50)
        d2 = np.asarray(shipped_raw(tt['con_full'], tt['evc_full'], nb,
                                    tgt_con=pe50, tgt_ev=pe50, n0_ev=20),
                        float)
        yv = np.array([nxt[h] for h in tt['hid']])
        for name, est in (('shipped', shipped_f), ('D1', d1), ('D2', d2),
                          ('prior alone', np.asarray(pr, float))):
            r = float(np.corrcoef(est, yv)[0, 1])
            rmse = float(np.sqrt(np.mean((yv - est) ** 2)))
            print(f'  {y0}->{y1} {name:<12} r={r:+.3f}  rmse={rmse:.2f}  '
                  f'(n={len(tt)})')


if __name__ == '__main__':
    main()
