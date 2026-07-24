#!/usr/bin/env python3
"""pitcherplus_v2_candidates.py — do sequencing and stamina add anything?

RESEARCH ONLY (2026-07-24). Evaluates the two Pitcher+ phase-2 ideas that
were never built, plus a handful of neighbours, against the SAME dual-panel
protocol the shipped metric used. Nothing here is wired to production.

Two families:

  MIX / SEQUENCING  (is he predictable?)
    condEntropy      mean H(pitch type | count), frequency-weighted.
                     Low = predictable.
    entropyBehind    H(pitch type) restricted to hitter counts (balls >
                     strikes) — the counts where predictability is
                     punished. entropyAhead is the pitcher-count twin.
    repeatRate       P(same type as the previous pitch of the PA).
    transEntropy     H(next type | previous type), first-order Markov.
    fbAfterFb        P(fastball | previous was a fastball).
    mixOptimality    usage-weighted covariance between a pitch type's
                     usage share and its xRV/100 — does he throw his best
                     pitches the most?

  STAMINA / STUFF RETENTION  (does he hold it?)
    veloSlope        OLS slope of VELO RESIDUAL (velo minus his own
                     season mean for that pitch type, so a late breaking
                     ball doesn't read as decay) on pitch index within
                     outing, mph per 100 pitches. Negative = fades.
    veloTTO          mean velo residual at TTO3+ minus TTO1.
    xrvSlope         same regression on per-pitch xRV, runs/100 per 100
                     pitches.
    xrvTTO           xRV/100 at TTO1 minus at TTO3+ (classic TTO penalty;
                     positive = fades).
    earlyLateXrv     xRV/100 on outing pitches 1-25 minus pitches 60+.

Pitch order: the season caches store Savant's native DESCENDING order, so
reversing within game_pk recovers chronological order. Verified on 2024:
within-PA ball+strike counts are non-decreasing 100.0% of transitions
(90.6% exactly +1, the remainder 2-strike fouls).

Reported per candidate: split-half reliability, stabilization n, both
panels' predictive r vs FUTURE xRV/100, and — the question that actually
matters — the MARGINAL out-of-fold gain when bolted onto the frozen
six-term Pitcher+.

Usage:
  python3 scripts/pitcherplus_v2_candidates.py --build    (~15-30 min)
  python3 scripts/pitcherplus_v2_candidates.py --screen
Outputs: data/_pplus_v2_tables.pkl, data/_pplus_v2_results.csv
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import leaderboard_metric_battery as bat      # noqa: E402
import pitcherplus_search as ps               # noqa: E402
import pitcherplus_combo as pc                # noqa: E402

V2_PKL = os.path.join(ROOT, 'data', '_pplus_v2_tables.pkl')
V2_CSV = os.path.join(ROOT, 'data', '_pplus_v2_results.csv')
SEASONS = [2021, 2022, 2023, 2024, 2025]
FB = {'FF', 'SI', 'FA', 'FC'}

MIX_FEATS = ['condEntropy', 'entropyBehind', 'entropyAhead', 'repeatRate',
             'transEntropy', 'fbAfterFb', 'mixOptimality']
STAM_FEATS = ['veloSlope', 'veloTTO', 'xrvSlope', 'xrvTTO', 'earlyLateXrv']
V2_FEATS = MIX_FEATS + STAM_FEATS


def chronological(df):
    """Savant order is descending within game — reverse to real pitch order,
    then stamp outing pitch index, PA id and times-through-order."""
    df = df.reset_index(drop=True)
    df['_ord'] = df.groupby('game_pk').cumcount(ascending=False)
    d = df.sort_values(['game_pk', '_ord']).reset_index(drop=True)
    d['pitchIdx'] = d.groupby(['game_pk', 'pitcher']).cumcount()
    # PA id: a new PA starts after every pa_end within the game
    d['_paid'] = (d.groupby('game_pk')['pa_end']
                  .transform(lambda s: s.shift(fill_value=False).cumsum()))
    # times through order = how many PAs this pitcher has already had vs
    # this batter in this game
    pa_first = d.drop_duplicates(['game_pk', 'pitcher', '_paid'])
    pa_first = pa_first[['game_pk', 'pitcher', '_paid', 'batter']].copy()
    pa_first['tto'] = pa_first.groupby(
        ['game_pk', 'pitcher', 'batter']).cumcount() + 1
    d = d.merge(pa_first[['game_pk', 'pitcher', '_paid', 'tto']],
                on=['game_pk', 'pitcher', '_paid'], how='left')
    # velo residual vs the pitcher's own season mean for that pitch type
    key = ['pitcher', 'pitch_type']
    d['veloResid'] = d['release_speed'] - d.groupby(key)['release_speed'] \
        .transform('mean')
    # previous pitch type within the same PA
    same_pa = ((d['pitcher'] == d['pitcher'].shift())
               & (d['_paid'] == d['_paid'].shift())
               & (d['game_pk'] == d['game_pk'].shift()))
    d['prevType'] = d['pitch_type'].shift().where(same_pa)
    return d


def _entropy(counts):
    tot = float(sum(counts))
    if tot <= 0:
        return np.nan
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / tot
            h -= p * math.log(p)
    return h


def _slope(x, y):
    """OLS slope, nan-safe."""
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return np.nan
    x, y = x[m], y[m]
    vx = ((x - x.mean()) ** 2).sum()
    if vx <= 1e-9:
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).sum() / vx)


def v2_aggs(g):
    # Fixed key order for EVERY group: pandas stacks groupby.apply results
    # into a long Series instead of a DataFrame when the returned Series'
    # index order varies between groups (filling gaps at the end with
    # setdefault silently did exactly that).
    out = {'n': len(g)}
    for f in V2_FEATS:
        out[f] = np.nan
    pt = g['pitch_type'].dropna()

    # ── mix / sequencing ──
    if len(pt) >= 100:
        # H(type | count), weighted by count frequency
        num, den = 0.0, 0
        for _c, gc in g.groupby(['balls', 'strikes']):
            v = gc['pitch_type'].dropna()
            if len(v) >= 25:
                h = _entropy(v.value_counts().values)
                if h == h:
                    num += h * len(v)
                    den += len(v)
        out['condEntropy'] = num / den if den else np.nan
        beh = g[g['balls'] > g['strikes']]['pitch_type'].dropna()
        ahd = g[g['strikes'] > g['balls']]['pitch_type'].dropna()
        out['entropyBehind'] = (_entropy(beh.value_counts().values)
                                if len(beh) >= 50 else np.nan)
        out['entropyAhead'] = (_entropy(ahd.value_counts().values)
                               if len(ahd) >= 50 else np.nan)
        pv = g[g['prevType'].notna() & g['pitch_type'].notna()]
        if len(pv) >= 100:
            out['repeatRate'] = float(
                (pv['pitch_type'] == pv['prevType']).mean())
            num2, den2 = 0.0, 0
            for _p, gp in pv.groupby('prevType'):
                if len(gp) >= 25:
                    h = _entropy(gp['pitch_type'].value_counts().values)
                    if h == h:
                        num2 += h * len(gp)
                        den2 += len(gp)
            out['transEntropy'] = num2 / den2 if den2 else np.nan
            fbprev = pv[pv['prevType'].isin(FB)]
            out['fbAfterFb'] = (float(fbprev['pitch_type'].isin(FB).mean())
                                if len(fbprev) >= 50 else np.nan)
        # usage-optimality: covariance of usage share with per-type xRV/100
        shares, vals = [], []
        for t, gt in g.groupby('pitch_type'):
            if len(gt) >= 50:
                v = gt['xrv_pitch'].dropna()
                if len(v) >= 30:
                    shares.append(len(gt) / len(g))
                    vals.append(v.sum() / len(gt) * 100)
        if len(shares) >= 3:
            s, v = np.array(shares), np.array(vals)
            out['mixOptimality'] = float(((s - s.mean())
                                          * (v - v.mean())).mean())

    # ── stamina / retention ──
    idx = g['pitchIdx'].to_numpy(float)
    out['veloSlope'] = _slope(idx, g['veloResid'].to_numpy(float)) * 100 \
        if len(g) >= 200 else np.nan
    out['xrvSlope'] = _slope(idx, g['xrv_pitch'].to_numpy(float)) * 100 * 100 \
        if len(g) >= 200 else np.nan
    t1 = g[g['tto'] == 1]
    t3 = g[g['tto'] >= 3]
    if len(t1) >= 100 and len(t3) >= 100:
        out['veloTTO'] = float(t3['veloResid'].mean()
                               - t1['veloResid'].mean())
        x1 = t1['xrv_pitch'].dropna()
        x3 = t3['xrv_pitch'].dropna()
        if len(x1) >= 50 and len(x3) >= 50:
            out['xrvTTO'] = float(x1.sum() / len(t1) * 100
                                  - x3.sum() / len(t3) * 100)
    early = g[g['pitchIdx'] < 25]
    late = g[g['pitchIdx'] >= 60]
    if len(early) >= 100 and len(late) >= 100:
        e = early['xrv_pitch'].dropna()
        l_ = late['xrv_pitch'].dropna()
        if len(e) >= 50 and len(l_) >= 50:
            out['earlyLateXrv'] = float(e.sum() / len(early) * 100
                                        - l_.sum() / len(late) * 100)
    return pd.Series(out)


def build():
    rows = []
    for year in SEASONS:
        print(f'── {year}', flush=True)
        df = bat.load_season(year)
        df = ps.add_extra_flags(df)
        df = ps.add_xrv(df, year)
        d = chronological(df)
        d = bat.add_half(d, 'pitcher')
        for half_label, sub in (('full', d), ('A', d[d['half'] == 'A']),
                                ('B', d[d['half'] == 'B'])):
            r = sub.groupby('pitcher', group_keys=False).apply(v2_aggs)
            r['season'] = year
            r['half'] = half_label
            r.index.name = 'pid'
            rows.append(r.reset_index())
        print(f'   {len(rows[-3])} pitchers', flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_pickle(V2_PKL)
    print(f'saved {V2_PKL} ({len(out)} rows)')


# ── screening + marginal value over the shipped Pitcher+ ────────────────
def screen():
    v2 = pd.read_pickle(V2_PKL)
    base = pc.load_tables()
    t = base.merge(v2[['pid', 'season', 'half'] + V2_FEATS],
                   on=['pid', 'season', 'half'], how='left')
    feats = [c for c in t.columns if c not in ps.META]
    for c in feats:
        t[c] = pd.to_numeric(t[c], errors='coerce')

    # 1. univariate screening, same protocol as phase 1
    full = t[(t['half'] == 'full') & (t['n'] >= ps.MIN_FULL)]
    A = t[(t['half'] == 'A') & (t['n'] >= ps.MIN_HALF)]
    B = t[(t['half'] == 'B') & (t['n'] >= ps.MIN_HALF)]
    ab = A.merge(B, on=['pid', 'season'], suffixes=('_a', '_b'))
    pairs = full.merge(full.assign(season=full['season'] - 1),
                       on=['pid', 'season'], suffixes=('', '_n1'))
    rows = []
    for m in V2_FEATS:
        sh, n_sh = bat.pear(ab[m + '_a'], ab[m + '_b'])
        half_n = float(np.nanmean(np.minimum(ab['n_a'], ab['n_b'])))
        stab = half_n * (1 - sh) / sh if sh and sh > 0 else np.nan
        pa_ = np.concatenate([pd.to_numeric(ab[m + '_a'], errors='coerce'),
                              pd.to_numeric(ab[m + '_b'], errors='coerce')])
        tb = np.concatenate([pd.to_numeric(ab['xrv100_b'], errors='coerce'),
                             pd.to_numeric(ab['xrv100_a'], errors='coerce')])
        pred_s, n_s = bat.pear(pa_, tb)
        pred_y, n_y = bat.pear(pairs[m], pairs['xrv100_n1'])
        cov = float(pd.to_numeric(full[m], errors='coerce').notna().mean())
        rows.append({'feature': m, 'coverage': cov, 'reliability_r': sh,
                     'stabilize_n': stab, 'pred_split_r': pred_s,
                     'pred_yoy_r': pred_y, 'n_split': n_s, 'n_pairs': n_y})
    res = pd.DataFrame(rows)

    # 2. marginal OOF gain on top of the frozen six-term Pitcher+
    kmap = pc.stab_constants()
    for f in V2_FEATS:
        scr = res[res.feature == f]['stabilize_n'].iloc[0]
        kmap[f] = float(scr) if scr == scr and scr > 0 else 1000.0
    allf = pc.SURVIVORS + V2_FEATS
    _orig = pc.SURVIVORS
    pc.SURVIVORS = allf
    try:
        (S_X, S_y, S_g), (Y_X, Y_y, Y_g) = pc.build_panels(t, kmap)
    finally:
        pc.SURVIVORS = _orig
    SHIPPED = ['stuffRaw', 'locRaw', 'kPct', 'izWhiffPct', 'xrv100', 'gbPct']
    base_cols = [allf.index(c) for c in SHIPPED]
    r_s0, _ = pc.oof_r(S_X, S_y, S_g, base_cols)
    r_y0, _ = pc.oof_r(Y_X, Y_y, Y_g, base_cols)
    b0 = (r_s0 + r_y0) / 2
    print(f'\nfrozen Pitcher+ baseline: S {r_s0:.4f}  Y {r_y0:.4f}  '
          f'combined {b0:.4f}')
    marg = []
    for f in V2_FEATS:
        cols = base_cols + [allf.index(f)]
        r_s, _ = pc.oof_r(S_X, S_y, S_g, cols)
        r_y, _ = pc.oof_r(Y_X, Y_y, Y_g, cols)
        marg.append({'feature': f, 'r_S': r_s, 'r_Y': r_y,
                     'combined': (r_s + r_y) / 2,
                     'gain': (r_s + r_y) / 2 - b0})
    mg = pd.DataFrame(marg)
    # all v2 features at once
    cols_all = base_cols + [allf.index(f) for f in V2_FEATS]
    r_sa, _ = pc.oof_r(S_X, S_y, S_g, cols_all)
    r_ya, _ = pc.oof_r(Y_X, Y_y, Y_g, cols_all)
    out = res.merge(mg, on='feature')
    out.to_csv(V2_CSV, index=False)
    pd.set_option('display.width', 220)
    print('\n════ v2 candidates — univariate + marginal over Pitcher+ ════')
    print(out.sort_values('gain', ascending=False).round(4).to_string(index=False))
    print(f'\nALL v2 features added at once: S {r_sa:.4f}  Y {r_ya:.4f}  '
          f'combined {(r_sa + r_ya) / 2:.4f}  (gain {(r_sa + r_ya) / 2 - b0:+.4f})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--screen', action='store_true')
    args = ap.parse_args()
    if args.build:
        build()
    if args.screen:
        screen()
