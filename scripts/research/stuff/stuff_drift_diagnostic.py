#!/usr/bin/env python3
"""stuff_drift_diagnostic.py — Goodhart/compression check for Stuff+
(after Andrews, FanGraphs 2026-01-20: FG Stuff+ pitcher SD fell 9.7 -> 8.8
and grade-to-wOBA correlation fell in 2024-25).

Uses the gate-v2 SHIPPED out-of-pair predictions (season Y scored by a
model that never saw Y or Y+1) so every season is measured by a model of
the same vintage rule, in one fixed currency: raw target runs per 100
pitches. 2026 is scored by a model fit on 2021-2024.

Per season: pitcher-level SD of mean prediction (>=300 pitches), pitch-level
SD, pitcher-level r(pred, -target) and r(pred, actual RV), and the
fastball-family share of the spread.

Usage: python3 scripts/research/stuff/stuff_drift_diagnostic.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stuff_plus.train_stuff as T                      # noqa: E402
import stuff_gate_v2 as G                               # noqa: E402

OUT = os.path.join(ROOT, 'data', '_stuff_drift_diagnostic.json')


def score_season(Y, train_years, frames):
    slopes = G.fit_vaa_slopes([frames[y] for y in train_years])
    P = {y: G.prepare(frames[y], slopes) for y in train_years + [Y]}
    tr = [G.apply_variant(P[y], {}) for y in train_years]
    feats = tr[0][1]
    Xtr = pd.concat([G.design(d, feats) for d, _ in tr], ignore_index=True)
    ytr = np.concatenate([d['target_xrv'].values for d, _ in tr])
    dY, _ = G.apply_variant(P[Y], {})
    m = xgb.XGBRegressor(**T._params_for(Xtr))
    m.fit(Xtr, ytr)
    return dY.assign(stuff=-m.predict(G.design(dY, feats)))


def season_stats(d):
    d = d.assign(neg=-d['target_xrv'], rv=-d['rv_raw'])
    g = d.groupby('pitcher').agg(s=('stuff', 'mean'), t=('neg', 'mean'),
                                 rv=('rv', 'mean'), n=('stuff', 'size'))
    g = g[g['n'] >= 300]
    fam = d[d['pitch_type'].isin({'FF', 'SI'})]
    gf = fam.groupby('pitcher').agg(s=('stuff', 'mean'), n=('stuff', 'size'))
    gf = gf[gf['n'] >= 150]
    return dict(n_pitchers=int(len(g)),
                pitcher_sd_per100=float(g['s'].std() * 100),
                pitch_sd_per100=float(d['stuff'].std() * 100),
                r_target=G.pear(g['s'], g['t']),
                r_rv=G.pear(g['s'], g['rv']),
                fb_pitcher_sd_per100=float(gf['s'].std() * 100),
                mean_per100=float(g['s'].mean() * 100))


def main():
    frames = {y: pd.read_pickle(G.season_path(y)) for y in G.SEASONS}
    G.set_arm_side_sign(list(frames.values()))
    out = {}
    for Y in (2021, 2022, 2023, 2024, 2025):
        # same recipe as the gate's SHIPPED, re-fit for the pitch-level frame
        train_years = [y for y in G.SEASONS if y not in (Y, Y + 1)]
        dY = score_season(Y, train_years, frames)
        out[Y] = season_stats(dY)
        print(Y, out[Y], flush=True)
    dY = score_season(2026, [2021, 2022, 2023, 2024], frames)
    out[2026] = season_stats(dY)
    print(2026, out[2026], flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
