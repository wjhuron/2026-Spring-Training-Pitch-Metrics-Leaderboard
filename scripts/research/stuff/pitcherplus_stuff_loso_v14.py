#!/usr/bin/env python3
"""pitcherplus_stuff_loso.py — leave-one-season-out Stuff+ raw scores, 2021-2025.

Pitcher+ project phase 1 (2026-07-24). The shipped v11 bundle trained on ALL
of 2021-2025 (prior rows join every fold), so scoring those seasons with it
would be in-sample and inflate Stuff+ in the candidate search. This script
holds each season out entirely: season Y is scored by a model trained on the
other four seasons (v11 architecture verbatim: build_df/design/_params_for
from stuff_plus/train_stuff.py, per-season FG Guts for targets).
Cross-season same-pitcher rows remain in training — that matches the v11
leakage standard (same-season outcomes are leakage; cross-season identity is
real support).

Output: data/_pplus_stuff_loso.csv with
  pid, season, half (A/B/full), stuffRaw (pitcher perspective: -mean
  predicted target_xrv, higher = better), stuffN
Halves reuse the odd/even game split from leaderboard_metric_battery
.add_half, joined on (player_name, game_date); doubleheader dates (two
game_pks for one pitcher-date) are dropped from the half aggregation but
kept in 'full'.

Usage: python3 scripts/pitcherplus_stuff_loso.py    (~1-2 h, background)
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'misc'))
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus'))

import leaderboard_metric_battery as bat        # noqa: E402
import stuff_plus.train_stuff as tv                    # noqa: E402

OUT_CSV = os.path.join(ROOT, 'data', '_pplus_stuff_loso.csv')
# per-outing means for the outing-grade research (Pitcher+ overhaul, 2026-08-28)
OUT_GAMES_CSV = os.path.join(ROOT, 'data', '_pplus_stuff_loso_games.csv')
SEASONS = [2021, 2022, 2023, 2024, 2025]
GUTS = dict(tv.HIST_GUTS)
GUTS[2025] = (tv.PRIOR_LG_WOBA, tv.PRIOR_WOBA_SCALE)


def build_year(year):
    pitches = pickle.load(open(
        os.path.join(ROOT, 'data', f'_pitches{year}_training.pkl'), 'rb'))
    lg, sc = GUTS[year]
    _lg0, _sc0 = tv.LG_WOBA, tv.WOBA_SCALE
    tv.LG_WOBA, tv.WOBA_SCALE = lg, sc
    d = tv.build_df(pitches)
    tv.LG_WOBA, tv.WOBA_SCALE = _lg0, _sc0
    d = d[d['target_xrv'].notna()].reset_index(drop=True)
    print(f'  {year}: {len(d)} training pitches, '
          f'{d.pitcher.nunique()} pitchers', flush=True)
    return d


def half_maps(year):
    """(player_name, date) -> half and player_name -> pid, from the cache.
    Doubleheader (name, date) pairs -> half None."""
    df = bat.load_season(year).reset_index(drop=True)
    df = bat.add_half(df, 'pitcher')
    g = df.drop_duplicates(['player_name', 'game_date', 'game_pk'])[
        ['player_name', 'game_date', 'game_pk', 'half']]
    per_date = g.groupby(['player_name', 'game_date'])
    hm = {}
    for (nm, dt), gg in per_date:
        hm[(nm, dt)] = gg['half'].iloc[0] if len(gg) == 1 else None
    pid_counts = df.groupby('player_name')['pitcher'].nunique()
    ambiguous = set(pid_counts[pid_counts > 1].index)
    pid_map = (df.drop_duplicates('player_name')
               .set_index('player_name')['pitcher'].to_dict())
    for nm in ambiguous:
        pid_map.pop(nm, None)
    if ambiguous:
        print(f'  {year}: {len(ambiguous)} ambiguous names dropped '
              f'({sorted(ambiguous)[:4]}...)', flush=True)
    return hm, pid_map


def main():
    dfs = {y: build_year(y) for y in SEASONS}
    rows = []
    game_rows = []
    for year in SEASONS:
        print(f'── LOSO {year}', flush=True)
        train = pd.concat([dfs[y] for y in SEASONS if y != year],
                          ignore_index=True)
        X = tv.design(train)
        y = train['target_xrv'].values
        model = XGBRegressor(**tv._params_for(X))
        model.fit(X, y)
        test = dfs[year]
        Xt = tv.design(test).reindex(columns=X.columns, fill_value=0)
        pred = model.predict(Xt)
        test = test.copy()
        test['stuff_p'] = -pred          # pitcher perspective, higher better
        hm, pid_map = half_maps(year)
        test['half'] = [hm.get((nm, dt))
                        for nm, dt in zip(test['pitcher'], test['date'])]
        matched = test['half'].notna().mean()
        print(f'  half-join match {matched:.3f}', flush=True)
        for (nm, dt), g in test.groupby(['pitcher', 'date']):
            pid = pid_map.get(nm)
            if pid is None:
                continue
            game_rows.append({'pid': pid, 'season': year, 'date': dt,
                              'stuffRaw': g['stuff_p'].mean(),
                              'stuffN': len(g)})
        for nm, g in test.groupby('pitcher'):
            pid = pid_map.get(nm)
            if pid is None:
                continue
            for h in ('full', 'A', 'B'):
                gg = g if h == 'full' else g[g['half'] == h]
                if not len(gg):
                    continue
                rows.append({'pid': pid, 'season': year, 'half': h,
                             'stuffRaw': gg['stuff_p'].mean(),
                             'stuffN': len(gg)})
        del train, X, Xt, model
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f'saved {OUT_CSV} ({len(rows)} rows)')
    pd.DataFrame(game_rows).to_csv(OUT_GAMES_CSV, index=False)
    print(f'saved {OUT_GAMES_CSV} ({len(game_rows)} rows)')


if __name__ == '__main__':
    main()
