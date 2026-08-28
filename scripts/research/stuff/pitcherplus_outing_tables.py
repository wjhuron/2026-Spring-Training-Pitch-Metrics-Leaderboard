#!/usr/bin/env python3
"""pitcherplus_outing_tables.py — outing-level component tables, 2021-2025.

Pitcher+ overhaul part 2 (2026-08-28): the daily-card outing grade needs a
league distribution of SINGLE OUTINGS, plus within-outing split halves to
fit weights ("which component mix best estimates tonight's quality,
validated by predicting the held-out half of the same outing").

Per (pitcher, game_pk) this script emits the season-Pitcher+ component set
at the outing grain, plus an odd/even-PA split within the outing:

  pid, season, date, game_pk, n, pa, nbip,
  kPct, bbPct, kbbPct, cswPct, whiffPct, izWhiffPct, chasePct, gbPct,
  xrv100, rv100, locRaw, locN
  + the same columns suffixed _o / _e for the odd/even PA halves
    (PA index parity within the outing; PA index from chronological
    order, reconstructed by reversing Savant's descending order within
    game_pk — the verified pitcherplus_v2 technique)

locRaw is the PRODUCTION Loc+ v2 per-pitch score (pipeline.locplus
build_surfaces/score_pitch on per-season league surfaces, same adaptation
as scripts/archive/pitcherplus_locplus_hist.py), pitcher perspective.
Outing stuffRaw merges in later from data/_pplus_stuff_loso_games.csv
(v14 LOSO, keyed pid+season+date) — not computed here.

Usage: PYTHONHASHSEED=0 python3 scripts/research/stuff/pitcherplus_outing_tables.py
Output: data/_pplus_outing_tables.pkl
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'misc'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import leaderboard_metric_battery as bat            # noqa: E402
import pitcherplus_search as ps                     # noqa: E402
from pipeline.locplus import (build_surfaces, score_pitch,  # noqa: E402
                              _is_scorable)

OUT_PKL = os.path.join(ROOT, 'data', '_pplus_outing_tables.pkl')
SEASONS = [2021, 2022, 2023, 2024, 2025]

# savant description -> sheet Description (Pitcher2026.simplify_description)
DESC_MAP = {
    'called_strike': 'Called Strike',
    'ball': 'Ball', 'blocked_ball': 'Ball', 'intent_ball': 'Ball',
    'pitchout': 'Pitchout',
    'swinging_strike': 'Swinging Strike',
    'swinging_strike_blocked': 'Swinging Strike',
    'foul_tip': 'Swinging Strike',
    'swinging_pitchout': 'Swinging Pitchout',
    'foul_pitchout': 'Foul Pitchout',
    'foul': 'Foul',
    'hit_into_play': 'In Play',
    'hit_by_pitch': 'Hit By Pitch',
    'foul_bunt': 'Foul Bunt', 'missed_bunt': 'Missed Bunt',
    'bunt_foul_tip': 'Bunt Foul Tip',
}


def add_loc(df, year):
    """Per-pitch production Loc+ v2 score (pitcher perspective) as a column.
    Surfaces are league-level per season, so same-season scoring carries no
    pitcher-level leakage (same argument as pitcherplus_locplus_hist)."""
    lg, scale = ps.GUTS[year]
    ev = df['events'].fillna('')
    desc = df['description'].fillna('').map(lambda d: DESC_MAP.get(d, d))
    bbtype = np.where(ev.isin(bat.SH_EV), 'bunt', df['bb_type'].fillna(''))
    count = (df['balls'].fillna(-1).astype(int).astype(str) + '-'
             + df['strikes'].fillna(-1).astype(int).astype(str))
    recs = pd.DataFrame({
        'Description': desc, 'Count': count,
        'PlateX': df['plate_x'], 'PlateZ': df['plate_z'],
        'SzTop': df['sz_top'], 'SzBot': df['sz_bot'],
        'Bats': df['stand'], 'Throws': df['p_throws'],
        'Pitch Type': df['pitch_type'], 'BBType': bbtype,
        'Event': np.where(ev == 'intent_walk', 'Intent Walk', ''),
        'RunExp': df['rv'],
        'xwOBA': df['estimated_woba_using_speedangle'],
    }, index=df.index)
    recs['_source'] = 'MLB'
    dicts = recs.to_dict('records')
    idx = list(recs.index)
    scorable = [(i, p) for i, p in zip(idx, dicts) if _is_scorable(p)]
    S = build_surfaces([p for _, p in scorable], lg, scale)
    loc = pd.Series(np.nan, index=df.index)
    for i, p in scorable:
        v = score_pitch(p, S)
        if v is not None:
            loc[i] = -v          # pitcher perspective, higher = better
    df['loc_pitch'] = loc
    return df


def add_pa_index(df):
    """Chronological PA index within (pitcher, game_pk). The cache stores
    games in Savant's descending order, so chronological = reversed row
    order within game_pk (verified technique from pitcherplus_v2)."""
    df = df.copy()
    df['_row'] = np.arange(len(df))
    df = df.sort_values(['pitcher', 'game_pk', '_row'],
                        ascending=[True, True, False], kind='stable')
    grp = df.groupby(['pitcher', 'game_pk'], sort=False)
    # PA index = number of completed PAs BEFORE this pitch
    df['pa_idx'] = grp['pa_end'].cumsum() - df['pa_end'].astype(int)
    return df.sort_values('_row').drop(columns=['_row'])


def outing_aggs(g):
    n = len(g)
    sw = g['swing'].sum()
    izn = g['iz'].sum()
    iz_sw = (g['iz'] & g['swing']).sum()
    ooz = g['ooz'].sum()
    pa = g['pa_end'].sum()
    bipn = g['bip'].sum()
    locv = g['loc_pitch'].dropna()
    xv = g['xrv_pitch'].dropna()
    out = {
        'n': n, 'pa': int(pa), 'nbip': int(bipn),
        'kPct': g['k'].sum() / pa if pa else np.nan,
        'bbPct': g['bbw'].sum() / pa if pa else np.nan,
        'kbbPct': (g['k'].sum() - g['bbw'].sum()) / pa if pa else np.nan,
        'cswPct': g['csw'].sum() / n if n else np.nan,
        'whiffPct': g['whiff'].sum() / sw if sw else np.nan,
        'izWhiffPct': (g['iz'] & g['whiff']).sum() / iz_sw
                      if iz_sw else np.nan,
        'chasePct': g['chase_sw'].sum() / ooz if ooz else np.nan,
        'gbPct': g['gb'].sum() / bipn if bipn else np.nan,
        'xrv100': xv.sum() / n * 100 if n and len(xv) else np.nan,
        'rv100': g['rv'].sum() / n * 100 if n else np.nan,
        'locRaw': locv.mean() if len(locv) else np.nan,
        'locN': len(locv),
    }
    return out


def main():
    rows = []
    for year in SEASONS:
        print(f'── {year}', flush=True)
        df = bat.load_season(year)
        df = ps.add_extra_flags(df)
        df = ps.add_xrv(df, year)
        print(f'   league xRV {df["xrv_pitch"].sum():+.0f} '
              f'vs RV {df["rv"].sum():+.0f} ({len(df)} pitches)', flush=True)
        df = add_loc(df, year)
        print(f'   loc scored {df["loc_pitch"].notna().mean():.3f} of pitches',
              flush=True)
        df = add_pa_index(df)
        for (pid, gpk), g in df.groupby(['pitcher', 'game_pk'], sort=False):
            row = {'pid': pid, 'season': year, 'game_pk': gpk,
                   'date': g['game_date'].iloc[0]}
            row.update(outing_aggs(g))
            odd = g[g['pa_idx'] % 2 == 1]
            even = g[g['pa_idx'] % 2 == 0]
            for suf, gg in (('_o', odd), ('_e', even)):
                for k, v in outing_aggs(gg).items():
                    row[k + suf] = v
            rows.append(row)
        print(f'   outings so far {len(rows)}', flush=True)
        del df
    out = pd.DataFrame(rows)
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(out, f)
    print(f'saved {OUT_PKL} ({len(out)} rows)')


if __name__ == '__main__':
    main()
