#!/usr/bin/env python3
"""pitcherplus_locplus_hist.py — historical Loc+ raw scores, 2021-2025.

Pitcher+ project phase 1 (2026-07-24). Scores every 2021-2025 season with
the PRODUCTION Loc+ code (pipeline_locplus.build_surfaces / score_pitch,
locked v2 config: anchoring off, plain-average canon) — no reimplementation.
Each season gets its own league surfaces (per the July multi-season audit:
pooling seasons HURTS Loc+), built from that season's Savant cache adapted
to sheet-style dicts. League-level surfaces make same-season scoring safe
(no pitcher-level leakage).

Savant tags for 2021-2025 (not retagged) — group_of() uses the tag only to
pick the surface group, same caveat as the rest of the search.

Output: data/_pplus_locplus_hist.csv with
  pid, season, half (A/B/full), locRaw (PITCHER perspective: -mean ExpRV,
  higher = better), locN
Halves use the same odd/even game split as pitcherplus_search.py
(leaderboard_metric_battery.add_half).

Usage: python3 scripts/pitcherplus_locplus_hist.py   (~10-30 min)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import leaderboard_metric_battery as bat  # noqa: E402
from pipeline.locplus import (build_surfaces, score_pitch,  # noqa: E402
                              _is_scorable)

OUT_CSV = os.path.join(ROOT, 'data', '_pplus_locplus_hist.csv')
SEASONS = [2021, 2022, 2023, 2024, 2025]
GUTS = {
    2021: (0.314, 1.209), 2022: (0.310, 1.259),
    2023: (0.318, 1.204), 2024: (0.310, 1.242),
    2025: (0.3131, 1.2317),
}

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
SH_EV = bat.SH_EV


def season_dicts(year):
    df = bat.load_season(year).reset_index(drop=True)
    df = bat.add_half(df, 'pitcher').reset_index(drop=True)
    ev = df['events'].fillna('')
    desc = df['description'].fillna('').map(lambda d: DESC_MAP.get(d, d))
    bbtype = np.where(ev.isin(SH_EV), 'bunt', df['bb_type'].fillna(''))
    count = (df['balls'].fillna(-1).astype(int).astype(str) + '-'
             + df['strikes'].fillna(-1).astype(int).astype(str))
    rv = df['rv']
    recs = pd.DataFrame({
        'pid': df['pitcher'], 'half': df['half'],
        'Description': desc, 'Count': count,
        'PlateX': df['plate_x'], 'PlateZ': df['plate_z'],
        'SzTop': df['sz_top'], 'SzBot': df['sz_bot'],
        'Bats': df['stand'], 'Throws': df['p_throws'],
        'Pitch Type': df['pitch_type'], 'BBType': bbtype,
        'Event': np.where(ev == 'intent_walk', 'Intent Walk', ''),
        'RunExp': rv,
        'xwOBA': df['estimated_woba_using_speedangle'],
    })
    recs['_source'] = 'MLB'
    return recs.to_dict('records')


def main():
    rows = []
    for year in SEASONS:
        print(f'── {year}', flush=True)
        pitches = season_dicts(year)
        lg, scale = GUTS[year]
        scorable = [p for p in pitches if _is_scorable(p)]
        S = build_surfaces(scorable, lg, scale)
        print(f'   surfaces built on {len(scorable)}/{len(pitches)} pitches',
              flush=True)
        acc = {}
        for p in scorable:
            v = score_pitch(p, S)
            if v is None:
                continue
            for h in ('full', p['half']):
                a = acc.setdefault((p['pid'], h), [0.0, 0])
                a[0] += v
                a[1] += 1
        for (pid, h), (s, n) in acc.items():
            rows.append({'pid': pid, 'season': year, 'half': h,
                         'locRaw': -s / n, 'locN': n})
        print(f'   scored {sum(1 for k in acc if k[1] == "full")} pitchers',
              flush=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f'saved {OUT_CSV} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
