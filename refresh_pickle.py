#!/usr/bin/env python3
"""refresh_pickle.py — fast refresh of data/all_pitches_rs_cache.pkl.

Why this exists
---------------
The full Huronalytics pipeline (process_data.py) runs on CI and writes
two outputs:

  1. data/hitter_leaderboard_rs.json (and friends) — committed to git,
     pulled to your laptop every time you `git pull`.
  2. data/all_pitches_rs_cache.pkl                — gitignored (~100 MB,
     too big), stays on the CI server, NEVER reaches your laptop.

HitterCards.py reads BOTH. When the JSON is fresh (today) but the pickle
is stale (last local pipeline run), the per-pitch heat maps and LA × Spray
scatter plot a stale snapshot while the aggregate stats show fresh
numbers. Confusing.

refresh_pickle.py rebuilds JUST the pickle by re-reading Google Sheets
and applying the same pre-pickle cleanup that process_data.py does.
Everything downstream of the pickle (stats / percentiles / JSON writes)
is skipped, so this runs in ~1-2 minutes instead of ~5-10.

Usage
-----
    python3 refresh_pickle.py

Requires the same Google Sheets credentials process_data.py uses
(~/.config/gspread, GOOGLE_SERVICE_ACCOUNT_JSON env var, or
service_account.json next to this script).
"""

import json
import os
import pickle
import sys

# Add script dir to path so we can import the shared pipeline modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_fetch import read_all_pitches_from_sheets
from pipeline_utils import DATA_DIR, MLB_TEAMS, compute_in_zone


def _apply_pre_pickle_cleanup(all_pitches):
    """Run the same in-place cleanup that process_data.py does between the
    Sheet read and the pickle write. Three steps:

    1. Recompute InZone from PlateX/PlateZ/SzTop/SzBot (ball-radius
       adjusted). Sheet data has a precomputed InZone but we re-derive it
       for consistency with the rest of the pipeline.
    2. Remap non-MLB BTeam entries to their MLB team based on the batter's
       known team (skip AAA-source rows — they really play in AAA).
    3. Tag ROC/AAA pitches so downstream consumers know which side of the
       at-bat is the AAA player.
    """
    # 1. InZone
    for p in all_pitches:
        p['InZone'] = compute_in_zone(p)

    # 2. BTeam remapping
    mlb_hitter_teams = {}
    for p in all_pitches:
        batter = p.get('Batter')
        b_team = p.get('BTeam')
        if batter and b_team and b_team in MLB_TEAMS:
            mlb_hitter_teams[batter] = b_team

    remapped = 0
    for p in all_pitches:
        b_team = p.get('BTeam')
        if b_team and b_team not in MLB_TEAMS:
            if p.get('_source') == 'AAA':
                continue
            batter = p.get('Batter')
            if batter and batter in mlb_hitter_teams:
                p['BTeam'] = mlb_hitter_teams[batter]
                remapped += 1
    if remapped:
        print(f"  Remapped {remapped} non-MLB BTeam entries")

    # 3. ROC tagging
    roc_p_count = 0
    roc_h_count = 0
    for p in all_pitches:
        src = p.get('_source', 'MLB')
        if src == 'ROC':
            p['_roc_pitcher_pitch'] = True
            roc_p_count += 1
        elif src == 'AAA':
            # e00bf72 data model (mirrors process_data): the batter is a
            # tracked Rochester hitter ONLY when BTeam == 'ROC'; every other
            # batter is opponent byproduct and is excluded from hitter stats
            # too. The old `BTeam 'AAA' -> 'ROC'` rewrite here silently
            # REVERTED sheet-side departure retags (released hitters moved
            # ROC -> AAA) in the local pickle, so cards and local analyses
            # kept showing departed players as ROC. Never rewrite BTeam.
            p['_roc_hitter_pitch'] = True
            if p.get('BTeam') != 'ROC':
                p['_roc_pitcher_pitch'] = True
            roc_h_count += 1
    if roc_p_count or roc_h_count:
        print(f"  Tagged {roc_p_count} ROC pitcher pitches, "
              f"{roc_h_count} ROC hitter pitches")


def main():
    print("\n=== Reading Regular Season data (6 division workbooks) ===")
    rs_pitches = read_all_pitches_from_sheets()
    print(f"  Read {len(rs_pitches)} RS pitches")

    print("\n=== Applying pre-pickle cleanup ===")
    _apply_pre_pickle_cleanup(rs_pitches)

    print("\n=== Writing pickle ===")
    cache_path = os.path.join(DATA_DIR, 'all_pitches_rs_cache.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(rs_pitches, f)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"  Wrote {len(rs_pitches)} pitches to {cache_path} ({size_mb:.1f} MB)")

    # Latest date in the refresh — confirms the pickle is fresh.
    dates = sorted({p.get('Game Date') for p in rs_pitches if p.get('Game Date')})
    if dates:
        print(f"  Date range: {dates[0]} → {dates[-1]} ({len(dates)} game dates)")


if __name__ == '__main__':
    main()
