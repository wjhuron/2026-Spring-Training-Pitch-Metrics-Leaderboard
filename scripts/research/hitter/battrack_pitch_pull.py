"""battrack_pitch_pull.py — pitch-level bat tracking for 2024 + 2025 MLB.

The BB+ bat prior must be fit in PRODUCTION currency: pipeline/compute.py
builds batSpeed (mean per-swing BatSpeed at bs >= 50) and squaredUpPct
(EV >= 0.80 * (0.212 * release velo + 1.23 * bat speed) among
blast-eligible swings) from per-pitch atoms. The 2026 sheets carry those
atoms; 2024-2025 need this pull. Statcast search CSV, date-chunked under
the 25k row cap, columns pruned to what the formulas need.

Coverage note: public bat tracking begins mid-April 2024, so the 2024
replicate starts when the data does.

Output: data/_battrack_pitch_{year}.pkl (DataFrame: game_date, batter,
bat_speed, launch_speed, release_speed, description)

Usage: python3 scripts/research/hitter/battrack_pitch_pull.py
"""
import io
import os
import pickle
import time
from datetime import date, timedelta

import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
URL = 'https://baseballsavant.mlb.com/statcast_search/csv'
KEEP = ['game_date', 'batter', 'bat_speed', 'launch_speed',
        'release_speed', 'description', 'game_type']
WINDOWS = {2024: (date(2024, 4, 1), date(2024, 10, 1)),
           2025: (date(2025, 3, 25), date(2025, 9, 29))}
CHUNK_DAYS = 4


def pull_chunk(sess, d0, d1):
    params = {
        'all': 'true', 'type': 'details',
        'game_date_gt': d0.isoformat(), 'game_date_lt': d1.isoformat(),
        'player_type': 'batter',
        'min_pitches': '0', 'min_results': '0',
        'sort_col': 'pitches', 'sort_order': 'desc',
    }
    for attempt in range(3):
        try:
            r = sess.get(URL, params=params, timeout=120)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
            return df
        except Exception as e:
            print(f'  {d0}..{d1} attempt {attempt + 1}: {e}', flush=True)
            time.sleep(8)
    raise RuntimeError(f'chunk {d0}..{d1} failed 3 times — aborting rather '
                       f'than writing a partial season')


def main():
    sess = requests.Session()
    sess.headers['Referer'] = 'https://baseballsavant.mlb.com/gamefeed'
    for year, (start, end) in WINDOWS.items():
        out_path = os.path.join(ROOT, 'data', f'_battrack_pitch_{year}.pkl')
        if os.path.exists(out_path):
            print(f'{year}: exists — skipped')
            continue
        parts = []
        d = start
        while d < end:
            d1 = min(d + timedelta(days=CHUNK_DAYS - 1), end)
            df = pull_chunk(sess, d, d1)
            if len(df) >= 24500:
                raise RuntimeError(f'{d}..{d1}: {len(df)} rows — at the '
                                   f'25k cap, shrink CHUNK_DAYS')
            cols = [c for c in KEEP if c in df.columns]
            parts.append(df[cols])
            print(f'  {year} {d}..{d1}: {len(df)} rows '
                  f'({sum(len(p) for p in parts)} total)', flush=True)
            d = d1 + timedelta(days=1)
            time.sleep(1.5)
        full = pd.concat(parts, ignore_index=True)
        if 'game_type' in full.columns:
            full = full[full['game_type'] == 'R']
        n_bs = full['bat_speed'].notna().sum()
        print(f'{year}: {len(full)} rows, {n_bs} with bat_speed')
        if n_bs < 100000:
            raise RuntimeError(f'{year}: only {n_bs} bat_speed rows — '
                               f'shape check failed, not writing')
        tmp = out_path + '.tmp'
        full.to_pickle(tmp)
        os.replace(tmp, out_path)
        print(f'{year}: saved {out_path}')


if __name__ == '__main__':
    main()
