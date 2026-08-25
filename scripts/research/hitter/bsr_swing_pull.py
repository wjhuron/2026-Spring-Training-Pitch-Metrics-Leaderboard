"""bsr_swing_pull.py — per-swing bat_speed + swing_length for 2024 + 2025.

The BSR screen (bsr_screen.py) needs the per-swing PAIR of bat_speed and
swing_length, plus location/velo controls. data/_battrack_pitch_{year}.pkl
kept bat_speed only, and the full statcast caches carry neither, so this
is a fresh date-chunked Savant search pull with the extra columns.

Output: data/_bsr_swing_{year}.pkl (DataFrame: game_date, batter, stand,
bat_speed, swing_length, plate_x, plate_z, release_speed, description)

Usage: python3 scripts/research/hitter/bsr_swing_pull.py
"""
import io
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
URL = 'https://baseballsavant.mlb.com/statcast_search/csv'
KEEP = ['game_date', 'batter', 'stand', 'bat_speed', 'swing_length',
        'plate_x', 'plate_z', 'release_speed', 'description', 'game_type']
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
            return pd.read_csv(io.StringIO(r.text), low_memory=False)
        except Exception as e:
            print(f'  {d0}..{d1} attempt {attempt + 1}: {e}', flush=True)
            time.sleep(8)
    raise RuntimeError(f'chunk {d0}..{d1} failed 3 times — aborting rather '
                       f'than writing a partial season')


def main():
    sess = requests.Session()
    sess.headers['Referer'] = 'https://baseballsavant.mlb.com/gamefeed'
    for year, (start, end) in WINDOWS.items():
        out_path = os.path.join(ROOT, 'data', f'_bsr_swing_{year}.pkl')
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
        tmp = out_path + '.tmp'
        full.to_pickle(tmp)
        os.replace(tmp, out_path)
        print(f'{year}: wrote {len(full)} rows -> {out_path}')


if __name__ == '__main__':
    main()
